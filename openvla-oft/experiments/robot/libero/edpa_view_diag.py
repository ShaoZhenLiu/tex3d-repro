#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""edpa_view_diag.py — C2 前置诊断：512×512 InfoNCE 的跨相机负样本结构（第12轮 P3）
三个测量（老师 §4 P3）：
  A) 固定 32 帧上，用 B2v2 最终纹理（tex_override 渲染 adv）计算 L_patch logits，
     对每个 query token 统计 softmax 质量分配到**另一相机** clean token 的比例
     （布面可见 query 单列；均匀基线 = 256/512 = 0.5）；
  B) 两视图布面可见率；
  C) 分项 θ 梯度范数：patch CE 全体 / 仅主相机 query / 仅腕相机 query（init θ，v2 诊断口径）。
预登记判读规则（先于看数固定，写入日志）：
  若 布面 query 的跨相机质量占比 > 0.5（即跨相机负样本比本相机更“吸引”）
  且 双视图平均布可见率 ≥ 20% → C2（视图内 InfoNCE）值得训练；
  否则跳过 C2，本轮不强行组合。"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import attack_oft as base            # noqa: E402
import transfer_pi2ov as TR          # noqa: E402
from attack_oft_cloth_edpa_v2 import (  # noqa: E402
    CKPT, ClothRendererOV, _norm_consts, tv_loss)
from attack_oft_cloth_upa import vis_token_mask  # noqa: E402

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
B2V2_PNG = ("/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/"
            "train_20261020_191751_edpav2/Universal_Cloth_OV.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.07)
    args = ap.parse_args()

    print("[预登记判读] 布面query跨相机质量>0.5 且 双视图均可见率≥20% → C2 值得训；否则跳过")

    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
    model = base.get_model(cfg)
    size = base.get_image_resize_size(cfg)
    model.eval()
    device = "cuda"
    sm, ss, dm, ds = _norm_consts(device)

    pool = torch.load(POOL, map_location=device, weights_only=False)
    frames = []
    for tid in sorted(pool.keys()):
        for fr in pool[tid]:
            if fr["mvp"] is not None and fr.get("wrist_mvp") is not None:
                if float(fr["seg"].mean()) > 0.05 and float(fr["wrist_seg"].mean()) > 0.02:
                    frames.append((tid, fr))
        if len(frames) >= args.frames:
            break
    frames = frames[:args.frames]
    print("[diag] 固定 %d 帧（双视图均见布）" % len(frames))

    renderer = ClothRendererOV(device=device, init_std=0.01, seed=7).to(device)
    t0_tex = renderer.orig_texture
    b2v2 = torch.from_numpy(
        np.asarray(Image.open(B2V2_PNG).convert("RGB"), dtype=np.float32) / 255.0
    ).to(device).unsqueeze(0)
    b2v2 = torch.flip(b2v2, dims=[1])          # 撤销导出时的垂直翻转

    def compose(fr, tex):
        rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                                  tex_override=tex)
        comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                           + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
        wrgba, _ = renderer.render(fr["wrist_mvp"], (256, 256), model_rot=fr["rot"],
                                   tex_override=tex)
        wrist = torch.clamp(wrgba.permute(0, 3, 1, 2) * fr["wrist_seg"]
                            + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
        return comp, wrist

    def to_pv(comp, wrist):
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=size, siglip_mean=sm, siglip_std=ss,
            dino_mean=dm, dino_std=ds, wrist_tensor=wrist).to(torch.bfloat16)

    # ---- A+B：最终纹理下的跨相机 softmax 质量 ----
    stats_all, stats_cloth, vis_rates = [], [], []
    CH = 8
    for c0 in range(0, len(frames), CH):
        chunk = frames[c0:c0 + CH]
        adv_pv = torch.cat([to_pv(*compose(fr, b2v2)) for _, fr in chunk], 0)
        cln_pv = torch.cat([to_pv(*compose(fr, t0_tex)) for _, fr in chunk], 0)
        with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
            za = model.projector(model.vision_backbone(adv_pv)).float()
            zc = model.projector(model.vision_backbone(cln_pv)).float()
        logits = torch.bmm(F.normalize(za, dim=-1),
                           F.normalize(zc, dim=-1).transpose(1, 2)) / args.temp
        probs = logits.softmax(dim=-1)                    # [B,512,512]
        B, P, _ = probs.shape
        g = P // 2
        own = torch.cat([probs[:, :g, :g].sum(-1), probs[:, g:, g:].sum(-1)], dim=1)
        oth = torch.cat([probs[:, :g, g:].sum(-1), probs[:, g:, :g].sum(-1)], dim=1)
        vm = vis_token_mask(chunk, P, device) > 0.5       # [B,P] 布面 token
        stats_all.append(((oth / (oth + own + 1e-12)).mean().item(), len(chunk) * P))
        cloth_ratio = (oth / (oth + own + 1e-12))[vm]
        stats_cloth.append((cloth_ratio.mean().item(), int(vm.sum().item())))
        vis_rates.append((float(vm[:, :g].float().mean()), float(vm[:, g:].float().mean())))
    wa = sum(v * n for v, n in stats_all) / sum(n for _, n in stats_all)
    wc = sum(v * n for v, n in stats_cloth) / sum(n for _, n in stats_cloth)
    v0 = np.mean([a for a, _ in vis_rates]); v1 = np.mean([b for _, b in vis_rates])
    print("[A] 跨相机质量占比 oth/(oth+own)：全 query 均值 %.4f（均匀基线 0.5）" % wa)
    print("[A] 布面 query 均值 %.4f（n=%d）" % (wc, sum(n for _, n in stats_cloth)))
    print("[B] 布面 token 可见率：主相机 %.3f / 腕相机 %.3f" % (v0, v1))

    # ---- C：init θ 分项梯度（patch CE 全体 / 仅主 / 仅腕）----
    batch = frames[:8]
    param = renderer.adv_noise
    uv = torch.eye(2, 3, device=device)
    uv[0, 0] = uv[1, 1] = 1.0
    adv_pv = torch.cat([to_pv(*compose(fr, None)) for _, fr in batch], 0)
    cln_pv = torch.cat([to_pv(*compose(fr, t0_tex)) for _, fr in batch], 0)
    with base.autocast(dtype=torch.bfloat16):
        za = model.projector(model.vision_backbone(adv_pv)).float()
    with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
        zc = model.projector(model.vision_backbone(cln_pv)).float()
    logits = torch.bmm(F.normalize(za, dim=-1),
                       F.normalize(zc, dim=-1).transpose(1, 2)) / args.temp
    labels = torch.arange(logits.size(1), device=device).unsqueeze(0).expand(
        logits.size(0), -1)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                         reduction="none").reshape(logits.size(0), logits.size(1))
    g = logits.size(1) // 2
    for nm, rows in (("全query", slice(None)), ("仅主相机query", slice(0, g)),
                     ("仅腕相机query", slice(g, None))):
        gr = torch.autograd.grad(ce[:, rows].mean(), param, retain_graph=True)[0]
        print("[C] θ 梯度范数 %-12s %.4e" % (nm, gr.norm().item()))
    tex = renderer.texture_tensor()
    print("[C] 正则参照: ref=%.4f tv=%.4f（init θ）" %
          (F.mse_loss(tex, t0_tex).item(), tv_loss(tex - t0_tex).item()))

    go = (wc > 0.5) and (v0 >= 0.20) and (v1 >= 0.20)
    print("[判定] %s：布面query跨相机占比 %.4f %s 0.5，可见率 %.2f/%.2f %s 0.20" %
          ("C2 值得训练" if go else "跳过 C2", wc, ">" if wc > 0.5 else "<=",
           v0, v1, "≥" if (v0 >= 0.2 and v1 >= 0.2) else "<"))


if __name__ == "__main__":
    main()
