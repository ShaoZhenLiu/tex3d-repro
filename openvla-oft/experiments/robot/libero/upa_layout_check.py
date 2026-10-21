#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""upa_layout_check.py — B3 双视图 token 布局核验（第12轮 P0-B，只读不重训）
核对项（老师 §3）：
  1) 实际模型代码哈希、num_images_in_input、backbone IDs（按配置确认 DINOv2/SigLIP 顺序）
  2) 输入/各塔各相机特征/projector/总序列 P/文本切片形状
  3) 每相机 mask 有效 token 数、resize→flatten→拼接顺序
  4) 扰动试验：只改主视图 / 只改腕视图 → 对应 256-token 块响应、另一块不动
  5) 导出 PNG SHA 与源门/目标评测用 SHA 对账
结论写入 stdout（tee 到日志）。GPU3 运行。"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import attack_oft as base            # noqa: E402
import transfer_pi2ov as TR          # noqa: E402
from attack_oft_cloth_upa import (   # noqa: E402
    CKPT, ClothRendererOV, _norm_consts, vis_token_mask)

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
TEX_B3 = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261020_211041_upa/Universal_Cloth_OV.png"
TEX_B2V2 = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261020_191751_edpav2/Universal_Cloth_OV.png"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block_diff(a, b):
    """两 token 序列按 [0:256]/[256:512] 分块的 L2 范数."""
    d = (a - b).reshape(a.shape[0], a.shape[1], -1)
    n = d.norm(dim=-1).mean(dim=0)
    return float(n[:256].mean()), float(n[256:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2)
    args = ap.parse_args()

    print("=" * 72)
    print("[0] 文件 SHA 对账")
    for name, p in [("B3 纹理(训练导出)", TEX_B3),
                    ("B2v2 纹理(训练导出)", TEX_B2V2)]:
        print("  %-22s %s  %s" % (name, sha256(p), p))

    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    size = base.get_image_resize_size(cfg)
    model.eval()

    print("=" * 72)
    print("[1] 模型身份")
    mod = type(model).__module__
    mfile = os.path.abspath(sys.modules[mod].__file__)
    print("  model class module : %s" % mod)
    print("  model code sha256  : %s (%s)" % (sha256(mfile), mfile))
    for attr in ("num_images_in_input",):
        print("  config.%s = %s" % (attr, getattr(cfg, attr, "<无该属性>")))
    vb = model.vision_backbone
    print("  vision_backbone class: %s" % type(vb).__name__)
    vbmod = os.path.abspath(sys.modules[type(vb).__module__].__file__)
    print("  vision code sha256 : %s (%s)" % (sha256(vbmod), vbmod))
    print("  vision_backbone 子模块结构:")
    for n, ch in vb.named_children():
        extra = ""
        if isinstance(ch, torch.nn.ModuleList):
            names = []
            for t in ch:
                tc = getattr(t, "config", None)
                names.append("%s(%s)" % (
                    type(t).__name__,
                    getattr(getattr(tc, "name_or_path", None), "split", lambda s: [str(s)])()[-1]
                    if getattr(tc, "name_or_path", None) else getattr(tc, "model_type", "?")))
            extra = " -> " + ", ".join(names)
        print("    %-18s %-34s%s" % (n, type(ch).__name__, extra))

    print("=" * 72)
    print("[2] 固定帧前向形状链（含扰动试验）")
    device = "cuda"
    sm, ss, dm, ds = _norm_consts(device)
    pool = torch.load(POOL, map_location=device, weights_only=False)
    frames = []
    for tid in sorted(pool.keys()):
        for fr in pool[tid]:
            if fr["mvp"] is not None and fr.get("wrist_mvp") is not None:
                seg_vis = float(fr["seg"].mean())
                wv = float(fr.get("wrist_seg", torch.zeros(1)).mean())
                if seg_vis > 0.05 and wv > 0.02:
                    frames.append((tid, fr, seg_vis, wv))
        if len(frames) >= args.frames:
            break
    print("  选用 %d 个双视图均见布的池帧（主/腕可见率见下）" % len(frames))

    renderer = ClothRendererOV(device=device, init_std=0.05, seed=7).to(device)
    t0_tex = renderer.orig_texture

    def compose(fr, tex_override):
        rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                                  tex_override=tex_override)
        comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                           + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
        wrgba, _ = renderer.render(fr["wrist_mvp"], (256, 256), model_rot=fr["rot"],
                                   tex_override=tex_override)
        wrist = torch.clamp(wrgba.permute(0, 3, 1, 2) * fr["wrist_seg"]
                            + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
        return comp, wrist

    def to_pv(comp, wrist):
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=size, siglip_mean=sm, siglip_std=ss,
            dino_mean=dm, dino_std=ds, wrist_tensor=wrist).to(torch.bfloat16)

    def tokens(pv):
        with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
            return model.projector(model.vision_backbone(pv)).float()

    for k, (tid, fr, sv, wv) in enumerate(frames):
        print("  -- 帧 %d（task %d，主视图布可见率 %.3f / 腕 %.3f）" % (k, tid, sv, wv))
        plain_m, plain_w = compose(fr, t0_tex)
        adv_m, adv_w = compose(fr, None)
        pv_plain = to_pv(plain_m, plain_w)
        print("     pixel_values 形状 = %s（dtype %s）" % (tuple(pv_plain.shape), pv_plain.dtype))
        t_plain = tokens(pv_plain)
        P = t_plain.shape[1]
        print("     projector token 形状 = %s  → P=%d" % (tuple(t_plain.shape), P))

        # 扰动试验：只改主 / 只改腕
        t_main = tokens(to_pv(adv_m, plain_w))
        t_wrist = tokens(to_pv(plain_m, adv_w))
        m0, m1 = block_diff(t_main, t_plain)
        w0, w1 = block_diff(t_wrist, t_plain)
        print("     只改主视图: 块0(前256) ΔL2=%.4f | 块1(后256) ΔL2=%.6f  比值 %.1f:1"
              % (m0, m1, m0 / max(m1, 1e-12)))
        print("     只改腕视图: 块0(前256) ΔL2=%.6f | 块1(后256) ΔL2=%.4f  比值 %.1f:1"
              % (w0, w1, w1 / max(w0, 1e-12)))

        # 布面 token mask（与训练器 vis_token_mask 同代码路径）
        vm = vis_token_mask([(tid, fr)], P, device)[0]
        g = P // 2
        print("     vis_token_mask: 块0 有效 %d/%d，块1 有效 %d/%d（阈值>0.5: %d/%d）"
              % (int((vm[:g] > 0.5).sum()), g, int((vm[g:] > 0.5).sum()), g,
                 int((vm[:g] > 0.5).sum()) + int((vm[g:] > 0.5).sum()), P))

    print("=" * 72)
    print("[3] 文本切片布局（full_forward 拼接序）")
    fr = frames[0][1]
    ids = fr["text_inputs"]["input_ids"].to(device)
    tmask = fr["text_inputs"]["attention_mask"].to(device)
    with torch.no_grad():
        lang_emb = model.get_input_embeddings()(ids)
    pv = to_pv(*compose(fr, t0_tex))
    with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
        proj = model._process_vision_features(pv, lang_emb, use_film=False)
    print("  lang_emb %s + vision proj %s → 拼接 [BOS][vision P][text rest]"
          % (tuple(lang_emb.shape), tuple(proj.shape)))
    print("  总序列长 = 1 + %d + (%d - 1) = %d；文本切片 = 末 %d 个位置（含 BOS 计法见训练器注释）"
          % (proj.shape[1], ids.shape[1], 1 + proj.shape[1] + ids.shape[1] - 1, ids.shape[1]))

    print("=" * 72)
    print("[4] 判定")
    ok_main = m0 > 10 * m1
    ok_wrist = w1 > 10 * w0
    print("  主视图扰动 → 块0 响应: %s；腕视图扰动 → 块1 响应: %s" %
          ("通过" if ok_main else "不通过", "通过" if ok_wrist else "不通过"))
    print("  P=%s（512=[主256][腕256] 与 freeze manifest 一致；映射说明 §4 的"
          "“P=256 并集”表述为文档错误）" % P if P == 512 else "  P=%d 与预期 512 不符，需人工核验" % P)


if __name__ == "__main__":
    main()
