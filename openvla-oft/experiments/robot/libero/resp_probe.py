#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""resp_probe.py — E 线探针②：源池 12 状态上 Plain/F0/B2v2/C1 的表征→动作响应（第12轮 §5.2）
每任务(0,2,6,8)×早/中/晚各 1 帧 = 12 状态；固定无 UV-EoT（可复现）。
对每状态每纹理测（相对 Plain）：
  SigLIP:  ||ΔS||/||S||（逐 token 平均）；ΔS 池化方向 d_S
  projector: ||ΔZ||/||Z||
  动作 chunk: ||Δa||/||a||（predict_action 归一化输出）
跨方法：cos(d_S(F0), d_S(B2v2)) 等两两方向一致性（SigLIP 与 projector 空间各一表）。
目标侧不用任何梯度/权重；全部只读前向。"""
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import attack_oft as base                     # noqa: E402
import attack_oft_cloth_edpa as E1            # noqa: E402
import attack_oft_cloth_edpa_v2 as E2         # noqa: E402
import transfer_pi2ov as TR                   # noqa: E402

EX = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth"
POOL = EX + "/oft_cloth/r0f_pool.pt"
TEXS = {
    "Plain": None,
    "F0": EX + "/texpush/f0_seed7.png",
    "B2v2": EX + "/oft_cloth/train_20261020_191751_edpav2/Universal_Cloth_OV.png",
    "C1": None,   # 训练完成后由 --c1-tex 传入
}
device = "cuda"


def load_tex(p):
    if p is None:
        return None
    t = torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).to(device).unsqueeze(0)
    return torch.flip(t, dims=[1])


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def pooled_delta(S_t, S_p):
    return (S_t - S_p).mean(dim=1).flatten()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-tex", default="")
    args = ap.parse_args()
    if args.c1_tex:
        TEXS["C1"] = args.c1_tex

    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=E2.CKPT, model_family="openvla")
    cfg.task_suite_name = "libero_spatial"
    model = base.get_model(cfg)
    base.check_unnorm_key(cfg, model)
    processor = base.get_processor(cfg)
    mis = base.get_image_resize_size(cfg)
    action_head, proprio_projector, noisy_action_projector = \
        E1._build_action_modules(cfg, model)
    sm, ss, dm, ds = E2._norm_consts(device)
    sig_tower = model.vision_backbone.fused_featurizer

    pool = torch.load(POOL, map_location=device, weights_only=False)
    states = []
    for tid in (0, 2, 6, 8):
        frs = [fr for fr in pool[tid] if fr["mvp"] is not None]
        n = len(frs)
        for ph, i in (("早", 0), ("中", n // 2), ("晚", n - 1)):
            states.append((tid, ph, frs[i]))
    print("[probe] %d 状态（4 任务 × 早/中/晚）" % len(states))

    renderer = E2.ClothRendererOV(device=device, epsilon=64.0 / 255.0,
                                  init_std=0.0, seed=7).to(device)
    t0_tex = renderer.orig_texture
    TEXS["Plain"] = ("t0", t0_tex)
    texs = {}
    for k, v in TEXS.items():
        if k == "Plain":
            texs[k] = t0_tex
        elif v and os.path.exists(v):
            texs[k] = load_tex(v)
        else:
            print("[probe] 跳过 %s（%s 不存在）" % (k, v))

    def compose(fr, tex):
        rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                                  tex_override=tex)
        comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                           + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
        wrist = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            wrgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                       model_rot=fr["rot"], tex_override=tex)
            wrist = torch.clamp(wrgba.permute(0, 3, 1, 2) * fr["wrist_seg"]
                                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
        return comp, wrist

    def to_pv(comp, wrist):
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=mis, siglip_mean=sm, siglip_std=ss,
            dino_mean=dm, dino_std=ds, wrist_tensor=wrist).to(torch.bfloat16)

    def sig_tokens(pv):
        outs = []
        for img in torch.split(pv, [6, 6], dim=1):
            _, fused = torch.split(img, [3, 3], dim=1)
            with torch.no_grad():
                outs.append(sig_tower(fused))
        return torch.cat(outs, dim=1).float()

    res = {k: {"S": [], "Z": [], "a": []} for k in texs if k != "Plain"}
    dS = {k: [] for k in texs if k != "Plain"}
    dZ = {k: [] for k in texs if k != "Plain"}
    for tid, ph, fr in states:
        with torch.no_grad():
            pv_p = to_pv(*compose(fr, t0_tex))
            S_p = sig_tokens(pv_p)
            Z_p = model.projector(model.vision_backbone(pv_p)).float()
            a_p, _ = model.predict_action(
                input_ids=fr["text_inputs"]["input_ids"],
                attention_mask=fr["text_inputs"]["attention_mask"],
                pixel_values=pv_p, unnorm_key=cfg.unnorm_key,
                proprio=fr.get("proprio"), proprio_projector=proprio_projector,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector)
            a_p = torch.as_tensor(np.asarray(a_p)).float()
        for k, tex in texs.items():
            if k == "Plain":
                continue
            with torch.no_grad():
                pv = to_pv(*compose(fr, tex))
                S = sig_tokens(pv)
                Z = model.projector(model.vision_backbone(pv)).float()
                a, _ = model.predict_action(
                    input_ids=fr["text_inputs"]["input_ids"],
                    attention_mask=fr["text_inputs"]["attention_mask"],
                    pixel_values=pv, unnorm_key=cfg.unnorm_key,
                    proprio=fr.get("proprio"), proprio_projector=proprio_projector,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector)
                a = torch.as_tensor(np.asarray(a)).float()
            res[k]["S"].append(rel(S, S_p))
            res[k]["Z"].append(rel(Z, Z_p))
            res[k]["a"].append(rel(a, a_p))
            dS[k].append(pooled_delta(S, S_p))
            dZ[k].append(pooled_delta(Z, Z_p))

    print("| 纹理 | SigLIP ‖ΔS‖/‖S‖ | projector ‖ΔZ‖/‖Z‖ | 动作 ‖Δa‖/‖a‖ |")
    print("|---|---:|---:|---:|")
    for k in res:
        f = lambda v: "—"
        print("| %s | %.4f（%.4f–%.4f） | %.4f（%.4f–%.4f） | %.4f（%.4f–%.4f） |" % (
            k,
            np.mean(res[k]["S"]), min(res[k]["S"]), max(res[k]["S"]),
            np.mean(res[k]["Z"]), min(res[k]["Z"]), max(res[k]["Z"]),
            np.mean(res[k]["a"]), min(res[k]["a"]), max(res[k]["a"])))

    ks = [k for k in res]
    print("\n跨方法 Δ方向一致性（12 状态平均 cos）：")
    print("| 空间 | " + " | ".join("%s·%s" % (a, b) for i, a in enumerate(ks)
                                    for b in ks[i + 1:]) + " |")
    for nm, dd in (("SigLIP d_S", dS), ("projector d_Z", dZ)):
        row = []
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                cs = [float(torch.nn.functional.cosine_similarity(dd[a][j], dd[b][j],
                                                                  dim=0))
                      for j in range(len(states))]
                row.append(np.mean(cs))
        print("| %s | " % nm + " | ".join("%.3f" % c for c in row) + " |")


if __name__ == "__main__":
    main()
