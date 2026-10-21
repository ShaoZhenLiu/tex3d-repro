#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_oft_tvfix_diag.py — 第13轮阶段 A：正确 TV 的固定小批短诊断（老师 §7-A）
两条线：
  [reg-only] 固定 1 帧 + 固定 UV，只对 λ_ref·ref + λ_TV·TV 做短步下降（30 步）——
             验证正确 TV 可微、梯度有限、更新方向确实抑制 TV（老师：判断符号用纯正则短步）。
  [full]     固定 4 帧（每任务 1 帧）+ 各自固定 UV，完整 F0 目标
             （attack = 0×action + 10×hidden MSE，缓存参考）+ 0.1·ref + λ_TV·TV，
             梯度下降 200 步；记录分项 θ 梯度范数 / TV / RMS / L∞ / 纹理快照。
正确 TV 定义（老师 §5.3，NHWC 两方向均值和）：
  L_TV = mean|δ[:,1:]-δ[:,:-1]| + mean|δ[:,:,1:]-δ[:,:,:-1]|
复用 F0 生产组件（attack_oft_cloth.ClothRendererOV 等），仅换 TV 函数与固定输入。"""
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
import attack_oft as base                                   # noqa: E402
import transfer_pi2ov as TR                                 # noqa: E402
import attack_oft_cloth as AC                               # noqa: E402
from attack_oft_cloth import (CKPT, ClothRendererOV,        # noqa: E402
                              _build_action_modules, _norm_consts,
                              sample_uv_xform)

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth"


def spatial_tv_nhwc(delta):
    """老师 §5.3 统一定义（NHWC，方向均值之和，0-1 域）。"""
    if delta.ndim != 4:
        raise ValueError("Expected NHWC tensor")
    if delta.shape[1] < 2 or delta.shape[2] < 2:
        raise ValueError("Both spatial dimensions must be >= 2")
    delta = delta.float()
    dy = delta[:, 1:, :, :] - delta[:, :-1, :, :]
    dx = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    return dy.abs().mean() + dx.abs().mean()


def stats(tex, t0):
    with torch.no_grad():
        d = (tex - t0) * 255.0
        tv = spatial_tv_nhwc(tex - t0).item() * 255.0
        return (float(torch.sqrt((d ** 2).mean())), float(d.abs().max()), tv)


def compose(renderer, fr, uv, size, sm, ss, dm, ds):
    rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                              uv_xform=uv)
    comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                       + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
    wrist = fr.get("wrist")
    if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
        w, _ = renderer.render(fr["wrist_mvp"], (256, 256), model_rot=fr["rot"],
                               uv_xform=uv)
        wrist = torch.clamp(w.permute(0, 3, 1, 2) * fr["wrist_seg"]
                            + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
    return base._compose_oft_pixel_values(
        adv_rgb=comp, model_input_size=size, siglip_mean=sm, siglip_std=ss,
        dino_mean=dm, dino_std=ds, wrist_tensor=wrist).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-file", default=POOL)
    ap.add_argument("--lambdas", default="0.1,1.0")
    ap.add_argument("--full-iters", type=int, default=200)
    ap.add_argument("--reg-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = "cuda"

    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
    model = base.get_model(cfg)
    base.check_unnorm_key(cfg, model)
    size = base.get_image_resize_size(cfg)
    ah, pp, nap = _build_action_modules(cfg, model)
    sm, ss, dm, ds = _norm_consts(device)
    pool = torch.load(args.pool_file, map_location=device, weights_only=False)

    stamp = args.lambdas.replace(",", "-")
    out_dir = os.path.join(OUT_ROOT, f"train_diag_tvfix_{stamp}_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "diag_log.txt"), "w")

    def w(line):
        print(line, flush=True)
        log.write(line + "\n")

    w(f"[tvfix-diag] 正确 TV（spatial_tv_nhwc, NHWC）| λ={args.lambdas} | "
      f"full {args.full_iters} iters / reg-only {args.reg_steps} 步")

    # 固定输入：4 任务各 1 帧中部 + 各自固定 UV（seed 派生，登记）
    g = torch.Generator().manual_seed(args.seed)
    fixed = []
    for tid in sorted(pool.keys()):
        frs = [f for f in pool[tid] if f["mvp"] is not None]
        fr = frs[len(frs) // 2]
        torch.manual_seed(int(torch.randint(0, 2**31, (1,), generator=g)))
        uv = sample_uv_xform(True, device)
        fixed.append((tid, fr, uv))
    w("[tvfix-diag] 固定输入: tasks=%s（每任务中部 1 帧，UV 固定）"
      % sorted(pool.keys()))

    for lam in [float(x) for x in args.lambdas.split(",")]:
        renderer = ClothRendererOV(device=device, epsilon=64.0 / 255.0,
                                   init_std=0.01, seed=args.seed).to(device)
        t0 = renderer.orig_texture
        param = renderer.adv_noise

        # ---------- [reg-only] 纯正则短步 ----------
        tid, fr, uv = fixed[0]
        w(f"---- λ_TV={lam} [reg-only] 固定帧 task{tid}，只降 ref+λ·TV，"
          f"{args.reg_steps} 步 ----")
        hist = []
        for st in range(args.reg_steps):
            param.grad = None
            tex = renderer.texture_tensor()
            ref = F.mse_loss(tex, t0)
            tv = spatial_tv_nhwc(tex - t0)
            (0.1 * ref + lam * tv).backward()
            with torch.no_grad():
                param.sub_(0.05 * param.grad.sign())
                param.clamp_(-5.0, 5.0)
            if st % 5 == 0 or st == args.reg_steps - 1:
                rms, linf, tvv = stats(renderer.texture_tensor(), t0)
                hist.append((st, tvv))
                w(f"  step {st:03d}  TV={tvv:.4f}(0-255口径)  RMS={rms:.2f} "
                  f"L∞={linf:.0f}")
        ok_mono = all(hist[i + 1][1] <= hist[i][1] + 1e-6
                      for i in range(len(hist) - 1))
        w(f"  [reg-only 判定] TV 序列{'单调不增 ✓' if ok_mono else '非单调 ✗'}"
          f"（{hist[0][1]:.3f} → {hist[-1][1]:.3f}）")

        # ---------- [full] 完整目标短诊断 ----------
        w(f"---- λ_TV={lam} [full] F0 目标 + 正确TV，{args.full_iters} iters ----")
        for it in range(args.full_iters):
            tid, fr, uv = fixed[it % len(fixed)]
            param.grad = None
            pv = compose(renderer, fr, uv, size, sm, ss, dm, ds)
            with base.autocast(dtype=torch.bfloat16):
                _, adv_hidden = model.predict_action(
                    input_ids=fr["text_inputs"]["input_ids"],
                    attention_mask=fr["text_inputs"]["attention_mask"],
                    pixel_values=pv, unnorm_key=cfg.unnorm_key,
                    proprio=fr["proprio"], proprio_projector=pp,
                    action_head=ah, noisy_action_projector=nap,
                    use_film=getattr(cfg, "use_film", False))
            attack_feat = -F.mse_loss(adv_hidden.float(),
                                      fr["clean_hidden"].float())
            attack = 0.0 * attack_feat + 10.0 * attack_feat   # α_act=0, α_feat=10
            tex = renderer.texture_tensor()
            ref = F.mse_loss(tex, t0)
            tv = spatial_tv_nhwc(tex - t0)
            if it in (0, args.full_iters // 2, args.full_iters - 1):
                ga = torch.autograd.grad(attack, param, retain_graph=True)[0]
                gr = torch.autograd.grad(0.1 * ref, param, retain_graph=True)[0]
                gt = torch.autograd.grad(lam * tv, param, retain_graph=True)[0]
                w(f"  it{it:03d} 分项θ梯度: attack={ga.norm().item():.3e} "
                  f"ref={gr.norm().item():.3e} λtv·TV={gt.norm().item():.3e} "
                  f"(λ={lam}) | 有限={bool(torch.isfinite(gt).all())}")
            obj = attack + 0.1 * ref + lam * tv
            obj.backward()
            with torch.no_grad():
                param.sub_(0.05 * param.grad.sign())
                param.clamp_(-5.0, 5.0)
            if it % 40 == 0 or it == args.full_iters - 1:
                rms, linf, tvv = stats(renderer.texture_tensor(), t0)
                w(f"  it{it:03d} attack={attack.item():+.4f} ref={ref.item():.4f} "
                  f"TV={tvv:.3f} RMS={rms:.2f} L∞={linf:.0f}")
        Image.fromarray(renderer.export_texture()).save(
            os.path.join(out_dir, f"tex_lam{lam}.png"))
        w(f"  [full 完成] 导出 tex_lam{lam}.png "
          f"(stats rms/linf/tv = {stats(renderer.texture_tensor(), t0)})")

    log.close()
    print(f"[tvfix-diag] 完成 → {out_dir}")


if __name__ == "__main__":
    main()
