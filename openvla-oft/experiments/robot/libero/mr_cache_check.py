#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mr_cache_check.py — 老师《第13轮评阅与完整执行计划_20261022》§6.3
MR 缓存一致性与源侧梯度分解（不新训、不用目标梯度、不改变任何产物）。

检查项（输出到 stdout 与 mr_cache_check.log）：
  A. θ=0 显式核验：MR 合成 Plain hidden（现场重算）vs 缓存参考 —— 逐帧
     rel/cos/MSE；同时对照池内原生 clean_hidden（=旧参考，即基差 b）。
     dtype/shape/eval 态/输入 pixel 一致性一并登记。
  B. 纯色 T0 的 UV 无关性：同一帧不同 UV（含镜像/旋转/平移）渲染 Plain
     逐像素一致 → 缓存按帧复用（训练时 UV 每步变）成立。
  C. 源侧梯度分解：固定帧 + 固定 UV，在 θ_start（seed7 随机初始化）与
     θ_final（MR 末步参数）两点，比较 gN=∇[−MSE(h,原生ref)] 与
     gR=∇[−MSE(h,缓存ref)]：范数、cos、sign 不一致比例、‖gN−gR‖。
     θ=0 处 gR 应≈0（匹配 MSE→0），cos 未定义留空（老师 §6.3 规定）。
用法（41，cwd=$D/code/tex3d-ws/g7，GPU2 空闲时）：
  CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python \
    $TRD/mr_cache_check.py --pool-file r0f_pool.pt \
    --mr-cache mr_plain_cache.pt --final-param <MR训练目录>/Ep0_Cloth_Delta_OV.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import attack_oft_cloth_tvfix_mr as M           # noqa: E402  同目录
import attack_oft as base                        # noqa: E402
import transfer_pi2ov as TR                      # noqa: E402

CKPT = "/data/shaozhen.liu/model/openvla-oft/ckpt_1280450_2025-04-22_04-35-28"
if hasattr(M, "CKPT") and getattr(M, "CKPT"):
    CKPT = M.CKPT


def sel_frames(pool):
    """按任务×轨迹分层取 16 帧：每任务 2 轨迹 × 早/晚各 1。"""
    idx = []
    for tid in list(pool.keys()):
        frs = [fr for fr in pool[tid] if fr["mvp"] is not None]
        n = len(frs)
        half = n // 2
        idx += [(tid, 0), (tid, max(1, half - 6)),
                (tid, half), (tid, n - 1)]
    return idx  # (task_id, 本任务内下标)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-file", required=True)
    ap.add_argument("--mr-cache", required=True)
    ap.add_argument("--final-param", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    was_clean = TR.patch_status() == "clean"
    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
    model = base.get_model(cfg)
    base.check_unnorm_key(cfg, model)
    model.eval()
    assert not model.training, "model 必须处于 eval 态"
    size = base.get_image_resize_size(cfg)
    action_head, proprio_projector, noisy_action_projector = \
        M._build_action_modules(cfg, model)
    sm, ss, dm, ds = M._norm_consts(device)

    pool = torch.load(args.pool_file, map_location=device, weights_only=False)
    task_ids = list(pool.keys())
    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]
                  if fr["mvp"] is not None]
    md = torch.load(args.mr_cache, map_location="cpu", weights_only=False)
    mr_hidden = md["hidden"]
    assert len(mr_hidden) == len(all_frames), (len(mr_hidden),
                                               len(all_frames))
    print("[check] 池 %d 帧 / 缓存 %d 帧（all_frames 顺序对齐）"
          % (len(all_frames), len(mr_hidden)), flush=True)

    renderer = M.ClothRendererOV(device=device, epsilon=64.0 / 255.0,
                                 init_std=0.01, seed=args.seed).to(device)
    t0_tex = renderer.orig_texture
    param = renderer.adv_noise
    # θ_start 备份后置零：F0 族 render 无 tex_override，θ=0 由参数置零实现
    # （与 mr_cache.py 构造方式一致：init_std=0 等效 θ=0）
    theta_start = param.detach().clone()
    with torch.no_grad():
        param.zero_()

    # 全局帧号映射：sel 返回 (tid, 局部下标) → 全局 fi
    base_off = {}
    off = 0
    for tid in task_ids:
        frs = [fr for fr in pool[tid] if fr["mvp"] is not None]
        base_off[tid] = off
        off += len(frs)
    sel = [(tid, li, base_off[tid] + li) for tid, li in sel_frames(pool)]
    print("[check] 分层选帧 %d 个: %s"
          % (len(sel), [(t, l) for t, l, _ in sel]), flush=True)

    def compose_plain_pv(fr, uv):
        plain_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                        model_rot=fr["rot"], uv_xform=uv)
        p_chw = plain_rgba.permute(0, 3, 1, 2)
        comp = torch.clamp(p_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                           0.0, 1.0)
        wrist = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            pw, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                    model_rot=fr["rot"], uv_xform=uv)
            pw_chw = pw.permute(0, 3, 1, 2)
            wrist = torch.clamp(pw_chw * fr["wrist_seg"]
                                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]),
                                0.0, 1.0)
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=size, siglip_mean=sm,
            siglip_std=ss, dino_mean=dm, dino_std=ds,
            wrist_tensor=wrist).to(torch.bfloat16)

    def adv_pv(fr, uv):
        adv_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                      model_rot=fr["rot"], uv_xform=uv)
        a_chw = adv_rgba.permute(0, 3, 1, 2)
        comp = torch.clamp(a_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                           0.0, 1.0)
        wrist = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            aw, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                    model_rot=fr["rot"], uv_xform=uv)
            aw_chw = aw.permute(0, 3, 1, 2)
            wrist = torch.clamp(aw_chw * fr["wrist_seg"]
                                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]),
                                0.0, 1.0)
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=size, siglip_mean=sm,
            siglip_std=ss, dino_mean=dm, dino_std=ds,
            wrist_tensor=wrist).to(torch.bfloat16)

    def run_hidden(pv):
        with base.autocast(dtype=torch.bfloat16):
            _, hid = model.predict_action(
                input_ids=fr["text_inputs"]["input_ids"],
                attention_mask=fr["text_inputs"]["attention_mask"],
                pixel_values=pv, unnorm_key=cfg.unnorm_key,
                proprio=fr["proprio"], proprio_projector=proprio_projector,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                use_film=getattr(cfg, "use_film", False))
        return hid.float()

    uv_fix = M.sample_uv_xform(True, device)  # 固定 UV（seed 决定）

    # ---------- A. θ=0 缓存一致性 ----------
    print("\n[A] θ=0 显式核验：现场合成 Plain hidden vs 缓存 vs 原生 clean_hidden",
          flush=True)
    print("| task | fi | shape/dtype | rel(现场,缓存) | cos | "
          "rel(缓存,原生) | cos | 输入一致 |", flush=True)
    print("|---|---|---|---:|---:|---:|---:|---|", flush=True)
    for tid, li, fi in sel:
        fr = all_frames[fi][1]
        with torch.no_grad():
            pv = compose_plain_pv(fr, None)
            h_syn = run_hidden(pv).detach()
            h_cache = mr_hidden[fi].to(device).float()
            pv2 = compose_plain_pv(fr, None)
            in_same = bool(torch.equal(pv, pv2))
            d1 = (h_syn - h_cache).norm() / (h_cache.norm() + 1e-12)
            c1 = F.cosine_similarity(h_syn.flatten().unsqueeze(0),
                                     h_cache.flatten().unsqueeze(0)).item()
            h_nat = fr["clean_hidden"].to(device).float()
            d2 = (h_cache - h_nat).norm() / (h_nat.norm() + 1e-12)
            c2 = F.cosine_similarity(h_cache.flatten().unsqueeze(0),
                                     h_nat.flatten().unsqueeze(0)).item()
        print("| %s | %d | %s/%s | %.3e | %.6f | %.3e | %.6f | %s |"
              % (tid, fi, tuple(h_cache.shape), mr_hidden[fi].dtype,
                 d1.item(), c1, d2.item(), c2, in_same), flush=True)

    # ---------- B. 纯色 T0 的 UV 无关性 ----------
    print("\n[B] 纯色 T0 × 5 种 UV（旋转/镜像/平移）→ Plain 渲染逐像素最大差",
          flush=True)
    fr = all_frames[sel[0][2]][1]
    ref_img = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                              uv_xform=None)[0]
    for k in range(5):
        uv = M.sample_uv_xform(True, device)
        img = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                              uv_xform=uv)[0]
        print("  UV#%d: max|Δ|=%.3e（<1e-5 判定 UV 无关，缓存可按帧复用）"
              % (k, (img - ref_img).abs().max().item()), flush=True)

    # ---------- C. 源侧梯度分解 ----------
    def grad_at(theta_name):
        print("\n[C@%s] gN(原生参考) vs gR(缓存参考)：逐帧范数/cos/sign不一致"
              % theta_name, flush=True)
        print("| task | fi | ‖gN‖ | ‖gR‖ | ‖gN−gR‖ | cos(gN,gR) | "
              "sign不一致率 |", flush=True)
        print("|---|---|---:|---:|---:|---:|---:|", flush=True)
        cos_list, sgn_list = [], []
        for tid, li, fi in sel[:8]:
            fr = all_frames[fi][1]
            pv = adv_pv(fr, uv_fix)
            h = run_hidden(pv)
            h_nat = fr["clean_hidden"].to(device).float()
            h_cache = mr_hidden[fi].to(device).float()
            gn = torch.autograd.grad((-F.mse_loss(h, h_nat)
                                      ).sum(), param,
                                     retain_graph=True)[0].flatten()
            gr = torch.autograd.grad((-F.mse_loss(h, h_cache)
                                      ).sum(), param)[0].flatten()
            del h
            cos = F.cosine_similarity(gn.unsqueeze(0),
                                      gr.unsqueeze(0)).item() \
                if gn.norm() > 0 and gr.norm() > 0 else float("nan")
            sgn = (torch.sign(gn) != torch.sign(gr)).float().mean().item()
            cos_list.append(cos)
            sgn_list.append(sgn)
            print("| %s | %d | %.3e | %.3e | %.3e | %s | %.4f |"
                  % (tid, fi, gn.norm().item(), gr.norm().item(),
                     (gn - gr).norm().item(),
                     ("%.6f" % cos) if cos == cos else "—(未定义)", sgn),
                  flush=True)
        print("  均值：cos=%.4f  sign不一致率=%.4f"
              % (float(np.nanmean(cos_list)), float(np.mean(sgn_list))),
              flush=True)

    # θ=0 处 gR 应≈0（匹配参考 → 梯度近零；cos 未定义，不填 0）
    # （参数自 A 段起保持置零 = 真实 θ=0，走 adv 可微路径渲染 = Plain）
    print("\n[C@θ=0] 匹配 MSE 与 gR 范数（预期：MSE≈0、‖gR‖≈0）", flush=True)
    for tid, li, fi in sel[:4]:
        fr = all_frames[fi][1]
        pv = adv_pv(fr, None)
        h = run_hidden(pv)
        h_cache = mr_hidden[fi].to(device).float()
        mse = F.mse_loss(h, h_cache).item()
        gr0 = torch.autograd.grad((-F.mse_loss(h, h_cache)).sum(),
                                  param)[0].norm().item()
        print("  fi=%d matched MSE=%.3e  ‖gR(θ=0)‖=%.3e" % (fi, mse, gr0),
              flush=True)
        del h
    with torch.no_grad():
        param.copy_(theta_start)

    grad_at("θ_start(seed7,init_std0.01)")
    final_delta = torch.load(args.final_param, map_location=device,
                             weights_only=False)
    with torch.no_grad():
        param.copy_(final_delta.to(device))
    print("  [info] θ_final 已载入 %s（RMS见下）"
          % os.path.basename(args.final_param), flush=True)
    tex = renderer.texture_tensor().detach()
    d = (tex - t0_tex) * 255.0
    print("  θ_final 纹理扰动 RMS=%.2f L∞=%.0f"
          % (float(torch.sqrt((d ** 2).mean())), float(d.abs().max())),
          flush=True)
    grad_at("θ_final(MR末步)")

    if was_clean:
        TR.ensure_patch(False)
    print("\n[check] 完成", flush=True)


if __name__ == "__main__":
    main()
