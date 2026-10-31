# -*- coding: utf-8 -*-
"""train_residual.py — 阶段五：受控训练（计划§8，方法身份分离）。

方法：matched（rank0≡生产 MR）/ residual（冻结 U 投影）/ random（同秩随机基）/
      scalar（matched×γ，仅当 γ<0.9 触发登记）。
循环与 attack_oft_cloth_ablate.py@train 逐行同数学：
  每帧 (α_feat·attack_feat)/batch 反传 → 正则一次 → sign 更新 → clamp±5。
attack_feat = −ResidualFilter.distance(h, ref)（k=0 时与 F.mse_loss 逐位相等）。
"""
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import adapters
from .adapters import RendererQ, ResidualCtx
from .residual_core import ResidualFilter
from .streams import load_stream_pt, load_stream_text

POOL_SHA = "d03d68591b9b820385d14be44758aefd95c2a63b8d68fc0926413a2eb1b04a93"
CACHE_SHA = "7ab459156068e23cd338bc33e100fb1113a3a367d8b020a80cdaf7cdb7249856"
CKPT_UPDATES = [0, 100, 250, 499, 500, 501, 1000, 2000, 3500, 5000]


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def active_frames(pool):
    task_ids = list(pool.keys())
    out, gids = [], []
    for fi, (tid, fr) in enumerate((t, f) for t in task_ids for f in pool[t]
                                   if f["mvp"] is not None):
        out.append((tid, fr))
        gids.append(fi)
    return out, gids


def build_filter(method, bases_dir, k, beta, gamma, device):
    if method == "matched":
        return None, 1.0
    if method == "scalar":
        assert gamma is not None and gamma > 0, "scalar 需 γ（fit 集冻结值）"
        return None, float(gamma)
    fn = {"residual": f"basis_k{k}.pt", "random": f"random_k{k}.pt"}[method]
    U = torch.load(os.path.join(bases_dir, fn), map_location="cpu",
                   weights_only=False)["U"]
    return ResidualFilter(U.to(device), beta), 1.0


def run_train(cfg, out_dir, method, seed=7, iters=5000, batch=8,
              stream_pt=None, stream_text=None, bases_dir=None,
              k=0, beta=1.0, gamma=None, dump_theta_iters=None,
              resume=False, device="cuda", log_every=50):
    """受控训练。stream 优先级：stream_pt > stream_text（二者需先验证一致）。"""
    import attack_oft_cloth_ablate as ABL

    assert sha256(cfg.paths.source_pool) == POOL_SHA
    assert sha256(cfg.paths.matched_reference_cache) == CACHE_SHA
    random.seed(seed)
    ctx = ResidualCtx(device=device, seed=seed)
    filt, scalar_w = build_filter(method, bases_dir, k, beta, gamma, device)
    if method in ("residual", "random"):
        assert filt is not None and filt.basis.shape[1] == k
    else:
        assert filt is None

    pool = torch.load(cfg.paths.source_pool, map_location="cpu",
                      weights_only=False)
    act, gids = active_frames(pool)
    full = torch.load(cfg.paths.matched_reference_cache, map_location="cpu",
                      weights_only=False)["hidden"]
    assert len(full) == 296
    mr_hidden = [full[g] for g in gids]
    n_active = len(act)

    if stream_pt:
        stream = load_stream_pt(stream_pt, iters)
    elif stream_text:
        stream = load_stream_text(stream_text, iters, batch)
    else:
        raise ValueError("需 stream_pt 或 stream_text（成对臂要求重放流）")

    # 生产 θ0：CPU generator(7)·std0.01（与 tvfix-mr / ablate 完全一致）
    g0 = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        ctx.renderer.adv_noise.copy_(
            torch.randn(ctx.renderer.adv_noise.shape, generator=g0
                        ).to(device) * cfg.training.init_std)
    param = ctx.renderer.adv_noise
    t0_tex = ctx.t0_tex

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "gradient_log.txt")
    torch.save(param.detach().cpu(), os.path.join(out_dir, "theta_updates000000.pt"))
    with open(os.path.join(out_dir, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(str(dict(method=method, seed=seed, iters=iters, batch=batch,
                         k=k, beta=beta, gamma=gamma,
                         stream_pt=stream_pt, stream_text=stream_text,
                         bases_dir=bases_dir,
                         entry="residual_lab.train_residual")) + "\n")
    samp_log = open(os.path.join(out_dir, "sampling_log.txt"), "w")
    samp_log.write("# it slot frame_idx(active) task uv00..uv12\n")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def save_ckpt(updates_completed, final=False):
        png = ctx.renderer.export_png()
        from PIL import Image
        Image.fromarray(png).save(os.path.join(
            out_dir, f"Universal_Cloth_OV_u{updates_completed:06d}.png"))
        torch.save(param.detach().cpu(), os.path.join(
            out_dir, f"theta_updates{updates_completed:06d}.pt"))
        if final:
            Image.fromarray(png).save(
                os.path.join(out_dir, "Universal_Cloth_OV.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
        log(f"[ckpt] updates_completed={updates_completed}")

    alpha_f = float(cfg.training.alpha_feature)
    lam_r = float(cfg.training.lambda_amplitude)
    lam_t = float(cfg.training.lambda_tv)
    lr = float(cfg.training.learning_rate)
    eps_b = float(cfg.training.parameter_bounds[1])
    dump = {}
    t_start = time.time()
    log(f"[train] {method} k={k} beta={beta} gamma={gamma} iters={iters} "
        f"stream={'pt' if stream_pt else 'text'} 开始")
    for it in range(iters):
        batch_idx, uv_a = stream[it]
        uv_a = uv_a.to(device) if uv_a is not None else None
        uvl = [float(x) for x in uv_a.cpu().numpy().reshape(-1)] \
            if uv_a is not None else [0.0] * 6
        param.grad = None
        for slot, fi in enumerate(batch_idx):
            samp_log.write("%d %d %d %d %s\n" % (
                it, slot, fi, act[fi][0], " ".join("%.6f" % x for x in uvl)))
            fr = ABL.frame_to_dev(act[fi][1], device)
            adv_rgba, _ = ctx.renderer.render(fr["mvp"], (256, 256),
                                              model_rot=fr["rot"], uv_xform=uv_a)
            comp = torch.clamp(
                adv_rgba.permute(0, 3, 1, 2) * fr["seg"]
                + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
            adv_wrist_t = fr.get("wrist")
            if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
                w, _ = ctx.renderer.render(fr["wrist_mvp"], (256, 256),
                                           model_rot=fr["rot"], uv_xform=uv_a)
                adv_wrist_t = torch.clamp(
                    w.permute(0, 3, 1, 2) * fr["wrist_seg"]
                    + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
            pv = adapters.base._compose_oft_pixel_values(
                adv_rgb=comp, model_input_size=ctx.size,
                siglip_mean=ctx.sm, siglip_std=ctx.ss,
                dino_mean=ctx.dm, dino_std=ctx.ds,
                wrist_tensor=adv_wrist_t).to(torch.bfloat16)
            with adapters.base.autocast(dtype=torch.bfloat16):
                _, adv_hidden = ctx.model.predict_action(
                    input_ids=fr["text_inputs"]["input_ids"],
                    attention_mask=fr["text_inputs"]["attention_mask"],
                    pixel_values=pv, unnorm_key=ctx.cfg.unnorm_key,
                    proprio=fr["proprio"], proprio_projector=ctx.pp,
                    action_head=ctx.ah, noisy_action_projector=ctx.nap,
                    use_film=getattr(ctx.cfg, "use_film", False))
            _ref = mr_hidden[fi].to(device)
            d = adv_hidden.float() - _ref.float()      # [1,56,4096]
            d_flat = d.reshape(-1)                      # [m]
            if filt is not None:
                d_f = filt(d_flat.unsqueeze(0)).squeeze(0)
                attack_feat = -(d_f.square().mean()) * scalar_w
                d_raw = float(F.mse_loss(adv_hidden.float(), _ref.float()))
            else:
                attack_feat = -(d_flat.square().mean()) * scalar_w
                d_raw = float(-attack_feat)
            (alpha_f * attack_feat / batch).backward()
        tex = ctx.renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = adapters.M.spatial_tv_nhwc(tex - t0_tex)
        (lam_r * ref + lam_t * tv).backward()
        with torch.no_grad():
            param.sub_(lr * param.grad.sign())
            param.clamp_(-eps_b, eps_b)
        if it % log_every == 0 or it == iters - 1:
            gn = float(param.grad.norm())
            assert np.isfinite(gn) and gn > 0, (it, gn)
            log(f"[train] it{it:05d} attack={alpha_f * float(attack_feat):+.5f} "
                f"D_raw={d_raw:.5f} ref={float(ref):.4f} tv={float(tv):.4f} "
                f"g={gn:.2e}")
        if dump_theta_iters and it in dump_theta_iters:
            dump[it] = param.detach().cpu().clone()
        if (it + 1) in CKPT_UPDATES or it == iters - 1:
            save_ckpt(it + 1, final=(it == iters - 1))
    samp_log.close()
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_final.pt"))
    delta = (torch.tanh(param.detach()) * cfg.training.epsilon) * 255.0
    tex = ctx.renderer.texture_tensor().detach()
    d255 = (tex - t0_tex).abs() * 255.0
    log(f"[final] deltaRMS={float(torch.sqrt((delta**2).mean())):.4f} "
        f"Linf={float(delta.abs().max()):.1f} "
        f"clipRMS={float(torch.sqrt((d255**2).mean())):.4f} "
        f"Linf={float(d255.max()):.1f} 用时={time.time()-t_start:.0f}s")
    ctx.cleanup()
    return dump
