# -*- coding: utf-8 -*-
"""one_step.py — 阶段四：原生目标上的单步收益检验（Gate C，计划§7）。

各起点 × 各方法在固定诊断 batch（seed7 流前 8 帧同 UV）生成 sign 更新
（真实 clamp 投影），导出真实 PNG；全部候选用同一原生评价目标打分：
  S_N(T) = ‖f_N(Q(T)) − f_N(T0)‖²/m   （验证 split 状态，replay 到同状态）
  J_N^true = 10·E[S_N] − 0.1·R_amp − 0.1·TV（部署 PNG 域）
同步记录各方法自身合成目标变化，识别"合成提高、原生不动"。
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import adapters
from .adapters import (RendererQ, ResidualCtx, png_delta_stats, quad_to_dev,
                       render_native_dual, strict_inject)
from .residual_core import ResidualFilter
from .train_residual import active_frames, load_stream_pt

STEP_SIZES = [0.001, 0.005, 0.01, 0.05]


def native_eval_states(ctx, pngs, t0_png, split="validation", device="cuda"):
    """replay 验证 split 的 rollout 到各采样状态，逐 PNG 原生评价 S_N。
    返回 {png_id: [S_N per state]} + T0 基线 A/A。"""
    import attack_oft as base
    import attack_oft_cloth_tvfix_mr as M
    import transfer_pi2ov as TR
    from libero.libero import benchmark
    from .capture import SPLIT_CFG, TASKS, _policy_step, capture_quad
    from .adapters import make_inputs, proprio_of

    sc = SPLIT_CFG[split]
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    results = {pid: [] for pid in pngs}
    aa_info = []
    for task_id in TASKS:
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        env, desc = base.get_libero_env(task, ctx.cfg.model_family, resolution=256)
        env.reset()
        obs = env.set_init_state(inits[sc["init"]])
        sim = env.env.sim
        sim.forward()
        TR.align_cloth(env)
        _, bid, _ = base.get_target_model_matrix(env, TR.SEARCH)
        tex_id = TR.ctu.find_target_texture_id(sim, bid, TR.SEARCH)
        ri0 = strict_inject(sim, tex_id, t0_png)
        n_state = 0
        try:
            for step in range(400):
                if n_state >= sc["states_per_task"]:
                    break
                if step % sc["stride"] == 0:
                    quad = capture_quad(env, obs, bid, device)
                    inputs = make_inputs(ctx, quad["native_main_px"], desc)
                    proprio = proprio_of(obs) if ctx.use_proprio else None
                    ri = strict_inject(sim, tex_id, t0_png)
                    n0, w0 = render_native_dual(sim)
                    TR.ctu.restore_mujoco_texture(sim, tex_id, *ri)
                    fN0 = ctx.hidden_from_arrays(n0, w0, inputs,
                                                 proprio).flatten()
                    fN0b = ctx.hidden_from_arrays(n0, w0, inputs,
                                                  proprio).flatten()
                    aa_info.append(float((fN0b - fN0).norm()
                                         / max(fN0.norm(), 1e-12)))
                    for pid, png in pngs.items():
                        rix = strict_inject(sim, tex_id, png)
                        a, w = render_native_dual(sim)
                        TR.ctu.restore_mujoco_texture(sim, tex_id, *rix)
                        fN = ctx.hidden_from_arrays(a, w, inputs,
                                                    proprio).flatten()
                        results[pid].append(
                            float((fN - fN0).square().mean()))
                    n_state += 1
                obs = _policy_step(ctx, env, obs, desc)
                if obs is None:
                    break
        finally:
            TR.ctu.restore_mujoco_texture(sim, tex_id, *ri0)
            env.close()
    return results, aa_info


def run_one_step(cfg, out_root, bases_dir, k, beta, gamma,
                 starts, methods, step_sizes=None, device="cuda"):
    """starts: {name: theta_spec}；methods: ['matched','residual','random'(,'scalar')]"""
    step_sizes = step_sizes or STEP_SIZES
    from .train_residual import build_filter, sha256, load_stream_pt
    import attack_oft_cloth_ablate as ABL

    ctx = ResidualCtx(device=device)
    filt_by_method = {}
    for mth in methods:
        f_, w_ = build_filter(mth, bases_dir, k, beta, gamma, device)
        filt_by_method[mth] = (f_, w_)

    pool = torch.load(cfg.paths.source_pool, map_location="cpu",
                      weights_only=False)
    act, gids = active_frames(pool)
    full = torch.load(cfg.paths.matched_reference_cache, map_location="cpu",
                      weights_only=False)["hidden"]
    mr_hidden = [full[g] for g in gids]

    stream = load_stream_pt(cfg.paths.stream_file, 1)   # 只取首个诊断 batch
    batch_idx, uv_a = stream[0]
    uv_a = uv_a.to(device) if uv_a is not None else None

    t0_png = ctx.t0_png
    png_dir = os.path.join(out_root, "one_step_pngs")
    os.makedirs(png_dir, exist_ok=True)
    rows = []
    cand_pngs = {}
    try:
        for sname, theta_spec in starts.items():
            if theta_spec == "init":
                g0 = torch.Generator(device="cpu").manual_seed(7)
                ctx.load_theta(
                    torch.randn(ctx.renderer.adv_noise.shape,
                                generator=g0) * cfg.training.init_std)
            else:
                ctx.load_theta(theta_spec)
            th_start = ctx.renderer.adv_noise.detach().clone()
            png0 = ctx.renderer.export_png()
            cand_pngs[f"{sname}__start"] = png0
            torch.save(ctx.renderer.adv_noise.detach().cpu(),
                       os.path.join(png_dir, f"{sname}__start_theta.pt"))
            Image.fromarray(png0).save(
                os.path.join(png_dir, f"{sname}__start.png"))
            stats0 = png_delta_stats(png0, t0_png)
            for mth in methods:
                filt, scalar_w = filt_by_method[mth]
                param = ctx.renderer.adv_noise
                param.grad = None
                for fi in batch_idx:
                    fr = ABL.frame_to_dev(act[fi][1], device)
                    adv_rgba, _ = ctx.renderer.render(
                        fr["mvp"], (256, 256), model_rot=fr["rot"], uv_xform=uv_a)
                    comp = torch.clamp(
                        adv_rgba.permute(0, 3, 1, 2) * fr["seg"]
                        + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
                    adv_wrist_t = fr.get("wrist")
                    if fr.get("wrist_mvp") is not None:
                        w, _ = ctx.renderer.render(
                            fr["wrist_mvp"], (256, 256), model_rot=fr["rot"],
                            uv_xform=uv_a)
                        adv_wrist_t = torch.clamp(
                            w.permute(0, 3, 1, 2) * fr["wrist_seg"]
                            + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
                    pv = adapters.base._compose_oft_pixel_values(
                        adv_rgb=comp, model_input_size=ctx.size,
                        siglip_mean=ctx.sm, siglip_std=ctx.ss,
                        dino_mean=ctx.dm, dino_std=ctx.ds,
                        wrist_tensor=adv_wrist_t).to(torch.bfloat16)
                    with adapters.base.autocast(dtype=torch.bfloat16):
                        _, h = ctx.model.predict_action(
                            input_ids=fr["text_inputs"]["input_ids"],
                            attention_mask=fr["text_inputs"]["attention_mask"],
                            pixel_values=pv, unnorm_key=ctx.cfg.unnorm_key,
                            proprio=fr["proprio"], proprio_projector=ctx.pp,
                            action_head=ctx.ah, noisy_action_projector=ctx.nap,
                            use_film=getattr(ctx.cfg, "use_film", False))
                    _ref = mr_hidden[fi].to(device)
                    d = (h.float() - _ref.float()).reshape(-1)
                    if filt is not None:
                        d = filt(d.unsqueeze(0)).squeeze(0)
                    attack_feat = -(d.square().mean()) * scalar_w
                    (float(cfg.training.alpha_feature) * attack_feat / len(
                        batch_idx)).backward()
                tex = ctx.renderer.texture_tensor()
                ref = F.mse_loss(tex, ctx.t0_tex)
                tv = adapters.M.spatial_tv_nhwc(tex - ctx.t0_tex)
                (float(cfg.training.lambda_amplitude) * ref
                 + float(cfg.training.lambda_tv) * tv).backward()
                gsign = param.grad.sign().detach().clone()
                for eta in step_sizes:
                    with torch.no_grad():
                        th_new = torch.clamp(
                            ctx.renderer.adv_noise - eta * gsign, -5.0, 5.0)
                    ctx.load_theta(th_new)
                    png = ctx.renderer.export_png()
                    pid = f"{sname}__{mth}__eta{eta}"
                    cand_pngs[pid] = png
                    torch.save(th_new.cpu(),
                               os.path.join(png_dir, pid + "_theta.pt"))
                    Image.fromarray(png).save(
                        os.path.join(png_dir, pid + ".png"))
                    st = png_delta_stats(png, png0)
                    rows.append({
                        "start": sname, "method": mth, "eta": eta,
                        "png_changed_px": int((png != png0).any(axis=-1).sum()),
                        "png_rms_vs_start": st["rms"],
                        **{"start_" + kk: vv for kk, vv in stats0.items()}})
                ctx.load_theta(th_start)
        # ---- 原生统一评价 ----
        print("[one-step] 候选 PNG 数", len(cand_pngs), "→ 原生评价", flush=True)
        sN, aa = native_eval_states(ctx, cand_pngs, t0_png, device=device)
        for r in rows:
            pid = f"{r['start']}__{r['method']}__eta{r['eta']}"
            vals = sN[pid]
            r["S_N_mean"] = float(np.mean(vals))
            r["S_N_states"] = [float(v) for v in vals]
        for sname in starts:
            vals = sN[f"{sname}__start"]
            rows.append({"start": sname, "method": "start", "eta": 0.0,
                         "S_N_mean": float(np.mean(vals)),
                         "S_N_states": [float(v) for v in vals]})
        out = {"rows": rows, "aa_forward_rms": [float(a) for a in aa],
               "k": k, "beta": beta, "gamma": gamma, "methods": methods,
               "step_sizes": step_sizes, "starts": list(starts)}
        diag = os.path.join(out_root, "diagnostics")
        os.makedirs(diag, exist_ok=True)
        with open(os.path.join(diag, "one_step_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        return out
    finally:
        ctx.cleanup()
