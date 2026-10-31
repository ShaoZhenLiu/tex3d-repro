# -*- coding: utf-8 -*-
"""capture.py — 阶段一/二：成对四路特征捕获（Gate A 数据 + 校准/验证/锁定测试集）。

同一停住状态内（不执行 env.step）依次：
  fN0      原生注入路由 @T0（主基线；obs 路由另存 fN0_obs 审计）
  fR0      合成路由 θ=0（float）
  fR0_Q    合成路由 θ=0 经真实 export→PNG 量化
  每纹理 ×3：fR_float(θ) / fR_Q(部署 PNG) / fN_Q(原生注入部署 PNG)
A/A：fN0 前向重复 5（固定图像）+ 完整注入路由重复 2；fR0 重复 2。
A→B→A：每任务首状态用高对比棋盘纹理验证注入/还原（双相机图像级）。
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from . import adapters
from .adapters import (ResidualCtx, capture_quad, make_inputs, proprio_of,
                       quad_to_dev, render_native_dual, strict_inject)

SPLIT_CFG = {
    "fit": {"init": 0, "states_per_task": 6, "stride": 15},
    "validation": {"init": 1, "states_per_task": 4, "stride": 20},
    "locked_test": {"init": 2, "states_per_task": 4, "stride": 20},
}
TASKS = [0, 2, 6, 8]


def _checker_png(size=256):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (((yy // 32) + (xx // 32)) % 2).astype(np.uint8)
    png = np.stack([c * 255, 255 - c * 255, (1 - c) * 255], axis=-1)
    return png.astype(np.uint8)


def _abab_check(sim, tex_id, tag, log):
    """高对比纹理 A→B→A：注入可见且还原干净（图像级，双相机）。"""
    base_imgs = render_native_dual(sim)
    ri = strict_inject(sim, tex_id, _checker_png())
    b_imgs = render_native_dual(sim)
    adapters.ctu.restore_mujoco_texture(sim, tex_id, *ri)
    a2_imgs = render_native_dual(sim)
    out = {}
    ok = True
    for name, (a, b, a2) in zip(("main", "wrist"),
                                zip(base_imgs, b_imgs, a2_imgs)):
        d_ab = float(np.abs(b.astype(np.int16) - a.astype(np.int16)).mean())
        d_ba = float(np.abs(a2.astype(np.int16) - a.astype(np.int16)).mean())
        out[name] = {"ab_meanpx": d_ab, "aba_meanpx": d_ba}
        ok = ok and d_ab > 5.0 and d_ba < 0.5
    out["pass"] = bool(ok)
    log(f"[abab] {tag}: {out}")
    if not ok:
        raise RuntimeError(f"A→B→A 注入验收失败 {tag}: {out}")
    return out


def run_capture(cfg, split, out_root, device="cuda"):
    assert split in SPLIT_CFG, split
    sc = SPLIT_CFG[split]
    tex_meta = json.load(open(os.path.join(out_root, "textures",
                                           "texture_table.json"),
                              encoding="utf-8"))
    texs = tex_meta["table"][split]
    t0_png = None

    import attack_oft as base
    import attack_oft_cloth_tvfix_mr as M
    import transfer_pi2ov as TR
    from libero.libero import benchmark

    ctx = ResidualCtx(device=device)
    t0_png = ctx.t0_png
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    pair_dir = os.path.join(out_root, "pairs", split)
    os.makedirs(pair_dir, exist_ok=True)
    log_path = os.path.join(pair_dir, "capture.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    records = []
    try:
        for task_id in TASKS:
            task = suite.get_task(task_id)
            inits = suite.get_task_init_states(task_id)
            env, task_description = base.get_libero_env(
                task, ctx.cfg.model_family, resolution=256)
            env.reset()
            obs = env.set_init_state(inits[sc["init"]])
            sim = env.env.sim
            sim.forward()
            TR.align_cloth(env)
            _, bid, _ = base.get_target_model_matrix(env, TR.SEARCH)
            tex_id = TR.ctu.find_target_texture_id(sim, bid, TR.SEARCH)
            ri0 = strict_inject(sim, tex_id, t0_png)
            n_state = 0
            abab = None
            try:
                for step in range(400):
                    if n_state >= sc["states_per_task"]:
                        break
                    if step % sc["stride"] == 0:
                        skey = f"t{task_id}_i{sc['init']}_s{step:03d}"
                        if abab is None:
                            abab = _abab_check(sim, tex_id, f"task{task_id}",
                                               log)
                        rec = _capture_state(
                            ctx, sim, obs, bid, tex_id, t0_png, texs,
                            task_id, sc["init"], step, skey, split,
                            task_description, log)
                        torch.save(rec, os.path.join(pair_dir, skey + ".pt"))
                        records.append({k: v for k, v in rec.items()
                                        if not torch.is_tensor(v)})
                        n_state += 1
                    obs = _policy_step(ctx, env, obs, task_description)
                    if obs is None:
                        break
            finally:
                TR.ctu.restore_mujoco_texture(sim, tex_id, *ri0)
                env.close()
            log(f"task{task_id}: {n_state} states captured")
    finally:
        ctx.cleanup()
    with open(os.path.join(pair_dir, "split_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"split": split, "split_cfg": sc, "records": records,
                   "textures": [{k: v for k, v in t.items()}
                                for t in texs]},
                  f, indent=1, default=str, ensure_ascii=False)
    log(f"[done] {split}: {len(records)} states × {len(texs)} textures")


def _capture_state(ctx, sim, obs, bid, tex_id, t0_png, texs,
                   task_id, init_idx, step, skey, split, task_description, log):
    device = ctx.device
    quad = capture_quad(env, obs, bid, device)
    quad_d = quad_to_dev(quad, device)
    inputs = make_inputs(ctx, quad["native_main_px"], task_description)
    proprio = proprio_of(obs) if ctx.use_proprio else None

    rec = {"record_id": skey, "split": split, "source_task_id": task_id,
           "source_init_index": init_idx, "policy_step": step,
           "seg_area": quad["seg_area"], "uv_id": "identity",
           "render_route": "native_inject+synthetic_nvdiffrast",
           "quantization_mode": "export_png_uint8", "precision": "bf16_autocast",
           "native_upload_ok": True, "status": "ok", "error_reason": "",
           "m": 229376, "timestamp": time.strftime("%F %T")}

    # ---- fN0：主基线（注入路由 @T0）+ obs 路由审计 ----
    ri = strict_inject(sim, tex_id, t0_png)
    n0, w0 = render_native_dual(sim)
    adapters.ctu.restore_mujoco_texture(sim, tex_id, *ri)
    fN0 = ctx.hidden_from_arrays(n0, w0, inputs, proprio).flatten().cpu()
    fN0_obs = ctx.hidden_from_arrays(
        quad["native_main_px"], quad["native_wrist_px"],
        inputs, proprio).flatten().cpu()
    rec["obs_vs_inject_pxgap"] = float(np.abs(
        n0.astype(np.int16) - quad["native_main_px"].astype(np.int16)).mean())

    # A/A：固定图像前向重复 5
    reps = [ctx.hidden_from_arrays(n0, w0, inputs, proprio).flatten().cpu()
            for _ in range(5)]
    rec["aa_forward_maxdiff"] = float(max(
        (r - fN0).abs().max() for r in reps))
    rec["aa_forward_rmsdiff"] = float(max(
        (r - fN0).norm() / max(fN0.norm(), 1e-12) for r in reps))
    # A/A：完整注入路由重复 2（渲染+前向）
    full_reps = []
    for _ in range(2):
        rix = strict_inject(sim, tex_id, t0_png)
        a, w = render_native_dual(sim)
        adapters.ctu.restore_mujoco_texture(sim, tex_id, *rix)
        full_reps.append(ctx.hidden_from_arrays(a, w, inputs,
                                                proprio).flatten().cpu())
    rec["aa_route_maxdiff"] = float(max(
        (r - fN0).abs().max() for r in full_reps))
    pxr = float(np.abs(a.astype(np.int16) - n0.astype(np.int16)).mean())
    rec["aa_route_pxgap"] = pxr

    # ---- fR0（θ=0 float）+ fR0_Q（θ=0 量化）----
    ctx.load_theta("zero")
    comp, wcomp = ctx.compose(quad_d)
    fR0 = ctx.hidden_from_tensors(comp, wcomp, inputs, proprio
                                  ).flatten().cpu()
    q0_png = ctx.renderer.export_png()
    fR0_Q = ctx.hidden_from_tensors(
        *ctx.compose(quad_d, tex_override=ctx.renderer.png_to_tex(q0_png)),
        inputs, proprio).flatten().cpu()
    rec["fR0_Q_vs_fR0_maxdiff"] = float((fR0_Q - fR0).abs().max())
    reps_r = [ctx.hidden_from_tensors(comp, wcomp, inputs, proprio
                                      ).flatten().cpu() for _ in range(2)]
    rec["aa_synth_maxdiff"] = float(max(
        (r - fR0).abs().max() for r in reps_r))

    rec["fN0"] = fN0
    rec["fN0_obs"] = fN0_obs
    rec["fR0"] = fR0
    rec["fR0_Q"] = fR0_Q

    # ---- 每纹理三路 ----
    for t in texs:
        th = torch.load(t["theta_path"], map_location="cpu",
                        weights_only=False)
        ctx.load_theta(th)
        png = ctx.renderer.export_png()
        tex_q = ctx.renderer.png_to_tex(png)
        fR_float = ctx.hidden_from_tensors(
            *ctx.compose(quad_d), inputs, proprio).flatten().cpu()
        fR_Q = ctx.hidden_from_tensors(
            *ctx.compose(quad_d, tex_override=tex_q),
            inputs, proprio).flatten().cpu()
        rix = strict_inject(sim, tex_id, png)
        a, w = render_native_dual(sim)
        adapters.ctu.restore_mujoco_texture(sim, tex_id, *rix)
        fN_Q = ctx.hidden_from_arrays(a, w, inputs, proprio).flatten().cpu()
        tid = t["texture_id"]
        rec[f"{tid}__fR_float"] = fR_float
        rec[f"{tid}__fR_Q"] = fR_Q
        rec[f"{tid}__fN_Q"] = fN_Q
        rec[f"{tid}__png_sha16"] = _sha16(png)
    ctx.load_theta("zero")

    # 四件套（离线诊断/单步复算用）
    rec["quad"] = {k: quad[k] for k in (
        "mvp", "rot", "seg", "bg", "wrist_seg", "wrist_bg", "wrist_mvp")}
    rec["inputs"] = {k: inputs[k].cpu() for k in
                     ("input_ids", "attention_mask")}
    rec["proprio"] = proprio
    rec["task_description"] = task_description
    log(f"{skey}: pxgap(obs/inj)={rec['obs_vs_inject_pxgap']:.2f} "
        f"aa_fwd={rec['aa_forward_maxdiff']:.2e} "
        f"aa_route={rec['aa_route_maxdiff']:.2e}")
    return rec


def _policy_step(ctx, env, obs, task_description):
    """源策略步进（与 t1_fix/生产采集逐行同链）。返回新 obs 或 None(done)。"""
    import attack_oft as base
    img_high = base.get_libero_image(obs)
    img_model = np.array(Image.fromarray(img_high).resize(
        (ctx.size, ctx.size)))
    observation = {"full_image": img_model}
    observation["state"] = proprio_of(obs)
    wnp_m = base._extract_wrist_image(obs, ctx.size)
    if wnp_m is not None:
        observation["wrist_image"] = wnp_m
    executed = base._query_action_with_compatible_signature(
        ctx.cfg, ctx.model, observation, task_description, ctx.processor,
        action_head=ctx.ah, proprio_projector=ctx.pp,
        noisy_action_projector=ctx.nap,
        use_film=getattr(ctx.cfg, "use_film", False))
    executed = np.asarray(executed[0] if isinstance(executed, list) else executed,
                          dtype=np.float32)
    executed = base.normalize_gripper_action(executed, binarize=True)
    if ctx.cfg.model_family == "openvla":
        executed = base.invert_gripper_action(executed)
    obs, _, done, _ = env.step(executed.tolist())
    if done or env.env._check_success():
        return None
    return obs


from .texutil import sha16 as _sha16
