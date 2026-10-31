#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_cloth_instrumented.py — 阶段⑤仪表化评测（全程暴露统计 + 连续运动轨迹）
复用 attack_pi0_cloth 部署版的回路与助手（env/注入/对齐/原生分割 mask）。
模式：
  exposure —— Plain 条件完整 rollout；每次策略查询记录双相机桌布可见像素占比 +
              任务物体可见占比（MuJoCo 原生分割，geom 级；失败回退 RGB 差分阈值10，披露）。
  traj     —— 三条件（plain/mr/random）连续轨迹：逐控制步记录 时间/env步/eef位姿/夹爪/
              任务物体位姿/action chunk 与执行索引/success；每次查询存双相机帧。
预登记（跑前写入 manifest，不看结果）：
  exposure: 4 suite × 10 任务 × init {10,11,12} = 120 回合/策略
  traj: 6 键 = (spatial,t0,i10) (spatial,t2,i10) (spatial,t0,i12) (goal,t5,i10) (goal,t7,i10) (goal,t5,i12) × 3 条件 = 18 回合/策略
用法（run_cloth330.sh 同环境）：
  python eval_cloth_instrumented.py exposure --ckpt <pi05> --out <dir> --limit-suites spatial,object,goal,10
  python eval_cloth_instrumented.py traj --ckpt <pi05> --out <dir> \
      --tex-mr <mr.png> --tex-random <s101.png>
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

import attack_pi0_cloth as APC

CAMERAS = ["agentview", "robot0_eye_in_hand"]
TRAJ_KEYS = [("spatial", 0, 10), ("spatial", 2, 10), ("spatial", 0, 12),
             ("goal", 5, 10), ("goal", 7, 10), ("goal", 5, 12)]
EXPOSURE_INITS = [10, 11, 12]
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "10": "libero_10"}


def geom_ids_of_body(sim, body_id):
    return [g for g in range(sim.model.ngeom)
            if int(sim.model.geom_bodyid[g]) == int(body_id)]


def object_of_interest_bodies(env, task):
    """任务物体 body id 列表：优先 task.object_of_interest 名称匹配；
    失败回退=所有非 robot0/非桌布/非桌面固定体且有 geom 的 body（口径写入记录）。"""
    sim = env.env.sim
    fallback = []
    for b in range(sim.model.nbody):
        try:
            name = sim.model.body_id2name(b) or ""
        except Exception:
            name = ""
        if (name.startswith("robot0") or name == "tablecloth"
                or name in ("table", "floor", "world", "root")
                or int(sim.model.body_dofnum[b]) <= 0):
            continue
        if geom_ids_of_body(sim, b):
            fallback.append(b)
    names = getattr(task, "object_of_interest", None)
    if names:
        if isinstance(names, str):
            names = [names]
        ids = []
        for nm in names:
            for b in range(sim.model.nbody):
                try:
                    bn = sim.model.body_id2name(b) or ""
                except Exception:
                    continue
                if nm in bn:
                    ids.append(b)
                    break
        if ids:
            return ids, "task_object_of_interest"
    return fallback, "all_movable_bodies_fallback"


def visible_fraction(env, body_ids, camera, res=256):
    sim = env.env.sim
    try:
        seg = APC._render_seg(env, res, camera)
    except Exception:
        seg = None
    if seg is not None:
        ids = set()
        for b in body_ids:
            ids.update(geom_ids_of_body(sim, b))
        if seg.ndim == 3:
            seg = seg[..., 0]
        m = np.isin(seg, list(ids)) if ids else np.zeros_like(seg, bool)
        return float(m.mean()), "segmentation"
    # 回退：alpha 隐藏差分（阈值同历史口径）
    hidden = []
    for b in body_ids:
        for g in geom_ids_of_body(sim, b):
            hidden.append((g, sim.model.geom_rgba[g].copy()))
            sim.model.geom_rgba[g, 3] = 0.0
    try:
        sim.forward()
        rgb_h = APC._render_now(env, res, camera)
    finally:
        for g, rgba in hidden:
            sim.model.geom_rgba[g] = rgba
        sim.forward()
    rgb_f = APC._render_now(env, res, camera)
    diff = np.abs(rgb_f.astype(np.int16) - rgb_h.astype(np.int16))
    return float((diff.max(axis=-1) > 10).mean()), "rgb_diff_th10"


def setup_episode(task, inits, init_idx, texture_path):
    tex = np.array(Image.open(texture_path).convert("RGB"))
    env, _ = APC._get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(inits[init_idx])
    sim = env.env.sim
    sim.forward()
    # Match the production evaluator's Object floor dispatch.
    floor_mode = task.problem_folder == "libero_object"
    APC.align_cloth_to_table(env, floor_mode=floor_mode)
    _, cloth_bid, _ = APC.get_target_model_matrix(env, APC.CLOTH_SEARCH)
    tex_id = APC.find_target_texture_id(sim, cloth_bid, APC.CLOTH_SEARCH)
    assert tex_id >= 0, "桌布纹理 id 未找到"
    restore = APC.inject_mujoco_texture(sim, tex_id, tex)
    obs = env.set_init_state(inits[init_idx])
    return env, sim, cloth_bid, restore, tex_id, obs


def run_episode(mode, policy, task, inits, init_idx, texture_path, out_rec,
                max_steps=400, num_steps_wait=10, replan_steps=5,
                save_frames=False, frames_dir=None):
    env, sim, cloth_bid, restore, tex_id, obs = setup_episode(
        task, inits, init_idx, texture_path)
    desc = task.language
    obj_bids, obj_scheme = object_of_interest_bodies(env, task)
    rec = {"mode": mode, "desc": desc, "init_index": init_idx,
           "object_scheme": obj_scheme, "queries": [], "steps": [],
           "texture": os.path.basename(texture_path)}
    success = False
    plan = deque()
    chunk = None
    chunk_idx = -1
    t0 = time.time()
    try:
        for step in range(max_steps + num_steps_wait):
            if step < num_steps_wait:
                obs, _, done, _ = env.step(APC.get_libero_dummy_action())
                if done or env.env._check_success():
                    success = True
                    break
                continue
            if not plan:
                # —— 策略查询时刻：记录 mask / 帧 ——
                q = {"step": step, "wall": round(time.time() - t0, 3)}
                for cam in CAMERAS:
                    try:
                        cf, scheme = visible_fraction(env, [cloth_bid], cam)
                    except Exception:
                        cf, scheme = float("nan"), "error"
                    try:
                        of, _ = visible_fraction(env, obj_bids, cam)
                    except Exception:
                        of = float("nan")
                    q["cloth_frac_%s" % cam] = round(cf, 5)
                    q["obj_frac_%s" % cam] = round(of, 5)
                    q["mask_scheme"] = scheme
                    if save_frames and frames_dir:
                        fr = APC._render_now(env, 256, cam)
                        Image.fromarray(fr).save(os.path.join(
                            frames_dir, "s%04d_%s.jpg" % (step, cam)))
                rec["queries"].append(q)
                raw = APC.build_raw_inputs(obs, desc)
                with torch.no_grad():
                    result = policy.infer(raw)
                chunk = result["actions"] if isinstance(result, dict) \
                    else result
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                plan.extend(chunk[:min(replan_steps, len(chunk))])
                chunk_idx = 0
            action = np.asarray(plan.popleft(), dtype=np.float32)
            if mode == "traj":
                srec = {"step": step, "wall": round(time.time() - t0, 3),
                        "chunk_idx": chunk_idx,
                        "chunk_len": int(chunk.shape[0]) if chunk is not None else 0,
                        "action": [round(float(x), 5) for x in action]}
                try:
                    srec["eef_pos"] = [round(float(x), 5)
                                       for x in obs["robot0_eef_pos"]]
                    srec["eef_quat"] = [round(float(x), 5)
                                        for x in obs["robot0_eef_quat"]]
                    srec["gripper"] = [round(float(x), 5)
                                       for x in obs["robot0_gripper_qpos"]]
                except Exception:
                    pass
                opos = []
                for b in obj_bids:
                    opos.append([round(float(x), 5) for x in sim.data.xpos[b]])
                srec["obj_pos"] = opos
                rec["steps"].append(srec)
                chunk_idx += 1
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success():
                success = True
                break
    finally:
        APC.restore_mujoco_texture(sim, tex_id, *restore)
        APC._close_libero_env(env)
    rec["success"] = bool(success)
    rec["n_queries"] = len(rec["queries"])
    qs = [q for q in rec["queries"]]
    for cam in CAMERAS:
        vals = [q["cloth_frac_%s" % cam] for q in qs
                if q.get("cloth_frac_%s" % cam) == q.get("cloth_frac_%s" % cam)]
        if vals:
            rec["cloth_frac_median_%s" % cam] = round(float(np.median(vals)), 5)
    with open(out_rec, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exposure", "traj"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", default="pi05_libero")
    ap.add_argument("--num-denoise", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex-plain", default=APC.CLOTH_T0)
    ap.add_argument("--tex-mr", default="")
    ap.add_argument("--tex-random", default="")
    ap.add_argument("--target-name", default="pi05")
    ap.add_argument("--limit-suites", default="spatial,object,goal,10")
    ap.add_argument("--limit-episodes", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    policy = APC.load_policy(args.ckpt, model_config=args.model_config,
                             num_denoise=args.num_denoise)
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()

    manifest = {"mode": args.mode, "target": args.target_name,
                "ckpt": args.ckpt, "num_denoise": args.num_denoise,
                "tex_plain": args.tex_plain,
                "tex_mr": args.tex_mr, "tex_random": args.tex_random,
                "registered_before_run": True}
    if args.mode == "exposure":
        manifest["protocol"] = ("4 suites x 10 tasks x inits 10/11/12, "
                                "Plain condition, per-query masks")
    else:
        manifest["protocol"] = ("6 pre-registered keys x 3 conditions "
                                "(plain/mr/random): " +
                                str(TRAJ_KEYS))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    summary = []
    n = 0
    if args.mode == "exposure":
        APC.ensure_patch(True)
        for sk in args.limit_suites.split(","):
            suite = bench[SUITES[sk]]()
            for tid in range(10):
                task = suite.get_task(tid)
                inits = suite.get_task_init_states(tid)
                for ii in EXPOSURE_INITS:
                    out_rec = os.path.join(
                        args.out, "exp_%s_t%02d_i%d.json" % (sk, tid, ii))
                    if os.path.exists(out_rec):
                        continue
                    rec = run_episode("exposure", policy, task, inits, ii,
                                      args.tex_plain, out_rec)
                    summary.append((out_rec, rec["success"],
                                    rec.get("n_queries", 0)))
                    n += 1
                    print("[exp] %s t%d i%d success=%s queries=%d" %
                          (sk, tid, ii, rec["success"], rec["n_queries"]),
                          flush=True)
                    if args.limit_episodes and n >= args.limit_episodes:
                        return
    else:
        assert args.tex_mr and args.tex_random, "traj 需 --tex-mr/--tex-random"
        APC.ensure_patch(True)
        conds = [("plain", args.tex_plain), ("mr", args.tex_mr),
                 ("random", args.tex_random)]
        for sk, tid, ii in TRAJ_KEYS:
            suite = bench[SUITES[sk]]()
            task = suite.get_task(tid)
            inits = suite.get_task_init_states(tid)
            for cname, tex in conds:
                fd = os.path.join(args.out, "frames_%s_t%02d_i%d_%s"
                                  % (sk, tid, ii, cname))
                os.makedirs(fd, exist_ok=True)
                out_rec = os.path.join(args.out,
                                       "traj_%s_t%02d_i%d_%s.json"
                                       % (sk, tid, ii, cname))
                if os.path.exists(out_rec):
                    continue
                rec = run_episode("traj", policy, task, inits, ii, tex,
                                  out_rec, save_frames=True, frames_dir=fd)
                summary.append((out_rec, rec["success"], rec["n_queries"]))
                print("[traj] %s t%d i%d %s success=%s" %
                      (sk, tid, ii, cname, rec["success"]), flush=True)
    with open(os.path.join(args.out, "summary.jsonl"), "a") as f:
        for r, s, q in summary:
            f.write(json.dumps({"file": r, "success": s,
                                "queries": q}) + "\n")


if __name__ == "__main__":
    main()
