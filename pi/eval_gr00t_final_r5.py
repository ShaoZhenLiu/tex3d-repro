#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_gr00t_final.py — GR00T 桌布评测（v4：flat→nested 观测转换 + LiberoEnv 格式）
"""
import argparse, math, os, sys, time
from collections import deque
import numpy as np
import torch
from PIL import Image

D = "/data/shaozhen.liu"
sys.path.insert(0, os.path.join(D, "code/isaac-gr00t"))
sys.path.insert(0, os.path.join(D, "code/isaac-gr00t/src"))
CLOTH_T0 = os.path.join(D, "code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth/cloth_base.png")


def quat2aa(q):
    if q[3] > 1: q[3] = 1
    elif q[3] < -1: q[3] = -1
    den = np.sqrt(1 - q[3] * q[3])
    if math.isclose(den, 0): return np.zeros(3)
    return (q[:3] * 2 * math.acos(q[3])) / den


def make_gr00t_obs(obs, desc):
    """LIBERO 原始 obs → GR00T 嵌套 batched 观测（匹配 LiberoEnv._process_observation + batch）"""
    img = obs["agentview_image"][::-1, ::-1].copy()
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
    xyz = obs["robot0_eef_pos"]
    aa = quat2aa(obs["robot0_eef_quat"])
    grip = obs["robot0_gripper_qpos"]

    return {
        "video": {
            "image": torch.from_numpy(img).unsqueeze(0).unsqueeze(0),        # (1,1,H,W,C)
            "wrist_image": torch.from_numpy(wrist).unsqueeze(0).unsqueeze(0), # (1,1,H,W,C)
        },
        "state": {
            "x": torch.tensor([[[xyz[0]]]], dtype=torch.float32),    # (1,1,1)
            "y": torch.tensor([[[xyz[1]]]], dtype=torch.float32),
            "z": torch.tensor([[[xyz[2]]]], dtype=torch.float32),
            "roll": torch.tensor([[[aa[0]]]], dtype=torch.float32),
            "pitch": torch.tensor([[[aa[1]]]], dtype=torch.float32),
            "yaw": torch.tensor([[[aa[2]]]], dtype=torch.float32),
            "gripper": torch.tensor([[grip.tolist()]], dtype=torch.float32),  # (1,1,2)
        },
        "language": {
            "annotation.human.action.task_description": [desc],
        },
    }


def extract_action(actions):
    """GR00T 动作 dict → 7 维向量（匹配 LiberoEnv.step 的拼接+夹爪变换）"""
    act = np.concatenate([
        np.atleast_1d(actions["x"].flatten()),
        np.atleast_1d(actions["y"].flatten()),
        np.atleast_1d(actions["z"].flatten()),
        np.atleast_1d(actions["roll"].flatten()),
        np.atleast_1d(actions["pitch"].flatten()),
        np.atleast_1d(actions["yaw"].flatten()),
        np.atleast_1d(actions["gripper"].flatten()),
    ])[:7].astype(np.float64)
    # 夹爪：[0,1] → [-1,+1] → 取反 → 二值化
    act[-1] = 2.0 * act[-1] - 1.0
    act[-1] = -act[-1]
    act[-1] = np.sign(act[-1])
    return act


def _get_env(task, resolution=256, seed=7):
    from pathlib import Path
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _find_cloth_texid(sim, bid):
    """MuJoCo 3.x: geom→material→texture（attack_pi0.py 生产同款路径）"""
    for g in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[g]) == int(bid):
            try:
                mat = int(sim.model.geom_matid[g])
            except Exception:
                mat = -1
            if mat < 0:
                continue
            tids = np.asarray(sim.model.mat_texid[mat]).reshape(-1)
            tids = tids[tids >= 0]
            if len(tids):
                return int(tids[0])
    return -1


def _tex_storage(sim):
    m = getattr(sim.model, "_model", sim.model)
    s = getattr(m, "tex_data", getattr(m, "tex_rgb", None))
    if s is None:
        raise RuntimeError("MuJoCo model exposes neither tex_data nor tex_rgb")
    return m, s


def _upload_texture(sim, tid):
    """把 CPU 侧纹理刷进所有 GL 渲染上下文（否则渲染器沿用旧 GPU 纹理）"""
    import mujoco
    ctxs = []
    for attr in ("render_contexts", "_render_context_offscreen", "render_context_offscreen"):
        v = getattr(sim, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            ctxs.extend(v)
        else:
            ctxs.append(v)
    seen, ok = set(), False
    for c in ctxs:
        if id(c) in seen:
            continue
        seen.add(id(c))
        con = getattr(c, "con", None)
        cm = getattr(c, "model", None)
        if con is None or cm is None:
            continue
        gl = getattr(c, "gl_ctx", None)
        if gl is not None:
            gl.make_current()
        mujoco.mjr_uploadTexture(getattr(cm, "_model", cm), con, tid)
        ok = True
    return ok


def inject_cloth(sim, texture_path):
    if texture_path is None: return None
    bid = -1
    for b in range(sim.model.nbody):
        nm = sim.model.body_id2name(b) or ""
        if "tablecloth" in nm: bid = b; break
    if bid < 0:
        raise RuntimeError("tablecloth body not found in scene")
    tid = _find_cloth_texid(sim, bid)
    if tid < 0:
        raise RuntimeError("cloth texture id not found via geom_matid/mat_texid")
    m, storage = _tex_storage(sim)
    th = int(m.tex_height[tid]); tw = int(m.tex_width[tid])
    adr = int(m.tex_adr[tid])
    nch = int(m.tex_nchannel[tid]) if hasattr(m, "tex_nchannel") else 3
    rgb = np.array(Image.open(texture_path).convert("RGB"))
    if rgb.shape[:2] != (th, tw):
        rgb = np.array(Image.fromarray(rgb).resize((tw, th), Image.BILINEAR))
    if nch == 4:
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        rgb = np.concatenate([rgb, alpha], axis=2)
    elif nch != 3:
        raise RuntimeError(f"unsupported tex_nchannel={nch}")
    n = th * tw * nch
    orig = np.asarray(storage[adr:adr + n], dtype=np.uint8).copy()
    storage[adr:adr + n] = rgb.reshape(-1)
    if not _upload_texture(sim, tid):
        print("[warn] no GL context uploaded; renderer may keep old texture", flush=True)
    sim.forward()
    return (tid, orig, adr, n)


def restore_cloth(sim, info):
    if info is None: return
    tid, orig, adr, n = info
    m, storage = _tex_storage(sim)
    storage[adr:adr + n] = orig
    _upload_texture(sim, tid)


def run_episode(policy, task, inits, init_idx, texture_path, max_steps=400, wait=10, replan=5):
    env, desc = _get_env(task)
    env.reset(); env.set_init_state(inits[init_idx])
    sim = env.env.sim; sim.forward()
    restore = inject_cloth(sim, texture_path)
    obs = env.set_init_state(inits[init_idx])
    success = False
    plan = deque()
    try:
        for step in range(max_steps + wait):
            if step < wait:
                obs, _, done, _ = env.step(np.zeros(7))
                if done or env.env._check_success(): success = True; break
                continue
            if not plan:
                try:
                    gr00t_obs = make_gr00t_obs(obs, desc)
                    actions, _ = policy._get_action(gr00t_obs)
                    # 从 16 步 chunk 按序取前 replan 步
                    for ci in range(min(replan, 16)):
                        vals = [float(actions[k].flatten()[ci]) for k in
                                ("x","y","z","roll","pitch","yaw","gripper")]
                        a = np.array(vals, dtype=np.float64)
                        a[-1] = 2.0 * a[-1] - 1.0
                        a[-1] = -a[-1]
                        a[-1] = np.sign(a[-1])
                        plan.append(a)
                except Exception as e:
                    if step == wait:
                        import traceback; traceback.print_exc()
                    print(f"[infer] step={step}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    break
            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success(): success = True; break
    finally:
        restore_cloth(sim, restore)
        try: env.close()
        except Exception: pass
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--condition", choices=["original", "plain", "adv"], default="original")
    ap.add_argument("--texture", default="")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--init-offset", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from gr00t.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    policy = Gr00tPolicy(model_path=args.ckpt_dir, embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                         device=args.device, strict=False)
    print("[gr00t] loaded", flush=True)

    texture = None
    if args.condition == "plain": texture = CLOTH_T0
    elif args.condition == "adv": texture = args.texture

    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[args.suite]()
    f = open(args.out, "w", encoding="utf-8")
    f.write(f"condition={args.condition}\nckpt={args.ckpt_dir}\n\n")
    total = succ = 0
    for tid in [int(t) for t in args.tasks.split(",")]:
        task = bench.get_task(tid)
        inits = bench.get_task_init_states(tid)
        n = min(args.trials, len(inits) - args.init_offset)
        s = 0
        for ep in range(n):
            ok = run_episode(policy, task, inits, args.init_offset + ep, texture,
                             max_steps=args.max_steps)
            s += int(ok); total += 1; succ += int(ok)
            line = f"Task: {tid} | Ep: {ep} | Success: {ok}"
            print(line, flush=True); f.write(line + "\n"); f.flush()
        f.write(f"task_{tid}: {s}/{n}\n"); print(f"[eval] task {tid}: {s}/{n}", flush=True)
    f.write(f"\noverall robot success: {succ}/{total} ({succ/total:.4f})\n"); f.close()
    print(f"[eval] 结果: {args.out}", flush=True)


if __name__ == "__main__":
    main()
