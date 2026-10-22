#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_patch_pi.py — 原生 2D patch 的目标侧闭环评测（第 14 轮 P1，老师 §7-P1/§6.3）
场景 = 无桌布（与 original 条件同 XML、同 O 环境）——patch 攻击不使用桌布，
Original（O）行直接复用既有 F0 O 条件结果，不重跑。
注入 = 官方 EDPA 评测约定（eval/simulation/Libero：每回合开头随机采样一次
位置、回合内固定；主/腕可各挂独立 patch）：本脚本主/腕各一张 patch、
位置各独立采样（pos-seed 固定可复现，逐回合登记到结果文件）。
条件：
  patch        Adv：官方训练得到的优化 patch（--patch-primary/--patch-wrist .pt）
  patch_random P_method：同尺寸均匀随机 [0,1) 对照（官方初始化分布，seed7，
               逐回合重采样位置同 Adv；作为该攻击形式的未优化载体对照）
输出与桌布评测同格式（Task: t | Ep: e | Success: bool；每格 100 行），
聚合与守恒断言复用 agg 系脚本；每回合位置写入结果文件头。
用法（经 run_patch_eval.sh）：
  eval --condition patch --suite libero_object --tasks 0,1,2,... --trials 10 \
       --num-denoise 10 --init-offset 10 --patch-primary <pt> --patch-wrist <pt> \
       --ckpt <RLinf ckpt> [--model-config pi05_libero] [--pos-seed 7]
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

import attack_pi0_cloth as APC          # noqa: E402  复用其加载/环境/输入接口
from attack_pi0_cloth import (          # noqa: E402
    build_raw_inputs, get_libero_dummy_action, get_libero_image)
from libero.libero import benchmark     # noqa: E402


def load_patch_hwc(path):
    t = torch.load(path, map_location="cpu", weights_only=False).detach()
    assert t.dim() == 3 and t.shape[0] == 3, (path, tuple(t.shape))
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round() \
        .astype(np.uint8)
    return arr


def random_patch_hwc(rng, size):
    t = torch.zeros(3, size, size).uniform_(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def evaluate_patch(policy, task, init_state, p_pri, p_wri, pos, max_steps=400,
                   num_steps_wait=10, replan_steps=5, save_video=None,
                   save_vis=None, desc=None):
    t0, l0, tw, lw = pos
    ph, pw = p_pri.shape[:2]
    desc = desc or task.language
    env, _ = APC._get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    success = False
    frames = []
    action_plan = deque()
    obs = env.set_init_state(init_state)
    try:
        for step in range(max_steps + num_steps_wait):
            if step < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done or env.env._check_success():
                    success = True
                    break
                continue
            if save_video is not None:
                frames.append(get_libero_image(obs, 256))
            if not action_plan:
                img = get_libero_image(obs, 256).copy()
                wrist = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy()
                assert img.shape[0] >= ph and wrist.shape[0] >= p_wri.shape[0]
                img[t0:t0 + ph, l0:l0 + pw] = p_pri
                wrist[tw:tw + p_wri.shape[0],
                      lw:lw + p_wri.shape[1]] = p_wri
                if save_vis is not None:
                    side = np.concatenate([img, wrist], axis=1)
                    Image.fromarray(side).save(save_vis)
                    save_vis = None  # 只存第一帧
                raw_inputs = build_raw_inputs(obs, desc, adv_image_np=img,
                                              adv_wrist_image_np=wrist)
                with torch.no_grad():
                    result = policy.infer(raw_inputs)
                chunk = result["actions"] if isinstance(result, dict) else result
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                action_plan.extend(chunk[:min(replan_steps, len(chunk))])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success():
                success = True
                break
    finally:
        APC._close_libero_env(env)
    if save_video is not None and frames:
        import imageio
        writer = imageio.get_writer(save_video, fps=15, codec="libx264",
                                    quality=8)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
    return success


def run_eval(args):
    APC.ensure_patch(False)  # 无桌布场景（同 original）
    suite = benchmark.get_benchmark_dict()[args.suite]()
    floor_mode = (args.suite == "libero_object")
    _ = floor_mode  # 无布条件下无需对齐
    if args.condition == "patch":
        if not (args.patch_primary and args.patch_wrist):
            raise ValueError("patch 条件需要 --patch-primary 与 --patch-wrist")
        p_pri = load_patch_hwc(args.patch_primary)
        p_wri = load_patch_hwc(args.patch_wrist)
    else:  # patch_random：官方初始化分布的随机对照（seed 固定）
        rng_p = torch.Generator().manual_seed(args.pos_seed)
        p_pri = (torch.zeros(3, args.random_size, args.random_size)
                 .uniform_(0, 1, generator=rng_p)
                 .permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        p_wri = (torch.zeros(3, args.random_size, args.random_size)
                 .uniform_(0, 1, generator=rng_p)
                 .permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    ph = p_pri.shape[0]

    policy = APC.load_policy(args.ckpt, model_config=args.model_config,
                             num_denoise=args.num_denoise)
    tasks = ([int(t) for t in args.tasks.split(",")]
             if args.tasks != "heldout" else APC.HELDOUT_TASKS)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    out_dir = os.path.join(APC.ARTIFACT_ROOT,
                           f"eval_{args.condition}_{args.suite}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "evaluation_result.txt")

    rng = np.random.default_rng(args.pos_seed)
    total_success = 0
    total_eps = 0
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"condition={args.condition}\n"
                f"patch_primary={args.patch_primary}\n"
                f"patch_wrist={args.patch_wrist}\n"
                f"patch_size={ph} pos_seed={args.pos_seed}\n"
                f"num_denoise={args.num_denoise} replan=5 "
                f"max_steps={args.max_steps} scene=no_cloth\n\n")
        for tid in tasks:
            task = suite.get_task(tid)
            init_states = suite.get_task_init_states(tid)
            n = min(args.trials, len(init_states))
            succ = 0
            for ep in range(n):
                t0 = int(rng.integers(0, 256 - ph + 1))
                l0 = int(rng.integers(0, 256 - ph + 1))
                tw = int(rng.integers(0, 256 - p_wri.shape[0] + 1))
                lw = int(rng.integers(0, 256 - p_wri.shape[1] + 1))
                f.write(f"pos task{tid} ep{ep}: "
                        f"pri({t0},{l0}) wrist({tw},{lw})\n")
                video = (os.path.join(out_dir, f"task{tid}_ep{ep:02d}.mp4")
                         if ep == 0 and args.save_videos else None)
                vis = (os.path.join(out_dir, "injection_check.png")
                       if tid == tasks[0] and ep == 0 else None)
                ok = evaluate_patch(policy, task,
                                    init_states[ep + args.init_offset],
                                    p_pri, p_wri, (t0, l0, tw, lw),
                                    max_steps=args.max_steps,
                                    num_steps_wait=10, replan_steps=5,
                                    save_video=video, save_vis=vis)
                succ += int(ok)
                total_success += int(ok)
                total_eps += 1
                line = f"Task: {tid} | Ep: {ep} | Success: {ok}"
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
            f.write(f"task_{tid}: {succ}/{n} ({succ / n:.4f})\n")
            f.flush()
            print(f"[eval] task {tid}: {succ}/{n} ({succ / n:.2%})")
        f.write(f"\noverall robot success: {total_success}/{total_eps} "
                f"({total_success / max(total_eps, 1):.4f})\n")
        f.write("attack failure rate = 1 - above\n")
    print(f"[eval] 结果: {result_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--condition", required=True,
                   choices=["patch", "patch_random"])
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--patch-primary", default=None)
    p.add_argument("--patch-wrist", default=None)
    p.add_argument("--random-size", type=int, default=57)
    p.add_argument("--pos-seed", type=int, default=7)
    p.add_argument("--ckpt", default=None, required=True)
    p.add_argument("--model-config", default="pi0_libero",
                   choices=["pi0_libero", "pi05_libero"])
    p.add_argument("--num-denoise", type=int, default=10)
    p.add_argument("--init-offset", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
    os._exit(0)
