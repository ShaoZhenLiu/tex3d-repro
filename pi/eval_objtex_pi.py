#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_objtex_pi.py — Tex3D 原生对象纹理的目标侧闭环评测（第 14 轮 P2）
场景 = 无桌布（与 original 同 XML/O 环境）。部署 = 每回合开始时对场景内
akita_black_bowl 的 MuJoCo 纹理做**运行时注入**（与桌布评测同一 inject/restore
机制，回合结束恢复；全程零共享资产文件写入——上游的文件覆写部署方式在并发
评测下不安全，此为部署等价、写入安全的替代，披露）。
条件：
  objtex  Adv：烘焙对抗纹理 PNG（--texture）
  objtex0 P_method：θ=0 烘焙纹理 PNG（--texture 传 θ0 产物）——隔离再烘焙伪影
  （Original 复用既有 F0 O 行：运行时注入+恢复不改变资产，O 环境一致）。
协议：与桌布/patch 评测同构（tasks、trials=10、init-offset 10、400+10 步、
replan 5、num-denoise 10）；任务 = Spatial {0,2,6,8}（碗所在任务，共同子集）。
用法（经 run_patch_eval.sh 同款包装）：
  eval --condition objtex --texture <baked.png> --tasks 0,2,6,8 ... --ckpt ... [--model-config pi05_libero]
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

import attack_pi0_cloth as APC          # noqa: E402
from attack_pi0_cloth import (          # noqa: E402
    build_raw_inputs, get_libero_dummy_action, get_libero_image)
from libero.libero import benchmark     # noqa: E402

OBJ_SEARCH = [["akita_black_bowl"], ["bowl"]]


def evaluate_objtex(policy, task, init_state, tex_arr, max_steps=400,
                    num_steps_wait=10, replan_steps=5, save_video=None):
    desc = task.language
    env, _ = APC._get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    restore_info = None
    tex_id = -1
    try:
        _, body_id, found = APC.get_target_model_matrix(env, OBJ_SEARCH)
        if body_id != -1:
            tex_id = APC.find_target_texture_id(sim, body_id, OBJ_SEARCH)
        if tex_id < 0:
            raise RuntimeError("未找到碗纹理 id（body=%s）" % found)
        restore_info = APC.inject_mujoco_texture(sim, tex_id, tex_arr)
        success = False
        frames = []
        action_plan = deque()
        obs = env.set_init_state(init_state)
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
                raw_inputs = build_raw_inputs(obs, desc)
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
        if restore_info is not None:
            APC.restore_mujoco_texture(sim, tex_id, *restore_info)
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
    APC.ensure_patch(False)  # 无桌布场景
    tex_arr = np.array(Image.open(args.texture).convert("RGB"))
    policy = APC.load_policy(args.ckpt, model_config=args.model_config,
                             num_denoise=args.num_denoise)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = [int(t) for t in args.tasks.split(",")]
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    out_dir = os.path.join(APC.ARTIFACT_ROOT,
                           f"eval_{args.condition}_{args.suite}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "evaluation_result.txt")
    total_success = 0
    total_eps = 0
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"condition={args.condition}\ntexture={args.texture}\n"
                f"object=akita_black_bowl inject=runtime_mujoco\n"
                f"num_denoise={args.num_denoise} replan=5 "
                f"max_steps={args.max_steps} scene=no_cloth\n\n")
        for tid in tasks:
            task = suite.get_task(tid)
            init_states = suite.get_task_init_states(tid)
            n = min(args.trials, len(init_states))
            succ = 0
            for ep in range(n):
                video = (os.path.join(out_dir, f"task{tid}_ep{ep:02d}.mp4")
                         if ep == 0 and args.save_videos else None)
                ok = evaluate_objtex(policy, task,
                                     init_states[ep + args.init_offset],
                                     tex_arr, max_steps=args.max_steps,
                                     num_steps_wait=10, replan_steps=5,
                                     save_video=video)
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
                   choices=["objtex", "objtex0"])
    p.add_argument("--tasks", default="0,2,6,8")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--texture", required=True)
    p.add_argument("--ckpt", required=True)
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
