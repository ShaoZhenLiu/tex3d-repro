#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_branch_case.py — MR 同初态行为案例与完整状态分支（第 15 轮 P2）

选例规则（先于运行登记，不挑"好看"）：
  案例1 = π0.5-Object (task0, init_state_id 10)：v8_1 中 Plain 成功 / MR 失败 / ETVF 成功
          （该格符合格 31 个，取最小任务号+最小初态）。
  案例2 = π0-Object (task0, init_state_id 12)：v8_1 中 MR 成功 / ETVF 失败
          （π0-Object 符合格 21 个，取最小任务号+最小初态）。
执行：
  A) 参考场：Plain / MR / ETVF 三条件完整回合（同初态、同推理协议），逐
     replan 查询前按 (task, ep, nq) 设 torch 种子（三分支同 nq 同噪声），
     逐策略步登记末端位置/夹爪/动作分解，MR 场在每个 replan 边界保存完整
     状态快照（MuJoCo state 深拷贝 + 当前观测各数组深拷贝 + 未执行动作队列
     + 步计数 + nq + torch/numpy/python RNG 状态；π 查询无跨查询记忆，
     当前观测即查询全部输入，如实登记）。
  B) 分支点（预登记规则）：MR 末端位置相对 Plain 参照（同策略步对齐）首次
     偏离 > 3cm 的时刻记为 t_dev；k* = t_dev 之前最后一个 replan 快照
     （若全程未超 3cm，取 replan 序列中位）。在 k* 处自快照恢复，分别以
     MR（续）/ 换回 Plain / 换成 ETVF 三分支继续到回合结束；恢复=新 env、
     set_init_state 后 sim.set_state(快照)+forward、观测用快照数组、动作
     队列/nq/计数恢复，三分支下一次查询同种子同噪声；纹理经同一 tex_id
     运行时替换。
  C) 未恢复分支如实保留；个别分支恢复仅作局部干预证据，不估计总体恢复率。
输出：OUT/<case>/result.json（全部逐 步日志 + k* + 分支结果）与三条件/三分支
      视频（agentview, fps15）。图形由本地脚本另绘（动作平移/旋转/夹爪分图）。
用法（41，cwd=$D/code/tex3d/pi，tex3d-pi0-mj330，GPU 由 wrapper/环境指定）：
  python eval_branch_case.py --case 1
"""
import argparse
import copy
import json
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

import attack_pi0_cloth as APC
from attack_pi0_cloth import (  # noqa: F401
    build_raw_inputs, find_target_texture_id, get_libero_dummy_action,
    get_libero_image, get_target_model_matrix, inject_mujoco_texture,
    restore_mujoco_texture)

D = "/data/shaozhen.liu"
OUT = os.path.join(D, "code/tex3d/experiments/branch_cases")
TEX = {
    "plain": APC.CLOTH_T0,
    "mr": os.path.join(D, "code/tex3d/experiments/pi0_cloth/oft_cloth/"
                          "train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"),
    "etvf": os.path.join(D, "code/tex3d/experiments/pi0_cloth/oft_cloth/"
                            "train_20261022_112717_edpatvfix/Universal_Cloth_OV.png"),
}
CKPTS = {
    "pi0": os.path.join(D, "model/RLinf-Pi0-LIBERO-130-fullshot-SFT"),
    "pi05": os.path.join(D, "model/RLinf-Pi05-LIBERO-130-fullshot-SFT"),
}
CASES = {
    1: dict(target="pi05", model_config="pi05_libero", suite="libero_object",
            tid=0, ep=0, expect=dict(plain=1, mr=0, etvf=1),
            note="辨别性失败例：Plain✓ MR✗ ETVF✓"),
    2: dict(target="pi0", model_config="pi0_libero", suite="libero_object",
            tid=0, ep=2, expect=dict(plain=None, mr=1, etvf=0),
            note="反例：MR✓ ETVF✗（Plain 参照同步给出）"),
}
MAX_STEPS, WAIT, REPLAN = 400, 10, 5
DEV_TH = 0.03  # 分支点规则：相对 Plain 参照末端偏离阈值 3cm


def logEE(obs, act):
    return dict(ee=[float(x) for x in obs["robot0_eef_pos"]],
                grip=[float(x) for x in obs["robot0_gripper_qpos"]],
                a_trans=[float(x) for x in act[:3]],
                a_rot=[float(x) for x in act[3:6]],
                a_grip=float(act[6]) if act.shape[0] > 6 else None)


def snapshot(env, obs, action_plan, step, nq):
    return dict(
        mj=copy.deepcopy(env.env.sim.get_state()),
        obs={k: (copy.deepcopy(v) if isinstance(v, np.ndarray) else v)
             for k, v in obs.items()},
        plan=[a.tolist() if isinstance(a, np.ndarray) else a
              for a in action_plan],
        step=step, nq=nq,
        torch_rng=copy.deepcopy(torch.get_rng_state()),
        np_rng=copy.deepcopy(np.random.get_state()),
        py_rng=copy.deepcopy(__import__("random").getstate()),
    )


def run_ep(policy, task, init_state, tex_arr, tid, ep, floor_mode,
           snap_every_replan=False, save_video=None, start_from=None):
    """完整回合（或自快照续跑）。返回 (success, steps_log, snapshots, dev_info)。"""
    desc = task.language
    env, _ = APC._get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    APC.align_cloth_to_table(env, floor_mode=floor_mode)
    _, bid, _ = get_target_model_matrix(env, APC.CLOTH_SEARCH)
    tex_id = find_target_texture_id(sim, bid, APC.CLOTH_SEARCH)
    if tex_id < 0:
        APC._close_libero_env(env)
        raise RuntimeError("未找到桌布纹理 id")
    restore = inject_mujoco_texture(sim, tex_id, tex_arr)
    success = False
    frames = []
    steps_log = []
    snaps = []
    action_plan = deque()
    nq = 0
    if start_from is None:
        obs = env.set_init_state(init_state)
        step0, nq = 0, 0
    else:
        sim.set_state(start_from["mj"])
        sim.forward()
        obs = start_from["obs"]
        for a in start_from["plan"]:
            action_plan.append(np.asarray(a, dtype=np.float32))
        step0, nq = start_from["step"], start_from["nq"]
        torch.set_rng_state(start_from["torch_rng"])
        np.random.set_state(start_from["np_rng"])
        __import__("random").setstate(start_from["py_rng"])
    try:
        step = step0
        while step < MAX_STEPS + WAIT:
            if step < WAIT:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                step += 1
                if done or env.env._check_success():
                    success = True
                    break
                continue
            if save_video is not None:
                frames.append(get_libero_image(obs, 256))
            if not action_plan:
                if snap_every_replan:
                    snaps.append(snapshot(env, obs, action_plan, step, nq))
                qseed = (tid * 1000003 + ep) * 1000003 + nq
                torch.manual_seed(qseed)  # 三条件/三分支同 nq 同噪声
                raw = build_raw_inputs(obs, desc)
                with torch.no_grad():
                    result = policy.infer(raw)
                chunk = result["actions"] if isinstance(result, dict) else result
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                action_plan.extend(chunk[:min(REPLAN, len(chunk))])
                nq += 1
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            steps_log.append(dict(step=step, **logEE(obs, action)))
            step += 1
            if done or env.env._check_success():
                success = True
                break
    finally:
        restore_mujoco_texture(sim, tex_id, *restore)
        APC._close_libero_env(env)
    if save_video is not None and frames:
        import imageio
        w = imageio.get_writer(save_video, fps=15, codec="libx264", quality=8)
        for fr in frames:
            w.append_data(fr)
        w.close()
    return success, steps_log, snaps


def pick_kstar(mr_log, plain_log, snaps):
    ee_mr = {d["step"]: np.array(d["ee"]) for d in mr_log}
    t_dev = None
    for d in plain_log:
        s = d["step"]
        if s in ee_mr and np.linalg.norm(ee_mr[s] - np.array(d["ee"])) > DEV_TH:
            t_dev = s
            break
    if t_dev is None:
        k = max(len(snaps) // 2, 0)
        return k, None, "no_dev→median"
    cand = [i for i, sn in enumerate(snaps) if sn["step"] < t_dev]
    k = cand[-1] if cand else max(len(snaps) // 2, 0)
    return k, t_dev, "last_replan_before_dev"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()
    C = CASES[args.case]
    cdir = os.path.join(OUT, "case%d" % args.case)
    os.makedirs(cdir, exist_ok=True)

    APC.ensure_patch(True)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[C["suite"]]()
    task = suite.get_task(C["tid"])
    init = suite.get_task_init_states(C["tid"])[C["ep"]]
    floor_mode = (C["suite"] == "libero_object")

    texs = {k: np.array(Image.open(p).convert("RGB")) for k, p in TEX.items()}
    policy = APC.load_policy(CKPTS[C["target"]],
                             model_config=C["model_config"], num_denoise=10)

    out = dict(case=args.case, **C,
               textures=dict(TEX),
               per_query_seed="(tid*1000003+ep)*1000003+nq",
               branch_rule="k*=last replan before EE deviates >3cm from Plain "
                           "ref; median fallback", dev_th=DEV_TH)

    # A) 参考场
    for cond in ("plain", "mr", "etvf"):
        t0 = time.time()
        ok, lg, _ = run_ep(policy, task, init, texs[cond], C["tid"], C["ep"],
                           floor_mode,
                           save_video=os.path.join(cdir, f"ref_{cond}.mp4"))
        out.setdefault("refs", {})[cond] = dict(
            success=int(ok), steps=len(lg), secs=round(time.time() - t0, 1),
            log=lg)
        print(f"[case{args.case}] ref {cond}: success={ok} steps={len(lg)}",
              flush=True)

    # B) MR 带 replan 快照
    ok, lg, snaps = run_ep(policy, task, init, texs["mr"], C["tid"], C["ep"],
                           floor_mode, snap_every_replan=True,
                           save_video=os.path.join(cdir, "ref_mr_rewatch.mp4"))
    out["mr_snapshots"] = len(snaps)
    k, t_dev, how = pick_kstar(lg, out["refs"]["plain"]["log"], snaps)
    out["kstar"] = dict(k=k, t_dev=t_dev, rule=how,
                        snap_step=snaps[k]["step"] if snaps else None)
    print(f"[case{args.case}] k*={k} (step {out['kstar']['snap_step']}) "
          f"t_dev={t_dev} rule={how}", flush=True)

    # C) 三分支自快照恢复
    for cond in ("mr", "plain", "etvf"):
        t0 = time.time()
        ok, lg, _ = run_ep(policy, task, init, texs[cond], C["tid"], C["ep"],
                           floor_mode, start_from=snaps[k],
                           save_video=os.path.join(cdir, f"branch_{cond}.mp4"))
        out.setdefault("branches", {})[cond] = dict(
            success=int(ok), steps=len(lg), secs=round(time.time() - t0, 1),
            log=lg)
        print(f"[case{args.case}] branch {cond}: success={ok} steps={len(lg)}",
              flush=True)

    with open(os.path.join(cdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[case{args.case}] DONE-BRANCH-CASE -> {cdir}/result.json",
          flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
