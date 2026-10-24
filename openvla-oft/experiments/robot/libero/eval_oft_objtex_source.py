#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_oft_objtex_source.py — Tex3D 原生对象纹理的源端最小闭环检查（第 15 轮 P1-2）
目的：诊断部署与源端作用（非收录分数门槛）。身份核对 §4 已确认无任何源侧闭环证据。
设计：
  源策略 = OpenVLA-OFT Spatial step150000（与全部训练同源，M.CKPT）；
  任务 = {0,8}（各用自己任务的冻结纹理：task0=5adecbb8…、task8_r2=5e778ec6…）；
  条件 = orig（注入原资产 texture.png，8a646d98…）/ th0（θ0 再烘焙，97c3ae59…）/ adv；
  初态 = 10–14（--init-offset 10，--trials 5）→ 2×3×5 = 30 回合；
  注入 = 运行时 tex 热替换（cloth_transfer_utils，与 π 侧 objtex 部署同机制，
        不写共享资产）；回合内绑定、finally 恢复；
  分辨率 = 256（与训练采池一致；上游 eval_libero 为 512，本检查为部署有效性诊断，
        分辨率选择如实登记）；
  同态证据（每任务每条件首个初态，注入后、动作前）：
        高分辨率帧 PNG + 模型输入（resize 到 model_input_size）PNG +
        对象区域差分 mask（|cond−orig|≥8 逐像素）与变化像素占比；
  回合协议 = 300 步 + num_steps_wait 哑动作（与 transfer_pi2oft/上游 live-test 同构）。
输出：OUT/ 下每条件一份 evaluation_result.txt（Task/Ep/Success 行）+ evidence/。
用法（41，cwd=$D/code/tex3d-ws/g7，tex3d-oft-mj330）：
  CUDA_VISIBLE_DEVICES=1 python $TRD/eval_oft_objtex_source.py
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_oft as UP                     # noqa: E402  上游（lab EGL 守卫版）
import attack_oft_cloth_tvfix_mr as M       # noqa: E402  同源 ckpt/动作模块
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import cloth_transfer_utils as ctu          # noqa: E402  注入四件套（attack_pi0 原样）

D = "/data/shaozhen.liu"
OUT = os.path.join(D, "code/tex3d/experiments/tex3d_native/source_check")
TNAT = os.path.join(D, "code/tex3d/experiments/tex3d_native")
TEX = {
    "orig": None,  # 运行时取 UP.OBJECTS 原资产
    "th0": os.path.join(TNAT, "train_20261023_072153_r1", "theta0_baked.png"),
    0: os.path.join(TNAT, "train_20261023_072153_r1", "akita_bowl_adv_baked.png"),
    8: os.path.join(TNAT, "train_20261023_203042_task8_r2", "akita_bowl_adv_baked.png"),
}
RES = 256
MAX_STEPS = 300


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_env(task):
    env, desc = UP.get_libero_env(task, "openvla", resolution=RES)
    return env, desc


def find_bowl_tex(sim, env, search):
    _, bid, _ = UP.get_target_model_matrix(env, search)
    tex_id = ctu.find_target_texture_id(sim, bid, search)
    if tex_id < 0 or bid == -1:
        raise RuntimeError("未找到 akita_black_bowl 纹理 id（部署失败）")
    return tex_id


def capture_evidence(task, tid, conditions, search, out_ev, f):
    """同一初态（init 10）下逐条件注入并捕获帧/差分 mask（注入后、动作前）。"""
    import attack_oft as _UP
    from libero.libero import benchmark as _bm
    suite = _bm.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    frames = {}
    for cond, tex_path in conditions.items():
        env, _ = build_env(task)
        env.reset()
        obs = env.set_init_state(inits[10])
        sim = env.env.sim
        sim.forward()
        tex_id = find_bowl_tex(sim, env, search)
        restore = ctu.inject_mujoco_texture(
            sim, tex_id, np.array(Image.open(tex_path).convert("RGB")))
        try:
            sim.forward()
            img = _UP.get_libero_image(obs)
            frames[cond] = img
            Image.fromarray(img).save(
                os.path.join(out_ev, f"frame_{cond}_t{tid}.png"))
        finally:
            ctu.restore_mujoco_texture(sim, tex_id, *restore)
            env.close()
    for cond in ("th0", "adv_t%d" % tid):
        d = np.abs(frames[cond].astype(np.int16) - frames["orig"].astype(np.int16))
        mask = (d >= 8).any(axis=2)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(out_ev, f"mask_{cond}_vs_orig_t{tid}.png"))
        frac = float(mask.mean())
        f.write(f"evidence t{tid}: mask {cond}-vs-orig changed={frac:.4f} "
                f"maxdiff={int(d.max())}\n")
        print(f"[evidence] t{tid} {cond}-vs-orig changed={frac:.4f} "
              f"maxdiff={int(d.max())}", flush=True)


def run_condition(cfg, model, processor, modules, cond, tex_path, tasks,
                  trials, init_offset, out_dir, f):
    action_head, proprio_projector, noisy_action_projector = modules
    use_proprio = bool(getattr(cfg, "use_proprio", False))
    use_wrist = getattr(cfg, "num_images_in_input", 1) > 1
    model_input_size = UP.get_image_resize_size(cfg)
    search = UP.OBJECTS["akita_black_bowl"]["search"]
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    total = 0
    succ_all = 0
    for tid in tasks:
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        n = min(trials, len(inits) - init_offset)
        succ = 0
        for ep in range(n):
            env, desc = build_env(task)
            env.reset()
            obs = env.set_init_state(inits[ep + init_offset])
            sim = env.env.sim
            sim.forward()
            restore, tex_id, done = None, -1, False
            try:
                tex_id = find_bowl_tex(sim, env, search)
                restore = ctu.inject_mujoco_texture(
                    sim, tex_id,
                    np.array(Image.open(tex_path).convert("RGB")))
                t = 0
                while t < MAX_STEPS + cfg.num_steps_wait:
                    if t < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(
                            UP.get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    img_high = UP.get_libero_image(obs)
                    img_model = np.array(Image.fromarray(img_high).resize(
                        (model_input_size, model_input_size)))
                    observation = {"full_image": img_model}
                    if use_proprio:
                        observation["state"] = np.concatenate((
                            obs["robot0_eef_pos"],
                            UP.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))
                    else:
                        observation["state"] = np.zeros(8)
                    if use_wrist:
                        wrist_m = UP._extract_wrist_image(obs, model_input_size)
                        if wrist_m is not None:
                            observation["wrist_image"] = wrist_m
                    action = UP._query_action_with_compatible_signature(
                        cfg, model, observation, desc, processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=getattr(cfg, "use_film", False))
                    action = np.asarray(
                        action[0] if isinstance(action, list) else action,
                        dtype=np.float32)
                    action = UP.normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = UP.invert_gripper_action(action)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                    t += 1
            finally:
                if restore is not None:
                    ctu.restore_mujoco_texture(sim, tex_id, *restore)
                env.close()
            succ += int(done)
            total += 1
            succ_all += int(done)
            line = f"Task: {tid} | Ep: {ep} | Success: {bool(done)}"
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
        f.write(f"task_{tid}: {succ}/{n} ({succ / n:.4f})\n")
        print(f"[eval] task {tid}: {succ}/{n}", flush=True)
    f.write(f"\noverall robot success: {succ_all}/{total} "
            f"({succ_all / max(total, 1):.4f})\n")
    print(f"[eval] 结果: condition={cond} {succ_all}/{total}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="0,8")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--init-offset", type=int, default=10)
    args = ap.parse_args()
    tasks = [int(t) for t in args.tasks.split(",")]

    obj = UP.OBJECTS["akita_black_bowl"]
    TEX["orig"] = obj["texture"]
    conditions = {"orig": TEX["orig"], "th0": TEX["th0"]}
    for tid in tasks:
        conditions[f"adv_t{tid}"] = TEX[tid]

    os.makedirs(os.path.join(OUT, "evidence"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary = os.path.join(OUT, f"source_check_{stamp}.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("Tex3D 源端最小检查（第15轮 P1-2）：2任务×3条件×初态10-14=30回合\n")
        f.write(f"ckpt={M.CKPT}\nres={RES} max_steps={MAX_STEPS}\n")
        for k, p in conditions.items():
            f.write(f"tex {k}: {p} sha256={sha256(p)}\n")
        f.flush()

        cfg = UP.GenerateConfig(
            pretrained_checkpoint=M.CKPT, model_family="openvla",
            task_suite_name="libero_spatial")
        cfg.use_proprio = False
        model = UP.get_model(cfg)
        UP.check_unnorm_key(cfg, model)
        processor = UP.get_processor(cfg)
        modules = M._build_action_modules(cfg, model)

        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        for tid in tasks:
            conds_t = {"orig": conditions["orig"], "th0": conditions["th0"],
                       f"adv_t{tid}": conditions[f"adv_t{tid}"]}
            capture_evidence(suite.get_task(tid), tid, conds_t,
                             obj["search"], os.path.join(OUT, "evidence"), f)

        for tid in tasks:
            for cond, tex_path in (
                    ("orig", conditions["orig"]),
                    ("th0", conditions["th0"]),
                    (f"adv_t{tid}", conditions[f"adv_t{tid}"])):
                res_path = os.path.join(
                    OUT, f"t{tid}_{cond}_evaluation_result.txt")
                with open(res_path, "w", encoding="utf-8") as rf:
                    rf.write(f"condition=src_{cond}\ntexture={tex_path}\n"
                             f"tasks={tid} trials={args.trials} "
                             f"init_offset={args.init_offset}\n\n")
                    rf.flush()
                    run_condition(cfg, model, processor, modules, cond,
                                  tex_path, [tid], args.trials,
                                  args.init_offset, OUT, rf)
        f.write("\nDONE-SOURCE-CHECK\n")
    print("[source-check] DONE-SOURCE-CHECK", flush=True)


if __name__ == "__main__":
    main()
