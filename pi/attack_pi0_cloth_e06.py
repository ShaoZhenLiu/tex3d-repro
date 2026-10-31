#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_pi0_cloth_e06.py — E06：π0 源反向迁移双臂训练（第18轮 §10）
对归档 π0 桌布分支的五项修复（§10.1）：
  1. TV 改为真实相邻像素差分（OFT spatial_tv 同语义：两方向均值之和，作用于 δ），
     不再用旧切片 abs（L_abs）实现；
  2. 主循环单帧更新 → 每 iter 8 帧，攻击梯度 1/8 累积、正则一次、sign 更新一次；
  3. 采池 query_key/nq 修正：不使用环境变量 qseed（采集路径显式走独立查询噪声，
     CLOTH_QUERY_SEED 不设置；本 runner 不注入该逻辑）；
  4. 采池去噪显式 10 步（load_policy(num_denoise=10)），登记采池策略协议；
  5. 仅替换主视图（双相机 π 源的首轮保留并披露）。
两臂（各对只变参考，§10.2）：
  R-N（--ref native ）= 池帧 clean_img_embs（原生 Plain 特征，采集时缓存）；
  R-M（--ref matched）= 同帧、同合成路径、θ=0 的参考特征（训练开始时
     临时置零 adv_noise 逐帧复合并缓存；MR 特征项在 θ=0 处距离=0、梯度≈0，
     有冒烟断言）。
  共享：新池（本 runner collect 采集）、同 θ0（seed7·std0.01）、同 frame/UV 流
  （先跑 fresh 臂写出 sampling_log+stream_full.pt，另一臂 replay）、batch8、
  5000 更新、lr0.05、ε=64/255、λmse=1.0、λref=λtv=0.1、seed7。
  可微图像 token embedding MSE（λmse=1），不回传去噪链；特征 MSE float32；
  参考 detach；只优化纹理参数，源权重冻结（requires_grad=False）。
用法（41，cwd=$D/code/tex3d/pi，经 run_cloth330.sh 环境）：
  python attack_pi0_cloth_e06.py collect --out e06_pool.pt ...
  python attack_pi0_cloth_e06.py check --pool e06_pool.pt ...      # 冒烟断言
  python attack_pi0_cloth_e06.py train --tag e06_rn --ref native --stream fresh ...
  python attack_pi0_cloth_e06.py train --tag e06_rm --ref matched --stream replay \
      --stream-file <e06_rn>/sampling_log.txt ...
"""
import argparse
import hashlib
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import openpi.models.model as _pi_model  # noqa: E402  Observation 命名空间
import attack_pi0_cloth as APC          # noqa: E402  π0 桌布管线（复用其加载/采集/合成/渲染器）
# ClothRenderer / extract_split_features / replace_image_in_processed_inputs
# 均经 APC 命名空间复用（ ClothRenderer 定义于 attack_pi0_cloth.py，参数=adv_vc_noise）

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/e06_pi0src"
E06_CKPT = "/data/shaozhen.liu/model/RLinf-Pi0-LIBERO-130-fullshot-SFT"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def real_tv(delta):
    """真实 TV（修复①）：NHWC/NCHW 兼容相邻差分，两方向均值之和（0-1 域）。"""
    d = delta.float()
    if d.ndim != 4:
        raise ValueError("Expected 4D tensor")
    dy = d[:, 1:, :, :] - d[:, :-1, :, :]
    dx = d[:, :, 1:, :] - d[:, :, :-1, :]
    return dy.abs().mean() + dx.abs().mean()


def uv_xform_seedless(rng_mod=random):
    """与生产 π0/OFT 同分布的 UV-EoT 仿射（旋转±镜像+±0.04 平移）。"""
    ang = rng_mod.uniform(-np.pi, np.pi)
    c, s_ = float(np.cos(ang)), float(np.sin(ang))
    A = np.array([[c, -s_, 0.5 - 0.5 * c + 0.5 * s_],
                  [s_, c, 0.5 - 0.5 * s_ - 0.5 * c]], dtype=np.float32)
    if rng_mod.random() < 0.5:
        if rng_mod.random() < 0.5:
            M = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], np.float32)
        else:
            M = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]], np.float32)
        A = (np.vstack([M, [0.0, 0.0, 1.0]])
             @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
    A[0, 2] += rng_mod.uniform(-0.04, 0.04)
    A[1, 2] += rng_mod.uniform(-0.04, 0.04)
    return A


def composite(renderer, frame, uv_xform):
    adv_rgba, _ = renderer.render(frame["mvp"], (256, 256), uv_xform=uv_xform)
    adv_rgb = adv_rgba.permute(0, 3, 1, 2)
    return torch.clamp(adv_rgb * frame["seg"] + frame["bg"] * (1.0 - frame["seg"]),
                       0.0, 1.0)


def emb_of(model, frame, renderer, uv_xform):
    adv_img = composite(renderer, frame, uv_xform)
    new_inputs, _, _ = APC.replace_image_in_processed_inputs(
        frame["processed_inputs"], adv_img, "cuda")
    obs = _pi_model.Observation.from_dict(new_inputs)
    embs, _, _ = APC.extract_split_features(model, obs)
    return embs


def collect_frames_e06(policy, task, init_state, num_frames, device,
                        stride=10, ep_tag=0):
    """修复③：重实现的 π0 闭环采帧（不经过旧采集路径的未声明 query_key/nq）。
    协议：T0 纯色注入、每 stride 步取 1 帧上限 num_frames、replan5、10 步去噪
    （由 policy 构造时的 num_denoise 决定）、独立查询噪声（不设 qseed）。
    帧字段与原路径一致：bg/seg/mvp/processed_inputs/clean_img_embs(+w/ep)。"""
    from collections import deque
    desc = task.language
    env, _ = APC._get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    APC.align_cloth_to_table(env, floor_mode=False)
    _, cloth_body_id, _ = APC.get_target_model_matrix(env, APC.CLOTH_SEARCH)
    assert cloth_body_id != -1, "未找到桌布 body"
    tex_id = APC.find_target_texture_id(sim, cloth_body_id, APC.CLOTH_SEARCH)
    assert tex_id >= 0, "未找到桌布纹理 id"
    t0_arr = np.array(Image.open(APC.CLOTH_T0).convert("RGB"))
    restore_info = APC.inject_mujoco_texture(sim, tex_id, t0_arr)
    model = policy._model
    frames = []
    action_plan = deque()
    try:
        obs = env.set_init_state(init_state)
        for _ in range(10):  # dummy 刷新：观测带上素色桌布
            obs, _, done, _ = env.step(APC.get_libero_dummy_action())
            if done or env.env._check_success():
                break
        step = 0
        while len(frames) < num_frames and step < 400:
            if stride <= 1 or step % stride == 0:
                bg_np = APC.render_mujoco_background_without_target(
                    env, cloth_body_id, 256,
                    search_keywords_list=APC.CLOTH_SEARCH)
                bg = (torch.from_numpy(bg_np).float().to(device) / 255.0
                      ).permute(2, 0, 1).unsqueeze(0)
                seg = APC.cloth_visibility_mask(env, cloth_body_id, 256)
                mm, _, _ = APC.get_target_model_matrix(env, APC.CLOTH_SEARCH)
                mvp = APC.build_mvp(env, mm, resolution=(256, 256))
                raw = APC.build_raw_inputs(obs, desc)
                with torch.no_grad():
                    pi_, obsn, _ = APC.get_processed_inputs_and_observation(
                        policy, raw)
                    embs, _, _ = APC.extract_split_features(model, obsn)
                frames.append({"bg": bg,
                               "seg": torch.from_numpy(
                                   seg.astype(np.float32)).to(device).view(
                                       1, 1, 256, 256),
                               "mvp": mvp, "processed_inputs": pi_,
                               "clean_img_embs": embs.detach(), "w": 1.0,
                               "ep": ep_tag})
            if not action_plan:
                raw = APC.build_raw_inputs(obs, desc)
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
                action_plan.extend(chunk[:min(5, len(chunk))])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success():
                break
            step += 1
    finally:
        APC.restore_mujoco_texture(sim, tex_id, *restore_info)
        APC._close_libero_env(env)
    return frames


def cmd_collect(args):
    device = "cuda"
    os.makedirs(OUT_ROOT, exist_ok=True)
    policy = APC.load_policy(args.ckpt, num_denoise=args.num_denoise)  # 修复④
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    pool = {}
    n_total = 0
    t0 = time.time()
    for tid in [int(t) for t in args.tasks.split(",")]:
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        pool[tid] = []
        for ep in range(args.episodes):
            frames = collect_frames_e06(policy, task, inits[ep],
                                        args.max_frames, device,
                                        stride=args.stride, ep_tag=ep)
            pool[tid].extend(frames)
            n_total += len(frames)
            print("[e06-collect] task%d ep%d: %d 帧（累计 %d）"
                  % (tid, ep, len(frames), n_total), flush=True)
    torch.save(pool, args.out)
    print("[e06-collect] 完成 %d 帧 -> %s（用时 %.0fs，num_denoise=%d，"
          "独立查询噪声：未设置 CLOTH_QUERY_SEED）"
          % (n_total, args.out, time.time() - t0, args.num_denoise))


def build_renderer(args, device):
    return APC.ClothRenderer(device=device, epsilon=args.epsilon,
                              init_std=args.init_std,
                              seed=args.seed).to(device)


def load_pool_frames(pool):
    act = []
    for tid in pool:
        for fr in pool[tid]:
            act.append((tid, fr))
    return act


def cmd_check(args):
    """冒烟断言（§10.2 验收）：16 帧 θ=0 MR 一致性 / MR 特征项 θ=0 梯度≈0 /
    初始化梯度非零 / 真TV方向检查。"""
    device = "cuda"
    policy = APC.load_policy(args.ckpt, num_denoise=10)
    model = policy._model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    pool = torch.load(args.pool, map_location=device, weights_only=False)
    act = load_pool_frames(pool)
    print("[e06-check] 池 %d 帧" % len(act))
    renderer = build_renderer(args, device)
    param = renderer.adv_vc_noise  # 顶点域 θ（Form-1 组合）
    t0_tex = renderer.orig_texture
    # (a) 真TV方向检查：构造整行常数、列递增的 δ，真TV>0 且方向项可分解
    ramp = torch.zeros(1, 16, 16, 3, device=device)
    ramp[:, :, 1:, :] = 1.0
    tv_real = float(real_tv(ramp))
    tv_old = float((ramp[:, 1:, :, :].abs().mean()
                    + ramp[:, :, 1:, :].abs().mean()) / 2.0)
    print("[e06-check] 真TV=%.4f（旧切片L_abs=%.4f；行常数、仅首列跃变时 "
          "真TV=dx均值=1/(W-1)=1/15）" % (tv_real, tv_old))
    assert abs(tv_real - 1.0 / 15.0) < 1e-4
    # (b) 16 帧 θ=0：matched 参考距离=0（构造性），对 native 的参考域差>0
    theta0 = param.detach().clone()
    gap = []
    with torch.no_grad():
        param.zero_()
        for i, (tid, fr) in enumerate(act[:16]):
            m = emb_of(model, fr, renderer, None)
            n = fr["clean_img_embs"]
            gap.append(float(F.mse_loss(m.float(), n.float())))
        mr_ref = [emb_of(model, fr, renderer, None).detach()
                  for _, fr in act[:16]]
    with torch.no_grad():
        param.copy_(theta0)
    print("[e06-check] θ=0 matched 距离=0（构造性✓）；θ=0 合成 vs native 参考 "
          "MSE：min=%.3e max=%.3e mean=%.3e（参考域差>0 即 R-N/R-M 区分）"
          % (min(gap), max(gap), sum(gap) / len(gap)))
    assert min(gap) > 0
    # (c) MR 特征项在 θ=0 的梯度 ≈ 0（数学上 (x−c)=0 → ∇=0）
    param.grad = None
    loss = -F.mse_loss(emb_of(model, act[0][1], renderer, None).float(),
                       mr_ref[0].to(device).float())
    loss.backward()
    g0 = float(param.grad.norm()) if param.grad is not None else 0.0
    print("[e06-check] MR 特征项 θ=0 梯度范数=%.3e（应≈0）" % g0)
    assert g0 < 1e-3, g0
    # (d) θ0（std0.01）初始化梯度非零且有限（对 matched 与 native 均')
    for refname, ref in (("matched", mr_ref[0].to(device)),
                         ("native", act[0][1]["clean_img_embs"])):
        param.grad = None
        loss = -F.mse_loss(emb_of(model, act[0][1], renderer, None).float(),
                           ref.float())
        loss.backward()
        g = float(param.grad.norm())
        print("[e06-check] θ0 梯度[%s]=%.3e（非零有限）" % (refname, g))
        assert np.isfinite(g) and g > 0
    print("[e06-check] 全部断言通过")


def train(args):
    device = "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(OUT_ROOT, exist_ok=True)
    APC.ensure_patch(True)
    policy = APC.load_policy(args.ckpt, num_denoise=10)   # 训练进程仅用于特征
    model = policy._model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    pool = torch.load(args.pool, map_location=device, weights_only=False)
    act = load_pool_frames(pool)
    n = len(act)
    print("[e06-train] 池 %d 帧" % n)

    renderer = build_renderer(args, device)
    param = renderer.adv_vc_noise  # 顶点域 θ（Form-1 组合）
    t0_tex = renderer.orig_texture

    refs = None
    if args.ref == "matched":
        theta0 = param.detach().clone()
        with torch.no_grad():
            param.zero_()
            refs = [emb_of(model, fr, renderer, None).detach()
                    for _, fr in act]
        with torch.no_grad():
            param.copy_(theta0)
        print("[e06-train] matched 参考（θ=0 同合成路径）%d 帧已缓存" % len(refs))

    stream = None
    full_uv = None
    if args.stream == "replay":
        stream = {}
        cnt = 0
        need = args.iters * args.batch
        for line in open(args.stream_file):
            if line.startswith("#"):
                continue
            if cnt >= need:
                break
            q = line.split()
            it, slot, fi = int(q[0]), int(q[1]), int(q[2])
            uv = None
            if len(q) >= 10 and any(x != "0.000000" for x in q[4:10]):
                uv = np.array([[float(q[4]), float(q[5]), float(q[6])],
                               [float(q[7]), float(q[8]), float(q[9])]],
                              dtype=np.float32)
            stream.setdefault(it, {})[slot] = (fi, uv)
            cnt += 1
        assert cnt == need, (cnt, need)
        pt = os.path.splitext(args.stream_file)[0] + "_full.pt"
        if os.path.isfile(pt) and args.uv == "on":
            d = torch.load(pt, map_location="cpu", weights_only=False)
            assert len(d["uv"]) == args.iters
            full_uv = d["uv"]
            print("[e06-train] replay %s + 全精度 UV" % args.stream_file)
        else:
            print("[e06-train] replay %s（UV 文本 6 位小数）" % args.stream_file)
        stream = {it: {"frames": [stream[it][s][0] for s in range(args.batch)],
                       "uv": stream[it][0][1]} for it in range(args.iters)}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"train_{stamp}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("fixes=realTV,batch8avg,collect-denoise10,no-qseed-env,"
                "mainview-only(disclosed)\n")
        if args.stream == "replay":
            f.write("stream_file=%s sha256=%s\n" % (args.stream_file,
                                                    sha256(args.stream_file)))
    log_path = os.path.join(out_dir, "gradient_log.txt")
    torch.save(param.detach().cpu(), os.path.join(out_dir, "theta_init.pt"))
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_init.pt"))
    with open(os.path.join(out_dir, "frame_map.txt"), "w") as f:
        f.write("# active_idx task ep_note\n")
        for ai, (tid, _fr) in enumerate(act):
            f.write("%d %d\n" % (ai, tid))

    samp = open(os.path.join(out_dir, "sampling_log.txt"), "w")
    samp.write("# it slot frame_idx task uv00..uv12 (六零=恒等哨兵)\n")
    fresh_uv_full = []
    print("[e06-train] %d iters × bs%d，ref=%s，stream=%s，uv=%s，开始"
          % (args.iters, args.batch, args.ref, args.stream, args.uv))
    t_start = time.time()

    def save_ckpt(it, final=False):
        suffix = f"_it{it:05d}" if args.ckpt_step_suffix else ""
        png = renderer.export_texture()
        if suffix:
            Image.fromarray(png).save(os.path.join(
                out_dir, "Universal_Cloth%s.png" % suffix))
            torch.save(param.detach().cpu(), os.path.join(
                out_dir, "Cloth_Delta%s.pt" % suffix))
        Image.fromarray(png).save(
            os.path.join(out_dir, "Universal_Cloth.png"))
        torch.save(param.detach().cpu(),
                   os.path.join(out_dir, "Cloth_Delta.pt"))
        with open(log_path, "a") as f:
            f.write("[ckpt%s] it%05d frames_accessed=%d\n"
                    % ("-final" if final else "", it, (it + 1) * args.batch))

    for it in range(args.iters):
        if stream is not None:
            batch_idx = stream[it]["frames"]
            if args.uv == "on":
                if full_uv is not None:
                    uv_a = torch.from_numpy(full_uv[it]).to(device)
                else:
                    assert stream[it]["uv"] is not None, "uv=on 但流为零哨兵"
                    uv_a = torch.from_numpy(stream[it]["uv"]).to(device)
            else:
                uv_a = None
        else:
            batch_idx = [random.randrange(n) for _ in range(args.batch)]
            uv_a = (torch.from_numpy(uv_xform_seedless()).to(device)
                    if args.uv == "on" else None)
        if stream is None and uv_a is not None:
            fresh_uv_full.append(uv_a.detach().cpu().numpy())
        uvl = [float(x) for x in uv_a.cpu().numpy().reshape(-1)] \
            if uv_a is not None else [0.0] * 6
        param.grad = None
        for slot, fi in enumerate(batch_idx):
            samp.write("%d %d %d %d %s\n" % (
                it, slot, fi, act[fi][0],
                " ".join("%.6f" % x for x in uvl)))
        attack_sum = 0.0
        for fi in batch_idx:
            tid, fr = act[fi]
            adv_img = composite(renderer, fr, uv_a)
            new_inputs, _, _ = APC.replace_image_in_processed_inputs(
                fr["processed_inputs"], adv_img, device)
            adv_obs = _pi_model.Observation.from_dict(new_inputs)
            adv_embs, _, _ = APC.extract_split_features(model, adv_obs)
            ref = (refs[fi].to(device) if refs is not None
                   else fr["clean_img_embs"])
            attack = -F.mse_loss(adv_embs.float(), ref.float())
            attack_sum += attack.item()
            (args.lambda_mse * attack / args.batch).backward()
        tex = renderer.texture_tensor()
        ref_l = F.mse_loss(tex, t0_tex)
        tv = real_tv(tex - t0_tex)
        (args.lambda_ref * ref_l + args.lambda_tv * tv).backward()
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            assert np.isfinite(g) and g > 0, (it, g)
            line = (f"[e06-train] it{it:05d} bs{args.batch} "
                    f"attack={attack_sum / args.batch:+.4f} "
                    f"ref={ref_l.item():.4f} tv={tv.item():.4f} "
                    f"g={g:.2e} frames_accessed={(it + 1) * args.batch}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            save_ckpt(it, final=(it == args.iters - 1))
    samp.close()
    if stream is None:
        torch.save({"uv": fresh_uv_full, "seed": args.seed, "n_active": n},
                   os.path.join(out_dir, "stream_full.pt"))
    delta = (torch.tanh(param.detach()) * args.epsilon) * 255.0
    tex = renderer.texture_tensor().detach()
    d255 = (tex - t0_tex).abs() * 255.0
    stats = ("[final] delta RMS=%.4f Linf=%.1f TV=%.4f 像素域RMS=%.4f "
             "Linf=%.1f 用时=%.0fs"
             % (float(torch.sqrt((delta ** 2).mean())), float(delta.abs().max()),
                float(real_tv(tex - t0_tex)),
                float(torch.sqrt((d255 ** 2).mean())), float(d255.max()),
                time.time() - t_start))
    with open(log_path, "a") as f:
        f.write(stats + "\n")
    print("[e06-train] " + stats)
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_final.pt"))
    APC.ensure_patch(False)
    print("[e06-train] 完成 %s" % out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", default="0,2,6,8")
    c.add_argument("--episodes", type=int, default=2)
    c.add_argument("--stride", type=int, default=10)
    c.add_argument("--max-frames", type=int, default=40)
    c.add_argument("--num-denoise", type=int, default=10)
    k = sub.add_parser("check")
    for p in (c, k):
        p.add_argument("--pool", default="")
        p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
        p.add_argument("--init-std", type=float, default=0.01)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--ckpt", default=E06_CKPT)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--ref", choices=["native", "matched"], required=True)
    t.add_argument("--pool", required=True)
    t.add_argument("--stream", choices=["fresh", "replay"], default="fresh")
    t.add_argument("--stream-file", default="")
    t.add_argument("--uv", choices=["on", "off"], default="on")
    t.add_argument("--iters", type=int, default=5000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--attack-lr", type=float, default=0.05)
    t.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    t.add_argument("--init-std", type=float, default=0.01)
    t.add_argument("--lambda-mse", type=float, default=1.0)
    t.add_argument("--lambda-ref", type=float, default=0.1)
    t.add_argument("--lambda-tv", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--checkpoint-every", type=int, default=625)
    t.add_argument("--ckpt-step-suffix", action="store_true")
    t.add_argument("--ckpt", default=E06_CKPT)
    args = ap.parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "check":
        cmd_check(args)
    else:
        if args.stream == "replay":
            assert args.stream_file and os.path.isfile(args.stream_file)
        train(args)


if __name__ == "__main__":
    main()
