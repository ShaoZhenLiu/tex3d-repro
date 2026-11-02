#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_oft_cloth_ablate.py — r18批次B/C 统一消融训练入口（E04/E05/E07）
复用生产配方 attack_oft_cloth_tvfix_mr.py@d1004416… 的训练循环（逐行同数学：
每 iter 攻击梯度 1/batch 累积 + 正则一次 + sign 更新 + clamp±5），仅参数化：
  --ref {mr, native}   参考来源：mr=mr_plain_cache.pt（同帧同渲染器 θ=0）；
                       native=池帧自带 clean_hidden（=Native8 臂 tvfix-bs8 同源）
  --stream {fresh, replay}
                       fresh = 生产式随机流（random.seed(s)：batch_idx 用
                       random.randrange、UV 用 sample_uv_xform，消耗顺序与生产一致）；
                       replay = 从流文件逐条重放 frame ID（+可选 UV 列）
  --stream-file PATH   replay 源（it slot frame task [uv×6]）；若有同名 .pt
                       （{"uv": [N,2,3] 全精度}）则 UV 用全精度版本
  --uv {on, off}       on=fresh 时逐 iter 采 UV；replay 时用流文件 UV 列。
                       off=render(uv_xform=None) 恒等分支（结构性无变换）
  --filter-task INT    只用该任务的池帧（E04 单任务臂）；MR 缓存按原始全局帧 ID
                       对齐子集，断言长度一致
  --ckpt-step-suffix   checkpoint 文件名带步数（E07：避免 625 步覆盖终点）
  成对共享初始化与采样流（老师 §9/§11）：先跑 fresh 臂（写出 sampling_log +
  stream_full.pt），另一臂 replay 该日志——两臂帧/UV 流逐条相同。
  额外登记（§8.2E）：theta_init.pt、rng_states_init/final.pt、frame_map.txt
  （active_idx→global_id→task/池内序）、终点 δ 统计（×255）。
用法（41，cwd=$D/code/tex3d-ws/g7，tex3d-oft-mj330）：
  python attack_oft_cloth_ablate.py genstream --filter-task 0 --iters 5000 ...
  python attack_oft_cloth_ablate.py train --tag e04-a --ref mr --filter-task 0 \
      --stream replay --stream-file <pregen> --uv on --iters 5000 --batch 8
"""
import argparse
import hashlib
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = '/data/shaozhen.liu/code/tex3d/openvla-oft/experiments/robot/libero'
sys.path.insert(0, HERE)
import attack_oft_cloth_tvfix_mr as M   # noqa: E402  生产配方入口（只读复用）
import attack_oft as base               # noqa: E402

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
MR_CACHE = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/mr_plain_cache.pt"
OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth"
POOL_SHA = "d03d68591b9b820385d14be44758aefd95c2a63b8d68fc0926413a2eb1b04a93"
CACHE_SHA = "7ab459156068e23cd338bc33e100fb1113a3a367d8b020a80cdaf7cdb7249856"
resource_ready = lambda: None
pause_requested = lambda: False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frame_to_dev(fr, device):
    """池驻 CPU，逐次访问搬运 GPU（降峰值显存：池≈6-8G 不再常驻）。
    text_inputs 等可能是 HF 容器（BatchEncoding 等）：优先用其自身 .to()；
    纯 dict 递归一层；其余原样。"""
    out = {}
    for k, v in fr.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device, non_blocking=True)
                           if torch.is_tensor(vv) else vv)
                      for kk, vv in v.items()}
        elif hasattr(v, "to") and callable(v.to):
            try:
                out[k] = v.to(device)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def active_frames(pool, filter_task):
    """返回 (active_list[(task,fr)], global_ids)。global_ids=未过滤池的全局下标。"""
    task_ids = list(pool.keys())
    out, gids = [], []
    for fi, (tid, fr) in enumerate((t, f) for t in task_ids for f in pool[t]
                                   if f["mvp"] is not None):
        if filter_task is None or tid == filter_task:
            out.append((tid, fr))
            gids.append(fi)
    return out, gids


def parse_stream(path, iters, batch):
    """流文件 → {it: {"frames": [...], "uv": 2x3 ndarray 或 None}}。
    只读前 iters*batch 条（允许源流更长，如 E07-625 重放 40000 条日志的前缀）。"""
    raw = {}
    n = 0
    need = iters * batch
    for line in open(path):
        if line.startswith("#"):
            continue
        if n >= need:
            break
        p = line.split()
        it, slot, fi = int(p[0]), int(p[1]), int(p[2])
        uv = None
        if len(p) >= 10 and any(x != "0.000000" for x in p[4:10]):
            uv = np.array([[float(p[4]), float(p[5]), float(p[6])],
                           [float(p[7]), float(p[8]), float(p[9])]],
                          dtype=np.float32)
        raw.setdefault(it, {})[slot] = (fi, uv)
        n += 1
    assert n == need, ("流长度不足", n, need)
    stream = {}
    for it in range(iters):
        d = raw[it]
        assert sorted(d) == list(range(batch)), (it, sorted(d))
        stream[it] = {"frames": [d[s][0] for s in range(batch)],
                      "uv": d[0][1]}   # 每 iter 共享 1 个 UV（生产语义）
    return stream


def load_full_uv(path_pt, iters):
    """stream_full.pt 的全精度 UV（若有）。"""
    d = torch.load(path_pt, map_location="cpu", weights_only=False)
    uv = d["uv"]
    assert len(uv) == iters, (len(uv), iters)
    return uv


def cmd_genstream(args):
    """在（可过滤的）活动帧表上预生成生产式流：random 消耗顺序与生产循环一致。"""
    random.seed(args.seed)
    pool = torch.load(POOL, map_location="cpu", weights_only=False, mmap=True)
    act, gids = active_frames(pool, args.filter_task)
    n = len(act)
    print("[genstream] 活动帧 %d（filter_task=%s）" % (n, args.filter_task))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    uv_full = []
    with open(args.out, "w") as f:
        f.write("# it slot frame_idx task uv00 uv01 uv02 uv10 uv11 uv12"
                "  (active 索引域；pregen seed=%d)\n" % args.seed)
        for it in range(args.iters):
            batch_idx = [random.randrange(n) for _ in range(args.batch)]
            uv_a = M.sample_uv_xform(True, "cpu")
            uvl = [float(x) for x in uv_a.numpy().reshape(-1)]
            uv_full.append(uv_a.numpy())
            for slot, fi in enumerate(batch_idx):
                f.write("%d %d %d %d %s\n" % (
                    it, slot, fi, act[fi][0],
                    " ".join("%.6f" % x for x in uvl)))
    torch.save({"uv": uv_full, "filter_task": args.filter_task,
                "seed": args.seed, "n_active": n},
               os.path.splitext(args.out)[0] + "_full.pt")
    print("[genstream] 写出 %s (+_full.pt 全精度 UV)" % args.out)


def train(args):
    device = "cuda"
    random.seed(args.seed)
    assert sha256(POOL) == POOL_SHA
    assert args.alpha_action == 0.0 and args.alpha_feature == 10.0
    _was_clean = M.TR.patch_status() == "clean"
    M.TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=M.CKPT, model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    base.check_unnorm_key(cfg, model)
    model_input_size = base.get_image_resize_size(cfg)
    action_head, proprio_projector, noisy_action_projector = \
        M._build_action_modules(cfg, model)
    sm, ss, dm, ds = M._norm_consts(device)

    pool = torch.load(POOL, map_location="cpu", weights_only=False, mmap=True)
    act, gids = active_frames(pool, args.filter_task)
    n_active = len(act)
    print(f"[pool] 活动帧 {n_active}"
          + (f"（filter_task={args.filter_task}）" if args.filter_task is not None
             else "（全池）"))

    mr_hidden = None
    if args.ref == "mr" or args.ref_switch > 0:
        assert sha256(MR_CACHE) == CACHE_SHA
        full = torch.load(MR_CACHE, map_location="cpu", weights_only=False, mmap=True)["hidden"]
        assert len(full) == 296
        # 按全局 ID 取子集：active[i] ↔ full[gids[i]]
        mr_hidden = [full[g] for g in gids]
        print(f"[train-mr] MR 参考缓存子集 {len(mr_hidden)} 帧"
              f"（全局ID {gids[0]}..{gids[-1]}，按 frame_map 对齐）")

    stream = None
    full_uv = None
    fresh_stream = None
    if args.stream == "fresh":
        # 全量预生成流（random 消耗顺序与逐 iter 生成逐位相同）：
        # 支持断点续训跳过前 N iter 而不改变后续采样
        fresh_stream = []
        for _it in range(args.iters):
            _b = [random.randrange(n_active) for _ in range(args.batch)]
            _u = M.sample_uv_xform(args.uv == "on", "cpu")                 if args.uv == "on" else None
            fresh_stream.append((_b, _u))
    if args.stream == "replay":
        stream = parse_stream(args.stream_file, args.iters, args.batch)
        pt = os.path.splitext(args.stream_file)[0] + "_full.pt"
        if os.path.isfile(pt) and args.uv == "on":
            full_uv = load_full_uv(pt, args.iters)
            print(f"[stream] replay {args.stream_file} + 全精度 UV {pt}")
        else:
            print(f"[stream] replay {args.stream_file}"
                  + ("（UV 用文本 6 位小数重放，浮点位级不主张）"
                     if args.uv == "on" else "（UV=恒等）"))

    renderer = M.ClothRendererOV(device=device, epsilon=args.epsilon,
                                 init_std=args.init_std,
                                 seed=args.seed).to(device)
    t0_tex = renderer.orig_texture
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"train_{stamp}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("entry=attack_oft_cloth_ablate.py（循环=生产配方逐行同数学）\n")
        if args.stream == "replay":
            f.write("stream_file=%s sha256=%s\n" % (args.stream_file,
                                                    sha256(args.stream_file)))
    log_path = os.path.join(out_dir, "gradient_log.txt")
    torch.save(renderer.adv_noise.detach().cpu(),
               os.path.join(out_dir, "theta_init.pt"))
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_init.pt"))
    with open(os.path.join(out_dir, "frame_map.txt"), "w") as f:
        f.write("# active_idx global_frame_id task local_idx_in_task\n")
        loc = {}
        for ai, (tid, _fr) in enumerate(act):
            loc.setdefault(tid, 0)
            f.write("%d %d %d %d\n" % (ai, gids[ai], tid, loc[tid]))
            loc[tid] += 1

    param = renderer.adv_noise
    backward_inputs = (param,) if getattr(args, "texture_only_grad", False) else None
    resume_from = 0
    if getattr(args, "resume_dir", ""):
        import glob as _glob
        cands = sorted(_glob.glob(os.path.join(args.resume_dir,
                                               "Ep0_Cloth_Delta_OV_it*.pt")),
                       key=lambda x: int(re.search(r"_it(\d+)\.pt$",
                                                   x).group(1)))
        assert cands, "resume-dir 无步数 checkpoint"
        last = getattr(args, 'resume_checkpoint', '') or cands[-1]
        assert os.path.dirname(last) == args.resume_dir
        it_saved = int(re.search(r"_it(\d+)\.pt$", last).group(1))
        with torch.no_grad():
            param.copy_(torch.load(last, map_location=device,
                                   weights_only=False).to(device))
        resume_from = it_saved + 1
        print(f"[resume] 从 {os.path.basename(last)} 恢复 θ，"
              f"自 it{resume_from:05d} 续跑（原计划 {args.iters} iters）")
    fresh_uv_full = []   # fresh 模式：全精度 UV 流（供成对臂 replay）
    samp_log = open(os.path.join(out_dir, "sampling_log.txt"), "w")
    samp_log.write("# it slot frame_idx(active) task uv00..uv12"
                   "  (uv 六零=恒等哨兵：uv_xform=None)\n")
    print(f"[train] {args.iters} iters × bs{args.batch}，ref={args.ref}，"
          f"stream={args.stream}，uv={args.uv}，开始")
    resource_ready()
    t_start = time.time()

    def save_ckpt(it, final=False):
        suffix = f"_it{it:05d}" if args.ckpt_step_suffix else ""
        png = renderer.export_texture()
        Image.fromarray(png).save(
            os.path.join(out_dir, "Universal_Cloth_OV%s.png" % suffix))
        torch.save(param.detach().cpu(),
                   os.path.join(out_dir, "Ep0_Cloth_Delta_OV%s.pt" % suffix))
        if final:  # 终点双写规范名（job_done/tex_ready/评测链固定取规范名）
            Image.fromarray(png).save(
                os.path.join(out_dir, "Universal_Cloth_OV.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
        with open(log_path, "a") as f:
            f.write("[ckpt%s] it%05d frames_accessed=%d\n"
                    % ("-final" if final else "", it, (it + 1) * args.batch))

    for it in range(resume_from, args.iters):
        if stream is not None:
            batch_idx = stream[it]["frames"]
            if args.uv == "on":
                if full_uv is not None:
                    uv_a = torch.from_numpy(full_uv[it]).to(device)
                else:
                    assert stream[it]["uv"] is not None,                         "uv=on 但流文件该行 UV 为零哨兵（源臂为 UV-off？）"
                    uv_a = torch.from_numpy(
                        np.asarray(stream[it]["uv"])).to(device)
            else:
                uv_a = None
        else:
            batch_idx, uv_a = fresh_stream[it]
            uv_a = uv_a.to(device) if uv_a is not None else None
        uvl = [float(x) for x in uv_a.cpu().numpy().reshape(-1)] \
            if uv_a is not None else [0.0] * 6
        param.grad = None
        for slot, fi in enumerate(batch_idx):
            samp_log.write("%d %d %d %d %s\n" % (
                it, slot, fi, act[fi][0],
                " ".join("%.6f" % x for x in uvl)))
        if args.ref_switch > 0 and it == args.ref_switch:
            print(f"[ref-switch] it={it} "
                  f"{'mr->native' if args.ref == 'mr' else 'native->mr'}",
                  flush=True)
        use_mr_now = ((args.ref == "mr")
                      != (args.ref_switch > 0 and it >= args.ref_switch))
        attack_sum = 0.0
        feat_val = float("nan")
        for fi in batch_idx:
            tid = act[fi][0]
            fr = frame_to_dev(act[fi][1], device)
            adv_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                          model_rot=fr["rot"], uv_xform=uv_a)
            adv_chw = adv_rgba.permute(0, 3, 1, 2)
            comp = torch.clamp(adv_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                               0.0, 1.0)
            adv_wrist_t = fr.get("wrist")
            if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
                adv_w_rgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                                model_rot=fr["rot"],
                                                uv_xform=uv_a)
                adv_w_chw = adv_w_rgba.permute(0, 3, 1, 2)
                adv_wrist_t = torch.clamp(
                    adv_w_chw * fr["wrist_seg"]
                    + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
            pv = base._compose_oft_pixel_values(
                adv_rgb=comp, model_input_size=model_input_size,
                siglip_mean=sm, siglip_std=ss, dino_mean=dm, dino_std=ds,
                wrist_tensor=adv_wrist_t).to(torch.bfloat16)
            with base.autocast(dtype=torch.bfloat16):
                _, adv_hidden = model.predict_action(
                    input_ids=fr["text_inputs"]["input_ids"],
                    attention_mask=fr["text_inputs"]["attention_mask"],
                    pixel_values=pv,
                    unnorm_key=cfg.unnorm_key,
                    proprio=fr["proprio"],
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector,
                    use_film=getattr(cfg, "use_film", False))
                adv_action = action_head.predict_action(adv_hidden)
            attack_action = -F.mse_loss(adv_action.float(),
                                        fr["clean_action_head"])
            _ref_h = (mr_hidden[fi].to(device)
                      if mr_hidden is not None and use_mr_now
                      else fr["clean_hidden"])
            attack_feat = -F.mse_loss(adv_hidden.float(), _ref_h.float())
            attack = (args.alpha_action * attack_action
                      + args.alpha_feature * attack_feat)
            attack_sum += attack.item()
            feat_val = attack_feat.item()
            (attack / args.batch).backward(inputs=backward_inputs)
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = M.spatial_tv_nhwc(tex - t0_tex)
        (args.lambda_ref * ref + args.lambda_tv * tv).backward(inputs=backward_inputs)
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            assert np.isfinite(g) and g > 0, (it, g)
            line = (f"[train] it{it:05d} bs{args.batch} "
                    f"attack={attack_sum / args.batch:+.4f}"
                    f" feat={feat_val:+.4f} ref={ref.item():.4f} "
                    f"tv={tv.item():.4f} g={g:.2e} frames_accessed="
                    f"{(it + 1) * args.batch}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            save_ckpt(it, final=(it == args.iters - 1))
        if pause_requested():
            if (it + 1) % args.checkpoint_every and it != args.iters - 1:
                save_ckpt(it)
            samp_log.close()
            print('[resource-pause] saved full theta at it%05d' % it, flush=True)
            return
    samp_log.close()
    if stream is None:
        fresh_uv_full = [((u.numpy() if hasattr(u, "numpy") else u)
                          if u is not None else None)
                         for _, u in fresh_stream]
        torch.save({"uv": fresh_uv_full, "filter_task": args.filter_task,
                    "seed": args.seed, "n_active": n_active},
                   os.path.join(out_dir, "stream_full.pt"))
    delta = (torch.tanh(param.detach()) * args.epsilon) * 255.0
    tex = renderer.texture_tensor().detach()
    d255 = (tex - t0_tex).abs() * 255.0
    stats = (f"[final] delta RMS={float(torch.sqrt((delta ** 2).mean())):.4f} "
             f"Linf={float(delta.abs().max()):.1f} "
             f"TV(0-1)={float(M.spatial_tv_nhwc(tex - t0_tex)):.4f} "
             f"clip后像素域 RMS={float(torch.sqrt((d255 ** 2).mean())):.4f} "
             f"Linf={float(d255.max()):.1f} 用时={time.time() - t_start:.0f}s")
    with open(log_path, "a") as f:
        f.write(stats + "\n")
    print("[train] " + stats)
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_final.pt"))
    if _was_clean:
        M.TR.ensure_patch(False)
    print(f"[train] 完成，产物在 {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genstream")
    g.add_argument("--filter-task", type=int, default=None)
    g.add_argument("--iters", type=int, default=5000)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--ref", choices=["mr", "native"], required=True)
    p.add_argument("--ref-switch", type=int, default=0,
                   help=">0 时在该 iter 切换参考（mr<->native）；θ/RNG/采样流/优化器均不动")
    p.add_argument("--stream", choices=["fresh", "replay"], default="fresh")
    p.add_argument("--stream-file", default="")
    p.add_argument("--uv", choices=["on", "off"], default="on")
    p.add_argument("--filter-task", type=int, default=None)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--alpha-action", type=float, default=0.0)
    p.add_argument("--alpha-feature", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=625)
    p.add_argument("--ckpt-step-suffix", action="store_true")
    p.add_argument("--texture-only-grad", action="store_true", help="Compute gradients only for the optimized texture; leave frozen network flags and forwards unchanged")
    p.add_argument("--resume-dir", default="",
                   help="断点续训：从该目录最近 it*.pt 恢复 θ 并续跑（重启容错）")
    args = ap.parse_args()
    if args.cmd == "genstream":
        cmd_genstream(args)
    else:
        if args.stream == "replay":
            assert args.stream_file and os.path.isfile(args.stream_file)
        train(args)


if __name__ == "__main__":
    main()
