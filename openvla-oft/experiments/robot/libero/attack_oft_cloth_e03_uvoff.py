#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_oft_cloth_e03_uvoff.py — E03：当前 MR 多任务配方的 UV-off 对照训练（第18轮 §8）
身份块：
  与 MR8 生产配方（attack_oft_cloth_tvfix_mr.py@d1004416…，train_20261021_224922_tvfix-mr）
  的全部要素逐项相同：同池 r0f_pool.pt（296 帧，tasks{0,2,6,8}）、同 MR 参考缓存
  mr_plain_cache.pt、α_act=0.0/α_feat=10.0、正确空间 TV λ_tv=λ_ref=0.1、seed7、
  θ0（CPU generator(seed=7)·std0.01）、ε=64/255、lr0.05 sign-PGD、batch8、5000 更新、
  clamp±5、ckpt_every=625、每 iter 攻击梯度 1/8 累积+正则一次。
  本臂的两处（且仅两处）差异（老师 §8.2C/D）：
    1. 帧访问序列不再随机采样，而是逐条重放 MR8 的 sampling_log.txt（40,000 条
       frame ID，it/slot 对齐；任务分布 10817/10785/10780/7618 已静态核验）；
    2. 训练期 UV 变换关闭 = render(uv_xform=None) 既有分支（不施加任何仿射；
       非全零仿射矩阵）。采样日志 uv 列沿用六个 0 的记录哨兵。
  额外登记（老师 §8.2E）：训练前保存 θ0 / python random / torch / numpy RNG 状态、
  全局 frame ID→(task,池内序) 映射、重放源 SHA 断言、终点 δ 统计（×255 口径）。
用法（41，cwd=$D/code/tex3d-ws/g7，tex3d-oft-mj330）：
  python attack_oft_cloth_e03_uvoff.py train --iters 5000 [--smoke-iters 100]
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
import attack_oft_cloth_tvfix_mr as M   # noqa: E402  生产配方入口（只读复用）
import attack_oft as base               # noqa: E402  （经 M 的 sys.path 注入）

REPLAY_LOG = ("/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/"
              "train_20261021_224922_tvfix-mr/sampling_log.txt")
REPLAY_SHA = "e25617c320efdb7bf7afc71f197c9a09120a12cc3a8e038014a650a8fac2bc5f"
POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
MR_CACHE = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/mr_plain_cache.pt"
OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_replay(path, iters, batch):
    """返回 {it: [batch 个 frame_idx（按 slot 排序）]}，并做完整性断言。"""
    stream = {}
    n = 0
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        it, slot, fi, task = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        stream.setdefault(it, {})[slot] = (fi, task)
        n += 1
    assert n == 40000, n
    assert len(stream) == 5000
    for it in range(iters):  # 冒烟只校验前缀；全量校验全部
        d = stream[it]
        assert sorted(d) == list(range(batch)), (it, sorted(d))
    return {it: [stream[it][s][0] for s in range(batch)] for it in range(iters)}


def uv_identity_check():
    """老师§8.2D 实证：render(uv_xform=None) vs 显式单位仿射 [[1,0,0],[0,1,0]]。
    None 分支=不进入仿射重映射代码路径（结构性恒等）；此检查给出像素级佐证。"""
    device = "cuda"
    pool = torch.load(POOL, map_location=device, weights_only=False)
    task_ids = list(pool.keys())
    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]
                  if fr["mvp"] is not None]
    renderer = M.ClothRendererOV(device=device, epsilon=64.0 / 255.0,
                                 init_std=0.01, seed=7).to(device)
    ident = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                         dtype=torch.float32, device=device)
    fr = all_frames[0][1]
    with torch.no_grad():
        a, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                               uv_xform=None)
        b, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                               uv_xform=ident)
    d = (a - b).abs()
    print("[uv-check] None vs 单位仿射: maxdiff=%.3e mean=%.3e 逐位一致=%s"
          % (float(d.max()), float(d.mean()), bool((d == 0).all())))
    # 反向对照：非单位仿射必须产生差异（证明 uv_xform 路径确实生效）
    rot = torch.tensor([[0.0, -1.0, 1.0], [1.0, 0.0, 0.0]],
                       dtype=torch.float32, device=device)
    with torch.no_grad():
        c, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                               uv_xform=rot)
    print("[uv-check] 反向对照(旋转90°): maxdiff=%.3e（应显著>0）"
          % float((a - c).abs().max()))


def train_replay(args):
    device = "cuda"
    random.seed(args.seed)
    assert sha256(REPLAY_LOG) == REPLAY_SHA, "重放源 SHA 不符"
    assert sha256(POOL) == "d03d68591b9b820385d14be44758aefd95c2a63b8d68fc0926413a2eb1b04a93"
    assert sha256(MR_CACHE) == "7ab459156068e23cd338bc33e100fb1113a3a367d8b020a80cdaf7cdb7249856"
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

    pool = torch.load(POOL, map_location=device, weights_only=False)
    task_ids = list(pool.keys())
    assert sorted(task_ids) == [0, 2, 6, 8], task_ids
    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]
                  if fr["mvp"] is not None]
    print(f"[pool] {len(all_frames)} 帧（任务 {sorted(task_ids)}）")
    replay = load_replay(REPLAY_LOG, args.iters, args.batch)

    mr_hidden = torch.load(MR_CACHE, map_location="cpu", weights_only=False)["hidden"]
    assert len(mr_hidden) == len(all_frames), (len(mr_hidden), len(all_frames))
    print(f"[train-mr] 匹配渲染 Plain 参考缓存 {len(mr_hidden)} 帧已载入")

    renderer = M.ClothRendererOV(device=device, epsilon=args.epsilon,
                                 init_std=args.init_std,
                                 seed=args.seed).to(device)
    t0_tex = renderer.orig_texture
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"train_{stamp}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("replay_source=%s sha256=%s\n" % (REPLAY_LOG, REPLAY_SHA))
        f.write("uv=identity(render None branch); 与 MR8 仅差帧流来源与UV两项\n")
    log_path = os.path.join(out_dir, "gradient_log.txt")

    # ---- 老师§8.2E：θ0 / RNG 状态 / 帧映射 全量登记 ----
    torch.save(renderer.adv_noise.detach().cpu(),
               os.path.join(out_dir, "theta_init.pt"))
    torch.save({"python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_init.pt"))
    with open(os.path.join(out_dir, "frame_map.txt"), "w") as f:
        f.write("# global_frame_id task local_idx_in_task\n")
        loc = {tid: 0 for tid in task_ids}
        for fi, (tid, _fr) in enumerate(all_frames):
            f.write("%d %d %d\n" % (fi, tid, loc[tid]))
            loc[tid] += 1

    param = renderer.adv_noise
    samp_log = open(os.path.join(out_dir, "sampling_log.txt"), "w")
    samp_log.write("# it slot frame_idx task uv00 uv01 uv02 uv10 uv11 uv12"
                   "  (uv 六零=恒等哨兵：本臂 uv_xform=None，不施加仿射)\n")
    print(f"[train-e03] 重放 MR8 帧流 × {args.iters} iters，UV=恒等，开始")
    t_start = time.time()
    for it in range(args.iters):
        batch_idx = replay[it]
        param.grad = None
        uvl = [0.0] * 6
        for slot, fi in enumerate(batch_idx):
            samp_log.write("%d %d %d %d %s\n" % (
                it, slot, fi, all_frames[fi][0],
                " ".join("%.6f" % x for x in uvl)))
        attack_sum = 0.0
        feat_val = float("nan")
        for fi in batch_idx:
            tid, fr = all_frames[fi]
            adv_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                          model_rot=fr["rot"], uv_xform=None)
            adv_chw = adv_rgba.permute(0, 3, 1, 2)
            comp = torch.clamp(adv_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                               0.0, 1.0)
            adv_wrist_t = fr.get("wrist")
            if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
                adv_w_rgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                                model_rot=fr["rot"],
                                                uv_xform=None)
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
            _ref_h = mr_hidden[fi].to(device)
            attack_feat = -F.mse_loss(adv_hidden.float(), _ref_h.float())
            attack = (args.alpha_action * attack_action
                      + args.alpha_feature * attack_feat)
            attack_sum += attack.item()
            feat_val = attack_feat.item()
            (attack / args.batch).backward()
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = M.spatial_tv_nhwc(tex - t0_tex)
        (args.lambda_ref * ref + args.lambda_tv * tv).backward()
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            assert np.isfinite(g) and g > 0, (it, g)
            acc = (it + 1) * args.batch
            line = (f"[train-e03] it{it:05d} bs{args.batch} "
                    f"attack={attack_sum / args.batch:+.4f}"
                    f" feat={feat_val:+.4f} ref={ref.item():.4f} "
                    f"tv={tv.item():.4f} g={g:.2e} frames_accessed={acc}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            Image.fromarray(renderer.export_texture()).save(
                os.path.join(out_dir, "Universal_Cloth_OV.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
            with open(log_path, "a") as f:
                f.write(f"[ckpt] it{it:05d} frames_accessed="
                        f"{(it + 1) * args.batch}\n")
    samp_log.close()
    # 终点 δ 统计（×255 口径，PNG 域）
    delta = (torch.tanh(param.detach()) * args.epsilon) * 255.0
    tex = renderer.texture_tensor().detach()
    d255 = (tex - t0_tex).abs() * 255.0
    stats = (f"delta RMS={float(torch.sqrt((delta ** 2).mean())):.4f} "
             f"Linf={float(delta.abs().max()):.1f} "
             f"TV(0-1)={float(M.spatial_tv_nhwc(tex - t0_tex)):.4f} "
             f"clip后像素域 RMS={float(torch.sqrt((d255 ** 2).mean())):.4f} "
             f"Linf={float(d255.max()):.1f} 用时={time.time() - t_start:.0f}s")
    with open(log_path, "a") as f:
        f.write("[final] " + stats + "\n")
    print("[train-e03] " + stats)
    torch.save({"python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
               os.path.join(out_dir, "rng_states_final.pt"))
    if _was_clean:
        M.TR.ensure_patch(False)
    print(f"[train-e03] 完成，产物在 {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--uv-check", action="store_true",
                   help="只做 UV 恒等实证检查后退出")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--alpha-action", type=float, default=0.0)
    p.add_argument("--alpha-feature", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=625)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--tag", default="e03-uvoff")
    args = ap.parse_args()
    if args.cmd == "train":
        if args.uv_check:
            uv_identity_check()
            return
        assert args.alpha_action == 0.0 and args.alpha_feature == 10.0
        train_replay(args)


if __name__ == "__main__":
    main()
