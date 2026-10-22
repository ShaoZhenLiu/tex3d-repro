#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_edpa_native2d.py — EDPA 原生 2D patch 基线（第 14 轮评阅 §7-P1）
身份块（老师 §5.3/§6.4）：
  逐字复用（零改动，import 官方克隆 /data/shaozhen.liu/code/EDPA_attack_defense@3b52691）：
    - VLAAttacker.pytorch.EDPA.EDPA：InfoNCE patch 损失 + 对齐损失、EMA(0.9/warmup100)
      归一、(1-α)·patch+α·align（α=0.2）、温度 0.07、梯度上升 sign-PGD 2/255、
      clamp[0,1]、每步每帧随机位置的 [0,1] 全像素方形贴片（非 ε 球）；
    - utils/openvla_oft 的 image_transform（letterbox/resize/双塔 normalize）、
      build_prompt、get_img_embedding（vision_backbone→projector）、get_lang_embedding；
    - 官方默认超参：ratio=0.05、alpha=0.2、iterations=1、max_steps=1200、batch=2、
      step_size=2/255、save_steps=10、camera_view=primary|wrist（官方默认只训主相机，
      评测支持主/腕各挂独立 patch——本脚本跑两遍得到两张原生 patch）。
  本项目适配（原文没有的部分，逐项披露）：
    1. 源模型 = 本项目冻结源 OpenVLA-OFT Spatial step150000（与全部方法同源；
       官方默认 HF hub 四套件 ckpt——同族不同权重，按"同 OFT 源"适配）；
    2. 数据 = 本项目源策略在**未打布补丁**（无桌布）LIBERO 场景的真实 rollout：
       任务 {0,2,6,8}×2 回合、stride10、每回合≤40 帧（与桌布冻结池同采样结构，
       但场景无布——原生 patch 用真实无布观测，老师 §7-P1）；官方用 RLDS 演示集；
    3. image_size=256（我们的观测分辨率）× ratio 0.05 → patch 57×57
       （官方 224→50×50；**比例保持原生**，绝对尺寸随观测分辨率）；
    4. bookkeeping：seed7、run_config.txt、采样日志（官方无显式 seed，披露为新增）；
    5. batch 采样 = 池内均匀随机（官方为 RLDS 流式 shuffle）。
  不做（保持原生身份）：不加桌布渲染器、不加 ε/TV/参考正则、不换优化器、
  不套 33×33 参数化；评测注入 = 官方约定（每回合开头随机采一次位置、回合内固定）。
用法（41，cwd=$D/code/tex3d-ws/g7）：
  采集（一次，两 patch 共用）：... attack_edpa_native2d.py train --collect-only \
      --pool-file /data/.../native_pool.pt
  训练：... attack_edpa_native2d.py train --camera primary|wrist --iters 1200 ...
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

OFFICIAL = "/data/shaozhen.liu/code/EDPA_attack_defense"
for p in (OFFICIAL,):
    if p not in sys.path:
        sys.path.insert(0, p)

import attack_oft as base                            # noqa: E402
import attack_oft_cloth_tvfix_mr as M                # noqa: E402  复用 _build_action_modules 等
from VLAAttacker.pytorch.EDPA import EDPA            # noqa: E402  官方攻击类（逐字）
from utils.openvla_oft import build_prompt           # noqa: E402  官方 prompt 模板

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/edpa_native2d"
TRAIN_TASKS = [0, 2, 6, 8]


def collect_native_pool(cfg, model, processor, action_head, proprio_projector,
                        noisy_action_projector, task, init_state, device,
                        stride=10, max_frames=40, ep_tag=0):
    """无布场景源策略 rollout 采帧（镜像桌布池 collect_frames，去掉全部布处理）。
    每帧只存：主/腕 256 观测（uint8 原图）与任务指令字符串——EDPA 原生攻击
    只需图像+指令，不需要 proprio/动作标签/隐层。"""
    from libero.libero import benchmark  # noqa: F401
    env, task_description = base.get_libero_env(task, cfg.model_family,
                                                resolution=256)
    env.reset()
    obs = env.set_init_state(init_state)
    use_wrist = getattr(cfg, "num_images_in_input", 1) > 1
    model_input_size = base.get_image_resize_size(cfg)
    frames = []
    for t in range(400):
        if len(frames) >= max_frames:
            break
        if t % stride == 0:
            img = base.get_libero_image(obs)
            wnp = base._extract_wrist_image(obs, 256) if use_wrist else None
            frames.append({"img": img, "wrist": wnp,
                           "instruction": task_description})
        img_high = base.get_libero_image(obs)
        img_model = np.array(Image.fromarray(img_high).resize(
            (model_input_size, model_input_size)))
        observation = {"full_image": img_model,
                       "state": np.zeros(8)}
        if use_wrist:
            wnp_m = base._extract_wrist_image(obs, model_input_size)
            if wnp_m is not None:
                observation["wrist_image"] = wnp_m
        executed = base._query_action_with_compatible_signature(
            cfg, model, observation, task_description, processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=getattr(cfg, "use_film", False))
        executed = np.asarray(executed[0] if isinstance(executed, list)
                              else executed, dtype=np.float32)
        executed = base.normalize_gripper_action(executed, binarize=True)
        if cfg.model_family == "openvla":
            executed = base.invert_gripper_action(executed)
        obs, _, done, _ = env.step(executed.tolist())
        if done:
            break
    env.close()
    print("[native-pool] task%s ep%d: +%d 帧" % (task, ep_tag, len(frames)),
          flush=True)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--pool-tasks", default=",".join(map(str, TRAIN_TASKS)))
    ap.add_argument("--pool-episodes", type=int, default=2)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames-ep", type=int, default=40)
    ap.add_argument("--pool-file",
                    default="/data/shaozhen.liu/code/tex3d/experiments/"
                            "edpa_native2d/native_pool.pt")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--load-pool", action="store_true")
    # —— 官方默认超参（cp_openvla_oft.py FinetuneConfig）——
    ap.add_argument("--camera", choices=["primary", "wrist"],
                    default="primary", help="官方 camera_view 语义")
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=1200,
                    help="官方 max_steps（外层 batch 步数）")
    ap.add_argument("--inner-iters", type=int, default=1,
                    help="官方 iterations（每步 PGD 次数）")
    ap.add_argument("--step-size", type=float, default=2 / 255)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = base.GenerateConfig(pretrained_checkpoint=M.CKPT,
                              model_family="openvla")
    model = base.get_model(cfg)
    base.check_unnorm_key(cfg, model)
    processor = base.get_processor(cfg)
    model.eval()
    action_head, proprio_projector, noisy_action_projector = \
        M._build_action_modules(cfg, model)

    if args.load_pool and os.path.exists(args.pool_file):
        pool = torch.load(args.pool_file, map_location="cpu",
                          weights_only=False)
        print("[native-pool] 缓存命中 %d 帧" % sum(len(v) for v in pool.values()))
    else:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        pool = {}
        for tid in [int(x) for x in args.pool_tasks.split(",")]:
            task = suite.get_task(tid)
            inits = suite.get_task_init_states(tid)
            pool[tid] = []
            for ep in range(args.pool_episodes):
                pool[tid].extend(collect_native_pool(
                    cfg, model, processor, action_head, proprio_projector,
                    noisy_action_projector, task, inits[ep], device,
                    stride=args.stride, max_frames=args.max_frames_ep,
                    ep_tag=ep))
        os.makedirs(os.path.dirname(args.pool_file), exist_ok=True)
        torch.save(pool, args.pool_file)
        print("[native-pool] 已保存 %s（%d 帧）"
              % (args.pool_file, sum(len(v) for v in pool.values())))
        if args.collect_only:
            return

    all_frames = [fr for tid in sorted(pool) for fr in pool[tid]]
    # 官方 EDPA 类：绕过 __init__ 的 HF processor/bootstrap（环境装配，非攻击机制），
    # 只手工绑定 __init__ 会设置的属性；generate_one_step/compute_loss/
    # apply_random_perturbation_to_tensor 三个攻击方法**逐字零改动**复用。
    from utils.openvla_oft import (image_transform, get_img_embedding,
                                   get_lang_embedding)
    attacker = EDPA.__new__(EDPA)
    attacker.max_steps = args.iters
    attacker.save_steps = args.save_steps
    attacker.step_size = args.step_size
    attacker.alpha = args.alpha
    attacker.iterations = args.inner_iters
    attacker.image_size = args.image_size
    attacker.perturbation_ratio = args.ratio
    attacker.ema_step = 0
    attacker.ema_warmup = 100
    attacker.ema_patch = 0
    attacker.ema_align = 0
    attacker.ema_decay = 0.9
    attacker.use_film = False
    attacker.cfg = None
    attacker.device_id = device
    attacker.processor = processor
    attacker.image_transform = image_transform
    attacker.get_img_embedding = get_img_embedding
    attacker.get_lang_embedding = get_lang_embedding

    patch_size = int(np.sqrt(args.image_size ** 2 * args.ratio))
    perturbation = torch.zeros((3, patch_size, patch_size),
                               dtype=torch.float32).uniform_(0, 1)
    print("[native2d] camera=%s patch=%d×%d ratio=%.2f steps=%d×batch%d "
          "step=%.5f α=%.2f（官方默认）"
          % (args.camera, patch_size, patch_size, args.ratio, args.iters,
             args.batch, args.step_size, args.alpha), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, "train_%s_cam-%s%s"
                           % (stamp, args.camera,
                              ("_" + args.tag) if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("official=EDPA_attack_defense@3b52691 verbatim EDPA class\n")
    log_path = os.path.join(out_dir, "train_log.txt")
    samp_log = open(os.path.join(out_dir, "sampling_log.txt"), "w")
    samp_log.write("# it frame_idx task\n")

    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    rng = random.Random(args.seed)
    for idx in range(args.iters):
        batch = [all_frames[rng.randrange(len(all_frames))]
                 for _ in range(args.batch)]
        for b_i, fr in enumerate(batch):
            samp_log.write("%d %d %s\n"
                           % (idx, b_i, fr["instruction"][:24]))
        images = torch.stack(
            [to_tensor(fr["img"] if args.camera == "primary" else fr["wrist"])
             for fr in batch])
        prompt = build_prompt([fr["instruction"] for fr in batch])
        instructions, instructions_masks = (
            processor.tokenizer(prompt, padding=True, return_tensors="pt")[k]
            for k in ("input_ids", "attention_mask"))
        perturbation, cost, patch_loss, align_loss = \
            attacker.generate_one_step(model, images, instructions,
                                       instructions_masks, perturbation)
        if idx % 10 == 0 or idx == args.iters - 1:
            line = ("[native2d] it%04d cost=%.4f patch=%.4f align=%.4f"
                    % (idx, cost.item(), float(patch_loss.mean()),
                       float(align_loss.mean())))
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if idx % args.save_steps == 0 or idx == args.iters - 1:
            torch.save(perturbation,
                       os.path.join(out_dir, "perturbation.pt"))
            from torchvision.utils import save_image
            save_image(perturbation,
                       os.path.join(out_dir, "perturbation.png"))
    samp_log.close()
    print("[native2d] 完成，产物在 %s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
