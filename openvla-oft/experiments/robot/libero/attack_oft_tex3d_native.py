#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attack_oft_tex3d_native.py — Tex3D 原生对象 3D 纹理基线训练（第 14 轮 P2）
身份块：
  逐字复用（零改动）上游 attack_oft.py@bfd0726c（lab 分支现文件与快照仅差
  EGL 环境变量守卫，攻击逻辑零改动）：
    - DifferentiableRenderer：对象 mesh 顶点色参数化 adv_noise（tanh·ε，ε=64/255）、
      nvdiffrast 光照 + 光度标定（calibrate_lighting）、get_baked_adv_texture
      顶点色烘焙回 UV 纹理；
    - train_adversarial_texture：原生采池（num_train_init_states=10 ×
      train_frames_per_state=1、frame_collect_with_policy=False 哑动作步进）、
      clean hidden/action_head 缓存、联合目标 α_act=1.0/α_feat=10.0、
      批=min(num_frames_to_attack=20, 池)、sign-PGD lr0.05（adv_noise -= lr·grad.sign
      等价于上升 −loss）、attack_iters=5000、seed=7——全部上游默认值；
    - OBJECTS['akita_black_bowl']（libero_spatial task0 黑碗，stable_scanned 资产）。
  本项目适配（逐项披露）：
    1. 源 ckpt = 本项目冻结源 OpenVLA-OFT Spatial step150000（与全部方法同源）；
    2. 不调用上游 eval_libero 主流程——其部署会**覆写共享资产 texture.png**
       （stable_scanned_objects/akita_black_bowl/），将污染并发评测。本 runner
       只调用 train_adversarial_texture（live_test_enabled=False 下训练全程
       零共享资产写入），目标侧部署改用逐回合运行时注入（eval_objtex_pi.py）；
    3. 输出目录/SHA/run_config/零扰动核验登记（bookkeeping 新增）。
  零扰动核验（先于训练，老师 §7-P2）：θ=0 烘焙纹理 vs 原资产 texture.png
  逐像素差——再烘焙（顶点色→UV 光栅化）可能非逐位一致；差值登记，若非零
  则 Adv 部署含再烘焙伪影，评测侧以 θ0 烘焙注入对照隔离。
用法（41，cwd=$D/code/tex3d-ws/g7，tex3d-oft-mj330）：
  python $TRD/attack_oft_tex3d_native.py --iters 5000
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_oft as UP                     # noqa: E402  上游（lab EGL 守卫版）
import attack_oft_cloth_tvfix_mr as M       # noqa: E402  同源 ckpt/动作模块

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/tex3d_native"
OBJ_KEY = "akita_black_bowl"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--suite", default="libero_spatial",
                    help="采帧/任务文本用的套件；cfg.task_suite_name 恒为 "
                         "libero_spatial（源模型与 unnorm 口径不变）")
    args = ap.parse_args()
    device = "cuda"

    cfg = UP.GenerateConfig(
        pretrained_checkpoint=M.CKPT, model_family="openvla",
        task_suite_name="libero_spatial", object_name=OBJ_KEY,
        enable_attack=True, attack_iters=args.iters, attack_lr=0.05,
        alpha_action=1.0, alpha_feature=10.0, num_frames_to_attack=20,
        num_train_init_states=10, train_frames_per_state=1,
        seed=args.seed, task_id=args.task_id)
    model = UP.get_model(cfg)
    UP.check_unnorm_key(cfg, model)
    processor = UP.get_processor(cfg)
    action_head, proprio_projector, noisy_action_projector = \
        M._build_action_modules(cfg, model)

    obj = UP.OBJECTS[OBJ_KEY]
    scale_xyz = UP.parse_mesh_scale(obj["xml"])
    renderer = UP.DifferentiableRenderer(
        mesh_path=obj["mesh"], orig_texture_path=obj["texture"],
        device=device, scale_xyz=scale_xyz).to(device)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, "train_%s_%s" % (stamp, args.tag))
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 零扰动核验（先于训练） ----------
    renderer.reset_texture()          # adv_noise = 0
    baked0 = renderer.get_baked_adv_texture().squeeze(0).cpu().numpy()
    orig = np.array(Image.open(obj["texture"]).convert("RGB"))
    b0 = (np.clip(baked0, 0, 1) * 255).round().astype(np.int16)
    o0 = orig.astype(np.int16)
    if b0.shape != o0.shape:
        img0 = Image.fromarray(b0.astype(np.uint8)).resize(
            (o0.shape[1], o0.shape[0]))
        b0 = np.array(img0).astype(np.int16)
    d = np.abs(b0 - o0)
    rel = float(np.sqrt(((b0 - o0) ** 2).mean()) /
                max(np.sqrt((o0.astype(float) ** 2).mean()), 1e-9))
    cos = float(np.dot(b0.flatten().astype(float), o0.flatten().astype(float)) /
                (np.linalg.norm(b0) * np.linalg.norm(o0) + 1e-9))
    theta0_png = os.path.join(out_dir, "theta0_baked.png")
    Image.fromarray((np.clip(baked0, 0, 1) * 255).astype(np.uint8)).save(
        theta0_png)
    print("[tex3d-native] 零扰动核验: shape=%s rel=%.4f cos=%.6f "
          "maxdiff=%d 逐位一致=%s" % (b0.shape, rel, cos, int(d.max()),
                                      bool((d == 0).all())), flush=True)

    # ---------- 原生训练（live_test_enabled=False → 零共享资产写入） ----------
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    inits = suite.get_task_init_states(args.task_id)
    cfg.use_proprio = False
    dummy_env, train_desc = UP.get_libero_env(task, cfg.model_family,
                                              resolution=256)
    dummy_env.close()
    UP.train_adversarial_texture(
        cfg, model, processor, renderer,
        action_head, proprio_projector, noisy_action_projector,
        inits[0], task, train_desc, out_dir, episode_idx=0,
        search_keywords_list=obj["search"], xml_path=obj["xml"],
        num_iters=args.iters, init_states=inits)

    # ---------- 产物保存与登记 ----------
    with torch.no_grad():
        baked = renderer.get_baked_adv_texture()
    adv_png = os.path.join(out_dir, "akita_bowl_adv_baked.png")
    Image.fromarray((baked.squeeze(0).clamp(0, 1).cpu().numpy() * 255)
                    .astype(np.uint8)).save(adv_png)
    noise_pt = os.path.join(out_dir, "adv_noise.pt")
    torch.save(renderer.adv_noise.detach().cpu(), noise_pt)
    delta = (torch.tanh(renderer.adv_noise.detach()) * renderer.epsilon) * 255.0
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("upstream=attack_oft.py@bfd0726c (lab EGL-guard only)\n")
        f.write("object=%s suite=%s task_id=%d mesh=%s\n"
                % (OBJ_KEY, args.suite, args.task_id, obj["mesh"]))
        f.write("adv_png=%s sha256=%s\n" % (adv_png, sha256(adv_png)))
        f.write("theta0_png=%s sha256=%s\n" % (theta0_png, sha256(theta0_png)))
        f.write("noise_pt sha256=%s\n" % sha256(noise_pt))
        f.write("theta0 check: rel=%.6f cos=%.6f maxdiff=%d bitwise=%s\n"
                % (rel, cos, int(d.max()), bool((d == 0).all())))
        f.write("delta RMS=%.4f Linf=%.1f (0-255)\n"
                % (float(torch.sqrt((delta ** 2).mean())),
                   float(delta.abs().max())))
    print("[tex3d-native] 完成 %s（adv sha=%s）"
          % (out_dir, sha256(adv_png)[:16]), flush=True)


if __name__ == "__main__":
    main()
