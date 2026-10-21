#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tv_sign_check.py — 补充符号验证（第13轮阶段A）：连续梯度（非 sign）短步下降
固定帧/UV，只对 0.1·ref + λ_TV·TV_new 做 15 步 θ -= η·grad（无 sign），
TV 应单调下降——排除 sign-PGD ±lr 振荡对"更新方向抑制 TV"判定的干扰。"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import transfer_pi2ov as TR                                  # noqa: E402
from attack_oft_tvfix_diag import (POOL, spatial_tv_nhwc,    # noqa: E402
                                   ClothRendererOV, sample_uv_xform)

device = "cuda"
TR.ensure_patch(True)
pool = torch.load(POOL, map_location=device, weights_only=False)
fr = None
for tid in sorted(pool.keys()):
    frs = [f for f in pool[tid] if f["mvp"] is not None]
    if frs:
        fr = frs[len(frs) // 2]
        break
renderer = ClothRendererOV(device=device, epsilon=64.0 / 255.0,
                           init_std=1.5, seed=7).to(device)   # 粗糙 init：从高 TV 出发验证下降
t0 = renderer.orig_texture
param = renderer.adv_noise
torch.manual_seed(11)
uv = sample_uv_xform(True, device)
print("[sign-check] 纯 TV sign 下降（粗糙 init_std=1.5 起点，10 步，无 ref/attack）：")
prev = None
mono = True
for st in range(10):
    param.grad = None
    tex = renderer.texture_tensor()
    tv = spatial_tv_nhwc(tex - t0)
    tv.backward()
    with torch.no_grad():
        param.sub_(0.05 * param.grad.sign())
        param.clamp_(-5.0, 5.0)
    with torch.no_grad():
        tvv = spatial_tv_nhwc(renderer.texture_tensor() - t0).item() * 255.0
    flag = ""
    if prev is not None:
        if tvv > prev + 1e-9:
            mono = False
            flag = " ↑"
        else:
            flag = " ↓"
    print("  step %02d  TV=%.4f%s" % (st, tvv, flag))
    prev = tvv
print("[sign-check] 判定: TV 序列%s" % ("单调下降 ✓（更新方向抑制 TV 得证）"
                                      if mono else "非单调 ✗"))
