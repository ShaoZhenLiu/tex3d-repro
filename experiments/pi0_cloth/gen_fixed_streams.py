#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_fixed_streams.py — 预固定 C1 的采样流（第12轮 C1，41 上 CPU 运行）
复刻 attack_oft_cloth_edpa_v2.py 在 seed7 下的精确 RNG 调用序列：
每 iter = 8×random.choice(all_frames) + 1×sample_uv_xform（uniform→[random→random]→uniform×2）
产出 fixed_streams_seed7.pt：[{idx:[8], uv:tensor(2,3)} × 5000]。
C1 与 B2v2 的唯一差异因此锁定为 L_patch 的特征接口（老师 §4 P2）。"""
import os
import random

import numpy as np
import torch

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
OUT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/fixed_streams_seed7.pt"
SEED, ITERS, BATCH = 7, 5000, 8

pool = torch.load(POOL, map_location="cpu", weights_only=False)
all_frames = [(tid, fr) for tid in pool.keys() for fr in pool[tid]
              if fr["mvp"] is not None]
n = len(all_frames)
print("[streams] 池帧 %d（v2 同构 all_frames 顺序：pool.keys() 插入序）" % n)

random.seed(SEED)
dummy = list(range(n))
streams = []
for it in range(ITERS):
    idx = [random.choice(dummy) for _ in range(BATCH)]
    # —— 逐行复刻 v2 的 sample_uv_xform(uv_eot=True) ——
    ang = random.uniform(-np.pi, np.pi)
    c, s_ = float(np.cos(ang)), float(np.sin(ang))
    A = np.array([[c, -s_, 0.5 - 0.5 * c + 0.5 * s_],
                  [s_, c, 0.5 - 0.5 * s_ - 0.5 * c]], dtype=np.float32)
    if random.random() < 0.5:
        if random.random() < 0.5:
            M = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], np.float32)
        else:
            M = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]], np.float32)
        A = (np.vstack([M, [0.0, 0.0, 1.0]])
             @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
    A[0, 2] += random.uniform(-0.04, 0.04)
    A[1, 2] += random.uniform(-0.04, 0.04)
    streams.append({"idx": idx, "uv": torch.from_numpy(A.copy())})

torch.save(streams, OUT)
print("[streams] 写出 %d 条 → %s（%.1f MB）" %
      (len(streams), OUT, os.path.getsize(OUT) / 1e6))
