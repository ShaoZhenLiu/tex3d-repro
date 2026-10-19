#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R4 对照纹理生成（导师 §7）：随机相位频谱匹配 + UV 块打乱，从 A1-seed7 派生。

- 频谱匹配随机相位：取 A1 纹理扰动的 FFT 幅度谱，随机相位重构（固定 seed），
  RMS 重标定 + L∞ 重约束到 ε——尽量保留功率谱/RMS，破坏空间结构。
- 块打乱：把扰动切成 8x8 UV 小块固定重排（保留像素直方图与幅度，破坏空间排列）。
生成后重测实际统计（mean|δ|/RMS/L∞/TV）并如实报告与 A1 的误差。
"""
import glob
import json

import numpy as np
from PIL import Image

BASE = '/data/shaozhen.liu/code/tex3d-deps/cloth_variants/60x40_solid/cloth_base.png'
A1 = sorted(glob.glob('/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/train_*_f4a1/Universal_Cloth.png'))[-1]
OUT = '/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/r4_controls'
EPS = 64.0 / 255.0


def stats(d):
    return dict(mean_abs=float(np.abs(d).mean()),
                rms=float(np.sqrt((d ** 2).mean())),
                linf=float(np.abs(d).max()),
                tv=float((np.abs(np.diff(d, axis=0)).mean()
                          + np.abs(np.diff(d, axis=1)).mean()) / 2))


base = np.asarray(Image.open(BASE).convert('RGB')).astype(np.float32) / 255.0
a1 = np.asarray(Image.open(A1).convert('RGB')).astype(np.float32) / 255.0
delta = np.clip(a1 - base, -EPS, EPS)
target = stats(delta)
print('A1-seed7 delta stats:', json.dumps(target, indent=1))

import os
os.makedirs(OUT, exist_ok=True)
report = {'target': target, 'controls': {}}

for seed in (201, 202):
    rng = np.random.RandomState(seed)
    F = np.fft.fft2(delta, axes=(0, 1))
    mag = np.abs(F)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, F.shape))
    rec = np.real(np.fft.ifft2(mag * phase, axes=(0, 1)))
    rec = rec / (np.sqrt((rec ** 2).mean()) + 1e-8) * target['rms']
    rec = np.clip(rec, -EPS, EPS)
    tex = np.clip(base + rec, 0, 1)
    Image.fromarray((tex * 255).round().astype(np.uint8)).save(f'{OUT}/specrand_s{seed}.png')
    report['controls'][f'specrand_{seed}'] = stats(tex - base)

for seed in (203, 204):
    rng = np.random.RandomState(seed)
    B = 8
    H, W = delta.shape[:2]
    hb, wb = H // B, W // B
    shuffled = np.zeros_like(delta)
    order = rng.permutation(B * B)
    for i, src in enumerate(order):
        r0, c0 = (src // B) * hb, (src % B) * wb
        d0, e0 = (i // B) * hb, (i % B) * wb
        shuffled[d0:d0 + hb, e0:e0 + wb] = delta[r0:r0 + hb, c0:c0 + wb]
    tex = np.clip(base + shuffled, 0, 1)
    Image.fromarray((tex * 255).round().astype(np.uint8)).save(f'{OUT}/shuffle_s{seed}.png')
    report['controls'][f'shuffle_{seed}'] = stats(tex - base)

for k, v in report['controls'].items():
    print(f'{k}: rms={v["rms"]:.4f}(Δ{v["rms"]-target["rms"]:+.4f}) '
          f'tv={v["tv"]:.5f}(Δ{v["tv"]-target["tv"]:+.5f}) linf={v["linf"]:.4f}')
json.dump(report, open(f'{OUT}/stats.json', 'w'), indent=1)
print('saved ->', OUT)
