# -*- coding: utf-8 -*-
"""texutil.py — 无重依赖的纹理工具（numpy/torch/PIL，不触发 nvdiffrast/GL）。"""
import hashlib

import numpy as np
import torch


def png_to_tex(png_uint8):
    """部署 PNG（uint8 HWC，export 方向）→ 渲染方向 float 张量 [1,H,W,3]。
    export_texture 做了 flip(dims=[1]) + round 量化；此处翻转对消。"""
    t = torch.from_numpy(np.asarray(png_uint8, dtype=np.float32) / 255.0)
    return t.unsqueeze(0).flip(dims=[1])


def png_delta_stats(png_uint8, t0_png):
    """部署域纹理统计（0-255）：RMS / L∞ / TV。仅像素统计，无模型参与。"""
    a = png_uint8.astype(np.float32)
    b = t0_png.astype(np.float32)
    d = a - b
    tv = float(np.abs(np.diff(d, axis=0)).mean()
               + np.abs(np.diff(d, axis=1)).mean())
    return {"rms": float(np.sqrt((d ** 2).mean())), "linf": float(np.abs(d).max()),
            "tv": tv}


def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
