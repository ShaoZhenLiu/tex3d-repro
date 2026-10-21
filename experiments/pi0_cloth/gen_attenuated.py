#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_attenuated.py — P1 幅度校准：把部署 PNG 线性衰减到 F0 实测 RMS（第12轮 P1）
T'_m = T0 + s_m·(T_m − T0)，s_m = R_F0 / R_m。保持图案布局与几何，只缩幅度。
8-bit 导出（母纹理不覆盖），按统一 PNG 口径复测 RMS/L∞/TV 并记量化误差。
RMS = sqrt(mean((T−T0)^2))（0–255 单位，全通道全像素）；
TV  = 横向相邻差绝对值均值 + 纵向相邻差绝对值均值（各方向对所有通道平均）——老师口径。"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

ASSET = "交给老师_材料包_20261021/03_图与纹理资产"
OUT = "cloth_work/p1_atten"
T0P = os.path.join(ASSET, "t0_solid_deployed.png")
F0P = os.path.join(ASSET, "f0_texture.png")
MOTHERS = [("edpav2", os.path.join(ASSET, "b2v2_edpa_fixed_texture.png")),
           ("upa", os.path.join(ASSET, "b3_upa_nopad_texture.png"))]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stats(t, t0):
    d = t - t0
    rms = float(np.sqrt((d ** 2).mean()))
    linf = float(np.abs(d).max())
    tv = float(np.abs(np.diff(t, axis=1)).mean() + np.abs(np.diff(t, axis=0)).mean())
    return rms, linf, tv


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = load(T0P)
    r_f0, linf_f0, tv_f0 = stats(load(F0P), t0)
    print("T0 均值 = %s" % np.round(t0.reshape(-1, 3).mean(axis=0), 2))
    print("F0  基准: RMS %.4f  L∞ %.0f  TV %.4f  sha %s" %
          (r_f0, linf_f0, tv_f0, sha256(F0P)[:16]))
    report = {"target_rms": r_f0, "t0_rgb": t0.reshape(-1, 3).mean(axis=0).tolist(),
              "textures": {}}
    for name, path in MOTHERS:
        tm = load(path)
        r_m, linf_m, tv_m = stats(tm, t0)
        s = r_f0 / r_m
        att = np.clip(np.round(t0 + s * (tm - t0)), 0, 255)
        outp = os.path.join(OUT, "%s_f0rms.png" % name)
        Image.fromarray(att.astype(np.uint8)).save(outp)
        r_a, linf_a, tv_a = stats(att, t0)
        row = dict(mother=path, mother_sha=sha256(path), s=s,
                   mother_stats=dict(rms=r_m, linf=linf_m, tv=tv_m),
                   out=outp, out_sha=sha256(outp),
                   out_stats=dict(rms=r_a, linf=linf_a, tv=tv_a),
                   quant_err_rms=r_a - r_f0)
        report["textures"][name] = row
        print("%-7s 母: RMS %.4f L∞ %.0f TV %.4f  s=%.5f → 衰减后: RMS %.4f "
              "(量化误差 %+.4f)  L∞ %.0f  TV %.4f  sha %s"
              % (name, r_m, linf_m, tv_m, s, r_a, r_a - r_f0, linf_a, tv_a,
                 row["out_sha"][:16]))
    with open(os.path.join(OUT, "atten_stats.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("saved → %s/atten_stats.json" % OUT)


if __name__ == "__main__":
    main()
