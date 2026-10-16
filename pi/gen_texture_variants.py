#!/usr/bin/env python
"""P2 探针纹理生成器：
  1) 幅度折减：T_new = T0_new + α·(T_long2 − T0_grid)   α ∈ {1.0, 0.5, 0.25}
     （把 long2 的扰动 δ 从旧格纹底图上摘下来，按比例贴到新载体底图上）
  2) 匹配随机对照：平滑随机场，匹配 δ 的平均绝对偏差并 clip 到 ±ε（默认 64/255），
     两个种子；与 long2 δ 同为"平滑低频"族，用于区分背景脆弱性 vs 优化特异性。
用法: python gen_texture_variants.py --out-dir <dir>
"""
import argparse
import os

import numpy as np
from PIL import Image

EX = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth"
LONG2 = f"{EX}/train_20261014_102227_long2/Universal_Cloth.png"
T0_GRID = "/data/shaozhen.liu/code/tex3d-deps/cloth_variants/60x40_grid/cloth_base.png"
STAGE = "/data/shaozhen.liu/code/tex3d-deps/cloth_variants"
EPS = 64.0 / 255.0


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier", default="30x20_solid", help="选定载体标签（决定 T0_new）")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = a.out_dir or os.path.join(EX, "p2_textures")
    os.makedirs(out, exist_ok=True)

    long2 = load(LONG2)
    t0_grid = load(T0_GRID)
    t0_new = load(os.path.join(STAGE, a.carrier, "cloth_base.png"))
    delta = long2 - t0_grid                      # 已训练的对抗扰动 δ
    mad = float(np.abs(delta).mean())
    print(f"[gen] δ 统计: mean|δ|={mad:.4f} ({mad*255:.1f}/255), max|δ|={np.abs(delta).max():.3f}")

    # 1) 幅度折减
    for alpha, name in [(1.0, "a100"), (0.5, "a50"), (0.25, "a25")]:
        tex = np.clip(t0_new + alpha * delta, 0, 1)
        p = os.path.join(out, f"long2_{name}_on_{a.carrier}.png")
        Image.fromarray((tex * 255).astype(np.uint8)).save(p)
        print(f"[gen] {p}  (α={alpha}, 实际 mean|扰动|={np.abs(tex-t0_new).mean()*255:.1f}/255)")

    # 2) 匹配随机对照（平滑低频随机场，匹配 mean|δ|，clip ±ε）
    for seed in (7, 8):
        rng = np.random.default_rng(seed)
        low = rng.normal(0, 1, (8, 8, 1)).astype(np.float32)
        field = np.asarray(
            Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8)[:, :, 0])
            .resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0 - 0.5
        for ch in range(3):
            pass  # 同一随机场三通道共用（保证是"平滑图案"而非彩色噪声）
        field3 = np.repeat(field[:, :, None], 3, axis=2)
        field3 *= rng.choice([-1, 1], size=(1, 1, 3)).astype(np.float32)  # 每通道随机符号
        field3 *= mad / (np.abs(field3).mean() + 1e-9)                    # 匹配 mean|δ|
        tex = np.clip(t0_new + np.clip(field3, -EPS, EPS), 0, 1)
        p = os.path.join(out, f"rand_s{seed}_on_{a.carrier}.png")
        Image.fromarray((tex * 255).astype(np.uint8)).save(p)
        print(f"[gen] {p}  (mean|扰动|={np.abs(tex-t0_new).mean()*255:.1f}/255, clip ±{EPS*255:.0f})")


if __name__ == "__main__":
    main()
