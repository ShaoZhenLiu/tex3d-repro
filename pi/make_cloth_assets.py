#!/usr/bin/env python
"""生成桌布资产三件套：tablecloth.obj（33×33 规则网格平面）+ cloth_base.png（底图 T_0）。

用法:
  python make_cloth_assets.py <输出目录> [--half-x 0.3] [--half-y 0.2]
                              [--style grid|solid|wood] [--cell-cm 5.0] [--seed 7]

尺寸与底图可参数化（P1 载体筛选：3 尺寸 × 3 底图）：
  grid  = 现行格纹（默认，RGB 210/198 双色，格宽 --cell-cm 物理厘米，保持物理正方形）
  solid = 同均值纯色（格纹两色的均值 (204,199,194)，只改图案不改亮度）
  wood  = 程序生成低对比哑光木纹（亮度与 grid 相近，对比度 ±6% 级）
默认参数 = 原版 60×40cm 格纹，行为完全不变。
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from PIL import Image

NX, NY = 33, 33              # 顶点数（3,267 参数；网格物理间距随尺寸变化，见输出提示）
TEX_RES = 256
COL_A = (210, 205, 200)
COL_B = (198, 193, 188)
SOLID = tuple((a + b) // 2 for a, b in zip(COL_A, COL_B))   # (204,199,194)
WOOD_BASE = np.array([200, 184, 162], dtype=np.float32)     # 哑光浅木色，亮度与格纹均值相近


def build_mesh(half_x, half_y, tex):
    xs = np.linspace(-half_x, half_x, NX)
    ys = np.linspace(-half_y, half_y, NY)
    vertices = np.zeros((NX * NY, 3), dtype=np.float64)
    uv = np.zeros((NX * NY, 2), dtype=np.float64)
    k = 0
    for j in range(NY):
        for i in range(NX):
            vertices[k] = (xs[i], ys[j], 0.0)
            uv[k] = ((xs[i] + half_x) / (2 * half_x), (ys[j] + half_y) / (2 * half_y))
            k += 1

    faces = []
    def vid(i, j):
        return j * NX + i
    for j in range(NY - 1):
        for i in range(NX - 1):
            v00, v10, v11, v01 = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.asarray(faces, dtype=np.int64)
    # 双面：复制一份反向绕序的面（MuJoCo 对 mesh 做背面剔除，单面平面可能整个不可见）
    faces = np.concatenate([faces, faces[:, ::-1]], axis=0)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    return mesh


def build_grid_texture(cells_u, cells_v):
    img = Image.new("RGB", (TEX_RES, TEX_RES))
    cu = TEX_RES / cells_u
    cv = TEX_RES / cells_v
    px = img.load()
    for y in range(TEX_RES):
        cell_v = int(y // cv)
        for x in range(TEX_RES):
            cell_u = int(x // cu)
            px[x, y] = COL_A if (cell_u + cell_v) % 2 == 0 else COL_B
    return img


def build_solid_texture():
    return Image.new("RGB", (TEX_RES, TEX_RES), SOLID)


def build_wood_texture(seed=7):
    """低对比哑光木纹：低频随机场调制亮度（±6%）+ 3 条暗纹条带，颜色浅木色。

    频率可控（--seed 不变时确定性），无周期格纹结构——用于检验格纹周期性是否为
    载体损害主因（导师分析 §5/§11 的 t6 诊断）。
    """
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 1, (8, 8)).astype(np.float32)
    low_img = Image.fromarray(
        ((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8)
    ).resize((TEX_RES, TEX_RES), Image.BILINEAR)
    field = np.asarray(low_img, dtype=np.float32) / 255.0 - 0.5   # [-0.5, 0.5]

    yy, xx = np.mgrid[0:TEX_RES, 0:TEX_RES].astype(np.float32)
    u = xx / TEX_RES
    v = yy / TEX_RES
    streaks = np.zeros_like(u)
    for k, phase in enumerate(rng.uniform(0, 2 * np.pi, 3)):
        # 竖向暗纹：沿 u 的低频正弦，带轻微 v 向弯曲（模拟木纹）
        streaks += 0.5 + 0.5 * np.sin(2 * np.pi * (2 + k) * u
                                      + 6.0 * np.sin(2 * np.pi * v + phase)
                                      + phase)
    streaks = (streaks / 3.0 - 0.5)                              # [-0.5, 0.5]

    mod = 0.06 * field + 0.04 * streaks                          # 合计 ±10% 上限，实际更低
    img = np.clip(WOOD_BASE[None, None, :] * (1.0 + mod[..., None]), 0, 255)
    return Image.fromarray(img.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--half-x", type=float, default=0.3, help="半宽（米），0.3=60cm")
    ap.add_argument("--half-y", type=float, default=0.2, help="半深（米），0.2=40cm")
    ap.add_argument("--style", choices=["grid", "solid", "wood"], default="grid")
    ap.add_argument("--cell-cm", type=float, default=5.0, help="格纹物理边长（厘米）")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w_cm, h_cm = args.half_x * 200, args.half_y * 200
    if args.style == "grid":
        cells_u = max(1, round(w_cm / args.cell_cm))
        cells_v = max(1, round(h_cm / args.cell_cm))
        tex = build_grid_texture(cells_u, cells_v)
    elif args.style == "solid":
        cells_u = cells_v = 1
        tex = build_solid_texture()
    else:
        cells_u = cells_v = 0
        tex = build_wood_texture(args.seed)

    mesh = build_mesh(args.half_x, args.half_y, tex)
    obj_path = os.path.join(args.out_dir, "tablecloth.obj")
    mesh.export(obj_path)
    png_path = os.path.join(args.out_dir, "cloth_base.png")
    tex.save(png_path)

    # 回读校验：obj 必须带 UV 且能被 attack 渲染器路径加载
    back = trimesh.load(obj_path, force="mesh")
    has_uv = getattr(back.visual, "uv", None) is not None and len(back.visual.uv) == len(back.vertices)
    uv_min = np.asarray(back.visual.uv).min(axis=0) if has_uv else None
    uv_max = np.asarray(back.visual.uv).max(axis=0) if has_uv else None
    print(f"[assets] {w_cm:.0f}x{h_cm:.0f}cm {args.style}"
          + (f" {cells_u}x{cells_v} 格" if args.style == "grid" else "")
          + f" | 网格间距 {2*args.half_x/(NX-1)*100:.2f}x{2*args.half_y/(NY-1)*100:.2f}cm")
    print(f"[assets] mesh: {len(back.vertices)} verts / {len(back.faces)} faces")
    print(f"[assets] uv present={has_uv}, range u[{uv_min[0]:.3f},{uv_max[0]:.3f}] v[{uv_min[1]:.3f},{uv_max[1]:.3f}]")
    print(f"[assets] 尺寸检查: x∈[{back.vertices[:,0].min():+.3f},{back.vertices[:,0].max():+.3f}] "
          f"y∈[{back.vertices[:,1].min():+.3f},{back.vertices[:,1].max():+.3f}] z≡{back.vertices[:,2].max():.4f}")
    print(f"[assets] 写出: {obj_path}\n[assets] 写出: {png_path}")
    assert has_uv, "OBJ 回读丢失 UV，渲染器将无法采样纹理"
    assert abs(back.vertices[:, 0].max() - args.half_x) < 1e-6
    assert abs(back.vertices[:, 1].max() - args.half_y) < 1e-6


if __name__ == "__main__":
    main()
