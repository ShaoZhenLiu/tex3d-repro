#!/usr/bin/env python
"""生成桌布资产三件套：tablecloth.obj（33×33 规则网格平面）+ cloth_base.png（素色格纹 T_0）。
物理尺寸 60×40cm（半宽 0.3/0.2），网格顶点 z=0、法向 +z，UV 覆盖 [0,1]²。
用法：python make_cloth_assets.py <输出目录>
"""
import sys
import os

import numpy as np
import trimesh
from PIL import Image

NX, NY = 33, 33              # 顶点数（3,267 参数）
HALF_X, HALF_Y = 0.3, 0.2    # 米（60×40cm 桌布）
TEX_RES = 256
CELLS_U, CELLS_V = 12, 8     # 5cm 方格：u 60/5=12，v 40/5=8（物理正方形）
COL_A = (210, 205, 200)
COL_B = (198, 193, 188)


def build_mesh():
    xs = np.linspace(-HALF_X, HALF_X, NX)
    ys = np.linspace(-HALF_Y, HALF_Y, NY)
    vertices = np.zeros((NX * NY, 3), dtype=np.float64)
    uv = np.zeros((NX * NY, 2), dtype=np.float64)
    k = 0
    for j in range(NY):
        for i in range(NX):
            vertices[k] = (xs[i], ys[j], 0.0)
            uv[k] = ((xs[i] + HALF_X) / (2 * HALF_X), (ys[j] + HALF_Y) / (2 * HALF_Y))
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

    tex = build_base_texture()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    return mesh


def build_base_texture():
    img = Image.new("RGB", (TEX_RES, TEX_RES))
    cu = TEX_RES / CELLS_U
    cv = TEX_RES / CELLS_V
    px = img.load()
    for y in range(TEX_RES):
        cell_v = int(y // cv)
        for x in range(TEX_RES):
            cell_u = int(x // cu)
            px[x, y] = COL_A if (cell_u + cell_v) % 2 == 0 else COL_B
    return img


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    mesh = build_mesh()
    obj_path = os.path.join(out_dir, "tablecloth.obj")
    mesh.export(obj_path)

    tex = build_base_texture()
    png_path = os.path.join(out_dir, "cloth_base.png")
    tex.save(png_path)

    # 回读校验：obj 必须带 UV 且能被 attack 渲染器路径加载
    back = trimesh.load(obj_path, force="mesh")
    has_uv = getattr(back.visual, "uv", None) is not None and len(back.visual.uv) == len(back.vertices)
    uv_min = np.asarray(back.visual.uv).min(axis=0) if has_uv else None
    uv_max = np.asarray(back.visual.uv).max(axis=0) if has_uv else None
    print(f"[assets] mesh: {len(back.vertices)} verts / {len(back.faces)} faces")
    print(f"[assets] uv present={has_uv}, range u[{uv_min[0]:.3f},{uv_max[0]:.3f}] v[{uv_min[1]:.3f},{uv_max[1]:.3f}]")
    print(f"[assets] 尺寸检查: x∈[{back.vertices[:,0].min():+.3f},{back.vertices[:,0].max():+.3f}] "
          f"y∈[{back.vertices[:,1].min():+.3f},{back.vertices[:,1].max():+.3f}] z≡{back.vertices[:,2].max():.4f}")
    print(f"[assets] 写出: {obj_path}\n[assets] 写出: {png_path}")
    assert has_uv, "OBJ 回读丢失 UV，渲染器将无法采样纹理"
    assert abs(back.vertices[:, 0].max() - HALF_X) < 1e-6
    assert abs(back.vertices[:, 1].max() - HALF_Y) < 1e-6


if __name__ == "__main__":
    main()
