"""render_r14_views.py — 周报示意图：三种布（Plain T0 / MR / EDPA-TVfix）
在三个代表任务第一帧的实际渲染对比（agentview 256）。
只读注入（渲染后还原 MuJoCo 纹理），不影响任何评测。
用法（41，GPU1 空槽时）：
  cd /data/shaozhen.liu/code/tex3d/pi && CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
  /data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python render_r14_views.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/pi")
import attack_pi0_cloth as C  # noqa: E402
from attack_pi0 import get_libero_dummy_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402

EX = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth"
TEXTURES = [
    ("Plain 基布 T0", C.CLOTH_T0),
    ("MR 主候选 (RMS 37.5)", os.path.join(EX, "oft_cloth",
                                           "train_20261021_224922_tvfix-mr",
                                           "Universal_Cloth_OV.png")),
    ("EDPA-TVfix (RMS 39.0)", os.path.join(EX, "oft_cloth",
                                            "train_20261022_112717_edpatvfix",
                                            "Universal_Cloth_OV.png")),
]
TASKS = [
    ("libero_spatial", 0, "Spatial t0"),
    ("libero_object", 5, "Object t5"),
    ("libero_goal", 5, "Goal t5"),
]
OUT = os.path.join(EX, "r14_render_views.png")


def render_task(suite, tid, tex_path):
    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(tid)
    init = s.get_task_init_states(tid)[0]
    env, _ = C._get_libero_env(task, resolution=256)
    env.reset(); env.set_init_state(init)
    sim = env.env.sim; sim.forward()
    C.align_cloth_to_table(env, floor_mode=(suite == "libero_object"))
    _, bid, _ = C.get_target_model_matrix(env, C.CLOTH_SEARCH)
    if bid == -1:
        C._close_libero_env(env)
        return None, "NO CLOTH BODY"
    tt = C.find_target_texture_id(sim, bid, C.CLOTH_SEARCH)
    ri = C.inject_mujoco_texture(sim, tt,
                                 np.array(Image.open(tex_path).convert("RGB")))
    env.set_init_state(init)
    for _ in range(10):
        env.step(get_libero_dummy_action())
    img = C._render_now(env, 256, camera_name="agentview")
    C.restore_mujoco_texture(sim, tt, *ri)
    C._close_libero_env(env)
    return img, "ok"


def main():
    W = 256
    label_h = 26
    top_h = 30
    grid = Image.new("RGB", (W * len(TEXTURES) + 12 * (len(TEXTURES) + 1),
                             top_h + len(TASKS) * (W + label_h)
                             + 12 * (len(TASKS) + 1)), (24, 24, 24))
    d = ImageDraw.Draw(grid)
    d.text((12, 8), "同一初始状态下的第一帧实际渲染（agentview，评测同管线）",
           fill=(230, 230, 230))
    for j, (tname, _) in enumerate(TEXTURES):
        d.text((12 + j * (W + 12) + 6, top_h), tname, fill=(255, 220, 120))
    for i, (suite, tid, tlabel) in enumerate(TASKS):
        y0 = top_h + 12 + i * (W + label_h + 12)
        d.text((12, y0), tlabel, fill=(160, 220, 255))
        for j, (_, tpath) in enumerate(TEXTURES):
            img, info = render_task(suite, tid, tpath)
            x0 = 12 + j * (W + 12)
            if img is None:
                d.text((x0 + 40, y0 + 120), info, fill=(255, 120, 120))
                continue
            grid.paste(Image.fromarray(img), (x0, y0 + label_h))
            print("[render] %s × %s ok" % (tlabel, TEXTURES[j][0]), flush=True)
    grid.save(OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
