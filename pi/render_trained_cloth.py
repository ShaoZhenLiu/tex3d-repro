"""渲染训练后对抗桌布在场景中的实际效果（评测运行期间安全：只读补丁，不还原）。

long2 v5 产物注入 6 个代表任务：long t0/t1/t2（训练集）、t6/t8（保留集）、
spatial t2（跨套件复核臂）。agentview + sideview 双视角拼图。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/pi")
import attack_pi0_cloth as C  # noqa: E402
from attack_pi0 import get_libero_dummy_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402

TEXTURE = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/train_20261014_102227_long2/Universal_Cloth.png"
OUT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/trained_cloth_views"
TASKS = [
    ("libero_10", 0, "long t0 (train)"),
    ("libero_10", 1, "long t1 (train)"),
    ("libero_10", 2, "long t2 (train)"),
    ("libero_10", 6, "long t6 (heldout)"),
    ("libero_10", 8, "long t8 (heldout)"),
    ("libero_spatial", 2, "spatial t2 (cross-suite)"),
]
CAMERAS = ["agentview", "sideview"]


def render_task(suite, tid):
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
        return None, None, "NO CLOTH BODY"
    pose = sim.model.body_pos[bid].copy()
    tt = C.find_target_texture_id(sim, bid, C.CLOTH_SEARCH)
    ri = C.inject_mujoco_texture(sim, tt, np.array(Image.open(TEXTURE).convert("RGB")))
    env.set_init_state(init)
    for _ in range(10):
        env.step(get_libero_dummy_action())
    imgs = [C._render_now(env, 256, camera_name=c) for c in CAMERAS]
    C.restore_mujoco_texture(sim, tt, *ri)
    C._close_libero_env(env)
    return imgs[0], imgs[1], f"z={pose[2]:.2f} xy=({pose[0]:.2f},{pose[1]:.2f})"


def main():
    os.makedirs(OUT, exist_ok=True)
    # 注意：评测链正在运行，场景补丁保持开启 —— 这里不调用 ensure_patch
    W, H = 400, 333
    canvas = Image.new("RGB", (W * 2 + 30, (H + 24) * len(TASKS) + 34), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), "long2 v5 trained Universal_Cloth  [left=agentview  right=sideview]", fill="black")
    for i, (suite, tid, label) in enumerate(TASKS):
        a, s_, info = render_task(suite, tid)
        if a is None:
            a = s_ = np.zeros((256, 256, 3), np.uint8)
        y = 30 + i * (H + 24)
        dr.text((6, y - 14), f"{label}  {info}", fill=(180, 0, 0))
        canvas.paste(Image.fromarray(a).resize((W, H)), (6, y))
        canvas.paste(Image.fromarray(s_).resize((W, H)), (W + 18, y))
        print(f"[trained] {label}: {info}", flush=True)
    path = os.path.join(OUT, "trained_cloth_sheet.png")
    canvas.save(path)
    print(f"[trained] saved {path}", flush=True)


if __name__ == "__main__":
    main()
