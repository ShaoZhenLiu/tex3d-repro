"""全套件桌布多视角参考图渲染（QA 用）。

对 libero_10 / spatial / object / goal 每个任务渲染
agentview + sideview（铺素色布，射线网格对齐），
按套件拼成 10 行 × 2 列带标注的对照总表。

用法: python render_all_suites.py  (GPU 环境变量由外层提供)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/pi")
import attack_pi0_cloth as C  # noqa: E402
from attack_pi0 import get_libero_dummy_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402

OUT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/suite_sheets"
SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
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
    pose = sim.model.body_pos[bid].copy()
    if bid == -1:
        C._close_libero_env(env)
        return None, None, "NO CLOTH BODY"
    tt = C.find_target_texture_id(sim, bid, C.CLOTH_SEARCH)
    ri = C.inject_mujoco_texture(sim, tt, np.array(Image.open(C.CLOTH_T0).convert("RGB")))
    obs = env.set_init_state(init)
    for _ in range(10):
        env.step(get_libero_dummy_action())
    imgs = [C._render_now(env, 256, camera_name=c) for c in CAMERAS]
    C.restore_mujoco_texture(sim, tt, *ri)
    C._close_libero_env(env)
    return imgs[0], imgs[1], f"z={pose[2]:.2f} xy=({pose[0]:.2f},{pose[1]:.2f})"


def main():
    os.makedirs(OUT, exist_ok=True)
    C.ensure_patch(True)
    W = 240
    for suite in SUITES:
        rows = []
        for tid in range(10):
            a, s_, info = render_task(suite, tid)
            if a is None:
                a = np.zeros((256, 256, 3), np.uint8)
                s_ = a
            rows.append((tid, a, s_, info))
            print(f"[sheet] {suite} t{tid}: {info}", flush=True)
        H = 200
        canvas = Image.new("RGB", (W * 2 + 30, (H + 22) * 10 + 30), "white")
        dr = ImageDraw.Draw(canvas)
        dr.text((6, 4), f"{suite}  [left=agentview  right=sideview]  cloth=T0", fill="black")
        for i, (tid, a, s_, info) in enumerate(rows):
            y = 28 + i * (H + 22)
            dr.text((6, y - 14), f"t{tid}  {info}", fill=(180, 0, 0))
            canvas.paste(Image.fromarray(a).resize((W, H)), (6, y))
            canvas.paste(Image.fromarray(s_).resize((W, H)), (W + 18, y))
        path = os.path.join(OUT, f"{suite}.png")
        canvas.save(path)
        print(f"[sheet] saved {path}", flush=True)
    C.ensure_patch(False)


if __name__ == "__main__":
    main()
