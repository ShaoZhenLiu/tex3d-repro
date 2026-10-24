#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""render_deploy_views.py — 图2 部署帧重渲染（第 15 轮 P3，重构 r14_render_views）
三套件（spatial t0 / object t5 / goal t5）× 三条件（素布 T0 / MR / EDPA-TVfix），
同初态（init 10）、同一对齐与注入路径（attack_pi0_cloth 原语）、注入后一步哑动作、
保存 256 agentview PNG。无策略加载（仅渲染）。"""
import os

import numpy as np
from PIL import Image

import attack_pi0_cloth as APC
from attack_pi0_cloth import (
    find_target_texture_id, get_target_model_matrix, inject_mujoco_texture,
    restore_mujoco_texture)

D = "/data/shaozhen.liu"
OUT = os.path.join(D, "code/tex3d/experiments/branch_cases/deploy_views")
TEX = {
    "plain": APC.CLOTH_T0,
    "mr": os.path.join(D, "code/tex3d/experiments/pi0_cloth/oft_cloth/"
                          "train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"),
    "etvf": os.path.join(D, "code/tex3d/experiments/pi0_cloth/oft_cloth/"
                            "train_20261022_112717_edpatvfix/Universal_Cloth_OV.png"),
}
ROWS = [("spatial", 0), ("object", 5), ("goal", 5)]


def main():
    os.makedirs(OUT, exist_ok=True)
    APC.ensure_patch(True)
    from libero.libero import benchmark
    for suite_name, tid in ROWS:
        suite = benchmark.get_benchmark_dict()["libero_" + suite_name]()
        task = suite.get_task(tid)
        init = suite.get_task_init_states(tid)[0]
        floor_mode = suite_name == "object"
        for cond, tex_path in TEX.items():
            arr = np.array(Image.open(tex_path).convert("RGB"))
            env, _ = APC._get_libero_env(task, resolution=256)
            env.reset()
            env.set_init_state(init)
            sim = env.env.sim
            sim.forward()
            APC.align_cloth_to_table(env, floor_mode=floor_mode)
            _, bid, _ = get_target_model_matrix(env, APC.CLOTH_SEARCH)
            tex_id = find_target_texture_id(sim, bid, APC.CLOTH_SEARCH)
            restore = inject_mujoco_texture(sim, tex_id, arr)
            try:
                obs, _, _, _ = env.step(APC.get_libero_dummy_action())
                img = APC.get_libero_image(obs, 256)
                p = os.path.join(OUT, "%s_t%d_%s.png" % (suite_name, tid, cond))
                Image.fromarray(img).save(p)
                print("saved", p, flush=True)
            finally:
                restore_mujoco_texture(sim, tex_id, *restore)
                APC._close_libero_env(env)
    print("DONE-DEPLOY-VIEWS", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
