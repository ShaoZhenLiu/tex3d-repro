#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""collect_r18_manifest.py — r18 批次A 身份采集（服务器端运行）
收集 E01/E02 的结果目录、纹理/权重/脚本 SHA、协议行 → identities/manifest.json
用法：python collect_r18_manifest.py > manifest.json （在 41 上，任意 cwd）
"""
import hashlib
import json
import os
import re
import subprocess
import time

D = "/data/shaozhen.liu"
EX = D + "/code/tex3d/experiments/r18_e01e02"
PI0_CKPT = D + "/model/RLinf-Pi0-LIBERO-130-fullshot-SFT/model.safetensors"
PI05_CKPT = D + "/model/RLinf-Pi05-LIBERO-130-fullshot-SFT/model.safetensors"
OFT_CKPT_DIR = D + "/code/tex3d-ws/g7/oft-spatial-ckpt"
N8 = D + "/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png"
MR = D + "/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"
T0_OFT = D + "/code/tex3d-deps/libero/libero/libero/assets/tablecloth/cloth_base.png"
T0_PI = D + "/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth/cloth_base.png"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def last_result_from_log(log):
    res = None
    if os.path.isfile(log):
        for line in open(log, errors="replace"):
            if "[eval] 结果:" in line:
                res = line.split("结果:")[-1].strip()
    return res


def main():
    man = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "batch": "r18-A (E01+E02)",
           "assets": {
               "native8_texture": {"path": N8, "sha256": sha256(N8)},
               "mr_texture": {"path": MR, "sha256": sha256(MR)},
               "plain_t0_oft_tree": {"path": T0_OFT, "sha256": sha256(T0_OFT)},
               "plain_t0_pi_tree": {"path": T0_PI, "sha256": sha256(T0_PI)},
               "pi0_ckpt_safetensors": {"path": PI0_CKPT, "sha256": sha256(PI0_CKPT)},
               "pi05_ckpt_safetensors": {"path": PI05_CKPT, "sha256": sha256(PI05_CKPT)},
               "oft_spatial_ckpt_dir": {
                   "path": OFT_CKPT_DIR,
                   "files": {f: sha256(os.path.join(OFT_CKPT_DIR, f))
                             for f in sorted(os.listdir(OFT_CKPT_DIR))
                             if f.endswith((".safetensors", ".bin", ".pt"))
                             and os.path.isfile(os.path.join(OFT_CKPT_DIR, f))}},
               "scripts": {
                   s: {"path": p, "sha256": sha256(p)} for s, p in {
                       "pi_eval_entry": D + "/code/tex3d/pi/attack_pi0_cloth.py",
                       "pi_wrapper": D + "/code/tex3d/pi/run_cloth330.sh",
                       "oft_eval_entry": D + "/code/tex3d/openvla-oft/experiments/"
                                          "robot/libero/transfer_pi2oft.py",
                       "e01_chain": EX + "/41_r18_e01_n8.sh",
                       "e02_chain": EX + "/41_r18_e02_oft.sh",
                       "agg": EX + "/agg_r18_ref8.py",
                   }.items()},
           },
           "protocol": {
               "e01": {"episodes_per_cell": 100, "tasks": "0-9",
                       "init_states": "10-19", "num_denoise": 10, "replan": 5,
                       "max_steps": 400, "wait_steps": 10, "mujoco": "3.3.0",
                       "query_noise": "independent (CLOTH_QUERY_SEED unset)",
                       "wrapper": "run_cloth330.sh (tex3d-pi0-mj330)"},
               "e02": {"episodes_per_cell": 100, "tasks": "0-9",
                       "init_states": "10-19", "max_steps": 400,
                       "wait_steps": "cfg.num_steps_wait=10",
                       "action_chain": "OFT native (predict_action + action_head, "
                                       "proprio projector, unnorm, gripper "
                                       "normalize+invert)",
                       "env_resolution": 512, "suite_ckpt": "oft-spatial-ckpt "
                       "(Spatial step150000, same as training source)"},
           },
           "runs": {"E01": {}, "E02": {}},
           "r1_reuse_verdict": {
               "historical_R1_O": D + "/code/tex3d/experiments/pi0_cloth/"
                                   "transfer_oft/original_20261018_160343 "
                                   "(99/100, init 0-9, 300 steps)",
               "historical_R1_P": D + "/code/tex3d/experiments/pi0_cloth/"
                                   "transfer_oft/plain_20261019_022907 "
                                   "(99/100, init 0-9, 300 steps)",
               "reused": False,
               "reason": "init-state range (0-9 vs 10-19) and step budget "
                         "(300 vs 400) differ from E02 protocol; all four "
                         "conditions re-run"},
           "git": {}}
    g = subprocess.run(["git", "-C", D + "/code/tex3d", "log", "-1",
                        "--format=%H %ci %s"], stdout=subprocess.PIPE)
    man["git"]["repo_head"] = g.stdout.decode().strip()

    for name in sorted(os.listdir(EX + "/e01_logs")):
        if not name.endswith(".log"):
            continue
        log = os.path.join(EX, "e01_logs", name)
        res = last_result_from_log(log)
        cell = {"log": log, "result": res, "episodes": None}
        if res and os.path.isfile(res):
            cell["episodes"] = sum(1 for l in open(res, errors="replace")
                                   if l.startswith("Task:"))
        man["runs"]["E01"][name[:-4]] = cell
    for name in sorted(os.listdir(EX + "/e02_logs")):
        if not name.endswith(".log"):
            continue
        log = os.path.join(EX, "e02_logs", name)
        res = last_result_from_log(log)
        cell = {"log": log, "result": res, "episodes": None}
        if res and os.path.isfile(res):
            cell["episodes"] = sum(1 for l in open(res, errors="replace")
                                   if l.startswith("Task:"))
        man["runs"]["E02"][name[:-4]] = cell
    print(json.dumps(man, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
