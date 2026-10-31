# -*- coding: utf-8 -*-
"""train_regress.py — Gate A §4.6 基线恒等回归。

三步：
  1. 流一致性：fixed_streams_seed7.pt 前 N iter 与 tvfix-mr 生产
     sampling_log.txt（前 N×8 行）帧序/UV（6dp）逐条相等。
  2. 生产参照：attack_oft_cloth_ablate.py --ref mr --stream replay
     （同一文本流）跑 30 iters（真实生产代码路径）。
  3. 本包 matched（rank0/beta0，stream_text 同源）跑 30 iters：
     θ 终点逐位对照 + 每 iter dump 的 θ 轨迹范数对照 + it0 loss 对照。
通过判据：max|Δθ| ≤ 1e-5（逐位相同时额外登记 exact=True）。
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

D = "/data/shaozhen.liu"
LIB = f"{D}/code/tex3d/openvla-oft/experiments/robot/libero"
PY = f"{D}/envs/tex3d-oft-mj330/bin/python"
TVFIX_MR_LOG = (f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/"
                "train_20261021_224922_tvfix-mr/sampling_log.txt")
STREAMS = (f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/"
           "fixed_streams_seed7.pt")


def check_stream_consistency(iters=30, batch=8):
    from .train_residual import load_stream_pt, load_stream_text
    pt = load_stream_pt(STREAMS, iters)
    tx = load_stream_text(TVFIX_MR_LOG, iters, batch)
    idx_mismatch, uv_mismatch = 0, 0
    max_uv_diff = 0.0
    for it in range(iters):
        if list(pt[it][0]) != list(tx[it][0]):
            idx_mismatch += 1
        u1 = pt[it][1]
        u2 = tx[it][1]
        if (u1 is None) != (u2 is None):
            uv_mismatch += 1
        elif u1 is not None:
            max_uv_diff = max(max_uv_diff,
                              float((u1 - u2).abs().max()))
    return {"iters": iters, "idx_mismatch_iters": idx_mismatch,
            "uv_presence_mismatch": uv_mismatch,
            "max_uv_absdiff_vs_6dp": max_uv_diff,
            "stream_identical": idx_mismatch == 0 and uv_mismatch == 0
            and max_uv_diff <= 5e-7}


def run_ablate_reference(iters, out_stamp, device_env, log):
    """生产代码路径参照（ablate replay）。返回输出目录。"""
    tag = f"regressA{out_stamp}"
    cmd = [PY, os.path.join(LIB, "attack_oft_cloth_ablate.py"), "train",
           "--tag", tag, "--ref", "mr", "--stream", "replay",
           "--stream-file", TVFIX_MR_LOG, "--uv", "on",
           "--iters", str(iters), "--batch", "8",
           "--checkpoint-every", str(iters)]
    log(f"[regress] ablate: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=LIB, env=device_env, capture_output=True,
                       text=True)
    with open(os.path.join(LIB, "../../..", "refcloth_regress_ablate.log"),
              "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("ablate 参照失败：" + r.stderr[-2000:])
    cands = sorted(glob.glob(
        f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/"
        f"train_*_{tag}"))
    assert cands, "ablate 输出目录未找到"
    return cands[-1]


def run_regression(cfg, ablate_out_hint=None, iters=30, device="cuda"):
    stamp = time.strftime("%H%M%S")
    out_root = cfg.paths.output_root
    reg_dir = os.path.join(out_root, "regression")
    os.makedirs(reg_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(os.path.join(reg_dir, "regress.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    report = {}
    # 1) 流一致性
    report["stream_check"] = check_stream_consistency(iters)
    log(f"[regress] 流一致性: {report['stream_check']}")

    # 2) 本包 matched（stream_text 同源 = 生产文本流）
    from .train_residual import run_train
    mine_dir = os.path.join(reg_dir, f"matched{stamp}")
    dump = run_train(cfg, mine_dir, "matched", seed=7, iters=iters,
                     stream_text=TVFIX_MR_LOG, dump_theta_iters=set(range(iters)),
                     device=device)

    # 3) ablate 生产参照
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",
                                                       "0"),
                "MUJOCO_GL": "egl", "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": f"{D}/code/tex3d:{D}/code/tex3d/openvla-oft",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
    abl_dir = run_ablate_reference(iters, stamp, env, log)
    th_abl = torch.load(os.path.join(abl_dir, "Ep0_Cloth_Delta_OV.pt"),
                        map_location="cpu", weights_only=False)
    th_mine = dump[iters - 1]
    diff = (th_abl.float() - th_mine.float()).abs()
    report["theta_final_maxdiff"] = float(diff.max())
    report["theta_final_exact"] = bool(torch.equal(th_abl, th_mine))
    report["theta_norm_abl"] = float(th_abl.float().norm())
    report["theta_norm_mine"] = float(th_mine.float().norm())
    # it0 loss 对照（日志抓取）
    def _it0_feat(path):
        for line in open(path, encoding="utf-8"):
            m = re.search(r"it00000 .*feat=([+-][0-9.]+)", line)
            if m:
                return float(m.group(1))
        return None
    report["it0_feat_abl"] = _it0_feat(os.path.join(abl_dir,
                                                    "gradient_log.txt"))
    report["it0_Draw_mine"] = None
    for line in open(os.path.join(mine_dir, "gradient_log.txt"),
                     encoding="utf-8"):
        if "it00000" in line and "D_raw=" in line:
            m = re.search(r"D_raw=([0-9.]+)", line)
            if m:
                report["it0_Draw_mine"] = float(m.group(1))
            break
    report["pass"] = bool(report["theta_final_maxdiff"] <= 1e-5
                          and report["stream_check"]["stream_identical"])
    report["ablate_dir"] = abl_dir
    report["mine_dir"] = mine_dir
    with open(os.path.join(reg_dir, "regression_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    log(f"[regress] 结果: max|Δθ|={report['theta_final_maxdiff']:.3e} "
        f"exact={report['theta_final_exact']} pass={report['pass']}")
    return report
