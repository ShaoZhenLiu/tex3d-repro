# -*- coding: utf-8 -*-
"""texture_gen.py — 预登记的校准纹理表（计划§5.2）。

结构先验固定：std × family{iid, lowpass} × 2 seeds，全部经同一顶点参数化与
输出函数（tanh(θ)·ε → texture_tensor → export PNG）。
高幅度组 std 由预登记规则从 MR8 终点 PNG 的 RMS（纯像素统计，无模型参与）
选定：candidates={0.45,0.6,0.8,1.2,1.6} 中取 RMS≥0.7×MR_RMS 的最小者。
纹理选择只用像素统计，不用任何模型分数。
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .texutil import png_delta_stats

SEED_TABLE = {
    # fit：3 幅度 × 2 结构 × 2 种子 = 12（std_hi 由规则填入）
    "fit": [("lo", 0.05, "iid", 101), ("lo", 0.05, "iid", 102),
            ("lo", 0.05, "lp", 103), ("lo", 0.05, "lp", 104),
            ("mid", 0.15, "iid", 105), ("mid", 0.15, "iid", 106),
            ("mid", 0.15, "lp", 107), ("mid", 0.15, "lp", 108),
            ("hi", None, "iid", 109), ("hi", None, "iid", 110),
            ("hi", None, "lp", 111), ("hi", None, "lp", 112)],
    "validation": [("mid", 0.10, "iid", 201), ("mid", 0.10, "lp", 202),
                   ("hi", 0.30, "iid", 203), ("hi", 0.30, "lp", 204)],
    "locked_test": [("mid", 0.10, "iid", 301), ("mid", 0.10, "lp", 302),
                    ("hi", 0.30, "iid", 303), ("hi", 0.30, "lp", 304)],
}
HI_CANDIDATES = [0.45, 0.6, 0.8, 1.2, 1.6]


def gen_theta(shape, family, std, seed):
    """隔离 CPU RNG（不消耗训练全局流）。lowpass：33×33 网格 3×3 box×2。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    th = torch.randn(shape, generator=g) * std
    if family == "lp":
        v = int(round(np.sqrt(shape[0])))
        assert v * v == shape[0], f"顶点数 {shape[0]} 非方阵网格"
        c = th.shape[-1]
        grid = th.reshape(1, v, v, c)
        grid = F.avg_pool2d(grid.permute(0, 3, 1, 2), 3, stride=1, padding=1)
        grid = F.avg_pool2d(grid, 3, stride=1, padding=1)
        th = grid.permute(0, 2, 3, 1).reshape(shape[0], c)
        # 平滑后方差缩减 → 重标定到目标 std（保持幅度登记口径）
        th = th * (std / max(float(th.std()), 1e-8))
    return th


def mr8_rms255(renderer_like_png, t0_png):
    from .texutil import png_delta_stats
    return png_delta_stats(renderer_like_png, t0_png)["rms"]


def build_table(ctx, out_dir, mr8_png=None):
    """生成三个 split 的 θ 表 + 部署 PNG + 像素统计。返回表 dict 并落盘。"""
    t0 = ctx.t0_png
    os.makedirs(out_dir, exist_ok=True)
    # 高幅度 std 规则（纯像素）
    hi_std = 0.8
    hi_rule = {"mr8_rms": None, "chosen": 0.8, "basis": "default"}
    if mr8_png is not None:
        mr_rms = mr8_rms255(mr8_png, t0)
        rms_by_std = {}
        for cand in HI_CANDIDATES:
            th = gen_theta(ctx.renderer.adv_noise.shape, "iid", cand, 9901)
            ctx.load_theta(th)
            rms_by_std[cand] = png_delta_stats(ctx.renderer.export_png(), t0)["rms"]
        hi_std = 0.8
        for cand in HI_CANDIDATES:      # 最小的 RMS≥0.7×mr_rms 者
            if rms_by_std[cand] >= 0.7 * mr_rms:
                hi_std = cand
                break
        hi_rule = {"mr8_rms": mr_rms, "rms_by_std": rms_by_std,
                   "chosen": hi_std, "basis": "min cand with rms>=0.7*mr8_rms"}
    table = {}
    for split, rows in SEED_TABLE.items():
        entries = []
        for amp, std, family, seed in rows:
            std_eff = hi_std if std is None else std
            th = gen_theta(ctx.renderer.adv_noise.shape, family, std_eff, seed)
            ctx.load_theta(th)
            png = ctx.renderer.export_png()
            tid = f"{split}_{amp}_{family}_s{seed}"
            Image.fromarray(png).save(os.path.join(out_dir, tid + ".png"))
            torch.save(th, os.path.join(out_dir, tid + "_theta.pt"))
            entries.append({
                "texture_id": tid, "split": split, "amplitude": amp,
                "std": std_eff, "family": family, "seed": seed,
                "png_path": os.path.join(out_dir, tid + ".png"),
                "theta_path": os.path.join(out_dir, tid + "_theta.pt"),
                **png_delta_stats(png, t0)})
        table[split] = entries
    ctx.load_theta("zero")
    meta = {"hi_std_rule": hi_rule, "table": table}
    with open(os.path.join(out_dir, "texture_table.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def load_table(out_dir):
    with open(os.path.join(out_dir, "texture_table.json"),
              encoding="utf-8") as f:
        return json.load(f)
