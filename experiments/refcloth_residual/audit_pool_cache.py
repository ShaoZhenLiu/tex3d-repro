#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_pool_cache.py — Gate A 前置只读审计（CPU-only，不加载模型权重到 GPU）

产出 experiments/refcloth_residual/audit/pool_cache_audit.json：
  1. r0f_pool.pt / mr_plain_cache.pt / fixed_streams_seed7.pt 的 SHA 与结构
  2. 每帧字段清单 + 张量 dtype/shape（clean_hidden 的 m 维 = 特征坐标基数）
  3. CKPT 身份（config.json 关键开关 + modeling_prismatic.py/action_head sha）
  4. libero_spatial 任务 {0,2,6,8} 的 init state 数量（fit=0/val=1/test=2 可用性）
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

D = "/data/shaozhen.liu"
POOL = f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
MR_CACHE = f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/mr_plain_cache.pt"
STREAMS = f"{D}/code/tex3d/experiments/pi0_cloth/oft_cloth/fixed_streams_seed7.pt"
CKPT_DIR = f"{D}/code/tex3d-ws/g7/oft-spatial-ckpt"
CLOTH_DIR = f"{D}/code/tex3d-deps/libero/libero/libero/assets/tablecloth"
OUT = f"{D}/code/tex3d/experiments/refcloth_residual/audit/pool_cache_audit.json"
POOL_SHA_REF = "d03d68591b9b820385d14be44758aefd95c2a63b8d68fc0926413a2eb1b04a93"
CACHE_SHA_REF = "7ab459156068e23cd338bc33e100fb1113a3a367d8b020a80cdaf7cdb7249856"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def describe(v, depth=0):
    if torch.is_tensor(v):
        return {"type": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, np.ndarray):
        return {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {"type": "dict", "keys": {k: describe(x, depth + 1)
                                         for k, x in list(v.items())[:8]}}
    if hasattr(v, "keys") and not isinstance(v, str):
        return {"type": type(v).__name__,
                "keys": {k: describe(v[k], depth + 1) for k in list(v.keys())[:8]}}
    return {"type": type(v).__name__}


def main():
    report = {}

    report["pool_sha256"] = sha256(POOL)
    report["pool_sha_match"] = report["pool_sha256"] == POOL_SHA_REF
    pool = torch.load(POOL, map_location="cpu", weights_only=False)
    tasks = {}
    first_frame = None
    for tid, frames in pool.items():
        active = [f for f in frames if f["mvp"] is not None]
        wrist_ok = sum(1 for f in active
                       if f.get("wrist_mvp") is not None
                       and f.get("wrist_bg") is not None
                       and f.get("wrist_seg") is not None)
        tasks[str(tid)] = {"n_frames": len(frames), "n_active": len(active),
                           "n_wrist_quad": wrist_ok}
        if first_frame is None and active:
            first_frame = (tid, active[0])
    report["pool_tasks"] = tasks
    report["pool_total_active"] = sum(t["n_active"] for t in tasks.values())
    tid0, fr0 = first_frame
    fields = {}
    for k, v in fr0.items():
        fields[k] = describe(v)
    report["pool_frame_example"] = {"task": int(tid0), "fields": fields}
    hs = fr0["clean_hidden"]
    report["clean_hidden"] = {"shape": list(hs.shape), "dtype": str(hs.dtype),
                              "m_prod": int(np.prod(hs.shape))}
    report["clean_action_head"] = describe(fr0["clean_action_head"])
    # hidden 形状跨任务一致性（所有活动帧）
    shapes = set()
    for tid, frames in pool.items():
        for f in frames:
            if f["mvp"] is not None:
                shapes.add(tuple(f["clean_hidden"].shape))
    report["clean_hidden_shapes_all"] = [list(s) for s in sorted(shapes)]
    # 每任务文本长度（input_ids 形状）——确认 hidden 不随 prompt 变（action-token 特征）
    text_shapes = {}
    for tid, frames in pool.items():
        for f in frames:
            if f["mvp"] is not None:
                ti = f["text_inputs"]
                ids = ti["input_ids"]
                shp = tuple(ids.shape) if torch.is_tensor(ids) else None
                text_shapes.setdefault(str(tid), set()).add(shp)
    report["input_ids_shapes_by_task"] = {
        k: [list(s) if s else None for s in sorted(v)]
        for k, v in text_shapes.items()}

    report["mr_cache_sha256"] = sha256(MR_CACHE)
    report["mr_cache_sha_match"] = report["mr_cache_sha256"] == CACHE_SHA_REF
    mr = torch.load(MR_CACHE, map_location="cpu", weights_only=False)
    report["mr_cache_top_keys"] = list(mr.keys())
    hid = mr["hidden"]
    if isinstance(hid, dict):
        entries = [hid[k] for k in sorted(hid.keys())]
        report["mr_cache"] = {"container": "dict",
                              "key_type": type(next(iter(hid.keys()))).__name__,
                              "len": len(hid),
                              "entry0": describe(entries[0]),
                              "entry_shapes": sorted({tuple(tuple(h.shape))
                                                      for h in entries})}
    else:
        report["mr_cache"] = {"container": "list", "len": len(hid),
                              "entry0": describe(hid[0]),
                              "entry_shapes": sorted({tuple(tuple(h.shape))
                                                      for h in hid[:40]})}

    if os.path.exists(STREAMS):
        st = torch.load(STREAMS, map_location="cpu", weights_only=False)
        if isinstance(st, dict):
            report["streams_keys"] = {k: describe(v) for k, v in st.items()}
        elif isinstance(st, list):
            report["streams_container"] = "list"
            report["streams_len"] = len(st)
            report["streams_elem0"] = describe(st[0])
            if st and isinstance(st[0], (list, tuple)):
                report["streams_elem0_parts"] = [describe(x) for x in st[0]]
        else:
            report["streams_container"] = type(st).__name__

    # CKPT 身份
    cfg_p = os.path.join(CKPT_DIR, "config.json")
    cfg = json.load(open(cfg_p))
    keep = {k: cfg[k] for k in cfg if any(
        s in k.lower() for s in ("l1", "diffusion", "proprio", "image",
                                 "chunk", "llm_dim", "action"))}
    report["ckpt_config_subset"] = keep
    report["ckpt_modeling_sha"] = sha256(os.path.join(CKPT_DIR,
                                                      "modeling_prismatic.py"))
    ah = os.path.join(CKPT_DIR, "action_head--150000_checkpoint.pt")
    report["ckpt_action_head_sha"] = sha256(ah)
    report["ckpt_action_head_size"] = os.path.getsize(ah)

    # 纹理资产
    t0 = Image.open(os.path.join(CLOTH_DIR, "cloth_base.png"))
    report["cloth_t0"] = {"size": t0.size, "mode": t0.mode}
    gm = os.path.join(CLOTH_DIR, "cloth_gain_map.npy")
    if os.path.exists(gm):
        report["gain_map_shape"] = list(np.load(gm).shape)

    # init state 数量（需要 libero；失败不致命）
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        inits = {}
        for tid in (0, 2, 6, 8):
            n = len(suite.get_task_init_states(tid))
            inits[str(tid)] = n
        report["libero_spatial_init_counts"] = inits
    except Exception as e:  # noqa: BLE001
        report["libero_spatial_init_counts"] = f"ERROR {type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:4000])
    print("[audit] saved", OUT)


if __name__ == "__main__":
    sys.exit(main())
