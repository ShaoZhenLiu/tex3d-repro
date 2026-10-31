# -*- coding: utf-8 -*-
"""streams.py — 成对采样流装载（torch-only，无重依赖）。

stream_pt  : fixed_streams_seed7.pt（5000×{idx:[8], uv:[2,3]|None}，全精度）
stream_text: 生产 sampling_log.txt（it slot fi task uv×6，6 位小数）
"""
import torch


def load_stream_pt(path, iters):
    st = torch.load(path, map_location="cpu", weights_only=False)
    assert len(st) >= iters, (len(st), iters)
    out = []
    for it in range(iters):
        d = st[it]
        uv = d.get("uv") if isinstance(d, dict) else None
        out.append((list(d["idx"]), uv.cpu() if uv is not None else None))
    return out


def load_stream_text(path, iters, batch=8):
    raw = {}
    n = 0
    for line in open(path):
        if line.startswith("#"):
            continue
        if n >= iters * batch:
            break
        p = line.split()
        it, slot, fi = int(p[0]), int(p[1]), int(p[2])
        uv = None
        if len(p) >= 10 and any(x != "0.000000" for x in p[4:10]):
            uv = torch.tensor([[float(p[4]), float(p[5]), float(p[6])],
                               [float(p[7]), float(p[8]), float(p[9])]],
                              dtype=torch.float32)
        raw.setdefault(it, {})[slot] = (fi, uv)
        n += 1
    assert n == iters * batch, ("流长度不足", n, iters * batch)
    return [([raw[it][s][0] for s in range(batch)], raw[it][0][1])
            for it in range(iters)]
