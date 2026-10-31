#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_r24_groot.py — GR00T-N1.7 桌布评测聚合（4条件×4套件×100回合）
从 r24_groot_eval/results/groot_{cond}_{suite}.txt 读逐回合键，输出：
- groot_episode_records.csv（逐回合：condition,suite,task,episode,success）
- groot_agg.json（每格 A/n、宏均值、载具效应/优化增量、配对 McNemar 精确检验）
用法： python agg_r24_groot.py <results_dir> <out_dir>
未满 100 唯一键的格标 status=incomplete，不进入宏均值。
"""
import json
import os
import sys
from math import comb

CONDS = ["orig", "plain", "nat", "mr"]
COND_NAMES = {"orig": "Original", "plain": "Plain", "nat": "NativeRef", "mr": "RefCloth"}
SUITES = [("sp", "spatial"), ("obj", "object"), ("go", "goal"), ("10", "long10")]


def parse(path):
    rec, header = {}, []
    if not os.path.exists(path):
        return rec, header
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(("condition=", "ckpt=", "texture=", "num_denoise")):
                header.append(line)
            if line.startswith("Task:"):
                p = [x.strip() for x in line.split("|")]
                t = int(p[0].split(":")[1])
                e = int(p[1].split(":")[1])
                rec[(t, e)] = p[2].split(":")[1].strip() == "True"
    return rec, header


def mcnemar_exact(a, b):
    keys = sorted(set(a) & set(b))
    x = sum(1 for k in keys if a[k] and not b[k])
    y = sum(1 for k in keys if not a[k] and b[k])
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p


def main():
    res_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    cells, rows, problems = {}, [], []
    for skey, sname in SUITES:
        for cond in CONDS:
            rec, hdr = parse(os.path.join(res_dir, f"groot_{cond}_{skey}.txt"))
            n, uniq = len(rec), len(set(rec))
            ok = (n == 100 and uniq == 100)
            if not ok:
                problems.append(f"groot_{cond}_{skey}: n={n} unique={uniq}")
            cells[(cond, skey)] = rec
            for (t, e), s in sorted(rec.items()):
                rows.append((COND_NAMES[cond], sname, t, e, int(s)))

    with open(os.path.join(out_dir, "groot_episode_records.csv"), "w", encoding="utf-8") as f:
        f.write("condition,suite,task,episode,success\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    agg = {"problems": problems, "cells": {}, "macros": {}, "pairwise": {}}
    for cond in CONDS:
        per = {}
        for skey, sname in SUITES:
            rec = cells[(cond, skey)]
            per[sname] = sum(rec.values()) if len(rec) == 100 else None
        agg["cells"][COND_NAMES[cond]] = per
        vals = [v for v in per.values() if v is not None]
        if len(vals) == 4:
            agg["macros"][COND_NAMES[cond]] = sum(vals) / 4

    # 效应量：clean 能力=orig；载具效应=orig−plain；RefCloth 优化增量=plain−mr 与 nat−mr
    def d(a, b, skey):
        ra, rb = cells.get((a, skey), {}), cells.get((b, skey), {})
        if len(ra) == 100 and len(rb) == 100:
            return sum(ra.values()) - sum(rb.values())
        return None

    eff = {}
    for skey, sname in SUITES:
        eff[sname] = {
            "clean": d("orig", "orig", skey) or (sum(cells[("orig", skey)].values()) if len(cells[("orig", skey)]) == 100 else None),
            "carrier_drop_O-P": d("orig", "plain", skey),
            "refcloth_drop_P-MR": d("plain", "mr", skey),
            "optim_increment_NAT-MR": d("nat", "mr", skey),
        }
    agg["effects"] = eff

    # 配对 McNemar（同初态键）：P vs MR、NAT vs MR、O vs P
    for skey, sname in SUITES:
        agg["pairwise"][sname] = {}
        for a, b, label in [("plain", "mr", "P_vs_MR"), ("nat", "mr", "NAT_vs_MR"),
                            ("orig", "plain", "O_vs_P")]:
            ra, rb = cells[(a, skey)], cells[(b, skey)]
            if len(ra) == 100 and len(rb) == 100:
                x, y, p = mcnemar_exact(ra, rb)
                agg["pairwise"][sname][label] = {"a_only": x, "b_only": y,
                                                 "p_exact": p}

    with open(os.path.join(out_dir, "groot_agg.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(json.dumps({"macros": agg["macros"], "problems": problems},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
