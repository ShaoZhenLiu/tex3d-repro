#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_mr800.py — MR 候选完整 800 聚合（第13轮收官，2026-10-22）
口径：左强/右强=左失败右成功/左成功右失败；断言 左强−右强=右成功数−左成功数；
McNemar 精确；Holm-8 一组。Plain 参照取 v5 CSV 的 F0/P 行；EDPA/F0/B1/B3 取 A 行。"""
import csv
import re

CSV5 = "交给老师_材料包_20261021/04_数据/episodes_merged_all_v5.csv"
EP = re.compile(r"Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)")
FILES = {
    ("pi0", "spatial"): "cloth_work/tvfix_raw/mr_sp.txt",   # 占位防错，实际 pi0 用 mr800_raw
    ("pi0", "object"): "cloth_work/mr800_raw/pi0_obj.txt",
    ("pi0", "goal"): "cloth_work/mr800_raw/pi0_go.txt",
    ("pi0", "10"): "cloth_work/mr800_raw/pi0_10.txt",
    ("pi05", "spatial"): "cloth_work/tvfix_raw/mr_sp.txt",
    ("pi05", "object"): "cloth_work/tvfix_raw/mr_obj.txt",
    ("pi05", "goal"): "cloth_work/mr800_raw/pi05_go.txt",
    ("pi05", "10"): "cloth_work/mr800_raw/pi05_10_merged.txt",
}
FILES[("pi0", "spatial")] = "cloth_work/mr800_raw/pi0_sp.txt"


def comb(n, k):
    r = 1
    for i in range(min(k, n - k)):
        r = r * (n - i) // (i + 1)
    return r


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / float(2 ** n))


def parse(p):
    d = {}
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = EP.search(line)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = m.group(3) == "True"
    return d


A, P = {}, {}
for row in csv.DictReader(open(CSV5, encoding="utf-8")):
    if row["cond"] == "A":
        A.setdefault((row["method"], row["target"], row["suite"]), {})[
            (int(row["task"]), int(row["episode_index"]))] = int(row["success"])
    elif row["method"] == "F0" and row["cond"] == "P":
        P.setdefault((row["target"], row["suite"]), {})[
            (int(row["task"]), int(row["episode_index"]))] = int(row["success"])

SU = {"spatial": "Spatial", "object": "Object", "goal": "Goal", "10": "Long"}
MR = {}
for (t, s), path in FILES.items():
    d = parse(path)
    assert len(d) == 100, (t, s, len(d))
    MR[(t, s)] = d

print("== MR 800：vs Plain（G/break-fix/p）==")
print("| 目标 | 套件 | A | G | b/f | p |")
print("|---|---|---:|---:|---|---|")
mac = {"pi0": [], "pi05": []}
for t in ("pi0", "pi05"):
    for s in ("spatial", "object", "goal", "10"):
        a = MR[(t, s)]
        p = P[(t, s)]
        ks = sorted(a)
        b = sum(1 for k in ks if p[k] and not a[k])
        f = sum(1 for k in ks if not p[k] and a[k])
        g = 100.0 * (b - f) / len(ks)
        mac[t].append(g)
        print("| π%s | %s | %d | %+.0f | %d/%d | %.1e |" %
              ("0.5" if t == "pi05" else "0", SU[s], sum(a.values()), g, b, f, mcn(b, f)))
for t in ("pi0", "pi05"):
    print("π%s 宏平均 G = %.2f" % ("0.5" if t == "pi05" else "0", sum(mac[t]) / 4.0))
print("两目标宏平均 = %.2f" % ((sum(mac["pi0"]) + sum(mac["pi05"])) / 8.0))

print()
print("== MR vs 参照（逐格直接配对，Holm-8 按每方法对一组）==")
for ref, rn in [("B2v2", "EDPA"), ("F0", "F0"), ("B1", "B1"), ("B3", "B3")]:
    ps = []
    detail = []
    for t in ("pi0", "pi05"):
        for s in ("spatial", "object", "goal", "10"):
            la = {(k): int(v) for k, v in MR[(t, s)].items()}
            ra = A[(ref, t, s)]
            ks = sorted(la)
            b = sum(1 for k in ks if not la[k] and ra[k])
            c = sum(1 for k in ks if la[k] and not ra[k])
            assert b - c == sum(ra.values()) - sum(la.values())
            ps.append(mcn(b, c))
            detail.append((t, s, b, c, mcn(b, c)))
    m = 8
    order = sorted(range(m), key=lambda i: ps[i])
    hp = [0.0] * m
    for r, i in enumerate(order):
        hp[i] = max(ps[order[j]] * (m - j) for j in range(r + 1))
    ls = rs = raw = 0
    print("-- MR vs %s（MR强/右强回合, ΔG=G_MR−G右, 原始p, Holm-8）:" % rn)
    for (t, s, b, c, pv), h in zip(detail, hp):
        dG = 100.0 * b / 100 - 100.0 * c / 100
        raw += pv < 0.05
        ls += b
        rs += c
        print("| π%s %s | %d/%d | %+.0f | %.3g | %.3g |" %
              ("0.5" if t == "pi05" else "0", SU[s], b, c, dG, pv, min(h, 1.0)))
    sig = 0
    for r, i in enumerate(order):
        if ps[i] <= 0.05 / (m - r):
            sig = r + 1
        else:
            break
    print("   汇总: MR强回合 %d / 右强回合 %d；原始 p<0.05 %d 格，Holm-8 %d 格" %
          (ls, rs, raw, sig))
