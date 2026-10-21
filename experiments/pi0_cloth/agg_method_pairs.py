#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_method_pairs.py — v3 CSV 全方法对×八格直接配对 + 各方法 vs Plain 的规范统计表
一切胜负/计数/显著性表述以此脚本输出为准（第12轮 P0-A）。
py3.6 兼容；McNemar 精确二项双侧；Holm 按每方法对八格一组。"""
import csv
from collections import defaultdict

CSV3 = "交给老师_材料包_20261021/04_数据/episodes_merged_all_v3.csv"
TARGETS = ["pi0", "pi05"]
SUITES = ["spatial", "object", "goal", "10"]
SUITE_NAME = {"spatial": "Spatial", "object": "Object", "goal": "Goal", "10": "Long"}
METHODS = ["F0", "B1", "R0", "B2v2", "B3"]
PAIRS = [("B2v2", "F0"), ("B3", "F0"), ("B2v2", "B1"), ("B3", "B1"), ("B2v2", "B3")]
PAIR_NAME = {("B2v2", "F0"): "EDPA-F0", ("B3", "F0"): "UPA-F0",
             ("B2v2", "B1"): "EDPA-B1", ("B3", "B1"): "UPA-B1",
             ("B2v2", "B3"): "EDPA-UPA"}


def comb(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(min(k, n - k)):
        r = r * (n - i) // (i + 1)
    return r


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    kk = min(b, c)
    p = sum(comb(n, i) for i in range(kk + 1)) / float(2 ** n)
    return min(1.0, 2 * p)


def holm_count(ps, alpha=0.05):
    """返回 Holm 校正后显著格数（逐对一组，m=8）。"""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    sig = 0
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        if ps[i] <= thr:
            sig = rank + 1
        else:
            break
    return sig


# 读 v3：A[method][target][suite][(task, ep)] = success；P/O 只在 F0 名下
A = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
P = defaultdict(dict)
for row in csv.DictReader(open(CSV3, encoding="utf-8")):
    t, s = row["target"], row["suite"]
    k = (int(row["task"]), int(row["episode_index"]))
    if row["cond"] == "A":
        A[row["method"]][t][s][k] = int(row["success"])
    elif row["cond"] == "P" and row["method"] == "F0":
        P[(t, s)][k] = int(row["success"])

for m in METHODS:
    for t in TARGETS:
        for s in SUITES:
            n = len(A[m][t][s])
            assert n == 100, (m, t, s, n)

# ---- 表1：各方法 vs Plain（攻击有效性），八格 ----
print("== 表1  各方法 Adv vs Plain：A 成功数 / G / break/fix / McNemar p ==")
print("| 方法 | 目标 | 套件 | A | P | G | break/fix | p |")
print("|---|---|---|---:|---:|---:|---|---|")
macro = defaultdict(lambda: defaultdict(list))
for m in METHODS:
    for t in TARGETS:
        for s in SUITES:
            a, p = A[m][t][s], P[(t, s)]
            b = sum(1 for k in a if p[k] and not a[k])
            f = sum(1 for k in a if not p[k] and a[k])
            G = 100.0 * (b - f) / len(a)
            macro[m][t].append(G)
            print("| %s | %s | %s | %d | %d | %+.2f | %d/%d | %.2e |" %
                  (m, t, SUITE_NAME[s], sum(a.values()), sum(p.values()),
                   G, b, f, mcnemar(b, f)))
for m in METHODS:
    print("%s 宏平均：π0 %.2f / π0.5 %.2f / 两目标 %.3f" %
          (m, sum(macro[m]["pi0"]) / 4.0, sum(macro[m]["pi05"]) / 4.0,
           (sum(macro[m]["pi0"]) + sum(macro[m]["pi05"])) / 8.0))

# ---- 表2：方法对×八格 Adv 直接配对 ----
print()
print("== 表2  方法对 Adv 逐集直接配对（左强=左失败右成功；Holm 按八格一组）==")
print("| 方法对 | 目标 | 套件 | ΔG(pp) | 左强/右强 | 原始 p | Holm-8 p |")
print("|---|---|---|---:|---|---|---|")
for lm, rm in PAIRS:
    ps, marks = [], []
    detail = []
    for t in TARGETS:
        for s in SUITES:
            la, ra = A[lm][t][s], A[rm][t][s]
            b = sum(1 for k in la if not la[k] and ra[k])   # 左强：左失败右成功
            c = sum(1 for k in la if la[k] and not ra[k])   # 右强
            dG = 100.0 * (sum(ra.values()) - sum(la.values())) / len(la)
            pv = mcnemar(b, c)
            ps.append(pv)
            detail.append((t, s, dG, b, c, pv))
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    holm_p = [None] * m
    for rank, i in enumerate(order):
        holm_p[i] = max(ps[order[j]] * (m - j) for j in range(rank + 1))
    ls = rs = 0
    for (t, s, dG, b, c, pv), hp in zip(detail, holm_p):
        ls += b
        rs += c
        marks.append((t, s, dG, b, c, pv, hp))
    for (t, s, dG, b, c, pv, hp) in marks:
        print("| %s | %s | %s | %+.0f | %d/%d | %.3g | %.3g |" %
              (PAIR_NAME[(lm, rm)], t, SUITE_NAME[s], dG, b, c, pv, min(hp, 1.0)))
    raw_sig = sum(1 for pv in ps if pv < 0.05)
    holm_sig = holm_count(ps)
    weaker = sum(1 for (_, _, dG, _, _, _) in detail if dG < 0)
    print("汇总 %s：左强/平/右强格 %d/%d/%d，合计左强/右强回合 %d/%d，原始 p<0.05 %d 格，Holm-8 后 %d 格" %
          (PAIR_NAME[(lm, rm)], sum(1 for d in detail if d[2] > 0),
           sum(1 for d in detail if d[2] == 0), weaker, ls, rs, raw_sig, holm_sig))
    print()
