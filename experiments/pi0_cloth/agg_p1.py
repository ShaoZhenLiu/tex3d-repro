#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_p1.py — P1 幅度校准与 C1 开发格聚合（第13轮修正版）
口径按老师 §6.3 统一："左 vs 右"比较中，左强回合=左失败右成功、右强回合=左成功右失败；
断言：左强 − 右强 = 右成功数 − 左成功数。p 为同 task/init 配对双侧精确 McNemar。
（第12轮版本表头计数反写，已纠正；p 因对称性不受影响。）"""
import csv

CSV4 = "交给老师_材料包_20261021/04_数据/episodes_merged_all_v4.csv"
# CSV method → (显示名, suite 限定)；None = 该方法仅一个 suite
ALIAS = {
    "F0": "F0", "B1": "B1", "B2v2": "EDPA", "B3": "UPA",
    "B2v2rms": "EDPA衰减", "B3rms": "UPA衰减", "C1rms": "C1衰减",
}


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


A = {}   # (显示名, suite) -> {(task,ep): adv_success}
P = {}   # (suite,) -> {(task,ep): plain_success}
for row in csv.DictReader(open(CSV4, encoding="utf-8")):
    if row["target"] != "pi05":
        continue
    k = (int(row["task"]), int(row["episode_index"]))
    s = row["suite"]
    if row["cond"] == "A":
        nm = ALIAS.get(row["method"], row["method"])
        if row["method"] == "C1":
            nm = "C1obj" if s == "object" else "C1sp"
        A.setdefault((nm, s), {})[k] = int(row["success"])
    elif row["method"] == "F0" and row["cond"] == "P":
        P.setdefault(s, {})[k] = int(row["success"])


def pair(lm, rm, suite, ln=None, rn=None):
    la, ra = A[(lm, suite)], A[(rm, suite)]
    ks = sorted(la)
    lstrong = sum(1 for k in ks if not la[k] and ra[k])   # 左失败右成功
    rstrong = sum(1 for k in ks if la[k] and not ra[k])   # 左成功右失败
    dl, dr = sum(la[k] for k in ks), sum(ra[k] for k in ks)
    assert lstrong - rstrong == dr - dl, (lm, rm, lstrong, rstrong, dl, dr)
    # 老师口径：ΔG(左−右) = G左 − G右 = (P−A左)−(P−A右) = 右成功数−左成功数 = dr−dl
    print("| %s vs %s | %d/%d | %+.0f | %.3g |" %
          (ln or lm, rn or rm, lstrong, rstrong, 100.0 * (dr - dl) / len(ks),
           mcnemar(lstrong, rstrong)))


print("== P1 配对表（π0.5-Object；老师 §4.2 口径）==")
print("| 左 vs 右 | 左强/右强 | ΔG(左−右) | p |")
print("|---|---|---:|---|")
for lm, rm in [("EDPA", "EDPA衰减"), ("EDPA衰减", "F0"),
               ("UPA", "UPA衰减"), ("UPA衰减", "F0"),
               ("EDPA衰减", "UPA衰减"),
               ("B1", "EDPA衰减"), ("B1", "UPA衰减")]:
    pair(lm, rm, "object")

print()
print("== C1 开发格配对（π0.5）==")
print("| 左 vs 右 | 左强/右强 | ΔG(左−右) | p |")
print("|---|---|---:|---|")
for lm, rm, s in [("C1obj", "F0", "object"), ("C1obj", "EDPA", "object"),
                  ("C1obj", "UPA", "object"),
                  ("C1sp", "F0", "spatial"), ("C1sp", "EDPA", "spatial"),
                  ("C1sp", "UPA", "spatial"),
                  ("C1obj", "C1衰减", "object"),
                  ("C1衰减", "EDPA衰减", "object")]:
    pair(lm, rm, s)

print()
print("== 各条件 vs Plain（π0.5）==")
print("| 条件 | suite | A | G | break/fix | p |")
print("|---|---|---:|---:|---|---|")
for (nm, s), a in sorted(A.items()):
    ks = sorted(a)
    p = P[s]
    b = sum(1 for k in ks if p[k] and not a[k])
    f = sum(1 for k in ks if not p[k] and a[k])
    print("| %s | %s | %d | %+.0f | %d/%d | %.2e |" %
          (nm, s, sum(a.values()), 100.0 * (b - f) / len(ks), b, f,
           mcnemar(b, f)))
