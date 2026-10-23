#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_native2d.py — EDPA-native-2D 原生 patch 基线聚合（第 14 轮 P1）
老师 §6.3 统一口径：每格报告
  Original（O，无布无 patch，复用 v6.1 F0 O 行）/ P_method（随机同尺寸 patch）/
  Adv（优化 patch）；总影响 SR(O)−SR(A)、优化增量 SR(P)−SR(A)、载体效应
  SR(O)−SR(P)；Adv vs P_method 直接配对（McNemar 精确）与 Adv vs O 配对。
数据源：41 上 $OUT/eval_logs/nat2d_{cond}_{target}_{suite}.log 指向的
evaluation_result.txt（每格恰 100 行 Task）；O 从 v6.1 CSV 读取。
本地运行：先把 16 份 result 拉到 r14_assets/native2d/logs/。
"""
import csv
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "r14_assets", "native2d", "logs")
CSV = None
for cand in (os.path.join(HERE, "..", "交给老师_材料包_20261022", "04_数据",
                          "episodes_merged_all_v6_1.csv"),):
    if os.path.exists(cand):
        CSV = cand
        break
EP = re.compile(r"Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)")
SUITES = ["spatial", "object", "goal", "10"]
TARGETS = ["pi0", "pi05"]


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
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / float(2 ** n))


ABBR = {"spatial": "sp", "object": "obj", "goal": "go", "10": "10"}


def load_log(name):
    # 扁平结果文件：nat2d_{cond}_{target}_{缩写}_evaluation_result.txt
    parts = name.split("_")
    res = os.path.join(EV, "nat2d_%s_%s_%s_evaluation_result.txt"
                       % (parts[1], parts[2], ABBR[parts[3]]))
    eps = {}
    for line in open(res, encoding="utf-8", errors="ignore"):
        m = EP.search(line)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            assert k not in eps
            eps[k] = 1 if m.group(3) == "True" else 0
    assert len(eps) == 100, (name, len(eps))
    return eps


cells = {}
for row in csv.DictReader(open(CSV, encoding="utf-8")):
    if row["method"] == "F0" and row["cond"] == "O":
        cells.setdefault((row["target"], row["suite"]),
                         {})[(int(row["task"]), int(row["episode_index"]))] = \
            int(row["success"])

print("== EDPA-native-2D：O / P_method(随机57×57) / Adv(优化) ==")
print("| 目标 | 套件 | O | P | A | 总影响 O−A | 优化增量 P−A | 载体效应 O−P | "
      "A vs P (b/c, p) |")
print("|---|---|---:|---:|---:|---:|---:|---:|---|")
macros = {"pi0": [], "pi05": []}
for t in TARGETS:
    for s in SUITES:
        O = cells[(t, s)]
        P = load_log("nat2d_rnd_%s_%s" % (t, s))
        A = load_log("nat2d_adv_%s_%s" % (t, s))
        assert set(O) == set(P) == set(A)
        o = sum(O.values()) / 100.0
        p = sum(P.values()) / 100.0
        a = sum(A.values()) / 100.0
        bp = sum(1 for k in O if P[k] and not A[k])   # P成A败
        ba = sum(1 for k in O if A[k] and not P[k])   # A成P败
        macros[t].append((o - a, p - a, o - p))
        print("| %s | %s | %.0f | %.0f | %.0f | %+.0f | %+.0f | %+.0f | "
              "%d/%d, p=%.3g |"
              % (t, s, o * 100, p * 100, a * 100, (o - a) * 100,
                 (p - a) * 100, (o - p) * 100, bp, ba, mcnemar(bp, ba)))
for t in TARGETS:
    m = macros[t]
    print("%s 宏平均：总影响 %.2f | 优化增量 %.2f | 载体效应 %.2f"
          % (t, sum(x[0] for x in m) / 4 * 100,
             sum(x[1] for x in m) / 4 * 100, sum(x[2] for x in m) / 4 * 100))
