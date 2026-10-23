#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_t3d_native.py — Tex3D 原生对象3D聚合（第14轮P2）
两个部署变体：
  (a) 通用部署：task0 纹理挂到 {0,2,6,8}×10初态 × {π0,π0.5} × {objtex(Adv), objtex0(θ0)} = 160
  (b) 逐任务原生粒度：每任务用自己的纹理 × {π0,π0.5} = 8 格 × 10 回合 = 80 Adv
Plain 基线：v7 CSV F0/O 行的 {0,2,6,8}×inits10-19 子集（每格 40 键，无重复）。
口径：G = SR(Plain) − SR(Adv)（§6.3）；θ0 对照给 rebake 工艺差；
配对检验 = McNemar 精确（逐回合同 (task,init) 键）。
本地运行：result 文件在 r14_assets/tex3d_native/logs/（缺格自动跳过并提示）。
"""
import csv
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "r14_assets", "tex3d_native", "logs")
CSV = None
for cand in (os.path.join(HERE, "..", "交给老师_材料包_20261022", "04_数据",
                          "episodes_merged_all_v7.csv"),
             os.path.join(HERE, "..", "交给老师_材料包_20261022", "04_数据",
                          "episodes_merged_all_v6_1.csv")):
    if os.path.exists(cand):
        CSV = cand
        break
EP = re.compile(r"Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)")
TARGETS = ["pi0", "pi05"]
TASKS = [0, 2, 6, 8]


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def mcnemar_exact(b, c):
    """同回合配对：b=只前者成功数, c=只后者成功数；双侧精确二项。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n) * 2.0
    return min(1.0, p)


def load_plain():
    """{(target,task,init): 0/1}"""
    out = {}
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["method"] == "F0" and r["cond"] == "O" and r["suite"] == "spatial"
                    and int(r["task"]) in TASKS
                    and 10 <= int(r["init_state_id"]) <= 19):
                out[(r["target"], int(r["task"]), int(r["init_state_id"]))] = int(r["success"])
    return out


def load_result(path):
    """{(task,ep): 0/1}，ep→init 映射 = init 10+ep（--init-offset 10）"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EP.match(line.strip())
            if m:
                out[(int(m.group(1)), 10 + int(m.group(2)))] = 1 if m.group(3) == "True" else 0
    return out


def pair(a, b):
    """a,b: {(task,init):0/1}；返回 (b_only, c_only)"""
    keys = set(a) & set(b)
    bo = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    co = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    return bo, co, len(keys), len(set(a) ^ set(b))


def sr(d):
    return sum(d.values()) / float(len(d)) if d else float("nan")


def main():
    plain = load_plain()
    print("Plain 基线（F0/O spatial {0,2,6,8}×inits10-19）")
    for t in TARGETS:
        sub = {k[1:]: v for k, v in plain.items() if k[0] == t}
        print("  %s: %d/%d = %.4f" % (t, sum(sub.values()), len(sub), sr(sub)))
    print()

    # —— 变体(a)：通用部署（task0 纹理）——
    print("=" * 72)
    print("变体(a) 通用部署：task0 纹理 → 4 任务 × 2 目标 × {Adv, θ0}")
    print("=" * 72)
    uni = {}
    for tgt in TARGETS:
        for cond, label in (("adv", "Adv"), ("th0", "θ0")):
            p = os.path.join(EV, "t3d_%s_%s.txt" % (cond, tgt))
            if not os.path.exists(p):
                print("  [缺] %s" % p)
                continue
            d = load_result(p)
            uni[(tgt, cond)] = d
            pl = {k[1:]: v for k, v in plain.items() if k[0] == tgt}
            bo, co, nk, sym = pair(pl, d)
            p_paired = mcnemar_exact(bo, co)
            print("  t3d_%s_%s: %d/%d = %.4f | G=Plain-Adv %+.2f | 配对 b=%d c=%d p=%.4f | 键差%d"
                  % (cond, tgt, sum(d.values()), len(d), sr(d),
                     (sr(pl) - sr(d)) * 100, bo, co, p_paired, sym))
        if (tgt, "adv") in uni and (tgt, "th0") in uni:
            bo, co, nk, sym = pair(uni[(tgt, "adv")], uni[(tgt, "th0")])
            print("    Adv vs θ0 配对: b=%d c=%d p=%.4f" % (bo, co, mcnemar_exact(bo, co)))

    # 逐任务分解（变体a）
    print("\n  逐任务 SR（变体a，每格每任务 10 回合）")
    for tgt in TARGETS:
        row = []
        for cond in ("adv", "th0"):
            if (tgt, cond) not in uni:
                continue
            for task in TASKS:
                sub = {k: v for k, v in uni[(tgt, cond)].items() if k[0] == task}
                row.append("t%d %s %d/10" % (task, cond, sum(sub.values())))
        print("  %s: %s" % (tgt, " | ".join(row)))

    # —— 加深口径（选项A，inits10-49；Plain/Adv/θ0 全部取自同一波 t3ddeep 文件）——
    print()
    print("=" * 72)
    print("加深口径 t3ddeep：inits10-49（40/任务）；同波内配对（不跨 v6.1 CSV——")
    print("  π0 回合级随机性已知，金丝雀先例 42 vs 35）")
    print("=" * 72)
    deep = {}
    for tgt in TARGETS:
        for cond in ("plain", "adv", "th0"):
            p = os.path.join(EV, "t3ddeep_%s_%s.txt" % (cond, tgt))
            if os.path.exists(p):
                deep[(tgt, cond)] = load_result(p)
    for tgt in TARGETS:
        row = []
        for cond in ("plain", "adv", "th0"):
            if (tgt, cond) in deep:
                d = deep[(tgt, cond)]
                row.append("%s %d/%d=%.4f" % (cond, sum(d.values()), len(d), sr(d)))
        print("  %s: %s" % (tgt, " | ".join(row) if row else "（未落地）"))
        if (tgt, "adv") in deep and (tgt, "plain") in deep:
            bo, co, nk, sym = pair(deep[(tgt, "plain")], deep[(tgt, "adv")])
            print("    G=Plain-Adv %+.2f | b=%d c=%d p=%.4f | 键差%d"
                  % ((sr(deep[(tgt, "plain")]) - sr(deep[(tgt, "adv")])) * 100,
                     bo, co, mcnemar_exact(bo, co), sym))
        if (tgt, "th0") in deep and (tgt, "plain") in deep:
            bo, co, nk, sym = pair(deep[(tgt, "plain")], deep[(tgt, "th0")])
            print("    G=Plain-θ0  %+.2f | b=%d c=%d p=%.4f | 键差%d"
                  % ((sr(deep[(tgt, "plain")]) - sr(deep[(tgt, "th0")])) * 100,
                     bo, co, mcnemar_exact(bo, co), sym))
        if (tgt, "adv") in deep and (tgt, "th0") in deep:
            bo, co, nk, sym = pair(deep[(tgt, "adv")], deep[(tgt, "th0")])
            print("    Adv vs θ0: b=%d c=%d p=%.4f" % (bo, co, mcnemar_exact(bo, co)))
        if (tgt, "adv") in deep:
            per = ["t%d %d/40" % (t, sum(v for (tt, _), v in deep[(tgt, "adv")].items() if tt == t))
                   for t in TASKS]
            print("    Adv 逐任务: %s" % " | ".join(per))

    # —— 变体(b)：逐任务粒度 ——
    print()
    print("=" * 72)
    print("变体(b) 逐任务原生粒度：task{0,2,6,8} 各用自己纹理 × 2 目标")
    print("=" * 72)
    pertask = {}
    for tgt in TARGETS:
        for task in TASKS:
            p = os.path.join(EV, "t3dp_t%d_%s.txt" % (task, tgt))
            if not os.path.exists(p):
                continue
            d = load_result(p)
            pertask[(tgt, task)] = d
            pl = {k[1:]: v for k, v in plain.items() if k[0] == tgt and k[1] == task}
            bo, co, nk, sym = pair(pl, d)
            print("  t3dp_t%d_%s: %d/%d = %.4f | G %+.2f | b=%d c=%d p=%.4f | 键差%d"
                  % (task, tgt, sum(d.values()), len(d), sr(d),
                     (sr(pl) - sr(d)) * 100, bo, co, mcnemar_exact(bo, co), sym))
    if pertask:
        for tgt in TARGETS:
            cells = [pertask[(tgt, t)] for t in TASKS if (tgt, t) in pertask]
            if len(cells) == len(TASKS):
                merged = {}
                for c in cells:
                    merged.update(c)
                pl = {k[1:]: v for k, v in plain.items() if k[0] == tgt}
                bo, co, nk, sym = pair(pl, merged)
                print("  %s 合并 %d 格: %d/%d = %.4f | G %+.2f | b=%d c=%d p=%.4f"
                      % (tgt, len(cells), sum(merged.values()), len(merged), sr(merged),
                         (sr(pl) - sr(merged)) * 100, bo, co, mcnemar_exact(bo, co)))
            else:
                print("  %s 已到 %d/%d 格（task8 训练中，未合并）"
                      % (tgt, len(cells), len(TASKS)))


if __name__ == "__main__":
    main()
