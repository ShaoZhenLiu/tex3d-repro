#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_csv_v8.py — v8 CSV 合并（第 14 轮原生形式收官）
v8 = v7 11000 + t3dp 80（T3DPT/T3D1） + t3ddeep 960（T3DADV/T3DTH0/T3DPLN/T3DD1）
      + UPA 800（UPANAT/UPAN1） = 12840 行。
cond 映射：T3DPLN→O（原始资产注入=无扰动）、T3DADV→A、T3DTH0→P（θ0 再烘焙工艺对照，
非随机 patch——manifest 已注明语义）；UPANAT→A。
suite 命名沿用 CSV 既有：spatial/object/goal/10（libero_10 的键为 "10"）。
断言：每格行数==Task 行数、新方法键无重复、总行数==12840。
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EV_T = os.path.join(HERE, "r14_assets", "tex3d_native", "logs")
EV_U = os.path.join(HERE, "r14_assets", "upa_native", "logs")
OUT = os.path.join(HERE, "..", "交给老师_材料包_20261022", "04_数据",
                   "episodes_merged_all_v8.csv")
V7 = os.path.join(HERE, "..", "交给老师_材料包_20261022", "04_数据",
                  "episodes_merged_all_v7.csv")
EP = re.compile(r"Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)")

SHA = {
    "t0": "5adecbb85b0861e560d1861f9f559e6b8461aa693301f191a2a60a726887b05b",
    "t2": "0ddd43c455b1eb1993fb9b65c241402e27390c23af0b8c9e9b5947035a6b219d",
    "t6": "278523518e967ed73b2d60b709f17232a5d492e860e4839358180035cbe91402",
    "t8": "5e778ec60b3ccbbfe579850c0c50cef0a319b2861c430ef024e065750bfeaba1",
    "adv0": "5adecbb85b0861e560d1861f9f559e6b8461aa693301f191a2a60a726887b05b",
    "th0": "97c3ae59325434edb004a3ed80e8543a3f01b3aba721ce7481db8d4f96cbffc3",
    "plain": "8a646d98b084b7400dd91beed9c83e10b4a4dca572896b3974356b7937a4bd85",
    "upa": "2799237dd3f25638c9331d517e9293c2f93fd8b2feb27e256f7ec2098b15bbdb",
}


def load(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EP.match(line.strip())
            if m:
                out.append((int(m.group(1)), 10 + int(m.group(2)),
                            1 if m.group(3) == "True" else 0))
    return out


def main():
    with open(V7, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = ["method", "target", "suite", "cond", "task", "episode_index",
                  "init_state_id", "success", "protocol_id", "texture_sha256",
                  "noise_policy_id"]
    base = len(rows)
    print("v7 rows:", base)

    new_keys = set()

    def add(method, target, suite, cond, eps, proto, sha, expect=None):
        n0 = len(rows)
        for task, init, succ in eps:
            key = (method, target, suite, cond, task, init)
            assert key not in new_keys, "dup key %s" % (key,)
            new_keys.add(key)
            rows.append({"method": method, "target": target, "suite": suite,
                         "cond": cond, "task": str(task),
                         "episode_index": str(init - 10),
                         "init_state_id": str(init), "success": str(succ),
                         "protocol_id": proto, "texture_sha256": sha,
                         "noise_policy_id": "query_seed_logged"})
        added = len(rows) - n0
        print("%-8s %-5s %-8s %s: +%d" % (method, target, suite, cond, added))
        if expect is not None:
            assert added == expect, (method, target, suite, added, expect)
        return added

    # t3dp 逐任务（variant b，inits10-19）
    for tgt in ("pi0", "pi05"):
        for t in (0, 2, 6, 8):
            eps = load(os.path.join(EV_T, "t3dp_t%d_%s.txt" % (t, tgt)))
            assert len(eps) == 10, (t, tgt, len(eps))
            add("T3DPT", tgt, "spatial", "A", eps, "T3D1", SHA["t%d" % t])

    # t3ddeep 加深（inits10-49）
    for tgt in ("pi0", "pi05"):
        for cond, method, sha in (("plain", "T3DPLN", SHA["plain"]),
                                  ("adv", "T3DADV", SHA["adv0"]),
                                  ("th0", "T3DTH0", SHA["th0"])):
            eps = load(os.path.join(EV_T, "t3ddeep_%s_%s.txt" % (cond, tgt)))
            assert len(eps) == 160, (cond, tgt, len(eps))
            csv_cond = {"plain": "O", "adv": "A", "th0": "P"}[cond]
            add(method, tgt, "spatial", csv_cond, eps, "T3DD1", sha)

    # UPA 800（四套件×两目标）
    for tgt, suffix in (("pi0", "pi0"), ("pi05", "pi05")):
        for suite, tag in (("spatial", "sp"), ("object", "obj"),
                           ("goal", "go"), ("10", "l10")):
            eps = load(os.path.join(EV_U, "upanat_%s_%s.txt" % (suffix, tag)))
            assert len(eps) == 100, (suffix, tag, len(eps))
            add("UPANAT", tgt, suite, "A", eps, "UPAN1", SHA["upa"])

    total = len(rows)
    print("v8 total:", total)
    assert total == 12840, total

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("written:", OUT)


if __name__ == "__main__":
    main()
