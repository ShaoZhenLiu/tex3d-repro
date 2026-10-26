#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_e03.py — 第18轮批次B（E03 UV消融）聚合入口
E03：当前 MR 多任务配方唯一变量 = 训练期 UV-EoT（on=MR8 既有 vs off=E03 重放臂）。
八格配对（左=UV-on/MR8，右=UV-off/MRUV0）：左强 = #(UV-on败 & UV-off成) =
UV-EoT 在该回合带来攻击成功的次数；断言 左强−右强 = UV-off成数−UV-on成数。
Holm-8 仅作用于本方法对八格族。Spatial 逐任务分组：{0} / 多任务额外{2,6,8} /
未参与优化{1,3,4,5,7,9}（§8.3；"未见"不表示 VLA 基座未训过这些任务）。
用法：
  python agg_e03.py --v81 <v8_1.csv> --e03-logdir <e03_logs> --out <refcloth_batch_b_e03>
兼容 py3.6；统计函数复用 agg_r18_ref8（已对教师数字自校验）。
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_r18_ref8 import (comb, holm, load_v81, parse_result_txt,  # noqa: E402
                          cell_pairs, sr, mcnemar_exact)

E03_CELLS = [
    ("e03_uv0_pi0_spatial", "pi0", "spatial"),
    ("e03_uv0_pi0_object", "pi0", "object"),
    ("e03_uv0_pi0_goal", "pi0", "goal"),
    ("e03_uv0_pi0_long", "pi0", "10"),
    ("e03_uv0_pi05_spatial", "pi05", "spatial"),
    ("e03_uv0_pi05_object", "pi05", "object"),
    ("e03_uv0_pi05_goal", "pi05", "goal"),
    ("e03_uv0_pi05_long", "pi05", "10"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v81", required=True)
    ap.add_argument("--e03-logdir", required=True)
    ap.add_argument("--tex-sha", default="", help="E03 冻结纹理 sha256（登记入逐集行）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    problems = []
    v81 = load_v81(args.v81)
    e03 = []
    for name, target, suite in E03_CELLS:
        log = os.path.join(args.e03_logdir, name + ".log")
        res = None
        if os.path.isfile(log):
            for line in open(log, encoding="utf-8", errors="replace"):
                m = re.search(r"\[eval\][^:]*:\s*(\S*/evaluation_result\.txt)", line)
                if m:
                    res = m.group(1)
        if not res or not os.path.isfile(res):
            problems.append("E03 %s: 结果文件缺失(%s)" % (name, res))
            continue
        eps = parse_result_txt(res)
        if len(eps) != 100:
            problems.append("E03 %s: 回合数 %d != 100" % (name, len(eps)))
        for (task, ep), succ in sorted(eps.items()):
            e03.append({"method": "MRUV0", "target": target, "suite": suite,
                        "cond": "A", "task": task, "episode_index": ep,
                        "init_state_id": 10 + ep, "success": int(succ),
                        "protocol_id": "F1", "texture_sha256": args.tex_sha or "-",
                        "noise_policy_id": "legacy_independent"})

    out_lines = []
    A = out_lines.append
    A("=== E03 UV 消融聚合（左=UV-on/MR8，右=UV-off/MRUV0；G=P−A） ===")
    A("v8.1 行数 %d" % len(v81))
    allrows = v81 + e03
    suite_disp = [("pi0", "spatial", "Spatial"), ("pi0", "object", "Object"),
                  ("pi0", "goal", "Goal"), ("pi0", "10", "Long"),
                  ("pi05", "spatial", "Spatial"), ("pi05", "object", "Object"),
                  ("pi05", "goal", "Goal"), ("pi05", "10", "Long")]
    ref_rows, pvals = [], []
    for target, suite, disp in suite_disp:
        P = cell_pairs(allrows, "F0", target, suite, "P")
        UVon = cell_pairs(allrows, "MR", target, suite, "A")
        UVoff = cell_pairs(allrows, "MRUV0", target, suite, "A")
        if not (len(P) == len(UVon) == len(UVoff) == 100):
            problems.append("%s-%s: 规模 P=%d on=%d off=%d"
                            % (target, suite, len(P), len(UVon), len(UVoff)))
            continue
        assert set(UVon) == set(UVoff), "键集合不一致 %s-%s" % (target, suite)
        b = sum(1 for k in UVon if UVon[k] == 0 and UVoff[k] == 1)   # UV-on强
        c = sum(1 for k in UVon if UVon[k] == 1 and UVoff[k] == 0)   # UV-off强
        assert b - c == sum(UVoff.values()) - sum(UVon.values())
        p = mcnemar_exact(b, c)
        pvals.append(p)
        pl, aon, aoff = sr(P), sr(UVon), sr(UVoff)
        ref_rows.append([disp, target, suite, pl, aon, pl - aon, aoff,
                         pl - aoff, (pl - aon) - (pl - aoff), b, c, p])
    if len(ref_rows) == 8:
        adj = holm(pvals)
        A("cell,target,suite,P,A_UVon,G_UVon,A_UVoff,G_UVoff,dG(UVon-UVoff),"
          "left_strong(UVon),right_strong(UVoff),p_exact,p_holm8")
        for i, r in enumerate(ref_rows):
            A("%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%+.2f,%d,%d,%.6g,%.6g"
              % tuple(r + [adj[i]]))
        def gv(sel, col):
            xs = [r[col] for r in ref_rows if sel(r)]
            return sum(xs) / max(len(xs), 1)
        A("")
        A("G 均值：UV-on 八格 %.3f | UV-off 八格 %.3f | ΔG 八格 %+.3f"
          % (gv(lambda r: True, 5), gv(lambda r: True, 7), gv(lambda r: True, 8)))
        A("ΔG：π0 四格 %+.3f | π0.5 四格 %+.3f"
          % (gv(lambda r: r[1] == "pi0", 8), gv(lambda r: r[1] == "pi05", 8)))
        # Spatial 逐任务分组（§8.3）
        sp = cell_pairs(allrows, "MRUV0", "pi0", "spatial", "A")
        sp_on = cell_pairs(allrows, "MR", "pi0", "spatial", "A")
        if len(sp) == 100:
            for grp, ids in (("共同单任务{0}", (0,)), ("多任务额外{2,6,8}", (2, 6, 8)),
                             ("未参与优化{1,3,4,5,7,9}", (1, 3, 4, 5, 7, 9))):
                on = sum(v for k, v in sp_on.items() if k[0] in ids)
                off = sum(v for k, v in sp.items() if k[0] in ids)
                n = 10 * len(ids)
                A("  Spatial-π0 %s: UV-on %d/%d vs UV-off %d/%d"
                  % (grp, on, n, off, n))
        sp5 = cell_pairs(allrows, "MRUV0", "pi05", "spatial", "A")
        sp5_on = cell_pairs(allrows, "MR", "pi05", "spatial", "A")
        if len(sp5) == 100:
            for grp, ids in (("共同单任务{0}", (0,)), ("多任务额外{2,6,8}", (2, 6, 8)),
                             ("未参与优化{1,3,4,5,7,9}", (1, 3, 4, 5, 7, 9))):
                on = sum(v for k, v in sp5_on.items() if k[0] in ids)
                off = sum(v for k, v in sp5.items() if k[0] in ids)
                n = 10 * len(ids)
                A("  Spatial-π0.5 %s: UV-on %d/%d vs UV-off %d/%d"
                  % (grp, on, n, off, n))
    else:
        problems.append("E03 八格不齐（%d），Holm-8 暂不执行" % len(ref_rows))

    A("")
    A("--- 唯一性 ---")
    keys = [(r["method"], r["target"], r["suite"], r["cond"], int(r["task"]),
             int(r["init_state_id"])) for r in e03]
    A("E03 行数 %d 唯一键 %d" % (len(keys), len(set(keys))))
    if len(keys) != len(set(keys)):
        problems.append("E03 重复键")
    if problems:
        A("问题清单:")
        for x in problems:
            A("  - " + x)

    flds = ["method", "target", "suite", "cond", "task", "episode_index",
            "init_state_id", "success", "protocol_id", "texture_sha256",
            "noise_policy_id"]
    with open(os.path.join(args.out, "data", "episodes_e03.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in e03:
            w.writerow(r)
    merged = list(v81) + e03
    with open(os.path.join(args.out, "data",
                           "episodes_merged_candidate_e03.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in merged:
            w.writerow({k: r[k] for k in flds})
    with open(os.path.join(args.out, "tables", "uv_ablation_8cells.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "target", "suite", "P", "A_UVon", "G_UVon",
                    "A_UVoff", "G_UVoff", "dG", "left_strong_UVon",
                    "right_strong_UVoff", "p_exact", "p_holm8"])
        if len(ref_rows) == 8:
            for i, r in enumerate(ref_rows):
                w.writerow(r + [adj[i]])
    with open(os.path.join(args.out, "agg_e03_output.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
