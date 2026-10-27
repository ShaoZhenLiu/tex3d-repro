#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_r18bc.py — r18批次B/C 总聚合（E04 任务×UV 矩阵 / E05 三种子 / E07 预算）
数据源：v8.1 + evalbc_logs（40 格评测日志）+ 训练目录 gradient_log 的 [final] 统计。
口径：G=P−A（π0.5 开发格 P 取 v8.1 F0/P；其余格同）。McNemar 只做预设配对，
Holm 不跨实验族。方向断言与 agg_r18_ref8 相同（左强−右强=右成−左成）。
用法：
  python agg_r18bc.py --v81 <csv> --evalbc-logdir <evalbc_logs> \
      --oftc <oft_cloth目录> --e06root <e06_pi0src目录> --out <dir>
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_r18_ref8 import (load_v81, parse_result_txt, cell_pairs, sr,  # noqa
                          mcnemar_exact, holm)

CELLS8 = [("pi0", "spatial"), ("pi0", "object"), ("pi0", "goal"), ("pi0", "10"),
          ("pi05", "spatial"), ("pi05", "object"), ("pi05", "goal"),
          ("pi05", "10")]
DEV = [("pi05", "spatial"), ("pi05", "object")]


def cellname(prefix, target, suite):
    return "%s_%s_%s" % (prefix, target, suite)


def load_evalbc(logdir, prefix, target, suite):
    """从 evalbc 日志取该格成败映射 {(task, init_id): 0/1}。"""
    log = os.path.join(logdir, cellname(prefix, target, suite) + ".log")
    res = None
    if os.path.isfile(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            m = re.search(r"\[eval\][^:]*:\s*(\S*/evaluation_result\.txt)", line)
            if m:
                res = m.group(1)
    if not res or not os.path.isfile(res):
        return None
    eps = parse_result_txt(res)
    if len(eps) != 100:
        return None
    return {(t, 10 + e): int(v) for (t, e), v in eps.items()}


def tex_final_stats(oftc, tag):
    """训练目录 gradient_log 的 [final] 行 → (RMS, Linf, TV)。"""
    import glob
    for d in sorted(glob.glob(os.path.join(oftc, "train_*_" + tag)),
                    reverse=True):
        if "smoke" in d:
            continue
        gl = os.path.join(d, "gradient_log.txt")
        if os.path.isfile(gl):
            for line in open(gl, errors="replace"):
                if line.startswith("[final]"):
                    rms = re.search(r"RMS=([\d.]+)", line)
                    linf = re.search(r"Linf=([\d.]+)", line)
                    tv = re.search(r"TV\(0-1\)=([\d.]+)", line)
                    if rms:
                        return (float(rms.group(1)), float(linf.group(1)),
                                float(tv.group(1)))
    return None


def paired(left, right, lname, rname, out):
    if left is None or right is None:
        return None
    assert set(left) == set(right)
    b = sum(1 for k in left if left[k] == 0 and right[k] == 1)
    c = sum(1 for k in left if left[k] == 1 and right[k] == 0)
    assert b - c == sum(right.values()) - sum(left.values())
    p = mcnemar_exact(b, c)
    out.append("  配对[%s vs %s] 左强=%d 右强=%d p=%.6g（方向断言✓）"
               % (lname, rname, b, c, p))
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v81", required=True)
    ap.add_argument("--evalbc-logdir", required=True)
    ap.add_argument("--oftc", required=True)
    ap.add_argument("--e06root", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    v81 = load_v81(args.v81)
    out = []
    A = out.append
    problems = []

    def v81cell(m, t, s, c):
        return cell_pairs(v81, m, t, s, c)

    # ---------------- E04：任务×UV 矩阵 ----------------
    A("=== E04 任务×UV 矩阵（8格 A / G=P−A；MR8=multi-on，MRuv0=multi-off(E03)，")
    A("    MRs0ON=e04a 单任务0 UV-on，MRs0OFF=e04b 单任务0 UV-off） ===")
    arms = [("MR", "multi-UVon(MR8)"), ("e03_uv0", "multi-UVoff(E03)"),
            ("e04a", "single0-UVon"), ("e04b", "single0-UVoff")]
    rows4 = []
    for target, suite in CELLS8:
        P = v81cell("F0", target, suite, "P")
        line = "%s-%s: P=%.0f" % (target, suite, sr(P))
        for pref, nm in arms:
            d = (v81cell("MR", target, suite, "A") if pref == "MR"
                 else load_evalbc(args.evalbc_logdir, pref, target, suite))
            if d is None:
                problems.append("E04 %s %s-%s 缺" % (pref, target, suite))
                line += " | %s=缺" % nm
                continue
            line += " | %s A=%.0f G=%.0f" % (nm, sr(d), sr(P) - sr(d))
            rows4.append([target, suite, nm, "%.2f" % sr(d),
                          "%.2f" % (sr(P) - sr(d))])
        A(line)
    # 预设配对：multi-on vs multi-off（=E03 结论，agg_e03 已出）；
    # single-on vs single-off；single-on vs multi-on（数据配方+UV共同影响，披露口径）
    for ta, tb, la, lb in (("e04a", "e04b", "single-on", "single-off"),
                           ("e04a", "MR", "single-on", "multi-on")):
        for target, suite in CELLS8:
            da = load_evalbc(args.evalbc_logdir, ta, target, suite)
            db = (v81cell("MR", target, suite, "A") if tb == "MR"
                  else load_evalbc(args.evalbc_logdir, tb, target, suite))
            if da is None or db is None:
                continue
            paired(da, db, la, lb, out)
    # Spatial 分组（单任务臂只训 task0；分组 {0}/{2,6,8}/{1,3,4,5,7,9}）
    A("  Spatial 分组（成功数/总数，越高=攻击越弱）：")
    for pref in ("MR", "e03_uv0", "e04a", "e04b"):
        for target in ("pi0", "pi05"):
            d = (v81cell("MR", target, "spatial", "A") if pref == "MR"
                 else load_evalbc(args.evalbc_logdir, pref, target, "spatial"))
            if d is None:
                continue
            g = {k: sum(v for (t, _), v in d.items() if t in ids)
                 for k, ids in (("{0}", (0,)), ("{2,6,8}", (2, 6, 8)),
                                ("{1,3,4,5,7,9}", (1, 3, 4, 5, 7, 9)))}
            A("    %s %s: {0}=%d/10 {2,6,8}=%d/30 其余=%d/60"
              % (pref, target, g["{0}"], g["{2,6,8}"], g["{1,3,4,5,7,9}"]))

    # ---------------- E05：三种子（7 既有 + 8/9 新，π0.5 两开发格） ----------------
    A("")
    A("=== E05 训练种子稳定性（π0.5-Spatial/Object；seed7=冻结主表） ===")
    seeds = {
        7: {"native": ("TVbs8", None), "mr": ("MR", None)},
        8: {"native": ("e05_s8n", None), "mr": ("e05_s8m", None)},
        9: {"native": ("e05_s9n", None), "mr": ("e05_s9m", None)},
    }
    e05_rows = []
    for target, suite in DEV:
        P = v81cell("F0", target, suite, "P")
        A("  %s-%s: P=%.0f" % (target, suite, sr(P)))
        for sd, armsd in seeds.items():
            line = "    seed%d:" % sd
            for kind, (pref, _) in armsd.items():
                d = (v81cell(pref, target, suite, "A")
                     if sd == 7 else
                     load_evalbc(args.evalbc_logdir, pref, target, suite))
                if d is None:
                    line += " %s=缺" % kind
                    problems.append("E05 seed%d %s %s-%s 缺"
                                    % (sd, kind, target, suite))
                    continue
                line += " %s A=%.0f G=%.0f" % (kind, sr(d), sr(P) - sr(d))
                e05_rows.append([target, suite, sd, kind, "%.2f" % sr(d),
                                 "%.2f" % (sr(P) - sr(d))])
            # seed 内 native vs mr 配对
            if sd == 7:
                dn, dm = (v81cell("TVbs8", target, suite, "A"),
                          v81cell("MR", target, suite, "A"))
            else:
                dn = load_evalbc(args.evalbc_logdir,
                                 "e05_s%dn" % sd, target, suite)
                dm = load_evalbc(args.evalbc_logdir,
                                 "e05_s%dm" % sd, target, suite)
            paired(dn, dm, "N8-s%d" % sd, "MR-s%d" % sd, out) \
                if dn is not None and dm is not None else None
            A(line)
    # 种子均值±样本标准差（G 口径，逐 seed 点保留）
    for kind in ("native", "mr"):
        for target, suite in DEV:
            gs = []
            for sd in (7, 8, 9):
                if sd == 7:
                    d = v81cell("TVbs8" if kind == "native" else "MR",
                                target, suite, "A")
                else:
                    d = load_evalbc(args.evalbc_logdir,
                                    "e05_s%d%s" % (sd, "n" if kind == "native" else "m"),
                                 target, suite)
                if d is not None:
                    P = v81cell("F0", target, suite, "P")
                    gs.append(sr(P) - sr(d))
            if len(gs) == 3:
                mu = sum(gs) / 3
                sd_ = (sum((x - mu) ** 2 for x in gs) / 2.0) ** 0.5
                A("    G[%s] %s-%s 三种子: %s 均值%.2f 样本std%.2f"
                  % (kind, target, suite,
                     "/".join("%.0f" % x for x in gs), mu, sd_))

    # ---------------- E07：帧访问预算矩阵 ----------------
    A("")
    A("=== E07 匹配帧访问预算（π0.5 两开发格；RMS×255 口径） ===")
    cfgs = [
        ("bs1×5000", "TVbs8" if False else "TVbs1", "TVbs1", None,
         "29.9145", 5000, 5000),
        ("bs8×625", "e07_625n", "e07_625m", None, None, 625, 5000),
        ("bs1×40000", "e07_bs1n", "e07_bs1m", None, None, 40000, 40000),
        ("bs8×5000", "TVbs8", "MR", None, None, 5000, 40000),
    ]
    e07_rows = []
    for label, npref, mpref, _, rms_fixed, upd, visits in cfgs:
        line = "  %s（更新%d，帧访问%d）:" % (label, upd, visits)
        for kind, pref in (("native", npref), ("mr", mpref)):
            sts = (rms_fixed if kind == "native" and rms_fixed
                   else None)
            for target, suite in DEV:
                P = v81cell("F0", target, suite, "P")
                if pref in ("TVbs1", "TVbs8", "MR"):
                    d = v81cell(pref, target, suite, "A")
                else:
                    d = load_evalbc(args.evalbc_logdir, pref, target, suite)
                if d is None:
                    line += " %s-%s=缺" % (kind, suite)
                    problems.append("E07 %s %s %s-%s 缺"
                                    % (label, kind, target, suite))
                    continue
                st = tex_final_stats(args.oftc,
                                     pref.replace("TVbs1", "tvfix-bs1")
                                         .replace("TVbs8", "tvfix-bs8")
                                         .replace("MR", "tvfix-mr")) \
                    if pref not in ("TVbs1", "TVbs8", "MR") else None
                line += " %s-%s A=%.0f G=%.0f" % (kind, suite, sr(d),
                                                  sr(P) - sr(d))
                e07_rows.append([label, kind, target, suite, upd, visits,
                                 "%.2f" % sr(d), "%.2f" % (sr(P) - sr(d)),
                                 ("%.4f" % st[0]) if st else "",
                                 ("%.1f" % st[1]) if st else "",
                                 ("%.4f" % st[2]) if st else ""])
        A(line)
    A("  （bs1×5000/bs8×5000 为既有终点；RMS 既有值 29.9145/37.4023(native)、"
      "28.6592/37.4991(matched) 按第18轮 §4.2）")
    A("  （bs8×625 为重放 MR8 采样流前 625 更新的短程纹理——不是原轨迹早停资产"
      "的逐位恢复，UV 为日志 6 位小数重放，浮点位级不主张）")

    if problems:
        A("")
        A("问题清单:")
        for x in problems:
            A("  - " + x)

    with open(os.path.join(args.out, "tables", "e04_task_uv_matrix.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "suite", "arm", "A", "G"])
        w.writerows(rows4)
    with open(os.path.join(args.out, "tables", "e05_seeds.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "suite", "seed", "arm", "A", "G"])
        w.writerows(e05_rows)
    with open(os.path.join(args.out, "tables", "e07_budget.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "ref", "target", "suite", "updates",
                    "frame_visits", "A", "G", "RMS255", "Linf", "TV01"])
        w.writerows(e07_rows)
    with open(os.path.join(args.out, "agg_r18bc_output.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
