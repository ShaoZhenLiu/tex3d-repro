#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agg_r18_ref8.py — 第18轮批次A（E01+E02）统一聚合入口
E01：Native8(tvfix_bs8_native, sha ee241502…) 补六格 → 完整八格 MR-vs-Native8 参考
     对照（每格 100 回合、tasks 0-9 × 初态 10-19、独立查询噪声）。
E02：OFT 源端(Spatial step150000)四条件闭环（O/P/N8/MR，初态 10-19、400+10 步）。

统计口径（老师 §6.2/§12.3）：
  - 左=MR、右=Native8；左强 b = #(MR败 & N8成)，右强 c = #(MR成 & N8败)；
    断言 b − c = succ_N8 − succ_MR（配对完整性 + 方向）。
  - 精确双侧 McNemar：n=b+c, p = 2·min{P(X≤min), P(X≥max)} 截断到 1（二项）。
  - Holm-8 仅对 MR-vs-Native8 这一方法对的八格执行，不得先挑显著格组族。
  - G = P − A（沿用主表共享 Plain，不重复评 Plain）；ΔG = G_MR − G_N8 = A_N8 − A_MR。
  - 多 seed/E02 源端单套件不并入八格族，单独成表。

用法：
  python agg_r18_ref8.py \
      --v81 交给老师_材料包_20261024/04_数据/episodes_merged_all_v8_1.csv \
      --e01-logdir <e01_logs 目录（含 e01_n8_*.log）> \
      --e02-dirs <transfer_oft 下若干结果目录…可多传> \
      --out <refcloth_next_batch>
产出：data/episodes_new.csv、data/episodes_merged_candidate.csv、
      tables/reference_8cells.csv、tables/oft_source_spatial.csv、agg_r18_output.txt
兼容 py3.6（math.comb 用阶乘实现）。
"""
import argparse
import csv
import math
import os
import re
import sys

N8_SHA = "ee241502c8d1523ae3fcb3b038304ff77c22772f5c0b4e56f7aa0a0bf71509ec"
MR_SHA = "51ad185f0495373784fcf39b48cdbcf75b32b947cdaabf71429e1868de863187"
T0_SHA = "9829a9529b74f02e61a87ee27c9acceddef9fed98a45c8bc829b8c233bda43c3"

E01_CELLS = [  # (log名, target, suite_key)
    ("e01_n8_pi0_spatial", "pi0", "spatial"),
    ("e01_n8_pi0_object", "pi0", "object"),
    ("e01_n8_pi0_goal", "pi0", "goal"),
    ("e01_n8_pi0_long", "pi0", "10"),
    ("e01_n8_pi05_goal", "pi05", "goal"),
    ("e01_n8_pi05_long", "pi05", "10"),
]
E02_COND = {  # 结果目录条件 → (method, cond)
    ("original", None): ("OFTSRC", "O"),
    ("plain", T0_SHA): ("OFTSRC", "P"),
    ("adv", N8_SHA): ("OFTSRC", "AN8"),
    ("adv", MR_SHA): ("OFTSRC", "AMR"),
}


def comb(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = den = 1.0
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return num / den


def mcnemar_exact(b, c):
    """精确双侧 McNemar（二项，p=0.5）。b=左强, c=右强。"""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p_le = sum(comb(n, i) for i in range(0, lo + 1)) * (0.5 ** n)
    hi = max(b, c)
    p_ge = sum(comb(n, i) for i in range(hi, n + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * min(p_le, p_ge))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        val = max(val, prev)
        prev = val
        adj[idx] = min(val, 1.0)
    return adj


def load_v81(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def parse_result_txt(path):
    """返回 {(task, ep): success_bool}；同时校验 100 行。"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^Task: (\d+) \| Ep: (\d+) \| Success: (True|False)", line)
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = (m.group(3) == "True")
    return out


def e01_rows_from_logs(logdir, csv_writer_sink, problems):
    rows = []
    for name, target, suite in E01_CELLS:
        log = os.path.join(logdir, name + ".log")
        res = None
        if os.path.isfile(log):
            for line in open(log, encoding="utf-8", errors="replace"):
                # 编码无关：匹配 [eval] …: <路径>/evaluation_result.txt
                m = re.search(r"\[eval\][^:]*:\s*(\S*/evaluation_result\.txt)",
                              line)
                if m:
                    res = m.group(1)
        if not res or not os.path.isfile(res):
            problems.append("E01 %s: 结果文件缺失(%s)" % (name, res))
            continue
        eps = parse_result_txt(res)
        if len(eps) != 100:
            problems.append("E01 %s: 回合数 %d != 100 (%s)" % (name, len(eps), res))
        # 纹理身份由链脚本启动时 SHA 断言保证（ee241502…），结果头不重复登记
        for (task, ep), succ in sorted(eps.items()):
            rows.append({
                "method": "TVbs8", "target": target, "suite": suite, "cond": "A",
                "task": task, "episode_index": ep, "init_state_id": 10 + ep,
                "success": int(succ), "protocol_id": "F1",
                "texture_sha256": N8_SHA,
                "noise_policy_id": "legacy_independent"})
    return rows


def e02_rows_from_dirs(dirs, problems):
    rows = []
    for d in dirs:
        res = os.path.join(d, "evaluation_result.txt")
        if not os.path.isfile(res):
            problems.append("E02 %s: 缺 evaluation_result.txt" % d)
            continue
        head = open(res, encoding="utf-8", errors="replace").read(2000)
        cond = re.search(r"^condition=(\w+)", head, re.M)
        sha = re.search(r"^texture_sha256=([0-9a-f]*)", head, re.M)
        ioff = re.search(r"^.*init_offset=(\d+)", head, re.M)
        msteps = re.search(r"^.*max_steps=(\d+)", head, re.M)
        suite = re.search(r"^suite=(\w+)", head, re.M)
        if not cond or not sha or not ioff or not msteps:
            problems.append("E02 %s: 结果头缺身份字段（旧版脚本产物？拒收）" % d)
            continue
        sha_v = sha.group(1) or None
        key = (cond.group(1), sha_v if cond.group(1) != "original" else None)
        # original 头 texture=None → sha 空；plain 头登记 T0 sha
        if key not in E02_COND:
            problems.append("E02 %s: 未知条件/纹理 %s %s" % (d, cond.group(1), sha_v))
            continue
        if int(ioff.group(1)) != 10 or int(msteps.group(1)) != 400:
            problems.append("E02 %s: 协议不符 init_offset=%s max_steps=%s（拒收）"
                            % (d, ioff.group(1), msteps.group(1)))
            continue
        method, csvcond = E02_COND[key]
        if cond.group(1) == "plain" and sha_v != T0_SHA:
            problems.append("E02 %s: plain 纹理 sha 非 T0（拒收）" % d)
            continue
        if cond.group(1) == "adv" and sha_v not in (N8_SHA, MR_SHA):
            problems.append("E02 %s: adv 纹理 sha 非 N8/MR（拒收）" % d)
            continue
        eps = parse_result_txt(res)
        if len(eps) != 100:
            problems.append("E02 %s: 回合数 %d != 100" % (d, len(eps)))
        for (task, ep), succ in sorted(eps.items()):
            rows.append({
                "method": method, "target": "oft", "suite": "spatial",
                "cond": csvcond, "task": task, "episode_index": ep,
                "init_state_id": 10 + ep, "success": int(succ),
                "protocol_id": "E02R18", "texture_sha256": sha_v or "-",
                "noise_policy_id": "legacy_independent"})
    return rows


def cell_pairs(rows, method, target, suite, cond):
    d = {}
    for r in rows:
        if (r["method"], r["target"], r["suite"], r["cond"]) == \
                (method, target, suite, cond):
            key = (int(r["task"]), int(r["init_state_id"]))
            if key in d:
                raise AssertionError("重复键 %s %s" % ((method, target, suite, cond), key))
            d[key] = int(r["success"])
    return d


def sr(d):
    return 100.0 * sum(d.values()) / max(len(d), 1)


def paired(b_succ, a_succ, name_left, name_right, out):
    """b_succ=左方法成败映射, a_succ=右方法；返回 (b, c, p)。"""
    assert set(b_succ) == set(a_succ), "键集合不一致: %s vs %s" % (name_left, name_right)
    b_strong = c_strong = 0
    sb = sa = 0
    for k in sorted(b_succ):
        lb, ra = b_succ[k], a_succ[k]
        sb += lb
        sa += ra
        if (not lb) and ra:
            b_strong += 1        # 左攻击失败 & 右成功 → 左强
        if lb and (not ra):
            c_strong += 1        # 右强
    assert b_strong - c_strong == sa - sb, \
        "方向断言失败 %s/%s: b-c=%d, 右成-左成=%d" % (name_left, name_right,
                                                 b_strong - c_strong, sa - sb)
    p = mcnemar_exact(b_strong, c_strong)
    out.append("  配对[%s vs %s] 左强(MR败N8成)=%d 右强=%d 精确p=%.6g "
               "(succ %d vs %d, 方向断言✓)" % (name_left, name_right, b_strong,
                                                c_strong, p, sb, sa))
    return b_strong, c_strong, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v81", required=True)
    ap.add_argument("--e01-logdir", required=True)
    ap.add_argument("--e02-dirs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    problems = []
    v81 = load_v81(args.v81)
    e01 = e01_rows_from_logs(args.e01_logdir, None, problems)
    e02 = e02_rows_from_dirs(args.e02_dirs, problems)
    out_lines = []
    A = out_lines.append

    A("=== r18 批次A 聚合（E01 Native8 六格 + E02 OFT 源端四条件） ===")
    A("v8.1 行数 %d（应 12840，仅对旧快照断言）" % len(v81))
    assert len(v81) == 12840, "v8.1 行数不符"

    # ---------- E01：八格 MR vs Native8 ----------
    allrows = v81 + e01
    suite_disp = [("pi0", "spatial", "Spatial"), ("pi0", "object", "Object"),
                  ("pi0", "goal", "Goal"), ("pi0", "10", "Long"),
                  ("pi05", "spatial", "Spatial"), ("pi05", "object", "Object"),
                  ("pi05", "goal", "Goal"), ("pi05", "10", "Long")]
    ref_rows = []
    pvals = []
    for target, suite, disp in suite_disp:
        O = cell_pairs(allrows, "F0", target, suite, "O")
        P = cell_pairs(allrows, "F0", target, suite, "P")
        MR = cell_pairs(allrows, "MR", target, suite, "A")
        N8 = cell_pairs(allrows, "TVbs8", target, suite, "A")
        if not (len(O) == len(P) == len(MR) == len(N8) == 100):
            problems.append("%s-%s: 单元规模 O=%d P=%d MR=%d N8=%d"
                            % (target, suite, len(O), len(P), len(MR), len(N8)))
            continue
        msgs = []
        b, c, p = paired(MR, N8, "MR", "N8", msgs)
        out_lines.extend(msgs)
        pvals.append(p)
        o, pl, amr, an8 = sr(O), sr(P), sr(MR), sr(N8)
        gmr, gn8 = pl - amr, pl - an8
        ref_rows.append([disp, target, suite, o, pl, amr, gmr, an8, gn8,
                         gmr - gn8, b, c, p])
    A("")
    A("--- E01 八格 MR vs Native8（左=MR，右=Native8；G=P−A，ΔG=G_MR−G_N8） ---")
    if len(ref_rows) == 8:
        adj = holm(pvals)
        A("cell,target,suite,O,P,A_MR,G_MR,A_N8,G_N8,dG,"
          "left_strong(MR),right_strong(N8),p_exact,p_holm8")
        for i, r in enumerate(ref_rows):
            A("%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%+.2f,%d,%d,%.6g,%.6g"
              % tuple(r + [adj[i]]))
        A("")
        A("--- 分组均值（G 口径） ---")
        def gv(sel, col):
            xs = [r[col] for r in ref_rows if sel(r)]
            return sum(xs) / max(len(xs), 1)
        A("MR: π0四格 %.3f | π0.5四格 %.3f | 八格 %.3f" %
          (gv(lambda r: r[1] == "pi0", 6), gv(lambda r: r[1] == "pi05", 6),
           gv(lambda r: True, 6)))
        dev = lambda r: (r[1], r[2]) in (("pi05", "spatial"), ("pi05", "object"))
        A("Native8: π0四格 %.3f | π0.5四格 %.3f | 八格 %.3f | 两个开发格 %.3f | "
          "六个新格 %.3f" %
          (gv(lambda r: r[1] == "pi0", 8), gv(lambda r: r[1] == "pi05", 8),
           gv(lambda r: True, 8), gv(dev, 8), gv(lambda r: not dev(r), 8)))
        A("ΔG(MR−N8): 八格均值 %+.3f | 六个新格均值 %+.3f" %
          (gv(lambda r: True, 9), gv(lambda r: not dev(r), 9)))
    else:
        problems.append("E01 八格不齐（当前 %d 格），Holm-8 暂不执行" % len(ref_rows))
    A("")
    A("教师对照：π0.5-Object 应 42/0 p≈4.5474735e-13；π0.5-Spatial 应 25/18 p≈0.3603777")

    # ---------- E02：OFT 源端 ----------
    A("")
    A("--- E02 OFT 源端（Spatial，初态10-19，400+10步） ---")
    e02_tbl_rows = []
    if e02:
        conds = [("O", "Original"), ("P", "Plain"), ("AN8", "Native8"),
                 ("AMR", "MR8")]
        cells = {}
        for c, nm in conds:
            cells[c] = cell_pairs(e02, "OFTSRC", "oft", "spatial", c) if \
                any(r["cond"] == c for r in e02) else {}
            A("%s: n=%d SR=%.2f" % (nm, len(cells[c]), sr(cells[c])))
        if all(len(cells[c]) == 100 for c, _ in conds):
            o, p = sr(cells["O"]), sr(cells["P"])
            for c, nm in (("AN8", "Native8"), ("AMR", "MR8")):
                a = sr(cells[c])
                A("%s: D=O−P=%.2f G=P−A=%.2f 总降幅 O−A=%.2f"
                  % (nm, o - p, p - a, o - a))
            msgs = []
            b, cp, pp = paired(cells["AMR"], cells["AN8"], "MR8源端", "N8源端", msgs)
            out_lines.extend(msgs)
            # 训练任务分组（{0,2,6,8} 共40回合；其余六任务共60回合）
            for grp, ids, tot in (("纹理训练任务{0,2,6,8}", (0, 2, 6, 8), 40),
                                  ("未参与优化{1,3,4,5,7,9}", (1, 3, 4, 5, 7, 9), 60)):
                sub = {c: sum(v for k, v in cells[c].items() if k[0] in ids)
                       for c, _ in conds}
                A("  %s: O %d/%d P %d/%d N8 %d/%d MR %d/%d"
                  % (grp, sub["O"], tot, sub["P"], tot,
                     sub["AN8"], tot, sub["AMR"], tot))
            hdr = None
            e02_tbl_rows.append(["summary", "%.2f" % o, "%.2f" % p,
                                 "%.2f" % sr(cells["AN8"]),
                                 "%.2f" % sr(cells["AMR"])])
            e02_tbl_rows.append(["paired", "left_strong(MR败N8成)=%d" % b,
                                 "right_strong=%d" % cp, "p=%.6g" % pp,
                                 "succ: O=%.0f P=%.0f" % (o, p)])
            for tid in range(10):
                e02_tbl_rows.append([
                    "task%d" % tid,
                    sum(cells["O"].get((tid, i), 0) for i in range(10, 20)),
                    sum(cells["P"].get((tid, i), 0) for i in range(10, 20)),
                    sum(cells["AN8"].get((tid, i), 0) for i in range(10, 20)),
                    sum(cells["AMR"].get((tid, i), 0) for i in range(10, 20))])
            e02_tbl_rows.append(["group", "trained{0,2,6,8}",
                                 "O=%d/40" % sum(v for k, v in cells["O"].items() if k[0] in (0, 2, 6, 8)),
                                 "N8=%d/40" % sum(v for k, v in cells["AN8"].items() if k[0] in (0, 2, 6, 8)),
                                 "MR=%d/40" % sum(v for k, v in cells["AMR"].items() if k[0] in (0, 2, 6, 8))])
            e02_tbl_rows.append(["group", "untrained{1,3,4,5,7,9}",
                                 "O=%d/60" % sum(v for k, v in cells["O"].items() if k[0] in (1, 3, 4, 5, 7, 9)),
                                 "N8=%d/60" % sum(v for k, v in cells["AN8"].items() if k[0] in (1, 3, 4, 5, 7, 9)),
                                 "MR=%d/60" % sum(v for k, v in cells["AMR"].items() if k[0] in (1, 3, 4, 5, 7, 9))])
        else:
            problems.append("E02 四条件未齐: %s" %
                            {c: len(cells[c]) for c, _ in conds})
    else:
        problems.append("E02: 无有效结果")
    with open(os.path.join(args.out, "tables", "oft_source_spatial.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "col1", "col2_O_or_P", "col3_N8", "col4_MR"])
        w.writerow(["# summary 行=四条件 SR%；task 行=各条件该任务 10 回合成功数",
                    "", "", "", ""])
        for r in e02_tbl_rows:
            w.writerow(r)

    # ---------- 汇总唯一性 ----------
    A("")
    A("--- 唯一性检查 ---")
    for label, rowsx in (("E01", e01), ("E02", e02)):
        keys = [(r["method"], r["target"], r["suite"], r["cond"],
                 int(r["task"]), int(r["init_state_id"])) for r in rowsx]
        A("%s 行数 %d 唯一键 %d" % (label, len(keys), len(set(keys))))
        if len(keys) != len(set(keys)):
            problems.append("%s 存在重复键" % label)

    if problems:
        A("")
        A("!!! 问题清单（新数据按计划作业与键集合验收，不硬套 12840）:")
        for x in problems:
            A("  - " + x)

    # ---------- 落盘 ----------
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    flds = ["method", "target", "suite", "cond", "task", "episode_index",
            "init_state_id", "success", "protocol_id", "texture_sha256",
            "noise_policy_id"]
    with open(os.path.join(args.out, "data", "episodes_new.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in e01 + e02:
            w.writerow(r)
    merged = list(v81) + e01 + e02
    with open(os.path.join(args.out, "data", "episodes_merged_candidate.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in merged:
            w.writerow({k: r[k] for k in flds})
    with open(os.path.join(args.out, "tables", "reference_8cells.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "target", "suite", "O", "P", "A_MR", "G_MR",
                    "A_N8", "G_N8", "dG", "left_strong_MR", "right_strong_N8",
                    "p_exact", "p_holm8"])
        if len(ref_rows) == 8:
            adj = holm(pvals) if pvals else []
            for i, r in enumerate(ref_rows):
                w.writerow(r[:13] + [adj[i] if adj else ""])
    A("")
    A("写出: episodes_new.csv(%d 行) / episodes_merged_candidate.csv(%d 行) / "
      "reference_8cells.csv" % (len(e01) + len(e02), len(merged)))
    txt = "\n".join(out_lines) + "\n"
    with open(os.path.join(args.out, "agg_r18_output.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
