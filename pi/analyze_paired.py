#!/usr/bin/env python3
"""O/P/A 三条件逐集配对分析（老师 2026-10-16 分析稿 §2 的指标）。

输入：若干 evaluation_result.txt（含逐集行 "Task: t | Ep: i | Success: True/False"）。
按 (suite, task, ep) 配对，输出：
  D  = SR_O − SR_P   （载体净损失）
  G  = SR_P − SR_A   （优化纹理净增益）
  q_break = Σ P_i(1−A_i) / Σ P_i   （plain 成功中被 adv 打失败的比例）
  修复数 fix = #{P=0,A=1}（adv 反而救回的回合）
仅对三条件共同覆盖的 (task, ep) 子集计算；公开覆盖数。

用法：
  python analyze_paired.py --suite libero_spatial \
      --original eval_original_libero_spatial_*/evaluation_result.txt \
      --plain    eval_plain_libero_spatial_*/evaluation_result.txt \
      --adv      eval_adv_libero_spatial_*/evaluation_result.txt
支持通配符（shell 展开多个文件时取并集）。
"""
import argparse
import glob
import re
import sys
from collections import defaultdict

EP_RE = re.compile(r"Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)")


def load(paths_glob):
    """多文件并集 -> {(task, ep): 0/1}；同键冲突时后读覆盖并告警。"""
    files = sorted(glob.glob(paths_glob))
    if not files:
        sys.exit(f"no file matches: {paths_glob}")
    out, conflicts = {}, 0
    for f in files:
        for t, e, s in EP_RE.findall(open(f, encoding="utf-8").read()):
            key, val = (int(t), int(e)), s == "True"
            if key in out and out[key] != val:
                conflicts += 1
            out[key] = val
    if conflicts:
        print(f"[warn] {paths_glob}: {conflicts} 个 (task,ep) 在多文件间结果冲突（取最后读取）",
              file=sys.stderr)
    return out


def rate(d, keys):
    return sum(d[k] for k in keys) / len(keys) if keys else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="仅用于输出标注")
    ap.add_argument("--original", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--adv", required=True)
    a = ap.parse_args()

    O, P, A = load(a.original), load(a.plain), load(a.adv)
    common = sorted(set(O) & set(P) & set(A))
    if not common:
        sys.exit("三条件无共同 (task, ep)")

    sr_o, sr_p, sr_a = rate(O, common), rate(P, common), rate(A, common)
    p_succ = [k for k in common if P[k]]
    q_break = (sum(1 for k in p_succ if not A[k]) / len(p_succ)) if p_succ else float("nan")
    n_fix = sum(1 for k in common if not P[k] and A[k])
    n_break = sum(1 for k in common if P[k] and not A[k])

    print(f"\n=== {a.suite}  配对子集 n={len(common)} 集（任务 {common[0][0]}..{common[-1][0]}）===")
    print(f"SR_O={sr_o:.3f}  SR_P={sr_p:.3f}  SR_A={sr_a:.3f}")
    print(f"D = O−P = {100*(sr_o-sr_p):+.1f}pp   G = P−A = {100*(sr_p-sr_a):+.1f}pp")
    print(f"q_break={q_break:.3f}（plain 成功 {len(p_succ)} 集中 adv 打失败 {n_break}）"
          f"   fix={n_fix}（adv 反而救回）")

    by_task = defaultdict(list)
    for k in common:
        by_task[k[0]].append(k)
    print("task |  O   P   A  | D(pp) G(pp) | n")
    for t in sorted(by_task):
        ks = by_task[t]
        o, p, aa = rate(O, ks), rate(P, ks), rate(A, ks)
        print(f"  t{t} | {o:4.2f} {p:4.2f} {aa:4.2f} | {100*(o-p):+5.1f} {100*(p-aa):+5.1f} | {len(ks)}")


if __name__ == "__main__":
    main()
