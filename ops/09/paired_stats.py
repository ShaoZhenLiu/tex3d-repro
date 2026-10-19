#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""McNemar 实算与配对统计（导师 §1.4 修正：用逐集 CSV/结果文件实算，不用下界）。

用法（09 上）: paired_stats.py <plain_result_txt> <adv_result_txt> [--csv out.csv]
输出: 各 SR、G=100*(b-f)/N、b/f、q_break=b/Plain成功数、精确二项 McNemar p（双侧）。
"""
import math
import re
import sys

EP = re.compile(r'Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)')


def load(path):
    d = {}
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = EP.search(line)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = m.group(3) == 'True'
    return d


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    plain, adv = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(plain) & set(adv))
    b = sum(1 for k in common if plain[k] and not adv[k])
    f = sum(1 for k in common if not plain[k] and adv[k])
    psr = sum(plain[k] for k in common)
    asr = sum(adv[k] for k in common)
    N = len(common)
    print(f'N={N}  Plain={psr} ({100*psr/N:.0f}%)  Adv={asr} ({100*asr/N:.0f}%)')
    print(f'G = 100*({b}-{f})/{N} = {100*(b-f)/N:+.1f}pp')
    print(f'break={b} fix={f} q_break={b}/{psr} = {b/psr if psr else float("nan"):.3f}')
    print(f'McNemar exact two-sided p = {mcnemar_exact(b, f):.4f}')
    per_task = {}
    for k in common:
        t = per_task.setdefault(k[0], [0, 0, 0])
        t[0] += plain[k]
        t[1] += adv[k]
        t[2] += 1
    print('task  plain  adv')
    for t in sorted(per_task):
        p, a, n = per_task[t]
        print(f'  {t}    {p}/{n}   {a}/{n}')


if __name__ == '__main__':
    main()
