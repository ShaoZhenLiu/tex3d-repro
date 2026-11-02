"""Render the complete paired study; values come only from finalized analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pm(mean,sd):
    return '$%.2f\\pm%.2f$'%(mean,sd)


def stats(values):
    return float(np.mean(values)),float(np.std(values,ddof=1))


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    result=json.loads((a.analysis/'summary.json').read_text())
    assert result['completed'] and result['episodes']==2600 and result['training_pairs']==6
    seeds=pd.read_csv(a.analysis/'seed_means.csv').sort_values('seed')
    suites=pd.read_csv(a.analysis/'paired_suite_results.csv').sort_values(['seed','suite'])
    assert seeds.seed.tolist()==[7,8,9,10,11,12] and len(suites)==12
    records=[]
    for suite,label in [('spatial','Spatial'),('object','Object'),(None,'Two-suite mean')]:
        t=seeds if suite is None else suites[suites.suite==suite]
        n_mean,n_sd=stats(t.Native_G.values);m_mean,m_sd=stats(t.Matched_G.values)
        d_mean,d_sd=stats(t.delta_G.values)
        records.append(dict(suite=label,Native_mean=n_mean,Native_sd=n_sd,
            Matched_mean=m_mean,Matched_sd=m_sd,delta_mean=d_mean,delta_sd=d_sd))
    aggregate=pd.DataFrame(records);aggregate.to_csv(str(a.out/'reference_seed_aggregate.csv'),index=False)
    table=r'''\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{5.5pt}
\caption{\textbf{Six paired texture-training replications.} Net reduction $G$ on $\pi_{0.5}$ (mean $\pm$ sample standard deviation across six trained textures per reference). Spatial/Object use 100 common-noise episodes per texture and suite. The last row averages the two suites within each training pair.}
\label{tab:seed_summary}
\begin{tabular}{@{}lrr@{}}
\toprule
Suite & Native reference & Matched reference\\
\midrule
'''
    for r in records:table+='%s & %s & %s\\\\\n'%(r['suite'],pm(r['Native_mean'],r['Native_sd']),pm(r['Matched_mean'],r['Matched_sd']))
    table+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (a.out/'reference_seed_summary.tex').write_text(table)
    plain={suite:int(suites[suites.suite==suite].Plain.iloc[0]) for suite in ['spatial','object']}
    assert all(suites[suites.suite==suite].Plain.nunique()==1 for suite in plain)
    detail=r'''\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4.2pt}
\caption{\textbf{Every paired replication.} Cells are successes out of 100 / net reduction $G$ (pp) on $\pi_{0.5}$. The final column is the within-seed, two-suite difference $G_M-G_N$ (pp). Plain succeeds in %d/100 Spatial and %d/100 Object episodes under this common-noise protocol.}
\label{tab:supp_seed_detail}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Spatial} & \multicolumn{2}{c}{Object} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Seed & Native & Matched & Native & Matched & $\Delta G$\\
\midrule
'''%(plain['spatial'],plain['object'])
    for seed in seeds.seed:
        row=suites[suites.seed==seed].set_index('suite')
        cells=[]
        for suite in ['spatial','object']:
            for method in ['Native','Matched']:
                cells.append('%d/%d'%(row.loc[suite,method],row.loc[suite,method+'_G']))
        delta=float(seeds[seeds.seed==seed].delta_G.iloc[0])
        detail+='%d & %s & %.1f\\\\\n'%(seed,' & '.join(cells),delta)
    detail+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (a.out/'reference_seed_detail.tex').write_text(detail)
    d=result['primary'];lo,hi=d['ci95'];p=d['exact_two_sided_sign_flip_p']
    primary=('Across six paired replications, the mean matched-reference advantage is '
        '$%.2f$ pp (seed-level 95\\%% $t$ interval $[%.2f,%.2f]$; exact two-sided '
        'sign-flip $p=%.4f$). The six paired differences range from $%.1f$ to $%.1f$ pp. '
        'Full paired outcomes and the evaluation protocol are in \\cref{supp:training_seeds}.\n')%(d['mean'],lo,hi,p,d['minimum'],d['maximum'])
    # A literal percent in TeX requires one backslash, unlike Python formatting.
    primary=primary.replace('\\\\%','\\%')
    (a.out/'reference_seed_primary.tex').write_text(primary)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(left,right)=plt.subplots(1,2,figsize=(6.5,2.35),gridspec_kw={'width_ratios':[1,1.25]})
    colors=['#446F99','#B34C3D','#56805D','#9465A7','#B28635','#518F98']
    for i,row in enumerate(seeds.itertuples()):
        left.plot([0,1],[row.Native_G,row.Matched_G],'-o',lw=.9,ms=3.4,color=colors[i],alpha=.9)
    left.set_xticks([0,1]);left.set_xticklabels(['Native','Matched'])
    left.set_ylabel('Net success reduction G (pp)');left.set_title('(a) Each trained pair, two-suite mean')
    left.grid(axis='y',alpha=.15);left.set_xlim(-.18,1.18)
    for i,row in enumerate(seeds.itertuples()):
        right.plot(row.delta_G,6-i,'o',ms=4,color=colors[i])
    right.errorbar(d['mean'],0,xerr=[[d['mean']-lo],[hi-d['mean']]],fmt='D',color='#222222',
        ms=4,capsize=3,lw=1.1)
    right.axvline(0,color='.55',ls='--',lw=.8)
    right.set_yticks(list(range(7)));right.set_yticklabels(['Mean [95% CI]']+['Seed %d'%s for s in reversed(seeds.seed.tolist())])
    right.set_ylim(-.7,6.7);right.set_xlabel('Matched minus Native G (pp)')
    right.set_title('(b) Paired differences; seed-level interval')
    right.grid(axis='x',alpha=.15);fig.subplots_adjust(wspace=.65,left=.09,right=.98,bottom=.2,top=.86)
    for ext in ['pdf','png']:fig.savefig(str(a.out/('reference_seed_pairs.'+ext)),dpi=260,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    lines=[
        '# RefCloth：六组成对复验结果',
        '',
        '本报告纳入预先固定的全部六组训练对照；没有根据目标结果选择种子、纹理或训练中间点。',
        '',
        '主要比较为 π0.5 Spatial/Object 上 matched 与 native 的净下降之差；正值表示 matched 攻击更强。每种纹理每个套件100回合，另有共用Plain 200回合，总计2600回合。',
        '',
        '| 训练种子 | Native 两套件平均 G | Matched 两套件平均 G | Matched − Native |',
        '| --- | ---: | ---: | ---: |'
    ]
    for row in seeds.itertuples():lines.append('| %d | %.2f | %.2f | %+.2f |'%(row.seed,row.Native_G,row.Matched_G,row.delta_G))
    leave=list(result['leave_one_out_means'].values())
    lines.extend([
        '',
        '全部六组的平均差值为 **%+.2f pp**，样本标准差 %.2f pp，种子层面的95%% t区间为 [%.2f, %.2f] pp，双侧精确符号翻转检验 p=%.4f。'%(d['mean'],d['sd'],lo,hi,p),
        '',
        'matched 更强／native 更强／持平的训练对数分别为 %d／%d／%d；留一组平均差值范围 [%.2f, %.2f] pp。'%(d['positive_pairs'],d['negative_pairs'],d['zero_pairs'],min(leave),max(leave)),
        '',
        '统计单位是六组已训练纹理。2600个评测回合没有被当成2600次独立训练。区间描述该固定任务和初始状态集合上的训练变化。',
        '',
        '现有三组纹理也在相同查询噪声下重新评测；本块不混入旧版独立噪声评测记录。新增三组从头执行5000次更新和40,000次观察访问，matched/native 对内初始化和完整采样流哈希一致。',
        '',
        '新增训练沿用冻结渲染器，未修改光照标定或洁净参考缓存。主视角校验通过，腕视角保留原配方已知的合成差距。只请求纹理梯度的内存优化通过独立帧验证；原反向传播自身存在数值不确定性，不能宣称逐位更新一致。新、旧实现组及硬件信息随原始记录保存。',
        '',
        'Spatial 平均配对差：%+.2f pp，95%%区间 [%.2f, %.2f]；Object：%+.2f pp，95%%区间 [%.2f, %.2f]。'%(
            result['suites']['spatial']['mean'],result['suites']['spatial']['ci95'][0],result['suites']['spatial']['ci95'][1],
            result['suites']['object']['mean'],result['suites']['object']['ci95'][0],result['suites']['object']['ci95'][1]),
        '',
        '现有纹理组三组平均配对差 %+.2f pp；新训练组三组为 %+.2f pp。这两个分组均完整保留，不用于选择主要结论。'%(result['existing_only']['mean'],result['new_only']['mean']),
        '',
        '双视角共享与独立优化仍应依据单独的训练对照判断。本轮不把输入替换或几何一致性当作共享训练效果优势的证明。',
        '',
        '可复算记录：`episode_outcomes.csv`、`paired_suite_results.csv`、`seed_means.csv`、`summary.json`；逐回合的初始化、图像、噪声和动作哈希保存在原始评测记录。'
    ])
    (a.out/'STABILITY_RESULTS_CN.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print('Rendered all six paired replications; no selection by effect size.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
