"""Create tables and plots from the full camera-control comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ORDER=['mr','uv_ind','consensus_raw','consensus_rms']
LABELS=['Shared UV','Independent UV','Independent views: mean','Independent views: RMS matched']
SHORT=['Shared UV','Independent UV','Mean consensus','RMS-matched consensus']


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.analysis/'summary.json').read_text());assert summary['completed'] and summary['new_episodes']==1800
    methods=pd.read_csv(a.analysis/'method_suite_results.csv');means=pd.read_csv(a.analysis/'seed_means.csv')
    aggregate=[]
    for method in ORDER:
        row={'method':method}
        for suite in ['spatial','object','mean']:
            values=(methods[methods.method==method].groupby('seed').G.mean().values if suite=='mean'
                else methods[(methods.method==method)&(methods.suite==suite)].sort_values('seed').G.values)
            assert len(values)==3
            row[suite+'_mean']=float(np.mean(values));row[suite+'_sd']=float(np.std(values,ddof=1))
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(str(a.out/'coupling_aggregate.csv'),index=False)
    table=r'''\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.1pt}
\caption{\textbf{Camera consistency and independent-view training.} Net success reduction $G$ (pp) on $\pi_{0.5}$; mean $\pm$ sample deviation over three paired seeds. Every row deploys one frozen cloth through the native simulator. Independent-view training uses two complete training arms per seed; the last row matches the shared texture's RMS using texture statistics only.}
\label{tab:camera_coupling}
\begin{tabular}{@{}lrrr@{}}
\toprule
Training / deployment & Spatial & Object & Mean\\
\midrule
'''
    tex_labels=['Shared UV','UV decoupled','Indep. views: mean','Indep. views: RMS']
    for label,row in zip(tex_labels,aggregate):
        cells=['$%.1f\\pm%.1f$'%(row[s+'_mean'],row[s+'_sd']) for s in ['spatial','object','mean']]
        table+=label+' & '+' & '.join(cells)+r'\\'+'\n'
    table+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (a.out/'camera_coupling.tex').write_text(table)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.45,2.65));colors=['#446F99','#B34C3D','#56805D']
    for index,seed in enumerate([10,11,12]):
        values=[float(methods[(methods.seed==seed)&(methods.method==method)].G.mean()) for method in ORDER]
        ax.plot(range(4),values,'-o',lw=.9,ms=3.5,color=colors[index],label='Seed %d'%seed)
    ax.set_xticks(range(4));ax.set_xticklabels(['Shared\nUV','Independent\nUV','Independent\nmean','Independent\nRMS match'])
    ax.set_ylabel('Two-suite net reduction G (pp)');ax.grid(axis='y',alpha=.16)
    ax.legend(frameon=False,ncol=3,loc='upper center',fontsize=7.1,columnspacing=.7)
    ax.set_ylim(min(0,float(methods.G.min())-5),max(100,float(methods.G.max())+5))
    fig.subplots_adjust(left=.17,right=.985,bottom=.2,top=.94)
    for ext in ['pdf','png']:fig.savefig(str(a.out/('camera_coupling.'+ext)),dpi=260,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    manifest=json.loads((a.root/'texture_manifest.json').read_text())
    base=np.asarray(Image.open(a.root/manifest['plain']['path']).convert('RGB')).astype(float)
    texture_rows=[]
    for seed in [10,11,12]:
        for method in ORDER:
            image=np.asarray(Image.open(a.root/manifest['s%d_%s'%(seed,method)]['path']).convert('RGB')).astype(float)
            delta=image-base
            texture_rows.append(dict(seed=seed,method=method,rms=float(np.sqrt(np.mean(delta**2))),
                linf=float(np.abs(delta).max()),spatial_tv=float(np.abs(np.diff(delta,axis=0)).mean()+np.abs(np.diff(delta,axis=1)).mean())))
    pd.DataFrame(texture_rows).to_csv(str(a.out/'texture_statistics.csv'),index=False)
    lines=['# 双视角对照：完整结果','','所有条件沿用原任务、相机、固定桌布载体和原生闭环部署。全部三组种子及两种合成版本均纳入分析。','',
        '| 条件 | Spatial G | Object G | 两套件平均 G |','| --- | ---: | ---: | ---: |']
    for label,row in zip(LABELS,aggregate):
        lines.append('| %s | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f |'%(label,row['spatial_mean'],row['spatial_sd'],row['object_mean'],row['object_sd'],row['mean_mean'],row['mean_sd']))
    lines.extend(['','下列差值为 Shared G 减去对照 G，正值表示共享同步训练的攻击更强：','',
        '| 种子 | 相比独立 UV | 相比原始均值合成 | 相比 RMS 匹配合成 |','| --- | ---: | ---: | ---: |'])
    for seed in [10,11,12]:
        t=means[means.seed==seed].set_index('variant')
        lines.append('| %d | %+.2f | %+.2f | %+.2f |'%(seed,t.loc['uv_ind','shared_advantage'],t.loc['consensus_raw','shared_advantage'],t.loc['consensus_rms','shared_advantage']))
    for variant,label in [('uv_ind','UV 同步对照'),('consensus_rms','独立训练后幅度匹配合成'),('consensus_raw','独立训练后原始合成')]:
        r=(summary['secondary'] if variant=='consensus_raw' else summary['primary'])[variant]
        lines.extend(['','%s：平均配对差 **%+.2f pp**，样本标准差 %.2f，95%% t 区间 [%.2f, %.2f]，三组种子的双侧精确符号翻转 p=%.3f。'%(label,r['mean'],r['sd'],r['ci95'][0],r['ci95'][1],r['exact_two_sided_sign_flip_p'])])
    lines.extend(['','UV 对照只改变两路变换的关联，参数、初始化、主相机 UV 和观察序列、更新预算均配对；它不单独检验任意两张纹理的参数自由度。',
        '', '独立视角条件分别优化主、腕两张纹理，每张均训练5000步，因此训练参数量和源前向次数为共享条件的两倍。两张纹理先做 UV 空间欧氏均值合成，再增加仅依据纹理统计的 RMS 匹配控制。二者都输出一张固定 PNG，由同一个场景表面同时进入两路相机。',
        '', '均值合成是纹理空间的投影，不是相机图像空间的最优拟合；RMS 匹配也不保证 TV 一致。完整纹理统计与合成残差已保留，结论应限定为这些实际比较。',
        '', '三组训练对照的精确双侧符号翻转检验最小 p 为0.25；不能把大量评测回合当成大量独立训练来证明总体显著性。逐套件配对 McNemar 检验只描述这些固定纹理，主要两种比较在12项检验内做 Holm 校正。',
        '', '本块包含2600条记录，其中800条为按纹理和协议逐字节核验的共用 Plain／共享纹理记录，新增1800回合。加上参考复验，两块合计4400个独立执行的回合。',
        '', '若某项结果为负或混合，不能据此宣称共享方式有性能优势；若共享更强，也不能扩大成所有独立视角攻击都更弱。'])
    (a.out/'COUPLING_RESULTS_CN.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print('Rendered every predeclared camera-control condition.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
