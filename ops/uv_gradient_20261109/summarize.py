"""Retain every predeclared pair and assemble mechanism evidence and its limits."""
import csv
import itertools
import json
import math
import statistics
import subprocess
import tarfile
import time
from pathlib import Path

from common import OPS,REPO,ROOT,read,sha,write


def save_csv(path,rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def signflip(values):
    observed=abs(statistics.mean(values))
    return sum(abs(statistics.mean(s*v for s,v in zip(signs,values)))>=observed-1e-12
        for signs in itertools.product([-1,1],repeat=len(values)))/2**len(values)


def aggregate(values):
    return dict(n=len(values),mean=statistics.mean(values),
        sd=statistics.stdev(values) if len(values)>1 else None,min=min(values),max=max(values))


def main():
    destination=ROOT/'analysis';destination.mkdir(exist_ok=True)
    trajectories=[];pairs=[];heldout=[];final=[];registrations=[]
    for seed in [7,8,9]:
        for uv in ['on','off']:
            directory=ROOT/'arms'/('s%d_%s'%(seed,uv))
            completed=read(directory/'completed.json');assert completed and completed['completed']
            registrations.append(completed)
            for update in range(0,5001,500):
                record=read(directory/'probes'/('u%05d.json'%update));assert record
                assert record['parameter_and_rng_state_restored']
                for policy,values in record['policies'].items():
                    common=dict(seed=seed,train_uv=uv,updates=update,probe_policy=policy)
                    row=dict(common,mean_pair_cosine=values['feature_pair_cosine_mean'],
                        pooled_pair_cosine=values['pooled_feature_pair_cosine_mean'],
                        conflict_ratio=values['conflict_numerator']/values['conflict_denominator'],
                        conflict_numerator=values['conflict_numerator'],
                        conflict_denominator=values['conflict_denominator'],
                        theta_boundary_fraction=record['theta_boundary_fraction'],
                        tanh_derivative_below_001=record['tanh_derivative_below_001'])
                    trajectories.append(row)
                    pairs.extend(dict(common,estimator='replicate',**r) for r in values['pair_rows'])
                    pairs.extend(dict(common,estimator='pooled',**r) for r in values['pooled_pair_rows'])
                    if update==5000:
                        baseline=read(directory/'probes/heldout_initial.json')['policies'][policy]
                        for phase,source in [('initial',baseline),('final',values)]:
                            heldout.extend(dict(common,phase=phase,estimator='replicate',**r)
                                for r in source['heldout_rows'])
                            heldout.extend(dict(common,phase=phase,estimator='pooled',**r)
                                for r in source['pooled_heldout_rows'])
                        final.append(dict(row,heldout_alignment=values['pooled_heldout_cosine_mean'],
                            heldout_alignment_initial=baseline['pooled_heldout_cosine_mean'],
                            heldout_mean_replicate_alignment=values['heldout_cosine_mean'],
                            train_feature_mse=statistics.mean(values['feature_mse'][str(t)] for t in [0,2,6,8]),
                            heldout_feature_mse=statistics.mean(values['feature_mse'][str(t)] for t in [1,3,4,5,7,9])))
    contrasts=[]
    for seed in [7,8,9]:
        for policy in ['identity','augmented']:
            on=next(r for r in final if r['seed']==seed and r['train_uv']=='on' and r['probe_policy']==policy)
            off=next(r for r in final if r['seed']==seed and r['train_uv']=='off' and r['probe_policy']==policy)
            initial={uv:next(r for r in trajectories if r['seed']==seed and r['train_uv']==uv
                and r['updates']==0 and r['probe_policy']==policy) for uv in ['on','off']}
            row=dict(seed=seed,probe_policy=policy,on_pooled_cosine=on['pooled_pair_cosine'],
                off_pooled_cosine=off['pooled_pair_cosine'],cosine_difference=on['pooled_pair_cosine']-off['pooled_pair_cosine'],
                difference_in_changes=(on['pooled_pair_cosine']-initial['on']['pooled_pair_cosine'])-
                    (off['pooled_pair_cosine']-initial['off']['pooled_pair_cosine']),
                on_heldout=on['heldout_alignment'],off_heldout=off['heldout_alignment'],
                heldout_difference=on['heldout_alignment']-off['heldout_alignment'])
            for uv in ['on','off']:
                sampled=[r for r in trajectories if r['seed']==seed and r['train_uv']==uv and
                    r['probe_policy']==policy and r['updates']>0]
                row[uv+'_trajectory_conflict_ratio']=sum(r['conflict_numerator'] for r in sampled)/sum(r['conflict_denominator'] for r in sampled)
            row['conflict_reduction']=row['off_trajectory_conflict_ratio']-row['on_trajectory_conflict_ratio']
            contrasts.append(row)
    save_csv(destination/'trajectory_summary.csv',trajectories)
    save_csv(destination/'all_pair_gradients.csv',pairs)
    save_csv(destination/'all_heldout_alignments.csv',heldout)
    save_csv(destination/'final_per_arm.csv',final)
    save_csv(destination/'paired_seed_contrasts.csv',contrasts)
    metrics=['cosine_difference','difference_in_changes','heldout_difference','conflict_reduction']
    stats={policy:{metric:dict(aggregate([r[metric] for r in contrasts if r['probe_policy']==policy]),
        exact_signflip_p=signflip([r[metric] for r in contrasts if r['probe_policy']==policy]))
        for metric in metrics} for policy in ['identity','augmented']}
    summary=dict(completed=True,seeds=[7,8,9],training_arms=6,updates_per_arm=5000,
        sampled_checkpoints=11,probe_replicates=8,gradient_replicate_pair_rows=len(pairs),
        heldout_rows=len(heldout),statistics=stats,
        protocol_sha256=sha(OPS/'protocol.json'),heldout=read(ROOT/'prepared/heldout.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,universal_newlines=True).strip(),
        time=time.time())
    write(destination/'summary.json',summary)
    plot(destination,trajectories,contrasts)
    report(destination,summary,contrasts,final)
    # Include original new observations once, not the duplicated per-task recovery fragments.
    archive=ROOT/'uv_gradient_complete.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
        for name in ['analysis','arms','streams','training','legacy']:
            if (ROOT/name).exists():stream.add(ROOT/name,arcname=name)
        for file in sorted((ROOT/'prepared').iterdir()):
            if file.name.startswith('heldout_task_'):continue
            stream.add(file,arcname='prepared/'+file.name)
        stream.add(OPS,arcname='source',filter=lambda info:None if '__pycache__' in info.name else info)
    receipt=dict(completed=True,archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        report='analysis/UV增广梯度机制实验报告.md',code_commit=summary['code_commit'],time=time.time())
    write(ROOT/'accepted.json',receipt)
    print('ALL_UV_EXPERIMENTS_ACCEPTED',json.dumps(receipt),flush=True)


def plot(destination,rows,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=ROOT/'prepared/times.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']='Times New Roman'
    colors={'on':'#1f77b4','off':'#d96a32'}
    figure,axes=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
    for uv in ['on','off']:
        values=[];conflicts=[]
        for update in range(0,5001,500):
            subset=[r for r in rows if r['train_uv']==uv and r['updates']==update and r['probe_policy']=='augmented']
            values.append(aggregate([r['pooled_pair_cosine'] for r in subset]))
            conflicts.append(aggregate([r['conflict_ratio']*100 for r in subset]))
        x=list(range(0,5001,500))
        for axis,data in [(axes[0],values),(axes[1],conflicts)]:
            means=[r['mean'] for r in data];spread=[r['sd'] for r in data]
            axis.plot(x,means,label='UV-'+uv.capitalize(),color=colors[uv],marker='o',markersize=3)
            axis.fill_between(x,[a-b for a,b in zip(means,spread)],[a+b for a,b in zip(means,spread)],
                alpha=.15,color=colors[uv])
    axes[0].set(ylabel='Task gradient cosine',xlabel='Training updates',title='(a) Common augmented probe')
    axes[1].set(ylabel='Sampled conflicts (%)',xlabel='Training updates',title='(b) Negative-cosine frequency')
    for uv,index in [('on',0),('off',1)]:
        values=[r[uv+'_heldout'] for r in contrasts if r['probe_policy']=='augmented']
        axes[2].scatter([index]*len(values),values,color=colors[uv],zorder=3)
        axes[2].plot([index-.15,index+.15],[statistics.mean(values)]*2,color=colors[uv],linewidth=3)
    for row in contrasts:
        if row['probe_policy']=='augmented':axes[2].plot([0,1],[row['on_heldout'],row['off_heldout']],color='.75',linewidth=.8)
    axes[2].set(xticks=[0,1],xticklabels=['UV-On','UV-Off'],ylabel='Train-to-held-out gradient cosine',title='(c) Six held-out source tasks')
    for axis in axes:axis.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    figure.savefig(destination/'uv_gradient_mechanism.png',dpi=220)
    figure.savefig(destination/'uv_gradient_mechanism.svg')
    plt.close(figure)


def report(destination,summary,contrasts,arms):
    lines=['# UV 表面增广的梯度机制对照实验','',
        '本次检验 UV 增广是否使源任务梯度更一致、减少梯度冲突，并改善对未参与纹理训练任务的梯度对齐。共三组配对训练（7、8、9），每组 UV-On/Off 各训练 5,000 步；四个训练任务为 0、2、6、8，六个保留任务为 1、3、4、5、7、9。','',
        '两臂共享初始化、逐步访问帧、步长、损失及渲染配方。只切换 UV 变换。每 500 步测量一次，两臂分别接受完全相同的固定 UV 和随机 UV 测量。测量不修改纹理、模型或训练 RNG。','',
        '![梯度轨迹、冲突频率与保留任务对齐](uv_gradient_mechanism.png)','',
        '## 1. 配对结果','',
        '下表采用共同的随机 UV 测量。任务梯度余弦为先在测量样本上平均各任务梯度、再计算六个任务对的余弦。冲突频率采用训练后十个采样检查点、每点八组测量和六对任务；不是逐步记录全部 5,000 步。','',
        '| 组 | On 终点余弦 | Off 终点余弦 | On−Off | On 冲突率 | Off 冲突率 | On 保留对齐 | Off 保留对齐 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in contrasts:
        if row['probe_policy']!='augmented':continue
        lines.append('| %d | %.4f | %.4f | %+.4f | %.2f%% | %.2f%% | %.4f | %.4f |'%(
            row['seed'],row['on_pooled_cosine'],row['off_pooled_cosine'],row['cosine_difference'],
            row['on_trajectory_conflict_ratio']*100,row['off_trajectory_conflict_ratio']*100,
            row['on_heldout'],row['off_heldout']))
    lines+=['','## 2. 汇总与可支持的结论','']
    for metric,label in [('cosine_difference','终点任务梯度余弦差（On−Off）'),
                         ('difference_in_changes','相对初始化变化的配对差'),
                         ('conflict_reduction','采样冲突率减少量（Off−On）'),
                         ('heldout_difference','保留任务对齐差（On−Off）')]:
        row=summary['statistics']['augmented'][metric]
        lines.append('- %s：均值 %+.4f，组间样本标准差 %.4f；范围 [%+.4f, %+.4f]。'%(label,row['mean'],row['sd'],row['min'],row['max']))
    positive=all(row['cosine_difference']>0 for row in contrasts if row['probe_policy']=='augmented')
    held=all(row['heldout_difference']>0 for row in contrasts if row['probe_policy']=='augmented')
    lines+=['',('三组终点梯度余弦均有利于 UV-On。' if positive else '终点任务梯度余弦没有在所有配对组中一致支持 UV-On。'),
        ('六个保留任务的平均对齐差在三组中均有利于 UV-On。' if held else '保留任务对齐差没有在所有配对组中一致支持 UV-On。'),'',
        '这些测量检验的是冻结源策略的纹理目标梯度；不能单独推出内部表征具有任务不变性，也不直接等同于闭环攻击成功率。三次配对训练用于方向和幅度复验，任务对及重复测量不作为独立训练样本。','',
        '## 3. 测量控制与复现','',
        '- 攻击项原始梯度为主指标；完整目标（含共同正则）和边界可行梯度分列保存，见 all_pair_gradients.csv。',
        '- 固定 UV 的敏感性结果完整保存在 paired_seed_contrasts.csv；随机 UV 与固定 UV 的效应不混合。',
        '- 保留任务仅用于冻结纹理后的诊断，采集初始纹理的结果也在训练完成后计算。',
        '- 旧 0.078→0.238 使用任务相关的 UV 测量；本次采用跨任务共同 UV，两个数值定义不同。legacy 目录单列原纹理复测，不并入新配对组。',
        '- 沿用冻结渲染器及已授权的腕部光度例外；没有重新标定渲染器。',
        '- 负余弦按 [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) 的梯度冲突定义计数；本实验没有执行梯度投影或修改优化器。','',
        '完整逐组、逐检查点和逐任务结果见同目录 CSV 与 summary.json。所有预先约定的训练组均保留。','']
    (destination/'UV增广梯度机制实验报告.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
