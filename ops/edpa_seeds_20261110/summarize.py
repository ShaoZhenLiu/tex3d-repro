"""Accept complete observations and generate the three-method Table S9 fragment."""
import csv
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from common import HERE,REPO,ROOT,SEEDS,SUITES,INITIALS,read,write,sha,jobs
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(accept=False):
    episode_files=list((ROOT/'evaluation/episodes').glob('*_edpa.json'))
    if len(episode_files)!=1600:return dict(completed=False,edpa_episodes=len(episode_files),planned=1600)
    if accept:
        assert read(ROOT/'evaluation_done_4090.json',{}).get('completed')
        assert read(ROOT/'evaluation_done_09.json',{}).get('completed')
        assert not any(e['root']==str(ROOT) for e in guard.active()),'Wait for native workers to exit'
    output=ROOT/'analysis';output.mkdir(exist_ok=True)
    protocol=read(HERE/'protocol.json');assert read(ROOT/'inputs.json')['protocol_sha256']==sha(HERE/'protocol.json')
    outcomes=list(csv.DictReader((ROOT/'baseline/outcomes.csv').open()))
    assert len(outcomes)==3600 and sha(ROOT/'baseline/outcomes.csv')==read(ROOT/'inputs.json')['baseline_csv_sha256']
    for name,entry in read(ROOT/'baseline/manifest.json').items():
        path=ROOT/('evaluation/episodes' if entry['method']=='plain' else 'baseline/episodes')/name
        assert sha(path)==entry['sha256']
    conditions=read(ROOT/'texture_manifest.json');source={}
    for seed in SEEDS:
        meta=read(ROOT/'arms'/('s%d_edpa'%seed)/'completed.json')
        assert meta['completed'] and meta['iterations']==5000 and meta['frame_visits']==40000
        assert meta['final_ema']['step']==5000 and meta['linf']<=64.0001
        assert sha(ROOT/'frozen'/('s%d_edpa.png'%seed))==meta['texture_sha256']
        source[seed]=meta
    for seed in SEEDS:
        for suite in SUITES:
            for task in range(10):
                for initial in INITIALS:
                    name='%s_t%d_i%d_s%d_edpa.json'%(suite,task,initial,seed)
                    d=read(ROOT/'evaluation/episodes'/name)
                    assert d['completed'] and d['training_seed']==seed and d['reference']=='cloth_edpa'
                    assert d['target']=='pi05' and d['texture_sha256']==source[seed]['texture_sha256']
                    assert d['host'].endswith('-41') if suite!='long' else d['host']=='zgc-33-9'
                    plain=read(ROOT/'evaluation/episodes'/('%s_t%d_i%d_plain.json'%(suite,task,initial)))
                    if d['trajectory'] and plain['trajectory']:
                        assert d['trajectory'][0]['proprio_sha256']==plain['trajectory'][0]['proprio_sha256']
                    else:assert not d['trajectory'] and not plain['trajectory'] and d['robot_success'] and plain['robot_success']
                    assert all(q['query']==i for i,q in enumerate(d['trajectory']))
                    assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(d['trajectory'],plain['trajectory']))
                    worker=ROOT/'evaluation/workers'/d['worker']
                    assert read(worker/'calibration.json')['passed']
                    assert read(worker/'registered.json')['protocol_sha256']==sha(HERE/'protocol.json')
                    outcomes.append(dict(seed=seed,method='edpa',suite=suite,task=task,initial=initial,success=int(d['robot_success']),file=name))
    assert len(outcomes)==5200
    suite_rows=[];means=[]
    for seed in SEEDS:
        record=dict(seed=seed)
        for suite in SUITES:
            row=dict(seed=seed,suite=suite,n=100)
            pp=[r for r in outcomes if r['method']=='plain' and r['suite']==suite];assert len(pp)==100
            plain_success=sum(int(r['success']) for r in pp);row['Plain']=plain_success
            for method in ['native','edpa','mr']:
                subset=[r for r in outcomes if r['method']==method and r['suite']==suite and int(r['seed'])==seed]
                assert len(subset)==100
                succeeded=sum(int(r['success']) for r in subset)
                row[method+'_success']=succeeded;row[method+'_G']=plain_success-succeeded
            suite_rows.append(row)
        for method in ['native','edpa','mr']:
            record[method+'_G']=statistics.mean(r[method+'_G'] for r in suite_rows if r['seed']==seed)
        record['mr_minus_native']=record['mr_G']-record['native_G']
        record['mr_minus_edpa']=record['mr_G']-record['edpa_G']
        means.append(record)
    # Compare with the already accepted reference archive; never substitute
    # expected numbers for a counted observation.
    expected={42:(25.75,35.5),7:(27.,45.5),66:(27.25,45.25),88:(30.5,32.)}
    for r in means:assert (r['native_G'],r['mr_G'])==expected[r['seed']]
    aggregate={method:dict(mean=statistics.mean(r[method+'_G'] for r in means),
        sample_sd=statistics.stdev(r[method+'_G'] for r in means)) for method in ['native','edpa','mr']}
    ordering=aggregate['mr']['mean']>aggregate['edpa']['mean']>aggregate['native']['mean']
    summary=dict(completed=True,seeds=SEEDS,suites=SUITES,new_training_arms=4,new_edpa_episodes=1600,
        reused_reference_episodes=3200,reused_plain_episodes=400,total_observations=5200,
        seed_means=means,aggregate=aggregate,anticipated_mean_ordering_observed=ordering,
        unit='Source-training run; sample SD uses ddof=1 over four seed macro-averages',
        mr_vs_edpa_wins=sum(r['mr_minus_edpa']>0 for r in means),
        mr_vs_edpa_ties=sum(r['mr_minus_edpa']==0 for r in means),
        baseline_counts_match_accepted_archive=True,
        current_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,universal_newlines=True).strip())
    csv_write(output/'episode_outcomes.csv',outcomes);csv_write(output/'four_suite_results.csv',suite_rows);csv_write(output/'seed_means.csv',means)
    write(output/'summary.json',summary)
    names={'native':'Native ref.','edpa':'Cloth-adapted EDPA','mr':'RefCloth (Matched)'}
    text=['# Cloth-adapted EDPA 四种子补充实验','',
        '固定训练组为 42、7、66、88。每张 EDPA 纹理训练 5,000 步、batch 8，只使用最终导出纹理，在 π0.5 的四套件、40 个任务上评测。每个套件为 100 回合，状态编号 10–19；G 为 Plain 成功率减去攻击后成功率，单位为百分点。','',
        '| Seed | Native ref. 平均 G | Cloth-EDPA 平均 G | RefCloth 平均 G | MR − EDPA |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for r in means:text.append('| {seed} | {native_G:.2f} | {edpa_G:.2f} | {mr_G:.2f} | {mr_minus_edpa:+.2f} |'.format(**r))
    text.append('| Mean ± SD | '+' | '.join('%.2f ± %.2f'%(aggregate[m]['mean'],aggregate[m]['sample_sd']) for m in ['native','edpa','mr'])+' | — |')
    text+=['','完成 4 个训练臂和 1,600 个新的 EDPA 评测回合；精确复用并校验 3,200 个 Native/Matched 回合与 400 个 Plain 回合。','',
        '四种子均值排序 RefCloth > Cloth-EDPA > Native：'+('本次成立。' if ordering else '本次不成立，以上表实测值为准。'),
        'MR 相对 EDPA 在 %d/4 组更强，%d/4 组持平。'%(summary['mr_vs_edpa_wins'],summary['mr_vs_edpa_ties']),'',
        '统计标准差以每个训练组的四套件平均 G 为单位，并非将评测回合当成独立训练重复。此处报告本次固定组的对照结果。','',
        '| Seed | EDPA 源设备 | 纹理 RMS | L∞ | 空间 TV |',
        '| ---: | --- | ---: | ---: | ---: |']
    for seed in SEEDS:
        e=source[seed];text.append('| %d | %s | %.3f | %.3f | %.3f |'%(seed,e['gpu_name'],e['rms'],e['linf'],e['tv']))
    text+=['','完整套件计数、逐回合结果、训练注册信息、完整 EMA 检查点和 SHA 校验清单随结果归档保存。',
        '输入路径、动作噪声和初始状态按既有配对协议核验；Spatial/Object/Goal 在 41，Long 在 09，与复用数据的推理硬件保持一致。']
    (output/'EDPA四种子复验.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    latex=[r'\subsection{Comparison of Source Objectives}',r'\label{supp:reference_runs}',
        r'To compare source objectives under a common cloth carrier and training budget, we evaluate Native reference, cloth-adapted EDPA, and RefCloth on four fixed source-training runs. EDPA receives 5,000 updates at batch size eight with the same observation pool, initialization and frame/UV stream. Each final texture is evaluated on all four $\pi_{0.5}$ suites with 100 episodes per suite. Reference outcomes and Plain controls are reused from the verified paired evaluation block.',
        r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{\textbf{Four-run source-objective comparison.} Four-suite macro-averaged $G$ (pp) on $\pi_{0.5}$. Higher is stronger. Mean and sample SD are computed over the four fixed source-training runs. Bold marks the strongest attack in each row.}',
        r'\label{tab:supp_reference_cases}',r'\begin{tabular}{@{}rrrr@{}}\toprule',
        r'Seed & Native ref. & Cloth-EDPA & RefCloth\\\midrule']
    for r in means:
        best=max(r[m+'_G'] for m in ['native','edpa','mr'])
        cells=[(r'\textbf{%.2f}'%r[m+'_G']) if r[m+'_G']==best else '%.2f'%r[m+'_G'] for m in ['native','edpa','mr']]
        latex.append(str(r['seed'])+' & '+' & '.join(cells)+r'\\')
    latex+=[r'\midrule',r'Mean $\pm$ SD & '+' & '.join('$%.2f\pm%.2f$'%(aggregate[m]['mean'],aggregate[m]['sample_sd']) for m in ['native','edpa','mr'])+r'\\\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'table_s9_edpa.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    if accept:
        write(ROOT/'protocol_snapshot.json',protocol)
        archive=ROOT/'edpa_four_seed_complete.tar.gz'
        if not archive.exists():
            temporary=archive.with_suffix('.tmp')
            with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
                for name in ['analysis','arms','streams','baseline','evaluation','frozen','logs','inputs.json','protocol_snapshot.json','texture_manifest.json','gradient_validation.json','loss_cpu_validation.json','remote_provenance','errors']:
                    path=ROOT/name
                    if path.exists():tar.add(path,arcname=name)
            temporary.replace(archive)
        receipt=dict(completed=True,training_arms=4,new_edpa_episodes=1600,reused_plain_episodes=400,
            reused_reference_episodes=3200,archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
            time=time.time(),aggregate=aggregate,anticipated_ordering_observed=ordering)
        write(ROOT/'accepted.json',receipt)
    return summary


if __name__=='__main__':
    print(json.dumps(analyze('--accept' in sys.argv),ensure_ascii=False))
