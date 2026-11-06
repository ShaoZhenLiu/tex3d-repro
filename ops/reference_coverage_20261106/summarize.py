# coding: utf-8
"""Validate the full extension and retain every suite and every suite-pair contrast."""
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from common import ROOT, SEEDS, INITIALS, NODES, jobs, sha, suite_root


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    records={};episode_rows=[];worker_cache={};all_hashes={};reused_count=0
    historical=ROOT/'existing_spatial_object_outcomes.csv'
    assert sha(historical)==sha(ROOT/'inputs'/historical.name)
    with historical.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key=(row['suite'],int(row['task']),int(row['initial']),row['condition'])
            assert key not in records and key[0] in ['spatial','object']
            records[key]=dict(success=int(row['success']))
            episode_rows.append(dict(suite=key[0],task=key[1],initial=key[2],condition=key[3],success=int(row['success']),
                worker=row['worker'],host=row.get('host',''),gpu=row.get('gpu',''),source='existing_spatial_object'))
    assert len(records)==7400
    for suite,node in NODES.items():
        root=suite_root(suite);manifest=json.loads((root/'texture_manifest.json').read_text())
        reused=json.loads((root/'evaluation/reused_manifest.json').read_text());reused_count+=len(reused)
        protocol=json.loads((root/'protocol.json').read_text())['evaluation']
        for condition,v in manifest.items():
            assert sha(root/v['path'])==v['sha256']
            assert all_hashes.setdefault(condition,v['sha256'])==v['sha256']
        files=list((root/'evaluation/episodes').glob('*.json'));assert len(files)==3700
        for file in files:
            e=json.loads(file.read_bytes());condition=e['condition']
            assert e['completed'] and e['suite']==suite and e.get('target')=='pi05'
            assert e['task'] in range(10) and e['initial'] in INITIALS and condition in manifest
            assert e['texture_sha256']==manifest[condition]['sha256'] and e['native_uploads']>0
            rows=e['trajectory'];assert e['queries']==len(rows)
            assert [r['query'] for r in rows]==list(range(len(rows)))
            for row in rows:
                assert len(row['actions'])==5 and all(len(a)==7 for a in row['actions'])
                assert all(math.isfinite(v) for a in row['actions'] for v in a)
            key=(suite,e['task'],e['initial'],condition);assert key not in records
            records[key]=dict(success=int(e['robot_success']),proprio=rows[0]['proprio_sha256'] if rows else None,
                              noise=[r['noise_sha256'] for r in rows],host=e['host'])
            wk=(suite,e['worker'])
            if wk not in worker_cache:
                p=root/'evaluation/workers'/e['worker'];reg=json.loads((p/'registered.json').read_text());cal=json.loads((p/'calibration.json').read_text())
                assert reg['checkpoint']==protocol['checkpoint'] and cal['passed']
                assert cal.get('trajectory_exact') if node=='4090' else (cal.get('success_repeatable') and cal.get('mode')=='success')
                worker_cache[wk]=dict(registration=reg,calibration=cal)
            assert e['host'].endswith('-41') if node=='4090' else e['host']=='zgc-33-9'
            if file.name in reused:assert sha(file)==reused[file.name]['sha256']
            episode_rows.append(dict(suite=suite,task=e['task'],initial=e['initial'],condition=condition,
                success=int(e['robot_success']),worker=e['worker'],host=e['host'],gpu=e['gpu'],
                source='exact_reuse' if file.name in reused else 'new_extension'))
        for job in jobs(suite):
            done=json.loads((root/'evaluation/completed'/(job['id']+'.json')).read_text())
            assert done['completed'] and done['job']==job
            assert {(r['condition'],r['initial']) for r in done['outcomes']}=={(c,i) for c in job['conditions'] for i in INITIALS}
            for r in done['outcomes']:assert int(r['robot_success'])==records[suite,job['task'],r['initial'],r['condition']]['success']
        for seed in SEEDS:
            for task in range(10):
                for initial in INITIALS:
                    p,n,m=[records[suite,task,initial,c] for c in ['plain','s%d_native'%seed,'s%d_mr'%seed]]
                    assert p['host']==n['host']==m['host']
                    for other in [p,m]:
                        assert n['proprio']==other['proprio']
                        assert all(a==b for a,b in zip(n['noise'],other['noise']))
    assert len(records)==14800 and len(episode_rows)==14800 and reused_count==1400
    suites=['spatial','object','goal','long'];tables=[];tasks=[]
    for suite in suites:
        for seed in SEEDS:
            counts={c:sum(records[suite,t,i,c]['success'] for t in range(10) for i in INITIALS)
                    for c in ['plain','s%d_native'%seed,'s%d_mr'%seed]}
            p,n,m=[counts[c] for c in ['plain','s%d_native'%seed,'s%d_mr'%seed]]
            tables.append(dict(seed=seed,suite=suite,n=100,Plain=p,Native=n,Matched=m,Native_G=p-n,Matched_G=p-m,delta_G=n-m))
            for task in range(10):
                p,n,m=[sum(records[suite,task,i,c]['success'] for i in INITIALS) for c in ['plain','s%d_native'%seed,'s%d_mr'%seed]]
                tasks.append(dict(seed=seed,suite=suite,task=task,n=10,Plain=p,Native=n,Matched=m,
                                  Native_G=10*(p-n),Matched_G=10*(p-m),delta_G=10*(n-m)))
    average=lambda block,key:statistics.mean(r[key] for r in block)
    seed_means=[dict(seed=s,**{k:average([r for r in tables if r['seed']==s],k) for k in ['Native_G','Matched_G','delta_G']}) for s in SEEDS]
    suite_means=[dict(suite=s,**{k:average([r for r in tables if r['suite']==s],k) for k in ['Native_G','Matched_G','delta_G']}) for s in suites]
    contrasts=[]
    for a,b in itertools.combinations(suites,2):
        block=[r for r in tables if r['suite'] in [a,b]]
        deltas=[average([r for r in block if r['seed']==s],'delta_G') for s in SEEDS]
        contrasts.append(dict(suite_pair=a+'+'+b,Native_G=average(block,'Native_G'),Matched_G=average(block,'Matched_G'),
            delta_G=average(block,'delta_G'),positive_pairs=sum(x>0 for x in deltas),negative_pairs=sum(x<0 for x in deltas),
            ties=sum(x==0 for x in deltas),interpretation='exploratory'))
    for name,rows in [('four_suite_results',tables),('four_suite_seed_means',seed_means),('suite_means',suite_means),
                      ('suite_pair_exploration',contrasts),('per_task_results',tasks),('episode_outcomes',episode_rows)]:
        save_csv(out/(name+'.csv'),rows)
    text=['# 十八组参考选择：四套件覆盖', '',
        'π0.5 的 Spatial/Object 沿用已核验的十八组成对记录；Goal 全部在 41，Long 全部在 09。新增 6,000 回合，精确复用 Goal/Long 各 700 回合。新增两套件共 7,400 回合，与此前两套件合计 14,800 个不同回合。', '',
        '主要汇总包含全部四套件。以下两套件组合是完整列出的探索性比较，不作为事前指定的稳定性终点。G 越大攻击越强，ΔG=MR G−Native G，单位为百分点。', '',
        '| Seed | 四套件 Native G | 四套件 MR G | ΔG |','| ---: | ---: | ---: | ---: |']
    for r in seed_means:text.append('| {seed} | {Native_G:.3f} | {Matched_G:.3f} | {delta_G:+.3f} |'.format(**r))
    text+=['','## 套件差异','','| 套件 | Native G | MR G | ΔG |','| --- | ---: | ---: | ---: |']
    for r in suite_means:text.append('| {suite} | {Native_G:.3f} | {Matched_G:.3f} | {delta_G:+.3f} |'.format(**r))
    text+=['','## 全部六种套件组合（探索分析）','','| 组合 | Native G | MR G | ΔG | MR较强／Native较强／持平种子数 |','| --- | ---: | ---: | ---: | --- |']
    for r in contrasts:text.append('| {suite_pair} | {Native_G:.3f} | {Matched_G:.3f} | {delta_G:+.3f} | {positive_pairs}/{negative_pairs}/{ties} |'.format(**r))
    (out/'十八组四套件参考比较.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(completed=True,extension_episodes=7400,new_episodes=6000,
        reused_extension_episodes=1400,four_suite_episodes=14800,seeds=SEEDS,seed_means=seed_means,suite_means=suite_means,
        suite_pair_exploration=contrasts,all_endpoints_retained=True,workers={s+'/'+w:v for (s,w),v in worker_cache.items()}),indent=2),encoding='utf-8')
    print(json.dumps(dict(completed=True,extension=7400,combined=14800)))


if __name__=='__main__':main()
