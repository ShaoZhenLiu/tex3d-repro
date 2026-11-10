"""Accept all 400 episodes, validate pairing and report every planned contrast."""
import csv
import itertools
import math
import os
import subprocess
import tarfile
import time
from common import HERE,ROOT,PROTOCOL,read,write,sha


def exact_mcnemar(a,b):
    stronger=sum(x and not y for x,y in zip(a,b))
    weaker=sum(y and not x for x,y in zip(a,b))
    n=stronger+weaker
    p=min(1.,2.*sum(math.comb(n,k) for k in range(min(stronger,weaker)+1))/2**n) if n else 1.
    return stronger,weaker,p


def main():
    p=read(PROTOCOL)['closed_loop'];rows=[];lookup={}
    assert read(ROOT/'smoke_passed.json')['passed']
    for task in p['tasks']:
        for initial in p['initials']:
            assert read(ROOT/'completed'/('t%d_i%d.json'%(task,initial)))['completed']
            group=[]
            for condition in p['conditions']:
                r=read(ROOT/'episodes'/('object_t%d_i%d_%s.json'%(task,initial,condition)))
                assert r['completed'] and r['condition']==condition and r['hook']
                assert r['protocol_sha256']==sha(PROTOCOL)
                assert all(q['measurement']['integration_exact'] and q['measurement']['cached_exposure_exact'] for q in r['trajectory'])
                for q in r['trajectory']:
                    m=q['measurement']
                    expected_main=m['ref_main_sha256'] if condition in ['main_only','both'] else m['plain_main_sha256']
                    expected_wrist=m['ref_wrist_sha256'] if condition in ['wrist_only','both'] else m['plain_wrist_sha256']
                    assert q['main_sha256']==expected_main and q['wrist_sha256']==expected_wrist
                lookup[(task,initial,condition)]=r;group.append(r)
                rows.append({k:r[k] for k in ['task','initial','condition','robot_success','queries','elapsed_seconds','worker','gpu','host']})
            assert len({r['gpu'] for r in group})==1 and len({r['host'] for r in group})==1
            assert len({r['inputs']['texture_pixel_sha256'] for r in group})==1
            for field in ['plain_main_sha256','plain_wrist_sha256','camera_qpos_sha256']:
                assert len({r['trajectory'][0]['measurement'][field] for r in group})==1
            for r in group[1:]:
                assert all(a['noise_sha256']==b['noise_sha256'] for a,b in zip(group[0]['trajectory'],r['trajectory']))
    assert len(rows)==400
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=True)
    with (analysis/'episode_outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=[(t,i) for t in p['tasks'] for i in p['initials']]
    outcomes={c:[bool(lookup[(t,i,c)]['robot_success']) for t,i in keys] for c in p['conditions']}
    successes={c:sum(v) for c,v in outcomes.items()}
    contrasts=[]
    for a,b in itertools.combinations(p['conditions'],2):
        ab,ba,pvalue=exact_mcnemar(outcomes[a],outcomes[b])
        contrasts.append(dict(first=a,second=b,first_success_second_fail=ab,first_fail_second_success=ba,
            success_difference_pp=successes[a]-successes[b],p=pvalue))
    ordered=sorted(contrasts,key=lambda r:r['p']);previous=0.
    for index,row in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-index)*row['p']));row['p_holm']=previous
    per_task={str(t):{c:sum(lookup[(t,i,c)]['robot_success'] for i in p['initials']) for c in p['conditions']} for t in p['tasks']}
    conditional=dict(
        wrist_drop_given_plain_main_pp=successes['plain']-successes['wrist_only'],
        wrist_drop_given_adversarial_main_pp=successes['main_only']-successes['both'],
        main_drop_given_plain_wrist_pp=successes['plain']-successes['main_only'],
        main_drop_given_adversarial_wrist_pp=successes['wrist_only']-successes['both'],
        joint_extra_drop_pp=successes['main_only']+successes['wrist_only']-successes['plain']-successes['both'],
        both_fail_main_only_success=sum(not b and m for b,m in zip(outcomes['both'],outcomes['main_only'])),
        both_fail_wrist_only_success=sum(not b and w for b,w in zip(outcomes['both'],outcomes['wrist_only'])))
    summary=dict(completed=True,episodes=400,paired_keys=100,successes=successes,
        net_drop_pp={c:successes['plain']-successes[c] for c in p['conditions']},
        contrasts=contrasts,per_task=per_task,conditional_camera_effects=conditional,protocol_sha256=sha(PROTOCOL),time=time.time())
    write(analysis/'closed_loop_summary.json',summary)
    report=['# 冻结 RefCloth 的目标策略相机干预实验','',
        'π₀.₅-Object，10 个任务 × 10 个初始状态；每个状态比较四个条件，共 400 回合。纹理、策略、部署几何和推理设置固定。',
        '所有条件按任务与初始状态配对，使用相同查询序号的显式去噪噪声。图像替换在各自轨迹的同一相机曝光状态完成。',
        '', '| 条件 | 成功 /100 | 相对 Plain 净下降 G（pp） |','|---|---:|---:|']
    for c in p['conditions']:report.append('| %s | %d | %d |'%(c,successes[c],successes['plain']-successes[c]))
    report.extend(['','| 配对比较 | 成功率差（前−后，pp） | 不一致回合（前成功/后成功） | 精确 p | Holm p |',
        '|---|---:|---:|---:|---:|'])
    for r in contrasts:report.append('| %s / %s | %+d | %d / %d | %.5g | %.5g |'%(r['first'],r['second'],r['success_difference_pp'],r['first_success_second_fail'],r['first_fail_second_success'],r['p'],r['p_holm']))
    report.extend(['','描述性条件效应：腕路扰动在主视角为 Plain / RefCloth 时分别使成功率下降 %d / %d pp；主路扰动在腕视角为 Plain / RefCloth 时分别下降 %d / %d pp。'%(conditional['wrist_drop_given_plain_main_pp'],conditional['wrist_drop_given_adversarial_main_pp'],conditional['main_drop_given_plain_wrist_pp'],conditional['main_drop_given_adversarial_wrist_pp']),
        '联合净下降超出两种单路净下降之和 %d pp。这是当前冻结纹理的成功率尺度输入交互，不用于推断哪种训练方式更优。'%conditional['joint_extra_drop_pp']])
    report.extend(['','Main-only/Wrist-only 是相机输入机制干预；它们的混合图像不表示一张物理桌布的独立部署。',
        'Plain 与 Both 已通过直接原生部署的完整轨迹逐查询一致性检查。显著性检验不用于宣布两条件等效。',
        '同目录保留所有任务、回合、输入 SHA、显式噪声 SHA、动作及实现校验。'])
    (analysis/'目标策略相机干预实验报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python',str(HERE/'plot_results.py')],env=env,check=True)
    archive=ROOT/'mechanism_complete.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['analysis','episodes','completed','workers','protocol.json','smoke_passed.json','run_manifest.json','fresh_feature_direction']:
            tar.add(ROOT/name,arcname=name)
        tar.add(HERE,arcname='source',filter=lambda item:None if '__pycache__' in item.name else item)
    write(ROOT/'accepted.json',dict(completed=True,episodes=400,archive=archive.name,
        sha256=sha(archive),bytes=archive.stat().st_size,time=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        inference_entry_sha256=sha(HERE/'evaluate.py'),summary_entry_sha256=sha(__file__)))
    print(summary,flush=True)


if __name__=='__main__':main()
