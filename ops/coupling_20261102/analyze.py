"""Recompute every camera-control result, including unfavorable comparisons."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


VARIANTS=['uv_ind','consensus_raw','consensus_rms']


def interval(values):
    v=np.asarray(values,dtype=float);mean=float(v.mean());sd=float(v.std(ddof=1))
    half=float(stats.t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v)))
    null=[abs(float(np.mean(v*np.asarray(s)))) for s in itertools.product([-1,1],repeat=len(v))]
    return dict(n=len(v),mean=mean,sd=sd,ci95=[mean-half,mean+half],minimum=float(v.min()),maximum=float(v.max()),
        exact_two_sided_sign_flip_p=float(np.mean(np.asarray(null)>=abs(mean)-1e-12)),
        positive_pairs=int((v>0).sum()),negative_pairs=int((v<0).sum()),zero_pairs=int((v==0).sum()))


def mcnemar(b,c):
    n=int(b+c)
    if not n:return 1.
    mass=sum(math.factorial(n)//(math.factorial(k)*math.factorial(n-k)) for k in range(min(b,c)+1))
    return min(1.,2.*mass/2**n)


def holm(values):
    order=np.argsort(values);out=np.zeros(len(values));last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,float(values[index])*(len(values)-rank)));out[index]=last
    return out


def main(a):
    paths=sorted((a.root/'evaluation/episodes').glob('*.json'));assert len(paths)==2600,len(paths)
    manifest=json.loads((a.root/'texture_manifest.json').read_text());assert len(manifest)==13
    for item in manifest.values():assert hashlib.sha256((a.root/item['path']).read_bytes()).hexdigest()==item['sha256']
    a.out.mkdir(parents=True,exist_ok=True);raw={};rows=[];reuse=0
    for path in paths:
        item=json.loads(path.read_text());assert item['completed'] and len(item['trajectory'])==item['queries'] and item['native_uploads']>=1
        key=tuple(item[k] for k in ['suite','task','initial','condition']);assert key not in raw
        assert item['texture_sha256']==manifest[item['condition']]['sha256']
        raw[key]=item
        if item['condition']=='plain' or item['condition'].endswith('_mr'):
            assert path.read_bytes()==(a.primary/'evaluation/episodes'/path.name).read_bytes();reuse+=1
        rows.append({k:item[k] for k in ['suite','task','initial','condition','robot_success','queries','worker','elapsed_seconds']})
    assert reuse==800
    frame=pd.DataFrame(rows);assert set(frame.groupby(['condition','suite']).size().values)=={100}
    integrity=[]
    for seed in [10,11,12]:
        base=json.loads((a.primary/'arms'/('s%d_mr'%seed)/'completed.json').read_text())
        arms=[json.loads((a.root/'arms'/('s%d_%s'%(seed,m))/'completed.json').read_text()) for m in ['uv_ind','main','wrist']]
        assert all(v['completed'] and v['iterations']==5000 and v['frame_visits']==40000 for v in arms)
        for v in arms:
            for field in ['theta_init_sha256','sampling_log_sha256','gpu_name','pytorch']:
                assert v[field]==base[field],(seed,v['mode'],field)
            assert v['stream']==base['stream'] and v['camera_control_validation']['passed']
            if v['mode']!='uv_ind':assert v['camera_control_validation']['untouched_camera_processed_exact']
            else:assert v['camera_control_validation']['wrist_uv_independent_draw']
        composition=json.loads((a.root/'compositions'/('s%d.json'%seed)).read_text());assert composition['completed']
        if composition['rms_attainable']:assert composition['rms_error']<.025
        integrity.append(dict(seed=seed,gpu_family=base['gpu_name'],paired_initialization_and_main_stream_exact=True,
            amplitude_match_attainable=composition['rms_attainable'],amplitude_error=composition['rms_error']))
    paired=[]
    for seed in [10,11,12]:
        for suite in ['spatial','object']:
            for variant in VARIANTS:
                plain=shared=other=b=c=0
                for task in range(10):
                    for initial in range(10,20):
                        pp=raw[(suite,task,initial,'plain')];ss=raw[(suite,task,initial,'s%d_mr'%seed)]
                        oo=raw[(suite,task,initial,'s%d_%s'%(seed,variant))]
                        for comparison in [ss,oo]:
                            pt,ct=pp['trajectory'],comparison['trajectory']
                            if pt and ct:assert pt[0]['proprio_sha256']==ct[0]['proprio_sha256']
                            else:assert not pt and not ct and pp['robot_success'] and comparison['robot_success']
                            assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(pt,ct))
                        ps,ss,os=int(pp['robot_success']),int(ss['robot_success']),int(oo['robot_success'])
                        plain+=ps;shared+=ss;other+=os;b+=os*(1-ss);c+=(1-os)*ss
                paired.append(dict(seed=seed,suite=suite,variant=variant,n=100,Plain=plain,Shared_success=shared,
                    Other_success=other,Shared_G=plain-shared,Other_G=plain-other,shared_advantage=other-shared,
                    shared_stronger=b,other_stronger=c,p=mcnemar(b,c)))
    table=pd.DataFrame(paired);table['p_holm']=np.nan
    for family in [table.variant!='consensus_raw',table.variant=='consensus_raw']:
        table.loc[family,'p_holm']=holm(table.loc[family,'p'].values)
    means=table.groupby(['seed','variant'])[['Shared_G','Other_G','shared_advantage']].mean().reset_index()
    methods=[]
    for seed in [10,11,12]:
        for condition in ['mr']+VARIANTS:
            for suite in ['spatial','object']:
                s=frame[(frame.condition=='s%d_%s'%(seed,condition))&(frame.suite==suite)].robot_success.astype(int).sum()
                p=frame[(frame.condition=='plain')&(frame.suite==suite)].robot_success.astype(int).sum()
                methods.append(dict(seed=seed,method=condition,suite=suite,success=int(s),Plain=int(p),G=int(p-s)))
    methods=pd.DataFrame(methods)
    result=dict(completed=True,analysis_episodes=2600,new_episodes=1800,reused_episodes=800,training_seeds=[10,11,12],
        primary={v:interval(means[means.variant==v].shared_advantage.values) for v in ['uv_ind','consensus_rms']},
        secondary={'consensus_raw':interval(means[means.variant=='consensus_raw'].shared_advantage.values)},
        integrity=integrity,
        source_budget='Shared and UV-ind each use 40000 source calls per seed; independently trained main/wrist textures together use 80000 and twice the parameters.',
        scope='All variants are frozen single-cloth native deployments. UV synchronization and separate-view optimization followed by specified UV-space consensus are different comparisons. Three training pairs imply exact sign-flip p>=0.25; episode tests are conditional on the frozen textures.')
    frame.to_csv(str(a.out/'episode_outcomes.csv'),index=False)
    table.to_csv(str(a.out/'paired_suite_results.csv'),index=False)
    means.to_csv(str(a.out/'seed_means.csv'),index=False)
    methods.to_csv(str(a.out/'method_suite_results.csv'),index=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
