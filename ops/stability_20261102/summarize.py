"""Analyze the full predeclared 2600-episode block; never select training seeds."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def interval(values):
    v=np.asarray(values,dtype=float);n=len(v);mean=float(v.mean())
    sd=float(v.std(ddof=1)) if n>1 else 0.
    half=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,sd=sd,ci95=[mean-half,mean+half] if half is not None else None,
                median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max()))


def sign_flip(values):
    v=np.asarray(values,dtype=float);observed=abs(v.mean())
    null=[abs(np.mean(v*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=len(v))]
    return float(np.mean(np.asarray(null)>=observed-1e-12))


def mcnemar(b,c):
    n=int(b+c)
    if not n:return 1.
    mass=sum(math.factorial(n)//(math.factorial(k)*math.factorial(n-k)) for k in range(min(b,c)+1))
    return min(1.,2.*mass/2**n)


def holm(values):
    values=np.asarray(values,dtype=float);order=np.argsort(values);out=np.zeros(len(values));last=0.
    for rank,idx in enumerate(order):
        last=max(last,min(1.,values[idx]*(len(values)-rank)));out[idx]=last
    return out


def main(a):
    followup=json.loads(a.followup_protocol.read_text()) if a.followup_protocol else None
    seeds=followup['new_training_seeds'] if followup else [7,8,9,10,11,12]
    new_seeds=seeds if followup else [10,11,12]
    episodes=200*(1+2*len(seeds))
    assert not followup or episodes==followup['evaluation']['planned_episodes']
    paths=sorted((a.root/'evaluation/episodes').glob('*.json'))
    if a.progress:
        print(json.dumps(dict(completed_episodes=len(paths),planned_episodes=episodes)))
        return
    assert len(paths)==episodes,('Full analysis waits for every planned case',len(paths))
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.root/'texture_manifest.json').read_text())
    assert len(manifest)==1+2*len(seeds)
    assert len({v['sha256'] for v in manifest.values()})==len(manifest)
    for item in manifest.values():
        assert hashlib.sha256((a.root/item['path']).read_bytes()).hexdigest()==item['sha256']
    rows=[];raw={}
    for path in paths:
        obj=json.loads(path.read_text());assert obj['completed']
        suite,task,initial,condition=[obj[k] for k in ['suite','task','initial','condition']]
        assert suite in ['spatial','object'] and task in range(10) and initial in range(10,20)
        assert obj['texture_sha256']==manifest[condition]['sha256']
        assert obj['queries']==len(obj['trajectory']) and obj['native_uploads']>=1
        assert [z['query'] for z in obj['trajectory']]==list(range(obj['queries']))
        assert all(np.asarray(z['actions']).shape==(5,7) and np.isfinite(z['actions']).all() for z in obj['trajectory'])
        key=(suite,task,initial,condition);assert key not in raw
        raw[key]=obj
        rows.append(dict(suite=suite,task=task,initial=initial,condition=condition,
            success=int(obj['robot_success']),queries=obj['queries'],elapsed_seconds=obj['elapsed_seconds'],worker=obj['worker'],host=obj['host'],gpu=obj['gpu']))
    frame=pd.DataFrame(rows)
    assert set(frame.groupby('condition').size().values)=={200}
    assert set(frame.groupby(['condition','suite']).size().values)=={100}
    workers={}
    for name in sorted(frame.worker.unique()):
        calibration=json.loads((a.root/'evaluation/workers'/name/'calibration.json').read_text())
        assert calibration['passed']
        if calibration.get('mode') == 'success':
            assert followup and a.followup_protocol
            evaluation_plan=json.loads(a.followup_protocol.with_name('evaluation_plan.json').read_text())
            assert evaluation_plan.get('user_authorized')
            assert calibration.get('node') in evaluation_plan['allow_stochastic_calibration_nodes']
            assert calibration.get('mode')=='success' and calibration.get('success_repeatable')
            assert calibration.get('canonical_success')
        else:
            assert calibration['trajectory_exact']
        workers[name]=calibration
    input_changes=[]
    for condition in sorted(set(frame.condition)-{'plain'}):
        main_changed=wrist_changed=comparable=initial_proprio_changed=0
        for suite in ['spatial','object']:
            for task in range(10):
                for initial in range(10,20):
                    p=raw[(suite,task,initial,'plain')]['trajectory']
                    t=raw[(suite,task,initial,condition)]['trajectory']
                    if p and t:
                        if p[0]['proprio_sha256']!=t[0]['proprio_sha256']:
                            calibration=workers[raw[(suite,task,initial,condition)]['worker']]
                            assert calibration.get('mode')=='success' and calibration.get('node')=='09'
                            initial_proprio_changed+=1
                        comparable+=1
                        main_changed+=p[0]['main_sha256']!=t[0]['main_sha256']
                        wrist_changed+=p[0]['wrist_sha256']!=t[0]['wrist_sha256']
                    else:assert not p and not t
                    assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(p,t))
        assert main_changed+wrist_changed>0
        input_changes.append(dict(condition=condition,comparable_initial_queries=comparable,
            main_changed=main_changed,wrist_changed=wrist_changed,initial_proprio_changed=initial_proprio_changed))
    paired=[]
    evaluation_hosts = {}
    for seed in seeds:
        hosts = frame[frame.condition.isin(['s%d_native'%seed, 's%d_mr'%seed])].host.unique()
        assert len(hosts)==1, (seed,hosts)
        evaluation_hosts[str(seed)] = str(hosts[0])
        for suite in ['spatial','object']:
            plain=frame[(frame.condition=='plain')&(frame.suite==suite)].success.sum()
            n_success=m_success=b=c=0
            for task in range(10):
                for initial in range(10,20):
                    n=raw[(suite,task,initial,'s%d_native'%seed)]
                    m=raw[(suite,task,initial,'s%d_mr'%seed)]
                    assert n['host']==m['host'],(seed,suite,task,initial)
                    nt,mt=n['trajectory'],m['trajectory']
                    if nt and mt:
                        assert nt[0]['proprio_sha256']==mt[0]['proprio_sha256']
                    else:assert not nt and not mt and n['robot_success'] and m['robot_success']
                    assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(nt,mt))
                    ns,ms=int(n['robot_success']),int(m['robot_success'])
                    n_success+=ns;m_success+=ms;b+=ns*(1-ms);c+=(1-ns)*ms
            paired.append(dict(seed=seed,suite=suite,n=100,Plain=int(plain),Native=n_success,Matched=m_success,
                Native_G=int(plain)-n_success,Matched_G=int(plain)-m_success,
                delta_G=n_success-m_success,matched_stronger=b,native_stronger=c,p=mcnemar(b,c)))
    table=pd.DataFrame(paired);table['p_holm%d'%len(table)]=holm(table.p.values)
    seed_means=table.groupby('seed')[['Native_G','Matched_G','delta_G']].mean().reset_index()
    delta=seed_means.delta_G.values
    result=dict(completed=True,episodes=episodes,training_pairs=len(seeds),new_pairs=len(new_seeds),
        primary=dict(interval(delta),exact_two_sided_sign_flip_p=sign_flip(delta),
            positive_pairs=int((delta>0).sum()),negative_pairs=int((delta<0).sum()),zero_pairs=int((delta==0).sum())),
        native=interval(seed_means.Native_G.values),matched=interval(seed_means.Matched_G.values),
        leave_one_out_means={str(seed):float(seed_means[seed_means.seed!=seed].delta_G.mean()) for seed in seed_means.seed},
        suites={suite:interval(table[table.suite==suite].delta_G.values) for suite in ['spatial','object']},
        scope='Predeclared trained reference pairs evaluated under one common-noise protocol with fixed benchmark states.')
    checks=[]
    for seed in new_seeds:
        arm=[json.loads((a.root/'arms'/('s%d_%s'%(seed,ref))/'completed.json').read_text()) for ref in ['native','mr']]
        assert all(x['completed'] and x['iterations']==5000 and x['frame_visits']==40000 for x in arm)
        assert arm[0]['theta_init_sha256']==arm[1]['theta_init_sha256']
        assert arm[0]['sampling_log_sha256']==arm[1]['sampling_log_sha256']
        assert arm[0]['stream']==arm[1]['stream']
        checks.append(dict(seed=seed,initialization_exact=True,sampling_stream_exact=True,
            device_families=[x['gpu_name'] for x in arm],runner_hashes=[x['runner_sha256'] for x in arm]))
    result['new_pair_integrity']=checks
    if followup:
        assert all(len(set(check['device_families'])) == 1 for check in checks)
        result['seed_order']=seeds
    else:
        result['existing_only']=interval(seed_means[seed_means.seed<10].delta_G.values)
        result['new_only']=interval(seed_means[seed_means.seed>=10].delta_G.values)
        prior=json.loads((a.root/'existing_stream_reconstruction.json').read_text())
        assert set(prior)=={'7','8','9'}
        assert all(v['rows_per_arm']==40000 and v['textual_stream_equal'] for v in prior.values())
        assert all(prior[str(seed)]['full_precision_uv_equal'] and prior[str(seed)]['initialization_exact'] for seed in [8,9])
        result['existing_pair_integrity']=prior
    result['evaluation_hosts']=evaluation_hosts
    result['calibration_scope']='Canonical 4090 Plain baseline; exact-trajectory calibration on 4090 and user-authorized success calibration on 09.'
    result['worker_calibrations']=workers
    result['initial_input_changes']=input_changes
    frame.to_csv(str(a.out/'episode_outcomes.csv'),index=False)
    table.to_csv(str(a.out/'paired_suite_results.csv'),index=False)
    seed_means.to_csv(str(a.out/'seed_means.csv'),index=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--followup-protocol',type=Path)
    p.add_argument('--progress',action='store_true');main(p.parse_args())
