"""Build episode-level Object statistics, integrity checks and paper-ready plots."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


TARGETS = ['pi0', 'pi05']
VIEWS = {'native': ['main', 'wrist'], 'model_input': ['base_0_rgb', 'left_wrist_0_rgb']}
METRICS = ['marker_fraction', 'plain_hidden_fraction', 'ref_plain_fraction',
           'hidden_missed_marker_fraction', 'ref_change_within_marker_fraction',
           'hidden_outside_marker_fraction', 'marker_hidden_iou']


def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csvwrite(rows, path):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def describe(a):
    a=np.asarray([x for x in a if x is not None], dtype=float)
    if not len(a):return dict(n=0,median=None,p25=None,p75=None,min=None,max=None,mean=None)
    return dict(n=len(a), median=float(np.median(a)), p25=float(np.percentile(a,25)),
                p75=float(np.percentile(a,75)), min=float(a.min()), max=float(a.max()), mean=float(a.mean()))


def mcnemar(b,c):
    n=b+c
    if n==0:return 1.
    k=min(b,c)
    # Exact two-sided conditional binomial test, safe for n<=30.
    return min(1.,2*sum(math.factorial(n)//(math.factorial(j)*math.factorial(n-j)) for j in range(k+1))/2**n)


def build(args):
    root=args.root;raw=root/'raw'
    if not raw.exists():raw=root
    (root/'figures').mkdir(exist_ok=True)
    episodes=[];registrations=[]
    for name in ['pi0_main','pi05_main','pi0_tail','pi05_tail']:
        r=raw/name
        if not (r/'registered.json').exists():continue
        registered=json.loads((r/'registered.json').read_text())
        registrations.append(registered)
        if not args.partial:assert (r/'results.json').exists(),name
        for f in sorted(r.glob('pi*_t*_i*_*.json')):
            x=json.loads(f.read_text());assert x['completed'];x['_folder']=str(r);episodes.append(x)
    casekeys={(e['target'],e['task'],e['initial'],e['condition']) for e in episodes}
    assert len(casekeys)==len(episodes)
    expected={(target,t,i,c) for target in TARGETS for t in range(10) for i in [10,11,12] for c in ['plain','refcloth']}
    assert casekeys.issubset(expected)
    if not args.partial:assert casekeys==expected,(len(casekeys),sorted(expected-casekeys))
    episode_rows=[];query_rows=[];same_state_rows=[];camera_delays=[];audit=dict(episodes=len(episodes),expected_episodes=120,plain_queries=0,
        aa_action_maxdiff=0.,cached_observation_maxdiff=0,aba_failures=0,integration_state_failures=0,
        hidden_marker_maxdiff=0,consumed_noise_pairs=0,official_resize_maxdiff=0.,smoke_checks={})
    pairs={}
    for e in episodes:
        assert e['queries']==len(e['trajectory']) and e['queries']>0
        first=e['trajectory'][0]
        audit['aa_action_maxdiff']=max(audit['aa_action_maxdiff'],first['aa_action_maxdiff'])
        pairs.setdefault((e['target'],e['task'],e['initial']),{})[e['condition']]=e
        common=dict(target=e['target'],task=e['task'],initial=e['initial'],condition=e['condition'])
        er=dict(common,robot_success=int(e['robot_success']),queries=e['queries'],elapsed_seconds=e['elapsed_seconds'])
        if e['condition']=='plain':
            assert e['probe']
            for q in e['trajectory']:
                m=q['measurement'];audit['plain_queries']+=1
                assert m['camera_views_same_state']
                camera_delays.append(m['camera_delay_seconds'])
                audit['cached_observation_maxdiff']=max(audit['cached_observation_maxdiff'],m['cached_observation_maxdiff'])
                audit['aba_failures']+=int(not m['aba_exact'])
                audit['integration_state_failures']+=int(not m['integration_state_exact'])
                audit['hidden_marker_maxdiff']=max(audit['hidden_marker_maxdiff'],m['hidden_marker_maxdiff'] or 0)
                assert set(m['model_specs'])=={'base_0_rgb','left_wrist_0_rgb'}
                for stage,views in VIEWS.items():
                    for view in views:
                        z=m[stage][view]
                        for threshold,values in z['thresholds'].items():
                            query_rows.append(dict(common,query=q['query'],action_step=q['query']*5,
                                stage=stage,view=view,threshold=int(threshold),**values,
                                marker_rms_rgb255=z['rms_rgb255']['marker'],
                                plain_hidden_rms_rgb255=z['rms_rgb255']['plain_hidden'],
                                ref_plain_rms_rgb255=z['rms_rgb255']['ref_plain']))
            for stage,views in VIEWS.items():
                for view in views:
                    z=[q['measurement'][stage][view] for q in e['trajectory']]
                    for threshold in ['1','5','10','20']:
                        for metric in METRICS:
                            vals=[x['thresholds'][threshold][metric] for x in z]
                            vals=[v for v in vals if v is not None]
                            er['%s__%s__t%s__%s'%(stage,view,threshold,metric)]=float(np.median(vals)) if vals else None
                        for metric in METRICS[:3]:
                            er['%s__%s__t%s__%s__query_fraction_gt_1pct'%(stage,view,threshold,metric)]=float(np.mean([x['thresholds'][threshold][metric]>.01 for x in z]))
                    for metric in ['marker','plain_hidden','ref_plain']:
                        er['%s__%s__%s_rms_rgb255'%(stage,view,metric)]=float(np.median([x['rms_rgb255'][metric] for x in z]))
        episode_rows.append(er)
    # Different row schemas are stored in separate tables, avoiding empty columns on Ref rows.
    flat=[{k:r[k] for k in ['target','task','initial','condition','robot_success','queries','elapsed_seconds']} for r in episode_rows]
    if flat:csvwrite(flat,root/'episode_outcomes.csv')
    plain=[r for r in episode_rows if r['condition']=='plain']
    if plain:csvwrite(plain,root/'plain_episode_measurements.csv')
    if query_rows:csvwrite(query_rows,root/'plain_query_measurements.csv')
    summary=dict(completed=casekeys==expected,counts=audit,attack={},exposure={},placement={},limitations=
        '30 paired episodes/target; only three initial states/task; simulator evidence; frozen MR texture; '
        'paired query noise differs from archived main-table randomness; no Original-condition rerun; '
        'marker contrast is a visibility proxy; no target gradients, training or high-sensitivity claim')
    for target in TARGETS:
        pp=[v for (tg,_,_),v in pairs.items() if tg==target and len(v)==2]
        if not pp:continue
        for v in pp:
            overlap=min(x['queries'] for x in v.values())
            a=[q['noise_sha256'] for q in v['plain']['trajectory'][:overlap]]
            b=[q['noise_sha256'] for q in v['refcloth']['trajectory'][:overlap]]
            assert a==b;audit['consumed_noise_pairs']+=1
            p0=v['plain']['trajectory'][0];r0=v['refcloth']['trajectory'][0]
            assert p0['simulator_state_sha256']==r0['simulator_state_sha256'], 'Paired initial physics differ'
            assert p0['proprio_sha256']==r0['proprio_sha256']
            key='%s_t%d_i%d_plain'%(target,v['plain']['task'],v['plain']['initial'])
            folder=Path(v['plain']['_folder'])/'images'/key
            for label,field in [('main','input_main_sha256'),('wrist','input_wrist_sha256')]:
                im=np.array(Image.open(str(folder/('refcloth_native_'+label+'.png'))))
                assert hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()==r0[field], 'Counterfactual does not match real Ref input at same state'
            pa=np.asarray(p0['actions']);ra=np.asarray(r0['actions']);d=ra-pa
            same_state_rows.append(dict(target=target,task=v['plain']['task'],initial=v['plain']['initial'],
                same_integration_state=True,same_proprio=True,same_noise=True,counterfactual_matches_actual_ref_images=True,
                first_five_action_rms=float(np.sqrt(np.mean(d*d))),
                first_five_xyz_action_rms=float(np.sqrt(np.mean(d[:,:3]*d[:,:3]))),
                first_five_rotation_action_rms=float(np.sqrt(np.mean(d[:,3:6]*d[:,3:6]))),
                first_five_gripper_sign_disagreements=int(np.sum(np.sign(pa[:,6])!=np.sign(ra[:,6])))))
        ps=sum(v['plain']['robot_success'] for v in pp)
        rs=sum(v['refcloth']['robot_success'] for v in pp)
        b=sum(v['plain']['robot_success'] and not v['refcloth']['robot_success'] for v in pp)
        c=sum(not v['plain']['robot_success'] and v['refcloth']['robot_success'] for v in pp)
        # Resample whole task clusters, retaining their three paired initial states.
        task_effects=[]
        for t in range(10):
            tt=[v for (tg,task,_),v in pairs.items() if tg==target and task==t and len(v)==2]
            if len(tt)==3:task_effects.append(np.mean([int(v['plain']['robot_success'])-int(v['refcloth']['robot_success']) for v in tt]))
        ci=None
        if len(task_effects)==10:
            rng=np.random.RandomState(20261102)
            boot=np.asarray(task_effects)[rng.randint(0,10,(50000,10))].mean(axis=1)
            ci=np.percentile(boot,[2.5,97.5]).tolist()
        summary['attack'][target]=dict(n=len(pp),plain_success=ps,ref_success=rs,
            plain_success_rate=ps/len(pp),ref_success_rate=rs/len(pp),net_attack_gain=(ps-rs)/len(pp),
            induced_failures=b,recovered_failures=c,exact_mcnemar_p=mcnemar(b,c),
            task_cluster_bootstrap_gain_ci95=ci)
        selected=[r for r in plain if r['target']==target]
        if not selected:continue
        summary['exposure'][target]={}
        for stage,views in VIEWS.items():
            summary['exposure'][target][stage]={}
            for view in views:
                prefix=stage+'__'+view+'__'
                summary['exposure'][target][stage][view]={k[len(prefix):]:describe([r[k] for r in selected])
                                                       for k in selected[0] if k.startswith(prefix)}
    # Holm correction over the two predeclared target comparisons.
    order=sorted(summary['attack'],key=lambda t:summary['attack'][t]['exact_mcnemar_p'])
    last=0.
    for j,target in enumerate(order):
        last=max(last,min(1.,summary['attack'][target]['exact_mcnemar_p']*(len(order)-j)))
        summary['attack'][target]['mcnemar_p_holm']=last
    placement=raw/'placement/placement.json'
    if placement.exists():
        a=json.loads(placement.read_text());assert a['completed'] and len(a['records'])==30
        summary['placement']['n']=30
        summary['placement']['old_ray_platform_count']=sum(x['placements']['legacy_ray']['body_pos'][2]>.1 for x in a['records'])
        summary['placement']['old_static_floor_count']=30-summary['placement']['old_ray_platform_count']
        summary['placement']['production_floor_count']=sum(abs(x['placements']['production_floor']['body_pos'][2]-.005)<1e-10 for x in a['records'])
        for stage in ['native','model_resize']:
            summary['placement'][stage]={}
            for view in ['main','wrist']:
                summary['placement'][stage][view]={}
                for label in ['legacy_ray','production_floor']:
                    summary['placement'][stage][view][label]={metric:describe([x['placements'][label][stage][view]['thresholds']['10'][metric] for x in a['records']])
                                                              for metric in METRICS[:4]}
    for target in TARGETS:
        f=raw/('smoke_'+target)/'smoke_passed.json'
        if f.exists():audit['smoke_checks'][target]=json.loads(f.read_text())
    old_baseline=raw/'diagnostics/failed_v1/smoke_pi05/smoke_uninstrumented.json'
    hook_baseline=raw/'smoke_pi05/smoke_uninstrumented.json'
    if old_baseline.exists() and hook_baseline.exists():
        a=json.loads(old_baseline.read_text());b=json.loads(hook_baseline.read_text())
        fields=['noise_sha256','input_main_sha256','input_wrist_sha256','proprio_sha256',
                'simulator_state_sha256','action_chunk_sha256','actions']
        assert a['queries']==b['queries'] and a['robot_success']==b['robot_success']
        assert all(all(x[k]==y[k] for k in fields) for x,y in zip(a['trajectory'],b['trajectory']))
        audit['camera_copy_hook_vs_original_production_exact']=True
    audit['same_state_real_counterfactual_pairs_checked']=len(same_state_rows)
    if same_state_rows:csvwrite(same_state_rows,root/'same_state_first_query_checks.csv')
    if camera_delays:audit['camera_delay_seconds']=describe(camera_delays)
    if not args.partial:
        assert len(audit['smoke_checks'])==2
        assert audit['aa_action_maxdiff']<=1e-6 and audit['cached_observation_maxdiff']==0
        assert audit['aba_failures']==audit['integration_state_failures']==audit['hidden_marker_maxdiff']==0
        assert audit['consumed_noise_pairs']==60
        assert summary['placement']['production_floor_count']==30
        assert audit['camera_copy_hook_vs_original_production_exact']
        assert len(same_state_rows)==60
    dump(summary,root/('summary.partial.json' if args.partial else 'summary.json'))
    print(json.dumps(dict(completed=summary['completed'],counts=audit,attack=summary['attack']),indent=2))
    if not args.partial:figures(root,summary,plain,query_rows)


def figures(root,s,plain,query_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#526d82','#cd8b38','#b63c50']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=False)
    for ax,view,title in zip(axes,VIEWS['model_input'],['Main camera (224 x 224)','Wrist camera (224 x 224)']):
        for j,metric in enumerate(METRICS[:3]):
            stats=[s['exposure'][t]['model_input'][view]['t10__'+metric] for t in TARGETS]
            x=np.arange(2)+(j-1)*.23
            med=[z['median'] for z in stats]
            err=np.array([[z['median']-z['p25'] for z in stats],[z['p75']-z['median'] for z in stats]])
            ax.bar(x,med,width=.21,color=colors[j],label=['Marker visibility','Plain - hidden','RefCloth - Plain'][j])
            ax.errorbar(x,med,yerr=err,fmt='none',ecolor='black',capsize=3,lw=1)
        ax.set_xticks(range(2));ax.set_xticklabels([r'$\pi_0$',r'$\pi_{0.5}$']);ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Changed pixel fraction')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,frameon=False,loc='upper left')
    fig.suptitle('Correct Object placement: 30 Plain trajectories per target\nEpisode medians, then median / IQR; RGB threshold > 10')
    fig.tight_layout();fig.savefig(str(root/'figures/model_input_three_measures.png'),dpi=180)
    fig.savefig(str(root/'figures/model_input_three_measures.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,target in zip(axes,TARGETS):
        a=s['attack'][target];vals=[a['plain_success_rate'],a['ref_success_rate']]
        ax.bar([0,1],vals,color=['#526d82','#b63c50'],width=.58)
        for x,v,n in zip([0,1],vals,[a['plain_success'],a['ref_success']]):ax.text(x,v+.025,'%d / %d'%(n,a['n']),ha='center')
        ax.set_xticks([0,1]);ax.set_xticklabels(['Plain','RefCloth']);ax.set_ylim(0,1.17)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Robot success rate')
        ax.set_title('%s: net attack gain %.1f pp'%(target,100*a['net_attack_gain']))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen original texture; same initial states and consumed noise')
    fig.tight_layout();fig.savefig(str(root/'figures/paired_attack_results.png'),dpi=180)
    fig.savefig(str(root/'figures/paired_attack_results.pdf'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for i,target in enumerate(TARGETS):
        for j,view in enumerate(VIEWS['model_input']):
            ax=axes[i,j]
            subset=[r for r in query_rows if r['target']==target and r['stage']=='model_input' and r['view']==view and r['threshold']==10]
            for k,metric in enumerate(METRICS[:3]):
                xs=sorted({r['action_step'] for r in subset})
                vals=[[r[metric] for r in subset if r['action_step']==x] for x in xs]
                med=[np.median(v) for v in vals]
                ax.plot(xs,med,color=colors[k],label=['Marker','Plain - hidden','RefCloth - Plain'][k])
            ax.set_title(target+' / '+['main','wrist'][j]);ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            if i==1:ax.set_xlabel('Action step (5 steps per policy query)')
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Per-query medians among still-running Plain episodes\nLater steps include fewer episodes; used for dynamics, not headline aggregation')
    fig.tight_layout();fig.savefig(str(root/'figures/trajectory_support_curves.png'),dpi=180);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--partial',action='store_true')
    build(ap.parse_args())
