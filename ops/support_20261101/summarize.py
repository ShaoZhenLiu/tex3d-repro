"""Predeclared source gates, non-additivity and held-out ranking diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from support_core import pooled_metrics


def load(p): return json.loads(p.read_text())


def finite_correlation(a,b):
    if np.std(a)==0 or np.std(b)==0: return None
    return float(spearmanr(a,b)[0])


def main(root,protocol):
    val=load(root/'validation_candidates.json')
    fit=load(root/'fit_candidates.json')
    candidates=load(root/'frozen_candidates.json')['candidates']
    manifest=load(root/'texture_manifest.json')
    train_tasks=protocol['fit']['tasks']
    report=dict(scope='native source action fidelity only; no target/closed-loop attack outcomes',
                counts=dict(fit_states=len(load(root/'fit_states.json')),validation_states=len(load(root/'validation_states.json'))),
                summaries={},gates=[])
    for label,rows in [('fit',fit),('validation',val),
                       ('validation_seen_tasks',[r for r in val if r['task'] in train_tasks]),
                       ('validation_new_tasks',[r for r in val if r['task'] not in train_tasks])]:
        report['summaries'][label]={c['name']:pooled_metrics([r for r in rows if r['candidate']==c['name']]) for c in candidates}
    summary=report['summaries']['validation']
    for k in protocol['kept_blocks']:
        g=summary['greedy_%02d'%k]
        random=[summary['random%d_%02d'%(s,k)]['pooled_relative_fidelity'] for s in protocol['random_seeds']]
        random_median=float(np.median(random))
        vis=summary['visibility_%02d'%k]['pooled_relative_fidelity']
        ratio_random=(g['pooled_relative_fidelity']/random_median if random_median>1e-14 else 1.0)
        ratio_visibility=(g['pooled_relative_fidelity']/vis if vis>1e-14 else
                          (1.0 if g['pooled_relative_fidelity']<1e-14 else None))
        flags=dict(relative_fidelity=g['pooled_relative_fidelity']<=.10,
                   state_fraction=g['fraction_fidelity_le_0_25']>=.75,
                   action_energy=g['pooled_energy_retention']>=.80,
                   cosine=g['mean_displacement_cosine'] is not None and g['mean_displacement_cosine']>=.90,
                   random_advantage=ratio_random<=.80,
                   visibility_comparison=ratio_visibility is not None and ratio_visibility<=1.05,
                   half_or_less_surface=k<=8)
        report['gates'].append(dict(kept=k,area_fraction=k/16.,metrics=g,random_relative_fidelities=random,
            random_median=random_median,relative_fidelity_vs_random=ratio_random,
            relative_fidelity_vs_visibility=ratio_visibility,flags=flags,pass_gate=all(flags.values())))
    loo_fit=load(root/'leave_one_out_fit.json')
    fit_scores=[next(x['loss'] for x in loo_fit if x['block']==j) for j in range(16)]
    val_scores=[sum(r['pose_error_sum'] for r in val if r['candidate']=='loo_%02d'%j)/report['counts']['validation_states'] for j in range(16)]
    meta=load(root/'fit_states.json')
    visibility=np.array([r['visibility'] for r in meta]).mean(axis=(0,2)).tolist()
    report['ranking']=dict(fit_scores=fit_scores,validation_scores=val_scores,
        fit_validation_spearman=finite_correlation(fit_scores,val_scores),
        fit_action_vs_visibility_spearman=finite_correlation(fit_scores,visibility),
        visibility=visibility,per_state=[])
    for state in load(root/'validation_states.json'):
        scores=[next(r['pose_error_sum'] for r in val if r['state']==state['key'] and r['candidate']=='loo_%02d'%j) for j in range(16)]
        report['ranking']['per_state'].append(dict(state=state['key'],spearman=finite_correlation(fit_scores,scores)))
    report['source_gate_passed']=any(x['pass_gate'] for x in report['gates'])
    report['texture_statistics']=manifest
    (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(gates=report['gates'],ranking={k:v for k,v in report['ranking'].items() if k not in ['per_state','fit_scores','validation_scores']},source_gate_passed=report['source_gate_passed']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args()
    main(a.root,load(a.protocol))
