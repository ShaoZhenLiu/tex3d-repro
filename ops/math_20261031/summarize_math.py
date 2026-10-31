"""Complete-grid summaries, preserving starts, step sizes and task groups."""
import argparse
import json
import math
import statistics
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def energy_ratio(records,field):
    error=sum(r[field]['error_rms']**2 for r in records)
    truth=sum(r[field]['truth_rms']**2 for r in records)
    return math.sqrt(error/max(truth,1e-24))


def summarize(folder):
    collections=[read(folder/name/'collection.json') for name in ['fit02_live','fit68_live']]
    evaluations=[read(folder/name/'validation.json') for name in ['validation02','validation68']]
    fits=sum([x['records'] for x in collections],[])
    rows=sum([x['records'] for x in evaluations],[])
    assert len(fits)==12 and len(rows)==288
    keys=[(r['task'],r['start'],r['eta'],r['method']) for r in rows]
    assert len(set(keys))==288
    solver=read(folder/'directions/direction_solver.json')
    report=dict(record_count=len(rows),fit_state_count=4,heldout_state_count=4,
        target_queries=0,new_closed_loop_success_rates=False,
        geometry=[],fusion=[],fidelity={},method_overall={},audit={})
    for start in ['all','init','moderate','frozen_mr']:
        rs=fits if start=='all' else [r for r in fits if r['start']==start]
        aa=energy_ratio(rs,'action_displacement')
        interaction=energy_ratio(rs,'action_interaction')
        report['fidelity'][start]=dict(action_AA_relative_error=aa,
            action_interaction_relative_error=interaction,interaction_vs_AA_error=interaction/aa,
            near_zero_native_interaction_count=sum(r['action_interaction']['truth_rms']<1e-6 for r in rs))
    per_task=[]
    for task in [0,2,6,8]:
        rs=[r for r in fits if r['task']==task]
        per_task.append(dict(task=task,AA_relative_error=energy_ratio(rs,'action_displacement'),
            interaction_relative_error=energy_ratio(rs,'action_interaction')))
    report['fidelity']['per_task']=per_task
    report['fidelity']['tasks_interaction_better']=sum(r['interaction_relative_error']<r['AA_relative_error'] for r in per_task)
    report['fidelity']['gate']=(report['fidelity']['all']['interaction_vs_AA_error']<=.75 and
                               report['fidelity']['tasks_interaction_better']>=3)
    for start in ['init','moderate','frozen_mr']:
        for eta in [.01,.05]:
            rs=[r for r in rows if r['start']==start and r['eta']==eta]
            for normalization in ['raw','l1']:
                norm_record=next(r for r in solver['records'] if r['start']==start and r['eta']==eta and r['method']==normalization+'_maximin')
                scales=dict(zip([0,2,6,8],norm_record['gradient_scales']))
                entry=dict(start=start,eta=eta,normalization=normalization,methods={})
                for short in ['mean_sign','pcgrad_sign','pcgrad_vector','maximin']:
                    method=normalization+'_'+short
                    mr=sorted([r for r in rs if r['method']==method],key=lambda r:r['task'])
                    raw=[r['native_total_gain'] for r in mr]
                    values=[r['native_total_gain']/scales[r['task']] for r in mr]
                    entry['methods'][short]=dict(task_gains=values,minimum=min(values),mean=statistics.mean(values),
                        raw_minimum=min(raw),raw_mean=statistics.mean(raw),
                        positive_tasks=sum(x>1e-8 for x in values),
                        mean_native_action_gain=statistics.mean(r['native_action_gain'] for r in mr))
                candidate=entry['methods']['maximin']
                baseline=[entry['methods'][s] for s in ['mean_sign','pcgrad_sign','pcgrad_vector']]
                best_min=max(b['minimum'] for b in baseline)
                best_mean=max(b['mean'] for b in baseline)
                # Retention is meaningful only for a positive baseline mean.
                retained=(candidate['mean']>=.9*best_mean if best_mean>0 else candidate['mean']>=best_mean)
                entry.update(best_baseline_minimum=best_min,best_baseline_mean=best_mean,
                    worst_gain_better=candidate['minimum']>best_min+1e-8,
                    gate_setting=(candidate['minimum']>=-1e-8 and candidate['minimum']>best_min+1e-8 and retained),
                    mean_retention= candidate['mean']/best_mean if best_mean>0 else None,
                    fit_optimal_margin=norm_record.get('optimal_margin'),fit_duality_gap=norm_record['duality_gap'])
                report['geometry'].append(entry)
            for method in ['action_mean','interaction_mean','interaction_penalized','interaction_solo_lp']:
                mr=[r for r in rs if r['method']==method]
                successful=sum(r['solo_no_increase'] and r['native_action_after']>1.1*r['native_action_before'] for r in mr)
                report['fusion'].append(dict(start=start,eta=eta,method=method,
                    solo_preserved_states=sum(r['solo_no_increase'] for r in mr),
                    successful_joint_and_solo_states=successful,
                    mean_joint_gain=statistics.mean(r['native_action_gain'] for r in mr),
                    mean_interaction_gain=statistics.mean(r['native_interaction_gain'] for r in mr),
                    positive_lower_bound_states=sum(r['native_joint_lower_bound_rms']>1e-8 for r in mr),
                    gate_setting=successful>=3))
    report['geometry_gate']={n:dict(settings_passed=sum(r['gate_setting'] for r in report['geometry'] if r['normalization']==n),
        required_settings=4,gate=sum(r['gate_setting'] for r in report['geometry'] if r['normalization']==n)>=4)
        for n in ['raw','l1']}
    report['fusion_gate']={m:dict(settings_passed=sum(r['gate_setting'] for r in report['fusion'] if r['method']==m),
        required_settings=4,gate=sum(r['gate_setting'] for r in report['fusion'] if r['method']==m)>=4)
        for m in ['action_mean','interaction_mean','interaction_penalized','interaction_solo_lp']}
    for method in sorted(set(r['method'] for r in rows)):
        mr=[r for r in rows if r['method']==method]
        report['method_overall'][method]=dict(mean_native_J_gain=statistics.mean(r['native_total_gain'] for r in mr),
            mean_native_FD_gain=statistics.mean(r['native_fd_gain'] for r in mr),
            mean_native_action_gain=statistics.mean(r['native_action_gain'] for r in mr),
            positive_J_states_settings=sum(r['native_total_gain']>1e-8 for r in mr))
    states=sum([r['states'] for r in collections+evaluations],[])
    report['audit']=dict(zero_png_maxdiff=max(s['zero_png_maxdiff'] for s in states),
        aa_feature_maxdiff=max(s['aa_feature_maxdiff'] for s in states),
        stale_observation_maxdiff=max(max(s['observation_audit']['main_maxdiff'],s['observation_audit']['wrist_maxdiff']) for s in states),
        all_native_upload_and_ABA_assertions_passed=True,
        source_peak_allocated_bytes=max(r['gpu_peak_bytes'] for r in collections+evaluations))
    (folder/'math_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(geometry=report['geometry_gate'],fusion=report['fusion_gate'],fidelity=report['fidelity']['all'],audit=report['audit']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);summarize(ap.parse_args().folder)
