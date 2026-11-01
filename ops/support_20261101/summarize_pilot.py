"""Paired exploratory outcomes; robot failure is the attack outcome.

Hoeffding intervals assume independent paired episodes, allow differing
per-case means, and deliberately do not return zero width for all ties.
The fixed, small task sample still does not support broad generalization.
"""
import argparse
import json
import math
from pathlib import Path


def main(combined,out):
    data=json.loads(combined.read_text())
    p=data['protocol'];rows=data['records']
    expected={(s,t,i,c) for s in p['suites'] for t in p['tasks'] for i in p['initials'] for c in p['conditions']}
    actual={(r['suite'],r['task'],r['initial'],r['condition']) for r in rows}
    assert actual==expected and len(rows)==len(expected)==72
    assert all(r['native_upload_ok'] and r['aa_action_maxdiff']<=1e-6 for r in rows)
    report=dict(exploratory=True,episodes=len(rows),source_strong_compression_gate_passed=False,
                candidate='75% retained surface, frozen before target queries',suites={},
                limitations='Six paired cases per suite; three tasks; not a noninferiority or full-suite estimate')
    for suite in p['suites']+['pooled_fixed_cases']:
        selected=[r for r in rows if suite=='pooled_fixed_cases' or r['suite']==suite]
        bycondition={c:{(r['suite'],r['task'],r['initial']):r for r in selected if r['condition']==c} for c in p['conditions']}
        full=bycondition['full'];plain=bycondition['plain'];n=len(full)
        outcome={}
        for condition,cases in bycondition.items():
            assert set(cases)==set(full)
            failures=sum(not r['robot_success'] for r in cases.values())
            d=[int(not cases[k]['robot_success'])-int(not full[k]['robot_success']) for k in full]
            delta=sum(d)/n
            width=math.sqrt(2*math.log(40)/n)
            losses=sum(x==-1 for x in d);wins=sum(x==1 for x in d)
            outcome[condition]=dict(n=n,robot_successes=n-failures,attack_failures_of_robot=failures,
                attack_rate=failures/n,attack_gain_vs_plain=(failures-sum(not r['robot_success'] for r in plain.values()))/n,
                paired_attack_rate_difference_vs_full=delta,improved_pairs=wins,degraded_pairs=losses,
                tied_pairs=n-losses-wins,
                conservative_95_difference_interval=[max(-1.,delta-width),min(1.,delta+width)],
                zero_degradation_one_sided_95_upper=(1-.05**(1/n) if losses==0 else None))
        report['suites'][suite]=outcome
    report['max_repeat_action_difference']=max(r['aa_action_maxdiff'] for r in rows)
    report['mean_elapsed_seconds_per_episode']=sum(r['elapsed_seconds'] for r in rows)/len(rows)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('combined',type=Path);ap.add_argument('out',type=Path)
    a=ap.parse_args();main(a.combined,a.out)
