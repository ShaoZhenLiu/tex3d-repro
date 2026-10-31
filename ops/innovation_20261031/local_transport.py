"""Test state-conditioned secant transport, preserving the uncalibrated space.

Unlike a projection, this estimates a mapping from synthesized to native feature
displacements. Calibration uses source-native probes at the SAME state. Its
query cost is explicitly part of the proposed training procedure.
"""
import argparse
import json
from pathlib import Path

import torch
from diagnose import load_split, vector_metrics


def transport(query, probes_r, probes_n, ridge=0.01):
    shape = query.shape
    q, r, n = query.flatten(1).double(), probes_r.flatten(1).double(), probes_n.flatten(1).double()
    gram = r @ r.T
    regularizer = ridge * gram.diag().mean().clamp_min(1e-20)
    coefficients = torch.linalg.solve(gram + regularizer * torch.eye(len(r),dtype=torch.float64),
                                      r @ q.T).T
    return (q + coefficients @ (n-r)).float().reshape(shape)


def run(root, output):
    torch.set_num_threads(2)
    report=dict(locked_test_consumed=False, ridge=.01,
                scope='within-state source-native calibration, unseen textures; no target-policy access',splits={})
    for split in ['fit','validation']:
        rows, r, n = load_split(root, split)
        native_predictions, rendered_tests, truth, states=[],[],[],[]
        for state in sorted(set(row['state'] for row in rows)):
            indices=[i for i,row in enumerate(rows) if row['state']==state]
            # One of each paired seed per source texture stratum, frozen rule.
            ordered=sorted(indices,key=lambda i:rows[i]['texture_seed'])
            calibration=ordered[::2]
            tests=ordered[1::2]
            pred=transport(r[tests],r[calibration],n[calibration])
            metrics=vector_metrics(pred,n[tests],r[tests])
            metrics['baseline_cosine']=vector_metrics(r[tests],n[tests],r[tests])['mean_cosine']
            states.append(dict(state=state,task=rows[tests[0]]['task'],calibration_n=len(calibration),test_n=len(tests),**metrics))
            native_predictions.append(pred);rendered_tests.append(r[tests]);truth.append(n[tests])
        pred=torch.cat(native_predictions);baseline=torch.cat(rendered_tests);target=torch.cat(truth)
        report['splits'][split]=dict(metrics=vector_metrics(pred,target,baseline),
            baseline=vector_metrics(baseline,target,baseline),states=states,
            calibration_queries=sum(v['calibration_n'] for v in states),test_count=len(pred))
    output.mkdir(parents=True,exist_ok=True)
    (output/'local_transport.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.root,args.out)
