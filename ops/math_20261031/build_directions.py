"""Solve source fit-state universal steps; no heldout native scores are read."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from box_maximin import update_box, normalize_rows, mean_sign, scaled_pcgrad, solve_box_maximin


def build(paths, output):
    banks=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    assert all(b['initial']==5 for b in banks)
    assert sum([b['tasks'] for b in banks],[])==[0,2,6,8]
    result=dict(fit_initial=5,tasks=[0,2,6,8],starts={})
    report=dict(fit_only=True,fit_initial=5,validation_initial=6,records=[])
    for start in ['init','moderate','frozen_mr']:
        first=banks[0]['starts'][start]
        theta=first['theta']
        assert all(torch.equal(theta,b['starts'][start]['theta']) for b in banks)
        gs={m:torch.cat([b['starts'][start]['gradients'][m] for b in banks]).double().flatten(1).numpy()
            for m in first['gradients']}
        penalty=first['penalty_gradient'].double().flatten().numpy()
        total=10*gs['fd']-penalty
        output_steps={}
        for eta in [.01,.05]:
            lo,hi=update_box(theta.flatten().double().tolist(),eta)
            steps={}
            for normalization in ['raw','l1']:
                q,scales=normalize_rows(total.tolist(),normalization)
                q=np.asarray(q)
                common_scale=max(np.max(np.abs(q)),1e-15)
                scaled=q/common_scale
                candidates=dict(mean_sign=mean_sign(q.tolist(),lo,hi),
                    pcgrad_sign=scaled_pcgrad(q.tolist(),lo,hi,take_sign=True),
                    pcgrad_vector=scaled_pcgrad(q.tolist(),lo,hi,take_sign=False))
                solved=solve_box_maximin(scaled.tolist(),lo,hi,method='scipy')
                candidates['maximin']=solved['delta']
                for name,delta in candidates.items():
                    delta=np.asarray(delta)
                    predicted=q@delta
                    record=dict(start=start,eta=eta,method=normalization+'_'+name,
                        predicted_gains=predicted.tolist(),minimum_predicted_gain=float(predicted.min()),
                        gradient_scales=scales,linf=float(abs(delta).max()),l2=float(np.linalg.norm(delta)),
                        fractional_coordinates=int(np.sum((delta>np.asarray(lo)+1e-8)&(delta<np.asarray(hi)-1e-8))))
                    if name=='maximin':
                        record.update(duality_gap=solved['duality_gap']*common_scale,
                                      dual_weights=solved['weights'])
                    report['records'].append(record)
                    steps[record['method']]=torch.tensor(delta,dtype=theta.dtype).reshape_as(theta)
            for name,g in [('action_mean',gs['action'].mean(0)),
                           ('interaction_mean',gs['interaction'].mean(0)),
                           ('interaction_penalized',gs['protected'].mean(0))]:
                steps[name]=torch.tensor(np.where(g>0,hi,np.where(g<0,lo,0)),dtype=theta.dtype).reshape_as(theta)
            # Simultaneous first-order solo protection on all fit-state views.
            objective=gs['interaction'].mean(0)
            solo=np.concatenate([gs['solo_main'],gs['solo_wrist']])
            solo=solo/np.maximum(np.sum(abs(solo),axis=1,keepdims=True),1e-15)
            if np.linalg.norm(objective)<1e-14:
                delta=np.zeros_like(objective)
                lp_message='zero interaction gradient; no arbitrary LP step'
            else:
                objective=objective/max(np.max(abs(objective)),1e-15)
                lp=linprog(-objective,A_ub=solo,b_ub=np.zeros(len(solo)),bounds=list(zip(lo,hi)),method='highs')
                assert lp.success,lp.message
                delta=lp.x
                assert np.max(solo@delta)<1e-6
                lp_message=lp.message
            steps['interaction_solo_lp']=torch.tensor(delta,dtype=theta.dtype).reshape_as(theta)
            report['records'].append(dict(start=start,eta=eta,method='interaction_solo_lp',
                maximum_solo_linear_gain=float(np.max(solo@delta)),status=lp_message,
                interaction_linear_gain=float(gs['interaction'].mean(0)@delta)))
            output_steps[str(eta)]=steps
        result['starts'][start]=dict(theta=theta,steps=output_steps)
    output.mkdir(parents=True,exist_ok=True)
    torch.save(result,output/'directions.partial.pt')
    (output/'directions.partial.pt').replace(output/'directions.pt')
    (output/'direction_solver.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('directions written',len(report['records']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--banks',type=Path,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    build(a.banks,a.out)
