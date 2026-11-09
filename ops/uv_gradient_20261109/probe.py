"""Observe raw per-task gradients without modifying the training state."""
import itertools
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, Context, read, sha, write

TRAIN=[0,2,6,8]
HELDOUT=[1,3,4,5,7,9]


@contextmanager
def preserve(ctx):
    param=ctx.renderer.adv_noise
    python=random.getstate();numpy=np.random.get_state()
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    previous=param.grad
    theta=param.detach().clone()
    try:yield
    finally:
        assert torch.equal(param.detach(),theta),'Measurement changed the training texture'
        param.grad=previous
        random.setstate(python);np.random.set_state(numpy)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)


def cosine(a,b):
    denominator=a.norm()*b.norm()
    if float(denominator)<1e-30:return None
    return float(torch.dot(a,b)/denominator)


def feasible(g,theta):
    return torch.where(((theta>=5)&(g>0))|((theta<=-5)&(g<0)),0.,g)


def pair_rows(grads,regularizer,theta,replicate):
    rows=[]
    for first,second in itertools.combinations(TRAIN,2):
        left,right=grads[first],grads[second]
        for objective,a,b in [('feature',left,right),
                ('full',left-regularizer,right-regularizer),
                ('feasible_feature',feasible(left,theta),feasible(right,theta))]:
            value=cosine(a,b)
            rows.append(dict(replicate=replicate,task_i=first,task_j=second,objective=objective,
                cosine=value,conflict=value<0 if value is not None else None,
                norm_i=float(a.norm()),norm_j=float(b.norm())))
    return rows


def measure(ctx,updates,*,heldout=False,only_heldout=False):
    started=time.monotonic()
    bank_path=ROOT/'prepared'/('heldout_probe_bank.pt' if heldout else 'probe_bank.pt')
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    theta=ctx.renderer.adv_noise.detach().flatten()
    regularizer=ctx.regularizer()
    groups=TRAIN+HELDOUT if heldout else TRAIN
    result=dict(updates=updates,policies={},elapsed_seconds=None,
        theta_boundary_fraction=float((theta.abs()>=5-1e-6).float().mean()),
        tanh_derivative_below_001=float(((1-torch.tanh(theta)**2)<.01).float().mean()),
        regularizer_norm=float(regularizer.norm()),bank_sha256=sha(bank_path))
    for policy in ['identity','augmented']:
        sums={};scores={};pairs=[];held=[]
        for rep in range(bank['replicates']):
            uv=None if policy=='identity' else bank['uv'][rep].to('cuda')
            gradients={}
            for task in groups:
                g,score=ctx.gradient(task,bank['frames'][task][rep],uv)
                gradients[task]=g
                sums[task]=g.clone() if task not in sums else sums[task]+g
                scores.setdefault(task,[]).append(score/10.)
            if not only_heldout:pairs.extend(pair_rows(gradients,regularizer,theta,rep))
            if heldout:
                average=sum(gradients[t] for t in TRAIN)/len(TRAIN)
                for task in HELDOUT:
                    held.append(dict(replicate=rep,task=task,cosine=cosine(average,gradients[task])))
            print('PROBE_STEP',updates,policy,'rep',rep+1,'/',bank['replicates'],flush=True)
        pooled={t:g/bank['replicates'] for t,g in sums.items()}
        pooled_pairs=[] if only_heldout else pair_rows(pooled,regularizer,theta,'pooled')
        pooled_held=[]
        if heldout:
            average=sum(pooled[t] for t in TRAIN)/len(TRAIN)
            pooled_held=[dict(task=t,cosine=cosine(average,pooled[t])) for t in HELDOUT]
        feature=[r['cosine'] for r in pairs if r['objective']=='feature' and r['cosine'] is not None]
        pooled_feature=[r['cosine'] for r in pooled_pairs if r['objective']=='feature' and r['cosine'] is not None]
        result['policies'][policy]=dict(pair_rows=pairs,pooled_pair_rows=pooled_pairs,
            heldout_rows=held,pooled_heldout_rows=pooled_held,
            feature_mse={t:float(np.mean(v)) for t,v in scores.items()},
            feature_pair_cosine_mean=float(np.mean(feature)) if feature else None,
            pooled_feature_pair_cosine_mean=float(np.mean(pooled_feature)) if pooled_feature else None,
            conflict_numerator=sum(v<0 for v in feature),conflict_denominator=len(feature),
            heldout_cosine_mean=float(np.mean([r['cosine'] for r in held])) if held else None,
            pooled_heldout_cosine_mean=float(np.mean([r['cosine'] for r in pooled_held])) if held else None)
    result['elapsed_seconds']=time.monotonic()-started
    result['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    result['observational_parameter_check_passed']=True
    return result


def callback_for(seed,uv,R):
    directory=ROOT/'arms'/('s%d_%s'%(seed,uv))/'probes'
    ctx=None
    def callback(state,updates):
        nonlocal ctx
        # A resource checkpoint can be between regular probe checkpoints.
        if updates%500:return
        if ctx is None:ctx=Context(state,R)
        destination=directory/('u%05d.json'%updates)
        if not destination.exists():
            with preserve(ctx):
                final=updates==5000
                if final:ctx.attach_heldout()
                result=measure(ctx,updates,heldout=final)
            result.update(seed=seed,train_uv=uv,parameter_and_rng_state_restored=True)
            write(destination,result)
        if updates==5000:
            initial=directory/'heldout_initial.json'
            if not initial.exists():
                import os
                theta_path=state['out_dir']+'/theta_init.pt'
                original=ctx.renderer.adv_noise.detach().clone()
                try:
                    with torch.no_grad():
                        ctx.renderer.adv_noise.copy_(torch.load(theta_path,map_location='cuda',weights_only=False))
                    with preserve(ctx):
                        result=measure(ctx,0,heldout=True,only_heldout=True)
                finally:
                    with torch.no_grad():ctx.renderer.adv_noise.copy_(original)
                result.update(seed=seed,train_uv=uv,parameter_and_rng_state_restored=True,
                    timing='Initial held-out diagnostic computed at end; never used by training.')
                write(initial,result)
    return callback
