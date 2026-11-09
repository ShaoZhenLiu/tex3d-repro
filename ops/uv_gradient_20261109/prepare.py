"""Recheck frozen smoke gates; prepare paired streams and disjoint held-out frames."""
import argparse
import gc
import random
import sys
import time

import numpy as np
import torch

from common import ENTRY, OPS, REPO, ROOT, read, runner, sha, standalone, write


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu(item) for item in value)
    if hasattr(value,'items'):return {key:cpu(item) for key,item in value.items()}
    return value


def smoke():
    sys.path.insert(0,str(REPO/'ops/stability_20261102'))
    import prepare as original
    original.main(argparse.Namespace(repo=REPO,protocol=OPS/'protocol_prepare.json',
        out=ROOT/'streams',verify_only=False))
    R=runner()
    assert sha(R.POOL)==R.POOL_SHA and sha(R.MR_CACHE)==R.CACHE_SHA
    renderer=R.M.ClothRendererOV(device='cuda',epsilon=64/255,init_std=0.,seed=7).to('cuda')
    pool=torch.load(R.POOL,map_location='cpu',weights_only=False,mmap=True)
    errors=[]
    for task,frames in pool.items():
        frame=R.frame_to_dev(frames[0],'cuda')
        for _ in range(3):
            uv=R.M.sample_uv_xform(True,'cuda')
            with torch.no_grad():
                a,_=renderer.render(frame['mvp'],(256,256),model_rot=frame['rot'],uv_xform=None)
                b,_=renderer.render(frame['mvp'],(256,256),model_rot=frame['rot'],uv_xform=uv)
            errors.append(float((a-b).abs().max()))
    assert max(errors)<1e-5,errors
    result=read(ROOT/'streams/prepared.json')
    result.update(plain_uv_invariance_maxdiff=max(errors),passed=True)
    write(ROOT/'prepared/smoke.json',result)
    print('SMOKE_COMPLETED',flush=True)


def heldout():
    import refcloth_resource as guard
    torch.set_num_threads(2)
    ctx,processor=standalone()
    guard.mark_ready()
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    frames={};provenance=[]
    output=ROOT/'prepared/heldout.pt'
    if output.exists():raise RuntimeError('Held-out data already exists; do not overwrite')
    R=ctx.R
    for task in [1,3,4,5,7,9]:
        task_file=ROOT/'prepared'/('heldout_task_%d.pt'%task)
        if task_file.exists():
            entry=torch.load(task_file,map_location='cpu',weights_only=False,mmap=True)
            frames[task]=entry['frames'];provenance+=entry['provenance'];continue
        values=[];records=[]
        for initial in [0,1]:
            random.seed(20261109+task*100+initial)
            np.random.seed(20261109+task*100+initial);torch.manual_seed(20261109+task*100+initial)
            batch=R.M.collect_frames(ctx.cfg,ctx.model,processor,ctx.action_head,
                ctx.proprio_projector,ctx.noisy_action_projector,suite.get_task(task),
                suite.get_task_init_states(task)[initial],24,'cuda',stride=10,max_frames=24,ep_tag=initial)
            values.extend(cpu(batch));records.append(dict(task=task,initial=initial,frames=len(batch)))
            del batch;gc.collect();torch.cuda.empty_cache()
        assert len(values)>=8,(task,len(values))
        frames[task]=values;provenance+=records
        torch.save(dict(frames=values,provenance=records),task_file)
    references={}
    with torch.no_grad():ctx.renderer.adv_noise.zero_()
    for task,items in frames.items():
        references[task]=[]
        for index,frame in enumerate(items):
            with torch.no_grad():hidden=ctx.forward(frame,None)
            references[task].append(hidden.cpu())
        print('HELDOUT_REFERENCES',task,len(items),flush=True)
    torch.save(dict(frames=frames,reference_hidden=references,provenance=provenance,
        source_only=True,used_in_training=False),output)
    pool=torch.load(R.POOL,map_location='cpu',weights_only=False,mmap=True)
    counts={task:len(items) for task,items in pool.items()}
    counts.update({task:len(items) for task,items in frames.items()})
    bank_frames={task:[random.Random(20261109+task*1000+rep).sample(range(counts[task]),4)
                      for rep in range(8)] for task in sorted(counts)}
    state=random.getstate();random.seed(20261109+1)
    uvs=[R.M.sample_uv_xform(True,'cpu') for _ in range(8)];random.setstate(state)
    bank=ROOT/'prepared/probe_bank.pt'
    preliminary=torch.load(bank,map_location='cpu',weights_only=False)
    assert all(preliminary['frames'][task]==bank_frames[task] for task in [0,2,6,8])
    assert all(torch.equal(a,b) for a,b in zip(preliminary['uv'],uvs))
    # Keep the immutable training probe bank; add a separate held-out bank.
    final_bank=ROOT/'prepared/heldout_probe_bank.pt'
    torch.save(dict(frames=bank_frames,uv=uvs,replicates=8,frames_per_replicate=4,counts=counts),final_bank)
    write(ROOT/'prepared/heldout.json',dict(completed=True,sha256=sha(output),
        counts=counts,provenance=provenance,probe_bank_sha256=sha(final_bank),time=time.time()))


def bank():
    R=runner()
    pool=torch.load(R.POOL,map_location='cpu',weights_only=False,mmap=True)
    counts={task:len(items) for task,items in pool.items()}
    sample_rng=random.Random(20261109)
    # Reserve the full ten-task ordering now; held-out counts are not yet known.
    # Separate per-task RNGs prevent held-out collection from changing train probes.
    frames={task:[random.Random(20261109+task*1000+rep).sample(range(counts[task]),4)
                  for rep in range(8)] for task in sorted(counts)}
    random.seed(20261109+1)
    uvs=[R.M.sample_uv_xform(True,'cpu') for _ in range(8)]
    (ROOT/'prepared').mkdir(exist_ok=True)
    torch.save(dict(frames=frames,uv=uvs,replicates=8,frames_per_replicate=4,counts=counts),
        ROOT/'prepared/probe_bank.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','heldout','bank'])
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='smoke':smoke();bank()
    elif args.stage=='bank':bank()
    else:heldout()
