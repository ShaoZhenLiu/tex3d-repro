"""A persistent single-GPU native query worker, coordinated by atomic JSON files."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from native_runtime import NativeRuntime
from support_core import array_sha, binary_texture, opacity_texture, action_metrics


def save(data,path):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def main(repo,out,worker_id,protocol):
    torch.manual_seed(7)
    np.random.seed(7)
    runtime=NativeRuntime(repo,protocol['gpus']['fraction_per_worker'],out)
    base_path=Path(runtime.A.M.CLOTH_T0)
    assert __import__('hashlib').sha256(base_path.read_bytes()).hexdigest()==protocol['base_texture_sha256']
    grid=protocol['grid']
    fit_tasks=[[0,1],[2,8]][worker_id]
    val_tasks=[[0,1,5,6],[2,8,7,9]][worker_id]
    def collect(split,tasks):
        cfg=protocol[split]
        states=runtime.collect(tasks,cfg['initial'],cfg['steps'])
        meta=[]
        for state in states:
            v=state.visibility(grid)
            record=dict(state.meta,plain_action=state.a0.tolist(),full_action=state.full_action.tolist(),visibility=v)
            meta.append(record)
            sample=out/'examples'/state.key
            sample.mkdir(parents=True,exist_ok=True)
            _,adv_pair=state.query(runtime.full)
            for view,p,a in zip(['main','wrist'],state.clean_pair,adv_pair):
                Image.fromarray(p).save(sample/(view+'_plain.png'))
                Image.fromarray(a).save(sample/(view+'_full.png'))
        return states,meta
    states,meta=collect('fit',fit_tasks)
    ready=dict(worker_id=worker_id,states=meta,scale=runtime.scale.tolist(),action_stats=runtime.action_stats,
               full_pixel_sha256=array_sha(runtime.full),base_pixel_sha256=array_sha(runtime.base),
               full_png_file_sha256=__import__('hashlib').sha256(Path(runtime.full_path).read_bytes()).hexdigest())
    save(ready,out/('ready%d.json'%worker_id))
    last=-1
    try:
        while True:
            request_path=out/'request.json'
            if not request_path.exists():
                time.sleep(.15);continue
            request=json.loads(request_path.read_text())
            seq=request['seq']
            if seq==last:
                time.sleep(.15);continue
            last=seq
            if request['op']=='stop':
                break
            if request['op']=='validation':
                states,meta=collect('validation',val_tasks)
                result=dict(seq=seq,worker=worker_id,states=meta)
            else:
                rows=[]
                for item in request['candidates']:
                    png=(binary_texture(runtime.full,runtime.base,item['keep'],grid) if 'keep' in item else
                         opacity_texture(runtime.full,runtime.base,item['opacity']))
                    for state in states:
                        action=state.action(png)
                        metrics=action_metrics(action,state.full_action,state.a0,runtime.scale)
                        rows.append(dict(state=state.key,task=state.meta['task'],candidate=item['name'],
                            action=action.tolist(),png_pixel_sha256=array_sha(png),**metrics))
                result=dict(seq=seq,worker=worker_id,rows=rows,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            save(result,out/('response%d.json'%worker_id))
            print('RESPONSE',seq,len(result.get('rows',[])),flush=True)
    finally:
        runtime.close_states()
    save(dict(stopped=True,pid=os.getpid()),out/('stopped%d.json'%worker_id))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--id',type=int,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args()
    main(args.repo,args.out,args.id,json.loads(args.protocol.read_text()))
