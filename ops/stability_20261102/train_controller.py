"""Resource-checked detached source replications on explicitly authorized devices."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO=Path('/data/shaozhen.liu/code/tex3d')
ROOT=REPO/'experiments/refcloth_stability/20261102'
OPS=REPO/'ops/stability_20261102'
PYTHON='/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'


def resources():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(row.split(',')[0]):int(row.split(',')[1]) for row in rows.splitlines()}
    memory={row.split(':')[0]:int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
    return free,memory['MemAvailable']/1024**2


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def main(a):
    protocol=json.loads((OPS/'protocol.json').read_text())
    assert protocol['frozen_renderer_exception']['authorized_by_user']
    assert json.loads((ROOT/'streams/prepared.json').read_text())['passed']
    if a.node=='09':
        assert json.loads((ROOT/'streams/preflight_09.json').read_text())['passed']
        jobs=[(12,'native',1),(12,'mr',2)];fraction=.37
    else:
        jobs=[(10,'native',0),(10,'mr',6),(11,'native',1),(11,'mr',5)];fraction=.62
    assert all(g!=4 for _,_,g in jobs)
    pending=list(jobs);active={};finished=[];failed=[];last_launch=0.
    (ROOT/'logs').mkdir(exist_ok=True)
    while pending or active:
        if pending and time.monotonic()-last_launch>=60:
            seed,ref,gpu=pending[0]
            free,ram=resources()
            if free[gpu]>=34000 and ram>=50:
                name='s%d_%s'%(seed,ref)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                    MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    MAX_JOBS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                    WANDB_MODE='disabled',USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                    PYTHONPATH=str(REPO)+':'+str(REPO/'openvla-oft'))
                command=[PYTHON,str(OPS/'train_pair_arm.py'),'--repo',str(REPO),'--root',str(ROOT),
                    '--seed',str(seed),'--ref',ref,'--memory-fraction',str(fraction),'--allow-frozen-wrist-gap']
                log=open(ROOT/'logs'/(name+'.log'),'a')
                proc=subprocess.Popen(command,cwd='/data/shaozhen.liu/code/tex3d-ws/g7',env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]=dict(proc=proc,job=(seed,ref,gpu),start=time.time())
                pending.pop(0);last_launch=time.monotonic()
                print('LAUNCH',name,proc.pid,'GPU',gpu,flush=True)
        for name,value in list(active.items()):
            code=value['proc'].poll()
            if code is None:continue
            result=ROOT/'arms'/name/'completed.json'
            if code==0 and result.exists() and json.loads(result.read_text())['completed']:
                finished.append(name);print('FINISHED',name,flush=True)
            else:
                failed.append(dict(name=name,exit_code=code));print('INFRASTRUCTURE_ERROR',name,code,flush=True)
            del active[name]
        write(ROOT/('train_status_'+a.node+'.json'),dict(node=a.node,pending=pending,
            active={n:dict(pid=v['proc'].pid,job=v['job'],start=v['start']) for n,v in active.items()},
            finished=finished,failed=failed,updated=time.time()))
        if pending or active:time.sleep(10)
    write(ROOT/('train_done_'+a.node+'.json'),dict(completed=not failed,finished=finished,failed=failed))
    if failed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090','09'],required=True)
    main(p.parse_args())
