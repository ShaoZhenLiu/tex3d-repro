"""Evaluate ready camera controls alongside resource-reserved source jobs."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

from publish import R,X,P,publish,write


def alive(pid):
    p=Path('/proc/%d/status'%pid)
    return p.exists() and not any(s.startswith('State:') and s.split()[1]=='Z' for s in p.read_text().splitlines())


def main(a):
    assert json.loads((P/'evaluation/workers/lowmem_smoke6/smoke_passed.json').read_text())['passed']
    active={};launched=set();failed=[];last=0.
    if a.resume:
        old=json.loads((X/'eval_controller_status.json').read_text());failed=old['failed']
        class ExistingProcess:
            def __init__(self,pid,name):self.pid=pid;self.name=name
            def poll(self):
                if alive(self.pid):return None
                return 0 if (X/'evaluation/workers'/self.name/'completed.json').exists() else 1
        for name,pid in old['active'].items():active[name]=ExistingProcess(pid,name);launched.add(name)
    lock=(R/'experiments/refcloth_gpu_budget_4090.lock').open('a')
    slots=[(g,'gpu%dc'%g) for g in [7,0,1,5,6]]+[(g,'gpu%dd'%g) for g in [7,0,1,5,6]]
    while True:
        manifest=publish();complete=len(list((X/'evaluation/completed').glob('*.json')))
        if complete==180:break
        jobs=json.loads((X/'evaluation/jobs.json').read_text());ready=[]
        for job in jobs:
            if (X/'evaluation/completed'/(job['id']+'.json')).exists():continue
            if any(n not in manifest for n in job['conditions']):continue
            if any(not (X/'evaluation/episodes'/n).exists() for n in job['requires_existing']):continue
            claim=X/'evaluation/claims'/(job['id']+'.json')
            if claim.exists() and alive(json.loads(claim.read_text())['pid']):continue
            ready.append(job['id'])
        fcntl.flock(lock,fcntl.LOCK_EX)
        free={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()}
        used={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True).splitlines() if s.split(',')[1].strip().isdigit()}
        ram=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2
        primary=json.loads((P/'train_status_4090.json').read_text())
        coupling=json.loads((X/'train_status_4090.json').read_text())
        loading=sum(not (X/'evaluation/workers'/n/'registered.json').exists() for n in active)
        for gpu,name in slots:
            if name in launched or len(ready)<=loading or ram<50 or time.monotonic()-last<90:continue
            reserve=0;source_on_card=False
            for info in primary['active'].values():
                if info['job'][2]==gpu and alive(info['pid']):
                    source_on_card=True;reserve+=max(0,23000-used.get(info['pid'],0))
            for info in coupling['active'].values():
                if info['gpu']==gpu and alive(info['pid']):
                    source_on_card=True;reserve+=max(0,23000-used.get(info['pid'],0))
            if coupling['pending'] and not source_on_card:reserve+=23000
            if not (P/'eval_controller_done.json').exists():
                for old,pid in json.loads((P/'eval_controller_status.json').read_text())['active'].items():
                    if int(re.match(r'gpu(\d+)',old).group(1))==gpu and alive(pid):
                        reserve+=max(0,(21900 if old in ['gpu7a','gpu7b','gpu1','gpu5'] else 17000)-used.get(pid,0))
            for old,proc in active.items():
                if int(re.match(r'gpu(\d+)',old).group(1))==gpu:reserve+=max(0,17000-used.get(proc.pid,0))
            if free[gpu]-reserve<17000:continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
                HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
                USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',TORCHDYNAMO_DISABLE='1',REFCLOTH_SURFACE_MODE='legacy',
                JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',
                MPLCONFIGDIR='/tmp/refcloth_stability_mpl',PYTHONPATH=':'.join([str(R),str(R/'pi'),str(R/'ops/stability_20261102'),'/data/shaozhen.liu/code/tex3d-deps/taming-transformers']))
            cmd=['/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python',str(R/'ops/stability_20261102/evaluate.py'),
                '--repo',str(R),'--root',str(X),'--worker',name,'--memory-fraction','.31']
            proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,
                stdout=open(X/('evaluation_'+name+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;launched.add(name);last=time.monotonic();loading+=1
            print('COUPLING_EVAL_LAUNCH',name,proc.pid,flush=True)
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if code!=0:failed.append(dict(worker=name,exit_code=code));print('COUPLING_EVAL_INFRASTRUCTURE_ERROR',name,code,flush=True)
            del active[name]
        write(X/'eval_controller_status.json',dict(active={n:p.pid for n,p in active.items()},failed=failed,
            completed_bundles=complete,total_bundles=180,available_bundles=len(ready),updated=time.time()))
        fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(15)
    assert len(list((X/'evaluation/episodes').glob('*.json')))==2600
    write(X/'eval_controller_done.json',dict(completed=True,bundles=180,episodes=2600,new_episodes=1800,reused_episodes=800,failed_attempts=failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args())
