"""Queue the nine controls without displacing the ongoing reference study."""
import argparse
import json
import os
import re
import subprocess
import time
import fcntl
from pathlib import Path


R=Path('/data/shaozhen.liu/code/tex3d')
X=R/'experiments/refcloth_coupling/20261102'
P=R/'experiments/refcloth_stability/20261102'
O=R/'ops/coupling_20261102'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def alive(pid):
    p=Path('/proc/%d/status'%pid)
    return p.exists() and not any(s.startswith('State:') and s.split()[1]=='Z' for s in p.read_text().splitlines())


def main(a):
    assert json.loads((X/'prepared.json').read_text())['passed']
    if a.node=='09':
        pending=[(12,'uv_ind'),(12,'main'),(12,'wrist')];gpus=[2,1];limit=1;required=24000;fraction=.27
    else:
        pending=[(10,'uv_ind'),(11,'uv_ind'),(10,'main'),(10,'wrist'),(11,'main'),(11,'wrist')]
        gpus=[6,0,1,5,7];limit=2;required=23000;fraction=.44
    active={};finished=[];failed=[];last=0.
    (X/'logs').mkdir(exist_ok=True);(X/'arms').mkdir(exist_ok=True)
    if a.resume:
        prior=json.loads((X/('train_status_'+a.node+'.json')).read_text())
        pending=[tuple(j) for j in prior['pending']];finished=prior['finished'];failed=prior['failed']
        class ExistingProcess:
            def __init__(self,pid,name):self.pid=pid;self.name=name
            def poll(self):
                if alive(self.pid):return None
                return 0 if (X/'arms'/self.name/'completed.json').exists() else 1
        for name,v in prior['active'].items():
            active[name]=dict(proc=ExistingProcess(v['pid'],name),gpu=v['gpu'],job=tuple(v['job']),start=v['start'])
    budget_lock=(R/('experiments/refcloth_gpu_budget_'+a.node+'.lock')).open('a')
    while pending or active:
        fcntl.flock(budget_lock,fcntl.LOCK_EX)
        primary=json.loads((P/('train_status_'+a.node+'.json')).read_text())
        free={int(row.split(',')[0]):int(row.split(',')[1]) for row in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()}
        mem_by_pid={int(row.split(',')[0]):int(row.split(',')[1]) for row in subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True).splitlines() if row.split(',')[1].strip().isdigit()}
        ram=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2
        counts={g:sum(v['gpu']==g for v in active.values()) for g in gpus}
        candidates=[]
        for gpu in sorted(gpus,key=lambda g:(counts[g],gpus.index(g))):
            if counts[gpu]>=limit:continue
            if a.node=='09' and gpu==1 and primary['active']:continue
            reserve=sum(max(0,(24000 if a.node=='09' else 23000)-mem_by_pid.get(v['pid'],0))
                for v in primary['active'].values() if v['job'][2]==gpu and alive(v['pid']))
            reserve+=sum(max(0,required-mem_by_pid.get(v['proc'].pid,0)) for v in active.values() if v['gpu']==gpu)
            if a.node=='4090':
                for path in [P/'eval_controller_status.json',X/'eval_controller_status.json']:
                    if not path.exists() or (path.parent/'eval_controller_done.json').exists():continue
                    for name,pid in json.loads(path.read_text())['active'].items():
                        if int(re.match(r'gpu(\d+)',name).group(1))==gpu and alive(pid):
                            reserve+=max(0,(21900 if name in ['gpu7a','gpu7b','gpu1','gpu5'] else 17000)-mem_by_pid.get(pid,0))
            if free[gpu]-reserve>=required:candidates.append(gpu)
        if pending and candidates and ram>=50 and time.monotonic()-last>=90:
            seed,mode=pending.pop(0);gpu=candidates[0];name='s%d_%s'%(seed,mode)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
                HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
                USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                PYTHONPATH=':'.join([str(O),str(R),str(R/'openvla-oft')]))
            cmd=['/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python',str(O/'train_arm.py'),
                '--repo',str(R),'--root',str(X),'--seed',str(seed),'--mode',mode,'--memory-fraction',str(fraction)]
            proc=subprocess.Popen(cmd,cwd='/data/shaozhen.liu/code/tex3d-ws/g7',env=env,
                stdin=subprocess.DEVNULL,stdout=open(X/'logs'/(name+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=dict(proc=proc,gpu=gpu,job=(seed,mode),start=time.time());last=time.monotonic()
            print('COUPLING_LAUNCH',name,gpu,proc.pid,flush=True)
        for name,item in list(active.items()):
            code=item['proc'].poll()
            if code is None:continue
            meta=X/'arms'/name/'completed.json'
            if code==0 and meta.exists():finished.append(name);print('COUPLING_FINISHED',name,flush=True)
            else:failed.append(dict(name=name,exit_code=code));print('COUPLING_INFRASTRUCTURE_ERROR',name,code,flush=True)
            del active[name]
        write(X/('train_status_'+a.node+'.json'),dict(node=a.node,pending=pending,
            active={name:dict(pid=v['proc'].pid,gpu=v['gpu'],job=v['job'],start=v['start']) for name,v in active.items()},
            finished=finished,failed=failed,allowed_gpus=gpus,required_mib=required,updated=time.time()))
        fcntl.flock(budget_lock,fcntl.LOCK_UN)
        if pending or active:time.sleep(10)
    write(X/('train_done_'+a.node+'.json'),dict(completed=not failed,finished=finished,failed=failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090','09'],required=True)
    p.add_argument('--resume',action='store_true');main(p.parse_args())
