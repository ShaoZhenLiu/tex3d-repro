"""Admit exact EDPA arms on allowed GPUs; resume only the same paused arm."""
import argparse
import fcntl
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from common import HERE,REPO,ROOT,SEEDS,TRAIN_NODES,read,write,sha
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard

SCRIPT=REPO/'ops/edpa_seeds_20261110/train_arm.py'
PYTHON='/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'


def launch(node,seed,gpu,budget,validate=False):
    total={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()}[gpu]
    fraction=(budget-768)/total
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
        WANDB_MODE='disabled',USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=':'.join([str(REPO/'ops'),str(HERE),str(REPO),str(REPO/'openvla-oft')]))
    cmd=[PYTHON,str(SCRIPT),'--seed',str(seed),'--node',node,'--memory-fraction',str(fraction)]
    name='validation' if validate else 's%d_edpa'%seed
    if validate:cmd.append('--validate-only')
    proc=subprocess.Popen(cmd,cwd='/data/shaozhen.liu/code/tex3d-ws/g7',env=env,
        stdin=subprocess.DEVNULL,stdout=open(ROOT/'logs'/(name+'.log'),'a'),
        stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+10
    while True:
        try:
            token=guard.register(proc.pid,SCRIPT,'source',gpu,budget,'edpa:'+name,ROOT);break
        except AssertionError:
            if proc.poll() is not None:raise RuntimeError('Source exited before registration: '+name)
            if time.monotonic()>deadline:proc.terminate();raise RuntimeError('Source command not visible: '+name)
            time.sleep(.05)
    return dict(pid=proc.pid,ticks=guard.identity(proc.pid)['ticks'],token=token,name=name,seed=seed,
        gpu=gpu,budget=budget,validate=validate,proc=proc)


def main(node):
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    owner=(ROOT/('source_'+node+'.lock')).open('a')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    assert read(ROOT/'inputs.json',{}).get('protocol_sha256')==sha(HERE/'protocol.json')
    assert read(ROOT/('smoke_'+node+'/prepared.json'),{}).get('passed')
    status_path=ROOT/('source_status_'+node+'.json')
    active={};pauses=read(status_path,{}).get('resource_pauses',[])
    for e in guard.active():
        if e['root']!=str(ROOT) or not e['label'].startswith('edpa:'):continue
        name=e['label'].split(':',1)[1];validate=name=='validation'
        active[name]=dict(pid=e['pid'],ticks=e['ticks'],token=e['token'],name=name,gpu=e['gpu'],
            budget=e['budget'],validate=validate,seed=7 if validate else int(name.split('_')[0][1:]),proc=None)
    assigned=[s for s in SEEDS if TRAIN_NODES[s]==node]
    budget_lock=(REPO/('experiments/refcloth_gpu_budget_'+node+'.lock')).open('a')
    try:
        while True:
            for name,e in list(active.items()):
                current=guard.identity(e['pid'])
                if current and current['ticks']==e['ticks']:continue
                done=read(ROOT/'gradient_validation.json',{}).get('passed') if e['validate'] else read(ROOT/'arms'/name/'completed.json',{}).get('completed')
                code=e['proc'].poll() if e['proc'] else None
                if done:print('FINISHED',name,flush=True)
                elif code==75 or guard.resource_stopped(e['pid'],e['ticks']):
                    pauses.append(dict(name=name,time=time.time(),pid=e['pid']));print('RESOURCE_REQUEUED',name,flush=True)
                else:raise RuntimeError('Source exited without a valid completion/checkpoint pause: %s exit=%r'%(name,code))
                del active[name]
            finished=[s for s in assigned if read(ROOT/'arms'/('s%d_edpa'%s)/'completed.json',{}).get('completed')]
            pending=[s for s in assigned if s not in finished and 's%d_edpa'%s not in active]
            validation=read(ROOT/'gradient_validation.json',{})
            if len(finished)==len(assigned):
                write(ROOT/('source_done_'+node+'.json'),dict(completed=True,seeds=assigned));return
            reasons=[]
            fcntl.flock(budget_lock,fcntl.LOCK_EX)
            try:
                allowed,reasons=guard.can_launch(cold_gib=10,max_models=6 if node=='09' else 5)
                if allowed and pending:
                    if not validation.get('passed'):
                        vgpus=[3] if node=='4090' else [0,1,2,3]
                        candidates=[g for g in vgpus if guard.gpu_headroom(g)>=(43000 if node=='4090' else 32000)]
                        if 'validation' not in active and candidates:
                            gpu=max(candidates,key=guard.gpu_headroom)
                            budget=min(43000,guard.gpu_headroom(gpu)-768)
                            e=launch(node,7,gpu,budget,True);active[e['name']]=e;print('LAUNCH_VALIDATION',e['pid'],flush=True)
                        else:reasons.append('waiting_for_verified_production_gradient')
                    else:
                        peak=max(validation['steady_peak_allocated_mib'],validation['steady_peak_reserved_mib'])
                        budget=int(math.ceil(peak+1536))
                        choices=[g for g in [0,1,2,3] if guard.gpu_headroom(g)>=budget]
                        if node=='4090':choices=[g for g in choices if not any(v['kind']=='source' and v['gpu']==g for v in guard.active())]
                        if choices:
                            gpu=max(choices,key=guard.gpu_headroom)
                            e=launch(node,pending[0],gpu,budget);active[e['name']]=e
                            print('LAUNCH',e['name'],e['pid'],'GPU',gpu,'budget_mib',budget,flush=True)
                        else:reasons.append('no_gpu_with_profiled_headroom')
            finally:fcntl.flock(budget_lock,fcntl.LOCK_UN)
            write(status_path,dict(node=node,pending=pending,finished=finished,active={k:{x:y for x,y in v.items() if x!='proc'} for k,v in active.items()},
                resource_pauses=pauses,admission_wait=reasons,updated=time.time()))
            time.sleep(10)
    except Exception as error:
        write(ROOT/('source_error_'+node+'.json'),dict(type=type(error).__name__,error=str(error),time=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090','09'],required=True);main(p.parse_args().node)
