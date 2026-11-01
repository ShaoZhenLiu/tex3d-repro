"""Detached, resource-checked Object jobs on authorized 4090 GPUs only."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo=Path('/data/shaozhen.liu/code/tex3d')
ops=repo/'ops/object_20261102'
root=repo/'experiments/refcloth_object/20261102'
python='/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python'
jobs=[('pi0_main','pi0','0,1,2,3,4,5,6',0,.55),
      ('pi05_main','pi05','0,1,2,3,4,5,6',6,.55),
      ('pi0_tail','pi0','7,8,9',3,.40),
      ('pi05_tail','pi05','7,8,9',3,.40)]


def resources():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in rows.splitlines()}
    mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
    return free,mem['MemAvailable']/1024**2


def launch(name,target,tasks,gpu,fraction,mode='run'):
    assert gpu in [0,3,6] and gpu!=4
    free,ram=resources()
    assert free[gpu]>=23000 and ram>=65,(free,ram)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
        WANDB_MODE='disabled',OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',MPLCONFIGDIR='/tmp/refcloth_object_mpl',
        TORCHDYNAMO_DISABLE='1',PYTHONPATH=':'.join([str(repo),str(repo/'pi'),str(ops),'/data/shaozhen.liu/code/tex3d-deps/taming-transformers']))
    out=root/name;out.mkdir(parents=True,exist_ok=True)
    log=open(str(out)+'.log','a')
    cmd=[python,str(ops/'run_object.py'),'--repo',str(repo),'--out',str(out),'--protocol',str(ops/'protocol.json'),
         '--target',target,'--tasks',tasks,'--memory-fraction',str(fraction),'--mode',mode]
    proc=subprocess.Popen(cmd,cwd=str(repo),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('LAUNCH',name,proc.pid,'GPU',gpu,flush=True)
    return proc


def main():
    # Existing pi05 smoke was started by the operator before this controller.
    start=time.monotonic()
    while not (root/'smoke_pi05/smoke_passed.json').exists():
        if time.monotonic()-start>900:
            raise RuntimeError('pi05 smoke did not pass; no evaluation jobs started')
        time.sleep(5)
    smoke=launch('smoke_pi0','pi0','0',6,.55,'smoke')
    assert smoke.wait()==0 and (root/'smoke_pi0/smoke_passed.json').exists(),'pi0 instrumentation parity failed'
    active={}
    pending=list(jobs)
    finished=[]
    while pending or active:
        for job in list(pending):
            name,target,tasks,gpu,fraction=job
            if any(x[0][3]==gpu for x in active.values()):continue
            free,ram=resources()
            if free[gpu]<23000 or ram<65:continue
            active[name]=(job,launch(*job));pending.remove(job)
        for name,(job,proc) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            assert code==0 and (root/name/'results.json').exists(),(name,code,'Inspect log; no invalid outcomes counted')
            finished.append(name);del active[name]
            print('FINISHED',name,flush=True)
        tmp=root/'controller_status.tmp'
        tmp.write_text(json.dumps(dict(pending=[x[0] for x in pending],active={k:v[1].pid for k,v in active.items()},finished=finished)))
        tmp.replace(root/'controller_status.json')
        time.sleep(5)
    (root/'controller_done.json').write_text(json.dumps(dict(completed=True,finished=finished)))


if __name__=='__main__':main()
