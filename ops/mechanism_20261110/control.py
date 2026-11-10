"""Resource-checked same-node inference, up to two workers on GPUs 0--3."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from common import HERE,PROTOCOL,REPO,ROOT,read,write,sha
import refcloth_resource as guard

PYTHON='/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python'
SCRIPT=HERE/'evaluate.py'


def launch(gpu,index,smoke=False):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
        OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',MPLCONFIGDIR='/tmp/refcloth_mechanism_mpl',
        TORCHDYNAMO_DISABLE='1',REFCLOTH_SURFACE_MODE='legacy',
        PYTHONPATH=':'.join([str(HERE),str(REPO/'ops'),str(REPO/'pi'),str(REPO),
            '/data/shaozhen.liu/code/tex3d-deps/taming-transformers']))
    name='w%02d_gpu%d'%(index,gpu)
    cmd=[PYTHON,str(SCRIPT),'--worker',name]+(['--smoke-first'] if smoke else [])
    log=(ROOT/'logs'/(name+'.log')).open('a')
    proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    guard.register(proc.pid,SCRIPT,'eval',gpu,23000,name,ROOT)
    print('LAUNCH',name,proc.pid,gpu,'smoke',smoke,flush=True)
    return dict(worker=name,pid=proc.pid,ticks=guard.identity(proc.pid)['ticks'],gpu=gpu)


def main(args):
    ROOT.mkdir(parents=True,exist_ok=True)
    for sub in ['logs','workers','episodes','claims','completed']:(ROOT/sub).mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert commit==args.commit
    old=read(ROOT/'protocol.json')
    p=read(PROTOCOL)
    assert old is None or old==p
    write(ROOT/'protocol.json',p)
    write(ROOT/'run_manifest.json',dict(commit=commit,protocol_sha256=sha(PROTOCOL),
        entry_sha256=sha(SCRIPT),allowed_gpus=[0,1,2,3],max_workers=8,per_gpu=2,started=time.time()))
    registry=[];index=0;failures=[]
    while True:
        active=[e for e in guard.active() if e['script']==str(SCRIPT) and e['root']==str(ROOT)]
        known={e['pid'] for e in registry}
        for e in active:
            if e['pid'] not in known:registry.append(dict(worker=e['label'],pid=e['pid'],ticks=e['ticks'],gpu=e['gpu']))
        for e in list(registry):
            if e.get('exited') or guard.process_runs_script(e['pid'],SCRIPT,e['ticks']):continue
            e['exited']=time.time()
            complete=read(ROOT/'workers'/e['worker']/'completed.json',{}).get('completed')
            if not complete:
                resource=guard.resource_stopped(e['pid'],e['ticks'])
                failures.append(dict(worker=e['worker'],resource_stop=resource,time=time.time()))
                print('WORKER_EXIT',e['worker'],'resource',resource,flush=True)
        done=len(list((ROOT/'completed').glob('*.json')))
        if done==100 and not active:
            subprocess.run(['python3',str(HERE/'summarize.py')],cwd=REPO,check=True)
            write(ROOT/'controller_done.json',dict(completed=True,pairs=100,episodes=400,time=time.time()))
            return
        unexplained=[f for f in failures if not f['resource_stop']]
        if unexplained:
            write(ROOT/'blocked.json',dict(reason='Worker failed; inspect logs before retrying',failures=unexplained))
            return
        ok,reasons=guard.can_launch(cold_gib=24,max_models=8)
        gate=read(ROOT/'smoke_passed.json',{}).get('passed',False)
        gpu=None
        if ok and done<100 and (gate or not active):
            counts={g:sum(e['gpu']==g for e in active) for g in range(4)}
            choices=[g for g in range(4) if counts[g]<2 and guard.gpu_headroom(g)>=23000]
            if choices:gpu=min(choices,key=lambda g:(counts[g],g))
        if gpu is not None:
            while any(r['worker']=='w%02d_gpu%d'%(index,gpu) for r in registry):index+=1
            registry.append(launch(gpu,index,smoke=not gate));index+=1
        write(ROOT/'status.json',dict(time=time.time(),pairs_complete=done,episodes_complete=len(list((ROOT/'episodes').glob('*.json'))),
            smoke_passed=gate,active=[dict(pid=e['pid'],worker=e['label'],gpu=e['gpu'],phase=e['phase']) for e in active],
            admission_reasons=reasons,failures=failures,workers=registry))
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
    main(parser.parse_args())
