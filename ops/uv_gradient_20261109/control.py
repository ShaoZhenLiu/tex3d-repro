"""Resource-aware scheduling with exact-arm checkpoint recovery and no duplicate worker."""
import fcntl
import os
import subprocess
import sys
import time

from common import OPS, PYTHON, REPO, ROOT, env_for, read, write

sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    owner=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    jobs=[('heldout',None,None)]+[('train',seed,uv) for seed in [7,8,9] for uv in ['on','off']]
    active={};attempts={};finished=[];procs={}
    for entry in guard.active():
        if entry['root']!=str(ROOT):continue
        name=entry['label'].split(':',1)[-1]
        active[name]=dict(pid=entry['pid'],ticks=entry['ticks'],gpu=entry['gpu'],job=None,
            start=entry.get('started',time.time()),token=entry['token'] if 'token' in entry else None)
    last=read(ROOT/'status.json',{})
    for name,item in active.items():
        previous=last.get('active',{}).get(name)
        assert previous and previous['pid']==item['pid'],('Live worker without job ledger',name)
        item['job']=previous['job']
    while True:
        for name,item in list(active.items()):
            identity=guard.identity(item['pid'])
            if identity and identity['ticks']==item['ticks']:continue
            result=read(ROOT/'prepared/heldout.json' if name=='heldout' else ROOT/'arms'/name/'completed.json',{})
            proc=procs.pop(name,None);code=proc.poll() if proc else None
            if result.get('completed'):
                print('FINISHED',name,flush=True)
            else:
                attempts[name]=attempts.get(name,0)+1
                write(ROOT/'last_errors'/('%s_%d.json'%(name,int(time.time()))),
                    dict(job=item['job'],pid=item['pid'],returncode=code,time=time.time()))
                print('REQUEUE',name,'attempt',attempts[name],flush=True)
            del active[name]
        pending=[];finished=[];failed=[]
        for kind,seed,uv in jobs:
            name='heldout' if kind=='heldout' else 's%d_%s'%(seed,uv)
            completed=read(ROOT/'prepared/heldout.json' if kind=='heldout' else ROOT/'arms'/name/'completed.json',{})
            if completed.get('completed'):finished.append(name)
            elif name not in active:
                if attempts.get(name,0)>=3:failed.append(name)
                else:pending.append((kind,seed,uv))
        reasons=[]
        if pending:
            allowed,reasons=guard.can_launch(cold_gib=12,max_models=6)
            if allowed:
                lock=(REPO/'experiments/refcloth_gpu_budget_4090.lock').open('a')
                with lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    counts={gpu:sum(item['gpu']==gpu for item in active.values()) for gpu in [0,1,2,3]}
                    choices=[gpu for gpu in [0,1,2,3] if counts[gpu]<2 and guard.gpu_headroom(gpu)>=23000]
                    choices.sort(key=lambda gpu:(counts[gpu],-guard.gpu_headroom(gpu),gpu))
                    if choices:
                        kind,seed,uv=pending[0];gpu=choices[0]
                        name='heldout' if kind=='heldout' else 's%d_%s'%(seed,uv)
                        script=OPS/('prepare.py' if kind=='heldout' else 'train.py')
                        cmd=[PYTHON,str(script)]+(['heldout'] if kind=='heldout' else ['--seed',str(seed),'--uv',uv])
                        log=ROOT/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
                        proc=subprocess.Popen(cmd,cwd='/data/shaozhen.liu/code/tex3d-ws/g7',env=env_for(gpu),
                            stdin=subprocess.DEVNULL,stdout=log.open('ab',buffering=0),stderr=subprocess.STDOUT,
                            start_new_session=True)
                        deadline=time.monotonic()+10
                        while True:
                            try:
                                token=guard.register(proc.pid,script,'source',gpu,23000,'uvgradient:'+name,ROOT)
                                break
                            except AssertionError:
                                if proc.poll() is not None or time.monotonic()>deadline:raise
                                time.sleep(.05)
                        active[name]=dict(pid=proc.pid,ticks=guard.identity(proc.pid)['ticks'],gpu=gpu,
                            job=[kind,seed,uv],start=time.time(),token=token)
                        procs[name]=proc
                        print('LAUNCH',name,proc.pid,'GPU',gpu,flush=True)
                    else:reasons.append('no_gpu_with_23000MiB_headroom')
        write(ROOT/'status.json',dict(active=active,pending=pending,finished=finished,failed=failed,
            retries=attempts,admission_wait=reasons,time=time.time()))
        if not active and not pending:
            write(ROOT/'training_done.json',dict(completed=not failed,finished=finished,failed=failed,time=time.time()))
            if not failed:
                subprocess.run([PYTHON,str(OPS/'summarize.py')],cwd=REPO,env=env_for(0),check=True)
            return
        time.sleep(10)


if __name__=='__main__':main()
