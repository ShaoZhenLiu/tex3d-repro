"""Publish completed textures and schedule persistent native evaluation workers."""
import json
import argparse
import hashlib
import os
import shutil
import subprocess
import time
import re
from pathlib import Path


R=Path('/data/shaozhen.liu/code/tex3d')
X=R/'experiments/refcloth_stability/20261102'
O=R/'ops/stability_20261102'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def publish():
    path=X/'texture_manifest.json';manifest=json.loads(path.read_text());changed=False
    for seed in [10,11,12]:
        for ref in ['native','mr']:
            name='s%d_%s'%(seed,ref);meta=X/'arms'/name/'completed.json'
            if name in manifest or not meta.exists():continue
            info=json.loads(meta.read_text());assert info['completed'] and info['iterations']==5000
            source=Path(info['texture']);dest=X/'frozen'/(name+'.png')
            if not source.exists():source=dest  # A100 results are transferred with their original provenance.
            if not source.exists():continue
            assert hashlib.sha256(source.read_bytes()).hexdigest()==info['texture_sha256']
            if source!=dest:shutil.copy2(source,dest)
            manifest[name]=dict(path=str(dest.relative_to(X)),sha256=info['texture_sha256'],
                seed=seed,reference=ref,training_metadata=str(meta.relative_to(X)),frozen_existing=False)
            changed=True;print('PUBLISHED',name,flush=True)
    if changed:write(path,manifest)
    for seed in [10,11,12]:
        pair=[X/'arms'/('s%d_%s'%(seed,ref))/'completed.json' for ref in ['native','mr']]
        if all(p.exists() for p in pair):
            left,right=[json.loads(p.read_text()) for p in pair]
            assert left['theta_init_sha256']==right['theta_init_sha256']
            assert left['sampling_log_sha256']==right['sampling_log_sha256']


def main(a):
    assert json.loads((X/'evaluation/workers/smoke7/smoke_passed.json').read_text())['passed']
    active={};launched=set();failed=[];last=0.
    if a.resume:
        prior=json.loads((X/'eval_controller_status.json').read_text())
        failed=prior['failed']
        class ExistingProcess:
            def __init__(self,pid,name):self.pid=pid;self.name=name
            def poll(self):
                p=Path('/proc/%d/status'%self.pid)
                if p.exists() and not any(row.startswith('State:') and 'Z' in row.split()[1] for row in p.read_text().splitlines()):return None
                return 0 if (X/'evaluation/workers'/self.name/'completed.json').exists() else 1
        for name,pid in prior['active'].items():
            active[name]=ExistingProcess(pid,name);launched.add(name)
    while True:
        publish()
        complete=len(list((X/'evaluation/completed').glob('*.json')))
        if complete==140:break
        source=json.loads((X/'train_status_4090.json').read_text())
        reserved={item['job'][2] for item in source['active'].values()}
        reserved.update(job[2] for job in source['pending'])
        compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        memory_by_pid={int(row.split(',')[0]):int(row.split(',')[1]) for row in compute_apps.splitlines() if row.split(',')[1].strip().isdigit()}
        free={int(row.split(',')[0]):int(row.split(',')[1]) for row in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        mem={row.split(':')[0]:int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
        ram=mem['MemAvailable']/1024**2
        coupling_path=R/'experiments/refcloth_coupling/20261102/train_status_4090.json'
        coupling=json.loads(coupling_path.read_text()) if coupling_path.exists() else None
        slots=[(7,'gpu7a',.44),(7,'gpu7b',.44),(0,'gpu0',.31),
               (1,'gpu1',.44),(5,'gpu5',.44),(6,'gpu6',.31),
               (0,'gpu0b',.31),(1,'gpu1b',.31),(5,'gpu5b',.31),(6,'gpu6b',.31)]
        for gpu,name,fraction in slots:
            remainder=0
            if gpu in reserved:
                if source.get('gradient_mode')!='texture_only':continue
                remainder=sum(max(0,23000-memory_by_pid.get(item['pid'],0)) for item in source['active'].values() if item['job'][2]==gpu)
                if any(job[2]==gpu for job in source['pending']):continue
            if coupling is not None:
                on_card=[v for v in coupling['active'].values() if v['gpu']==gpu]
                remainder+=sum(max(0,23000-memory_by_pid.get(v['pid'],0)) for v in on_card)
                if coupling['pending'] and not on_card and gpu not in reserved:
                    remainder+=23000
            # A loading worker may still occupy only a CUDA context while its
            # checkpoint is on CPU. Reserve its eventual full budget now.
            for other,proc in active.items():
                if int(re.match(r'gpu(\d+)',other).group(1))!=gpu:continue
                budget=next(17000 if f==.31 else 21900 for g,n,f in slots if n==other)
                remainder+=max(0,budget-memory_by_pid.get(proc.pid,0))
            # Full-trajectory low-memory smoke: 14,789 MB peak allocated,
            # 15,089 MB reserved; .31 cap passed clean and adversarial cases.
            # Include CUDA/MuJoCo context overhead and the separate source
            # reserve. A second worker on these cards waits for source exit.
            required=17000 if fraction==.31 else 21900
            if name in launched or free[gpu]-remainder<required or ram<50 or time.monotonic()-last<90:continue
            if fraction==.31:
                assert json.loads((X/'evaluation/workers/lowmem_smoke6/smoke_passed.json').read_text())['passed']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
                HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
                USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',TORCHDYNAMO_DISABLE='1',REFCLOTH_SURFACE_MODE='legacy',
                JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',MPLCONFIGDIR='/tmp/refcloth_stability_mpl',
                PYTHONPATH=':'.join([str(R),str(R/'pi'),str(O),'/data/shaozhen.liu/code/tex3d-deps/taming-transformers']))
            command=['/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python',str(O/'evaluate.py'),
                     '--repo',str(R),'--root',str(X),'--worker',name,'--memory-fraction',str(fraction)]
            proc=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,
                stdout=open(X/('evaluation_'+name+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;launched.add(name);last=time.monotonic()
            print('EVAL_LAUNCH',name,proc.pid,flush=True)
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if code!=0:failed.append(dict(worker=name,exit_code=code));print('EVAL_INFRASTRUCTURE_ERROR',name,code,flush=True)
            del active[name]
        write(X/'eval_controller_status.json',dict(completed_bundles=complete,total_bundles=140,
            active={str(g):p.pid for g,p in active.items()},failed=failed,updated=time.time()))
        if not active and failed:raise RuntimeError('All workers failed; inspect before retrying same registered jobs')
        time.sleep(15)
    write(X/'eval_controller_done.json',dict(completed=True,bundles=140,episodes=2600,failed_attempts=failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args())
