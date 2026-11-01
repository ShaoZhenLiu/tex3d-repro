"""Resume the same fixed 72 episodes, distributing task shards over 3 GPUs."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from controller import preflight


def main(repo,out):
    op=repo/'ops/support_20261101'
    base=repo/'experiments/refcloth_support/20261101'
    source=base/'screen'
    protocol=op/'pilot_protocol.json';p=json.loads(protocol.read_text())
    out.mkdir(parents=True,exist_ok=False)
    gpus=[1,6,7]
    audit=preflight(gpus,28000)
    audit.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                 reason='Scheduling change only: preserve completed episodes; no masks, noise keys, cases or conditions changed',
                 utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    py='/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python'
    def env(gpu):
        e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
            PYTHONUNBUFFERED='1',OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',
            MPLCONFIGDIR='/tmp/refcloth_support_mpl',TORCHDYNAMO_DISABLE='1',
            PYTHONPATH=':'.join(map(str,[repo,repo/'pi',op,Path('/data/shaozhen.liu/code/tex3d-deps/taming-transformers')])))
        return e
    # Queue longest remaining Object shards first. Exactly the same case set.
    jobs=[];oldfiles={}
    for suite in ['libero_object','libero_spatial']:
        old=base/'pilot'/suite/'results.resume.json'
        oldfiles[suite]=old
        previous=json.loads(old.read_text())['records'] if old.exists() else []
        for task in p['tasks']:
            done=sum(r['task']==task for r in previous)
            if done<len(p['initials'])*len(p['conditions']):
                jobs.append(dict(suite=suite,task=task,remaining=12-done,resume=str(old)))
    jobs.sort(key=lambda x:(x['suite']!='libero_object',-x['remaining'],x['task']))
    audit['jobs']=jobs.copy();audit['workers']=[]
    running={};logs=[]
    try:
        while jobs or running:
            for gpu in gpus:
                if gpu in running or not jobs:continue
                job=jobs.pop(0);name='%s_t%d'%(job['suite'],job['task'])
                log=open(out/(name+'.log'),'w');logs.append(log)
                args=[py,str(op/'target_pilot.py'),'--repo',str(repo),'--source',str(source),
                    '--out',str(out/name),'--suite',job['suite'],'--protocol',str(protocol),
                    '--tasks',str(job['task']),'--resume-file',job['resume']]
                proc=subprocess.Popen(args,cwd=repo,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL)
                running[gpu]=(proc,name)
                audit['workers'].append(dict(pid=proc.pid,gpu=gpu,name=name))
                (out/'audit.json').write_text(json.dumps(audit,indent=2))
                print('START',gpu,name,flush=True)
            for gpu,(proc,name) in list(running.items()):
                if proc.poll() is None:continue
                assert proc.returncode==0, (name,proc.returncode)
                print('DONE',gpu,name,flush=True);del running[gpu]
            time.sleep(1)
        records={}
        for suite,old in oldfiles.items():
            if old.exists():
                for r in json.loads(old.read_text())['records']:
                    records[(suite,r['task'],r['initial'],r['condition'])]=dict(r,suite=suite,execution='initial')
        for file in out.glob('*/results.json'):
            data=json.loads(file.read_text());suite=data['suite']
            for r in data['records']:
                key=(suite,r['task'],r['initial'],r['condition'])
                if key in records:
                    assert records[key]['robot_success']==r['robot_success']
                    assert records[key]['noise_sha256']==r['noise_sha256']
                else:records[key]=dict(r,suite=suite,execution=file.parent.name)
        expected={(s,t,i,c) for s in p['suites'] for t in p['tasks'] for i in p['initials'] for c in p['conditions']}
        assert set(records)==expected, (len(records),len(expected))
        report=dict(completed=True,protocol=p,records=list(records.values()),
                    utc_end=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (out/'combined.json').write_text(json.dumps(report,indent=2))
    finally:
        for proc,name in running.values():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();main(a.repo,a.out)
