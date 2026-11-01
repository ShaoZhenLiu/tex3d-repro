"""Own the target asset patch and supervise the two exploratory suite workers."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from controller import preflight


def main(repo,out):
    source=repo/'experiments/refcloth_support/20261101/screen'
    op=repo/'ops/support_20261101'
    protocol=op/'pilot_protocol.json'
    p=json.loads(protocol.read_text())
    out.mkdir(parents=True,exist_ok=False)
    audit=preflight(p['gpus'],34000)
    audit['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    audit['utc_start']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    py='/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python'
    def env(gpu):
        e=os.environ.copy()
        e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
            PYTHONUNBUFFERED='1',OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi',
            MPLCONFIGDIR='/tmp/refcloth_support_mpl',TORCHDYNAMO_DISABLE='1',
            PYTHONPATH=':'.join(map(str,[repo,repo/'pi',op,Path('/data/shaozhen.liu/code/tex3d-deps/taming-transformers')])))
        return e
    patch="import attack_pi0_cloth as A,json; s=A.patch_status(); A.ensure_patch(True); print('PATCH_OWNER_JSON='+json.dumps({'initial':s,'current':A.patch_status()}))"
    result=subprocess.check_output([py,'-c',patch],cwd=repo,env=env(p['gpus'][0]),text=True)
    owner=json.loads(next(x.split('=',1)[1] for x in result.splitlines() if x.startswith('PATCH_OWNER_JSON=')))
    audit['patch_owner']=owner
    procs=[];logs=[]
    try:
        for suite,gpu in zip(p['suites'],p['gpus']):
            log=open(out/(suite+'.log'),'w');logs.append(log)
            args=[py,str(op/'target_pilot.py'),'--repo',str(repo),'--source',str(source),
                  '--out',str(out/suite),'--suite',suite,'--protocol',str(protocol)]
            proc=subprocess.Popen(args,cwd=repo,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL)
            procs.append(proc)
        audit['workers']=[dict(pid=x.pid,gpu=g,suite=s) for x,g,s in zip(procs,p['gpus'],p['suites'])]
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        while any(x.poll() is None for x in procs):
            bad=[x.returncode for x in procs if x.poll() not in (None,0)]
            if bad: raise RuntimeError('Target pilot worker failed: %s'%bad)
            time.sleep(2)
        assert all(x.returncode==0 for x in procs)
        (out/'completed.json').write_text(json.dumps(dict(completed=True,utc_end=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
        for log in logs: log.close()
        if owner['initial']=='clean':
            subprocess.check_call([py,'-c','import attack_pi0_cloth as A; A.ensure_patch(False)'],cwd=repo,env=env(p['gpus'][0]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();main(a.repo,a.out)
