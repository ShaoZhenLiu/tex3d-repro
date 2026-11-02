"""Wait for healthy host, synchronize the reviewed code, then recover controllers."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import refcloth_resource as guard

R = Path('/data/shaozhen.liu/code/tex3d')


def git(*args):
    return subprocess.check_output(['git','-c','core.packedGitWindowSize=1m',
        '-c','core.packedGitLimit=8m']+list(args),cwd=R,universal_newlines=True,stderr=subprocess.STDOUT,timeout=90)


def main(a):
    guard.BASE.mkdir(exist_ok=True)
    lock=(guard.BASE/'bootstrap.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    expected=guard.read(a.manifest)
    assert expected['commit']==a.commit and expected['files']
    last_reason=None
    while True:
        allowed,reasons=guard.can_launch()
        guard.write(guard.BASE/'bootstrap.json',dict(waiting=not allowed,admission_wait=reasons,
                    required_commit=a.commit,time=time.time()))
        if not allowed:
            if reasons!=last_reason:print('BOOTSTRAP_WAIT',','.join(reasons),flush=True);last_reason=reasons
            time.sleep(10);continue
        for relative,digest in expected['files'].items():
            assert hashlib.sha256((R/relative).read_bytes()).hexdigest()==digest,relative
        try:
            if git('rev-parse','HEAD').strip()!=a.commit:
                print(git('fetch','lab','reference-stability-20261102'),flush=True)
                # Only our reviewed files may enter the index. Their bytes must
                # equal the incoming commit before a fast-forward is attempted.
                for relative,digest in expected['files'].items():
                    data=subprocess.check_output(['git','show',a.commit+':'+relative],cwd=R)
                    assert hashlib.sha256(data).hexdigest()==digest,relative
                git('add','--',*sorted(expected['files']))
                staged=set(git('diff','--cached','--name-only').splitlines())
                assert staged<=set(expected['files']),staged-set(expected['files'])
                print(git('merge','--ff-only',a.commit),flush=True)
            assert git('rev-parse','HEAD').strip()==a.commit
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            print('SYNC_RETRY',str(error)[-1000:],flush=True)
            time.sleep(30);continue
        # Recheck admission after Git/SSH, before any controller may load models.
        if not guard.can_launch()[0]:continue
        launched=[]
        commands=[('stability','train_controller.py'),('coupling','train_controller.py'),
                  ('stability','eval_controller.py'),('coupling','eval_controller.py')]
        for study,script in commands:
            path=R/'ops'/('%s_20261102'%study)/script
            cmd=[sys.executable,str(path),'--resume']
            if script=='train_controller.py':cmd+=['--node',a.node]
            log=open(guard.BASE/('%s_%s.log'%(study,script)),'ab',buffering=0)
            proc=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launched.append(dict(study=study,script=script,pid=proc.pid))
        guard.write(guard.BASE/'bootstrap.json',dict(waiting=False,started=True,commit=a.commit,
                    controllers=launched,time=time.time()))
        print('CONTROLLERS_RECOVERED',json.dumps(launched),flush=True)
        return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090'],required=True)
    p.add_argument('--commit',required=True);p.add_argument('--manifest',type=Path,required=True)
    main(p.parse_args())
