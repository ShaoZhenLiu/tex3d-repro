"""Finish the result archive after a controller that predates the summarizer exits."""
import fcntl
import subprocess
import time

from common import OPS,PYTHON,REPO,ROOT,env_for,read

ROOT.mkdir(parents=True,exist_ok=True)
owner=(ROOT/'finish.lock').open('a')
try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
while not read(ROOT/'accepted.json',{}).get('completed'):
    status=read(ROOT/'training_done.json',{})
    if status.get('completed'):
        subprocess.run([PYTHON,str(OPS/'summarize.py')],cwd=REPO,env=env_for(0),check=True)
        break
    if status.get('failed'):raise RuntimeError('Training stopped with unresolved infrastructure failures')
    time.sleep(20)
