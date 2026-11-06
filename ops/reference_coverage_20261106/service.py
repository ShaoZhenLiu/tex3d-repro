"""Peer recovery, immutable bundle collection, and full-result acceptance."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from common import HERE, REPO, ROOT, NODES, PEERS, jobs, suite_root, sha
sys.path.insert(0,str(REPO/'ops'))
sys.path.insert(0,str(REPO/'ops/stability_seeds_20261105'))
import refcloth_resource as guard
import transfer_evaluation
from status import snapshot


def remote(node,command,timeout=45):
    payload=base64.b64encode(('set -e\ncd '+str(REPO)+'\n'+command).encode()).decode()
    p=subprocess.run(['ssh','-n','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=7',
        '-o','ServerAliveInterval=7','-o','ServerAliveCountMax=2',PEERS[node],
        'echo '+payload+' | base64 -d | bash'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-1800:])
    return p.stdout


def recover_load_oom(root):
    path=root/'controller_error.json';failure=guard.read(path,{})
    if not failure:return
    worker=failure.get('detail','').rsplit(': ',1)[-1]
    if '/' not in worker and worker.startswith('gpu'):
        log=root/('evaluation_'+worker+'.log')
        if log.exists() and not (root/'evaluation/workers'/worker/'registered.json').exists():
            if 'torch.OutOfMemoryError: CUDA out of memory.' in log.read_text(errors='replace'):
                history=guard.read(root/'load_recoveries.json',[])
                if sum(time.time()-x['time']<3600 for x in history)<3:
                    folder=root/'recovery';folder.mkdir(exist_ok=True)
                    out=folder/('controller_error_'+sha(path)+'.json')
                    if out.exists():assert out.read_bytes()==path.read_bytes()
                    else:out.write_bytes(path.read_bytes())
                    history.append(dict(time=time.time(),worker=worker,error=str(out)))
                    guard.write(root/'load_recoveries.json',history);path.unlink()
                    print('RECOVERED_SHARED_GPU_LOAD_OOM',worker,flush=True);return
    raise RuntimeError('Controller requires inspection: '+str(failure))


def ensure_local(suite):
    root=suite_root(suite)
    if guard.read(root/'complete.json',{}).get('completed'):return
    workers=[e for e in guard.active() if e['root']==str(root)]
    process=guard.read(root/'controller_process.json',{})
    alive=process and guard.process_runs_script(process['pid'],HERE/'control.py',process.get('ticks'))
    if workers and all(guard.progress_age(e,time.time())>900 for e in workers):
        failure=dict(type='NoEvaluationProgress',detail='All model outputs stale for fifteen minutes',time=time.time())
        guard.write(root/'controller_error.json',failure)
        if alive:os.kill(process['pid'],signal.SIGTERM)
        with guard.locked():
            stops=guard.read(guard.BASE/'stops.json',{})
            for e in workers:stops[e['token']]=dict(e,time=time.time(),reason='no_evaluation_progress')
            guard.write(guard.BASE/'stops.json',stops)
        for e in workers:guard.stop_entry(e,signal.SIGTERM)
        raise RuntimeError(str(failure))
    if alive:return
    recover_load_oom(root)
    log=(root/'controller.log').open('a')
    child=subprocess.Popen(['python3','-u',str(HERE/'control.py'),'--suite',suite],cwd=REPO,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    guard.write(root/'controller_process.json',dict(pid=child.pid,ticks=guard.identity(child.pid)['ticks'],started=time.time(),suite=suite))
    print('CONTROLLER_STARTED',suite,child.pid,flush=True)


def collect_long():
    root=suite_root('long');receipt=ROOT/'imported09.json'
    seen=set(guard.read(receipt,{}).get('jobs',[]))
    if not seen:
        seen={j['id'] for j in jobs('long') if (root/'evaluation/completed'/(j['id']+'.json')).exists()}
    if len(seen)==190:return False
    encoded=base64.b64encode(json.dumps(sorted(seen)).encode()).decode()
    packet=json.loads(remote('09','PYTHONPATH="$PWD/ops" python3 ops/stability_seeds_20261105/transfer_evaluation.py export --root '+str(root)+' --seen-base64 '+encoded,timeout=90))
    if packet['pending']:return False
    folder=ROOT/'staging';folder.mkdir(exist_ok=True);archive=folder/(packet['sha256']+'.tar.gz')
    if not archive.exists():
        origin=packet['archive'] if packet['archive'].startswith('/') else str(REPO)+'/'+packet['archive']
        subprocess.run(['scp','-B','-o','ConnectTimeout=7',PEERS['09']+':'+origin,str(archive.with_suffix('.part'))],
                       stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=120)
        assert sha(archive.with_suffix('.part'))==packet['sha256'];archive.with_suffix('.part').replace(archive)
    imported=transfer_evaluation.install(root,archive,packet['sha256'])
    assert set(imported['imported'])==set(packet['jobs']);seen.update(imported['imported'])
    guard.write(receipt,dict(jobs=sorted(seen),count=len(seen),updated=time.time()))
    print('IMPORTED_LONG',len(seen),'/190',flush=True)
    if len(seen)==190:
        assert len(list((root/'evaluation/episodes').glob('*.json')))==3700
        guard.write(root/'complete.json',dict(completed=True,episodes=3700,new_episodes=3000,reused_episodes=700,imported_from='09'))
    return True


def accept():
    log=(ROOT/'analysis.log').open('a')
    subprocess.run(['python3',str(HERE/'summarize.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,
                   stderr=subprocess.STDOUT,check=True,timeout=600)
    assert guard.read(ROOT/'analysis/summary.json')['completed']
    archive=ROOT/'reference_coverage_complete.tar.gz'
    if not archive.exists():
        with tarfile.open(str(archive.with_suffix('.tmp')),'w:gz') as stream:
            for name in ['suites','analysis','inputs','plan.json','existing_spatial_object_outcomes.csv','existing_sources.json','imported09.json']:
                stream.add(str(ROOT/name),arcname=name)
        archive.with_suffix('.tmp').replace(archive)
    guard.write(ROOT/'accepted.json',dict(completed=True,extension_episodes=7400,four_suite_episodes=14800,
        new_episodes=6000,reused_extension_episodes=1400,archive=archive.name,sha256=sha(archive),accepted_at=time.time()))


def main(node,commit):
    suite=next(s for s,n in NODES.items() if n==node);peer='09' if node=='4090' else '4090'
    while True:
        try:
            if guard.read(ROOT/'accepted.json',{}).get('completed'):return
            ensure_local(suite)
            state=json.loads(remote(peer,'PYTHONPATH="$PWD/ops" python3 ops/reference_coverage_20261106/status.py --node '+peer))
            if state['accepted']:
                guard.write(ROOT/'accepted_peer.json',dict(completed=True,time=time.time()));return
            if not state['supervisor'] and not (peer=='09' and state['complete']):
                command='test "$(git rev-parse HEAD)" = '+commit+'\nnohup setsid bash "$PWD/ops/reference_coverage_20261106/run.sh" '+peer+' '+commit+' >>"'+str(ROOT)+'/supervisor_'+peer+'.log" 2>&1 </dev/null &\necho PEER_RESUMED'
                print(remote(peer,command).strip(),flush=True)
            progressed=collect_long() if node=='4090' else False
            guard.write(ROOT/('service_status_'+node+'.json'),dict(local=snapshot(node),peer=state,updated=time.time()))
            if node=='4090' and all(guard.read(suite_root(s)/'complete.json',{}).get('completed') for s in NODES):
                accept();return
            time.sleep(1 if progressed else 30)
        except (OSError,ValueError,KeyError,RuntimeError,AssertionError,subprocess.SubprocessError) as e:
            guard.write(ROOT/('last_error_'+node+'.json'),dict(type=type(e).__name__,detail=str(e)[-1800:],time=time.time()))
            print('SERVICE_RETRY',type(e).__name__,str(e)[-400:],flush=True);time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();main(a.node,a.commit)
