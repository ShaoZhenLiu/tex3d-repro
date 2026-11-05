"""Remote peer supervision and SHA-verified result collection; no Windows dependency."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from common import HERE, NODES, PEERS, REPO, ROOT, jobs, sha
sys.path.insert(0, str(REPO/'ops'))
sys.path.insert(0, str(REPO/'ops/stability_seeds_20261105'))
import refcloth_resource as guard
import transfer_evaluation
from status import snapshot


def remote(node, command, timeout=45):
    body = base64.b64encode(('set -e\ncd '+str(REPO)+'\n'+command).encode()).decode()
    proc = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=6', '-o', 'ServerAliveInterval=7', '-o', 'ServerAliveCountMax=2',
        PEERS[node], 'echo '+body+' | base64 -d | bash'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1500:])
    return proc.stdout


def peer_status(node):
    return json.loads(remote(node, 'python3 ops/full_reference_20261105/status.py --node '+node))


def ensure_local(target):
    folder = ROOT/'targets'/target
    if guard.read(folder/'complete.json', {}).get('completed'):
        return
    process = guard.read(folder/'controller_process.json', {})
    if process and guard.process_runs_script(process['pid'], HERE/'control.py', process.get('ticks')):
        return
    failure = guard.read(folder/'controller_error.json', {})
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if failure.get('commit') == current_commit:
        raise RuntimeError('Controller needs inspection before retry: '+str(failure))
    log = (folder/'controller.log').open('a')
    child = subprocess.Popen(['python3', str(REPO/'ops/full_reference_20261105/control.py'), '--target', target],
        cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    guard.write(folder/'controller_process.json', dict(pid=child.pid, ticks=guard.identity(child.pid)['ticks'],
                target=target, started=time.time()))
    print('CONTROLLER_STARTED', target, child.pid, flush=True)


def ensure_peer(node, state, commit):
    if state['accepted'] or state['supervisor'] or (node == '09' and state['complete']):
        return
    cmd = ('test "$(git rev-parse HEAD)" = '+commit+'\n'
           'nohup setsid bash "$PWD/ops/full_reference_20261105/run.sh" '+node+' '+commit+
           ' >>"'+str(ROOT)+'/supervisor_'+node+'.log" 2>&1 </dev/null &\necho PEER_RESUMED')
    print(remote(node, cmd).strip(), flush=True)


def collect_09():
    target = ROOT/'targets/pi0'
    receipt = ROOT/'imported09.json'
    seen = set(guard.read(receipt, {}).get('jobs', []))
    if len(seen) == 160:
        return False
    encoded = base64.b64encode(json.dumps(sorted(seen)).encode()).decode()
    command = ('PYTHONPATH="$PWD/ops" python3 ops/stability_seeds_20261105/transfer_evaluation.py export --root '+
               str(target)+' --seen-base64 '+encoded)
    packet = json.loads(remote('09', command, timeout=90))
    if packet['pending']:
        return False
    staging = ROOT/'staging'; staging.mkdir(exist_ok=True)
    archive = staging/(packet['sha256']+'.tar.gz')
    if not archive.exists():
        origin = packet['archive']
        if not origin.startswith('/'):
            origin = str(REPO)+'/'+origin
        proc = subprocess.run(['scp', '-B', '-o', 'ConnectTimeout=6', '-o', 'ServerAliveInterval=10',
            PEERS['09']+':'+origin, str(archive.with_suffix('.part'))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode(errors='replace')[-1000:])
        assert sha(archive.with_suffix('.part')) == packet['sha256']
        archive.with_suffix('.part').replace(archive)
    imported = transfer_evaluation.install(target, archive, packet['sha256'])
    assert set(imported['imported']) == set(packet['jobs'])
    seen.update(imported['imported'])
    guard.write(receipt, dict(jobs=sorted(seen), count=len(seen), updated=time.time()))
    print('IMPORTED_09', len(seen), '/160', flush=True)
    if len(seen) == 160:
        assert len(list((target/'evaluation/episodes').glob('*.json'))) == 2800
        guard.write(target/'complete.json', dict(completed=True, target='pi0', episodes=2800,
                    new_episodes=2800, reused_episodes=0, imported_from='09'))
    return True


def analyze_and_archive():
    summary = ROOT/'analysis/summary.json'
    if not summary.exists():
        log = (ROOT/'analysis.log').open('a')
        subprocess.run(['/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python',
                        str(HERE/'summarize.py')], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True, timeout=300)
    assert guard.read(summary)['completed']
    archive = ROOT/'refcloth_full_reference_complete.tar.gz'
    if not archive.exists():
        temporary = archive.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as stream:
            for name in ['analysis', 'targets', 'protocols', 'selection.json', 'imported09.json',
                         'preparation4090.json']:
                stream.add(ROOT/name, arcname=name)
        temporary.replace(archive)
    guard.write(ROOT/'accepted.json', dict(completed=True, episodes=5600, new_episodes=4600,
        reused_episodes=1000, archive=archive.name, sha256=sha(archive), accepted_at=time.time()))
    print('ACCEPTED_FULL_BENCHMARK', flush=True)


def main(node, commit):
    target = next(t for t, n in NODES.items() if n == node)
    peer = '09' if node == '4090' else '4090'
    while True:
        try:
            if guard.read(ROOT/'accepted.json', {}).get('completed'):
                return
            ensure_local(target)
            state = peer_status(peer)
            if state['accepted']:
                guard.write(ROOT/'accepted_peer.json', dict(completed=True, peer=peer, time=time.time()))
                return
            ensure_peer(peer, state, commit)
            progressed = collect_09() if node == '4090' else False
            local = snapshot(node)
            guard.write(ROOT/('service_status_'+node+'.json'), dict(local=local, peer=state, updated=time.time()))
            if node == '4090' and all(guard.read(ROOT/'targets'/t/'complete.json', {}).get('completed') for t in NODES):
                analyze_and_archive()
                return
            time.sleep(1 if progressed else 30)
        except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as error:
            guard.write(ROOT/('last_error_'+node+'.json'), dict(type=type(error).__name__, detail=str(error)[-1500:], time=time.time()))
            print('SERVICE_RETRY', type(error).__name__, str(error)[-500:], flush=True)
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--node', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args(); main(args.node, args.commit)
