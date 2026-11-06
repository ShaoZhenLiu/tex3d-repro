"""Remote peer supervision and SHA-verified result collection; no Windows dependency."""
import argparse
import base64
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from common import HERE, LONG_ROOT, NODES, PEERS, REPO, ROOT, jobs, sha, long_jobs, migration_enabled
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
    if target == 'pi05' and migration_enabled():
        return
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


def ensure_long_local():
    if guard.read(LONG_ROOT/'complete.json', {}).get('completed'):
        return
    process = guard.read(LONG_ROOT/'controller_process.json', {})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    workers = [e for e in guard.active() if e['root'] == str(LONG_ROOT)]
    if workers and all(guard.progress_age(e, time.time()) > 900 for e in workers):
        guard.write(LONG_ROOT/'controller_error.json', dict(type='NoEvaluationProgress',
            detail='All workers have had no output for 15 minutes; inspect renderer before retry',
            commit=commit, time=time.time()))
        if process and guard.process_runs_script(process['pid'], HERE/'long_control.py', process.get('ticks')):
            os.kill(process['pid'], signal.SIGTERM)
        with guard.locked():
            stops = guard.read(guard.BASE/'stops.json', {})
            for entry in workers:
                stops[entry['token']] = dict(entry, time=time.time(), reason='no_evaluation_progress')
            guard.write(guard.BASE/'stops.json', stops)
        for entry in workers:
            guard.stop_entry(entry, signal.SIGTERM)
        raise RuntimeError('Long evaluation has stopped making progress; paused for inspection')
    if process and guard.process_runs_script(process['pid'], HERE/'long_control.py', process.get('ticks')):
        return
    failure = guard.read(LONG_ROOT/'controller_error.json', {})
    if failure.get('commit') == commit:
        raise RuntimeError('Long controller needs inspection before retry: '+str(failure))
    log = (LONG_ROOT/'controller.log').open('a')
    child = subprocess.Popen(['python3', str(HERE/'long_control.py')], cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    guard.write(LONG_ROOT/'controller_process.json', dict(pid=child.pid,
                ticks=guard.identity(child.pid)['ticks'], started=time.time()))
    print('LONG_CONTROLLER_STARTED', child.pid, flush=True)


def ensure_peer(node, state, commit):
    if state['accepted'] or state['supervisor'] or (node == '09' and state['work_complete']):
        return
    cmd = ('test "$(git rev-parse HEAD)" = '+commit+'\n'
           'nohup setsid bash "$PWD/ops/full_reference_20261105/run.sh" '+node+' '+commit+
           ' >>"'+str(ROOT)+'/supervisor_'+node+'.log" 2>&1 </dev/null &\necho PEER_RESUMED')
    print(remote(node, cmd).strip(), flush=True)


def collect_09(long=False):
    target = LONG_ROOT if long else ROOT/'targets/pi0'
    receipt = ROOT/('imported09_long.json' if long else 'imported09.json')
    total, episodes = (40, 700) if long else (160, 2800)
    seen = set(guard.read(receipt, {}).get('jobs', []))
    if len(seen) == total:
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
    print('IMPORTED_09', 'pi05_long' if long else 'pi0', len(seen), '/', total, flush=True)
    if len(seen) == total:
        assert len(list((target/'evaluation/episodes').glob('*.json'))) == episodes
        guard.write(target/'complete.json', dict(completed=True, target='pi05' if long else 'pi0', episodes=episodes,
                    new_episodes=episodes, reused_episodes=0, imported_from='09'))
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
            names = ['analysis', 'targets', 'protocols', 'selection.json', 'imported09.json', 'preparation4090.json']
            if migration_enabled():
                names += ['segments', 'long_migration_20261106.json', 'imported09_long.json',
                          'preserved_long_41_manifest.json']
            for name in names:
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
            if node == '09' and migration_enabled():
                ensure_long_local()
            state = peer_status(peer)
            if state['accepted']:
                guard.write(ROOT/'accepted_peer.json', dict(completed=True, peer=peer, time=time.time()))
                return
            ensure_peer(peer, state, commit)
            progressed = collect_09() if node == '4090' else False
            if node == '4090' and migration_enabled():
                progressed = collect_09(long=True) or progressed
                if guard.read(LONG_ROOT/'complete.json', {}).get('completed'):
                    guard.write(ROOT/'targets/pi05/complete.json', dict(completed=True, target='pi05',
                        episodes=2800, new_episodes=1800, reused_episodes=1000, long_source=str(LONG_ROOT),
                        evaluation_nodes={'spatial':'4090','object':'4090','goal':'4090','long':'09'}))
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
