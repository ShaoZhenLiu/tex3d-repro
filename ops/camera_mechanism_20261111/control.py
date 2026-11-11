"""Admit the replay gate, then dispatch the frozen Object comparisons."""
import argparse
import fcntl
import os
import shutil
import subprocess
import time
from types import SimpleNamespace

from common import GPUS, HERE, INITIALS, PRIOR, PROTOCOL, REPO, ROOT, SEEDS, STUDY, TASKS
from common import controls, episode_name, guard, local_node, new_conditions, sha
import refcloth_eval_scheduler as scheduler


def available(gpu):
    try:
        return guard.gpu_headroom(gpu)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return -1


def run_pilot(node, gpus, budget):
    proof = ROOT / 'camera_preflight.json'
    with (REPO / f'experiments/refcloth_gpu_budget_{node}.lock').open('a') as lock:
        while not proof.exists():
            process = None
            with lock_guard(lock):
                ready, reasons = guard.can_launch(max_models=6 if node == '09' else 5)
                if any(e['label'].startswith(STUDY + ':preflight_') for e in guard.active()):
                    ready = False
                    reasons.append('camera_preflight_running')
                free = [(available(gpu), gpu) for gpu in gpus] if ready else []
                eligible = [entry for entry in free if entry[0] >= budget]
                if eligible:
                    _, gpu = min(eligible)
                    name = f'preflight_{node}_gpu{gpu}_{int(time.time())}'
                    command, env = scheduler.worker_spec(ROOT, node, gpu, name, PROTOCOL, node == '09')
                    with (ROOT / ('evaluation_' + name + '.log')).open('a') as log:
                        process = subprocess.Popen(command + ['--camera-preflight'], cwd=REPO, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                    scheduler.register_started_worker(process, gpu, budget, STUDY + ':' + name, ROOT)
                    ticks = guard.identity(process.pid)['ticks']
                    print('PREFLIGHT_LAUNCH', name, process.pid, flush=True)
                else:
                    guard.write(ROOT / 'preflight_wait.json', dict(reasons=reasons or ['no_eligible_gpu'],
                                                                  updated=time.time()))
            if process is None:
                time.sleep(15)
                continue
            code = process.wait()
            if code != 0 and not guard.resource_stopped(process.pid, ticks):
                raise RuntimeError(f'Camera preflight exited {code}. See evaluation_{name}.log')
    result = guard.read(proof)
    assert result['passed'] and result['protocol_sha256'] == sha(PROTOCOL)
    return result


class lock_guard:
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def __exit__(self, *_):
        fcntl.flock(self.handle, fcntl.LOCK_UN)


def publish_jobs(node, preflight):
    seeds = SEEDS[node]
    reuse = preflight['can_reuse_controls']
    manifest = guard.read(ROOT / 'texture_manifest.json')
    reused = {}
    jobs = []
    existing = ROOT / 'evaluation' / 'episodes'
    existing.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for condition in controls(seeds):
            if not reuse:
                continue
            for initial in INITIALS:
                name = episode_name(task, initial, condition)
                old = PRIOR / 'evaluation' / 'episodes' / name
                record = guard.read(old)
                assert record['completed'] and record['suite'] == 'object' and record['condition'] == condition
                assert (record['task'], record['initial']) == (task, initial)
                assert record['texture_sha256'] == manifest[condition]['sha256']
                assert record['host'] == os.uname().nodename
                digest = sha(old)
                dest = existing / name
                if not dest.exists():
                    temporary = dest.with_suffix('.reused.tmp')
                    shutil.copyfile(old, temporary)
                    temporary.replace(dest)
                assert sha(dest) == digest
                reused[name] = dict(source=str(old), sha256=digest)
        if not reuse:
            jobs.append(dict(id=f'object_t{task}_plain', suite='object', task=task, node=node,
                             conditions=['plain']))
        for seed in seeds:
            conditions = new_conditions(seed)
            if not reuse:
                conditions += [f's{seed}_main', f's{seed}_wrist']
            jobs.append(dict(id=f'object_t{task}_s{seed}_camera', suite='object', task=task, node=node,
                             conditions=conditions,
                             requires_existing=[episode_name(task, i, 'plain') for i in INITIALS]))
    fresh = sum(10 * len(job['conditions']) for job in jobs)
    assert fresh + len(reused) == 1100
    plan = dict(node=node, seeds=seeds, fresh=fresh, reused=len(reused), protocol_sha256=sha(PROTOCOL),
                preflight_sha256=sha(ROOT / 'camera_preflight.json'))
    old_plan = guard.read(ROOT / 'eval_plan.json')
    if old_plan is not None:
        assert old_plan == dict(plan, seeds=list(seeds))
    guard.write(ROOT / 'reused_rollouts.json', reused)
    guard.write(ROOT / 'evaluation' / 'jobs.json', jobs)
    guard.write(ROOT / 'eval_plan.json', plan)
    print('CAMERA_QUEUE_READY', node, fresh, len(reused), flush=True)


def main(a):
    assert a.node == local_node()
    assert set(a.gpus) <= set(GPUS[a.node]) and a.gpus
    prepared = guard.read(ROOT / 'prepared.json', {})
    assert prepared.get('protocol_sha256') == sha(PROTOCOL)
    with (ROOT / 'controller.lock').open('a') as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        budget = 18000 if a.node == '09' else 17000
        preflight = run_pilot(a.node, a.gpus, budget)
        publish_jobs(a.node, preflight)
        scheduler.main(STUDY, lambda: None, SimpleNamespace(gpus=a.gpus, share_gpu3=False, resume=True),
                       root=ROOT, node=a.node, allow_stochastic_calibration=a.node == '09',
                       evaluation_protocol=PROTOCOL, reused_episode_count=0,
                       memory_budget_mib=budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=SEEDS, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    main(parser.parse_args())
