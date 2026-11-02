"""One-at-a-time cold loads, bounded concurrency and resource-safe target recovery."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import refcloth_resource as guard

R = Path('/data/shaozhen.liu/code/tex3d')
P = R / 'experiments/refcloth_stability/20261102'
C = R / 'experiments/refcloth_coupling/20261102'
SCRIPT = R / 'ops/stability_20261102/evaluate.py'


def claim_alive(path):
    claim = guard.read(path)
    if not claim:
        return False
    current = guard.identity(claim['pid'])
    return bool(current and current['uid'] == os.getuid() and str(SCRIPT) in current['cmd']
                and (claim.get('ticks') is None or current['ticks'] == claim['ticks']))


def main(study, publish, a):
    coupling = study == 'coupling'
    if a.share_gpu3 and (not coupling or not os.uname().nodename.endswith('-41')):
        raise ValueError('The shared GPU 3 option is for camera evaluation on host 41')
    x = C if coupling else P
    total = 180 if coupling else 140
    assert guard.read(P / 'evaluation/workers/lowmem_smoke6/smoke_passed.json')['passed']
    singleton = guard.BASE / (study + '_eval.lock'); singleton.parent.mkdir(exist_ok=True)
    owner = singleton.open('a')
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('An evaluation controller for this study is already running')
    old = guard.read(x / 'eval_controller_status.json', {}) if a.resume else {}
    failed = old.get('failed', []); holds = old.get('resource_pauses', []); actors = {}
    for name, pid in old.get('active', {}).items():
        current = guard.identity(pid)
        if not (current and current['uid'] == os.getuid() and str(SCRIPT) in current['cmd']
                and str(x) in current['cmd'] and '--worker ' + name in current['cmd']):
            continue
        gpu = int(name.split('gpu')[1].split('_')[0][0])
        budget = 19000 if gpu == 3 else 17000
        token = guard.register(pid, SCRIPT, 'eval', gpu, budget, study + ':' + name, x, phase='warm')
        actors[name] = dict(pid=pid, ticks=current['ticks'], proc=None, token=token, gpu=gpu)
    budget_lock = (R / 'experiments/refcloth_gpu_budget_4090.lock').open('a')
    jobs = guard.read(x / 'evaluation/jobs.json')
    while True:
        publish()
        completed = len(list((x / 'evaluation/completed').glob('*.json')))
        if completed == total:
            break
        manifest = guard.read(x / 'texture_manifest.json')
        ready = [j for j in jobs if not (x / 'evaluation/completed' / (j['id'] + '.json')).exists()
                 and all(c in manifest for c in j['conditions'])
                 and all((x / 'evaluation/episodes' / n).exists() for n in j.get('requires_existing', []))
                 and not claim_alive(x / 'evaluation/claims' / (j['id'] + '.json'))]
        for name, entry in list(actors.items()):
            current = guard.identity(entry['pid'])
            if current and current['ticks'] == entry['ticks']:
                continue
            code = entry['proc'].poll() if entry['proc'] else None
            if guard.resource_stopped(entry['pid'], entry['ticks']):
                holds.append(dict(worker=name, pid=entry['pid'], time=time.time()))
                print('EVAL_RESOURCE_REQUEUED', name, flush=True)
            elif code not in [0, None]:
                failed.append(dict(worker=name, exit_code=code))
                raise RuntimeError('Evaluation infrastructure failed; inspect before retry: ' + name)
            del actors[name]
        fcntl.flock(budget_lock, fcntl.LOCK_EX)
        try:
            allowed, reasons = guard.can_launch(max_models=6 if a.share_gpu3 else None)
            source_pending = any(guard.read(root / 'train_status_4090.json', {}).get('pending') for root in [P, C])
            loading = sum(e['kind'] == 'eval' and e['phase'] == 'loading' and e['root'] == str(x) for e in guard.active())
            if ready and len(ready) > loading and allowed:
                entries = guard.active()
                normal_workers = sum(e['gpu'] != 3 for e in entries)
                gpus = [0, 1, 5, 6, 7] if normal_workers < 5 else []
                if a.share_gpu3 and not any(e['gpu'] == 3 for e in entries):
                    gpus.append(3)
                candidates = []
                for gpu in gpus:
                    source_here = any(e['kind'] == 'source' and e['gpu'] == gpu for e in entries)
                    reserve = 23000 if gpu != 3 and source_pending and not source_here else 0
                    budget = 19000 if gpu == 3 else 17000
                    if guard.gpu_headroom(gpu) - reserve >= budget:
                        candidates.append(gpu)
                if candidates:
                    gpu = min(candidates, key=lambda g: sum(e['gpu'] == g for e in entries))
                    budget = 19000 if gpu == 3 else 17000
                    name = 'gpu%d_%s_%d' % (gpu, study, int(time.time()))
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        MAX_JOBS='2', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                        WANDB_MODE='disabled', USE_TF='0', TF_CPP_MIN_LOG_LEVEL='2', TORCHDYNAMO_DISABLE='1',
                        REFCLOTH_SURFACE_MODE='legacy', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        OPENPI_DATA_HOME='/data/shaozhen.liu/.cache/openpi', MPLCONFIGDIR='/tmp/refcloth_stability_mpl',
                        PYTHONPATH=':'.join([str(R / 'ops'), str(R), str(R / 'pi'), str(R / 'ops/stability_20261102'),
                                           '/data/shaozhen.liu/code/tex3d-deps/taming-transformers']))
                    cmd = ['/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python', str(SCRIPT), '--repo', str(R),
                           '--root', str(x), '--worker', name, '--memory-fraction', '.31']
                    proc = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.DEVNULL,
                        stdout=open(x / ('evaluation_' + name + '.log'), 'a'), stderr=subprocess.STDOUT,
                        start_new_session=True)
                    token = guard.register(proc.pid, SCRIPT, 'eval', gpu, budget, study + ':' + name, x)
                    actors[name] = dict(pid=proc.pid, ticks=guard.identity(proc.pid)['ticks'],
                                        gpu=gpu, token=token, proc=proc)
                    print('EVAL_LAUNCH', name, proc.pid, flush=True)
            guard.write(x / 'eval_controller_status.json', dict(active={n: e['pid'] for n, e in actors.items()},
                failed=failed, resource_pauses=holds, completed_bundles=completed, total_bundles=total,
                available_bundles=len(ready), admission_wait=reasons, updated=time.time()))
        finally:
            fcntl.flock(budget_lock, fcntl.LOCK_UN)
        time.sleep(15)
    result = dict(completed=True, bundles=total, episodes=2600, failed_attempts=failed, resource_pauses=holds)
    if coupling:
        result.update(new_episodes=1800, reused_episodes=800)
    guard.write(x / 'eval_controller_done.json', result)


def entry(study, publish):
    p = argparse.ArgumentParser(); p.add_argument('--resume', action='store_true')
    p.add_argument('--share-gpu3', action='store_true',
                   help='Allow one extra camera worker on host 41 GPU 3 with at least 19000 MiB free')
    main(study, publish, p.parse_args())
