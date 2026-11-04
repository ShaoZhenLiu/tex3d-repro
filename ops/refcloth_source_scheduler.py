"""Shared source scheduler: resource pauses requeue the same experimental arm."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import refcloth_resource as guard

R = Path('/data/shaozhen.liu/code/tex3d')
PYTHON = '/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'


def main(study, a, *, root=None, jobs_override=None, gpus_override=None,
         gpu_required_mib=None, model_limit=None):
    coupling = study == 'coupling'
    x = R / ('experiments/refcloth_coupling/20261102' if coupling else 'experiments/refcloth_stability/20261102')
    ops = R / ('ops/coupling_20261102' if coupling else 'ops/stability_20261102')
    script = ops / ('train_arm.py' if coupling else 'train_pair_arm.py')
    x = root if root is not None else x
    status = x / ('train_status_%s.json' % a.node)
    if coupling:
        jobs = [(12, m) for m in ['uv_ind', 'main', 'wrist']] if a.node == '09' else [
            (10, 'uv_ind'), (11, 'uv_ind'), (10, 'main'), (10, 'wrist'), (11, 'main'), (11, 'wrist')]
        preferred = {}; fraction = .27 if a.node == '09' else .44
    else:
        jobs = [(12, 'native'), (12, 'mr')] if a.node == '09' else [
            (10, 'native'), (10, 'mr'), (11, 'native'), (11, 'mr')]
        preferred = {(10, 'native'): 0, (10, 'mr'): 6, (11, 'native'): 1, (11, 'mr'): 5}
        fraction = .285 if a.node == '09' else .51
    gpus = [2, 1] if a.node == '09' else [0, 1, 5, 6, 7]
    if jobs_override is not None:
        jobs = jobs_override
        preferred = {}
    if gpus_override is not None:
        gpus = gpus_override
    required = 24000 if a.node == '09' else 23000
    share_source_gpu = root is not None and a.node == '09'
    if model_limit is None and share_source_gpu:
        model_limit = len(gpus)
    required_by_gpu = dict.fromkeys(gpus, required)
    required_by_gpu.update(gpu_required_mib or {})
    singleton = (guard.BASE / ('%s_source.lock' % study))
    singleton.parent.mkdir(exist_ok=True)
    owner = singleton.open('a')
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A source controller for this study is already running')
    prior = guard.read(status, {}) if a.resume else {}
    pending = []; active = {}; finished = []; failed = list(prior.get('failed', []))
    holds = list(prior.get('resource_pauses', []))
    for job in jobs:
        name = 's%d_%s' % job
        if guard.read(x / 'arms' / name / 'completed.json', {}).get('completed'):
            finished.append(name)
            continue
        old = prior.get('active', {}).get(name)
        current = guard.identity(old['pid']) if old else None
        if (current and current['uid'] == os.getuid() and str(script) in current['cmd']
                and str(x) in current['cmd'] and '--seed %d' % job[0] in current['cmd']
                and ('--mode ' if coupling else '--ref ') + job[1] in current['cmd']):
            gpu = old.get('gpu', old['job'][2] if len(old['job']) == 3 else None)
            token = guard.register(old['pid'], script, 'source', gpu, required_by_gpu.get(gpu, required),
                                   study + ':' + name, x, phase='warm', started=old['start'])
            active[name] = dict(pid=old['pid'], proc=None, gpu=gpu, job=job,
                                start=old['start'], token=token, ticks=current['ticks'])
        else:
            pending.append(job)
    (x / 'logs').mkdir(exist_ok=True)
    (x / 'arms').mkdir(exist_ok=True)
    lock = (R / ('experiments/refcloth_gpu_budget_%s.lock' % a.node)).open('a')
    while pending or active:
        for name, item in list(active.items()):
            current = guard.identity(item['pid'])
            if current and current['ticks'] == item['ticks']:
                continue
            code = item['proc'].poll() if item['proc'] else None
            if guard.read(x / 'arms' / name / 'completed.json', {}).get('completed'):
                finished.append(name)
                print('FINISHED', name, flush=True)
            elif code == 75 or guard.resource_stopped(item['pid'], item['ticks']):
                pending.insert(0, item['job'])
                holds.append(dict(name=name, pid=item['pid'], time=time.time(), reason='resource_pressure'))
                print('RESOURCE_REQUEUED', name, flush=True)
            else:
                failed.append(dict(name=name, pid=item['pid'], exit_code=code))
                print('INFRASTRUCTURE_ERROR', name, code, flush=True)
            del active[name]
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            allowed, reasons = guard.can_launch(cold_gib=10, max_models=model_limit)
            if pending and allowed:
                job = pending[0]; seed, mode = job
                choices = sorted(gpus, key=lambda g: (g != preferred.get(job), sum(v['gpu'] == g for v in active.values()), gpus.index(g)))
                choices = [g for g in choices if guard.gpu_headroom(g) >= required_by_gpu[g]
                           and (root is None or share_source_gpu or not any(e['gpu'] == g and e['kind'] == 'source'
                                                        for e in guard.active()))]
                if not choices:
                    reasons.append('no_eligible_gpu')
                if choices:
                    gpu = choices[0]; name = 's%d_%s' % job
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        MAX_JOBS='2', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                        WANDB_MODE='disabled', USE_TF='0', TF_CPP_MIN_LOG_LEVEL='2',
                        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                        PYTHONPATH=':'.join([str(R / 'ops'), str(ops), str(R), str(R / 'openvla-oft')]))
                    cmd = [PYTHON, str(script), '--repo', str(R), '--root', str(x), '--seed', str(seed),
                           '--mode' if coupling else '--ref', mode, '--memory-fraction', str(fraction)]
                    if not coupling:
                        cmd.append('--allow-frozen-wrist-gap')
                    proc = subprocess.Popen(cmd, cwd='/data/shaozhen.liu/code/tex3d-ws/g7', env=env,
                        stdin=subprocess.DEVNULL, stdout=open(x / 'logs' / (name + '.log'), 'a'),
                        stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            token = guard.register(proc.pid, script, 'source', gpu, required_by_gpu[gpu], study + ':' + name, x)
                            break
                        except AssertionError:
                            if proc.poll() is not None:
                                raise RuntimeError('Source process exited before registration: ' + name)
                            if time.monotonic() >= deadline:
                                proc.terminate()
                                raise RuntimeError('Source process command did not become visible: ' + name)
                            time.sleep(.05)
                    active[name] = dict(proc=proc, pid=proc.pid, ticks=guard.identity(proc.pid)['ticks'],
                                        gpu=gpu, job=job, start=time.time(), token=token)
                    pending.pop(0)
                    print('LAUNCH', name, proc.pid, 'GPU', gpu, flush=True)
            # Keep the previous status schema for the transport/evaluation readers.
            pending_rows = [list(j) if coupling else list(j) + [preferred.get(j, 1)] for j in pending]
            guard.write(status, dict(node=a.node, pending=pending_rows,
                active={name: dict(pid=v['pid'], gpu=v['gpu'], job=list(v['job']) if coupling else list(v['job']) + [v['gpu']],
                                  start=v['start']) for name, v in active.items()},
                finished=finished, failed=failed, resource_pauses=holds, allowed_gpus=gpus,
                required_mib=required, required_mib_by_gpu=required_by_gpu, model_limit=model_limit,
                gradient_mode='texture_only', admission_wait=reasons,
                updated=time.time()))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if pending or active:
            time.sleep(10)
    guard.write(x / ('train_done_%s.json' % a.node), dict(completed=not failed, finished=finished,
                failed=failed, resource_pauses=holds))


def entry(study):
    p = argparse.ArgumentParser(); p.add_argument('--node', choices=['4090', '09'], required=True)
    p.add_argument('--resume', action='store_true')
    main(study, p.parse_args())
