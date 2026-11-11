"""Run one read-only source probe per host with the shared resource guard."""
import argparse
import fcntl
import importlib.util
import subprocess
import time

from common import GPUS, HERE, REPO, ROOT, SEEDS, STUDY, guard, local_node
from control import available, lock_guard

PYTHON = '/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'
SCRIPT = HERE / 'source_probe.py'
LABEL = STUDY + ':source_probe'


def main(a):
    assert a.node == local_node()
    assert a.gpus and set(a.gpus) <= set(GPUS[a.node])
    source = ROOT / 'source'
    source.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        'existing_source_environment', REPO / 'ops/uv_gradient_20261109/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    budget = 24000 if a.node == '09' else 22500
    with (source / 'controller.lock').open('a') as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with (REPO / f'experiments/refcloth_gpu_budget_{a.node}.lock').open('a') as lock:
            while not guard.read(source / 'done.json', {}).get('completed'):
                process = None
                with lock_guard(lock):
                    allowed, reasons = guard.can_launch(cold_gib=10, max_models=6 if a.node == '09' else 5)
                    if any(e['label'] == LABEL for e in guard.active()):
                        allowed = False
                        reasons.append('source_probe_running')
                    choices = [(available(g), g) for g in a.gpus] if allowed else []
                    candidates = [item for item in choices if item[0] >= budget]
                    if candidates:
                        _, gpu = max(candidates)
                        env = module.env_for(gpu)
                        command = [PYTHON, str(SCRIPT), '--node', a.node]
                        with (ROOT / 'logs/source_probe.log').open('a') as log:
                            process = subprocess.Popen(command, cwd='/data/shaozhen.liu/code/tex3d-ws/g7',
                                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                guard.register(process.pid, SCRIPT, 'source', gpu, budget, LABEL, ROOT)
                                break
                            except AssertionError:
                                if process.poll() is not None:
                                    raise RuntimeError('Source probe exited before registration')
                                if time.monotonic() >= deadline:
                                    process.terminate()
                                    raise RuntimeError('Source probe command is unavailable')
                                time.sleep(.05)
                        ticks = guard.identity(process.pid)['ticks']
                        print('SOURCE_PROBE_LAUNCH', gpu, process.pid, flush=True)
                    else:
                        guard.write(source / 'wait.json', dict(reasons=reasons or ['no_eligible_gpu'],
                                                              updated=time.time()))
                if process is None:
                    time.sleep(15)
                    continue
                code = process.wait()
                if code not in (0, 75) and not guard.resource_stopped(process.pid, ticks):
                    raise RuntimeError(f'Source gradient probe exited {code}. See logs/source_probe.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=SEEDS, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    main(parser.parse_args())
