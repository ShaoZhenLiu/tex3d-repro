"""Continue a predeclared reference comparison without changing old results."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'ops'))
import refcloth_resource as guard
import refcloth_source_scheduler as source
import refcloth_eval_scheduler as evaluator

NAME = 'stability_seeds_20261104'
ROOT = REPO / 'experiments/refcloth_stability/20261104'
OLD = REPO / 'experiments/refcloth_stability/20261102'
PROTOCOL_PATH = HERE / 'protocol.json'
PROTOCOL = guard.read(PROTOCOL_PATH)
SEEDS = PROTOCOL['new_training_seeds']
SCOPE = {'4090': [0, 1, 2, 3], '09': [0, 1, 2, 3]}
BASELINE = REPO / 'ops/stability_20261102'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source_path, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        assert sha(source_path) == sha(target_path), target_path
        return
    tmp = target_path.with_suffix('.copying')
    shutil.copyfile(source_path, tmp)
    tmp.replace(target_path)


def prepare(node):
    streams = ROOT / 'streams'
    old = guard.read(streams / 'prepared.json', {})
    if old.get('passed'):
        assert sorted(r['seed'] for r in old['streams']) == sorted(SEEDS)
        for row in old['streams']:
            path = streams / ('seed_%d.txt' % row['seed'])
            assert sha(path) == row['text_sha256']
            assert sha(path.with_name(path.stem+'_full.pt')) == row['uv_sha256']
        return

    budget = (REPO / ('experiments/refcloth_gpu_budget_%s.lock' % node)).open('a')
    while True:
        fcntl.flock(budget, fcntl.LOCK_EX)
        try:
            allowed, reasons = guard.can_launch(cold_gib=10)
            candidates = [gpu for gpu in SCOPE[node] if allowed and guard.gpu_headroom(gpu) >= 6000]
            if candidates:
                gpu = candidates[0]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                    MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', MAX_JOBS='2', HF_HUB_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
                    PYTHONPATH=':'.join([str(REPO / 'ops'), str(BASELINE), str(REPO), str(REPO / 'openvla-oft')]))
                script = BASELINE / 'prepare.py'
                command = [source.PYTHON, str(script), '--repo', str(REPO), '--protocol',
                           str(PROTOCOL_PATH), '--out', str(streams)]
                subprocess.run(command, cwd='/data/shaozhen.liu/code/tex3d-ws/g7',
                               env=env, check=True, timeout=240, stdin=subprocess.DEVNULL)
                guard.write(ROOT / 'preparation.json', dict(passed=True, gpu=gpu, node=node,
                                                            protocol_sha256=sha(PROTOCOL_PATH)))
                return
            guard.write(ROOT / 'preparation.json', dict(passed=False, node=node,
                        reasons=reasons or ['no_gpu_with_6000MiB_for_smoke']))
        finally:
            fcntl.flock(budget, fcntl.LOCK_UN)
        time.sleep(30)


def initialize_evaluation():
    old = guard.read(OLD / 'texture_manifest.json')['plain']
    copy(OLD / old['path'], ROOT / old['path'])
    assert sha(ROOT / old['path']) == old['sha256']
    manifest_path = ROOT / 'texture_manifest.json'
    if not manifest_path.exists():
        guard.write(manifest_path, dict(plain=old))
    jobs = []
    ep = PROTOCOL['evaluation']
    for suite in ep['suites']:
        for task in ep['tasks']:
            plain_id = '%s_t%d_plain' % (suite, task)
            plain_keys = ['%s_t%d_i%d_plain.json' % (suite, task, initial)
                          for initial in ep['initials']]
            for key in plain_keys:
                copy(OLD / 'evaluation/episodes' / key, ROOT / 'evaluation/episodes' / key)
            rel = Path('evaluation/completed') / (plain_id + '.json')
            copy(OLD / rel, ROOT / rel)
            jobs.append(dict(id=plain_id, suite=suite, task=task, conditions=['plain']))
            for seed in SEEDS:
                jobs.append(dict(id='%s_t%d_s%d' % (suite, task, seed), suite=suite, task=task,
                    conditions=['s%d_native' % seed, 's%d_mr' % seed], requires_existing=plain_keys))
    assert len(jobs) == len(ep['suites']) * len(ep['tasks']) * (1 + len(SEEDS))
    assert sum(len(ep['initials']) * len(job['conditions']) for job in jobs) == ep['planned_episodes']
    for row in (ROOT / 'evaluation/episodes').glob('*_plain.json'):
        name = guard.read(row)['worker']
        rel = Path('evaluation/workers') / name / 'calibration.json'
        copy(OLD / rel, ROOT / rel)
    guard.write(ROOT / 'evaluation/jobs.json', jobs)


def publish():
    manifest_path = ROOT / 'texture_manifest.json'
    manifest = guard.read(manifest_path)
    changed = False
    for seed in SEEDS:
        names = ['s%d_native' % seed, 's%d_mr' % seed]
        if all(name in manifest for name in names):
            continue
        metas = [ROOT / 'arms' / name / 'completed.json' for name in names]
        if not all(path.exists() for path in metas):
            continue
        infos = [guard.read(path) for path in metas]
        if not all(info and info.get('completed') for info in infos):
            continue
        assert all(info['iterations'] == 5000 and
                   info['frame_visits'] == 40000 for info in infos)
        for key in ['theta_init_sha256', 'sampling_log_sha256', 'stream', 'gpu_name']:
            assert infos[0][key] == infos[1][key], (seed, key)
        for name, path, info in zip(names, metas, infos):
            target = ROOT / 'frozen' / (name + '.png')
            texture = Path(info['texture'])
            if texture.exists():
                copy(texture, target)
            if not target.exists():
                break
            assert sha(target) == info['texture_sha256']
        else:
            for name, path, info in zip(names, metas, infos):
                manifest[name] = dict(path='frozen/' + name + '.png', sha256=info['texture_sha256'],
                    seed=seed, reference=info['reference'],
                    training_metadata=str(path.relative_to(ROOT)), frozen_existing=False)
            changed = True
    if changed:
        guard.write(manifest_path, manifest)


def main(args):
    assert os.uname().nodename.endswith('-41') if args.node == '4090' else 'zgc-33-9' in os.uname().nodename
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        with (ROOT / 'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args.node)
    elif args.stage == 'source':
        prepare_file = guard.read(ROOT / 'streams/prepared.json', {})
        assert prepare_file.get('passed')
        args.seeds = args.seeds or SEEDS
        jobs = [(seed, ref) for seed in args.seeds for ref in ['native', 'mr']]
        source.main(NAME, args, root=ROOT, jobs_override=jobs, gpus_override=SCOPE[args.node])
    elif args.stage == 'eval':
        assert args.node == '4090'
        initialize_evaluation()
        args.gpus, args.share_gpu3 = SCOPE[args.node], False
        evaluator.main(NAME, publish, args, root=ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['prepare', 'source', 'eval'])
    parser.add_argument('--node', required=True, choices=['4090', '09'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    PROTOCOL_PATH = args.protocol.resolve()
    assert PROTOCOL_PATH.parent.parent == REPO / 'ops'
    NAME = PROTOCOL_PATH.parent.name
    prefix = 'stability_seeds_'
    date = NAME[len(prefix):]
    assert NAME.startswith(prefix) and date.isdigit() and len(date) == 8
    ROOT = REPO / 'experiments/refcloth_stability' / date
    PROTOCOL = guard.read(PROTOCOL_PATH)
    SEEDS = PROTOCOL['new_training_seeds']
    if args.seeds and (len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(SEEDS)):
        parser.error('--seeds must be a nonrepeating subset of the protocol seeds')
    main(args)
