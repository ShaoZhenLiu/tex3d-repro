"""Native two-camera evaluation using the established paired-noise worker."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from source import HERE, REPO, ROOT, SEEDS, freeze, guard, sha
import refcloth_eval_scheduler as scheduler

SUITES = ('spatial', 'object')
INITIALS = range(10, 20)
PRIOR = REPO / 'experiments/refcloth_edpa_seeds/20261110'
EVAL_PROTOCOL = ROOT / 'evaluation_protocol.json'


def existing_joint(seed):
    key = f's{seed}_mr'
    imported = guard.read(ROOT / 'original_joint' / (key + '.json'))
    if imported is not None:
        assert imported['seed'] == seed and imported['reused_training']
        assert sha(Path(imported['texture'])) == imported['texture_sha256']
        return imported
    prior = REPO / 'experiments/refcloth_stability' / ('20261102' if seed == 7 else '20261104')
    manifest = guard.read(prior / 'texture_manifest.json')
    entry = manifest[key]
    path = prior / entry['path']
    assert sha(path) == entry['sha256']
    receipt = guard.read(prior / 'arms' / key / 'completed.json', {})
    return dict(receipt, completed=True, seed=seed, mode='mr', iterations=5000,
                texture=str(path), texture_sha256=entry['sha256'], prior_study=str(prior),
                reused_training=True, frozen_manifest=entry)


def stage_joint(seeds):
    for seed in seeds:
        receipt = existing_joint(seed)
        key = f's{seed}_mr'
        target = ROOT / 'original_joint' / (key + '.png')
        freeze(Path(receipt['texture']), target, receipt['texture_sha256'])
        receipt['texture'] = str(target)
        guard.write(target.with_suffix('.json'), receipt)
        print('ORIGINAL_JOINT_READY', seed, receipt['texture_sha256'], flush=True)


def prepare(node, original_joint, seeds, shrink_unstarted=False):
    plan_path = ROOT / 'evaluation_plan.json'
    assert len(set(seeds)) == len(seeds) and set(original_joint) <= set(seeds)
    plan = {'node': node, 'original_joint': list(original_joint), 'seeds': list(seeds)}
    saved = guard.read(plan_path)
    frozen = ROOT / 'frozen'
    outcomes = ROOT / 'evaluation/episodes'
    conditions = ['plain'] + [f's{s}_{mode}' for s in seeds for mode in ('mr', 'main', 'wrist')]
    if saved is not None:
        saved.setdefault('seeds', list(SEEDS))
        if saved != plan:
            assert shrink_unstarted and saved['node'] == node
            assert set(seeds) < set(saved['seeds']) and saved['original_joint'] == list(original_joint)
            retained = {f'{suite}_t{task}_i{i}_{condition}.json' for suite in SUITES
                        for task in range(10) for i in INITIALS for condition in conditions}
            assert all(p.name in retained for p in outcomes.glob('*.json'))
    plain = PRIOR / 'frozen/plain.png'
    reference = PRIOR / 'evaluation/episodes/spatial_t0_i10_plain.json'
    plain_sha = guard.read(reference)['texture_sha256']
    freeze(plain, frozen / 'plain.png', plain_sha)
    manifest = guard.read(ROOT / 'texture_manifest.json', {})
    assert set(manifest) <= set(conditions)
    manifest['plain'] = dict(path='frozen/plain.png', sha256=plain_sha, reference='plain')
    freeze(reference, ROOT / 'calibration/spatial_t0_i10_plain.json', sha(reference))
    cfg = guard.read(HERE / 'protocol.json')
    cfg['evaluation']['calibration_reference'] = 'calibration/spatial_t0_i10_plain.json'
    guard.write(EVAL_PROTOCOL, cfg)
    reused = {}
    jobs = []
    for seed in original_joint:
        receipt = existing_joint(seed)
        key = f's{seed}_mr'
        old = guard.read(ROOT / 'arms' / key / 'completed.json')
        assert old is None or old['texture_sha256'] == receipt['texture_sha256']
        guard.write(ROOT / 'arms' / key / 'completed.json', receipt)
    manifest_baseline = guard.read(PRIOR / 'baseline/manifest.json') if node == '4090' else {}
    reused_conditions = {'plain'} | {f's{s}_mr' for s in original_joint}
    for suite in SUITES:
        for task in range(10):
            for condition in conditions:
                reuse = node == '4090' and condition in reused_conditions
                if reuse:
                    for initial in INITIALS:
                        name = f'{suite}_t{task}_i{initial}_{condition}.json'
                        evidence = manifest_baseline[name]
                        freeze(Path(evidence['source']), outcomes / name, evidence['sha256'])
                        reused[name] = evidence
                else:
                    jobs.append(dict(id=f'{suite}_t{task}_{condition}', target='pi05',
                                     suite=suite, task=task, node=node, conditions=[condition]))
    assert 10 * len(jobs) + len(reused) == 200 * len(conditions)
    guard.write(ROOT / 'texture_manifest.json', manifest)
    guard.write(ROOT / 'evaluation/jobs.json', jobs)
    guard.write(ROOT / 'reused_rollouts.json', reused)
    guard.write(plan_path, plan)
    print('EVALUATION_READY', node, 'new', 10 * len(jobs), 'reused', len(reused), flush=True)


def publish():
    manifest = guard.read(ROOT / 'texture_manifest.json')
    changed = False
    plan = guard.read(ROOT / 'evaluation_plan.json')
    for seed in plan.get('seeds', SEEDS):
        for mode in ('mr', 'main', 'wrist'):
            name = f's{seed}_{mode}'
            if name in manifest:
                continue
            receipt = guard.read(ROOT / 'arms' / name / 'completed.json', {})
            if not receipt.get('completed'):
                continue
            assert receipt['seed'] == seed and receipt['mode'] == mode and receipt['iterations'] == 5000
            texture = ROOT / 'frozen' / (name + '.png')
            freeze(Path(receipt['texture']), texture, receipt['texture_sha256'])
            manifest[name] = dict(path=str(texture.relative_to(ROOT)), sha256=receipt['texture_sha256'],
                                  seed=seed, reference=mode)
            changed = True
    if changed:
        guard.write(ROOT / 'texture_manifest.json', manifest)


def run(node, gpus):
    plan = guard.read(ROOT / 'evaluation_plan.json')
    assert node == plan['node']
    assert (os.uname().nodename.endswith('-41')) == (node == '4090')
    assert set(gpus) <= ({0, 1, 2, 3, 5, 6, 7} if node == '4090' else {0, 1, 2, 3})
    original_headroom = guard.gpu_headroom
    if node == '09':
        def headroom(gpu):
            free = original_headroom(gpu)
            pending = guard.read(ROOT / 'train_status_09.json', {}).get('pending')
            if pending and free >= 24000:
                free -= 24000
            return free
        guard.gpu_headroom = headroom
    try:
        scheduler.main('single_view_20261111', publish,
                       SimpleNamespace(gpus=gpus, share_gpu3=False, resume=True),
                       root=ROOT, node=node, evaluation_protocol=EVAL_PROTOCOL,
                       reused_episode_count=0, allow_stochastic_calibration=node == '09',
                       memory_budget_mib=18000 if node == '09' else None)
    finally:
        guard.gpu_headroom = original_headroom


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'publish', 'stage-original'))
    parser.add_argument('--node', choices=('4090', '09'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--original-joint', nargs='+', type=int, choices=SEEDS, default=[])
    parser.add_argument('--gpus', nargs='+', type=int, default=[])
    parser.add_argument('--shrink-unstarted', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.node, args.original_joint, args.seeds, args.shrink_unstarted)
    elif args.stage == 'run':
        assert args.gpus
        run(args.node, args.gpus)
    elif args.stage == 'stage-original':
        stage_joint(args.original_joint)
    else:
        publish()
