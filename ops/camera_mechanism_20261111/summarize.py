"""Verify every planned rollout and source-gradient checkpoint before archiving."""
import csv
import math
import os
import statistics
import tarfile

from common import INITIALS, PROTOCOL, ROOT, SEEDS, SOURCE_STEPS, TASKS
from common import controls, episode_name, guard, local_node, new_conditions, sha

VIEWS = ('main', 'wrist')
PAIR_FIELDS = ('main_norm', 'wrist_norm', 'joint_norm', 'cosine', 'main_share', 'wrist_share')


def table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contrast(first, second):
    assert len(first) == len(second) == 100
    ab = sum(a and not b for a, b in zip(first, second))
    ba = sum(b and not a for a, b in zip(first, second))
    n = ab + ba
    p = min(1., 2. * sum(math.comb(n, k) for k in range(min(ab, ba) + 1)) / 2**n) if n else 1.
    return dict(diff_pp=ab - ba, first_only=ab, second_only=ba, exact_p=p)


def checked_episode(name, condition, manifest, reused):
    path = ROOT / 'evaluation' / 'episodes' / name
    episode = guard.read(path)
    assert episode and episode['completed'] and episode['suite'] == 'object'
    assert episode['condition'] == condition and episode['host'] == os.uname().nodename
    assert isinstance(episode['robot_success'], bool) and episode['native_uploads'] > 0
    assert episode['texture_sha256'] == manifest[condition]['sha256']
    trace = episode['trajectory']
    assert len(trace) == episode['queries']
    if name in reused:
        assert sha(path) == reused[name]['sha256']
    else:
        assert episode['protocol_sha256'] == sha(PROTOCOL)
        worker = ROOT / 'evaluation' / 'workers' / episode['worker']
        assert guard.read(worker / 'calibration.json')['passed']
        assert guard.read(worker / 'registered.json')['protocol_sha256'] == sha(PROTOCOL)
    mode = manifest[condition].get('camera_mode')
    if mode is not None:
        assert episode['camera_mode'] == mode
        for q, query in enumerate(trace):
            m = query['camera_measurement']
            assert q == query['query']
            assert all(m[k] for k in ('integration_exact', 'cached_exposure_exact', 'aba_exact', 'same_camera_state'))
            for view in VIEWS:
                reference = 'ref' if view == mode else 'plain'
                assert query[f'{view}_sha256'] == m[f'{reference}_{view}_sha256']
            if mode == 'main' and q in (0, 10, 30):
                assert set(m['policy_input_rms']) == {'main', 'main_low', 'wrist'}
    return episode


def verify_eval(node):
    plan = guard.read(ROOT / 'eval_plan.json')
    assert plan['node'] == node and plan['protocol_sha256'] == sha(PROTOCOL)
    assert plan['preflight_sha256'] == sha(ROOT / 'camera_preflight.json')
    preflight = guard.read(ROOT / 'camera_preflight.json')
    assert preflight['passed']
    assert preflight['can_reuse_controls'] == (plan['reused'] == 500)
    assert guard.read(ROOT / 'eval_controller_done.json')['completed']
    seeds = SEEDS[node]
    conditions = controls(seeds) + [condition for seed in seeds for condition in new_conditions(seed)]
    manifest = guard.read(ROOT / 'texture_manifest.json')
    assert set(manifest) == set(conditions)
    for item in manifest.values():
        assert sha(ROOT / item['path']) == item['sha256']
    reused = guard.read(ROOT / 'reused_rollouts.json')
    keys = [(t, i) for t in TASKS for i in INITIALS]
    lookup, episodes, inputs = {}, [], []
    for task, initial in keys:
        group = {}
        for condition in conditions:
            name = episode_name(task, initial, condition)
            ep = checked_episode(name, condition, manifest, reused)
            assert (ep['task'], ep['initial']) == (task, initial)
            group[condition] = ep
            lookup[task, initial, condition] = ep['robot_success']
            episodes.append(dict(task=task, initial=initial, condition=condition,
                                 success=int(ep['robot_success']), queries=ep['queries'],
                                 worker=ep['worker'], reused=int(name in reused)))
        baseline = group['plain']['trajectory']
        for condition, record in group.items():
            trace = record['trajectory']
            assert bool(baseline) == bool(trace)
            if not trace:
                assert record['robot_success']
                continue
            assert all(x['noise_sha256'] == y['noise_sha256'] for x, y in zip(baseline, trace))
            assert baseline[0]['proprio_sha256'] == trace[0]['proprio_sha256']
            if not manifest[condition].get('camera_mode'):
                continue
            first = trace[0]['camera_measurement']
            seed = manifest[condition]['seed']
            original_main = group[f's{seed}_main']['trajectory'][0]
            for view in VIEWS:
                assert first[f'plain_{view}_sha256'] == baseline[0][f'{view}_sha256']
                assert first[f'ref_{view}_sha256'] == original_main[f'{view}_sha256']
            for q in trace:
                for variant, views in q['camera_measurement'].get('policy_input_rms', {}).items():
                    inputs.append(dict(seed=seed, task=task, initial=initial, query=q['query'],
                                       texture=variant, main=views['main'], wrist=views['wrist']))
    assert len(episodes) == 1100 == plan['fresh'] + plan['reused']
    assert len(reused) == plan['reused']
    assert len(list((ROOT / 'evaluation' / 'episodes').glob('*.json'))) == len(episodes)

    scores, comparisons = [], []
    for seed in seeds:
        names = dict(pp='plain', ap=f's{seed}_ap', pa=f's{seed}_pa', aa=f's{seed}_main',
                     low=f's{seed}_main_low', wrist=f's{seed}_wrist')
        outcomes = {k: [lookup[t, i, name] for t, i in keys] for k, name in names.items()}
        successes = {key: sum(values) for key, values in outcomes.items()}
        scores.append(dict(seed=seed, **successes, wrist_only_pp=successes['pp'] - successes['pa'],
                           wrist_given_main_pp=successes['ap'] - successes['aa'],
                           interaction_pp=successes['ap'] + successes['pa'] - successes['pp'] - successes['aa'],
                           g_main_pp=successes['pp'] - successes['aa'],
                           g_low_pp=successes['pp'] - successes['low'],
                           g_wrist_pp=successes['pp'] - successes['wrist']))
        for a, b in [('pp', 'pa'), ('ap', 'aa'), ('aa', 'low'), ('low', 'wrist')]:
            comparisons.append(dict(seed=seed, first=a, second=b, **contrast(outcomes[a], outcomes[b])))
    return plan, episodes, scores, comparisons, inputs


def verify_source(node):
    source = ROOT / 'source'
    done = guard.read(source / 'done.json')
    assert done['completed'] and tuple(done['steps']) == SOURCE_STEPS
    assert tuple(done['seeds']) == SEEDS[node]
    assert done['bank_sha256'] == sha(source / 'bank.json')
    gradients = []
    for seed in SEEDS[node]:
        for update in SOURCE_STEPS:
            folder = source / f's{seed}_u{update:05d}'
            row = guard.read(folder / 'result.json')
            assert row['completed'] and row['seed'] == seed and row['updates'] == update
            assert row['bank_sha256'] == done['bank_sha256']
            assert row['gradient_sha256'] == sha(folder / 'batch_gradients.npz')
            assert len(row['batches']) == 8 and len(row['frames']) == 64
            residuals = [r['decomposition_residual'] for r in row['batches']]
            assert max(residuals) <= .002
            for r in row['batches']:
                if r['joint_norm'] > 1e-15:
                    assert abs(r['main_share'] + r['wrist_share'] - 1.) < 1e-8
            medians = {}
            for key in PAIR_FIELDS:
                values = [r[key] for r in row['batches'] if r[key] is not None]
                medians[key] = statistics.median(values) if values else None
            gradients.append(dict(seed=seed, updates=update, batches=8, **medians,
                                  max_decomposition_residual=max(residuals)))
    return done, gradients


def main():
    node = local_node()
    plan, episodes, scores, comparisons, inputs = verify_eval(node)
    source, gradients = verify_source(node)
    output = ROOT / 'analysis'
    output.mkdir(exist_ok=True)
    table(output / 'episode_outcomes.csv', episodes)
    table(output / 'closed_loop.csv', scores)
    table(output / 'paired_contrasts.csv', comparisons)
    table(output / 'policy_input_rms.csv', inputs)
    table(output / 'source_gradient.csv', gradients)
    summary = dict(node=node, plan=plan, closed_loop=scores, source=source,
                   texture_amplitudes=guard.read(ROOT / 'texture_amplitudes.json'))
    guard.write(output / 'summary.json', summary)
    path = ROOT / 'camera_mechanism_complete.tar.gz'
    temporary = path.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as tar:
        for name in ('analysis', 'source', 'frozen', 'prepared.json', 'camera_preflight.json',
                     'texture_manifest.json', 'texture_amplitudes.json', 'evaluation_protocol.json',
                     'eval_plan.json', 'reused_rollouts.json', 'evaluation', 'eval_controller_done.json'):
            tar.add(ROOT / name, arcname=name)
    temporary.replace(path)
    guard.write(ROOT / 'accepted.json', dict(summary, completed=True, archive=path.name,
                                             archive_bytes=path.stat().st_size, sha256=sha(path)))
    print('ACCEPTED', node, plan['fresh'], plan['reused'], sha(path), flush=True)


if __name__ == '__main__':
    main()
