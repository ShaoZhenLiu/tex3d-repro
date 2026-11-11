"""Accept the full native comparison and archive the keyed observations."""
import csv
import json
import statistics
import tarfile
from pathlib import Path

from evaluation import EVAL_PROTOCOL, INITIALS, SUITES
from source import HERE, ROOT, SEEDS, guard, sha

MODES = ('mr', 'main', 'wrist')


def conditions(seeds):
    return ('plain',) + tuple(f's{s}_{m}' for s in seeds for m in MODES)


def table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs(manifest, seeds):
    assert set(manifest) == set(conditions(seeds))
    inputs = guard.read(ROOT / 'prepared.json')['paired_replay']
    receipts = {}
    for name, entry in manifest.items():
        assert sha(ROOT / entry['path']) == entry['sha256']
        if name == 'plain':
            continue
        record = guard.read(ROOT / 'arms' / name / 'completed.json')
        assert record['completed'] and record['iterations'] == 5000
        assert record['texture_sha256'] == entry['sha256']
        if not record.get('reused_training'):
            expected = inputs[str(entry['seed'])]
            assert record['stream']['text_sha256'] == expected['text_sha256']
            assert record['stream']['uv_sha256'] == expected['uv_sha256']
            validation = record['camera_control_validation']
            assert validation['passed']
            if record['mode'] in ('main', 'wrist'):
                assert validation['untouched_camera_processed_exact']
            else:
                assert validation['wrist_uv_shared_draw']
        receipts[name] = record
    for seed in seeds:
        arms = [receipts[f's{seed}_{m}'] for m in MODES]
        assert len({arm['gpu_name'] for arm in arms}) == 1
        verified_init = {arm['theta_init_sha256'] for arm in arms if arm.get('theta_init_sha256')}
        assert len(verified_init) == 1
    return receipts


def verified_episodes(manifest, plan, reused, seeds):
    directory = ROOT / 'evaluation/episodes'
    evaluator_hash = sha(EVAL_PROTOCOL)
    workers = set()
    rows = []
    names = set()
    for suite in SUITES:
        for task in range(10):
            for initial in INITIALS:
                baseline = None
                for condition in conditions(seeds):
                    name = f'{suite}_t{task}_i{initial}_{condition}.json'
                    path = directory / name
                    record = guard.read(path)
                    assert record and record['completed'], name
                    assert (record['suite'], record['task'], record['initial'], record['condition']) == (
                        suite, task, initial, condition), name
                    assert record['texture_sha256'] == manifest[condition]['sha256'], name
                    assert record['native_uploads'] > 0 and isinstance(record['robot_success'], bool), name
                    if plan['node'] == '09':
                        assert record['host'] == 'zgc-33-9', name
                    else:
                        assert record['host'].endswith('-41'), name
                    trajectory = record['trajectory']
                    assert record['queries'] == len(trajectory), name
                    if baseline is None:
                        baseline = trajectory
                    else:
                        if baseline and trajectory:
                            assert baseline[0]['proprio_sha256'] == trajectory[0]['proprio_sha256'], name
                        assert all(x['noise_sha256'] == y['noise_sha256']
                                   for x, y in zip(baseline, trajectory)), name
                    if name in reused:
                        assert sha(path) == reused[name]['sha256'], name
                    else:
                        workers.add(record['worker'])
                    names.add(name)
                    rows.append(dict(suite=suite, task=task, initial=initial, condition=condition,
                                     success=int(record['robot_success']), worker=record['worker'],
                                     reused=int(name in reused)))
    assert names == {path.name for path in directory.glob('*.json')}
    assert len(rows) == 200 * len(conditions(seeds)) and workers
    for worker in workers:
        path = ROOT / 'evaluation/workers' / worker
        assert guard.read(path / 'calibration.json')['passed']
        assert guard.read(path / 'registered.json')['protocol_sha256'] == evaluator_hash
    return rows, sorted(workers)


def estimates(rows, seeds=SEEDS):
    counts = {(suite, condition): 0 for suite in SUITES for condition in conditions(seeds)}
    for row in rows:
        counts[row['suite'], row['condition']] += row['success']
    suites = []
    seed_rows = []
    for seed in seeds:
        for suite in SUITES:
            plain = counts[suite, 'plain']
            row = dict(seed=seed, suite=suite, plain_successes=plain)
            for mode in MODES:
                success = counts[suite, f's{seed}_{mode}']
                row[f'{mode}_successes'] = success
                row[f'{mode}_G_pp'] = plain - success
            suites.append(row)
        means = {m: statistics.mean(r[f'{m}_G_pp'] for r in suites if r['seed'] == seed)
                 for m in MODES}
        seed_rows.append(dict(seed=seed, **{f'{m}_G_pp': means[m] for m in MODES},
                              joint_minus_main_pp=means['mr'] - means['main'],
                              joint_minus_wrist_pp=means['mr'] - means['wrist'],
                              joint_minus_best_single_pp=means['mr'] - max(means['main'], means['wrist'])))
    metrics = {}
    for column in seed_rows[0]:
        if column == 'seed':
            continue
        values = [r[column] for r in seed_rows]
        metrics[column] = dict(mean=statistics.mean(values),
                               sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                               positive=sum(x > 0 for x in values), tied=values.count(0),
                               negative=sum(x < 0 for x in values))
    return suites, seed_rows, metrics


def archive(inputs, metadata):
    path = ROOT / 'single_view_four_seeds.tar.gz'
    temporary = path.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as tar:
        for name in ('analysis', 'frozen', 'original_joint', 'arms', 'streams',
                     'prepared.json', 'source_preflight.json', 'texture_manifest.json',
                     'evaluation_protocol.json', 'evaluation_plan.json', 'reused_rollouts.json',
                     'evaluation/episodes', 'evaluation/workers', 'evaluation/jobs.json',
                     'eval_controller_done.json'):
            source = ROOT / name
            if source.exists():
                tar.add(source, arcname=name)
        tar.add(HERE / 'protocol.json', arcname='training_protocol.json')
        for name, receipt in inputs.items():
            directory = receipt.get('directory')
            if not directory or receipt.get('reused_training'):
                continue
            for filename in ('Ep0_Cloth_Delta_OV.pt', 'theta_init.pt', 'sampling_log.txt',
                             'gradient_log.txt', 'camera_control_validation.json'):
                source = Path(directory) / filename
                if source.is_file():
                    tar.add(source, arcname=f'final_training/{name}/{filename}')
    temporary.replace(path)
    guard.write(ROOT / 'accepted.json', dict(metadata, completed=True, archive=path.name,
                                             sha256=sha(path), archive_bytes=path.stat().st_size))


def main():
    done = guard.read(ROOT / 'eval_controller_done.json', {})
    assert done.get('completed')
    plan = guard.read(ROOT / 'evaluation_plan.json')
    seeds = plan.get('seeds', SEEDS)
    manifest = guard.read(ROOT / 'texture_manifest.json')
    inputs = verify_inputs(manifest, seeds)
    reused = guard.read(ROOT / 'reused_rollouts.json')
    rows, workers = verified_episodes(manifest, plan, reused, seeds)
    suites, seed_rows, metrics = estimates(rows, seeds)
    output = ROOT / 'analysis'
    output.mkdir(exist_ok=True)
    table(output / 'episode_outcomes.csv', rows)
    table(output / 'suite_results.csv', suites)
    table(output / 'seed_means.csv', seed_rows)
    summary = dict(plan=plan, seeds=seed_rows, estimates=metrics, calibrated_workers=workers,
                   fresh_episodes=len(rows) - len(reused), reused_episodes=len(reused),
                   source_training_hardware={k: inputs[k]['gpu_name'] for k in inputs})
    guard.write(output / 'summary.json', summary)
    archive(inputs, summary)
    print(json.dumps(guard.read(ROOT / 'accepted.json'), indent=2), flush=True)


if __name__ == '__main__':
    main()
