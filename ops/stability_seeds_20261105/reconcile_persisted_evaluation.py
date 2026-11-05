"""Reconcile completion indexes against unmodified recorded episode files.

Used after quiescing duplicate workers. No episode observations are edited.
Every original file is archived before any derived completion index is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import refcloth_resource as guard


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(root, apply=False):
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid == os.getuid():
                args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
                assert not (any(a.endswith('/evaluate.py') for a in args)
                            and any(str(root.resolve()) == str(Path(a).resolve())
                                    for a in args if 'experiments/refcloth_stability/' in a)), 'Quiesce study evaluators first'
        except (OSError, ValueError):
            pass
    jobs = {j['id']: j for j in guard.read(root/'evaluation/jobs.json') if j.get('node') == '09'}
    textures = guard.read(root/'texture_manifest.json')
    changes = []
    for file in sorted((root/'evaluation/completed').glob('*.json')):
        old = guard.read(file)
        if old['job']['id'] not in jobs:
            continue
        job = jobs[old['job']['id']]
        assert old['job'] == job and old['completed']
        expected = {(i, c) for i in range(10, 20) for c in job['conditions']}
        assert {(row['initial'], row['condition']) for row in old['outcomes']} == expected
        episodes = {}
        paths = {}
        for initial, condition in sorted(expected):
            path = root/'evaluation/episodes'/('%s_t%d_i%d_%s.json' % (job['suite'], job['task'], initial, condition))
            episode = guard.read(path)
            assert episode and episode['completed'] and episode['host'] == 'zgc-33-9'
            assert (episode['suite'], episode['task'], episode['initial'], episode['condition']) == (job['suite'], job['task'], initial, condition)
            assert episode['texture_sha256'] == textures[condition]['sha256']
            worker = root/'evaluation/workers'/episode['worker']
            assert guard.read(worker/'calibration.json')['passed']
            assert (worker/'registered.json').exists()
            episodes[(initial, condition)] = episode
            paths[str(path.relative_to(root))] = digest(path)
        for initial in range(10, 20):
            pair = [episodes[(initial, c)] for c in job['conditions']]
            left, right = pair[0]['trajectory'], pair[1]['trajectory']
            if left and right:
                assert left[0]['proprio_sha256'] == right[0]['proprio_sha256']
            else:
                assert not left and not right and all(p['robot_success'] for p in pair)
            assert all(x['noise_sha256'] == y['noise_sha256'] for x, y in zip(left, right))
        differences = [row for row in old['outcomes']
                       if row['robot_success'] != episodes[(row['initial'], row['condition'])]['robot_success']]
        if differences:
            updated = dict(old)
            updated['outcomes'] = [dict(row, robot_success=episodes[(row['initial'], row['condition'])]['robot_success'])
                                   for row in old['outcomes']]
            updated['index_reconciliation'] = dict(original_index_sha256=digest(file), episode_sha256=paths,
                rule='Use the unchanged persisted episode records in the quiescent snapshot for every condition',
                original_episode_observations_modified=False, mismatch_count=len(differences))
            changes.append((file, updated))
    report = dict(changed_indexes=[str(p.relative_to(root)) for p, _ in changes],
                  mismatched_index_rows=sum(x['index_reconciliation']['mismatch_count'] for _, x in changes),
                  episode_observations_modified=False)
    if not apply:
        return report
    recovery = root/'recovery'/('duplicate_claims_%d' % time.time())
    recovery.mkdir(parents=True)
    original_files = [p for p in (root/'evaluation').rglob('*') if p.is_file()]
    original_files += list(root.glob('evaluation_*.log'))
    originals = {str(p.relative_to(root)): digest(p) for p in original_files}
    archive = recovery/'original_evaluation.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in original_files:
            stream.add(path, arcname=str(path.relative_to(root)))
    guard.write(recovery/'original_sha256.json', originals)
    report.update(archive=str(archive), archive_sha256=digest(archive), snapshot_files=len(original_files))
    for file, updated in changes:
        guard.write(file, updated)
    for name, old_digest in originals.items():
        if name.startswith('evaluation/episodes/'):
            assert digest(root/name) == old_digest
    guard.write(recovery/'reconciliation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(reconcile(args.root, args.apply)))
