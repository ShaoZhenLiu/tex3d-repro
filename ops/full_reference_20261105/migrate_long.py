"""Preserve the partial 41 record and prepare an entirely new Long block on 09."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from common import HERE, LONG_ROOT, REPO, ROOT, jobs, long_jobs, sha
sys.path.insert(0, str(REPO/'ops'))
import refcloth_resource as guard


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha(source) == sha(destination), destination
    else:
        shutil.copyfile(source, destination)


def prepare(node):
    assert os.uname().nodename.endswith('-41') if node == '4090' else os.uname().nodename == 'zgc-33-9'
    if node == '09':
        assert guard.read(ROOT/'long_migration_20261106.json')['authorized']
        assert guard.read(LONG_ROOT/'evaluation/jobs.json') == long_jobs()
        manifest = guard.read(LONG_ROOT/'texture_manifest.json')
        assert len(manifest) == 7
        for info in manifest.values():
            assert sha(LONG_ROOT/info['path']) == info['sha256']
        assert guard.read(LONG_ROOT/'protocol.json') == guard.read(HERE/'pi05_long09.json')
        guard.write(LONG_ROOT/'prepared09.json', dict(passed=True, time=time.time()))
        return
    assert not any('refcloth_full_reference' in e['root'] for e in guard.active())
    original = ROOT/'targets/pi05'
    for job in jobs('pi05'):
        if job['suite'] != 'long':
            done = guard.read(original/'evaluation/completed'/(job['id']+'.json'))
            assert done['completed'] and done['job'] == job
    archive = ROOT/'preserved_long_41_20261106.tar.gz'
    index_path = ROOT/'preserved_long_41_manifest.json'
    if not archive.exists():
        selected = list((original/'evaluation/episodes').glob('long_*.json'))
        assert selected
        payloads = {str(f.relative_to(original)): sha(f) for f in selected}
        for f in (original/'evaluation/completed').glob('long_*.json'):
            payloads[str(f.relative_to(original))] = sha(f)
        guard.write(index_path, dict(episodes=len(selected), source=str(original), files=payloads,
                    reason='Driver failure; complete Long block will be evaluated on 09', time=time.time()))
        with tarfile.open(str(archive.with_suffix('.tmp')), 'w:gz') as stream:
            for name in payloads:
                stream.add(str(original/name), arcname=name)
            stream.add(str(index_path), arcname='preservation_manifest.json')
            stream.add(str(original/'texture_manifest.json'), arcname='texture_manifest.json')
            stream.add(str(original/'evaluation/workers'), arcname='evaluation/workers')
        archive.with_suffix('.tmp').replace(archive)
    index = guard.read(index_path)
    assert all(sha(original/name) == digest for name, digest in index['files'].items())
    for directory in ['evaluation/episodes', 'evaluation/completed', 'evaluation/claims',
                      'evaluation/workers', 'frozen', 'arms']:
        (LONG_ROOT/directory).mkdir(parents=True, exist_ok=True)
    manifest = guard.read(original/'texture_manifest.json')
    for condition, info in manifest.items():
        copy(original/info['path'], LONG_ROOT/info['path'])
        if condition != 'plain':
            copy(original/'arms'/condition/'completed.json', LONG_ROOT/'arms'/condition/'completed.json')
    guard.write(LONG_ROOT/'texture_manifest.json', manifest)
    guard.write(LONG_ROOT/'evaluation/jobs.json', long_jobs())
    guard.write(LONG_ROOT/'evaluation/reused_manifest.json', {})
    copy(HERE/'pi05_long09.json', LONG_ROOT/'protocol.json')
    copy(HERE/'pi05_long09.json', ROOT/'protocols/pi05_long09.json')
    marker = ROOT/'long_migration_20261106.json'
    if not marker.exists():
        guard.write(marker, dict(authorized=True, authorization='User selected entire pi05 Long suite on 09',
            time=time.time(), target='pi05', suites=['long'], node='09', planned_episodes=700,
            preserved_partial_episodes=index['episodes'], preserved_archive=archive.name,
            preserved_archive_sha256=sha(archive), original_files_unchanged=True,
            raw_source=str(original), formal_long_source=str(LONG_ROOT),
            unchanged='Textures, checkpoint, states, suite noise indices, rollout horizon and action recipe'))
    bootstrap = ROOT/'bootstrap_pi05_long09_20261106.tar.gz'
    if not bootstrap.exists():
        with tarfile.open(str(bootstrap.with_suffix('.tmp')), 'w:gz') as stream:
            stream.add(str(LONG_ROOT), arcname='segments/pi05_long_09')
            stream.add(str(marker), arcname=marker.name)
            stream.add(str(ROOT/'protocols/pi05_long09.json'), arcname='protocols/pi05_long09.json')
        bootstrap.with_suffix('.tmp').replace(bootstrap)
    print(json.dumps(dict(passed=True, archive=str(bootstrap), sha256=sha(bootstrap),
               preserved_episodes=index['episodes'], new_episodes=700)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--node', choices=['4090', '09'], required=True)
    prepare(parser.parse_args().node)
