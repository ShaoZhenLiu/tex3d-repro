"""Freeze six existing exports and exactly reuse 1000 compatible observations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from common import HERE, INITIALS, REPO, ROOT, SEEDS, compatible_reuse, episode_name, jobs, sha, source_root
sys.path.insert(0, str(REPO/'ops'))
import refcloth_resource as guard


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), target
    else:
        shutil.copyfile(source, target)


def initialize(node):
    assert (os.uname().nodename.endswith('-41') if node == '4090' else os.uname().nodename == 'zgc-33-9')
    if node == '09':
        target = ROOT/'targets/pi0'
        manifest = guard.read(target/'texture_manifest.json')
        assert len(manifest) == 7 and guard.read(target/'evaluation/jobs.json') == jobs('pi0')
        for info in manifest.values():
            assert sha(target/info['path']) == info['sha256']
        guard.write(ROOT/'preparation09.json', dict(passed=True, target='pi0', textures=7))
        return
    for name in ['pi0', 'pi05']:
        target = ROOT/'targets'/name
        for folder in ['evaluation/episodes', 'evaluation/completed', 'evaluation/claims', 'evaluation/workers', 'frozen', 'arms']:
            (target/folder).mkdir(parents=True, exist_ok=True)
        manifest = {}
        for seed in SEEDS:
            original = source_root(seed)
            previous = guard.read(original/'texture_manifest.json')
            for ref in ['native', 'mr']:
                label = 's%d_%s' % (seed, ref)
                item = previous[label]
                assert sha(original/item['path']) == item['sha256']
                copy(original/item['path'], target/'frozen'/(label+'.png'))
                meta_path = original/'arms'/label/'completed.json'
                meta = guard.read(meta_path)
                assert meta['completed'] and meta['iterations'] == 5000 and meta['frame_visits'] == 40000
                assert meta['texture_sha256'] == item['sha256']
                copy(meta_path, target/'arms'/label/'completed.json')
                manifest[label] = dict(item, path='frozen/'+label+'.png', frozen_existing=True,
                                      original_root=str(original), training_metadata='arms/'+label+'/completed.json')
        old = source_root(10)
        plain = guard.read(old/'texture_manifest.json')['plain']
        copy(old/plain['path'], target/'frozen/plain.png')
        manifest['plain'] = dict(plain, path='frozen/plain.png', frozen_existing=True)
        guard.write(target/'texture_manifest.json', manifest)
        guard.write(target/'evaluation/jobs.json', jobs(name))
        reused = {}
        if name == 'pi05':
            protocol = guard.read(HERE/'pi05.json')['evaluation']
            assert compatible_reuse(guard.read(REPO/'ops/stability_20261102/protocol.json')['evaluation'], protocol)
            for job in jobs(name):
                if job['suite'] not in ['spatial', 'object'] or any(c.startswith('s3_') for c in job['conditions']):
                    continue
                rows = []
                for initial in INITIALS:
                    for condition in job['conditions']:
                        filename = episode_name(job['suite'], job['task'], initial, condition)
                        path = old/'evaluation/episodes'/filename
                        episode = guard.read(path)
                        assert episode['completed'] and episode['host'].endswith('-41')
                        assert episode['texture_sha256'] == manifest[condition]['sha256']
                        worker = episode['worker']
                        for field in ['calibration.json', 'registered.json']:
                            copy(old/'evaluation/workers'/worker/field, target/'evaluation/workers'/worker/field)
                        registration = guard.read(old/'evaluation/workers'/worker/'registered.json')
                        assert registration['checkpoint'] == protocol['checkpoint']
                        copy(path, target/'evaluation/episodes'/filename)
                        reused[filename] = dict(source=str(path), sha256=sha(path))
                        rows.append(dict(condition=condition, initial=initial, robot_success=episode['robot_success']))
                guard.write(target/'evaluation/completed'/(job['id']+'.json'), dict(
                    completed=True, job=job, outcomes=rows, worker='reuse', reused=True,
                    provenance='Unmodified observations from the exact common-noise 4090 evaluation'))
            assert len(reused) == 1000
        guard.write(target/'evaluation/reused_manifest.json', reused)
    guard.write(ROOT/'selection.json', guard.read(HERE/'selection.json'))
    for name in ['pi0.json', 'pi05.json']:
        copy(HERE/name, ROOT/'protocols'/name)
    guard.write(ROOT/'preparation4090.json', dict(passed=True, textures=6, reused_episodes=1000,
                code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    archive = ROOT/'bootstrap09.tar.gz'
    if not archive.exists():
        with tarfile.open(archive.with_suffix('.tmp'), 'w:gz') as stream:
            stream.add(ROOT/'targets/pi0', arcname='targets/pi0')
            stream.add(ROOT/'selection.json', arcname='selection.json')
        archive.with_suffix('.tmp').replace(archive)
    print(json.dumps(dict(passed=True, bootstrap=str(archive), sha256=sha(archive))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=['4090', '09'], required=True)
    initialize(parser.parse_args().node)
