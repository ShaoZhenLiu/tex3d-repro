"""Transfer immutable completed A100 evaluation bundles to the canonical ledger."""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile
import time

import refcloth_resource as guard


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root, seen):
    jobs = {j['id']: j for j in guard.read(root / 'evaluation/jobs.json')
            if j.get('node') == '09'}
    selected = sorted(jid for jid in jobs if jid not in seen
                      and (root / 'evaluation/completed' / (jid + '.json')).exists())[:4]
    if not selected:
        return dict(pending=True)
    files = {}
    for jid in selected:
        job = jobs[jid]
        completed = Path('evaluation/completed') / (jid + '.json')
        info = guard.read(root / completed)
        assert info['completed'] and info['job'] == job
        files[str(completed)] = root / completed
        for row in info['outcomes']:
            name = '%s_t%d_i%d_%s.json' % (job['suite'], job['task'],
                                          row['initial'], row['condition'])
            path = Path('evaluation/episodes') / name
            episode = guard.read(root / path)
            assert episode['completed'] and episode['host'] == 'zgc-33-9'
            assert episode['robot_success'] == row['robot_success']
            files[str(path)] = root / path
            for filename in ('calibration.json', 'registered.json'):
                path = Path('evaluation/workers') / episode['worker'] / filename
                assert (root / path).is_file()
                files[str(path)] = root / path
    index = dict(jobs=selected, files={p: sha(f) for p, f in files.items()},
                 node='09', created_at=time.time())
    label = hashlib.sha256('\n'.join(selected).encode()).hexdigest()[:16]
    out = root / 'evaluation/exports09'
    out.mkdir(exist_ok=True)
    archive = out / (label + '.tar.gz')
    if not archive.exists():
        index_path = out / (label + '.json')
        guard.write(index_path, index)
        temporary = archive.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as stream:
            for name, path in sorted(files.items()):
                stream.add(str(path), arcname=name)
            stream.add(str(index_path), arcname='transfer_manifest.json')
        temporary.replace(archive)
    return dict(pending=False, jobs=selected, archive=str(archive), sha256=sha(archive))


def install(root, archive_path, expected):
    assert sha(archive_path) == expected
    jobs = {j['id']: j for j in guard.read(root / 'evaluation/jobs.json')}
    textures = guard.read(root / 'texture_manifest.json')
    with tarfile.open(archive_path) as stream:
        members = {m.name: m for m in stream if m.isfile()}
        index = json.load(stream.extractfile(members['transfer_manifest.json']))
        assert index['node'] == '09' and index['jobs']
        data = {}
        for name, digest in index['files'].items():
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[:2] in (('evaluation','episodes'), ('evaluation','completed'),
                                      ('evaluation','workers'))
            payload = stream.extractfile(members[name]).read()
            assert hashlib.sha256(payload).hexdigest() == digest
            data[path] = payload
    for jid in index['jobs']:
        assert jid in jobs and jobs[jid].get('node') == '09'
        completed = json.loads(data[Path('evaluation/completed') / (jid + '.json')])
        assert completed['completed'] and completed['job'] == jobs[jid]
        expected_rows = {(i, c) for i in range(10,20) for c in jobs[jid]['conditions']}
        assert {(r['initial'], r['condition']) for r in completed['outcomes']} == expected_rows
        for row in completed['outcomes']:
            path = Path('evaluation/episodes') / ('%s_t%d_i%d_%s.json' %
                (jobs[jid]['suite'], jobs[jid]['task'], row['initial'], row['condition']))
            episode = json.loads(data[path])
            assert episode['completed'] and episode['host'] == 'zgc-33-9'
            assert episode['robot_success'] == row['robot_success']
            assert episode['condition'] == row['condition'] and episode['initial'] == row['initial']
            assert episode['suite'] == jobs[jid]['suite'] and episode['task'] == jobs[jid]['task']
            assert episode['texture_sha256'] == textures[row['condition']]['sha256']
    with (root / 'evaluation/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ordered = sorted(data, key=lambda p: p.parts[1] == 'completed')
        for relative in ordered:
            target = root / relative
            if target.exists():
                assert target.read_bytes() == data[relative], target
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.from09')
            temporary.write_bytes(data[relative])
            temporary.replace(target)
        receipt = root / 'evaluation/imported09' / (expected + '.json')
        guard.write(receipt, dict(complete=True, archive_sha256=expected,
                                  jobs=index['jobs'], imported_at=time.time()))
    return dict(imported=index['jobs'], sha256=expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('export', 'import'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seen-base64', default='W10=')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.mode == 'export':
        result = export(args.root, set(json.loads(base64.b64decode(args.seen_base64))))
    else:
        result = install(args.root, args.archive, args.sha256)
    print(json.dumps(result))
