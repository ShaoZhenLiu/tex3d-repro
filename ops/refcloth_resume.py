"""Recover full-precision iterates and preserve the exact prescribed sample stream."""
import ast
import hashlib
import re
from pathlib import Path


def select_checkpoint(training, tag):
    candidates = []
    for directory in Path(training).glob('train_*_' + tag):
        log = directory / 'gradient_log.txt'
        if not log.exists():
            continue
        saved = set(int(x) for x in re.findall(r'\[ckpt(?:-final)?\] it(\d+)', log.read_text()))
        for path in directory.glob('Ep0_Cloth_Delta_OV_it*.pt'):
            iteration = int(re.search(r'_it(\d+)\.pt$', path.name).group(1))
            if iteration in saved and path.stat().st_size:
                candidates.append((iteration, path.stat().st_mtime, path))
    return max(candidates)[2] if candidates else None


def _rows(directory, stop, seen=None):
    directory = Path(directory)
    seen = set() if seen is None else seen
    assert str(directory) not in seen, 'Resume chain cycle'
    seen.add(str(directory))
    options = ast.literal_eval((directory / 'run_config.txt').read_text().splitlines()[0])
    result = {}
    if options.get('resume_dir'):
        if options.get('resume_checkpoint'):
            first = 1 + int(re.search(r'_it(\d+)\.pt$', options['resume_checkpoint']).group(1))
        else:
            checkpoint = select_checkpoint(Path(options['resume_dir']).parent, Path(options['resume_dir']).name.split('_', 3)[-1])
            assert checkpoint is not None
            first = 1 + int(re.search(r'_it(\d+)\.pt$', checkpoint.name).group(1))
        result.update(_rows(options['resume_dir'], min(stop, first), seen))
    log = directory / 'sampling_log_segment.txt'
    if not log.exists():
        log = directory / 'sampling_log.txt'
    for row in log.read_text().splitlines():
        if row.startswith('#') or not row.strip():
            continue
        cols = row.split(); key = int(cols[0]), int(cols[1])
        if key[0] < stop:
            result[key] = row
    return result


def consolidate(directory, stream, iterations=5000, batch=8):
    directory = Path(directory)
    rows = _rows(directory, iterations)
    expected = [r for r in Path(stream).read_text().splitlines() if r.strip() and not r.startswith('#')]
    assert len(expected) == iterations * batch
    keys = [(i, s) for i in range(iterations) for s in range(batch)]
    assert sorted(rows) == keys, 'Missing or duplicate iteration slots in resumed stream'
    for key, prescribed in zip(keys, expected):
        assert rows[key].split() == prescribed.split(), (key, rows[key], prescribed)
    target = directory / 'sampling_log.txt'
    header = target.read_text().splitlines()[0]
    raw = directory / 'sampling_log_segment.txt'
    if not raw.exists():
        raw.write_bytes(target.read_bytes())
    target.write_text(header + '\n' + '\n'.join(rows[k] for k in keys) + '\n')
    return dict(full_sampling_log_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                raw_segment=str(raw), reconstructed_visits=len(keys), exact_prescribed_stream=True)
