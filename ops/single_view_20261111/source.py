"""Train paired camera ablations against the already verified source streams."""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path('/data/shaozhen.liu/code/tex3d')
ROOT = REPO / 'experiments/refcloth_single_view/20261111'
HERE = Path(__file__).resolve().parent
SEEDS = (42, 7, 66, 88)
sys.path.insert(0, str(REPO / 'ops'))
import refcloth_resource as guard
import refcloth_source_scheduler as scheduler


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze(source, target, expected):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.tmp')
        shutil.copyfile(source, temporary)
        assert sha(temporary) == expected, source
        temporary.replace(target)
    assert sha(target) == expected, target


def prepare(node):
    prior = REPO / 'experiments/refcloth_edpa_seeds/20261110'
    coupling = REPO / 'experiments/refcloth_coupling/20261102'
    preflight = prior / ('smoke_' + node) / 'prepared.json'
    calibration = guard.read(preflight)
    inputs = guard.read(prior / 'inputs.json')
    cache = guard.read(coupling / 'prepared.json')
    assert calibration['passed'] and calibration['frozen_wrist_gap_authorized']
    assert inputs['passed'] and cache['passed']
    assert calibration['pool_sha256'] == inputs['pool_sha256'] == cache['pool_sha256']
    assert cache['frames'] == 296 and cache['maximum_uniform_cloth_uv_pixel_difference'] < 1e-6
    for seed in SEEDS:
        record = inputs['streams'][str(seed)]
        assert record['rows'] == 40000
        for path_key, hash_key in (('text_path', 'text_sha256'), ('uv_path', 'uv_sha256')):
            name = record[path_key]
            freeze(prior / name, ROOT / name, record[hash_key])
    freeze(preflight, ROOT / 'source_preflight.json', sha(preflight))
    plain = ROOT / 'plain_image_cache.pt'
    cached_plain = coupling / plain.name
    if not plain.exists():
        plain.symlink_to(cached_plain)
    assert sha(plain) == cache['same_path_plain_cache_sha256']
    guard.write(ROOT / 'prepared.json', {
        'passed': True, 'paired_replay': inputs['streams'],
        'source_preflight_sha256': sha(preflight), 'pool_sha256': inputs['pool_sha256'],
        'same_path_plain_cache_sha256': cache['same_path_plain_cache_sha256'],
        'renderer_sha256': cache['renderer_sha256'],
        'protocol_sha256': sha(HERE / 'protocol.json'),
    })
    print('PAIRED_SOURCE_READY', node, flush=True)


def main(a):
    node = '4090' if os.uname().nodename.endswith('-41') else '09'
    assert a.node == node
    allowed = {0, 1, 2, 3, 5, 6, 7} if node == '4090' else {0, 1, 2, 3}
    assert a.gpus and set(a.gpus) <= allowed
    assert set(a.train_joint) <= set(a.seeds)
    if a.min_free_mib is not None:
        assert node == '4090' and a.min_free_mib >= 22500
    required = dict.fromkeys(a.gpus, a.min_free_mib) if a.min_free_mib else None
    prepare(node)
    jobs = [(seed, mode) for seed in a.seeds
            for mode in (('main', 'wrist', 'mr') if seed in a.train_joint else ('main', 'wrist'))]
    scheduler.main('single_view_20261111', SimpleNamespace(node=node, resume=True), root=ROOT,
                   jobs_override=jobs, gpus_override=a.gpus, camera_modes=True,
                   gpu_required_mib=required,
                   model_limit=6 if node == '09' else 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', required=True, choices=('4090', '09'))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--train-joint', nargs='+', type=int, choices=SEEDS, default=[])
    parser.add_argument('--min-free-mib', type=int)
    main(parser.parse_args())
