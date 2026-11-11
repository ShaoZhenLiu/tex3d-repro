"""Locations and the fixed seed/hardware assignments for this follow-up."""
import hashlib
import os
import sys
from pathlib import Path

REPO = Path('/data/shaozhen.liu/code/tex3d')
ROOT = REPO / 'experiments/refcloth_camera_mechanism/20261111'
PRIOR = REPO / 'experiments/refcloth_single_view/20261111'
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'evaluation_protocol.json'
STUDY = 'camera_mechanism_20261111'
SEEDS = {'09': (42, 7), '4090': (66, 88)}
GPUS = {'09': (0, 1, 2, 3), '4090': (6, 5, 7, 3, 0, 1, 2)}
TASKS = tuple(range(10))
INITIALS = tuple(range(10, 20))
SOURCE_STEPS = (0, 1875, 3750, 5000)
sys.path.insert(0, str(REPO / 'ops'))
import refcloth_resource as guard


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def local_node():
    host = os.uname().nodename
    assert host.endswith('-41') or host == 'zgc-33-9'
    return '4090' if host.endswith('-41') else '09'


def controls(seeds):
    return ['plain'] + [f's{seed}_{view}' for seed in seeds for view in ('main', 'wrist')]


def new_conditions(seed):
    return [f's{seed}_{suffix}' for suffix in ('ap', 'pa', 'main_low')]


def episode_name(task, initial, condition):
    return f'object_t{task}_i{initial}_{condition}.json'
