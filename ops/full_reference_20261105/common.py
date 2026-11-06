"""Immutable study configuration and job construction; no model imports."""
import hashlib
import json
from pathlib import Path

REPO = Path('/data/shaozhen.liu/code/tex3d')
HERE = Path(__file__).resolve().parent
ROOT = REPO/'experiments/refcloth_full_reference/20261105'
LONG_ROOT = ROOT/'segments/pi05_long_09'
SEEDS = [3, 12, 10]
SUITES = ['spatial', 'object', 'goal', 'long']
INITIALS = list(range(10, 20))
NODES = {'pi05': '4090', 'pi0': '09'}
HOSTS = {'4090': 'hr-RTX4090-132-41', '09': 'zgc-33-9'}
PEERS = {'4090': 'shaozhen.liu@172.18.132.41', '09': 'shaozhen.liu@172.18.33.9'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs(target):
    node = NODES[target]
    result = []
    for suite in SUITES:
        for task in range(10):
            base = dict(target=target, suite=suite, task=task, node=node)
            result.append(dict(base, id='%s_t%d_plain' % (suite, task), conditions=['plain']))
            keys = ['%s_t%d_i%d_plain.json' % (suite, task, initial) for initial in INITIALS]
            for seed in SEEDS:
                result.append(dict(base, id='%s_t%d_s%d' % (suite, task, seed),
                    conditions=['s%d_native' % seed, 's%d_mr' % seed], requires_existing=keys))
    return result


def compatible_reuse(old, new):
    """Preserve every inference setting and the two original suite/noise indices."""
    keys = ['target', 'config', 'checkpoint', 'tasks', 'initials', 'max_steps', 'settle_steps',
            'replan_steps', 'num_denoise', 'surface_mode', 'floor_mode', 'noise']
    return (all(old[key] == new[key] for key in keys)
            and new['suites'][:len(old['suites'])] == old['suites'])


def source_root(seed):
    date = '20261105' if seed == 3 else '20261102'
    return REPO/'experiments/refcloth_stability'/date


def episode_name(suite, task, initial, condition):
    return '%s_t%d_i%d_%s.json' % (suite, task, initial, condition)


def long_jobs():
    return [dict(row, node='09') for row in jobs('pi05') if row['suite'] == 'long']


def migration_enabled():
    return (ROOT/'long_migration_20261106.json').is_file()


def evaluation_sources(target, migrated=None):
    if migrated is None:
        migrated = migration_enabled()
    if target == 'pi05' and migrated:
        return [(ROOT/'targets/pi05', ['spatial', 'object', 'goal'], '4090'),
                (LONG_ROOT, ['long'], '09')]
    return [(ROOT/'targets'/target, SUITES, NODES[target])]
