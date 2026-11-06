"""Fixed all-start, two-suite extension of the existing reference comparison."""
import hashlib
from pathlib import Path

REPO = Path('/data/shaozhen.liu/code/tex3d')
HERE = Path(__file__).resolve().parent
ROOT = REPO/'experiments/refcloth_reference_coverage/20261106'
FULL = REPO/'experiments/refcloth_full_reference/20261105'
SEEDS = [3,7,8,9,10,11,12,19,36,42,51,52,55,58,63,66,88,99]
INITIALS = list(range(10,20))
NODES = {'goal':'4090','long':'09'}
PEERS = {'4090':'shaozhen.liu@172.18.132.41','09':'shaozhen.liu@172.18.33.9'}
REUSED_SEEDS = [3,10,12]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_root(seed):
    date = '20261102' if seed in [7,8,9,10,11,12] else ('20261104' if seed in [66,88,99,42] else '20261105')
    return REPO/'experiments/refcloth_stability'/date


def jobs(suite):
    rows=[]
    for task in range(10):
        base=dict(target='pi05',suite=suite,task=task,node=NODES[suite])
        rows.append(dict(base,id='%s_t%d_plain'%(suite,task),conditions=['plain']))
        requirements=['%s_t%d_i%d_plain.json'%(suite,task,i) for i in INITIALS]
        for seed in SEEDS:
            rows.append(dict(base,id='%s_t%d_s%d'%(suite,task,seed),
                conditions=['s%d_native'%seed,'s%d_mr'%seed],requires_existing=requirements))
    return rows


def suite_root(suite):
    return ROOT/'suites'/suite


def old_suite_root(suite):
    return FULL/('targets/pi05' if suite=='goal' else 'segments/pi05_long_09')
