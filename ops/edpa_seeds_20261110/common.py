"""Fixed follow-up: four existing reference pairs and four EDPA source runs."""
import hashlib
import json
from pathlib import Path

REPO=Path('/data/shaozhen.liu/code/tex3d')
HERE=Path(__file__).resolve().parent
ROOT=REPO/'experiments/refcloth_edpa_seeds/20261110'
SEEDS=[42,7,66,88]
SUITES=['spatial','object','goal','long']
INITIALS=list(range(10,20))
TRAIN_NODES={7:'4090',66:'4090',88:'4090',42:'09'}
EVAL_NODES={s:('09' if s=='long' else '4090') for s in SUITES}
POOL=REPO/'experiments/pi0_cloth/oft_cloth/r0f_pool.pt'
POOL_SHA='d03d68591b9b820385d14be44758aefd95c2a63b8d68fc0926413a2eb1b04a93'
PRODUCTION=REPO/'openvla-oft/experiments/robot/libero/attack_oft_cloth_edpa_tvfix.py'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)

def jobs():
    rows=[]
    for seed in SEEDS:
        for suite in SUITES:
            for task in range(10):
                rows.append(dict(id='%s_t%d_s%d_edpa'%(suite,task,seed),target='pi05',
                    suite=suite,task=task,node=EVAL_NODES[suite],conditions=['s%d_edpa'%seed],
                    requires_existing=['%s_t%d_i%d_plain.json'%(suite,task,i) for i in INITIALS]))
    return rows

def source_root(seed):
    return REPO/'experiments/refcloth_stability'/('20261102' if seed==7 else '20261104')

def baseline_root(seed,suite):
    if suite in ['spatial','object']:return source_root(seed)
    return REPO/'experiments/refcloth_reference_coverage/20261106/suites'/suite
