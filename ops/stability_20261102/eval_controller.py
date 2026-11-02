"""Publish completed textures and schedule persistent native evaluation workers."""
import json
import argparse
import hashlib
import os
import shutil
import subprocess
import time
import re
import fcntl
from pathlib import Path


R=Path('/data/shaozhen.liu/code/tex3d')
X=R/'experiments/refcloth_stability/20261102'
O=R/'ops/stability_20261102'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def publish():
    path=X/'texture_manifest.json';manifest=json.loads(path.read_text());changed=False
    for seed in [10,11,12]:
        for ref in ['native','mr']:
            name='s%d_%s'%(seed,ref);meta=X/'arms'/name/'completed.json'
            if name in manifest or not meta.exists():continue
            info=json.loads(meta.read_text());assert info['completed'] and info['iterations']==5000
            source=Path(info['texture']);dest=X/'frozen'/(name+'.png')
            if not source.exists():source=dest  # A100 results are transferred with their original provenance.
            if not source.exists():continue
            assert hashlib.sha256(source.read_bytes()).hexdigest()==info['texture_sha256']
            if source!=dest:shutil.copy2(source,dest)
            manifest[name]=dict(path=str(dest.relative_to(X)),sha256=info['texture_sha256'],
                seed=seed,reference=ref,training_metadata=str(meta.relative_to(X)),frozen_existing=False)
            changed=True;print('PUBLISHED',name,flush=True)
    if changed:write(path,manifest)
    for seed in [10,11,12]:
        pair=[X/'arms'/('s%d_%s'%(seed,ref))/'completed.json' for ref in ['native','mr']]
        if all(p.exists() for p in pair):
            left,right=[json.loads(p.read_text()) for p in pair]
            assert left['theta_init_sha256']==right['theta_init_sha256']
            assert left['sampling_log_sha256']==right['sampling_log_sha256']


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refcloth_eval_scheduler import entry
if __name__ == '__main__':
    entry('stability',publish)
