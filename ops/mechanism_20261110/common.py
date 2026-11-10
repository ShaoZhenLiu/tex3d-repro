"""Frozen inputs and atomic records for the two mechanism measurements."""
import hashlib
import json
import os
from pathlib import Path

REPO=Path('/data/shaozhen.liu/code/tex3d')
HERE=Path(__file__).resolve().parent
ROOT=REPO/'experiments/refcloth_mechanism/20261110'
PROTOCOL=HERE/'protocol.json'


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
