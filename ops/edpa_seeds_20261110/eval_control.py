"""Use the established paired native evaluator as soon as an EDPA export exists."""
import argparse
import sys
from types import SimpleNamespace
from common import HERE,REPO,ROOT,SEEDS,read,write,sha
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard
import refcloth_eval_scheduler as scheduler


def publish():
    manifest=read(ROOT/'texture_manifest.json') or {}
    if 'plain' not in manifest:
        plain=ROOT/'frozen/plain.png'
        expected=read(ROOT/'evaluation/episodes/spatial_t0_i10_plain.json')['texture_sha256']
        assert sha(plain)==expected
        manifest['plain']=dict(path='frozen/plain.png',sha256=expected,reference='plain')
    for seed in SEEDS:
        e=read(ROOT/'arms'/('s%d_edpa'%seed)/'completed.json',{})
        if not e.get('completed'):continue
        path=ROOT/'frozen'/('s%d_edpa.png'%seed)
        assert sha(path)==e['texture_sha256']
        manifest['s%d_edpa'%seed]=dict(path=str(path.relative_to(ROOT)),sha256=e['texture_sha256'],seed=seed,reference='cloth_edpa')
    guard.write(ROOT/'texture_manifest.json',manifest)


def main(node):
    scheduler.main('edpa_seeds_20261110',publish,
        SimpleNamespace(gpus=[0,1,2,3],share_gpu3=False,resume=True),root=ROOT,node=node,
        allow_stochastic_calibration=node=='09',evaluation_protocol=HERE/'protocol.json',
        reused_episode_count=0,memory_budget_mib=18000 if node=='09' else None)
    write(ROOT/('evaluation_done_'+node+'.json'),dict(completed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090','09'],required=True);main(p.parse_args().node)
