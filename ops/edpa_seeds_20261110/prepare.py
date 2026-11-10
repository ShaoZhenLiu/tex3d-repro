"""Verify existing observations/episodes and freeze all inputs before new training."""
import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

from common import HERE,REPO,ROOT,SEEDS,SUITES,INITIALS,POOL,POOL_SHA,read,write,sha,jobs,source_root,baseline_root


def copy(src,dst):
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),dst
    else:shutil.copyfile(src,dst)


def inputs():
    import numpy as np
    import torch
    sys.path.insert(0,str(REPO/'ops/stability_20261102'))
    import attack_oft_cloth_ablate as shared
    assert sha(POOL)==POOL_SHA
    for name in ['streams','arms','frozen','logs','baseline/episodes','evaluation/episodes','evaluation/completed','evaluation/workers','evaluation/claims']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    pool=torch.load(str(POOL),map_location='cpu',weights_only=False,mmap=True)
    active,_=shared.active_frames(pool,None);assert len(active)==296
    streams={}
    for seed in SEEDS:
        text=ROOT/'streams'/('seed_%d.txt'%seed);full=text.with_name(text.stem+'_full.pt')
        existing=read(source_root(seed)/'arms'/('s%d_mr'%seed)/'completed.json',{})
        if seed==7:
            legacy=REPO/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/sampling_log.txt'
            copy(legacy,text);replay=shared.parse_stream(str(text),5000,8)
            random.seed(seed);uvs=[];maximum_log_error=0.
            for it in range(5000):
                indices=[random.randrange(296) for _ in range(8)]
                uv=shared.M.sample_uv_xform(True,'cpu').numpy()
                assert indices==replay[it]['frames'],('legacy frame sequence',it)
                # The archived six-decimal text was parsed as float32. Its
                # decimal rounding plus float32 conversion can exceed 5e-7.
                error=float(np.max(np.abs(uv-replay[it]['uv'])))
                maximum_log_error=max(maximum_log_error,error)
                assert error<=1e-6,('legacy UV sequence',it,error)
                uvs.append(uv)
            if not full.exists():torch.save(dict(uv=uvs,seed=seed,n_active=296),full)
            loaded=torch.load(full,map_location='cpu',weights_only=False)['uv']
            assert all(np.array_equal(x,y) for x,y in zip(loaded,uvs))
            origin='Seed-7 full precision reconstructed from its original PRNG and verified against every archived frame/rounded-UV row'
        else:
            original=source_root(seed)/'streams'/text.name
            copy(original,text);copy(original.with_name(full.name),full)
            assert existing['stream']['text_sha256']==sha(text)
            assert existing['stream']['uv_sha256']==sha(full)
            origin='Byte-identical copy of the Native/Matched paired source stream'
        streams[str(seed)]=dict(text_path=str(text.relative_to(ROOT)),uv_path=str(full.relative_to(ROOT)),
            text_sha256=sha(text),uv_sha256=sha(full),rows=40000,origin=origin,
            theta_init_sha256=existing.get('theta_init_sha256'))
        if seed==7:streams[str(seed)]['maximum_archived_uv_log_error']=maximum_log_error
    plain=read(source_root(7)/'texture_manifest.json')['plain']
    copy(source_root(7)/plain['path'],ROOT/'frozen/plain.png')
    write(ROOT/'texture_manifest.json',dict(plain=dict(path='frozen/plain.png',sha256=plain['sha256'],reference='plain')))
    write(ROOT/'evaluation/jobs.json',jobs())
    rows=[];refs={}
    for suite in SUITES:
        shared_root=baseline_root(7,suite)
        for task in range(10):
            for initial in INITIALS:
                plain_name='%s_t%d_i%d_plain.json'%(suite,task,initial)
                plain_path=shared_root/'evaluation/episodes'/plain_name
                d=read(plain_path);assert d['completed'] and d['texture_sha256']==plain['sha256']
                assert d['host'].endswith('-41') if suite!='long' else d['host']=='zgc-33-9'
                copy(plain_path,ROOT/'evaluation/episodes'/plain_name)
                refs[plain_name]=dict(source=str(plain_path),sha256=sha(plain_path),method='plain')
                rows.append(dict(seed='',method='plain',suite=suite,task=task,initial=initial,success=int(d['robot_success']),file=plain_name))
                for seed in SEEDS:
                    origin=baseline_root(seed,suite)
                    assert sha(origin/'evaluation/episodes'/plain_name)==sha(plain_path)
                    manifest=read(origin/'texture_manifest.json')
                    for method in ['native','mr']:
                        name='%s_t%d_i%d_s%d_%s.json'%(suite,task,initial,seed,method)
                        path=origin/'evaluation/episodes'/name;e=read(path)
                        assert e['completed'] and e['texture_sha256']==manifest['s%d_%s'%(seed,method)]['sha256']
                        assert e['host'].endswith('-41') if suite!='long' else e['host']=='zgc-33-9'
                        if e['trajectory'] and d['trajectory']:
                            assert e['trajectory'][0]['proprio_sha256']==d['trajectory'][0]['proprio_sha256']
                        assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(e['trajectory'],d['trajectory']))
                        copy(path,ROOT/'baseline/episodes'/name)
                        refs[name]=dict(source=str(path),sha256=sha(path),method=method,seed=seed)
                        rows.append(dict(seed=seed,method=method,suite=suite,task=task,initial=initial,success=int(e['robot_success']),file=name))
    assert len(rows)==3600 and len(refs)==3600
    with (ROOT/'baseline/outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method','suite','task','initial','success','file']);writer.writeheader();writer.writerows(rows)
    write(ROOT/'baseline/manifest.json',refs)
    write(ROOT/'inputs.json',dict(passed=True,seeds=SEEDS,pool_sha256=POOL_SHA,streams=streams,
        baseline_reference_episodes=3200,baseline_plain_episodes=400,baseline_csv_sha256=sha(ROOT/'baseline/outcomes.csv'),
        protocol_sha256=sha(HERE/'protocol.json')))
    archive=ROOT/'bootstrap09.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['streams','frozen','inputs.json','texture_manifest.json','evaluation/jobs.json','evaluation/episodes','baseline/outcomes.csv','baseline/manifest.json']:
            tar.add(ROOT/name,arcname=name)
    print(json.dumps(dict(passed=True,bootstrap=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)),flush=True)


def smoke(node):
    assert read(ROOT/'inputs.json',{}).get('passed')
    cfg=read(HERE/'protocol.json');cfg['new_training_seeds']=[]
    path=ROOT/('smoke_'+node)/'protocol_smoke.json';write(path,cfg)
    spec=importlib.util.spec_from_file_location('_edpa_frozen_smoke',REPO/'ops/stability_20261102/prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(SimpleNamespace(repo=REPO,protocol=path,out=path.parent,verify_only=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inputs','smoke']);p.add_argument('--node',choices=['4090','09'],required=True)
    a=p.parse_args()
    if a.stage=='inputs':assert a.node=='4090';inputs()
    else:smoke(a.node)
