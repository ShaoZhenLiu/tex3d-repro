"""Freeze all 36 final exports and exactly reuse 700 observations per suite."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from common import HERE, REPO, ROOT, FULL, SEEDS, INITIALS, REUSED_SEEDS, jobs, sha, source_root, suite_root, old_suite_root
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),dst
    else:shutil.copyfile(src,dst)


def prepare(node):
    assert (os.uname().nodename.endswith('-41') if node=='4090' else os.uname().nodename=='zgc-33-9')
    if node=='09':
        root=suite_root('long');manifest=guard.read(root/'texture_manifest.json')
        assert len(manifest)==37 and guard.read(root/'evaluation/jobs.json')==jobs('long')
        assert len(guard.read(root/'evaluation/reused_manifest.json'))==700
        for v in manifest.values():assert sha(root/v['path'])==v['sha256']
        assert guard.read(root/'protocol.json')==guard.read(HERE/'long.json')
        guard.write(ROOT/'prepared09.json',dict(passed=True,time=time.time()))
        return
    assert not any(str(ROOT) in e['root'] for e in guard.active())
    legacy=guard.read(ROOT/'inputs/legacy_training_provenance.json')
    for suite in ['goal','long']:
        root=suite_root(suite)
        for d in ['frozen','provenance','calibration','evaluation/episodes','evaluation/completed','evaluation/workers','evaluation/claims']:
            (root/d).mkdir(parents=True,exist_ok=True)
        manifest={}
        for seed in SEEDS:
            original=source_root(seed);previous=guard.read(original/'texture_manifest.json')
            for ref in ['native','mr']:
                key='s%d_%s'%(seed,ref);info=previous[key]
                assert sha(original/info['path'])==info['sha256']
                copy(original/info['path'],root/'frozen'/(key+'.png'))
                meta=original/'arms'/key/'completed.json'
                if meta.exists():
                    v=guard.read(meta);assert v['completed'] and v['iterations']==5000 and v['frame_visits']==40000
                    assert v['texture_sha256']==info['sha256']
                    copy(meta,root/'provenance'/(key+'.json'))
                else:
                    tag='tvfix-bs8' if seed==7 and ref=='native' else ('tvfix-mr' if seed==7 else 'e05_s%d_%s'%(seed,ref))
                    records=[v for v in legacy if v['directory'].endswith('_'+tag)]
                    assert len(records)==1,(key,tag)
                    cfg=ast.literal_eval(records[0]['config'].splitlines()[0])
                    assert cfg['seed']==seed and cfg['iters']==5000 and cfg['batch']==8
                    guard.write(root/'provenance'/(key+'.json'),dict(legacy_record=records[0],texture_sha256=info['sha256']))
                manifest[key]=dict(info,path='frozen/'+key+'.png',frozen_existing=True,original_root=str(original))
        plain_source=source_root(7);plain=guard.read(plain_source/'texture_manifest.json')['plain']
        copy(plain_source/plain['path'],root/'frozen/plain.png')
        manifest['plain']=dict(plain,path='frozen/plain.png',frozen_existing=True)
        guard.write(root/'texture_manifest.json',manifest)
        guard.write(root/'evaluation/jobs.json',jobs(suite))
        copy(HERE/(suite+'.json'),root/'protocol.json')
        copy(FULL/'targets/pi05/evaluation/episodes/spatial_t0_i10_plain.json',root/'calibration/reference_plain_spatial_t0_i10.json')
        old=old_suite_root(suite);reused={}
        for job in jobs(suite):
            if job['conditions']!=['plain'] and not any(job['conditions'][0].startswith('s%d_'%s) for s in REUSED_SEEDS):continue
            rows=[]
            for initial in INITIALS:
                for condition in job['conditions']:
                    name='%s_t%d_i%d_%s.json'%(suite,job['task'],initial,condition)
                    file=old/'evaluation/episodes'/name;e=guard.read(file)
                    assert e['completed'] and e['texture_sha256']==manifest[condition]['sha256']
                    assert (e['host'].endswith('-41') if suite=='goal' else e['host']=='zgc-33-9')
                    for f in ['registered.json','calibration.json']:
                        copy(old/'evaluation/workers'/e['worker']/f,root/'evaluation/workers'/e['worker']/f)
                    copy(file,root/'evaluation/episodes'/name)
                    reused[name]=dict(source=str(file),sha256=sha(file))
                    rows.append(dict(initial=initial,condition=condition,robot_success=e['robot_success']))
            guard.write(root/'evaluation/completed'/(job['id']+'.json'),dict(completed=True,job=job,outcomes=rows,reused=True))
        assert len(reused)==700
        guard.write(root/'evaluation/reused_manifest.json',reused)
    for name in ['existing_spatial_object_outcomes.csv','existing_sources.json']:
        copy(ROOT/'inputs'/name,ROOT/name)
    guard.write(ROOT/'plan.json',dict(seeds=SEEDS,suites=['goal','long'],target='pi05',
        formal_extension_episodes=7400,new_episodes=6000,reused_episodes=1400,
        combined_four_suite_episodes=14800,all_four_suites_retained=True,
        suite_pair_contrasts='Exploratory; no outcome-based endpoint replacement',
        assignment={'goal':'4090','long':'09'},gpu_ids=[0,1,2,3]))
    archive=ROOT/'bootstrap09.tar.gz'
    if not archive.exists():
        with tarfile.open(str(archive.with_suffix('.tmp')),'w:gz') as stream:
            stream.add(str(suite_root('long')),arcname='suites/long')
            for name in ['plan.json','existing_spatial_object_outcomes.csv','existing_sources.json']:
                stream.add(str(ROOT/name),arcname=name)
        archive.with_suffix('.tmp').replace(archive)
    guard.write(ROOT/'prepared4090.json',dict(passed=True,time=time.time()))
    print(json.dumps(dict(passed=True,archive=str(archive),sha256=sha(archive),new=6000,reused=1400)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['4090','09'],required=True)
    prepare(p.parse_args().node)
