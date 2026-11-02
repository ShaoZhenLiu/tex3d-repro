"""Freeze controls, project independent textures, and reuse keyed source cases."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


R=Path('/data/shaozhen.liu/code/tex3d')
X=R/'experiments/refcloth_coupling/20261102'
P=R/'experiments/refcloth_stability/20261102'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def initialize():
    for name in ['frozen','compositions','evaluation/episodes','evaluation/claims','evaluation/completed','evaluation/workers']:
        (X/name).mkdir(parents=True,exist_ok=True)
    if not (X/'texture_manifest.json').exists():write(X/'texture_manifest.json',{})
    if not (X/'protocol.json').exists():shutil.copy2(str(R/'ops/coupling_20261102/protocol.json'),str(X/'protocol.json'))
    if not (X/'evaluation/jobs.json').exists():
        jobs=[]
        for suite in ['spatial','object']:
            for task in range(10):
                for seed in [10,11,12]:
                    for variant in ['uv_ind','consensus_raw','consensus_rms']:
                        shared='s%d_mr'%seed;other='s%d_%s'%(seed,variant)
                        jobs.append(dict(id='%s_t%d_%s'%(suite,task,other),suite=suite,task=task,
                            conditions=[shared,other],requires_existing=[
                                '%s_t%d_i%d_%s.json'%(suite,task,i,shared) for i in range(10,20)]))
        assert len(jobs)==180;write(X/'evaluation/jobs.json',jobs)


def consensus(seed,manifest):
    shared='s%d_mr'%seed;left='s%d_main'%seed;right='s%d_wrist'%seed
    metadata=[X/'arms'/name/'completed.json' for name in [left,right]]
    if shared not in manifest or not all(p.exists() for p in metadata):return False
    record=X/'compositions'/('s%d.json'%seed)
    if record.exists():
        restored=False
        for suffix,item in json.loads(record.read_text())['textures'].items():
            name='s%d_%s'%(seed,suffix)
            assert sha(X/item['path'])==item['sha256']
            if name not in manifest:manifest[name]=item;restored=True
        return restored
    arms=[json.loads(p.read_text()) for p in metadata]
    for name,meta in zip([left,right],arms):
        png=X/'frozen'/(name+'.png');assert png.exists() and sha(png)==meta['texture_sha256']
    base=np.asarray(Image.open(X/'frozen/plain.png').convert('RGB')).astype(float)
    a,b=[np.asarray(Image.open(X/'frozen'/(name+'.png')).convert('RGB')).astype(float) for name in [left,right]]
    shared_rgb=np.asarray(Image.open(X/manifest[shared]['path']).convert('RGB')).astype(float)
    mean=(a+b)*.5;delta=mean-base
    assert np.abs(a-base).max()<=64 and np.abs(b-base).max()<=64
    target=float(np.sqrt(np.mean((shared_rgb-base)**2)))
    def render(scale):return np.rint(np.clip(base+np.clip(scale*delta,-64,64),0,255)).astype(np.uint8)
    def rms(rgb):return float(np.sqrt(np.mean((rgb.astype(float)-base)**2)))
    raw=render(1.);upper=256.;attainable=rms(render(upper))>=target
    if attainable:
        low=0.
        for _ in range(60):
            middle=(low+upper)*.5
            if rms(render(middle))<target:low=middle
            else:upper=middle
        scale=min([low,upper],key=lambda c:abs(rms(render(c))-target))
    else:scale=upper
    normalized=render(scale)
    projection=float(np.mean((mean-a)**2+(mean-b)**2))
    assert abs(projection-.5*float(np.mean((a-b)**2)))<1e-9
    result=dict(seed=seed,completed=True,source_arms=arms,
        method='Equal-view Euclidean consensus in UV texture space, plus a separate amplitude control',
        raw_rms=rms(raw),shared_rms=target,scaled_rms=rms(normalized),scale=scale,
        rms_attainable=bool(attainable),rms_error=abs(rms(normalized)-target),
        independent_disagreement_rms=float(np.sqrt(np.mean((a-b)**2))),
        prequantization_projection_objective=projection,
        raw_export_projection_objective=float(np.mean((raw-a)**2+(raw-b)**2)),
        textures={})
    if attainable:assert result['rms_error']<.025,result
    for suffix,rgb in [('consensus_raw',raw),('consensus_rms',normalized)]:
        name='s%d_%s'%(seed,suffix);path=X/'frozen'/(name+'.png')
        assert abs(rgb.astype(float)-base).max()<=64
        Image.fromarray(rgb).save(path)
        manifest[name]=dict(path=str(path.relative_to(X)),sha256=sha(path),seed=seed,reference=suffix)
        result['textures'][suffix]=manifest[name]
    write(record,result);return True


def publish():
    initialize();manifest=json.loads((X/'texture_manifest.json').read_text());old=json.loads((P/'texture_manifest.json').read_text())
    changed=False
    for name in ['plain']+['s%d_mr'%s for s in [10,11,12]]:
        if name in manifest or name not in old:continue
        source=P/old[name]['path'];dest=X/'frozen'/(name+'.png');assert sha(source)==old[name]['sha256']
        shutil.copy2(str(source),str(dest));manifest[name]=dict(old[name],path=str(dest.relative_to(X)),reused_from=str(source));changed=True
    for seed in [10,11,12]:
        for mode in ['uv_ind','main','wrist']:
            name='s%d_%s'%(seed,mode);meta=X/'arms'/name/'completed.json'
            if not meta.exists():continue
            info=json.loads(meta.read_text());assert info['completed'] and info['iterations']==5000 and info['frame_visits']==40000
            source=Path(info['texture']);dest=X/'frozen'/(name+'.png')
            if not source.exists():source=dest
            assert source.exists() and sha(source)==info['texture_sha256']
            if source!=dest and (not dest.exists() or sha(dest)!=info['texture_sha256']):
                temp=dest.with_suffix('.incoming');shutil.copy2(str(source),str(temp));temp.replace(dest)
            if mode=='uv_ind' and name not in manifest:
                manifest[name]=dict(path=str(dest.relative_to(X)),sha256=info['texture_sha256'],seed=seed,reference=mode);changed=True
        if consensus(seed,manifest):changed=True
    if changed:write(X/'texture_manifest.json',manifest)
    # Imported cases retain their original worker and complete query traces.
    for source in (P/'evaluation/episodes').glob('*.json'):
        if not (source.stem.endswith('_plain') or any(source.stem.endswith('_s%d_mr'%s) for s in [10,11,12])):continue
        dest=X/'evaluation/episodes'/source.name
        if not dest.exists():
            temp=dest.with_suffix('.incoming');shutil.copy2(str(source),str(temp));temp.replace(dest)
    return manifest


if __name__=='__main__':
    manifest=publish();print('Published',len(manifest),'analysis textures')
