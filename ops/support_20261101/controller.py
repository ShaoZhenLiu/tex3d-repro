"""Two persistent GPU workers; adaptive queries remain source-only.

The controller serializes accepted deletions and freezes all masks before
opening the held-out source split. No target-policy imports occur here.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from support_core import (array_sha,binary_texture,opacity_texture,texture_stats,
                          best_deletion,pooled_metrics)


def save(obj,path):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def environment(repo,gpu):
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
        PYTHONPATH=':'.join(map(str,[repo,repo/'openvla-oft',repo/'openvla-oft/experiments/robot/libero',
                                    repo/'ops/math_20261031',Path(__file__).parent])))
    return e


def preflight(gpus,min_free):
    assert 4 not in gpus
    v=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in v.strip().splitlines()}
    mem=Path('/proc/meminfo').read_text().splitlines()
    available=next(int(x.split()[1]) for x in mem if x.startswith('MemAvailable:'))/1024**2
    assert available>=40, 'Insufficient available host RAM for two workers'
    for g in gpus:
        assert free[g]>=min_free, 'GPU %s only has %s MiB free' % (g,free[g])
    return dict(gpu_free_mib=free,host_available_gib=available)


def main(repo,out,protocol_path):
    out.mkdir(parents=True,exist_ok=False)
    p=read(protocol_path)
    gpus=p['gpus']['allowed_for_this_run']
    audit=preflight(gpus,p['gpus']['min_free_mib_before_start'])
    audit['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    audit['protocol_sha256']=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    audit['utc_start']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    py='/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'
    base_env=environment(repo,gpus[0])
    patch_code="from residual_lab import adapters as A; import json; s=A.TR.patch_status(); A.TR.ensure_patch(True); print('PATCH_OWNER_JSON='+json.dumps({'initial':s,'current':A.TR.patch_status()}))"
    patch=subprocess.check_output([py,'-c',patch_code],cwd=repo,env=base_env,text=True)
    owner=json.loads(next(x.split('=',1)[1] for x in patch.splitlines() if x.startswith('PATCH_OWNER_JSON=')))
    audit['patch_owner']=owner
    save(audit,out/'audit.json')
    procs=[];logs=[]
    seq=-1
    def alive():
        for i,proc in enumerate(procs):
            if proc.poll() is not None:
                raise RuntimeError('Worker %d exited %s; inspect worker%d.log' % (i,proc.returncode,i))
    def wait_for(paths,predicate=lambda values:True,timeout=1200):
        start=time.monotonic()
        while True:
            alive()
            if all(x.exists() for x in paths):
                vals=[read(x) for x in paths]
                if predicate(vals): return vals
            if time.monotonic()-start>timeout: raise TimeoutError('Timed out waiting for native workers')
            time.sleep(.15)
    def request(op='query',candidates=None):
        nonlocal seq
        seq+=1
        req=dict(seq=seq,op=op)
        if candidates is not None: req['candidates']=candidates
        save(req,out/'request.json')
        vals=wait_for([out/('response%d.json'%i) for i in range(2)],
                      lambda values:all(v['seq']==seq for v in values))
        save(dict(request=req,responses=vals),out/('queries/q%03d.json'%seq))
        return vals
    try:
        (out/'queries').mkdir()
        for i,gpu in enumerate(gpus):
            log=open(out/('worker%d.log'%i),'w')
            logs.append(log)
            proc=subprocess.Popen([py,str(Path(__file__).parent/'worker.py'),'--repo',str(repo),
                '--out',str(out),'--id',str(i),'--protocol',str(protocol_path)],cwd=repo,
                env=environment(repo,gpu),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            procs.append(proc)
        audit['workers']=[dict(pid=x.pid,gpu=g) for x,g in zip(procs,gpus)]
        save(audit,out/'audit.json')
        ready=wait_for([out/('ready%d.json'%i) for i in range(2)])
        assert len({r['full_pixel_sha256'] for r in ready})==1
        assert len({r['base_pixel_sha256'] for r in ready})==1
        fit_meta=sum([r['states'] for r in ready],[])
        save(fit_meta,out/'fit_states.json')
        n=p['grid']**2
        keep=tuple(range(n));path=[];loo=None
        for iteration in range(n-min(p['kept_blocks'])):
            candidates=[dict(name='delete%d_from_%04x'%(j,sum(1<<b for b in keep)),
                             keep=[b for b in keep if b!=j],removed=j) for j in keep]
            vals=request(candidates=candidates)
            rows=sum([v['rows'] for v in vals],[])
            byname={c['name']:[r for r in rows if r['candidate']==c['name']] for c in candidates}
            loss={c['removed']:sum(r['pose_error_sum'] for r in byname[c['name']])/len(fit_meta) for c in candidates}
            if loo is None:
                loo=[dict(block=c['removed'],loss=loss[c['removed']],rows=byname[c['name']]) for c in candidates]
                save(loo,out/'leave_one_out_fit.json')
            keep,removed=best_deletion(keep,loss)
            path.append(dict(iteration=iteration,removed=removed,keep=list(keep),loss=loss[removed],all_deletions=loss))
            save(path,out/'greedy_path.json')
            print('GREEDY',len(keep),'removed',removed,'pose_sum_error',loss[removed],flush=True)
        # All candidate definitions are frozen before source held-out data.
        ranking=[r['block'] for r in sorted(loo,key=lambda r:(-r['loss'],r['block']))]
        visibility=np.array([s['visibility'] for s in fit_meta]).mean(axis=(0,2))
        vrank=sorted(range(n),key=lambda j:(-visibility[j],j))
        candidates=[dict(name='full',family='full',keep=list(range(n))),
                    dict(name='plain',family='plain',keep=[])]
        for k in p['kept_blocks']:
            chosen=next(r['keep'] for r in path if len(r['keep'])==k)
            candidates.extend([dict(name='greedy_%02d'%k,family='greedy',keep=chosen),
                dict(name='ranking_%02d'%k,family='ranking',keep=ranking[:k]),
                dict(name='visibility_%02d'%k,family='visibility',keep=vrank[:k])])
            for seed in p['random_seeds']:
                order=np.random.RandomState(seed).permutation(n).tolist()
                candidates.append(dict(name='random%d_%02d'%(seed,k),family='random',seed=seed,keep=order[:k]))
            candidates.append(dict(name='opacity_%02d'%k,family='opacity',opacity=k/n))
        frozen=dict(candidates=candidates,fit_visibility=visibility.tolist(),fit_loo_ranking=ranking,
                    fit_visibility_ranking=vrank,source_validation_not_yet_opened=True)
        save(frozen,out/'frozen_candidates.json')
        full=np.array(Image.open(repo/p['texture']).convert('RGB'))
        base=np.array(Image.open('/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/assets/tablecloth/cloth_base.png').convert('RGB'))
        assert array_sha(full)==ready[0]['full_pixel_sha256']
        assert array_sha(base)==ready[0]['base_pixel_sha256']
        artifacts=[]
        (out/'textures').mkdir()
        for c in candidates:
            png=(binary_texture(full,base,c['keep'],p['grid']) if 'keep' in c else opacity_texture(full,base,c['opacity']))
            texpath=out/'textures'/(c['name']+'.png')
            Image.fromarray(png).save(texpath)
            artifacts.append(dict(c,**texture_stats(png,base,c.get('keep'),p['grid']),
                                  file_sha256=hashlib.sha256(texpath.read_bytes()).hexdigest()))
        save(artifacts,out/'texture_manifest.json')
        fit_results=request(candidates=candidates)
        save(sum([v['rows'] for v in fit_results],[]),out/'fit_candidates.json')
        val=request('validation')
        val_meta=sum([v['states'] for v in val],[])
        save(val_meta,out/'validation_states.json')
        # Validation leave-one-out is diagnostic only and never revises masks.
        val_candidates=candidates+[dict(name='loo_%02d'%j,family='loo',keep=[b for b in range(n) if b!=j]) for j in range(n)]
        values=request(candidates=val_candidates)
        save(sum([v['rows'] for v in values],[]),out/'validation_candidates.json')
        save(dict(completed=True,utc_end=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  peak_cuda_bytes=[v['peak_cuda_bytes'] for v in values]),out/'completed.json')
    finally:
        save(dict(seq=999999,op='stop'),out/'request.json')
        for proc in procs:
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
        for log in logs: log.close()
        if owner['initial']=='clean':
            subprocess.check_call([py,'-c','from residual_lab import adapters as A; A.TR.ensure_patch(False)'],cwd=repo,env=base_env)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args()
    main(args.repo,args.out,args.protocol)
