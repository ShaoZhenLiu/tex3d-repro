"""Common-noise native evaluation of every planned frozen reference pair."""
import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False));temp.replace(path)


def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','5','6','7']
    torch.set_num_threads(2);torch.manual_seed(7)
    torch.cuda.set_per_process_memory_fraction(.44,0)
    import attack_pi0_cloth as A
    import attack_pi0 as B
    import mujoco
    from libero.libero import benchmark
    protocol=json.loads((a.repo/'ops/stability_20261102/protocol.json').read_text())
    assert A.patch_status()=='patched' and os.environ.get('REFCLOTH_SURFACE_MODE','legacy')=='legacy'
    assert mujoco.__version__=='3.3.0'
    p=protocol['evaluation']
    policy=A.load_policy(p['checkpoint'],model_config=p['config'],num_denoise=p['num_denoise'])
    policy._model.eval().requires_grad_(False)
    cfg=A.training_config.get_config(p['config']).model
    noise_shape=(cfg.action_horizon,cfg.action_dim)
    worker=a.root/'evaluation/workers'/a.worker;worker.mkdir(parents=True,exist_ok=True)
    outcomes=a.root/'evaluation/episodes';outcomes.mkdir(parents=True,exist_ok=True)
    complete=a.root/'evaluation/completed';complete.mkdir(parents=True,exist_ok=True)
    claims=a.root/'evaluation/claims';claims.mkdir(parents=True,exist_ok=True)
    suites={name:benchmark.get_benchmark_dict()['libero_'+name]() for name in p['suites']}
    current={};original_env=A._get_libero_env;original_inject=A.inject_mujoco_texture
    def remember(*args,**kwargs):
        env,desc=original_env(*args,**kwargs);current['env']=env;return env,desc
    def inject(sim,tid,png):
        saved=original_inject(sim,tid,png)
        assert B._upload_mujoco_texture(sim,tid),'Native texture upload did not reach the renderer'
        current['uploads']=current.get('uploads',0)+1
        return saved
    A._get_libero_env=remember;A.inject_mujoco_texture=inject
    class Paired:
        def reset(self,suite,task,initial):
            self.suite,self.task,self.initial=suite,task,initial;self.rows=[]
        def infer(self,raw):
            q=len(self.rows)
            seed=6810000+p['suites'].index(self.suite)*1000000+self.task*10000+self.initial*100+q
            noise=np.random.default_rng(seed).standard_normal(noise_shape).astype(np.float32)
            row=dict(query=q,noise_sha256=sha(noise),main_sha256=sha(raw['observation/image']),
                wrist_sha256=sha(raw['observation/wrist_image']),proprio_sha256=sha(raw['observation/state']))
            with torch.no_grad():result=policy.infer(copy.deepcopy(raw),noise=noise.copy())
            actions=np.asarray(result['actions']);assert actions.ndim==2 and actions.shape[1]==7 and np.isfinite(actions).all()
            row.update(action_sha256=sha(actions),actions=actions[:5].tolist())
            self.rows.append(row)
            return result
    paired=Paired();pixels={}
    def texture(name):
        manifest=json.loads((a.root/'texture_manifest.json').read_text())
        item=manifest[name];path=a.root/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        if name not in pixels:pixels[name]=np.array(Image.open(path).convert('RGB'))
        return pixels[name],item
    def episode(suite,task,initial,condition,calibration=False):
        path=outcomes/('%s_t%d_i%d_%s.json'%(suite,task,initial,condition))
        if path.exists() and not calibration:
            data=json.loads(path.read_text());assert data['completed'];return data
        image,meta=texture(condition)
        np.random.seed(7300000+p['suites'].index(suite)*100000+task*1000+initial)
        paired.reset(suite,task,initial);before=current.get('uploads',0);start=time.monotonic()
        task_obj=suites[suite].get_task(task);state=suites[suite].get_task_init_states(task)[initial]
        success=A.evaluate_native(paired,task_obj,state,image,max_steps=400,num_steps_wait=10,
            replan_steps=5,floor_mode=(suite=='object'))
        assert paired.rows and current['uploads']>before
        data=dict(completed=True,suite=suite,task=task,initial=initial,condition=condition,
            training_seed=meta.get('seed'),reference=meta.get('reference'),texture_sha256=meta['sha256'],
            robot_success=bool(success),queries=len(paired.rows),elapsed_seconds=time.monotonic()-start,
            native_uploads=current['uploads']-before,trajectory=paired.rows,worker=a.worker,
            host=os.uname().nodename,gpu=os.environ['CUDA_VISIBLE_DEVICES'])
        if not calibration:write(path,data)
        print('EPISODE',suite,task,initial,condition,int(success),len(paired.rows),flush=True)
        return data
    # Each freshly loaded model repeats an entire prescribed Plain trajectory.
    cal0=episode('spatial',0,10,'plain',True)
    cal1=episode('spatial',0,10,'plain',True)
    assert cal0['robot_success']==cal1['robot_success'] and cal0['trajectory']==cal1['trajectory']
    write(worker/'calibration.json',dict(passed=True,queries=cal0['queries'],trajectory_exact=True))
    registration=dict(worker=a.worker,protocol_sha256=hashlib.sha256((a.repo/'ops/stability_20261102/protocol.json').read_bytes()).hexdigest(),
        checkpoint=p['checkpoint'],noise_shape=noise_shape,mujoco=mujoco.__version__,pytorch=torch.__version__,
        gpu_name=torch.cuda.get_device_name(),host=os.uname().nodename,
        source_hashes={str(Path(inspect.getfile(v))):hashlib.sha256(Path(inspect.getfile(v)).read_bytes()).hexdigest() for v in [A,B,type(policy),type(policy._model)]},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip())
    write(worker/'registered.json',registration)
    if a.smoke:
        for condition in ['s7_native','s7_mr']:
            row=episode('spatial',0,10,condition,True)
            assert row['trajectory'][0]['main_sha256']!=cal0['trajectory'][0]['main_sha256']
        write(worker/'smoke_passed.json',dict(passed=True,plain_trajectory_identity=True,
                                            adversarial_main_input_changed=True))
        return
    lock=a.root/'evaluation/queue.lock';jobs=json.loads((a.root/'evaluation/jobs.json').read_text())
    done=[]
    while True:
        chosen=None
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            manifest=json.loads((a.root/'texture_manifest.json').read_text())
            for job in jobs:
                jid=job['id'];claim=claims/(jid+'.json')
                if (complete/(jid+'.json')).exists():continue
                if any(name not in manifest for name in job['conditions']):continue
                if claim.exists():
                    existing=json.loads(claim.read_text())
                    try:os.kill(existing['pid'],0);continue
                    except ProcessLookupError:pass
                write(claim,dict(pid=os.getpid(),worker=a.worker,time=time.time()))
                chosen=job;break
            fcntl.flock(f,fcntl.LOCK_UN)
        if chosen is None:
            count=len(list(complete.glob('*.json')))
            write(worker/'status.json',dict(waiting=True,finished_bundles=count,total_bundles=len(jobs),updated=time.time()))
            if count==len(jobs):break
            time.sleep(15);continue
        rows=[]
        for initial in p['initials']:
            conditions=list(chosen['conditions'])
            if (initial+chosen['task'])%2:conditions.reverse()
            pair=[episode(chosen['suite'],chosen['task'],initial,c) for c in conditions]
            if len(pair)==2:
                assert pair[0]['trajectory'][0]['proprio_sha256']==pair[1]['trajectory'][0]['proprio_sha256']
                assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(pair[0]['trajectory'],pair[1]['trajectory']))
            rows.extend(dict(condition=x['condition'],initial=initial,robot_success=x['robot_success']) for x in pair)
        write(complete/(chosen['id']+'.json'),dict(completed=True,job=chosen,outcomes=rows,worker=a.worker))
        done.append(chosen['id'])
        write(worker/'status.json',dict(waiting=False,last_completed=chosen['id'],completed=done,updated=time.time()))
    write(worker/'completed.json',dict(completed=True,bundles=done,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',required=True);p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
