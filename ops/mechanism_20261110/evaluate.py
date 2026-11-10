"""Same-state native camera interventions; frozen pi05 and frozen texture."""
import argparse
import copy
import ctypes
import fcntl
import gc
import hashlib
import inspect
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from common import HERE,PROTOCOL,REPO,ROOT,read,write,sha
import refcloth_resource as guard


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','2','3']
    torch.set_num_threads(2);torch.manual_seed(7)
    torch.cuda.set_per_process_memory_fraction(.47,0)
    guard.install_pause_handler()
    import mujoco
    import attack_pi0_cloth as A
    import attack_pi0 as B
    from libero.libero import benchmark
    p=read(PROTOCOL)['closed_loop']
    assert A.patch_status()=='patched' and mujoco.__version__=='3.3.0'
    assert os.environ.get('REFCLOTH_SURFACE_MODE','legacy')==p['surface_mode']
    original_create=A.policy_config.create_trained_policy
    assert 'pytorch_device' in inspect.signature(original_create).parameters
    def create_cpu(*a,**kw):
        kw['pytorch_device']='cpu'
        return original_create(*a,**kw)
    A.policy_config.create_trained_policy=create_cpu
    try:policy=A.load_policy(p['checkpoint'],model_config=p['config'],num_denoise=p['num_denoise'])
    finally:A.policy_config.create_trained_policy=original_create
    policy._model.to('cuda');policy._pytorch_device='cuda'
    policy._model.eval().requires_grad_(False)
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    guard.mark_ready()
    print('MODEL_READY',args.worker,flush=True)
    cfg=A.training_config.get_config(p['config']).model
    noise_shape=(cfg.action_horizon,cfg.action_dim)
    suite=benchmark.get_benchmark_dict()[p['suite']]()
    base=np.array(Image.open(A.CLOTH_T0).convert('RGB'))
    adversarial=np.array(Image.open(REPO/p['texture']).convert('RGB'))
    frozen=dict(base_file_sha256=sha(A.CLOTH_T0),texture_file_sha256=sha(REPO/p['texture']),
        base_pixel_sha256=array_sha(base),texture_pixel_sha256=array_sha(adversarial))
    assert frozen['texture_pixel_sha256'].startswith('8a87319e2cea903c')
    worker=ROOT/'workers'/args.worker;worker.mkdir(parents=True,exist_ok=True)
    write(worker/'registered.json',dict(completed=False,worker=args.worker,pid=os.getpid(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        protocol_sha256=sha(PROTOCOL),checkpoint=p['checkpoint'],noise_shape=noise_shape,
        gpu=os.environ['CUDA_VISIBLE_DEVICES'],host=os.uname().nodename,
        cuda_name=torch.cuda.get_device_name(),pytorch=torch.__version__,mujoco=mujoco.__version__,
        inputs=frozen,source_hashes={inspect.getfile(v):sha(inspect.getfile(v)) for v in [A,B,type(policy),type(policy._model)]}))
    current={};original_env=A._get_libero_env;original_inject=A.inject_mujoco_texture

    def inject(sim,tid,png):
        previous=original_inject(sim,tid,png)
        assert B._upload_mujoco_texture(sim,tid),'Native GL texture upload failed'
        return previous
    A.inject_mujoco_texture=inject

    def remember(*a,**kw):
        env,desc=original_env(*a,**kw);current['env']=env;current['cameras']={}
        original_step=env.step
        def step(*sa,**sk):
            sim=env.env.sim
            if not getattr(sim,'_mechanism_camera_hook',False):
                original_render=sim.render;current['original_render']=original_render
                def sensor_render(*ra,**rk):
                    result=original_render(*ra,**rk)
                    name=rk.get('camera_name')
                    if name in ['agentview','robot0_eye_in_hand'] and rk.get('width')==256 and not rk.get('segmentation',False):
                        assert not rk.get('depth',False)
                        native=getattr(sim.data,'_data',sim.data)
                        current['cameras'][name]=dict(data=copy.copy(native),image=np.ascontiguousarray(result[::-1,::-1]))
                    return result
                sim.render=sensor_render;sim._mechanism_camera_hook=True
            return original_step(*sa,**sk)
        env.step=step
        return env,desc
    A._get_libero_env=remember

    def integration(sim):
        m=getattr(sim.model,'_model',sim.model);d=getattr(sim.data,'_data',sim.data)
        flag=mujoco.mjtState.mjSTATE_INTEGRATION
        values=np.empty(mujoco.mj_stateSize(m,flag),dtype=np.float64)
        mujoco.mj_getState(m,d,values,flag)
        return values

    def render_cached(sim):
        context=sim._render_context_offscreen;previous=context.data
        result=[]
        try:
            for camera in ['agentview','robot0_eye_in_hand']:
                context.data=SimpleNamespace(_data=current['cameras'][camera]['data'])
                im=current['original_render'](width=256,height=256,camera_name=camera,mode='offscreen')
                result.append(np.ascontiguousarray(im[::-1,::-1]))
        finally:context.data=previous
        return tuple(result)

    def intervene(raw,condition,first):
        sim=current['env'].env.sim
        actual=(raw['observation/image'],raw['observation/wrist_image'])
        cameras=[current['cameras'][name] for name in ['agentview','robot0_eye_in_hand']]
        assert cameras[0]['data'].time==cameras[1]['data'].time
        assert np.array_equal(cameras[0]['data'].qpos,cameras[1]['data'].qpos)
        assert np.array_equal(cameras[0]['data'].qvel,cameras[1]['data'].qvel)
        assert all(np.array_equal(a,c['image']) for a,c in zip(actual,cameras))
        before=integration(sim)
        repeated=render_cached(sim)
        assert all(np.array_equal(a,b) for a,b in zip(actual,repeated)),'Cached sensor exposure mismatch'
        output=list(actual)
        alternate=None
        if condition!='plain':
            _,bid,_=A.get_target_model_matrix(current['env'],A.CLOTH_SEARCH)
            tid=A.find_target_texture_id(sim,bid,A.CLOTH_SEARCH)
            saved=inject(sim,tid,adversarial)
            try:alternate=render_cached(sim)
            finally:A.restore_mujoco_texture(sim,tid,*saved)
            assert all(np.array_equal(a,b) for a,b in zip(actual,render_cached(sim))),'A-B-A changed images'
            if condition in ['main_only','both']:output[0]=alternate[0]
            if condition in ['wrist_only','both']:output[1]=alternate[1]
        assert np.array_equal(before,integration(sim)),'Rendering changed live physics'
        row=dict(integration_exact=True,cached_exposure_exact=True,aba_exact=True,
            same_camera_state=True,camera_time=float(cameras[0]['data'].time),
            query_time=float(sim.data.time),camera_qpos_sha256=array_sha(cameras[0]['data'].qpos),
            plain_main_sha256=array_sha(actual[0]),plain_wrist_sha256=array_sha(actual[1]))
        if alternate is not None:
            row.update(ref_main_sha256=array_sha(alternate[0]),ref_wrist_sha256=array_sha(alternate[1]))
        raw['observation/image'],raw['observation/wrist_image']=output
        if first:
            folder=worker/'input_examples'/current['episode'];folder.mkdir(parents=True,exist_ok=True)
            for name,pair in [('plain',actual),('consumed',output)]+([('refcloth',alternate)] if alternate is not None else []):
                for view,im in zip(['main','wrist'],pair):Image.fromarray(im).save(folder/(name+'_'+view+'.png'))
        return row

    class Paired:
        def reset(self,task,initial,condition,hook):
            self.task,self.initial,self.condition,self.hook=task,initial,condition,hook;self.rows=[]
        def infer(self,raw):
            q=len(self.rows);raw=copy.deepcopy(raw)
            seed=3710000+self.task*10000+self.initial*100+q
            noise=np.random.default_rng(seed).standard_normal(noise_shape).astype(np.float32)
            extra={}
            if self.hook:
                rng=np.random.get_state();trng=torch.get_rng_state();crng=torch.cuda.get_rng_state();prng=random.getstate()
                extra=intervene(raw,self.condition,q==0)
                assert np.array_equal(rng[1],np.random.get_state()[1]) and rng[2:]==np.random.get_state()[2:]
                assert torch.equal(trng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
                assert prng==random.getstate(),'Counterfactual renderer consumed RNG'
            row=dict(query=q,noise_sha256=array_sha(noise),main_sha256=array_sha(raw['observation/image']),
                wrist_sha256=array_sha(raw['observation/wrist_image']),proprio_sha256=array_sha(raw['observation/state']))
            with torch.no_grad():result=policy.infer(raw,noise=noise.copy())
            actions=np.asarray(result['actions']);assert actions.ndim==2 and actions.shape[1]==7 and np.isfinite(actions).all()
            row.update(action_sha256=array_sha(actions),actions=actions[:p['replan_steps']].tolist(),measurement=extra)
            self.rows.append(row)
            return result
    paired=Paired()

    def episode(task,initial,condition,hook=True,calibration=False):
        name='object_t%d_i%d_%s'%(task,initial,condition)
        path=ROOT/'episodes'/(name+'.json')
        if path.exists() and not calibration:
            old=read(path);assert old['completed'] and old['inputs']==frozen
            assert old['protocol_sha256']==sha(PROTOCOL)
            return old
        np.random.seed(700000+task*1000+initial)
        torch.manual_seed(7);random.seed(7)
        paired.reset(task,initial,condition,hook);current['episode']=name+('_direct' if not hook else '')
        obj=suite.get_task(task);state=suite.get_task_init_states(task)[initial]
        pixels=adversarial if condition=='both' and not hook else base
        started=time.monotonic()
        success=A.evaluate_native(paired,obj,state,pixels,max_steps=p['max_steps'],
            num_steps_wait=p['settle_steps'],replan_steps=p['replan_steps'],floor_mode=p['floor_mode'])
        assert paired.rows or success
        record=dict(completed=True,target=p['target'],suite='object',task=task,initial=initial,condition=condition,
            robot_success=bool(success),queries=len(paired.rows),trajectory=paired.rows,inputs=frozen,
            elapsed_seconds=time.monotonic()-started,worker=args.worker,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            host=os.uname().nodename,protocol_sha256=sha(PROTOCOL),hook=hook)
        if not calibration:write(path,record)
        print('EPISODE',task,initial,condition,int(success),len(paired.rows),round(record['elapsed_seconds'],2),flush=True)
        return record

    if args.smoke_first:
        fields=['noise_sha256','main_sha256','wrist_sha256','proprio_sha256','action_sha256','actions']
        outcomes=[]
        for condition in ['plain','both']:
            direct=episode(0,10,condition,False,True)
            hook=episode(0,10,condition,True,True)
            assert direct['robot_success']==hook['robot_success'] and direct['queries']==hook['queries']
            assert all(all(x[k]==y[k] for k in fields) for x,y in zip(direct['trajectory'],hook['trajectory'])), 'Hook changed a native control rollout'
            write(worker/(condition+'_direct.json'),direct);write(worker/(condition+'_hook.json'),hook)
            outcomes.append(dict(condition=condition,robot_success=direct['robot_success'],queries=direct['queries'],trajectory_exact=True))
        write(ROOT/'smoke_passed.json',dict(passed=True,worker=args.worker,conditions=outcomes,
            protocol_sha256=sha(PROTOCOL),inputs=frozen,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    assert read(ROOT/'smoke_passed.json',{}).get('passed'), 'Smoke is required before endpoint evaluation'
    for sub in ['episodes','completed','claims']:(ROOT/sub).mkdir(exist_ok=True)
    job_list=[(task,initial) for task in p['tasks'] for initial in p['initials']]
    while True:
        chosen=None
        with (ROOT/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for task,initial in job_list:
                key='t%d_i%d'%(task,initial)
                if (ROOT/'completed'/(key+'.json')).exists():continue
                previous=read(ROOT/'claims'/(key+'.json'),{})
                if previous and guard.process_runs_script(previous['pid'],__file__,previous['ticks']):continue
                write(ROOT/'claims'/(key+'.json'),dict(pid=os.getpid(),ticks=guard.identity(os.getpid())['ticks'],worker=args.worker,time=time.time()))
                chosen=(task,initial,key);break
        if chosen is None:
            if len(list((ROOT/'completed').glob('*.json')))==len(job_list):break
            if guard.pause_requested():raise SystemExit(75)
            time.sleep(5);continue
        task,initial,key=chosen
        conditions=p['conditions'];shift=(task*10+initial-10)%len(conditions)
        conditions=conditions[shift:]+conditions[:shift]
        records=[episode(task,initial,c) for c in conditions]
        for other in records[1:]:
            assert all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(records[0]['trajectory'],other['trajectory']))
            assert records[0]['trajectory'][0]['proprio_sha256']==other['trajectory'][0]['proprio_sha256']
        write(ROOT/'completed'/(key+'.json'),dict(completed=True,task=task,initial=initial,
            outcomes={r['condition']:r['robot_success'] for r in records},worker=args.worker,gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        write(worker/'status.json',dict(last_key=key,updated=time.time(),pairs_complete=len(list((ROOT/'completed').glob('*.json')))))
        if guard.pause_requested():raise SystemExit(75)
    write(worker/'completed.json',dict(completed=True,time=time.time(),peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True)
    parser.add_argument('--smoke-first',action='store_true')
    main(parser.parse_args())
