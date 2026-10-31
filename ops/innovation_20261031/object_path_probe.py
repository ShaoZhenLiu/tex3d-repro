"""Same-state pi0.5 Object camera intervention, with explicit common noise.

No texture optimization or task-success evaluation. At a frozen simulator state
we mix native Plain/RefCloth images across the two views, audit transformed
inputs, and compare action chunks. Policy noise is an explicit fixed argument.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def array_tree(tree,prefix=''):
    out={}
    if isinstance(tree,dict):
        for key,value in tree.items():out.update(array_tree(value,prefix+'/'+str(key)))
    elif isinstance(tree,(list,tuple)):
        for i,value in enumerate(tree):out.update(array_tree(value,prefix+'/'+str(i)))
    elif isinstance(tree,np.ndarray):out[prefix]=tree
    return out


def main(repo,output):
    import attack_pi0_cloth as APC
    import attack_pi0 as BASE
    from libero.libero import benchmark
    torch.set_num_threads(2);np.random.seed(7);torch.manual_seed(7)
    checkpoint='/data/shaozhen.liu/model/RLinf-Pi05-LIBERO-130-fullshot-SFT'
    policy=APC.load_policy(checkpoint,model_config='pi05_libero',num_denoise=10)
    config=APC.training_config.get_config('pi05_libero').model
    noise_shape=(config.action_horizon,config.action_dim)
    base=np.array(Image.open(APC.CLOTH_T0).convert('RGB'))
    mr=np.array(Image.open(repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png').convert('RGB'))
    output.mkdir(parents=True,exist_ok=True)
    report=dict(checkpoint=checkpoint,model_config='pi05_libero',num_denoise=10,
        scope='same-state input-path diagnostic; no closed-loop success claims',
        noise_shape=list(noise_shape),texture_optimized=False,records=[])
    was_clean=APC.patch_status()=='clean';APC.ensure_patch(True)
    suite=benchmark.get_benchmark_dict()['libero_object']()
    def render(sim):
        return tuple(np.ascontiguousarray(sim.render(width=256,height=256,camera_name=cam,mode='offscreen')[::-1,::-1])
                     for cam in ['agentview','robot0_eye_in_hand'])
    def inject(sim,tid,png):
        model=getattr(sim.model,'_model',sim.model)
        assert png.shape[:2]==(int(model.tex_height[tid]),int(model.tex_width[tid]))
        handle=APC.inject_mujoco_texture(sim,tid,png)
        if not BASE._upload_mujoco_texture(sim,tid):raise RuntimeError('texture upload failed')
        return handle
    try:
        for task_id in [0,4,8]:
            task=suite.get_task(task_id);inits=suite.get_task_init_states(task_id)
            for initial in [10,11]:
                env,desc=APC._get_libero_env(task,resolution=256)
                env.reset();obs=env.set_init_state(inits[initial]);sim=env.env.sim;sim.forward()
                APC.align_cloth_to_table(env)
                _,bid,_=APC.get_target_model_matrix(env,APC.CLOTH_SEARCH)
                tid=APC.find_target_texture_id(sim,bid,APC.CLOTH_SEARCH)
                assert tid>=0
                render(sim)
                handle=inject(sim,tid,base)
                try:
                    for _ in range(10):obs,_,_,_=env.step(APC.get_libero_dummy_action())
                    p0,p1=render(sim)
                    temp=inject(sim,tid,mr);a0,a1=render(sim);APC.restore_mujoco_texture(sim,tid,*temp)
                    again0,again1=render(sim)
                    assert np.array_equal(again0,p0) and np.array_equal(again1,p1),'A-B-A pixels changed'
                    model=getattr(sim.model,'_model',sim.model)
                    users=[]
                    for gid in range(model.ngeom):
                        matid=int(model.geom_matid[gid])
                        if matid>=0 and tid in np.asarray(model.mat_texid[matid]).reshape(-1):
                            try:name=sim.model.geom_id2name(gid)
                            except Exception:name=str(gid)
                            users.append(dict(geom=name,body_id=int(model.geom_bodyid[gid])))
                    assert all(u['body_id']==int(bid) for u in users),('texture shared with other body',users)
                    noise=np.random.default_rng(61000+task_id*100+initial).standard_normal(noise_shape).astype(np.float32)
                    images={'PP':(p0,p1),'AP':(a0,p1),'PA':(p0,a1),'AA':(a0,a1)}
                    actions={};transformed={};raws={}
                    key='object_t%d_i%d'%(task_id,initial)
                    for condition,(main_image,wrist_image) in images.items():
                        raw=APC.build_raw_inputs(obs,desc,adv_image_np=main_image,adv_wrist_image_np=wrist_image)
                        raws[condition]=raw
                        transformed[condition]=array_tree(policy._input_transform(copy.deepcopy(raw)))
                        with torch.no_grad():result=policy.infer(raw,noise=noise.copy())
                        actions[condition]=np.asarray(result['actions'])
                        assert actions[condition].ndim==2 and actions[condition].shape[1]==7
                        assert np.isfinite(actions[condition]).all()
                        for view,image in [('main',main_image),('wrist',wrist_image)]:
                            Image.fromarray(image).save(output/(key+'_'+condition+'_'+view+'.png'))
                    with torch.no_grad():repeat=np.asarray(policy.infer(raws['PP'],noise=noise.copy())['actions'])
                    aa_floor=float(np.max(np.abs(repeat-actions['PP'])))
                    assert aa_floor<1e-6,aa_floor
                    input_diffs={}
                    for condition,tree in transformed.items():
                        input_diffs[condition]={}
                        for path,value in tree.items():
                            reference=transformed['PP'][path]
                            if np.issubdtype(value.dtype,np.number):
                                diff=value.astype(np.float64)-reference.astype(np.float64)
                                input_diffs[condition][path]=dict(shape=list(value.shape),maxabs=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff**2))))
                    action_diffs={}
                    for condition,value in actions.items():
                        delta=value-actions['PP']
                        action_diffs[condition]=dict(all_rmse=float(np.sqrt(np.mean(delta**2))),
                            executed_translation_rmse=float(np.sqrt(np.mean(delta[:5,:3]**2))),
                            executed_rotation_rmse=float(np.sqrt(np.mean(delta[:5,3:6]**2))),
                            executed_gripper_max=float(np.max(np.abs(delta[:5,6]))))
                    rec=dict(task=task_id,initial=initial,description=desc,texture_users=users,aa_action_maxdiff=aa_floor,
                        aba_pixels_equal=True,noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),input_diffs=input_diffs,
                        main_changed_fraction=float(np.mean(np.max(np.abs(a0.astype(int)-p0.astype(int)),axis=-1)>1)),
                        wrist_changed_fraction=float(np.mean(np.max(np.abs(a1.astype(int)-p1.astype(int)),axis=-1)>1)),
                        action_diffs=action_diffs)
                    report['records'].append(rec)
                    np.savez_compressed(output/(key+'_actions.npz'),noise=noise,**actions)
                    (output/'object_path.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                    print(json.dumps(dict(task=task_id,initial=initial,aa=aa_floor,main_fraction=rec['main_changed_fraction'],wrist_fraction=rec['wrist_changed_fraction'],actions=action_diffs)),flush=True)
                finally:APC.restore_mujoco_texture(sim,tid,*handle);APC._close_libero_env(env)
        report['completed']=True
        (output/'object_path.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        if was_clean:APC.ensure_patch(False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.repo,args.out)
