"""Same-state native Object measurement at the real policy/model input stages."""
import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from measurement_core import measure_three


def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(obj,path):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def main(repo,out,target,protocol_path):
    import attack_pi0_cloth as A
    import attack_pi0 as BASE
    from libero.libero import benchmark
    from openpi.models import model as model_module
    from openpi.models_pytorch import preprocessing_pytorch as preprocessing
    p=json.loads(protocol_path.read_text());tp=p['targets'][target]
    assert A.patch_status()=='patched', 'Shared asset patch is read-only for this audit'
    assert os.environ.get('REFCLOTH_SURFACE_MODE','legacy')=='legacy'
    torch.set_num_threads(2);torch.manual_seed(7);np.random.seed(7)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    out.mkdir(parents=True,exist_ok=False)
    policy=A.load_policy(tp['checkpoint'],model_config=tp['config'],num_denoise=10)
    policy._model.eval().requires_grad_(False)
    # Only these two production methods are called. No actions are computed.
    def process(raw):
        transformed=policy._input_transform(copy.deepcopy(raw))
        intermediate={k:v.copy() for k,v in transformed['image'].items()}
        import jax
        tensors=jax.tree.map(lambda x:torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None,...],transformed)
        observation=model_module.Observation.from_dict(tensors)
        with torch.no_grad():imgs,masks,tokens,token_masks,state=policy._model._preprocess_observation(observation,train=False)
        names=list(preprocessing.IMAGE_KEYS)
        assert len(names)==len(imgs)==len(masks)
        result={};spec={}
        for key,image,mask in zip(names,imgs,masks):
            z=image.detach().float().cpu().numpy()
            assert z.shape==(1,3,224,224),(key,z.shape)
            rgb=(z[0].transpose(1,2,0).astype(np.float64)+1)*127.5
            assert np.max(np.abs(rgb-intermediate[key].astype(float)))<1e-3
            result[key]=rgb
            spec[key]=dict(active=bool(mask.item()),tensor_shape=list(z.shape),dtype=str(image.dtype),
                           min=float(z.min()),max=float(z.max()),normalized_tensor_sha256=sha(z))
        return intermediate,result,spec,state.detach().cpu().numpy(),tokens.detach().cpu().numpy()
    base=np.array(Image.open(A.CLOTH_T0).convert('RGB'))
    ref=np.array(Image.open(repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png').convert('RGB'))
    assert base.shape==ref.shape
    marker_a=np.full_like(base,p['marker_a_rgb']);marker_b=np.full_like(base,p['marker_b_rgb'])
    suite=benchmark.get_benchmark_dict()['libero_object']()
    report=dict(target=target,protocol=p,records=[],completed=False,actions_computed=False,
        model_weights_trained=False,target_gradients_used=False,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        pipeline_sources={})
    for obj in [A,BASE,type(policy),type(policy._model),model_module,preprocessing]:
        path=Path(inspect.getfile(obj));report['pipeline_sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    code=out/'pipeline_sources';code.mkdir()
    for name,fn in [('build_raw_inputs',A.build_raw_inputs),('render_now',A._render_now),
                    ('policy_infer',policy.infer),('model_preprocess',policy._model._preprocess_observation),
                    ('observation_from_dict',model_module.Observation.from_dict),
                    ('preprocess_function',preprocessing.preprocess_observation_pytorch)]:
        (code/(name+'.txt')).write_text(inspect.getsource(fn))
    for task_id in p['tasks']:
        task=suite.get_task(task_id);inits=suite.get_task_init_states(task_id)
        for initial in p['initials']:
            env,_=A._get_libero_env(task,resolution=256)
            env.reset();env.set_init_state(inits[initial]);sim=env.env.sim;sim.forward()
            A.align_cloth_to_table(env,floor_mode=True)
            _,bid,_=A.get_target_model_matrix(env,A.CLOTH_SEARCH)
            tid=A.find_target_texture_id(sim,bid,A.CLOTH_SEARCH);assert tid>=0
            geoms=[g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g])==int(bid)]
            assert geoms
            native=getattr(sim.model,'_model',sim.model)
            def render():return tuple(A._render_now(env,256,c) for c in ['agentview','robot0_eye_in_hand'])
            def inject(png):
                assert png.shape[:2]==(int(native.tex_height[tid]),int(native.tex_width[tid]))
                h=A.inject_mujoco_texture(sim,tid,png);assert BASE._upload_mujoco_texture(sim,tid);return h
            render();h0=inject(base)
            try:
                obs=env.set_init_state(inits[initial])
                for _ in range(p['settle_steps']):obs,_,_,_=env.step(A.get_libero_dummy_action())
                sim.forward();obs=env.env._get_observations(force_update=True)
                frozen=sim.get_state().flatten().copy()
                plain=render();assert all(np.array_equal(x,y) for x,y in zip(plain,render()))
                def query(png):
                    h=inject(png)
                    try:pair=render()
                    finally:A.restore_mujoco_texture(sim,tid,*h)
                    assert all(np.array_equal(x,y) for x,y in zip(plain,render()))
                    assert np.array_equal(sim.get_state().flatten(),frozen)
                    return pair
                images=dict(plain=plain,refcloth=query(ref),marker_a=query(marker_a),marker_b=query(marker_b))
                # Match the old alpha-only hiding operation, independently
                # verify that hidden colors disappear. Never silently redefine it.
                old_rgba={g:sim.model.geom_rgba[g].copy() for g in geoms}
                for g in geoms:sim.model.geom_rgba[g,3]=0
                try:
                    images['hidden']=render()
                    h=inject(marker_a)
                    try:ha=render()
                    finally:A.restore_mujoco_texture(sim,tid,*h)
                    h=inject(marker_b)
                    try:hb=render()
                    finally:A.restore_mujoco_texture(sim,tid,*h)
                    hidden_marker_max=max(int(np.abs(a.astype(int)-b.astype(int)).max()) for a,b in zip(ha,hb))
                    assert hidden_marker_max==0, 'Old alpha-only hiding does not hide cloth'
                finally:
                    for g,value in old_rgba.items():sim.model.geom_rgba[g]=value
                assert all(np.array_equal(x,y) for x,y in zip(plain,render()))
                key='object_t%d_i%d'%(task_id,initial);folder=out/key;folder.mkdir()
                record=dict(task=task_id,initial=initial,cloth_world=sim.data.body_xpos[bid].tolist(),
                    simulator_state_sha256=sha(frozen),hidden_marker_maxdiff=hidden_marker_max,
                    aba_exact=True,native={},input_transform={},model_input={},model_specs={})
                processed={};intermediates={};states={};tokens={}
                for condition,pair in images.items():
                    raw=A.build_raw_inputs(obs,task.language,adv_image_np=pair[0],adv_wrist_image_np=pair[1])
                    mid,final,spec,state,token=process(raw)
                    intermediates[condition]=mid;processed[condition]=final
                    states[condition]=state;tokens[condition]=token
                    record['model_specs'][condition]=spec
                    for view,image in zip(['main','wrist'],pair):Image.fromarray(image).save(folder/(condition+'_native_'+view+'.png'))
                    for view,image in final.items():Image.fromarray(np.rint(image).clip(0,255).astype(np.uint8)).save(folder/(condition+'_model_'+view+'.png'))
                assert all(np.array_equal(x,states['plain']) for x in states.values())
                assert all(np.array_equal(x,tokens['plain']) for x in tokens.values())
                for j,view in enumerate(['main','wrist']):
                    record['native'][view]=measure_three({k:v[j] for k,v in images.items()})
                for view in processed['plain']:
                    record['input_transform'][view]=measure_three({k:v[view] for k,v in intermediates.items()})
                    record['model_input'][view]=measure_three({k:v[view] for k,v in processed.items()})
                np.savez_compressed(folder/'final_model_rgb255.npz',**{k+'__'+v:a for k,x in processed.items() for v,a in x.items()})
                report['records'].append(record)
                write(report,out/'measurement.partial.json')
                print('MEASURED',target,key,json.dumps(record['model_input']['base_0_rgb']['thresholds']['10']),flush=True)
            finally:A.restore_mujoco_texture(sim,tid,*h0);A._close_libero_env(env)
    report['completed']=True;report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    write(report,out/'measurement.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--target',choices=['pi0','pi05'],required=True);ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args();main(a.repo,a.out,a.target,a.protocol)
