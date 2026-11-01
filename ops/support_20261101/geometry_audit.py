"""Native mask exposure at target episode starts, without loading a policy.

Static geometric diagnostic; no target actions and no mask selection.
Checks actual color changes and native marker-defined visible support.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from support_core import block_ids


def arrays(tree,prefix=''):
    out={}
    if isinstance(tree,dict):
        for key,value in tree.items():out.update(arrays(value,prefix+'/'+str(key)))
    elif isinstance(tree,np.ndarray) and tree.ndim==3:out[prefix]=tree
    return out


def main(repo,source,out,protocol_path):
    import attack_pi0_cloth as A
    import attack_pi0 as BASE
    from libero.libero import benchmark
    p=json.loads(protocol_path.read_text())
    assert A.patch_status()=='patched'
    out.mkdir(parents=True,exist_ok=False)
    base=np.array(Image.open(A.CLOTH_T0).convert('RGB'))
    definitions={x['name']:x for x in json.loads((source/'texture_manifest.json').read_text())}
    names=p['conditions']+['greedy_08','greedy_04']
    labels=block_ids(base.shape,4)
    report=dict(scope='native target-start geometry only; no policy loaded or queried',records=[],completed=False)
    def render(sim):
        return [np.ascontiguousarray(sim.render(width=256,height=256,camera_name=cam,mode='offscreen')[::-1,::-1])
                for cam in ['agentview','robot0_eye_in_hand']]
    for suite_name in p['suites']:
        suite=benchmark.get_benchmark_dict()[suite_name]()
        for task_id in p['tasks']:
            task=suite.get_task(task_id)
            for initial in p['initials']:
                env,_=A._get_libero_env(task,resolution=256)
                env.reset();env.set_init_state(suite.get_task_init_states(task_id)[initial])
                sim=env.env.sim;sim.forward()
                A.align_cloth_to_table(env,floor_mode=suite_name=='libero_object')
                _,bid,_=A.get_target_model_matrix(env,A.CLOTH_SEARCH)
                tid=A.find_target_texture_id(sim,bid,A.CLOTH_SEARCH)
                assert tid>=0
                model=getattr(sim.model,'_model',sim.model)
                def inject(png):
                    assert png.shape[:2]==(int(model.tex_height[tid]),int(model.tex_width[tid]))
                    h=A.inject_mujoco_texture(sim,tid,png)
                    assert BASE._upload_mujoco_texture(sim,tid)
                    return h
                render(sim);h0=inject(base)
                obs=env.set_init_state(suite.get_task_init_states(task_id)[initial])
                try:
                    for _ in range(p['settle_steps']):obs,_,_,_=env.step(A.get_libero_dummy_action())
                    plain=render(sim)
                    raw0=arrays(A.build_raw_inputs(obs,task.language,adv_image_np=plain[0],adv_wrist_image_np=plain[1]))
                    for name in names:
                        definition=definitions[name]
                        path=source/'textures'/(name+'.png')
                        assert hashlib.sha256(path.read_bytes()).hexdigest()==definition['file_sha256']
                        png=np.array(Image.open(path).convert('RGB'))
                        support=np.isin(labels,definition['keep'])
                        marker=base.copy();marker[support]=[255,0,255]
                        def query(image):
                            h=inject(image)
                            try:pair=render(sim)
                            finally:A.restore_mujoco_texture(sim,tid,*h)
                            assert all(np.array_equal(a,b) for a,b in zip(plain,render(sim)))
                            return pair
                        actual=query(png);visible=query(marker)
                        raw=arrays(A.build_raw_inputs(obs,task.language,adv_image_np=visible[0],adv_wrist_image_np=visible[1]))
                        rawarea={}
                        for key,value in raw.items():
                            delta=np.abs(value.astype(float)-raw0[key].astype(float))
                            axis=0 if delta.shape[0]==3 else -1
                            threshold=1 if np.issubdtype(value.dtype,np.integer) else 1/255.
                            rawarea[key]=float(np.any(delta>threshold,axis=axis).mean())
                        row=dict(suite=suite_name,task=task_id,initial=initial,candidate=name,
                            surface_fraction=len(definition['keep'])/16.,
                            native_visible_fraction=[float(np.any(np.abs(a.astype(float)-b)>1,axis=-1).mean()) for a,b in zip(visible,plain)],
                            actual_changed_fraction=[float(np.any(np.abs(a.astype(float)-b)>1,axis=-1).mean()) for a,b in zip(actual,plain)],
                            rgb_rms=[float(np.sqrt(np.mean(np.square((a.astype(float)-b)/255.)))) for a,b in zip(actual,plain)],
                            raw_input_marker_fraction=rawarea,aba_exact=True)
                        report['records'].append(row)
                    print('GEOMETRY',suite_name,task_id,initial,flush=True)
                    (out/'geometry.partial.json').write_text(json.dumps(report,indent=2))
                finally:A.restore_mujoco_texture(sim,tid,*h0);A._close_libero_env(env)
    report['completed']=True
    (out/'geometry.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args();main(a.repo,a.source,a.out,a.protocol)
