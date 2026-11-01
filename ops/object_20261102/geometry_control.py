"""Placement-only Object control; no policy actions or model weights."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from measurement_core import measure_three


def main(args):
    import attack_pi0_cloth as A
    import attack_pi0 as BASE
    from libero.libero import benchmark
    from openpi_client import image_tools
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '4'
    assert A.patch_status() == 'patched'
    assert os.environ.get('REFCLOTH_SURFACE_MODE', 'legacy') == 'legacy'
    p = json.loads(args.protocol.read_text())
    base = np.array(Image.open(A.CLOTH_T0).convert('RGB'))
    ref = np.array(Image.open(args.repo/p['texture']).convert('RGB'))
    markers = [np.full_like(base, p[k]) for k in ['marker_a_rgb', 'marker_b_rgb']]
    suite = benchmark.get_benchmark_dict()['libero_object']()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=p, completed=False, records=[], actions_computed=False,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip(),
        pipeline='Same native two-axis flip, official resize_with_pad(224,224); model equivalence checked by policy probe')

    def write():
        tmp=args.out/'placement.partial.tmp'
        tmp.write_text(json.dumps(report, separators=(',',':'), allow_nan=False))
        tmp.replace(args.out/'placement.partial.json')

    for t in p['tasks']:
        task = suite.get_task(t)
        for initial in p['initials']:
            np.random.seed(700000+t*1000+initial)
            env,_ = A._get_libero_env(task,resolution=256)
            try:
                env.reset();env.set_init_state(suite.get_task_init_states(t)[initial]);sim=env.env.sim
                sim.forward()
                _,bid,_=A.get_target_model_matrix(env,A.CLOTH_SEARCH)
                initial_pos=sim.model.body_pos[bid].copy()
                old_pos=A.align_cloth_to_table(env,floor_mode=False).copy()
                sim.model.body_pos[bid]=initial_pos;sim.forward()
                prod_pos=A.align_cloth_to_table(env,floor_mode=True).copy()
                tid=A.find_target_texture_id(sim,bid,A.CLOTH_SEARCH)
                assert tid>=0
                geoms=np.flatnonzero(np.asarray(sim.model.geom_bodyid)==bid)
                def render():
                    return tuple(np.ascontiguousarray(sim.render(width=256,height=256,camera_name=c,mode='offscreen')[::-1,::-1])
                                 for c in ['agentview','robot0_eye_in_hand'])
                def inject(png):
                    m=getattr(sim.model,'_model',sim.model)
                    assert png.shape[:2]==(int(m.tex_height[tid]),int(m.tex_width[tid]))
                    h=A.inject_mujoco_texture(sim,tid,png)
                    assert BASE._upload_mujoco_texture(sim,tid)
                    return h
                render();h0=inject(base)
                try:
                    env.set_init_state(suite.get_task_init_states(t)[initial])
                    for _ in range(p['settle_steps']):env.step(A.get_libero_dummy_action())
                    frozen=sim.get_state().flatten().copy()
                    row=dict(task=t,initial=initial,placements={},simulator_state_sha256=hashlib.sha256(frozen.tobytes()).hexdigest())
                    for label,pos in [('legacy_ray',old_pos),('production_floor',prod_pos)]:
                        sim.model.body_pos[bid]=pos;sim.forward()
                        assert np.array_equal(frozen,sim.get_state().flatten())
                        plain=render()
                        def changed(png):
                            h=inject(png)
                            try:return render()
                            finally:A.restore_mujoco_texture(sim,tid,*h)
                        images=dict(plain=plain,refcloth=changed(ref),marker_a=changed(markers[0]),marker_b=changed(markers[1]))
                        rgba=sim.model.geom_rgba[geoms].copy();sim.model.geom_rgba[geoms,3]=0
                        try:images['hidden']=render()
                        finally:sim.model.geom_rgba[geoms]=rgba
                        assert all(np.array_equal(x,y) for x,y in zip(plain,render()))
                        assert np.array_equal(frozen,sim.get_state().flatten())
                        res=dict(body_pos=pos.tolist(),native={},model_resize={})
                        folder=args.out/('t%d_i%d'%(t,initial))/label
                        folder.mkdir(parents=True,exist_ok=True)
                        for j,view in enumerate(['main','wrist']):
                            res['native'][view]=measure_three({k:v[j] for k,v in images.items()})
                            # Compare colors after the official resize, then take support.
                            resized={k:image_tools.resize_with_pad(v[j],224,224) for k,v in images.items()}
                            res['model_resize'][view]=measure_three(resized)
                            for c,pair in images.items():Image.fromarray(pair[j]).save(folder/(c+'_'+view+'.png'))
                        row['placements'][label]=res
                    report['records'].append(row);write()
                    print('PLACEMENT',t,initial,json.dumps({k:v['body_pos'] for k,v in row['placements'].items()}),flush=True)
                finally:A.restore_mujoco_texture(sim,tid,*h0)
            finally:A._close_libero_env(env)
    report['completed']=True;write()
    (args.out/'placement.partial.json').replace(args.out/'placement.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    main(ap.parse_args())
