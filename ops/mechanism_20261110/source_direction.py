"""Independent same-state forward confirmation with exact native A/A controls."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import HERE,REPO,ROOT,write,sha
import refcloth_resource as guard


def score(prediction,truth):
    p=prediction.detach().double().flatten();t=truth.detach().double().flatten()
    pn,tn=float(p.norm()),float(t.norm())
    return dict(cosine=float(F.cosine_similarity(p,t,dim=0)) if pn>1e-12 and tn>1e-12 else None,
        error_norm=float((p-t).norm()),relative_error=float((p-t).norm())/tn if tn>1e-12 else None,
        truth_norm=tn,prediction_norm=pn)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','2','3']
    for path in [REPO/'openvla-oft/experiments/robot/libero',REPO/'ops/math_20261031',REPO/'ops/support_20261101']:
        sys.path.insert(0,str(path))
    from native_runtime import NativeRuntime
    out=ROOT/'fresh_feature_direction';out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1234);np.random.seed(1234)
    runtime=NativeRuntime(REPO,.47,out)
    guard.mark_ready();ctx,A=runtime.ctx,runtime.A
    assert A.TR.patch_status()=='patched' and not ctx.cfg.use_diffusion
    write(out/'registered.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        frozen_texture_file_sha256=sha(runtime.full_path),tasks=[0,2,6,8],initial=0,steps=list(range(0,71,10)),
        source='Same 32 task/step keys, with fresh canonical simulator snapshots; archive raw states independently',
        primary='The identical deployed uint8 PNG on both paths; no UV augmentation at measurement',
        secondary='Original floating trained theta on the differentiable path, deployed uint8 PNG natively',
        renderer='Frozen production renderer; no recalibration',gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        legacy_zero_control='Archived zero-texture native displacement was nonzero; retained separately, independently confirm with exact A/A controls'))
    rows=[]
    try:
        states=runtime.collect([0,2,6,8],0,list(range(0,71,10)))
        assert len(states)==32
        for st in states:
            folder=out/st.key;folder.mkdir(exist_ok=True)
            st.restore()
            obs=st.env.env._get_observations(force_update=True)
            pair0=st.pixels(runtime.base);pair1=st.pixels(runtime.full)
            for name,pair in [('plain',pair0),('refcloth',pair1)]:
                for view,image in zip(['main','wrist'],pair):Image.fromarray(image).save(folder/(name+'_'+view+'.png'))
            st.restore()
            _,bid,_=A.base.get_target_model_matrix(st.env,A.TR.SEARCH)
            obs=dict(obs)
            obs['agentview_image']=np.ascontiguousarray(pair0[0][::-1,::-1])
            obs['robot0_eye_in_hand_image']=np.ascontiguousarray(pair0[1][::-1,::-1])
            quad=A.quad_to_dev(A.capture_quad(st.env,obs,bid,ctx.device),ctx.device)
            assert np.array_equal(quad['native_main_px'],pair0[0])
            assert np.array_equal(quad['native_wrist_px'],pair0[1])
            def native(pair):return ctx.hidden_from_arrays(*pair,st.inputs,st.proprio).detach()
            def synth(tex=None):
                a,b=ctx.compose(quad,tex_override=tex)
                return ctx.hidden_from_tensors(a,b,st.inputs,st.proprio).detach()
            with torch.no_grad():
                n0=native(pair0);n1=native(pair1);n0_repeat=native(st.pixels(runtime.base))
                assert torch.equal(n0,n0_repeat),'Native zero-texture A/A feature discrepancy'
                ctx.load_theta('zero')
                r0=synth();r0_repeat=synth()
                assert torch.equal(r0,r0_repeat),'Differentiable zero-texture A/A discrepancy'
                zero_export=ctx.renderer.export_png()
                assert np.array_equal(zero_export,runtime.base),'Zero export differs from original Plain pixels'
                r0_png=synth(ctx.renderer.png_to_tex(runtime.base).to(ctx.device))
                assert torch.equal(r0,r0_png),'PNG Plain and zero-theta source features differ'
                r1_png=synth(ctx.renderer.png_to_tex(runtime.full).to(ctx.device))
                ctx.load_theta(str(REPO/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Ep0_Cloth_Delta_OV.pt'))
                assert np.array_equal(ctx.renderer.export_png(),runtime.full)
                r1_float=synth()
                for route,r1 in [('png',r1_png),('float',r1_float)]:
                    dm,dn,truth=r1-r0,r1-n0,n1-n0
                    matched,native_ref=score(dm,truth),score(dn,truth)
                    row=dict(state=st.key,task=st.meta['task'],step=st.meta['step'],route=route,
                        matched_cosine=matched['cosine'],native_cosine=native_ref['cosine'],
                        cosine_difference=matched['cosine']-native_ref['cosine'],
                        matched_error=matched['error_norm'],native_error=native_ref['error_norm'],
                        relative_matched_error=matched['relative_error'],relative_native_error=native_ref['relative_error'],
                        true_displacement_norm=matched['truth_norm'],baseline_offset_norm=float((r0-n0).double().norm()),
                        baseline_relative_offset=float((r0-n0).double().norm()/n0.double().norm()),
                        native_aa_exact=True,render_aa_exact=True,native_aba_exact=True)
                    rows.append(row)
                torch.save(dict(native0=n0.cpu(),native1=n1.cpu(),render0=r0.cpu(),render1_png=r1_png.cpu(),
                    render1_float=r1_float.cpu()),str(folder/'features.pt'))
            np.save(folder/'simulator_state.npy',st.snapshot)
            write(folder/'state.json',st.meta)
            write(out/'partial.json',dict(completed=False,rows=rows))
            print('DIRECTION',st.key,rows[-2]['matched_cosine'],rows[-2]['native_cosine'],flush=True)
        with (out/'state_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries={}
        for route in ['png','float']:
            subset=[r for r in rows if r['route']==route]
            summaries[route]={k:float(np.mean([r[k] for r in subset])) for k in ['matched_cosine','native_cosine',
                'cosine_difference','matched_error','native_error','relative_matched_error','relative_native_error','baseline_relative_offset']}
            summaries[route]['matched_cosine_wins']=sum(r['cosine_difference']>0 for r in subset)
            summaries[route]['matched_error_wins']=sum(r['matched_error']<r['native_error'] for r in subset)
        write(out/'results.json',dict(completed=True,states=32,rows=rows,summary=summaries,
            all_zero_controls_exact=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),time=time.time()))
    finally:runtime.close_states()


if __name__=='__main__':main()
