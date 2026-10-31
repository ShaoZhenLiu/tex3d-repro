"""Fresh-state native one-step test for a source-conditioned feature transport.

All methods use the production renderer, source forward, regularizers, and real
PNG export. Each PNG is scored by the SAME native-source feature objective.
The old residual training entry is not used or declared regression-equivalent.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from local_transport import transport


def run(repo, output):
    from residual_lab import adapters as A
    from residual_lab.adapters import ResidualCtx, capture_quad, quad_to_dev, make_inputs, proprio_of, strict_inject, render_native_dual
    from libero.libero import benchmark
    torch.set_num_threads(2)
    torch.manual_seed(7);np.random.seed(7)
    device='cuda:0'
    ctx=ResidualCtx(device=device)
    for module in [ctx.model,ctx.ah,ctx.pp,ctx.nap]:
        if module is not None: module.eval().requires_grad_(False)
    source=repo/'experiments/refcloth_residual/20261031_run1'
    texture_table=json.loads((source/'textures/texture_table.json').read_text())
    # Asset paths frozen in plan, never selected by native or target scores.
    probe_ids=['fit_hi_iid_s109','fit_hi_lp_s111']
    probe_paths=[source/'textures'/(tid+'_theta.pt') for tid in probe_ids]
    starts={'init':None,
            'moderate':source/'textures/fit_mid_lp_s107_theta.pt',
            'frozen_mr':repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Ep0_Cloth_Delta_OV.pt'}
    output.mkdir(parents=True,exist_ok=True)
    report=dict(scope='fresh source states, identity UV, native one-step only',
        init_index=3,tasks=[0,2,6,8],probe_ids=probe_ids,eta=[.01,.05],
        no_target_queries=True,locked_test_consumed=False,rows=[],states=[],
        policy='production source; frozen parameters',branch='innovation-20261031')
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    def quad_feature(quad,inputs,proprio,grad=False):
        return (ctx.adv_forward_grad(quad,inputs,proprio) if grad else
                ctx.hidden_from_tensors(*ctx.compose(quad),inputs,proprio)).reshape(1,56,4096)
    try:
        for task_id in report['tasks']:
            task=suite.get_task(task_id);inits=suite.get_task_init_states(task_id)
            env,desc=A.base.get_libero_env(task,ctx.cfg.model_family,resolution=256)
            env.reset();obs=env.set_init_state(inits[3]);sim=env.env.sim;sim.forward();A.TR.align_cloth(env)
            _,bid,_=A.base.get_target_model_matrix(env,A.TR.SEARCH)
            tex_id=A.TR.ctu.find_target_texture_id(sim,bid,A.TR.SEARCH)
            original=strict_inject(sim,tex_id,ctx.t0_png)
            try:
                quad=capture_quad(env,obs,bid,device);quad=quad_to_dev(quad,device)
                inputs=make_inputs(ctx,quad['native_main_px'],desc)
                proprio=proprio_of(obs) if ctx.use_proprio else None
                def native_feature(png):
                    handle=strict_inject(sim,tex_id,png)
                    try:
                        main,wrist=render_native_dual(sim)
                        return ctx.hidden_from_arrays(main,wrist,inputs,proprio).reshape(1,56,4096)
                    finally:A.ctu.restore_mujoco_texture(sim,tex_id,*handle)
                n0=native_feature(ctx.t0_png).detach()
                repeat=native_feature(ctx.t0_png).detach()
                aa=float((n0-repeat).abs().max());assert aa==0,aa
                ctx.load_theta('zero');r0=quad_feature(quad,inputs,proprio).detach()
                bank_r=[];bank_n=[]
                for path in probe_paths:
                    ctx.load_theta(str(path))
                    bank_r.append((quad_feature(quad,inputs,proprio)-r0).detach())
                    bank_n.append((native_feature(ctx.renderer.export_png())-n0).detach())
                bank_r=torch.cat(bank_r);bank_n=torch.cat(bank_n)
                report['states'].append(dict(task=task_id,init=3,aa_maxdiff=aa,
                    baseline_feature_gap=float((r0-n0).square().mean())))
                for start_name,path in starts.items():
                    if path is None:
                        gen=torch.Generator(device='cpu').manual_seed(7)
                        theta=torch.randn(ctx.renderer.adv_noise.shape,generator=gen)*.01
                        ctx.load_theta(theta)
                    else:ctx.load_theta(str(path))
                    start_theta=ctx.renderer.adv_noise.detach().clone()
                    start_png=ctx.renderer.export_png()
                    native_start=float((native_feature(start_png)-n0).square().mean())
                    synth_start=quad_feature(quad,inputs,proprio).detach()-r0
                    start_prediction=transport(synth_start,bank_r,bank_n)
                    true_displacement=(native_feature(start_png)-n0).detach()
                    pred_ratio=float((start_prediction-true_displacement).square().sum()/
                                     (synth_start-true_displacement).square().sum().clamp_min(1e-20))
                    Image.fromarray(start_png).save(output/('t%d_%s_start.png'%(task_id,start_name)))
                    gradients={}
                    for method in ['matched','transport']:
                        ctx.load_theta(start_theta);ctx.renderer.adv_noise.grad=None
                        h=quad_feature(quad,inputs,proprio,grad=True)
                        displacement=h-r0
                        if method=='transport':displacement=transport(displacement,bank_r,bank_n)
                        feature_loss=(F.mse_loss(h,r0) if method=='matched' else
                                      F.mse_loss(displacement,torch.zeros_like(displacement)))
                        tex=ctx.renderer.texture_tensor()
                        amp=F.mse_loss(tex,ctx.t0_tex)
                        tv=A.M.spatial_tv_nhwc(tex-ctx.t0_tex)
                        loss=-10*feature_loss+.1*amp+.1*tv
                        loss.backward()
                        g=ctx.renderer.adv_noise.grad.detach().clone();gradients[method]=g
                        assert torch.isfinite(g).all()
                        for eta in report['eta']:
                            with torch.no_grad():new_theta=(start_theta-eta*g.sign()).clamp(-5,5)
                            ctx.load_theta(new_theta)
                            png=ctx.renderer.export_png()
                            native_score=float((native_feature(png)-n0).square().mean())
                            row=dict(task=task_id,start=start_name,method=method,eta=eta,
                                native_before=native_start,native_after=native_score,
                                native_gain=native_score-native_start,gradient_norm=float(g.norm()),
                                feature_prediction_error_ratio_at_start=pred_ratio,
                                changed_pixels=int(np.any(png!=start_png,axis=-1).sum()),
                                sha256=hashlib.sha256(png.tobytes()).hexdigest())
                            report['rows'].append(row)
                            Image.fromarray(png).save(output/('t%d_%s_%s_%s.png'%(task_id,start_name,method,eta)))
                            print(json.dumps(row),flush=True)
                    gm,gt=gradients['matched'].flatten(),gradients['transport'].flatten()
                    report['states'][-1][start_name+'_gradient_cosine']=float(F.cosine_similarity(gm,gt,dim=0))
                    (output/'gradient_probe.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            finally:
                A.ctu.restore_mujoco_texture(sim,tex_id,*original);env.close()
        report['completed']=True
        (output/'gradient_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:ctx.cleanup()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.repo,args.out)
