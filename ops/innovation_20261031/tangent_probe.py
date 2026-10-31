"""Current-texture native tangent correction, including a BPDA anchor control.

This is a new exploratory hypothesis after zero-anchored feature transport failed
the native-update gate. It uses fresh source initial state 4, never targets.
The standard BPDA/least-squares components are controls, not novelty claims.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from local_transport import transport


def run(repo,output):
    from residual_lab import adapters as A
    from residual_lab.adapters import ResidualCtx,capture_quad,quad_to_dev,make_inputs,proprio_of,strict_inject,render_native_dual
    from libero.libero import benchmark
    torch.set_num_threads(2);torch.manual_seed(7);np.random.seed(7)
    device='cuda:0';ctx=ResidualCtx(device=device)
    for module in [ctx.model,ctx.ah,ctx.pp,ctx.nap]:
        if module is not None:module.eval().requires_grad_(False)
    source=repo/'experiments/refcloth_residual/20261031_run1'
    starts={'init':None,'moderate':source/'textures/fit_mid_lp_s107_theta.pt',
            'frozen_mr':repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Ep0_Cloth_Delta_OV.pt'}
    report=dict(init_index=4,tasks=[0,2,6,8],eta=[.01,.05],probe_radius=.05,
                methods=['matched','native_anchor','local_tangent'],rows=[],states=[],
                target_queries=0,locked_test_consumed=False,
                scope='native one-step direction probe; not training or closed-loop evaluation')
    output.mkdir(parents=True,exist_ok=True)
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    try:
        for task_id in report['tasks']:
            task=suite.get_task(task_id);inits=suite.get_task_init_states(task_id)
            env,desc=A.base.get_libero_env(task,ctx.cfg.model_family,resolution=256)
            env.reset();obs=env.set_init_state(inits[4]);sim=env.env.sim;sim.forward();A.TR.align_cloth(env)
            _,bid,_=A.base.get_target_model_matrix(env,A.TR.SEARCH)
            tex_id=A.TR.ctu.find_target_texture_id(sim,bid,A.TR.SEARCH)
            original=strict_inject(sim,tex_id,ctx.t0_png)
            try:
                quad=capture_quad(env,obs,bid,device);quad=quad_to_dev(quad,device)
                inputs=make_inputs(ctx,quad['native_main_px'],desc)
                proprio=proprio_of(obs) if ctx.use_proprio else None
                def native(png):
                    handle=strict_inject(sim,tex_id,png)
                    try:
                        main,wrist=render_native_dual(sim)
                        return ctx.hidden_from_arrays(main,wrist,inputs,proprio).reshape(1,56,4096)
                    finally:A.ctu.restore_mujoco_texture(sim,tex_id,*handle)
                def synth(grad=False):
                    return (ctx.adv_forward_grad(quad,inputs,proprio) if grad else
                            ctx.hidden_from_tensors(*ctx.compose(quad),inputs,proprio)).reshape(1,56,4096)
                def regularizer():
                    tex=ctx.renderer.texture_tensor()
                    return .1*F.mse_loss(tex,ctx.t0_tex)+.1*A.M.spatial_tv_nhwc(tex-ctx.t0_tex)
                n0=native(ctx.t0_png).detach();aa=float((native(ctx.t0_png)-n0).abs().max());assert aa==0,aa
                ctx.load_theta('zero');r0=synth().detach()
                report['states'].append(dict(task=task_id,init=4,aa_maxdiff=aa))
                for start_name,path in starts.items():
                    if path is None:
                        gen=torch.Generator(device='cpu').manual_seed(7)
                        ctx.load_theta(torch.randn(ctx.renderer.adv_noise.shape,generator=gen)*.01)
                    else:ctx.load_theta(str(path))
                    theta=ctx.renderer.adv_noise.detach().clone();png0=ctx.renderer.export_png()
                    rc=synth().detach();nc=native(png0).detach();dnc=nc-n0
                    native_before=float(dnc.square().mean())
                    ctx.renderer.adv_noise.grad=None
                    loss=-10*F.mse_loss(synth(grad=True),r0)+regularizer();loss.backward()
                    matched=ctx.renderer.adv_noise.grad.detach().clone()
                    v0=matched.sign()
                    gen=torch.Generator(device='cpu').manual_seed(1900+task_id)
                    v1=torch.randn(theta.shape,generator=gen).sign().to(device)
                    v1=v1-(v1*v0).sum()/v0.square().sum().clamp_min(1)*v0
                    v1=v1/v1.abs().max().clamp_min(1e-20)
                    bank_r=[];bank_n=[]
                    for direction in [v0,v1]:
                        rf=[];nf=[]
                        for sign in [-1,1]:
                            ctx.load_theta((theta+sign*.05*direction).clamp(-5,5))
                            rf.append(synth().detach());nf.append(native(ctx.renderer.export_png()).detach())
                        bank_r.append((rf[1]-rf[0])/.1);bank_n.append((nf[1]-nf[0])/.1)
                    bank_r=torch.cat(bank_r);bank_n=torch.cat(bank_n)
                    for method in report['methods']:
                        ctx.load_theta(theta);ctx.renderer.adv_noise.grad=None
                        if method=='matched':g=matched
                        else:
                            displacement=synth(grad=True)-rc
                            if method=='local_tangent':displacement=transport(displacement,bank_r,bank_n)
                            prediction=dnc+displacement
                            loss=-10*F.mse_loss(prediction,torch.zeros_like(prediction))+regularizer()
                            loss.backward();g=ctx.renderer.adv_noise.grad.detach().clone()
                        assert torch.isfinite(g).all()
                        for eta in report['eta']:
                            ctx.load_theta((theta-eta*g.sign()).clamp(-5,5))
                            png=ctx.renderer.export_png();after=float((native(png)-n0).square().mean())
                            row=dict(task=task_id,start=start_name,method=method,eta=eta,native_before=native_before,
                                native_after=after,native_gain=after-native_before,gradient_norm=float(g.norm()),
                                changed_pixels=int(np.any(png!=png0,axis=-1).sum()))
                            report['rows'].append(row);print(json.dumps(row),flush=True)
                            Image.fromarray(png).save(output/('t%d_%s_%s_%s.png'%(task_id,start_name,method,eta)))
                    (output/'tangent_probe.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            finally:A.ctu.restore_mujoco_texture(sim,tex_id,*original);env.close()
        report['completed']=True
        (output/'tangent_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:ctx.cleanup()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.repo,args.out)
