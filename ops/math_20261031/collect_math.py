"""Fit-state source gradients and native/synthesis cross-view fidelity.

Only fit initial index5 is used to construct universal steps. No target policy.
Native actions use the frozen production head, first five translations only.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from probe_common import SourceRuntime, fidelity, rms


def main(repo, output, tasks, initial, memory_fraction):
    torch.manual_seed(7)
    np.random.seed(7)
    runtime = SourceRuntime(repo, memory_fraction)
    ctx = runtime.ctx
    output.mkdir(parents=True, exist_ok=True)
    report = dict(tasks=tasks, initial=initial, scope='source-only gradients and native fidelity, not closed-loop',
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
                  records=[], states=[], gpu_peak_bytes=None)
    bank = dict(tasks=tasks, initial=initial, starts={})
    for start in ['init', 'moderate', 'frozen_mr']:
        theta = runtime.start_theta(start)
        ctx.load_theta(theta)
        texture=ctx.renderer.texture_tensor()
        amp=(texture-ctx.t0_tex).square().mean()
        tv=runtime.A.M.spatial_tv_nhwc(texture-ctx.t0_tex)
        penalty=.1*amp+.1*tv
        gpenalty=torch.autograd.grad(penalty,ctx.renderer.adv_noise)[0].detach().cpu()
        bank['starts'][start] = dict(theta=theta.cpu(), gradients={}, penalty_gradient=gpenalty,
            amp=float(amp.detach()),tv=float(tv.detach()))
    for task in tasks:
        with runtime.state(task, initial) as st:
            report['states'].append(dict(task=task, zero_png_maxdiff=st.zero_png_maxdiff,
                seg_area=st.quad['seg_area'], aa_feature_maxdiff=0))
            for start in bank['starts']:
                theta = bank['starts'][start]['theta'].to(ctx.device)
                ctx.load_theta(theta)
                png = ctx.renderer.export_png()
                n = st.native_all(png)
                r = {v: st.synth(v).detach() for v in ['PP','AP','PA','AA']}
                with torch.no_grad():
                    na = {k: runtime.action(v) for k,v in n.items()}
                    ra = {k: runtime.action(v) for k,v in r.items()}
                inat = na['AA']-na['AP']-na['PA']+na['PP']
                iren = ra['AA']-ra['AP']-ra['PA']+ra['PP']
                rec = dict(task=task,start=start,action_displacement=fidelity(ra['AA']-ra['PP'],na['AA']-na['PP']),
                    action_interaction=fidelity(iren,inat),
                    native_joint_rms=rms(na['AA']-na['PP']),
                    native_solo_main_rms=rms(na['AP']-na['PP']),
                    native_solo_wrist_rms=rms(na['PA']-na['PP']),
                    feature_displacement=fidelity(r['AA']-r['PP'],n['AA']-n['PP']),
                    feature_interaction=fidelity(r['AA']-r['AP']-r['PA']+r['PP'],n['AA']-n['AP']-n['PA']+n['PP']))
                gradients = {}
                # Head-sized graph determines per-view VJP coefficients. Each
                # backbone backward is separate to cap shared-GPU memory use.
                for method in ['fd','action','interaction','solo_main','solo_wrist']:
                    ctx.load_theta(theta)
                    if method == 'fd':
                        h = st.synth('AA', grad=True)
                        loss = (h-st.render0).square().mean()
                        gradients[method] = torch.autograd.grad(loss,ctx.renderer.adv_noise)[0].detach().cpu()
                    else:
                        leaves = {v: ra[v].detach().clone().requires_grad_(v!='PP') for v in ra}
                        interaction = leaves['AA']-leaves['AP']-leaves['PA']+leaves['PP']
                        solo = (leaves['AP']-leaves['PP']).square().mean()+(leaves['PA']-leaves['PP']).square().mean()
                        loss = ((leaves['AA']-leaves['PP']).square().mean() if method=='action' else
                                (leaves['AP']-leaves['PP']).square().mean() if method=='solo_main' else
                                (leaves['PA']-leaves['PP']).square().mean() if method=='solo_wrist' else
                                interaction.square().mean())
                        views = (['AA'] if method=='action' else ['AP'] if method=='solo_main' else
                                 ['PA'] if method=='solo_wrist' else ['AA','AP','PA'])
                        coeffs = torch.autograd.grad(loss,[leaves[v] for v in views])
                        total = torch.zeros_like(ctx.renderer.adv_noise)
                        for v, coeff in zip(views, coeffs):
                            h = st.synth(v, grad=True)
                            a = runtime.action(h)
                            total += torch.autograd.grad(a, ctx.renderer.adv_noise, grad_outputs=coeff)[0].detach()
                        gradients[method] = total.cpu()
                    assert torch.isfinite(gradients[method]).all()
                    bank['starts'][start]['gradients'].setdefault(method,[]).append(gradients[method])
                gradients['protected']=gradients['interaction']-gradients['solo_main']-gradients['solo_wrist']
                bank['starts'][start]['gradients'].setdefault('protected',[]).append(gradients['protected'])
                rec['gradient_norms']={k:float(v.norm()) for k,v in gradients.items()}
                report['records'].append(rec)
                print(json.dumps(rec),flush=True)
                (output/'collection.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for entry in bank['starts'].values():
        entry['gradients']={k:torch.stack(v) for k,v in entry['gradients'].items()}
    report['gpu_peak_bytes']=torch.cuda.max_memory_allocated()
    report['completed']=True
    torch.save(bank,output/'gradient_bank.partial.pt')
    (output/'gradient_bank.partial.pt').replace(output/'gradient_bank.pt')
    (output/'collection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tasks',default='0,2,6,8')
    ap.add_argument('--initial',type=int,default=5)
    ap.add_argument('--memory-fraction',type=float,default=.49)
    a=ap.parse_args()
    main(a.repo,a.out,[int(t) for t in a.tasks.split(',')],a.initial,a.memory_fraction)
