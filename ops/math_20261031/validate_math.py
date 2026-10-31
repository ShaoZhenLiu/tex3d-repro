"""Export fit-only universal steps, evaluate on independent source initial6."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from probe_common import SourceRuntime, rms


def main(repo,directions,output,tasks,memory_fraction):
    torch.manual_seed(7)
    np.random.seed(7)
    runtime=SourceRuntime(repo,memory_fraction)
    ctx=runtime.ctx
    fitted=torch.load(directions,map_location='cpu',weights_only=False)
    assert fitted['fit_initial']==5
    output.mkdir(parents=True,exist_ok=True)
    report=dict(scope='native one-step, fit initial5 -> source heldout initial6; no closed-loop',
                tasks=tasks,validation_initial=6,records=[],states=[],completed=False)
    def tex_stats():
        with torch.no_grad():
            tex=ctx.renderer.texture_tensor()
            return dict(amp=float((tex-ctx.t0_tex).square().mean()),
                        tv=float(runtime.A.M.spatial_tv_nhwc(tex-ctx.t0_tex)))
    for task in tasks:
        with runtime.state(task,6) as st:
            with torch.no_grad():a0=runtime.action(st.native0)
            report['states'].append(dict(task=task,zero_png_maxdiff=st.zero_png_maxdiff,
                seg_area=st.quad['seg_area'],aa_feature_maxdiff=0))
            for start,entry in fitted['starts'].items():
                theta=entry['theta'].to(ctx.device)
                ctx.load_theta(theta)
                before_png=ctx.renderer.export_png()
                before_h=st.native_aa(before_png)
                with torch.no_grad():before_action=runtime.action(before_h)
                before_tex=tex_stats()
                before_fd=float((before_h-st.native0).square().mean())
                before_act=float((before_action-a0).square().mean())
                before_j=10*before_fd-.1*before_tex['amp']-.1*before_tex['tv']
                # Used only for a diagnostic inequality; it is not task success.
                bn=st.native_all(before_png)
                with torch.no_grad():ba={v:runtime.action(h) for v,h in bn.items()}
                before_solo={v:float((ba[v]-a0).square().mean()) for v in ['AP','PA']}
                for eta,methods in entry['steps'].items():
                    for method,delta in methods.items():
                        ctx.load_theta((theta+delta.to(ctx.device)).clamp(-5,5))
                        png=ctx.renderer.export_png()
                        nh=st.native_aa(png)
                        with torch.no_grad():na=runtime.action(nh)
                        stats=tex_stats()
                        fd=float((nh-st.native0).square().mean())
                        act=float((na-a0).square().mean())
                        j=10*fd-.1*stats['amp']-.1*stats['tv']
                        record=dict(task=task,start=start,eta=float(eta),method=method,
                            native_fd_before=before_fd,native_fd_after=fd,native_fd_gain=fd-before_fd,
                            native_action_before=before_act,native_action_after=act,native_action_gain=act-before_act,
                            native_total_before=before_j,native_total_after=j,native_total_gain=j-before_j,
                            amp=stats['amp'],tv=stats['tv'],
                            png_changed_pixels=int(np.any(png!=before_png,axis=-1).sum()),
                            png_sha256=hashlib.sha256(png.tobytes()).hexdigest())
                        if method.startswith('interaction') or method=='action_mean':
                            native=st.native_all(png)
                            with torch.no_grad():actions={v:runtime.action(h) for v,h in native.items()}
                            inter=actions['AA']-actions['AP']-actions['PA']+actions['PP']
                            record['native_interaction_mse']=float(inter.square().mean())
                            record['solo_before']=before_solo
                            record['solo_after']={v:float((actions[v]-a0).square().mean()) for v in ['AP','PA']}
                            record['solo_no_increase']=all(record['solo_after'][v]<=before_solo[v]+1e-8 for v in before_solo)
                        report['records'].append(record)
                        Image.fromarray(png).save(output/('t%d_%s_%s_%s.png'%(task,start,method,eta)))
                        print(json.dumps(record),flush=True)
                    (output/'validation.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['completed']=True
    report['gpu_peak_bytes']=torch.cuda.max_memory_allocated()
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--directions',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tasks',default='0,2,6,8')
    ap.add_argument('--memory-fraction',type=float,default=.48)
    a=ap.parse_args()
    main(a.repo,a.directions,a.out,[int(t) for t in a.tasks.split(',')],a.memory_fraction)
