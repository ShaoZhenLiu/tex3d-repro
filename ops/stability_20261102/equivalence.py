"""Check that omitting unused frozen-model gradients preserves texture updates."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['1','2']
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.60,0)
    sys.path.insert(0,str(a.repo/'openvla-oft/experiments/robot/libero'))
    import attack_oft_cloth_ablate as R
    out=a.root/'equivalence_backward';out.mkdir(exist_ok=True)
    full=torch.load(a.root/'streams/seed_10_full.pt',map_location='cpu',weights_only=False)
    stream=out/'short.txt'
    lines=(a.root/'streams/seed_10.txt').read_text().splitlines()
    stream.write_text('\n'.join(lines[:17])+'\n')
    torch.save(dict(uv=full['uv'][:2]),out/'short_full.pt')
    original_model=R.base.get_model;original_modules=R.M._build_action_modules
    original_renderer=R.M.ClothRendererOV
    cached={};renderers=[];handles=[]
    original_backward=torch.Tensor.backward
    def configure(module):
        if module is not None:
            module.eval()
            for p in module.parameters():p.grad=None
        return module
    def get_model(cfg):
        if 'model' not in cached:cached['model']=original_model(cfg)
        return configure(cached['model'])
    def modules(cfg,model):
        if 'modules' not in cached:cached['modules']=original_modules(cfg,model)
        return tuple(configure(m) for m in cached['modules'])
    def renderer(*args,**kwargs):
        value=original_renderer(*args,**kwargs);renderers.append(value);return value
    R.base.get_model=get_model;R.M._build_action_modules=modules;R.M.ClothRendererOV=renderer
    R.OUT_ROOT=str(out/'training')
    outcomes={}
    for ref in ['native','mr']:
        arrays={}
        for mode in ['original','inputs_only','discard_unused']:
            for h in handles:h.remove()
            handles=[];torch.Tensor.backward=original_backward
            for module in ([cached['model']] if 'model' in cached else [])+list(cached.get('modules',[])):
                configure(module)
            if mode=='inputs_only':
                def restricted_backward(tensor,*args,**kwargs):
                    kwargs['inputs']=(renderers[-1].adv_noise,)
                    return original_backward(tensor,*args,**kwargs)
                torch.Tensor.backward=restricted_backward
            elif mode=='discard_unused':
                for module in [cached['model']]+list(cached.get('modules',[])):
                    if module is not None:
                        for param in module.parameters():
                            if param.requires_grad:
                                handles.append(param.register_post_accumulate_grad_hook(lambda p:setattr(p,'grad',None)))
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(10)
            opt=argparse.Namespace(tag='%s_%s'%(ref,mode),ref=ref,
                ref_switch=0,stream='replay',stream_file=str(stream),uv='on',filter_task=None,
                iters=2,batch=8,attack_lr=.05,epsilon=64/255,init_std=.01,
                lambda_ref=.1,lambda_tv=.1,alpha_action=0.,alpha_feature=10.,seed=10,
                checkpoint_every=2,ckpt_step_suffix=True,resume_dir='')
            start=time.monotonic();R.train(opt)
            rendered=renderers[-1]
            arrays[mode]=dict(theta=rendered.adv_noise.detach().float().cpu().clone(),
                gradient=rendered.adv_noise.grad.detach().float().cpu().clone(),
                seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        outcomes[ref]={}
        for mode in ['inputs_only','discard_unused']:
            old,new=arrays['original'],arrays[mode]
            relative=float(torch.linalg.vector_norm(old['gradient']-new['gradient'])/torch.linalg.vector_norm(old['gradient']))
            exact=bool(torch.equal(old['theta'],new['theta']))
            same_sign=float((old['gradient'].sign()==new['gradient'].sign()).float().mean())
            outcomes[ref][mode]=dict(passed=relative<1e-6 and exact and same_sign==1.,relative_gradient_l2=relative,
                texture_parameters_bitwise_equal=exact,gradient_sign_agreement=same_sign,
                original_seconds=old['seconds'],optimized_seconds=new['seconds'],
                original_peak_cuda_bytes=old['peak_cuda_bytes'],optimized_peak_cuda_bytes=new['peak_cuda_bytes'])
    modes=[mode for mode in ['inputs_only','discard_unused'] if all(row[mode]['passed'] for row in outcomes.values())]
    result=dict(passed=bool(modes),validated_modes=modes,paired_updates=2,batch=8,references=outcomes,
        interpretation='Network parameter requires-grad flags and all forward computations are preserved; unused model gradients are pruned or immediately released.')
    (out/'equivalence.json').write_text(json.dumps(result,indent=2))
    print('EQUIVALENCE_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
