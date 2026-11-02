"""Compare backward requests on the SAME tape before accepting a memory optimization."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78,0)
    sys.path.insert(0,str(a.repo/'openvla-oft/experiments/robot/libero'))
    import attack_oft_cloth_ablate as R
    out=a.root/'same_graph';out.mkdir(exist_ok=True)
    full=torch.load(a.root/'streams/seed_10_full.pt',map_location='cpu',weights_only=False)
    stream=out/'short.txt';stream.write_text('\n'.join((a.root/'streams/seed_10.txt').read_text().splitlines()[:9])+'\n')
    torch.save(dict(uv=full['uv'][:1]),out/'short_full.pt')
    model_get=R.base.get_model;module_get=R.M._build_action_modules;renderer_class=R.M.ClothRendererOV
    cached={};renderers=[];comparisons=[];backward=torch.Tensor.backward
    def get_model(cfg):
        if 'model' not in cached:cached['model']=model_get(cfg)
        return cached['model']
    def get_modules(cfg,model):
        if 'modules' not in cached:cached['modules']=module_get(cfg,model)
        return cached['modules']
    def get_renderer(*args,**kwargs):
        value=renderer_class(*args,**kwargs);renderers.append(value);return value
    def clear():
        for module in [cached['model']]+list(cached.get('modules',[])):
            if module is not None:
                for param in module.parameters():param.grad=None
    def compare(tensor,*args,**kwargs):
        param=renderers[-1].adv_noise
        prior=None if param.grad is None else param.grad.detach().clone()
        def reset():
            clear();param.grad=None if prior is None else prior.clone()
        options=dict(kwargs);options['retain_graph']=True
        backward(tensor,*args,**options);original=param.grad.detach().clone()
        reset();backward(tensor,*args,**options);repeat=param.grad.detach().clone()
        reset();options['retain_graph']=False;options['inputs']=(param,)
        backward(tensor,*args,**options);optimized=param.grad.detach().clone()
        denominator=torch.linalg.vector_norm(original).clamp_min(1e-30)
        def stats(candidate):
            different=candidate.sign()!=original.sign()
            return dict(relative_l2=float(torch.linalg.vector_norm(candidate-original)/denominator),
                sign_disagreement=float(different.float().mean()),
                weighted_sign_disagreement=float((original.abs()*different).sum()/original.abs().sum().clamp_min(1e-30)),
                bitwise_equal=bool(torch.equal(original,candidate)))
        row=dict(reference=current_ref,repeat=stats(repeat),inputs_only=stats(optimized))
        comparisons.append(row)
        param.grad=original
        return None
    R.base.get_model=get_model;R.M._build_action_modules=get_modules;R.M.ClothRendererOV=get_renderer
    R.OUT_ROOT=str(out/'training')
    try:
        torch.Tensor.backward=compare
        for current_ref in ['native','mr']:
            if 'model' in cached:clear()
            torch.manual_seed(10)
            options=argparse.Namespace(tag=current_ref,ref=current_ref,ref_switch=0,stream='replay',
                stream_file=str(stream),uv='on',filter_task=None,iters=1,batch=8,attack_lr=.05,
                epsilon=64/255,init_std=.01,lambda_ref=.1,lambda_tv=.1,alpha_action=0.,alpha_feature=10.,
                seed=10,checkpoint_every=1,ckpt_step_suffix=True,resume_dir='')
            R.train(options)
    finally:torch.Tensor.backward=backward
    passed=all(r['inputs_only']['relative_l2']<=max(2e-6,5*r['repeat']['relative_l2'])
               and r['inputs_only']['weighted_sign_disagreement']<=max(2e-6,5*r['repeat']['weighted_sign_disagreement'])
               for r in comparisons)
    result=dict(passed=passed,comparisons=comparisons,
        all_bitwise_equal=all(r['inputs_only']['bitwise_equal'] for r in comparisons),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='Same renderer parameters and same model-forward tape; compare full, repeated-full, and texture-only backward requests')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('SAME_GRAPH_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
