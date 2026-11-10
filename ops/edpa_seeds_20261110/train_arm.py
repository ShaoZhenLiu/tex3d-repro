"""Production EDPA forwards/loss, exact replay, texture-only backward and recovery."""
import argparse
import ast
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from common import HERE,REPO,ROOT,POOL,POOL_SHA,PRODUCTION,SEEDS,read,sha,write
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard


def load_modules():
    sys.path.insert(0,str(PRODUCTION.parent))
    import attack_oft_cloth_edpa_tvfix as M
    sys.path.insert(0,str(REPO/'ops/stability_20261102'))
    import attack_oft_cloth_ablate as shared
    return M,shared


def exact_loss_code():
    source=PRODUCTION.read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='it')
    first=next(n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='logits' for t in n.targets))
    last=next(n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='objective' for t in n.targets))
    fragment=textwrap.dedent(''.join(source.splitlines(keepends=True)[first.lineno-1:last.end_lineno]))
    return compile(fragment,str(PRODUCTION),'exec'),hashlib.sha256(fragment.encode()).hexdigest()


def to_device(value):
    # Clone mapping containers; BatchEncoding.to() mutates its own dictionary.
    if torch.is_tensor(value):return value.to('cuda',non_blocking=True)
    if isinstance(value,Mapping):return {k:to_device(v) for k,v in value.items()}
    return value


def build_loss(M,model,renderer,frames,uv,ema,code):
    base=M.base;device='cuda';sm,ss,dm,ds=M._norm_consts(device)
    adv_pvs=[];clean_pvs=[];ids_list=[];mask_list=[]
    for _,raw in frames:
        fr={k:to_device(raw[k]) for k in ['mvp','rot','seg','bg','text_inputs','wrist','wrist_mvp','wrist_seg','wrist_bg'] if k in raw}
        def compose(clean):
            tex=renderer.orig_texture if clean else None
            adv,_=renderer.render(fr['mvp'],(256,256),model_rot=fr['rot'],uv_xform=uv,tex_override=tex)
            comp=(adv.permute(0,3,1,2)*fr['seg']+fr['bg']*(1-fr['seg'])).clamp(0,1)
            wrist=fr.get('wrist')
            if fr.get('wrist_mvp') is not None and fr.get('wrist_bg') is not None:
                image,_=renderer.render(fr['wrist_mvp'],(256,256),model_rot=fr['rot'],uv_xform=uv,tex_override=tex)
                wrist=(image.permute(0,3,1,2)*fr['wrist_seg']+fr['wrist_bg']*(1-fr['wrist_seg'])).clamp(0,1)
            return base._compose_oft_pixel_values(adv_rgb=comp,model_input_size=224,
                siglip_mean=sm,siglip_std=ss,dino_mean=dm,dino_std=ds,wrist_tensor=wrist).to(torch.bfloat16)
        adv_pvs.append(compose(False))
        with torch.no_grad():clean_pvs.append(compose(True))
        ids_list.append(fr['text_inputs']['input_ids'][0]);mask_list.append(fr['text_inputs']['attention_mask'][0])
    maxlen=max(int(x.shape[0]) for x in ids_list)
    ids=torch.zeros((len(ids_list),maxlen),dtype=torch.long,device=device)
    tmask=torch.zeros_like(ids)
    for i,(ii,mm) in enumerate(zip(ids_list,mask_list)):
        ids[i,:ii.shape[0]]=ii;tmask[i,:mm.shape[0]]=mm
    with base.autocast(dtype=torch.bfloat16):
        adv_embed=model.projector(model.vision_backbone(torch.cat(adv_pvs,dim=0)))
    with torch.no_grad(),base.autocast(dtype=torch.bfloat16):
        clean_embed=model.projector(model.vision_backbone(torch.cat(clean_pvs,dim=0)))
        lang_embed=model.get_input_embeddings()(ids)
    scope=dict(torch=torch,F=M.F,adv_embed=adv_embed.float(),clean_embed=clean_embed.float(),
        lang_embed=lang_embed.float(),tmask=tmask,ema_step=ema['step'],ema_warmup=100,ema_decay=.9,
        ema_patch_v=ema['patch'],ema_align_v=ema['align'],renderer=renderer,t0_tex=renderer.orig_texture,
        args=SimpleNamespace(edpa_alpha=.2,lambda_ref=.1,lambda_tv=.1))
    exec(code,scope)
    return scope


def validation(model,renderer,frames,uv,M,code,loss_sha):
    ctx=build_loss(M,model,renderer,frames,uv,dict(step=0,patch=0.,align=0.),code)
    param=renderer.adv_noise;full=[]
    for _ in range(2):
        model.zero_grad(set_to_none=True);param.grad=None
        ctx['objective'].backward(retain_graph=True)
        full.append(param.grad.detach().float().cpu().clone())
    model.zero_grad(set_to_none=True);param.grad=None
    requested=torch.autograd.grad(ctx['objective'],param)[0].detach().float().cpu()
    def compare(a,b):
        return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-20)),
            sign_disagreement=float((a.sign()!=b.sign()).float().mean()))
    repeat=compare(full[0],full[1]);delta=compare(requested,full[1])
    assert requested.norm()>0 and torch.isfinite(requested).all()
    assert delta['relative_l2']<=max(5*repeat['relative_l2'],1e-4),(repeat,delta)
    assert delta['sign_disagreement']<=max(5*repeat['sign_disagreement'],.002),(repeat,delta)
    del ctx,requested,full;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    ctx=build_loss(M,model,renderer,frames,uv,dict(step=0,patch=0.,align=0.),code)
    gradient=torch.autograd.grad(ctx['objective'],param)[0]
    assert torch.isfinite(gradient).all()
    result=dict(passed=True,repeat_backward=repeat,texture_request=delta,loss_ast_sha256=loss_sha,
        steady_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        steady_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        objective=float(ctx['objective'].detach()),patch=float(ctx['patch_loss'].mean().detach()),
        alignment=float(ctx['align_loss'].mean().detach()),device=torch.cuda.get_device_name(),time=time.time())
    write(ROOT/'gradient_validation.json',result);print('GRADIENT_VALIDATED '+json.dumps(result),flush=True)


def main(a):
    guard.install_pause_handler();torch.set_num_threads(2)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','2','3']
    assert read(ROOT/('smoke_'+a.node+'/prepared.json'),{}).get('passed')
    inputs=read(ROOT/'inputs.json');assert inputs['passed']
    if not a.validate_only:assert read(ROOT/'gradient_validation.json',{}).get('passed')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,0)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    M,shared=load_modules();assert M.TR.patch_status()=='patched'
    record=inputs['streams'][str(a.seed)];stream=ROOT/record['text_path'];uv_path=ROOT/record['uv_path']
    assert sha(stream)==record['text_sha256'] and sha(uv_path)==record['uv_sha256']
    replay=shared.parse_stream(str(stream),5000,8)
    full=torch.load(uv_path,map_location='cpu',weights_only=False)['uv']
    assert len(full)==5000
    pool=torch.load(str(POOL),map_location='cpu',weights_only=False,mmap=True)
    frames,_=shared.active_frames(pool,None);assert len(frames)==296
    cfg=M.base.GenerateConfig(pretrained_checkpoint=M.CKPT,model_family='openvla')
    model=M.base.get_model(cfg);model.eval();M.base.check_unnorm_key(cfg,model)
    assert M.base.get_image_resize_size(cfg)==224
    renderer=M.ClothRendererOV(device='cuda',epsilon=64/255,init_std=.01,seed=a.seed).to('cuda')
    param=renderer.adv_noise
    theta_sha=hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
    if record.get('theta_init_sha256'):assert theta_sha==record['theta_init_sha256']
    code,loss_sha=exact_loss_code()
    guard.mark_ready()
    def batch(it):
        return [frames[i] for i in replay[it]['frames']],torch.as_tensor(full[it],device='cuda')
    if a.validate_only:
        sample,uv=batch(0);validation(model,renderer,sample,uv,M,code,loss_sha);return
    out=ROOT/'arms'/('s%d_edpa'%a.seed);out.mkdir(parents=True,exist_ok=True)
    assert not read(out/'completed.json',{}).get('completed')
    registration=dict(seed=a.seed,method='cloth_edpa',iterations=5000,batch=8,frame_visits=40000,
        source_production_sha256=sha(PRODUCTION),loss_ast_sha256=loss_sha,stream=record,
        protocol_sha256=sha(HERE/'protocol.json'),theta_init_sha256=theta_sha,
        host=os.uname().nodename,gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu_name=torch.cuda.get_device_name(),
        pytorch=torch.__version__,cuda=torch.version.cuda,memory_fraction=a.memory_fraction,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,universal_newlines=True).strip(),
        gradient_mode='texture_only',gradient_validation_sha256=sha(ROOT/'gradient_validation.json'),
        pool_mmap=True,time=time.time())
    write(out/'registrations'/('%d.json'%time.time_ns()),registration);write(out/'registered.json',registration)
    initial=0;ema=dict(step=0,patch=0.,align=0.);checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==registration['protocol_sha256']
        assert state['loss_ast_sha256']==loss_sha and state['stream_sha256']==record['text_sha256']
        with torch.no_grad():param.copy_(state['parameter'].to('cuda'))
        initial=state['next_step'];ema=state['ema'];assert ema['step']==initial
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state(state['rng_cuda'])
    def save(next_step):
        state=dict(parameter=param.detach().cpu(),next_step=next_step,ema=ema,
            protocol_sha256=registration['protocol_sha256'],loss_ast_sha256=loss_sha,
            stream_sha256=record['text_sha256'],rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state())
        temporary=checkpoint.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint)
        write(out/'progress.json',dict(next_step=next_step,iterations=5000,ema=ema,time=time.time(),host=os.uname().nodename))
    start=time.monotonic()
    with (out/'gradient_log.txt').open('a',buffering=1) as log:
        for it in range(initial,5000):
            selected,uv=batch(it);ctx=build_loss(M,model,renderer,selected,uv,ema,code)
            gradient=torch.autograd.grad(ctx['objective'],param)[0]
            assert torch.isfinite(gradient).all() and torch.isfinite(ctx['objective'])
            ema=dict(step=it+1,patch=ctx['ema_patch_v'],align=ctx['ema_align_v'])
            with torch.no_grad():param.add_(.05*gradient.sign());param.clamp_(-5,5)
            if it%50==0 or it==4999:
                stats=dict(it=it,cost=float(ctx['edpa_cost'].detach()),patch=float(ctx['patch_loss'].mean().detach()),
                    alignment=float(ctx['align_loss'].mean().detach()),amplitude=float(ctx['ref'].detach()),
                    tv=float(ctx['tv'].detach()),gradient_norm=float(gradient.norm()),
                    wall_seconds=time.monotonic()-start,ema=ema)
                line=json.dumps(stats);print('TRAIN '+line,flush=True);log.write(line+'\n')
                write(out/'progress.json',dict(stats,next_step=it+1))
            del ctx,gradient,selected,uv
            if (it+1)%250==0 or it==4999 or guard.pause_requested():save(it+1)
            if guard.pause_requested():raise SystemExit(75)
    image=out/'Universal_Cloth_OV.png';Image.fromarray(renderer.export_texture()).save(image)
    target=ROOT/'frozen'/('s%d_edpa.png'%a.seed);target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.tmp');temporary.write_bytes(image.read_bytes());temporary.replace(target)
    rms,mab,linf,tv=M.texture_stats(renderer);assert linf<=64.0001
    result=dict(registration,completed=True,iterations=5000,texture=str(image),texture_sha256=sha(image),
        frozen_path=str(target.relative_to(ROOT)),rms=rms,linf=linf,tv=tv,
        elapsed_segment_seconds=time.monotonic()-start,resumed_from=initial,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,final_ema=ema)
    write(out/'completed.json',result);print('ARM_COMPLETED '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--node',choices=['4090','09'],required=True)
    p.add_argument('--memory-fraction',type=float,default=.60)
    p.add_argument('--validate-only',action='store_true')
    main(p.parse_args())
