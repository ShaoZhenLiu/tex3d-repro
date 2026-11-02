"""Frozen-source view interventions and action-token attention reconstruction."""
import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from native_runtime import NativeRuntime


def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(x,path):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False));temp.replace(path)


def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','6']
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(7);np.random.seed(7)
    runtime=NativeRuntime(a.repo,.65,a.out)
    ctx=runtime.ctx
    assert ctx.use_wrist and ctx.size==224 and not ctx.cfg.use_diffusion
    textures={name:np.array(Image.open(a.assets/(file+'.png')).convert('RGB')) for name,file in
              [('plain','Plain'),('refcloth','RefCloth'),('native_ref','Native_reference'),('no_uv','No_UV'),
               ('random101','random_cloth_s101'),('random102','random_cloth_s102'),('random103','random_cloth_s103')]}
    assert np.array_equal(textures['plain'],runtime.base) and np.array_equal(textures['refcloth'],runtime.full)
    att=ctx.model.language_model.model.layers[-1].self_attn
    captured={}
    original=att.forward
    assert 'Sdpa' in type(att).__name__,type(att).__name__
    def trace_attention(*args,**kwargs):
        kernel=F.scaled_dot_product_attention
        def capture(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,**kw):
            result=kernel(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,**kw)
            captured.update(q=q.detach().transpose(1,2).clone(),k=k.detach().transpose(1,2).clone(),
                v=v.detach().transpose(1,2).clone(),output=result.detach().transpose(1,2).clone(),
                mask=None if attn_mask is None else attn_mask.detach().clone(),
                length=int(q.shape[-2]),dropout=dropout_p,scale=kw.get('scale'),causal=bool(is_causal))
            return result
        F.scaled_dot_product_attention=capture
        try:return original(*args,**kwargs)
        finally:F.scaled_dot_product_attention=kernel
    att.forward=trace_attention
    method=ctx.model._regression_or_discrete_prediction
    signature=inspect.signature(method)
    def trace(*args,**kwargs):
        values=signature.bind(*args,**kwargs).arguments
        captured['patches']=int(values['NUM_PATCHES']);captured['prompt']=int(values['NUM_PROMPT_TOKENS'])
        return method(*args,**kwargs)
    ctx.model._regression_or_discrete_prediction=trace
    (a.out/'attention_implementation.txt').write_text(inspect.getsource(original))
    report=dict(completed=False,tasks=a.tasks,initial=21,steps=[10,50],source_only=True,
        attention='Last language-model SDPA layer; mean over heads and 56 action-token queries; raw visual-key probabilities',
        intervention='Counterfactual main/wrist inputs at frozen physical states; no texture training or closed-loop evaluation',
        semantic_mechanism_tested=False,records=[],commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),
        texture_sha256={k:sha(v) for k,v in textures.items()},gpu=os.environ['CUDA_VISIBLE_DEVICES'])

    def attention():
        q,k,v=captured['q'],captured['k'],captured['v']
        assert q.shape[0]==k.shape[0]==1 and q.shape[1]==k.shape[1]==captured['length']
        assert captured['dropout']==0
        begin=captured['patches']+captured['prompt'];end=begin+56
        assert end<=q.shape[1] and ctx.model.vision_backbone.get_num_patches()==256
        q=q[:,begin:end].transpose(1,2).float();k=k.transpose(1,2).float();v=v.transpose(1,2).float()
        factor=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(factor,dim=1);v=v.repeat_interleave(factor,dim=1)
        logits=q@k.transpose(-1,-2)
        logits*=captured['scale'] if captured['scale'] is not None else 1/math.sqrt(q.shape[-1])
        causal=captured['causal']
        if causal:
            keypos=torch.arange(k.shape[-2],device=k.device)[None,:]
            querypos=torch.arange(begin,end,device=k.device)[:,None]
            logits.masked_fill_((keypos>querypos)[None,None],float('-inf'))
        mask=captured['mask']
        if mask is not None:
            assert mask.ndim==4,mask.shape
            mask=mask if mask.shape[-2]==1 else mask[...,begin:end,:]
            if mask.dtype==torch.bool:logits.masked_fill_(~mask,float('-inf'))
            else:logits+=mask.float()
        weights=logits.softmax(dim=-1)
        reconstruction=(weights@v).transpose(1,2)
        actual=captured['output'][:,begin:end].float()
        relative=float(torch.linalg.vector_norm(reconstruction-actual)/torch.linalg.vector_norm(actual))
        assert relative<.01,('Attention reconstruction differs from SDPA output',relative)
        visual=weights[0,:,:,1:513].mean(dim=(0,1)).reshape(2,16,16).cpu().numpy()
        return visual,dict(relative_output_reconstruction_rms=relative,causal=causal,
                           action_token_start=begin,visual_tokens=512,action_queries=56)

    def action(pair,state):
        h=ctx.hidden_from_arrays(pair[0],pair[1],state.inputs,state.proprio)
        result=ctx.ah.predict_action(h.reshape(1,56,4096).to(next(ctx.ah.parameters()).dtype))
        result=result.float()[0].cpu().numpy()
        weights,spec=attention()
        return result,weights,spec

    try:
        states=runtime.collect(a.tasks,21,[10,50])
        for state in states:
            folder=a.out/state.key;folder.mkdir()
            basepair=state.pixels(runtime.base)
            actualbase,baseattn,spec=action(basepair,state)
            assert np.array_equal(actualbase.astype(np.float64),state.a0),'Attention hooks changed original policy action'
            marker=[]
            for color in [[255,0,255],[0,255,0]]:marker.append(state.pixels(np.full_like(runtime.base,color)))
            fractions=[]
            for j in range(2):
                resized=[F.interpolate(torch.from_numpy(p[j].copy()).permute(2,0,1).float()[None],size=(224,224),mode='bilinear',align_corners=False) for p in marker]
                mask=(resized[0]-resized[1]).abs().amax(dim=1,keepdim=True)>10
                fractions.append(F.avg_pool2d(mask.float(),14,14)[0,0].numpy())
            fractions=np.stack(fractions)
            np.save(folder/'cloth_patch_fractions.npy',fractions)
            pairref=state.pixels(runtime.full)
            pairs={k:state.pixels(v) for k,v in textures.items()}
            pairs['main_only']=(pairref[0],basepair[1]);pairs['wrist_only']=(basepair[0],pairref[1])
            row=dict(state.meta,conditions={},original_action_identity=True)
            actions={}
            for name,pair in pairs.items():
                with torch.no_grad():out,attn,spec=action(pair,state)
                actions[name]=out
                np.save(folder/(name+'_attention.npy'),attn)
                for j,view in enumerate(['main','wrist']):Image.fromarray(pair[j]).save(folder/(name+'_'+view+'.png'))
                masses=attn.sum(axis=(1,2));cloth=(attn*fractions).sum(axis=(1,2));area=fractions.mean(axis=(1,2))
                density=cloth/(masses*area+1e-30)
                row['conditions'][name]=dict(action=out.tolist(),attention_view_mass=masses.tolist(),attention_cloth_mass=cloth.tolist(),
                    cloth_patch_area_fraction=area.tolist(),cloth_attention_density_relative_to_uniform=density.tolist(),
                    relative_output_reconstruction_rms=spec['relative_output_reconstruction_rms'])
            assert np.array_equal(actions['plain'],actualbase)
            shape=lambda x:np.asarray(x[:5,:6],dtype=float).reshape(-1)
            plain=shape(actions['plain']);full=shape(actions['refcloth']);delta=full-plain;energy=float(delta@delta)
            main=shape(actions['main_only']);wrist=shape(actions['wrist_only'])
            phi_main=.5*((main-plain)+(full-wrist));phi_wrist=.5*((wrist-plain)+(full-main))
            assert np.max(np.abs(phi_main+phi_wrist-delta))<1e-12
            row['action_interventions']=dict(full_pose_action_rms=float(np.sqrt(np.mean(delta**2))),
                main_only_pose_action_rms=float(np.sqrt(np.mean((main-plain)**2))),wrist_only_pose_action_rms=float(np.sqrt(np.mean((wrist-plain)**2))),
                main_symmetric_projection_share=float(phi_main@delta/energy) if energy>1e-10 else None,
                wrist_symmetric_projection_share=float(phi_wrist@delta/energy) if energy>1e-10 else None,
                interaction_pose_action_rms=float(np.sqrt(np.mean((full-main-wrist+plain)**2))))
            report['records'].append(row);write(report,a.out/'results.partial.json')
            print('MEASURED',state.key,json.dumps(row['action_interventions']),flush=True)
        report['completed']=True;report['peak_cuda_bytes']=torch.cuda.max_memory_allocated();write(report,a.out/'results.json')
    finally:runtime.close_states()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tasks',type=lambda s:[int(v) for v in s.split(',')],required=True)
    main(p.parse_args())
