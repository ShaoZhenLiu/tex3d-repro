"""Frozen paths and source-forward assembly for the UV mechanism controls."""
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

REPO = Path('/data/shaozhen.liu/code/tex3d')
ROOT = REPO/'experiments/refcloth_uv_gradient/20261109'
OPS = REPO/'ops/uv_gradient_20261109'
PYTHON = '/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python'
ENTRY = REPO/'openvla-oft/experiments/robot/libero'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def runner():
    sys.path.insert(0,str(ENTRY))
    spec=importlib.util.spec_from_file_location('uv_instrumented_runner',str(OPS/'trainer_probe.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def env_for(gpu):
    value=os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        MAX_JOBS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
        WANDB_MODE='disabled',USE_TF='0',TF_CPP_MIN_LOG_LEVEL='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=':'.join([str(REPO/'ops'),str(OPS),str(REPO),str(REPO/'openvla-oft')]))
    return value


class Context:
    def __init__(self,state,R):
        self.R=R
        for key in ['model','cfg','model_input_size','action_head','proprio_projector',
                    'noisy_action_projector','sm','ss','dm','ds','renderer']:
            setattr(self,key,state[key])
        self.frames={}
        self.refs={}
        for fi,(tid,fr) in enumerate(state['act']):
            self.frames.setdefault(tid,[]).append(fr)
            self.refs.setdefault(tid,[]).append(state['mr_hidden'][fi])

    def render_views(self,frame,uv):
        R=self.R;fr=R.frame_to_dev(frame,'cuda')
        rgba,_=self.renderer.render(fr['mvp'],(256,256),model_rot=fr['rot'],uv_xform=uv)
        comp=(rgba.permute(0,3,1,2)*fr['seg']+fr['bg']*(1-fr['seg'])).clamp(0,1)
        wrist=fr.get('wrist')
        if fr.get('wrist_mvp') is not None and fr.get('wrist_bg') is not None:
            rgba,_=self.renderer.render(fr['wrist_mvp'],(256,256),model_rot=fr['rot'],uv_xform=uv)
            wrist=(rgba.permute(0,3,1,2)*fr['wrist_seg']+fr['wrist_bg']*(1-fr['wrist_seg'])).clamp(0,1)
        return fr,comp,wrist

    def hidden_from_views(self,fr,comp,wrist):
        import torch
        R=self.R
        pv=R.base._compose_oft_pixel_values(adv_rgb=comp,model_input_size=self.model_input_size,
            siglip_mean=self.sm,siglip_std=self.ss,dino_mean=self.dm,dino_std=self.ds,
            wrist_tensor=wrist).to(torch.bfloat16)
        with R.base.autocast(dtype=torch.bfloat16):
            _,hidden=self.model.predict_action(input_ids=fr['text_inputs']['input_ids'],
                attention_mask=fr['text_inputs']['attention_mask'],pixel_values=pv,
                unnorm_key=self.cfg.unnorm_key,proprio=fr['proprio'],
                proprio_projector=self.proprio_projector,action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                use_film=getattr(self.cfg,'use_film',False))
        return hidden.float()

    def forward(self,frame,uv):
        return self.hidden_from_views(*self.render_views(frame,uv))

    def gradient(self,task,indices,uv):
        import torch
        import torch.nn.functional as F
        param=self.renderer.adv_noise
        gradient=torch.zeros_like(param);score=0.
        for index in indices:
            hidden=self.forward(self.frames[task][index],uv)
            loss=10.*F.mse_loss(hidden,self.refs[task][index].to('cuda').float())
            gradient+=torch.autograd.grad(loss/len(indices),param)[0].detach()
            score+=float(loss.detach())/len(indices)
        assert torch.isfinite(gradient).all() and gradient.norm()>0,(task,score)
        return gradient.flatten().float(),score

    def regularizer(self):
        import torch
        import torch.nn.functional as F
        tex=self.renderer.texture_tensor()
        delta=tex-self.renderer.orig_texture
        penalty=.1*F.mse_loss(tex,self.renderer.orig_texture)+.1*self.R.M.spatial_tv_nhwc(delta)
        return torch.autograd.grad(penalty,self.renderer.adv_noise)[0].detach().flatten().float()

    def attach_heldout(self):
        import torch
        data=torch.load(str(ROOT/'prepared/heldout.pt'),map_location='cpu',weights_only=False,mmap=True)
        self.frames.update(data['frames']);self.refs.update(data['reference_hidden'])


def standalone(seed=7):
    import numpy as np
    import torch
    R=runner();torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    assert R.M.TR.patch_status()=='patched','No shared XML mutations in this study'
    cfg=R.base.GenerateConfig(pretrained_checkpoint=R.M.CKPT,model_family='openvla')
    model=R.base.get_model(cfg);processor=R.base.get_processor(cfg)
    R.base.check_unnorm_key(cfg,model)
    ah,pp,nap=R.M._build_action_modules(cfg,model)
    sm,ss,dm,ds=R.M._norm_consts('cuda')
    pool=torch.load(R.POOL,map_location='cpu',weights_only=False,mmap=True)
    act,gids=R.active_frames(pool,None)
    full=torch.load(R.MR_CACHE,map_location='cpu',weights_only=False,mmap=True)['hidden']
    renderer=R.M.ClothRendererOV(device='cuda',epsilon=64/255,init_std=.01,seed=seed).to('cuda')
    state=dict(model=model,cfg=cfg,model_input_size=R.base.get_image_resize_size(cfg),
        action_head=ah,proprio_projector=pp,noisy_action_projector=nap,
        sm=sm,ss=ss,dm=dm,ds=ds,renderer=renderer,act=act,
        mr_hidden=[full[g] for g in gids])
    return Context(state,R),processor
