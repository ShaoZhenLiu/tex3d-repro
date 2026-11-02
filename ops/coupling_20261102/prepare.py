"""Cache exact same-path Plain RGBs and independent wrist UV streams."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08)
    sys.path.insert(0,str(a.repo/'openvla-oft/experiments/robot/libero'))
    import attack_oft_cloth_ablate as R
    assert sha(Path(R.POOL))==R.POOL_SHA
    pool=torch.load(R.POOL,map_location='cpu',weights_only=False)
    active,gids=R.active_frames(pool,None);assert gids==list(range(296))
    a.out.mkdir(parents=True,exist_ok=True);streams=a.out/'streams';streams.mkdir(exist_ok=True)
    renderer=R.M.ClothRendererOV(device='cuda',init_std=0.,seed=7).to('cuda')
    cache=[];largest=0.;checks=[]
    for index,(task,frame) in enumerate(active):
        fr={k:v.to('cuda') for k,v in frame.items() if k in ['mvp','seg','bg','rot','wrist_mvp','wrist_seg','wrist_bg'] and torch.is_tensor(v)}
        pair={}
        with torch.no_grad():
            for view,prefix in [('main',''),('wrist','wrist_')]:
                assert prefix+'mvp' in fr
                rgb,_=renderer.render(fr[prefix+'mvp'],(256,256),model_rot=fr['rot'])
                rgb=(rgb.permute(0,3,1,2)*fr[prefix+'seg']+fr[prefix+'bg']*(1-fr[prefix+'seg'])).clamp(0,1)
                pair[view]=rgb.detach().cpu().contiguous()
                if index in [0,37,74,111,148,185,222,259,295]:
                    for seed in [10,11,12]:
                        random.seed(991000+seed);uv=R.M.sample_uv_xform(True,'cuda')
                        changed,_=renderer.render(fr[prefix+'mvp'],(256,256),model_rot=fr['rot'],uv_xform=uv)
                        changed=(changed.permute(0,3,1,2)*fr[prefix+'seg']+fr[prefix+'bg']*(1-fr[prefix+'seg'])).clamp(0,1)
                        err=float((changed-rgb).abs().max());largest=max(largest,err)
                        assert err<1e-6,(index,view,seed,err)
                        checks.append(dict(frame=index,view=view,seed=seed,maxdiff=err))
        cache.append(pair)
    cache_path=a.out/'plain_image_cache.pt'
    torch.save(dict(frames=cache,pool_sha256=R.POOL_SHA,renderer_sha256=sha(Path(R.M.__file__))),cache_path)
    records=[]
    for seed in [10,11,12]:
        random.seed(991000+seed)
        uv=[R.M.sample_uv_xform(True,'cpu').numpy() for _ in range(5000)]
        path=streams/('wrist_%d.pt'%seed)
        torch.save(dict(uv=uv,seed=seed,aux_seed=991000+seed,n_active=296),path)
        records.append(dict(seed=seed,aux_seed=991000+seed,path=str(path),sha256=sha(path)))
    result=dict(passed=True,frames=296,pool_sha256=R.POOL_SHA,
        same_path_plain_cache_sha256=sha(cache_path),same_path_plain_cache_bytes=cache_path.stat().st_size,
        maximum_uniform_cloth_uv_pixel_difference=largest,uniform_checks=checks,streams=records,
        renderer_sha256=sha(Path(R.M.__file__)),pytorch=torch.__version__,device=torch.cuda.get_device_name())
    (a.out/'prepared.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='uniform_checks'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
