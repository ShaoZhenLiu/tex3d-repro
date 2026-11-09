"""Historical endpoints, kept separate from the new prospective paired experiment."""
import json
import time
from pathlib import Path

import torch

from common import REPO,ROOT,sha,write
from probe import measure,preserve


def run(ctx):
    directory=REPO/'experiments/pi0_cloth/oft_cloth'
    states=[('init',directory/'train_20261027_075225_e03-uvoff/theta_init.pt'),
            ('on',directory/'train_20261021_224922_tvfix-mr/Ep0_Cloth_Delta_OV.pt'),
            ('off',directory/'train_20261027_075225_e03-uvoff/Ep0_Cloth_Delta_OV.pt')]
    for name,path in states:
        output=ROOT/'legacy'/(name+'.json')
        if output.exists():continue
        with torch.no_grad():ctx.renderer.adv_noise.copy_(torch.load(path,map_location='cuda',weights_only=False))
        with preserve(ctx):result=measure(ctx,0 if name=='init' else 5000)
        result.update(historical=True,state=name,theta_path=str(path),theta_sha256=sha(path),
            provenance='Archived original seed-7 On and frame-stream-replayed Off; separate from new three paired trainings.',
            time=time.time())
        write(output,result)
        print('LEGACY_DIAGNOSTIC',name,json.dumps({p:v['pooled_feature_pair_cosine_mean']
            for p,v in result['policies'].items()}),flush=True)
