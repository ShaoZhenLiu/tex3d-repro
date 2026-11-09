"""An exact-stream UV pair, using the unchanged production update plus read-only probes."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from common import OPS, REPO, ROOT, read, runner, sha, write
from probe import callback_for


def main(args):
    sys.path.insert(0,str(REPO/'ops'))
    import refcloth_resource as guard
    from refcloth_resume import select_checkpoint
    guard.install_pause_handler()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.61,0)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','2','3']
    prepared=read(ROOT/'prepared/smoke.json');assert prepared['passed']
    stream=ROOT/'streams'/('seed_%d.txt'%args.seed)
    record=next(row for row in prepared['streams'] if row['seed']==args.seed)
    assert sha(stream)==record['text_sha256']
    assert sha(stream.with_name(stream.stem+'_full.pt'))==record['uv_sha256']
    R=runner();assert R.M.TR.patch_status()=='patched'
    assert sha(R.POOL)==R.POOL_SHA and sha(R.MR_CACHE)==R.CACHE_SHA
    tag='uvgradient_s%d_%s'%(args.seed,args.uv)
    output=ROOT/'arms'/('s%d_%s'%(args.seed,args.uv));output.mkdir(parents=True,exist_ok=True)
    assert not (output/'completed.json').exists()
    R.OUT_ROOT=str(ROOT/'training');(ROOT/'training').mkdir(exist_ok=True)
    # Numbered checkpoints are immutable, so every restart has explicit provenance.
    latest=select_checkpoint(ROOT/'training',tag)
    while latest is not None:
        try:
            check=torch.load(latest,map_location='cpu',weights_only=False)
            assert torch.is_tensor(check) and torch.isfinite(check).all() and check.abs().max()<=5.00001
            del check
            break
        except Exception as error:
            # Preserve the incomplete bytes, then recover an earlier fully logged checkpoint.
            quarantine=latest.with_suffix('.pt.incomplete_%d'%time.time_ns())
            latest.rename(quarantine)
            write(output/'recovery'/('%d.json'%time.time_ns()),dict(incomplete=str(quarantine),error=str(error)))
            latest=select_checkpoint(ROOT/'training',tag)
    resume_updates=int(latest.stem.rsplit('_it',1)[1])+1 if latest else 0
    later=[p for p in (output/'probes').glob('u*.json') if int(p.stem[1:])>resume_updates]
    if later:
        preserved=output/('probes_before_resume_%d'%time.time_ns());preserved.mkdir()
        for path in later:path.rename(preserved/path.name)
    options=argparse.Namespace(tag=tag,ref='mr',ref_switch=0,stream='replay',stream_file=str(stream),
        uv=args.uv,filter_task=None,iters=5000,batch=8,attack_lr=.05,epsilon=64/255,init_std=.01,
        lambda_ref=.1,lambda_tv=.1,alpha_action=0.,alpha_feature=10.,seed=args.seed,
        checkpoint_every=500,ckpt_step_suffix=True,resume_dir=str(latest.parent) if latest else '',
        resume_checkpoint=str(latest) if latest else '',texture_only_grad=True)
    registration=dict(seed=args.seed,uv=args.uv,options=vars(options),stream=record,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,universal_newlines=True).strip(),
        host=os.uname().nodename,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        gpu_name=torch.cuda.get_device_name(),pytorch=torch.__version__,cuda=torch.version.cuda,
        runner_sha256=sha(OPS/'trainer_probe.py'),protocol_sha256=sha(OPS/'protocol.json'),
        resume_checkpoint=str(latest) if latest else None)
    write(output/'registrations'/('%d.json'%time.time_ns()),registration)
    write(output/'registered.json',registration)
    R.resource_ready=guard.mark_ready;R.pause_requested=guard.pause_requested
    R.measurement_callback=callback_for(args.seed,args.uv,R)
    start=time.monotonic()
    R.train(options)
    if guard.pause_requested():raise SystemExit(75)
    candidates=list((ROOT/'training').glob('train_*_'+tag+'/Ep0_Cloth_Delta_OV_it04999.pt'))
    assert candidates,candidates
    trained=sorted(candidates,key=lambda p:p.parent.name)[-1].parent
    assert all((output/'probes'/('u%05d.json'%update)).exists() for update in range(0,5001,500))
    assert (output/'probes/heldout_initial.json').exists()
    result=dict(registration,completed=True,updates=5000,directory=str(trained),
        texture_sha256=sha(trained/'Universal_Cloth_OV.png'),
        theta_sha256=sha(trained/'Ep0_Cloth_Delta_OV_it04999.pt'),
        theta_init_sha256=sha(trained/'theta_init.pt'),elapsed_seconds=time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),time=time.time())
    write(output/'completed.json',result)
    print('UV_ARM_COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--uv',choices=['on','off'],required=True);main(parser.parse_args())
