"""One frozen-stream camera-control arm; exports only the full final iterate."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refcloth_resource as guard
from refcloth_resume import select_checkpoint, consolidate

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main(a):
    guard.install_pause_handler()
    assert json.loads((a.repo/'ops/refcloth_mmap_validation.json').read_text())['passed']
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    primary=a.repo/'experiments/refcloth_stability/20261102'
    prepared=json.loads((a.root/'prepared.json').read_text())
    old=json.loads((primary/'streams/prepared.json').read_text())
    assert prepared['passed'] and old['passed'] and old['frozen_wrist_gap_authorized']
    assert json.loads((a.repo/'ops/stability_20261102/gradient_validation_holdout.json').read_text())['passed']
    record=next(r for r in old['streams'] if r['seed']==a.seed)
    stream=primary/'streams'/('seed_%d.txt'%a.seed)
    assert sha(stream)==record['text_sha256']
    assert sha(stream.with_name(stream.stem+'_full.pt'))==record['uv_sha256']
    cache=a.root/'plain_image_cache.pt'
    if a.mode!='uv_ind':assert sha(cache)==prepared['same_path_plain_cache_sha256']
    uv=a.root/'streams'/('wrist_%d.pt'%a.seed)
    aux=next(r for r in prepared['streams'] if r['seed']==a.seed)
    assert sha(uv)==aux['sha256']
    import attack_oft_cloth_coupling as R
    R.OUT_ROOT=str(a.root/'training');Path(R.OUT_ROOT).mkdir(exist_ok=True)
    assert R.M.TR.patch_status()=='patched'
    tag='coupling_s%d_%s'%(a.seed,a.mode)
    checkpoint=select_checkpoint(R.OUT_ROOT,tag)
    options=argparse.Namespace(tag=tag,ref='mr',ref_switch=0,
        stream='replay',stream_file=str(stream),uv='on',filter_task=None,iters=5000,batch=8,
        attack_lr=.05,epsilon=64/255,init_std=.01,lambda_ref=.1,lambda_tv=.1,
        alpha_action=0.,alpha_feature=10.,seed=a.seed,checkpoint_every=625,ckpt_step_suffix=True,
        resume_dir=str(checkpoint.parent) if checkpoint else '',
        resume_checkpoint=str(checkpoint) if checkpoint else '',
        texture_only_grad=True,view_mode='both' if a.mode=='uv_ind' else a.mode,
        wrist_uv_file=str(uv) if a.mode=='uv_ind' else '',plain_image_cache=str(cache))
    arm=a.root/'arms'/('s%d_%s'%(a.seed,a.mode));arm.mkdir(parents=True,exist_ok=True)
    assert not (arm/'completed.json').exists()
    registration=dict(seed=a.seed,mode=a.mode,training_options=vars(options),stream=record,
        wrist_stream=aux if a.mode=='uv_ind' else None,
        plain_image_cache_sha256=prepared['same_path_plain_cache_sha256'],
        host=os.uname().nodename,gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu_name=torch.cuda.get_device_name(),
        pytorch=torch.__version__,cuda=torch.version.cuda,memory_fraction=a.memory_fraction,
        runner_sha256=sha(Path(R.__file__)),renderer_sha256=sha(Path(R.M.__file__)),
        protocol_sha256=sha(a.repo/'ops/coupling_20261102/protocol.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,universal_newlines=True).strip())
    (arm/'registrations').mkdir(exist_ok=True)
    if (arm/'registered.json').exists():
        (arm/'registrations'/('prior_%d.json'%time.time_ns())).write_bytes((arm/'registered.json').read_bytes())
    (arm/'registrations'/('%d.json'%time.time_ns())).write_text(json.dumps(registration,indent=2))
    (arm/'registered.json').write_text(json.dumps(registration,indent=2))
    R.resource_ready=guard.mark_ready;R.pause_requested=guard.pause_requested
    start=time.monotonic()
    if not checkpoint or checkpoint.name!='Ep0_Cloth_Delta_OV_it04999.pt':R.train(options)
    if guard.pause_requested():raise SystemExit(75)
    candidates=[p for p in sorted(Path(R.OUT_ROOT).glob('train_*_'+options.tag))
        if (p/'gradient_log.txt').exists() and '[ckpt-final] it04999' in (p/'gradient_log.txt').read_text()]
    assert len(candidates)==1;trained=candidates[0]
    validation=json.loads((trained/'camera_control_validation.json').read_text());assert validation['passed']
    recovery=consolidate(trained,stream)
    theta=torch.load(trained/'theta_init.pt',map_location='cpu',weights_only=False).numpy()
    result=dict(registration,completed=True,iterations=5000,frame_visits=40000,
        directory=str(trained),texture=str(trained/'Universal_Cloth_OV.png'),
        texture_sha256=sha(trained/'Universal_Cloth_OV.png'),
        theta_init_sha256=hashlib.sha256(theta.tobytes()).hexdigest(),
        sampling_log_sha256=sha(trained/'sampling_log.txt'),
        camera_control_validation=validation,
        elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        resume_provenance=recovery,immutable_cache_mmap=True)
    (arm/'completed.json').write_text(json.dumps(result,indent=2));print('COUPLING_ARM_COMPLETED '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[10,11,12],required=True)
    p.add_argument('--mode',choices=['uv_ind','main','wrist'],required=True)
    p.add_argument('--memory-fraction',type=float,required=True);main(p.parse_args())
