"""Run one unmodified production reference arm with a predeclared exact stream."""
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
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main(a):
    guard.install_pause_handler()
    assert json.loads((a.repo/'ops/refcloth_mmap_validation.json').read_text())['passed']
    assert os.environ['CUDA_VISIBLE_DEVICES'] != '4'
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    prepared=json.loads((a.root/'streams/prepared.json').read_text())
    assert prepared['passed']
    validation=json.loads((a.repo/'ops/stability_20261102/gradient_validation_holdout.json').read_text())
    assert validation['passed'] and validation['seed']==11
    assert prepared.get('all_view_photometric_gate_passed',False) or (
        a.allow_frozen_wrist_gap and prepared.get('frozen_wrist_gap_authorized',False))
    record=next(r for r in prepared['streams'] if r['seed']==a.seed)
    stream=a.root/'streams'/('seed_%d.txt'%a.seed)
    assert sha(stream)==record['text_sha256']
    assert sha(stream.with_name(stream.stem+'_full.pt'))==record['uv_sha256']
    sys.path.insert(0,str(a.repo/'openvla-oft/experiments/robot/libero'))
    import importlib.util
    source=a.repo/'ops/stability_20261102/attack_oft_cloth_ablate.py'
    spec=importlib.util.spec_from_file_location('refcloth_stability_runner',str(source))
    R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    assert R.M.TR.patch_status()=='patched'
    R.OUT_ROOT=str(a.root/'training')
    Path(R.OUT_ROOT).mkdir(parents=True,exist_ok=True)
    tag='stability_s%d_%s'%(a.seed,a.ref)
    checkpoint=select_checkpoint(R.OUT_ROOT, tag)
    options=argparse.Namespace(tag=tag,ref=a.ref,
        ref_switch=0,stream='replay',stream_file=str(stream),uv='on',filter_task=None,
        iters=5000,batch=8,attack_lr=.05,epsilon=64/255,init_std=.01,
        lambda_ref=.1,lambda_tv=.1,alpha_action=0.,alpha_feature=10.,
        seed=a.seed,checkpoint_every=625,ckpt_step_suffix=True,
        resume_dir=str(checkpoint.parent) if checkpoint else '',
        resume_checkpoint=str(checkpoint) if checkpoint else '',texture_only_grad=True)
    out=a.root/'arms'/('s%d_%s'%(a.seed,a.ref))
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completed.json').exists()
    registration=dict(seed=a.seed,reference=a.ref,training_options=vars(options),
        stream=record,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,universal_newlines=True).strip(),
        host=os.uname().nodename,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        gpu_name=torch.cuda.get_device_name(),pytorch=torch.__version__,cuda=torch.version.cuda,
        runner_sha256=sha(Path(R.__file__)),protocol_sha256=sha(a.repo/'ops/stability_20261102/protocol.json'),
        frozen_renderer_gap_explicitly_authorized=a.allow_frozen_wrist_gap)
    (out/'registrations').mkdir(exist_ok=True)
    if (out/'registered.json').exists():
        (out/'registrations'/('prior_%d.json'%time.time_ns())).write_bytes((out/'registered.json').read_bytes())
    (out/'registrations'/('%d.json'%time.time_ns())).write_text(json.dumps(registration,indent=2))
    (out/'registered.json').write_text(json.dumps(registration,indent=2))
    start=time.monotonic()
    R.resource_ready=guard.mark_ready;R.pause_requested=guard.pause_requested
    if not checkpoint or checkpoint.name!='Ep0_Cloth_Delta_OV_it04999.pt':R.train(options)
    if guard.pause_requested():raise SystemExit(75)
    candidates=[p for p in sorted(Path(R.OUT_ROOT).glob('train_*_'+options.tag))
                if (p/'gradient_log.txt').exists() and '[ckpt-final] it04999' in (p/'gradient_log.txt').read_text()]
    assert len(candidates)==1,candidates
    trained=candidates[0]
    assert '[ckpt-final] it04999' in (trained/'gradient_log.txt').read_text()
    recovery=consolidate(trained,stream)
    theta=torch.load(trained/'theta_init.pt',map_location='cpu',weights_only=False)
    result=dict(registration,completed=True,iterations=5000,frame_visits=40000,
        directory=str(trained),texture=str(trained/'Universal_Cloth_OV.png'),
        texture_sha256=sha(trained/'Universal_Cloth_OV.png'),theta_init_sha256=hashlib.sha256(theta.numpy().tobytes()).hexdigest(),
        sampling_log_sha256=sha(trained/'sampling_log.txt'),elapsed_seconds=time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),resume_provenance=recovery,
        immutable_cache_mmap=True)
    (out/'completed.json').write_text(json.dumps(result,indent=2))
    print('ARM_COMPLETED '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[10,11,12],required=True)
    p.add_argument('--ref',choices=['native','mr'],required=True)
    p.add_argument('--memory-fraction',type=float,default=.78)
    p.add_argument('--allow-frozen-wrist-gap',action='store_true')
    main(p.parse_args())
