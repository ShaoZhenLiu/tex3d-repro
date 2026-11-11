"""Decompose the same joint feature displacement gradient by image branch."""
import argparse
import hashlib
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import GPUS, PRIOR, REPO, ROOT, SEEDS, SOURCE_STEPS, guard, local_node, sha

TASKS = (0, 2, 6, 8)
BATCHES = 8


def load_context_module():
    path = REPO / 'ops/uv_gradient_20261109/common.py'
    spec = importlib.util.spec_from_file_location('existing_source_context', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_path(directory, update):
    if update == 0:
        return directory / 'theta_init.pt'
    if update == 5000:
        return directory / 'Ep0_Cloth_Delta_OV.pt'
    return directory / f'Ep0_Cloth_Delta_OV_it{update-1:05d}.pt'


def bank(ctx):
    rng = random.Random(20261111)
    indices = {task: rng.sample(range(len(ctx.frames[task])), 2 * BATCHES) for task in TASKS}
    random.seed(20261112)
    return [dict(frames=[(t, i) for t in TASKS for i in indices[t][2*b:2*b+2]],
                 uv=ctx.R.M.sample_uv_xform(True, 'cpu').tolist()) for b in range(BATCHES)]


def metrics(gm, gw):
    main, wrist = gm.double().flatten(), gw.double().flatten()
    total = main + wrist
    nm, nw, nt = (float(x.norm()) for x in (main, wrist, total))
    return dict(main_norm=nm, wrist_norm=nw, joint_norm=nt,
        cosine=float(torch.dot(main, wrist)) / (nm * nw) if nm * nw > 1e-30 else None,
        main_share=float(torch.dot(main, total)) / nt**2 if nt > 1e-15 else None,
        wrist_share=float(torch.dot(wrist, total)) / nt**2 if nt > 1e-15 else None)


def split(ctx, task, index, uv, check):
    param = ctx.renderer.adv_noise
    fr, main, wrist = ctx.render_views(ctx.frames[task][index], uv)
    assert main.requires_grad and wrist.requires_grad
    hidden = ctx.hidden_from_views(fr, main, wrist)
    loss = 10. * F.mse_loss(hidden, ctx.refs[task][index].to('cuda').float())
    dm, dw = torch.autograd.grad(loss, (main, wrist), retain_graph=True)
    gm = torch.autograd.grad(main, param, dm, retain_graph=True)[0].detach()
    gw = torch.autograd.grad(wrist, param, dw, retain_graph=check)[0].detach()
    assert torch.isfinite(gm).all() and torch.isfinite(gw).all()
    residual = None
    if check:
        direct = torch.autograd.grad(loss, param)[0].detach()
        residual = float((gm + gw - direct).norm()) / max(float(gm.norm() + gw.norm()), 1e-15)
        assert residual <= .002, ('camera branch sum', task, index, residual)
    return gm, gw, residual


def measure(ctx, seed, update, checkpoint, frame_bank, receipt):
    out = ROOT / 'source' / f's{seed}_u{update:05d}'
    output = out / 'result.json'
    if output.exists():
        record = guard.read(output)
        assert record['completed'] and record['checkpoint_sha256'] == sha(checkpoint)
        assert record['bank_sha256'] == sha(ROOT / 'source/bank.json')
        return
    out.mkdir(parents=True, exist_ok=True)
    theta = torch.load(checkpoint, map_location='cpu', weights_only=False).contiguous()
    param = ctx.renderer.adv_noise
    assert theta.shape == param.shape and theta.dtype == param.dtype
    if update == 0:
        assert hashlib.sha256(theta.numpy().tobytes()).hexdigest() == receipt['theta_init_sha256']
    with torch.no_grad():
        param.copy_(theta.to(param.device))
    if update == 5000:
        frozen = np.array(Image.open(PRIOR / 'frozen' / f's{seed}_mr.png').convert('RGB'))
        assert np.array_equal(ctx.renderer.export_texture(), frozen), ('checkpoint export', seed)

    rows, frames, vectors = [], [], []
    for batch, spec in enumerate(frame_bank):
        uv = torch.tensor(spec['uv'], dtype=torch.float32, device='cuda')
        main, wrist = torch.zeros_like(param), torch.zeros_like(param)
        residual = None
        for position, (task, index) in enumerate(spec['frames']):
            gm, gw, error = split(ctx, task, index, uv, position == 0)
            main += gm / len(spec['frames'])
            wrist += gw / len(spec['frames'])
            frames.append(dict(batch=batch, task=task, index=index, **metrics(gm, gw)))
            if error is not None:
                residual = error
        rows.append(dict(batch=batch, decomposition_residual=residual, **metrics(main, wrist)))
        vectors.append([main.cpu().numpy(), wrist.cpu().numpy()])
        print('GRADIENT_BATCH', seed, update, batch + 1, '/', BATCHES, flush=True)
        if guard.pause_requested():
            raise SystemExit(75)
    assert torch.equal(param.detach().cpu(), theta), 'Probe changed the saved parameter'
    vector_file = out / 'batch_gradients.npz'
    temporary = out / 'batch_gradients.partial.npz'
    np.savez_compressed(temporary, gradients=np.array(vectors))
    temporary.replace(vector_file)
    guard.write(output, dict(completed=True, seed=seed, updates=update, batches=rows, frames=frames,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), gradient_sha256=sha(vector_file),
        bank_sha256=sha(ROOT / 'source/bank.json'), peak_cuda_bytes=torch.cuda.max_memory_allocated()))


def main(a):
    node = local_node()
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert node == a.node and gpu in GPUS[node]
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.27 if node == '09' else .44)
    guard.install_pause_handler()
    module = load_context_module()
    ctx, _ = module.standalone()
    ctx.model.eval()
    assert len(ctx.frames) == 4 and sum(len(x) for x in ctx.frames.values()) == 296
    assert module.sha(ctx.R.POOL) == ctx.R.POOL_SHA and module.sha(ctx.R.MR_CACHE) == ctx.R.CACHE_SHA
    guard.mark_ready()
    frame_bank = bank(ctx)
    path = ROOT / 'source/bank.json'
    guard.write(path, frame_bank)
    guard.write(ROOT / 'source/registered.json', dict(node=node, gpu=gpu, pytorch=torch.__version__,
        gpu_name=torch.cuda.get_device_name(), source_pool_sha256=module.sha(ctx.R.POOL),
        matched_cache_sha256=module.sha(ctx.R.MR_CACHE), steps=SOURCE_STEPS,
        seeds=SEEDS[node], batch_size=8, batches=BATCHES, model_checkpoint=str(ctx.R.M.CKPT)))
    for seed in SEEDS[node]:
        receipt = guard.read(PRIOR / 'arms' / f's{seed}_mr' / 'completed.json')
        assert receipt['completed'] and receipt['iterations'] == 5000
        assert receipt['gpu_name'] == torch.cuda.get_device_name()
        directory = Path(receipt['directory'])
        for update in SOURCE_STEPS:
            measure(ctx, seed, update, snapshot_path(directory, update), frame_bank, receipt)
    guard.write(ROOT / 'source/done.json', dict(completed=True, seeds=SEEDS[node],
                steps=SOURCE_STEPS, bank_sha256=sha(path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=SEEDS, required=True)
    main(parser.parse_args())
