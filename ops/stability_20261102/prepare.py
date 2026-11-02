"""Verify the existing renderer and predeclare exact paired training streams."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] != '4'
    torch.set_num_threads(2)
    entry = a.repo/'openvla-oft/experiments/robot/libero'
    sys.path.insert(0, str(entry))
    import attack_oft_cloth_ablate as R
    a.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(a.protocol.read_text())
    assert R.M.TR.patch_status() == 'patched', 'Do not mutate shared scene assets'
    assert sha(R.POOL) == R.POOL_SHA and sha(R.MR_CACHE) == R.CACHE_SHA
    pool = torch.load(R.POOL, map_location='cpu', weights_only=False)
    act, gids = R.active_frames(pool, None)
    assert len(act) == 296 and sorted(pool) == [0, 2, 6, 8]
    renderer = R.M.ClothRendererOV(device='cuda', init_std=0., seed=7).to('cuda')
    base = np.array(Image.open(R.M.CLOTH_T0).convert('RGB'))
    export_error = int(np.abs(renderer.export_texture().astype(int)-base.astype(int)).max())
    assert export_error <= 1
    gates = []
    for task, frames in pool.items():
        fr = R.frame_to_dev(frames[0], 'cuda')
        row = dict(task=task, views={})
        for name, mvp_key, seg_key, bg_key in [
            ('main', 'mvp', 'seg', 'bg'), ('wrist', 'wrist_mvp', 'wrist_seg', 'wrist_bg')]:
            if fr.get(mvp_key) is None:
                continue
            with torch.no_grad():
                rendered, _ = renderer.render(fr[mvp_key], (256, 256), model_rot=fr['rot'])
                bg, seg = fr[bg_key], fr[seg_key]
                comp = (rendered.permute(0, 3, 1, 2)*seg+bg*(1-seg)).clamp(0, 1)
                mask = seg.expand_as(bg)>0.5
                visible = float(seg.mean())
                # bg is the cloth-HIDDEN render, not the native Plain image.
                # The main RGB is retained in the processor's SigLIP channels;
                # the full native wrist RGB is stored directly in the pool.
                if name == 'main':
                    pixels=fr['text_inputs']['pixel_values']
                    assert pixels.shape[1] == 6 and tuple(pixels.shape[-2:]) == (224,224)
                    processor_config=json.loads((Path(R.M.CKPT)/'preprocessor_config.json').read_text())
                    siglip_index=next(i for i,(mean,std) in enumerate(zip(processor_config['means'],processor_config['stds']))
                                      if mean==[.5,.5,.5] and std==[.5,.5,.5])
                    reference=(pixels[:,3*siglip_index:3*(siglip_index+1)].float()*.5+.5).clamp(0,1)
                    candidate=F.interpolate(comp,size=(224,224),mode='bilinear',align_corners=False)
                    metric_mask=F.interpolate(seg,size=(224,224),mode='nearest').expand_as(reference)>.5
                    mse=float(((candidate-reference)[metric_mask]**2).mean())
                    reference_path='stored native-Plain processor RGB at 224; inverse SigLIP normalization'
                else:
                    mse=float(((comp-fr['wrist'])[mask]**2).mean()) if mask.any() else 0.
                    reference_path='stored native-Plain wrist RGB at 256'
                psnr = float(-10*np.log10(max(mse, 1e-20)))
                outside = float((comp-bg)[~mask].abs().max()) if (~mask).any() else 0.
            row['views'][name] = dict(visible_fraction=visible, cloth_psnr_db=psnr,
                                     foreground_maxdiff=outside,reference_path=reference_path)
            if name == 'main':
                assert visible > .05 and psnr >= 25 and outside <= 1e-6, row
        gates.append(row)
    streams = []
    if a.verify_only:
        existing=json.loads((a.out/'prepared.json').read_text())
        for row in existing['streams']:
            path=a.out/('seed_%d.txt'%row['seed'])
            assert sha(path)==row['text_sha256']
            assert sha(path.with_name(path.stem+'_full.pt'))==row['uv_sha256']
        streams=existing['streams']
    for seed in ([] if a.verify_only else protocol['new_training_seeds']):
        random.seed(seed)
        path = a.out/('seed_%d.txt'%seed)
        uv_full = []
        with path.open('w') as f:
            f.write('# it slot frame_idx task uv00 uv01 uv02 uv10 uv11 uv12\n')
            for it in range(5000):
                frames = [random.randrange(len(act)) for _ in range(8)]
                uv = R.M.sample_uv_xform(True, 'cpu').numpy()
                uv_full.append(uv)
                values = ' '.join('%.6f'%v for v in uv.reshape(-1))
                for slot, fi in enumerate(frames):
                    f.write('%d %d %d %d %s\n'%(it, slot, fi, act[fi][0], values))
        pt = path.with_name(path.stem+'_full.pt')
        torch.save(dict(uv=uv_full, seed=seed, n_active=296, filter_task=None), pt)
        streams.append(dict(seed=seed, text_sha256=sha(path), uv_sha256=sha(pt), rows=40000))
    result = dict(passed=True, gate_scope='original main-view smoke; wrist photometry is an explicitly authorized frozen-recipe exception',
                  all_view_photometric_gate_passed=all(v['cloth_psnr_db']>=25 for row in gates for v in row['views'].values()),
                  frozen_wrist_gap_authorized=protocol['frozen_renderer_exception']['authorized_by_user'],
                  source='existing frozen renderer and pool',
                  pool_sha256=R.POOL_SHA, reference_cache_sha256=R.CACHE_SHA,
                  zero_export_maxdiff=export_error, visibility_photometry=gates,
                  streams=streams, pytorch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (a.out/('preflight_09.json' if a.verify_only else 'prepared.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    main(p.parse_args())
