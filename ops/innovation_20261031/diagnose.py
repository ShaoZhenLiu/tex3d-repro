"""Source-only offline exploration of deployment-feature fidelity.

Reads fit and validation captures only. The original locked_test remains sealed.
No model, simulator, target policy, or GPU is used. Each exploratory model is
evaluated on a crossed split and on the already-inspected validation set.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from gate_audit import candidate_passes


def load_split(root, split):
    directory = root / 'pairs' / split
    manifest = json.loads((directory / 'split_manifest.json').read_text())
    texture_meta = {v['texture_id']: v for v in manifest['textures']}
    rows, rendered, native = [], [], []
    for path in sorted(directory.glob('*.pt')):
        record = torch.load(path, map_location='cpu', weights_only=False)
        for key in sorted(record):
            if not key.endswith('__fR_float'):
                continue
            tid = key[:-len('__fR_float')]
            meta = texture_meta[tid]
            rendered.append((record[key] - record['fR0']).float())
            native.append((record[tid + '__fN_Q'] - record['fN0']).float())
            rows.append(dict(state=record['record_id'], task=int(record['source_task_id']),
                             step=int(record['policy_step']), texture=tid,
                             texture_seed=int(meta['seed']), rms=float(meta['rms']),
                             family=meta['family']))
    return rows, torch.stack(rendered).reshape(-1, 56, 4096), \
        torch.stack(native).reshape(-1, 56, 4096)


def crossed_masks(rows):
    # Defined before examining feature scores: early states versus late states,
    # odd versus even seeds within every amplitude/frequency stratum.
    masks = {name: [] for name in ['fit', 'new_texture', 'new_state', 'both_new']}
    for row in rows:
        seen_state = row['step'] <= 45
        seen_texture = row['texture_seed'] % 2 == 1
        group = ('fit' if seen_texture else 'new_texture') if seen_state else (
            'new_state' if seen_texture else 'both_new')
        for name in masks:
            masks[name].append(group == name)
    return {name: torch.tensor(value) for name, value in masks.items()}


def energy(value):
    return value.double().square().sum()


def vector_metrics(prediction, truth, baseline):
    difference = prediction - truth
    numerator = energy(difference)
    denominator = energy(baseline - truth)
    dot = (prediction * truth).flatten(1).sum(1)
    norm = prediction.flatten(1).norm(dim=1) * truth.flatten(1).norm(dim=1)
    cosine = dot / norm.clamp_min(1e-20)
    return dict(relative_error_vs_rendered=float(numerator / denominator),
                relative_error_vs_native=float(numerator / energy(truth)),
                mean_cosine=float(cosine.mean()),
                prediction_energy_vs_native=float(energy(prediction) / energy(truth)))


def pca_basis(residual, rank):
    flat = residual.flatten(1).double()
    gram = flat @ flat.T
    values, vectors = torch.linalg.eigh(gram)
    count = int((values > values[-1] * 1e-10).sum())
    rank = min(rank, count)
    order = torch.arange(len(values) - 1, len(values) - rank - 1, -1)
    return (flat.T @ (vectors[:, order] / values[order].sqrt())).float()


def projection_metrics(rendered, native, basis):
    residual = (rendered - native).flatten(1)
    truth = native.flatten(1)
    r_filtered = residual - (residual @ basis) @ basis.T
    n_filtered = truth - (truth @ basis) @ basis.T
    return dict(q_err=float(energy(r_filtered) / energy(residual)),
                q_signal=float(energy(n_filtered) / energy(truth)))


def correlation(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 0 else None


def fit_transport(rendered, native, mode):
    if mode == 'scalar':
        numerator = (rendered * native).sum()
        denominator = rendered.square().sum()
    elif mode == 'token':
        numerator = (rendered * native).sum(dim=(0, 2)).reshape(1, 56, 1)
        denominator = rendered.square().sum(dim=(0, 2)).reshape(1, 56, 1)
    elif mode == 'channel':
        numerator = (rendered * native).sum(dim=(0, 1)).reshape(1, 1, 4096)
        denominator = rendered.square().sum(dim=(0, 1)).reshape(1, 1, 4096)
    else:
        raise ValueError(mode)
    return (numerator / denominator.clamp_min(1e-20)).clamp(0, 2)


def fit_energy_metric(rendered, native):
    # Positive diagonal action-token metric, 56 parameters. No feature mixing.
    x = rendered.double().square().mean(2)
    y = native.double().square().mean((1, 2))
    scale = x.mean(0).clamp_min(1e-10)
    xs = x / scale
    gram = xs.T @ xs / len(x)
    rhs = xs.T @ y / len(x)
    ridge = 0.01
    hessian = gram + ridge * torch.eye(56, dtype=torch.float64)
    # Anchored at a global scalar baseline. Projected convex optimization.
    global_weight = float((x.mean(1) * y).sum() / x.mean(1).square().sum())
    anchor = global_weight * scale / 56
    rhs = rhs + ridge * anchor
    weights = anchor.clone()
    step = 1 / float(torch.linalg.eigvalsh(hessian)[-1])
    for _ in range(1000):
        weights = (weights - step * (hessian @ weights - rhs)).clamp_min(0)
    return weights / scale


def energy_metrics(rendered, native, weights):
    x = rendered.double().square().mean(2)
    y = native.double().square().mean((1, 2))
    pred = x @ weights.double()
    base = x.mean(1)
    return dict(mse=float((pred - y).square().mean()),
                baseline_mse=float((base - y).square().mean()),
                pearson=correlation(pred, y), baseline_pearson=correlation(base, y),
                relative_mse=float((pred-y).square().sum()/(base-y).square().sum()))


def run(root, output):
    torch.set_num_threads(2)
    rows, rendered, native = load_split(root, 'fit')
    val_rows, val_rendered, val_native = load_split(root, 'validation')
    masks = crossed_masks(rows)
    report = dict(scope='source-only exploratory; not a closed-loop attack result',
                  locked_test_consumed=False, feature_layout=[56, 4096],
                  predeclared_split='step<=45 and odd seed fit; all other crosses held out',
                  counts={name: int(mask.sum()) for name, mask in masks.items()},
                  validation_count=len(val_rows), crossed={}, validation={}, gate_audit={})
    train_r, train_n = rendered[masks['fit']], native[masks['fit']]
    models = {mode: fit_transport(train_r, train_n, mode)
              for mode in ['scalar', 'token', 'channel']}
    metric = fit_energy_metric(train_r, train_n)
    bases = {rank: pca_basis(train_r - train_n, rank) for rank in [8, 32, 64]}
    groups = [(name, rendered[mask], native[mask]) for name, mask in masks.items()]
    groups.append(('validation', val_rendered, val_native))
    for name, r, n in groups:
        result = dict(baseline=vector_metrics(r, n, r), transport={}, projection={})
        for mode, weights in models.items():
            result['transport'][mode] = vector_metrics(r * weights, n, r)
        for rank, basis in bases.items():
            result['projection'][str(rank)] = projection_metrics(r, n, basis)
        result['positive_token_energy_metric'] = energy_metrics(r, n, metric)
        if name == 'validation':
            report['validation'] = result
        else:
            report['crossed'][name] = result
    # Inspect the source-task heterogeneity without retuning on validation.
    report['validation_by_task'] = {}
    for task in sorted(set(row['task'] for row in val_rows)):
        mask = torch.tensor([row['task'] == task for row in val_rows])
        r, n = val_rendered[mask], val_native[mask]
        report['validation_by_task'][str(task)] = dict(
            baseline=vector_metrics(r,n,r),
            channel=vector_metrics(r*models['channel'],n,r),
            energy_metric=energy_metrics(r,n,metric))
    old_report = json.loads((root/'diagnostics/gate_b_report.json').read_text())
    candidates = old_report['candidates']
    changed = []
    for c in candidates:
        passed = candidate_passes(c, old_report['gate'])
        if passed != c['gate_pass']:
            changed.append(dict(k=c['k'],beta=c['beta'],new_pass=passed))
    report['gate_audit'] = dict(n_candidates=len(candidates),
        passing_after_fix=sum(candidate_passes(c,old_report['gate']) for c in candidates),
        changed=changed, minimum_error=min(c['q_err'] for c in candidates))
    report['energy_metric_weights'] = metric.tolist()
    report['transport_weight_summary'] = {mode:dict(min=float(w.min()),max=float(w.max()),mean=float(w.mean()))
                                          for mode,w in models.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save(dict(transport=models,energy_metric=metric,
                    training_rows=[rows[i] for i in range(len(rows)) if masks['fit'][i]]),
               output/'calibration_fit.pt')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run(args.root,args.out)
