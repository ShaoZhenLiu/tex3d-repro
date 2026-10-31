"""Fit renderer uncertainty in action space from the existing source cache.

Consumes fit/validation only. Source action head only; no 7B model, simulator,
target query, or original locked_test read. Native energy is always evaluated in
the original unweighted 15 translation coordinates, not the learned metric.
"""
import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from uncertainty_metric import (fit_uncertainty_model, solve_worst_action,
                                quadratic_metric_score, mahalanobis_squared)


def load_head(source, checkpoint, device):
    names = {'MLPResNetBlock', 'MLPResNet', 'L1RegressionActionHead'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(torch=torch, nn=nn, ACTION_DIM=7, NUM_ACTIONS_CHUNK=8)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    head = namespace['L1RegressionActionHead']()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head.load_state_dict({k.removeprefix('module.'): v for k, v in weights.items()}, strict=True)
    return head.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)


def load_translation_scale(statistics_path=None, statistics_key=None):
    if statistics_path is None:
        return [1.0, 1.0, 1.0], dict(units='normalized source-head action coordinates', statistics=None)
    stats = json.loads(statistics_path.read_text())
    if statistics_key is not None:
        action = stats[statistics_key]['action']
        key = statistics_key
    elif 'action' in stats:
        action, key = stats['action'], '<root>'
    else:
        candidates = [(key, value['action']) for key, value in stats.items()
                      if isinstance(value, dict) and isinstance(value.get('action'), dict)
                      and 'q01' in value['action'] and 'q99' in value['action']]
        if len(candidates) != 1:
            raise ValueError('Specify --statistics-key; action statistics keys: '+str([k for k, _ in candidates]))
        key, action = candidates[0]
    mask = action.get('mask', [True]*len(action['q01']))
    scale = [(float(action['q99'][i])-float(action['q01'][i]))/2 if mask[i] else 1.0 for i in range(3)]
    if not all(value > 0 for value in scale):
        raise ValueError('Translation unnormalization scales must be positive')
    return scale, dict(units='source translation control units after affine q01/q99 unnormalization',
        statistics=str(statistics_path), statistics_key=key,
        statistics_sha256=hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        affine_scale=scale, note='Offsets cancel in displacement. This assumes production affine q01/q99 unnormalization without additional clipping.')


def extract(root, split, head, device, translation_scale):
    metadata, predicted, truth, feature_scores = [], [], [], []
    for path in sorted((root/'pairs'/split).glob('*.pt')):
        rec = torch.load(path, map_location='cpu', weights_only=False)
        tids = sorted(k[:-len('__fR_float')] for k in rec if k.endswith('__fR_float'))
        r = torch.stack([rec[t+'__fR_float']-rec['fR0'] for t in tids]).reshape(-1, 56, 4096)
        native_features = torch.stack([rec[t+'__fN_Q'] for t in tids]).reshape(-1, 56, 4096)
        native0 = rec['fN0'].reshape(1, 56, 4096)

        def predict(x):
            with torch.no_grad():
                return head.predict_action(x.to(device=device, dtype=torch.bfloat16)).float().cpu()[:, :5, :3].flatten(1)

        a0 = predict(native0)
        ar = predict(native0+r)
        an = predict(native_features)
        repeat = predict(native_features)
        assert float((repeat-an).abs().max()) == 0, 'head repeat floor is nonzero'
        scale = torch.tensor(translation_scale, dtype=torch.double).repeat(5)
        predicted.append((ar-a0).double()*scale)
        truth.append((an-a0).double()*scale)
        feature_scores.append(r.float().square().mean(dim=(1, 2)))
        metadata.extend(dict(state=rec['record_id'], task=int(rec['source_task_id']),
                             step=int(rec['policy_step']), texture=t) for t in tids)
        print('extracted', split, rec['record_id'], len(tids), flush=True)
    return metadata, torch.cat(predicted).double(), torch.cat(truth).double(), torch.cat(feature_scores).double()


def ranks(xs):
    xs = list(xs)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rank = [0.0]*len(xs)
    lo = 0
    while lo < len(xs):
        hi = lo+1
        while hi < len(xs) and xs[order[hi]] == xs[order[lo]]:
            hi += 1
        for i in order[lo:hi]:
            rank[i] = (lo+hi-1)/2
        lo = hi
    return torch.tensor(rank, dtype=torch.double)


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    a, b = a-a.mean(), b-b.mean()
    if float(a.norm()*b.norm()) == 0:
        return None
    return float(a@b/(a.norm()*b.norm()))


def assess(metadata, u, n, feature_scores, model):
    assert torch.isfinite(u).all() and torch.isfinite(n).all()
    true_energy = n.square().mean(-1)
    scores = dict(direct_action=u.square().mean(-1),
                  generalized=quadratic_metric_score(u, model, reduction='none').double())
    if feature_scores is not None:
        scores['feature_fd'] = feature_scores
    coverage = {}
    for mode in ['absolute', 'relative']:
        entry = model[mode]
        value, _, info = solve_worst_action(u, entry, entry['radius'], relative=mode == 'relative')
        scores['robust_'+mode] = value/u.shape[-1]
        error_norm = mahalanobis_squared(n-u, entry)
        covered = error_norm <= info['radius_squared']*(1+1e-10)
        violation = scores['robust_'+mode]-true_energy
        tolerance = 1e-10+1e-7*torch.maximum(scores['robust_'+mode], true_energy)
        violation_mask = violation > tolerance
        coverage[mode] = dict(coverage=float(covered.double().mean()),
            positive_fraction=float((value/u.shape[-1] > 1e-14).double().mean()),
            exact_positive_fraction=float((value > 0).double().mean()),
            all_bound_violations=int(violation_mask.sum()),
            covered_bound_violations=int((violation_mask&covered).sum()),
            maximum_covered_violation=float(violation[covered].max()) if bool(covered.any()) else None)
        assert coverage[mode]['covered_bound_violations'] == 0
    states = sorted(set(r['state'] for r in metadata))
    report = dict(n=len(metadata), state_count=len(states), methods={}, uncertainty_coverage=coverage,
        zero_score_threshold=1e-14,
        selection_rule='Highest surrogate score; ties within 1e-14+1e-10*maxabs use lexicographically first texture ID; also report uniform tie expectation',
        evaluation_warning='States, not texture pairs, are the independent grouping unit. Undefined flat-state correlations are additionally counted as zero to avoid selective exclusion.')
    for name, score in scores.items():
        assert torch.isfinite(score).all()
        per_state = []
        for state in states:
            indices = [i for i, r in enumerate(metadata) if r['state'] == state]
            local_score = score[indices]
            maximum = float(local_score.max())
            tied = [i for i in indices if abs(float(score[i])-maximum) <= 1e-14+1e-10*abs(maximum)]
            chosen = min(tied, key=lambda i: metadata[i]['texture'])
            oracle = float(true_energy[indices].max())
            expected = statistics.mean(float(true_energy[i]) for i in tied)
            selectable = maximum > 1e-14 and float(local_score.max()-local_score.min()) > 1e-14
            per_state.append(dict(state=state,
                spearman=spearman(local_score.tolist(), true_energy[indices].tolist()),
                chosen_texture=metadata[chosen]['texture'], chosen_native_energy=float(true_energy[chosen]),
                max_score=maximum, score_range=float(local_score.max()-local_score.min()),
                all_zero_scores=bool(local_score.abs().max() <= 1e-14),
                selection_informative=selectable, max_tie_count=len(tied),
                tied_textures=[metadata[i]['texture'] for i in tied],
                tie_expected_native_energy=expected,
                tie_min_native_energy=min(float(true_energy[i]) for i in tied),
                tie_max_native_energy=max(float(true_energy[i]) for i in tied),
                oracle_native_energy=oracle,
                chosen_over_oracle=float(true_energy[chosen])/oracle if oracle > 1e-20 else None,
                tie_expected_over_oracle=expected/oracle if oracle > 1e-20 else None))
        rank_values = [r['spearman'] for r in per_state if r['spearman'] is not None]
        oracle_ratios = [r['chosen_over_oracle'] for r in per_state if r['chosen_over_oracle'] is not None]
        expected_oracle_ratios = [r['tie_expected_over_oracle'] for r in per_state if r['tie_expected_over_oracle'] is not None]
        report['methods'][name] = dict(
            pooled_spearman=spearman(score.tolist(), true_energy.tolist()),
            mean_within_state_spearman=statistics.mean(rank_values) if rank_values else None,
            defined_state_spearman_count=len(rank_values),
            mean_within_state_spearman_zero_for_undefined=statistics.mean(
                r['spearman'] if r['spearman'] is not None else 0 for r in per_state),
            mean_selected_native_energy=statistics.mean(r['chosen_native_energy'] for r in per_state),
            mean_tie_expected_native_energy=statistics.mean(r['tie_expected_native_energy'] for r in per_state),
            mean_chosen_over_oracle=statistics.mean(oracle_ratios) if oracle_ratios else None,
            mean_tie_expected_over_oracle=statistics.mean(expected_oracle_ratios) if expected_oracle_ratios else None,
            nonzero_oracle_states=len(oracle_ratios),
            all_zero_states=sum(r['all_zero_scores'] for r in per_state),
            informative_selection_states=sum(r['selection_informative'] for r in per_state),
            unique_max_states=sum(r['max_tie_count'] == 1 for r in per_state),
            zero_score_count=int((score <= 1e-14).sum()), per_state=per_state)
    direct = {r['state']: r for r in report['methods']['direct_action']['per_state']}
    for name, method in report['methods'].items():
        comparisons = []
        for r in method['per_state']:
            baseline = direct[r['state']]['tie_expected_native_energy']
            difference = r['tie_expected_native_energy']-baseline
            tolerance = 1e-12+1e-7*abs(baseline)
            outcome = 'win' if difference > tolerance else ('loss' if difference < -tolerance else 'tie')
            comparisons.append(dict(state=r['state'], outcome=outcome,
                selection_informative=r['selection_informative'],
                expected_native_energy=r['tie_expected_native_energy'], baseline_expected_native_energy=baseline))
        baseline_mean = report['methods']['direct_action']['mean_tie_expected_native_energy']
        method['vs_direct_action'] = dict(
            expected_energy_ratio=method['mean_tie_expected_native_energy']/baseline_mean if baseline_mean > 1e-20 else None,
            wins=sum(r['outcome'] == 'win' for r in comparisons),
            losses=sum(r['outcome'] == 'loss' for r in comparisons),
            ties=sum(r['outcome'] == 'tie' for r in comparisons),
            informative_wins=sum(r['outcome'] == 'win' and r['selection_informative'] for r in comparisons),
            states=comparisons)
    return report


def reassess_cached_actions(pairs_path, model_path, output_path):
    """CPU-only assessment of an already completed cache job, without the head.

    Older pair files did not retain feature_fd. In that case the FD baseline is
    explicitly omitted here, and remains available in the original report.
    """
    pairs = json.loads(Path(pairs_path).read_text())
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    report = dict(scope='CPU reassessment of frozen cached action pairs; no refit or new policy query',
                  pair_sha256=hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest())
    for split in ['fit', 'validation']:
        rows = pairs[split]
        metadata = [r['meta'] for r in rows]
        predicted = torch.tensor([r['predicted'] for r in rows], dtype=torch.double)
        native = torch.tensor([r['native'] for r in rows], dtype=torch.double)
        feature = torch.tensor([r['feature_fd'] for r in rows], dtype=torch.double) if all('feature_fd' in r for r in rows) else None
        report[split] = assess(metadata, predicted, native, feature, model)
        report[split]['feature_fd_available'] = feature is not None
    Path(output_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    head = load_head(args.head_source, args.checkpoint, args.device)
    translation_scale, action_units = load_translation_scale(args.statistics, args.statistics_key)
    fit_meta, fit_u, fit_n, fit_f = extract(args.root, 'fit', head, args.device, translation_scale)
    val_meta, val_u, val_n, val_f = extract(args.root, 'validation', head, args.device, translation_scale)
    shape_states = set()
    for task in sorted(set(r['task'] for r in fit_meta)):
        states = sorted(set((r['step'], r['state']) for r in fit_meta if r['task'] == task))
        shape_states.update(state for _, state in states[::2])
    shape_mask = torch.tensor([r['state'] in shape_states for r in fit_meta])
    calibration_mask = ~shape_mask
    assert bool(shape_mask.any()) and bool(calibration_mask.any())
    model = fit_uncertainty_model(fit_u, fit_n, shape_mask, calibration_mask,
                                 ridge=args.ridge, quantile=args.quantile, generalized_rank=5)
    model['translation_scale'] = torch.tensor(translation_scale, dtype=torch.double)
    model['action_units'] = action_units
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.out/'uncertainty_model.pt')
    report = dict(scope='source cached-feature exploration; no native update or closed-loop success claim',
                  locked_test_consumed=False, target_queries=0,
                  action_subset='first 5 execution actions, translation only (15 coordinates)',
                  action_units=action_units,
                  action_prediction='H(native_clean_feature + synthetic_feature_displacement) - H(native_clean_feature)',
                  fit_split='alternate time-sorted states within each source task: even shape, odd radius',
                  shape_states=sorted(shape_states), calibration_states=sorted(set(r['state'] for r in fit_meta)-shape_states),
                  shape_rows=int(shape_mask.sum()), calibration_rows=int(calibration_mask.sum()),
                  ridge=args.ridge, quantile=args.quantile,
                  covariance_eigenvalues={m: model[m]['values'].tolist() for m in ['absolute', 'relative']},
                  radii={m: model[m]['radius'] for m in ['absolute', 'relative']},
                  head_source_sha256=hashlib.sha256(args.head_source.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  mathematical_guarantee='conditional on actual error lying in fitted ellipsoid; random probe coverage does not certify optimized textures',
                  fit=assess(fit_meta, fit_u, fit_n, fit_f, model),
                  validation=assess(val_meta, val_u, val_n, val_f, model))
    (args.out/'metric_cache_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pairs = dict(fit=[dict(meta=m, predicted=u.tolist(), native=n.tolist(), feature_fd=float(f)) for m, u, n, f in zip(fit_meta, fit_u, fit_n, fit_f)],
                 validation=[dict(meta=m, predicted=u.tolist(), native=n.tolist(), feature_fd=float(f)) for m, u, n, f in zip(val_meta, val_u, val_n, val_f)])
    (args.out/'action_displacement_pairs.json').write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    print(json.dumps(dict(radii=report['radii'], validation=report['validation']), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--head-source', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ridge', type=float, default=0.01)
    ap.add_argument('--quantile', type=float, default=0.9)
    ap.add_argument('--statistics', type=Path, help='dataset_statistics.json with production q01/q99 action bounds')
    ap.add_argument('--statistics-key', help='dataset key in the statistics file; auto-detected only if unique')
    main(ap.parse_args())
