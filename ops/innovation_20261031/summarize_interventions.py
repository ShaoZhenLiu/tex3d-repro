"""Summarize the complete frozen intervention grid without case selection."""
import argparse
import json
import statistics
from pathlib import Path


def median(rows, field):
    return statistics.median(r[field] for r in rows)


def summarize(folder):
    rows = json.loads((folder/'multiview_control.json').read_text(encoding='utf-8'))['records']
    expected = {(t, i, n, x) for t in [0, 4, 8] for i in [10, 11]
                for n in [0, 1, 2] for x in ['MR', 'random101', 'random102']}
    actual = [(r['task'], r['initial'], r['noise_id'], r['texture']) for r in rows]
    assert set(actual) == expected and len(actual) == len(expected)
    assert max(r['aa_maxdiff'] for r in rows) < 1e-6
    result = dict(record_count=len(rows), independent_scene_count=6, noise_count=3,
                  warning='Noise repeats are not independent scenes. No closed-loop task-success claim.',
                  by_texture={}, by_state=[])
    for texture in ['MR', 'random101', 'random102']:
        r = [x for x in rows if x['texture'] == texture]
        result['by_texture'][texture] = dict(
            median_interaction_rms=median(r, 'interaction_rms'),
            median_amplification=median(r, 'amplification'),
            amplification_over_1p1=sum(x['amplification'] > 1.1 for x in r))
    for task, initial in [(t, i) for t in [0, 4, 8] for i in [10, 11]]:
        r = [x for x in rows if x['task'] == task and x['initial'] == initial]
        entry = dict(task=task, initial=initial, textures={})
        for texture in ['MR', 'random101', 'random102']:
            tr = sorted([x for x in r if x['texture'] == texture], key=lambda x: x['noise_id'])
            entry['textures'][texture] = dict(median_interaction_rms=median(tr, 'interaction_rms'),
                median_amplification=median(tr, 'amplification'),
                amplification_per_noise=[x['amplification'] for x in tr],
                amplified_noises=sum(x['amplification'] > 1.1 for x in tr))
        result['by_state'].append(entry)
    result['stable_MR_amplification_states'] = sum(x['textures']['MR']['amplified_noises'] >= 2 for x in result['by_state'])
    mr = result['by_texture']['MR']['median_interaction_rms']
    result['MR_interaction_vs_random'] = {x: mr/result['by_texture'][x]['median_interaction_rms'] for x in ['random101', 'random102']}
    result['broad_mechanism_gate'] = result['stable_MR_amplification_states'] >= 5 and min(result['MR_interaction_vs_random'].values()) >= 1.25
    (folder/'multiview_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    path = folder/'instruction_counterfactual.json'
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding='utf-8'))['records']
    actual = [(r['task'], r['initial'], r['noise_id'], r['texture']) for r in rows]
    assert set(actual) == expected and len(actual) == len(expected)
    assert max(r['aa_maxdiff'] for r in rows) < 1e-6
    result = dict(record_count=len(rows), independent_scene_count=6, noise_count=3,
                  warning='Action conditioning is not semantic competence. The alternative Plain goal needs closed-loop validation.',
                  by_texture={}, by_state=[])
    for texture in ['MR', 'random101', 'random102']:
        r = [x for x in rows if x['texture'] == texture]
        result['by_texture'][texture] = dict(
            median_language_texture_interaction_rms=median(r, 'language_texture_interaction_rms'),
            median_original_texture_effect_rms=median(r, 'original_texture_effect_rms'),
            median_alternative_texture_effect_rms=median(r, 'alternative_texture_effect_rms'))
    for task, initial in [(t, i) for t in [0, 4, 8] for i in [10, 11]]:
        r = [x for x in rows if x['task'] == task and x['initial'] == initial]
        entry = dict(task=task, initial=initial, textures={},
                     plain_instruction_response_rms=median(r, 'plain_instruction_response_rms'))
        for texture in ['MR', 'random101', 'random102']:
            tr = [x for x in r if x['texture'] == texture]
            entry['textures'][texture] = dict(
                median_language_texture_interaction_rms=median(tr, 'language_texture_interaction_rms'))
        result['by_state'].append(entry)
    field = 'median_language_texture_interaction_rms'
    mr = result['by_texture']['MR'][field]
    result['MR_interaction_vs_random'] = {x: mr/result['by_texture'][x][field] for x in ['random101', 'random102']}
    result['states_MR_exceeds_both_controls'] = sum(
        x['textures']['MR'][field] > max(x['textures'][c][field] for c in ['random101', 'random102'])
        for x in result['by_state'])
    result['exploratory_screen'] = min(result['MR_interaction_vs_random'].values()) >= 1.25 and result['states_MR_exceeds_both_controls'] >= 5
    (folder/'instruction_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', type=Path)
    summarize(ap.parse_args().folder)
