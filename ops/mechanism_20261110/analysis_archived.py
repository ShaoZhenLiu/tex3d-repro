"""Recompute frozen-texture direction and residuals from all archived vectors."""
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from load_feature_cache import load

HERE = Path(__file__).resolve().parent
SOURCE = Path('E:/Linux/zcode_test/refcloth_theory_results/t1_fix_20261027_233045')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cos(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator < 1e-20 else float(np.dot(a, b) / denominator)


def main(source_path=SOURCE, output_path=HERE/'feature_direction', protocol_path=HERE/'repo/ops/mechanism_20261110/protocol.json'):
    SOURCE=Path(source_path)
    source = json.loads((SOURCE/'t1fix_results.json').read_text(encoding='utf-8'))
    data = load(SOURCE/'t1fix_features.pt')
    expected = [s['state'] for s in source['states']]
    assert list(data) == expected and len(expected) == 32
    metadata = {t['name']: t for t in source['textures']}
    original = {(r['state'], r['texture'], r['baseline']): r for r in source['rows']}
    rows = []
    deviations = []
    for state in expected:
        z = data[state]
        for texture in metadata:
            attack = z[texture]
            fr, fn = attack['f_r'].astype(np.float64), attack['f_n'].astype(np.float64)
            fr0 = z['f_r0'].astype(np.float64)
            for route, clean in [('inject', 'f_n0_inj'), ('obs', 'f_n0_obs')]:
                fn0 = z[clean].astype(np.float64)
                dm, dn, ds = fr-fr0, fr-fn0, fn-fn0
                residual, error, bias = dm-ds, dn-ds, fr0-fn0
                norm = float(np.linalg.norm(ds))
                em, en = float(np.linalg.norm(residual)), float(np.linalg.norm(error))
                old = original[(state, texture, route)]
                deviations.extend([abs(em-old['E_M']), abs(en-old['E_N'])])
                assert abs(em-old['E_M']) < max(.02,1e-4*em) and abs(en-old['E_N']) < max(.02,1e-4*en), (state,texture,route,em,en,old['E_M'],old['E_N'])
                row = dict(state=state, task=int(state.split('_')[0][1:]), texture=texture,
                    baseline=route, cosine_matched=cos(dm, ds), cosine_native=cos(dn, ds),
                    matched_error=em, native_error=en, true_displacement_norm=norm,
                    baseline_offset_norm=float(np.linalg.norm(bias)),
                    baseline_relative_offset=float(np.linalg.norm(bias)/np.linalg.norm(fn0)),
                    relative_matched_error=em/norm if norm > 1e-12 else None,
                    relative_native_error=en/norm if norm > 1e-12 else None,
                    matched_norm_ratio=float(np.linalg.norm(dm)/norm) if norm > 1e-12 else None,
                    native_norm_ratio=float(np.linalg.norm(dn)/norm) if norm > 1e-12 else None)
                row['cosine_difference'] = (row['cosine_matched']-row['cosine_native']
                    if row['cosine_matched'] is not None and row['cosine_native'] is not None else None)
                rows.append(row)
    out = Path(output_path);out.mkdir(parents=True,exist_ok=True)
    with (out/'all_state_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = {}
    for route in ['inject', 'obs']:
        subset = [r for r in rows if r['texture']=='mr8' and r['baseline']==route]
        fields = ['cosine_matched','cosine_native','cosine_difference','matched_error','native_error',
                  'relative_matched_error','relative_native_error','baseline_relative_offset',
                  'matched_norm_ratio','native_norm_ratio']
        summary[route] = {f:dict(mean=float(np.mean([r[f] for r in subset])),
            median=float(np.median([r[f] for r in subset])),
            min=float(min(r[f] for r in subset)),max=float(max(r[f] for r in subset))) for f in fields}
        summary[route]['states'] = len(subset)
        summary[route]['matched_cosine_wins'] = sum(r['cosine_difference']>0 for r in subset)
        summary[route]['matched_error_wins'] = sum(r['matched_error']<r['native_error'] for r in subset)
        summary[route]['per_task'] = {str(task):{f:float(np.mean([r[f] for r in subset if r['task']==task]))
            for f in fields} for task in [0,2,6,8]}
    receipt = dict(completed=True, all_rows=len(rows), frozen_texture=metadata['mr8']['sha'],
        source_files={p.name:file_sha(p) for p in [SOURCE/'manifest.json', SOURCE/'t1fix_results.json',SOURCE/'t1fix_features.pt']},
        protocol_sha256=file_sha(Path(protocol_path)),
        max_norm_reconstruction_error=max(deviations), results=summary,
        scope='Reanalysis of archived forward features; no additional model calls or texture selection')
    (out/'summary.json').write_text(json.dumps(receipt, ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--out',type=Path,default=HERE/'feature_direction')
    ap.add_argument('--protocol',type=Path,default=HERE/'repo/ops/mechanism_20261110/protocol.json')
    a=ap.parse_args();main(a.source,a.out,a.protocol)
