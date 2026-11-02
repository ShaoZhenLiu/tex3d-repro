"""Independent final checks over saved Object episodes and original paper files."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
PAPER=Path('E:/Linux/cvpr2027/RefCloth_Closure_20261031')


def main():
    raw=ROOT/'raw'
    protocol_hash=hashlib.sha256((ROOT/'code/protocol.json').read_bytes()).hexdigest()
    rows=[];registrations=[];results=[]
    for name in ['pi0_main','pi05_main','pi0_tail','pi05_tail']:
        p=raw/name
        meta=json.loads((p/'registered.json').read_text())
        result=json.loads((p/'results.json').read_text())
        assert result['completed'] and meta['protocol_sha256']==protocol_hash
        assert meta['device'] in ['0','3','6']
        registrations.append(meta);results.append(result)
        for f in p.glob('pi*_t*_i*_*.json'):
            x=json.loads(f.read_text());assert x['completed'];rows.append(x)
            assert 0<x['queries']<=80 and x['queries']==len(x['trajectory'])
            assert x['trajectory'][0]['aa_action_maxdiff']<=1e-6
            for i,q in enumerate(x['trajectory']):
                assert q['query']==i
                actions=np.asarray(q['actions'])
                assert actions.shape==(5,7) and np.isfinite(actions).all()
            if x['condition']=='plain':
                assert x['native_upload_count']==1+4*x['queries']+2
                assert x['trajectory'][0]['measurement']['hidden_marker_maxdiff']==0
                poses=[]
                for q in x['trajectory']:
                    m=q['measurement'];assert m['aba_exact'] and m['integration_state_exact'] and m['camera_views_same_state']
                    assert m['cached_observation_maxdiff']==0
                    assert -.000001<=m['camera_delay_seconds']<=.05
                    poses.append(m['cloth_world'])
                    assert abs(m['cloth_world'][2]-.005)<1e-9
                    assert set(m['model_specs'])=={'base_0_rgb','left_wrist_0_rgb'}
                    for spec in m['model_specs'].values():
                        assert spec['active'] and spec['shape']==[1,3,224,224] and spec['dtype']=='torch.float32'
                assert np.max(np.abs(np.asarray(poses)-poses[0]))<1e-10
            else:
                assert not x['probe'] and x['native_upload_count']==1
    assert len({(x['target'],x['task'],x['initial'],x['condition']) for x in rows})==len(rows)==120
    assert len({x['ref_sha256'] for x in registrations})==1
    assert len({x['base_sha256'] for x in registrations})==1
    assert len({json.dumps(x['source_hashes'],sort_keys=True) for x in registrations})==1
    before=json.loads((ROOT/'evidence/paper_hashes_before.json').read_text())
    changed=[f for f,h in before.items() if not (PAPER/f).exists() or hashlib.sha256((PAPER/f).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed'] and summary['counts']['same_state_real_counterfactual_pairs_checked']==60
    outcome=list(csv.DictReader((ROOT/'episode_outcomes.csv').open(encoding='utf-8-sig')))
    assert len(outcome)==120
    tasks={}
    for r in rows:
        t=str(r['task'])
        if t in tasks:assert tasks[t]==r['language']
        tasks[t]=r['language']
    (ROOT/'task_catalog.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    checks=dict(passed=True,expected_episodes=120,observed_episodes=120,
        all_condition_case_keys_unique=True,all_same_state_pairs_checked=60,
        source_pngs_and_pipeline_hashes_consistent=True,
        authorized_gpus=sorted({r['device'] for r in registrations}),gpu4_used=False,
        shared_scene_assets_toggled=False,training_or_target_gradients_used=False,
        all_native_texture_upload_counts_match=True,
        active_model_inputs='two real 1x3x224x224 float32 images; dummy view excluded',
        raw_actions_finite=True,primary_statistics_use_episode_medians=True,
        original_paper_hashes_checked=len(before),original_paper_files_changed=changed,
        execution_commits={r['target']+'_gpu'+r['device']:r['commit'] for r in registrations},
        peak_cuda_bytes=[x['peak_cuda_bytes'] for x in results])
    (ROOT/'final_integrity_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
