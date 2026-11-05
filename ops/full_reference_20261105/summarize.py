# coding: utf-8
"""Validate every planned observation and report all three frozen candidates."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from common import HERE, INITIALS, ROOT, SEEDS, SUITES, episode_name, jobs, sha


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    out = ROOT/'analysis'
    out.mkdir(exist_ok=True)
    tables = []
    all_episodes = []
    texture_hashes = {}
    worker_records = {}
    reuse_total = 0
    for target in ['pi05', 'pi0']:
        folder = ROOT/'targets'/target
        protocol = json.loads((HERE/(target+'.json')).read_text())['evaluation']
        manifest = json.loads((folder/'texture_manifest.json').read_text())
        reused = json.loads((folder/'evaluation/reused_manifest.json').read_text())
        reuse_total += len(reused)
        for condition, info in manifest.items():
            assert sha(folder/info['path']) == info['sha256']
            assert texture_hashes.setdefault(condition, info['sha256']) == info['sha256']
        files = list((folder/'evaluation/episodes').glob('*.json'))
        assert len(files) == 2800
        records = {}
        for file in files:
            episode = json.loads(file.read_text())
            suite, task, initial, condition = [episode[k] for k in ['suite', 'task', 'initial', 'condition']]
            assert episode['completed'] and episode.get('target', target) == target
            assert suite in SUITES and task in range(10) and initial in INITIALS and condition in manifest
            assert episode['texture_sha256'] == manifest[condition]['sha256']
            assert episode['queries'] == len(episode['trajectory']) and episode['native_uploads'] > 0
            assert [r['query'] for r in episode['trajectory']] == list(range(episode['queries']))
            for row in episode['trajectory']:
                assert np.asarray(row['actions']).shape == (5, 7) and np.isfinite(row['actions']).all()
            key = suite, task, initial, condition
            assert key not in records
            records[key] = episode
            worker = folder/'evaluation/workers'/episode['worker']
            registration = json.loads((worker/'registered.json').read_text())
            calibration = json.loads((worker/'calibration.json').read_text())
            assert registration['checkpoint'] == protocol['checkpoint'] and calibration['passed']
            if target == 'pi05':
                assert episode['host'].endswith('-41') and calibration['trajectory_exact']
            else:
                assert episode['host'] == 'zgc-33-9'
                assert calibration.get('success_repeatable') and calibration.get('mode') == 'success'
            worker_records[target+'/'+episode['worker']] = dict(registration=registration, calibration=calibration)
            all_episodes.append(dict(target=target, suite=suite, task=task, initial=initial,
                condition=condition, success=int(episode['robot_success']), queries=episode['queries'],
                host=episode['host'], gpu=episode['gpu'], worker=episode['worker'], reused=file.name in reused))
            if file.name in reused:
                assert sha(file) == reused[file.name]['sha256']
        for job in jobs(target):
            done = json.loads((folder/'evaluation/completed'/(job['id']+'.json')).read_text())
            assert done['completed'] and done['job'] == job
            expected = {(initial, c) for initial in INITIALS for c in job['conditions']}
            assert {(r['initial'], r['condition']) for r in done['outcomes']} == expected
            for row in done['outcomes']:
                assert row['robot_success'] == records[(job['suite'], job['task'], row['initial'], row['condition'])]['robot_success']
        for suite in SUITES:
            plain = sum(records[suite, task, initial, 'plain']['robot_success'] for task in range(10) for initial in INITIALS)
            for seed in SEEDS:
                n_success = m_success = native_stronger = matched_stronger = 0
                for task in range(10):
                    for initial in INITIALS:
                        n = records[suite, task, initial, 's%d_native' % seed]
                        m = records[suite, task, initial, 's%d_mr' % seed]
                        p = records[suite, task, initial, 'plain']
                        assert n['host'] == m['host'] == p['host']
                        for other in [m, p]:
                            if n['trajectory'] and other['trajectory']:
                                assert n['trajectory'][0]['proprio_sha256'] == other['trajectory'][0]['proprio_sha256']
                            else:
                                assert not n['trajectory'] and not other['trajectory'] and n['robot_success'] and other['robot_success']
                            assert all(x['noise_sha256'] == y['noise_sha256'] for x, y in zip(n['trajectory'], other['trajectory']))
                        ns, ms = int(n['robot_success']), int(m['robot_success'])
                        n_success += ns; m_success += ms
                        native_stronger += (1-ns)*ms; matched_stronger += ns*(1-ms)
                tables.append(dict(target=target, suite=suite, seed=seed, n=100, Plain=int(plain),
                    Native=n_success, Matched=m_success, Native_G=int(plain)-n_success,
                    Matched_G=int(plain)-m_success, native_advantage=m_success-n_success,
                    native_stronger=native_stronger, matched_stronger=matched_stronger,
                    selection_condition=(target == 'pi05' and suite in ['spatial', 'object'])))
    assert len(all_episodes) == 5600 and reuse_total == 1000
    seed_means = []
    for seed in SEEDS:
        for target in ['pi05', 'pi0', 'both']:
            block = [row for row in tables if row['seed'] == seed and (target == 'both' or row['target'] == target)]
            fresh_domains = [row for row in block if not row['selection_condition']]
            mean = lambda rows, key: sum(row[key] for row in rows)/len(rows)
            seed_means.append(dict(seed=seed, target=target, Native_G=mean(block, 'Native_G'),
                Matched_G=mean(block, 'Matched_G'), native_advantage=mean(block, 'native_advantage'),
                extended_conditions_Native_G=mean(fresh_domains, 'Native_G'),
                extended_conditions_Matched_G=mean(fresh_domains, 'Matched_G')))
    save_csv(out/'paired_suite_results.csv', tables)
    save_csv(out/'episode_outcomes.csv', all_episodes)
    save_csv(out/'seed_means.csv', seed_means)
    result = dict(completed=True, episodes=5600, new_episodes=4600, reused_episodes=1000,
        seeds=SEEDS, targets=['pi05', 'pi0'], suites=SUITES, texture_hashes=texture_hashes,
        same_texture_across_all_conditions=True, paired_states_and_noise_validated=True,
        evaluation_hosts={'pi05': '4090', 'pi0': '09'}, seed_means=seed_means,
        selection='Post-hoc Native-favorable candidates from pi05 Spatial/Object; not random-seed evidence',
        worker_records=worker_records)
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    text = ['# Native 三候选完整基准复验', '',
        '选择 3、12、10，是因为它们在先前 π0.5-Spatial/Object 上具有最大的 Native 优势，且 Native 强度均位于十八组前五。此次为事后筛选候选的完整基准扩展，不作为随机种子稳定性证明。', '',
        '六张 Native/Matched 冻结纹理均沿用原导出。π0.5 全部在 41 上评估，π0 全部在 09 上评估；每套件十个任务，各初始状态 10–19。每组共用任务状态与查询噪声。完整数据为 5,600 回合，其中新评测 4,600、精确复用 1,000。', '',
        'G 为相对同策略 Plain 的成功率净下降，越大攻击越强。Native 优势为 Native G − Matched G。', '',
        '| seed | 目标策略 | Native 平均 G | Matched 平均 G | Native 优势 |',
        '| ---: | --- | ---: | ---: | ---: |']
    for row in seed_means:
        text.append('| {} | {} | {:.2f} | {:.2f} | {:+.2f} |'.format(row['seed'], row['target'], row['Native_G'], row['Matched_G'], row['native_advantage']))
    text += ['', '## 逐套件结果', '', '| seed | 策略 | 套件 | Plain | Native 成功 | Matched 成功 | Native G | Matched G | Native 优势 |',
             '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in tables:
        text.append('| {seed} | {target} | {suite} | {Plain} | {Native} | {Matched} | {Native_G} | {Matched_G} | {native_advantage:+d} |'.format(**row))
    text += ['', '## 未用于此次筛选的扩展条件', '',
        '单列 π0 的全部四套件，以及 π0.5 的 Goal/Long，避免把已用于选候选的 Spatial/Object 与新条件混为一体。', '',
        '| seed | Native G | Matched G |', '| ---: | ---: | ---: |']
    for row in seed_means:
        if row['target'] == 'both':
            text.append('| {} | {:.2f} | {:.2f} |'.format(row['seed'], row['extended_conditions_Native_G'], row['extended_conditions_Matched_G']))
    (out/'Native三候选完整基准复验.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(dict(completed=True, episodes=5600, new=4600, reused=1000)))


if __name__ == '__main__':
    main()
