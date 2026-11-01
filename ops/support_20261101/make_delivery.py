# -*- coding: utf-8 -*-
"""Integrity checks, concise decisions, paired case CSV and a small git payload."""
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from support_core import binary_texture,opacity_texture,array_sha


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(obj,p):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main(root):
    source=root/'results/screen';target=root/'results/pilot_parallel/combined.json'
    summary=load(source/'summary.json');pilot=load(root/'results/pilot_summary.json')
    combined=load(target);p=combined['protocol'];rows=combined['records']
    expected={(s,t,i,c) for s in p['suites'] for t in p['tasks'] for i in p['initials'] for c in p['conditions']}
    by={(r['suite'],r['task'],r['initial'],r['condition']):r for r in rows}
    assert set(by)==expected and len(by)==len(rows)==72
    assert all(r['native_upload_ok'] and r['aa_action_maxdiff']==0 for r in rows)
    for suite,task,initial in {(r['suite'],r['task'],r['initial']) for r in rows}:
        hashes=[by[(suite,task,initial,c)]['noise_sha256'] for c in p['conditions']]
        overlap=min(map(len,hashes));assert all(x[:overlap]==hashes[0][:overlap] for x in hashes)
    audit=load(source/'audit.json')
    assert sha(root/'code/protocol.json')==audit['protocol_sha256']
    for reg in (root/'results/pilot').glob('*/registered.json'):
        assert sha(root/'code/pilot_protocol.json')==load(reg)['protocol_sha256']
    full=np.array(Image.open(source/'textures/full.png').convert('RGB'))
    base=np.array(Image.open(source/'textures/plain.png').convert('RGB'))
    for d in load(source/'texture_manifest.json'):
        tex=source/'textures'/(d['name']+'.png')
        arr=np.array(Image.open(tex).convert('RGB'))
        assert sha(tex)==d['file_sha256'] and array_sha(arr)==d['pixel_sha256']
        wanted=(binary_texture(full,base,d['keep']) if 'keep' in d else opacity_texture(full,base,d['opacity']))
        assert np.array_equal(arr,wanted)
    qrows=sum(len(r.get('rows',[])) for q in (source/'queries').glob('q*.json') for r in load(q)['responses'])
    checks=dict(source_protocol_hash_verified=True,pilot_protocol_hash_verified=True,
        all_29_png_hashes_and_exact_frozen_colors_verified=True,unique_target_episodes=72,
        unique_paired_target_cases=12,conditions_per_case=6,all_target_native_uploads_passed=True,
        max_target_repeat_action_difference=0,common_consumed_noise_hash_prefixes_verified=True,
        fit_states=8,validation_states=16,source_candidate_state_rows=qrows,
        candidate_state_rows_include_cached_repeats=True,
        geometry_initial_states=12,geometry_candidate_records=len(load(root/'results/geometry/geometry.json')['records']))
    dump(checks,root/'integrity_checks.json')
    changed=[]
    with (root/'results/paired_target_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['suite','task','initial']+['robot_failure_'+c for c in p['conditions']])
        for suite in p['suites']:
            for task in p['tasks']:
                for initial in p['initials']:
                    outcomes=[int(not by[(suite,task,initial,c)]['robot_success']) for c in p['conditions']]
                    writer.writerow([suite,task,initial]+outcomes)
                    if outcomes[p['conditions'].index('greedy_12')]!=outcomes[p['conditions'].index('full')]:
                        changed.append(dict(suite=suite,task=task,initial=initial,
                            plain_robot_failure=outcomes[0],full_robot_failure=outcomes[1],greedy75_robot_failure=outcomes[2]))
    decisions=dict(engineering_feasible=True,source_strong_compression_gate_passed=False,
        source_greedy75_relative_fidelity=summary['summaries']['validation']['greedy_12']['pooled_relative_fidelity'],
        source_greedy75_action_energy_retention=summary['summaries']['validation']['greedy_12']['pooled_energy_retention'],
        greedy75_equals_one_shot_ranking=next(x for x in load(source/'texture_manifest.json') if x['name']=='greedy_12')['pixel_sha256']==
                                       next(x for x in load(source/'texture_manifest.json') if x['name']=='ranking_12')['pixel_sha256'],
        target_full_robot_failures=4,target_greedy75_robot_failures=2,target_visible75_robot_failures=4,
        paired_cases=12,changed_greedy_cases=changed,
        interpretation='Source fidelity did not reliably preserve target closed-loop outcomes. No support for upgrading this implementation to the paper main novelty.',
        recommendation='Keep the diagnostic and boundary findings. Preserve the original method and main tables.',
        untested='50%/25% target closed-loop performance; 8x8 masks; full-suite noninferiority; real hardware printing',
        not_claimed='All possible surface compression fails; source fidelity is a necessary condition for attack; masks/greedy or triangle inequalities are new mathematics')
    dump(decisions,root/'decisions.json')
    run=dict(baseline_commit='832c426c49fef4d9667af63424677732796a5aa1',
        source_execution_commit=audit['commit'],initial_target_commit='125a50a66d516ae705e3cacaac5b8a6aab2ffcbe',
        resumed_target_commit=load(root/'results/pilot_parallel/audit.json')['commit'],
        source_audit=audit,source_completed=load(source/'completed.json'),
        target_parallel_audit=load(root/'results/pilot_parallel/audit.json'),target_utc_end=combined['utc_end'],
        gpus=dict(source=[6,7],target=[1,6,7],geometry_only=[0],forbidden=[4],gpu4_used=False),
        source_state_count=24,target_episodes=72,source_candidate_state_rows=qrows,
        tasks_unchanged=True,texture_colors_retrained=False,policy_weights_trained=False,
        original_paper_edited=False,main_tables_edited=False,
        server_raw_root='/data/shaozhen.liu/code/tex3d/experiments/refcloth_support/20261101',
        code_branch='support-20261101',shared_source_patch_initial='patched',shared_target_patch_initial='patched',
        scheduling='Initial two-suite workers interrupted only to shard the same case set over three GPUs; 24 completed rows preserved. Unfinished calls not counted.',
        cpu_tests='5 meaningful mask/action tests passed on local CPU, 4090 and 09')
    dump(run,root/'run_manifest.json')
    text='''本轮结论：工程上可执行，但当前动作保真选区不宜升级为主创新点。

先打开《区域精简探索报告.html》。图表在 figures/；每个回合的配对结果在 results/paired_target_cases.csv。

1. 不重新优化颜色，不改变任务。4×4 区域恢复底布；单一掩码作用于整张三维桌布和全部相机。
2. 源侧 8 个选区状态、16 个验证状态。保留 75% 面积时源动作能量保留 86.3%；50% 和 25% 分别为 57.4%、23.0%，未通过较强精简门槛。
3. 额外探索性目标闭环：π0.5，Spatial/Object 各 6 个配对初态，6 种纹理，共 72 回合。完整纹理条件下机器人失败 4/12 次；75% 动作选区为 2/12 次；75% 可见面积选区为 4/12 次。底布已有 1/12 次自然失败，不能把所有失败因果归因于攻击。动作选区没有建立优于简单对照的优势。
4. 75% 时贪心与一次性排序是同一张纹理。源侧保真优于动作选区的 random102 也未可靠保持目标效果，说明源侧代理目标不能充当迁移保证。
5. 可见面积基线虽然合并失败次数与完整纹理相同，Spatial 中有 1 对变弱和 1 对变强，不能称为逐回合无损。样本也不足以证明非劣效。
6. Spatial/0/初态23 是完整纹理诱导的失败被动作选区恢复；Object/4/初态23 则是底布和完整纹理均失败、动作选区成功。详细结果不把自然失败全部归因于攻击。

code/ 包含源查询、贪心、对照、几何检查、目标评测和汇总代码。code/protocol.json 与 pilot_protocol.json 保留原门槛及额外探索性登记；没有把 75% 版本重标成通过 50%/25% 的原门槛。
results/ 包含全部原始查询和回合记录、首次失败及修正后的冒烟日志、断点恢复记录；未完成调用没有计作回合。integrity_checks.json 记录 PNG/噪声/回合完整性验证。
source_candidate_state_rows=1960 包含缓存复用，不是 1960 个独立状态。几何 96 条记录来自 12 个起始状态，不是完整回合暴露统计。

原论文和主表未修改；原纹理保留。未新增长训练；未使用 GPU4。GPU6/7 跑源侧，GPU1/6/7 跑目标，GPU0 仅做轻量无策略渲染。09 进行了代码同步和 CPU 测试。
完整 GitHub 仓库分支：https://github.com/ShaoZhenLiu/tex3d-repro/tree/support-20261101
代码依赖既有 ops/math_20261031/probe_common.py 和生产 residual_lab；权重、环境和场景资产使用服务器现有版本，未打包模型文件。
'''
    (root/'阅读说明.txt').write_text(text,encoding='utf-8')
    mappings={
        source/'summary.json':'results/summary.json',source/'audit.json':'results/source_audit.json',
        source/'completed.json':'results/source_completed.json',root/'results/pilot_summary.json':'results/pilot_summary.json',
        target:'results/target_combined.json',root/'results/geometry/geometry.json':'results/geometry.json',
        root/'decisions.json':'results/decisions.json'}
    for name in ['fit_states','validation_states','leave_one_out_fit','greedy_path','frozen_candidates','texture_manifest','fit_candidates','validation_candidates']:
        mappings[source/(name+'.json')]='results/'+name+'.json'
    for file in (root/'figures').glob('*'):mappings[file]='figures/'+file.name
    for name in ['plain','full','greedy_12','greedy_08','greedy_04','visibility_12','random101_12','random102_12']:
        mappings[source/'textures'/(name+'.png')]='textures/'+name+'.png'
    with tarfile.open(root/'public_payload.tgz','w:gz') as tf:
        for file,name in mappings.items():tf.add(file,arcname=name)
    print(json.dumps(dict(checks=checks,public_payload_bytes=(root/'public_payload.tgz').stat().st_size)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();main(a.root)
