#!/usr/bin/env python
"""按导师执行清单 §14.1 生成实验 manifest YAML + §14.2 逐集 CSV。

用法:
  gen_manifest.py <train_dir>            # 训练 run -> manifest.yaml + episodes.csv
  gen_manifest.py --eval <result_txt>    # 评测结果 -> episodes.csv
"""
import ast
import csv
import glob
import hashlib
import os
import re
import sys

EX_ROOT = '/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth'
EP_RE = re.compile(r'Task:\s*(\d+)\s*\|\s*Ep:\s*(\d+)\s*\|\s*Success:\s*(True|False)')


def sha256(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(_buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_run_config(train_dir):
    p = os.path.join(train_dir, 'run_config.txt')
    return ast.literal_eval(open(p).read()) if os.path.exists(p) else {}


def manifest_for_train(train_dir):
    cfg = load_run_config(train_dir)
    tex = os.path.join(train_dir, 'Universal_Cloth.png')
    pool = cfg.get('pool_file', '')
    y = []
    y.append(f'experiment_id: {os.path.basename(train_dir)}')
    y.append('protocol_version: r3-dev-1step' )
    y.append('code_commit: f27ee8a0ebb0beb53df332f11a2a88110252c490')
    y.append(f'changed_factor: {cfg.get("tag", "")}')
    y.append('')
    y.append('source_model:')
    y.append('  repo_id: RLinf/RLinf-Pi0-LIBERO-130-fullshot-SFT')
    y.append(f'  weights_dir: {cfg.get("ckpt", "")}')
    y.append('')
    y.append('attack:')
    y.append('  carrier_cm: [60, 40]')
    y.append(f'  base_texture_sha256: {sha256("/data/shaozhen.liu/code/tex3d-deps/cloth_variants/60x40_solid/cloth_base.png")}')
    y.append(f'  mesh_sha256: {sha256("/data/shaozhen.liu/code/tex3d-deps/cloth_variants/60x40_solid/tablecloth.obj")}')
    y.append(f'  epsilon: {cfg.get("epsilon")}')
    y.append(f'  texture_train_seed: {cfg.get("seed")}')
    y.append(f'  iterations: {cfg.get("iters")}')
    y.append(f'  optimizer: sign-SGD lr={cfg.get("attack_lr")} clamp=5')
    y.append(f'  effective_frames_per_update: 1')
    y.append(f'  lambda_mse: {cfg.get("lambda_mse")}')
    y.append(f'  lambda_con: {cfg.get("lambda_con")}')
    y.append(f'  lambda_task: {cfg.get("lambda_task")}')
    y.append(f'  task_eval_every: {cfg.get("task_every_k")}  # 无频率补偿')
    y.append(f'  lambda_ref: {cfg.get("lambda_ref")}')
    y.append(f'  lambda_tv: {cfg.get("lambda_tv")}')
    y.append(f'  texture_sha256: {sha256(tex) if os.path.exists(tex) else "NA"}')
    y.append('')
    y.append('data:')
    y.append(f'  attack_train_tasks: {cfg.get("pool_tasks")}')
    y.append(f'  pool_file: {pool}')
    y.append(f'  pool_sha256: {sha256(pool) if os.path.exists(pool) else "NA"}')
    y.append(f'  negative_bank_size: {"=pool_frames" if cfg.get("lambda_con", 0) > 0 else 0}')
    y.append('')
    y.append('augmentation:')
    y.append(f'  collect_pose_jitter: {"True (collect-time)" if "fs6040" in str(pool) else cfg.get("place_jitter")}')
    y.append(f'  train_uv_eot: {cfg.get("uv_eot")}')
    y.append('  uv_rotation_range: +-pi; mirror p=0.5; uv_translate 0.04; boundary: wrap')
    out = os.path.join(train_dir, 'manifest.yaml')
    open(out, 'w').write('\n'.join(y) + '\n')
    print('wrote', out)


def csv_for_eval(result_txt):
    run_id = os.path.basename(os.path.dirname(result_txt))
    rows = []
    for line in open(result_txt, encoding='utf-8', errors='ignore'):
        m = EP_RE.search(line)
        if m:
            rows.append([run_id, 'r3-dev-1step', '', '', '', 'pi0_fullshot',
                         'libero_spatial', m.group(1), m.group(2), '7', 'nominal',
                         'adv-or-see-log', m.group(3), '', '', 'ok'])
    out = result_txt.replace('evaluation_result.txt', 'episodes.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'protocol_version', 'method', 'train_seed',
                    'texture_sha256', 'victim_id', 'suite', 'task_id',
                    'init_state_id', 'inference_seed', 'pose_config_id',
                    'condition', 'success', 'num_steps', 'termination_reason', 'status'])
        w.writerows(rows)
    print('wrote', out, len(rows), 'rows')


if __name__ == '__main__':
    if sys.argv[1] == '--eval':
        csv_for_eval(sys.argv[2])
    else:
        manifest_for_train(sys.argv[1])
