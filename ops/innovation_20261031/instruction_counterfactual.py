"""Same frozen scene/noise, two physically valid object instructions.

Action interaction is not evidence of instruction correctness or task success.
All six exploratory target states and both random controls are reported.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main(repo, output, reference):
    import attack_pi0_cloth as APC
    import attack_pi0 as BASE
    from libero.libero import benchmark
    torch.set_num_threads(2)
    np.random.seed(7)
    torch.manual_seed(7)
    paths = dict(MR=repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png',
                 random101=repo/'experiments/pi0_cloth/r19_random_ctrl/textures/random_cloth_s101.png',
                 random102=repo/'experiments/pi0_cloth/r19_random_ctrl/textures/random_cloth_s102.png')
    textures = {key: np.array(Image.open(path).convert('RGB')) for key, path in paths.items()}
    base = np.array(Image.open(APC.CLOTH_T0).convert('RGB'))
    policy = APC.load_policy('/data/shaozhen.liu/model/RLinf-Pi05-LIBERO-130-fullshot-SFT',
                            model_config='pi05_libero', num_denoise=10)
    config = APC.training_config.get_config('pi05_libero').model
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='exploratory language-texture action interaction; not semantic success or optimization',
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip(),
                  texture_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                  object_floor_dispatch=True, records=[], checks=[])
    suite = benchmark.get_benchmark_dict()['libero_object']()
    was_clean = APC.patch_status() == 'clean'
    APC.ensure_patch(True)

    def render(sim):
        return tuple(np.ascontiguousarray(sim.render(width=256, height=256, camera_name=cam,
                     mode='offscreen')[::-1, ::-1]) for cam in ['agentview', 'robot0_eye_in_hand'])

    def inject(sim, tid, png):
        handle = APC.inject_mujoco_texture(sim, tid, png)
        assert BASE._upload_mujoco_texture(sim, tid), 'native upload failed'
        return handle

    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64)**2)))
    try:
        for task_id in [0, 4, 8]:
            task = suite.get_task(task_id)
            bddl_root = Path('/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/bddl_files')
            bddl_path = bddl_root/task.problem_folder/task.bddl_file
            objects = bddl_path.read_text().split('(:objects')[1].split(')')[0]
            assert 'salad_dressing_1 - salad_dressing' in objects, 'counterfactual object absent'
            inits = suite.get_task_init_states(task_id)
            for initial in [10, 11]:
                env, desc = APC._get_libero_env(task, resolution=256)
                prompts = dict(original=desc, alternative='pick up the salad dressing and place it in the basket')
                env.reset()
                obs = env.set_init_state(inits[initial])
                sim = env.env.sim
                sim.forward()
                APC.align_cloth_to_table(env, floor_mode=True)
                _, bid, _ = APC.get_target_model_matrix(env, APC.CLOTH_SEARCH)
                tid = APC.find_target_texture_id(sim, bid, APC.CLOTH_SEARCH)
                assert tid >= 0
                render(sim)
                handle = inject(sim, tid, base)
                try:
                    for _ in range(10):
                        obs, _, _, _ = env.step(APC.get_libero_dummy_action())
                    plain = render(sim)
                    images = dict(Plain=plain)
                    key = 'object_t%d_i%d' % (task_id, initial)
                    for view, im in zip(['main', 'wrist'], plain):
                        previous = np.array(Image.open(reference/(key+'_PP_'+view+'.png')).convert('RGB'))
                        assert np.array_equal(im, previous), 'audited Plain state not reproduced'
                    for texture, png in textures.items():
                        temp = inject(sim, tid, png)
                        images[texture] = render(sim)
                        APC.restore_mujoco_texture(sim, tid, *temp)
                        assert all(np.array_equal(x, y) for x, y in zip(plain, render(sim)))
                    for noise_id in [0, 1, 2]:
                        seed = 61000 + task_id*100 + initial + noise_id*100000
                        noise = np.random.default_rng(seed).standard_normal((config.action_horizon, config.action_dim)).astype(np.float32)
                        actions = {}
                        for prompt_key, prompt in prompts.items():
                            for texture, pair in images.items():
                                raw = APC.build_raw_inputs(obs, prompt, adv_image_np=pair[0], adv_wrist_image_np=pair[1])
                                with torch.no_grad():
                                    a = np.asarray(policy.infer(raw, noise=noise.copy())['actions'])
                                assert a.ndim == 2 and a.shape[1] == 7 and np.isfinite(a).all()
                                actions[prompt_key+'_'+texture] = a
                        raw = APC.build_raw_inputs(obs, desc, adv_image_np=plain[0], adv_wrist_image_np=plain[1])
                        with torch.no_grad():
                            repeat = np.asarray(policy.infer(raw, noise=noise.copy())['actions'])
                        floor = float(np.max(np.abs(repeat-actions['original_Plain'])))
                        assert floor < 1e-6
                        if noise_id == 0:
                            old = np.load(reference/(key+'_actions.npz'))
                            error = max(float(np.max(np.abs(old[k]-actions['original_'+v]))) for k, v in [('PP','Plain'),('AA','MR')])
                            assert error < 1e-5, ('original actions not reproduced', error)
                            report['checks'].append(dict(task=task_id, initial=initial, reference_maxdiff=error))
                        subset = {k: v[:5, :3].astype(np.float64) for k, v in actions.items()}
                        plain_response = subset['alternative_Plain']-subset['original_Plain']
                        for texture in textures:
                            original_effect = subset['original_'+texture]-subset['original_Plain']
                            alternative_effect = subset['alternative_'+texture]-subset['alternative_Plain']
                            row = dict(task=task_id, initial=initial, noise_id=noise_id, texture=texture,
                                       prompts=prompts, aa_maxdiff=floor, plain_instruction_response_rms=rms(plain_response),
                                       original_texture_effect_rms=rms(original_effect),
                                       alternative_texture_effect_rms=rms(alternative_effect),
                                       language_texture_interaction_rms=rms(alternative_effect-original_effect),
                                       perturbed_instruction_response_rms=rms(subset['alternative_'+texture]-subset['original_'+texture]))
                            report['records'].append(row)
                            print(json.dumps(row), flush=True)
                        np.savez_compressed(output/(key+'_n%d.npz' % noise_id), noise=noise, **actions)
                    (output/'instruction_counterfactual.partial.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                finally:
                    APC.restore_mujoco_texture(sim, tid, *handle)
                    APC._close_libero_env(env)
        report['completed'] = True
        (output/'instruction_counterfactual.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        if was_clean:
            APC.ensure_patch(False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    args = ap.parse_args()
    main(args.repo, args.out, args.reference)
