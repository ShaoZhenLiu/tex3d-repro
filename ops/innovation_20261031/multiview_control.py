"""Confirm an exploratory cross-view observation with frozen textures/noises.

This reads six already-explored target states. It does not train or select any
texture, and these states must not later be described as untouched target tests.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def metrics(actions):
    a = {key: value[:5, :3].astype(np.float64) for key, value in actions.items()}
    dm, dw, dj = (a[key] - a['PP'] for key in ['AP', 'PA', 'AA'])
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    return dict(main_rms=rms(dm), wrist_rms=rms(dw), joint_rms=rms(dj),
                interaction_rms=rms(dj - dm - dw),
                amplification=rms(dj) / max(rms(dm) + rms(dw), 1e-12))


def main(repo, output, reference):
    import attack_pi0_cloth as APC
    import attack_pi0 as BASE
    from libero.libero import benchmark

    torch.set_num_threads(2)
    np.random.seed(7)
    torch.manual_seed(7)
    texture_paths = dict(
        MR=repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png',
        random101=repo/'experiments/pi0_cloth/r19_random_ctrl/textures/random_cloth_s101.png',
        random102=repo/'experiments/pi0_cloth/r19_random_ctrl/textures/random_cloth_s102.png')
    textures = {key: np.array(Image.open(path).convert('RGB')) for key, path in texture_paths.items()}
    base = np.array(Image.open(APC.CLOTH_T0).convert('RGB'))
    policy = APC.load_policy('/data/shaozhen.liu/model/RLinf-Pi05-LIBERO-130-fullshot-SFT',
                             model_config='pi05_libero', num_denoise=10)
    config = APC.training_config.get_config('pi05_libero').model
    noise_shape = (config.action_horizon, config.action_dim)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='exploratory frozen-target diagnosis, no optimization or closed-loop claims',
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip(),
                  object_floor_dispatch=True, surface_mode='production_centroid',
                  texture_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in texture_paths.items()},
                  records=[], checks=[])
    suite = benchmark.get_benchmark_dict()['libero_object']()
    was_clean = APC.patch_status() == 'clean'
    APC.ensure_patch(True)

    def render(sim):
        return tuple(np.ascontiguousarray(sim.render(width=256, height=256, camera_name=cam,
                     mode='offscreen')[::-1, ::-1]) for cam in ['agentview', 'robot0_eye_in_hand'])

    def inject(sim, tid, png):
        model = getattr(sim.model, '_model', sim.model)
        assert png.shape[:2] == (int(model.tex_height[tid]), int(model.tex_width[tid]))
        handle = APC.inject_mujoco_texture(sim, tid, png)
        assert BASE._upload_mujoco_texture(sim, tid), 'native texture upload failed'
        return handle

    try:
        for task_id in [0, 4, 8]:
            task = suite.get_task(task_id)
            inits = suite.get_task_init_states(task_id)
            for initial in [10, 11]:
                env, desc = APC._get_libero_env(task, resolution=256)
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
                    images = {}
                    key = 'object_t%d_i%d' % (task_id, initial)
                    for view, im in zip(['main', 'wrist'], plain):
                        previous = np.array(Image.open(reference/(key+'_PP_'+view+'.png')).convert('RGB'))
                        assert np.array_equal(im, previous), 'state did not reproduce the audited pixels'
                    for texture, png in textures.items():
                        temp = inject(sim, tid, png)
                        images[texture] = render(sim)
                        APC.restore_mujoco_texture(sim, tid, *temp)
                        restored = render(sim)
                        assert all(np.array_equal(x, y) for x, y in zip(plain, restored)), 'A-B-A pixels changed'
                    for noise_id in [0, 1, 2]:
                        seed = 61000 + task_id*100 + initial + noise_id*100000
                        noise = np.random.default_rng(seed).standard_normal(noise_shape).astype(np.float32)

                        def infer(pair):
                            raw = APC.build_raw_inputs(obs, desc, adv_image_np=pair[0], adv_wrist_image_np=pair[1])
                            with torch.no_grad():
                                result = np.asarray(policy.infer(raw, noise=noise.copy())['actions'])
                            assert result.ndim == 2 and result.shape[1] == 7 and np.isfinite(result).all()
                            return result

                        pp = infer(plain)
                        floor = float(np.max(np.abs(infer(plain) - pp)))
                        assert floor < 1e-6
                        for texture, adv in images.items():
                            actions = dict(PP=pp, AP=infer((adv[0], plain[1])),
                                           PA=infer((plain[0], adv[1])), AA=infer(adv))
                            if texture == 'MR' and noise_id == 0:
                                previous = np.load(reference/(key+'_actions.npz'))
                                maxdiff = max(float(np.max(np.abs(actions[k]-previous[k]))) for k in actions)
                                assert maxdiff < 1e-5, ('MR reference did not reproduce', maxdiff)
                                report['checks'].append(dict(task=task_id, initial=initial, reference_maxdiff=maxdiff))
                            row = dict(task=task_id, initial=initial, texture=texture, noise_id=noise_id,
                                       noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(), aa_maxdiff=floor,
                                       **metrics(actions))
                            report['records'].append(row)
                            np.savez_compressed(output/(key+'_'+texture+'_n%d.npz' % noise_id), noise=noise, **actions)
                            print(json.dumps(row), flush=True)
                    (output/'multiview_control.partial.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                finally:
                    APC.restore_mujoco_texture(sim, tid, *handle)
                    APC._close_libero_env(env)
        report['completed'] = True
        (output/'multiview_control.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
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
