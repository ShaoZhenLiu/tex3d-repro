"""Native Object evaluation with non-invasive, same-state appearance probes."""
import argparse
import copy
import hashlib
import inspect
import json
import os
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from measurement_core import measure_three


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False, separators=(',', ':')))
    tmp.replace(path)


def main(args):
    import attack_pi0_cloth as A
    import attack_pi0 as BASE
    import mujoco
    from libero.libero import benchmark
    from openpi.models import model as model_module
    from openpi.models_pytorch import preprocessing_pytorch as preprocessing

    p = json.loads(args.protocol.read_text())
    assert A.patch_status() == 'patched'
    assert os.environ.get('REFCLOTH_SURFACE_MODE', 'legacy') == p['surface_mode']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '4', 'GPU4 is prohibited'
    torch.set_num_threads(2)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    tp = p['targets'][args.target]
    policy = A.load_policy(tp['checkpoint'], model_config=tp['config'], num_denoise=p['num_denoise'])
    policy._model.eval().requires_grad_(False)
    cfg = A.training_config.get_config(tp['config']).model
    noise_shape = (cfg.action_horizon, cfg.action_dim)
    base = np.array(Image.open(A.CLOTH_T0).convert('RGB'))
    ref = np.array(Image.open(args.repo / p['texture']).convert('RGB'))
    assert base.shape == ref.shape
    textures = {'plain': base, 'refcloth': ref}
    markers = [np.full_like(base, p[k]) for k in ['marker_a_rgb', 'marker_b_rgb']]
    real_env = A._get_libero_env
    real_inject = A.inject_mujoco_texture
    current = {}
    injection_count = [0]

    def checked_inject(sim, tid, png):
        m = getattr(sim.model, '_model', sim.model)
        assert png.shape[:2] == (int(m.tex_height[tid]), int(m.tex_width[tid]))
        h = real_inject(sim, tid, png)
        assert BASE._upload_mujoco_texture(sim, tid), 'Native texture upload failed'
        injection_count[0] += 1
        return h

    def remember_env(*a, **kw):
        env, desc = real_env(*a, **kw)
        current['env'] = env
        current['cameras'] = {}
        original_step = env.step
        def step(*sa, **sk):
            sim = env.env.sim
            if not getattr(sim, '_object_camera_hook', False):
                original_render = sim.render
                current['original_render'] = original_render
                def sensor_render(*ra, **rk):
                    result = original_render(*ra, **rk)
                    cam = rk.get('camera_name')
                    if cam in ['agentview', 'robot0_eye_in_hand'] and rk.get('width') == 256 and not rk.get('segmentation', False):
                        assert not rk.get('depth', False)
                        native = getattr(sim.data, '_data', sim.data)
                        current['cameras'][cam] = dict(data=copy.copy(native),
                            image=np.ascontiguousarray(result[::-1, ::-1]))
                    return result
                sim.render = sensor_render
                sim._object_camera_hook = True
            return original_step(*sa, **sk)
        env.step = step
        return env, desc

    A._get_libero_env = remember_env
    A.inject_mujoco_texture = checked_inject

    def process(raw):
        import jax
        from openpi_client import image_tools
        z = policy._input_transform(copy.deepcopy(raw))
        middle = {k: v.copy() for k, v in z['image'].items()}
        for key, source in [('base_0_rgb', 'observation/image'), ('left_wrist_0_rgb', 'observation/wrist_image')]:
            assert np.array_equal(middle[key], image_tools.resize_with_pad(raw[source], 224, 224))
        tensors = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...], z)
        obs = model_module.Observation.from_dict(tensors)
        with torch.no_grad():
            images, masks, _, _, _ = policy._model._preprocess_observation(obs, train=False)
        result = {}
        specs = {}
        for key, image, mask in zip(preprocessing.IMAGE_KEYS, images, masks):
            if not bool(mask.item()):
                continue
            t = image.detach().float().cpu().numpy()
            assert t.shape == (1, 3, 224, 224)
            rgb = (t[0].transpose(1, 2, 0).astype(np.float64) + 1) * 127.5
            assert np.max(np.abs(rgb - middle[key].astype(float))) < 1e-3
            result[key] = rgb
            specs[key] = dict(shape=list(t.shape), dtype=str(image.dtype), tensor_sha256=sha(t), active=True)
        assert set(result) == {'base_0_rgb', 'left_wrist_0_rgb'}
        return result, specs

    def integration_state(sim):
        m = getattr(sim.model, '_model', sim.model)
        d = getattr(sim.data, '_data', sim.data)
        flag = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, flag), dtype=np.float64)
        mujoco.mj_getState(m, d, state, flag)
        return state

    def render(sim):
        # Render copied sensor-exposure MjData; live physics and camera cache stay intact.
        context = sim._render_context_offscreen
        live_data = context.data
        result = []
        try:
            for camera in ['agentview', 'robot0_eye_in_hand']:
                context.data = SimpleNamespace(_data=current['cameras'][camera]['data'])
                result.append(np.ascontiguousarray(current['original_render'](width=256, height=256,
                              camera_name=camera, mode='offscreen')[::-1, ::-1]))
        finally:
            context.data = live_data
        return tuple(result)

    def capture(raw, folder=None):
        env = current['env']
        sim = env.env.sim
        _, bid, _ = A.get_target_model_matrix(env, A.CLOTH_SEARCH)
        tid = A.find_target_texture_id(sim, bid, A.CLOTH_SEARCH)
        geoms = np.flatnonzero(np.asarray(sim.model.geom_bodyid) == bid)
        assert tid >= 0 and len(geoms)
        frozen = integration_state(sim)
        actual = (raw['observation/image'], raw['observation/wrist_image'])
        cameras = [current['cameras'][name] for name in ['agentview', 'robot0_eye_in_hand']]
        assert cameras[0]['data'].time == cameras[1]['data'].time
        assert np.array_equal(cameras[0]['data'].qpos, cameras[1]['data'].qpos)
        assert np.array_equal(cameras[0]['data'].qvel, cameras[1]['data'].qvel)
        assert all(np.array_equal(x, y['image']) for x, y in zip(actual, cameras)), 'Hook did not capture consumed cached frame'
        native_current = render(sim)
        maxdiff = max(int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(actual, native_current))
        assert maxdiff == 0, ('cached production observation differs from render', maxdiff)

        def changed(png):
            h = checked_inject(sim, tid, png)
            try:
                return render(sim)
            finally:
                A.restore_mujoco_texture(sim, tid, *h)

        images = dict(plain=changed(base), refcloth=changed(ref),
                      marker_a=changed(markers[0]), marker_b=changed(markers[1]))
        old_rgba = sim.model.geom_rgba[geoms].copy()
        sim.model.geom_rgba[geoms, 3] = 0
        try:
            images['hidden'] = render(sim)
            # Validate alpha hiding independently on the first query of each episode.
            if folder is not None:
                h_a, h_b = changed(markers[0]), changed(markers[1])
                hidden_max = max(int(np.abs(x.astype(int)-y.astype(int)).max()) for x, y in zip(h_a, h_b))
                assert hidden_max == 0, hidden_max
            else:
                hidden_max = None
        finally:
            sim.model.geom_rgba[geoms] = old_rgba
        assert all(np.array_equal(x, y) for x, y in zip(actual, render(sim))), 'A-B-A image mismatch'
        assert np.array_equal(frozen, integration_state(sim)), 'Probe changed integration state'
        transformed = {}
        specs = None
        for condition, pair in images.items():
            r = copy.deepcopy(raw)
            r['observation/image'], r['observation/wrist_image'] = pair
            transformed[condition], spec = process(r)
            if condition == 'plain':
                specs = spec
        actual_processed, actual_specs = process(raw)
        for k in actual_processed:
            assert np.array_equal(actual_processed[k], transformed['plain'][k])
        row = dict(native={}, model_input={}, model_specs=specs,
                   aba_exact=True, integration_state_exact=True,
                   cached_observation_maxdiff=maxdiff, hidden_marker_maxdiff=hidden_max,
                   camera_time=float(cameras[0]['data'].time), query_time=float(sim.data.time),
                   camera_delay_seconds=float(sim.data.time-cameras[0]['data'].time),
                   camera_views_same_state=True,
                   camera_qpos_sha256=sha(cameras[0]['data'].qpos),
                   cloth_world=np.asarray(sim.data.body_xpos[bid]).tolist())
        for j, view in enumerate(['main', 'wrist']):
            row['native'][view] = measure_three({k: v[j] for k, v in images.items()})
        for view in transformed['plain']:
            row['model_input'][view] = measure_three({k: v[view] for k, v in transformed.items()})
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            for c, pair in images.items():
                for view, image in zip(['main', 'wrist'], pair):
                    Image.fromarray(image).save(folder / (c + '_native_' + view + '.png'))
                for view, image in transformed[c].items():
                    Image.fromarray(np.rint(image).clip(0, 255).astype(np.uint8)).save(folder / (c+'_model_'+view+'.png'))
            np.savez_compressed(folder / 'actual_model_rgb255.npz', **actual_processed)
        return row

    suite = benchmark.get_benchmark_dict()[p['suite']]()
    metadata = dict(protocol=p, protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        target=args.target, tasks=args.tasks, noise_shape=list(noise_shape),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip(),
        device=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_name=torch.cuda.get_device_name(),
        base_sha256=hashlib.sha256(Path(A.CLOTH_T0).read_bytes()).hexdigest(),
        ref_sha256=hashlib.sha256((args.repo / p['texture']).read_bytes()).hexdigest(),
        source_hashes={str(Path(inspect.getfile(o))): hashlib.sha256(Path(inspect.getfile(o)).read_bytes()).hexdigest()
                       for o in [A, BASE, type(policy), type(policy._model), model_module, preprocessing]})
    write(metadata, args.out / 'registered.json')

    class PairedPolicy:
        def reset(self, task, initial, condition, probe, save, key):
            self.task, self.initial, self.condition = task, initial, condition
            self.probe, self.save, self.key = probe, save, key
            self.rows, self.frames = [], []
            self.nq = 0

        def infer(self, raw):
            raw = copy.deepcopy(raw)
            seed = 2710000 + ['pi0', 'pi05'].index(args.target)*1000000 + self.task*10000 + self.initial*100 + self.nq
            noise = np.random.default_rng(seed).standard_normal(noise_shape).astype(np.float32)
            row = dict(query=self.nq, noise_sha256=sha(noise),
                input_main_sha256=sha(raw['observation/image']), input_wrist_sha256=sha(raw['observation/wrist_image']),
                proprio_sha256=sha(raw['observation/state']),
                simulator_state_sha256=sha(integration_state(current['env'].env.sim)))
            if self.probe:
                row['measurement'] = capture(raw, args.out/'images'/self.key if self.nq == 0 else None)
            if self.save:
                self.frames.append(np.concatenate([raw['observation/image'], raw['observation/wrist_image']], axis=1))
            with torch.no_grad():
                result = policy.infer(raw, noise=noise.copy())
                if self.nq == 0:
                    repeat = policy.infer(copy.deepcopy(raw), noise=noise.copy())
                    row['aa_action_maxdiff'] = float(np.max(np.abs(result['actions'] - repeat['actions'])))
                    assert row['aa_action_maxdiff'] <= 1e-6, row['aa_action_maxdiff']
            actions = np.asarray(result['actions'])
            assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
            row['actions'] = actions[:p['replan_steps']].tolist()
            row['action_chunk_sha256'] = sha(actions)
            self.rows.append(row)
            self.nq += 1
            if self.nq % 20 == 0:
                print('QUERY', self.key, self.nq, flush=True)
            return result

    wrapper = PairedPolicy()

    def episode(t, initial, c, probe, key, video=False):
        task = suite.get_task(t)
        state = suite.get_task_init_states(t)[initial]
        np.random.seed(700000 + t*1000 + initial)
        wrapper.reset(t, initial, c, probe, video, key)
        before = injection_count[0]
        start = time.monotonic()
        ok = A.evaluate_native(wrapper, task, state, textures[c], max_steps=p['max_steps'],
                num_steps_wait=p['settle_steps'], replan_steps=p['replan_steps'], floor_mode=True)
        assert wrapper.nq > 0
        row = dict(target=args.target, task=t, initial=initial, condition=c, robot_success=bool(ok),
                   queries=wrapper.nq, elapsed_seconds=time.monotonic()-start,
                   native_upload_count=injection_count[0]-before, trajectory=wrapper.rows,
                   language=task.language, probe=probe, completed=True)
        write(row, args.out / (key+'.json'))
        if video and wrapper.frames:
            import imageio
            (args.out/'videos').mkdir(exist_ok=True)
            with imageio.get_writer(str(args.out/'videos'/(key+'.mp4')), fps=4, codec='libx264', quality=7) as writer:
                for frame in wrapper.frames:
                    writer.append_data(frame)
        print('EPISODE '+json.dumps({k:v for k,v in row.items() if k!='trajectory'}), flush=True)
        return row

    if args.mode == 'smoke':
        a = episode(0, 10, 'plain', False, 'smoke_uninstrumented')
        b = episode(0, 10, 'plain', True, 'smoke_instrumented')
        fields = ['noise_sha256', 'input_main_sha256', 'input_wrist_sha256', 'proprio_sha256',
                  'simulator_state_sha256', 'action_chunk_sha256', 'actions']
        assert a['queries'] == b['queries'] and a['robot_success'] == b['robot_success']
        for x, y in zip(a['trajectory'], b['trajectory']):
            assert all(x[k] == y[k] for k in fields), ('Instrumentation changed closed loop', x['query'])
        write(dict(passed=True, target=args.target, queries=a['queries'], fields=fields,
                   robot_success=a['robot_success']), args.out/'smoke_passed.json')
        return

    done = []
    for t in args.tasks:
        for initial in p['initials']:
            pair = []
            for c in p['conditions']:
                key = '%s_t%d_i%d_%s' % (args.target, t, initial, c)
                path = args.out/(key+'.json')
                if path.exists():
                    row = json.loads(path.read_text())
                    assert row['completed'] and row['target'] == args.target
                else:
                    row = episode(t, initial, c, c == 'plain', key,
                                  video=t in p['video_cases']['tasks'] and initial == p['video_cases']['initial'])
                pair.append(row)
                done.append({k:v for k,v in row.items() if k!='trajectory'})
            overlap = min(x['queries'] for x in pair)
            assert [x['noise_sha256'] for x in pair[0]['trajectory'][:overlap]] == [x['noise_sha256'] for x in pair[1]['trajectory'][:overlap]]
    write(dict(completed=True, records=done, peak_cuda_bytes=torch.cuda.max_memory_allocated(), metadata=metadata),
          args.out/'results.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--target', choices=['pi0', 'pi05'], required=True)
    ap.add_argument('--tasks', type=lambda s: [int(x) for x in s.split(',')], default=list(range(10)))
    ap.add_argument('--mode', choices=['smoke', 'run'], default='run')
    ap.add_argument('--memory-fraction', type=float, default=.55)
    main(ap.parse_args())
