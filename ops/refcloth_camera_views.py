"""Render camera counterfactuals at their actual sensor exposure states."""
import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

CAMERAS = ('agentview', 'robot0_eye_in_hand')
INPUTS = ('observation/image', 'observation/wrist_image')


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class Exposures:
    def __init__(self, cloth, mujoco):
        self.cloth = cloth
        self.mujoco = mujoco

    def attach(self, env):
        self.env = env
        self.cameras = {}
        original_step = env.step

        def step(*args, **kwargs):
            sim = env.env.sim
            if not getattr(sim, '_refcloth_camera_hook', False):
                render = sim.render
                self.original_render = render

                def sensor_render(*ra, **rk):
                    result = render(*ra, **rk)
                    name = rk.get('camera_name')
                    if name in CAMERAS and rk.get('width') == 256 and not rk.get('segmentation', False):
                        assert not rk.get('depth', False)
                        data = getattr(sim.data, '_data', sim.data)
                        self.cameras[name] = (copy.copy(data), np.ascontiguousarray(result[::-1, ::-1]))
                    return result

                sim.render = sensor_render
                sim._refcloth_camera_hook = True
            return original_step(*args, **kwargs)

        env.step = step

    def integration(self, sim):
        m = getattr(sim.model, '_model', sim.model)
        d = getattr(sim.data, '_data', sim.data)
        flag = self.mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(self.mujoco.mj_stateSize(m, flag), dtype=np.float64)
        self.mujoco.mj_getState(m, d, values, flag)
        return values

    def render(self, sim):
        context = sim._render_context_offscreen
        previous = context.data
        result = []
        try:
            for camera in CAMERAS:
                context.data = SimpleNamespace(_data=self.cameras[camera][0])
                image = self.original_render(width=256, height=256, camera_name=camera, mode='offscreen')
                result.append(np.ascontiguousarray(image[::-1, ::-1]))
        finally:
            context.data = previous
        return tuple(result)

    def compare(self, raw, candidates):
        sim = self.env.env.sim
        actual = tuple(raw[key] for key in INPUTS)
        data0, data1 = (self.cameras[name][0] for name in CAMERAS)
        assert data0.time == data1.time
        assert np.array_equal(data0.qpos, data1.qpos) and np.array_equal(data0.qvel, data1.qvel)
        assert all(np.array_equal(a, self.cameras[name][1]) for a, name in zip(actual, CAMERAS))
        before = self.integration(sim)
        assert all(np.array_equal(a, b) for a, b in zip(actual, self.render(sim))), 'Sensor exposure differs'
        alternates = {}
        if candidates:
            _, body, _ = self.cloth.get_target_model_matrix(self.env, self.cloth.CLOTH_SEARCH)
            texture_id = self.cloth.find_target_texture_id(sim, body, self.cloth.CLOTH_SEARCH)
            assert texture_id >= 0
            for name, image in candidates.items():
                saved = self.cloth.inject_mujoco_texture(sim, texture_id, image)
                try:
                    alternates[name] = self.render(sim)
                finally:
                    self.cloth.restore_mujoco_texture(sim, texture_id, *saved)
            assert all(np.array_equal(a, b) for a, b in zip(actual, self.render(sim))), 'A-B-A changed images'
        assert np.array_equal(before, self.integration(sim)), 'Rendering changed live physics'
        evidence = dict(integration_exact=True, cached_exposure_exact=True, aba_exact=True,
                        same_camera_state=True, camera_time=float(data0.time),
                        query_time=float(sim.data.time), camera_qpos_sha256=array_sha(data0.qpos),
                        plain_main_sha256=array_sha(actual[0]), plain_wrist_sha256=array_sha(actual[1]))
        return actual, alternates, evidence


class Diagnostic:
    def __init__(self, cloth, mujoco, pi, policy, root):
        self.exposures = Exposures(cloth, mujoco)
        self.pi = pi
        self.policy = policy
        self.root = Path(root)
        self.images = {}
        self.mode = None

    def image(self, name):
        if name not in self.images:
            item = json.loads((self.root / 'texture_manifest.json').read_text())[name]
            path = self.root / item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            self.images[name] = np.array(Image.open(path).convert('RGB'))
        return self.images[name]

    def select(self, image, metadata):
        self.mode = metadata.get('camera_mode')
        if self.mode is None:
            return image
        assert self.mode in ('main', 'wrist')
        self.seed, self.texture = int(metadata['seed']), image
        return self.image('plain')

    def attach(self, env):
        if self.mode is not None:
            self.exposures.attach(env)

    def model_images(self, raw, images):
        inputs = copy.deepcopy(raw)
        inputs.update(zip(INPUTS, images))
        _, observation, _ = self.pi.get_processed_inputs_and_observation(self.policy, inputs)
        assert list(observation.images)[:2] == ['base_0_rgb', 'left_wrist_0_rgb']
        processed, _, _, _, _ = self.policy._model._preprocess_observation(observation, train=False)
        assert len(processed) >= 2
        return [value.detach().float() for value in processed]

    def input_rms(self, raw, plain, candidates):
        with torch.no_grad():
            base = self.model_images(raw, plain)
            full = candidates['main']
            for index in range(2):
                trial = list(plain)
                trial[index] = full[index]
                changed = self.model_images(raw, trial)
                assert all(torch.equal(base[i], changed[i]) for i in range(len(base)) if i != index)
            result = {}
            for name, pair in candidates.items():
                images = self.model_images(raw, pair)
                result[name] = {view: float((images[i] - base[i]).square().mean().sqrt())
                                for i, view in enumerate(('main', 'wrist'))}
            return result

    def intervene(self, raw, query):
        rng = np.random.get_state()
        torch_rng, cuda_rng, python_rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
        candidates = {'main': self.texture}
        measure = query in (0, 10, 30) and self.mode == 'main'
        if measure:
            candidates['main_low'] = self.image(f's{self.seed}_main_low')
            candidates['wrist'] = self.image(f's{self.seed}_wrist')
        actual, alternatives, evidence = self.exposures.compare(raw, candidates)
        alternate = alternatives['main']
        evidence.update(ref_main_sha256=array_sha(alternate[0]), ref_wrist_sha256=array_sha(alternate[1]))
        if measure:
            evidence['policy_input_rms'] = self.input_rms(raw, actual, alternatives)
        now = np.random.get_state()
        assert rng[0] == now[0] and np.array_equal(rng[1], now[1]) and rng[2:] == now[2:]
        assert torch.equal(torch_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert python_rng == random.getstate()
        result = dict(raw)
        index = 0 if self.mode == 'main' else 1
        result[INPUTS[index]] = alternate[index]
        return result, evidence
