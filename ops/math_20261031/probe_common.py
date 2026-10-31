"""Shared source-only probes. Controller owns the shared LIBERO XML patch."""
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


class SourceRuntime:
    def __init__(self, repo, memory_fraction=.49):
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        from residual_lab import adapters as A
        self.A = A
        assert A.TR.patch_status() != 'clean', 'controller must prepare shared XML patch first'
        self.ctx = A.ResidualCtx(device='cuda:0')
        self.ctx._was_clean = False
        for module in [self.ctx.model, self.ctx.ah, self.ctx.pp, self.ctx.nap]:
            if module is not None:
                module.eval().requires_grad_(False)
        self.repo = repo
        self.base_root = repo/'experiments/refcloth_residual/20261031_run1'
        stats=json.loads(Path('/data/shaozhen.liu/code/tex3d-ws/g7/oft-spatial-ckpt/dataset_statistics.json').read_text())
        action_stats=stats[self.ctx.cfg.unnorm_key]['action']
        self.translation_scale=torch.tensor([(b-a+1e-8)/2 for a,b in
            zip(action_stats['q01'][:3],action_stats['q99'][:3])],device=self.ctx.device)

    def start_theta(self, name):
        ctx = self.ctx
        if name == 'init':
            g = torch.Generator(device='cpu').manual_seed(7)
            return torch.randn(ctx.renderer.adv_noise.shape, generator=g).to(ctx.device)*.01
        paths = dict(moderate=self.base_root/'textures/fit_mid_lp_s107_theta.pt',
                     frozen_mr=self.repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Ep0_Cloth_Delta_OV.pt')
        return torch.load(paths[name], map_location=ctx.device, weights_only=False)

    def action(self, features):
        ah = self.ctx.ah
        dtype = next(ah.parameters()).dtype
        normalized=ah.predict_action(features.reshape(-1, 56, 4096).to(dtype)).float()[:, :5, :3]
        # The production action unnormalization is affine. Its offset cancels
        # from every displacement/interaction, so retain the exact linear part.
        return (normalized*self.translation_scale).reshape(-1)

    @contextlib.contextmanager
    def state(self, task_id, initial):
        from libero.libero import benchmark
        A, ctx = self.A, self.ctx
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        task = suite.get_task(task_id)
        env, desc = A.base.get_libero_env(task, ctx.cfg.model_family, resolution=256)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[initial])
        sim = env.env.sim
        sim.forward()
        A.TR.align_cloth(env)
        _, bid, _ = A.base.get_target_model_matrix(env, A.TR.SEARCH)
        tid = A.TR.ctu.find_target_texture_id(sim, bid, A.TR.SEARCH)
        assert tid >= 0
        A.render_native_dual(sim)
        handle = A.strict_inject(sim, tid, ctx.t0_png)
        try:
            quad = A.quad_to_dev(A.capture_quad(env, obs, bid, ctx.device), ctx.device)
            inputs = A.make_inputs(ctx, quad['native_main_px'], desc)
            proprio = A.proprio_of(obs) if ctx.use_proprio else None
            ctx.load_theta('zero')
            zero_png = ctx.renderer.export_png()
            max_zero = int(np.max(np.abs(zero_png.astype(int)-ctx.t0_png.astype(int))))
            assert max_zero <= 1
            clean = tuple(x.detach() if x is not None else None for x in ctx.compose(quad))
            native_clean = A.render_native_dual(sim)

            class State:
                def synth(self, view='AA', grad=False):
                    with torch.set_grad_enabled(grad):
                        pair = ctx.compose(quad)
                        main = pair[0] if view[0] == 'A' else clean[0]
                        wrist = pair[1] if view[1] == 'A' else clean[1]
                        pv = A.base._compose_oft_pixel_values(adv_rgb=main, model_input_size=ctx.size,
                            siglip_mean=ctx.sm, siglip_std=ctx.ss, dino_mean=ctx.dm, dino_std=ctx.ds,
                            wrist_tensor=wrist).to(torch.bfloat16)
                        with A.base.autocast(dtype=torch.bfloat16):
                            _, h = ctx.model.predict_action(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'], pixel_values=pv,
                                unnorm_key=ctx.cfg.unnorm_key, proprio=proprio, proprio_projector=ctx.pp,
                                action_head=ctx.ah, noisy_action_projector=ctx.nap,
                                use_film=getattr(ctx.cfg, 'use_film', False))
                        return h.float().reshape(1, 56, 4096)

                def native_all(self, png):
                    temp = A.strict_inject(sim, tid, png)
                    try:
                        adv = A.render_native_dual(sim)
                    finally:
                        A.ctu.restore_mujoco_texture(sim, tid, *temp)
                    restored = A.render_native_dual(sim)
                    assert all(np.array_equal(x, y) for x, y in zip(native_clean, restored)), 'A-B-A pixel failure'
                    result = {}
                    for view in ['PP', 'AP', 'PA', 'AA']:
                        pair = (adv[0] if view[0] == 'A' else native_clean[0],
                                adv[1] if view[1] == 'A' else native_clean[1])
                        result[view] = ctx.hidden_from_arrays(*pair, inputs, proprio).reshape(1, 56, 4096)
                    return result

                def native_aa(self, png):
                    temp = A.strict_inject(sim, tid, png)
                    try:
                        pair = A.render_native_dual(sim)
                        return ctx.hidden_from_arrays(*pair, inputs, proprio).reshape(1, 56, 4096)
                    finally:
                        A.ctu.restore_mujoco_texture(sim, tid, *temp)

            st = State()
            st.quad = quad
            st.zero_png_maxdiff = max_zero
            st.native0 = st.native_aa(ctx.t0_png).detach()
            repeat = st.native_aa(ctx.t0_png).detach()
            assert torch.equal(st.native0, repeat), 'native A/A feature floor nonzero'
            st.render0 = st.synth('PP').detach()
            yield st
        finally:
            A.ctu.restore_mujoco_texture(sim, tid, *handle)
            env.close()


def rms(x):
    return float(x.double().square().mean().sqrt())


def fidelity(prediction, truth):
    import torch.nn.functional as F
    p, t = prediction.flatten().double(), truth.flatten().double()
    return dict(error_rms=rms(p-t), truth_rms=rms(t), prediction_rms=rms(p),
                relative_error=rms(p-t)/max(rms(t), 1e-12),
                cosine=float(F.cosine_similarity(p, t, dim=0)))
