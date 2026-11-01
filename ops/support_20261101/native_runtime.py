"""Native source-only queries of frozen simulator snapshots and frozen PNGs."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from probe_common import SourceRuntime
from support_core import array_sha, block_ids


class NativeState:
    def __init__(self, runtime, env, desc, task, initial, step, obs, tex_id):
        self.r, self.env, self.sim = runtime, env, env.env.sim
        self.tex_id = tex_id
        self.snapshot = self.sim.get_state().flatten().copy()
        self.key = 't%d_i%d_s%03d' % (task, initial, step)
        self.meta = dict(key=self.key, task=task, initial=initial, step=step, description=desc,
                         simulator_state_sha256=array_sha(self.snapshot))
        A, ctx = runtime.A, runtime.ctx
        pair = A.render_native_dual(self.sim)
        self.clean_pair = tuple(x.copy() for x in pair)
        obs = dict(obs)
        obs['agentview_image'] = np.ascontiguousarray(pair[0][::-1, ::-1])
        obs['robot0_eye_in_hand_image'] = np.ascontiguousarray(pair[1][::-1, ::-1])
        self.inputs = A.make_inputs(ctx, pair[0], desc)
        self.proprio = A.proprio_of(obs) if ctx.use_proprio else None
        self.meta['proprio'] = self.proprio.tolist() if self.proprio is not None else None
        self.a0, _ = self.query(runtime.base)
        self.full_action, full_pair = self.query(runtime.full)
        a0rep, _ = self.query(runtime.base)
        afrep, _ = self.query(runtime.full)
        self.meta['plain_repeat_maxdiff'] = float(np.max(np.abs(self.a0-a0rep)))
        self.meta['full_repeat_maxdiff'] = float(np.max(np.abs(self.full_action-afrep)))
        assert self.meta['plain_repeat_maxdiff'] == self.meta['full_repeat_maxdiff'] == 0
        self.meta['full_changed_pixel_fraction'] = [float(np.any(x != y, axis=-1).mean())
                                                   for x, y in zip(self.clean_pair, full_pair)]
        self.cache = {array_sha(runtime.base): self.a0, array_sha(runtime.full): self.full_action}

    def restore(self):
        self.sim.set_state_from_flattened(self.snapshot)
        self.sim.forward()

    def pixels(self, png):
        self.restore()
        A = self.r.A
        before = A.render_native_dual(self.sim)
        assert all(np.array_equal(x, y) for x, y in zip(before, self.clean_pair)), 'Snapshot restoration changed clean RGB'
        h = A.strict_inject(self.sim, self.tex_id, png)
        try:
            pair = A.render_native_dual(self.sim)
        finally:
            A.ctu.restore_mujoco_texture(self.sim, self.tex_id, *h)
        after = A.render_native_dual(self.sim)
        assert all(np.array_equal(x, y) for x, y in zip(before, after)), 'Native A-B-A pixel failure'
        assert np.array_equal(self.sim.get_state().flatten(), self.snapshot), 'Query stepped physics'
        return pair

    @torch.no_grad()
    def query(self, png):
        pair = self.pixels(png)
        ctx = self.r.ctx
        hidden = ctx.hidden_from_arrays(pair[0], pair[1], self.inputs, self.proprio)
        dtype = next(ctx.ah.parameters()).dtype
        action = ctx.ah.predict_action(hidden.reshape(1, 56, 4096).to(dtype)).float()
        action = action[0].cpu().numpy().astype(np.float64)
        assert action.ndim == 2 and action.shape[1] == 7 and np.isfinite(action).all()
        return action, pair

    def action(self, png):
        key = array_sha(png)
        if key not in self.cache:
            self.cache[key] = self.query(png)[0]
        return self.cache[key]

    def visibility(self, grid):
        def resize(pair):
            return [F.interpolate(torch.from_numpy(x.copy()).permute(2,0,1).float()[None]/255.,
                size=(self.r.ctx.size,self.r.ctx.size), mode='bilinear', align_corners=False)[0].numpy()
                    for x in pair]
        base = resize(self.clean_pair)
        labels = block_ids(self.r.base.shape, grid)
        values = []
        for j in range(grid*grid):
            marker = self.r.base.copy()
            marker[labels == j] = [255, 0, 255]
            rgb = resize(self.pixels(marker))
            values.append([float(np.max(np.abs(a-b),axis=0).__gt__(1/255.).mean()) for a,b in zip(rgb,base)])
        return values


class NativeRuntime(SourceRuntime):
    def __init__(self, repo, fraction, output):
        super().__init__(repo, fraction)
        self.output = Path(output)
        self.base = self.ctx.t0_png.copy()
        path = repo/'experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png'
        self.full = np.array(Image.open(path).convert('RGB'))
        self.full_path = str(path)
        assert self.base.shape == self.full.shape
        assert not self.ctx.cfg.use_diffusion, 'This protocol assumes deterministic L1 action inference'
        import json
        stats=json.loads(Path('/data/shaozhen.liu/code/tex3d-ws/g7/oft-spatial-ckpt/dataset_statistics.json').read_text())
        self.action_stats=stats[self.ctx.cfg.unnorm_key]['action']
        self.scale=(np.array(self.action_stats['q99'])-np.array(self.action_stats['q01'])+1e-8)/2.
        self.envs = []

    def close_states(self):
        for env in self.envs:
            env.close()
        self.envs = []

    def collect(self, tasks, initial, steps):
        from libero.libero import benchmark
        from residual_lab.capture import _policy_step
        self.close_states()
        suite=benchmark.get_benchmark_dict()['libero_spatial']()
        states=[]
        A, ctx = self.A, self.ctx
        for task_id in tasks:
            task=suite.get_task(task_id)
            env,desc=A.base.get_libero_env(task,ctx.cfg.model_family,resolution=256)
            self.envs.append(env)
            env.seed(7)
            env.reset()
            obs=env.set_init_state(suite.get_task_init_states(task_id)[initial])
            sim=env.env.sim
            sim.forward()
            A.TR.align_cloth(env)
            _, bid, _=A.base.get_target_model_matrix(env,A.TR.SEARCH)
            tid=A.TR.ctu.find_target_texture_id(sim,bid,A.TR.SEARCH)
            assert tid>=0
            A.render_native_dual(sim)
            A.strict_inject(sim,tid,self.base)  # private env remains at base until closed
            # Rebuild the observations after alignment/upload before policy stepping.
            pair=A.render_native_dual(sim)
            obs=dict(obs)
            obs['agentview_image']=np.ascontiguousarray(pair[0][::-1,::-1])
            obs['robot0_eye_in_hand_image']=np.ascontiguousarray(pair[1][::-1,::-1])
            for step in range(max(steps)+1):
                if step in steps:
                    state=NativeState(self,env,desc,task_id,initial,step,obs,tid)
                    states.append(state)
                    print('STATE',state.meta,flush=True)
                if step == max(steps):
                    break
                if step<10:
                    obs,_,done,_=env.step([0,0,0,0,0,0,-1])
                    assert not done
                else:
                    obs=_policy_step(ctx,env,obs,desc)
                    if obs is None:
                        raise RuntimeError('Clean trajectory terminated before predeclared snapshot')
        return states
