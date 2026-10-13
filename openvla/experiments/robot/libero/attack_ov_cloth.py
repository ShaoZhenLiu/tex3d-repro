"""OpenVLA 侧通用桌布训练（反向迁移：OV 训布 → 打 π0 / OV 自评）.

与 π0 侧 attack_pi0_cloth.py 同构的池化训练：
  collect: OV 环境真实策略轨迹（素色布注入），按 stride 存帧
           [bg(无布渲染)/seg(RGB差分)/mvp/rot/clean_ids/clean_hidden]
  train:   sign-PGD，attack = -MSE(adv_hidden.mean(1), clean_hidden.mean(1))
           （与 attack_openvla_taao 的 adv_action_proxy 同口径）+ λ_ref/λ_tv 外观正则
产物: Universal_Cloth_OV.png + Ep0_Cloth_Delta_OV.pt

用法:
  python attack_ov_cloth.py train --pool-tasks 0,1,2,8 [--iters 2000] \
      [--pool-file ov_cloth_pool.pt] [--collect-only|--load-pool] [--tag s7]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_openvla_taao as base            # noqa: E402
import transfer_pi2ov as TR                    # noqa: E402  补丁/对齐/注入/常量

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/ov_cloth"
CKPT = TR.CKPT
CLOTH_MESH = os.path.join(TR.CLOTH_DIR, "tablecloth.obj")
CLOTH_T0 = TR.CLOTH_T0
TRAIN_TASKS = [0, 1, 2, 8]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]
DINO_MEAM = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


class ClothRendererOV(base.DifferentiableRenderer):
    """桌布参数化（移植自 π0 侧 ClothRenderer，参数名对齐 OV 基类 adv_noise）。"""

    def __init__(self, device="cuda", epsilon=64.0 / 255.0, init_std=0.01, seed=7):
        super().__init__(mesh_path=CLOTH_MESH, orig_texture_path=CLOTH_T0,
                         device=device, scale_xyz=[1.0, 1.0, 1.0],
                         pos_offset=[0.0, 0.0, 0.0], epsilon=epsilon)
        if init_std > 0:
            g = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                self.adv_noise.copy_(
                    torch.randn(self.adv_noise.shape, generator=g).to(device)
                    * init_std)
        gm_path = os.path.join(TR.CLOTH_DIR, "cloth_gain_map.npy")
        self.gain_map = None
        if os.path.exists(gm_path):
            gm = np.load(gm_path)
            self.gain_map = torch.from_numpy(gm.astype(np.float32)).to(device)

    def texture_tensor(self):
        noise = torch.tanh(self.adv_noise) * self.epsilon
        uv_clip = torch.zeros(self.uv.shape[0], 4, device=self.device)
        uv_clip[:, 0] = self.uv[:, 0] * 2.0 - 1.0
        uv_clip[:, 1] = self.uv[:, 1] * 2.0 - 1.0
        uv_clip[:, 3] = 1.0
        rast, _ = base.dr.rasterize(self.glctx, uv_clip.unsqueeze(0),
                                    self.uv_idx.int(),
                                    resolution=(self.tex_h, self.tex_w))
        field, _ = base.dr.interpolate(noise.unsqueeze(0).contiguous(), rast,
                                       self.uv_idx.int())
        cover = (rast[..., 3] > 0).float().unsqueeze(-1)
        return torch.clamp(self.orig_texture + field * cover, 0.0, 1.0)

    def export_texture(self):
        tex = torch.flip(self.texture_tensor().detach(), dims=[1])
        arr = (tex.squeeze(0).cpu().numpy() * 255.0).round().clip(0, 255)
        return arr.astype(np.uint8)

    def render(self, mvp, resolution=(256, 256), model_rot=None):
        tex = self.texture_tensor()
        pos_homo = torch.cat([self.pos, torch.ones_like(self.pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())
        rast, _ = base.dr.rasterize(self.glctx, pos_clip.unsqueeze(0),
                                    self.faces.int(), resolution=resolution)
        tex_uv, _ = base.dr.interpolate(self.uv.unsqueeze(0), rast,
                                        self.uv_idx.int())
        color = base.dr.texture(tex.contiguous(), tex_uv.contiguous(),
                                filter_mode="linear")
        vn_world = (model_rot @ self.vn.T).T if model_rot is not None else self.vn
        vn_interp, _ = base.dr.interpolate(vn_world.unsqueeze(0).contiguous(),
                                           rast, self.faces.int())
        vn_interp = F.normalize(vn_interp, p=2, dim=-1)
        diffuse = torch.clamp(
            (vn_interp * self.light_dir.view(1, 1, 1, 3)).sum(-1, keepdim=True),
            0.0, 1.0)
        lit = torch.clamp(color * (0.6 + 0.4 * diffuse), 0.0, 1.0)
        if self.gain_map is not None:
            lit = lit * self.gain_map.unsqueeze(0)
        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        return lit, mask


def tv_loss(delta):
    return (delta[:, 1:, :, :].abs().mean()
            + delta[:, :, 1:, :].abs().mean()) / 2.0


def collect_frames(cfg, model, processor, task, init_state, num_frames, device,
                   stride=10, max_frames=40, ep_tag=0):
    from libero.libero import benchmark  # noqa: F401
    env, task_description = base.get_libero_env(task, cfg.model_family,
                                                 resolution=256)
    env.reset()
    obs = env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    TR.align_cloth(env)
    _, bid, _ = base.get_target_model_matrix(env, TR.SEARCH)
    if bid == -1:
        env.close()
        raise RuntimeError("未找到桌布 body（XML 未打补丁？）")
    tex_id = TR.ctu.find_target_texture_id(sim, bid, TR.SEARCH)
    ri = TR.ctu.inject_mujoco_texture(sim, tex_id,
                                      np.array(Image.open(CLOTH_T0).convert("RGB")))
    model_input_size = base.get_image_resize_size(cfg)
    sm = torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1)
    ss = torch.tensor(SIGLIP_STD, device=device).view(1, 3, 1, 1)
    dm = torch.tensor(DINO_MEAM, device=device).view(1, 3, 1, 1)
    ds = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    frames = []
    try:
        for t in range(400):
            if len(frames) >= max_frames:
                break
            if t % stride == 0:
                img = base.get_libero_image(obs, 256)
                bg_np = base._render_bg_without_target(env, bid, 256)
                seg = (np.abs(img.astype(np.int16)
                              - bg_np.astype(np.int16)).max(axis=-1) > 10)
                mm, _, _ = base.get_target_model_matrix(env, TR.SEARCH)
                mvp = base.get_render_mvp_from_matrix(
                    env, mm, resolution=(256, 256))
                pil = Image.fromarray(img).resize(
                    (model_input_size, model_input_size))
                prompt = (f"In: What action should the robot take to "
                          f"{task_description.lower()}?\nOut:")
                inputs = processor(prompt, images=pil).to(device)
                with torch.no_grad(), torch.autocast(dtype=torch.bfloat16):
                    ids = model.generate(
                        **inputs, max_new_tokens=7, do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id)
                    pv224 = F.interpolate(
                        (torch.from_numpy(img.astype(np.float32) / 255.0)
                         ).permute(2, 0, 1).unsqueeze(0).to(device),
                        size=(model_input_size, model_input_size),
                        mode="bilinear", align_corners=False)
                    pv = torch.cat([(pv224 - sm) / ss, (pv224 - dm) / ds],
                                   dim=1).to(torch.bfloat16)
                    fwd = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                pixel_values=pv, output_hidden_states=True)
                frames.append({
                    "bg": (torch.from_numpy(bg_np.copy()).float().to(device)
                           / 255.0).permute(2, 0, 1).unsqueeze(0),
                    "seg": torch.from_numpy(seg.astype(np.float32)
                                            ).to(device).view(1, 1, 256, 256),
                    "mvp": mvp, "rot": mm[:3, :3].detach().float(),
                    "ids": ids.detach(),
                    "clean_hidden": fwd.hidden_states[-1].detach().float(),
                })
            # 策略步进（与 transfer_pi2ov 同链）
            img_high = base.get_libero_image(obs, 256)
            img_model = np.array(Image.fromarray(img_high).resize(
                (model_input_size, model_input_size)))
            observation = {
                "full_image": img_model,
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    base.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }
            action = base.get_action(cfg, model, observation, task_description,
                                     processor=processor)
            action = base.normalize_gripper_action(action, binarize=True)
            action = base.invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
    finally:
        TR.ctu.restore_mujoco_texture(sim, tex_id, *ri)
        env.close()
    print(f"[pool] task ep{ep_tag}: +{len(frames)} 帧")
    return frames


def train(args):
    device = "cuda"
    random.seed(args.seed)
    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT,
                              model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    model_input_size = base.get_image_resize_size(cfg)
    sm = torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1)
    ss = torch.tensor(SIGLIP_STD, device=device).view(1, 3, 1, 1)
    dm = torch.tensor(DINO_MEAM, device=device).view(1, 3, 1, 1)
    ds = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)

    if args.load_pool and os.path.exists(args.pool_file):
        pool = torch.load(args.pool_file, map_location=device, weights_only=False)
        print(f"[pool] 缓存命中 {sum(len(v) for v in pool.values())} 帧")
    else:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        pool = {}
        for tid in args.pool_tasks:
            task = suite.get_task(tid)
            inits = suite.get_task_init_states(tid)
            pool[tid] = []
            for ep in range(args.pool_episodes):
                pool[tid].extend(collect_frames(
                    cfg, model, processor, task, inits[ep],
                    args.frames_per_ep, device, stride=args.stride,
                    max_frames=args.max_frames_ep, ep_tag=ep))
        if args.collect_only:
            os.makedirs(os.path.dirname(args.pool_file), exist_ok=True)
            torch.save(pool, args.pool_file)
            print(f"[pool] 已保存 {args.pool_file}（--collect-only 退出）")
            return

    renderer = ClothRendererOV(device=device, epsilon=args.epsilon,
                               init_std=args.init_std, seed=args.seed).to(device)
    t0_tex = renderer.orig_texture
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"train_{stamp}"
                           + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
    log_path = os.path.join(out_dir, "gradient_log.txt")

    param = renderer.adv_noise
    task_ids = list(pool.keys())
    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]]
    for it in range(args.iters):
        tid, fr = random.choice(all_frames)
        param.grad = None
        adv_rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"])
        comp = torch.clamp(adv_rgba * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                           0.0, 1.0)
        pv224 = F.interpolate(comp, size=(model_input_size, model_input_size),
                              mode="bilinear", align_corners=False)
        pv = torch.cat([(pv224 - sm) / ss, (pv224 - dm) / ds],
                       dim=1).to(torch.bfloat16)
        fwd = model(input_ids=fr["ids"], attention_mask=torch.ones_like(fr["ids"]),
                    pixel_values=pv, output_hidden_states=True)
        adv_h = fwd.hidden_states[-1].float().mean(dim=1)
        clean_h = fr["clean_hidden"].float().mean(dim=1)
        attack = -F.mse_loss(adv_h, clean_h)
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = tv_loss(tex - t0_tex)
        (attack + args.lambda_ref * ref + args.lambda_tv * tv).backward()
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            line = (f"[train] it{it:05d} task{tid} attack={attack.item():+.4f} "
                    f"ref={ref.item():.4f} tv={tv.item():.4f} g={g:.2e}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            Image.fromarray(renderer.export_texture()).save(
                os.path.join(out_dir, "Universal_Cloth_OV.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
    TR.ensure_patch(False)
    print(f"[train] 完成，产物在 {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--pool-tasks", default=",".join(map(str, TRAIN_TASKS)))
    p.add_argument("--pool-episodes", type=int, default=2)
    p.add_argument("--frames-per-ep", type=int, default=10)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-frames-ep", type=int, default=40)
    p.add_argument("--pool-file",
                   default=os.path.join(OUT_ROOT, "ov_cloth_pool.pt"))
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--load-pool", action="store_true")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cmd == "train":
        args.pool_tasks = [int(t) for t in args.pool_tasks.split(",")]
        train(args)


if __name__ == "__main__":
    main()
