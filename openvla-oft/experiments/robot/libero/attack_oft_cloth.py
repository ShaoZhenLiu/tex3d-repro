"""OFT 侧通用桌布训练（与 OV 版 attack_ov_cloth.py 同构，动作链适配 OFT）.

机制照抄 attack_oft.py 生产路径（绝不走 model.generate——OFT 并行解码
action chunk，HF 缓存里 generate 分支有 torch 兼容 bug）：
  前向  model.predict_action(input_ids, attention_mask, pixel_values,
        unnorm_key, proprio, proprio_projector, action_head,
        noisy_action_projector, use_film) -> (action_tokens, hidden)
  动作  action_head.predict_action(hidden) —— 可微 action chunk
  像素  base._compose_oft_pixel_values（siglip+dino 双塔归一化拼接）
  精度  torch.cuda.amp.autocast(dtype=bfloat16)（老 API，不带 device_type）

池化训练：
  collect: OFT 真实策略轨迹（素色布 T0 注入），按 stride 存帧
           [bg(无布渲染)/seg(RGB差分)/mvp/rot/text_inputs/proprio/wrist/
            clean_action_head]
  train:   sign-PGD，attack = -MSE(adv_action_chunk, clean_action_chunk)
           + λ_ref/λ_tv 外观正则
产物: Universal_Cloth_OV.png + Ep0_Cloth_Delta_OV.pt

用法:
  python attack_oft_cloth.py train --pool-tasks 0,1,2,8 [--iters 2000] \
      [--pool-file oft_cloth_pool.pt] [--collect-only|--load-pool] [--tag s7]
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
import attack_oft as base            # noqa: E402
import sys as _s; _s.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import transfer_pi2ov as TR                    # noqa: E402  补丁/对齐/注入/常量

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth"
CKPT = "/data/shaozhen.liu/code/tex3d-ws/g7/oft-spatial-ckpt"
CLOTH_MESH = os.path.join(TR.CLOTH_DIR, "tablecloth.obj")
CLOTH_T0 = TR.CLOTH_T0
TRAIN_TASKS = [0, 1, 2, 8]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def _norm_consts(device):
    sm = torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1)
    ss = torch.tensor(SIGLIP_STD, device=device).view(1, 3, 1, 1)
    dm = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    ds = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    return sm, ss, dm, ds


def _build_action_modules(cfg, model):
    """OFT 动作头三件套（调用方式与 attack_oft.eval_libero 一致）."""
    if base.get_action_head is None:
        raise RuntimeError("OFT 需 openvla_utils.get_action_head（导入失败）")
    action_head = None
    proprio_projector = None
    noisy_action_projector = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = base.get_action_head(cfg, model.llm_dim)
    if cfg.use_proprio and base.get_proprio_projector is not None:
        proprio_projector = base.get_proprio_projector(
            cfg, model.llm_dim, proprio_dim=8)
    if cfg.use_diffusion and base.get_noisy_action_projector is not None:
        noisy_action_projector = base.get_noisy_action_projector(
            cfg, model.llm_dim)
    return action_head, proprio_projector, noisy_action_projector


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


# ------------------------------------------------------------ 腕相机视图
def _wrist_camera_id(sim):
    for name in ("robot0_eye_in_hand", "robot0_eye", "eye_in_hand"):
        try:
            return sim.model.camera_name2id(name)
        except Exception:
            continue
    raise RuntimeError("未找到腕相机（robot0_eye_in_hand）")


def get_wrist_mvp_from_matrix(env, model_matrix, resolution=(256, 256),
                              proj_flip_x: float = -1.0,
                              proj_flip_y: float = -1.0):
    """与 base.get_render_mvp_from_matrix 同构，但取腕相机 robot0_eye_in_hand。"""
    sim = env.sim if not hasattr(env, "unwrapped") else env.unwrapped.sim
    W, H = resolution
    cam_id = _wrist_camera_id(sim)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    view_rot = cam_xmat.T
    view_mtx = np.eye(4, dtype=np.float32)
    view_mtx[:3, :3] = view_rot
    view_mtx[:3, 3] = -(view_rot @ cam_pos)
    fovy_deg = float(sim.model.cam_fovy[cam_id])
    aspect = float(W) / float(H)
    near, far = 0.01, 10.0
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    proj_mtx = np.zeros((4, 4), dtype=np.float32)
    proj_mtx[0, 0] = proj_flip_x * f / aspect
    proj_mtx[1, 1] = proj_flip_y * f
    proj_mtx[2, 2] = (far + near) / (near - far)
    proj_mtx[2, 3] = (2 * far * near) / (near - far)
    proj_mtx[3, 2] = -1.0
    return (torch.from_numpy(proj_mtx).cuda()
            @ torch.from_numpy(view_mtx).cuda() @ model_matrix)


def _render_wrist_bg_without_target(env, body_id: int, resolution: int):
    """腕视图下隐藏桌布的无布渲染（照 base._render_bg_without_target，换相机）。"""
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    geom_ids = [g for g in range(sim.model.ngeom)
                if sim.model.geom_bodyid[g] == body_id]
    if not geom_ids:
        return None
    orig_alpha = [float(sim.model.geom_rgba[g, 3]) for g in geom_ids]
    try:
        for g in geom_ids:
            sim.model.geom_rgba[g, 3] = 0.0
        img = sim.render(width=resolution, height=resolution,
                         camera_name="robot0_eye_in_hand", mode="offscreen")
        img = img[::-1, ::-1]
    finally:
        for g, a in zip(geom_ids, orig_alpha):
            sim.model.geom_rgba[g, 3] = a
    return img


def collect_frames(cfg, model, processor, action_head, proprio_projector,
                   noisy_action_projector, task, init_state, num_frames, device,
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
    use_proprio = bool(getattr(cfg, "use_proprio", False))
    use_wrist = getattr(cfg, "num_images_in_input", 1) > 1
    sm, ss, dm, ds = _norm_consts(device)
    frames = []
    try:
        for t in range(400):
            if len(frames) >= max_frames:
                break
            if t % stride == 0:
                img = base.get_libero_image(obs)
                bg_np = base._render_bg_without_target(env, bid, 256)
                seg = (np.abs(img.astype(np.int16)
                              - bg_np.astype(np.int16)).max(axis=-1) > 10)
                mm, _, _ = base.get_target_model_matrix(env, TR.SEARCH)
                mvp = base.get_render_mvp_from_matrix(
                    env, mm, resolution=(256, 256))
                pil = Image.fromarray(img).resize(
                    (model_input_size, model_input_size))
                prompt = base._build_openvla_prompt(task_description)
                inputs = processor(prompt, pil).to(device)
                full_t = (torch.from_numpy(img.astype(np.float32) / 255.0)
                          ).permute(2, 0, 1).unsqueeze(0).to(device)
                wrist_t = None
                wrist_bg_t = None
                wrist_seg_t = None
                wrist_mvp = None
                if use_wrist:
                    wnp = base._extract_wrist_image(obs, 256)
                    if wnp is not None:
                        wrist_t = (torch.from_numpy(
                            wnp.astype(np.float32) / 255.0)
                            ).permute(2, 0, 1).unsqueeze(0).to(device)
                        # 腕视图对抗合成四件套：无布渲染 / RGB 差分 seg / 腕 mvp
                        wrist_bg_np = _render_wrist_bg_without_target(
                            env, bid, 256)
                        if wrist_bg_np is not None:
                            wrist_seg = (np.abs(
                                wnp.astype(np.int16)
                                - wrist_bg_np.astype(np.int16)
                                ).max(axis=-1) > 10)
                            wrist_bg_t = (
                                torch.from_numpy(wrist_bg_np.copy()).float()
                                .to(device) / 255.0
                                ).permute(2, 0, 1).unsqueeze(0)
                            wrist_seg_t = torch.from_numpy(
                                wrist_seg.astype(np.float32)
                                ).to(device).view(1, 1, 256, 256)
                            wrist_mvp = get_wrist_mvp_from_matrix(
                                env, mm, resolution=(256, 256))
                            print(f"    [帧{t}] 主seg {seg.mean():.3f} "
                                  f"腕seg {wrist_seg.mean():.3f}")
                proprio = None
                if use_proprio:
                    proprio = np.concatenate((
                        obs["robot0_eef_pos"],
                        base.quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                # clean 侧：真实观测（素色布）→ predict_action + action_head
                with torch.no_grad():
                    with base.autocast(dtype=torch.bfloat16):
                        clean_pv = base._compose_oft_pixel_values(
                            adv_rgb=full_t,
                            model_input_size=model_input_size,
                            siglip_mean=sm, siglip_std=ss,
                            dino_mean=dm, dino_std=ds,
                            wrist_tensor=wrist_t).to(torch.bfloat16)
                        _, clean_hidden = model.predict_action(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            pixel_values=clean_pv,
                            unnorm_key=cfg.unnorm_key,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            action_head=action_head,
                            noisy_action_projector=noisy_action_projector,
                            use_film=getattr(cfg, "use_film", False),
                        )
                        clean_hidden = clean_hidden.detach()
                        clean_action_head = action_head.predict_action(
                            clean_hidden).detach()
                frames.append({
                    "bg": (torch.from_numpy(bg_np.copy()).float().to(device)
                           / 255.0).permute(2, 0, 1).unsqueeze(0),
                    "seg": torch.from_numpy(seg.astype(np.float32)
                                            ).to(device).view(1, 1, 256, 256),
                    "mvp": mvp, "rot": mm[:3, :3].detach().float(),
                    "text_inputs": inputs,
                    "proprio": proprio,
                    "wrist": wrist_t,
                    "wrist_bg": wrist_bg_t,
                    "wrist_seg": wrist_seg_t,
                    "wrist_mvp": wrist_mvp,
                    "clean_action_head": clean_action_head.float(),
                    "clean_hidden": clean_hidden.to(torch.bfloat16),
                })
            # 策略步进（与 transfer_pi2ov 同链，OFT 走签名兼容包装）
            img_high = base.get_libero_image(obs)
            img_model = np.array(Image.fromarray(img_high).resize(
                (model_input_size, model_input_size)))
            observation = {"full_image": img_model}
            if use_proprio:
                observation["state"] = np.concatenate((
                    obs["robot0_eef_pos"],
                    base.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
            else:
                observation["state"] = np.zeros(8)
            if use_wrist:
                wnp_m = base._extract_wrist_image(obs, model_input_size)
                if wnp_m is not None:
                    observation["wrist_image"] = wnp_m
            executed = base._query_action_with_compatible_signature(
                cfg, model, observation, task_description, processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=getattr(cfg, "use_film", False))
            if isinstance(executed, list):
                executed = np.asarray(executed[0], dtype=np.float32)
            else:
                executed = np.asarray(executed, dtype=np.float32)
            executed = base.normalize_gripper_action(executed, binarize=True)
            if cfg.model_family == "openvla":
                executed = base.invert_gripper_action(executed)
            obs, _, done, _ = env.step(executed.tolist())
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
    # 只在结束时撤"自己打的"补丁：若启动时补丁已被其他评测持有（patched），
    # 盲目 unpatch 会打崩并发使用 XML 的 transfer_pi2ov 评测（2026-10-12 事故）
    _was_clean = TR.patch_status() == "clean"
    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT,
                              model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    base.check_unnorm_key(cfg, model)
    model_input_size = base.get_image_resize_size(cfg)
    action_head, proprio_projector, noisy_action_projector = \
        _build_action_modules(cfg, model)
    sm, ss, dm, ds = _norm_consts(device)

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
                    cfg, model, processor, action_head, proprio_projector,
                    noisy_action_projector, task, inits[ep],
                    args.frames_per_ep, device, stride=args.stride,
                    max_frames=args.max_frames_ep, ep_tag=ep))
        if args.collect_only:
            os.makedirs(os.path.dirname(args.pool_file), exist_ok=True)
            torch.save(pool, args.pool_file)
            print(f"[pool] 已保存 {args.pool_file}（--collect-only 退出）")
            return
        # 非 collect-only 也落盘，供后续 --load-pool 复用（重采一轮要 12 分钟）
        os.makedirs(os.path.dirname(args.pool_file), exist_ok=True)
        torch.save(pool, args.pool_file)
        print(f"[pool] 已保存 {args.pool_file}")

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
    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]
                  if fr["mvp"] is not None]
    print(f"[train] 池 {len(all_frames)} 帧 × {args.iters} iters 开始")
    for it in range(args.iters):
        tid, fr = random.choice(all_frames)
        param.grad = None
        # 主视图对抗合成
        adv_rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"])
        adv_chw = adv_rgba.permute(0, 3, 1, 2)
        comp = torch.clamp(adv_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                           0.0, 1.0)
        # 腕视图对抗合成（OFT 双相机：只打主相机时动作头几乎不动）
        adv_wrist_t = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            adv_w_rgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                            model_rot=fr["rot"])
            adv_w_chw = adv_w_rgba.permute(0, 3, 1, 2)
            adv_wrist_t = torch.clamp(
                adv_w_chw * fr["wrist_seg"]
                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
        pv = base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=model_input_size,
            siglip_mean=sm, siglip_std=ss, dino_mean=dm, dino_std=ds,
            wrist_tensor=adv_wrist_t).to(torch.bfloat16)
        with base.autocast(dtype=torch.bfloat16):
            _, adv_hidden = model.predict_action(
                input_ids=fr["text_inputs"]["input_ids"],
                attention_mask=fr["text_inputs"]["attention_mask"],
                pixel_values=pv,
                unnorm_key=cfg.unnorm_key,
                proprio=fr["proprio"],
                proprio_projector=proprio_projector,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                use_film=getattr(cfg, "use_film", False),
            )
            adv_action = action_head.predict_action(adv_hidden)
        # 生产配方（attack_oft.py）：动作损失 + alpha_feature×hidden 特征损失
        attack_action = -F.mse_loss(adv_action.float(), fr["clean_action_head"])
        if "clean_hidden" in fr:
            attack_feat = -F.mse_loss(adv_hidden.float(),
                                      fr["clean_hidden"].float())
            attack = attack_action + args.alpha_feature * attack_feat
        else:
            attack_feat = None
            attack = attack_action
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = tv_loss(tex - t0_tex)
        (attack + args.lambda_ref * ref + args.lambda_tv * tv).backward()
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            extra = (f" feat={attack_feat.item():+.4f}"
                     if attack_feat is not None else "")
            line = (f"[train] it{it:05d} task{tid} attack={attack.item():+.4f}"
                    f"{extra} ref={ref.item():.4f} tv={tv.item():.4f} "
                    f"g={g:.2e}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            Image.fromarray(renderer.export_texture()).save(
                os.path.join(out_dir, "Universal_Cloth_OV.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
    if _was_clean:
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
                   default=os.path.join(OUT_ROOT, "oft_cloth_pool.pt"))
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--load-pool", action="store_true")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--alpha-feature", type=float, default=10.0,
                   help="hidden 特征损失权重（attack_oft.py 生产默认 10.0）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cmd == "train":
        args.pool_tasks = [int(t) for t in args.pool_tasks.split(",")]
        train(args)


if __name__ == "__main__":
    main()
