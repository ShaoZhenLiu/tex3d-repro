# -*- coding: utf-8 -*-
"""attack_oft_cloth_edpa_c1.py — C1 候选：EDPA(B2v2) 唯一改动 = L_patch 特征接口
从融合 projector token 换成源 OFT 内 SigLIP 塔 token（共享视觉分支假设，老师 §4 P2）。
J_C1 = (1-α)·L_patch(S_A,S_P)/EMA_patch + α·L_align(Z_A,Z_P,E(c))/EMA_align
       − λ_ref·L_ref − λ_tv·L_tv      （攻击项上升，正则仍下降；α/EMA/优化器/预算同 v2）
S = vision_backbone.fused_featurizer 输出（按 config 核实为 SigLIP；DINOv2 在前），
   两次塔调用（主/腕相机）token 维拼接；语言对齐项仍在 projector 空间（维度不混用）。
采样流固定：--streams fixed_streams_seed7.pt（gen_fixed_streams.py 复刻 v2 seed7 的
RNG 调用序列）——C1 与 B2v2 的差异因此唯一锁定为特征接口。不加载任何目标模型梯度。
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
CLOTH_DIR = "/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/assets/tablecloth"
CLOTH_MESH = os.path.join(CLOTH_DIR, "tablecloth.obj")
CLOTH_T0 = os.path.join(CLOTH_DIR, "cloth_base.png")
TRAIN_TASKS = [0, 2, 6, 8]
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


class ClothRendererOV(base.DifferentiableRenderer):
    """桌布参数化（与 v1 相同；render 增加 tex_override 供 Plain 参考渲染）."""

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
        gm_path = os.path.join(CLOTH_DIR, "cloth_gain_map.npy")
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

    def render(self, mvp, resolution=(256, 256), model_rot=None, uv_xform=None,
               tex_override=None):
        # tex_override：绕过 adv_noise，直接用给定纹理（Plain 参考 = orig_texture）
        tex = self.texture_tensor() if tex_override is None else tex_override
        pos_homo = torch.cat([self.pos, torch.ones_like(self.pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())
        rast, _ = base.dr.rasterize(self.glctx, pos_clip.unsqueeze(0),
                                    self.faces.int(), resolution=resolution)
        tex_uv, _ = base.dr.interpolate(self.uv.unsqueeze(0), rast,
                                        self.uv_idx.int())
        if uv_xform is not None:
            u, v = tex_uv[..., 0], tex_uv[..., 1]
            nu = uv_xform[0, 0] * u + uv_xform[0, 1] * v + uv_xform[0, 2]
            nv = uv_xform[1, 0] * u + uv_xform[1, 1] * v + uv_xform[1, 2]
            tex_uv = torch.stack([torch.remainder(nu, 1.0),
                                  torch.remainder(nv, 1.0)], dim=-1)
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


def sample_uv_xform(uv_eot, device):
    if not uv_eot:
        return None
    ang = random.uniform(-np.pi, np.pi)
    c, s_ = float(np.cos(ang)), float(np.sin(ang))
    A = np.array([[c, -s_, 0.5 - 0.5 * c + 0.5 * s_],
                  [s_, c, 0.5 - 0.5 * s_ - 0.5 * c]], dtype=np.float32)
    if random.random() < 0.5:
        if random.random() < 0.5:
            M = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], np.float32)
        else:
            M = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]], np.float32)
        A = (np.vstack([M, [0.0, 0.0, 1.0]])
             @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
    A[0, 2] += random.uniform(-0.04, 0.04)
    A[1, 2] += random.uniform(-0.04, 0.04)
    return torch.from_numpy(A).to(device)


def texture_stats(renderer):
    """θ 纹理 vs T0 的扰动统计（0-255 尺度，导出前、与翻转无关）."""
    with torch.no_grad():
        tex = renderer.texture_tensor().detach()
        d = (tex - renderer.orig_texture) * 255.0
        rms = float(torch.sqrt((d ** 2).mean()))
        mab = float(d.abs().mean())
        linf = float(d.abs().max())
        tvv = float(tv_loss(tex - renderer.orig_texture)) * 255.0
    return rms, mab, linf, tvv


def save_ckpt(renderer, param, out_dir, it):
    Image.fromarray(renderer.export_texture()).save(
        os.path.join(out_dir, "Universal_Cloth_OV.png"))
    torch.save(param.detach().cpu(),
               os.path.join(out_dir, "Ep0_Cloth_Delta_OV.pt"))
    rms, mab, linf, tvv = texture_stats(renderer)
    return f"[ckpt] it{it:05d} rms={rms:.2f} mean_abs={mab:.2f} " \
           f"linf={linf:.1f} tv={tvv:.2f}"


def train(args):
    device = "cuda"
    random.seed(args.seed)
    _was_clean = TR.patch_status() == "clean"
    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT,
                              model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    base.check_unnorm_key(cfg, model)
    model_input_size = base.get_image_resize_size(cfg)

    # ---- C1: SigLIP 塔定位（按配置/维度核实，不按变量名猜）----
    def _tower_id(t):
        c = getattr(t, "config", None)
        for k in ("name_or_path", "model_type", "_name_or_path"):
            v = getattr(c, k, None)
            if v:
                return str(v)
        return "?"
    _vb = model.vision_backbone
    for _nm in ("featurizer", "fused_featurizer"):
        _t = getattr(_vb, _nm, None)
        print("[c1] tower %-16s id=%s patches=%s" %
              (_nm, _tower_id(_t), _t.patch_embed.num_patches))
    _sid = _tower_id(_vb.fused_featurizer).lower()
    assert "siglip" in _sid, "fused_featurizer 不是 SigLIP(%s)，中止" % _sid
    print("[c1] vision_backbone_id=%s（DINOv2 在前 / SigLIP 在后）" %
          getattr(cfg, "vision_backbone_id", "?"))
    sig_cap = []
    _hook = _vb.fused_featurizer.register_forward_hook(
        lambda m, i, o: sig_cap.append(o))

    if not (args.load_pool and os.path.exists(args.pool_file)):
        raise SystemExit("v2 仅支持 --load-pool 冻结池（与 F0/B1/B2v1 同池）")
    pool = torch.load(args.pool_file, map_location=device, weights_only=False)
    print(f"[pool] 缓存命中 {sum(len(v) for v in pool.values())} 帧")
    if not args.streams or not os.path.exists(args.streams):
        raise SystemExit("C1 需要 --streams（gen_fixed_streams.py 生成，seed7 复刻）")
    streams = torch.load(args.streams, map_location=device, weights_only=False)
    assert len(streams) == args.iters, (len(streams), args.iters)
    print(f"[c1] 固定采样流 {len(streams)} iters × batch {args.edpa_batch} 已载入")
    sm, ss, dm, ds = _norm_consts(device)

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
    print(f"[train-c1] 池 {len(all_frames)} 帧 × {args.iters} iters "
          f"× batch {args.edpa_batch} 开始（C1: L_patch@SigLIP, align@projector, "
          f"ascent + clean=Plain + 固定流）")
    ema_step, ema_warmup, ema_decay = 0, 100, 0.9
    ema_patch_v, ema_align_v = 0.0, 0.0
    fwd_cnt = bwd_cnt = 0
    diag_probes = {0, args.iters // 2, args.iters - 1} if args.diag else set()
    import time as _t
    t_start = _t.time()

    def compose_plain(fr, uv):
        """Plain 参考合成（θ=0 → orig_texture），主+腕；与 adv 仅差纹理."""
        plain_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                        model_rot=fr["rot"], uv_xform=uv,
                                        tex_override=t0_tex)
        p_chw = plain_rgba.permute(0, 3, 1, 2)
        comp = torch.clamp(p_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                           0.0, 1.0)
        wrist = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            pw_rgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                         model_rot=fr["rot"], uv_xform=uv,
                                         tex_override=t0_tex)
            pw_chw = pw_rgba.permute(0, 3, 1, 2)
            wrist = torch.clamp(pw_chw * fr["wrist_seg"]
                                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]),
                                0.0, 1.0)
        return comp, wrist

    for it in range(args.iters):
        st = streams[it]
        batch = [all_frames[i] for i in st["idx"]]
        param.grad = None
        uv_a = st["uv"].to(device)
        adv_pvs, clean_pvs, ids_list, mask_list = [], [], [], []
        for tid, fr in batch:
            adv_rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                          model_rot=fr["rot"], uv_xform=uv_a)
            adv_chw = adv_rgba.permute(0, 3, 1, 2)
            comp = torch.clamp(adv_chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]),
                               0.0, 1.0)
            adv_wrist_t = fr.get("wrist")
            if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
                adv_w_rgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                                model_rot=fr["rot"], uv_xform=uv_a)
                adv_w_chw = adv_w_rgba.permute(0, 3, 1, 2)
                adv_wrist_t = torch.clamp(
                    adv_w_chw * fr["wrist_seg"]
                    + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
            adv_pvs.append(base._compose_oft_pixel_values(
                adv_rgb=comp, model_input_size=model_input_size,
                siglip_mean=sm, siglip_std=ss, dino_mean=dm, dino_std=ds,
                wrist_tensor=adv_wrist_t).to(torch.bfloat16))
            # clean = Plain 合成帧（修正 2），no_grad 与官方一致
            with torch.no_grad():
                plain_comp, plain_wrist = compose_plain(fr, uv_a)
            clean_pvs.append(base._compose_oft_pixel_values(
                adv_rgb=plain_comp, model_input_size=model_input_size,
                siglip_mean=sm, siglip_std=ss, dino_mean=dm, dino_std=ds,
                wrist_tensor=plain_wrist).to(torch.bfloat16))
            ids_list.append(fr["text_inputs"]["input_ids"][0])
            mask_list.append(fr["text_inputs"]["attention_mask"][0])
        maxlen = max(int(x.shape[0]) for x in ids_list)
        ids = torch.zeros((len(ids_list), maxlen), dtype=torch.long, device=device)
        tmask = torch.zeros((len(mask_list), maxlen), dtype=torch.long, device=device)
        for i, (ii, mm) in enumerate(zip(ids_list, mask_list)):
            ids[i, :ii.shape[0]] = ii
            tmask[i, :mm.shape[0]] = mm
        pv = torch.cat(adv_pvs, dim=0)
        pv_clean = torch.cat(clean_pvs, dim=0)
        del sig_cap[:]
        with base.autocast(dtype=torch.bfloat16):
            adv_embed = model.projector(model.vision_backbone(pv))
            fwd_cnt += 1
        sig_adv = torch.cat(sig_cap[-2:], dim=1).float()
        del sig_cap[:]
        with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
            clean_embed = model.projector(model.vision_backbone(pv_clean))
            lang_embed = model.get_input_embeddings()(ids)
            fwd_cnt += 1
        sig_clean = torch.cat(sig_cap[-2:], dim=1).float()
        adv_embed = adv_embed.float()
        clean_embed = clean_embed.float()
        lang_embed = lang_embed.float()
        # ---- EDPA compute_loss（与官方/v1 相同：token 级 InfoNCE + 对齐 + EMA）----
        logits = torch.bmm(F.normalize(sig_adv, dim=-1),
                           F.normalize(sig_clean, dim=-1).transpose(1, 2)) / 0.07
        labels = torch.arange(logits.size(1), device=logits.device
                              ).unsqueeze(0).expand(logits.size(0), -1)
        patch_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            reduction="none").reshape(logits.size(0), logits.size(1)).mean(dim=1)
        adv_sim = torch.bmm(F.normalize(adv_embed, dim=-1),
                            F.normalize(lang_embed, dim=-1).transpose(1, 2))
        clean_sim = torch.bmm(F.normalize(clean_embed, dim=-1),
                              F.normalize(lang_embed, dim=-1).transpose(1, 2))
        tm = tmask.float().unsqueeze(1)
        adv_sim, clean_sim = adv_sim * tm, clean_sim * tm
        valid = torch.clamp(tm.sum(dim=-1), min=1e-8)
        align_loss = torch.sum(torch.abs(adv_sim - clean_sim), dim=(1, 2)) / valid.squeeze(-1)
        adjusted_decay = ema_decay * min(ema_step / ema_warmup, 1.0)
        ema_patch_v = float(adjusted_decay * ema_patch_v
                            + (1 - adjusted_decay) * patch_loss.mean().item())
        ema_align_v = float(adjusted_decay * ema_align_v
                            + (1 - adjusted_decay) * align_loss.mean().item())
        edpa_cost = ((1 - args.edpa_alpha) * patch_loss / (ema_patch_v + 1e-8)
                     + args.edpa_alpha * align_loss / (ema_align_v + 1e-8)).mean()
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = tv_loss(tex - t0_tex)
        # ---- 修正 1：官方为梯度上升最大化 EDPA cost；ref/TV 仍取下降方向 ----
        objective = edpa_cost - args.lambda_ref * ref - args.lambda_tv * tv
        # diag：分项 θ 梯度范数（探针迭代；retain 后由总目标 backward 收尾）
        if it in diag_probes:
            gp = torch.autograd.grad(((1 - args.edpa_alpha) * patch_loss
                                      / (ema_patch_v + 1e-8)).mean(),
                                     param, retain_graph=True)[0].norm().item()
            ga = torch.autograd.grad((args.edpa_alpha * align_loss
                                      / (ema_align_v + 1e-8)).mean(),
                                     param, retain_graph=True)[0].norm().item()
            gr = torch.autograd.grad(args.lambda_ref * ref, param,
                                     retain_graph=True)[0].norm().item()
            gt = torch.autograd.grad(args.lambda_tv * tv, param,
                                     retain_graph=True)[0].norm().item()
            rms, mab, linf, tvv = texture_stats(renderer)
            line = (f"[diag] it{it:05d} 分项θ梯度: patch={gp:.3e} "
                    f"align={ga:.3e} ref={gr:.3e} tv={gt:.3e} | "
                    f"texRMS={rms:.2f} linf={linf:.1f}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        objective.backward()
        bwd_cnt += 1
        with torch.no_grad():
            param.add_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        ema_step += 1
        log_now = args.diag or it % 50 == 0 or it == args.iters - 1
        if log_now:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            line = (f"[train-c1] it{it:05d} cost={edpa_cost.item():+.4f} "
                    f"patchS={patch_loss.mean().item():+.4f} "
                    f"align={align_loss.mean().item():+.4f} "
                    f"emaP={ema_patch_v:.4f} emaA={ema_align_v:.4f} "
                    f"ref={ref.item():.4f} tv={tv.item():.4f} g={g:.2e} "
                    f"S={tuple(sig_adv.shape[1:])}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0:
            line = save_ckpt(renderer, param, out_dir, it)
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
    _hook.remove()
    wall = (_t.time() - t_start) / 60.0
    line = save_ckpt(renderer, param, out_dir, args.iters - 1)
    print(line, flush=True)
    print(f"[train-edpav2] 成本记录: 更新={args.iters} 前向={fwd_cnt} "
          f"反向={bwd_cnt} 墙钟={wall:.1f}min batch={args.edpa_batch}", flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")
        f.write(f"[cost] updates={args.iters} fwd={fwd_cnt} bwd={bwd_cnt} "
                f"wall_min={wall:.1f} batch={args.edpa_batch} "
                f"direction=ascent clean=plain feat=siglip streams=fixed\n")
    if _was_clean:
        TR.ensure_patch(False)
    print(f"[train] 完成，产物在 {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--pool-tasks", default=",".join(map(str, TRAIN_TASKS)))
    p.add_argument("--pool-file",
                   default=os.path.join(OUT_ROOT, "oft_cloth_pool.pt"))
    p.add_argument("--load-pool", action="store_true")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--uv-eot", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--edpa-alpha", type=float, default=0.2)
    p.add_argument("--edpa-batch", type=int, default=8)
    p.add_argument("--diag", action="store_true",
                   help="诊断模式：固定批+固定UV，逐迭代记录，探针分项梯度")
    p.add_argument("--streams", default="",
                   help="固定采样流 .pt（gen_fixed_streams.py）")
    p.add_argument("--tag", default="c1")
    args = ap.parse_args()
    if args.cmd == "train":
        args.pool_tasks = [int(t) for t in args.pool_tasks.split(",")]
        train(args)


if __name__ == "__main__":
    main()
