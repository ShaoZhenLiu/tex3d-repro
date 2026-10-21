# -*- coding: utf-8 -*-
"""OFT 侧通用桌布训练（UPA-RFAS 适配 B3；映射说明《B3_UPA适配映射说明_20261020.md》先行冻结）.

九月复现（41 /data/shaozhen.liu/code/UPA-RFAS @a066bb4，OpenVLA-7B 源+50×50 贴片）
的损失模块移植到 OFT+桌布基座。实现门结果（diag 探针 upa_attn_probe）：
OFT 定制 modeling 在 eager 与 sdpa 两后端下注意力图结构性不同（末层注意力
cos=0.055，embedding/projector 完全一致），无法在不修改共享 ckpt 代码的前提
下取到与生产路径一致的注意力——按映射说明 §4 预登记分支，**PAD 缺失，降级
命名为 "UPA-RFAS, cloth-adapted (budgeted, no-PAD variant)"**。保留三核心：
  1) 特征/对比代理  BiContrastive（双向 InfoNCE，T=0.07，token 级 [B,N,N]，最大化推离）
                    + featL1（L1 最大化），表征=projector(vision_backbone(pv)) token
  2) PSM            布面 token 均值表征靠近 probe 短语（input-embedding 原型、
                    LogSumExp），远离当前指令末层文本表征（文本表征经复刻的
                    multimodal 前向取，no_grad 参考，偏离披露于映射说明）
  3) 鲁棒内循环     每批 dual_step 步 PGD 训练 ε=2/255 辅助扰动（内层最小化 con+L1，
                    即把表征拉回 clean；仅作用于布面之外区域），θ 须在其上存活；
                    辅助扰动逐批重建、绝不部署
裁剪：动作 token 损失（OpenVLA 专属）、九月 A1/A3/A4/F 扩展、PAD（实现门失败）。
λ（con 1.0 / featL1 0.1 / text 0.5；λ_attn 项结构性移除）。
优化器 sign-PGD lr 0.05（θ 域，同 F0/B1/B2）；预算 5000 更新 × batch 8；
clean 参考 = Plain 合成帧（B2v1 教训）；模型全程 sdpa 生产后端。

用法（41，cwd=$TEX3D_WS/g7）:
  python attack_oft_cloth_upa.py train --pool-file r0f_pool.pt --load-pool \
      --iters 5000 --seed 7 --uv-eot [--diag] --tag upa
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
import transfer_pi2ov as TR                    # noqa: E402

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
PHRASES = ("put", "pick up", "place", "turn on", "push", "open", "close")


def _norm_consts(device):
    sm = torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1)
    ss = torch.tensor(SIGLIP_STD, device=device).view(1, 3, 1, 1)
    dm = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    ds = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    return sm, ss, dm, ds


class ClothRendererOV(base.DifferentiableRenderer):
    """与 attack_oft_cloth_edpa_v2.ClothRendererOV 相同（tex_override 供 Plain 参考）."""

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


# ---------------------------------------------------------------- UPA 损失
def bi_contrastive(adv_tok, clean_tok, temp=0.07):
    """九月 BiContrastiveLoss 同构：token 级对角标签 CE（[B,N,N]），返回 CE 本身.
    攻击侧最大化（推离 clean 对应位 token）。"""
    logits = torch.bmm(F.normalize(adv_tok, dim=-1),
                       F.normalize(clean_tok, dim=-1).transpose(1, 2)) / temp
    labels = torch.arange(logits.size(1), device=logits.device
                          ).unsqueeze(0).expand(logits.size(0), -1)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                         reduction="none").reshape(logits.size(0), logits.size(1))
    return ce.mean(dim=1).mean()


def text_hidden_forward(mdl, ids, tmask, pv, proprio, pproj):
    """复刻 OFT multimodal 前向取**末层文本段 hidden**（PSM 的指令表征参考）：
    lang_emb → _process_vision_features → 追加 proprio → [BOS][vision][propio][text]
    → 内层 language_model（sdpa 生产后端；无动作查询，文本段布局精确已知）。
    仅 no_grad 使用（参考侧 detach，映射说明披露）。

    第13轮核查（psm_slice_check）：历史版本取 [:, S-L:] 混入 proj 末 token
    （B3 有 proprio → proprio 末位），漏 BOS——off-by-one 缺陷；已修正为
    [:, 1+pn:]（纯 text rest 区）。历史 B3 结果以"PSM 参考含 1/34 proprio
    token"限定保留（PSM 分项梯度最小，方向性影响小），不重跑。"""
    lang_emb = mdl.get_input_embeddings()(ids)
    proj = mdl._process_vision_features(pv, lang_emb, use_film=False)
    if pproj is not None and proprio is not None:
        proj = mdl._process_proprio_features(proj, proprio, pproj)
    pn = proj.shape[1]
    embeds = torch.cat([lang_emb[:, :1, :], proj, lang_emb[:, 1:, :]], dim=1)
    am = torch.cat([tmask[:, :1],
                    torch.ones(tmask.shape[0], pn, dtype=tmask.dtype,
                               device=tmask.device),
                    tmask[:, 1:]], dim=1)
    out = mdl.language_model(inputs_embeds=embeds, attention_mask=am,
                             output_hidden_states=True)
    return out.hidden_states[-1][:, 1 + int(pn):, :]  # 纯 text rest 区（不含BOS）


def psm_text_loss(mdl, adv_tok, txt_hidden, vis_mask, tokenizer, tau=0.3,
                  alpha_pos=1.0, beta_neg=0.5):
    """PSM：布面 token 均值表征靠近 probe 短语原型、远离当前指令文本表征（待最小化）."""
    B, P, D = adv_tok.shape
    m = (vis_mask / (vis_mask.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)
    v_patch = F.normalize((adv_tok.float() * m).sum(dim=1), dim=-1)   # [B,D]
    E = mdl.get_input_embeddings().weight.detach().to(adv_tok.device).float()
    prots = []
    for p in PHRASES:
        ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        prots.append(E[torch.tensor(ids, device=E.device)].mean(dim=0))
    Pk = F.normalize(torch.stack(prots, dim=0), dim=-1)               # [K,D]
    pos = torch.matmul(v_patch, Pk.t()) / tau
    L_pos = -torch.logsumexp(pos, dim=-1).mean()
    v_text = F.normalize(txt_hidden.float().mean(dim=1), dim=-1)      # [B,D] no_grad
    L_neg = (v_patch * v_text).sum(dim=-1).mean()
    return alpha_pos * L_pos + beta_neg * L_neg


def vis_token_mask(batch, P, device):
    """布面 token mask：OFT fused 骨干把双视图按 [主 16×16][腕 16×16] 拼接
    （modeling_prismatic FusedVisionTower.forward 的 cat(dim=1) 核实，P=512）——
    主/腕 seg 分别下采样到各自 g×g 后拼接。batch 元素为 (task_id, frame) 元组。"""
    n_img = 2
    g = int(round((P // n_img) ** 0.5))
    parts = []
    for seg_key in ("seg", "wrist_seg"):
        m = None
        for _, fr in batch:
            s = fr.get(seg_key)
            if s is None:
                s = torch.zeros_like(fr["seg"])
            m = s if m is None else torch.cat([m, s], dim=0)
        small = F.interpolate(m, size=(g, g), mode="bilinear", align_corners=False)
        parts.append(small.reshape(small.size(0), -1))
    return torch.cat(parts, dim=1).to(device)                        # [B,P]


def _stack_proprio(frames):
    ps = [fr.get("proprio") for fr in frames]
    if any(p is None for p in ps):
        return None
    return torch.tensor(np.stack(ps), dtype=torch.float32, device="cuda")


def train(args):
    device = "cuda"
    random.seed(args.seed)
    _was_clean = TR.patch_status() == "clean"
    TR.ensure_patch(True)
    cfg = base.GenerateConfig(pretrained_checkpoint=CKPT,
                              model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    model_input_size = base.get_image_resize_size(cfg)
    sm, ss, dm, ds = _norm_consts(device)

    if not (args.load_pool and os.path.exists(args.pool_file)):
        raise SystemExit("B3 仅支持 --load-pool 冻结池（与 F0/B1/B2 同池）")
    pool = torch.load(args.pool_file, map_location=device, weights_only=False)
    print(f"[pool] 缓存命中 {sum(len(v) for v in pool.values())} 帧")

    renderer = ClothRendererOV(device=device, epsilon=args.epsilon,
                               init_std=args.init_std, seed=args.seed).to(device)
    t0_tex = renderer.orig_texture
    pproj = None
    if getattr(cfg, "use_proprio", False) and base.get_proprio_projector is not None:
        try:
            pproj = base.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
        except Exception as _e:
            print(f"[warn] proprio_projector 加载失败（按无 proprio 前向）：{_e}")

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
    print(f"[train-upa] 池 {len(all_frames)} 帧 × {args.iters} iters × batch "
          f"{args.batch}（con {args.lambda_con} / featL1 {args.lambda_1} / "
          f"text {args.lambda_text}；PAD 已按实现门降级移除；dual {args.dual_step}"
          f"步×ε{args.dual_eps:.4f}）")
    fwd_cnt = bwd_cnt = dual_cnt = 0
    diag_batch = [random.choice(all_frames) for _ in range(args.batch)] \
        if args.diag else None
    diag_uv = sample_uv_xform(args.uv_eot, device) if args.diag else None
    diag_probes = {0, args.iters // 2, args.iters - 1} if args.diag else set()
    tokenizer = processor.tokenizer
    import time as _t
    t_start = _t.time()

    def compose_frame(fr, tex_override, uv):
        rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                                  uv_xform=uv, tex_override=tex_override)
        chw = rgba.permute(0, 3, 1, 2)
        comp = torch.clamp(chw * fr["seg"] + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
        wrist = fr.get("wrist")
        if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
            wrgba, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                       model_rot=fr["rot"], uv_xform=uv,
                                       tex_override=tex_override)
            wchw = wrgba.permute(0, 3, 1, 2)
            wrist = torch.clamp(wchw * fr["wrist_seg"]
                                + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]), 0.0, 1.0)
        return comp, wrist

    def to_pv(comp, wrist):
        return base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=model_input_size,
            siglip_mean=sm, siglip_std=ss, dino_mean=dm, dino_std=ds,
            wrist_tensor=wrist).to(torch.bfloat16)

    def proj_tokens(pv):
        with base.autocast(dtype=torch.bfloat16):
            return model.projector(model.vision_backbone(pv)).float()

    for it in range(args.iters):
        batch = diag_batch if diag_batch is not None else \
            [random.choice(all_frames) for _ in range(args.batch)]
        param.grad = None
        uv_a = diag_uv if diag_uv is not None else \
            sample_uv_xform(getattr(args, "uv_eot", False), device)

        ids_list = [fr["text_inputs"]["input_ids"][0] for _, fr in batch]
        mask_list = [fr["text_inputs"]["attention_mask"][0] for _, fr in batch]
        maxlen = max(int(x.shape[0]) for x in ids_list)
        ids = torch.zeros((len(ids_list), maxlen), dtype=torch.long, device=device)
        tmask = torch.zeros((len(mask_list), maxlen), dtype=torch.long, device=device)
        for i, (ii, mm) in enumerate(zip(ids_list, mask_list)):
            ids[i, :ii.shape[0]] = ii
            tmask[i, :mm.shape[0]] = mm
        proprio_b = _stack_proprio([fr for _, fr in batch])

        # ---- clean = Plain 合成（θ=0），projector token（no_grad）----
        with torch.no_grad():
            clean_pvs = [to_pv(*compose_frame(fr, t0_tex, uv_a))
                         for _, fr in batch]
            clean_tok = proj_tokens(torch.cat(clean_pvs, dim=0)).detach()
            fwd_cnt += 1

        # ---- adv 合成（θ 纹理，带梯度）----
        comps, wrists = [], []
        for tid, fr in batch:
            ac, aw = compose_frame(fr, None, uv_a)
            comps.append(ac)
            wrists.append(aw)
        segs_main = torch.cat([fr["seg"] for _, fr in batch], dim=0)
        noncloth_main = 1.0 - segs_main

        # ---- 鲁棒内循环：辅助扰动（仅布面之外），内层最小化 con+L1 ----
        # 合成帧先 detach：内层反向只许触达 pert，不得把"最小化"梯度泄进 θ
        comps_d = [c.detach() for c in comps]
        wrists_d = [w.detach() if w is not None else None for w in wrists]
        pert = torch.zeros_like(segs_main).repeat(1, 3, 1, 1).requires_grad_(True)
        lr_p = 2.0 * args.dual_eps / max(args.dual_step, 1)
        for _ in range(args.dual_step):
            pert.grad = None
            pm = pert * noncloth_main
            pvs = torch.cat([to_pv(torch.clamp(c + pm[i:i + 1], 0.0, 1.0), w)
                             for i, (c, w) in enumerate(zip(comps_d, wrists_d))],
                            dim=0)
            tok_p = proj_tokens(pvs)
            dual_cnt += 1
            d_con = bi_contrastive(tok_p, clean_tok)
            d_l1 = F.l1_loss(tok_p, clean_tok)
            (d_con + d_l1).backward()
            with torch.no_grad():
                pert.add_(-lr_p * pert.grad.sign())
                pert.clamp_(-args.dual_eps, args.dual_eps)
        pert_d = pert.detach()
        adv_pvs = torch.cat([to_pv(torch.clamp(c + (pert_d * noncloth_main)[i:i + 1],
                                                0.0, 1.0), w)
                             for i, (c, w) in enumerate(zip(comps, wrists))], dim=0)

        # ---- adv projector token（带梯度）+ PSM 文本参考（no_grad 复刻前向）----
        adv_tok = proj_tokens(adv_pvs)
        fwd_cnt += 1
        with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
            txt_h = text_hidden_forward(model, ids, tmask, adv_pvs, proprio_b,
                                        pproj)
            fwd_cnt += 1
        vis_mask = vis_token_mask(batch, adv_tok.shape[1], device)
        if it == 0:
            line = (f"[layout] projector token P={adv_tok.shape[1]} "
                    f"（g={int(round(adv_tok.shape[1] ** 0.5))}） "
                    f"txt_hidden={tuple(txt_h.shape)} vis_mask mean="
                    f"{vis_mask.mean():.4f} max={vis_mask.max():.4f} "
                    f"非零批={int((vis_mask.sum(1) > 0).sum())}/{args.batch}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")

        # ---- 损失（ascent 目标：最大化 con/featL1，最小化 psm/正则）----
        con = bi_contrastive(adv_tok, clean_tok)
        featL1 = F.l1_loss(adv_tok, clean_tok)
        psm = psm_text_loss(model, adv_tok, txt_h, vis_mask, tokenizer)
        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = tv_loss(tex - t0_tex)
        objective = (args.lambda_con * con + args.lambda_1 * featL1
                     - args.lambda_text * psm
                     - args.lambda_ref * ref - args.lambda_tv * tv)
        if it in diag_probes:
            terms = {"con": args.lambda_con * con, "featL1": args.lambda_1 * featL1,
                     "psm(-)": -args.lambda_text * psm,
                     "ref(-)": -args.lambda_ref * ref, "tv(-)": -args.lambda_tv * tv}
            parts = []
            for name, term in terms.items():
                g = torch.autograd.grad(term, param, retain_graph=True)[0]
                parts.append(f"{name}={g.norm().item():.3e}")
            rms, mab, linf, tvv = texture_stats(renderer)
            line = f"[diag] it{it:05d} 分项θ梯度: " + " ".join(parts) + \
                   f" | texRMS={rms:.2f} linf={linf:.1f}"
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        objective.backward()
        bwd_cnt += 1
        with torch.no_grad():
            param.add_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        if args.diag or it % 50 == 0 or it == args.iters - 1:
            g = param.grad.norm().item() if param.grad is not None else 0.0
            line = (f"[train-upa] it{it:05d} obj={objective.item():+.4f} "
                    f"con={con.item():.4f} featL1={featL1.item():.4f} "
                    f"psm={psm.item():+.4f} ref={ref.item():.4f} "
                    f"tv={tv.item():.4f} g={g:.2e}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if (it + 1) % args.checkpoint_every == 0:
            line = save_ckpt(renderer, param, out_dir, it)
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        del adv_tok, clean_tok, txt_h, pert, adv_pvs

    wall = (_t.time() - t_start) / 60.0
    line = save_ckpt(renderer, param, out_dir, args.iters - 1)
    print(line, flush=True)
    print(f"[train-upa] 成本: 更新={args.iters} 前向={fwd_cnt} 反向={bwd_cnt} "
          f"内循环步={dual_cnt} 墙钟={wall:.1f}min batch={args.batch}", flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")
        f.write(f"[cost] updates={args.iters} fwd={fwd_cnt} bwd={bwd_cnt} "
                f"dual_steps={dual_cnt} wall_min={wall:.1f} batch={args.batch} "
                f"variant=no-PAD\n")
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
    p.add_argument("--lambda-con", type=float, default=1.0)
    p.add_argument("--lambda-text", type=float, default=0.5)
    p.add_argument("--lambda-1", type=float, default=0.1,
                   help="featL1 权重（trans.py 默认 0.1）")
    p.add_argument("--dual-step", type=int, default=10,
                   help="鲁棒内循环步数（trans.py --step 默认 10）")
    p.add_argument("--dual-eps", type=float, default=2.0 / 255.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--uv-eot", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--diag", action="store_true")
    p.add_argument("--tag", default="upa")
    args = ap.parse_args()
    if args.cmd == "train":
        args.pool_tasks = [int(t) for t in args.pool_tasks.split(",")]
        train(args)


if __name__ == "__main__":
    main()
