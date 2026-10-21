#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""psm_slice_check.py — B3/PSM 文本切片 off-by-one 核查（第13轮 第0步，老师 §6.2）
静态代数：序列 = [BOS(1)][proj pn][text rest L-1]，总长 S = pn+L。
生产代码取 [:, S-L:] = [:, pn:] → 首位是 proj 末 token（B3 有 proprio → proprio 末位；
无 proprio → 最后一个 vision token），漏 BOS。
本脚本：固定池帧实际前向，打印 pn/S/切片首尾 token 与 input_ids 的对齐检查。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import attack_oft as base            # noqa: E402
import transfer_pi2ov as TR          # noqa: E402
from attack_oft_cloth_upa import (   # noqa: E402
    CKPT, ClothRendererOV, _norm_consts, text_hidden_forward)

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
device = "cuda"

TR.ensure_patch(True)
cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
model = base.get_model(cfg)
processor = base.get_processor(cfg)
sm, ss, dm, ds = _norm_consts(device)
pool = torch.load(POOL, map_location=device, weights_only=False)
fr = next(f for tid in pool for f in pool[tid] if f["mvp"] is not None)
pproj = base.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
proprio_t = (torch.tensor(fr["proprio"], dtype=torch.float32, device=device)
             if fr.get("proprio") is not None else None)

renderer = ClothRendererOV(device=device, init_std=0.0, seed=7).to(device)
t0 = renderer.orig_texture
rgba, _ = renderer.render(fr["mvp"], (256, 256), model_rot=fr["rot"],
                          tex_override=t0)
comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                   + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
mis = base.get_image_resize_size(cfg)
pv = base._compose_oft_pixel_values(
    adv_rgb=comp, model_input_size=mis, siglip_mean=sm, siglip_std=ss,
    dino_mean=dm, dino_std=ds, wrist_tensor=fr.get("wrist")).to(torch.bfloat16)

ids = fr["text_inputs"]["input_ids"].to(device)
tmask = fr["text_inputs"]["attention_mask"].to(device)
L = int(ids.shape[1])
with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
    lang_emb = model.get_input_embeddings()(ids)
    proj = model._process_vision_features(pv, lang_emb, use_film=False)
    proj = model._process_proprio_features(proj, proprio_t, pproj)
pn = int(proj.shape[1])
with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
    seg = text_hidden_forward(model, ids, tmask, pv, proprio_t, pproj)
S = 1 + pn + (L - 1)
print("[psm-check] L(text ids)=%d  pn(vision+proprio)=%d  期望总长 S=1+pn+(L-1)=%d"
      % (L, pn, S))
print("[psm-check] 生产切片 text_hidden_forward 输出形状 = %s（长度 %d）"
      % (tuple(seg.shape), seg.shape[1]))
print("[psm-check] 生产切片覆盖位置 = [%d, %d)；正确 text 区 = [%d, %d)"
      % (S - L, S, 1 + pn, S))
print("[psm-check] 位置 %d 归属：%s（应为 text 区首，实际=%s）"
      % (S - L, "text rest 首" if S - L == 1 + pn else "proj 末 token ← off-by-one",
         "text" if S - L == 1 + pn else "proj末token"))
tok = processor.tokenizer.convert_ids_to_tokens(
    [int(x) for x in ids[0, 1:3]] + ["..."] + [int(x) for x in ids[0, -2:]])
print("[psm-check] 指令文本首尾 tokens: %s" % tok)
verdict = "PASS（切片正确）" if S - L == 1 + pn else \
    "FAIL（off-by-one：切片含 proj 末 token、漏 BOS；影响=PSM 参考表征稀释 1/%d" % (seg.shape[1],)
print("[psm-check] 判定: %s" % verdict)
