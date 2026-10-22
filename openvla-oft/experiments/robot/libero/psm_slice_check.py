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

# 位置级验证：复刻函数体取完整末层 hidden（长度 S），断言生产切片 == full[:, 1+pn:]
with torch.no_grad(), base.autocast(dtype=torch.bfloat16):
    le = model.get_input_embeddings()(ids)
    pj = model._process_vision_features(pv, le, use_film=False)
    pj = model._process_proprio_features(pj, proprio_t, pproj)
    embeds = torch.cat([le[:, :1, :], pj, le[:, 1:, :]], dim=1)
    am = torch.cat([tmask[:, :1],
                    torch.ones(tmask.shape[0], pj.shape[1], dtype=tmask.dtype,
                               device=tmask.device),
                    tmask[:, 1:]], dim=1)
    full = model.language_model(inputs_embeds=embeds, attention_mask=am,
                                output_hidden_states=True).hidden_states[-1]
print("[psm-check] 完整序列长度=%d（应=S=%d）；text rest 区 = [%d, %d)"
      % (full.shape[1], S, 1 + pn, S))
ok_len = seg.shape[1] == L - 1
ok_pos = bool(torch.allclose(seg.float(), full[:, 1 + pn:, :].float(),
                             atol=0.0, rtol=0.0))
tok = processor.tokenizer.convert_ids_to_tokens(
    [int(x) for x in ids[0, 1:4]]) + ["..."] + processor.tokenizer.convert_ids_to_tokens(
    [int(x) for x in ids[0, -2:]])
print("[psm-check] 指令文本首尾 tokens: %s" % tok)
print("[psm-check] 断言1 切片长度==L-1(33): %s" % ("PASS" if ok_len else "FAIL"))
print("[psm-check] 断言2 生产切片逐元素==full[:,1+pn:547]: %s"
      % ("PASS" if ok_pos else "FAIL"))
print("[psm-check] 断言3 pn==513(512 vision+1 proprio), S==547: %s"
      % ("PASS" if (pn == 513 and S == 547 and full.shape[1] == 547)
         else "FAIL"))
verdict = "PASS（修后切片正确：纯 text rest 区 [1+pn, S)，不含 BOS/proprio）" \
    if (ok_len and ok_pos and pn == 513 and S == 547) else "FAIL"
print("[psm-check] 判定: %s" % verdict)
