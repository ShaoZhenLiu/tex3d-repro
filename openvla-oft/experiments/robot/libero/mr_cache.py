#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mr_cache.py — 第13轮 阶段D 预备材料（只缓存，不启动 MR 训练）
为"匹配渲染 Plain 参考"预计算：r0f 池全部 296 帧在**同渲染器** tex_override=T0
下的合成 Plain 帧 → predict_action 末层 hidden（与 fr["clean_hidden"] 同口径）。
纯色 T0 下 UV-EoT 恒等（edpa_gap_check B≡A 已证）→ 无 UV 渲染即可，缓存键=帧索引
（与 tvfix 训练器 sampling_log 的 all_frames 序一致）。
产出：mr_plain_cache.pt（296 × [1, action_chunk, 4096] float32 CPU）。
若阶段 D 不触发，此缓存仅作 θ=0 域差的完整版记录，不进入任何比较。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import attack_oft as base                                  # noqa: E402
import transfer_pi2ov as TR                                # noqa: E402
import attack_oft_cloth as AC                              # noqa: E402
from attack_oft_cloth import (CKPT, ClothRendererOV,       # noqa: E402
                              _build_action_modules, _norm_consts)

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/r0f_pool.pt"
OUT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/oft_cloth/mr_plain_cache.pt"
device = "cuda"

TR.ensure_patch(True)
cfg = base.GenerateConfig(pretrained_checkpoint=CKPT, model_family="openvla")
model = base.get_model(cfg)
base.check_unnorm_key(cfg, model)
size = base.get_image_resize_size(cfg)
ah, pp, nap = _build_action_modules(cfg, model)
sm, ss, dm, ds = _norm_consts(device)
pool = torch.load(POOL, map_location=device, weights_only=False)

# F0 生产版 render 无 tex_override 参数；init_std=0 → θ=0 → texture_tensor()=T0（等效）
renderer = ClothRendererOV(device=device, init_std=0.0, seed=7).to(device)
t0 = renderer.orig_texture

cache = {}
idx = 0
with torch.no_grad():
    for tid in pool.keys():
        for fr in pool[tid]:
            if fr["mvp"] is None:
                continue
            rgba, _ = renderer.render(fr["mvp"], (256, 256),
                                      model_rot=fr["rot"])
            comp = torch.clamp(rgba.permute(0, 3, 1, 2) * fr["seg"]
                               + fr["bg"] * (1.0 - fr["seg"]), 0.0, 1.0)
            wrist = fr.get("wrist")
            if fr.get("wrist_mvp") is not None and fr.get("wrist_bg") is not None:
                w, _ = renderer.render(fr["wrist_mvp"], (256, 256),
                                       model_rot=fr["rot"])
                wrist = torch.clamp(w.permute(0, 3, 1, 2) * fr["wrist_seg"]
                                    + fr["wrist_bg"] * (1.0 - fr["wrist_seg"]),
                                    0.0, 1.0)
            pv = base._compose_oft_pixel_values(
                adv_rgb=comp, model_input_size=size, siglip_mean=sm,
                siglip_std=ss, dino_mean=dm, dino_std=ds,
                wrist_tensor=wrist).to(torch.bfloat16)
            _, h = model.predict_action(
                input_ids=fr["text_inputs"]["input_ids"],
                attention_mask=fr["text_inputs"]["attention_mask"],
                pixel_values=pv, unnorm_key=cfg.unnorm_key,
                proprio=fr["proprio"], proprio_projector=pp,
                action_head=ah, noisy_action_projector=nap,
                use_film=getattr(cfg, "use_film", False))
            cache[idx] = h.float().cpu()
            idx += 1
            if idx % 50 == 0:
                print("[mr-cache] %d 帧" % idx, flush=True)

torch.save({"frame_index_order": "all_frames = [(tid, fr) for tid in pool.keys() "
                                  "for fr in pool[tid] if fr['mvp'] is not None]",
            "hidden": cache}, OUT)
print("[mr-cache] 完成 %d 帧 → %s（%.1f MB）"
      % (idx, OUT, os.path.getsize(OUT) / 1e6))
