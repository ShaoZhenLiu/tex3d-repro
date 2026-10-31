# -*- coding: utf-8 -*-
"""adapters.py — 源特征接口与原生部署接口（计划§8.2 契约的生产实现）。

SourceAdapter：OFT 模型 + ClothRendererOV 子类（tex_override 支持量化合成路径），
             特征 = predict_action 第二返回值（action-token hidden [1,56,4096]，
             形状与 prompt 长度无关——20261031 audit 已核）。
NativeAdapter：MuJoCo 同状态停机换纹理（严格注入/还原，禁止静默 resize），
             双相机 offscreen 渲染，与生产评测同链。
两条路径共用：text_inputs / proprio / 归一化 / 预处理（同输入处理成对性）。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

D = "/data/shaozhen.liu"
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, f"{D}/code/tex3d/openvla-oft/experiments/robot/libero",
           f"{D}/code/tex3d/openvla/experiments/robot/libero", f"{D}/code/tex3d"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import attack_oft as base                   # noqa: E402
import attack_oft_cloth_tvfix_mr as M       # noqa: E402  生产配方（只读复用）
import transfer_pi2ov as TR                 # noqa: E402
import cloth_transfer_utils as ctu          # noqa: E402
from .texutil import png_delta_stats, png_to_tex, sha16   # noqa: E402,F401


# ---------------------------------------------------------------- 渲染器扩展
class RendererQ(M.ClothRendererOV):
    """生产渲染器 + 量化/外源纹理渲染。不改父类行为（tex_override=None 时逐行同父）。"""

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

    # ---- 部署量化路径（两条路径必须读同一烘焙结果，计划§4.4） ----
    def export_png(self):
        """真实部署 artifact：uint8 [H,W,3]（含垂直翻转 + round 量化）。"""
        return self.export_texture()

    @staticmethod
    def png_to_tex(png_uint8):
        """部署 PNG → 渲染方向 float 张量 [1,H,W,3]（与原生注入同一张图）。"""
        return png_to_tex(png_uint8)


# ---------------------------------------------------------------- 严格注入
def strict_inject(sim, tex_id, png_uint8):
    """注入 + fail-fast：尺寸必须精确一致（ctu 的静默 resize 分支禁止触达），
    GPU 上传必须成功（ctu 只 warning，这里复检并抛错）。返回还原句柄。"""
    native = getattr(sim.model, "_model", sim.model)
    h = int(native.tex_height[tex_id])
    w = int(native.tex_width[tex_id])
    if png_uint8.shape[:2] != (h, w):
        raise RuntimeError(
            f"native tex {w}x{h} != png {png_uint8.shape[1]}x{png_uint8.shape[0]}"
            "（禁止静默 resize，计划§4.3）")
    ri = ctu.inject_mujoco_texture(sim, tex_id, png_uint8)
    if not ctu._upload_mujoco_texture(sim, tex_id):
        raise RuntimeError("MuJoCo GPU 纹理上传失败（无可用 GL 上下文）")
    return ri


def render_native_dual(sim):
    """双相机 offscreen（与生产/评测同链：[::-1,::-1]）。"""
    a = sim.render(width=256, height=256, camera_name="agentview",
                   mode="offscreen")[::-1, ::-1]
    w = sim.render(width=256, height=256, camera_name="robot0_eye_in_hand",
                   mode="offscreen")[::-1, ::-1]
    return a, w


# ---------------------------------------------------------------- 源接口
class ResidualCtx:
    """模型 + 处理器 + 动作头 + 渲染器（一次装配，整个实验复用）。
    capture 用 no_grad 前向；训练用 adv_forward_grad（θ 可微）。"""

    def __init__(self, device="cuda", seed=7):
        self.device = device
        self._was_clean = TR.patch_status() == "clean"
        TR.ensure_patch(True)
        cfg = base.GenerateConfig(pretrained_checkpoint=M.CKPT,
                                  model_family="openvla")
        self.cfg = cfg
        self.model = base.get_model(cfg)
        self.processor = base.get_processor(cfg)
        base.check_unnorm_key(cfg, self.model)
        self.size = base.get_image_resize_size(cfg)
        self.ah, self.pp, self.nap = M._build_action_modules(cfg, self.model)
        self.sm, self.ss, self.dm, self.ds = M._norm_consts(device)
        self.use_proprio = bool(getattr(cfg, "use_proprio", False))
        self.use_wrist = bool(getattr(cfg, "num_images_in_input", 1) > 1)
        self.renderer = RendererQ(device=device, epsilon=64.0 / 255.0,
                                  init_std=0.0, seed=seed).to(device)
        self.t0_tex = self.renderer.orig_texture
        self.t0_png = np.array(Image.open(M.CLOTH_T0).convert("RGB"))

    # ---- 特征前向（no_grad；capture/评测用） ----
    def hidden_from_arrays(self, img_main, img_wrist, inputs, proprio):
        full_t = (torch.from_numpy(img_main.astype(np.float32) / 255.0)
                  ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        wrist_t = None
        if img_wrist is not None:
            wrist_t = (torch.from_numpy(img_wrist.astype(np.float32) / 255.0)
                       ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pv = base._compose_oft_pixel_values(
                adv_rgb=full_t, model_input_size=self.size,
                siglip_mean=self.sm, siglip_std=self.ss,
                dino_mean=self.dm, dino_std=self.ds,
                wrist_tensor=wrist_t).to(torch.bfloat16)
            with base.autocast(dtype=torch.bfloat16):
                _, h = self.model.predict_action(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=pv, unnorm_key=self.cfg.unnorm_key,
                    proprio=proprio, proprio_projector=self.pp,
                    action_head=self.ah, noisy_action_projector=self.nap,
                    use_film=getattr(self.cfg, "use_film", False))
        return h.float()

    def hidden_from_tensors(self, comp_main, comp_wrist, inputs, proprio):
        """合成路由：comp_* 为 [1,3,256,256] clamp 后合成图（no_grad 版）。"""
        with torch.no_grad():
            pv = base._compose_oft_pixel_values(
                adv_rgb=comp_main, model_input_size=self.size,
                siglip_mean=self.sm, siglip_std=self.ss,
                dino_mean=self.dm, dino_std=self.ds,
                wrist_tensor=comp_wrist).to(torch.bfloat16)
            with base.autocast(dtype=torch.bfloat16):
                _, h = self.model.predict_action(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=pv, unnorm_key=self.cfg.unnorm_key,
                    proprio=proprio, proprio_projector=self.pp,
                    action_head=self.ah, noisy_action_projector=self.nap,
                    use_film=getattr(self.cfg, "use_film", False))
        return h.float()

    def compose(self, quad, uv_xform=None, tex_override=None):
        """quad = capture 保存的合成四件套（mvp/rot/seg/bg + 腕四件套）。
        返回 (comp_main, comp_wrist) [1,3,256,256]。"""
        r = self.renderer
        adv_rgba, _ = r.render(quad["mvp"], (256, 256), model_rot=quad["rot"],
                               uv_xform=uv_xform, tex_override=tex_override)
        comp = torch.clamp(
            adv_rgba.permute(0, 3, 1, 2) * quad["seg"]
            + quad["bg"] * (1.0 - quad["seg"]), 0.0, 1.0)
        wrist_comp = quad.get("wrist")
        if quad.get("wrist_mvp") is not None and quad.get("wrist_bg") is not None:
            w_rgba, _ = r.render(quad["wrist_mvp"], (256, 256),
                                 model_rot=quad["rot"], uv_xform=uv_xform,
                                 tex_override=tex_override)
            wrist_comp = torch.clamp(
                w_rgba.permute(0, 3, 1, 2) * quad["wrist_seg"]
                + quad["wrist_bg"] * (1.0 - quad["wrist_seg"]), 0.0, 1.0)
        return comp, wrist_comp

    # ---- θ 管理 ----
    def load_theta(self, theta):
        """'zero' | tensor[Like renderer.adv_noise] | 文件路径。"""
        r = self.renderer
        with torch.no_grad():
            if isinstance(theta, str) and theta == "zero":
                r.adv_noise.zero_()
            elif isinstance(theta, str):
                r.adv_noise.copy_(torch.load(theta, map_location=self.device,
                                             weights_only=False).to(self.device))
            else:
                r.adv_noise.copy_(theta.to(self.device))
        return r

    # ---- 可微前向（one-step / 训练用；θ 保留梯度） ----
    def adv_forward_grad(self, quad, inputs, proprio, uv_xform=None):
        comp, wrist_comp = self.compose(quad, uv_xform=uv_xform)
        pv = base._compose_oft_pixel_values(
            adv_rgb=comp, model_input_size=self.size,
            siglip_mean=self.sm, siglip_std=self.ss,
            dino_mean=self.dm, dino_std=self.ds,
            wrist_tensor=wrist_comp).to(torch.bfloat16)
        with base.autocast(dtype=torch.bfloat16):
            _, h = self.model.predict_action(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=pv, unnorm_key=self.cfg.unnorm_key,
                proprio=proprio, proprio_projector=self.pp,
                action_head=self.ah, noisy_action_projector=self.nap,
                use_film=getattr(self.cfg, "use_film", False))
        return h.float()

    def cleanup(self):
        if self._was_clean:
            TR.ensure_patch(False)


# ---------------------------------------------------------------- 状态四件套采集
def capture_quad(env, obs, bid, device):
    """与生产池采集/t1_fix 完全同链的合成四件套 + 腕四件套 + text/proprio。"""
    img = base.get_libero_image(obs)
    bg_np = base._render_bg_without_target(env, bid, 256)
    seg = (np.abs(img.astype(np.int16) - bg_np.astype(np.int16)
                  ).max(axis=-1) > 10)
    mm, _, _ = base.get_target_model_matrix(env, TR.SEARCH)
    mvp = base.get_render_mvp_from_matrix(env, mm, resolution=(256, 256))
    quad = {
        "mvp": mvp, "rot": mm[:3, :3].detach().float().cpu(),
        "seg": torch.from_numpy(seg.astype(np.float32)).view(1, 1, 256, 256),
        "bg": (torch.from_numpy(bg_np.copy()).float() / 255.0
               ).permute(2, 0, 1).unsqueeze(0),
        "seg_area": float(seg.mean()),
        "native_main_px": img.copy(),
    }
    wnp = base._extract_wrist_image(obs, 256)
    quad["native_wrist_px"] = wnp.copy() if wnp is not None else None
    if wnp is not None:
        wbg = M._render_wrist_bg_without_target(env, bid, 256)
        if wbg is not None:
            wseg = (np.abs(wnp.astype(np.int16) - wbg.astype(np.int16)
                           ).max(axis=-1) > 10)
            quad["wrist_seg"] = torch.from_numpy(wseg.astype(np.float32)
                                                 ).view(1, 1, 256, 256)
            quad["wrist_bg"] = (torch.from_numpy(wbg.copy()).float() / 255.0
                                ).permute(2, 0, 1).unsqueeze(0)
            quad["wrist_mvp"] = M.get_wrist_mvp_from_matrix(
                env, mm, resolution=(256, 256)).cpu()
    return quad


def quad_to_dev(quad, device):
    out = dict(quad)
    for k in ("mvp", "rot", "seg", "bg", "wrist_seg", "wrist_bg", "wrist_mvp"):
        if out.get(k) is not None and torch.is_tensor(out[k]):
            out[k] = out[k].to(device)
    return out


def proprio_of(obs):
    return np.concatenate((
        obs["robot0_eef_pos"],
        base.quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"]))


def make_inputs(ctx, env_obs_img, task_description):
    pil = Image.fromarray(env_obs_img).resize((ctx.size, ctx.size))
    prompt = base._build_openvla_prompt(task_description)
    return ctx.processor(prompt, pil).to(ctx.device)
