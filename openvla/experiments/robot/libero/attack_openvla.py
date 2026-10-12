import os
import sys
import glob
import ctypes
import traceback
import time
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import tqdm
from torch.cuda.amp import autocast
from PIL import Image
from scipy.spatial.transform import Rotation as R
import draccus
import nvdiffrast.torch as dr
import trimesh
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

if 'MUJOCO_GL' not in os.environ:
    print("[INFO] Setting up OSMesa for CPU Rendering...")
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
else:
    print(f"[INFO] Using preconfigured MUJOCO_GL={os.environ['MUJOCO_GL']}")

try:
    ctypes.CDLL("libOSMesa.so")
except Exception as e:
    print(f"[WARNING] Failed to load libOSMesa.so: {e}")

PATH_TO_LIBERO_ROOT = "./libero-eval"
ASSET_ROOT_SCANNED  = "./libero-eval/libero/libero/assets/stable_scanned_objects"
ASSET_ROOT_HOPE     = "./libero-eval/libero/libero/assets/stable_hope_objects"

if PATH_TO_LIBERO_ROOT not in sys.path:
    sys.path.append(PATH_TO_LIBERO_ROOT)

from libero.libero import benchmark
import imageio


sys.path.append(str(Path(__file__).parent))
from libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    quat2axisangle, save_rollout_video,
)
sys.path.append(str(Path(__file__).parent.parent))

OPENVLA_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if OPENVLA_REPO_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_REPO_ROOT)

from openvla_utils import get_processor
from robot_utils import (
    DATE_TIME, get_action, get_image_resize_size, get_model,
    invert_gripper_action, normalize_gripper_action, set_seed_everywhere,
)


def _scanned(name, mesh_file=None, tex_file="texture.png"):
    base = f"{ASSET_ROOT_SCANNED}/{name}"
    return {
        "xml":     f"{base}/{name}.xml",
        "mesh":    f"{base}/{mesh_file or name + '.obj'}",
        "texture": f"{base}/{tex_file}",
    }


def _hope(name, mesh_file=None, tex_file="texture_map.png"):
    base = f"{ASSET_ROOT_HOPE}/{name}"
    return {
        "xml":     f"{base}/{name}.xml",
        "mesh":    f"{base}/{mesh_file or 'textured.obj'}",
        "texture": f"{base}/{tex_file}",
    }


OBJECTS = {
    "akita_black_bowl": {
        **_scanned("akita_black_bowl"),
        "search":     [["akita_black_bowl"], ["bowl"]],
        "task_suite": "libero_spatial",
        "task_id":    0,
    },
    "alphabet_soup": {
        **_hope("alphabet_soup", mesh_file="textured.obj"),
        "search":     [["alphabet_soup"], ["soup"]],
        "task_suite": "libero_object",
        "task_id":    0,
    },
    "cream_cheese": {
        **_hope("cream_cheese", mesh_file="cream_cheese.obj"),
        "search":     [["cream_cheese"], ["cheese"]],
        "task_suite": "libero_object",
        "task_id":    1,
    },
    "salad_dressing": {
        **_hope("salad_dressing", mesh_file="textured.obj"),
        "search":     [["salad_dressing"], ["dressing"]],
        "task_suite": "libero_object",
        "task_id":    2,
    },
    "bbq_sauce": {
        **_hope("bbq_sauce", mesh_file="bbq_sauce.obj"),
        "search":     [["bbq_sauce"], ["bbq"], ["sauce"]],
        "task_suite": "libero_object",
        "task_id":    3,
    },
    "ketchup": {
        **_hope("ketchup", mesh_file="textured.obj"),
        "search":     [["ketchup"]],
        "task_suite": "libero_object",
        "task_id":    4,
    },
    "tomato_sauce": {
        **_hope("tomato_sauce"),
        "search":     [["tomato_sauce"], ["tomato"]],
        "task_suite": "libero_object",
        "task_id":    5,
    },
    "butter": {
        **_hope("butter", mesh_file="butter.obj"),
        "search":     [["butter"]],
        "task_suite": "libero_object",
        "task_id":    6,
    },
    "milk": {
        **_hope("milk", mesh_file="textured.obj"),
        "search":     [["milk"]],
        "task_suite": "libero_object",
        "task_id":    7,
    },
    "chocolate_pudding": {
        **_hope("chocolate_pudding", mesh_file="textured.obj"),
        "search":     [["chocolate_pudding"], ["chocolate"], ["pudding"]],
        "task_suite": "libero_object",
        "task_id":    8,
    },
    "orange_juice": {
        **_hope("orange_juice", mesh_file="textured.obj"),
        "search":     [["orange_juice"], ["orange"], ["juice"]],
        "task_suite": "libero_object",
        "task_id":    9,
    },
}


def parse_mesh_scale(xml_path: str) -> List[float]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for mesh_elem in root.findall(".//mesh"):
        scale_str = mesh_elem.get("scale")
        if scale_str:
            vals = [float(v) for v in scale_str.strip().split()]
            return vals if len(vals) == 3 else [vals[0]] * 3
    return [1.0, 1.0, 1.0]


class DifferentiableRenderer(nn.Module):
    def __init__(self, mesh_path, orig_texture_path=None, device="cuda",
                 scale_xyz=None, pos_offset=None, epsilon=128.0 / 255.0):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.tex_h = 256
        self.tex_w = 256
        self.pos_offset = torch.tensor(
            pos_offset or [0.02, 0.01, 0.025], dtype=torch.float32, device=device
        )

        if scale_xyz is None:
            scale_xyz = [1.0, 1.0, 1.0]
            
        scale_arr = np.array(scale_xyz, dtype=np.float64)

        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception:
            print(f"[WARNING] Failed to load mesh from {mesh_path}, using dummy box.")
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.001])

        vertices = mesh.vertices * scale_arr[None, :]
        self.num_vertices = len(vertices)
        self.register_buffer("pos",   torch.from_numpy(vertices.astype(np.float32)).to(device))
        self.register_buffer("faces", torch.from_numpy(mesh.faces.astype(np.int32)).to(device))

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            uv = mesh.visual.uv.astype(np.float32)
            self.register_buffer("uv", torch.from_numpy(uv).to(device))
            if (hasattr(mesh.visual, "face_uv") and mesh.visual.face_uv is not None
                    and len(mesh.visual.face_uv) == len(mesh.faces)):
                self.register_buffer("uv_idx",
                    torch.from_numpy(mesh.visual.face_uv.astype(np.int32)).to(device))
            else:
                self.register_buffer("uv_idx", self.faces)
        else:
            v  = torch.from_numpy(vertices).to(device)
            uv = v[:, :2]
            if uv.numel() > 0:
                mn = uv.min(0, keepdim=True)[0]
                mx = uv.max(0, keepdim=True)[0]
                uv = (uv - mn) / (mx - mn + 1e-8)
            else:
                uv = torch.zeros((len(vertices), 2), device=device)
            self.register_buffer("uv",     uv.float())
            self.register_buffer("uv_idx", self.faces)

        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            mesh.fix_normals()
        vn = np.array(getattr(mesh, "vertex_normals", np.zeros_like(vertices)))
        if vn.sum() == 0:
            vn[:, 2] = 1.0
        vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
        self.register_buffer("vn", torch.from_numpy(vn.astype(np.float32)).to(device))

        self.glctx = dr.RasterizeCudaContext()

        if orig_texture_path and os.path.exists(orig_texture_path):
            img_raw = Image.open(orig_texture_path).convert("RGB")
            self.tex_w, self.tex_h = img_raw.size
            img = img_raw.transpose(Image.FLIP_TOP_BOTTOM)
            tex = torch.from_numpy(np.array(img)).float() / 255.0
            self.register_buffer("orig_texture",
                tex.unsqueeze(0).to(device).contiguous())
        else:
            fb = (torch.tensor([0.45, 0.45, 0.45], device=device)
                    .view(1, 1, 1, 3)
                    .expand(1, self.tex_h, self.tex_w, 3)
                    .contiguous())
            self.register_buffer("orig_texture", fb)

        orig_vc = self._sample_uv_texture_at_vertices()
        self.register_buffer("orig_vertex_colors", orig_vc)
        self.adv_noise = nn.Parameter(
            torch.zeros(self.num_vertices, 3, dtype=torch.float32, device=device)
        )
        self.light_dir = F.normalize(
            torch.tensor([0.2, 0.2, 1.0], device=device), dim=0
        )
        self.register_buffer("calib_scale", torch.ones(3, device=device))
        self.register_buffer("calib_bias",  torch.zeros(3, device=device))
        self.register_buffer("calib_gamma", torch.ones(3, device=device))
        self.ambient_strength = 0.42
        self.diffuse_strength = 0.48
        self.specular_strength = 0.05
        self.specular_shininess = 24.0
        self.shadow_strength = 0.15
        self.shadow_gamma = 1.8
        self.min_light = 0.16

    def _sample_uv_texture_at_vertices(self):
        with torch.no_grad():
            if len(self.uv) == self.num_vertices:
                uv_verts = self.uv
            else:
                uv_sum   = torch.zeros(self.num_vertices, 2, device=self.device)
                uv_count = torch.zeros(self.num_vertices, 1, device=self.device)
                face_vi  = self.faces.long()
                face_ui  = self.uv_idx.long()
                ones_f   = torch.ones(len(face_vi), 1, device=self.device)
                for local in range(3):
                    vi = face_vi[:, local]
                    ui = face_ui[:, local]
                    uv_sum.scatter_add_(0, vi.unsqueeze(1).expand(-1, 2), self.uv[ui])
                    uv_count.scatter_add_(0, vi.unsqueeze(1), ones_f)
                uv_verts = uv_sum / uv_count.clamp_min(1)

            uv_query = uv_verts.unsqueeze(0).unsqueeze(0)
            colors   = dr.texture(self.orig_texture.contiguous(),
                                  uv_query.contiguous(), filter_mode="linear")
            return colors.squeeze(0).squeeze(0).contiguous()

    def get_texture_param(self):
        return self.adv_noise

    def reset_texture(self):
        with torch.no_grad():
            self.adv_noise.data.fill_(0.0)

    def calibrate_lighting(self, mvp, mujoco_clean_rgb, ema: float = 0.0, model_rot=None):
        with torch.no_grad():
            H, W = mujoco_clean_rgb.shape[-2], mujoco_clean_rgb.shape[-1]
            _, clean_lit, mask = self.render(mvp, resolution=(H, W), return_clean=True,
                                             model_rot=model_rot)
            m      = (mask.squeeze(0).squeeze(-1) > 0.5)
            pred   = clean_lit.squeeze(0)
            target = mujoco_clean_rgb.squeeze(0).permute(1, 2, 0)

            if m.sum() < 50:
                print("[WARN] 光照校准: 目标物体在画面里的像素太少，跳过校准，沿用默认光照参数。")
                return

            scales, biases, gammas = [], [], []
            for c in range(3):
                x = pred[..., c][m].clamp(1e-4, 1.0 - 1e-4)
                y = target[..., c][m]
                gamma_grid = torch.tensor(
                    [1.00, 1.15, 1.30, 1.45, 1.60], device=x.device, dtype=x.dtype
                )
                best_err = None
                best_a = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                best_b = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                best_g = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                for g in gamma_grid:
                    xg = x.pow(g)
                    x_mean, y_mean = xg.mean(), y.mean()
                    denom = ((xg - x_mean) ** 2).sum().clamp_min(1e-6)
                    a = (((xg - x_mean) * (y - y_mean)).sum() / denom).clamp(0.85, 1.35)
                    b = (y_mean - a * x_mean).clamp(-0.12, 0.0)
                    y_hat = torch.clamp(a * xg + b, 0.0, 1.0)
                    err = F.mse_loss(y_hat, y)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        best_g = g
                scales.append(best_a)
                biases.append(best_b)
                gammas.append(best_g)

            new_scale = torch.stack(scales).to(self.calib_scale.device)
            new_bias = torch.stack(biases).to(self.calib_bias.device)
            new_gamma = torch.stack(gammas).to(self.calib_gamma.device)

            if ema > 0.0:
                m_ = float(max(0.0, min(0.999, ema)))
                self.calib_scale = m_ * self.calib_scale + (1.0 - m_) * new_scale
                self.calib_bias = m_ * self.calib_bias + (1.0 - m_) * new_bias
                self.calib_gamma = m_ * self.calib_gamma + (1.0 - m_) * new_gamma
            else:
                self.calib_scale = new_scale
                self.calib_bias = new_bias
                self.calib_gamma = new_gamma
            self.calib_bias = self.calib_bias.clamp(-0.12, 0.0)

    def render(self, mvp, resolution=(256, 256), return_clean=False, model_rot=None):
        pos      = self.pos + self.pos_offset
        pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())

        rast, _ = dr.rasterize(self.glctx, pos_clip.unsqueeze(0),
                                self.faces, resolution=resolution)

        noise_param = torch.tanh(self.adv_noise) * self.epsilon
        adv_vc_raw  = self.orig_vertex_colors + noise_param
        adv_vc      = adv_vc_raw + (adv_vc_raw.clamp(0, 1) - adv_vc_raw).detach()

        clean_color, _ = dr.interpolate(
            self.orig_vertex_colors.unsqueeze(0).contiguous(), rast, self.faces)
        adv_color, _   = dr.interpolate(
            adv_vc.unsqueeze(0).contiguous(), rast, self.faces)

        vn_world = (model_rot @ self.vn.T).T if model_rot is not None else self.vn
        vn_interp, _ = dr.interpolate(vn_world.unsqueeze(0).contiguous(), rast, self.faces)
        nrm = F.normalize(vn_interp, dim=-1, eps=1e-6)

        l = self.light_dir.view(1, 1, 1, 3)
        v = F.normalize(torch.tensor([0.0, 0.0, 1.0], device=nrm.device), dim=0).view(1, 1, 1, 3)
        h = F.normalize(l + v, dim=-1, eps=1e-6)
        ndotl = torch.clamp((nrm * l).sum(dim=-1, keepdim=True), 0.0, 1.0)
        ndoth = torch.clamp((nrm * h).sum(dim=-1, keepdim=True), 0.0, 1.0)
        spec  = ndoth.pow(self.specular_shininess)
        light = (
            self.ambient_strength
            + self.diffuse_strength * ndotl
            + self.specular_strength * spec
        )
        shadow_mask = (1.0 - ndotl).clamp(0.0, 1.0).pow(self.shadow_gamma)
        light = light * (1.0 - self.shadow_strength * shadow_mask)
        light = torch.clamp(light, self.min_light, 1.35)

        clean_shaded = torch.clamp(clean_color * light, 0.0, 1.0)
        adv_shaded   = torch.clamp(adv_color   * light, 0.0, 1.0)

        calib_scale = self.calib_scale.view(1, 1, 1, 3)
        calib_bias  = self.calib_bias.view(1, 1, 1, 3)
        calib_gamma = self.calib_gamma.view(1, 1, 1, 3)
        clean_base = torch.clamp(clean_shaded, 1e-6, 1.0).pow(calib_gamma)
        clean_lit  = torch.clamp(clean_base * calib_scale + calib_bias, 0.0, 1.0)
        adv_base   = torch.clamp(adv_shaded,  1e-6, 1.0).pow(calib_gamma)
        adv_lit    = torch.clamp(adv_base   * calib_scale + calib_bias, 0.0, 1.0)

        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            return adv_lit, clean_lit, mask
        return adv_lit, mask

    def get_baked_adv_texture(self):
        """UV-space 光栅化：把对抗顶点颜色完整 bake 到 UV atlas，覆盖全 UV 岛。"""
        with torch.no_grad():
            noise_param = torch.tanh(self.adv_noise) * self.epsilon
            adv_vc = (self.orig_vertex_colors + noise_param).clamp(0, 1)

            H, W   = self.tex_h, self.tex_w
            n_uv   = self.uv.shape[0]
            device = self.device

            uv_to_pos = torch.zeros(n_uv, dtype=torch.long, device=device)
            face_vi = self.faces.long()
            face_ui = self.uv_idx.long()
            for c in range(3):
                uv_to_pos[face_ui[:, c]] = face_vi[:, c]
            uv_vc = adv_vc[uv_to_pos]

            uv = self.uv
            uv_clip = torch.zeros(n_uv, 4, device=device)
            uv_clip[:, 0] = 2.0 * uv[:, 0] - 1.0
            uv_clip[:, 1] = 2.0 * uv[:, 1] - 1.0
            uv_clip[:, 3] = 1.0

            rast_uv, _ = dr.rasterize(self.glctx, uv_clip.unsqueeze(0),
                                       self.uv_idx.int(), resolution=[H, W])

            baked_colors, _ = dr.interpolate(
                uv_vc.unsqueeze(0).contiguous(), rast_uv, self.uv_idx.int()
            )

            mask = (rast_uv[..., 3] > 0).unsqueeze(-1).float()
            baked = mask * baked_colors + (1.0 - mask) * self.orig_texture

            return torch.flip(baked, dims=[1])

    def bake_vertex_colors_to_texture(self, resolution=(256, 256)):
        return self.get_baked_adv_texture()


def get_obj_name(mod, idx, obj_type):
    short_type_map = {
        "texture": "tex", "material": "mat",
        "geom": "geom", "body": "body", "camera": "cam",
    }
    short_type = short_type_map.get(obj_type, obj_type)
    func_name = f"{short_type}_id2name"
    if hasattr(mod, func_name):
        try:
            return getattr(mod, func_name)(idx)
        except Exception:
            pass
    if hasattr(mod, "id2name"):
        try:
            return mod.id2name(idx, obj_type)
        except Exception:
            pass
    return None


def get_target_model_matrix(env, search_keywords_list: List[List[str]]):
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    target_body_id = -1
    found_name = None

    for keywords in search_keywords_list:
        if hasattr(sim.model, "nbody"):
            for i in range(sim.model.nbody):
                name = get_obj_name(sim.model, i, "body")
                if not name or "vis" in name or "site" in name:
                    continue
                if all(k in name for k in keywords):
                    target_body_id = i
                    found_name = name
                    break
        if target_body_id != -1:
            break

    if target_body_id == -1:
        print(f"[WARNING] Could not find target body for {search_keywords_list}. Using fallback matrix.")
        fallback = torch.eye(4).cuda()
        fallback[2, 3] = 0.85
        return fallback, -1, None

    pos  = sim.data.body_xpos[target_body_id]
    quat = sim.data.body_xquat[target_body_id]
    rot  = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat  = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    mat[:3,  3] = pos
    return torch.from_numpy(mat).cuda(), target_body_id, found_name


def get_render_mvp_from_matrix(
    env,
    model_matrix,
    resolution=(256, 256),
    proj_flip_x: float = -1.0,
    proj_flip_y: float = -1.0,
):
    sim = env.sim if not hasattr(env, "unwrapped") else env.unwrapped.sim
    W, H = resolution

    cam_id = 0
    try:
        cam_id = sim.model.camera_name2id("agentview")
    except Exception:
        print("[WARNING] Camera 'agentview' not found, using camera 0.")

    cam_pos  = sim.data.cam_xpos[cam_id]
    cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    view_rot = cam_xmat.T
    view_mtx = np.eye(4, dtype=np.float32)
    view_mtx[:3, :3] = view_rot
    view_mtx[:3,  3] = -(view_rot @ cam_pos)

    fovy_deg = float(sim.model.cam_fovy[cam_id])
    aspect   = float(W) / float(H)
    near, far = 0.01, 10.0
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)

    proj_mtx = np.zeros((4, 4), dtype=np.float32)
    proj_mtx[0, 0] = proj_flip_x * f / aspect
    proj_mtx[1, 1] = proj_flip_y * f
    proj_mtx[2, 2] = (far + near) / (near - far)
    proj_mtx[2, 3] = (2 * far * near) / (near - far)
    proj_mtx[3, 2] = -1.0

    return (
        torch.from_numpy(proj_mtx).cuda()
        @ torch.from_numpy(view_mtx).cuda()
        @ model_matrix
    )


def _render_bg_without_target(env, body_id: int, resolution: int):
    """临时把目标 body 所有 geom 的 alpha 设为 0，渲一帧无目标物体的背景，渲完立刻还原。
    不影响物理状态/obs，MuJoCo 原生视频/物理推进完全不受影响。"""
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    geom_ids = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == body_id]
    if not geom_ids:
        return None
    orig_alpha = [float(sim.model.geom_rgba[g, 3]) for g in geom_ids]
    try:
        for g in geom_ids:
            sim.model.geom_rgba[g, 3] = 0.0
        img = sim.render(width=resolution, height=resolution, camera_name="agentview", mode="offscreen")
        img = img[::-1, ::-1]
    finally:
        for g, a in zip(geom_ids, orig_alpha):
            sim.model.geom_rgba[g, 3] = a
    return img


def _composite(adv_rgba, mask, bg_tensor):
    mask_t = mask.permute(0, 3, 1, 2)
    return torch.clamp(
        adv_rgba.permute(0, 3, 1, 2) * mask_t + bg_tensor * (1 - mask_t),
        0.0, 1.0,
    )


def render_and_composite(renderer, bg_tensor, mvp, resolution=(256, 256), model_rot=None):
    if mvp is None:
        return bg_tensor
    adv_rgba, mask = renderer.render(mvp, resolution=resolution, model_rot=model_rot)
    return _composite(adv_rgba, mask, bg_tensor)
def get_attack_loss(logits, clean_labels):
    ACTION_START = 31744
    ACTION_END   = 32000
    NUM_BINS     = 256

    if logits.shape[1] > clean_labels.shape[1]:
        logits = logits[:, -clean_labels.shape[1]:, :]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = clean_labels[:, 1:].contiguous().to(logits.device)

    action_mask = (
        (shift_labels >= ACTION_START) &
        (shift_labels <  ACTION_END)   &
        (shift_labels != -100)
    )
    if not action_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits  = shift_logits[action_mask]
    valid_labels  = shift_labels[action_mask]
    action_logits = valid_logits[:, ACTION_START:ACTION_END]
    correct_bins  = valid_labels - ACTION_START
    opposite_bins = (NUM_BINS - 1 - correct_bins)

    return F.cross_entropy(action_logits, opposite_bins)


def decode_action_from_generated_ids(model, generated_ids, unnorm_key=None):
    action_dim = model.get_action_dim(unnorm_key)
    predicted_action_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
def _build_adv_samples(renderer, fdata, RENDER_RES: int) -> List[torch.Tensor]:
    adv_rgba, mask = renderer.render(fdata["mvp"], resolution=(RENDER_RES, RENDER_RES),
                                     model_rot=fdata.get("model_rot"))
    bg = fdata.get("bg_tensor_no_obj") if fdata.get("bg_tensor_no_obj") is not None else fdata["bg_tensor"]
    return [_composite(adv_rgba, mask, bg)]


def train_adversarial_texture(
    cfg, model, processor, renderer,
    initial_obs_state, task, task_description,
    save_dir, episode_idx,
    search_keywords_list: List[List[str]],
    xml_path,
    num_iters=20,
    init_states=None,
):
    print(f"[ATTACK] Training Ep {episode_idx} | {cfg.num_frames_to_attack}-Frame Optimization...")
    os.makedirs(save_dir, exist_ok=True)

    RENDER_RES       = 256
    model_input_size = get_image_resize_size(cfg)
    device           = model.device

    siglip_mean = torch.tensor([0.5,   0.5,   0.5  ], device=device).view(1, 3, 1, 1)
    siglip_std  = torch.tensor([0.5,   0.5,   0.5  ], device=device).view(1, 3, 1, 1)
    dino_mean   = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dino_std    = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    _avail = init_states if (init_states is not None and len(init_states) > 0) else [initial_obs_state]
    n_train_states   = min(cfg.num_train_init_states, len(_avail))
    train_states     = [_avail[i] for i in range(n_train_states)]
    frames_per_state = max(1, cfg.train_frames_per_state)

    frame_data   = []
    calib_done   = False
    calib_count  = 0

    for _si, _train_state in enumerate(train_states):
        env, _ = get_libero_env(task, cfg.model_family, resolution=RENDER_RES)
        env.reset()
        obs = env.set_init_state(_train_state)
        env.env.sim.forward()

        if cfg.frame_collect_with_policy:
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        if cfg.collect_grasp_frames:
            state_frames = deque(maxlen=cfg.grasp_pre_frames)
            loop_max     = cfg.grasp_max_steps
        else:
            state_frames = []
            loop_max     = frames_per_state

        grasp_detected = False
        post_count     = 0

        for t in range(loop_max):
            img_np = get_libero_image(obs, RENDER_RES)
            bg_tensor = (torch.from_numpy(img_np).float().to(device) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0)

            model_matrix, body_id, found_name = get_target_model_matrix(env, search_keywords_list)
            mvp = (get_render_mvp_from_matrix(env, model_matrix, resolution=(RENDER_RES, RENDER_RES))
                   if body_id != -1 else None)
            if body_id != -1:
                print(f"  [状态{_si} 步{t}] 目标 body: '{found_name}'")
            else:
                print(f"  [状态{_si} 步{t}] 未找到目标 body")

            _bg_no_obj_np = _render_bg_without_target(env, body_id, RENDER_RES) if body_id != -1 else None
            bg_tensor_no_obj = (
                (torch.from_numpy(_bg_no_obj_np.copy()).float().to(device) / 255.0
                 ).permute(2, 0, 1).unsqueeze(0)
                if _bg_no_obj_np is not None else None
            )

            if mvp is not None and not calib_done and calib_count < cfg.photometric_calib_frames:
                renderer.calibrate_lighting(mvp, bg_tensor, ema=(0.8 if calib_count > 0 else 0.0),
                                            model_rot=model_matrix[:3, :3])
                calib_count += 1
                if calib_count >= cfg.photometric_calib_frames:
                    calib_done = True

            image_pil    = Image.fromarray(img_np).resize((model_input_size, model_input_size))
            prompt       = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            clean_inputs = processor(prompt, images=image_pil).to(device)
            if "pixel_values" in clean_inputs:
                clean_inputs["pixel_values"] = clean_inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                with autocast(dtype=torch.bfloat16):
                    clean_output_ids = model.generate(
                        **clean_inputs, max_new_tokens=7,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                    clean_224 = F.interpolate(bg_tensor, size=(model_input_size, model_input_size),
                                               mode="bilinear", align_corners=False)
                    clean_pv  = torch.cat(
                        [(clean_224 - siglip_mean) / siglip_std,
                         (clean_224 - dino_mean)   / dino_std], dim=1
                    ).to(torch.bfloat16)
                    clean_fwd = model(
                        input_ids=clean_output_ids,
                        attention_mask=torch.ones_like(clean_output_ids),
                        pixel_values=clean_pv,
                        output_hidden_states=True,
                    )
                    clean_hidden = clean_fwd.hidden_states[-1].detach()
                    clean_action = decode_action_from_generated_ids(model, clean_output_ids, cfg.unnorm_key)
            executed_action = None
            if cfg.frame_collect_with_policy:
                executed_action = normalize_gripper_action(clean_action.copy(), binarize=True)
                if cfg.model_family == "openvla":
                    executed_action = invert_gripper_action(executed_action)


            frame_entry = {
                "bg_tensor":        bg_tensor,
                "bg_tensor_no_obj": bg_tensor_no_obj,
                "mvp":              mvp,
                "model_rot":        model_matrix[:3, :3] if model_matrix is not None else None,
                "clean_output_ids": clean_output_ids,
                "prompt_ids":       clean_inputs["input_ids"],
                "clean_action":     clean_action,
                "executed_action":  executed_action,
                "clean_hidden":     clean_hidden,
                "siglip_mean":      siglip_mean,
                "siglip_std":       siglip_std,
                "dino_mean":        dino_mean,
                "dino_std":         dino_std,
                "model_input_size": model_input_size,
            }

            if cfg.collect_grasp_frames:
                _gqpos = obs.get("robot0_gripper_qpos", None)
                gripper_close = (cfg.frame_collect_with_policy
                                 and _gqpos is not None
                                 and float(np.max(_gqpos)) < cfg.grasp_qpos_threshold)
                if not grasp_detected:
                    state_frames.append(frame_entry)
                    if gripper_close and len(state_frames) >= cfg.grasp_pre_frames:
                        grasp_detected = True
                else:
                    if post_count < cfg.grasp_post_frames:
                        state_frames.append(frame_entry)
                        post_count += 1
                    if post_count >= cfg.grasp_post_frames:
                        break
            else:
                state_frames.append(frame_entry)
                if cfg.frame_collect_with_policy and float(executed_action[-1]) > 0:
                    print(f"  [状态{_si} 步{t}] 夹爪闭合，停止采帧（{t+1} 帧）")
                    break

            if cfg.frame_collect_with_policy:
                obs, _, _, _ = env.step(executed_action.tolist())
            else:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        if cfg.collect_grasp_frames:
            state_frames = list(state_frames)
            if not grasp_detected:
                print(f"[ATTACK] 状态{_si}: 未检测到夹爪闭合，使用末尾 {len(state_frames)} 帧")
            print(f"[ATTACK] 状态{_si} grasp-window 帧数 = {len(state_frames)}")

        frame_data.extend(state_frames)
        env.close()

    frame_pool = frame_data
    pool_size  = len(frame_pool)
    batch_size = min(cfg.num_frames_to_attack, pool_size)


    pgd_step = cfg.attack_lr
    loss_history = []

    grad_log_path = os.path.join(save_dir, f"Ep{episode_idx}_gradient_log.txt")
    with open(grad_log_path, "w") as f:
        f.write("Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | LR\n")

    def _bake_and_inject_eval_texture(tag):
        with torch.no_grad():
            baked_tex = renderer.get_baked_adv_texture()
        tex_path = os.path.join(save_dir, f"Ep{episode_idx}_LiveTexture_{tag}.png")
        Image.fromarray(
            (baked_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        ).save(tex_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        tex_name = f"tex-{cfg.object_name}"
        mat_name = f"mat-{cfg.object_name}"
        for asset_elem in root.findall("asset"):
            for tex_elem in asset_elem.findall("texture"):
                if tex_elem.get("name") == tex_name:
                    tex_elem.set("file", str(Path(tex_path).resolve()))
                    tex_elem.set("type", "2d")
                    break
        for mat_elem in root.findall(".//material"):
            if mat_elem.get("name") == mat_name:
                mat_elem.set("texuniform", "false")
        tree.write(xml_path)
        return tex_path

    def _run_live_test(i):
        tex_path = _bake_and_inject_eval_texture(f"iter{i:04d}")

        noise_pt_path = os.path.join(save_dir, f"Ep{episode_idx}_noise_iter{i:04d}.pt")
        torch.save(renderer.adv_noise.data.cpu(), noise_pt_path)

        if init_states is not None and len(init_states) > 1:
            chosen_state = init_states[np.random.randint(len(init_states))]
        else:
            chosen_state = initial_obs_state

        test_env, _ = get_libero_env(
            task, cfg.model_family, resolution=cfg.live_test_resolution
        )
        test_env.reset()
        test_obs = test_env.set_init_state(chosen_state)
        test_env.env.sim.forward()

        t2, max_steps, done2 = 0, cfg.live_test_max_steps, False
        frames = []
        while t2 < max_steps + cfg.num_steps_wait:
            if t2 < cfg.num_steps_wait:
                test_obs, _, _, _ = test_env.step(get_libero_dummy_action(cfg.model_family))
                t2 += 1
                continue

            img_np = get_libero_image(test_obs, cfg.live_test_resolution)
            bg_tensor = (torch.from_numpy(img_np).float().to(device) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0)

            model_matrix, body_id, _ = get_target_model_matrix(test_env, search_keywords_list)
            mvp = (get_render_mvp_from_matrix(
                test_env, model_matrix,
                resolution=(cfg.live_test_resolution, cfg.live_test_resolution),
            ) if body_id != -1 else None)

            _bg_no_obj = _render_bg_without_target(test_env, body_id, cfg.live_test_resolution) if body_id != -1 else None
            composite_bg = (
                (torch.from_numpy(_bg_no_obj.copy()).float().to(device) / 255.0
                 ).permute(2, 0, 1).unsqueeze(0)
                if _bg_no_obj is not None else bg_tensor
            )

            with torch.no_grad():
                if mvp is not None:
                    adv_rgba, mask = renderer.render(
                        mvp, resolution=(cfg.live_test_resolution, cfg.live_test_resolution),
                        model_rot=model_matrix[:3, :3]
                    )
                    composited = _composite(adv_rgba, mask, composite_bg)
                else:
                    composited = bg_tensor

            perceived_np = (
                composited[0].permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            frames.append(img_np)

            img_model = np.array(
                Image.fromarray(perceived_np).resize((model_input_size, model_input_size))
            )
            observation = {
                "full_image": img_model,
                "state": np.concatenate((
                    test_obs["robot0_eef_pos"],
                    quat2axisangle(test_obs["robot0_eef_quat"]),
                    test_obs["robot0_gripper_qpos"],
                )),
            }
            act = get_action(cfg, model, observation, task_description, processor=processor)
            act = normalize_gripper_action(act, binarize=True)
            if cfg.model_family == "openvla":
                act = invert_gripper_action(act)

            test_obs, _, done2, _ = test_env.step(act.tolist())
            if done2:
                break
            t2 += 1
        test_env.close()

        video_path = os.path.join(
            save_dir, f"Ep{episode_idx}_LiveTest_iter{i:04d}_success={done2}.mp4"
        )
        writer = imageio.get_writer(video_path, fps=30)
        for fr in frames:
            writer.append_data(fr)
        writer.close()

        print(f"[LIVE-TEST] iter={i} success={done2} | video={video_path}")
        return done2

    iterator = tqdm.tqdm(range(num_iters), desc="Optimizing", leave=False)
    for i in iterator:
        renderer.adv_noise.grad = None
        avg_total = avg_action = avg_feat = 0.0
        valid = 0

        batch_idx  = np.random.choice(pool_size, batch_size, replace=False)
        batch_frames = [frame_pool[j] for j in batch_idx]

        for t, fdata in enumerate(batch_frames):
            if fdata["mvp"] is None:
                continue

            w_t = torch.tensor(1.0 / batch_size, device=device)

            adv_samples = _build_adv_samples(renderer, fdata, RENDER_RES)

            sample_action_losses = []
            sample_feat_losses   = []

            for sample_idx, adv_img in enumerate(adv_samples):
                adv_224 = F.interpolate(
                    adv_img,
                    size=(fdata["model_input_size"], fdata["model_input_size"]),
                    mode="bilinear", align_corners=False,
                )

                pv = torch.cat(
                    [(adv_224 - fdata["siglip_mean"]) / fdata["siglip_std"],
                     (adv_224 - fdata["dino_mean"])   / fdata["dino_std"]], dim=1
                )

                with autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=fdata["clean_output_ids"],
                        attention_mask=torch.ones_like(fdata["clean_output_ids"]),
                        pixel_values=pv.to(torch.bfloat16),
                        output_hidden_states=True,
                    )

                sample_action_losses.append(get_attack_loss(outputs.logits, fdata["clean_output_ids"]))
                sample_feat_losses.append(-F.mse_loss(outputs.hidden_states[-1], fdata["clean_hidden"]))

            loss_action  = torch.stack(sample_action_losses).mean()
            loss_feature = torch.stack(sample_feat_losses).mean()

            frame_loss = w_t * (
                cfg.alpha_action  * loss_action +
                cfg.alpha_feature * loss_feature
            )
            frame_loss.backward()

            avg_total  += frame_loss.item()
            avg_action += loss_action.item()
            avg_feat   += loss_feature.item()
            valid += 1

        if valid > 0:
            avg_total  /= valid
            avg_action /= valid
            avg_feat   /= valid
        loss_history.append(avg_total)

        grad   = renderer.adv_noise.grad
        g_norm = grad.norm().item() if grad is not None else 0.0

        with open(grad_log_path, "a") as f:
            f.write(f"{i:02d} | {avg_total:.6f} | {avg_action:.6f} | "
                    f"{avg_feat:.6f} | {g_norm:.6e} | {pgd_step:.6e}\n")

        with torch.no_grad():
            renderer.adv_noise.data -= pgd_step * grad.sign()
        iterator.set_postfix(
            act=f"{avg_action:.4f}",
            feat=f"{avg_feat:.4f}",
            gnorm=f"{g_norm:.4f}",
        )

        if (cfg.live_test_enabled and cfg.live_test_every_n_iters > 0
                and (i + 1) % cfg.live_test_every_n_iters == 0):
            _run_live_test(i + 1)

    if cfg.save_attack_artifacts:
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            os.path.join(save_dir, f"Ep{episode_idx}_Vertex_Noise.pt"),
        )
        Image.fromarray(
            (renderer.get_baked_adv_texture().squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(save_dir, f"Ep{episode_idx}_UV_Map.png"))
        np.save(
            os.path.join(save_dir, f"Ep{episode_idx}_loss_history.npy"),
            np.array(loss_history),
        )

    return env, loss_history


@dataclass
class GenerateConfig:
    model_family:          str            = "openvla"
    pretrained_checkpoint: Union[str, Path] = "./openvla-7b-finetuned-libero-spatial"
    load_in_8bit:          bool           = False
    load_in_4bit:          bool           = False
    center_crop:           bool           = True

    object_name:           str           = "akita_black_bowl"
    override_mesh_path:    Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path:     Optional[str] = None

    task_suite_name:     str           = "libero_spatial"
    task_id:             Optional[int] = 7
    num_steps_wait:      int           = 10
    num_trials_per_task: int           = 50

    enable_attack:        bool           = True
    attack_iters:         int            = 10
    attack_lr:            float          = 0.05
    num_frames_to_attack: int            = 20
    num_train_init_states: int           = 10
    train_frames_per_state: int          = 1
    alpha_action:         float          = 1.0
    alpha_feature:        float          = 10.0
    frame_collect_with_policy: bool       = False
    collect_grasp_frames:  bool  = False
    grasp_pre_frames:      int   = 40
    grasp_post_frames:     int   = 0
    grasp_max_steps:       int   = 400
    grasp_qpos_threshold:  float = 0.02

    photometric_calib_frames: int = 5

    live_test_enabled:        bool  = True
    live_test_every_n_iters:  int   = 20
    live_test_resolution:     int   = 256
    live_test_max_steps:      int   = 300

    save_attack_artifacts: bool           = True
    load_texture_path:     Optional[str]  = None
    local_log_dir:         str            = "./experiments/logs"

    use_wandb:     bool           = False
    wandb_project: str            = "openvla_attack"
    wandb_entity:  str            = "user"

    seed:        int            = 7
    run_id_note: Optional[str]  = None
    unnorm_key:  Optional[str]  = None


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)


    if cfg.object_name not in OBJECTS:
        raise ValueError(f"未知物体 '{cfg.object_name}'，可选: {list(OBJECTS.keys())}")
    obj_cfg = OBJECTS[cfg.object_name]
    mesh_path    = cfg.override_mesh_path    or obj_cfg["mesh"]
    texture_path = cfg.override_texture_path or obj_cfg["texture"]
    xml_path     = cfg.override_xml_path     or obj_cfg["xml"]
    search_kw    = obj_cfg["search"]
    task_suite_name = cfg.task_suite_name or obj_cfg["task_suite"]

    scale_xyz = parse_mesh_scale(xml_path)

    run_id = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id = f"{cfg.run_id_note}-{run_id}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    artifact_dir = os.path.join(cfg.local_log_dir, "attack_artifacts", run_id)
    if cfg.enable_attack and cfg.save_attack_artifacts:
        os.makedirs(artifact_dir, exist_ok=True)
    log_file     = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    original_xml = Path(xml_path)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_id)

    if not original_xml.exists():
        raise FileNotFoundError(f"XML asset not found at {original_xml}")
    global_clean_backup = original_xml.with_name(
        f"{original_xml.stem}_clean_backup_{DATE_TIME}{original_xml.suffix}"
    )
    shutil.copy(original_xml, global_clean_backup)

    _real_tex_path = Path(texture_path)
    if not _real_tex_path.is_absolute():
        _real_tex_path = (Path.cwd() / _real_tex_path).resolve()
    _real_tex_backup = None
    if _real_tex_path.exists():
        _real_tex_backup = _real_tex_path.with_name(f"texture_clean_backup_{DATE_TIME}.png")
        shutil.copy(_real_tex_path, _real_tex_backup)
    else:
        print(f"[WARNING] Real MuJoCo texture not found at {_real_tex_path}, MuJoCo video may look clean")

    import signal, atexit

    def _restore_clean_assets(context: str, remove_backups: bool = False):
        if global_clean_backup.exists():
            shutil.copy(global_clean_backup, original_xml)
            if remove_backups:
                global_clean_backup.unlink(missing_ok=True)
            print(f"[INFO] {context} Original XML restored.")
        if _real_tex_backup is not None and _real_tex_backup.exists():
            shutil.copy(_real_tex_backup, _real_tex_path)
            if remove_backups:
                _real_tex_backup.unlink(missing_ok=True)
            print(f"[INFO] {context} Real MuJoCo texture restored.")

    def _restore_xml_on_exit():
        _restore_clean_assets("(exit handler)", remove_backups=True)

    def _sig_handler(signum, frame):
        _restore_xml_on_exit()
        raise SystemExit(f"Caught signal {signum}, exiting cleanly.")
    atexit.register(_restore_xml_on_exit)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT,  _sig_handler)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite_name]()
    target_tasks   = ([cfg.task_id] if cfg.task_id is not None
                      else range(task_suite_obj.n_tasks))

    model     = get_model(cfg)
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    renderer = None
    if cfg.enable_attack:
        print("[INFO] Initializing DifferentiableRenderer...")
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=scale_xyz,
        ).to(model.device)
    total_episodes  = 0
    total_successes = 0

    try:
        VIDEO_RES = 512

        for task_id in tqdm.tqdm(target_tasks, desc="Tasks"):
            _restore_clean_assets(f"Before Task {task_id}", remove_backups=False)

            task        = task_suite_obj.get_task(task_id)
            init_states = task_suite_obj.get_task_init_states(task_id)

            if cfg.enable_attack and cfg.load_texture_path is not None:
                print(f"[INFO] Loading pre-trained adversarial noise from {cfg.load_texture_path}")
                if cfg.load_texture_path.endswith(".pt"):
                    noise_t = torch.load(cfg.load_texture_path,
                                         map_location=renderer.adv_noise.device)
                    renderer.adv_noise.data.copy_(noise_t)
                    with torch.no_grad():
                        delta = torch.tanh(noise_t) * renderer.epsilon
                    print(f"[INFO] adv_noise loaded from .pt: "
                          f"max_delta={delta.abs().max():.4f}, "
                          f"nonzero={(delta.abs() > 1e-3).float().mean() * 100:.1f}%")
                else:
                    baked_np  = np.array(Image.open(cfg.load_texture_path)).astype(np.float32) / 255.0
                    baked_t   = torch.from_numpy(baked_np).unsqueeze(0).to(renderer.adv_noise.device)
                    baked_stored = torch.flip(baked_t, dims=[1])
                    saved_orig = renderer.orig_texture
                    renderer.orig_texture = baked_stored.contiguous()
                    adv_vc_loaded = renderer._sample_uv_texture_at_vertices()
                    renderer.orig_texture = saved_orig
                    delta   = (adv_vc_loaded - renderer.orig_vertex_colors).clamp(
                        -renderer.epsilon + 1e-6, renderer.epsilon - 1e-6)
                    noise_t = torch.atanh(delta / renderer.epsilon)
                    renderer.adv_noise.data.copy_(noise_t)
                    print(f"[INFO] adv_noise loaded from PNG: "
                          f"max_delta={delta.abs().max():.4f}, "
                          f"nonzero={(delta.abs() > 1e-3).float().mean() * 100:.1f}%")
                adv_tex_inj_path = os.path.join(artifact_dir, f"task_{task_id}_adv_tex_loaded.png")
                with torch.no_grad():
                    baked_vis = renderer.get_baked_adv_texture()
                Image.fromarray(
                    (baked_vis.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                ).save(adv_tex_inj_path)
                tree = ET.parse(original_xml)
                root = tree.getroot()
                for asset_elem in root.findall("asset"):
                    for tex_elem in asset_elem.findall("texture"):
                        if tex_elem.get("name") == f"tex-{cfg.object_name}":
                            tex_elem.set("file", str(Path(adv_tex_inj_path).resolve()))
                            tex_elem.set("type", "2d")
                            break
                for mat_elem in root.findall(".//material"):
                    if mat_elem.get("name") == f"mat-{cfg.object_name}":
                        mat_elem.set("texuniform", "false")
                tree.write(original_xml)
                print(f"[INFO] MuJoCo XML updated with adversarial texture → {adv_tex_inj_path}")
                if _real_tex_path.exists() or _real_tex_backup is not None:
                    shutil.copy(adv_tex_inj_path, _real_tex_path)
                    print(f"[INFO] Real MuJoCo texture overwritten → {_real_tex_path}")

            if cfg.enable_attack and cfg.load_texture_path is None:
                print(f"[INFO] Attack training for Task {task_id}...")
                renderer.reset_texture()

                dummy_env, train_task_desc = get_libero_env(
                    task, cfg.model_family, resolution=256
                )
                dummy_env.close()

                _, _ = train_adversarial_texture(
                    cfg, model, processor, renderer,
                    init_states[0], task, train_task_desc,
                    artifact_dir, episode_idx=task_id,
                    search_keywords_list=search_kw,
                    xml_path=original_xml,
                    num_iters=cfg.attack_iters,
                    init_states=init_states,
                )

                with torch.no_grad():
                    baked_tex = renderer.get_baked_adv_texture()
                    trained_tex_path = os.path.join(
                        artifact_dir, f"task_{task_id}_adv_texture_{DATE_TIME}.png"
                    )
                    Image.fromarray(
                        (baked_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    ).save(trained_tex_path)
                print(f"[INFO] Task {task_id} texture saved → {trained_tex_path}")

                tree = ET.parse(original_xml)
                root = tree.getroot()
                tex_name = f"tex-{cfg.object_name}"
                mat_name = f"mat-{cfg.object_name}"
                for asset_elem in root.findall("asset"):
                    for tex_elem in asset_elem.findall("texture"):
                        if tex_elem.get("name") == tex_name:
                            tex_elem.set("file", str(Path(trained_tex_path).resolve()))
                            tex_elem.set("type", "2d")
                            break
                for mat_elem in root.findall(".//material"):
                    if mat_elem.get("name") == mat_name:
                        mat_elem.set("texuniform", "false")
                tree.write(original_xml)
                print(f"[INFO] XML updated for Task {task_id}.")
                if _real_tex_path.exists() or _real_tex_backup is not None:
                    shutil.copy(trained_tex_path, _real_tex_path)
                    print(f"[INFO] Real MuJoCo texture overwritten → {_real_tex_path}")

            n_eval = min(cfg.num_trials_per_task, len(init_states))
            for ep in tqdm.tqdm(range(n_eval),
                                 desc=f"Task {task_id} Episodes"):
                env, task_description = get_libero_env(
                    task, cfg.model_family, resolution=VIDEO_RES
                )
                env.reset()
                obs = env.set_init_state(init_states[ep])
                env.env.sim.forward()

                t, max_steps, done, replay_images = 0, 300, False, []

                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(
                                get_libero_dummy_action(cfg.model_family)
                            )
                            t += 1
                            continue

                        img_high = get_libero_image(obs, VIDEO_RES)
                        model_input_size = get_image_resize_size(cfg)

                        if renderer is not None:
                            bg = (torch.from_numpy(img_high).float().to(model.device) / 255.0
                                  ).permute(2, 0, 1).unsqueeze(0)
                            mm, bid, _ = get_target_model_matrix(env, search_kw)
                            mvp = (get_render_mvp_from_matrix(env, mm, resolution=(VIDEO_RES, VIDEO_RES))
                                   if bid != -1 else None)
                            _bg_np = _render_bg_without_target(env, bid, VIDEO_RES) if bid != -1 else None
                            composite_bg = (
                                (torch.from_numpy(_bg_np.copy()).float().to(model.device) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0)
                                if _bg_np is not None else bg
                            )
                            with torch.no_grad():
                                if mvp is not None:
                                    adv_rgba, mask = renderer.render(
                                        mvp, resolution=(VIDEO_RES, VIDEO_RES),
                                        model_rot=mm[:3, :3])
                                    comp = _composite(adv_rgba, mask, composite_bg)
                                else:
                                    comp = bg
                            frame_np = (comp[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                            img_model = np.array(Image.fromarray(frame_np).resize(
                                (model_input_size, model_input_size)))
                        else:
                            frame_np = img_high
                            img_model = np.array(
                                Image.fromarray(img_high).resize((model_input_size, model_input_size)))

                        replay_images.append(img_high)

                        observation = {
                            "full_image": img_model,
                            "state": np.concatenate((
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                        }
                        action = get_action(
                            cfg, model, observation, task_description, processor=processor
                        )
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            total_successes += 1
                            break
                        t += 1
                    except Exception as e:
                        print(f"[ERROR] Task {task_id} Ep {ep} step {t}: {e}")
                        traceback.print_exc()
                        break

                total_episodes += 1
                log_str = f"Task: {task_id} | Ep: {ep} | Success: {done}"
                print(log_str)
                log_file.write(log_str + "\n")
                log_file.flush()

                save_rollout_video(
                    replay_images, total_episodes,
                    success=done, task_description=task_description,
                    log_file=log_file,
                )
                env.close()

        avg_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
        print(f"\n[DONE] Episodes: {total_episodes} | Attack success rate: {avg_sr:.2%}")
        log_file.write(f"\nFINAL AVG SUCCESS RATE: {avg_sr:.2%}\n")

    finally:
        _restore_clean_assets("Final cleanup", remove_backups=True)
        log_file.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    eval_libero()
