import os
import sys
import math
import time
import random
from collections import deque
import xml.etree.ElementTree as ET
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import tqdm
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
import draccus
from omegaconf import OmegaConf

PATH_TO_LIBERO_ROOT = "/data/shaozhen.liu/code/tex3d-deps/libero-pi"
if PATH_TO_LIBERO_ROOT not in sys.path:
    sys.path.append(PATH_TO_LIBERO_ROOT)

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

try:
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    import openpi.models.model as _model
except ImportError:
    raise ImportError("请确保已在 openpi 环境下运行此脚本")

ASSET_ROOT_SCANNED = (
    "/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/assets/stable_scanned_objects"
)
ASSET_ROOT_HOPE = (
    "/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/assets/stable_hope_objects"
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
        **_hope("alphabet_soup", mesh_file="alphabet_soup_col.obj"),
        "search":     [["alphabet_soup"], ["soup"]],
        "task_suite": "libero_object",
        "task_id":    0,
    },
    "cream_cheese": {
        **_hope("cream_cheese", mesh_file="hope_cream_cheese_coll.obj"),
        "search":     [["cream_cheese"], ["cheese"]],
        "task_suite": "libero_object",
        "task_id":    1,
    },
    "salad_dressing": {
        **_hope("salad_dressing", mesh_file="salad_dressing_col.obj"),
        "search":     [["salad_dressing"], ["dressing"]],
        "task_suite": "libero_object",
        "task_id":    2,
    },
    "bbq_sauce": {
        **_hope("bbq_sauce", mesh_file="hope_bbq_coll.obj"),
        "search":     [["bbq_sauce"], ["bbq"], ["sauce"]],
        "task_suite": "libero_object",
        "task_id":    3,
    },
    "ketchup": {
        **_hope("ketchup", mesh_file="ketchup_col.obj"),
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
        **_hope("butter", mesh_file="hope_butter_coll.obj"),
        "search":     [["butter"]],
        "task_suite": "libero_object",
        "task_id":    6,
    },
    "milk": {
        **_hope("milk", mesh_file="milk_col.obj"),
        "search":     [["milk"]],
        "task_suite": "libero_object",
        "task_id":    7,
    },
    "chocolate_pudding": {
        **_hope("chocolate_pudding", mesh_file="chocolate_pudding_col.obj"),
        "search":     [["chocolate_pudding"], ["chocolate"], ["pudding"]],
        "task_suite": "libero_object",
        "task_id":    8,
    },
    "orange_juice": {
        **_hope("orange_juice", mesh_file="orange_juice_col.obj"),
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


def get_target_model_matrix(env, search_keywords_list: List[List[str]]):
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    target_body_id = -1
    found_name = None

    for keywords in search_keywords_list:
        for i in range(sim.model.nbody):
            try:
                name = sim.model.body_id2name(i)
            except Exception:
                name = None
            if not name or "vis" in name or "site" in name:
                continue
            if all(k in name for k in keywords):
                target_body_id = i
                found_name = name
                break
        if target_body_id != -1:
            break

    if target_body_id == -1:
        return None, -1, None

    pos  = sim.data.body_xpos[target_body_id]
    quat = sim.data.body_xquat[target_body_id]
    rot  = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat  = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    mat[:3,  3] = pos
    return torch.from_numpy(mat).cuda(), target_body_id, found_name


def build_mvp(
    env, model_matrix: torch.Tensor, resolution=(256, 256), camera_name="agentview"
) -> torch.Tensor:
    sim = env.sim if not hasattr(env, "unwrapped") else env.unwrapped.sim
    W, H = resolution

    cam_id = 0
    try:
        cam_id = sim.model.camera_name2id(camera_name)
    except Exception:
        pass

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
    proj_mtx[0, 0] = -f / aspect
    proj_mtx[1, 1] = -f
    proj_mtx[2, 2] = (far + near) / (near - far)
    proj_mtx[2, 3] = (2 * far * near) / (near - far)
    proj_mtx[3, 2] = -1.0

    return (
        torch.from_numpy(proj_mtx).cuda()
        @ torch.from_numpy(view_mtx).cuda()
        @ model_matrix
    )


def get_libero_dummy_action():
    return [0.0] * 6 + [-1.0]


def _get_libero_env(task, resolution, seed=7):
    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths":  resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _close_libero_env(env):
    """Release MuJoCo's EGL context before dropping the LIBERO wrapper."""
    if env is None:
        return
    try:
        sim = getattr(getattr(env, "env", None), "sim", None)
        if sim is not None:
            sim.free()
            env.env.sim = None
    finally:
        env.close()


def get_libero_image(obs, resolution=256):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _quat2axisangle(quat):
    quat = np.clip(quat, -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_raw_inputs(
    obs_env,
    task_description,
    adv_image_np=None,
    adv_wrist_image_np=None,
    resolution=256,
):
    img = get_libero_image(obs_env, resolution) if adv_image_np is None else adv_image_np
    if adv_wrist_image_np is None:
        wrist = np.ascontiguousarray(obs_env["robot0_eye_in_hand_image"][::-1, ::-1])
    else:
        wrist = np.ascontiguousarray(adv_wrist_image_np)
    state = np.concatenate((
        obs_env["robot0_eef_pos"],
        _quat2axisangle(obs_env["robot0_eef_quat"]),
        obs_env["robot0_gripper_qpos"],
    ))
    return {
        "observation/image":       img,
        "observation/wrist_image": wrist,
        "observation/state":       state,
        "prompt":                  str(task_description),
    }


class DifferentiableRenderer(nn.Module):
    def __init__(self, mesh_path, orig_texture_path=None, device="cuda",
                 scale_xyz=None, pos_offset=None, epsilon=64.0 / 255.0):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.texture_res = 256
        self.pos_offset = torch.tensor(
            pos_offset or [0.02, 0.01, 0.025], dtype=torch.float32, device=device
        )

        if scale_xyz is None:
            scale_xyz = [1.0, 1.0, 1.0]
        scale_arr = np.array(scale_xyz, dtype=np.float64)

        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception:
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
                self.register_buffer("uv_idx", torch.from_numpy(mesh.visual.face_uv.astype(np.int32)).to(device))
            else:
                self.register_buffer("uv_idx", self.faces)
        else:
            v  = torch.from_numpy(vertices).to(device)
            uv = v[:, :2]
            if uv.numel() > 0:
                mn, mx = uv.min(0, keepdim=True)[0], uv.max(0, keepdim=True)[0]
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
            img = (Image.open(orig_texture_path).convert("RGB")
                        .resize((self.texture_res, self.texture_res))
                        .transpose(Image.FLIP_TOP_BOTTOM))
            tex = torch.from_numpy(np.array(img)).float() / 255.0
            self.register_buffer("orig_texture", tex.unsqueeze(0).to(device).contiguous())
        else:
            fb = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 1, 3).expand(
                1, self.texture_res, self.texture_res, 3).contiguous()
            self.register_buffer("orig_texture", fb)

        self.adv_vc_noise = nn.Parameter(
            torch.zeros((self.num_vertices, 3), dtype=torch.float32, device=device)
        )
        self.light_dir = F.normalize(torch.tensor([0.5, 0.5, 1.0], device=device), dim=0)

    def get_texture_param(self):
        return self.adv_vc_noise

    def render(self, mvp, resolution=(256, 256), return_clean=False, model_rot=None):
        pos      = self.pos + self.pos_offset
        pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())

        rast, _         = dr.rasterize(self.glctx, pos_clip.unsqueeze(0), self.faces, resolution=resolution)
        tex_uv, _       = dr.interpolate(self.uv.unsqueeze(0), rast, self.uv_idx)
        clean_color     = dr.texture(self.orig_texture.contiguous(), tex_uv.contiguous(), filter_mode="linear")

        noise           = torch.tanh(self.adv_vc_noise) * self.epsilon
        noise_interp, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast, self.faces)
        adv_color       = torch.clamp(clean_color + noise_interp, 0.0, 1.0)

        vn_world = (model_rot @ self.vn.T).T if model_rot is not None else self.vn
        vn_interp, _ = dr.interpolate(vn_world.unsqueeze(0).contiguous(), rast, self.faces)
        vn_interp    = F.normalize(vn_interp, p=2, dim=-1)
        diffuse      = torch.clamp((vn_interp * self.light_dir.view(1, 1, 1, 3)).sum(-1, keepdim=True), 0.0, 1.0)
        lighting     = 0.6 + 0.4 * diffuse

        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            return adv_color * lighting, clean_color * lighting, mask
        return adv_color * lighting, mask

    def get_baked_adv_texture(self):
        with torch.no_grad():
            noise = torch.tanh(self.adv_vc_noise) * self.epsilon
            adv_vertex_colors = torch.clamp(
                self._sample_original_vertex_colors() + noise, 0.0, 1.0
            )

            uv_to_vertex = torch.zeros(
                self.uv.shape[0], dtype=torch.long, device=self.device
            )
            face_vertices = self.faces.long()
            face_uvs = self.uv_idx.long()
            for corner in range(3):
                uv_to_vertex[face_uvs[:, corner]] = face_vertices[:, corner]
            uv_colors = adv_vertex_colors[uv_to_vertex]

            uv_clip = torch.zeros(
                self.uv.shape[0], 4, device=self.device, dtype=self.uv.dtype
            )
            uv_clip[:, 0] = self.uv[:, 0] * 2.0 - 1.0
            uv_clip[:, 1] = self.uv[:, 1] * 2.0 - 1.0
            uv_clip[:, 3] = 1.0
            rast, _ = dr.rasterize(
                self.glctx,
                uv_clip.unsqueeze(0),
                self.uv_idx.int(),
                resolution=(self.texture_res, self.texture_res),
            )
            baked, _ = dr.interpolate(
                uv_colors.unsqueeze(0).contiguous(), rast, self.uv_idx.int()
            )
            mask = (rast[..., 3] > 0).float().unsqueeze(-1)
            result = mask * baked + (1.0 - mask) * self.orig_texture
            return torch.flip(result.clamp(0.0, 1.0), dims=[1])

    def _sample_original_vertex_colors(self):
        if len(self.uv) == self.num_vertices:
            vertex_uv = self.uv
        else:
            uv_sum = torch.zeros(self.num_vertices, 2, device=self.device)
            uv_count = torch.zeros(self.num_vertices, 1, device=self.device)
            face_vertices = self.faces.long()
            face_uvs = self.uv_idx.long()
            ones = torch.ones(len(face_vertices), 1, device=self.device)
            for corner in range(3):
                vertex_ids = face_vertices[:, corner]
                uv_ids = face_uvs[:, corner]
                uv_sum.scatter_add_(
                    0, vertex_ids[:, None].expand(-1, 2), self.uv[uv_ids]
                )
                uv_count.scatter_add_(0, vertex_ids[:, None], ones)
            vertex_uv = uv_sum / uv_count.clamp_min(1.0)
        sampled = dr.texture(
            self.orig_texture.contiguous(),
            vertex_uv[None, None].contiguous(),
            filter_mode="linear",
        )
        return sampled[0, 0].contiguous()


def hide_object_geoms(sim, search_keywords_list: List[List[str]]):
    hidden = []
    for i in range(sim.model.ngeom):
        try:
            name = sim.model.geom_id2name(i)
        except Exception:
            name = None
        if not name:
            continue
        matched = any(all(k in name for k in kws) for kws in search_keywords_list)
        if matched:
            orig = sim.model.geom_rgba[i].copy()
            sim.model.geom_rgba[i, 3] = 0.0
            hidden.append((i, orig))
    return hidden


def restore_object_geoms(sim, hidden):
    for i, orig in hidden:
        sim.model.geom_rgba[i] = orig


def render_mujoco_background_without_target(
    env,
    body_id: int,
    resolution: int,
    search_keywords_list: Optional[List[List[str]]] = None,
    camera_name="agentview",
):
    """Render MuJoCo normally, but temporarily hide the target object.

    The clean observation always comes directly from MuJoCo. This background is only
    used as the plate on which the nvdiffrast adversarial object is composited.
    """
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    hidden = []

    if body_id >= 0:
        geom_ids = [
            geom_id for geom_id in range(sim.model.ngeom)
            if int(sim.model.geom_bodyid[geom_id]) == int(body_id)
        ]
        for geom_id in geom_ids:
            original_rgba = sim.model.geom_rgba[geom_id].copy()
            sim.model.geom_rgba[geom_id, 3] = 0.0
            hidden.append((geom_id, original_rgba))

    # Some LIBERO assets put visual geoms under child bodies. Fall back to the
    # name-based search so that those geoms are removed from the background too.
    if search_keywords_list:
        already_hidden = {geom_id for geom_id, _ in hidden}
        for geom_id, original_rgba in hide_object_geoms(sim, search_keywords_list):
            if geom_id not in already_hidden:
                hidden.append((geom_id, original_rgba))

    try:
        sim.forward()
        image = sim.render(
            camera_name=camera_name,
            height=resolution,
            width=resolution,
        )
        return np.ascontiguousarray(image[::-1, ::-1])
    finally:
        restore_object_geoms(sim, hidden)
        sim.forward()


def find_target_texture_id(
    sim,
    body_id: int,
    search_keywords_list: List[List[str]],
) -> int:
    """Find the MuJoCo texture used by the target object's visual material."""
    candidate_geoms = []
    for geom_id in range(sim.model.ngeom):
        name = None
        try:
            name = sim.model.geom_id2name(geom_id)
        except Exception:
            pass
        belongs_to_body = body_id >= 0 and int(sim.model.geom_bodyid[geom_id]) == int(body_id)
        name_matches = bool(
            name
            and any(all(keyword in name for keyword in keywords)
                    for keywords in search_keywords_list)
        )
        if belongs_to_body or name_matches:
            candidate_geoms.append(geom_id)

    for geom_id in candidate_geoms:
        try:
            material_id = int(sim.model.geom_matid[geom_id])
        except Exception:
            material_id = -1
        if material_id < 0:
            continue
        tex_ids = np.asarray(sim.model.mat_texid[material_id]).reshape(-1)
        valid_tex_ids = tex_ids[tex_ids >= 0]
        if len(valid_tex_ids):
            return int(valid_tex_ids[0])

    # Fallback for assets whose geom/body names do not carry the object name.
    for material_id in range(sim.model.nmat):
        name = None
        try:
            name = sim.model.mat_id2name(material_id)
        except Exception:
            pass
        if not name or not any(
            all(keyword in name for keyword in keywords)
            for keywords in search_keywords_list
        ):
            continue
        tex_ids = np.asarray(sim.model.mat_texid[material_id]).reshape(-1)
        valid_tex_ids = tex_ids[tex_ids >= 0]
        if len(valid_tex_ids):
            return int(valid_tex_ids[0])
    return -1


def _upload_mujoco_texture(sim, texture_id: int) -> bool:
    """Upload modified model.tex_rgb data to every available MuJoCo GL context."""
    import mujoco

    contexts = []
    for attr in ("render_contexts", "_render_context_offscreen", "render_context_offscreen"):
        value = getattr(sim, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            contexts.extend(value)
        else:
            contexts.append(value)

    uploaded = False
    seen = set()
    for context in contexts:
        if id(context) in seen:
            continue
        seen.add(id(context))
        con = getattr(context, "con", None)
        context_model = getattr(context, "model", None)
        if con is None or context_model is None:
            continue
        gl_context = getattr(context, "gl_ctx", None)
        if gl_context is not None:
            gl_context.make_current()
        native_model = getattr(context_model, "_model", context_model)
        mujoco.mjr_uploadTexture(native_model, con, texture_id)
        uploaded = True
    return uploaded


def inject_mujoco_texture(sim, texture_id: int, texture_rgb: np.ndarray):
    """Replace one MuJoCo texture in memory and return data needed for restoration."""
    if texture_id < 0:
        raise ValueError("Invalid MuJoCo texture id")

    # robosuite's compatibility MjModel forwards most MuJoCo arrays, but not
    # tex_rgb in this release. Read all texture storage from the native model.
    native_model = getattr(sim.model, "_model", sim.model)
    height = int(native_model.tex_height[texture_id])
    width = int(native_model.tex_width[texture_id])
    offset = int(native_model.tex_adr[texture_id])
    channels = int(native_model.tex_nchannel[texture_id]) if hasattr(
        native_model, "tex_nchannel"
    ) else 3
    byte_count = height * width * channels
    texture_storage = getattr(
        native_model, "tex_data", getattr(native_model, "tex_rgb", None)
    )
    if texture_storage is None:
        raise RuntimeError("MuJoCo model exposes neither tex_data nor tex_rgb")

    texture_rgb = np.asarray(texture_rgb, dtype=np.uint8)
    if texture_rgb.ndim != 3 or texture_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB texture, got {texture_rgb.shape}")
    if texture_rgb.shape[:2] != (height, width):
        texture_rgb = np.asarray(
            Image.fromarray(texture_rgb).resize((width, height), Image.BILINEAR),
            dtype=np.uint8,
        )
    if channels == 4:
        alpha = np.full((*texture_rgb.shape[:2], 1), 255, dtype=np.uint8)
        texture_rgb = np.concatenate([texture_rgb, alpha], axis=2)
    elif channels != 3:
        raise ValueError(f"Unsupported MuJoCo texture channel count: {channels}")

    original = np.asarray(
        texture_storage[offset:offset + byte_count], dtype=np.uint8
    ).copy()
    texture_storage[offset:offset + byte_count] = texture_rgb.reshape(-1)
    uploaded = _upload_mujoco_texture(sim, texture_id)
    if not uploaded:
        print(
            "[warning] MuJoCo texture data changed, but no upload_texture context "
            "was found; the video renderer may keep the old GPU texture"
        )
    return original, offset, byte_count


def restore_mujoco_texture(sim, texture_id: int, original, offset: int, byte_count: int):
    native_model = getattr(sim.model, "_model", sim.model)
    texture_storage = getattr(
        native_model, "tex_data", getattr(native_model, "tex_rgb", None)
    )
    if texture_storage is None:
        raise RuntimeError("MuJoCo model exposes neither tex_data nor tex_rgb")
    texture_storage[offset:offset + byte_count] = original
    _upload_mujoco_texture(sim, texture_id)


def render_hires(renderer, bg_tensor, mvp, hires=1024, render_res=256, model_rot=None):
    if mvp is None:
        return bg_tensor
    bg_hires = F.interpolate(bg_tensor, size=(hires, hires), mode="bilinear", align_corners=False)
    adv_rgba, mask = renderer.render(mvp, resolution=(hires, hires), model_rot=model_rot)
    adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
    mask_t   = mask.permute(0, 3, 1, 2)
    return torch.clamp(adv_rgb * mask_t + bg_hires * (1 - mask_t), 0.0, 1.0)


def save_hires(tensor, path, hires=1024):
    if tensor.shape[-1] != hires or tensor.shape[-2] != hires:
        tensor = F.interpolate(tensor, size=(hires, hires), mode="bilinear", align_corners=False)
    arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def annotate_video_frames(frames, label):
    """Burn a short evaluation label into each saved video frame."""
    annotated = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width, 360), 24), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        annotated.append(np.asarray(image))
    return annotated


def render_and_composite(
    renderer,
    bg_tensor,
    mvp,
    resolution=(256, 256),
    return_clean=False,
    model_rot=None,
):
    if mvp is None:
        return (bg_tensor, bg_tensor) if return_clean else bg_tensor
    if return_clean:
        adv_rgba, clean_rgba, mask = renderer.render(
            mvp, resolution=resolution, return_clean=True, model_rot=model_rot
        )
        clean_rgb = clean_rgba.permute(0, 3, 1, 2)
    else:
        adv_rgba, mask = renderer.render(mvp, resolution=resolution, model_rot=model_rot)
    adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
    mask_t   = mask.permute(0, 3, 1, 2)
    comp_adv = torch.clamp(adv_rgb * mask_t + bg_tensor * (1 - mask_t), 0.0, 1.0)
    if return_clean:
        comp_clean = torch.clamp(clean_rgb * mask_t + bg_tensor * (1 - mask_t), 0.0, 1.0)
        return comp_adv, comp_clean
    return comp_adv


def inspect_image_structure(processed_inputs, verbose=True):
    img_top_key = None
    for key in ["image", "base_0_rgb", "cam_high", "observation/image"]:
        if key in processed_inputs:
            img_top_key = key
            break
    if img_top_key is None:
        raise RuntimeError(f"找不到图像键！所有键: {list(processed_inputs.keys())}")
    img_val = processed_inputs[img_top_key]
    if isinstance(img_val, dict):
        cam_key = None
        for ck in ["base_0_rgb", "cam_high", "agentview_rgb", "exterior_image_1_left", "image"]:
            if ck in img_val:
                cam_key = ck
                break
        if cam_key is None:
            cam_key = next(iter(img_val))
        img_tensor = img_val[cam_key]
        if verbose:
            print(f"  [图像结构] image 是 dict，包含: {list(img_val.keys())}")
            print(f"  [图像结构] 主相机键: '{cam_key}'，shape={img_tensor.shape}")
    else:
        cam_key = None
        img_tensor = img_val
    with torch.no_grad():
        vmax = img_tensor.float().max().item()
        vmin = img_tensor.float().min().item()
    if vmax > 2.0:
        value_range = "0_255"
    elif vmin < -0.5:
        value_range = "-1_1"
    else:
        value_range = "0_1"
    return img_top_key, cam_key, img_tensor, value_range


def _adapt_adv_image(adv_image_01, orig_tensor, value_range):
    orig_h, orig_w = orig_tensor.shape[-2], orig_tensor.shape[-1]
    if adv_image_01.shape[-2:] != (orig_h, orig_w):
        adv = F.interpolate(adv_image_01, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    else:
        adv = adv_image_01
    if value_range == "0_255":
        adv = adv * 255.0
    elif value_range == "-1_1":
        adv = adv * 2.0 - 1.0
    adv = adv.to(dtype=orig_tensor.dtype)
    if orig_tensor.dim() == 5:
        T = orig_tensor.shape[1]
        adv = adv.unsqueeze(1).expand(-1, T, -1, -1, -1)
    return adv


def replace_image_in_processed_inputs(processed_inputs, adv_image_01, device, verbose=False):
    img_top_key, cam_key, orig_tensor, value_range = inspect_image_structure(
        processed_inputs, verbose=verbose
    )
    adv_resized = _adapt_adv_image(adv_image_01, orig_tensor, value_range)
    new_inputs  = dict(processed_inputs)
    img_val     = new_inputs[img_top_key]
    if isinstance(img_val, dict):
        new_img_dict = dict(img_val)
        new_img_dict[cam_key] = adv_resized
        new_inputs[img_top_key] = new_img_dict
    else:
        new_inputs[img_top_key] = adv_resized
    return new_inputs, img_top_key, cam_key


def get_processed_inputs_and_observation(policy, raw_inputs_dict):
    import jax
    device = policy._pytorch_device
    inputs = jax.tree.map(lambda x: x, raw_inputs_dict)
    inputs = policy._input_transform(inputs)
    processed_inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device).unsqueeze(0),
        inputs
    )
    observation = _model.Observation.from_dict(processed_inputs)
    return processed_inputs, observation, device


def extract_split_features(model, observation):
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    img_embs = []
    for img, img_mask in zip(images, img_masks):
        img_emb = model.paligemma_with_expert.embed_image(img)
        img_embs.append(img_emb)
    img_embs = torch.cat(img_embs, dim=1)

    lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

    return img_embs, lang_emb, lang_masks


def compute_image_feature_loss(model, adv_observation, clean_img_embs_detached):
    adv_img_embs, _, _ = extract_split_features(model, adv_observation)
    D = adv_img_embs.shape[-1]
    return F.mse_loss(
        adv_img_embs.float(),
        clean_img_embs_detached.float(),
        reduction="mean"
    ) * D


def _extract_first_action(policy_output):
    action = policy_output["actions"] if isinstance(policy_output, dict) else policy_output
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)
    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]
    return torch.from_numpy(action.astype(np.float32))


def load_latent_encoder(config_path, ckpt_path, device):
    taming_candidates = [
        Path("/data/shaozhen.liu/code/tex3d-deps/taming-transformers"),
        Path("/data/shaozhen.liu/code/tex3d-deps/taming-transformers"),
        Path("./taming-transformers"),
        Path("./taming-transformers-master"),
    ]
    for taming_root in taming_candidates:
        if taming_root.exists():
            taming_root_resolved = str(taming_root.resolve())
            if taming_root_resolved not in sys.path:
                sys.path.insert(0, taming_root_resolved)
            break

    from taming.models.vqgan import VQModel

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.params if "model" in cfg and "params" in cfg.model else cfg.model
    # We only use VQModel.encoder for a frozen perceptual feature. The original
    # training loss constructs LPIPS/VGG16 and downloads separate VGG weights,
    # which is unnecessary for encoding and breaks on read-only home caches.
    model_cfg.lossconfig = OmegaConf.create({"target": "torch.nn.Identity"})
    latent_model = VQModel(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    latent_model.load_state_dict(state_dict, strict=False)
    latent_model.eval().to(device)
    for p in latent_model.parameters():
        p.requires_grad = False
    return latent_model.encoder


def extract_latent(encoder, img_np, device):
    img = Image.fromarray(img_np).resize((256, 256))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = x * 2.0 - 1.0
    with torch.no_grad():
        feat_map = encoder(x)
        feat = feat_map.mean(dim=[2, 3]).squeeze(0)
    return feat.detach()



# --- LIBERO-goal suite extension (added for goal-suite reproduction) ---
GOAL_TASKS = {
    "goal_bowl_t1":  dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_goal", task_id=1),
    "goal_bowl_t3":  dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_goal", task_id=3),
    "goal_bowl_t4":  dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_goal", task_id=4),
    "goal_plate_t5": dict(_scanned("plate"), search=[["plate"]], task_suite="libero_goal", task_id=5),
    "goal_cream_t6": dict(_hope("cream_cheese", mesh_file="hope_cream_cheese_coll.obj"), search=[["cream_cheese"], ["cheese"]], task_suite="libero_goal", task_id=6),
    "goal_bowl_t8":  dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_goal", task_id=8),
}
OBJECTS.update(GOAL_TASKS)
# --- end extension ---


# --- LIBERO-spatial per-task entries (pi0 spatial attack suite) ---
SP_TASKS = {
    "sp_bowl_t0": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=0),
    "sp_bowl_t1": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=1),
    "sp_bowl_t2": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=2),
    "sp_bowl_t3": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=3),
    "sp_bowl_t4": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=4),
    "sp_bowl_t5": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=5),
    "sp_bowl_t6": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=6),
    "sp_bowl_t7": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=7),
    "sp_bowl_t8": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=8),
    "sp_bowl_t9": dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_spatial", task_id=9),
}
OBJECTS.update(SP_TASKS)
# --- end spatial extension ---


# --- LIBERO-10 (long) suite extension ---
LONG_TASKS = {
    "lg_soup_t0":  dict(_hope("alphabet_soup", mesh_file="alphabet_soup_col.obj"), search=[["alphabet_soup"], ["soup"]], task_suite="libero_10", task_id=0),
    "lg_cream_t1": dict(_hope("cream_cheese", mesh_file="hope_cream_cheese_coll.obj"), search=[["cream_cheese"], ["cheese"]], task_suite="libero_10", task_id=1),
    "lg_bowl_t3":  dict(_scanned("akita_black_bowl"), search=[["akita_black_bowl"], ["bowl"]], task_suite="libero_10", task_id=3),
    "lg_choc_t6":  dict(_hope("chocolate_pudding"), search=[["chocolate_pudding"], ["chocolate"], ["pudding"]], task_suite="libero_10", task_id=6),
    "lg_soup_t7":  dict(_hope("alphabet_soup", mesh_file="alphabet_soup_col.obj"), search=[["alphabet_soup"], ["soup"]], task_suite="libero_10", task_id=7),
}
OBJECTS.update(LONG_TASKS)
# --- end long extension ---


# --- turbosquid asset root helper (long/goal 补齐) ---
ASSET_ROOT_TS = "/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/assets/turbosquid_objects"
def _ts(name, mesh_file=None, tex_file=None):
    base = f"{ASSET_ROOT_TS}/{name}"
    return {
        "xml":     f"{base}/{name}.xml",
        "mesh":    f"{base}/{mesh_file or (name + '.obj')}",
        "texture": f"{base}/{tex_file or 'texture.png'}",
    }

TS_TASKS = {
    "lg_moka_t2":  dict(_ts("moka_pot", mesh_file="moka_pot_body.obj", tex_file="metal_diff.png"), search=[["moka_pot"], ["moka"]], task_suite="libero_10", task_id=2),
    "lg_pmug_t4":  dict(_ts("porcelain_mug", tex_file="porcelain_mug_texture.png"), search=[["porcelain_mug"], ["mug"]], task_suite="libero_10", task_id=4),
    "lg_book_t5":  dict(_ts("black_book", tex_file="black_book_texture.png"), search=[["black_book"], ["book"]], task_suite="libero_10", task_id=5),
    "lg_moka_t8":  dict(_ts("moka_pot", mesh_file="moka_pot_body.obj", tex_file="metal_diff.png"), search=[["moka_pot"], ["moka"]], task_suite="libero_10", task_id=8),
    "lg_wmug_t9":  dict(_ts("white_yellow_mug", tex_file="white_yellow_mug_texture.png"), search=[["white_yellow_mug"], ["yellow"]], task_suite="libero_10", task_id=9),
    "gwine_t2":    dict(_ts("wine_bottle", mesh_file="wine_bottle_red_logo.obj", tex_file="label_wine.png"), search=[["wine_bottle"], ["wine"]], task_suite="libero_goal", task_id=2),
    "gwine_t9":    dict(_ts("wine_bottle", mesh_file="wine_bottle_red_logo.obj", tex_file="label_wine.png"), search=[["wine_bottle"], ["wine"]], task_suite="libero_goal", task_id=9),
}
OBJECTS.update(TS_TASKS)
# --- end turbosquid extension ---

def compute_frame_weights(latent_features: List[torch.Tensor], tau: float) -> torch.Tensor:
    if len(latent_features) == 0:
        return torch.tensor([], dtype=torch.float32)
    if len(latent_features) == 1:
        return torch.ones(1, dtype=torch.float32)

    feats = [f.detach().float() for f in latent_features]
    device = feats[0].device
    n = len(feats)

    v = torch.zeros(n, device=device, dtype=torch.float32)
    a = torch.zeros(n, device=device, dtype=torch.float32)
    for t in range(1, n):
        v[t] = torch.norm(feats[t] - feats[t - 1], p=2) / 2.0
    for t in range(1, n):
        a[t] = torch.abs(v[t] - v[t - 1])

    def _minmax(x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        denom = torch.clamp(x_max - x_min, min=1e-8)
        return (x - x_min) / denom

    v_hat = _minmax(v)
    a_hat = _minmax(a)
    s = torch.maximum(v_hat, a_hat)
    tau = float(max(tau, 1e-6))
    w = torch.softmax(s / tau, dim=0)
    return w.detach()


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype):
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / torch.sum(g)
    k2d = torch.outer(g, g)
    k2d = k2d / torch.sum(k2d)
    return k2d


def apply_eot_transforms(img_tensor, num_samples) -> torch.Tensor:
    x = img_tensor
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected input shape [1,3,H,W], got {tuple(x.shape)}")
    device, dtype = x.device, x.dtype
    outs = []
    for _ in range(num_samples):
        y = x.clone()
        b = random.uniform(-0.2, 0.2)
        c = random.uniform(0.8, 1.2)
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * c + mean
        y = y + b

        k = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        kernel = _gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        kernel = kernel.view(1, 1, k, k).repeat(3, 1, 1, 1)
        y = F.conv2d(y, kernel, padding=k // 2, groups=3)
        y = torch.clamp(y, 0.0, 1.0)
        outs.append(y.squeeze(0))
    return torch.stack(outs, dim=0)


def perturb_mvp(
    mvp: torch.Tensor,
    rot_std_deg: float = 5.0,
    trans_std: float = 0.02,
    scale_range: tuple = (0.9, 1.1),
) -> torch.Tensor:
    device = mvp.device
    dtype = mvp.dtype

    angles = torch.randn(3, device=device, dtype=dtype) * math.radians(rot_std_deg)
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)

    Rx = torch.eye(4, device=device, dtype=dtype)
    Ry = torch.eye(4, device=device, dtype=dtype)
    Rz = torch.eye(4, device=device, dtype=dtype)

    Rx[1, 1] = cx
    Rx[1, 2] = -sx
    Rx[2, 1] = sx
    Rx[2, 2] = cx

    Ry[0, 0] = cy
    Ry[0, 2] = sy
    Ry[2, 0] = -sy
    Ry[2, 2] = cy

    Rz[0, 0] = cz
    Rz[0, 1] = -sz
    Rz[1, 0] = sz
    Rz[1, 1] = cz

    R_perturb = Rz @ Ry @ Rx

    T_perturb = torch.eye(4, device=device, dtype=dtype)
    T_perturb[:3, 3] = torch.randn(3, device=device, dtype=dtype) * trans_std

    scale = random.uniform(*scale_range)
    S_perturb = torch.eye(4, device=device, dtype=dtype)
    S_perturb[0, 0] = scale
    S_perturb[1, 1] = scale
    S_perturb[2, 2] = scale

    return mvp @ R_perturb @ T_perturb @ S_perturb


def _build_target_action(clean_action: torch.Tensor, mode: str) -> torch.Tensor:
    target = clean_action.clone()
    if mode == "sign_flip_xyz":
        target[:3] = -target[:3]
    else:
        raise ValueError(f"Unsupported target_action_mode: {mode}")
    return target


def collect_frame_data(policy, task, task_description, initial_obs_state,
                       num_frames, RENDER_RES, device, cfg, save_dir, episode_idx,
                       search_keywords_list: List[List[str]]):
    model = policy._model

    env, _ = _get_libero_env(task, resolution=RENDER_RES)
    env.reset()
    obs = env.set_init_state(initial_obs_state)
    env.env.sim.forward()

    print(f"[攻击] 收集 {num_frames} 帧数据并预计算干净图像特征...")
    frame_data   = []
    verbose_done = False
    latent_encoder = load_latent_encoder(
        cfg.latent_encoder_config,
        cfg.latent_encoder_ckpt,
        device,
    )

    for t in range(num_frames):
        img_np     = get_libero_image(obs, RENDER_RES)
        raw_inputs = build_raw_inputs(obs, task_description)

        sim     = env.env.sim
        hidden  = hide_object_geoms(sim, search_keywords_list)
        sim.forward()
        obs_hidden  = sim.render(camera_name="agentview", height=RENDER_RES, width=RENDER_RES)
        bg_np_clean = obs_hidden[::-1, ::-1].copy()
        restore_object_geoms(sim, hidden)

        bg_tensor = (torch.from_numpy(bg_np_clean).float().to(device) / 255.0
                     ).permute(2, 0, 1).unsqueeze(0)

        model_matrix, body_id, found_name = get_target_model_matrix(env, search_keywords_list)
        if body_id != -1:
            print(f"  [帧 {t}] 找到目标 body: '{found_name}'")
            mvp = build_mvp(env, model_matrix, resolution=(RENDER_RES, RENDER_RES))
        else:
            print(f"  [帧 {t}] 未找到目标 body，跳过该帧渲染")
            mvp = None

        with torch.no_grad():
            processed_inputs, observation, _ = get_processed_inputs_and_observation(policy, raw_inputs)
            clean_img_embs, _, _ = extract_split_features(model, observation)
            clean_img_embs = clean_img_embs.detach()
            clean_action = _extract_first_action(policy.infer(raw_inputs))
            latent_feat = extract_latent(latent_encoder, img_np, device)

        if not verbose_done:
            print(f"  [特征] clean_img_embs shape={clean_img_embs.shape}, "
                  f"norm={clean_img_embs.norm():.2f}")
            inspect_image_structure(processed_inputs, verbose=True)
            verbose_done = True

        frame_data.append({
            "bg_tensor":        bg_tensor,
            "mvp":              mvp,
            "processed_inputs": processed_inputs,
            "clean_img_embs":   clean_img_embs,
            "clean_action":     clean_action,
            "latent_feat":      latent_feat,
            "raw_inputs":       raw_inputs,
        })

        obs, _, _, _ = env.step(get_libero_dummy_action())

    return env, frame_data


def collect_frame_data_dual_renderer(
    policy,
    task,
    task_description,
    initial_obs_state,
    num_frames,
    render_res,
    device,
    cfg,
    save_dir,
    episode_idx,
    search_keywords_list: List[List[str]],
):
    """Collect white-box attack frames using MuJoCo-clean + nvdiffrast-adv branches."""
    model = policy._model
    env, _ = _get_libero_env(task, resolution=render_res)
    env.reset()
    obs = env.set_init_state(initial_obs_state)
    env.env.sim.forward()

    frame_data = []
    verbose_done = False
    latent_encoder = load_latent_encoder(
        cfg.latent_encoder_config,
        cfg.latent_encoder_ckpt,
        device,
    )

    print(f"[attack] collecting {num_frames} dual-renderer frames")
    for t in range(num_frames):
        # The clean branch is the untouched MuJoCo camera observation.
        clean_img_np = get_libero_image(obs, render_res)
        raw_inputs = build_raw_inputs(obs, task_description)
        model_matrix, body_id, found_name = get_target_model_matrix(
            env, search_keywords_list
        )
        if body_id == -1:
            print(f"  [frame {t}] target body not found; frame skipped")
            mvp = None
            model_rot = None
        else:
            print(f"  [frame {t}] target body: '{found_name}'")
            mvp = build_mvp(env, model_matrix, resolution=(render_res, render_res))
            model_rot = model_matrix[:3, :3]

        # MuJoCo supplies the background plate; nvdiffrast supplies only the
        # adversarial target object during optimization.
        bg_np = render_mujoco_background_without_target(
            env,
            body_id,
            render_res,
            search_keywords_list=search_keywords_list,
        )
        bg_tensor = (
            torch.from_numpy(bg_np).float().to(device) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        clean_tensor = (
            torch.from_numpy(clean_img_np).float().to(device) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            processed_inputs, observation, _ = get_processed_inputs_and_observation(
                policy, raw_inputs
            )
            clean_img_embs, _, _ = extract_split_features(model, observation)
            clean_img_embs = clean_img_embs.detach()
            clean_action = _extract_first_action(policy.infer(raw_inputs))
            latent_feat = extract_latent(latent_encoder, clean_img_np, device)

        if not verbose_done:
            print(
                f"  [features] clean_img_embs shape={clean_img_embs.shape}, "
                f"norm={clean_img_embs.norm():.2f}"
            )
            inspect_image_structure(processed_inputs, verbose=True)
            verbose_done = True

        frame_data.append({
            "bg_tensor": bg_tensor,
            "clean_tensor": clean_tensor,
            "mvp": mvp,
            "model_rot": model_rot,
            "processed_inputs": processed_inputs,
            "clean_img_embs": clean_img_embs,
            "clean_action": clean_action,
            "latent_feat": latent_feat,
            "raw_inputs": raw_inputs,
        })
        obs, _, _, _ = env.step(get_libero_dummy_action())

    return env, frame_data


def train_adversarial_texture_feature_attack(cfg, policy, renderer, initial_obs_state,
                                              task, task_description, save_dir, episode_idx,
                                              search_keywords_list: List[List[str]],
                                              num_iters=50):
    os.makedirs(save_dir, exist_ok=True)
    RENDER_RES = 256
    model      = policy._model
    device     = policy._pytorch_device

    print(f"[攻击] 图像特征空间攻击 | 模型: {type(model).__name__}, Device: {device}")
    print(f"[攻击] 攻击目标关键词: {search_keywords_list}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    lambda_nat  = getattr(cfg, "lambda_nat",  0.1)
    num_frames  = getattr(cfg, "num_frames",  5)

    env, frame_data = collect_frame_data_dual_renderer(
        policy, task, task_description, initial_obs_state,
        num_frames, RENDER_RES, device, cfg, save_dir, episode_idx,
        search_keywords_list=search_keywords_list,
    )
    frame_weights = compute_frame_weights(
        [f["latent_feat"] for f in frame_data], tau=cfg.tau
    )
    valid_frame_mask = torch.tensor(
        [f["mvp"] is not None for f in frame_data],
        device=frame_weights.device,
        dtype=frame_weights.dtype,
    )
    frame_weights = frame_weights * valid_frame_mask
    if frame_weights.sum() <= 0:
        raise RuntimeError("没有找到包含目标物体的有效攻击帧")
    frame_weights = frame_weights / frame_weights.sum()
    valid_frame_count = int(valid_frame_mask.sum().item())
    print(f"[攻击] 关键帧权重: {frame_weights.cpu().numpy().round(4).tolist()}")

    optimizer_mode = getattr(cfg, "optimizer_mode", "sign_pgd")
    if optimizer_mode == "adam":
        optimizer = torch.optim.Adam([renderer.get_texture_param()], lr=cfg.attack_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iters, eta_min=cfg.attack_lr * 0.1
        )
    elif optimizer_mode == "sign_pgd":
        optimizer = None
        scheduler = None
    else:
        raise ValueError("optimizer_mode must be 'sign_pgd' or 'adam'")

    loss_history  = []
    grad_log_path = os.path.join(save_dir, f"Ep{episode_idx}_gradient_log.txt")
    with open(grad_log_path, "w") as f:
        f.write("Iter | Total Loss | TAAO Loss | Nat Loss | Grad Norm | LR\n")

    print(f"\n[攻击] 开始图像特征空间优化，共 {num_iters} 轮...")
    iterator = tqdm.tqdm(range(num_iters), desc="Image Feature Attack", leave=False)

    for i in iterator:
        renderer.get_texture_param().grad = None
        if optimizer is not None:
            optimizer.zero_grad()
        avg_taao = avg_nat = avg_total = 0.0
        valid = 0

        for t, fdata in enumerate(frame_data):
            if fdata["mvp"] is None:
                continue

            adv_diff = render_and_composite(
                renderer, fdata["bg_tensor"], fdata["mvp"],
                resolution=(RENDER_RES, RENDER_RES),
                model_rot=fdata.get("model_rot"),
            )
            # The naturalness/reference target is the real MuJoCo clean frame,
            # not a second nvdiffrast rendering.
            nat_loss = F.mse_loss(adv_diff, fdata["clean_tensor"])

            if cfg.use_eot:
                adv_imgs_list = []
                for _ in range(cfg.eot_num_samples):
                    mvp_perturbed = perturb_mvp(
                        fdata["mvp"],
                        rot_std_deg=cfg.eot_rot_std_deg,
                        trans_std=cfg.eot_trans_std,
                        scale_range=(cfg.eot_scale_min, cfg.eot_scale_max),
                    )
                    adv_rgba_k, mask_k = renderer.render(
                        mvp_perturbed,
                        resolution=(RENDER_RES, RENDER_RES),
                        model_rot=fdata.get("model_rot"),
                    )
                    adv_rgb_k = adv_rgba_k.permute(0, 3, 1, 2)
                    mask_k = mask_k.permute(0, 3, 1, 2)
                    adv_k = torch.clamp(
                        adv_rgb_k * mask_k + fdata["bg_tensor"] * (1 - mask_k), 0.0, 1.0
                    )
                    adv_k = apply_eot_transforms(adv_k, num_samples=1)[0:1]
                    adv_imgs_list.append(adv_k)
                eot_imgs = torch.cat(adv_imgs_list, dim=0)
            else:
                eot_imgs = adv_diff

            taao_sample_losses = []
            for k in range(eot_imgs.shape[0]):
                img_k = eot_imgs[k:k + 1]
                new_inputs, _, _ = replace_image_in_processed_inputs(
                    fdata["processed_inputs"], img_k, device, verbose=False
                )
                adv_obs = _model.Observation.from_dict(new_inputs)

                adv_img_embs, _, _ = extract_split_features(model, adv_obs)
                adv_action_proxy = adv_img_embs.mean(dim=1)
                clean_action_proxy = fdata["clean_img_embs"].mean(dim=1).to(device)

                if cfg.attack_mode == "untargeted":
                    taao_k = -F.mse_loss(adv_action_proxy, clean_action_proxy.detach())
                elif cfg.attack_mode == "targeted":
                    target_proxy = -clean_action_proxy.detach()
                    taao_k = F.mse_loss(adv_action_proxy, target_proxy)
                else:
                    raise ValueError(f"Unsupported attack_mode: {cfg.attack_mode}")
                taao_sample_losses.append(taao_k)

            taao_loss = torch.stack(taao_sample_losses).mean()

            # frame_weights already sum to one. Dividing the whole expression
            # by num_frames again would suppress the attack term by another
            # factor of N while leaving the accumulated naturalness term at
            # its mean, making the regularizer dominate for multi-frame runs.
            weighted_taao = frame_weights[t] * taao_loss
            weighted_nat = lambda_nat * nat_loss / valid_frame_count
            total_loss = weighted_taao + weighted_nat
            total_loss.backward()

            avg_taao  += weighted_taao.item()
            avg_nat   += nat_loss.item() / valid_frame_count
            avg_total += total_loss.item()
            valid += 1

        if valid > 0:
            loss_history.append(avg_total)

        grad       = renderer.adv_vc_noise.grad
        g_norm     = grad.norm().item() if grad is not None else 0.0
        current_lr = (
            optimizer.param_groups[0]["lr"] if optimizer is not None else cfg.attack_lr
        )
        with open(grad_log_path, "a") as f:
            f.write(f"{i:02d} | {avg_total:.6f} | {avg_taao:.6f} | {avg_nat:.6f} "
                    f"| {g_norm:.6e} | {current_lr:.6e}\n")

        if optimizer_mode == "sign_pgd":
            # Match the OpenVLA/OFT attack: fixed-step sign update. This keeps
            # progress visible even when the model feature gradient is tiny.
            with torch.no_grad():
                renderer.get_texture_param().sub_(
                    cfg.attack_lr * renderer.get_texture_param().grad.sign()
                )
                renderer.get_texture_param().clamp_(-5.0, 5.0)
        else:
            optimizer.step()
            scheduler.step()
        iterator.set_postfix(TAAO=f"{avg_taao:.3f}", Nat=f"{avg_nat:.4f}", GNorm=f"{g_norm:.4f}")

    if cfg.save_attack_artifacts:
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            os.path.join(save_dir, f"Ep{episode_idx}_Texture_Noise.pt"),
        )
        baked_texture = renderer.get_baked_adv_texture()
        baked_texture_np = (
            baked_texture.squeeze(0).detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(baked_texture_np).save(
            os.path.join(save_dir, f"Ep{episode_idx}_Adversarial_Texture.png")
        )
        np.save(os.path.join(save_dir, f"Ep{episode_idx}_loss_history.npy"),
                np.array(loss_history))

        print("\n[诊断] 最终图像特征偏移量（越大表示攻击越有效）:")
        with torch.no_grad():
            for t, fdata in enumerate(frame_data):
                if fdata["mvp"] is not None:
                    adv_final = render_and_composite(
                        renderer, fdata["bg_tensor"], fdata["mvp"],
                        resolution=(RENDER_RES, RENDER_RES),
                        model_rot=fdata.get("model_rot"),
                    )
                    new_inputs, _, _ = replace_image_in_processed_inputs(
                        fdata["processed_inputs"], adv_final, device
                    )
                    adv_obs = _model.Observation.from_dict(new_inputs)
                    adv_img_embs, _, _ = extract_split_features(model, adv_obs)
                    delta = (adv_img_embs - fdata["clean_img_embs"]).norm(dim=-1).mean().item()
                    print(f"  帧 {t}: 图像特征 L2 偏移 = {delta:.4f}")

    return env, loss_history


def evaluate_adversarial_policy(env, policy, renderer, task_description,
                                  initial_state, search_keywords_list: List[List[str]],
                                  max_steps=400):
    # Compatibility wrapper: texture injection is no longer used.
    return evaluate_dual_renderer_policy(
        env,
        policy,
        renderer,
        task_description,
        initial_state,
        search_keywords_list,
        use_adversarial=True,
        max_steps=max_steps,
    )

    print("\n--- 对抗环境 Rollout 评估 ---")
    env.reset()
    obs = env.set_init_state(initial_state)
    sim = env.env.sim
    sim.forward()

    tex_id = find_object_tex_id(sim, search_keywords_list)
    mujoco_tex_injected = False
    tex_restore_info    = None

    if tex_id >= 0:
        baked      = renderer.get_baked_adv_texture()
        adv_tex_np = (baked.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        orig_data, offset, n_bytes = replace_mujoco_texture(sim, tex_id, adv_tex_np)
        tex_restore_info    = (orig_data, offset, n_bytes)
        mujoco_tex_injected = True
        print(f"[评估] MuJoCo 纹理注入成功（tex_id={tex_id}）")
    else:
        print("[评估] ⚠️ 未找到目标纹理 ID，降级为 nvdiffrast 叠加方案")

    device     = policy._pytorch_device
    RENDER_RES = 256
    success    = False
    frames     = []

    try:
        for step in tqdm.tqdm(range(max_steps), desc="评估中"):
            if mujoco_tex_injected:
                img_np  = get_libero_image(obs, RENDER_RES)
                raw_inp = build_raw_inputs(obs, task_description)
            else:
                img_np = get_libero_image(obs, RENDER_RES)
                model_matrix, body_id, _ = get_target_model_matrix(env, search_keywords_list)
                mvp = build_mvp(env, model_matrix, resolution=(RENDER_RES, RENDER_RES)) if body_id != -1 else None
                if mvp is not None:
                    hidden = hide_object_geoms(sim, search_keywords_list)
                    sim.forward()
                    obs_h  = sim.render(camera_name="agentview", height=RENDER_RES, width=RENDER_RES)
                    bg_np  = obs_h[::-1, ::-1].copy()
                    restore_object_geoms(sim, hidden)
                    bg = (torch.from_numpy(bg_np).float().to(device) / 255.0
                          ).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        adv    = render_and_composite(renderer, bg, mvp, resolution=(RENDER_RES, RENDER_RES))
                        img_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                raw_inp = build_raw_inputs(obs, task_description, adv_image_np=img_np)

            frames.append(img_np)
            with torch.no_grad():
                result = policy.infer(raw_inp)
                action = result["actions"] if isinstance(result, dict) else result
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim == 3:
                    action = action[0, 0]
                elif action.ndim == 2:
                    action = action[0]

            obs, _, _, _ = env.step(action.tolist())
            if env.env._check_success():
                print(f"\n[结果] 第 {step} 步成功 —— 攻击失败 ❌")
                success = True
                break
    finally:
        if mujoco_tex_injected and tex_restore_info is not None:
            restore_mujoco_texture(sim, tex_id, *tex_restore_info)
            print("[评估] MuJoCo 纹理已恢复")

    if not success:
        print("\n[结果] 达到最大步数，任务失败 —— 攻击成功 ✅")
    return success, frames


def evaluate_dual_renderer_policy(
    env,
    policy,
    renderer,
    task_description,
    initial_state,
    search_keywords_list: List[List[str]],
    *,
    use_adversarial: bool,
    max_steps=400,
    num_steps_wait=10,
    replan_steps=5,
):
    """Evaluate MuJoCo-clean or nvdiffrast-adversarial observations."""
    mode = "adversarial/nvdiffrast" if use_adversarial else "clean/MuJoCo"
    print(f"\n--- PI0 rollout: {mode} ---")

    env.reset()
    obs = env.set_init_state(initial_state)
    env.env.sim.forward()

    device = policy._pytorch_device
    render_res = 256
    success = False
    frames = []
    policy_frames = []
    wrist_policy_frames = []
    action_plan = deque()
    sim = env.env.sim
    injected_texture = None

    if use_adversarial:
        _, initial_body_id, _ = get_target_model_matrix(
            env, search_keywords_list
        )
        texture_id = find_target_texture_id(
            sim, initial_body_id, search_keywords_list
        )
        if texture_id < 0:
            raise RuntimeError(
                "Could not find the target MuJoCo texture required for rollout video"
            )
        with torch.no_grad():
            baked_texture = renderer.get_baked_adv_texture()
            baked_texture_np = (
                baked_texture[0].detach().cpu().numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
        original, offset, byte_count = inject_mujoco_texture(
            sim, texture_id, baked_texture_np
        )
        injected_texture = (texture_id, original, offset, byte_count)
        print(f"[evaluation] adversarial texture injected into MuJoCo tex_id={texture_id}")

    for step in tqdm.tqdm(
        range(max_steps + num_steps_wait), desc=f"rollout {mode}"
    ):
        if step < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done or env.env._check_success():
                success = True
                break
            continue

        # This is the video frame. In adversarial mode MuJoCo has already loaded
        # the baked adversarial texture, so the saved video is 100% MuJoCo render.
        mujoco_video_frame = get_libero_image(obs, render_res)
        model_img_np = mujoco_video_frame
        wrist_input = None

        if use_adversarial:
            model_matrix, body_id, _ = get_target_model_matrix(
                env, search_keywords_list
            )
            if body_id != -1:
                bg_np = render_mujoco_background_without_target(
                    env,
                    body_id,
                    render_res,
                    search_keywords_list=search_keywords_list,
                )
                bg_tensor = (
                    torch.from_numpy(bg_np).float().to(device) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                mvp = build_mvp(
                    env, model_matrix, resolution=(render_res, render_res)
                )
                with torch.no_grad():
                    adv_tensor = render_and_composite(
                        renderer,
                        bg_tensor,
                        mvp,
                        resolution=(render_res, render_res),
                        model_rot=model_matrix[:3, :3],
                    )
                model_img_np = (
                    adv_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0
                ).round().clip(0, 255).astype(np.uint8)
            else:
                print("[warning] target body not found; using MuJoCo frame")

        # The policy sees model_img_np (dual renderer in adversarial mode), while
        # the video always records the complete MuJoCo camera render.
        frames.append(mujoco_video_frame.copy())
        policy_frames.append(model_img_np.copy())

        if not action_plan:
            wrist_input = None
            if use_adversarial:
                wrist_camera = "robot0_eye_in_hand"
                try:
                    sim.model.camera_name2id(wrist_camera)
                    # Keep the complete MuJoCo scene, including the plate.
                    # Hide only the target and replace it with the nvdiffrast
                    # adversarial rendering for the wrist policy input.
                    wrist_bg_np = render_mujoco_background_without_target(
                        env,
                        body_id,
                        render_res,
                        search_keywords_list=search_keywords_list,
                        camera_name=wrist_camera,
                    )
                    wrist_bg = (
                        torch.from_numpy(wrist_bg_np).float().to(device) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0)
                    wrist_mvp = build_mvp(
                        env,
                        model_matrix,
                        resolution=(render_res, render_res),
                        camera_name=wrist_camera,
                    )
                    with torch.no_grad():
                        wrist_adv = render_and_composite(
                            renderer,
                            wrist_bg,
                            wrist_mvp,
                            resolution=(render_res, render_res),
                            model_rot=model_matrix[:3, :3],
                        )
                    wrist_input = (
                        wrist_adv[0].permute(1, 2, 0).cpu().numpy() * 255.0
                    ).round().clip(0, 255).astype(np.uint8)
                except Exception as exc:
                    print(
                        "[warning] wrist nvdiffrast view unavailable; "
                        f"using MuJoCo wrist image: {exc}"
                    )
            raw_inputs = build_raw_inputs(
                obs,
                task_description,
                adv_image_np=model_img_np,
                adv_wrist_image_np=wrist_input,
                resolution=render_res,
            )
            with torch.no_grad():
                result = policy.infer(raw_inputs)
            action_chunk = result["actions"] if isinstance(result, dict) else result
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.detach().cpu().numpy()
            action_chunk = np.asarray(action_chunk)
            if action_chunk.ndim == 3:
                action_chunk = action_chunk[0]
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            if len(action_chunk) == 0:
                raise RuntimeError("PI0 returned an empty action chunk")
            action_plan.extend(action_chunk[:min(replan_steps, len(action_chunk))])

        # Retained for rollout diagnostics, although policy-input videos are
        # no longer written to disk.
        wrist_policy_frames.append(
            wrist_input.copy()
            if wrist_input is not None
            else np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        )

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        if done or env.env._check_success():
            success = True
            print(f"[result] {mode} succeeded at step {step - num_steps_wait}")
            break

    if not success:
        print(f"[result] {mode} failed after {max_steps} policy steps")
    if injected_texture is not None:
        texture_id, original, offset, byte_count = injected_texture
        restore_mujoco_texture(
            sim, texture_id, original, offset, byte_count
        )
        print(f"[evaluation] restored MuJoCo texture tex_id={texture_id}")
    return success, frames, policy_frames, wrist_policy_frames


@dataclass
class GenerateConfig:
    pretrained_checkpoint: str = (
        "/data/shaozhen.liu/model/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
    )
    # draccus 0.8.0 cannot decode typing.Literal values from the CLI.
    # Keep these as strings and validate them explicitly at startup instead.
    model_config: str = "pi0_libero"
    run_mode: str = "all"
    load_texture_path: Optional[str] = None

    object_name:           str          = "akita_black_bowl"
    override_mesh_path:    Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path:     Optional[str] = None

    attack_iters: int   = 5000
    attack_lr:    float = 0.05
    optimizer_mode: str = "sign_pgd"
    num_frames:   int   = 20
    lambda_feat:  float = 1.0
    lambda_nat:   float = 0.01

    save_attack_artifacts: bool = True
    local_log_dir: str = "/data/shaozhen.liu/code/tex3d/experiments/pi0_attacks"
    latent_encoder_config: str = (
        "/data/shaozhen.liu/code/tex3d-deps/taming-transformers/configs/"
        "vqgan_imagenet_f16_16384.yaml"
    )
    latent_encoder_ckpt: str = (
        "/data/shaozhen.liu/code/tex3d-deps/taming-transformers/checkpoints/"
        "vqgan_imagenet_f16_16384.ckpt"
    )
    tau:                   float = 1.0
    attack_mode:           str   = "untargeted"  # "untargeted" or "targeted"
    target_action_mode:    str   = "sign_flip_xyz"
    use_eot:               bool  = False
    eot_num_samples:       int   = 4
    eot_rot_std_deg:       float = 5.0
    eot_trans_std:         float = 0.02
    eot_scale_min:         float = 0.9
    eot_scale_max:         float = 1.1
    eval_max_steps:        int   = 400
    eval_num_trials:       int   = 1
    eval_all_tasks:        bool  = False
    # If set, test only this task id. It takes precedence over eval_all_tasks.
    eval_task_id:          Optional[int] = None
    num_steps_wait:        int   = 10
    # Execute one action from each PI0 50-step prediction, then replan from
    # the newest observation. The model horizon itself remains 50.
    replan_steps:          int   = 1
    evaluate_clean:        bool  = True


@draccus.wrap()
def run_whitebox_attack(cfg: GenerateConfig):
    valid_model_configs = {"pi0_libero", "pi05_libero"}
    if cfg.model_config not in valid_model_configs:
        raise ValueError(
            f"未知 model_config '{cfg.model_config}'，可选: "
            f"{sorted(valid_model_configs)}"
        )

    valid_run_modes = {"train", "adv_test", "clean_test", "all"}
    if cfg.run_mode not in valid_run_modes:
        raise ValueError(
            f"未知 run_mode '{cfg.run_mode}'，可选: {sorted(valid_run_modes)}"
        )

    if cfg.object_name not in OBJECTS:
        raise ValueError(f"未知物体 '{cfg.object_name}'，可选: {list(OBJECTS.keys())}")
    obj_cfg = OBJECTS[cfg.object_name]

    mesh_path    = cfg.override_mesh_path    or obj_cfg["mesh"]
    texture_path = cfg.override_texture_path or obj_cfg["texture"]
    xml_path     = cfg.override_xml_path     or obj_cfg["xml"]
    task_suite   = obj_cfg["task_suite"]
    task_id      = obj_cfg["task_id"]
    search_kw    = obj_cfg["search"]

    print(f"[配置] 物体: {cfg.object_name}")
    print(f"[配置] mesh:    {mesh_path}")
    print(f"[配置] texture: {texture_path}")
    print(f"[配置] xml:     {xml_path}")
    print(f"[配置] task_suite={task_suite}, task_id={task_id}")
    print(f"[配置] 搜索关键词: {search_kw}")

    scale_xyz = parse_mesh_scale(xml_path)
    print(f"[配置] mesh scale (from xml): {scale_xyz}")

    print(
        f"\n加载 {cfg.model_config} checkpoint: "
        f"{cfg.pretrained_checkpoint}"
    )
    config = training_config.get_config(cfg.model_config)
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.disable = True

    policy = policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=cfg.pretrained_checkpoint,
    )
    policy._model.float()
    default_sample_kwargs = dict(policy._sample_kwargs)

    assert policy._is_pytorch_model, "需要 PyTorch 版本的 checkpoint！"

    device = policy._pytorch_device

    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(device),
        scale_xyz=scale_xyz,
    ).to(device)

    benchmark_dict    = benchmark.get_benchmark_dict()
    task_suite_obj    = benchmark_dict[task_suite]()
    train_task        = task_suite_obj.get_task(task_id)
    train_init_states = task_suite_obj.get_task_init_states(task_id)
    # The description is already stored on the task. Creating an environment
    # merely to read it leaks an offscreen EGL context until interpreter exit.
    train_task_desc = train_task.language

    DATE_TIME    = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(cfg.local_log_dir, f"attack_{cfg.object_name}_{DATE_TIME}")

    if cfg.run_mode != "all":
        mode_dir_name = {
            "train": "train",
            "adv_test": "adv_test",
            "clean_test": "clean_test",
        }[cfg.run_mode]
        artifact_dir = os.path.join(
            cfg.local_log_dir,
            f"{mode_dir_name}_{cfg.object_name}_{DATE_TIME}",
        )
        os.makedirs(artifact_dir, exist_ok=True)

        if cfg.run_mode == "train":
            policy._sample_kwargs["num_steps"] = 1
            print("[train] PI0 denoising steps: 1")
            print(f"[train] task: {train_task_desc}")
            env, _ = train_adversarial_texture_feature_attack(
                cfg,
                policy,
                renderer,
                train_init_states[0],
                train_task,
                train_task_desc,
                artifact_dir,
                episode_idx=0,
                search_keywords_list=search_kw,
                num_iters=cfg.attack_iters,
            )
            print(f"[train] artifacts saved to: {artifact_dir}")
            try:
                _close_libero_env(env)
            except Exception:
                pass
            return

        if cfg.run_mode == "adv_test":
            policy._sample_kwargs["num_steps"] = 1
            print("[adv_test] PI0 denoising steps: 1")
            if not cfg.load_texture_path:
                raise ValueError(
                    "adv_test requires --load_texture_path pointing to "
                    "Ep0_Texture_Noise.pt"
                )
            loaded_noise = torch.load(
                cfg.load_texture_path, map_location=device, weights_only=True
            )
            if isinstance(loaded_noise, dict):
                loaded_noise = loaded_noise.get(
                    "adv_vc_noise", loaded_noise.get("texture_noise")
                )
            if loaded_noise is None or loaded_noise.shape != renderer.adv_vc_noise.shape:
                raise ValueError(
                    "Loaded texture noise shape does not match renderer: "
                    f"loaded={getattr(loaded_noise, 'shape', None)}, "
                    f"expected={renderer.adv_vc_noise.shape}"
                )
            renderer.adv_vc_noise.data.copy_(loaded_noise.to(device))
            use_adversarial = True
            video_name = "Adversarial_Rollout.mp4"
        else:
            policy._sample_kwargs = dict(default_sample_kwargs)
            use_adversarial = False
            video_name = "Clean_MuJoCo_Rollout.mp4"

        if cfg.eval_num_trials < 1:
            raise ValueError("eval_num_trials must be at least 1")
        if cfg.eval_task_id is not None:
            if not 0 <= cfg.eval_task_id < task_suite_obj.n_tasks:
                raise ValueError(
                    f"eval_task_id must be in [0, {task_suite_obj.n_tasks - 1}], "
                    f"got {cfg.eval_task_id}"
                )
            eval_task_ids = [cfg.eval_task_id]
        elif cfg.eval_all_tasks:
            eval_task_ids = list(range(task_suite_obj.n_tasks))
        else:
            eval_task_ids = [task_id]
        total_successes = 0
        total_episodes = 0
        task_results = []
        result_path = os.path.join(artifact_dir, "evaluation_result.txt")
        with open(result_path, "w", encoding="utf-8") as result_file:
            result_file.write(f"run_mode={cfg.run_mode}\n")
            result_file.write(f"object_name={cfg.object_name}\n")
            result_file.write(f"task_suite={task_suite}\n")
            result_file.write(f"eval_task_ids={eval_task_ids}\n")
            result_file.write(f"eval_num_trials={cfg.eval_num_trials}\n")
            if cfg.load_texture_path:
                result_file.write(f"texture_noise={cfg.load_texture_path}\n")
            result_file.write("\n# Live results (updated after every episode)\n")
            result_file.flush()
            os.fsync(result_file.fileno())

        for eval_task_id in eval_task_ids:
            eval_task = task_suite_obj.get_task(eval_task_id)
            eval_init_states = task_suite_obj.get_task_init_states(eval_task_id)
            eval_task_desc = eval_task.language
            if cfg.eval_num_trials > len(eval_init_states):
                raise ValueError(
                    f"Task {eval_task_id} only has {len(eval_init_states)} init "
                    f"states, but eval_num_trials={cfg.eval_num_trials}"
                )
            task_successes = 0
            task_dir = os.path.join(artifact_dir, f"task_{eval_task_id:02d}")
            os.makedirs(task_dir, exist_ok=True)

            for episode_idx in range(cfg.eval_num_trials):
                print(
                    f"[test] task={eval_task_id}/{eval_task_ids[-1]} "
                    f"episode={episode_idx + 1}/{cfg.eval_num_trials}: "
                    f"{eval_task_desc}"
                )
                eval_env, _ = _get_libero_env(eval_task, resolution=256)
                try:
                    success, frames, policy_frames, wrist_policy_frames = evaluate_dual_renderer_policy(
                        eval_env,
                        policy,
                        renderer,
                        eval_task_desc,
                        eval_init_states[episode_idx],
                        search_keywords_list=search_kw,
                        use_adversarial=use_adversarial,
                        max_steps=cfg.eval_max_steps,
                        num_steps_wait=cfg.num_steps_wait,
                        replan_steps=cfg.replan_steps,
                    )
                finally:
                    _close_libero_env(eval_env)

                task_successes += int(success)
                total_successes += int(success)
                total_episodes += 1

                # Record the current success rate immediately, so partial runs
                # still leave an up-to-date result on disk.
                with open(result_path, "a", encoding="utf-8") as result_file:
                    result_file.write(
                        f"task_{eval_task_id} episode={episode_idx + 1}/"
                        f"{cfg.eval_num_trials} success={success} "
                        f"task_rate={task_successes}/{episode_idx + 1} "
                        f"({task_successes / (episode_idx + 1):.6f}) "
                        f"overall={total_successes}/{total_episodes} "
                        f"({total_successes / total_episodes:.6f})\n"
                    )
                    result_file.flush()
                    os.fsync(result_file.fileno())

                if cfg.save_attack_artifacts and frames:
                    import imageio

                    stem = os.path.splitext(video_name)[0]
                    video_path = os.path.join(
                        task_dir,
                        f"episode_{episode_idx:02d}_{stem}_success_{success}.mp4",
                    )
                    labeled_frames = annotate_video_frames(
                        frames, f"{cfg.run_mode} | success={success}"
                    )
                    writer = imageio.get_writer(
                        video_path, fps=15, codec="libx264", quality=8
                    )
                    for frame in labeled_frames:
                        writer.append_data(frame)
                    writer.close()

            task_results.append(
                (eval_task_id, eval_task_desc, task_successes, cfg.eval_num_trials)
            )
            print(
                f"[test] task {eval_task_id} success: "
                f"{task_successes}/{cfg.eval_num_trials} "
                f"({task_successes / cfg.eval_num_trials:.2%})"
            )

        with open(result_path, "a", encoding="utf-8") as result_file:
            result_file.write("\n# Final summary\n")
            for result_task_id, desc, successes, episodes in task_results:
                result_file.write(
                    f"task_{result_task_id}={successes}/{episodes} "
                    f"({successes / episodes:.6f}) | {desc}\n"
                )
            result_file.write(
                f"overall={total_successes}/{total_episodes} "
                f"({total_successes / total_episodes:.6f})\n"
            )
            result_file.flush()
            os.fsync(result_file.fileno())
        print(
            f"[test] overall success: {total_successes}/{total_episodes} "
            f"({total_successes / total_episodes:.2%})"
        )
        print(f"[test] result saved to: {result_path}")
        return

    print(f"\n开始图像特征空间攻击: [{train_task_desc}]")
    policy._sample_kwargs["num_steps"] = 1
    print("[train] PI0 denoising steps: 1")
    env, loss_history = train_adversarial_texture_feature_attack(
        cfg, policy, renderer, train_init_states[0],
        train_task, train_task_desc, artifact_dir, episode_idx=0,
        search_keywords_list=search_kw,
        num_iters=cfg.attack_iters,
    )
    print(f"\n攻击完成，产物保存至: {artifact_dir}")

    clean_success = None
    clean_frames = []
    if cfg.evaluate_clean:
        policy._sample_kwargs = dict(default_sample_kwargs)
        clean_success, clean_frames, clean_policy_frames, clean_wrist_policy_frames = evaluate_dual_renderer_policy(
            env,
            policy,
            renderer,
            train_task_desc,
            train_init_states[0],
            search_keywords_list=search_kw,
            use_adversarial=False,
            max_steps=cfg.eval_max_steps,
            num_steps_wait=cfg.num_steps_wait,
            replan_steps=cfg.replan_steps,
        )

    policy._sample_kwargs["num_steps"] = 1
    print("[adv_test] PI0 denoising steps: 1")
    is_success, frames, policy_frames, wrist_policy_frames = evaluate_dual_renderer_policy(
        env,
        policy,
        renderer,
        train_task_desc,
        train_init_states[0],
        search_keywords_list=search_kw,
        use_adversarial=True,
        max_steps=cfg.eval_max_steps,
        num_steps_wait=cfg.num_steps_wait,
        replan_steps=cfg.replan_steps,
    )
    print(
        f"[evaluation] clean_success={clean_success}, "
        f"adversarial_success={is_success}"
    )

    if cfg.save_attack_artifacts and frames:
        import imageio
        labeled_frames = annotate_video_frames(
            frames, f"adv_test | success={is_success} | MuJoCo"
        )
        mp4_path = os.path.join(
            artifact_dir, f"Adversarial_Rollout_success_{is_success}.mp4"
        )
        writer   = imageio.get_writer(mp4_path, fps=15, codec="libx264", quality=8)
        for frame in labeled_frames:
            writer.append_data(frame)
        writer.close()
        if clean_frames:
            clean_mp4_path = os.path.join(artifact_dir, "Clean_MuJoCo_Rollout.mp4")
            clean_writer = imageio.get_writer(
                clean_mp4_path, fps=15, codec="libx264", quality=8
            )
            for frame in clean_frames:
                clean_writer.append_data(frame)
            clean_writer.close()
            print(f"clean rollout video saved: {clean_mp4_path}")
        print(f"录像保存: {mp4_path}")

    try:
        _close_libero_env(env)
    except Exception:
        pass


if __name__ == "__main__":
    run_whitebox_attack()
    os._exit(0)
