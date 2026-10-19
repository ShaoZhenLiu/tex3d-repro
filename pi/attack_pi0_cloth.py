#!/usr/bin/env python
"""π0 通用桌布对抗纹理（TRAP 启发的"攻击面=桌面覆盖物"，Tex3D 协议兼容）。

与 attack_pi0.py 的关系：复用其渲染/注入/评测基础设施，但
  1) 攻击目标从物体 mesh 换成桌面上新增的独立视觉桌布（60×40cm 细分平面）；
  2) 纹理参数化统一为 T_θ(u,v)=clip(T_0+Interp(ε·tanh θ),0,1)，训练与导出走
     同一条 texture_tensor() 路径（消灭 bake/train 不等价，即审计#13 类问题）；
  3) 多任务联合优化一张通用桌布（先均匀抽任务再抽帧），导出一张固定 PNG；
  4) 合成 mask 用 MuJoCo 分割出的"桌布可见区域"，防止把纹理画到前景物体上；
  5) 评测为原生闭环（eval_renderer=mujoco）：注入最终 PNG 后模型直接读 MuJoCo
     观测，clean/adv 统一去噪步数（修审计#7 的不对称）；
  6) 外观约束在纹理空间：λ_ref·MSE(T_θ,T_0)+λ_tv·TV(T_θ−T_0)；
     腕部相机按 Tex3D 设计（issue #3）保持素色桌布视图，v1 不做腕部合成。

子命令：
  patch --on/--off/--status   场景 XML 加/去桌布（带 .precloth 备份）
  smoke [--smoke-task N]       F1-F3 三门冒烟 + 渲染增益标定
  train                        多任务联合训练（产出 Universal_Cloth.png）
  eval --condition {original,plain,adv} [--tasks 0,1,2,8|heldout]
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PI_DIR = os.path.dirname(os.path.abspath(__file__))
if PI_DIR not in sys.path:
    sys.path.insert(0, PI_DIR)

from attack_pi0 import (  # noqa: E402
    DifferentiableRenderer,
    _close_libero_env,
    _get_libero_env,
    _model,
    build_mvp,
    build_raw_inputs,
    extract_split_features,
    find_target_texture_id,
    get_libero_dummy_action,
    get_libero_image,
    get_processed_inputs_and_observation,
    get_target_model_matrix,
    inject_mujoco_texture,
    render_mujoco_background_without_target,
    replace_image_in_processed_inputs,
    restore_mujoco_texture,
)

import nvdiffrast.torch as dr  # noqa: E402

try:
    from openpi.policies import policy_config
    from openpi.training import config as training_config
except ImportError:
    raise ImportError("请在 tex3d-pi0 venv（便携 glibc 包装器）下运行本脚本")

from libero.libero import benchmark  # noqa: E402

TEX3D_ROOT = "/data/shaozhen.liu/code/tex3d"
CLOTH_ASSET_DIR = (
    "/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth"
)
_SCENES_DIR = (
    "/data/shaozhen.liu/code/tex3d-deps/libero-pi/libero/libero/assets/scenes"
)
# (场景 XML, 桌布 body z)：桌面场景贴桌面(运行时仍由 SurfaceAdapter 精对齐)，
# floor 场景贴地。多场景覆盖 libero_10 等混合套件。
SCENE_XMLS = [
    (os.path.join(_SCENES_DIR, "libero_tabletop_base_style.xml"), 0.905),
    (os.path.join(_SCENES_DIR, "libero_kitchen_tabletop_base_style.xml"), 0.905),
    (os.path.join(_SCENES_DIR, "libero_living_room_tabletop_base_style.xml"), 0.905),
    (os.path.join(_SCENES_DIR, "libero_study_base_style.xml"), 0.905),
    (os.path.join(_SCENES_DIR, "libero_coffee_table_base_style.xml"), 0.905),
    (os.path.join(_SCENES_DIR, "libero_floor_base_style.xml"), 0.005),
]
SCENE_XML = SCENE_XMLS[0][0]
SCENE_XML_BACKUP = SCENE_XML + ".precloth"
CLOTH_MESH = os.path.join(CLOTH_ASSET_DIR, "tablecloth.obj")
CLOTH_T0 = os.path.join(CLOTH_ASSET_DIR, "cloth_base.png")
GAIN_JSON = os.path.join(CLOTH_ASSET_DIR, "cloth_gain.json")
DEFAULT_CKPT = "/data/shaozhen.liu/model/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
ARTIFACT_ROOT = os.path.join(TEX3D_ROOT, "experiments", "pi0_cloth")

CLOTH_SEARCH = [["tablecloth"]]
TRAIN_POOL_TASKS = [0, 1, 2, 8]
HELDOUT_TASKS = [5, 6, 7, 9]


# --------------------------------------------------------------------------
# 场景 XML 补丁（文件级，进程内唯一；带备份/恢复）
# --------------------------------------------------------------------------
def patch_status():
    return "patched" if all(os.path.exists(x + ".precloth")
                            for x, _ in SCENE_XMLS) else "clean"


def patch_scene(on: bool):
    if on:
        if patch_status() == "patched":
            return
        for xml, cloth_z in SCENE_XMLS:
            shutil.copy2(xml, xml + ".precloth")
            tree = ET.parse(xml)
            root = tree.getroot()
            asset = root.find("asset")
            worldbody = root.find("worldbody")
            if asset is None or worldbody is None:
                raise RuntimeError(f"场景 XML 结构异常：{xml}")
            asset.append(ET.Element("mesh", {
                "file": "../tablecloth/tablecloth.obj", "name": "tablecloth_mesh",
                "inertia": "shell"}))
            asset.append(ET.Element("texture", {
                "file": "../tablecloth/cloth_base.png", "type": "2d",
                "name": "tex-tablecloth"}))
            asset.append(ET.Element("material", {
                "name": "tablecloth_mat", "texture": "tex-tablecloth",
                "texuniform": "false", "reflectance": "0.0",
                "shininess": "0.0", "specular": "0.0"}))
            xy = "0.4 0" if cloth_z < 0.5 else "0 0"  # floor 场景布铺在机器人前方地面
            body = ET.SubElement(worldbody, "body", {
                "name": "tablecloth", "pos": f"{xy} {cloth_z}"})
            ET.SubElement(body, "geom", {
                "name": "tablecloth_visual", "type": "mesh",
                "mesh": "tablecloth_mesh", "material": "tablecloth_mat",
                "contype": "0", "conaffinity": "0", "group": "1"})
            tree.write(xml)
        print(f"[patch] 桌布已写入 {len(SCENE_XMLS)} 个场景（备份 .precloth；"
              f"桌面场景 z 由 SurfaceAdapter 运行时对齐）")
    else:
        if patch_status() == "clean":
            return
        for xml, _ in SCENE_XMLS:
            bak = xml + ".precloth"
            if os.path.exists(bak):
                shutil.move(bak, xml)
        print("[patch] 全部场景 XML 已还原为无桌布版本")


def ensure_patch(on: bool):
    cur = patch_status()
    want = "patched" if on else "clean"
    if cur != want:
        patch_scene(on)


def align_cloth_to_table(env, lift=0.005, floor_mode=False):
    """SurfaceAdapter：运行时把桌布对齐真实桌面（arena 会改写桌子尺寸/位置，
    场景 XML 里的静态坐标不可信）。读 table_top site，把布摆在桌面上方 lift 米。
    floor_mode=True（地面操作套件）：直接用 XML 静态贴地位置。"""
    sim = env.env.sim
    table_top = None
    for s in range(sim.model.nsite):
        try:
            name = sim.model.site_id2name(s)
        except Exception:
            name = None
        if name == "table_top":
            table_top = s
            break
    _, cloth_bid, _ = get_target_model_matrix(env, CLOTH_SEARCH)
    if cloth_bid == -1:
        raise RuntimeError("SurfaceAdapter：找不到桌布 body")
    sim.forward()
    if floor_mode:
        # 地面操作套件：布铺在动态物体的质心处（盖住大多数物体，天然靠机械臂
        # 工作区），高度贴地；夹紧在可达范围内。
        cen, _, _ = _scene_object_bbox_xy(sim, cloth_bid)
        if cen is not None:
            cx = float(np.clip(cen[0], -0.75, 0.75))
            cy = float(np.clip(cen[1], -0.75, 0.75))
            sim.model.body_pos[cloth_bid, 0] = cx
            sim.model.body_pos[cloth_bid, 1] = cy
            sim.model.body_pos[cloth_bid, 2] = 0.005
            sim.forward()
        return sim.model.body_pos[cloth_bid].copy()
    if table_top is not None:
        pos = np.array(sim.data.site_xpos[table_top], dtype=float)
    else:
        # 泛化回退（射线网格版）：在场景 XY 平面撒 21×21 垂直射线网，
        # 找"高度一致的最大平坦高台"(0.25-1.2m)作为桌面——xy 与 z 全部
        # 从视觉几何直接测得。教训：碰撞盒的 z 可差 20cm、xy 也可偏移，
        # 一律不可信；桌上物体(框/碗)形成的小平台会被"最大平台"逻辑排除。
        # floor 等无高台场景沿用 XML 静态位置（贴地）。
        import mujoco as _mj
        m = getattr(sim.model, "_model", sim.model)
        d_ = getattr(sim.data, "_data", sim.data)
        cells = []
        for gx in np.arange(-1.2, 1.21, 0.12):
            for gy in np.arange(-1.2, 1.21, 0.12):
                pnt = np.array([gx, gy, 1.8])
                try:
                    gid = np.zeros(1, dtype=np.int32)
                    dist = _mj.mj_ray(m, d_, pnt, np.array([0.0, 0.0, -1.0]),
                                      None, 1, int(cloth_bid), gid)
                except Exception:
                    dist = -1.0
                if dist >= 0:
                    z = 1.8 - float(dist)
                    if 0.25 <= z <= 1.2:
                        cells.append((float(gx), float(gy), z))
        if len(cells) < 8:
            print(f"[align] 无 table_top site 且高台射线不足({len(cells)})，沿用静态位置")
            return sim.model.body_pos[cloth_bid].copy()
        # ±1.5cm 容差聚类取最大平台（0.1m 分桶会把布压沉 3-4cm，弃用）
        zs_sorted = sorted(c[2] for c in cells)
        groups, cur = [], [zs_sorted[0]]
        for z in zs_sorted[1:]:
            if z - cur[-1] <= 0.015:
                cur.append(z)
            else:
                groups.append(cur); cur = [z]
        groups.append(cur)
        grp_zs = max(groups, key=len)
        z_top = float(np.median(grp_zs))
        thr = z_top + 0.02
        grp = [c for c in cells if abs(c[2] - z_top) <= 0.02]
        xs = np.array([c[0] for c in grp]); ys = np.array([c[1] for c in grp])
        cx, cy = float(xs.mean()), float(ys.mean())
        cx = float(np.clip(cx, np.percentile(xs, 10), np.percentile(xs, 90)))
        cy = float(np.clip(cy, np.percentile(ys, 10), np.percentile(ys, 90)))
        print(f"[align] 无 table_top site，射线网格：平台 z={z_top:.3f} "
              f"(n={len(grp)}/{len(cells)})，布心=({cx:.2f},{cy:.2f})")
        pos = np.array([cx, cy, z_top], dtype=float)
    sim.model.body_pos[cloth_bid, 0] = pos[0]
    sim.model.body_pos[cloth_bid, 1] = pos[1]
    sim.model.body_pos[cloth_bid, 2] = pos[2] + lift
    sim.forward()
    return sim.model.body_pos[cloth_bid].copy()


# --------------------------------------------------------------------------
# 桌布渲染器：统一 texture_tensor()/export_texture()
# --------------------------------------------------------------------------
class ClothRenderer(DifferentiableRenderer):
    """在共享 T_0 上加规则网格顶点扰动；训练与导出共用 texture_tensor()。"""

    def __init__(self, mesh_path=CLOTH_MESH, base_png=CLOTH_T0,
                 device="cuda", epsilon=64.0 / 255.0, init_std=0.01, seed=7):
        super().__init__(
            mesh_path=mesh_path,
            orig_texture_path=base_png,
            device=device,
            scale_xyz=[1.0, 1.0, 1.0],
            pos_offset=[0.0, 0.0, 0.0],
            epsilon=epsilon,
        )
        if init_std > 0:
            g = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                self.adv_vc_noise.copy_(
                    torch.randn(self.adv_vc_noise.shape, generator=g).to(device)
                    * init_std)
        self.gain = torch.ones(3, device=device)
        self.bias = torch.zeros(3, device=device)
        # 逐像素光照校正（H,W,3）：MuJoCo 点光源在布面上有空间渐变，
        # 全局仿射修不了阴影方向，用 T_0 比值图做空间增益。
        self.gain_map = None
        gain_map_path = os.path.join(os.path.dirname(base_png), "cloth_gain_map.npy")
        if os.path.exists(gain_map_path):
            gm = np.load(gain_map_path)
            self.gain_map = torch.from_numpy(gm.astype(np.float32)).to(device)
            print(f"[cloth] 已加载逐像素增益图 {gain_map_path} {tuple(gm.shape)}")
        elif os.path.exists(GAIN_JSON):
            with open(GAIN_JSON) as f:
                d = json.load(f)
            self.gain = torch.tensor(d["gain"], device=device)
            self.bias = torch.tensor(d["bias"], device=device)

    def set_gain(self, gain, bias):
        self.gain = torch.as_tensor(gain, dtype=torch.float32, device=self.device)
        self.bias = torch.as_tensor(bias, dtype=torch.float32, device=self.device)

    def set_gain_map(self, arr_hw3):
        self.gain_map = torch.as_tensor(arr_hw3, dtype=torch.float32,
                                        device=self.device)
        self.gain = torch.ones(3, device=self.device)
        self.bias = torch.zeros(3, device=self.device)

    def _apply_gain(self, lit):
        if self.gain_map is not None:
            return lit * self.gain_map.unsqueeze(0)
        return lit * self.gain.view(1, 1, 1, 3) + self.bias.view(1, 1, 1, 3)

    def texture_tensor(self):
        """可微：返回完整 T_θ = clip(T_0 + UV 光栅化的 ε·tanh θ, 0, 1)。"""
        noise = torch.tanh(self.adv_vc_noise) * self.epsilon
        uv_clip = torch.zeros(self.uv.shape[0], 4, device=self.device)
        uv_clip[:, 0] = self.uv[:, 0] * 2.0 - 1.0
        uv_clip[:, 1] = self.uv[:, 1] * 2.0 - 1.0
        uv_clip[:, 3] = 1.0
        rast, _ = dr.rasterize(self.glctx, uv_clip.unsqueeze(0),
                               self.uv_idx.int(),
                               resolution=(self.texture_res, self.texture_res))
        field, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast,
                                  self.uv_idx.int())
        cover = (rast[..., 3] > 0).float().unsqueeze(-1)
        return torch.clamp(self.orig_texture + field * cover, 0.0, 1.0)

    def export_texture(self):
        """导出：同一 texture_tensor() 路径 + detach + 量化（PIL 方位）。"""
        tex = torch.flip(self.texture_tensor().detach(), dims=[1])
        arr = (tex.squeeze(0).cpu().numpy() * 255.0).round().clip(0, 255)
        return arr.astype(np.uint8)

    def render(self, mvp, resolution=(256, 256), return_clean=False, model_rot=None,
               uv_xform=None):
        tex = self.texture_tensor()
        pos = self.pos + self.pos_offset
        pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())
        rast, _ = dr.rasterize(self.glctx, pos_clip.unsqueeze(0), self.faces,
                               resolution=resolution)
        tex_uv, _ = dr.interpolate(self.uv.unsqueeze(0), rast, self.uv_idx)
        if uv_xform is not None:
            # UV 域 EoT：旋转/镜像/平移纹理采样坐标（模式级位置鲁棒性）
            u, v = tex_uv[..., 0], tex_uv[..., 1]
            nu = uv_xform[0, 0] * u + uv_xform[0, 1] * v + uv_xform[0, 2]
            nv = uv_xform[1, 0] * u + uv_xform[1, 1] * v + uv_xform[1, 2]
            tex_uv = torch.stack([torch.remainder(nu, 1.0),
                                  torch.remainder(nv, 1.0)], dim=-1)
        color = dr.texture(tex.contiguous(), tex_uv.contiguous(),
                           filter_mode="linear")

        vn_world = (model_rot @ self.vn.T).T if model_rot is not None else self.vn
        vn_interp, _ = dr.interpolate(vn_world.unsqueeze(0).contiguous(), rast,
                                      self.faces)
        vn_interp = F.normalize(vn_interp, p=2, dim=-1)
        diffuse = torch.clamp(
            (vn_interp * self.light_dir.view(1, 1, 1, 3)).sum(-1, keepdim=True),
            0.0, 1.0)
        lighting = 0.6 + 0.4 * diffuse
        shaded = torch.clamp(self._apply_gain(color * lighting), 0.0, 1.0)

        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            base = self.orig_texture.expand_as(color)
            clean = torch.clamp(self._apply_gain(base * lighting), 0.0, 1.0)
            return shaded, clean, mask
        return shaded, mask


# --------------------------------------------------------------------------
# 桌布可见 mask：优先 MuJoCo 分割渲染差分，失败降级 RGB 差分
# --------------------------------------------------------------------------
def _render_now(env, resolution=256, camera_name="agentview"):
    sim = env.env.sim
    sim.forward()
    return np.ascontiguousarray(
        sim.render(camera_name=camera_name, height=resolution,
                   width=resolution)[::-1, ::-1])


def _render_seg(env, resolution=256, camera_name="agentview"):
    sim = env.env.sim
    try:
        out = sim.render(camera_name=camera_name, height=resolution,
                         width=resolution, segmentation=True)
    except TypeError:
        return None
    if isinstance(out, tuple) and len(out) == 3:
        return np.asarray(out[2])
    return None


def cloth_visibility_mask(env, cloth_body_id, resolution=256, camera_name="agentview"):
    """桌布当前状态下在该相机里真正可见（未被前景遮挡）的像素 mask（bool）。"""
    sim = env.env.sim
    seg_full = _render_seg(env, resolution, camera_name)

    geom_ids = [g for g in range(sim.model.ngeom)
                if int(sim.model.geom_bodyid[g]) == int(cloth_body_id)]
    hidden = []
    for g in geom_ids:
        rgba = sim.model.geom_rgba[g].copy()
        sim.model.geom_rgba[g, 3] = 0.0
        hidden.append((g, rgba))
    try:
        sim.forward()
        if seg_full is not None:
            seg_hidden = _render_seg(env, resolution, camera_name)
            diff = seg_full != seg_hidden
            mask = np.any(diff, axis=-1) if diff.ndim == 3 else np.asarray(diff)
            if mask.ndim > 2:
                mask = mask[..., 0]
            return mask.astype(bool)
        rgb_hidden = _render_now(env, resolution, camera_name)
    finally:
        for g, rgba in hidden:
            sim.model.geom_rgba[g] = rgba
        sim.forward()

    rgb_full = _render_now(env, resolution, camera_name)
    diff = np.abs(rgb_full.astype(np.int16) - rgb_hidden.astype(np.int16))
    return diff.max(axis=-1) > 10


# --------------------------------------------------------------------------
# 策略加载
# --------------------------------------------------------------------------
def load_policy(ckpt, model_config="pi0_libero", num_denoise=1):
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.disable = True
    config = training_config.get_config(model_config)
    policy = policy_config.create_trained_policy(
        train_config=config, checkpoint_dir=ckpt)
    policy._model.float()
    policy._sample_kwargs = dict(policy._sample_kwargs)
    policy._sample_kwargs["num_steps"] = num_denoise
    assert policy._is_pytorch_model
    return policy


# --------------------------------------------------------------------------
# 帧采集：真实策略轨迹 + 素色桌布已注入
# --------------------------------------------------------------------------
def _scene_object_bbox_xy(sim, cloth_bid):
    """收集场景中动态物体（排除机械臂/桌布）的世界 XY，返回 (质心, x展宽, y展宽)。"""
    xs, ys = [], []
    for b in range(sim.model.nbody):
        try:
            name = sim.model.body_id2name(b) or ""
        except Exception:
            name = ""
        if name.startswith("robot0") or name == "tablecloth" or int(b) == int(cloth_bid):
            continue
        if int(sim.model.body_dofnum[b]) <= 0:
            continue
        p = sim.data.xpos[b]
        xs.append(float(p[0])); ys.append(float(p[1]))
    if not xs:
        return None, 0.0, 0.0
    return (np.array([np.mean(xs), np.mean(ys)]),
            float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))


def jitter_cloth_placement(env, rng=None):
    """EoT 几何抖动（自适应版）：随机 z 轴旋转（顺/逆时针各 0-180°，即 ±180°），
    平移幅度按物体包围盒各轴展宽的 30% 自适应（截断 1-8cm，x/y 独立采样）——
    物体散得开多抖、紧凑少抖。采集期调用。"""
    import numpy.random as npr
    r = rng or npr
    sim = env.env.sim
    _, bid, _ = get_target_model_matrix(env, CLOTH_SEARCH)
    theta = float(r.uniform(-np.pi, np.pi))
    cen, ex, ey = _scene_object_bbox_xy(sim, bid)
    sx = float(np.clip(0.3 * (ex if ex > 0 else 0.2), 0.01, 0.08))
    sy = float(np.clip(0.3 * (ey if ey > 0 else 0.2), 0.01, 0.08))
    dx, dy = float(r.uniform(-sx, sx)), float(r.uniform(-sy, sy))
    q = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    sim.model.body_quat[bid] = q
    sim.model.body_pos[bid, 0] += dx
    sim.model.body_pos[bid, 1] += dy
    sim.forward()


def collect_cloth_frames(policy, task, init_state, num_frames, device,
                         max_steps=400, replan_steps=5, num_steps_wait=10,
                         stride=1, max_frames_ep=None, weighting=None,
                         tau=1.0, ep_tag=0, place_jitter=False,
                         floor_mode=False, jitter_rng=None):
    desc = task.language
    env, _ = _get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    align_cloth_to_table(env, floor_mode=floor_mode)
    if place_jitter:
        jitter_cloth_placement(env, rng=jitter_rng)

    _, cloth_body_id, _ = get_target_model_matrix(env, CLOTH_SEARCH)
    if cloth_body_id == -1:
        _close_libero_env(env)
        raise RuntimeError("未找到桌布 body——场景 XML 未打补丁？")
    tex_id = find_target_texture_id(sim, cloth_body_id, CLOTH_SEARCH)
    if tex_id < 0:
        _close_libero_env(env)
        raise RuntimeError("未找到桌布 MuJoCo 纹理 id")
    t0_arr = np.array(Image.open(CLOTH_T0).convert("RGB"))
    restore_info = inject_mujoco_texture(sim, tex_id, t0_arr)

    model = policy._model
    frames = []
    action_plan = deque()
    obs = None
    # short 模式：攒够 num_frames 即停（v1 行为，帧集中于接近阶段）；
    # full 模式：跑完整条轨迹（至成功/超时），每 stride 步取 1 帧，上限 max_frames_ep
    cap = max_frames_ep if max_frames_ep is not None else num_frames
    try:
        # 预热等待期（dummy 步）刷新 obs 图像，使其带上素色桌布
        obs = env.set_init_state(init_state)
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done or env.env._check_success():
                break

        for step in range(max_steps):
            if (len(frames) < cap
                    and (stride <= 1 or step % stride == 0)):
                bg_np = render_mujoco_background_without_target(
                    env, cloth_body_id, 256, search_keywords_list=CLOTH_SEARCH)
                bg = (torch.from_numpy(bg_np).float().to(device) / 255.0
                      ).permute(2, 0, 1).unsqueeze(0)
                seg_mask = cloth_visibility_mask(env, cloth_body_id, 256)
                model_matrix, _, _ = get_target_model_matrix(env, CLOTH_SEARCH)
                mvp = build_mvp(env, model_matrix, resolution=(256, 256))
                raw_inputs = build_raw_inputs(obs, desc)
                with torch.no_grad():
                    processed_inputs, observation, _ = \
                        get_processed_inputs_and_observation(policy, raw_inputs)
                    clean_img_embs, _, _ = extract_split_features(model, observation)
                frames.append({
                    "bg": bg,
                    "seg": torch.from_numpy(seg_mask.astype(np.float32)
                                            ).to(device).view(1, 1, 256, 256),
                    "mvp": mvp,
                    "processed_inputs": processed_inputs,
                    "clean_img_embs": clean_img_embs.detach(),
                })

            if not action_plan:
                raw_inputs = build_raw_inputs(obs, desc)
                with torch.no_grad():
                    result = policy.infer(raw_inputs)
                chunk = result["actions"] if isinstance(result, dict) else result
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                action_plan.extend(chunk[:min(replan_steps, len(chunk))])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success() or len(frames) >= cap:
                break
    finally:
        restore_mujoco_texture(sim, tex_id, *restore_info)
        _close_libero_env(env)
    # 轨迹内 TAAO 权重：softmax 后 ×n 归一到均值 1（保持与 uniform 相同的损失量级，
    # 否则 Σw=1 会把单帧攻击项压到 ~1/n、破坏与外观正则的配比）
    if weighting == "taao" and len(frames) > 1:
        feats = [f["clean_img_embs"] for f in frames]
        w = compute_frame_weights(feats, tau=tau) * len(frames)
        for f, wi in zip(frames, w):
            f["w"] = float(wi)
        print(f"[pool] ep{ep_tag} TAAO 权重: min={min(f['w'] for f in frames):.3f} "
              f"max={max(f['w'] for f in frames):.3f} 均值=1")
    for f in frames:
        f.setdefault("w", 1.0)
        f["ep"] = ep_tag
    return frames


# --------------------------------------------------------------------------
# 通用训练
# --------------------------------------------------------------------------
def composite_cloth(renderer, frame, uv_xform=None):
    adv_rgba, _ = renderer.render(frame["mvp"], resolution=(256, 256),
                                  uv_xform=uv_xform)
    adv_rgb = adv_rgba.permute(0, 3, 1, 2)
    seg = frame["seg"]
    return torch.clamp(adv_rgb * seg + frame["bg"] * (1.0 - seg), 0.0, 1.0)


def tv_loss(delta):
    return (delta[:, 1:, :, :].abs().mean()
            + delta[:, :, 1:, :].abs().mean()) / 2.0


def compute_frame_weights(latent_features, tau: float):
    """TAAO dynamics-guided frame weights（移植自 attack_pi0.py / attack_openvla_taao.py）：
    v=特征速度, a=特征加速度, s=max(minmax(v),minmax(a)), w=softmax(s/tau)。
    仅在单条轨迹内部计算（多任务池禁止跨轨迹 softmax）。"""
    if len(latent_features) == 0:
        return torch.tensor([], dtype=torch.float32)
    if len(latent_features) == 1:
        return torch.ones(1, dtype=torch.float32)
    feats = [f.detach().float().flatten() for f in latent_features]
    device = feats[0].device
    n = len(feats)
    v = torch.zeros(n, device=device, dtype=torch.float32)
    a = torch.zeros(n, device=device, dtype=torch.float32)
    for t in range(1, n):
        v[t] = torch.norm(feats[t] - feats[t - 1], p=2) / 2.0
    for t in range(1, n):
        a[t] = torch.abs(v[t] - v[t - 1])
    def _minmax(x):
        x_min = torch.min(x); x_max = torch.max(x)
        return (x - x_min) / torch.clamp(x_max - x_min, min=1e-8)
    s = torch.maximum(_minmax(v), _minmax(a))
    return torch.softmax(s / float(max(tau, 1e-6)), dim=0)


def train_universal(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ensure_patch(True)
    policy = load_policy(args.ckpt, num_denoise=1)
    model = policy._model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 帧池获取：--load-pool 优先读磁盘缓存（训练进程零环境、零 MuJoCo/EGL，
    # 内存占用最小）；缓存缺失或未指定时现场采集。--collect-only 采集完落盘即退出，
    # 便于"采集"与"训练"分进程执行，避免峰值叠加。
    pool_file = args.pool_file
    pool = None
    total = 0
    if args.load_pool and os.path.exists(pool_file):
        pool = torch.load(pool_file, map_location=device, weights_only=False)
        total = sum(len(v) for v in pool.values())
        print(f"[pool] 从缓存载入 {pool_file}：{total} 帧，任务 {list(pool.keys())}")
    else:
        suite = benchmark.get_benchmark_dict()[
            getattr(args, "suite", "libero_spatial")]()
        pool = {}
        started = time.time()
        for task_id in args.pool_tasks:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            frames = []
            for ep in range(args.pool_episodes):
                got = collect_cloth_frames(
                    policy, task, init_states[ep], args.frames_per_ep, device,
                    stride=args.stride,
                    max_frames_ep=(args.max_frames_ep
                                   if args.pool_mode == "full" else None),
                    weighting=args.frame_weighting, tau=args.tau, ep_tag=ep,
                    place_jitter=args.place_jitter,
                    floor_mode=(args.suite == "libero_object"),
                    jitter_rng=np.random.RandomState(args.seed + 1000 + ep))
                frames.extend(got)
                print(f"[pool] task {task_id} ep{ep}: +{len(got)} 帧 "
                      f"({time.time() - started:.0f}s)")
            pool[task_id] = frames
            total += len(frames)
        print(f"[pool] 总帧数 {total}，任务 {list(pool.keys())}")
        if args.collect_only:
            os.makedirs(os.path.dirname(pool_file), exist_ok=True)
            torch.save(pool, pool_file)
            print(f"[pool] 帧池已保存到 {pool_file}（--collect-only，进程退出）")
            return
    if total == 0:
        raise RuntimeError("帧池为空")

    renderer = ClothRenderer(device=device, epsilon=args.epsilon,
                             init_std=args.init_std, seed=args.seed).to(device)
    t0_tex = renderer.orig_texture

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        ARTIFACT_ROOT, f"train_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
    log_path = os.path.join(out_dir, "gradient_log.txt")
    with open(log_path, "w") as f:
        f.write("iter | task | attack | ref | tv | grad_norm\n")

    param = renderer.adv_vc_noise
    task_ids = list(pool.keys())
    history = []

    def sample_uv_xform():
        """UV 域 EoT：绕(0.5,0.5)随机旋转（顺/逆时针各 0-180°）+ 50% 镜像
        （随机沿 u 轴水平翻或 v 轴垂直翻）+ x/y 独立的 ±0.04 平移。"""
        if not getattr(args, "uv_eot", False):
            return None
        ang = random.uniform(-np.pi, np.pi)
        c, s = float(np.cos(ang)), float(np.sin(ang))
        A = np.array([[c, -s, 0.5 - 0.5 * c + 0.5 * s],
                      [s, c, 0.5 - 0.5 * s - 0.5 * c]], dtype=np.float32)
        if random.random() < 0.5:
            if random.random() < 0.5:  # 沿 v 轴镜像（水平翻 u）
                M = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], np.float32)
            else:                       # 沿 u 轴镜像（垂直翻 v）
                M = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]], np.float32)
            # 仿射合成须在 3x3 齐次坐标下做（(2,3)@(2,3) 的 matmul 维度不合法）
            A = (np.vstack([M, [0.0, 0.0, 1.0]])
                 @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
        A[0, 2] += random.uniform(-0.04, 0.04)
        A[1, 2] += random.uniform(-0.04, 0.04)
        return torch.from_numpy(A).to(device)

    all_frames = [(tid, fr) for tid in task_ids for fr in pool[tid]]

    # ---------- 扩展目标（导师方案）：J = λ_mse·J_mse + λ_con·J_con + λ_task·J_task − λ_reg·R ----------
    frame_index = {id(fr): i for i, (_, fr) in enumerate(all_frames)}

    def pooled_feat(embs):
        """(1, N, D) 图像 token 均值池化 → 归一化 (D,) 向量。"""
        v = embs[0].mean(dim=0).float()
        return v / (v.norm() + 1e-8)

    # 对比项负样本库：池内全部帧的 clean 特征（采集时已缓存，零额外前向）。
    # 目标 = 最小化 log[exp(sim(z_a, z_own)/τ) / Σ_j exp(sim(z_a, z_j)/τ)]：
    # 把受扰特征推离自身 clean、推向他帧分布 —— 约束方向而非范数。
    con_bank = None
    if getattr(args, "lambda_con", 0.0) > 0:
        con_bank = torch.stack(
            [pooled_feat(fr["clean_img_embs"]) for _, fr in all_frames]).to(device)

    def contrast_loss(adv_embs, fr):
        z_a = pooled_feat(adv_embs).to(con_bank.dtype)
        sims = con_bank @ z_a / args.con_tau
        own = torch.tensor([frame_index[id(fr)]], device=device)
        return F.cross_entropy(sims[None], own)

    # 任务条件项：指令 token 在语言融合后的 Gemma 隐藏状态（前缀全前向，无 suffix，
    # 与 sample_actions 的 prefill 同构）。clean 参考在开跑前一次性缓存（no_grad）。
    clean_task_vec = None
    _task_fwd_ready = False

    def _to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_to_device(v) for v in obj)
        return obj

    def task_fused_hidden(observation):
        """返回 (D,) 指令 token 融合隐藏向量（对图像可微）。"""
        nonlocal _task_fwd_ready
        if not _task_fwd_ready:
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            _task_fwd_ready = True
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(
            observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        if (model.paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = model._prepare_attention_masks_4d(att_2d)
        (prefix_out, _), _ = model.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=pos_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=False)
        prefix_out = prefix_out.to(dtype=torch.float32)
        n_lang = lang_tokens.shape[1]
        lang_hidden = prefix_out[0, -n_lang:]           # 前缀尾部 = 语言段
        m = lang_masks[0].to(prefix_out.dtype)[:, None]
        v = (lang_hidden * m).sum(dim=0) / (m.sum() + 1e-8)
        return v

    if getattr(args, "lambda_task", 0.0) > 0:
        with torch.no_grad():
            clean_task_vec = torch.stack([
                task_fused_hidden(_model.Observation.from_dict(
                    _to_device(fr["processed_inputs"])))
                for _, fr in all_frames]).to(device)
        print(f"[train] 任务条件项 clean 参考已缓存 {tuple(clean_task_vec.shape)}")

    # UV 覆盖日志：顶点梯度幅值累积（33x33 热图，回答"哪个 UV 区域吃到多少梯度"）
    grad_cov = torch.zeros(param.shape[0], device=device)

    for it in range(args.iters):
        tid, fr = random.choice(all_frames)
        param.grad = None
        uv_a = sample_uv_xform()
        adv_img = composite_cloth(renderer, fr, uv_xform=uv_a)
        new_inputs, _, _ = replace_image_in_processed_inputs(
            fr["processed_inputs"], adv_img, device)
        adv_obs = _model.Observation.from_dict(new_inputs)
        adv_embs, _, _ = extract_split_features(model, adv_obs)
        attack_loss = -F.mse_loss(adv_embs, fr["clean_img_embs"])
        fw = float(fr.get("w", 1.0)) if args.frame_weighting == "taao" else 1.0

        loss = fw * getattr(args, "lambda_mse", 1.0) * attack_loss
        con_val = task_val = None
        if con_bank is not None:
            con_val = contrast_loss(adv_embs, fr)
            loss = loss + fw * args.lambda_con * con_val
        if clean_task_vec is not None and (it % args.task_every_k == 0
                                           or it == args.iters - 1):
            z_task = task_fused_hidden(adv_obs)
            task_val = -F.mse_loss(z_task, clean_task_vec[frame_index[id(fr)]])
            loss = loss + fw * args.lambda_task * task_val

        tex = renderer.texture_tensor()
        ref = F.mse_loss(tex, t0_tex)
        tv = tv_loss(tex - t0_tex)
        (loss + args.lambda_ref * ref + args.lambda_tv * tv).backward()

        gnorm = param.grad.norm().item() if param.grad is not None else 0.0
        grad_cov += param.grad.detach().abs().max(dim=-1)[0] if param.grad is not None else 0.0
        with torch.no_grad():
            param.sub_(args.attack_lr * param.grad.sign())
            param.clamp_(-5.0, 5.0)
        history.append(float(attack_loss.item()))
        if it % 50 == 0 or it == args.iters - 1:
            extra = ""
            if con_val is not None:
                extra += f" con={con_val.item():+.4f}"
            if task_val is not None:
                extra += f" task={task_val.item():+.4f}"
            with open(log_path, "a") as f:
                f.write(f"{it:05d} | {tid} | {attack_loss.item():+.5f} | "
                        f"{ref.item():.5f} | {tv.item():.5f} | {gnorm:.3e}{extra}\n")
            print(f"[train] it{it:05d} task{tid} attack={attack_loss.item():+.4f} "
                  f"ref={ref.item():.4f} tv={tv.item():.4f} "
                  f"w={fw:.2f} g={gnorm:.2e}{extra}")
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            Image.fromarray(renderer.export_texture()).save(
                os.path.join(out_dir, "Universal_Cloth.png"))
            torch.save(param.detach().cpu(),
                       os.path.join(out_dir, "Ep0_Cloth_Delta.pt"))
            np.save(os.path.join(out_dir, "loss_history.npy"), np.array(history))
            side = int(round(math.sqrt(param.shape[0])))
            cov = grad_cov.cpu().numpy().reshape(side, side)
            np.save(os.path.join(out_dir, "grad_coverage.npy"), cov)
            lo, hi = float(cov.min()), float(cov.max()) + 1e-12
            vis = (np.log1p(cov - lo) / np.log1p(hi - lo) * 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(out_dir, "grad_coverage.png"))
    print(f"[train] 完成，产物在 {out_dir}")


# --------------------------------------------------------------------------
# 原生闭环评测
# --------------------------------------------------------------------------
def evaluate_native(policy, task, init_state, texture_arr=None, max_steps=400,
                    num_steps_wait=10, replan_steps=5, save_video=None,
                    floor_mode=False):
    desc = task.language
    env, _ = _get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    # original 条件无桌布（XML clean），对齐仅在有桌布的 plain/adv 条件下执行
    if texture_arr is not None:
        align_cloth_to_table(env, floor_mode=floor_mode)

    restore_info = None
    tex_id = -1
    if texture_arr is not None:
        _, cloth_body_id, _ = get_target_model_matrix(env, CLOTH_SEARCH)
        tex_id = find_target_texture_id(sim, cloth_body_id, CLOTH_SEARCH)
        if tex_id < 0:
            _close_libero_env(env)
            raise RuntimeError("评测注入失败：未找到桌布纹理 id（XML 未打补丁？）")
        restore_info = inject_mujoco_texture(sim, tex_id, texture_arr)

    success = False
    frames = []
    action_plan = deque()
    obs = env.set_init_state(init_state)
    try:
        for step in range(max_steps + num_steps_wait):
            if step < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done or env.env._check_success():
                    success = True
                    break
                continue
            if save_video is not None:
                frames.append(get_libero_image(obs, 256))
            if not action_plan:
                raw_inputs = build_raw_inputs(obs, desc)
                with torch.no_grad():
                    result = policy.infer(raw_inputs)
                chunk = result["actions"] if isinstance(result, dict) else result
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk)
                if chunk.ndim == 3:
                    chunk = chunk[0]
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                action_plan.extend(chunk[:min(replan_steps, len(chunk))])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            if done or env.env._check_success():
                success = True
                break
    finally:
        if restore_info is not None:
            restore_mujoco_texture(sim, tex_id, *restore_info)
        _close_libero_env(env)

    if save_video is not None and frames:
        import imageio
        writer = imageio.get_writer(save_video, fps=15, codec="libx264", quality=8)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
    return success


def run_eval(args):
    want_patch = args.condition in ("plain", "adv")
    ensure_patch(want_patch)
    texture_arr = None
    if args.condition == "plain":
        texture_arr = np.array(Image.open(CLOTH_T0).convert("RGB"))
    elif args.condition == "adv":
        if not args.texture:
            raise ValueError("adv 条件需要 --texture Universal_Cloth.png")
        texture_arr = np.array(Image.open(args.texture).convert("RGB"))

    policy = load_policy(args.ckpt, model_config=args.model_config,
                         num_denoise=args.num_denoise)
    suite = benchmark.get_benchmark_dict()[
        getattr(args, "suite", "libero_spatial")]()
    floor_mode = (getattr(args, "suite", "libero_spatial") == "libero_object")

    tasks = HELDOUT_TASKS if args.tasks == "heldout" else \
        [int(t) for t in args.tasks.split(",")]
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    suite_name = getattr(args, "suite", "libero_spatial")
    out_dir = os.path.join(ARTIFACT_ROOT,
                           f"eval_{args.condition}_{suite_name}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "evaluation_result.txt")

    total_success = 0
    total_eps = 0
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"condition={args.condition}\ntexture={args.texture}\n"
                f"num_denoise={args.num_denoise} replan=5 max_steps={args.max_steps}\n\n")
        for tid in tasks:
            task = suite.get_task(tid)
            init_states = suite.get_task_init_states(tid)
            n = min(args.trials, len(init_states))
            succ = 0
            for ep in range(n):
                video = (os.path.join(out_dir, f"task{tid}_ep{ep:02d}.mp4")
                         if ep == 0 and args.save_videos else None)
                ok = evaluate_native(policy, task,
                                     init_states[ep + getattr(args, "init_offset", 0)],
                                     texture_arr,
                                     max_steps=args.max_steps,
                                     num_steps_wait=10, replan_steps=5,
                                     save_video=video, floor_mode=floor_mode)
                succ += int(ok)
                total_success += int(ok)
                total_eps += 1
                line = f"Task: {tid} | Ep: {ep} | Success: {ok}"
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
            f.write(f"task_{tid}: {succ}/{n} ({succ / n:.4f})\n")
            f.flush()
            print(f"[eval] task {tid}: {succ}/{n} ({succ / n:.2%})")
        f.write(f"\noverall robot success: {total_success}/{total_eps} "
                f"({total_success / max(total_eps, 1):.4f})\n")
        f.write("attack failure rate = 1 - above\n")
    print(f"[eval] 结果: {result_path}")


# --------------------------------------------------------------------------
# 冒烟三门 + 增益标定
# --------------------------------------------------------------------------
def run_smoke(args):
    device = "cuda"
    ensure_patch(True)
    print(f"[smoke] 场景状态: {patch_status()}")

    renderer = ClothRenderer(device=device, init_std=0.0, seed=0).to(device)
    # 标定必须从"中性渲染"出发：清掉历史遗留的 gain.json/gain_map，
    # 否则比值图会把旧增益算进去，F3 应用时双重计费。
    renderer.gain = torch.ones(3, device=device)
    renderer.bias = torch.zeros(3, device=device)
    renderer.gain_map = None

    # ---- F1: 零扰动恢复 ----
    exported = renderer.export_texture()
    t0_arr = np.array(Image.open(CLOTH_T0).convert("RGB").resize((256, 256)))
    diff = np.abs(exported.astype(np.int16) - t0_arr.astype(np.int16))
    print(f"[F1] 零扰动导出 vs T_0: max|diff|={diff.max()} mean={diff.mean():.4f}")
    assert diff.max() <= 1, "F1 失败：θ=0 时导出未能恢复底图"

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.smoke_task)
    init_state = suite.get_task_init_states(args.smoke_task)[0]
    env, _ = _get_libero_env(task, resolution=256)
    env.reset()
    env.set_init_state(init_state)
    sim = env.env.sim
    sim.forward()
    cloth_pos = align_cloth_to_table(env)
    print(f"[smoke] SurfaceAdapter 对齐后桌布位姿: {cloth_pos}")
    model_matrix, cloth_body_id, found = get_target_model_matrix(env, CLOTH_SEARCH)
    print(f"[smoke] 桌布 body: id={cloth_body_id} name={found}")
    assert cloth_body_id != -1
    tex_id = find_target_texture_id(sim, cloth_body_id, CLOTH_SEARCH)
    print(f"[smoke] 桌布纹理 id: {tex_id}")
    assert tex_id >= 0

    restore_info = inject_mujoco_texture(sim, tex_id, t0_arr)

    mujoco_full = _render_now(env, 256)
    bg_np = render_mujoco_background_without_target(
        env, cloth_body_id, 256, search_keywords_list=CLOTH_SEARCH)
    seg_mask = cloth_visibility_mask(env, cloth_body_id, 256)
    print(f"[F2] 桌布可见像素: {seg_mask.mean() * 100:.1f}% of frame")
    assert seg_mask.mean() > 0.05, \
        f"F2 失败：桌布可见像素仅 {seg_mask.mean()*100:.2f}%（<5%），检查 mesh 绕序/位置"

    mvp = build_mvp(env, model_matrix, resolution=(256, 256))
    bg = (torch.from_numpy(bg_np).float().to(device) / 255.0
          ).permute(2, 0, 1).unsqueeze(0)
    seg_t = torch.from_numpy(seg_mask.astype(np.float32)).to(device).view(1, 1, 256, 256)

    with torch.no_grad():
        clean_rgba, _, _ = renderer.render(mvp, resolution=(256, 256),
                                           return_clean=True)
    comp_clean = torch.clamp(clean_rgba.permute(0, 3, 1, 2) * seg_t
                             + bg * (1.0 - seg_t), 0.0, 1.0)

    debug_dir = os.path.join(ARTIFACT_ROOT, "smoke_debug")
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(mujoco_full).save(os.path.join(debug_dir, "mujoco_full_T0.png"))
    Image.fromarray(bg_np).save(os.path.join(debug_dir, "bg_cloth_hidden.png"))
    Image.fromarray((seg_mask * 255).astype(np.uint8)).save(
        os.path.join(debug_dir, "seg_mask.png"))
    Image.fromarray((comp_clean[0].permute(1, 2, 0).cpu().numpy() * 255)
                    .round().clip(0, 255).astype(np.uint8)).save(
        os.path.join(debug_dir, "composite_clean.png"))

    # ---- 光照标定 v2：逐像素增益图（修 MuJoCo 点光源的空间渐变/阴影方向）----
    # 布与相机皆静态 → 光照场恒定。比值图 = MuJoCo原生(T_0) / nvdiffrast合成(T_0)，
    # 限幅防边缘噪点，5×5 均值平滑。
    mj = torch.from_numpy(mujoco_full.astype(np.float32) / 255.0).to(device)
    mj = mj.permute(2, 0, 1)  # HWC → CHW
    nd = comp_clean[0]
    m = seg_t[0, 0] > 0.5
    ratio = mj / torch.clamp(nd, min=1e-3)
    ratio = torch.where(m.unsqueeze(0), ratio, torch.ones_like(ratio))
    ratio = torch.clamp(ratio, 0.5, 2.0)
    k = 5
    box = torch.ones(3, 1, k, k, device=device) / (k * k)
    ratio_smooth = F.conv2d(ratio.unsqueeze(0), box, padding=k // 2, groups=3)[0]
    gain_map = ratio_smooth.permute(1, 2, 0).contiguous()  # H,W,3
    renderer.set_gain_map(gain_map)
    np.save(os.path.join(CLOTH_ASSET_DIR, "cloth_gain_map.npy"),
            gain_map.cpu().numpy())
    print(f"[calib] 逐像素增益图已保存（范围 "
          f"{gain_map.min().item():.2f}~{gain_map.max().item():.2f}），"
          f"5×5 平滑、限幅 [0.5,2.0]")

    # ---- F3: 导出-运行一致性（固定测试扰动）----
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        renderer.adv_vc_noise.copy_(
            torch.randn(renderer.adv_vc_noise.shape, generator=g).to(device) * 0.5)
    test_arr = renderer.export_texture()
    Image.fromarray(test_arr).save(os.path.join(debug_dir, "T_test.png"))
    inject_mujoco_texture(sim, tex_id, test_arr)
    mujoco_test = _render_now(env, 256)
    with torch.no_grad():
        adv_rgba, _ = renderer.render(mvp, resolution=(256, 256))
    comp_adv = torch.clamp(adv_rgba.permute(0, 3, 1, 2) * seg_t
                           + bg * (1.0 - seg_t), 0.0, 1.0)
    Image.fromarray(mujoco_test).save(os.path.join(debug_dir, "mujoco_T_test.png"))
    Image.fromarray((comp_adv[0].permute(1, 2, 0).cpu().numpy() * 255)
                    .round().clip(0, 255).astype(np.uint8)).save(
        os.path.join(debug_dir, "composite_T_test.png"))

    mj_t = torch.from_numpy(mujoco_test.astype(np.float32) / 255.0).to(device)
    mj_t = mj_t.permute(2, 0, 1)  # HWC → CHW
    mse_cloth = F.mse_loss(comp_adv[0][:, m], mj_t[:, m]).item()
    psnr = 10 * math.log10(1.0 / max(mse_cloth, 1e-12))
    print(f"[F3] 桌布区域 PSNR(合成 vs MuJoCo原生) = {psnr:.2f} dB")

    restore_mujoco_texture(sim, tex_id, *restore_info)
    _close_libero_env(env)
    print(f"[smoke] 调试图: {debug_dir}")
    print(f"[smoke] F1 通过; F2 mask={seg_mask.mean() * 100:.1f}%; "
          f"F3 PSNR={psnr:.2f}dB（人工核对调试图后放行）")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("patch")
    p.add_argument("--on", action="store_true")
    p.add_argument("--off", action="store_true")
    p.add_argument("--status", action="store_true")

    p = sub.add_parser("smoke")
    p.add_argument("--smoke-task", type=int, default=1)

    p = sub.add_parser("train")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--pool-tasks", default=",".join(map(str, TRAIN_POOL_TASKS)))
    p.add_argument("--pool-episodes", type=int, default=2)
    p.add_argument("--frames-per-ep", type=int, default=10)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--attack-lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=64.0 / 255.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-ref", type=float, default=0.1)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--lambda-mse", type=float, default=1.0,
                   help="特征 MSE 攻击项权重（四臂对比中 A4 置 0）")
    p.add_argument("--lambda-con", type=float, default=0.0,
                   help="排斥 InfoNCE 对比项权重（负样本=池内他帧 clean 特征）")
    p.add_argument("--lambda-task", type=float, default=0.0,
                   help="任务条件项权重（指令 token 融合后隐藏状态偏移）")
    p.add_argument("--con-tau", type=float, default=0.1)
    p.add_argument("--task-every-k", type=int, default=4,
                   help="任务条件项每 k 个迭代计一次（7B 全前向控算力）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--pool-file",
                   default=os.path.join(ARTIFACT_ROOT, "cloth_pool.pt"))
    p.add_argument("--collect-only", action="store_true",
                   help="只采集帧池并保存到 --pool-file，然后退出")
    p.add_argument("--load-pool", action="store_true",
                   help="从 --pool-file 载入帧池（不存在则回退现场采集）")
    p.add_argument("--pool-mode", choices=["short", "full"], default="short",
                   help="short=每集攒够即停(v1)；full=跑满轨迹按 stride 取帧")
    p.add_argument("--stride", type=int, default=10,
                   help="full 模式每 stride 环境步取 1 帧")
    p.add_argument("--max-frames-ep", type=int, default=40,
                   help="full 模式每集帧数上限")
    p.add_argument("--frame-weighting", choices=["uniform", "taao"],
                   default="uniform", help="TAAO 轨迹内帧权重（加权损失）")
    p.add_argument("--tau", type=float, default=1.0, help="TAAO softmax 温度")
    p.add_argument("--tag", default="", help="训练输出目录后缀")
    p.add_argument("--suite", default="libero_spatial",
                   help="采集帧池的 benchmark 套件名")
    p.add_argument("--place-jitter", action="store_true",
                   help="采集期几何 EoT：桌布随机旋转(0-180°)+平移(±3cm)")
    p.add_argument("--uv-eot", action="store_true",
                   help="训练期 UV 域 EoT：纹理随机旋转/镜像/平移")

    p = sub.add_parser("eval")
    p.add_argument("--condition", required=True,
                   choices=["original", "plain", "adv"])
    p.add_argument("--tasks", default="0,1,2,8",
                   help="逗号分隔 task id，或 'heldout'")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--texture", default=None)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--model-config", default="pi0_libero",
                   choices=["pi0_libero", "pi05_libero"],
                   help="openpi 训练配置：π0 用 pi0_libero，RLinf π0.5 用 pi05_libero")
    p.add_argument("--num-denoise", type=int, default=1)
    p.add_argument("--init-offset", type=int, default=0,
                   help="初态编号偏移（R1 冻结确认用 10-19 → 传 10）")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--suite", default="libero_spatial",
                   help="benchmark 套件名（libero_spatial/object/goal/10...）")

    args = ap.parse_args()
    if args.cmd == "patch":
        if args.status:
            print(patch_status())
        if args.on:
            patch_scene(True)
        if args.off:
            patch_scene(False)
    elif args.cmd == "smoke":
        run_smoke(args)
    elif args.cmd == "train":
        args.pool_tasks = [int(t) for t in args.pool_tasks.split(",")]
        train_universal(args)
    elif args.cmd == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
    os._exit(0)
