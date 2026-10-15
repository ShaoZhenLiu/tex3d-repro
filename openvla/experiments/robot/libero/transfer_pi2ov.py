"""π0 训练的通用桌布纹理 → OpenVLA 三条件迁移评测（MuJoCo 原生闭环）.

在 OpenVLA 的 LIBERO 树上铺同一块桌布，检验 π0 侧优化出的对抗纹理能否跨模型迁移。
三条件：original（无布）/ plain（素色 T0）/ adv（π0 的 Universal_Cloth.png）。
注入机制与 π0 侧同源（cloth_transfer_utils 热替换）；对齐逻辑与 attack_pi0_cloth.py
同版本（table_top site → 射线网格回退 → object 套件 floor_mode 贴物体质心）。

用法:
  python transfer_pi2ov.py patch --on|--off|--status
  python transfer_pi2ov.py eval --condition original|plain|adv \
      --suite libero_spatial --tasks 0,2,8 --trials 10 [--texture path.png] [--save-frame]
"""
import argparse
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cloth_transfer_utils as ctu            # noqa: E402  注入四件套（attack_pi0 原样）
import attack_openvla_taao as base            # noqa: E402  复用其 env/模型/动作链

OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/transfer_ov"
CKPT = "/data/shaozhen.liu/code/tex3d-ws/g5/ckpt"
RES = 512  # 与 attack_openvla_taao 的 eval_libero 同分辨率
SEARCH = [["tablecloth"]]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

_ASSETS_ROOT_CACHE = []


def libero_assets_root():
    """从实际 import 到的 libero 包解析 assets 根目录——补丁必须落在
    评测真正使用的树上（09=tex3d-deps/libero，41 由 venv 决定），不能写死。"""
    if _ASSETS_ROOT_CACHE:
        return _ASSETS_ROOT_CACHE[0]
    try:
        import libero.libero as _lib
        root = os.path.abspath(os.path.join(os.path.dirname(_lib.__file__), "assets"))
    except Exception:
        root = ("/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/assets")
    _ASSETS_ROOT_CACHE.append(root)
    return root


def _scene_xmls():
    root = libero_assets_root()
    return [
        (os.path.join(root, "scenes", "libero_tabletop_base_style.xml"), 0.905),
        (os.path.join(root, "scenes", "libero_kitchen_tabletop_base_style.xml"), 0.905),
        (os.path.join(root, "scenes", "libero_living_room_tabletop_base_style.xml"), 0.905),
        (os.path.join(root, "scenes", "libero_study_base_style.xml"), 0.905),
        (os.path.join(root, "scenes", "libero_coffee_table_base_style.xml"), 0.905),
        (os.path.join(root, "scenes", "libero_floor_base_style.xml"), 0.005),
    ]


# ---------------------------------------------------------------- 场景补丁
def patch_status():
    xmls = _scene_xmls()
    return ("patched" if all(os.path.exists(x + ".precloth") for x, _ in xmls)
            else "clean")


def patch_scene(on: bool):
    if on:
        if patch_status() == "patched":
            return
        for xml, cloth_z in _scene_xmls():
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
        print(f"[patch] 桌布已写入 {len(_scene_xmls())} 个场景 @ {libero_assets_root()}"
              f"（备份 .precloth；桌面场景 z 由 SurfaceAdapter 运行时对齐）")
    else:
        if patch_status() == "clean":
            return
        for xml, _ in _scene_xmls():
            bak = xml + ".precloth"
            if os.path.exists(bak):
                shutil.move(bak, xml)
        print("[patch] 全部场景 XML 已还原为无桌布版本")


def ensure_patch(on: bool):
    cur = patch_status()
    want = "patched" if on else "clean"
    if cur != want:
        patch_scene(on)


# ---------------------------------------------------------------- 对齐（π0 侧同版本）
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
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    if not xs:
        return None, 0.0, 0.0
    return (np.array([np.mean(xs), np.mean(ys)]),
            float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))


def align_cloth(env, lift=0.005, floor_mode=False):
    """SurfaceAdapter（与 attack_pi0_cloth.py 同版本）：table_top site →
    射线网格回退（21×21、±1.5cm 聚类取最大平台）→ floor_mode 贴物体质心。
    返回布的世界坐标；失败时抛异常由调用方决定是否终止。"""
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
    _, cloth_bid, _ = base.get_target_model_matrix(env, SEARCH)
    if cloth_bid == -1:
        raise RuntimeError("SurfaceAdapter：找不到桌布 body")
    sim.forward()
    if floor_mode:
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
        zs_sorted = sorted(c[2] for c in cells)
        groups, cur = [], [zs_sorted[0]]
        for z in zs_sorted[1:]:
            if z - cur[-1] <= 0.015:
                cur.append(z)
            else:
                groups.append(cur)
                cur = [z]
        groups.append(cur)
        grp_zs = max(groups, key=len)
        z_top = float(np.median(grp_zs))
        grp = [c for c in cells if abs(c[2] - z_top) <= 0.02]
        xs = np.array([c[0] for c in grp])
        ys = np.array([c[1] for c in grp])
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


# ---------------------------------------------------------------- 评测
def run_eval(args):
    want_patch = args.condition in ("plain", "adv")
    ensure_patch(want_patch)
    texture_path = None
    if args.condition == "plain":
        texture_path = os.path.join(libero_assets_root(), "tablecloth", "cloth_base.png")
        if not os.path.exists(texture_path):
            texture_path = ("/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/"
                            "assets/tablecloth/cloth_base.png")
    elif args.condition == "adv":
        if not args.texture:
            raise ValueError("adv 条件需要 --texture 指向 π0 的 Universal_Cloth.png")
        texture_path = args.texture

    cfg = base.GenerateConfig(
        pretrained_checkpoint=args.ckpt, model_family="openvla",
        unnorm_key=args.unnorm_key)
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    model_input_size = base.get_image_resize_size(cfg)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    floor_mode = args.suite == "libero_object"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"{args.condition}_{args.suite}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "evaluation_result.txt")

    tasks = [int(t) for t in args.tasks.split(",")]
    total_succ = total_eps = 0
    frame_saved = False
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"condition={args.condition} suite={args.suite}\ntexture={texture_path}\n"
                f"tasks={tasks} trials={args.trials} max_steps={args.max_steps}\n\n")
        for tid in tasks:
            task = suite.get_task(tid)
            init_states = suite.get_task_init_states(tid)
            n = min(args.trials, len(init_states))
            succ = 0
            for ep in range(n):
                env, task_description = base.get_libero_env(
                    task, cfg.model_family, resolution=RES)
                env.reset()
                obs = env.set_init_state(init_states[ep])
                sim = env.env.sim
                sim.forward()
                restore_info, tex_id, done = None, -1, False
                try:
                    if want_patch:
                        align_cloth(env, floor_mode=floor_mode)
                        _, bid, _ = base.get_target_model_matrix(env, SEARCH)
                        tex_id = ctu.find_target_texture_id(sim, bid, SEARCH)
                        if tex_id < 0:
                            raise RuntimeError("未找到桌布纹理 id（XML 未打补丁？）")
                        restore_info = ctu.inject_mujoco_texture(
                            sim, tex_id,
                            np.array(Image.open(texture_path).convert("RGB")))

                    t, max_steps = 0, args.max_steps
                    while t < max_steps + cfg.num_steps_wait:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(
                                base.get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue
                        img_high = base.get_libero_image(obs, RES)
                        if args.save_frame and not frame_saved:
                            Image.fromarray(img_high).save(
                                os.path.join(out_dir, f"task{tid}_frame.png"))
                            frame_saved = True
                        img_model = np.array(Image.fromarray(img_high).resize(
                            (model_input_size, model_input_size)))
                        observation = {
                            "full_image": img_model,
                            "state": np.concatenate((
                                obs["robot0_eef_pos"],
                                base.quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                        }
                        action = base.get_action(
                            cfg, model, observation, task_description,
                            processor=processor)
                        action = base.normalize_gripper_action(action, binarize=True)
                        action = base.invert_gripper_action(action)
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                        t += 1
                finally:
                    if restore_info is not None:
                        ctu.restore_mujoco_texture(sim, tex_id, *restore_info)
                    env.close()
                succ += int(done)
                line = f"Task: {tid} | Ep: {ep} | Success: {done}"
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
            total_succ += succ
            total_eps += n
            f.write(f"task_{tid}: {succ}/{n} ({succ / n:.4f})\n")
            print(f"[eval] task {tid}: {succ}/{n}", flush=True)
        rate = total_succ / total_eps if total_eps else 0.0
        f.write(f"\noverall robot success: {total_succ}/{total_eps} ({rate:.4f})\n"
                f"attack failure rate = 1 - above\n")
    print(f"[eval] 结果: {result_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patch")
    p.add_argument("--on", action="store_true")
    p.add_argument("--off", action="store_true")
    p.add_argument("--status", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--condition", required=True,
                   choices=["original", "plain", "adv"])
    p.add_argument("--suite", default="libero_spatial", choices=SUITES)
    p.add_argument("--tasks", default="0,2,8")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=400,
                   help="与 π0 侧 evaluate_native 同默认，保证跨模型协议一致")
    p.add_argument("--texture", default=None)
    p.add_argument("--ckpt", default=CKPT,
                   help="OV checkpoint 目录（每套件用各自 SFT，避免 unnorm 统计错配）")
    p.add_argument("--unnorm-key", default=None,
                   help="动作反归一化统计键，如 libero_object；缺省用 checkpoint 唯一键")
    p.add_argument("--save-frame", action="store_true")
    args = ap.parse_args()
    if args.cmd == "patch":
        if args.status:
            print(patch_status())
        if args.on:
            patch_scene(True)
        if args.off:
            patch_scene(False)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
