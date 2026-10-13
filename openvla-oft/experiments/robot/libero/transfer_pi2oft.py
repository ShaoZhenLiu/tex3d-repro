"""π0/OV 训练的桌布纹理 → OFT 三条件迁移评测（MuJoCo 原生闭环）.

在 OFT 的 LIBERO 树上铺同一块桌布，检验 π0/OV 侧优化出的对抗纹理能否跨模型迁移。
三条件：original（无布）/ plain（素色 T0）/ adv（Universal_Cloth.png）。
注入与对齐机制与 π0/OV 侧完全同源（SurfaceAdapter + tex_data 热替换）。
动作链走 attack_oft.py 的 predict_action + action_head 路径
（_query_action_with_compatible_signature 包装，不走 model.generate）。

用法:
  python transfer_pi2oft.py patch --on|--off|--status
  python transfer_pi2oft.py eval --condition original|plain|adv \
      --tasks 0,2,8 --trials 25 [--texture path.png] [--save-frame]
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
import sys as _s; _s.path.insert(0, "/data/shaozhen.liu/code/tex3d/openvla/experiments/robot/libero")
import cloth_transfer_utils as ctu            # noqa: E402  注入四件套（attack_pi0 原样）
import attack_oft as base            # noqa: E402  复用其 env/模型/动作链

LIBERO_XML = (
    "/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/assets/"
    "scenes/libero_tabletop_base_style.xml"
)
CLOTH_DIR = "/data/shaozhen.liu/code/tex3d-deps/libero/libero/libero/assets/tablecloth"
CLOTH_T0 = os.path.join(CLOTH_DIR, "cloth_base.png")
BACKUP = LIBERO_XML + ".precloth"
SEARCH = [["tablecloth"]]
OUT_ROOT = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/transfer_oft"
CKPT = "/data/shaozhen.liu/code/tex3d-ws/g7/oft-spatial-ckpt"
RES = 512  # 与 attack_oft 的 eval_libero 同分辨率


# ---------------------------------------------------------------- 场景补丁
def patch_status():
    return "patched" if os.path.exists(BACKUP) else "clean"


def patch_scene(on: bool):
    if on:
        if patch_status() == "patched":
            return
        shutil.copy2(LIBERO_XML, BACKUP)
        tree = ET.parse(LIBERO_XML)
        root, asset, worldbody = tree.getroot(), tree.find("asset"), tree.find("worldbody")
        if asset is None or worldbody is None:
            raise RuntimeError("场景 XML 结构异常：缺 asset/worldbody")
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
        body = ET.SubElement(worldbody, "body", {
            "name": "tablecloth", "pos": "0 0 0.905"})
        ET.SubElement(body, "geom", {
            "name": "tablecloth_visual", "type": "mesh",
            "mesh": "tablecloth_mesh", "material": "tablecloth_mat",
            "contype": "0", "conaffinity": "0", "group": "1"})
        tree.write(LIBERO_XML)
        print(f"[patch] 桌布已写入 LIBERO 树 {LIBERO_XML}（运行时将对齐桌面）")
    else:
        if patch_status() == "clean":
            return
        shutil.move(BACKUP, LIBERO_XML)
        print("[patch] LIBERO 树场景 XML 已还原")


def ensure_patch(on: bool):
    if patch_status() != ("patched" if on else "clean"):
        patch_scene(on)


def align_cloth(env, lift=0.005):
    """SurfaceAdapter（与 π0 侧同逻辑）：读 table_top site 对齐真实桌面。"""
    sim = env.env.sim
    site_id = None
    for s in range(sim.model.nsite):
        try:
            if sim.model.site_id2name(s) == "table_top":
                site_id = s
                break
        except Exception:
            pass
    _, cloth_bid, _ = base.get_target_model_matrix(env, SEARCH)
    if site_id is None or cloth_bid == -1:
        raise RuntimeError("SurfaceAdapter：找不到 table_top site 或桌布 body")
    sim.forward()
    pos = np.array(sim.data.site_xpos[site_id], dtype=float)
    sim.model.body_pos[cloth_bid, 0] = pos[0]
    sim.model.body_pos[cloth_bid, 1] = pos[1]
    sim.model.body_pos[cloth_bid, 2] = pos[2] + lift
    sim.forward()


# ---------------------------------------------------------------- 评测
def run_eval(args):
    want_patch = args.condition in ("plain", "adv")
    ensure_patch(want_patch)
    texture_path = None
    if args.condition == "plain":
        texture_path = CLOTH_T0
    elif args.condition == "adv":
        if not args.texture:
            raise ValueError("adv 条件需要 --texture 指向 Universal_Cloth.png")
        texture_path = args.texture

    cfg = base.GenerateConfig(
        pretrained_checkpoint=CKPT, model_family="openvla")
    model = base.get_model(cfg)
    processor = base.get_processor(cfg)
    base.check_unnorm_key(cfg, model)
    model_input_size = base.get_image_resize_size(cfg)
    use_proprio = bool(getattr(cfg, "use_proprio", False))
    use_wrist = getattr(cfg, "num_images_in_input", 1) > 1

    # OFT 动作头三件套（与 attack_oft.eval_libero 同调用方式）
    if base.get_action_head is None:
        raise RuntimeError("OFT 需 openvla_utils.get_action_head（导入失败）")
    action_head = proprio_projector = noisy_action_projector = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = base.get_action_head(cfg, model.llm_dim)
    if cfg.use_proprio and base.get_proprio_projector is not None:
        proprio_projector = base.get_proprio_projector(
            cfg, model.llm_dim, proprio_dim=8)
    if cfg.use_diffusion and base.get_noisy_action_projector is not None:
        noisy_action_projector = base.get_noisy_action_projector(
            cfg, model.llm_dim)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"{args.condition}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "evaluation_result.txt")

    tasks = [int(t) for t in args.tasks.split(",")]
    total_succ = total_eps = 0
    frame_saved = False
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"condition={args.condition}\ntexture={texture_path}\n"
                f"tasks={tasks} trials={args.trials}\n\n")
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
                        align_cloth(env)
                        _, bid, _ = base.get_target_model_matrix(env, SEARCH)
                        tex_id = ctu.find_target_texture_id(sim, bid, SEARCH)
                        if tex_id < 0:
                            raise RuntimeError("未找到桌布纹理 id（XML 未打补丁？）")
                        restore_info = ctu.inject_mujoco_texture(
                            sim, tex_id,
                            np.array(Image.open(texture_path).convert("RGB")))

                    t, max_steps = 0, 300
                    while t < max_steps + cfg.num_steps_wait:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(
                                base.get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue
                        img_high = base.get_libero_image(obs)
                        if args.save_frame and not frame_saved:
                            Image.fromarray(img_high).save(
                                os.path.join(out_dir, f"task{tid}_frame.png"))
                            frame_saved = True
                        img_model = np.array(Image.fromarray(img_high).resize(
                            (model_input_size, model_input_size)))
                        observation = {"full_image": img_model}
                        if use_proprio:
                            observation["state"] = np.concatenate((
                                obs["robot0_eef_pos"],
                                base.quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ))
                        else:
                            observation["state"] = np.zeros(8)
                        if use_wrist:
                            wrist_m = base._extract_wrist_image(
                                obs, model_input_size)
                            if wrist_m is not None:
                                observation["wrist_image"] = wrist_m
                        action = base._query_action_with_compatible_signature(
                            cfg, model, observation, task_description,
                            processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=getattr(cfg, "use_film", False))
                        if isinstance(action, list):
                            action = np.asarray(action[0], dtype=np.float32)
                        else:
                            action = np.asarray(action, dtype=np.float32)
                        action = base.normalize_gripper_action(
                            action, binarize=True)
                        if cfg.model_family == "openvla":
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
    p.add_argument("--tasks", default="0,2,8")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--texture", default=None)
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
