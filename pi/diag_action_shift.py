"""动作分布诊断：embedding 攻击是否真的动摇 π0 的动作输出？

在既有帧池上做纯前向配对实验：同一帧、同一份去噪噪声 z，
分别以素色布(T0)和对抗布(Tθ)合成图像过 π0，
量动作 chunk 的偏移（L2/max/mean）并与 embedding 偏移关联。
不触碰 MuJoCo 环境，与 XML 状态无关。

用法: python diag_action_shift.py [--frames-per-task 8]
"""
import argparse
import os
import random

import numpy as np
import torch

import attack_pi0_cloth as C
from attack_pi0 import replace_image_in_processed_inputs, extract_split_features

POOL = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/cloth_pool_v2.pt"
TRAIN_DIR = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/train_20261012_101656"
OUT_DIR = "/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-task", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)

    pool = torch.load(POOL, map_location="cuda", weights_only=False)
    policy = C.load_policy(C.DEFAULT_CKPT, num_denoise=args.num_steps)
    model = policy._model

    renderer = C.ClothRenderer(device="cuda", epsilon=64.0 / 255.0,
                               init_std=0.01, seed=7).to("cuda")
    param = torch.load(os.path.join(TRAIN_DIR, "Ep0_Cloth_Delta.pt"),
                       map_location="cuda")
    renderer.adv_vc_noise.data.copy_(param)
    backup = renderer.adv_vc_noise.data.clone()

    rows = []
    shared_noise = None
    for tid, frames in sorted(pool.items()):
        for f in random.sample(frames, min(args.frames_per_task, len(frames))):
            adv_img = C.composite_cloth(renderer, f)
            renderer.adv_vc_noise.data.zero_()
            plain_img = C.composite_cloth(renderer, f)
            renderer.adv_vc_noise.data.copy_(backup)

            acts, embs = {}, {}
            for name, img in (("plain", plain_img), ("adv", adv_img)):
                ni, _, _ = replace_image_in_processed_inputs(
                    f["processed_inputs"], img, "cuda")
                obs = C._model.Observation.from_dict(ni)
                with torch.no_grad():
                    if shared_noise is None:
                        act = model.sample_actions("cuda", obs, num_steps=args.num_steps)
                        shared_noise = torch.randn(act.shape, device="cuda")
                    act = model.sample_actions("cuda", obs, noise=shared_noise,
                                               num_steps=args.num_steps)
                    emb, _, _ = extract_split_features(model, obs)
                acts[name] = act.detach().float().cpu().numpy()
                embs[name] = emb.detach().float().cpu()
            da = acts["adv"] - acts["plain"]
            emb_shift = (embs["adv"] - f["clean_img_embs"].detach().cpu().float()
                         ).pow(2).mean().sqrt().item()
            rows.append(dict(task=tid,
                             act_l2=float(np.linalg.norm(da)),
                             act_max=float(np.abs(da).max()),
                             act_mean=float(np.abs(da).mean()),
                             act_range=float(acts["plain"].std()),
                             emb_rmse=emb_shift))
        per = [r for r in rows if r["task"] == tid]
        print(f"[diag] task {tid}: n={len(per)} "
              f"act_l2={np.mean([r['act_l2'] for r in per]):.4f} "
              f"act_max={np.mean([r['act_max'] for r in per]):.4f} "
              f"emb_rmse={np.mean([r['emb_rmse'] for r in per]):.4f}")

    np.save(os.path.join(OUT_DIR, "action_shift.npy"), np.array(rows, dtype=object),
            allow_pickle=True)
    l2 = np.array([r["act_l2"] for r in rows])
    mx = np.array([r["act_max"] for r in rows])
    em = np.array([r["emb_rmse"] for r in rows])
    std = np.array([r["act_range"] for r in rows])
    corr = float(np.corrcoef(l2, em)[0, 1]) if len(rows) > 2 else float("nan")
    print(f"[diag] 总计 {len(rows)} 帧 | 动作L2均值={l2.mean():.4f} (策略自身std={std.mean():.4f}) "
          f"| 最大分量={mx.mean():.4f} | emb_rmse={em.mean():.4f} | corr(act,emb)={corr:.3f}")
    print(f"[diag] 产物: {OUT_DIR}/action_shift.npy")


if __name__ == "__main__":
    main()
