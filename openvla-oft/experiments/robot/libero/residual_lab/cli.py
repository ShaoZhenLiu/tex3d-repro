# -*- coding: utf-8 -*-
"""cli.py — residual_lab 命令行入口（计划§14 的接口规格）。

用法（cwd=任意，PYTHONPATH 含本包父目录）：
  python -m residual_lab.cli textures  --config <yaml>
  python -m residual_lab.cli capture   --config <yaml> --split fit
  python -m residual_lab.cli fit       --config <yaml>
  python -m residual_lab.cli one-step  --config <yaml> --k 8 --beta 1.0
  python -m residual_lab.cli train     --config <yaml> --method matched --seed 7
  python -m residual_lab.cli summarize --config <yaml>
"""
import argparse
import json
import os
import subprocess
import sys


def _cfg(args):
    from .config import load_config
    return load_config(args.config)


def cmd_textures(args):
    import numpy as np
    from PIL import Image
    from .adapters import ResidualCtx
    from . import texture_gen
    cfg = _cfg(args)
    ctx = ResidualCtx(device="cuda")
    try:
        mr8 = cfg.paths.get("mr8_png")
        mr8_png = np.array(Image.open(mr8).convert("RGB")) if mr8 else None
        meta = texture_gen.build_table(ctx, os.path.join(
            cfg.paths.output_root, "textures"), mr8_png)
        print(json.dumps(meta["hi_std_rule"], indent=1))
        for split, entries in meta["table"].items():
            print(f"[{split}] " + "; ".join(
                f"{e['texture_id']}: rms={e['rms']:.1f}" for e in entries))
    finally:
        ctx.cleanup()


def cmd_capture(args):
    from .capture import run_capture
    cfg = _cfg(args)
    run_capture(cfg, args.split, cfg.paths.output_root)


def cmd_fit(args):
    from .fit_basis import fit_and_report
    cfg = _cfg(args)
    rep = fit_and_report(cfg.paths.output_root)
    print(json.dumps({k: rep[k] for k in (
        "n_fit", "m", "num_rank", "gate_b", "selection_ranked",
        "gamma_scalar_control", "abs_rms") if k in rep}, indent=1,
        ensure_ascii=False))


def cmd_one_step(args):
    from .one_step import run_one_step
    cfg = _cfg(args)
    starts = {"init": "init", "zero": "zero"}
    for name, path in (("mr8", cfg.paths.get("mr8_theta")),
                       ("native8", cfg.paths.get("native8_theta")),
                       ("bs8", cfg.paths.get("bs8_theta"))):
        if path:
            starts[name] = path
    methods = ["matched", "residual", "random"]
    gamma = None
    gate_b_path = os.path.join(cfg.paths.output_root, "diagnostics",
                               "gate_b_report.json")
    if os.path.exists(gate_b_path):
        rep = json.load(open(gate_b_path, encoding="utf-8"))
        gamma = rep.get("gamma_scalar_control")
        if gamma is not None and gamma < cfg.training.get(
                "conditional_scalar_control_if_gamma_below", 0.9):
            methods.append("scalar")
    run_one_step(cfg, cfg.paths.output_root, os.path.join(
        cfg.paths.output_root, "bases"), args.k, args.beta, gamma,
        starts, methods)


def cmd_train(args):
    from .train_residual import run_train
    cfg = _cfg(args)
    out_dir = os.path.join(cfg.paths.output_root, "training",
                           f"{args.method}_seed{args.seed}")
    gate_b_path = os.path.join(cfg.paths.output_root, "diagnostics",
                               "gate_b_report.json")
    gamma = None
    if args.method == "scalar":
        rep = json.load(open(gate_b_path, encoding="utf-8"))
        gamma = rep["gamma_scalar_control"]
    dump = run_train(
        cfg, out_dir, args.method, seed=args.seed, iters=args.iters,
        batch=int(cfg.training.batch_size),
        stream_pt=cfg.paths.get("stream_file"),
        bases_dir=os.path.join(cfg.paths.output_root, "bases"),
        k=args.k, beta=args.beta, gamma=gamma,
        dump_theta_iters=set(range(args.dump_iters)) if args.dump_iters else None)
    if dump:
        torch_path = os.path.join(out_dir, "theta_dump.pt")
        import torch
        torch.save(dump, torch_path)
        print(f"[train] θ dump → {torch_path}")


def cmd_regress(args):
    """Gate A §4.6：rank0/beta0 ≡ 生产 MR 的 30-iter 逐位回归。
    需要 --ablate-dir（ablate 参照输出目录）。"""
    from .train_regress import run_regression
    cfg = _cfg(args)
    run_regression(cfg, args.ablate_dir, iters=args.iters)


def cmd_summarize(args):
    cfg = _cfg(args)
    root = cfg.paths.output_root
    out = {"output_root": root}
    for name, fn in (("gate_b", "diagnostics/gate_b_report.json"),
                     ("one_step", "diagnostics/one_step_report.json")):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:8000])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("textures").set_defaults(func=cmd_textures)
    p = sub.add_parser("capture")
    p.add_argument("--split", required=True,
                   choices=["fit", "validation", "locked_test"])
    p.set_defaults(func=cmd_capture)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    p = sub.add_parser("one-step")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.set_defaults(func=cmd_one_step)
    p = sub.add_parser("train")
    p.add_argument("--method", required=True,
                   choices=["matched", "residual", "random", "scalar"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--dump-iters", type=int, default=0,
                   help="前 N iter 逐 iter dump θ（回归对照用）")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("regress")
    p.add_argument("--ablate-dir", default=None)
    p.add_argument("--iters", type=int, default=30)
    p.set_defaults(func=cmd_regress)
    sub.add_parser("summarize").set_defaults(func=cmd_summarize)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
