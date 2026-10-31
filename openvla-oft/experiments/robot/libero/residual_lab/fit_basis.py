# -*- coding: utf-8 -*-
"""fit_basis.py — 阶段三：子空间拟合与 Gate B 诊断（计划§6）。

残差主定义（deployment-path residual）：
  d_R_float = fR_float − fR0        （训练路径位移）
  d_R_Q     = fR_Q     − fR0        （同参考；量化纹理进合成渲染）
  d_N_Q     = fN_Q     − fN0        （原生部署位移）
  r_render  = d_R_Q − d_N_Q
  r_export  = d_R_float − d_R_Q
  r_deploy  = d_R_float − d_N_Q = r_render + r_export   ← 拟合目标
指标：q_err / q_signal / q_synth（总能量比，非样本比值均值），
     分 task 与幅度组分层报告；绝对 RMS-per-feature 同步给出。
对照：k=0、同秩随机基×5种子、原生信号 PCA 同秩移除、打乱配对破坏对照。
"""
import json
import os

import torch

from . import residual_core
from .residual_core import ResidualFilter, fit_uncentered_basis, random_orthobasis

RANKS = [0, 1, 2, 4, 8, 16, 32, 64]
RANDOM_SEEDS = [1001, 1002, 1003, 1004, 1005]
GATE = {"max_q_err": 0.5, "min_q_signal": 0.8, "min_group_q_signal": 0.6}


def load_split_rows(out_root, split):
    """读取 pairs/{split}/*.pt → rows: dict 含各向量的 flatten fp32 + 分组键。"""
    d = os.path.join(out_root, "pairs", split)
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pt"):
            continue
        rec = torch.load(os.path.join(d, fn), map_location="cpu",
                         weights_only=False)
        fN0, fR0 = rec["fN0"], rec["fR0"]
        for key in list(rec.keys()):
            if key.endswith("__fR_float"):
                tid = key[: -len("__fR_float")]
                amp = _amp_of(tid)
                rows.append({
                    "state": rec["record_id"], "task": rec["source_task_id"],
                    "texture": tid, "amp": amp,
                    "dR_float": rec[key] - fR0,
                    "dR_Q": rec[f"{tid}__fR_Q"] - fR0,
                    "dN_Q": rec[f"{tid}__fN_Q"] - fN0,
                    "aa_forward_maxdiff": rec.get("aa_forward_maxdiff", 0.0),
                })
    return rows


def _amp_of(tid):
    for a in ("lo", "mid", "hi"):
        if f"_{a}_" in tid:
            return a
    return "na"


def stack(rows, key):
    return torch.stack([r[key] for r in rows])   # [n, m] fp32


def q_ratio(w_d, d):
    num = w_d.square().sum()
    den = d.square().sum()
    if float(den) <= 0:
        return None
    return float(num / den)


def rms_per_feat(d):
    return float(d.square().sum().div(d.numel()).sqrt())


def fit_and_report(out_root, device="cpu", betas_extra=(0.25, 0.5, 0.75)):
    fit_rows = load_split_rows(out_root, "fit")
    val_rows = load_split_rows(out_root, "validation")
    assert fit_rows and val_rows, "capture 数据缺失"
    for r in fit_rows + val_rows:
        r["r_deploy"] = r["dR_float"] - r["dN_Q"]
        r["r_render"] = r["dR_Q"] - r["dN_Q"]
        r["r_export"] = r["dR_float"] - r["dR_Q"]

    X = stack(fit_rows, "r_deploy").to(device)
    n, m = X.shape
    dev = X.device
    # 数值秩上限
    with torch.no_grad():
        _, s_full, _ = torch.linalg.svd(X.to(torch.float64) if device == "cpu"
                                        else X, full_matrices=False)
    num_rank = int((s_full > max(n, m) * torch.finfo(
        s_full.dtype).eps * float(s_full[0])).sum())

    bases_dir = os.path.join(out_root, "bases")
    os.makedirs(bases_dir, exist_ok=True)
    torch.save({"s": s_full.float().cpu(), "n": n, "m": m, "num_rank": num_rank},
               os.path.join(bases_dir, "spectrum.pt"))

    Vd = stack(val_rows, "r_deploy").to(dev)
    Sd = stack(val_rows, "dR_float").to(dev)      # 合成信号
    Nd = stack(val_rows, "dN_Q").to(dev)          # 原生信号
    groups = {
        "by_task": [("task", t) for t in sorted({r["task"] for r in val_rows})],
        "by_amp": [("amp", a) for a in ("lo", "mid", "hi")],
    }
    gmask = {}
    for gname, pairs_ in groups.items():
        for field, val in pairs_:
            gmask[f"{gname}:{val}"] = torch.tensor(
                [r[field] == val for r in val_rows])

    report = {"n_fit": n, "m": m, "num_rank": num_rank, "gate": GATE,
              "candidates": [], "abs_rms": {
                  "r_deploy": rms_per_feat(Vd.cpu()),
                  "dR_float": rms_per_feat(Sd.cpu()),
                  "dN_Q": rms_per_feat(Nd.cpu())}}
    aa = [r["aa_forward_maxdiff"] for r in val_rows]
    report["aa_forward_maxdiff_val_max"] = float(max(aa)) if aa else None

    feasible = [k for k in RANKS if k <= num_rank]
    for k in feasible:
        if k == 0:
            U = torch.empty(m, 0)
            sv = torch.empty(0)
        else:
            U, sv = fit_uncentered_basis(X.cpu(), k)
            U = U.to(dev)
        for beta in ([1.0] if k == 0 else [1.0] + list(betas_extra)):
            filt = ResidualFilter(U, beta)
            cand = {"k": k, "beta": beta,
                    "q_err": q_ratio(filt(Vd), Vd),
                    "q_signal": q_ratio(filt(Nd), Nd),
                    "q_synth": q_ratio(filt(Sd), Sd)}
            for gname, mask in gmask.items():
                cand[f"{gname}.q_signal"] = q_ratio(filt(Nd[mask.to(dev)]),
                                                    Nd[mask.to(dev)])
                cand[f"{gname}.q_err"] = q_ratio(filt(Vd[mask.to(dev)]),
                                                 Vd[mask.to(dev)])
            report["candidates"].append(cand)
        if k > 0:
            torch.save({"U": U.cpu(), "s_all": sv.cpu(), "k": k,
                        "fit_rows": n, "beta_default": 1.0,
                        "source": "uncentered_thin_svd_r_deploy_fit"},
                       os.path.join(bases_dir, f"basis_k{k}.pt"))

    # ---- 对照 1：同秩随机基（5 种子；报告分布，训练控制用固定映射 seed） ----
    rand_ctrl = []
    for k in [kk for kk in feasible if kk > 0]:
        for sd in RANDOM_SEEDS:
            Ur = random_orthobasis(m, k, sd).to(dev)
            fr = ResidualFilter(Ur, 1.0)
            rand_ctrl.append({"k": k, "seed": sd,
                              "q_err": q_ratio(fr(Vd), Vd),
                              "q_signal": q_ratio(fr(Nd), Nd),
                              "q_synth": q_ratio(fr(Sd), Sd)})
        torch.save({"U": random_orthobasis(m, k, RANDOM_SEEDS[0]).cpu(),
                    "k": k, "seed": RANDOM_SEEDS[0], "seeds_all": RANDOM_SEEDS},
                   os.path.join(bases_dir, f"random_k{k}.pt"))
    report["random_control"] = rand_ctrl

    # ---- 对照 2：原生信号 PCA 同秩移除（离线诊断） ----
    Xn = stack(fit_rows, "dN_Q").to(dev)
    pca_ctrl = []
    for k in [kk for kk in feasible if kk > 0]:
        Un, _ = fit_uncentered_basis(Xn.cpu(), k)
        fn_ = ResidualFilter(Un.to(dev), 1.0)
        pca_ctrl.append({"k": k,
                         "q_err": q_ratio(fn_(Vd), Vd),
                         "q_signal": q_ratio(fn_(Nd), Nd)})
    report["native_pca_control"] = pca_ctrl

    # ---- 对照 3：打乱配对（破坏性 sanity，仅报告） ----
    perm = torch.randperm(len(fit_rows), generator=torch.Generator().manual_seed(7))
    Xs = stack(fit_rows, "dR_float") - stack(fit_rows, "dN_Q")[perm]
    shuf = []
    for k in [kk for kk in feasible if 0 < kk <= 32]:
        Us, _ = fit_uncentered_basis(Xs, k)
        fs = ResidualFilter(Us.to(dev), 1.0)
        shuf.append({"k": k, "q_err": q_ratio(fs(Vd), Vd)})
    report["shuffled_pair_control"] = shuf

    # ---- Gate B 判定与候选选择（source-only） ----
    passing = []
    for c in report["candidates"]:
        ok = (c["q_err"] is not None and c["q_err"] <= GATE["max_q_err"]
              and c["q_signal"] is not None
              and c["q_signal"] >= GATE["min_q_signal"])
        groups_ok = all(v is not None and v >= GATE["min_group_q_signal"]
                        for kk, v in c.items() if kk.startswith("by_"))
        c["gate_pass"] = bool(ok and groups_ok)
        if c["gate_pass"]:
            passing.append(c)
    passing.sort(key=lambda c: (c["q_err"], c["k"], c["beta"]))
    report["selection_ranked"] = [
        {"k": c["k"], "beta": c["beta"], "q_err": c["q_err"],
         "q_signal": c["q_signal"]} for c in passing[:6]]
    report["gate_b"] = "PASS" if passing else (
        "FAIL" if feasible and report["candidates"] else "INCONCLUSIVE")

    # γ（scalar 控制触发量，fit 集上冻结）——最优候选的 Σ||W dR||²/Σ||dR||²
    if passing:
        best = passing[0]
        Ub = torch.load(os.path.join(bases_dir, f"basis_k{best['k']}.pt"),
                        map_location=dev, weights_only=False)["U"].to(dev)
        fb = ResidualFilter(Ub, best["beta"])
        Sfit = stack(fit_rows, "dR_float").to(dev)
        report["gamma_scalar_control"] = q_ratio(fb(Sfit), Sfit)
        report["selected"] = {"k": best["k"], "beta": best["beta"]}

    diag_dir = os.path.join(out_root, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    with open(os.path.join(diag_dir, "gate_b_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return report
