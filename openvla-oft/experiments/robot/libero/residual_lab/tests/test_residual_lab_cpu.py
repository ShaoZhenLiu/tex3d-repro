# -*- coding: utf-8 -*-
"""test_residual_lab_cpu.py — residual_lab CPU 侧快速集成测试（服务器运行）。

覆盖（计划§13 子集，无 GPU 部分）：
  1. vendored core 10 项原样通过（直接引用包内 tests）；
  2. rank0 filter 与 F.mse_loss 的 loss/梯度逐位相等（train_residual 数学）；
  3. 量化往返：export→PNG→png_to_tex 逐位等于 round(tex·255)/255（flip 对消）；
  4. 流解析：文本 6dp 与 pt 全精度解析一致；6dp 舍入 ≤5e-7；
  5. 纹理表确定性：同 seed 两次 gen_theta 逐位相同且不消耗全局 RNG；
  6. config fail-fast：缺文件抛 FileNotFoundError。
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # residual_lab/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), ".."))  # libero dir

from residual_lab import residual_core as rc      # noqa: E402
from residual_lab.residual_core import ResidualFilter  # noqa: E402


def test_vendored_core_suite():
    pkg_tests = os.path.join(os.path.dirname(HERE), "tests_pkg",
                             "test_residual_core.py")
    if not os.path.exists(pkg_tests):        # 服务器布局：tests_pkg 与 tests 同级
        pkg_tests = os.path.join(HERE, "tests_pkg", "test_residual_core.py")
    if os.path.exists(pkg_tests):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", pkg_tests],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(pkg_tests),
                           env={**os.environ,
                                "PYTHONPATH": os.path.dirname(HERE)})
        assert r.returncode == 0, r.stdout + r.stderr


def test_rank0_equals_mse_loss_and_grad():
    # 生产口径：f32 前向（adv_hidden.float()）→ f32 MSE
    torch.manual_seed(3)
    h = torch.randn(1, 56, 8, dtype=torch.float32, requires_grad=True)
    ref = torch.randn(1, 56, 8, dtype=torch.float32)
    f0 = ResidualFilter(torch.empty(8, 0, dtype=torch.float32))
    mine = f0.distance(h, ref)
    prod = F.mse_loss(h, ref)
    assert torch.equal(mine, prod) or torch.allclose(mine, prod, atol=1e-8)
    g1 = torch.autograd.grad(mine, h, retain_graph=True)[0]
    g2 = torch.autograd.grad(prod, h)[0]
    assert torch.allclose(g1, g2, atol=1e-8)
    # 与训练循环一致：d_flat.square().mean()（逐位）
    d = h.reshape(-1) - ref.reshape(-1).detach()
    assert torch.equal(mine, d.square().mean())


def test_png_quantization_roundtrip():
    from residual_lab.texutil import png_to_tex
    tex = torch.rand(1, 256, 256, 3)
    png = (torch.flip(tex, dims=[1]) * 255.0).round().clamp(
        0, 255).to(torch.uint8).squeeze(0).numpy()
    back = png_to_tex(png)
    expect = (tex * 255.0).round().clamp(0, 255) / 255.0
    assert torch.equal(back, expect), "翻转+量化必须逐位对消"


def test_stream_text_pt_consistency(tmp_path):
    from residual_lab.streams import load_stream_pt, load_stream_text
    iters, batch = 4, 8
    # 造 pt 流
    st = []
    for it in range(iters):
        idx = list(range(it, it + batch))
        uv = torch.randn(2, 3) if it % 2 == 0 else None
        st.append({"idx": idx, "uv": uv})
    pt = tmp_path / "s.pt"
    torch.save(st, pt)
    # 造文本流（uv 打 6dp）
    tx = tmp_path / "s.log"
    with open(tx, "w") as f:
        f.write("# hdr\n")
        for it in range(iters):
            if st[it]["uv"] is None:
                uvl = ["0.000000"] * 6
            else:
                uvl = ["%.6f" % x for x in st[it]["uv"].numpy().reshape(-1)]
            for slot, fi in enumerate(st[it]["idx"]):
                f.write(f"{it} {slot} {fi} {0} " + " ".join(uvl) + "\n")
    a = load_stream_pt(str(pt), iters)
    b = load_stream_text(str(tx), iters, batch)
    for (ia, ua), (ib, ub) in zip(a, b):
        assert ia == ib
        assert (ua is None) == (ub is None)
        if ua is not None:
            assert float((ua - ub).abs().max()) <= 5e-7


def test_gen_theta_deterministic_and_rng_isolated():
    from residual_lab.texture_gen import gen_theta
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    t1 = gen_theta((1089, 3), "iid", 0.15, 105)
    t2 = gen_theta((1089, 3), "iid", 0.15, 105)
    assert torch.equal(t1, t2)
    assert torch.equal(before, torch.get_rng_state())
    tl = gen_theta((1089, 3), "lp", 0.15, 105)
    assert abs(float(tl.std()) - 0.15) < 1e-5


def test_config_fail_fast(tmp_path):
    from residual_lab.config import load_config
    bad = tmp_path / "bad.yaml"
    bad_yaml = (
        "paths:\n"
        "  source_pool: /nonexistent/x.pt\n"
        "  source_checkpoint: /nonexistent/y\n"
        "  matched_reference_cache: /nonexistent/z\n"
        "  cloth_mesh: /nonexistent/m\n"
        "  base_texture: /nonexistent/b\n"
        "  output_root: /nonexistent/o\n")
    bad.write_text(bad_yaml, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(str(bad))
