import pytest
import torch
from residual_core import ResidualFilter, fit_uncentered_basis, random_orthobasis, retained_energy


def test_zero_rank_matches_mr():
    torch.manual_seed(0)
    z = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    z0 = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    filt = ResidualFilter(torch.empty(17, 0, dtype=torch.float64))
    actual = filt.distance(z, z0)
    expected = (z - z0.detach()).square().mean()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, z, retain_graph=True)[0],
                       torch.autograd.grad(expected, z)[0])
    assert z0.grad is None


def test_beta_zero_is_identity():
    u = random_orthobasis(17, 4, 11, dtype=torch.float64)
    x = torch.randn(3, 17, dtype=torch.float64)
    assert torch.equal(ResidualFilter(u, 0.0)(x), x)


def test_projector_idempotence_and_explicit_matrix():
    u = random_orthobasis(17, 4, 12, dtype=torch.float64)
    f = ResidualFilter(u)
    x = torch.randn(3, 17, dtype=torch.float64)
    p = torch.eye(17, dtype=x.dtype) - u @ u.T
    assert torch.allclose(f(x), x @ p, atol=1e-12)
    assert torch.allclose(f(f(x)), f(x), atol=1e-12)
    assert torch.allclose(f(x) @ u, torch.zeros(3, 4, dtype=x.dtype), atol=1e-12)


def test_soft_filter_energy_and_gradient():
    u = random_orthobasis(17, 4, 13, dtype=torch.float64)
    x = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    beta = 0.4
    f = ResidualFilter(u, beta)
    loss = f(x).square().sum()
    rhs = x.square().sum() - (2*beta-beta**2) * (x @ u).square().sum()
    assert torch.allclose(loss, rhs, atol=1e-12)
    expected = 2 * f(f(x.detach()))
    assert torch.allclose(torch.autograd.grad(loss, x)[0], expected, atol=1e-12)
    assert torch.autograd.gradcheck(lambda q: f(q).square().sum(), (x,))


def test_uncentered_svd_keeps_constant_error():
    v = torch.arange(1., 18., dtype=torch.float64)
    residuals = v[None, :].repeat(12, 1)
    u, s = fit_uncentered_basis(residuals, 1)
    assert retained_energy(residuals, ResidualFilter(u)) < 1e-20
    assert s[0] > 0
    with pytest.raises(ValueError):
        fit_uncentered_basis(residuals, 2)


def test_conditional_native_bound():
    u = random_orthobasis(17, 4, 14, dtype=torch.float64)
    f = ResidualFilter(u)
    dn = torch.randn(20, 17, dtype=torch.float64)
    coeff = torch.randn(20, 4, dtype=torch.float64)
    xi = torch.randn(20, 17, dtype=torch.float64) * .01
    residual = coeff @ u.T + xi
    dr = dn + residual
    eps = f(residual).norm(dim=-1)
    lower = torch.relu(f(dr).norm(dim=-1) - eps).square()
    assert torch.all(dn.square().sum(-1) + 1e-10 >= lower)


def test_random_control_does_not_change_global_rng():
    torch.manual_seed(55)
    before = torch.get_rng_state().clone()
    u1 = random_orthobasis(17, 4, 1001)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(u1, random_orthobasis(17, 4, 1001))


def test_invalid_basis_and_data_rejected():
    with pytest.raises(ValueError):
        ResidualFilter(torch.ones(7, 3))
    with pytest.raises(ValueError):
        fit_uncentered_basis(torch.tensor([[float('nan'), 0.]]), 1)
    with pytest.raises(ValueError):
        retained_energy(torch.zeros(2, 7), ResidualFilter(torch.empty(7, 0)))


def test_reference_is_detached_and_basis_not_trained():
    u = random_orthobasis(17, 3, 10)
    f = ResidualFilter(u)
    assert list(f.parameters()) == []
    z = torch.randn(2, 17, requires_grad=True)
    r = torch.randn(2, 17, requires_grad=True)
    f.distance(z, r).backward()
    assert z.grad is not None and r.grad is None
    assert f.basis.grad is None


def test_bfloat16_difference_is_promoted_first():
    z = torch.randn(2, 17).bfloat16().requires_grad_()
    r = torch.randn(2, 17).bfloat16()
    f = ResidualFilter(torch.empty(17, 0))
    loss = f.distance(z, r)
    assert loss.dtype == torch.float32
    assert torch.equal(loss, (z.float()-r.float()).square().mean())
