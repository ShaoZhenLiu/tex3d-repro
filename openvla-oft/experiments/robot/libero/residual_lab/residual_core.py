"""Small, simulator-independent reference core; not a complete RefCloth trainer.

Rows are samples, columns are FIXED, aligned source-feature coordinates.
The filter is frozen. No feature whitening, norm renormalization, or learned
projector is performed. float16/bfloat16 inputs are promoted before algebra.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


def _analysis_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def fit_uncentered_basis(residuals: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Return U [m,k] and all singular values, using ONLY calibration-fit rows.

    This deliberately uses uncentered SVD. It is a small-data reference
    implementation; use a separately validated streamed method for large data.
    """
    if residuals.ndim != 2 or not residuals.is_floating_point():
        raise ValueError('residuals must be a floating [n_samples, n_features] tensor')
    n, m = residuals.shape
    if min(n, m) < 1 or not 0 <= rank <= min(n, m):
        raise ValueError('rank outside valid range')
    x = residuals.detach().to(dtype=_analysis_dtype(residuals))
    if not bool(torch.isfinite(x).all()):
        raise ValueError('non-finite residual')
    if rank == 0:
        return x.new_empty((m, 0)), x.new_empty((0,))
    with torch.no_grad():
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        tol = max(n, m) * torch.finfo(x.dtype).eps * s[0]
        numerical_rank = int((s > tol).sum())
        if rank > numerical_rank:
            raise ValueError(f'rank {rank} exceeds numerical rank {numerical_rank}')
        basis = vh[:rank].T.contiguous()
    return basis, s


def random_orthobasis(m: int, rank: int, seed: int, *, dtype=torch.float32) -> Tensor:
    """An independent CPU RNG avoids perturbing the texture-training stream."""
    if m < 1 or not 0 <= rank <= m:
        raise ValueError('invalid dimensions')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('QR requires float32 or float64 here')
    if rank == 0:
        return torch.empty((m, 0), dtype=dtype)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn((m, rank), generator=gen, dtype=dtype)
    return torch.linalg.qr(x, mode='reduced').Q.contiguous()


class ResidualFilter(nn.Module):
    """W d = d - beta U(U^T d), applied to row vectors without an m-by-m matrix.

    beta=1: orthogonal projection; beta=0: identity. Intermediate beta uses
    the squared norm ||W d||^2, NOT d^T W d. The basis is never optimized.
    """
    def __init__(self, basis: Tensor, beta: float = 1.0):
        super().__init__()
        if basis.ndim != 2 or not basis.is_floating_point() or basis.shape[0] < 1:
            raise ValueError('basis must be floating [m, k]')
        if basis.shape[1] > basis.shape[0] or not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('invalid rank or beta')
        u = basis.detach().clone().to(dtype=_analysis_dtype(basis))
        if not bool(torch.isfinite(u).all()):
            raise ValueError('non-finite basis')
        eye = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
        if not torch.allclose(u.T @ u, eye, atol=2e-5, rtol=2e-5):
            raise ValueError('basis columns must be orthonormal')
        self.register_buffer('basis', u)
        self.beta = float(beta)

    def forward(self, displacement: Tensor) -> Tensor:
        if displacement.ndim < 1 or displacement.shape[-1] != self.basis.shape[0]:
            raise ValueError('last dimension does not match aligned feature space')
        if not displacement.is_floating_point():
            raise ValueError('displacement must be floating')
        d = displacement.to(dtype=_analysis_dtype(displacement))
        u = self.basis.to(device=d.device, dtype=d.dtype)
        if self.beta == 0.0 or u.shape[1] == 0:
            return d
        return d - self.beta * ((d @ u) @ u.T)

    def distance(self, adversarial: Tensor, reference: Tensor) -> Tensor:
        """Scalar elementwise-mean squared distance; reference always detached.

        Flatten token/hidden coordinates per observation BEFORE calling this.
        Pass the same source feature space for every sample.
        """
        if adversarial.shape != reference.shape:
            raise ValueError('feature/reference shapes differ')
        dt = _analysis_dtype(adversarial)
        d = adversarial.to(dt) - reference.detach().to(device=adversarial.device, dtype=dt)
        return self(d).square().mean()


def retained_energy(displacements: Tensor, filt: ResidualFilter) -> float:
    """Ratio of total filtered energy to total original energy; not mean ratios."""
    d = displacements.detach().to(dtype=_analysis_dtype(displacements))
    denom = d.square().sum()
    if not bool(torch.isfinite(d).all()) or float(denom) <= 0:
        raise ValueError('energy ratio undefined for zero/non-finite displacement')
    with torch.no_grad():
        return float(filt(d).square().sum() / denom)
