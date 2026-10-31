"""Math helpers for a protected cross-view interaction experiment.

No tensor library is imported: inclusion_exclusion works on Torch tensors;
the tiny exact cone solver is plain Python, intended for low-dimensional
gradient-Gram diagnostics or independently testing the tensor implementation.

The cone projection is standard quadratic programming (as in GEM), not a new
optimizer. The experimental hypothesis concerns native/synthesis mixed error.
"""
import itertools
import math


def inclusion_exclusion(a11, a10, a01, a00):
    """Preserve autograd when inputs are tensors. All four need same reference."""
    return a11 - a10 - a01 + a00


def rms(values):
    values = list(values)
    if not values:
        raise ValueError('empty action subset')
    return math.sqrt(sum(float(x)**2 for x in values) / len(values))


def four_corner_stats(corners_r, corners_n, denominator_floor=1e-6):
    """All values are flat action lists in one fixed common coordinate system.

    The denominator floor must be fixed before looking at errors; a ratio for
    below-floor native interactions is returned as None rather than inflated.
    """
    if denominator_floor <= 0:
        raise ValueError('denominator_floor must be positive')
    keys = ['11', '10', '01', '00']
    dim = len(corners_n['00'])
    assert all(len(c[k]) == dim for c in [corners_r, corners_n] for k in keys)
    assert all(math.isfinite(float(x)) for c in [corners_r, corners_n]
               for k in keys for x in c[k])

    def diffs(c):
        d1 = [x-y for x, y in zip(c['10'], c['00'])]
        d2 = [x-y for x, y in zip(c['01'], c['00'])]
        dj = [x-y for x, y in zip(c['11'], c['00'])]
        it = [a-b-c+d for a, b, c, d in zip(c['11'], c['10'], c['01'], c['00'])]
        return d1, d2, dj, it

    dr1, dr2, drj, ir = diffs(corners_r)
    dn1, dn2, dnj, inn = diffs(corners_n)
    err_i = rms(x-y for x, y in zip(ir, inn))
    err_j = rms(x-y for x, y in zip(drj, dnj))
    norm_i, norm_j = rms(inn), rms(dnj)
    bound = max(0.0, rms(ir)-err_i-rms(dn1)-rms(dn2))
    return dict(
        source_native_interaction_rms=norm_i,
        synthesis_interaction_rms=rms(ir),
        interaction_prediction_error_rms=err_i,
        interaction_relative_error=None if norm_i < denominator_floor else err_i/norm_i,
        source_native_joint_rms=norm_j,
        synthesis_joint_rms=rms(drj),
        joint_prediction_error_rms=err_j,
        joint_relative_error=None if norm_j < denominator_floor else err_j/norm_j,
        native_solo_main_rms=rms(dn1), native_solo_wrist_rms=rms(dn2),
        synthesis_solo_main_rms=rms(dr1), synthesis_solo_wrist_rms=rms(dr2),
        native_joint_lower_bound_from_calibrated_interaction=bound,
        verified_bound=bound <= norm_j+1e-12,
        denominator_floor=denominator_floor)


def protected_cone_direction(gradient, constraint_gradients, radius, tol=1e-10):
    """Exact linear-objective optimum on an L2 ball intersected with <=2 cones.

    max <g,d> s.t. <h_i,d> <= 0 and ||d||_2 <= radius.
    Return d = radius * Pi_C(g) / ||Pi_C(g)||, or zero when Pi_C(g)=0.
    Projection onto the cone enumerates the 0/1/2 active-set candidates.
    Protection is first-order only; caller needs a finite-step constraint check.
    """
    g = [float(x) for x in gradient]
    hs = [[float(x) for x in h] for h in constraint_gradients]
    if len(hs) > 2 or any(len(h) != len(g) for h in hs):
        raise ValueError('solver supports zero, one, or two same-sized constraints')
    if radius < 0 or not math.isfinite(radius):
        raise ValueError('invalid radius')
    if not all(math.isfinite(x) for row in [g]+hs for x in row):
        raise ValueError('nonfinite gradient')
    dot = lambda x, y: sum(a*b for a, b in zip(x, y))
    # Normalize h so rank tolerance and feasibility tolerance do not depend on
    # arbitrary positive scaling of the squared-action constraints.
    hs = [[x/math.sqrt(dot(h, h)) for x in h] for h in hs if dot(h, h) > tol**2]
    candidates = []
    for k in range(len(hs)+1):
        for active in itertools.combinations(range(len(hs)), k):
            if k == 0:
                multipliers = []
            elif k == 1:
                multipliers = [dot(hs[active[0]], g)]
            else:
                h0, h1 = (hs[j] for j in active)
                c = dot(h0, h1)
                det = 1-c*c
                if det <= tol:
                    # Parallel/opposite constraints have a valid single-active
                    # representation unless g is already feasible.
                    continue
                b0, b1 = dot(h0, g), dot(h1, g)
                multipliers = [(b0-c*b1)/det, (b1-c*b0)/det]
            if any(x < -tol for x in multipliers):
                continue
            candidate = [x-sum(lam*hs[j][d] for j, lam in zip(active, multipliers))
                         for d, x in enumerate(g)]
            if any(dot(h, candidate) > tol*(1+math.sqrt(dot(g, g))) for h in hs):
                continue
            distance = sum((x-y)**2 for x, y in zip(candidate, g))
            candidates.append((distance, candidate))
    if not candidates:
        raise ArithmeticError('no KKT-feasible projection; inspect conditioning')
    projected = min(candidates, key=lambda pair: pair[0])[1]
    norm = math.sqrt(dot(projected, projected))
    if norm <= tol or radius == 0:
        return [0.0 for _ in g]
    return [radius*x/norm for x in projected]


def gradient_gram_cone_direction(g, h1, h2, radius, eps=1e-10):
    """Tensor implementation without stacking texture-sized gradients.

    Uses three texture-sized tensors and scalar Gram products. Supports CUDA
    and autograd-produced gradients; the returned update is detached.
    Import torch only when this function is used by the experiment controller.
    """
    import torch
    with torch.no_grad():
        gs = [x.detach().float() for x in [g, h1, h2]]
        if not all(bool(torch.isfinite(x).all()) for x in gs):
            raise ValueError('nonfinite gradient')
        if radius < 0 or not math.isfinite(radius):
            raise ValueError('invalid radius')
        g = gs[0]
        hs = [h/h.norm() for h in gs[1:] if float(h.norm()) > eps]
        dot = lambda a, b: float((a*b).sum())
        candidates = []
        gn = float(g.norm())
        for k in range(len(hs)+1):
            for active in itertools.combinations(range(len(hs)), k):
                if k == 0:
                    multipliers = []
                elif k == 1:
                    multipliers = [dot(hs[active[0]], g)]
                else:
                    u, v = (hs[j] for j in active)
                    c = dot(u, v)
                    det = 1-c*c
                    if det <= eps:
                        continue
                    b0, b1 = dot(u, g), dot(v, g)
                    multipliers = [(b0-c*b1)/det, (b1-c*b0)/det]
                if any(x < -eps*(1+gn) for x in multipliers):
                    continue
                p = g.clone()
                for j, lam in zip(active, multipliers):
                    p.sub_(hs[j], alpha=lam)
                if any(dot(h, p) > 1e-5*(1+gn) for h in hs):
                    continue
                candidates.append((float((p-g).square().sum()), p))
        if not candidates:
            raise ArithmeticError('no KKT-feasible projection')
        p = min(candidates, key=lambda pair: pair[0])[1]
        norm = float(p.norm())
        return torch.zeros_like(p) if norm <= eps else p*(radius/norm)
