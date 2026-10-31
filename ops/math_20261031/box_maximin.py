"""Max-min common ascent in the FINAL executable parameter-update box.

No model or simulator dependency. scipy.optimize.linprog is preferred; a pure
Python no-regret fallback returns a feasible fractional update and an explicit
primal/dual gap. Neither output may be passed through sign() afterwards.

The LP and its minimax dual are standard convex optimization, not a new theorem.
"""
import math


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def update_box(theta, eta, parameter_limit=5.0):
    """Actual delta box after the production theta clip, NOT an ideal eta cube."""
    if eta <= 0 or parameter_limit <= 0:
        raise ValueError('eta and parameter_limit must be positive')
    if any(not math.isfinite(x) or abs(x) > parameter_limit+1e-10 for x in theta):
        raise ValueError('theta must be finite and within parameter bounds')
    lower = [max(-eta, -parameter_limit-x) for x in theta]
    upper = [min(eta, parameter_limit-x) for x in theta]
    return lower, upper


def normalize_rows(gradients, mode='l1', eps=1e-12):
    """Keep zero rows: they are genuine obstructions to strict common ascent."""
    if mode not in ['raw', 'l1']:
        raise ValueError('mode must be raw or l1')
    scales = [1.0 if mode == 'raw' else max(sum(abs(x) for x in g), eps)
              for g in gradients]
    return [[x/s for x in g] for g, s in zip(gradients, scales)], scales


def mean_sign(gradients, lower, upper):
    combined = [sum(g[j] for g in gradients)/len(gradients) for j in range(len(lower))]
    return [u if h > 0 else l if h < 0 else min(u, max(l, 0.0))
            for h, l, u in zip(combined, lower, upper)]


def pcgrad_direction(gradients, order=None):
    """Deterministic PCGrad baseline with projections onto ORIGINAL other rows.

    The order is explicitly supplied/frozen rather than optimized on outcomes.
    This is a baseline implementation, not a new algorithm.
    """
    n = len(gradients)
    order = list(range(n)) if order is None else list(order)
    if sorted(order) != list(range(n)):
        raise ValueError('order must be a permutation')
    projected = []
    for i, original in enumerate(gradients):
        row = list(original)
        for j in order:
            if i == j:
                continue
            other = gradients[j]
            product = _dot(row, other)
            square = _dot(other, other)
            if product < 0 and square > 0:
                row = [x-product*y/square for x, y in zip(row, other)]
        projected.append(row)
    return [sum(g[j] for g in projected)/n for j in range(len(gradients[0]))]


def scaled_pcgrad(gradients, lower, upper, take_sign=False):
    """Fit PCGrad to the same box, with and without the lossy sign operation."""
    v = pcgrad_direction(gradients)
    if take_sign:
        return [u if h > 0 else l if h < 0 else min(u, max(l, 0.0))
                for h, l, u in zip(v, lower, upper)]
    ratios = [(u/h if h > 0 else l/h) for h, l, u in zip(v, lower, upper) if h != 0]
    if not ratios:
        return [0.0]*len(v)
    scale = max(0.0, min(ratios))
    return [scale*x for x in v]


def _validate(gradients, lower, upper, offsets):
    if not gradients or not lower or len(lower) != len(upper):
        raise ValueError('nonempty gradients and consistent bounds required')
    if any(len(g) != len(lower) for g in gradients) or len(offsets) != len(gradients):
        raise ValueError('dimension mismatch')
    if not all(math.isfinite(x) for g in gradients for x in g):
        raise ValueError('nonfinite gradients')
    if not all(math.isfinite(x) for x in list(lower)+list(upper)+list(offsets)):
        raise ValueError('nonfinite bounds or offsets')
    if any(l > u for l, u in zip(lower, upper)):
        raise ValueError('empty update box')


def _dual(gradients, lower, upper, offsets, weights):
    mixed = [sum(w*g[j] for w, g in zip(weights, gradients)) for j in range(len(lower))]
    direction = [u if h > 0 else l if h < 0 else min(u, max(l, 0.0))
                 for h, l, u in zip(mixed, lower, upper)]
    value = _dot(weights, offsets)+_dot(mixed, direction)
    return value, direction


def solve_box_maximin(gradients, lower, upper, offsets=None, method='auto',
                      iterations=10000, tolerance=1e-5):
    """Maximize min_i(<g_i,delta> + offsets_i), lower <= delta <= upper.

    offsets can encode FIXED-radius uncertainty/remainder penalties. They are
    not a certified native bound unless their assumptions are independently
    justified. A direction-dependent uncertainty set requires a larger LP.
    """
    gradients = [[float(x) for x in row] for row in gradients]
    lower, upper = list(lower), list(upper)
    offsets = [0.0]*len(gradients) if offsets is None else list(offsets)
    _validate(gradients, lower, upper, offsets)
    if method not in ['auto', 'scipy', 'mw']:
        raise ValueError('unknown solver')
    if method != 'mw':
        try:
            import numpy as np
            import scipy
            from scipy.optimize import linprog
        except ImportError:
            if method == 'scipy':
                raise
        else:
            if tuple(int(x) for x in scipy.__version__.split('.')[:2]) < (1, 6):
                if method == 'scipy':
                    raise RuntimeError('SciPy >=1.6 is required for HiGHS and its dual marginals')
                return solve_box_maximin(gradients, lower, upper, offsets, method='mw',
                                         iterations=iterations, tolerance=tolerance)
            n, d = len(gradients), len(lower)
            matrix = np.column_stack([-np.asarray(gradients), np.ones(n)])
            result = linprog([0.0]*d+[-1.0], A_ub=matrix, b_ub=offsets,
                             bounds=list(zip(lower, upper))+[(None, None)], method='highs')
            if not result.success:
                raise RuntimeError('HiGHS failed: '+result.message)
            delta = result.x[:-1].tolist()
            weights = (-result.ineqlin.marginals).clip(0).tolist()
            norm = sum(weights)
            weights = [w/norm for w in weights]
            gains = [_dot(g, delta)+c for g, c in zip(gradients, offsets)]
            upper_bound, _ = _dual(gradients, lower, upper, offsets, weights)
            return dict(delta=delta, gains=gains, objective=min(gains), dual_upper=upper_bound,
                        duality_gap=max(0.0, upper_bound-min(gains)), weights=weights,
                        method='scipy-highs', iterations=int(result.nit))
    if iterations < 1:
        raise ValueError('iterations must be positive')
    n, d = len(gradients), len(lower)
    max_gain = max(abs(c)+sum(abs(g[j])*max(abs(lower[j]), abs(upper[j])) for j in range(d))
                   for g, c in zip(gradients, offsets))
    max_gain = max(max_gain, 1e-20)
    learning_rate = math.sqrt(2*math.log(max(n, 2))/iterations)
    log_weights = [0.0]*n
    accumulated = [0.0]*d
    initial = [min(u, max(l, 0.0)) for l, u in zip(lower, upper)]
    best_delta = initial
    best_lower = min(_dot(g, initial)+c for g, c in zip(gradients, offsets))
    best_upper, best_weights = float('inf'), [1.0/n]*n
    used = 0
    for k in range(1, iterations+1):
        maximum = max(log_weights)
        exp_weights = [math.exp(x-maximum) for x in log_weights]
        weights = [x/sum(exp_weights) for x in exp_weights]
        dual, direction = _dual(gradients, lower, upper, offsets, weights)
        if dual < best_upper:
            best_upper, best_weights = dual, list(weights)
        gains = [_dot(g, direction)+c for g, c in zip(gradients, offsets)]
        accumulated = [x+y for x, y in zip(accumulated, direction)]
        log_weights = [x-learning_rate*y/max_gain for x, y in zip(log_weights, gains)]
        used = k
        if k % 8 == 0 or k == iterations:
            average = [x/k for x in accumulated]
            primal = min(_dot(g, average)+c for g, c in zip(gradients, offsets))
            if primal > best_lower:
                best_lower, best_delta = primal, average
            if best_upper-best_lower <= tolerance:
                break
    final_gains = [_dot(g, best_delta)+c for g, c in zip(gradients, offsets)]
    return dict(delta=best_delta, gains=final_gains, objective=min(final_gains),
                dual_upper=best_upper, duality_gap=max(0.0, best_upper-min(final_gains)),
                weights=best_weights, method='pure-python-multiplicative-weights', iterations=used)
