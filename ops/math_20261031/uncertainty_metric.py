"""Decision-space ellipsoidal renderer uncertainty, with envelope gradients.

These are standard robust optimization / trust-region calculations. Any useful
contribution must come from a validated renderer-error model and native policy
updates, not from naming this solver as a new theorem.
"""
import torch


def fit_shape(error, ridge=0.01):
    """Origin-centered second moment: a systematic error is not subtracted."""
    error = error.detach().double()
    covariance = error.T @ error / len(error)
    scale = covariance.diag().mean().clamp_min(1e-12)
    covariance = covariance + ridge * scale * torch.eye(error.shape[1], device=error.device, dtype=error.dtype)
    values, vectors = torch.linalg.eigh(covariance)
    return dict(values=values.clamp_min(1e-15).cpu(), vectors=vectors.cpu())


def radius_from_errors(error, shape, quantile=0.9):
    values, vectors = shape['values'], shape['vectors']
    coordinates = error.detach().double().cpu() @ vectors
    scores = (coordinates.square()/values).sum(-1).sqrt()
    # Empirical calibration; this is not an adversarial coverage certificate.
    return float(torch.quantile(scores, quantile))


def mahalanobis_squared(error, shape):
    x = error.detach().double()
    values = shape['values'].to(x.device)
    vectors = shape['vectors'].to(x.device)
    return ((x @ vectors).square()/values).sum(-1)


def solve_worst_action(displacement, shape, radius, relative=False, iterations=64):
    """Solve min_e ||u+e||^2 subject to e'C^-1 e <= r(u)^2.

    r(u)=radius for absolute, radius*||u|| for relative uncertainty. The root and
    inner optimizer are intentionally detached; differentiation is via envelope.
    Returns the exact value and its gradient with respect to u, both detached.
    """
    x = displacement.detach().double()
    values = shape['values'].to(x.device)
    vectors = shape['vectors'].to(x.device)
    v = x @ vectors
    radius_squared = torch.full_like(v[..., 0], float(radius)**2)
    if relative:
        radius_squared = radius_squared * x.square().sum(-1)
    inside = (v.square()/values).sum(-1) <= radius_squared
    zero_radius = radius_squared <= 0
    # f(mu)=sum c_i*v_i^2/(c_i+mu)^2 decreases monotonically. This upper
    # bracket follows from c_i+mu >= mu and avoids an unbounded search loop.
    high = ((values*v.square()).sum(-1)/radius_squared.clamp_min(1e-300)).sqrt()
    high = high.clamp_min(1e-15)
    low = torch.zeros_like(high)
    for _ in range(iterations):
        middle = (low+high)/2
        constraint = (values*v.square()/(values+middle[..., None]).square()).sum(-1)
        too_small = constraint > radius_squared
        low = torch.where(too_small, middle, low)
        high = torch.where(too_small, high, middle)
    multiplier = (low+high)/2
    factors = multiplier[..., None]/(values+multiplier[..., None])
    factors = torch.where(inside[..., None], torch.zeros_like(factors), factors)
    factors = torch.where(zero_radius[..., None], torch.ones_like(factors), factors)
    multiplier = torch.where(inside | zero_radius, torch.zeros_like(multiplier), multiplier)
    worst = (factors*v) @ vectors.T
    value = worst.square().sum(-1)
    gradient = 2*worst
    if relative:
        # The uncertainty radius depends on u. Omitting this term produces an
        # incorrect gradient despite the correct forward robust score.
        gradient = gradient - 2*multiplier[..., None]*float(radius)**2*x
    error = worst-x
    info = dict(worst_action=worst, adversarial_error=error, multiplier=multiplier,
                inside=inside, radius_squared=radius_squared,
                error_mahalanobis_squared=((error@vectors).square()/values).sum(-1))
    return value, gradient, info


def robust_action_score(displacement, model, mode='relative', rho_scale=1.0, reduction='mean'):
    """Drop-in differentiable score for a [..., 15] action displacement.

    Sum of coordinate squares is divided by 15, matching ordinary action MSE.
    reduction='none' returns one scalar per input; 'mean' averages samples.
    """
    if mode not in ['absolute', 'relative']:
        raise ValueError(mode)
    entry = model[mode]
    value, gradient, _ = solve_worst_action(displacement, entry, entry['radius']*rho_scale,
                                           relative=mode == 'relative')
    x = displacement.float()
    v = value.to(device=x.device, dtype=x.dtype)
    g = gradient.to(device=x.device, dtype=x.dtype)
    # This expression has the exact inner optimum as forward value and the
    # envelope gradient in backward; no derivative through bisection is needed.
    tangent = (x*g.detach()).sum(-1)
    score = (v.detach()+(tangent-tangent.detach()))/x.shape[-1]
    if reduction == 'none':
        return score
    if reduction == 'mean':
        return score.mean()
    raise ValueError(reduction)


def quadratic_metric_score(displacement, model, reduction='mean'):
    x = displacement.float()
    matrix = model['generalized_metric'].to(device=x.device, dtype=x.dtype)
    score = ((x@matrix)*x).sum(-1)/x.shape[-1]
    return score if reduction == 'none' else score.mean()


def fit_uncertainty_model(predicted, native, shape_mask, calibration_mask,
                          ridge=0.01, quantile=0.9, generalized_rank=5):
    predicted, native = predicted.detach().double().cpu(), native.detach().double().cpu()
    error = native-predicted
    norm = predicted.norm(dim=-1).clamp_min(1e-8)
    result = dict(dimension=predicted.shape[-1], ridge=ridge, quantile=quantile)
    for mode in ['absolute', 'relative']:
        errors = error if mode == 'absolute' else error/norm[:, None]
        shape = fit_shape(errors[shape_mask], ridge=ridge)
        shape['radius'] = radius_from_errors(errors[calibration_mask], shape, quantile=quantile)
        result[mode] = shape
    # A fixed generalized-Rayleigh metric is an inexpensive contrast with the
    # adaptive ellipsoidal score. This does not constitute new eigentheory.
    c = result['absolute']
    invsqrt = (c['vectors']/c['values'].sqrt()) @ c['vectors'].T
    signal = native[shape_mask].T@native[shape_mask]/int(shape_mask.sum())
    eigenvalues, eigenvectors = torch.linalg.eigh(invsqrt@signal@invsqrt)
    rank = min(generalized_rank, len(eigenvalues))
    directions = invsqrt@eigenvectors[:, -rank:]
    metric = directions@directions.T
    result['generalized_metric'] = metric*(predicted.shape[-1]/metric.trace())
    result['generalized_eigenvalues'] = eigenvalues
    result['generalized_rank'] = rank
    return result
