"""Frozen-color surface interventions and dimensionless action diagnostics.

No renderer/model imports: these routines also run on the local CPU.
Masks index exported PNG rows (not a second, independently flipped UV grid).
"""
import hashlib
import numpy as np


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def block_ids(shape, grid=4):
    h, w = shape[:2]
    if h % grid or w % grid:
        raise ValueError('Equal-area grid must divide texture dimensions')
    y, x = np.indices((h, w))
    return (y // (h // grid)) * grid + x // (w // grid)


def binary_texture(full, base, keep, grid=4):
    full, base = np.asarray(full), np.asarray(base)
    if full.shape != base.shape or full.dtype != np.uint8 or base.dtype != np.uint8:
        raise ValueError('Expected matching uint8 RGB exported textures')
    keep = tuple(sorted(set(int(x) for x in keep)))
    if any(x < 0 or x >= grid * grid for x in keep):
        raise ValueError('Invalid block id')
    mask = np.isin(block_ids(full.shape, grid), keep)
    return np.where(mask[..., None], full, base).astype(np.uint8)


def opacity_texture(full, base, opacity):
    if not 0 <= opacity <= 1:
        raise ValueError('Invalid opacity')
    return np.rint(base.astype(float) + opacity * (full.astype(float)-base)).clip(0, 255).astype(np.uint8)


def texture_stats(png, base, keep=None, grid=4):
    d = (png.astype(float)-base.astype(float))/255.
    out = dict(rms=float(np.sqrt(np.mean(d*d))), linf=float(np.max(np.abs(d))),
               tv=float(np.abs(np.diff(d, axis=0)).mean()+np.abs(np.diff(d, axis=1)).mean()),
               nonzero_png_fraction=float(np.any(png != base, axis=-1).mean()),
               pixel_sha256=array_sha(png))
    if keep is not None:
        m = np.isin(np.arange(grid*grid).reshape(grid, grid), keep).astype(int)
        # Includes the exterior boundary, measured in grid-cell edge units.
        padded = np.pad(m, 1, mode='constant')
        out.update(kept_blocks=list(sorted(keep)), area_fraction=len(keep)/(grid*grid),
                   perimeter_cell_edges=int(np.abs(np.diff(padded, axis=0)).sum()+
                                            np.abs(np.diff(padded, axis=1)).sum()))
    return out


def pose(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] != 7 or not np.isfinite(a).all():
        raise ValueError('Expected finite action chunk with at least five 7-D steps')
    return a[:5, :6]


def action_metrics(candidate, teacher, plain, scale):
    c, t, p = [np.asarray(a, dtype=np.float64)[:5] for a in (candidate, teacher, plain)]
    pc, pt, pp = [pose(a) for a in (c, t, p)]
    e, signal, actual = pc-pt, pt-pp, pc-pp
    den = float(np.square(signal).sum())
    err = float(np.square(e).sum())
    val = float(np.square(actual).sum())
    norm_product = np.sqrt(den*val)
    scale = np.asarray(scale, dtype=np.float64)[:6]
    return dict(pose_error_sum=err, teacher_signal_sum=den, candidate_signal_sum=val,
                relative_fidelity=(err/den if den > 1e-12 else None),
                action_energy_retention=(val/den if den > 1e-12 else None),
                displacement_cosine=(float(np.sum(actual*signal)/norm_product) if norm_product > 1e-12 else None),
                normalized_pose_mse=float(np.mean(e*e)),
                translation_control_mse=float(np.mean(np.square(e[:, :3]*scale[:3]))),
                rotation_control_mse=float(np.mean(np.square(e[:, 3:]*scale[3:]))),
                normalized_gripper_sign_disagreement=float(np.mean((c[:, 6] > 0)!=(t[:, 6] > 0))))


def pooled_metrics(rows):
    if not rows:
        raise ValueError('Empty state group')
    err = sum(r['pose_error_sum'] for r in rows)
    den = sum(r['teacher_signal_sum'] for r in rows)
    act = sum(r['candidate_signal_sum'] for r in rows)
    defined = [r['displacement_cosine'] for r in rows if r['displacement_cosine'] is not None]
    return dict(states=len(rows), pooled_relative_fidelity=(err/den if den > 1e-12 else None),
                pooled_energy_retention=(act/den if den > 1e-12 else None),
                mean_displacement_cosine=(float(np.mean(defined)) if defined else None),
                undefined_cosine_states=len(rows)-len(defined),
                fraction_fidelity_le_0_25=float(np.mean([r['relative_fidelity'] is not None and
                                                       r['relative_fidelity'] <= .25 for r in rows])),
                pose_error_sum=err, teacher_signal_sum=den)


def best_deletion(keep, losses):
    """Tie break deterministically by removed id; exact native loss is supplied."""
    choices = [(float(losses[j]), int(j)) for j in keep]
    if not all(np.isfinite(x[0]) for x in choices):
        raise ValueError('Non-finite native query loss')
    _, removed = min(choices)
    return tuple(j for j in keep if j != removed), removed
