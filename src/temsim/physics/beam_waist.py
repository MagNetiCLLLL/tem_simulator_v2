from __future__ import annotations
import numpy as np

def branch_waist_candidates(branch, z_min_mm=-np.inf, z_max_mm=np.inf, min_rays=5):
    """Current-weighted plot-node diagnostics with one population per bracket.

    Splitting a ray into copies with divided weights must not move a waist.
    Stops cannot create one by removing the broad part of a bundle. Precise
    focus qualification uses independently executed checkpoints, not markers.
    """
    z = np.asarray(branch.z, float)
    arrays = [np.asarray(getattr(branch, key), float) for key in ("x", "y", "tx", "ty")]
    blocked = np.asarray(branch.blocked_z, float)
    raw_weight = getattr(branch, "ray_weight", None)
    weight = np.ones(blocked.size) if raw_weight is None else np.asarray(raw_weight, float)
    if (z.ndim != 1 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0)
            or blocked.ndim != 1 or any(a.shape != (z.size, blocked.size) for a in arrays)
            or weight.shape != blocked.shape or np.any(~np.isfinite(weight)) or np.any(weight < 0)):
        raise ValueError("Invalid current-weighted waist history")
    for j in range(1, z.size-1):
        if not z_min_mm < z[j] < z_max_mm:
            continue
        valid = (weight > 0) & (np.isnan(blocked) | (blocked >= z[j+1]))
        for a in arrays:
            valid &= np.all(np.isfinite(a[j-1:j+2]), axis=0)
        if valid.sum() < min_rays:
            continue
        w = weight[valid]/weight[valid].sum()
        x, y, tx, ty = [a[j-1:j+2, valid] for a in arrays]
        x, y, tx, ty = [a-(a@w)[:, None] for a in (x, y, tx, ty)]
        rms = np.sqrt((x*x+y*y)@w)
        corr = (x*tx+y*ty)@w
        if rms[1] <= rms[0] and rms[1] <= rms[2] and corr[0] < 0 < corr[2]:
            yield dict(z_mm=float(z[j]), rms_radius_mm=float(rms[1]*1e3),
                correlation=float(corr[1]), ray_count=int(valid.sum()),
                kind="current-weighted ensemble RMS beam waist")


def detect_beam_waist(branch, z_min_mm, z_max_mm, min_rays=5):
    return min(branch_waist_candidates(branch, z_min_mm, z_max_mm, min_rays),
               key=lambda row: row["rms_radius_mm"], default=None)
