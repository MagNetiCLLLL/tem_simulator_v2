import numpy as np



def _rms_and_correlation(branch, j, valid):

    x = np.asarray(branch.x, float)

    y = np.asarray(branch.y, float)

    tx = np.asarray(branch.tx, float)

    ty = np.asarray(branch.ty, float)

    dx = x[j, valid] - x[j, valid].mean()

    dy = y[j, valid] - y[j, valid].mean()

    dtx = tx[j, valid] - tx[j, valid].mean()

    dty = ty[j, valid] - ty[j, valid].mean()

    rms = np.sqrt(np.mean(dx * dx + dy * dy))

    correlation = np.mean(dx * dtx + dy * dty)

    return float(rms), float(correlation)



def first_crossover_after_lens(branch, lens_z_mm, name, stop_z_mm=None,

                               minimum_rays=5, start_margin_mm=0.25):

    """Return the first verified real crossover downstream of a lens.


    A marker is accepted only when the RMS beam radius is a local minimum and

    the ensemble changes from converging to diverging. The function does not

    require the waist to be 'between' any two named components; stop_z_mm is

    only an optional search limit.

    """

    z = np.asarray(branch.z, float)

    x = np.asarray(branch.x, float)

    y = np.asarray(branch.y, float)

    tx = np.asarray(branch.tx, float)

    ty = np.asarray(branch.ty, float)

    blocked = np.asarray(branch.blocked_z, float)


    mask = z > float(lens_z_mm) + float(start_margin_mm)

    if stop_z_mm is not None:

        mask &= z < float(stop_z_mm)

    indices = np.flatnonzero(mask)


    for j in indices:

        if j <= 0 or j >= len(z) - 1:

            continue

        valid = (

            np.isfinite(x[j]) & np.isfinite(y[j]) &

            np.isfinite(tx[j]) & np.isfinite(ty[j]) &

            (np.isnan(blocked) | (blocked >= z[j]))

        )

        if int(valid.sum()) < int(minimum_rays):

            continue

        left_rms, left_corr = _rms_and_correlation(branch, j - 1, valid)

        rms, _ = _rms_and_correlation(branch, j, valid)

        right_rms, right_corr = _rms_and_correlation(branch, j + 1, valid)

        if rms <= left_rms and rms <= right_rms and left_corr < 0.0 < right_corr:

            return {

                "name": name,

                "z_mm": float(z[j]),

                "rms_radius_mm": rms * 1000.0,

                "verified": True,

                "source_lens_z_mm": float(lens_z_mm),

            }

    return None

