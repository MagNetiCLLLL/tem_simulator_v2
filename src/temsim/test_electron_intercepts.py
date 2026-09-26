"""Optional compiled chronological contacts for captured diagnostic hardware.

Coordinates remain global XYZ metres. Aperture masks retain their existing
millimetre arithmetic, including the original zero-radius distinctions. This
module packs executed scene geometry, never a particle source or checkpoint.
Unsupported tip surfaces or aperture masks use the complete reference method.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _jit(function):
    return function if njit is None else njit(cache=True, nogil=True, fastmath=False)(function)


def prepare_compiled_intercepts(scene):
    """Return ``((rows,), reasons)`` or None for unrepresented geometry.

    Each immutable float64 row stores kind and the original scalar geometry;
    its index identifies a reason. Row order preserves reference tie-breaking:
    unsupported forward planes, flat tip, bores, then active apertures.
    """
    if njit is None:
        return None
    from temsim.test_electron_scene import TestElectronScene, _Bore
    from temsim.optics.electron_gun.aperture import GunAperture
    from temsim.optics.condenser_aperture import ContinuousApertureComponent
    from temsim.optics.objective_aperture import ObjectiveApertureComponent
    from temsim.optics.selected_area_aperture import SelectedAreaApertureComponent
    from temsim.optics.energy_filter_entrance_aperture import EnergyFilterEntranceApertureComponent

    if (type(scene) is not TestElectronScene or not scene._flat_cathode
            or getattr(scene.diagnostic_segment_stop, "__func__", None) is not TestElectronScene.diagnostic_segment_stop
            or getattr(scene._tip_intercept, "__func__", None) is not TestElectronScene._tip_intercept):
        return None
    circles = (ContinuousApertureComponent, ObjectiveApertureComponent,
               SelectedAreaApertureComponent, EnergyFilterEntranceApertureComponent)
    rows, reasons = [], []
    for plane, reason in scene._unsupported_stops:
        if not math.isfinite(plane):
            return None
        rows.append((0., plane, 0., 0., 0., 0., 0., 0.))
        reasons.append(reason)
    rows.append((1., 0., 0., 0., 0., 0., 0., 0.))
    reasons.append("tip_return")
    for bore in scene._bores:
        if type(bore) is not _Bore:
            return None
        if (not np.isfinite((bore.lower_m, bore.upper_m, bore.inner_m)).all()
                or math.isnan(bore.outer_m) or bore.outer_m <= bore.inner_m
                or bore.lower_m > bore.upper_m or bore.inner_m <= 0.):
            return None
        rows.append((2., bore.lower_m, bore.upper_m, bore.inner_m, bore.outer_m, 0., 0., 0.))
        reasons.append(f"hardware:{bore.key}")
    for aperture in scene._apertures:
        mask = getattr(aperture, "transmission_mask", None)
        block_nonpositive, passes_all = True, False
        if callable(mask):
            cls = type(aperture)
            if cls not in (*circles, GunAperture) or getattr(mask, "__func__", None) is not cls.transmission_mask:
                return None
            if cls is GunAperture:
                if getattr(aperture, "_slit_mode", False) and getattr(aperture, "_slit_profile", None) is not None:
                    return None
                passes_all = not aperture.enabled
            else:
                block_nonpositive = False
        try:
            values = (float(aperture.z_mm)*1e-3, float(aperture.offset_x_mm),
                      float(aperture.offset_y_mm), float(aperture.radius_mm))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(values).all():
            return None
        rows.append((3., *values, float(block_nonpositive), float(passes_all), 0.))
        reasons.append(f"aperture:{aperture.key}")
    packed = np.array(rows, dtype=np.float64).reshape(-1, 8)
    packed.setflags(write=False)
    return (packed,), tuple(reasons)


@_jit
def _circle_passes(x, y, row):
    if row[6] != 0.:
        return True
    if row[5] != 0. and row[4] <= 0.:
        return False
    return math.hypot(x*1e3-row[2], y*1e3-row[3]) <= row[4]


@_jit
def _add_root(cuts, count, root):
    if 0. <= root <= 1.:
        cuts[count] = root
        count += 1
    return count


@_jit
def _compiled_bore(start, end, row):
    lower, upper, inner, outer = row[1], row[2], row[3], row[4]
    if max(start[2], end[2]) < lower or min(start[2], end[2]) > upper:
        return -1.
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    if lower == upper and dz != 0.:
        fraction = (lower-start[2])/dz
        radius = math.hypot(start[0]+fraction*dx, start[1]+fraction*dy)
        if 0. <= fraction <= 1. and inner <= radius <= outer:
            return fraction
        return -1.
    # 2 endpoints + 2 axial faces + 2 roots for each radial shell surface.
    cuts = np.empty(8)
    cuts[0], cuts[1] = 0., 1.
    count = 2
    if dz != 0.:
        count = _add_root(cuts, count, (lower-start[2])/dz)
        count = _add_root(cuts, count, (upper-start[2])/dz)
    a = dx*dx+dy*dy
    b = 2.*(start[0]*dx+start[1]*dy)
    radius_squared = start[0]*start[0]+start[1]*start[1]
    for radius in (inner, outer):
        if not math.isfinite(radius):
            continue
        c = radius_squared-radius*radius
        if a == 0.:
            if b != 0.:
                count = _add_root(cuts, count, -c/b)
            continue
        discriminant = b*b-4.*a*c
        if discriminant < 0.:
            continue
        root = math.sqrt(discriminant)
        q = -.5*(b+math.copysign(root, b))
        if q == 0.:
            count = _add_root(cuts, count, -b/(2.*a))
        else:
            count = _add_root(cuts, count, q/a)
            count = _add_root(cuts, count, c/q)
    ordered = np.sort(cuts[:count])
    for index in range(len(ordered)-1):
        first, last = ordered[index], ordered[index+1]
        if first == last:
            continue
        fraction = .5*(first+last)
        z = start[2]+fraction*dz
        radial = math.hypot(start[0]+fraction*dx, start[1]+fraction*dy)
        if lower <= z <= upper and inner <= radial <= outer:
            return first
    radial = math.hypot(end[0], end[1])
    if lower <= end[2] <= upper and inner <= radial <= outer:
        return 1.
    return -1.


@_jit
def compiled_intercept(start, end, data):
    """Return earliest ``(fraction, row_index)``; index -1 means no contact."""
    rows, = data
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    best, reason_index = math.inf, -1
    for index in range(len(rows)):
        row = rows[index]
        fraction = -1.
        kind = row[0]
        if kind == 0.:
            if dz > 0. and start[2] <= row[1] <= end[2]:
                fraction = (row[1]-start[2])/dz
        elif kind == 1.:
            if min(start[2], end[2]) < 0.:
                if start[2] < 0.:
                    fraction = 0.
                elif dz < 0.:
                    fraction = -start[2]/dz
        elif kind == 2.:
            fraction = _compiled_bore(start, end, row)
        else:
            plane = row[1]
            if dz == 0.:
                if start[2] != plane:
                    continue
                if not _circle_passes(start[0], start[1], row):
                    fraction = 0.
                elif _circle_passes(end[0], end[1], row):
                    continue
                else:
                    low, high = 0., 1.
                    for _ in range(48):
                        mid = .5*(low+high)
                        if _circle_passes(start[0]+mid*dx, start[1]+mid*dy, row):
                            low = mid
                        else:
                            high = mid
                    fraction = high
            else:
                fraction = (plane-start[2])/dz
                if not 0. <= fraction <= 1.:
                    continue
                if _circle_passes(start[0]+fraction*dx, start[1]+fraction*dy, row):
                    continue
        if fraction >= 0. and fraction < best:
            best, reason_index = fraction, index
    return best, reason_index
