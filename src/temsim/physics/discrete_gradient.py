"""Shared relativistic static-field update for one electron or an array.

All callers use u = p/(m*c), SI position/time and potential in volts.  The
electric discrete gradient enforces work against the sampled potential; the
magnetic Cayley solve does no work.  Field sampling and adaptive scheduling
belong to the caller, so batching never changes the force law.
"""
from __future__ import annotations

import numpy as np
from math import sqrt

try:
    from numba.extending import register_jitable
except ImportError:
    def register_jitable(function=None, **kwargs):
        return function if callable(function) else lambda wrapped: wrapped

from .relativistic_lorentz import (
    ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
)

ITERATION_TOLERANCE = 1e-11
MAXIMUM_ITERATIONS = 64
NORMALIZED_MOMENTUM_FLOOR = 1e-9


@register_jitable(inline="always")
def dot3(first, second):
    """Fixed scalar order for adaptive decisions in native/compiled pushers."""
    return first[0]*second[0]+first[1]*second[1]+first[2]*second[2]


@register_jitable(inline="always")
def norm3(value):
    return sqrt(dot3(value, value))


@register_jitable(inline="always")
def discrete_gradient_update(
    ux, uy, uz, gamma_sum, dx, dy, dz,
    ex, ey, ez, bx, by, bz, phi0, phi1, dt,
):
    """Return the next fixed-point iterate; arguments may be scalars/arrays.

    At an unresolved turning displacement, use the limiting midpoint E.
    Dividing the rounded potential difference by displacement squared there
    would spuriously cancel the force and strand a reversing electron.
    """
    length2 = dx*dx + dy*dy + dz*dz
    work = ex*dx + ey*dy + ez*dz
    delta = phi1-phi0
    resolution = 64.*np.finfo(np.float64).eps*np.maximum(
        np.maximum(np.abs(phi0), np.abs(phi1)), 1.)
    resolved = (length2 > 0.) & (np.abs(delta)+np.abs(work) > resolution)
    divisor = np.where(resolved, length2, 1.)
    correction = (delta+work)/divisor * resolved
    ex, ey, ez = ex-correction*dx, ey-correction*dy, ez-correction*dz

    alpha = -ELEMENTARY_CHARGE_C*dt/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)
    scale = alpha*SPEED_OF_LIGHT_M_PER_S/gamma_sum
    tx, ty, tz = scale*bx, scale*by, scale*bz
    ax, ay, az = 2.*ux+alpha*ex, 2.*uy+alpha*ey, 2.*uz+alpha*ez
    dot = ax*tx + ay*ty + az*tz
    denominator = 1.+tx*tx+ty*ty+tz*tz
    return (
        (ax + ay*tz-az*ty + tx*dot)/denominator-ux,
        (ay + az*tx-ax*tz + ty*dot)/denominator-uy,
        (az + ax*ty-ay*tx + tz*dot)/denominator-uz,
    )
