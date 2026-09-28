"""A proved query envelope for production gun integration, not a field cutoff.

The electric solution and magnetic provider retain their complete domains.
Only trial durations are bounded here so a gun cache can depend on the fields
that its transport can actually query, including rejected integration trials.
"""
from __future__ import annotations

import math

from .relativistic_lorentz import SPEED_OF_LIGHT_M_PER_S

try:
    from numba.extending import register_jitable
except ImportError:  # Numba remains optional.
    def register_jitable(function):
        return function


GUN_QUERY_SCHEMA = "subluminal-gun-trial-envelope-v1"
_ROUNDOFF_MARGIN = 64. * 2.220446049250313e-16


def gun_magnetic_query_upper_m(gun):
    """Keep the existing gun lookahead independently of the full E solve.

    The old outlet extension remains useful as numerical trial space. It is
    neither a grounded boundary nor a truncation of magnetic/electric tails.
    Positive tracing steps provide room when that extension is explicitly zero.
    """
    values = (float(gun.exit_plane_z_mm),
              float(getattr(gun, "_gun_field_exit_extension_mm", 100.)),
              float(gun.trace_step_mm), float(gun.drift_step_mm))
    if (not all(map(math.isfinite, values)) or values[0] <= 0.
            or values[1] < 0. or min(values[2:]) <= 0.):
        raise ValueError("Gun transport requires finite positive steps and a non-negative query extension")
    return (values[0] + max(values[1:])) * 1e-3


@register_jitable
def bounded_gun_time_step(upper_m, maximum_z_m, requested_dt):
    """Keep every trial field query inside its declared axial upper bound.

    For every discrete-gradient iterate, |c*(u0+u1)/(gamma0+gamma1)| < c.
    Relativistic Boris uses subluminal velocities as well. Thus all full/half
    trial endpoints, midpoints and fractional plane-crossing re-integrations
    lie within c*dt of their initial position, including rejected trials.
    A small coordinate-roundoff margin makes the floating-point bound strict.
    Standalone, unbound numerical fixtures keep their existing step schedule.
    """
    if upper_m == math.inf:
        return requested_dt
    if (not math.isfinite(upper_m) or not math.isfinite(maximum_z_m)
            or not math.isfinite(requested_dt) or requested_dt <= 0.):
        raise ValueError("Gun trial envelope requires finite positions and a positive time step")
    margin = _ROUNDOFF_MARGIN * max(1., abs(upper_m), abs(maximum_z_m))
    distance = upper_m - maximum_z_m - margin
    if distance <= 0.:
        raise ValueError("Active gun particle is outside its declared magnetic query envelope")
    return min(requested_dt, distance / SPEED_OF_LIGHT_M_PER_S)
