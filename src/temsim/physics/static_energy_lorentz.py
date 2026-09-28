"""Implicit discrete-gradient Lorentz step for a static electric potential.

SI units, electron charge q=-e. Let vbar=(p1+p0)/(m*(gamma1+gamma0)).
Then K1-K0=vbar.dot(p1-p0) exactly. With x1-x0=dt*vbar and
Ebar.dot(x1-x0)=-(phi1-phi0), the update
p1-p0=q*dt*(Ebar+vbar cross Bmid) preserves K-e*phi to solve tolerance.
No momentum rescaling, energy injection or source-parameter adjustment occurs.
The symmetric Gonzalez discrete gradient is second-order for smooth fields;
piecewise grid fields require independent step/grid convergence checks.
"""
from __future__ import annotations

import numpy as np
from temsim.physics.relativistic_lorentz import (
    SPEED_OF_LIGHT_M_PER_S as c,
    ELEMENTARY_CHARGE_C as e,
    ELECTRON_MASS_KG as m_e,
)

from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace
from .discrete_gradient import (
    discrete_gradient_update, ITERATION_TOLERANCE, MAXIMUM_ITERATIONS,
    NORMALIZED_MOMENTUM_FLOOR,
)


def static_energy_step(phase, dt_s, magnetic, electric, *,
                       tolerance=ITERATION_TOLERANCE, maximum_iterations=MAXIMUM_ITERATIONS):
    dt = float(dt_s)
    if not np.isfinite(dt) or dt == 0 or not 0 < tolerance < 1e-3:
        raise ValueError("Invalid discrete-gradient step or tolerance")
    x0 = np.asarray(phase.position_m, float)
    p0 = np.asarray(phase.momentum_kg_m_per_s, float)
    if x0.shape != p0.shape or x0.shape[-1:] != (3,) or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(p0)):
        raise ValueError("Invalid particle state")
    from .grounded_particle_step import try_step
    compiled = try_step(phase, dt, magnetic, electric, tolerance, maximum_iterations)
    if compiled is not None:
        return compiled
    phi = getattr(electric, "potential_rise_v_at_global_positions", electric.potential_v_at_global_positions)
    phi0 = phi(x0)
    u0 = p0/(m_e*c)
    gamma0 = np.sqrt(1+np.sum(u0*u0, axis=-1))
    # The identical electric predictor and normalized Cayley iteration are
    # used by the diagnostic scalar and compiled batch implementations.
    u1 = u0-e*dt/(m_e*c)*electric.field_at_global_positions_v_per_m(x0)
    for _ in range(maximum_iterations):
        gamma1 = np.sqrt(1+np.sum(u1*u1, axis=-1))
        vbar = c*(u1+u0)/(gamma1+gamma0)[..., None]
        dx = dt*vbar
        x1 = x0+dx
        midpoint = .5*(x0+x1)
        electric_mid = np.asarray(electric.field_at_global_positions_v_per_m(midpoint), float)
        magnetic_mid = np.asarray(magnetic.field_at_global_positions_t(midpoint), float)
        updated = np.stack(discrete_gradient_update(
            u0[..., 0], u0[..., 1], u0[..., 2], gamma1+gamma0,
            dx[..., 0], dx[..., 1], dx[..., 2],
            electric_mid[..., 0], electric_mid[..., 1], electric_mid[..., 2],
            magnetic_mid[..., 0], magnetic_mid[..., 1], magnetic_mid[..., 2],
            phi0, phi(x1), dt), axis=-1)
        if not np.all(np.isfinite(updated)):
            raise ValueError("Non-finite discrete-gradient iteration")
        scale = np.maximum(np.maximum(np.linalg.norm(u0, axis=-1), np.linalg.norm(updated, axis=-1)), NORMALIZED_MOMENTUM_FLOOR)
        error = np.max(np.linalg.norm(updated-u1, axis=-1)/scale, initial=0)
        if error <= tolerance:
            # Return the same x/p pair used in the last equation residual.
            return RelativisticPhaseSpace(x1, updated*(m_e*c), phase.time_s+dt)
        u1 = updated
    raise ValueError("Discrete-gradient Lorentz iteration did not converge")
