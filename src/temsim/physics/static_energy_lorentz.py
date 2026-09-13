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
from scipy.constants import c, e, m_e

from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace


def static_energy_step(phase, dt_s, magnetic, electric, *, tolerance=1e-11, maximum_iterations=64):
    dt = float(dt_s)
    if not np.isfinite(dt) or dt == 0 or not 0 < tolerance < 1e-3:
        raise ValueError("Invalid discrete-gradient step or tolerance")
    x0 = np.asarray(phase.position_m, float)
    p0 = np.asarray(phase.momentum_kg_m_per_s, float)
    if x0.shape != p0.shape or x0.shape[-1:] != (3,) or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(p0)):
        raise ValueError("Invalid particle state")
    phi = getattr(electric, "potential_rise_v_at_global_positions", electric.potential_v_at_global_positions)
    phi0 = phi(x0)
    gamma0 = np.sqrt(1+np.sum((p0/(m_e*c))**2, axis=-1))
    p1 = p0.copy()
    for _ in range(maximum_iterations):
        gamma1 = np.sqrt(1+np.sum((p1/(m_e*c))**2, axis=-1))
        vbar = (p1+p0)/(m_e*(gamma1+gamma0)[..., None])
        dx = dt*vbar
        x1 = x0+dx
        midpoint = .5*(x0+x1)
        electric_mid = np.asarray(electric.field_at_global_positions_v_per_m(midpoint), float)
        magnetic_mid = np.asarray(magnetic.field_at_global_positions_t(midpoint), float)
        potential_difference = phi(x1)-phi0
        length2 = np.sum(dx*dx, axis=-1)
        defect = potential_difference+np.sum(electric_mid*dx, axis=-1)
        correction = np.divide(defect, length2, out=np.zeros_like(defect), where=length2>0)
        electric_discrete = electric_mid-correction[..., None]*dx
        updated = p0-e*dt*(electric_discrete+np.cross(vbar, magnetic_mid))
        if not np.all(np.isfinite(updated)):
            raise ValueError("Non-finite discrete-gradient iteration")
        scale = np.maximum(np.maximum(np.linalg.norm(p0, axis=-1), np.linalg.norm(updated, axis=-1)), 1e-30)
        error = np.max(np.linalg.norm(updated-p1, axis=-1)/scale, initial=0)
        if error <= tolerance:
            # Return the same x/p pair used in the last equation residual.
            return RelativisticPhaseSpace(x1, updated, phase.time_s+dt)
        p1 = updated
    raise ValueError("Discrete-gradient Lorentz iteration did not converge")
