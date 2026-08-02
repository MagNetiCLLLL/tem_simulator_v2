"""High-accuracy reference checks for the production FEG Boris trace."""

from __future__ import annotations

import numpy as np

from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    integrate_adaptive_reference,
    momentum_from_kinetic_energy_ev,
)


def trace_reference_ray_to_c1(gun, *, x_m=0.0, y_m=0.0, tx_rad=0.0, ty_rad=0.0):
    direction = np.array([tx_rad, ty_rad, 1.0], dtype=float)
    momentum = momentum_from_kinetic_energy_ev(
        gun.emitter.emission_energy_ev, direction
    )
    initial = RelativisticPhaseSpace(
        np.array([x_m, y_m, 0.0], dtype=float), momentum
    )
    exit_z_m = gun.exit_plane_z_mm * 1e-3

    def stop(position, _momentum, _time):
        return float(position[2] - exit_z_m)

    stop.terminal = True
    stop.direction = 1.0
    duration = gun.exit_plane_z_mm * 1e-3 / 1.0e5
    return integrate_adaptive_reference(
        initial,
        duration,
        gun.magnetic_field,
        electric_field=gun.electric_field,
        relative_tolerance=1.0e-10,
        position_tolerance_m=1.0e-12,
        momentum_tolerance_kg_m_per_s=1.0e-32,
        maximum_step_s=1.0e-11,
        stop_event=stop,
    )
