"""Explicit high-accuracy validation of Energy Filter Boris tracing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter_raytrace import (
    trace_energy_filter_batch,
)
from temsim.optics.energy_filter_sector import (
    magnetic_field_from_energy_filter,
)
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    integrate_adaptive_reference,
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


@dataclass(frozen=True)
class EnergyFilterIntegratorValidation:
    boris_step_mm: float
    adaptive_maximum_step_mm: float
    relative_tolerance: float
    boris_dispersive_um: float
    adaptive_dispersive_um: float
    dispersive_error_nm: float
    boris_non_dispersive_um: float
    adaptive_non_dispersive_um: float
    non_dispersive_error_nm: float
    direction_error_urad: float
    adaptive_step_count: int
    adaptive_function_evaluations: int

    def summary(self):
        return (
            f"Boris Δs {self.boris_step_mm:.4g} mm vs adaptive DOP853 "
            f"(max Δs {self.adaptive_maximum_step_mm:.4g} mm, "
            f"rtol {self.relative_tolerance:.1e}) | slit error: "
            f"dispersive {self.dispersive_error_nm:+.4g} nm, "
            f"non-dispersive {self.non_dispersive_error_nm:+.4g} nm | "
            f"direction {self.direction_error_urad:.4g} µrad | "
            f"{self.adaptive_step_count} adaptive steps / "
            f"{self.adaptive_function_evaluations} field evaluations"
        )


def validate_energy_filter_boris_reference(
    state,
    *,
    adaptive_maximum_step_mm=1.0,
    relative_tolerance=1.0e-10,
):
    """Compare the zero-loss Boris ray with an event-aligned DOP853 ray."""

    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    maximum_step_mm = float(adaptive_maximum_step_mm)
    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(maximum_step_mm) or maximum_step_mm <= 0.0:
        raise ValueError("Adaptive maximum spatial step must be positive.")
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
    ):
        raise ValueError("Adaptive relative tolerance must be positive.")

    boris = trace_energy_filter_batch(
        state,
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        apply_slit=False,
        stop_at_slit=True,
    )
    if not bool(boris.reached_slit[0]):
        raise RuntimeError(
            "The current zero-loss Boris reference ray does not reach "
            "the slit plane; adaptive comparison was not run."
        )

    magnetic_field = magnetic_field_from_energy_filter(energy_filter)
    sector = magnetic_field.sector
    exit_origin = sector.exit_point_m
    outgoing_s = sector.exit_frame.rotation_local_to_global[:, 2]
    outgoing_x = sector.exit_frame.rotation_local_to_global[:, 0]
    outgoing_y = sector.exit_frame.rotation_local_to_global[:, 1]
    slit_distance_m = float(
        energy_filter.energy_slit.distance_from_sector_exit_m
    )
    direction = np.array([
        1.0,
        float(energy_filter.alignment_y_mrad) * 1.0e-3,
        float(energy_filter.alignment_x_mrad) * 1.0e-3,
    ])
    momentum = momentum_from_kinetic_energy_ev(
        float(state.beam_voltage_kv) * 1000.0,
        direction,
    )
    initial = RelativisticPhaseSpace(np.zeros(3), momentum)
    speed = float(
        np.linalg.norm(velocity_from_momentum_m_per_s(momentum))
    )
    path_length_m = (
        float(energy_filter.prism_entrance_s_mm) * 1.0e-3
        + sector.arc_length_m
        + slit_distance_m
        + 4.0 * float(energy_filter.prism_fringe_mm) * 1.0e-3
    )
    duration_s = 1.25 * path_length_m / speed
    maximum_step_s = maximum_step_mm * 1.0e-3 / speed

    def slit_event(position_m, _momentum, _time_s):
        return float(
            (np.asarray(position_m) - exit_origin) @ outgoing_s
            - slit_distance_m
        )

    slit_event.terminal = True
    slit_event.direction = 1.0
    adaptive = integrate_adaptive_reference(
        initial,
        duration_s,
        magnetic_field,
        relative_tolerance=relative_tolerance,
        position_tolerance_m=1.0e-12,
        momentum_tolerance_kg_m_per_s=1.0e-34,
        maximum_step_s=maximum_step_s,
        stop_event=slit_event,
    )
    if not adaptive.terminated_by_event:
        raise RuntimeError(
            "Adaptive reference ray did not reach the slit plane."
        )

    adaptive_position = adaptive.position_m[-1]
    adaptive_delta = adaptive_position - exit_origin
    adaptive_x_m = float(adaptive_delta @ outgoing_x)
    adaptive_y_m = float(adaptive_delta @ outgoing_y)
    boris_x_m = float(boris.slit_dispersive_m[0])
    boris_y_m = float(boris.slit_non_dispersive_m[0])

    boris_direction = np.asarray(
        boris.final_momentum_kg_m_per_s[0],
        dtype=float,
    )
    adaptive_direction = np.asarray(
        adaptive.momentum_kg_m_per_s[-1],
        dtype=float,
    )
    boris_direction /= np.linalg.norm(boris_direction)
    adaptive_direction /= np.linalg.norm(adaptive_direction)
    direction_error_rad = math.atan2(
        float(np.linalg.norm(
            np.cross(boris_direction, adaptive_direction)
        )),
        float(np.dot(boris_direction, adaptive_direction)),
    )
    return EnergyFilterIntegratorValidation(
        boris_step_mm=float(energy_filter.ray_step_mm),
        adaptive_maximum_step_mm=maximum_step_mm,
        relative_tolerance=relative_tolerance,
        boris_dispersive_um=boris_x_m * 1.0e6,
        adaptive_dispersive_um=adaptive_x_m * 1.0e6,
        dispersive_error_nm=(boris_x_m - adaptive_x_m) * 1.0e9,
        boris_non_dispersive_um=boris_y_m * 1.0e6,
        adaptive_non_dispersive_um=adaptive_y_m * 1.0e6,
        non_dispersive_error_nm=(boris_y_m - adaptive_y_m) * 1.0e9,
        direction_error_urad=direction_error_rad * 1.0e6,
        adaptive_step_count=len(adaptive.time_s) - 1,
        adaptive_function_evaluations=adaptive.function_evaluations,
    )
