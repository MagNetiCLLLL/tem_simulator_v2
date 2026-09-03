"""Specimen-local transport in the objective-lens axial magnetic field.

The global column solver already integrates the continuous round-lens field on
both sides of the specimen plane.  The finite specimen Monte Carlo uses much
shorter (nm--um) material flights, so it uses the local-uniform ``Bz`` limit of
the same solver field.  This preserves the incident kinetic energy and applies
the exact helical drift for each straight-flight arc before a collision.

This module intentionally does *not* infer a field from the displayed pole
piece geometry or material.  Those data are mechanical reconstruction inputs;
the active optical field remains the TOML-configured axial field provider until
a measured or FEM field map is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.core import fields
from temsim.physics.relativistic_lorentz import (
    ELECTRON,
    momentum_from_kinetic_energy_ev,
)


@dataclass(frozen=True, slots=True)
class SampleAxialFieldDiagnostic:
    """Auditable local-field values used by finite specimen transport."""

    sample_z_mm: float
    total_field_t: float
    objective_field_t: float
    face_fields_t: tuple[float, float]
    active_lens_contributions_t: tuple[tuple[str, float], ...]
    transport_model: str = "local_uniform_axial_Bz_exact_helical_drift"
    field_source: str = "same parameterised axial providers as global column solver"
    geometry_material_coupled: bool = False

    @property
    def face_variation_t(self) -> float:
        return abs(float(self.face_fields_t[1]) - float(self.face_fields_t[0]))


def sample_axial_field_diagnostic(state) -> SampleAxialFieldDiagnostic:
    """Evaluate total and per-lens ``Bz`` at the finite specimen."""

    sample = state.sample
    sample_z_mm = float(sample.z_mm)
    thickness_nm = max(float(getattr(sample, "thickness_nm", 0.0)), 0.0)
    half_thickness_mm = 0.5 * thickness_nm * 1.0e-6
    axial_points = np.asarray(
        (
            sample_z_mm - half_thickness_mm,
            sample_z_mm,
            sample_z_mm + half_thickness_mm,
        ),
        dtype=float,
    )
    total_values = np.asarray(fields(axial_points, state)[0], dtype=float)

    objective = getattr(state, "objective_lens", None)
    if objective is not None and hasattr(objective, "magnetic_field_t"):
        objective_field_t = float(
            np.asarray(objective.magnetic_field_t((sample_z_mm,)), dtype=float)[0]
        )
    else:
        objective_field_t = 0.0

    contributions: list[tuple[str, float]] = []
    for lens in getattr(state, "lenses", ()):
        if not bool(getattr(lens, "enabled", True)):
            continue
        provider = (
            state.condenser_system[lens.key]
            if lens.key in CONDENSER_LENS_KEYS
            else lens
        )
        evaluator = getattr(provider, "magnetic_field_t", None)
        if not callable(evaluator):
            continue
        value = float(np.asarray(evaluator((sample_z_mm,)), dtype=float)[0])
        if abs(value) > 1.0e-15:
            contributions.append((str(getattr(lens, "key", lens)), value))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    return SampleAxialFieldDiagnostic(
        sample_z_mm=sample_z_mm,
        total_field_t=float(total_values[1]),
        objective_field_t=objective_field_t,
        face_fields_t=(float(total_values[0]), float(total_values[2])),
        active_lens_contributions_t=tuple(contributions),
    )


def axial_rotation_rate_rad_per_nm(
    magnetic_field_t: float,
    kinetic_energy_ev: float,
) -> float:
    """Return signed electron cyclotron rotation per path length in rad/nm."""

    field_t = float(magnetic_field_t)
    energy_ev = float(kinetic_energy_ev)
    if not math.isfinite(field_t):
        raise ValueError("Axial magnetic field must be finite")
    if not math.isfinite(energy_ev) or energy_ev <= 0.0:
        raise ValueError("Electron kinetic energy must be finite and positive")
    momentum = momentum_from_kinetic_energy_ev(
        energy_ev,
        np.asarray((0.0, 0.0, 1.0), dtype=float),
    )
    momentum_magnitude = float(np.linalg.norm(momentum))
    return (
        float(ELECTRON.charge_c)
        * field_t
        / momentum_magnitude
        * 1.0e-9
    )


def advance_in_uniform_axial_field(
    position_nm,
    direction,
    path_length_nm: float,
    *,
    rotation_rate_rad_per_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one electron by arc length in a locally uniform axial field.

    ``direction`` is the unit mechanical-momentum direction.  The analytic
    update is reversible, conserves its norm and leaves the longitudinal
    component unchanged.  A zero field is exactly the ordinary straight drift.
    """

    position = np.asarray(position_nm, dtype=float)
    unit = np.asarray(direction, dtype=float)
    length = float(path_length_nm)
    rate = float(rotation_rate_rad_per_nm)
    if (
        position.shape != (3,)
        or unit.shape != (3,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(unit))
        or not math.isfinite(length)
        or not math.isfinite(rate)
    ):
        raise ValueError("Axial-field transport requires finite 3-vectors")
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("Electron direction must be non-zero")
    unit = unit / norm
    angle = rate * length
    if abs(angle) < 1.0e-10:
        # Series forms retain the first magnetic correction instead of
        # numerically cancelling 1-cos(angle).
        sin_over_rate = length * (1.0 - angle * angle / 6.0)
        one_minus_cos_over_rate = length * (
            0.5 * angle - angle**3 / 24.0
        )
    else:
        sin_over_rate = math.sin(angle) / rate
        one_minus_cos_over_rate = (1.0 - math.cos(angle)) / rate

    ux, uy, uz = (float(value) for value in unit)
    endpoint = position + np.asarray(
        (
            ux * sin_over_rate + uy * one_minus_cos_over_rate,
            uy * sin_over_rate - ux * one_minus_cos_over_rate,
            uz * length,
        ),
        dtype=float,
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    final_direction = np.asarray(
        (
            ux * cosine + uy * sine,
            uy * cosine - ux * sine,
            uz,
        ),
        dtype=float,
    )
    final_direction /= np.linalg.norm(final_direction)
    return endpoint, final_direction


def axial_field_polyline(
    position_nm,
    direction,
    path_length_nm: float,
    *,
    rotation_rate_rad_per_nm: float,
    point_count: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return display samples and the final direction for one helical drift."""

    count = max(int(point_count), 2)
    lengths = np.linspace(0.0, float(path_length_nm), count)
    points = []
    final_direction = np.asarray(direction, dtype=float)
    for length in lengths:
        point, final_direction = advance_in_uniform_axial_field(
            position_nm,
            direction,
            float(length),
            rotation_rate_rad_per_nm=rotation_rate_rad_per_nm,
        )
        points.append(point)
    return np.asarray(points, dtype=float), final_direction
