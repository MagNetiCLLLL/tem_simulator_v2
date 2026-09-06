"""Axial field diagnostics and an analytical uniform-field test reference.

Production finite-specimen transport uses vector_field_transport and the same
registered fields as the column. The exact axial helix below is retained as an
independent analytical reference, not a second production transport model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.core import (
    _LegacyAxialFieldProvider,
    _support_mask,
    fields,
)
from temsim.physics.lens_field_provider import (
    MappedLensFieldProvider,
    runtime_axial_magnetic_field_t,
)
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
    transport_model: str = "shared_vector_field_relativistic_boris"
    field_source: str = "same parameterised axial providers as global column solver"
    geometry_material_coupled: bool = False
    objective_field_model_status: str = "no_active_objective_field_provider"
    objective_field_source: str = ""
    active_lens_field_sources: tuple[tuple[str, str, str], ...] = ()

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
    objective_key = str(getattr(objective, "key", ""))
    objective_field_t = 0.0
    objective_status = "no_active_objective_field_provider"
    objective_source = ""
    objective_geometry_coupled = False
    contributions: list[tuple[str, float]] = []
    sources: list[tuple[str, str, str]] = []
    for lens in getattr(state, "lenses", ()):
        if not bool(getattr(lens, "enabled", True)):
            continue
        native_provider = (
            state.condenser_system[lens.key]
            if lens.key in CONDENSER_LENS_KEYS
            else lens
        )
        if not callable(getattr(native_provider, "magnetic_field_t", None)):
            native_provider = _LegacyAxialFieldProvider(lens)
        values, provider = runtime_axial_magnetic_field_t(
            state,
            lens.key,
            native_provider,
            np.asarray((sample_z_mm,), dtype=float),
        )
        value = float(np.where(
            _support_mask(np.asarray((sample_z_mm,)), provider),
            values,
            0.0,
        )[0])
        status = str(
            getattr(provider, "model_status", "native_field_provider")
        )
        if isinstance(provider, MappedLensFieldProvider):
            provenance = provider.field_map.provenance
            source = (
                f"{provenance.kind}:{provenance.source_path}; "
                f"sha256={provenance.source_sha256}"
            )
        else:
            reason = str(getattr(provider, "fallback_reason", ""))
            source = (
                f"{status}; {reason}" if reason else status
            )
        sources.append((str(lens.key), status, source))
        if abs(value) > 1.0e-15:
            contributions.append((str(getattr(lens, "key", lens)), value))
        if str(lens.key) == objective_key:
            objective_field_t = value
            objective_status = status
            objective_source = source
            objective_geometry_coupled = isinstance(
                provider, MappedLensFieldProvider
            )
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    any_imported = any(
        status == "measured_or_fem_geometry_bound"
        for _key, status, _source in sources
    )

    return SampleAxialFieldDiagnostic(
        sample_z_mm=sample_z_mm,
        total_field_t=float(total_values[1]),
        objective_field_t=objective_field_t,
        face_fields_t=(float(total_values[0]), float(total_values[2])),
        active_lens_contributions_t=tuple(contributions),
        field_source=(
            "same runtime measured/FEM and fallback providers as global "
            "column solver"
            if any_imported
            else "same provisional TOML analytic providers as global column "
            "solver; not a pole-shape magnetostatic solve"
        ),
        geometry_material_coupled=objective_geometry_coupled,
        objective_field_model_status=objective_status,
        objective_field_source=objective_source,
        active_lens_field_sources=tuple(sources),
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
