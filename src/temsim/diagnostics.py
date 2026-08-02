"""Shared mechanical, magnetic-field and ray-stop diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.lens_focal_length import focal_length_mm
from temsim.physics.core import electron, fields


@dataclass(frozen=True, slots=True)
class PhysicalLayoutRecord:
    key: str
    name: str
    kind: str
    profile: str
    branch: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    outer_diameter_mm: float
    bore_diameter_mm: float
    vacuum_inner_diameter_mm: float
    pole_gap_mm: float
    pole_tip_diameter_mm: float
    optical_references_mm: tuple[float, ...]
    excitation_enabled: bool | None


@dataclass(frozen=True, slots=True)
class LensFieldRecord:
    key: str
    name: str
    colour: object
    formula_key: str
    formula_label: str
    formula_expression: str
    formula_colour: str
    enabled: bool
    excitation_percent: float
    z_mm: np.ndarray
    field_t: np.ndarray
    peak_t: float
    support_mm: tuple[float, float]
    focal_length_mm: float
    signed_field_integral_t_m: float
    larmor_rotation_deg: float
    spherical_aberration_mm: float | None


@dataclass(frozen=True, slots=True)
class RayStopRecord:
    bundle: str
    ray_index: int
    key: str
    z_mm: float
    x_mm: float
    y_mm: float
    radial_mm: float


def _optical_references(part) -> tuple[float, ...]:
    local_center = float(part.data["local_center_z_mm"])
    references = []
    for field, value in part.data.items():
        if field == "optical_reference_local_z_mm":
            references.append(part.center_z_mm + float(value) - local_center)
        elif field == "interaction_centers_local_z_mm":
            references.extend(
                part.center_z_mm + float(item) - local_center
                for item in value
            )
        elif field.endswith("_field_reference_local_z_mm"):
            references.append(part.center_z_mm + float(value) - local_center)
        elif field == "virtual_reference_local_z_mm":
            references.append(part.center_z_mm + float(value) - local_center)
    return tuple(sorted(set(float(item) for item in references)))


def physical_layout_records(result) -> tuple[PhysicalLayoutRecord, ...]:
    """Return physical drawing records from one resolved calculation."""

    if result.assembly is None or result.layout is None:
        return ()
    layout_by_key = {component.key: component for component in result.layout}
    records = []
    for part in result.assembly.parts:
        component = layout_by_key.get(part.key)
        shape = getattr(component, "mechanical_shape", None)
        outer = getattr(shape, "outer_diameter_mm", None)
        if outer is None:
            outer = part.data.get(
                "mechanical_outer_diameter_mm",
                part.data.get("outer_diameter_mm", 1.0),
            )
        active = getattr(shape, "active_diameter_mm", None)
        effective_radius = getattr(
            component, "effective_aperture_radius_mm", None
        )
        if active is not None and float(active) > 0.0:
            bore = float(active)
        elif effective_radius is not None and float(effective_radius) > 0.0:
            bore = 2.0 * float(effective_radius)
        else:
            bore = min(float(outer), float(
                part.data.get(
                    "mechanical_inner_diameter_mm",
                    part.data.get(
                        "mechanical_bore_diameter_mm",
                        part.data.get("bore_diameter_mm", 0.0),
                    ),
                )
            ))
        profile = part.data.get(
            "mechanical_profile",
            getattr(shape, "profile", "axial_envelope"),
        )
        kind = getattr(
            component,
            "kind",
            part.data.get("mechanical_part_role", "component"),
        )
        excitation_source = component
        if excitation_source is None and "field_source_key" in part.data:
            excitation_source = layout_by_key.get(part.data["field_source_key"])
        records.append(PhysicalLayoutRecord(
            key=str(part.key),
            name=str(part.name),
            kind=str(kind),
            profile=str(profile),
            branch=str(getattr(
                getattr(component, "branch", None),
                "value",
                part.branch,
            )),
            start_z_mm=float(part.start_z_mm),
            center_z_mm=float(part.center_z_mm),
            end_z_mm=float(part.end_z_mm),
            outer_diameter_mm=max(float(outer), 0.001),
            bore_diameter_mm=max(min(float(bore), float(outer)), 0.0),
            vacuum_inner_diameter_mm=float(
                part.data["vacuum_inner_diameter_mm"]
            ),
            pole_gap_mm=max(float(part.data.get("pole_gap_mm", 0.0)), 0.0),
            pole_tip_diameter_mm=max(float(part.data.get(
                "mechanical_tip_diameter_mm",
                part.data.get("pole_piece_tip_diameter_mm", 0.0),
            )), 0.0),
            optical_references_mm=_optical_references(part),
            excitation_enabled=getattr(
                excitation_source, "excitation_enabled", None
            ),
        ))
    return tuple(records)


def vacuum_bore_plot_points(assembly):
    """Return stepped upper/lower 2-D projections of the vacuum cylinders."""

    segments = tuple(getattr(assembly, "vacuum_bore_segments", ()))
    if not segments:
        return np.empty(0), np.empty(0), np.empty(0)
    z_values = []
    radius_values = []
    for segment in segments:
        radius = 0.5 * float(segment.inner_diameter_mm)
        start = float(segment.start_z_mm)
        end = float(segment.end_z_mm)
        if z_values and not math.isclose(z_values[-1], start, abs_tol=1.0e-12):
            z_values.extend((float("nan"), start))
            radius_values.extend((float("nan"), radius))
        elif z_values and not math.isclose(
            radius_values[-1], radius, abs_tol=1.0e-12
        ):
            z_values.append(start)
            radius_values.append(radius)
        else:
            z_values.append(start)
            radius_values.append(radius)
        z_values.append(end)
        radius_values.append(radius)
    z = np.asarray(z_values, dtype=float)
    upper = np.asarray(radius_values, dtype=float)
    return z, upper, -upper


def _lens_provider(state, lens):
    if lens.key in CONDENSER_LENS_KEYS:
        return state.condenser_system[lens.key]
    return lens


FIELD_FORMULAS = {
    "three_gaussian": (
        "Three-Gaussian axial field",
        "Bz(z) = P B100 sum_i a_i exp[-(z-z_i)^2/(2 sigma_i^2)]",
        "#38bdf8",
    ),
    "peak_normalised_three_gaussian": (
        "Peak-normalised three-Gaussian field",
        "Bz(z) = P B100 G(z) / max|G(z)|",
        "#f59e0b",
    ),
    "dual_pole_gaussian": (
        "Dual-pole Gaussian field",
        "Bz(z) = P [B_upper G_upper(z) + B_lower G_lower(z)]",
        "#e879f9",
    ),
    "solver_provider": (
        "Solver-owned magnetic-field provider",
        "Bz(z) = provider.magnetic_field_t(z)",
        "#4ade80",
    ),
    "no_field_provider": (
        "No axial-field formula",
        "Bz(z) = 0",
        "#64748b",
    ),
}


def _field_formula(lens, provider):
    if all(
        hasattr(provider, attribute)
        for attribute in ("upper_gaussian", "lower_gaussian")
    ):
        key = "dual_pole_gaussian"
    elif bool(getattr(lens, "normalise_profile_peak", False)):
        key = "peak_normalised_three_gaussian"
    elif hasattr(lens, "gaussian") or hasattr(provider, "gaussian"):
        key = "three_gaussian"
    elif hasattr(provider, "magnetic_field_t"):
        key = "solver_provider"
    else:
        key = "no_field_provider"
    label, expression, colour = FIELD_FORMULAS[key]
    return key, label, expression, colour


def lens_field_records(
    state, z_mm: np.ndarray
) -> tuple[np.ndarray, tuple[LensFieldRecord, ...]]:
    """Evaluate the total and per-lens Bz using solver-owned field methods."""

    z_mm = np.asarray(z_mm, dtype=float)
    total = fields(z_mm, state)[0]
    records = []
    charge_c, momentum, _ = electron(state)
    for lens in state.lenses:
        provider = _lens_provider(state, lens)
        formula_key, formula_label, formula_expression, formula_colour = (
            _field_formula(lens, provider)
        )
        if hasattr(provider, "magnetic_field_t"):
            field_t = np.asarray(provider.magnetic_field_t(z_mm), dtype=float)
        else:
            field_t = np.zeros_like(z_mm)
        if hasattr(provider, "field_support_mm"):
            support = tuple(float(value) for value in provider.field_support_mm())
        else:
            nonzero = np.flatnonzero(np.abs(field_t) > 1.0e-12)
            support = (
                (float(z_mm[nonzero[0]]), float(z_mm[nonzero[-1]]))
                if nonzero.size else (float(lens.z_mm), float(lens.z_mm))
            )
        try:
            focal = float(focal_length_mm(lens, state.beam_voltage_kv))
        except Exception:
            focal = math.nan
        integral = (
            float(np.trapezoid(field_t, z_mm * 1.0e-3))
            if field_t.size > 1 else 0.0
        )
        rotation_rad = -charge_c * integral / (2.0 * momentum)
        cs_value = getattr(lens, "cs_mm", None)
        records.append(LensFieldRecord(
            key=str(lens.key),
            name=str(lens.name),
            colour=getattr(lens, "colour", "#38bdf8"),
            formula_key=formula_key,
            formula_label=formula_label,
            formula_expression=formula_expression,
            formula_colour=formula_colour,
            enabled=bool(getattr(lens, "enabled", True)),
            excitation_percent=float(getattr(lens, "percent", 0.0)),
            z_mm=z_mm,
            field_t=field_t,
            peak_t=float(np.max(np.abs(field_t))) if field_t.size else 0.0,
            support_mm=(float(support[0]), float(support[1])),
            focal_length_mm=focal,
            signed_field_integral_t_m=integral,
            larmor_rotation_deg=float(np.degrees(rotation_rad)),
            spherical_aberration_mm=(
                None if cs_value is None else float(cs_value)
            ),
        ))
    return total, tuple(records)


def _interpolate_stop(branch, ray_index: int, stop_z_mm: float):
    z = np.asarray(branch.z, dtype=float)
    x_mm = 1.0e3 * np.asarray(branch.x[:, ray_index], dtype=float)
    y_mm = 1.0e3 * np.asarray(branch.y[:, ray_index], dtype=float)
    return (
        float(np.interp(stop_z_mm, z, x_mm)),
        float(np.interp(stop_z_mm, z, y_mm)),
    )


def ray_stop_records(
    simulation, *, maximum_records: int | None = None
) -> tuple[RayStopRecord, ...]:
    """Return first-intercept coordinates without duplicate branch copies."""

    records = []
    incident_end = float(simulation.incident.z[-1])
    bundles = (("Incident", simulation.incident), *(
        (branch.name, branch) for branch in simulation.branches.values()
    ))
    per_bundle_limit = None
    if maximum_records is not None:
        per_bundle_limit = max(1, int(maximum_records) // max(len(bundles), 1))
    for bundle_name, branch in bundles:
        keys = np.asarray(branch.blocked_key, dtype=object)
        stop_values = np.asarray(branch.blocked_z, dtype=float)
        valid = (keys != "") & np.isfinite(stop_values)
        if branch is not simulation.incident:
            valid &= stop_values > incident_end + 1.0e-9
        indices = np.flatnonzero(valid)
        if per_bundle_limit is not None and indices.size > per_bundle_limit:
            indices = indices[np.unique(np.linspace(
                0, indices.size - 1, per_bundle_limit, dtype=int
            ))]
        for ray_index in indices:
            key = keys[ray_index]
            stop_z = stop_values[ray_index]
            x_mm, y_mm = _interpolate_stop(branch, ray_index, float(stop_z))
            records.append(RayStopRecord(
                bundle=str(bundle_name),
                ray_index=int(ray_index),
                key=str(key),
                z_mm=float(stop_z),
                x_mm=x_mm,
                y_mm=y_mm,
                radial_mm=float(math.hypot(x_mm, y_mm)),
            ))
    return tuple(records)
