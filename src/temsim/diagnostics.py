"""Shared mechanical, magnetic-field and ray-stop diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.lens_focal_length import focal_length_mm
from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
from temsim.physics.core import electron, fields
from temsim.physics.first_order import (
    DetectorFrameCalibration,
    LinearMapProperties,
    TransverseTransfer,
    detector_frame_from_component,
    linear_map_properties,
    trace_transverse_transfers,
)


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
    pole_nose_axial_length_mm: float
    pole_cone_angle_to_axis_deg: float
    pole_face_land_axial_thickness_mm: float
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
    polarity: int
    field_polarity_status: str
    field_polarity_source: str
    center_z_mm: float
    z_mm: np.ndarray
    field_t: np.ndarray
    peak_t: float
    support_mm: tuple[float, float]
    focal_length_mm: float
    signed_field_integral_t_m: float
    larmor_rotation_deg: float
    cumulative_column_rotation_deg: float
    spherical_aberration_mm: float | None


@dataclass(frozen=True, slots=True)
class ImagePlaneRotationRecord:
    key: str
    name: str
    z_mm: float
    magnification: float
    image_rotation_from_sample_deg: float
    larmor_rotation_from_sample_deg: float
    conjugacy_error_m: float
    anisotropy_ratio: float


@dataclass(frozen=True, slots=True)
class OpticalTransferRecord:
    """Signed sample-to-plane Jacobian blocks and their coordinate metadata."""

    key: str
    name: str
    z_mm: float
    plane_role: str
    inserted: bool | None
    transfer: TransverseTransfer
    image_properties: LinearMapProperties
    diffraction_properties: LinearMapProperties
    detector_frame: DetectorFrameCalibration


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
        # Curvilinear branch internals have their own Energy Filter view.  A
        # zero-thickness marker at the main-column interface would duplicate
        # and visually flatten their TOML path geometry.
        if bool(part.data.get("branch_path_only", False)):
            continue
        component = layout_by_key.get(part.key)
        shape = getattr(component, "mechanical_shape", None)
        profile = part.data.get(
            "mechanical_profile",
            getattr(shape, "profile", "axial_envelope"),
        )
        recording_surface = profile in {
            "retractable_detector_plane",
            "camera_sensor_plane",
        }
        outer = getattr(shape, "outer_diameter_mm", None)
        if outer is None:
            outer = part.data.get(
                "mechanical_outer_diameter_mm",
                part.data.get(
                    "outer_diameter_mm",
                    part.data.get("outer_width_mm", 1.0),
                ),
            )
        if recording_surface:
            outer = part.data.get(
                "outer_width_mm",
                getattr(shape, "active_diameter_mm", outer),
            )
        active = getattr(shape, "active_diameter_mm", None)
        effective_radius = getattr(
            component, "effective_aperture_radius_mm", None
        )
        if recording_surface:
            bore = float(part.data.get("inner_diameter_mm", 0.0))
        elif active is not None and float(active) > 0.0:
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
            pole_nose_axial_length_mm=max(float(part.data.get(
                "pole_nose_axial_length_mm", 0.0,
            )), 0.0),
            pole_cone_angle_to_axis_deg=max(float(part.data.get(
                "pole_cone_angle_to_axis_deg", 0.0,
            )), 0.0),
            pole_face_land_axial_thickness_mm=max(float(part.data.get(
                "pole_face_land_axial_thickness_mm", 0.0,
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
        cs_value = spherical_aberration_mm(lens, state.beam_voltage_kv)
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
            polarity=int(getattr(lens, "polarity", 1)),
            field_polarity_status=str(getattr(
                lens, "field_polarity_status", "untracked"
            )),
            field_polarity_source=str(getattr(
                lens,
                "field_polarity_source",
                "No selected-manifest provenance is attached.",
            )),
            center_z_mm=float(getattr(lens, "z_mm", 0.0)),
            z_mm=z_mm,
            field_t=field_t,
            peak_t=float(np.max(np.abs(field_t))) if field_t.size else 0.0,
            support_mm=(float(support[0]), float(support[1])),
            focal_length_mm=focal,
            signed_field_integral_t_m=integral,
            larmor_rotation_deg=float(np.degrees(rotation_rad)),
            cumulative_column_rotation_deg=0.0,
            spherical_aberration_mm=(
                None if cs_value is None else float(cs_value)
            ),
        ))
    cumulative_deg = 0.0
    for index in sorted(
        range(len(records)), key=lambda item: records[item].center_z_mm
    ):
        cumulative_deg += records[index].larmor_rotation_deg
        records[index] = replace(
            records[index],
            cumulative_column_rotation_deg=cumulative_deg,
        )
    return total, tuple(records)


def _sample_to_plane_larmor_rotation_deg(state, plane_z_mm: float) -> float:
    sample_z_mm = float(state.sample.z_mm)
    plane_z_mm = float(plane_z_mm)
    if plane_z_mm <= sample_z_mm:
        return 0.0
    from temsim.optics.equivalent_image_lenses import (
        equivalent_image_events,
        equivalent_image_lenses_enabled,
    )
    if equivalent_image_lenses_enabled(state):
        return float(np.degrees(sum(
            event.rotation_rad
            for event in equivalent_image_events(
                state, sample_z_mm, plane_z_mm
            )
        )))
    step_mm = max(min(float(getattr(state, "step_mm", 0.1)), 0.25), 0.01)
    count = max(2, int(math.ceil((plane_z_mm - sample_z_mm) / step_mm)) + 1)
    z_mm = np.linspace(sample_z_mm, plane_z_mm, count)
    magnetic_t = fields(z_mm, state)[0]
    charge_c, momentum, _ = electron(state)
    integral_t_m = float(np.trapezoid(magnetic_t, z_mm * 1.0e-3))
    return float(np.degrees(-charge_c * integral_t_m / (2.0 * momentum)))


def _sample_to_plane_image_maps(state, plane_z_values_mm):
    """Trace a reference plus four transverse bases at requested planes."""

    sample_z_mm = float(state.sample.z_mm)
    transfers = trace_transverse_transfers(
        state,
        sample_z_mm,
        (
            float(value) for value in plane_z_values_mm
            if float(value) > sample_z_mm
        ),
    )
    return {
        z_mm: (transfer.j_img, transfer.j_diff_m_per_rad)
        for z_mm, transfer in transfers.items()
    }


def optical_transfer_records(state) -> tuple[OpticalTransferRecord, ...]:
    """Return full signed J_img/J_diff data at named downstream planes.

    Objective reference planes use the simulator's laboratory X-Y frame.
    Physical recording planes additionally carry their detector/display-axis
    calibration.  The curved Energy Filter branch is not folded into these
    straight-column matrices; its entrance is exposed as a chain boundary.
    """

    sample_z_mm = float(state.sample.z_mm)
    candidates: list[
        tuple[str, str, float, str, bool | None, object]
    ] = []
    objective_planes = (
        (
            "objective_back_focal_plane",
            "Objective back focal plane",
            state.objective_back_focal_plane_z_mm,
            "diffraction_reference",
        ),
        (
            "objective_image_plane",
            "Objective image plane",
            state.objective_image_plane_z_mm,
            "image_reference",
        ),
    )
    for key, name, value, role in objective_planes:
        if value is not None and math.isfinite(float(value)):
            candidates.append(
                (key, name, float(value), role, None, None)
            )
    if bool(getattr(state, "image_corrector_installed", False)):
        sad = state.image_corrector_system.sad_plane
        candidates.append((
            str(sad.key),
            str(sad.name),
            float(sad.z_mm),
            "image_reference",
            None,
            None,
        ))
    candidates.extend(
        (
            str(plane.key),
            str(plane.name),
            float(plane.z_mm),
            "recording_plane",
            bool(getattr(plane, "inserted", True)),
            plane,
        )
        for plane in getattr(state, "recording_planes", ())
    )
    if bool(getattr(state, "energy_filter_installed", False)):
        entrance = getattr(state, "energy_filter_entrance_aperture", None)
        if entrance is not None:
            candidates.append((
                str(entrance.key),
                "Energy Filter entrance",
                float(entrance.z_mm),
                "energy_filter_chain_boundary",
                bool(getattr(entrance, "enabled", True)),
                None,
            ))

    prepared = []
    used_keys = set()
    for candidate in candidates:
        key, _name, z_mm, _role, _inserted, _component = candidate
        if z_mm <= sample_z_mm or key in used_keys:
            continue
        prepared.append(candidate)
        used_keys.add(key)
    transfers = trace_transverse_transfers(
        state,
        sample_z_mm,
        (candidate[2] for candidate in prepared),
    )
    records = []
    for key, name, z_mm, role, inserted, component in prepared:
        transfer = transfers[z_mm]
        records.append(OpticalTransferRecord(
            key=key,
            name=name,
            z_mm=z_mm,
            plane_role=role,
            inserted=inserted,
            transfer=transfer,
            image_properties=linear_map_properties(transfer.j_img),
            diffraction_properties=linear_map_properties(
                transfer.j_diff_m_per_rad
            ),
            detector_frame=detector_frame_from_component(component),
        ))
    return tuple(sorted(records, key=lambda record: (record.z_mm, record.key)))


def image_plane_rotation_records(state) -> tuple[ImagePlaneRotationRecord, ...]:
    """Return paraxial image orientation at named planes below the sample.

    Rotation is the orthogonal factor of the full 2-D sample-to-plane spatial
    map.  This remains meaningful in the presence of quadrupoles, where the X
    and Y magnifications need not be identical.  The angular-to-spatial norm is
    retained so a plane that is not actually conjugate to the sample is not
    presented as exact.
    """

    candidates = []
    objective_image_z_mm = state.objective_image_plane_z_mm
    if objective_image_z_mm is not None and math.isfinite(
        float(objective_image_z_mm)
    ):
        candidates.append((
            "objective_image_plane",
            "Objective image plane",
            float(objective_image_z_mm),
        ))
    if bool(getattr(state, "image_corrector_installed", False)):
        sad = state.image_corrector_system.sad_plane
        candidates.append((str(sad.key), str(sad.name), float(sad.z_mm)))
    selected_area = state.selected_area_aperture
    if getattr(selected_area, "conjugate_to", None) == "objective_image_plane":
        candidates.append(
            (str(selected_area.key), str(selected_area.name), float(selected_area.z_mm))
        )
    if str(getattr(state, "projector_mode", "")) == "image":
        candidates.extend(
            (str(plane.key), str(plane.name), float(plane.z_mm))
            for plane in getattr(state, "recording_planes", ())
            if bool(getattr(plane, "inserted", True))
        )

    sample_z_mm = float(state.sample.z_mm)
    prepared_candidates = []
    used_z_mm = []
    for key, name, plane_z_mm in candidates:
        if plane_z_mm <= sample_z_mm:
            continue
        if any(abs(plane_z_mm - used) <= 1.0e-9 for used in used_z_mm):
            continue
        prepared_candidates.append((key, name, plane_z_mm))
        used_z_mm.append(plane_z_mm)

    plane_maps = _sample_to_plane_image_maps(
        state, (item[2] for item in prepared_candidates)
    )
    records = []
    for key, name, plane_z_mm in prepared_candidates:
        spatial, angular = plane_maps[plane_z_mm]
        properties = linear_map_properties(spatial)
        rotation_deg = (
            float(properties.orientation_deg)
            if properties.orientation_deg is not None
            else math.nan
        )
        records.append(ImagePlaneRotationRecord(
            key=key,
            name=name,
            z_mm=plane_z_mm,
            magnification=properties.isotropic_scale,
            image_rotation_from_sample_deg=rotation_deg,
            larmor_rotation_from_sample_deg=(
                _sample_to_plane_larmor_rotation_deg(state, plane_z_mm)
            ),
            conjugacy_error_m=float(np.linalg.norm(angular, ord=2)),
            anisotropy_ratio=properties.anisotropy_ratio,
        ))
    return tuple(records)


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
