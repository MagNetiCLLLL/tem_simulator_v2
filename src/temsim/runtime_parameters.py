"""Discover editable operating parameters without making them geometry owners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SCALAR_TYPES = (bool, int, float, str)
IDENTITY_FIELDS = frozenset({
    "key", "name", "label", "display_name", "colour", "color", "type_key",
    "corrector", "owner", "kind", "shape_profile", "interaction_kind",
})
INTERNAL_FIELDS = frozenset({
    "active_backend",
    "active_installation",
    "accelerator_restore_profile",
    "calibrated_dispersion_um_per_ev",
    "centre_m",
    "column_mode",
    "corrector_mode",
    "energy_filter_installed",
    "energy_filter_mode",
    "image_corrector_installed",
    "installation_model_version",
    "installed",
    "gap_m",
    "m12_frames_placed",
    "monochromator_installed",
    "probe_corrector_installed",
    "schema_version",
    "zero_loss_offset_m",
})
TOML_OWNED_FIELDS = frozenset({
    "a_mm",
    "b0_t",
    "blade_thickness_m",
    "clear_height_m",
    "distance_from_sector_exit_m",
    "detector_axis_rotation_deg",
    "detector_flip_x",
    "detector_flip_y",
    "detector_orientation_source",
    "detector_orientation_status",
    "detector_orientation_uncertainty_deg",
    "point_spread_model",
    "point_spread_sigma_x_mm",
    "point_spread_sigma_y_mm",
    "point_spread_rotation_deg",
    "point_spread_status",
    "point_spread_source",
    "signal_collection_surface",
    "field_polarity_source",
    "field_polarity_status",
    "housing_length_mm",
    "inner_face_gap_mm",
    "lower_a_mm",
    "lower_b0_t",
    "lower_objective_lens_axial_length_mm",
    "max_percent",
    "maximum_gap_m",
    "maximum_abs_offset_ev",
    "offset_range_status",
    "maximum_kick_mrad",
    "maximum_strength_m2",
    "maximum_strength_m3",
    "electrode_length_mm",
    "electrode_gap_mm",
    "nominal_focal_length_mm",
    "nominal_voltage_kv",
    "sample_axial_offset_mm",
    "upper_a_mm",
    "upper_b0_t",
    "upper_objective_lens_axial_length_mm",
    "virtual_lens_offset_below_lower_surface_mm",
    "spectral_clear_height_mm",
    "strip_count",
    "pixels_per_strip",
    "strip_height_pixels",
    "alignment_pixels_x",
    "alignment_pixels_y",
    "pixel_size_um",
    "maximum_spectra_per_s",
    "provisional_strip_center_pitch_mm",
    "strip_center_pitch_status",
    "external_envelope_status",
})
GEOMETRY_MARKERS = (
    "z_mm", "mechanical_", "optical_reference", "field_center",
    "pole_piece", "assembly_length", "assembly_outer", "anchor_key",
    "downstream_of_anchor", "upstream_gap", "layout_", "maximum_radius_mm",
    "plate_thickness_mm", "outer_width_mm",
    "inner_diameter_mm", "bore_radius", "bore_diameter", "pole_gap_mm",
)


@dataclass(frozen=True, slots=True)
class RuntimeTarget:
    key: str
    label: str
    obj: object


@dataclass(frozen=True, slots=True)
class RuntimeParameter:
    name: str
    value: object


def _label(obj: object, fallback: str) -> str:
    for attribute in ("name", "label", "display_name"):
        value = getattr(obj, attribute, None)
        if isinstance(value, str) and value:
            return value
    return fallback.replace("_", " ").title()


def runtime_targets(state) -> dict[str, RuntimeTarget]:
    targets: dict[str, RuntimeTarget] = {}

    def add(key: str, obj: object) -> None:
        targets.setdefault(key, RuntimeTarget(key, _label(obj, key), obj))

    add("simulation", state)
    add("electron_gun", state.electron_gun)

    def add_children(parent_key: str, parent: object) -> None:
        for attribute, obj in vars(parent).items():
            if attribute.startswith("_") or isinstance(
                obj, (type(None), bool, int, float, str, bytes, tuple, list, dict)
            ):
                continue
            child_key = str(
                getattr(obj, "key", None) or f"{parent_key}.{attribute}"
            )
            add(child_key, obj)

    add_children("electron_gun", state.electron_gun)
    monochromator = getattr(state.electron_gun, "monochromator", None)
    if monochromator is not None:
        add("electron_gun.monochromator", monochromator)
        add_children("electron_gun.monochromator", monochromator)
    for collection in (
        state.lenses,
        state.apertures,
        state.stigmators,
        state.deflectors,
        getattr(state, "corrector_elements", ()),
        getattr(state, "recording_planes", ()),
    ):
        for obj in collection:
            key = getattr(obj, "key", None)
            if key:
                add(str(key), obj)
    add("sample", state.sample)
    camera = getattr(state, "camera", None)
    if camera is not None:
        add("camera", camera)
    energy_filter = getattr(state, "energy_filter", None)
    if (
        energy_filter is not None
        and bool(getattr(state, "energy_filter_installed", False))
        and getattr(state, "energy_filter_mode", "no_energy_filter")
        == "energy_filter"
    ):
        add("energy_filter", energy_filter)
        add_children("energy_filter", energy_filter)
        for element in getattr(energy_filter, "multipoles", ()) or ():
            key = getattr(element, "key", None)
            if key:
                add(str(key), element)
    return targets


def is_geometry_owned(name: str) -> bool:
    return any(marker in name for marker in GEOMETRY_MARKERS)


def editable_parameters(target: RuntimeTarget) -> tuple[RuntimeParameter, ...]:
    result = []
    for name, value in vars(target.obj).items():
        if (
            name.startswith("_")
            or name in IDENTITY_FIELDS
            or name in INTERNAL_FIELDS
            or name in TOML_OWNED_FIELDS
            or is_geometry_owned(name)
        ):
            continue
        if value is None or isinstance(value, SCALAR_TYPES):
            result.append(RuntimeParameter(name, value))
    return tuple(sorted(result, key=lambda parameter: parameter.name))


def convert_runtime_value(old_value: Any, text: str) -> object:
    if isinstance(old_value, bool):
        normalised = text.strip().lower()
        if normalised not in {"true", "false"}:
            raise ValueError("Boolean values must be true or false")
        return normalised == "true"
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(text)
    if isinstance(old_value, float):
        return float(text)
    if old_value is None:
        return None if text.strip().lower() == "none" else float(text)
    return text


def validate_runtime_assignment(
    target: RuntimeTarget, name: str, value: object
) -> object:
    """Type-check and domain-check one profile/runtime assignment."""

    old_value = getattr(target.obj, name)
    if isinstance(old_value, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{target.key}.{name} must be a Boolean")
        converted = value
    elif isinstance(old_value, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{target.key}.{name} must be an integer")
        converted = int(value)
    elif isinstance(old_value, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{target.key}.{name} must be numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{target.key}.{name} must be finite")
    elif old_value is None:
        if value is None:
            converted = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{target.key}.{name} must be finite")
        else:
            raise ValueError(f"{target.key}.{name} must be numeric or none")
    elif isinstance(old_value, str):
        if not isinstance(value, str):
            raise ValueError(f"{target.key}.{name} must be text")
        converted = value
    else:
        raise ValueError(f"{target.key}.{name} has an unsupported value type")

    if name in {"step_mm", "history_step_mm", "trace_step_mm", "ray_step_mm"}:
        if float(converted) <= 0.0:
            raise ValueError(f"{target.key}.{name} must be positive")
    if name in {"ray_count", "maximum_trace_rays"} and int(converted) <= 0:
        raise ValueError(f"{target.key}.{name} must be positive")
    if name == "polarity" and int(converted) not in (-1, 1):
        raise ValueError(f"{target.key}.{name} must be +1 or -1")
    if name == "specimen_mode" and str(converted).lower() not in {
        "atomic",
        "virtual",
    }:
        raise ValueError("sample.specimen_mode must be atomic or virtual")
    if name in {"radius_mm", "thickness_nm", "rocking_width_inv_nm"}:
        if float(converted) < 0.0:
            raise ValueError(f"{target.key}.{name} cannot be negative")
    if name in {"size_x_nm", "size_y_nm"} and float(converted) <= 0.0:
        raise ValueError(f"{target.key}.{name} must be positive")
    if name in {
        "centre_x_nm",
        "centre_y_nm",
        "scan_origin_x_nm",
        "scan_origin_y_nm",
    } and not math.isfinite(float(converted)):
        raise ValueError(f"{target.key}.{name} must be finite")
    if (
        name == "wave_grid_pixels"
        and int(converted) != 0
        and int(converted) < 32
    ):
        raise ValueError(f"{target.key}.{name} must be 0 or at least 32")
    if name == "wave_field_of_view_angstrom" and float(converted) < 0.0:
        raise ValueError(f"{target.key}.{name} cannot be negative")
    if (
        name == "wave_slice_thickness_angstrom"
        and float(converted) <= 0.0
    ):
        raise ValueError(f"{target.key}.{name} must be positive")
    if name == "wave_bandwidth_fraction" and not (
        0.0 < float(converted) <= 1.0
    ):
        raise ValueError(f"{target.key}.{name} must be in (0, 1]")
    if name == "wave_frozen_phonon_configurations" and not (
        1 <= int(converted) <= 64
    ):
        raise ValueError(
            f"{target.key}.{name} must be between 1 and 64"
        )
    if (
        name == "wave_frozen_phonon_sigma_angstrom"
        and float(converted) < 0.0
    ):
        raise ValueError(f"{target.key}.{name} cannot be negative")
    if name == "wave_frozen_phonon_seed" and int(converted) < 0:
        raise ValueError(f"{target.key}.{name} cannot be negative")
    if name == "stem_poisson_seed" and int(converted) < 0:
        raise ValueError("sample.stem_poisson_seed cannot be negative")
    if name == "wave_probe_padding_factor" and float(converted) < 0.0:
        raise ValueError("sample.wave_probe_padding_factor cannot be negative")
    if name in {
        "real_plasmon_mean_free_path_nm",
        "real_ionisation_mean_free_path_nm",
        "real_other_inelastic_mean_free_path_nm",
        "real_absorption_mean_free_path_nm",
        "real_plasmon_energy_ev",
        "real_ionisation_energy_ev",
    } and float(converted) < 0.0:
        raise ValueError(f"sample.{name} cannot be negative")
    if (
        name == "real_other_inelastic_energy_ev"
        and float(converted) <= 0.0
    ):
        raise ValueError("sample.real_other_inelastic_energy_ev must be positive")
    if name == "real_tail_atomic_number" and not 1 <= int(converted) <= 118:
        raise ValueError("sample.real_tail_atomic_number must be between 1 and 118")
    if name == "real_tail_areal_density_atoms_nm2" and float(converted) < 0.0:
        raise ValueError("sample.real_tail_areal_density_atoms_nm2 cannot be negative")
    if name in {
        "real_tail_screening_angle_mrad",
        "real_tail_max_angle_mrad",
    } and float(converted) <= 0.0:
        raise ValueError(f"sample.{name} must be positive")
    if name in {
        "virtual_diffraction_angle_mrad",
        "virtual_scattering_angle_mrad",
    } and not 0.0 <= float(converted) <= 200.0:
        raise ValueError(
            f"{target.key}.{name} must be between 0 and 200 mrad"
        )
    if name in {
        "virtual_diffraction_relative_weight",
        "virtual_scattering_relative_weight",
    } and float(converted) < 0.0:
        raise ValueError(f"{target.key}.{name} cannot be negative")
    if name == "virtual_scattering_azimuth_samples" and not (
        4 <= int(converted) <= 128
    ):
        raise ValueError(
            "sample.virtual_scattering_azimuth_samples must be between 4 and 128"
        )
    if name == "percent":
        maximum = min(
            100.0,
            float(getattr(target.obj, "max_percent", 100.0)),
        )
        if float(converted) < 0.0 or float(converted) > maximum:
            raise ValueError(
                f"{target.key}.{name} must be between 0 and {maximum:g}"
            )
    return converted
