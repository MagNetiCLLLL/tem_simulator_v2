"""TOML persistence for operating parameters."""

from __future__ import annotations

import os
import math
from pathlib import Path
import tempfile
import tomllib
from copy import copy, deepcopy
from functools import lru_cache
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

import tomli_w

from temsim.assembly_catalog import AssemblySelection
from temsim.runtime_parameters import (
    RETIRED_SAMPLE_FIELDS,
    SAMPLE_SOURCE_FIELDS,
    editable_parameters,
    runtime_targets,
    validate_runtime_assignment,
)
from temsim.specimen.geometry import (
    normalise_quaternion_wxyz,
    quaternion_from_euler_xyz_deg,
    sample_orientation_quaternion,
    set_sample_orientation,
)
from temsim.specimen.source import migrate_legacy_structure_source


PROFILE_FORMAT_VERSION = 7
_GUN_SOURCE_MODEL_KEY = "__gun_source_model__"
_SAMPLE_MODEL_KEY = "__sample_model__"
_SIMULATION_MODEL_KEY = "__simulation_model__"
_PROFILE_VERSION_KEY = "__profile_format_version__"


@lru_cache(maxsize=None)
def _nullable_fields(component_type: type) -> frozenset[str]:
    """Use declared optional fields, independently of their current values."""

    return frozenset(
        name
        for name, annotation in get_type_hints(component_type).items()
        if get_origin(annotation) in (Union, UnionType)
        and type(None) in get_args(annotation)
    )


def _atomic_write_profile(path: Path, document: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            tomli_w.dump(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_profile(path: str | Path, state, selection: AssemblySelection) -> None:
    from dataclasses import asdict
    from temsim.simulation_modes import capture_mode_settings, mode_key
    devices = {}
    none_values = {}
    for key, target in runtime_targets(state).items():
        values = {}
        for parameter in editable_parameters(target):
            if parameter.value is None:
                if parameter.name not in _nullable_fields(type(target.obj)):
                    raise ValueError(f"{key}.{parameter.name} cannot be none")
                none_values.setdefault(key, []).append(parameter.name)
            else:
                values[parameter.name] = parameter.value
        if key == "sample":
            values.update({name: getattr(state.sample, name) for name in sorted(SAMPLE_SOURCE_FIELDS)})
        if values:
            devices[key] = values
    document = {
        "format_version": PROFILE_FORMAT_VERSION,
        "assembly": {
            "gun": selection.gun,
            "column": selection.column,
            "recording": selection.recording,
            "beam_blanker": selection.beam_blanker,
        },
        "devices": devices,
        "gun_source_model": {
            "representation": getattr(state.electron_gun, "source_representation", "classical_particles"),
            **({"effective": asdict(state.electron_gun.effective_source)}
               if getattr(state.electron_gun, "effective_source", None) is not None else {}),
        },
        # TOML has no null literal. Keep absence (legacy/default behaviour)
        # distinct from explicitly clearing an optional runtime coefficient.
        "none_values": none_values,
        "simulation_model": {
            "mode": mode_key(state),
            "profiles": deepcopy(state.simulation_mode_profiles),
            "settings": capture_mode_settings(state),
        },
        "sample_model": {
            "wave_illumination": deepcopy(state.sample.wave_illumination),
            "orientation_quaternion_wxyz": list(
                normalise_quaternion_wxyz(
                    sample_orientation_quaternion(state.sample)
                )
            ),
            "zone_axis_uvw": list(state.sample.zone_axis_uvw),
            "in_plane_axis_uvw": list(state.sample.in_plane_axis_uvw),
            "frozen_phonon_sigma_by_element_angstrom": dict(
                state.sample.wave_frozen_phonon_sigma_by_element_angstrom
            ),
        },
    }
    _atomic_write_profile(Path(path), document)


def read_profile(path: str | Path) -> tuple[AssemblySelection, dict]:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if not isinstance(document, dict):
        raise ValueError("Operating profile must be a TOML table")
    format_version = int(document.get("format_version", 0))
    if format_version not in {1, 2, 3, 4, 5, 6, PROFILE_FORMAT_VERSION}:
        raise ValueError("Unsupported operating-profile format")
    assembly = document.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("Operating profile is missing the assembly table")
    selection = AssemblySelection(
        gun=str(assembly["gun"]),
        column=str(assembly["column"]),
        recording=str(assembly["recording"]),
        beam_blanker=str(assembly.get("beam_blanker", "None")),
    )
    devices = document.get("devices", {})
    if not isinstance(devices, dict):
        raise ValueError("Operating profile devices must be a table")
    values = dict(devices)
    none_values = document.get("none_values", {})
    if not isinstance(none_values, dict):
        raise ValueError("Operating profile none_values must be a table")
    if none_values and format_version < 4:
        raise ValueError("Operating profile none_values requires format version 4")
    for key, names in none_values.items():
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(f"Operating profile none_values.{key} must be an array of names")
        if len(set(names)) != len(names):
            raise ValueError(f"Operating profile none_values.{key} contains duplicate names")
        attributes = values.get(key, {})
        if not isinstance(attributes, dict):
            raise ValueError(f"Operating profile device {key} must be a table")
        if set(names) & attributes.keys():
            raise ValueError(f"Operating profile device {key} has conflicting none values")
        values[key] = {**attributes, **dict.fromkeys(names)}
    values[_PROFILE_VERSION_KEY] = format_version
    model = document.get("simulation_model", {"mode": "custom"})
    if not isinstance(model, dict):
        raise ValueError("Operating profile simulation_model must be a table")
    values[_SIMULATION_MODEL_KEY] = model
    values[_GUN_SOURCE_MODEL_KEY] = document.get("gun_source_model", {"representation": "classical_particles"})
    if format_version >= 2:
        sample_model = document.get("sample_model", {})
        if not isinstance(sample_model, dict):
            raise ValueError("Operating profile sample_model must be a table")
        values[_SAMPLE_MODEL_KEY] = dict(sample_model)
    return selection, values


def _apply_sample_model(sample, model: dict) -> None:
    from temsim.physics.illumination import validate_illumination_config, require_production_illumination
    if not isinstance(model, dict):
        raise ValueError("Operating profile sample_model must be a table")
    illumination = validate_illumination_config(model.get("wave_illumination", sample.wave_illumination))
    require_production_illumination(illumination)
    quaternion = normalise_quaternion_wxyz(
        model.get(
            "orientation_quaternion_wxyz",
            sample.specimen_orientation_quaternion_wxyz,
        )
    )
    zone = tuple(int(value) for value in model.get("zone_axis_uvw", sample.zone_axis_uvw))
    in_plane = tuple(
        int(value)
        for value in model.get("in_plane_axis_uvw", sample.in_plane_axis_uvw)
    )
    if len(zone) != 3 or len(in_plane) != 3 or zone == (0, 0, 0):
        raise ValueError("Operating profile has invalid sample zone-axis metadata")
    element_sigma = model.get(
        "frozen_phonon_sigma_by_element_angstrom",
        sample.wave_frozen_phonon_sigma_by_element_angstrom,
    )
    if not isinstance(element_sigma, dict):
        raise ValueError("Sample frozen-phonon element RMS values must be a table")
    converted_sigma = {}
    for symbol, value in element_sigma.items():
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0.0:
            raise ValueError(
                f"Frozen-phonon RMS for {symbol} must be finite and positive"
            )
        converted_sigma[str(symbol)] = converted
    set_sample_orientation(sample, quaternion)
    sample.zone_axis_uvw = zone
    sample.in_plane_axis_uvw = in_plane
    sample.virtual_interactions = []
    sample.virtual_regions = []
    sample.wave_frozen_phonon_sigma_by_element_angstrom = converted_sigma
    sample.wave_illumination = illumination


def apply_profile_values(state, values: dict) -> list[str]:
    if not isinstance(values, dict):
        raise ValueError("Operating profile devices must be a table")
    values = dict(values)
    format_version = int(values.pop(_PROFILE_VERSION_KEY, 1))
    sample_model = values.pop(_SAMPLE_MODEL_KEY, None)
    gun_source = values.pop(_GUN_SOURCE_MODEL_KEY, {"representation": "classical_particles"})
    if not isinstance(gun_source, dict) or set(gun_source)-{"representation", "effective"}:
        raise ValueError("Operating profile has invalid electron-gun source fields")
    representation = gun_source.get("representation")
    if representation not in {"classical_particles", "effective_gaussian_schell"}:
        raise ValueError("Operating profile has an unknown electron-gun source model")
    parameters = getattr(state.electron_gun, "effective_source", None)
    if "effective" in gun_source:
        from temsim.optics.electron_gun.effective_source import EffectiveGunSource
        parameters = EffectiveGunSource(**gun_source["effective"])
    if representation == "effective_gaussian_schell" and (
            parameters is None or state.electron_gun.type_key != "cold_feg"):
        raise ValueError("Effective gun source requires its saved configuration and a cold FEG")
    from temsim.simulation_modes import validate_mode, normalise_profiles, MODEL_SETTINGS
    model = values.pop(_SIMULATION_MODEL_KEY, {"mode": "custom"})
    if not isinstance(model, dict):
        raise ValueError("Operating profile simulation_model must be a table")
    selected_mode = validate_mode(model.get("mode", "custom"))
    model_profiles = normalise_profiles(model.get("profiles", {}))
    model_settings = normalise_profiles({selected_mode: model.get("settings", {})})[selected_mode]
    targets = runtime_targets(state)
    skipped = []
    pending = []
    legacy_sample_source = ""
    sample_mode_was_explicit = False
    sample_cif_was_explicit = False
    sample_preset_was_explicit = False
    sample_attributes = values.get("sample", {})
    for key, attributes in values.items():
        target = targets.get(key)
        if target is None:
            if isinstance(attributes, dict) and any(
                value is None for value in attributes.values()
            ):
                raise ValueError(f"Unknown operating-profile device for none values: {key}")
            skipped.append(key)
            continue
        if not isinstance(attributes, dict):
            raise ValueError(f"Operating profile device {key} must be a table")
        allowed = {parameter.name for parameter in editable_parameters(target)}
        if key == "sample":
            allowed.update(SAMPLE_SOURCE_FIELDS)
        for name, value in attributes.items():
            if value is None:
                if name not in allowed or name not in _nullable_fields(type(target.obj)):
                    raise ValueError(f"{key}.{name} cannot be none")
                pending.append((target.obj, name, None))
                continue
            if key == "sample" and name == "atomic_structure_source":
                legacy_sample_source = str(value).strip().lower()
                if legacy_sample_source not in {"preset", "cif"}:
                    raise ValueError(
                        "Legacy sample.atomic_structure_source must be "
                        "preset or cif"
                    )
                continue
            if key == "sample" and name == "specimen_mode":
                sample_mode_was_explicit = True
                if format_version < 5 and str(value).lower() == "virtual":
                    pending.append((target.obj, name, "virtual"))
                    continue
            if key == "sample" and name == "cif_path":
                sample_cif_was_explicit = True
            if key == "sample" and name == "specimen_preset_key":
                sample_preset_was_explicit = True
                if format_version < 5:
                    pending.append((target.obj, name, validate_runtime_assignment(target, name, value)))
                continue
            if key == "sample" and (name in RETIRED_SAMPLE_FIELDS or name.startswith("virtual_")):
                # Retired idealized controls have no role in real CIF samples.
                continue
            if key == "sample" and name == "eds_elastic_trajectory_count":
                # Retired in schema 69: EDS histories now come from the exact
                # upstream ray bundle reaching the physical sample plane.
                # Old profiles remain loadable without reporting this known
                # no-op field as an unrelated unsupported parameter.
                continue
            if name not in allowed:
                skipped.append(f"{key}.{name}")
                continue
            converted = validate_runtime_assignment(target, name, value)
            pending.append((target.obj, name, converted))
    # Validate sample tables and legacy migration on a separate sample before
    # committing any device changes, including optional coefficient resets.
    candidate_sample = copy(state.sample)
    migration_notes = []
    if format_version < 6:
        migration_notes.append("Model switches preserve current lens excitation and polarity; historical shelves do not retune hardware.")
        if not isinstance(sample_model, dict) or "wave_illumination" not in sample_model:
            candidate_sample.wave_illumination = {"model": "ray_conditioned_reduced_order"}
            migration_notes.append("Missing illumination definition retained as legacy ray-conditioned reduced-order mode.")
        if "stem_execution_policy" not in sample_attributes:
            candidate_sample.stem_execution_policy = "auto"
            migration_notes.append("STEM execution uses the existing toolbar backend (auto policy).")
    for obj, name, value in pending:
        if obj is state.sample:
            setattr(candidate_sample, name, value)
    legacy_orientation_fields = tuple(f"specimen_rotation_{axis}_deg" for axis in "xyz")
    if format_version < 5 and (
        legacy_sample_source in {"preset", "cif"}
        or (sample_mode_was_explicit and candidate_sample.specimen_mode in {"atomic", "virtual"})
        or (sample_cif_was_explicit and not sample_mode_was_explicit)
        or (sample_preset_was_explicit and not sample_mode_was_explicit and not sample_cif_was_explicit)
        or any(name in sample_attributes for name in legacy_orientation_fields)
    ):
        # Legacy presets started from an identity *relative* rotation, while
        # a new reference sample already contains its CIF zone alignment.
        # Never use that new default as an extra legacy tilt.
        set_sample_orientation(candidate_sample, quaternion_from_euler_xyz_deg(
            tuple(sample_attributes.get(name, 0.0) for name in legacy_orientation_fields)
        ))
    if sample_model is not None:
        _apply_sample_model(candidate_sample, sample_model)
    if format_version < 5:
        if not sample_mode_was_explicit:
            if sample_cif_was_explicit and str(candidate_sample.cif_path).strip():
                candidate_sample.specimen_mode = "atomic"
            elif sample_preset_was_explicit:
                candidate_sample.specimen_mode = "virtual"
        # Older presets applied their crystal zone before the saved relative
        # orientation. Migrate only after the sample-model quaternion is loaded.
        migrated = migrate_legacy_structure_source(
            deepcopy(vars(candidate_sample)), legacy_source=legacy_sample_source,
            infer_implicit_atomic_preset=False,
        )
        vars(candidate_sample).update(migrated)
        migration_notes.append("Legacy specimen selection/orientation converted to the real-structure schema; virtual interaction controls retired.")
        if sample_attributes or sample_model is not None:
            for name in ("real_tail_material_source", "real_tail_screening_source"):
                if name not in sample_attributes:
                    setattr(candidate_sample, name, "manual")
    candidate_sample.virtual_interactions = []
    candidate_sample.virtual_regions = []
    for obj, name, value in pending:
        if obj is not state.sample:
            setattr(obj, name, value)
    vars(state.sample).update(vars(candidate_sample))
    state.simulation_mode = selected_mode
    state.simulation_mode_profiles = model_profiles
    for name in MODEL_SETTINGS:
        if name in model_settings:
            setattr(state, name, deepcopy(model_settings[name]))
    for lens in state.lenses:
        row = model_settings.get("lens_excitation", {}).get(lens.key)
        if row is not None:
            lens.percent, lens.polarity = row["percent"], row["polarity"]
    state._runtime_lens_field_provider_cache = {}
    state._lens_field_map_bindings = {}
    state._simulation_mode_maps = {}
    if hasattr(state.electron_gun, "source_representation"):
        state.electron_gun.source_representation = representation
        state.electron_gun.effective_source = parameters
    if representation == "effective_gaussian_schell":
        from temsim.optics.electron_gun.effective_source import validate_binding
        try:
            validate_binding(state.electron_gun, parameters)
        except ValueError:
            migration_notes.append("Saved effective source retained unchanged; its gun binding is stale. Rebind explicitly before new calculations.")
    state._profile_migration_report = {
        "from_version":format_version,"to_version":PROFILE_FORMAT_VERSION,
        "status":"migrated" if format_version < PROFILE_FORMAT_VERSION else "current",
        "notes":migration_notes,"skipped_fields":list(skipped),
        "hardware_policy":"Only explicit profile operating values applied; no automatic geometry or lens retuning to preserve an image",
        "geometry_policy":"Catalog TOML remains authoritative",
    }
    return skipped
