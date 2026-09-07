"""TOML persistence for operating parameters."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tomllib
from copy import deepcopy

import tomli_w

from temsim.assembly_catalog import AssemblySelection
from temsim.runtime_parameters import (
    editable_parameters,
    runtime_targets,
    validate_runtime_assignment,
)
from temsim.specimen.geometry import (
    normalise_quaternion_wxyz,
    sample_orientation_quaternion,
    set_sample_orientation,
)
from temsim.specimen.source import migrate_legacy_structure_source


PROFILE_FORMAT_VERSION = 3
_SAMPLE_MODEL_KEY = "__sample_model__"
_SIMULATION_MODEL_KEY = "__simulation_model__"
_PROFILE_VERSION_KEY = "__profile_format_version__"


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
    from temsim.simulation_modes import capture_mode_settings, mode_key
    devices = {}
    for key, target in runtime_targets(state).items():
        values = {
            parameter.name: parameter.value
            for parameter in editable_parameters(target)
            if parameter.value is not None
        }
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
        "simulation_model": {
            "mode": mode_key(state),
            "profiles": deepcopy(state.simulation_mode_profiles),
            "settings": capture_mode_settings(state),
        },
        "sample_model": {
            "orientation_quaternion_wxyz": list(
                normalise_quaternion_wxyz(
                    sample_orientation_quaternion(state.sample)
                )
            ),
            "zone_axis_uvw": list(state.sample.zone_axis_uvw),
            "in_plane_axis_uvw": list(state.sample.in_plane_axis_uvw),
            "virtual_interactions": deepcopy(
                state.sample.virtual_interactions
            ),
            "virtual_regions": deepcopy(state.sample.virtual_regions),
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
    if format_version not in {1, 2, PROFILE_FORMAT_VERSION}:
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
    values[_PROFILE_VERSION_KEY] = format_version
    model = document.get("simulation_model", {"mode": "custom"})
    if not isinstance(model, dict):
        raise ValueError("Operating profile simulation_model must be a table")
    values[_SIMULATION_MODEL_KEY] = model
    if format_version >= 2:
        sample_model = document.get("sample_model", {})
        if not isinstance(sample_model, dict):
            raise ValueError("Operating profile sample_model must be a table")
        values[_SAMPLE_MODEL_KEY] = dict(sample_model)
    return selection, values


def _apply_sample_model(state, model: dict) -> None:
    if not isinstance(model, dict):
        raise ValueError("Operating profile sample_model must be a table")
    sample = state.sample
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
    interactions = model.get("virtual_interactions", sample.virtual_interactions)
    regions = model.get("virtual_regions", sample.virtual_regions)
    element_sigma = model.get(
        "frozen_phonon_sigma_by_element_angstrom",
        sample.wave_frozen_phonon_sigma_by_element_angstrom,
    )
    if not isinstance(interactions, list) or not all(
        isinstance(row, dict) for row in interactions
    ):
        raise ValueError("Sample virtual_interactions must be an array of tables")
    if not isinstance(regions, list) or not all(
        isinstance(row, dict) for row in regions
    ):
        raise ValueError("Sample virtual_regions must be an array of tables")
    if not isinstance(element_sigma, dict):
        raise ValueError("Sample frozen-phonon element RMS values must be a table")
    converted_sigma = {}
    for symbol, value in element_sigma.items():
        converted = float(value)
        if converted <= 0.0:
            raise ValueError(
                f"Frozen-phonon RMS for {symbol} must be positive"
            )
        converted_sigma[str(symbol)] = converted
    set_sample_orientation(sample, quaternion)
    sample.zone_axis_uvw = zone
    sample.in_plane_axis_uvw = in_plane
    sample.virtual_interactions = deepcopy(interactions)
    sample.virtual_regions = deepcopy(regions)
    sample.wave_frozen_phonon_sigma_by_element_angstrom = converted_sigma


def apply_profile_values(state, values: dict) -> list[str]:
    if not isinstance(values, dict):
        raise ValueError("Operating profile devices must be a table")
    values = dict(values)
    format_version = int(values.pop(_PROFILE_VERSION_KEY, 1))
    sample_model = values.pop(_SAMPLE_MODEL_KEY, None)
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
    for key, attributes in values.items():
        target = targets.get(key)
        if target is None:
            skipped.append(key)
            continue
        if not isinstance(attributes, dict):
            raise ValueError(f"Operating profile device {key} must be a table")
        allowed = {parameter.name for parameter in editable_parameters(target)}
        for name, value in attributes.items():
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
            if key == "sample" and name == "cif_path":
                sample_cif_was_explicit = True
            if key == "sample" and name == "specimen_preset_key":
                sample_preset_was_explicit = True
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
    for obj, name, value in pending:
        setattr(obj, name, value)
    if legacy_sample_source:
        migrated = migrate_legacy_structure_source(
            {
                "specimen_mode": state.sample.specimen_mode,
                "cif_path": state.sample.cif_path,
            },
            legacy_source=legacy_sample_source,
        )
        state.sample.specimen_mode = migrated["specimen_mode"]
    elif not sample_mode_was_explicit:
        if sample_cif_was_explicit and str(state.sample.cif_path).strip():
            state.sample.specimen_mode = "atomic"
        elif sample_preset_was_explicit:
            state.sample.specimen_mode = "virtual"
    if sample_model is not None:
        _apply_sample_model(state, sample_model)
    elif format_version == 1:
        # V1 stored only the two legacy relative sliders.  Convert them once
        # to absolute-probability rows while retaining the scalar compatibility
        # fields for older scripts.
        from temsim.specimen.virtual import legacy_virtual_interaction_rows

        state.sample.virtual_interactions = legacy_virtual_interaction_rows(
            state.sample
        )
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
    return skipped
