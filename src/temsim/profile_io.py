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
    SAMPLE_SOURCE_FIELDS,
    editable_parameters,
    runtime_targets,
    validate_runtime_assignment,
)
from temsim.specimen.geometry import (
    normalise_quaternion_wxyz,
    sample_orientation_quaternion,
    set_sample_orientation,
)


PROFILE_FORMAT_VERSION = 12
_GUN_SOURCE_MODEL_KEY = "__gun_source_model__"
_ENERGY_FILTER_MODEL_KEY = "__energy_filter_model__"
_GUN_CONTROLS_KEY = "__gun_controls__"
_SAMPLE_MODEL_KEY = "__sample_model__"
_SIMULATION_MODEL_KEY = "__simulation_model__"
_PROFILE_VERSION_KEY = "__profile_format_version__"
_SLIT_PHYSICAL_FIELDS = frozenset({"gap_m", "centre_m", "zero_loss_offset_m", "calibrated_dispersion_um_per_ev"})


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
    _validate_profile_assembly(state, selection)
    from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
    from temsim.component_keys import ENERGY_FILTER_MULTIPOLE_KEYS
    devices = {}
    none_values = {}
    for key, target in runtime_targets(state).items():
        if key in ENERGY_FILTER_MULTIPOLE_KEYS:
            # One complete operating record owns every carrier field and calibration.
            continue
        values = {}
        for parameter in editable_parameters(target):
            if parameter.value is None:
                if parameter.name not in _nullable_fields(type(target.obj)):
                    raise ValueError(f"{key}.{parameter.name} cannot be none")
                none_values.setdefault(key, []).append(parameter.name)
            else:
                values[parameter.name] = parameter.value
        if key == "energy_filter_slit":
            values.update({name: getattr(target.obj, name) for name in sorted(_SLIT_PHYSICAL_FIELDS)})
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
        "gun_controls": capture_gun_profile_controls(state.electron_gun),
        "vacuum_map": state.vacuum_map.to_dict(),
        "gun_source_model": {
            "representation": getattr(state.electron_gun, "source_representation", "classical_particles"),
            **({"effective": asdict(state.electron_gun.effective_source)}
               if getattr(state.electron_gun, "effective_source", None) is not None else {}),
            **({"tip_coherence": asdict(state.electron_gun.emitter.coherence)}
               if getattr(state.electron_gun.emitter, "coherence", None) is not None else {}),
            **({"surface_model": state.electron_gun.emitter.surface_model.to_dict()}
               if getattr(state.electron_gun.emitter, "surface_model", None) is not None else {}),
            **({"curvature_model": state.electron_gun.emitter.curvature_model}
               if getattr(state.electron_gun.emitter, "curvature_nm_inv", 0.0) else {}),
        },
        # TOML has no null literal; clearing an optional coefficient is explicit.
        "none_values": none_values,
        "simulation_model": {
            "mode": mode_key(state),
            "profiles": deepcopy(state.simulation_mode_profiles),
            # Active lens controls already have one owner in devices. Mode
            # shelves retain their own excitations for explicit mode changes.
            "settings": {key: value for key, value in capture_mode_settings(state).items()
                         if key != "lens_excitation"},
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
    if state.energy_filter_installed:
        if tuple(element.key for element in state.energy_filter.multipoles) != ENERGY_FILTER_MULTIPOLE_KEYS:
            raise ValueError("Energy filter profile requires the ten installed multipole carriers in order")
        from temsim.optics.energy_filter_m12 import serialise_energy_filter_m12
        document["energy_filter_model"] = {
            "multipoles": [serialise_energy_filter_m12(element)
                           for element in state.energy_filter.multipoles],
        }
    _atomic_write_profile(Path(path), document)


def _validate_profile_assembly(state, selection):
    """Refuse an unreadable mixture of installed hardware and another catalog choice."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    catalog = AssemblyCatalog(getattr(state.electron_gun, "_manifest_catalog_root", None) or INSTRUMENT_CONFIG_ROOT)
    selection = catalog.normalise_selection(selection)
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None or catalog.selection_for_resolved(assembly) != selection:
        raise ValueError("Profile assembly selection does not match the resolved instrument")
    recording = next(option for option in catalog.recording_systems if option.name == selection.recording)
    has_filter = bool(recording.properties["energy_filter"])
    gun = next(option for option in catalog.guns if option.name == selection.gun)
    column = next(option for option in catalog.columns if option.name == selection.column)
    expected = {
        "energy_filter_installed": has_filter,
        "energy_filter_mode": "energy_filter" if has_filter else "no_energy_filter",
        "monochromator_installed": bool(gun.properties.get("monochromator", False)),
        "probe_corrector_installed": bool(column.properties["probe_corrector"]),
        "image_corrector_installed": bool(column.properties["image_corrector"]),
    }
    mismatches = [name for name, value in expected.items() if getattr(state, name) != value]
    if state.energy_filter.enabled != has_filter:
        mismatches.append("energy_filter.enabled")
    expected_gun = "thermionic" if gun.properties["electron_gun"] == "Thermionic" else "cold_feg"
    if state.electron_gun.type_key != expected_gun:
        mismatches.append("electron_gun")
    if bool(state.nanopulser.installed) != (selection.beam_blanker != "None"):
        mismatches.append("beam_blanker")
    if mismatches:
        raise ValueError("Profile hardware flags disagree with the selected assembly: " + ", ".join(mismatches))


def read_profile(path: str | Path) -> tuple[AssemblySelection, dict]:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if not isinstance(document, dict):
        raise ValueError("Operating profile must be a TOML table")
    format_version = document.get("format_version")
    if type(format_version) is not int or format_version != PROFILE_FORMAT_VERSION:
        raise ValueError(f"Unsupported operating-profile format {format_version!r}; expected {PROFILE_FORMAT_VERSION}")
    unknown = set(document) - {"format_version", "assembly", "devices", "none_values",
        "vacuum_map", "simulation_model", "gun_source_model", "sample_model", "energy_filter_model", "gun_controls"}
    if unknown:
        raise ValueError(f"Unknown operating-profile tables: {', '.join(sorted(unknown))}")
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
    for table, target in (("vacuum_map", "__vacuum_map__"),
                         ("simulation_model", _SIMULATION_MODEL_KEY),
                         ("gun_source_model", _GUN_SOURCE_MODEL_KEY),
                         ("sample_model", _SAMPLE_MODEL_KEY),
                         ("energy_filter_model", _ENERGY_FILTER_MODEL_KEY),
                         ("gun_controls", _GUN_CONTROLS_KEY)):
        if table in document:
            if not isinstance(document[table], dict):
                raise ValueError(f"Operating profile {table} must be a table")
            values[target] = dict(document[table])
    return selection, values


def _apply_sample_model(sample, model: dict) -> None:
    from temsim.physics.illumination import validate_illumination_config, require_production_illumination
    if not isinstance(model, dict):
        raise ValueError("Operating profile sample_model must be a table")
    unknown = set(model) - {"wave_illumination", "orientation_quaternion_wxyz",
        "zone_axis_uvw", "in_plane_axis_uvw", "frozen_phonon_sigma_by_element_angstrom"}
    if unknown:
        raise ValueError(f"Unknown sample-model fields: {', '.join(sorted(unknown))}")
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


def apply_profile_values(state, values: dict) -> None:
    """Apply validated current controls atomically; omitted controls stay unchanged."""
    if not isinstance(values, dict):
        raise ValueError("Operating profile devices must be a table")
    values = dict(values)
    gun_controls_present = _GUN_CONTROLS_KEY in values
    gun_controls = values.pop(_GUN_CONTROLS_KEY, None)
    if gun_controls_present and not isinstance(gun_controls, dict):
        raise ValueError("Operating profile gun_controls must be a table")
    filter_model_present = _ENERGY_FILTER_MODEL_KEY in values
    filter_model = values.pop(_ENERGY_FILTER_MODEL_KEY, None)
    if filter_model_present and not isinstance(filter_model, dict):
        raise ValueError("Operating profile energy_filter_model must be a table")
    from temsim.vacuum import VacuumMap
    vacuum_data = values.pop("__vacuum_map__", None)
    vacuum_candidate = VacuumMap.from_dict(vacuum_data) if vacuum_data is not None else state.vacuum_map
    format_version = values.pop(_PROFILE_VERSION_KEY, PROFILE_FORMAT_VERSION)
    if type(format_version) is not int or format_version != PROFILE_FORMAT_VERSION:
        raise ValueError(f"Unsupported operating-profile format {format_version!r}; expected {PROFILE_FORMAT_VERSION}")
    sample_model = values.pop(_SAMPLE_MODEL_KEY, None)
    from dataclasses import asdict
    emitter = state.electron_gun.emitter
    current_source = {"representation": getattr(state.electron_gun, "source_representation", "classical_particles")}
    current_effective = getattr(state.electron_gun, "effective_source", None)
    if current_effective is not None:
        current_source["effective"] = asdict(current_effective)
    if getattr(emitter, "coherence", None) is not None:
        current_source["tip_coherence"] = asdict(emitter.coherence)
    if getattr(emitter, "surface_model", None) is not None:
        current_source["surface_model"] = emitter.surface_model.to_dict()
    if hasattr(emitter, "curvature_model"):
        current_source["curvature_model"] = emitter.curvature_model
    gun_source = values.pop(_GUN_SOURCE_MODEL_KEY, current_source)
    if not isinstance(gun_source, dict) or set(gun_source)-{"representation", "effective", "tip_coherence", "surface_model", "curvature_model"}:
        raise ValueError("Operating profile has invalid electron-gun source fields")
    representation = gun_source.get("representation")
    if representation not in {"classical_particles", "effective_gaussian_schell"}:
        raise ValueError("Operating profile has an unknown electron-gun source model")
    if representation != "classical_particles":
        from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel, EXIT_SOURCE_REJECTION
        raise UnsupportedSourceModel(EXIT_SOURCE_REJECTION + " This historical profile cannot be activated.")
    parameters = getattr(state.electron_gun, "effective_source", None)
    if "effective" in gun_source:
        from temsim.optics.electron_gun.effective_source import EffectiveGunSource
        parameters = EffectiveGunSource(**gun_source["effective"])
    tip_coherence = None
    surface_model = None
    if "surface_model" in gun_source:
        from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
        if state.electron_gun.type_key != "cold_feg" or "tip_coherence" in gun_source:
            raise ValueError("Select one FEG tip emission model; surface and historical coherent inputs cannot coexist")
        surface_model = TipSurfaceModel.from_dict(gun_source["surface_model"])
    if "tip_coherence" in gun_source:
        from temsim.optics.electron_gun.tip_coherence import TipCoherence
        if state.electron_gun.type_key != "cold_feg" or not isinstance(gun_source["tip_coherence"], dict):
            raise ValueError("Tip coherence requires a cold FEG parameter table")
        try:
            tip_coherence = TipCoherence(**gun_source["tip_coherence"]).validate()
        except TypeError as error:
            raise ValueError("Operating profile has invalid tip coherence fields") from error
    from temsim.simulation_modes import validate_mode, normalise_profiles, MODEL_SETTINGS
    model = values.pop(_SIMULATION_MODEL_KEY, {"mode": state.simulation_mode,
        "profiles": state.simulation_mode_profiles, "settings": {}})
    if not isinstance(model, dict):
        raise ValueError("Operating profile simulation_model must be a table")
    if set(model) - {"mode", "profiles", "settings"}:
        raise ValueError("Unknown simulation-model fields")
    selected_mode = validate_mode(model.get("mode", "custom"))
    model_profiles = normalise_profiles(model.get("profiles", {}))
    model_settings = normalise_profiles({selected_mode: model.get("settings", {})})[selected_mode]
    if "lens_excitation" in model_settings:
        raise ValueError("Active lens excitation belongs only in profile devices, not simulation_model.settings")
    targets = runtime_targets(state)
    pending = []
    for key, attributes in values.items():
        target = targets.get(key)
        if target is None:
            raise ValueError(f"Unknown operating-profile device: {key}")
        if not isinstance(attributes, dict):
            raise ValueError(f"Operating profile device {key} must be a table")
        allowed = {parameter.name for parameter in editable_parameters(target)}
        if key == getattr(state.electron_gun.emitter, "key", None) and state.electron_gun.type_key == "cold_feg":
            # Validate emission controls against the incoming physical model.
            from temsim.runtime_parameters import RuntimeTarget
            incoming_emitter = copy(target.obj)
            incoming_emitter.surface_model = surface_model
            incoming_emitter.coherence = tip_coherence
            allowed = {p.name for p in editable_parameters(RuntimeTarget(key, target.label, incoming_emitter))}
            if "curvature_nm_inv" in attributes:
                # Validate even when the incoming model hides the
                # control. Incompatible inputs must not be silently skipped.
                allowed.add("curvature_nm_inv")
        if surface_model is None and target.hidden_parameters:
            from temsim.runtime_parameters import RuntimeTarget
            allowed = {p.name for p in editable_parameters(RuntimeTarget(key, target.label, target.obj))}
        if key == "energy_filter_slit":
            allowed.update(_SLIT_PHYSICAL_FIELDS)
        if key == "sample":
            allowed.update(SAMPLE_SOURCE_FIELDS)
        for name, value in attributes.items():
            if value is None:
                if name not in allowed or name not in _nullable_fields(type(target.obj)):
                    raise ValueError(f"{key}.{name} cannot be none")
                pending.append((target.obj, name, None))
                continue
            if name not in allowed:
                raise ValueError(f"Unknown operating-profile field: {key}.{name}")
            # Coupled tip inputs are validated together below, against the
            # incoming model/width, never against partially restored values.
            converted = validate_runtime_assignment(target, name, value,
                                                     validate_source_geometry=False)
            pending.append((target.obj, name, converted))
    # Validate coupled controls on a separate sample before
    # committing any device changes, including optional coefficient resets.
    candidate_sample = copy(state.sample)
    candidate_emitter = None
    if state.electron_gun.type_key == "cold_feg":
        candidate_emitter = copy(state.electron_gun.emitter)
        for obj, name, value in pending:
            if obj is state.electron_gun.emitter:
                setattr(candidate_emitter, name, value)
        candidate_emitter.coherence = tip_coherence
        candidate_emitter.surface_model = surface_model
        from temsim.optics.electron_gun.tip_curvature import MODEL
        candidate_emitter.curvature_model = gun_source.get("curvature_model", MODEL)
    prepared_gun_controls = None
    if gun_controls is not None:
        from temsim.optics.electron_gun.profile_controls import prepare_gun_profile_controls
        candidate_accelerator = copy(state.electron_gun.accelerator)
        for obj, name, value in pending:
            if obj is state.electron_gun.accelerator:
                setattr(candidate_accelerator, name, value)
        prepared_gun_controls = prepare_gun_profile_controls(
            state.electron_gun, gun_controls,
            emitter=candidate_emitter, accelerator=candidate_accelerator,
        )
        if candidate_emitter is not None:
            candidate_emitter.quadrature = prepared_gun_controls.tip_quadrature
    if candidate_emitter is not None:
        candidate_emitter.validate()
    for obj, name, value in pending:
        if obj is state.sample:
            setattr(candidate_sample, name, value)
    if sample_model is not None:
        _apply_sample_model(candidate_sample, sample_model)
    # Scan and wobble are a coupled constraint: checking individual fields
    # against the old object would permit an invalid simultaneous enable.
    for pair in (state.ac_deflector, state.descan_deflector):
        candidate_pair = copy(pair)
        for obj, name, value in pending:
            if obj is pair:
                setattr(candidate_pair, name, value)
        candidate_pair.validate()
    filter_candidate = _filter_controls_candidate(state, values, targets, pending, filter_model)
    candidate_sample.virtual_interactions = []
    candidate_sample.virtual_regions = []
    for obj, name, value in pending:
        if obj is not state.sample:
            setattr(obj, name, value)
    if state.electron_gun.type_key == "cold_feg":
        state.electron_gun.emitter.coherence = tip_coherence
        state.electron_gun.emitter.surface_model = surface_model
        state.electron_gun.emitter.curvature_nm_inv = candidate_emitter.curvature_nm_inv
        state.electron_gun.emitter.curvature_model = candidate_emitter.curvature_model
    if prepared_gun_controls is not None:
        from temsim.optics.electron_gun.profile_controls import apply_prepared_gun_profile_controls
        apply_prepared_gun_profile_controls(state.electron_gun, prepared_gun_controls)
    if filter_candidate is not None:
        for original, candidate in filter_candidate:
            vars(original).update(vars(candidate))
    vars(state.sample).update(vars(candidate_sample))
    state.vacuum_map = vacuum_candidate
    state.simulation_mode = selected_mode
    state.simulation_mode_profiles = model_profiles
    for name in MODEL_SETTINGS:
        if name in model_settings:
            setattr(state, name, deepcopy(model_settings[name]))
    state._runtime_lens_field_provider_cache = {}
    state._lens_field_map_bindings = {}
    state._simulation_mode_maps = {}
    if hasattr(state.electron_gun, "source_representation"):
        state.electron_gun.source_representation = representation
        state.electron_gun.effective_source = parameters


def _filter_controls_candidate(state, values, targets, pending, filter_model=None):
    """Apply explicit software requests on detached physical filter components."""
    ef = state.energy_filter
    if ef is None or "energy_filter" not in targets:
        if filter_model is not None:
            raise ValueError("Energy filter model requires an installed energy filter")
        return None
    restored_multipoles = None
    if filter_model is not None:
        from temsim.component_keys import ENERGY_FILTER_MULTIPOLE_KEYS
        from temsim.optics.energy_filter_m12 import energy_filter_multipole_from_dict
        if set(filter_model) != {"multipoles"}:
            raise ValueError("Energy filter model requires only the current multipoles table")
        saved = filter_model["multipoles"]
        if not isinstance(saved, list) or len(saved) != len(ENERGY_FILTER_MULTIPOLE_KEYS):
            raise ValueError("Energy filter model requires exactly ten current multipole records")
        conflicts = set(ENERGY_FILTER_MULTIPOLE_KEYS) & values.keys()
        if conflicts:
            raise ValueError("Multipole controls appear in both devices and energy_filter_model: " + ", ".join(sorted(conflicts)))
        restored_multipoles = [
            energy_filter_multipole_from_dict(record, index, ef.voltage_reference_kv)
            for index, record in enumerate(saved, start=1)
        ]
        for current, restored in zip(ef.multipoles, restored_multipoles, strict=True):
            if current.name != restored.name:
                raise ValueError(f"Multipole name must match the installed component {current.key}")
    children = [ef, ef.energy_slit, ef.bias_tube, ef.fast_shutter,
                ef.camera_deflector, ef.zebra_detector, *ef.multipoles]
    if restored_multipoles is None and not any(obj is child for obj, _, _ in pending for child in children):
        return None
    candidate = copy(ef)
    for attribute in ("energy_slit", "bias_tube", "fast_shutter", "camera_deflector", "zebra_detector", "multipoles"):
        setattr(candidate, attribute, deepcopy(getattr(ef, attribute)))
    copies = [candidate, candidate.energy_slit, candidate.bias_tube,
              candidate.fast_shutter, candidate.camera_deflector,
              candidate.zebra_detector, *candidate.multipoles]
    pairs = list(zip(children, copies))
    by_id = {id(original): cloned for original, cloned in pairs}
    parent_values = values.get("energy_filter", {})
    for obj, name, value in pending:
        if obj is ef:
            setattr(candidate, name, value)
    from temsim.optics.energy_filter import configure_energy_filter_operating_mode
    mode_requested = {"operating_mode", "multi_eels_enabled", "multi_eels_region_count"} & parent_values.keys()
    if mode_requested:
        configure_energy_filter_operating_mode(candidate, candidate.operating_mode)
    # Complete records override mode defaults. Partial mode commands apply them.
    for obj, name, value in pending:
        if id(obj) in by_id:
            setattr(by_id[id(obj)], name, value)
    slit_values = values.get(candidate.energy_slit.key, {})
    physical_present = _SLIT_PHYSICAL_FIELDS & slit_values.keys()
    if physical_present and physical_present != _SLIT_PHYSICAL_FIELDS:
        raise ValueError("Physical slit state requires gap, centre, zero-loss offset and dispersion together")
    if not physical_present and {"requested_width_ev", "requested_centre_loss_ev"} & slit_values.keys():
        candidate.energy_slit.configure_energy_window(
            candidate.energy_slit.requested_centre_loss_ev,
            candidate.energy_slit.requested_width_ev,
        )
    if restored_multipoles is not None:
        from temsim.physics.finite_multipole_field import FiniteMultipoleField
        for component, restored in zip(candidate.multipoles, restored_multipoles, strict=True):
            # Preserve the installed field envelope and coordinate frame. The
            # profile owns excitation, not a replacement mechanical layout.
            component.field_backend = FiniteMultipoleField(
                restored.multipole_field,
                component.field_backend.envelope,
                fringe_expansion_order=restored.field_backend.fringe_expansion_order,
            )
            component.calibration = restored.calibration
            component.enabled = restored.enabled
            component.__post_init__()
    candidate.energy_slit.__post_init__()
    for component in (candidate.bias_tube, candidate.fast_shutter,
                      candidate.camera_deflector, candidate.zebra_detector):
        component.validate()
    # Keep original component identities; candidates never become a second tree.
    for attribute in ("energy_slit", "bias_tube", "fast_shutter", "camera_deflector", "zebra_detector", "multipoles"):
        setattr(candidate, attribute, getattr(ef, attribute))
    return pairs
