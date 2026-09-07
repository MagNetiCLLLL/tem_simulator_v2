"""Stable cache identities for expensive simulator calculation products.

The GUI may calculate a fast Preview and a complete High accuracy result for
the same editable microscope state.  These identities keep those two result
streams separate and describe which expensive products can be reused after a
targeted setting change.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Iterable


_SAMPLE_REGION_PREFIXES = ("sample_region_",)
_EDS_PREFIXES = ("eds_",)
_STEM_PREFIXES = ("stem_",)
_FOURDSTEM_CAPTURE_FIELDS = frozenset({
    "stem_wave_enabled",
})
_FOURDSTEM_RESPONSE_FIELDS = frozenset({
    "stem_fourdstem_response_mode",
    "stem_fourdstem_quantum_efficiency",
    "stem_fourdstem_charge_spread_sigma_px",
    "stem_fourdstem_dark_electrons_per_pixel",
    "stem_fourdstem_read_noise_electrons_rms",
    "stem_fourdstem_saturation_electrons",
    "stem_fourdstem_gain_counts_per_electron",
    "stem_fourdstem_offset_counts",
    "stem_fourdstem_poisson_enabled",
    "stem_fourdstem_seed",
})
_FOURDSTEM_VIRTUAL_FIELDS = frozenset({
    "stem_fourdstem_virtual_inner_mrad",
    "stem_fourdstem_virtual_outer_mrad",
})
_POST_SAMPLE_PROJECTION_LENS_KEYS = frozenset({
    "diffraction_lens",
    "intermediate_lens",
    "projector_lens_1",
    "projector_lens_2",
})
_LOADED_INPUT_DIGEST_CACHE: dict[str, tuple[object, str]] = {}
_WAVE_COORDINATE_SCHEMA = "centred-real-space-v2"


def _wave_digest(payload):
    # The old wave solver mixed corner-origin probes and centred specimen
    # arrays. Invalidate those wave/cube products only; incident, interaction
    # and EDS checkpoints remain reusable across this numerical correction.
    return _digest({"wave_coordinate_schema": _WAVE_COORDINATE_SCHEMA, "parameters": payload})


def _drop_sample_fields(
    payload: dict[str, object],
    *,
    prefixes: Iterable[str] = (),
    names: Iterable[str] = (),
    keep: Iterable[str] = (),
) -> dict[str, object]:
    result = deepcopy(payload)
    sample = dict(result.get("sample", {}))
    exact = set(names)
    retained = set(keep)
    for name in tuple(sample):
        if name in retained:
            continue
        if name in exact or any(name.startswith(prefix) for prefix in prefixes):
            sample.pop(name, None)
    result["sample"] = sample
    return result


def _drop_runtime_solver_state(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result.pop("active_backend", None)
    # Inactive fidelity shelves do not change the active calculation. Keep
    # simulation_mode itself in every affected product identity.
    result.pop("simulation_mode_profiles", None)
    if result.get("simulation_mode", "custom") in {"ideal", "analytical"}:
        result["lens_field_map_descriptors"] = {}
    # This map only remembers enabled preferences while optional hardware is
    # removed/reinserted.  The current components already carry the effective
    # installed/enabled state consumed by every solver.
    result.pop("layout_reference_enabled", None)
    # AC/Descan lower-coil gains are compatibility read-backs of the
    # calibrated 2x2 coupling matrices.  Solving scan geometry updates those
    # read-backs, but they are not independent user inputs.  Keeping them in
    # an identity made an otherwise unchanged result invalidate every cache
    # signature merely because its scan calibration had been evaluated.
    correctors = []
    for item in result.get("corrector_elements", ()):
        row = dict(item)
        key = str(row.get("key", ""))
        if key in {"ac_deflector", "descan_deflector"}:
            row.pop("lower_coil_gain", None)
        if key == "ac_deflector":
            # The specimen pixel pitch and active optical response determine
            # the calibrated AC command matrix.  These legacy angular
            # amplitudes only initialise the private matrix before that solve
            # and are overwritten before every scan calculation.
            row.pop("scan_amplitude_x_mrad", None)
            row.pop("scan_amplitude_y_mrad", None)
        if key == "descan_deflector":
            # Descan consumes the opposite calibrated AC raster.  These
            # duplicate values are overwritten from AC before every use and
            # therefore cannot be independent cache dependencies.
            for name in (
                "scan_amplitude_x_mrad",
                "scan_amplitude_y_mrad",
                "scan_frame_period_s",
                "scan_pixels_x",
                "scan_lines",
                "scan_pixel_size_nm",
            ):
                row.pop(name, None)
        correctors.append(row)
    if "corrector_elements" in result:
        result["corrector_elements"] = correctors
    return result


def _drop_physical_current_scale(
    payload: dict[str, object]
) -> dict[str, object]:
    """Remove dose-only controls from geometry/trajectory identities."""

    result = deepcopy(payload)
    result.pop("column_current_limit_percent", None)
    return result


def _drop_emitter_ray_count(
    payload: dict[str, object],
) -> dict[str, object]:
    """Remove source sampling density while retaining optical integration.

    Scan calibration geometry is a response calculation, not a Monte-Carlo
    sample of the emitter.  Its identity must still retain ``step_mm`` and
    ``history_step_mm`` because those controls set the response-path grid.
    """

    result = deepcopy(payload)
    gun = dict(result.get("electron_gun", {}))
    components = {
        str(key): dict(value)
        for key, value in dict(gun.get("components", {})).items()
    }
    for component in components.values():
        component.pop("ray_count", None)
    gun["components"] = components
    result["electron_gun"] = gun
    return result


def _drop_energy_filter_controls(
    payload: dict[str, object],
) -> dict[str, object]:
    """Remove spectrometer-internal settings from non-filter products."""

    result = deepcopy(payload)
    result.pop("energy_filter", None)
    return result


def _drop_post_column_specimen_content(
    payload: dict[str, object],
) -> dict[str, object]:
    """Remove external inputs not consumed by the column ray calculation.

    CIF atoms, Virtual-region density maps, reference specimen data,
    holder/support data and Bote-Salvat coefficients are consumed by
    wave/specimen/EDS stages after the electron column has been traced.
    """

    result = deepcopy(payload)
    result.pop("_cif_content_identity", None)
    result.pop("_virtual_region_map_identities", None)
    result.pop("_loaded_solver_input_identities", None)
    return result


def _incident_only_payload(payload: dict[str, object]) -> dict[str, object]:
    """Keep only dependencies consumed before the specimen boundary."""

    result = _drop_post_column_specimen_content(payload)
    result["sample"] = {}
    result.pop("_loaded_solver_input_identities", None)
    return result


def _select_geometry_scope(
    payload: dict[str, object],
    scope: str,
) -> dict[str, object]:
    """Replace full assembly data with one stage-bounded geometry identity."""

    result = deepcopy(payload)
    scopes = result.pop("_resolved_assembly_geometry_fingerprints", {})
    if isinstance(scopes, Mapping) and scope in scopes:
        result.pop("_resolved_assembly", None)
        result.pop("_resolved_assembly_geometry_fingerprint", None)
        result["_resolved_assembly_geometry_fingerprint"] = scopes[scope]
    if scope == "upstream":
        # The D/I/P lens rows have already been removed by
        # _drop_post_sample_projection_controls.  Imported field-map
        # descriptors are serialized separately at State root, so filter both
        # those contracts and their live file identities to the same retained
        # lens set.  Otherwise merely importing/editing a downstream P1 map
        # would falsely invalidate incident/specimen-local products.
        retained_lens_keys = {
            str(row.get("key", ""))
            for row in result.get("lenses", ())
            if isinstance(row, Mapping) and str(row.get("key", ""))
        }
        descriptors = result.get("lens_field_map_descriptors")
        if isinstance(descriptors, Mapping):
            result["lens_field_map_descriptors"] = {
                str(lens_key): descriptor
                for lens_key, descriptor in descriptors.items()
                if str(lens_key) in retained_lens_keys
            }
        field_maps = result.get("_lens_field_map_content_identities")
        if isinstance(field_maps, (tuple, list)):
            result["_lens_field_map_content_identities"] = tuple(
                row
                for row in field_maps
                if isinstance(row, Mapping)
                and str(row.get("lens_key", "")) in retained_lens_keys
            )
    return result


def _drop_post_sample_projection_controls(
    payload: dict[str, object],
) -> dict[str, object]:
    """Remove only the independently repeatable D/I/P projection controls.

    The D, I, P1 and P2 excitations and the image/diffraction selection affect
    propagation after the Objective-side wave/specimen checkpoints.  Other
    post-sample components are intentionally retained: the Objective aperture,
    image corrector and image-aberration controls already contribute to the
    stored complex wave, while SampleRegionResult currently also stores full
    downstream electron branches.
    """

    result = deepcopy(payload)
    result.pop("projector_mode", None)
    result.pop("recording_planes", None)
    result["lenses"] = [
        lens
        for lens in result.get("lenses", [])
        if str(lens.get("key", ""))
        not in _POST_SAMPLE_PROJECTION_LENS_KEYS
    ]
    placements = result.get("component_placements")
    if isinstance(placements, Mapping):
        result["component_placements"] = {
            key: value
            for key, value in placements.items()
            if str(key) not in _POST_SAMPLE_PROJECTION_LENS_KEYS
        }
    return _select_geometry_scope(result, "upstream")


def _drop_numerical_resolution(payload: dict[str, object]) -> dict[str, object]:
    result = _drop_runtime_solver_state(payload)
    result.pop("step_mm", None)
    result.pop("history_step_mm", None)
    gun = dict(result.get("electron_gun", {}))
    components = {
        str(key): dict(value)
        for key, value in dict(gun.get("components", {})).items()
    }
    for component in components.values():
        component.pop("ray_count", None)
    gun["components"] = components
    result["electron_gun"] = gun
    return result


def _file_content_identity(raw_path: object) -> dict[str, object] | None:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    try:
        data = path.read_bytes()
        stat = path.stat()
    except OSError:
        return {"path": str(path), "available": False}
    return {
        "path": str(path.resolve()),
        "available": True,
        "size": int(stat.st_size),
        "sha256": sha256(data).hexdigest(),
    }


def _cif_content_identity(payload: dict[str, object]) -> dict[str, object] | None:
    sample = dict(payload.get("sample", {}))
    if str(sample.get("specimen_mode", "")).strip().lower() != "atomic":
        return None
    return _file_content_identity(sample.get("cif_path", ""))


def _virtual_region_map_identities(
    payload: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Identify density-map bytes consumed by Virtual specimen regions."""

    sample = dict(payload.get("sample", {}))
    if str(sample.get("specimen_mode", "")).strip().lower() != "virtual":
        return ()
    identities = []
    for index, row in enumerate(sample.get("virtual_regions", ()) or ()):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("kind", "")).strip().lower() != "map":
            continue
        identity = _file_content_identity(row.get("map_path", ""))
        if identity is not None:
            identities.append({"region_index": index, **identity})
    return tuple(identities)


def _lens_field_map_content_identities(
    payload: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Identify current bytes behind serialized imported-field descriptors."""

    descriptors = payload.get("lens_field_map_descriptors", {})
    if not isinstance(descriptors, Mapping):
        return ()
    identities = []
    for lens_key, descriptor in sorted(
        descriptors.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(descriptor, Mapping):
            continue
        identity = _file_content_identity(descriptor.get("source_path", ""))
        if identity is not None:
            identities.append({"lens_key": str(lens_key), **identity})
    return tuple(identities)


def _loaded_input_digest(cache_key: str, value: object) -> str:
    """Hash the parsed object actually consumed by a cached data loader."""

    cached = _LOADED_INPUT_DIGEST_CACHE.get(str(cache_key))
    if cached is not None and cached[0] is value:
        return cached[1]
    digest = _digest({"value": _json_value(value)})
    _LOADED_INPUT_DIGEST_CACHE[str(cache_key)] = (value, digest)
    return digest


def _loaded_solver_input_identities(state) -> dict[str, str]:
    """Return identities for parsed external data used by specimen solvers.

    These are based on the loader-owned immutable objects rather than directly
    hashing their files.  If a loader is still serving an older cached object,
    the identity therefore describes the same data the solver will really use.
    """

    sample = getattr(state, "sample", None)
    if sample is None:
        return {}
    identities: dict[str, str] = {}
    try:
        from temsim.specimen.source import wave_template_preset_key
        from temsim.specimen.presets import load_specimen_preset

        preset = load_specimen_preset(
            wave_template_preset_key(
                sample,
                inserted=bool(getattr(sample, "inserted", True)),
            )
        )
        identities["specimen_preset"] = _loaded_input_digest(
            f"specimen_preset:{preset.key}", preset
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        identities["specimen_preset_error"] = _digest({
            "type": type(exc).__name__,
            "message": str(exc),
        })

    try:
        from temsim.specimen.support import load_support_catalog

        support = load_support_catalog()
        identities["support_catalog"] = _loaded_input_digest(
            "support_catalog", support
        )
    except (OSError, TypeError, ValueError) as exc:
        identities["support_catalog_error"] = _digest({
            "type": type(exc).__name__,
            "message": str(exc),
        })

    if bool(getattr(sample, "eds_enabled", False)):
        try:
            from temsim.detector.eds_atomic import (
                load_bote_salvat_coefficients,
            )

            coefficients = load_bote_salvat_coefficients()
            identities["bote_salvat"] = _loaded_input_digest(
                "bote_salvat", coefficients
            )
        except (OSError, TypeError, ValueError) as exc:
            identities["bote_salvat_error"] = _digest({
                "type": type(exc).__name__,
                "message": str(exc),
            })
    return identities


def _json_value(value):
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _state_payload(state) -> dict[str, object]:
    payload = _drop_runtime_solver_state(state.to_dict())
    # Dynamic scan/deflector phase is deliberately not persisted by State,
    # but it is read by ray, wave, scan and detector propagation.
    payload["simulation_time_s"] = float(
        getattr(state, "simulation_time_s", 0.0)
    )
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        payload["_resolved_assembly"] = _json_value(assembly)
        # Include final live positions and resolved pole/vacuum geometry.  TOML
        # owned coordinates are intentionally absent from State.to_dict(), so
        # the resolved assembly alone would miss a permitted live Z change.
        # The helper excludes lens excitation percent.
        from temsim.calculation_manifest import (
            resolved_assembly_geometry_fingerprints,
        )

        geometry_fingerprints = resolved_assembly_geometry_fingerprints(
            state
        )
        payload["_resolved_assembly_geometry_fingerprint"] = (
            geometry_fingerprints["complete"]
        )
        payload["_resolved_assembly_geometry_fingerprints"] = (
            geometry_fingerprints
        )
    cif_identity = _cif_content_identity(payload)
    if cif_identity is not None:
        payload["_cif_content_identity"] = cif_identity
    region_maps = _virtual_region_map_identities(payload)
    if region_maps:
        payload["_virtual_region_map_identities"] = region_maps
    lens_field_maps = _lens_field_map_content_identities(payload)
    if lens_field_maps:
        payload["_lens_field_map_content_identities"] = lens_field_maps
    loaded_inputs = _loaded_solver_input_identities(state)
    if loaded_inputs:
        payload["_loaded_solver_input_identities"] = loaded_inputs
    return payload


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def state_model_signature(state) -> str:
    """Identify one editable physical state, independent of solver density."""

    return _digest(_drop_numerical_resolution(_state_payload(state)))


def external_model_signature(state) -> str:
    """Identify loaded assembly/CIF content independently of live controls."""

    payload = _state_payload(state)
    external = {
        key: payload[key]
        for key in (
            "_resolved_assembly",
            "_resolved_assembly_geometry_fingerprint",
            "_resolved_assembly_geometry_fingerprints",
            "_cif_content_identity",
            "_virtual_region_map_identities",
            "_lens_field_map_content_identities",
            "_loaded_solver_input_identities",
        )
        if key in payload
    }
    return _digest(external)


def calculation_signatures(state) -> dict[str, str]:
    """Return dependency-scoped identities for reusable calculation products."""

    return _calculation_signatures_from_payload(_state_payload(state))


def calculation_signatures_for_request(
    state,
    *,
    ray_count: int,
    step_mm: float,
    minimum_history_step_mm: float = 0.5,
) -> dict[str, str]:
    """Hash a solver request without rebuilding the complete State object.

    The GUI already owns a validated live state.  High-accuracy status checks
    only override the emitter sample count and integration/history steps, so
    reconstructing every optical component through ``State.from_dict`` is
    unnecessary and made ordinary edits visibly block the interface.
    """

    rays = int(ray_count)
    step = float(step_mm)
    if rays <= 0 or step <= 0.0:
        raise ValueError("Calculation resolution must be positive")
    full = _state_payload(state)
    full["step_mm"] = step
    full["history_step_mm"] = max(
        step, float(minimum_history_step_mm)
    )
    gun = dict(full.get("electron_gun", {}))
    components = {
        str(key): dict(value)
        for key, value in dict(gun.get("components", {})).items()
    }
    emitter = getattr(getattr(state, "electron_gun", None), "emitter", None)
    emitter_key = str(getattr(emitter, "key", ""))
    if emitter_key in components:
        components[emitter_key]["ray_count"] = rays
    else:
        ray_components = [
            component
            for component in components.values()
            if "ray_count" in component
        ]
        if len(ray_components) != 1:
            raise ValueError("Could not identify the active electron emitter")
        ray_components[0]["ray_count"] = rays
    gun["components"] = components
    full["electron_gun"] = gun
    return _calculation_signatures_from_payload(full)


def _calculation_signatures_from_payload(
    full: dict[str, object],
) -> dict[str, str]:
    """Build scoped hashes from one canonical calculation payload."""

    no_region = _drop_sample_fields(
        full,
        prefixes=_SAMPLE_REGION_PREFIXES,
    )
    non_filter = _drop_energy_filter_controls(no_region)
    non_filter_geometric = deepcopy(non_filter)
    non_filter_geometric.pop("image_aberrations", None)
    column = _drop_post_column_specimen_content(
        _drop_physical_current_scale(_drop_sample_fields(
            non_filter_geometric,
            prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
        ))
    )
    incident = _incident_only_payload(
        _drop_post_sample_projection_controls(column)
    )
    # Elastic particle transport uses holder/support geometry and its own
    # stochastic controls, but not spectrum binning, detector response or
    # Poisson presentation settings.
    elastic_keep = {
        "eds_transport_mode",
        "eds_elastic_seed",
        "eds_elastic_max_events",
        "eds_support_material_key",
        "eds_support_mesh_key",
        "eds_support_offset_x_um",
        "eds_support_offset_y_um",
        "eds_support_rotation_deg",
    }
    elastic = _drop_post_sample_projection_controls(
        _drop_physical_current_scale(
            _drop_sample_fields(
                non_filter_geometric,
                prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
                keep=elastic_keep,
            )
        )
    )
    wave = _drop_physical_current_scale(_drop_sample_fields(
        non_filter,
        prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
        keep={"stem_wave_enabled"},
    ))
    wave_source = _drop_post_sample_projection_controls(wave)
    fourdstem_cube = _drop_post_sample_projection_controls(
        _drop_physical_current_scale(
            _drop_sample_fields(
                non_filter_geometric,
                prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
                keep=_FOURDSTEM_CAPTURE_FIELDS,
            )
        )
    )
    fourdstem_virtual_detectors = _drop_post_sample_projection_controls(
        _drop_sample_fields(
            non_filter_geometric,
            prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
            keep=(
                *_FOURDSTEM_CAPTURE_FIELDS,
                *_FOURDSTEM_RESPONSE_FIELDS,
                *_FOURDSTEM_VIRTUAL_FIELDS,
            ),
        )
    )
    fourdstem_physical_recording = _drop_physical_current_scale(
        _drop_sample_fields(
            non_filter_geometric,
            prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
            keep=_FOURDSTEM_CAPTURE_FIELDS,
        )
    )
    eds = _drop_post_sample_projection_controls(
        _drop_sample_fields(
            non_filter_geometric,
            prefixes=("wave_", *_STEM_PREFIXES),
        )
    )
    scan_ray_paths = _drop_emitter_ray_count(
        _drop_physical_current_scale(
            _drop_sample_fields(
                non_filter_geometric,
                prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
            )
        )
    )
    scan_geometry = deepcopy(scan_ray_paths)
    # Geometry evaluates the complete raster response and never samples the
    # dynamic command at the currently displayed playback time.
    scan_geometry.pop("simulation_time_s", None)
    stem = _drop_sample_fields(
        non_filter_geometric,
        prefixes=(*_EDS_PREFIXES, "stem_fourdstem_"),
        keep=elastic_keep,
    )
    stem_transport = _drop_physical_current_scale(stem)
    sample_region_base = _drop_energy_filter_controls(full)
    sample_region_base.pop("image_aberrations", None)
    sample_region_full = _drop_sample_fields(
        sample_region_base,
        prefixes=("wave_", *_STEM_PREFIXES),
    )
    # The local electron/X-ray paths end at the specimen-region boundaries.
    # D/I/P excitation, projector mode and recording-plane selection only
    # affect the separately repeatable downstream reinjection.
    sample_region = _drop_post_sample_projection_controls(
        sample_region_full
    )
    sample_region_downstream = _select_geometry_scope(
        _drop_physical_current_scale(
            _drop_sample_fields(
                sample_region_base,
                prefixes=("wave_", *_STEM_PREFIXES, *_EDS_PREFIXES),
                names=(
                    "sample_region_upstream_distance_um",
                    "sample_region_photon_path_count",
                    "sample_region_secondary_path_count",
                    "sample_region_seed",
                ),
                keep=elastic_keep,
            )
        ),
        "post_sample",
    )
    # EnergyFilterResult and SampleRegionResult both carry absolute current or
    # photon counts.  Their identities must retain the current scale even
    # though their geometric trajectories could be split out in the future.
    raw_energy_filter = no_region.get("energy_filter", {})
    energy_filter_mode = str(
        raw_energy_filter.get("operating_mode", "eels")
        if isinstance(raw_energy_filter, Mapping)
        else "eels"
    ).strip().lower()
    energy_filter_base = deepcopy(no_region)
    if energy_filter_mode == "eftem":
        # EFTEM carries the TEM camera-plane wave intensity into the filter;
        # wave and image-aberration inputs are therefore real dependencies.
        energy_filter = _drop_sample_fields(
            energy_filter_base,
            prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
        )
    else:
        # EELS transport remains independent of coherent TEM image formation.
        energy_filter_base.pop("image_aberrations", None)
        energy_filter = _drop_sample_fields(
            energy_filter_base,
            prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
        )
    return {
        "request": _digest(full),
        "column": _digest(column),
        "incident": _digest(incident),
        "elastic": _digest(elastic),
        "wave": _wave_digest(wave),
        "wave_source": _wave_digest(wave_source),
        "fourdstem_cube": _wave_digest(fourdstem_cube),
        "fourdstem_virtual_detectors": _wave_digest(fourdstem_virtual_detectors),
        "fourdstem_physical_recording": _wave_digest(
            fourdstem_physical_recording
        ),
        "eds": _digest(eds),
        "energy_filter": _digest(energy_filter),
        "scan_geometry": _digest(scan_geometry),
        "scan_ray_paths": _digest(scan_ray_paths),
        "stem": _wave_digest(stem),
        "stem_transport": _wave_digest(stem_transport),
        "sample_region": _digest(sample_region),
        "sample_downstream": _digest(sample_region_downstream),
    }


def matching_products(
    previous: dict[str, str] | None,
    current: dict[str, str],
) -> frozenset[str]:
    """Return products whose complete dependency signatures still match."""

    old = previous or {}
    return frozenset(
        name
        for name, signature in current.items()
        if name != "request" and old.get(name) == signature
    )
