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
_POST_SAMPLE_PROJECTION_LENS_KEYS = frozenset({
    "diffraction_lens",
    "intermediate_lens",
    "projector_lens_1",
    "projector_lens_2",
})


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
    return result


def _drop_physical_current_scale(
    payload: dict[str, object]
) -> dict[str, object]:
    """Remove dose-only controls from geometry/trajectory identities."""

    result = deepcopy(payload)
    result.pop("column_current_limit_percent", None)
    return result


def _drop_post_sample_projection_controls(
    payload: dict[str, object],
) -> dict[str, object]:
    """Remove controls downstream of the reusable specimen checkpoint.

    The D, I, P1 and P2 excitations and the image/diffraction selection affect
    propagation after the objective source planes.  They cannot change the
    incident bundle or any interaction that has already happened inside the
    specimen.
    """

    result = deepcopy(payload)
    result.pop("projector_mode", None)
    # FluScreen/Camera selection and detector sampling are downstream of the
    # specimen/Objective checkpoint.  Switching the physical recording target
    # must only reproject the saved complex wave.
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
    return result


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


def _cif_content_identity(payload: dict[str, object]) -> dict[str, object] | None:
    sample = dict(payload.get("sample", {}))
    if str(sample.get("specimen_mode", "")).strip().lower() != "atomic":
        return None
    raw_path = str(sample.get("cif_path", "")).strip()
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
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        payload["_resolved_assembly"] = _json_value(assembly)
    cif_identity = _cif_content_identity(payload)
    if cif_identity is not None:
        payload["_cif_content_identity"] = cif_identity
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


def calculation_signatures(state) -> dict[str, str]:
    """Return dependency-scoped identities for reusable calculation products."""

    full = _state_payload(state)
    no_region = _drop_sample_fields(
        full,
        prefixes=_SAMPLE_REGION_PREFIXES,
    )
    column = _drop_physical_current_scale(_drop_sample_fields(
        no_region,
        prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
    ))
    incident = _drop_post_sample_projection_controls(column)
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
                no_region,
                prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
                keep=elastic_keep,
            )
        )
    )
    wave = _drop_physical_current_scale(_drop_sample_fields(
        no_region,
        prefixes=(*_EDS_PREFIXES, *_STEM_PREFIXES),
        keep={"stem_wave_enabled"},
    ))
    wave_source = _drop_post_sample_projection_controls(wave)
    eds = _drop_post_sample_projection_controls(
        _drop_sample_fields(
            no_region,
            prefixes=("wave_", *_STEM_PREFIXES),
        )
    )
    scan = _drop_physical_current_scale(_drop_sample_fields(
        no_region,
        prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
    ))
    stem = _drop_sample_fields(
        no_region,
        prefixes=_EDS_PREFIXES,
    )
    sample_region = _drop_post_sample_projection_controls(
        _drop_physical_current_scale(
            _drop_sample_fields(
                full,
                prefixes=("wave_", *_STEM_PREFIXES),
            )
        )
    )
    energy_filter = _drop_physical_current_scale(_drop_sample_fields(
        no_region,
        prefixes=(*_EDS_PREFIXES, "wave_", *_STEM_PREFIXES),
    ))
    return {
        "request": _digest(full),
        "column": _digest(column),
        "incident": _digest(incident),
        "elastic": _digest(elastic),
        "wave": _digest(wave),
        "wave_source": _digest(wave_source),
        "eds": _digest(eds),
        "energy_filter": _digest(energy_filter),
        "scan": _digest(scan),
        "stem": _digest(stem),
        "sample_region": _digest(sample_region),
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
