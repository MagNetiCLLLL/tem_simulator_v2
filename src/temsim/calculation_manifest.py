"""Immutable provenance for one scientific calculation request.

The editable :class:`~temsim.optics.model.State` is intentionally mutable.
Workers and persistent caches must not infer their identity from that live
object after a calculation starts.  A :class:`CalculationManifest` freezes the
validated state, final resolved assembly geometry, external file content and
solver/schema identity at submission time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import platform

import numpy as np

from temsim import __version__
from temsim.calculation_cache import (
    calculation_signatures,
    calculation_signatures_for_request,
    external_model_signature,
    state_model_signature,
)
from temsim.immutable_json import freeze_json, json_digest, thaw_json


CALCULATION_MANIFEST_SCHEMA_VERSION = 1
SOLVER_IMPLEMENTATION_SCHEMA = "temsim-solver-2026-09-static-bh-v1"

_GEOMETRY_NAME_PARTS = (
    "anchor",
    "angle",
    "aperture",
    "axis",
    "bore",
    "centre",
    "center",
    "diameter",
    "distance",
    "gap",
    "height",
    "interface",
    "length",
    "material",
    "offset",
    "outer",
    "parent",
    "pole",
    "position",
    "profile",
    "radius",
    "rotation",
    "shape",
    "sigma",
    "start",
    "stop",
    "style",
    "taper",
    "thickness",
    "vacuum",
    "width",
    "z_mm",
)
_OPERATING_GEOMETRY_EXCLUSIONS = frozenset({
    "default_excitation_percent",
    "excitation_percent",
    "nominal_excitation_percent",
    "percent",
})


def _normalise_geometry_numbers(value: object) -> object:
    """Remove arithmetic round-off without hiding physical geometry edits.

    Resolved module offsets can produce values such as ``3025.9`` and
    ``3025.8999999999996`` for the same plane.  Fifteen significant decimal
    digits collapse that binary-arithmetic residue while retaining geometry
    changes many orders below the simulator's nanometre-scale controls.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _normalise_geometry_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_normalise_geometry_numbers(item) for item in value)
    if isinstance(value, list):
        return [_normalise_geometry_numbers(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            return float(format(numeric, ".15g"))
    return value


def _is_geometry_key(name: object) -> bool:
    key = str(name).strip().lower()
    if key in _OPERATING_GEOMETRY_EXCLUSIONS or key.endswith("_percent"):
        return False
    return key in {"key", "branch", "type", "installed"} or any(
        token in key for token in _GEOMETRY_NAME_PARTS
    )


def _geometry_mapping(values: object) -> dict[str, object]:
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): value
        for key, value in values.items()
        if _is_geometry_key(key)
    }


def _assembly_scope_context(
    assembly: object,
) -> tuple[str | None, float, float]:
    """Return the sample module and safe axial boundaries for cache scopes."""

    parts = tuple(getattr(assembly, "parts", ()))
    sample_part = next(
        (
            part
            for part in parts
            if str(getattr(part, "key", "")) == "sample"
        ),
        None,
    )
    if sample_part is None:
        return None, float("-inf"), float("inf")
    module_key = str(getattr(sample_part, "module_key", ""))
    module_parts = tuple(
        part
        for part in parts
        if str(getattr(part, "module_key", "")) == module_key
    )
    upstream_end = max(
        (
            float(getattr(part, "end_z_mm"))
            for part in module_parts
        ),
        default=float(getattr(sample_part, "end_z_mm")),
    )
    return (
        module_key,
        float(getattr(sample_part, "center_z_mm")),
        upstream_end,
    )


def _resolved_assembly_geometry_payload(
    assembly: object,
    *,
    scope: str = "complete",
) -> dict[str, object]:
    if assembly is None:
        return {}
    if scope not in {"complete", "upstream", "post_sample"}:
        raise ValueError(f"Unknown assembly geometry scope: {scope}")
    sample_module_key, sample_z_mm, upstream_end_z_mm = (
        _assembly_scope_context(assembly)
    )
    all_parts = tuple(getattr(assembly, "parts", ()))

    def include_part(part: object) -> bool:
        if scope == "complete" or sample_module_key is None:
            return True
        if scope == "upstream":
            return float(getattr(part, "end_z_mm")) <= (
                upstream_end_z_mm + 1.0e-9
            )
        return float(getattr(part, "end_z_mm")) >= (
            sample_z_mm - 1.0e-9
        )

    scoped_parts = tuple(part for part in all_parts if include_part(part))
    included_module_keys = {
        str(getattr(part, "module_key", "")) for part in scoped_parts
    }
    modules = []
    for module in getattr(assembly, "modules", ()):
        if (
            scope != "complete"
            and sample_module_key is not None
            and str(getattr(module, "key", ""))
            not in included_module_keys
        ):
            continue
        modules.append({
            "type": str(getattr(module, "type", "")),
            "key": str(getattr(module, "key", "")),
            "entrance_interface": str(
                getattr(module, "entrance_interface", "")
            ),
            "entrance_z_mm": float(getattr(module, "entrance_z_mm")),
            "exit_interface": str(getattr(module, "exit_interface", "")),
            "exit_z_mm": float(getattr(module, "exit_z_mm")),
            "length_mm": float(getattr(module, "length_mm")),
            "geometry": _geometry_mapping(
                getattr(module, "geometry", {})
            ),
        })
    parts = []
    for part in scoped_parts:
        parts.append({
            "module_key": str(getattr(part, "module_key", "")),
            "key": str(getattr(part, "key", "")),
            "branch": str(getattr(part, "branch", "")),
            "parent_key": getattr(part, "parent_key", None),
            "start_z_mm": float(getattr(part, "start_z_mm")),
            "center_z_mm": float(getattr(part, "center_z_mm")),
            "end_z_mm": float(getattr(part, "end_z_mm")),
            "length_mm": float(getattr(part, "length_mm")),
            "geometry": _geometry_mapping(getattr(part, "data", {})),
        })
    def include_segment(segment: object) -> bool:
        if scope == "complete" or sample_module_key is None:
            return True
        if scope == "upstream":
            return float(getattr(segment, "end_z_mm")) <= (
                upstream_end_z_mm + 1.0e-9
            )
        return float(getattr(segment, "end_z_mm")) >= (
            sample_z_mm - 1.0e-9
        )

    vacuum_bore = [
        {
            "key": str(getattr(segment, "key", "")),
            "start_z_mm": float(getattr(segment, "start_z_mm")),
            "end_z_mm": float(getattr(segment, "end_z_mm")),
            "inner_diameter_mm": float(
                getattr(segment, "inner_diameter_mm")
            ),
        }
        for segment in getattr(assembly, "vacuum_bore_segments", ())
        if include_segment(segment)
    ]
    vacuum_liner = [
        {
            "key": str(getattr(segment, "key", "")),
            "start_z_mm": float(getattr(segment, "start_z_mm")),
            "end_z_mm": float(getattr(segment, "end_z_mm")),
            "inner_diameter_mm": float(
                getattr(segment, "inner_diameter_mm")
            ),
            "outer_diameter_mm": float(
                getattr(segment, "outer_diameter_mm")
            ),
            "wall_thickness_mm": float(
                getattr(segment, "wall_thickness_mm")
            ),
        }
        for segment in getattr(assembly, "vacuum_liner_segments", ())
        if include_segment(segment)
    ]
    return {
        "modules": modules,
        "parts": parts,
        "vacuum_bore_segments": vacuum_bore,
        "vacuum_liner_segments": vacuum_liner,
        "scope": scope,
        "exit_z_mm": (
            float(getattr(assembly, "exit_z_mm", 0.0))
            if scope == "complete"
            else (
                upstream_end_z_mm
                if scope == "upstream"
                else float(getattr(assembly, "exit_z_mm", 0.0))
            )
        ),
    }


def _live_component_geometry(
    state: object,
    *,
    scope: str = "complete",
) -> tuple[dict[str, object], ...]:
    rows: dict[str, dict[str, object]] = {}
    collections = (
        "lenses",
        "apertures",
        "stigmators",
        "deflectors",
        "corrector_elements",
        "recording_planes",
        "stem_detectors",
    )
    scalar_names = (
        "z_mm",
        "upper_z_mm",
        "lower_z_mm",
        "start_z_mm",
        "end_z_mm",
        "length_mm",
        "effective_length_mm",
        "active_length_mm",
        "thickness_mm",
        "a_mm",
        "radius_mm",
        "diameter_mm",
        "inner_diameter_mm",
        "outer_diameter_mm",
        "mechanical_outer_diameter_mm",
        "mechanical_bore_diameter_mm",
        "mechanical_clear_bore_diameter_mm",
        "pole_gap_mm",
        "pole_piece_tip_diameter_mm",
        "pole_piece_bore_diameter_mm",
        "pole_piece_center_separation_mm",
        "installed",
        "b0_t",
        "max_percent",
    )
    for collection_name in collections:
        for component in getattr(state, collection_name, ()) or ():
            key = str(getattr(component, "key", "")).strip()
            if not key:
                continue
            row = rows.setdefault(key, {"key": key})
            for name in scalar_names:
                if hasattr(component, name):
                    value = getattr(component, name)
                    if value is not None:
                        row[name] = value
            gaussian = getattr(component, "gaussian", None)
            if gaussian:
                row["field_profile_geometry"] = [
                    {
                        "offset": float(getattr(item, "offset")),
                        "sigma": float(getattr(item, "sigma")),
                    }
                    for item in gaussian
                ]
    sample = getattr(state, "sample", None)
    if sample is not None:
        rows["sample"] = {
            "key": "sample",
            "z_mm": float(getattr(sample, "z_mm")),
        }
    if scope == "complete":
        return tuple(rows[key] for key in sorted(rows))
    assembly = getattr(state, "_resolved_assembly", None)
    _, sample_z_mm, upstream_end_z_mm = _assembly_scope_context(assembly)

    def include_row(row: Mapping[str, object]) -> bool:
        z_values = tuple(
            float(row[name])
            for name in (
                "z_mm",
                "upper_z_mm",
                "lower_z_mm",
                "start_z_mm",
                "end_z_mm",
            )
            if name in row
        )
        if not z_values:
            return True
        if scope == "upstream":
            return min(z_values) <= upstream_end_z_mm + 1.0e-9
        return max(z_values) >= sample_z_mm - 1.0e-9

    return tuple(
        rows[key]
        for key in sorted(rows)
        if include_row(rows[key])
    )


def resolved_assembly_geometry_fingerprints(
    state_or_assembly: object,
) -> dict[str, str]:
    """Return complete and stage-scoped final-geometry identities.

    ``upstream`` includes the complete resolved module that owns the sample,
    so Objective-pole and local-field geometry remain dependencies while the
    separately repeatable D/I/P module does not. ``post_sample`` identifies
    geometry from the specimen plane downstream. The complete identity is the
    authoritative manifest/recipe fingerprint.
    """

    assembly = getattr(state_or_assembly, "_resolved_assembly", None)
    state = state_or_assembly if assembly is not None else None
    if assembly is None and hasattr(state_or_assembly, "parts"):
        assembly = state_or_assembly
    if assembly is None:
        raise ValueError("No resolved assembly geometry is available")
    result: dict[str, str] = {}
    for scope in ("upstream", "post_sample", "complete"):
        payload: dict[str, object] = {
            "resolved_assembly": _resolved_assembly_geometry_payload(
                assembly,
                scope=scope,
            ),
        }
        if state is not None:
            payload["live_components"] = _live_component_geometry(
                state,
                scope=scope,
            )
            if scope in {"upstream", "complete"}:
                payload["resolved_anchors"] = {
                    "upper_objective_package": dict(getattr(
                        state,
                        "_upper_objective_package_resolved_positions_mm",
                        {},
                    )),
                    "ac_downstream": dict(getattr(
                        state,
                        "_ac_downstream_resolved_positions_mm",
                        {},
                    )),
                }
        result[scope] = json_digest(_normalise_geometry_numbers(payload))
    return result


def resolved_assembly_geometry_fingerprint(state_or_assembly: object) -> str:
    """Identify final installed geometry without hashing lens excitation.

    The payload contains resolved part anchors, pole-piece dimensions, vacuum
    bores and final live component positions.  Lens ``percent`` is excluded so
    a strength adjustment does not make a geometrically valid operating preset
    look as though it belongs to different hardware.
    """

    return resolved_assembly_geometry_fingerprints(state_or_assembly)[
        "complete"
    ]


@dataclass(frozen=True, slots=True)
class SolverIdentity:
    package_version: str
    state_schema_version: int
    manifest_schema_version: int = CALCULATION_MANIFEST_SCHEMA_VERSION
    implementation_schema: str = SOLVER_IMPLEMENTATION_SCHEMA
    python_version: str = platform.python_version()
    numpy_version: str = np.__version__

    @property
    def digest(self) -> str:
        return json_digest({
            "package_version": self.package_version,
            "state_schema_version": self.state_schema_version,
            "manifest_schema_version": self.manifest_schema_version,
            "implementation_schema": self.implementation_schema,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
        })


@dataclass(frozen=True, slots=True)
class ExternalInputIdentity:
    role: str
    path: str
    available: bool
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.available:
            if self.size_bytes < 0 or len(self.sha256) != 64:
                raise ValueError("Available external inputs need size and SHA-256")
        elif self.size_bytes != 0 or self.sha256:
            raise ValueError("Unavailable inputs cannot claim file content")


def _file_identity(role: str, raw_path: object) -> ExternalInputIdentity:
    path = Path(str(raw_path)).expanduser()
    try:
        data = path.read_bytes()
        resolved = path.resolve()
    except OSError:
        return ExternalInputIdentity(str(role), str(path), False, 0, "")
    return ExternalInputIdentity(
        str(role),
        str(resolved),
        True,
        len(data),
        sha256(data).hexdigest(),
    )


def _external_inputs(state: object) -> tuple[ExternalInputIdentity, ...]:
    rows: dict[tuple[str, str], ExternalInputIdentity] = {}
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        root = Path(getattr(assembly, "root", "."))
        for module_type, relative in getattr(
            assembly, "selected_module_paths", ()
        ):
            row = _file_identity(f"assembly:{module_type}", root / relative)
            rows[(row.role, row.path)] = row
        catalog = _file_identity("assembly:catalog", root / "catalog.toml")
        rows[(catalog.role, catalog.path)] = catalog
    sample = getattr(state, "sample", None)
    if sample is not None:
        mode = str(getattr(sample, "specimen_mode", "")).lower()
        if mode == "atomic" and str(getattr(sample, "cif_path", "")).strip():
            row = _file_identity("specimen:cif", sample.cif_path)
            rows[(row.role, row.path)] = row
        if mode == "virtual":
            for index, region in enumerate(
                getattr(sample, "virtual_regions", ()) or ()
            ):
                if not isinstance(region, Mapping):
                    continue
                if str(region.get("kind", "")).lower() != "map":
                    continue
                raw_path = str(region.get("map_path", "")).strip()
                if raw_path:
                    row = _file_identity(
                        f"specimen:region_map:{index}", raw_path
                    )
                    rows[(row.role, row.path)] = row
    descriptors = getattr(state, "lens_field_map_descriptors", {})
    if isinstance(descriptors, Mapping):
        for lens_key, descriptor in sorted(
            descriptors.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(descriptor, Mapping):
                continue
            raw_path = str(descriptor.get("source_path", "")).strip()
            if raw_path:
                row = _file_identity(
                    f"lens_field_map:{lens_key}", raw_path
                )
                rows[(row.role, row.path)] = row
    return tuple(rows[key] for key in sorted(rows))


def capture_external_input_identities(
    state: object,
) -> tuple[ExternalInputIdentity, ...]:
    """Freeze the path and content identity of every external model input.

    This deliberately exposes the same inventory used by calculation
    manifests so design recipes cannot develop a second, subtly different
    definition of CIF, assembly-TOML, region-map, or lens-field-map identity.
    """

    return _external_inputs(state)


def changed_external_input_identities(
    expected_inputs: Sequence[ExternalInputIdentity],
) -> tuple[ExternalInputIdentity, ...]:
    """Return captured external inputs whose path availability/bytes drifted."""

    changed = []
    for expected in tuple(expected_inputs):
        if not isinstance(expected, ExternalInputIdentity):
            raise TypeError(
                "Expected external inputs must be ExternalInputIdentity rows"
            )
        actual = _file_identity(expected.role, expected.path)
        if actual != expected:
            changed.append(actual)
    return tuple(changed)


def assert_external_input_identities_unchanged(
    expected_inputs: Sequence[ExternalInputIdentity],
) -> None:
    """Reject an operation when any previously captured input has drifted."""

    changed = changed_external_input_identities(expected_inputs)
    if changed:
        roles = ", ".join(sorted({row.role for row in changed}))
        raise RuntimeError(
            "Captured external inputs changed: " + roles
        )


def _selection_payload(selection: object | None) -> dict[str, str]:
    if selection is None:
        return {}
    names = ("gun", "column", "recording", "beam_blanker")
    if isinstance(selection, Mapping):
        return {
            name: str(selection.get(name, "None" if name == "beam_blanker" else ""))
            for name in names
        }
    return {
        name: str(getattr(
            selection, name, "None" if name == "beam_blanker" else ""
        ))
        for name in names
    }


@dataclass(frozen=True, slots=True)
class CalculationManifest:
    created_at_utc: str
    state_payload: Mapping[str, object]
    selection: Mapping[str, str]
    calculation_signatures: Mapping[str, str]
    state_model_signature: str
    external_model_signature: str
    geometry_fingerprint: str
    solver: SolverIdentity
    external_inputs: tuple[ExternalInputIdentity, ...]
    schema_version: int = CALCULATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != CALCULATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported calculation-manifest schema")
        object.__setattr__(self, "state_payload", freeze_json(self.state_payload))
        object.__setattr__(self, "selection", freeze_json(self.selection))
        object.__setattr__(
            self,
            "calculation_signatures",
            freeze_json(self.calculation_signatures),
        )
        object.__setattr__(self, "external_inputs", tuple(self.external_inputs))

    @property
    def identity_payload(self) -> dict[str, object]:
        """Return deterministic scientific identity, excluding wall-clock time."""

        return {
            "schema_version": self.schema_version,
            "state_payload": self.state_payload,
            "selection": self.selection,
            "calculation_signatures": self.calculation_signatures,
            "state_model_signature": self.state_model_signature,
            "external_model_signature": self.external_model_signature,
            "geometry_fingerprint": self.geometry_fingerprint,
            "solver_digest": self.solver.digest,
            "external_inputs": self.external_inputs,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.identity_payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at_utc": self.created_at_utc,
            **thaw_json(self.identity_payload),
            "solver": thaw_json(freeze_json(self.solver)),
            "digest": self.digest,
        }


def capture_calculation_manifest(
    state: object,
    *,
    selection: object | None = None,
    ray_count: int | None = None,
    step_mm: float | None = None,
    created_at_utc: str | None = None,
) -> CalculationManifest:
    """Freeze the exact mutable inputs at calculation submission time."""

    serializer = getattr(state, "to_dict", None)
    if not callable(serializer):
        raise TypeError("Calculation state must provide to_dict()")
    state_payload = dict(serializer())
    state_payload["simulation_time_s"] = float(
        getattr(state, "simulation_time_s", 0.0)
    )
    if (ray_count is None) != (step_mm is None):
        raise ValueError("ray_count and step_mm must be supplied together")
    if ray_count is None:
        signatures = calculation_signatures(state)
    else:
        signatures = calculation_signatures_for_request(
            state,
            ray_count=int(ray_count),
            step_mm=float(step_mm),
        )
    solver = SolverIdentity(
        package_version=str(__version__),
        state_schema_version=int(getattr(state, "schema_version", 0)),
    )
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return CalculationManifest(
        created_at_utc=timestamp,
        state_payload=state_payload,
        selection=_selection_payload(selection),
        calculation_signatures=signatures,
        state_model_signature=state_model_signature(state),
        external_model_signature=external_model_signature(state),
        geometry_fingerprint=resolved_assembly_geometry_fingerprint(state),
        solver=solver,
        external_inputs=_external_inputs(state),
    )


def changed_external_inputs(
    manifest: CalculationManifest,
) -> tuple[ExternalInputIdentity, ...]:
    """Return files whose current bytes no longer match the frozen manifest."""

    return changed_external_input_identities(manifest.external_inputs)


def assert_external_inputs_unchanged(manifest: CalculationManifest) -> None:
    """Fail before publishing an artifact if an input changed during work."""

    changed = changed_external_inputs(manifest)
    if changed:
        roles = ", ".join(sorted({row.role for row in changed}))
        raise RuntimeError(
            "Calculation inputs changed after submission: " + roles
        )


__all__ = (
    "CALCULATION_MANIFEST_SCHEMA_VERSION",
    "SOLVER_IMPLEMENTATION_SCHEMA",
    "CalculationManifest",
    "ExternalInputIdentity",
    "SolverIdentity",
    "assert_external_input_identities_unchanged",
    "assert_external_inputs_unchanged",
    "capture_external_input_identities",
    "capture_calculation_manifest",
    "changed_external_input_identities",
    "changed_external_inputs",
    "resolved_assembly_geometry_fingerprint",
    "resolved_assembly_geometry_fingerprints",
)
