"""Pure data models for comparing microscope designs and cached products.

The GUI owns the live microscope state and :class:`CalculationResult` objects.
This module deliberately stores neither: a design capture contains a recursively
immutable, JSON-like state payload, while a result summary contains only small
identity strings and product-presence flags.  A/B comparisons therefore cannot
pin ray histories, wave arrays, spectra, or detector frames in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from temsim.calculation_cache import (
    calculation_signatures,
    external_model_signature,
    state_model_signature,
)
from temsim.calculation_manifest import (
    ExternalInputIdentity,
    capture_external_input_identities,
    resolved_assembly_geometry_fingerprint,
)
from temsim.specimen.downstream_transport import (
    validated_geometric_specimen_exit,
)
from temsim.specimen.sample_region import validated_sample_region_exit


_SELECTION_FIELDS = ("gun", "column", "recording", "beam_blanker")


@dataclass(frozen=True, slots=True)
class HighAccuracyRequest:
    """Numerical controls that complete one High accuracy cache identity."""

    ray_count: int
    step_mm: float

    def __post_init__(self) -> None:
        if isinstance(self.ray_count, bool) or int(self.ray_count) <= 0:
            raise ValueError("High-accuracy ray count must be positive")
        step = float(self.step_mm)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("High-accuracy step must be finite and positive")
        object.__setattr__(self, "ray_count", int(self.ray_count))
        object.__setattr__(self, "step_mm", step)


@dataclass(frozen=True, slots=True)
class DesignSnapshot:
    """One independent, recursively immutable A/B design capture."""

    slot: str
    captured_at_utc: str
    selection: Mapping[str, str]
    state_payload: Mapping[str, object]
    request: HighAccuracyRequest
    model_signature: str
    request_signatures: Mapping[str, str]
    external_model_signature: str = ""
    geometry_fingerprint: str = ""
    external_inputs: tuple[ExternalInputIdentity, ...] = ()

    def __post_init__(self) -> None:
        slot = str(self.slot).strip().upper()
        if slot not in {"A", "B"}:
            raise ValueError("Design snapshot slot must be A or B")
        if not isinstance(self.request, HighAccuracyRequest):
            raise TypeError("Design snapshot request must be HighAccuracyRequest")
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "captured_at_utc", str(self.captured_at_utc))
        object.__setattr__(self, "selection", _freeze_mapping(self.selection))
        object.__setattr__(
            self, "state_payload", _freeze_mapping(self.state_payload)
        )
        object.__setattr__(self, "model_signature", str(self.model_signature))
        object.__setattr__(
            self,
            "external_model_signature",
            str(self.external_model_signature),
        )
        object.__setattr__(
            self,
            "geometry_fingerprint",
            str(self.geometry_fingerprint),
        )
        external_inputs = tuple(self.external_inputs)
        if any(
            not isinstance(row, ExternalInputIdentity)
            for row in external_inputs
        ):
            raise TypeError(
                "Design snapshot external inputs must be identity records"
            )
        object.__setattr__(self, "external_inputs", external_inputs)
        object.__setattr__(
            self,
            "request_signatures",
            _freeze_string_mapping(self.request_signatures),
        )


@dataclass(frozen=True, slots=True)
class EmptyContainer:
    """Stable leaf used when an empty mapping or sequence is flattened."""

    kind: str


EMPTY_MAPPING = EmptyContainer("mapping")
EMPTY_SEQUENCE = EmptyContainer("sequence")


@dataclass(frozen=True, slots=True)
class MissingValue:
    """Marker that distinguishes a missing path from a path containing None."""

    label: str = "<missing>"


MISSING_VALUE = MissingValue()


@dataclass(frozen=True, slots=True)
class FlatParameter:
    """One stable leaf in a normalized snapshot payload."""

    path: str
    value: object


@dataclass(frozen=True, slots=True)
class DesignDifference:
    """One changed A/B value, including additions and removals."""

    path: str
    value_a: object
    value_b: object
    kind: str = "parameter"


@dataclass(frozen=True, slots=True)
class ProductStage:
    """Public declaration of one dependency-scoped calculation product."""

    key: str
    label: str
    signature_key: str
    depends_on: tuple[str, ...] = ()
    result_product_keys: tuple[str, ...] = ()


# ``depends_on`` describes the physical/data flow for presentation only.  Cache
# validity is always decided from ``signature_key``; it must never be inherited
# from a parent status.  For example, downstream projector changes invalidate
# the full column while leaving the specimen-incident checkpoint reusable.
PRODUCT_STAGES: tuple[ProductStage, ...] = (
    ProductStage(
        "incident",
        "Incident beam at specimen",
        "incident",
        result_product_keys=("incident",),
    ),
    ProductStage(
        "column",
        "Full column trajectories",
        "column",
        depends_on=("incident",),
        result_product_keys=("column",),
    ),
    ProductStage(
        "diagnostics",
        "Optical diagnostics",
        "column",
        depends_on=("column",),
        result_product_keys=("diagnostics",),
    ),
    ProductStage(
        "elastic",
        "Elastic specimen transport",
        "elastic",
        depends_on=("incident",),
        result_product_keys=("elastic",),
    ),
    ProductStage(
        "wave_source",
        "Objective exit-wave checkpoint",
        "wave_source",
        depends_on=("incident",),
        result_product_keys=("wave_source",),
    ),
    ProductStage(
        "wave",
        "TEM recording-plane image",
        "wave",
        depends_on=("wave_source",),
        result_product_keys=("wave", "wave_projection"),
    ),
    ProductStage(
        "eds",
        "EDS spectrum",
        "eds",
        depends_on=("elastic",),
        result_product_keys=("eds",),
    ),
    ProductStage(
        "sample_region",
        "Sample-local 3D paths",
        "sample_region",
        depends_on=("elastic", "eds"),
        result_product_keys=("sample_region",),
    ),
    ProductStage(
        "sample_downstream",
        "Specimen-exit downstream transport",
        "sample_downstream",
        depends_on=("elastic", "column"),
        result_product_keys=("sample_downstream",),
    ),
    ProductStage(
        "scan_geometry",
        "Scan calibration geometry",
        "scan_geometry",
        result_product_keys=("scan_geometry",),
    ),
    ProductStage(
        "scan_ray_paths",
        "Ray-diagram scan playback",
        "scan_ray_paths",
        depends_on=("column", "scan_geometry"),
        result_product_keys=("scan_ray_paths",),
    ),
    ProductStage(
        "stem",
        "STEM detector frame",
        "stem",
        depends_on=("column", "scan_geometry", "elastic"),
        result_product_keys=("stem",),
    ),
    ProductStage(
        "fourdstem_cube",
        "4D-STEM diffraction cube",
        "fourdstem_cube",
        depends_on=("wave_source", "scan_geometry"),
        result_product_keys=("fourdstem_cube",),
    ),
    ProductStage(
        "fourdstem_virtual_detectors",
        "Virtual-detector integration",
        "fourdstem_virtual_detectors",
        depends_on=("fourdstem_cube",),
        result_product_keys=("fourdstem_virtual_detectors",),
    ),
    ProductStage(
        "fourdstem_physical_recording",
        "Physical record-plane re-integration",
        "fourdstem_physical_recording",
        depends_on=("fourdstem_cube", "column"),
        result_product_keys=("fourdstem_physical_recording",),
    ),
    ProductStage(
        "energy_filter",
        "Energy-filter transport",
        "energy_filter",
        depends_on=("column",),
        result_product_keys=("energy_filter",),
    ),
)


class ProductStatus(str, Enum):
    """User-facing reuse state for one declared product stage."""

    READY = "ready"
    REUSABLE = "reusable"
    RECALCULATE = "recalculate"
    OFF = "off"


class ArtifactState(str, Enum):
    """Canonical provenance/validity state of one result artifact."""

    CALCULATED = "calculated"
    REUSED = "reused"
    STALE = "stale"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ResultProductSummary:
    """Small immutable metadata extracted from a calculation result."""

    model_signature: str
    request_signature: str
    signatures: Mapping[str, str]
    available_stage_keys: frozenset[str]
    calculated_products: frozenset[str]
    reused_products: frozenset[str]
    terminal_stage_keys: frozenset[str] = frozenset()
    cache_hit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_signature", str(self.model_signature))
        object.__setattr__(
            self, "request_signature", str(self.request_signature)
        )
        object.__setattr__(
            self, "signatures", _freeze_string_mapping(self.signatures)
        )
        object.__setattr__(
            self,
            "available_stage_keys",
            frozenset(str(value) for value in self.available_stage_keys),
        )
        object.__setattr__(
            self,
            "calculated_products",
            frozenset(str(value) for value in self.calculated_products),
        )
        object.__setattr__(
            self,
            "reused_products",
            frozenset(str(value) for value in self.reused_products),
        )
        object.__setattr__(
            self,
            "terminal_stage_keys",
            frozenset(str(value) for value in self.terminal_stage_keys),
        )
        object.__setattr__(self, "cache_hit", bool(self.cache_hit))


@dataclass(frozen=True, slots=True)
class ProductStageStatus:
    """Evaluated current-state status for one product declaration."""

    stage: ProductStage
    artifact_state: ArtifactState
    reason: str
    source_request_signature: str = ""
    requested: bool = True
    current_request_exact: bool = False

    @property
    def status(self) -> ProductStatus:
        """Return the former coarse state for API compatibility."""

        if not self.requested:
            return ProductStatus.OFF
        if (
            self.artifact_state is ArtifactState.REUSED
            and self.current_request_exact
        ):
            return ProductStatus.READY
        return {
            ArtifactState.CALCULATED: ProductStatus.READY,
            ArtifactState.REUSED: ProductStatus.REUSABLE,
            ArtifactState.STALE: ProductStatus.RECALCULATE,
            ArtifactState.MISSING: ProductStatus.RECALCULATE,
        }[self.artifact_state]


def _freeze_string_mapping(values: Mapping[str, object]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("Expected a mapping")
    return MappingProxyType({
        str(key): str(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    })


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("Expected a mapping")
    frozen: dict[str, object] = {}
    for raw_key, value in sorted(values.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if key in frozen:
            raise ValueError(f"Mapping keys collide after normalization: {key!r}")
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    """Convert a JSON-like value to recursively immutable small objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        frozen = tuple(_freeze_value(item) for item in value)
        return tuple(sorted(frozen, key=repr))

    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy":
        # NumPy scalars are safe after conversion; arrays must never be pinned
        # by a design capture or result summary.
        ndim = getattr(value, "ndim", None)
        if ndim == 0 and hasattr(value, "item"):
            return _freeze_value(value.item())
        raise TypeError("Design snapshots cannot contain NumPy arrays")
    raise TypeError(
        "Design snapshots support only JSON-like values, not "
        f"{value_type.__name__}"
    )


def _plain_value(value: object) -> object:
    """Return a JSON-serializable copy of a recursively frozen value."""

    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _digest_payload(value: object) -> str:
    encoded = json.dumps(
        _plain_value(_freeze_value(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _selection_payload(selection: object) -> dict[str, str]:
    if isinstance(selection, Mapping):
        source = selection
        getter = lambda name, default=None: source.get(name, default)
    else:
        getter = lambda name, default=None: getattr(selection, name, default)
    payload = {
        name: str(getter(name, "None" if name == "beam_blanker" else ""))
        for name in _SELECTION_FIELDS
    }
    for name in _SELECTION_FIELDS[:3]:
        if not payload[name]:
            raise ValueError(f"Assembly selection is missing {name}")
    return payload


def _state_payload(state: object) -> Mapping[str, object]:
    if isinstance(state, Mapping):
        return state
    serializer = getattr(state, "to_dict", None)
    if not callable(serializer):
        raise TypeError("Microscope state must provide to_dict()")
    payload = serializer()
    if not isinstance(payload, Mapping):
        raise TypeError("Microscope state to_dict() must return a mapping")
    payload = dict(payload)
    payload["simulation_time_s"] = float(
        getattr(state, "simulation_time_s", 0.0)
    )
    return payload


def _high_accuracy_signature_state(
    state: object, request: HighAccuracyRequest
) -> object:
    """Build the same numerical request snapshot as CalculationController."""

    state_type = type(state)
    loader = getattr(state_type, "from_dict", None)
    if not callable(loader):
        return state
    payload = dict(_state_payload(state))
    payload.pop("simulation_time_s", None)
    snapshot = loader(payload)
    if hasattr(state, "simulation_time_s"):
        snapshot.simulation_time_s = float(state.simulation_time_s)
    gun = getattr(snapshot, "electron_gun", None)
    emitter = getattr(gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = int(request.ray_count)
    elif gun is not None:
        gun.ray_count = int(request.ray_count)
    if hasattr(snapshot, "step_mm"):
        snapshot.step_mm = float(request.step_mm)
    if hasattr(snapshot, "history_step_mm"):
        snapshot.history_step_mm = max(float(request.step_mm), 0.5)
    return snapshot


def capture_design_snapshot(
    state: object,
    selection: object,
    *,
    slot: str,
    request: HighAccuracyRequest,
    request_signatures: Mapping[str, str] | None = None,
    model_signature: str | None = None,
    external_signature: str | None = None,
    geometry_fingerprint: str | None = None,
    external_inputs: Sequence[ExternalInputIdentity] | None = None,
    captured_at_utc: str | None = None,
) -> DesignSnapshot:
    """Capture current settings without retaining the live state or results.

    ``request_signatures`` may be supplied by the calculation controller when
    it already prepared a request.  Otherwise they are calculated from an
    independent High accuracy state snapshot using the same ray-count and step
    overrides as the controller.
    """

    payload = _state_payload(state)
    if model_signature is None:
        try:
            model_signature = state_model_signature(state)
        except (AttributeError, TypeError):
            model_signature = _digest_payload(payload)
    if external_signature is None:
        if isinstance(state, Mapping):
            external_signature = ""
        else:
            try:
                external_signature = external_model_signature(state)
            except (AttributeError, TypeError):
                external_signature = ""
    if geometry_fingerprint is None:
        if isinstance(state, Mapping):
            geometry_fingerprint = ""
        else:
            try:
                geometry_fingerprint = (
                    resolved_assembly_geometry_fingerprint(state)
                )
            except (AttributeError, TypeError, ValueError):
                geometry_fingerprint = ""
    if external_inputs is None:
        if isinstance(state, Mapping):
            external_inputs = ()
        else:
            try:
                external_inputs = capture_external_input_identities(state)
            except (AttributeError, OSError, TypeError, ValueError):
                external_inputs = ()
    if request_signatures is None:
        try:
            signature_state = _high_accuracy_signature_state(state, request)
            request_signatures = calculation_signatures(signature_state)
        except (AttributeError, TypeError):
            base = _digest_payload(payload)
            request_key = _digest_payload({
                "state": payload,
                "ray_count": request.ray_count,
                "step_mm": request.step_mm,
            })
            request_signatures = {
                "request": request_key,
                **{
                    stage.signature_key: base
                    for stage in PRODUCT_STAGES
                },
            }
    timestamp = captured_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return DesignSnapshot(
        slot=slot,
        captured_at_utc=timestamp,
        selection=_selection_payload(selection),
        state_payload=payload,
        request=request,
        model_signature=str(model_signature),
        request_signatures=request_signatures,
        external_model_signature=str(external_signature),
        geometry_fingerprint=str(geometry_fingerprint),
        external_inputs=tuple(external_inputs),
    )


def _join_path(prefix: str, segment: str) -> str:
    return f"{prefix}.{segment}" if prefix else segment


def _keyed_sequence(values: Sequence[object]) -> dict[str, object] | None:
    if not values or not all(isinstance(item, Mapping) for item in values):
        return None
    keyed: dict[str, object] = {}
    for item in values:
        if "key" not in item:
            return None
        key = str(item["key"])
        if not key or key in keyed:
            return None
        keyed[key] = item
    return keyed


def flatten_payload(
    payload: Mapping[str, object], *, prefix: str = ""
) -> tuple[FlatParameter, ...]:
    """Flatten JSON-like data with component lists addressed by stable keys."""

    leaves: list[FlatParameter] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                leaves.append(FlatParameter(path, EMPTY_MAPPING))
                return
            for key in sorted(value, key=str):
                visit(value[key], _join_path(path, str(key)))
            return
        if isinstance(value, tuple):
            if not value:
                leaves.append(FlatParameter(path, EMPTY_SEQUENCE))
                return
            keyed = _keyed_sequence(value)
            if keyed is not None:
                for key in sorted(keyed):
                    visit(keyed[key], f"{path}[{key}]")
            else:
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
            return
        leaves.append(FlatParameter(path, value))

    visit(_freeze_mapping(payload), str(prefix))
    return tuple(sorted(leaves, key=lambda leaf: leaf.path))


def flatten_snapshot(snapshot: DesignSnapshot) -> tuple[FlatParameter, ...]:
    """Return stable leaves for assembly, physical state, and request inputs."""

    combined = {
        "assembly": snapshot.selection,
        "state": snapshot.state_payload,
        "calculation": {
            "high_accuracy_ray_count": snapshot.request.ray_count,
            "high_accuracy_step_mm": snapshot.request.step_mm,
        },
    }
    hidden = {
        "state.active_backend",
        "state.history_step_mm",
        "state.schema_version",
        "state.step_mm",
    }
    return tuple(
        leaf
        for leaf in flatten_payload(combined)
        if leaf.path not in hidden
        and not (
            leaf.path.startswith("state.electron_gun.")
            and leaf.path.endswith(".ray_count")
        )
    )


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return bool(left == right)


def diff_design_snapshots(
    snapshot_a: DesignSnapshot,
    snapshot_b: DesignSnapshot,
) -> tuple[DesignDifference, ...]:
    """Return deterministic changed-only rows for two independent captures."""

    values_a = {leaf.path: leaf.value for leaf in flatten_snapshot(snapshot_a)}
    values_b = {leaf.path: leaf.value for leaf in flatten_snapshot(snapshot_b)}
    differences = []
    for path in sorted(set(values_a) | set(values_b)):
        value_a = values_a.get(path, MISSING_VALUE)
        value_b = values_b.get(path, MISSING_VALUE)
        if not _values_equal(value_a, value_b):
            differences.append(DesignDifference(path, value_a, value_b))

    if (
        snapshot_a.external_model_signature
        != snapshot_b.external_model_signature
    ):
        differences.append(DesignDifference(
            "identity.external_model_toml_or_cif",
            snapshot_a.external_model_signature,
            snapshot_b.external_model_signature,
            kind="external_identity",
        ))
    if snapshot_a.geometry_fingerprint != snapshot_b.geometry_fingerprint:
        differences.append(DesignDifference(
            "identity.resolved_assembly_geometry",
            snapshot_a.geometry_fingerprint,
            snapshot_b.geometry_fingerprint,
            kind="external_identity",
        ))
    request_a = snapshot_a.request_signatures.get("request", "")
    request_b = snapshot_b.request_signatures.get("request", "")
    if request_a != request_b and not differences:
        differences.append(DesignDifference(
            "identity.calculation_request",
            request_a,
            request_b,
            kind="external_identity",
        ))
    return tuple(sorted(differences, key=lambda difference: difference.path))


def _present(value: object, attribute: str) -> bool:
    return value is not None and getattr(value, attribute, None) is not None


def summarise_calculation_result(result: object) -> ResultProductSummary:
    """Extract small immutable cache metadata without retaining ``result``."""

    raw_signatures = getattr(result, "signatures", None) or {}
    signatures = (
        dict(raw_signatures) if isinstance(raw_signatures, Mapping) else {}
    )
    available: set[str] = set()
    terminal: set[str] = set()
    simulation = getattr(result, "simulation", None)
    if simulation is not None:
        available.add("column")
        if getattr(simulation, "incident", None) is not None:
            available.add("incident")
        metrics = getattr(simulation, "metrics", None)
        if (
            isinstance(metrics, Mapping)
            and str(metrics.get("sample_illumination_status", ""))
            == "no incident current"
        ):
            terminal.update((
                "elastic",
                "wave_source",
                "wave",
                "eds",
                "sample_downstream",
                "stem",
                "fourdstem_cube",
                "fourdstem_virtual_detectors",
                "fourdstem_physical_recording",
            ))

    interactions = getattr(result, "specimen_interactions", None)
    if _present(interactions, "elastic_transport"):
        available.add("elastic")
    if _present(interactions, "eds_spectrum"):
        available.add("eds")

    wave = getattr(result, "wave_imaging", None)
    if wave is not None:
        available.add("wave")
        if getattr(wave, "projector_checkpoint", None) is not None:
            available.add("wave_source")
    sample_region = getattr(result, "sample_region", None)
    sample_region_signature = str(signatures.get("sample_region", ""))
    sample_region_metrics = dict(
        getattr(sample_region, "metrics", {}) or {}
    )
    sample_region_valid = bool(
        sample_region is not None
        and sample_region_signature
        and str(sample_region_metrics.get("sample_region_signature", ""))
        == sample_region_signature
    )
    if sample_region_valid:
        available.add("sample_region")
    downstream_signature = str(signatures.get("sample_downstream", ""))
    specimen_exit = validated_geometric_specimen_exit(
        getattr(result, "specimen_exit", None),
        downstream_signature,
    )
    if specimen_exit is None and sample_region is not None:
        specimen_exit = validated_sample_region_exit(
            sample_region,
            downstream_signature,
        )
    if specimen_exit is not None:
        available.add("sample_downstream")
    if getattr(result, "energy_filter", None) is not None:
        available.add("energy_filter")
    if getattr(result, "scan_geometry", None) is not None:
        available.add("scan_geometry")
    if getattr(result, "scan_ray_paths", None) is not None:
        available.add("scan_ray_paths")
    stem_scan = getattr(result, "stem_scan", None)
    fourdstem_cube_available = False
    virtual_product_available = False
    physical_product_available = False
    if stem_scan is not None:
        available.add("stem")
        artifact = getattr(stem_scan, "fourdstem_artifact", None)
        artifact_metadata = getattr(artifact, "metadata", {}) or {}
        provenance = (
            artifact_metadata.get("provenance", {})
            if isinstance(artifact_metadata, Mapping)
            else {}
        )
        stored_cube_signature = str(
            provenance.get("fourdstem_cube_state_signature", "")
            if isinstance(provenance, Mapping)
            else ""
        )
        result_cube_signature = str(signatures.get("fourdstem_cube", ""))
        fourdstem_cube_available = bool(
            artifact is not None
            and stored_cube_signature
            and result_cube_signature
            and stored_cube_signature == result_cube_signature
        )
        if fourdstem_cube_available:
            available.add("fourdstem_cube")
            stem_metrics = getattr(stem_scan, "metrics", {}) or {}
            if isinstance(stem_metrics, Mapping):
                virtual_signature = str(stem_metrics.get(
                    "fourdstem_virtual_detectors_signature", ""
                ))
                physical_signature = str(stem_metrics.get(
                    "fourdstem_physical_recording_signature", ""
                ))
                if virtual_signature:
                    signatures["fourdstem_virtual_detectors"] = (
                        virtual_signature
                    )
                    available.add("fourdstem_virtual_detectors")
                    virtual_product_available = True
                if physical_signature:
                    signatures["fourdstem_physical_recording"] = (
                        physical_signature
                    )
                    available.add("fourdstem_physical_recording")
                    physical_product_available = True
    if (
        hasattr(result, "lens_crossovers")
        and hasattr(result, "aperture_stops")
    ):
        available.add("diagnostics")

    calculated = {
        str(value)
        for value in (getattr(result, "calculated_products", ()) or ())
    }
    reused = {
        str(value)
        for value in (getattr(result, "reused_products", ()) or ())
    }
    if fourdstem_cube_available:
        (reused if "stem" in reused else calculated).add("fourdstem_cube")
    if virtual_product_available:
        calculated.add("fourdstem_virtual_detectors")
    if physical_product_available:
        calculated.add("fourdstem_physical_recording")
    return ResultProductSummary(
        model_signature=str(getattr(result, "model_signature", "")),
        request_signature=str(signatures.get("request", "")),
        signatures=signatures,
        available_stage_keys=frozenset(available),
        calculated_products=frozenset(calculated),
        reused_products=frozenset(reused),
        terminal_stage_keys=frozenset(terminal),
        cache_hit=bool(getattr(result, "cache_hit", False)),
    )


def evaluate_product_statuses(
    current_signatures: Mapping[str, str],
    summary: ResultProductSummary | None = None,
    *,
    requested_stage_keys: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> tuple[ProductStageStatus, ...]:
    """Evaluate ready/reusable/recalculation states for the current request.

    ``requested_stage_keys`` separates a disabled physical feature (``OFF``)
    from an enabled feature whose product is absent (``RECALCULATE``).  When it
    is omitted all declared stages are treated as requested.
    """

    if not isinstance(current_signatures, Mapping):
        raise TypeError("Current calculation signatures must be a mapping")
    requested = (
        {stage.key for stage in PRODUCT_STAGES}
        if requested_stage_keys is None
        else {str(key) for key in requested_stage_keys}
    )
    known = {stage.key for stage in PRODUCT_STAGES}
    unknown = requested - known
    if unknown:
        raise ValueError(
            "Unknown calculation product stages: " + ", ".join(sorted(unknown))
        )

    current_request = str(current_signatures.get("request", ""))
    exact_request = bool(
        summary is not None
        and current_request
        and current_request == summary.request_signature
    )
    statuses = []
    for stage in PRODUCT_STAGES:
        if stage.key not in requested:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.MISSING,
                "Not requested by the current microscope mode",
                requested=False,
            ))
            continue
        if summary is None:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.MISSING,
                "No cached product is available",
            ))
            continue

        source = summary.request_signature
        if exact_request and stage.key in summary.terminal_stage_keys:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.CALCULATED,
                "Completed with no incident current at the specimen",
                source,
                current_request_exact=True,
            ))
            continue
        old_signature = summary.signatures.get(stage.signature_key, "")
        new_signature = str(
            current_signatures.get(stage.signature_key, "")
        )
        if (
            not exact_request
            and stage.key in summary.terminal_stage_keys
            and old_signature
            and new_signature
            and old_signature == new_signature
        ):
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.REUSED,
                "The matching dependencies still produce no incident current",
                source,
            ))
            continue
        if stage.key not in summary.available_stage_keys:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.MISSING,
                "No cached product is available",
                source,
            ))
            continue

        if exact_request:
            product_keys = set(stage.result_product_keys)
            explicitly_calculated = bool(
                product_keys & summary.calculated_products
            )
            explicitly_reused = bool(
                product_keys & summary.reused_products
            )
            calculated_with_parent = bool(
                (
                    stage.key == "incident"
                    and "column" in summary.calculated_products
                )
                or (
                    stage.key == "wave_source"
                    and "wave" in summary.calculated_products
                )
            )
            if explicitly_calculated:
                artifact_state = ArtifactState.CALCULATED
                reason = "Calculated in the matching completed request"
            elif explicitly_reused:
                artifact_state = ArtifactState.REUSED
                reason = "Reused in the matching completed request"
            elif calculated_with_parent:
                artifact_state = ArtifactState.CALCULATED
                reason = "Calculated with its matching parent result"
            elif summary.cache_hit:
                artifact_state = ArtifactState.REUSED
                reason = "Returned from the complete result cache"
            else:
                artifact_state = ArtifactState.CALCULATED
                reason = "Present in the matching completed request"
            statuses.append(ProductStageStatus(
                stage,
                artifact_state,
                reason,
                source,
                current_request_exact=True,
            ))
            continue

        if old_signature and new_signature and old_signature == new_signature:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.REUSED,
                "Dependency signature matches the completed checkpoint",
                source,
            ))
        else:
            statuses.append(ProductStageStatus(
                stage,
                ArtifactState.STALE,
                "Dependencies changed since the cached result",
                source,
            ))
    return tuple(statuses)


__all__ = (
    "DesignDifference",
    "DesignSnapshot",
    "ArtifactState",
    "EMPTY_MAPPING",
    "EMPTY_SEQUENCE",
    "EmptyContainer",
    "FlatParameter",
    "HighAccuracyRequest",
    "MISSING_VALUE",
    "MissingValue",
    "PRODUCT_STAGES",
    "ProductStage",
    "ProductStageStatus",
    "ProductStatus",
    "ResultProductSummary",
    "capture_design_snapshot",
    "diff_design_snapshots",
    "evaluate_product_statuses",
    "flatten_payload",
    "flatten_snapshot",
    "summarise_calculation_result",
)
