"""Signed mixed-plane propagation to physical recording stops.

The specimen-exit phase-space convention is ``(x, y, theta_x, theta_y)``
in SI units, with electrons travelling downstream along +Z.  Every physical
plane is evaluated from the same specimen reference using the complete signed
first-order map::

    r_plane = J_img @ r_sample + J_diff @ theta_sample

This module deliberately reads the resolved runtime state.  It does not use
bootstrap/preset detector coordinates.  A plan fingerprint includes the
resolved assembly geometry, immutable stop snapshots, and the actual traced
transfer matrices so that a cached result cannot survive a pole-piece,
component-position, or assembly edit unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal

import numpy as np

from temsim.physics.first_order import (
    TransverseTransfer,
    trace_transverse_transfers,
)


PlaneKind = Literal["aperture", "detector"]


def _readonly(values, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _normalise_json(value):
    """Convert resolved dataclasses/mappings/arrays to stable JSON values."""

    if isinstance(value, np.ndarray):
        return [_normalise_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            field.name: _normalise_json(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Paths and enums have a stable textual representation.  Avoid object
    # repr, which may contain a process-specific memory address.
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    return str(value)


def _digest(payload) -> str:
    encoded = json.dumps(
        _normalise_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PlaneStop:
    """Immutable physical aperture/detector geometry at one runtime Z plane."""

    key: str
    name: str
    z_mm: float
    kind: PlaneKind
    geometry: str
    radius_mm: float = 0.0
    outer_width_mm: float = 0.0
    inner_diameter_mm: float = 0.0
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    readout_enabled: bool = False
    non_blocking: bool = False
    authority: str = "runtime_state"

    def __post_init__(self) -> None:
        if not str(self.key).strip():
            raise ValueError("Plane-stop key must not be empty")
        if self.kind not in {"aperture", "detector"}:
            raise ValueError("Plane-stop kind must be aperture or detector")
        for label in (
            "z_mm",
            "radius_mm",
            "outer_width_mm",
            "inner_diameter_mm",
            "offset_x_mm",
            "offset_y_mm",
        ):
            _finite(getattr(self, label), f"{self.key} {label}")
        if self.kind == "aperture":
            if self.radius_mm < 0.0:
                raise ValueError(f"{self.key}: aperture radius cannot be negative")
        else:
            geometry = str(self.geometry).lower()
            if geometry not in {"disk", "annulus", "square", "rectangle", "camera"}:
                raise ValueError(f"{self.key}: unsupported detector geometry {geometry!r}")
            if self.outer_width_mm <= 0.0:
                raise ValueError(f"{self.key}: detector width must be positive")
            if self.inner_diameter_mm < 0.0:
                raise ValueError(f"{self.key}: detector inner diameter cannot be negative")
            if geometry == "annulus" and self.inner_diameter_mm >= self.outer_width_mm:
                raise ValueError(f"{self.key}: detector annulus has no active area")
        if not str(self.authority).strip():
            raise ValueError("Plane-stop authority must not be empty")

    def transmission_mask(self, x_m, y_m) -> np.ndarray:
        """Return the open aperture region for coordinates expressed in metres."""

        if self.kind != "aperture":
            raise TypeError("Only an aperture has a transmission mask")
        x_mm = np.asarray(x_m, dtype=float) * 1.0e3 - self.offset_x_mm
        y_mm = np.asarray(y_m, dtype=float) * 1.0e3 - self.offset_y_mm
        if self.radius_mm == 0:
            return np.zeros(np.broadcast_shapes(x_mm.shape, y_mm.shape), dtype=bool)
        return np.hypot(x_mm, y_mm) <= self.radius_mm

    def hit_mask(self, x_m, y_m) -> np.ndarray:
        """Return the active detector region for coordinates expressed in metres."""

        if self.kind != "detector":
            raise TypeError("Only a detector has a hit mask")
        x_mm = np.asarray(x_m, dtype=float) * 1.0e3 - self.offset_x_mm
        y_mm = np.asarray(y_m, dtype=float) * 1.0e3 - self.offset_y_mm
        geometry = str(self.geometry).lower()
        half_width = 0.5 * self.outer_width_mm
        if geometry in {"square", "rectangle", "camera"}:
            return (np.abs(x_mm) <= half_width) & (np.abs(y_mm) <= half_width)
        radius = np.hypot(x_mm, y_mm)
        if geometry == "annulus":
            return (
                (radius >= 0.5 * self.inner_diameter_mm)
                & (radius <= half_width)
            )
        return radius <= half_width


@dataclass(frozen=True, slots=True)
class ProjectedPhaseSpace:
    """Position in metres and paraxial angle in radians at a target plane."""

    position_m: np.ndarray
    angle_rad: np.ndarray

    def __post_init__(self) -> None:
        position = _readonly(self.position_m, dtype=float)
        angle = _readonly(self.angle_rad, dtype=float)
        if position.shape != angle.shape or position.shape[-1:] != (2,):
            raise ValueError("Projected position and angle must share shape (..., 2)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(angle)):
            raise ValueError("Projected phase-space coordinates must be finite")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "angle_rad", angle)


def project_sample_phase_space(
    transfer: TransverseTransfer,
    sample_position_m,
    sample_angle_rad,
) -> ProjectedPhaseSpace:
    """Apply the full signed 4-D map to broadcast-compatible sample arrays."""

    position = np.asarray(sample_position_m, dtype=float)
    angle = np.asarray(sample_angle_rad, dtype=float)
    if position.shape[-1:] != (2,) or angle.shape[-1:] != (2,):
        raise ValueError("Sample positions and angles must have trailing dimension 2")
    try:
        leading = np.broadcast_shapes(position.shape[:-1], angle.shape[:-1])
    except ValueError as error:
        raise ValueError("Sample positions and angles are not broadcast-compatible") from error
    position = np.broadcast_to(position, leading + (2,))
    angle = np.broadcast_to(angle, leading + (2,))
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(angle)):
        raise ValueError("Sample phase-space coordinates must be finite")
    projected_position = (
        np.einsum("ij,...j->...i", transfer.j_img, position)
        + np.einsum("ij,...j->...i", transfer.j_diff_m_per_rad, angle)
        + np.asarray(transfer.position_offset_m)
    )
    projected_angle = (
        np.einsum("ij,...j->...i", transfer.k_img_rad_per_m, position)
        + np.einsum("ij,...j->...i", transfer.k_diff, angle)
        + np.asarray(transfer.angle_offset_rad)
    )
    return ProjectedPhaseSpace(projected_position, projected_angle)


@dataclass(frozen=True, slots=True)
class RecordPlanePlan:
    """Frozen runtime geometry and traced maps used by one routing calculation."""

    source_z_mm: float
    planes: tuple[PlaneStop, ...]
    transfers: tuple[TransverseTransfer, ...]
    resolved_geometry_fingerprint: str
    fingerprint: str
    scan_times_s: np.ndarray | None = None
    scan_position_offsets_m: tuple[np.ndarray, ...] = ()
    time_dependent_deflection: bool = False

    def __post_init__(self) -> None:
        _finite(self.source_z_mm, "Record-plane source Z")
        if len(self.planes) != len(self.transfers):
            raise ValueError("Every record plane must have one transfer")
        if len({plane.key for plane in self.planes}) != len(self.planes):
            raise ValueError("Record-plane keys must be unique")
        previous = -math.inf
        for plane, transfer in zip(self.planes, self.transfers):
            if plane.z_mm < previous:
                raise ValueError("Record planes must be sorted downstream")
            if not math.isclose(transfer.source_z_mm, self.source_z_mm, abs_tol=1.0e-9):
                raise ValueError("Record-plane transfer has the wrong source Z")
            if not math.isclose(transfer.target_z_mm, plane.z_mm, abs_tol=1.0e-9):
                raise ValueError("Record-plane transfer has the wrong target Z")
            previous = plane.z_mm
        for label in (self.resolved_geometry_fingerprint, self.fingerprint):
            if len(label) != 64:
                raise ValueError("Record-plane fingerprints must be SHA-256 hex digests")
        if self.scan_times_s is None:
            if self.scan_position_offsets_m:
                raise ValueError("Scan-dependent recording offsets require scan times")
        else:
            times = _readonly(self.scan_times_s, dtype=float)
            if times.ndim != 2 or not times.size or not np.all(np.isfinite(times)):
                raise ValueError("Record-plane scan times must be a finite non-empty 2-D array")
            offsets = tuple(_readonly(value, dtype=float) for value in self.scan_position_offsets_m)
            if (offsets and len(offsets) != len(self.planes)) or any(
                value.shape != times.shape + (2,) or not np.all(np.isfinite(value))
                for value in offsets
            ):
                raise ValueError("Every recording plane requires finite offsets matching the scan")
            object.__setattr__(self, "scan_times_s", times)
            object.__setattr__(self, "scan_position_offsets_m", offsets)


@dataclass(frozen=True, slots=True)
class PlaneInteraction:
    """Sequential routing event for one physical stop."""

    plane: PlaneStop
    projected_position_m: np.ndarray
    incident_mask: np.ndarray
    intercepted_mask: np.ndarray
    signal_mask: np.ndarray
    outgoing_mask: np.ndarray
    incident_weight: float
    intercepted_weight: float
    signal_weight: float
    outgoing_weight: float

    def __post_init__(self) -> None:
        position = _readonly(self.projected_position_m, dtype=float)
        masks = []
        for value in (
            self.incident_mask,
            self.intercepted_mask,
            self.signal_mask,
            self.outgoing_mask,
        ):
            masks.append(_readonly(value, dtype=bool))
        shape = position.shape[:-1]
        if position.shape[-1:] != (2,) or any(mask.shape != shape for mask in masks):
            raise ValueError("Plane-interaction arrays do not share one event shape")
        object.__setattr__(self, "projected_position_m", position)
        for name, mask in zip(
            ("incident_mask", "intercepted_mask", "signal_mask", "outgoing_mask"),
            masks,
        ):
            object.__setattr__(self, name, mask)


@dataclass(frozen=True, slots=True)
class RecordPlaneResult:
    """Conservative physical routing result with detector readout weights."""

    plan_fingerprint: str
    event_shape: tuple[int, ...]
    interactions: tuple[PlaneInteraction, ...]
    surviving_mask: np.ndarray
    initial_weight: float
    physically_intercepted_weight: float
    surviving_weight: float
    balance_error: float

    def __post_init__(self) -> None:
        surviving = _readonly(self.surviving_mask, dtype=bool)
        if surviving.shape != self.event_shape:
            raise ValueError("Surviving mask has the wrong event shape")
        if len(self.plan_fingerprint) != 64:
            raise ValueError("Result plan fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "surviving_mask", surviving)

    @property
    def signal_weights(self) -> Mapping[str, float]:
        return MappingProxyType({
            event.plane.key: float(event.signal_weight)
            for event in self.interactions
            if event.plane.kind == "detector"
        })


def _authority_for(state, key: str) -> str:
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        try:
            return str(assembly.part(key).definition_id)
        except (AttributeError, KeyError):
            pass
    return "runtime_state"


def _aperture_stop(state, component) -> PlaneStop:
    radius = getattr(component, "effective_aperture_radius_mm", None)
    if radius is None:
        radius = getattr(component, "radius_mm")
    return PlaneStop(
        key=str(component.key),
        name=str(getattr(component, "name", component.key)),
        z_mm=float(component.z_mm),
        kind="aperture",
        geometry="disk",
        radius_mm=float(radius),
        offset_x_mm=float(getattr(component, "offset_x_mm", 0.0)),
        offset_y_mm=float(getattr(component, "offset_y_mm", 0.0)),
        authority=_authority_for(state, str(component.key)),
    )


def _detector_stop(state, component) -> PlaneStop:
    return PlaneStop(
        key=str(component.key),
        name=str(getattr(component, "name", component.key)),
        z_mm=float(component.z_mm),
        kind="detector",
        geometry=str(getattr(component, "geometry", "disk")),
        outer_width_mm=float(
            getattr(component, "outer_width_mm", getattr(component, "outer_diameter_mm", 0.0))
        ),
        inner_diameter_mm=float(getattr(component, "inner_diameter_mm", 0.0)),
        offset_x_mm=float(getattr(component, "centre_offset_x_mm", 0.0)),
        offset_y_mm=float(getattr(component, "centre_offset_y_mm", 0.0)),
        readout_enabled=bool(getattr(component, "readout_enabled", True)),
        non_blocking=bool(getattr(component, "NON_BLOCKING", False)),
        authority=_authority_for(state, str(component.key)),
    )


def runtime_recording_stops(state, source_z_mm: float | None = None) -> tuple[PlaneStop, ...]:
    """Snapshot enabled downstream apertures and inserted detector surfaces."""

    source = float(state.sample.z_mm if source_z_mm is None else source_z_mm)
    entries: list[tuple[float, int, int, PlaneStop]] = []
    seen: set[str] = set()
    for index, component in enumerate(getattr(state, "apertures", ())):
        key = str(getattr(component, "key", ""))
        if (
            not key
            or key in seen
            or not bool(getattr(component, "enabled", False))
            or not bool(getattr(component, "installed", True))
            or float(component.z_mm) < source - 1.0e-9
        ):
            continue
        stop = _aperture_stop(state, component)
        entries.append((stop.z_mm, 0, index, stop))
        seen.add(key)
    for index, component in enumerate(getattr(state, "recording_planes", ())):
        key = str(getattr(component, "key", ""))
        if (
            not key
            or key in seen
            or not bool(getattr(component, "inserted", False))
            or float(component.z_mm) < source - 1.0e-9
        ):
            continue
        stop = _detector_stop(state, component)
        entries.append((stop.z_mm, 1, index, stop))
        seen.add(key)
    return tuple(item[3] for item in sorted(entries, key=lambda item: item[:3]))


def resolved_runtime_geometry_fingerprint(state, planes: Sequence[PlaneStop]) -> str:
    """Hash the resolved structural geometry that owns the current ray plan."""

    assembly = getattr(state, "_resolved_assembly", None)
    assembly_payload = None
    if assembly is not None:
        assembly_payload = {
            "selected_module_paths": getattr(assembly, "selected_module_paths", ()),
            "parts": getattr(assembly, "parts", ()),
            "vacuum_bore_segments": getattr(assembly, "vacuum_bore_segments", ()),
            "vacuum_liner_segments": getattr(assembly, "vacuum_liner_segments", ()),
            "exit_z_mm": getattr(assembly, "exit_z_mm", None),
        }
    return _digest({
        "axis": "laboratory +Z downstream",
        "sample_z_mm": float(state.sample.z_mm),
        "beam_voltage_kv": float(state.beam_voltage_kv),
        "resolved_assembly": assembly_payload,
        "runtime_planes": tuple(planes),
    })


def _dynamic_column_coils(state, source_z_mm, target_z_mm):
    """Captured dynamic coils whose physical support intersects this span."""
    from temsim.physics.instrument_magnetic import column_dipole_fields
    keys = {str(component.key)
        for collection in ("deflectors", "corrector_elements", "stigmators")
        for component in getattr(state, collection, ())
        if getattr(component, "enabled", False)
        and (getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False))}
    return tuple(coil for coil in column_dipole_fields(state)
        if coil.key.rsplit(":", 1)[0] in keys
        and coil.lower_m < target_z_mm*1e-3 and coil.upper_m > source_z_mm*1e-3)


def _scan_deflection_offsets(state, source, planes, times, maximum_step_mm,
                             reference_transfers):
    """Trace finite-coil responses through overlapping fields, then combine.

    Linear analytic fields need at most two reference trajectories per driven
    coil, independent of raster size. Electric fields and imported maps trace the central
    orbit at each distinct time. Their Jacobian remains the declared local map
    at the reference working point; this is not a nonlinear per-pixel map.
    """
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.instrument_magnetic import active_column_events
    from temsim.physics.lens_field_provider import active_mapped_providers
    from temsim.physics.electrostatic_column_transport import active_electric_field

    if not planes:
        return ()
    reference_coils = _dynamic_column_coils(state, source, planes[-1].z_mm)
    if not reference_coils:
        return ()
    unique_times, inverse = np.unique(times, return_inverse=True)
    timed_states, timed_coils = [], []
    for time in unique_times:
        working = copy(state)
        working.simulation_time_s = float(time)
        timed_states.append(working)
        coils = _dynamic_column_coils(working, source, planes[-1].z_mm)
        if tuple((c.key, c.lower_m, c.upper_m) for c in coils) != tuple(
                (c.key, c.lower_m, c.upper_m) for c in reference_coils):
            raise ValueError("Scan-dependent magnetic coil geometry requires a time-dependent transport model")
        timed_coils.append(coils)
    deltas = np.asarray([[(c.bx_t-r.bx_t, c.by_t-r.by_t)
        for c, r in zip(coils, reference_coils)] for coils in timed_coils])
    if not np.any(deltas):
        return ()
    targets = tuple(float(plane.z_mm) for plane in planes)
    offsets = [np.zeros(times.shape + (2,)) for _ in planes]
    zero = np.zeros(1)
    plan = build_propagation_plan(state, source, targets[-1], active_column_events(state),
        include_spherical_aberration=False, include_hexapole=False,
        checkpoint_z_mm=targets, maximum_step_mm=maximum_step_mm)
    if active_mapped_providers(state) or active_electric_field(plan) is not None:
        positions = []
        for working in timed_states:
            plan = build_propagation_plan(working, source, targets[-1],
                active_column_events(working), include_spherical_aberration=False,
                include_hexapole=False, checkpoint_z_mm=targets,
                maximum_step_mm=maximum_step_mm)
            points = execute_propagation_plan(working, plan, zero, zero, zero, zero)[-1]
            indices = np.searchsorted(points.z_mm, targets)
            positions.append(np.column_stack((points.x_m[indices, 0], points.y_m[indices, 0])))
        for index, transfer in enumerate(reference_transfers):
            delta = np.asarray(positions)[:, index]-np.asarray(transfer.position_offset_m)
            offsets[index] = delta[inverse].reshape(times.shape+(2,))
        return tuple(offsets)

    sample = np.zeros((len(plan.step_m), 3))
    sample[:, 2] = .5*(plan.z_mm[:-1]+plan.z_mm[1:])*1e-3
    for coil_index, coil in enumerate(reference_coils):
        for axis in (0, 1):
            change = deltas[:, coil_index, axis]
            scale = float(np.max(np.abs(change)))
            if scale == 0.:
                continue
            # Normalise to the actual command range, avoiding an artificially
            # huge unit field. This is a response basis, never a beam source.
            basis_coil = replace(coil, bx_t=scale if axis == 0 else 0.,
                                 by_t=scale if axis == 1 else 0.)
            values = basis_coil.field_at_global_positions_t(sample)
            response_plan = replace(plan,
                dipole_bx_t=np.repeat(values[:, 0], 3),
                dipole_by_t=np.repeat(values[:, 1], 3),
                kick_x_rad=np.zeros_like(plan.kick_x_rad),
                kick_y_rad=np.zeros_like(plan.kick_y_rad),
                signature=_digest((plan.signature, "finite-coil-response", coil.key, axis, scale)))
            points = execute_propagation_plan(state, response_plan, zero, zero, zero, zero)[-1]
            indices = np.searchsorted(points.z_mm, targets)
            response = np.column_stack((points.x_m[indices, 0], points.y_m[indices, 0]))
            command = (change[inverse]/scale).reshape(times.shape)
            for index in range(len(planes)):
                offsets[index] += command[..., None]*response[index]
    return tuple(offsets)


def build_record_plane_plan(
    state,
    *,
    source_z_mm: float | None = None,
    maximum_step_mm: float | None = None,
    scan_times_s=None,
    recalibrate_scan: bool = False,
) -> RecordPlanePlan:
    """Trace all active physical surfaces from the current resolved state once."""

    if recalibrate_scan and scan_times_s is not None:
        # Serialized calculation snapshots do not retain the private AC/descan
        # coupling matrices. Rebuild those derived values on private components;
        # keep the caller's microscope settings and resolved geometry untouched.
        from temsim.physics.scan_geometry import calibrate_scan_system

        state = copy(state)
        state.corrector_elements = [copy(component) for component in state.corrector_elements]
        state._ac_deflector = None
        state._descan_deflector = None
        calibrate_scan_system(state)
    source = float(state.sample.z_mm if source_z_mm is None else source_z_mm)
    planes = runtime_recording_stops(state, source)
    transfers_by_z = trace_transverse_transfers(
        state,
        source,
        (plane.z_mm for plane in planes),
        maximum_step_mm=maximum_step_mm,
    )
    transfers = tuple(transfers_by_z[plane.z_mm] for plane in planes)
    times = None if scan_times_s is None else np.asarray(scan_times_s, dtype=float)
    if times is not None and (times.ndim != 2 or not times.size or not np.all(np.isfinite(times))):
        raise ValueError("Record-plane scan times must be a finite non-empty 2-D array")
    scan_offsets = () if times is None else _scan_deflection_offsets(
        state, source, planes, times, maximum_step_mm, transfers,
    )
    time_dependent = bool(planes and _dynamic_column_coils(state, source, planes[-1].z_mm))
    geometry_fingerprint = resolved_runtime_geometry_fingerprint(state, planes)
    fingerprint = _digest({
        "model": "finite_coil_first_order_recording_v3",
        "source_z_mm": source,
        "resolved_geometry_fingerprint": geometry_fingerprint,
        "planes": planes,
        "transfers": tuple(transfer.matrix for transfer in transfers),
        "offsets": tuple((transfer.position_offset_m, transfer.angle_offset_rad) for transfer in transfers),
        "scan_times_s": times,
        "scan_position_offsets_m": scan_offsets,
        "time_dependent_deflection": time_dependent,
    })
    return RecordPlanePlan(
        source_z_mm=source,
        planes=planes,
        transfers=transfers,
        resolved_geometry_fingerprint=geometry_fingerprint,
        fingerprint=fingerprint,
        scan_times_s=times,
        scan_position_offsets_m=scan_offsets,
        time_dependent_deflection=time_dependent,
    )


def record_plane_plan_provenance(plan: RecordPlanePlan) -> Mapping[str, object]:
    """Return finite JSON data for the exact runtime stops and signed maps."""

    return MappingProxyType({
        "model": "finite_coil_first_order_recording_v3",
        "scan_mapping_scope": "Local fixed Jacobian at the reference working point; finite-coil affine responses. Electric-field and imported-map central orbits are traced at each distinct scan time, without claiming the off-axis Jacobian stays constant.",
        "axis": "laboratory +Z downstream",
        "source_z_mm": float(plan.source_z_mm),
        "fingerprint": plan.fingerprint,
        "resolved_geometry_fingerprint": plan.resolved_geometry_fingerprint,
        "scan_times_s": _normalise_json(plan.scan_times_s),
        "scan_position_offsets_m": _normalise_json(plan.scan_position_offsets_m),
        "time_dependent_deflection": plan.time_dependent_deflection,
        "planes": tuple(_normalise_json(plane) for plane in plan.planes),
        "transfers": tuple({
            "source_z_mm": float(transfer.source_z_mm),
            "target_z_mm": float(transfer.target_z_mm),
            "matrix": np.asarray(transfer.matrix, dtype=float).tolist(),
            "position_offset_m": transfer.position_offset_m,
            "angle_offset_rad": transfer.angle_offset_rad,
            "j_img": np.asarray(transfer.j_img, dtype=float).tolist(),
            "j_diff_m_per_rad": np.asarray(
                transfer.j_diff_m_per_rad, dtype=float
            ).tolist(),
        } for transfer in plan.transfers),
    })


def prepare_record_plane_detector_masks(
    plan: RecordPlanePlan,
    sample_angle_rad,
) -> Callable[..., dict[str, np.ndarray]]:
    """Prepare exact detector acceptance on a fixed specimen angular grid.

    The returned callable accepts broadcast-compatible positions in metres
    and an optional flattened ``scan_slice`` for the plan's raster offsets.
    It returns one Boolean signal mask per detector, including disabled
    readouts.  Angles use radians and a trailing XY axis of length two.

    Only ``J_diff @ angle`` is cached.  Each call evaluates
    ``(J_img @ position + cached_angle) + affine + raster_offset`` in the
    reference router's arithmetic order.  All physical stop methods and
    sequential blocking/readout rules are shared with the full router;
    unused projected angles, per-plane histories and weighted diagnostics
    are not constructed.  This is not a new transport approximation.
    """

    angle = np.asarray(sample_angle_rad, dtype=float)
    if angle.shape[-1:] != (2,):
        raise ValueError("Sample angles must have trailing dimension 2")
    if not np.all(np.isfinite(angle)):
        raise ValueError("Sample phase-space coordinates must be finite")
    angular_positions = tuple(
        np.einsum("ij,...j->...i", transfer.j_diff_m_per_rad, angle)
        for transfer in plan.transfers
    )
    for values in angular_positions:
        if not np.all(np.isfinite(values)):
            raise ValueError("Projected phase-space coordinates must be finite")
        values.setflags(write=False)
    angle_shape = angle.shape[:-1]

    def detector_masks(
        sample_position_m, *, scan_slice: slice | None = None,
    ) -> dict[str, np.ndarray]:
        position = np.asarray(sample_position_m, dtype=float)
        if position.shape[-1:] != (2,):
            raise ValueError("Sample positions must have trailing dimension 2")
        try:
            event_shape = np.broadcast_shapes(position.shape[:-1], angle_shape)
        except ValueError as error:
            raise ValueError("Sample positions and angles are not broadcast-compatible") from error
        if not np.all(np.isfinite(position)):
            raise ValueError("Sample phase-space coordinates must be finite")
        active = np.ones(event_shape, dtype=bool)
        masks = {}
        for index, (plane, transfer, angular_position) in enumerate(
            zip(plan.planes, plan.transfers, angular_positions)
        ):
            projected = (
                np.einsum("ij,...j->...i", transfer.j_img, position)
                + angular_position
                + np.asarray(transfer.position_offset_m)
            )
            if plan.scan_position_offsets_m:
                delta = plan.scan_position_offsets_m[index].reshape(-1, 2)
                if scan_slice is not None:
                    delta = delta[scan_slice]
                if not event_shape or delta.shape[0] != event_shape[0]:
                    raise ValueError("Record-plane scan offsets do not match the routed scan batch")
                delta = delta.reshape((delta.shape[0],) + (1,) * (len(event_shape) - 1) + (2,))
                projected = projected + delta
            if not np.all(np.isfinite(projected)):
                raise ValueError("Projected phase-space coordinates must be finite")
            if plane.kind == "aperture":
                active &= plane.transmission_mask(projected[..., 0], projected[..., 1])
            else:
                hits = plane.hit_mask(projected[..., 0], projected[..., 1])
                signal = active & hits if plane.readout_enabled else np.zeros(event_shape, dtype=bool)
                signal.setflags(write=False)
                masks[plane.key] = signal
                if not plane.non_blocking:
                    active &= ~hits
        return masks

    return detector_masks


def route_record_planes(
    plan: RecordPlanePlan,
    sample_position_m,
    sample_angle_rad,
    *,
    weights=None,
    scan_slice: slice | None = None,
) -> RecordPlaneResult:
    """Route weighted phase-space samples through all surfaces in physical order."""

    position = np.asarray(sample_position_m, dtype=float)
    angle = np.asarray(sample_angle_rad, dtype=float)
    if position.shape[-1:] != (2,) or angle.shape[-1:] != (2,):
        raise ValueError("Sample position and angle arrays must end in dimension 2")
    try:
        event_shape = np.broadcast_shapes(position.shape[:-1], angle.shape[:-1])
    except ValueError as error:
        raise ValueError("Sample position and angle arrays cannot be broadcast") from error
    position = np.broadcast_to(position, event_shape + (2,))
    angle = np.broadcast_to(angle, event_shape + (2,))
    if weights is None:
        weight = np.ones(event_shape, dtype=float)
    else:
        try:
            weight = np.broadcast_to(np.asarray(weights, dtype=float), event_shape)
        except ValueError as error:
            raise ValueError("Record-plane weights cannot be broadcast to event shape") from error
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("Record-plane weights must be finite and non-negative")

    active = np.ones(event_shape, dtype=bool)
    interactions: list[PlaneInteraction] = []
    physically_intercepted = 0.0
    for index, (plane, transfer) in enumerate(zip(plan.planes, plan.transfers)):
        projected = project_sample_phase_space(transfer, position, angle)
        if plan.scan_position_offsets_m:
            delta = plan.scan_position_offsets_m[index].reshape(-1, 2)
            if scan_slice is not None:
                delta = delta[scan_slice]
            if not event_shape or delta.shape[0] != event_shape[0]:
                raise ValueError("Record-plane scan offsets do not match the routed scan batch")
            delta = delta.reshape((delta.shape[0],) + (1,) * (len(event_shape) - 1) + (2,))
            projected = ProjectedPhaseSpace(projected.position_m + delta, projected.angle_rad)
        incident = active.copy()
        if plane.kind == "aperture":
            passes = plane.transmission_mask(
                projected.position_m[..., 0], projected.position_m[..., 1]
            )
            intercepted = incident & ~passes
            signal = np.zeros(event_shape, dtype=bool)
            outgoing = incident & passes
        else:
            hits = plane.hit_mask(
                projected.position_m[..., 0], projected.position_m[..., 1]
            )
            intercepted = incident & hits
            signal = intercepted if plane.readout_enabled else np.zeros(event_shape, dtype=bool)
            outgoing = incident if plane.non_blocking else incident & ~hits
        incident_weight = float(np.sum(weight[incident], dtype=np.float64))
        intercepted_weight = float(np.sum(weight[intercepted], dtype=np.float64))
        signal_weight = float(np.sum(weight[signal], dtype=np.float64))
        outgoing_weight = float(np.sum(weight[outgoing], dtype=np.float64))
        if plane.kind == "aperture" or not plane.non_blocking:
            physically_intercepted += intercepted_weight
        interactions.append(PlaneInteraction(
            plane=plane,
            projected_position_m=projected.position_m,
            incident_mask=incident,
            intercepted_mask=intercepted,
            signal_mask=signal,
            outgoing_mask=outgoing,
            incident_weight=incident_weight,
            intercepted_weight=intercepted_weight,
            signal_weight=signal_weight,
            outgoing_weight=outgoing_weight,
        ))
        active = outgoing

    initial_weight = float(np.sum(weight, dtype=np.float64))
    surviving_weight = float(np.sum(weight[active], dtype=np.float64))
    balance_error = initial_weight - physically_intercepted - surviving_weight
    tolerance = max(initial_weight, 1.0) * 2.0e-12
    if abs(balance_error) > tolerance:
        raise RuntimeError(
            "Record-plane routing did not conserve physical weight: "
            f"error={balance_error:.6g}, tolerance={tolerance:.6g}"
        )
    return RecordPlaneResult(
        plan_fingerprint=plan.fingerprint,
        event_shape=event_shape,
        interactions=tuple(interactions),
        surviving_mask=active,
        initial_weight=initial_weight,
        physically_intercepted_weight=physically_intercepted,
        surviving_weight=surviving_weight,
        balance_error=balance_error,
    )


def result_matches_plan(result: RecordPlaneResult, plan: RecordPlanePlan) -> bool:
    """Return whether a cached result belongs to the exact current ray plan."""

    return bool(result.plan_fingerprint == plan.fingerprint)
