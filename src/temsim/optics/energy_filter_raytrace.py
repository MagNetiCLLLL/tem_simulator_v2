"""Continuous relativistic ray tracing through the Energy Filter."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
from threading import RLock

import numpy as np
from temsim.physics.backend_execution import cpu_call

from temsim.detector.stem_signal import (
    measure_aperture_transmitted_current,
    source_current_pa,
)
from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter import EnergyFilterSystem
from temsim.optics.energy_filter_detector import (
    EnergyFilterBiasTube, EnergyFilterCameraDeflector, EnergyFilterShutter,
    ZebraEELSDetector,
)
from temsim.optics.energy_filter_m12 import EnergyFilterM12Component, M12VoltageCalibration
from temsim.optics.energy_filter_slit import EnergySelectionSlitComponent
from temsim.optics.twelve_pole_element import LocalCoordinateFrame, TwelvePoleElement
from temsim.optics.energy_filter_sector import (
    EnergyFilterMagneticField, SectorMagnetElement,
    magnetic_field_from_energy_filter,
    sector_from_energy_filter,
)
from temsim.physics.finite_multipole_field import FiniteMultipoleField, SoftEdgeEnvelope
from temsim.physics.multipole_field import MultipoleField
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    boris_step,
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


# Only an already executed, independent diagnostic reference may enter this
# process-local cache. The physical incident population is always traced.
_REFERENCE_CACHE_SCHEMA = "executed-filter-reference-boris-tof-v1"
_REFERENCE_CACHE_MAX_ENTRIES = 8
_REFERENCE_CACHE_MAX_BYTES = 8 * 1024 * 1024
_REFERENCE_CACHE = OrderedDict()
_REFERENCE_CACHE_LOCK = RLock()
_REFERENCE_MODEL_TYPES = (
    EnergyFilterSystem, EnergyFilterM12Component, M12VoltageCalibration,
    TwelvePoleElement, LocalCoordinateFrame, FiniteMultipoleField,
    SoftEdgeEnvelope, MultipoleField, EnergySelectionSlitComponent,
    EnergyFilterBiasTube, EnergyFilterShutter, EnergyFilterCameraDeflector,
    ZebraEELSDetector, EnergyFilterMagneticField, SectorMagnetElement,
)
_REFERENCE_MODEL_METHODS = tuple(
    (cls, name, value)
    for cls in _REFERENCE_MODEL_TYPES for name, value in vars(cls).items()
    if callable(value) or isinstance(value, (property, staticmethod, classmethod))
)
_REFERENCE_FIELD_FACTORY = magnetic_field_from_energy_filter
_REFERENCE_BORIS = boris_step


def _reference_model_payload(value):
    """Serialize actual fields/frames, including geometry omitted by profiles.

    Unknown providers or instance method overrides are deliberately uncached.
    The bound entrance aperture is upstream of this zero-offset reference;
    its transmitted population is separately traced by the ordinary path.
    """
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biuf":
            raise TypeError("Unrecognized reference input array")
        return {"dtype": value.dtype.str, "shape": value.shape, "values": value.tolist()}
    if isinstance(value, Mapping):
        return {key: _reference_model_payload(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_reference_model_payload(item) for item in value]
    if type(value) in _REFERENCE_MODEL_TYPES:
        return {"type": type(value).__module__ + "." + type(value).__qualname__,
                "values": {name: _reference_model_payload(item) for name, item in vars(value).items()
                           if not (type(value) is EnergyFilterSystem and name == "_entrance_aperture_component")}}
    raise TypeError("Unrecognized reference field or component")


def _reference_cache_key(state):
    if (magnetic_field_from_energy_filter is not _REFERENCE_FIELD_FACTORY
            or boris_step is not _REFERENCE_BORIS
            or any(vars(cls).get(name) is not expected for cls, name, expected in _REFERENCE_MODEL_METHODS)):
        return None
    from temsim.calculation_manifest import solver_source_identity
    from temsim.immutable_json import json_digest
    try:
        payload = {"beam_voltage_kv": float(state.beam_voltage_kv),
                   "energy_filter": _reference_model_payload(state.energy_filter),
                   "method": "boris", "apply_slit": True, "stop_at_slit": False,
                   "initial_position_slope_energy_offset": [0.] * 5,
                   "implementation": solver_source_identity()}
        return (_REFERENCE_CACHE_SCHEMA, trace_energy_filter_batch, json_digest(payload))
    except (TypeError, ValueError, AttributeError):
        return None


def _copy_reference_batch(batch, *, readonly=False):
    values = {}
    for field in fields(EnergyFilterTraceBatch):
        value = getattr(batch, field.name)
        if value is not None:
            value = deepcopy(value)
            if readonly:
                value.setflags(write=False)
        values[field.name] = value
    return EnergyFilterTraceBatch(**values)


def _trace_energy_filter_reference(state, *, use_cache=True):
    """Return a detached reference trace; never a configurable beam source."""
    ensure_energy_filter(state)
    key = _reference_cache_key(state) if use_cache else None
    if key is not None:
        with _REFERENCE_CACHE_LOCK:
            cached = _REFERENCE_CACHE.get(key)
            if cached is not None:
                _REFERENCE_CACHE.move_to_end(key)
                return _copy_reference_batch(cached[0])
    result = trace_energy_filter_batch(state, [0.], [0.], [0.], [0.], [0.])
    if key is not None and _reference_cache_key(state) == key:
        retained = _copy_reference_batch(result, readonly=True)
        size = sum(getattr(retained, field.name).nbytes for field in fields(EnergyFilterTraceBatch)
                   if getattr(retained, field.name) is not None)
        if size <= _REFERENCE_CACHE_MAX_BYTES:
            with _REFERENCE_CACHE_LOCK:
                _REFERENCE_CACHE[key] = (retained, size)
                _REFERENCE_CACHE.move_to_end(key)
                while (len(_REFERENCE_CACHE) > _REFERENCE_CACHE_MAX_ENTRIES
                       or sum(item[1] for item in _REFERENCE_CACHE.values()) > _REFERENCE_CACHE_MAX_BYTES):
                    _REFERENCE_CACHE.popitem(last=False)
    return result


@dataclass(frozen=True)
class EntranceRay:
    x_mm: float
    y_mm: float
    tx_rad: float
    ty_rad: float
    energy_offset_ev: float
    colour: str
    source_fraction: float
    source_ray_id: int = -1
    flight_time_s: float = math.nan
    source_branch: str = ""
    source_path_index: int = -1


@dataclass
class EnergyFilterTraceBatch:
    positions_m: np.ndarray
    stop_step: np.ndarray
    stop_key: np.ndarray
    reached_slit: np.ndarray
    passed_slit: np.ndarray
    reached_output: np.ndarray
    reached_eels: np.ndarray
    zebra_recorded: np.ndarray
    slit_dispersive_m: np.ndarray
    slit_non_dispersive_m: np.ndarray
    output_dispersive_m: np.ndarray
    output_non_dispersive_m: np.ndarray
    final_momentum_kg_m_per_s: np.ndarray
    # Elapsed laboratory time from this batch's filter-entrance event. These
    # local clocks alone never establish a tip-origin arrival time.
    time_s: np.ndarray | None = None
    slit_time_s: np.ndarray | None = None
    output_time_s: np.ndarray | None = None
    eels_time_s: np.ndarray | None = None
    slit_tx_rad: np.ndarray | None = None
    slit_ty_rad: np.ndarray | None = None
    output_tx_rad: np.ndarray | None = None
    output_ty_rad: np.ndarray | None = None
    eels_dispersive_m: np.ndarray | None = None
    eels_non_dispersive_m: np.ndarray | None = None
    eels_tx_rad: np.ndarray | None = None
    eels_ty_rad: np.ndarray | None = None


@dataclass(frozen=True)
class FilterPlaneArrival:
    """A physical crossing in the bent filter's outgoing local frame.

    Time is seconds from simultaneous tip emission, or NaN when the incoming
    path lacks that clock. X/Y are dispersive/non-dispersive metres; slopes
    are transverse momentum divided by momentum along the sector-exit axis.
    Reached is a geometric crossing, independent of detector acceptance and
    of whether an upstream tip-origin time is available.
    """

    key: str
    time_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    reached: np.ndarray
    coordinate_frame: str = "sector_exit_local"
    time_reference: str = "simultaneous_tip_emission"


@dataclass
class EnergyFilterResult:
    paths_u_mm: list
    paths_v_mm: list
    paths_y_mm: list
    colours: list
    losses_ev: np.ndarray
    entrance_ray_count: int
    eels_ray_count: int
    stopped_at_output_detector: bool
    entrance_signal: object
    status: str
    stop_keys: tuple
    slit_transmitted_fraction: float
    slit_transmitted_current_pa: float
    camera_recorded_fraction: float
    camera_recorded_current_pa: float
    eels_transmitted_fraction: float
    eels_transmitted_current_pa: float
    metrics: object = None
    eels_forward: object = None
    eftem_image: object = None
    # These arrays and timed_planes share the sampled physical-ray order.
    # The extra diagnostic reference polyline in paths_* is excluded.
    source_ray_id: np.ndarray | None = None
    source_fraction: np.ndarray | None = None
    source_branch: tuple[str, ...] = ()
    source_path_index: np.ndarray | None = None
    entrance_time_s: np.ndarray | None = None
    timed_planes: tuple[FilterPlaneArrival, ...] = ()
    entrance_provenance: str = "historical_unspecified"
    entrance_dependency_signature: str = ""


def branch_items(simulation):
    values = [("incident", getattr(simulation, "incident", None))]
    values += list(getattr(simulation, "branches", {}).items())
    return [(name, branch) for name, branch in values if branch is not None]


def colour(name, branch):
    value = getattr(branch, "colour", None)
    if isinstance(value, str):
        return value
    if value is not None and len(value) >= 3:
        rgb = (
            np.clip(np.asarray(value[:3]), 0.0, 1.0) * 255
        ).astype(int)
        return "#%02x%02x%02x" % tuple(rgb)
    if "+g" in name:
        return "#d32f2f"
    if "-g" in name:
        return "#1976d2"
    return "#f2c600"


def at_plane(branch, plane_z_mm):
    z = np.asarray(branch.z, dtype=float)
    if (
        z.size < 2
        or plane_z_mm < z[0] - 1.0e-9
        or plane_z_mm > z[-1] + 1.0e-9
    ):
        return None
    upper = min(
        max(int(np.searchsorted(z, plane_z_mm, "left")), 1),
        len(z) - 1,
    )
    lower = upper - 1
    fraction = (plane_z_mm - z[lower]) / max(
        z[upper] - z[lower],
        1.0e-12,
    )

    def interpolate(values):
        values = np.asarray(values, dtype=float)
        return (
            values[lower]
            + fraction * (values[upper] - values[lower])
        )

    blocked_z = np.asarray(branch.blocked_z, dtype=float)
    valid = np.isnan(blocked_z) | (blocked_z >= plane_z_mm - 1.0e-9)
    return (
        interpolate(branch.x),
        interpolate(branch.y),
        interpolate(branch.tx),
        interpolate(branch.ty),
        valid,
    )


def _branch_probabilities(simulation):
    branches = tuple(getattr(simulation, "branches", {}).values())
    if not branches:
        return {}
    raw = np.asarray(
        [
            max(float(getattr(branch, "weight", 1.0)), 0.0)
            for branch in branches
        ],
        dtype=float,
    )
    if bool(
        getattr(simulation, "metrics", {}).get(
            "branch_weights_are_absolute", False
        )
    ):
        if float(raw.sum()) > 1.0 + 1.0e-10:
            raise ValueError(
                "Absolute simulation branch probabilities exceed one."
            )
        return {
            id(branch): float(probability)
            for branch, probability in zip(branches, raw)
        }
    total = float(raw.sum())
    if total <= 0.0:
        raw[:] = 1.0
        total = float(len(raw))
    return {
        id(branch): float(probability)
        for branch, probability in zip(branches, raw / total)
    }


def extract_entrance_rays(state, simulation):
    from temsim.physics.flight_time import sample_flight_time
    from temsim.physics.ray_identity import branch_identity
    energy_filter = state.energy_filter
    aperture = state.energy_filter_entrance_aperture
    probabilities = _branch_probabilities(simulation)
    output = []
    provenance = str(getattr(simulation, "metrics", {}).get(
        "energy_filter_entrance_provenance", "optical_column_reference"))
    items = (list(getattr(simulation, "branches", {}).items())
             if provenance in {"validated_specimen_exit", "no_incident_illumination"}
             else branch_items(simulation))
    for name, branch in items:
        data = at_plane(branch, float(energy_filter.entrance_z_mm))
        if data is None:
            continue
        x, y, tx, ty, valid = data
        blocked_z = np.asarray(branch.blocked_z, dtype=float)
        # Observing a ray's arrival at an absorbing entrance-plane device does
        # not grant that ray permission to propagate through the filter.
        valid &= np.isnan(blocked_z) | (blocked_z > float(energy_filter.entrance_z_mm) + 1e-9)
        flight_times = sample_flight_time(branch, float(energy_filter.entrance_z_mm))
        source_ids, _ = branch_identity(branch, simulation)
        # Source azimuth is optional for a clock; a separately stored valid
        # ancestor ID must not be discarded merely because azimuth is absent.
        stored_ids = np.asarray(getattr(branch, "source_ray_id", None))
        if (stored_ids.shape == x.shape and stored_ids.dtype.kind in "iu"
                and np.all(stored_ids >= -1)
                and np.all(stored_ids <= np.iinfo(np.int64).max)):
            source_ids = stored_ids
        x_mm = x * 1.0e3
        y_mm = y * 1.0e3
        valid &= np.isfinite(x_mm) & np.isfinite(y_mm)
        if aperture.enabled and aperture.installed:
            valid &= aperture.transmission_mask(x_mm, y_mm)
        offsets = np.asarray(
            getattr(branch, "energy_offset_ev", np.zeros_like(x)),
            dtype=float,
        )
        ray_weights = getattr(branch, "ray_weight", None)
        if ray_weights is None:
            ray_weights = np.full(
                len(x),
                1.0 / max(len(x), 1),
                dtype=float,
            )
        else:
            ray_weights = np.asarray(ray_weights, dtype=float)
        branch_probability = probabilities.get(id(branch), 1.0)
        for index in np.flatnonzero(valid):
            output.append(
                EntranceRay(
                    x_mm=float(x_mm[index]),
                    y_mm=float(y_mm[index]),
                    tx_rad=float(tx[index]),
                    ty_rad=float(ty[index]),
                    energy_offset_ev=float(offsets[index]),
                    colour=colour(name, branch),
                    source_fraction=(
                        branch_probability
                        * float(ray_weights[index])
                    ),
                    source_ray_id=int(source_ids[index]),
                    flight_time_s=float(flight_times[index]),
                    source_branch=str(name),
                    source_path_index=int(index),
                )
            )
    return output


def _representative_entrance_rays(rays, maximum_count):
    """Bound branch-memory use while preserving total current weight."""

    maximum = max(1, int(maximum_count))
    if len(rays) <= maximum:
        return list(rays), len(rays)
    indices = np.unique(np.linspace(
        0, len(rays) - 1, maximum, dtype=int
    ))
    selected = [rays[index] for index in indices]
    total_weight = sum(ray.source_fraction for ray in rays)
    selected_weight = sum(ray.source_fraction for ray in selected)
    scale = total_weight / max(selected_weight, np.finfo(float).tiny)
    selected = [
        replace(ray, source_fraction=ray.source_fraction * scale)
        for ray in selected
    ]
    return selected, len(rays)


def _m12_bore_blocked(element, positions_m):
    local = element.local_positions_m(positions_m)
    within_length = (
        np.abs(local[..., 2]) <= 0.5 * float(element.length_m)
    )
    outside_bore = np.hypot(
        local[..., 0],
        local[..., 1],
    ) > float(element.bore_radius_m)
    return within_length & outside_bore


def _crossing_fraction(previous, current, plane):
    denominator = current - previous
    return np.divide(
        plane - previous,
        denominator,
        out=np.zeros_like(current),
        where=np.abs(denominator) > 1.0e-18,
    )


def trace_energy_filter_batch(
    state,
    x_mm,
    y_mm,
    tx_rad,
    ty_rad,
    energy_offset_ev,
    *,
    method="boris",
    apply_slit=True,
    stop_at_slit=False,
):
    """Trace one ray population, retaining elapsed clocks from its entrance.

    Boundary clocks use the same accepted-step crossing fraction as position;
    no displayed path length or reference-trajectory time is substituted.
    """

    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    if str(method).lower() != "boris":
        raise ValueError(
            "Energy Filter production tracing currently uses Boris."
        )
    arrays = np.broadcast_arrays(
        np.asarray(x_mm, dtype=float),
        np.asarray(y_mm, dtype=float),
        np.asarray(tx_rad, dtype=float),
        np.asarray(ty_rad, dtype=float),
        np.asarray(energy_offset_ev, dtype=float),
    )
    x_mm, y_mm, tx_rad, ty_rad, energy_offset_ev = [
        value.reshape(-1) for value in arrays
    ]
    ray_count = len(x_mm)
    kinetic_energy_ev = (
        float(state.beam_voltage_kv) * 1000.0
        + energy_offset_ev
    )
    if np.any(kinetic_energy_ev <= 0.0):
        raise ValueError("Energy Filter ray energy must be positive.")

    directions = np.column_stack(
        (
            np.ones(ray_count),
            ty_rad
            + float(energy_filter.alignment_y_mrad) * 1.0e-3,
            tx_rad
            + float(energy_filter.alignment_x_mrad) * 1.0e-3,
        )
    )
    position = np.column_stack(
        (
            np.zeros(ray_count),
            y_mm * 1.0e-3,
            x_mm * 1.0e-3,
        )
    )
    momentum = momentum_from_kinetic_energy_ev(
        kinetic_energy_ev,
        directions,
    )
    state_now = RelativisticPhaseSpace(position, momentum)
    magnetic_field = magnetic_field_from_energy_filter(energy_filter)
    sector = magnetic_field.sector
    exit_frame = sector.exit_frame
    exit_origin = sector.exit_point_m
    outgoing_s = exit_frame.rotation_local_to_global[:, 2]
    outgoing_x = exit_frame.rotation_local_to_global[:, 0]
    outgoing_y = exit_frame.rotation_local_to_global[:, 1]

    step_m = max(
        float(getattr(energy_filter, "ray_step_mm", 0.5)) * 1.0e-3,
        1.0e-6,
    )
    speeds = np.linalg.norm(
        velocity_from_momentum_m_per_s(momentum),
        axis=-1,
    )
    time_step_s = step_m / max(float(np.max(speeds)), 1.0)
    total_length_m = (
        float(energy_filter.prism_entrance_s_mm) * 1.0e-3
        + sector.arc_length_m
        + (
            float(energy_filter.output_detector_d_mm)
            + float(energy_filter.eels_plane_offset_mm)
            + 20.0
        )
        * 1.0e-3
    )
    step_count = int(math.ceil(1.25 * total_length_m / step_m))

    positions = np.empty(
        (step_count + 1, ray_count, 3),
        dtype=float,
    )
    positions[0] = position
    elapsed_times = np.empty(step_count + 1, dtype=np.float64)
    elapsed_times[0] = float(state_now.time_s)
    alive = np.ones(ray_count, dtype=bool)
    stop_step = np.full(ray_count, step_count, dtype=int)
    stop_key = np.full(ray_count, "", dtype=object)
    reached_slit = np.zeros(ray_count, dtype=bool)
    passed_slit = np.zeros(ray_count, dtype=bool)
    reached_output = np.zeros(ray_count, dtype=bool)
    reached_eels = np.zeros(ray_count, dtype=bool)
    zebra_recorded = np.zeros(ray_count, dtype=bool)
    slit_dispersive = np.full(ray_count, np.nan)
    slit_non_dispersive = np.full(ray_count, np.nan)
    output_dispersive = np.full(ray_count, np.nan)
    output_non_dispersive = np.full(ray_count, np.nan)
    slit_time = np.full(ray_count, np.nan)
    output_time = np.full(ray_count, np.nan)
    eels_time = np.full(ray_count, np.nan)
    slit_tx, slit_ty, output_tx, output_ty, eels_tx, eels_ty = (
        np.full(ray_count, np.nan) for _ in range(6)
    )
    eels_dispersive = np.full(ray_count, np.nan)
    eels_non_dispersive = np.full(ray_count, np.nan)

    previous_delta = position - exit_origin
    previous_along = previous_delta @ outgoing_s
    slit_plane = float(
        energy_filter.energy_slit.distance_from_sector_exit_m
    )
    output_plane = (
        float(energy_filter.output_detector_d_mm) * 1.0e-3
    )
    eels_plane = output_plane + (
        float(energy_filter.eels_plane_offset_mm) * 1.0e-3
    )
    shutter_plane = (
        float(energy_filter.fast_shutter_d_mm) * 1.0e-3
    )

    def crossing_slopes(indices, fraction, before, after):
        momentum = before[indices] + fraction[:, None] * (after[indices] - before[indices])
        along = momentum @ outgoing_s
        return tuple(np.divide(momentum @ axis, along,
                     out=np.full(indices.size, np.nan), where=along != 0.)
                     for axis in (outgoing_x, outgoing_y))

    for step_index in range(1, step_count + 1):
        previous_time_s = float(state_now.time_s)
        previous_position = state_now.position_m.copy()
        previous_momentum = state_now.momentum_kg_m_per_s.copy()
        advanced = boris_step(
            state_now,
            time_step_s,
            magnetic_field,
        )
        advanced.position_m[~alive] = previous_position[~alive]
        advanced.momentum_kg_m_per_s[~alive] = previous_momentum[~alive]
        state_now = advanced
        elapsed_times[step_index] = float(state_now.time_s)
        current_position = state_now.position_m
        positions[step_index] = current_position

        newly_blocked = np.zeros(ray_count, dtype=bool)
        for multipole in energy_filter.multipoles:
            carrier_blocked = (
                alive
                & ~newly_blocked
                & _m12_bore_blocked(multipole, current_position)
            )
            stop_key[carrier_blocked] = multipole.key
            newly_blocked |= carrier_blocked

        sector_blocked = (
            alive
            & ~newly_blocked
            & sector.aperture_blocked_mask(current_position)
        )
        stop_key[sector_blocked] = "energy_filter_sector_bore"
        newly_blocked |= sector_blocked

        delta = current_position - exit_origin
        current_along = delta @ outgoing_s

        slit_crossing = (
            alive
            & ~newly_blocked
            & ~reached_slit
            & (previous_along < slit_plane)
            & (current_along >= slit_plane)
        )
        if np.any(slit_crossing):
            fraction = _crossing_fraction(
                previous_along[slit_crossing],
                current_along[slit_crossing],
                slit_plane,
            )
            crossing_position = (
                previous_position[slit_crossing]
                + fraction[:, np.newaxis]
                * (
                    current_position[slit_crossing]
                    - previous_position[slit_crossing]
                )
            )
            crossing_delta = crossing_position - exit_origin
            dispersive = crossing_delta @ outgoing_x
            non_dispersive = crossing_delta @ outgoing_y
            indices = np.flatnonzero(slit_crossing)
            reached_slit[indices] = True
            slit_time[indices] = previous_time_s + fraction * (float(state_now.time_s) - previous_time_s)
            slit_tx[indices], slit_ty[indices] = crossing_slopes(
                indices, fraction, previous_momentum, state_now.momentum_kg_m_per_s)
            slit_dispersive[indices] = dispersive
            slit_non_dispersive[indices] = non_dispersive
            transmitted = (
                energy_filter.energy_slit.transmission_mask(
                    dispersive,
                    non_dispersive,
                )
                if apply_slit
                else np.ones(dispersive.shape, dtype=bool)
            )
            passed_slit[indices] = transmitted
            rejected_indices = indices[~transmitted]
            if rejected_indices.size:
                newly_blocked[rejected_indices] = True
                stop_key[rejected_indices] = (
                    energy_filter.energy_slit.key
                )
                current_position[rejected_indices] = crossing_position[
                    ~transmitted
                ]
            if stop_at_slit:
                newly_blocked[indices] = True
                stop_key[indices] = "slit_measurement_plane"
                current_position[indices] = crossing_position

        shutter_crossing = (
            alive
            & ~newly_blocked
            & (previous_along < shutter_plane)
            & (current_along >= shutter_plane)
        )
        shutter = energy_filter.fast_shutter
        if (
            np.any(shutter_crossing)
            and shutter.enabled
            and not shutter.open
        ):
            indices = np.flatnonzero(shutter_crossing)
            fraction = _crossing_fraction(
                previous_along[indices],
                current_along[indices],
                shutter_plane,
            )
            crossing_position = (
                previous_position[indices]
                + fraction[:, np.newaxis]
                * (current_position[indices] - previous_position[indices])
            )
            newly_blocked[indices] = True
            stop_key[indices] = shutter.key
            current_position[indices] = crossing_position

        output_crossing = (
            alive
            & ~newly_blocked
            & ~reached_output
            & (previous_along < output_plane)
            & (current_along >= output_plane)
        )
        if np.any(output_crossing):
            fraction = _crossing_fraction(
                previous_along[output_crossing],
                current_along[output_crossing],
                output_plane,
            )
            crossing_position = (
                previous_position[output_crossing]
                + fraction[:, np.newaxis]
                * (
                    current_position[output_crossing]
                    - previous_position[output_crossing]
                )
            )
            crossing_delta = crossing_position - exit_origin
            dispersive = crossing_delta @ outgoing_x
            non_dispersive = crossing_delta @ outgoing_y
            indices = np.flatnonzero(output_crossing)
            reached_output[indices] = True
            output_time[indices] = previous_time_s + fraction * (float(state_now.time_s) - previous_time_s)
            output_tx[indices], output_ty[indices] = crossing_slopes(
                indices, fraction, previous_momentum, state_now.momentum_kg_m_per_s)
            output_dispersive[indices] = dispersive
            output_non_dispersive[indices] = non_dispersive
            if energy_filter.output_detector_inserted:
                half_width = (
                    float(energy_filter.output_detector_width_mm)
                    * 0.5
                    * 1.0e-3
                )
                recorded = (
                    (np.abs(dispersive) <= half_width)
                    & (np.abs(non_dispersive) <= half_width)
                )
                recorded_indices = indices[recorded]
                if recorded_indices.size:
                    newly_blocked[recorded_indices] = True
                    stop_key[recorded_indices] = (
                        "energy_filter_output_detector"
                    )

        eels_crossing = (
            alive
            & ~newly_blocked
            & ~reached_eels
            & (previous_along < eels_plane)
            & (current_along >= eels_plane)
        )
        if np.any(eels_crossing):
            indices = np.flatnonzero(eels_crossing)
            fraction = _crossing_fraction(
                previous_along[indices],
                current_along[indices],
                eels_plane,
            )
            crossing_position = (
                previous_position[indices]
                + fraction[:, np.newaxis]
                * (current_position[indices] - previous_position[indices])
            )
            crossing_delta = crossing_position - exit_origin
            dispersive = crossing_delta @ outgoing_x
            non_dispersive = crossing_delta @ outgoing_y
            eels_time[indices] = previous_time_s + fraction * (float(state_now.time_s) - previous_time_s)
            eels_dispersive[indices] = dispersive
            eels_non_dispersive[indices] = non_dispersive
            eels_tx[indices], eels_ty[indices] = crossing_slopes(
                indices, fraction, previous_momentum, state_now.momentum_kg_m_per_s)
            bias = energy_filter.bias_tube
            if (
                energy_filter.multi_eels_enabled
                and bias.enabled
                and abs(float(bias.offset_ev)) > 0.0
            ):
                dispersion_um_per_ev = float(
                    energy_filter.energy_slit
                    .calibrated_dispersion_um_per_ev
                )
                dispersive = (
                    dispersive
                    - float(bias.offset_ev)
                    * dispersion_um_per_ev
                    * 1.0e-6
                )
            zebra = energy_filter.zebra_detector
            recorded = zebra.recording_mask(
                dispersive, non_dispersive
            )
            reached_eels[indices] = True
            zebra_recorded[indices] = recorded
            newly_blocked[indices] = True
            current_position[indices] = crossing_position
            stop_key[indices] = "energy_filter_zebra_miss"
            if np.any(recorded):
                active_strip = (
                    int(energy_filter.camera_deflector.active_strip)
                    if energy_filter.camera_deflector.enabled
                    else 1
                )
                stop_key[indices[recorded]] = (
                    f"{zebra.key}_strip_{active_strip}"
                    if not zebra.alignment_mode
                    else f"{zebra.key}_alignment"
                )

        if np.any(newly_blocked):
            stop_step[newly_blocked] = step_index
            alive[newly_blocked] = False
            positions[step_index] = current_position
        previous_along = current_along
        if not np.any(alive):
            positions = positions[: step_index + 1]
            elapsed_times = elapsed_times[: step_index + 1]
            break

    return EnergyFilterTraceBatch(
        positions_m=positions,
        stop_step=stop_step,
        stop_key=stop_key,
        reached_slit=reached_slit,
        passed_slit=passed_slit,
        reached_output=reached_output,
        reached_eels=reached_eels,
        zebra_recorded=zebra_recorded,
        slit_dispersive_m=slit_dispersive,
        slit_non_dispersive_m=slit_non_dispersive,
        output_dispersive_m=output_dispersive,
        output_non_dispersive_m=output_non_dispersive,
        final_momentum_kg_m_per_s=state_now.momentum_kg_m_per_s.copy(),
        time_s=elapsed_times,
        slit_time_s=slit_time,
        output_time_s=output_time,
        eels_time_s=eels_time,
        slit_tx_rad=slit_tx,
        slit_ty_rad=slit_ty,
        output_tx_rad=output_tx,
        output_ty_rad=output_ty,
        eels_dispersive_m=eels_dispersive,
        eels_non_dispersive_m=eels_non_dispersive,
        eels_tx_rad=eels_tx,
        eels_ty_rad=eels_ty,
    )


def _paths_from_batch(batch):
    paths_u = []
    paths_v = []
    paths_y = []
    for ray_index, stop_index in enumerate(batch.stop_step):
        upper = min(int(stop_index) + 1, batch.positions_m.shape[0])
        path = batch.positions_m[:upper, ray_index]
        paths_u.append(path[:, 0] * 1.0e3)
        paths_v.append(path[:, 2] * 1.0e3)
        paths_y.append(path[:, 1] * 1.0e3)
    return paths_u, paths_v, paths_y


def trace_one(state, x0, tx0, energy_offset_ev):
    """Compatibility wrapper returning the longitudinal diagram path."""

    batch = trace_energy_filter_batch(
        state,
        [x0],
        [0.0],
        [tx0],
        [0.0],
        [energy_offset_ev],
    )
    paths_u, paths_v, _ = _paths_from_batch(batch)
    return paths_u[0], paths_v[0], bool(batch.reached_eels[0])


def _timed_filter_planes(energy_filter, batch, entrance_time_s):
    """Attach boundary clocks to their executed upstream ancestry only."""
    from temsim.component_keys import ENERGY_FILTER_EFTEM_OUTPUT_PLANE
    initial = np.asarray(entrance_time_s, dtype=np.float64)
    planes = []
    for key, prefix in (
        (energy_filter.energy_slit.key, "slit"),
        (ENERGY_FILTER_EFTEM_OUTPUT_PLANE, "output"),
        (energy_filter.zebra_detector.key, "eels"),
    ):
        def values(suffix):
            raw = getattr(batch, f"{prefix}_{suffix}", None)
            return (np.full(initial.shape, np.nan, dtype=np.float64) if raw is None
                    else np.asarray(raw, dtype=np.float64).copy())

        elapsed = values("time_s")
        reached = np.asarray(getattr(batch, f"reached_{prefix}"), dtype=bool).copy()
        known = (reached & np.isfinite(initial) & (initial >= 0.)
                 & np.isfinite(elapsed) & (elapsed >= 0.))
        absolute = np.full(initial.shape, np.nan, dtype=np.float64)
        absolute[known] = initial[known] + elapsed[known]
        planes.append(FilterPlaneArrival(
            key=str(key), time_s=absolute,
            x_m=values("dispersive_m"), y_m=values("non_dispersive_m"),
            tx_rad=values("tx_rad"), ty_rad=values("ty_rad"), reached=reached,
        ))
    return tuple(planes)


@cpu_call("energy_filter")
def simulate_energy_filter(
    state,
    simulation,
    inelastic_distribution=None,
    *,
    eftem_source_image=None,
):
    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    if not energy_filter.enabled:
        return None
    source_metrics = dict(getattr(simulation, "metrics", {}) or {})
    entrance_provenance = str(source_metrics.get(
        "energy_filter_entrance_provenance", "optical_column_reference"))
    entrance_dependency_signature = str(source_metrics.get(
        "energy_filter_entrance_dependency_signature", ""))
    entrance_signal = measure_aperture_transmitted_current(
        simulation,
        state,
        state.energy_filter_entrance_aperture,
    )
    eels_forward = None
    eftem_image = None
    if inelastic_distribution is not None:
        # This spectrum consumes the exact immutable distribution already
        # produced by the shared specimen-interaction result.  It is
        # normalised per electron at the specimen boundary; the separate ray
        # counters below remain authoritative for physical entrance/output
        # interception until an energy-resolved Boris transfer is available.
        from temsim.detector.eels_forward import simulate_eels_forward

        eels_forward = simulate_eels_forward(
            state,
            incident_electrons=1.0,
            simulation=simulation,
            distribution=inelastic_distribution,
        )
        if (
            str(getattr(energy_filter, "operating_mode", "eels")).lower()
            == "eftem"
            and eftem_source_image is not None
        ):
            from temsim.detector.eels_forward import (
                eftem_image_from_shared_spectrum,
            )

            eftem_image = eftem_image_from_shared_spectrum(
                eftem_source_image,
                eels_forward,
            )
    rays, total_entrance_count = _representative_entrance_rays(
        extract_entrance_rays(state, simulation),
        energy_filter.maximum_trace_rays,
    )
    if not rays:
        return EnergyFilterResult(
            paths_u_mm=[],
            paths_v_mm=[],
            paths_y_mm=[],
            colours=[],
            losses_ev=np.array([]),
            entrance_ray_count=0,
            eels_ray_count=0,
            stopped_at_output_detector=(
                energy_filter.output_detector_inserted
            ),
            entrance_signal=entrance_signal,
            status=(
                "No TEM rays reached and passed the "
                "Spectrometer entrance aperture"
            ),
            stop_keys=(),
            slit_transmitted_fraction=0.0,
            slit_transmitted_current_pa=0.0,
            camera_recorded_fraction=0.0,
            camera_recorded_current_pa=0.0,
            eels_transmitted_fraction=0.0,
            eels_transmitted_current_pa=0.0,
            eels_forward=eels_forward,
            eftem_image=eftem_image,
            source_ray_id=np.empty(0, dtype=np.int64),
            source_fraction=np.empty(0, dtype=np.float64),
            source_path_index=np.empty(0, dtype=np.int64),
            entrance_time_s=np.empty(0, dtype=np.float64),
            entrance_provenance=entrance_provenance,
            entrance_dependency_signature=entrance_dependency_signature,
        )

    reference_batch = _trace_energy_filter_reference(state)
    batch = trace_energy_filter_batch(
        state,
        [ray.x_mm for ray in rays],
        [ray.y_mm for ray in rays],
        [ray.tx_rad for ray in rays],
        [ray.ty_rad for ray in rays],
        [ray.energy_offset_ev for ray in rays],
    )
    reference_paths = _paths_from_batch(reference_batch)
    ray_paths = _paths_from_batch(batch)
    paths_u = reference_paths[0] + ray_paths[0]
    paths_v = reference_paths[1] + ray_paths[1]
    paths_y = reference_paths[2] + ray_paths[2]
    weights = np.asarray(
        [ray.source_fraction for ray in rays],
        dtype=float,
    )
    transmitted_fraction = float(weights[batch.passed_slit].sum())
    transmitted_current_pa = (
        source_current_pa(state) * transmitted_fraction
    )
    camera_recorded = (
        batch.stop_key == "energy_filter_output_detector"
    )
    camera_fraction = float(weights[camera_recorded].sum())
    eels_fraction = float(weights[batch.zebra_recorded].sum())
    source_pa = source_current_pa(state)
    eels_count = int(np.count_nonzero(batch.zebra_recorded))
    stop_counts = {
        str(key): int(np.count_nonzero(batch.stop_key == key))
        for key in np.unique(batch.stop_key)
        if str(key)
    }
    counts_text = ", ".join(
        f"{key}: {count}"
        for key, count in sorted(stop_counts.items())
    )
    status = (
        f"{eels_count}/{len(rays)} sampled trajectories recorded by EELS camera"
        + (
            f" ({total_entrance_count} entrance rays represented)"
            if total_entrance_count != len(rays)
            else ""
        )
        + (f" | stops: {counts_text}" if counts_text else "")
    )
    entrance_times = np.asarray([ray.flight_time_s for ray in rays], dtype=np.float64)
    return EnergyFilterResult(
        paths_u_mm=paths_u,
        paths_v_mm=paths_v,
        paths_y_mm=paths_y,
        colours=["#263238"] + [ray.colour for ray in rays],
        losses_ev=np.asarray(
            [0.0] + [-ray.energy_offset_ev for ray in rays],
            dtype=float,
        ),
        entrance_ray_count=len(rays),
        eels_ray_count=eels_count,
        stopped_at_output_detector=(
            energy_filter.output_detector_inserted
        ),
        entrance_signal=entrance_signal,
        status=status,
        stop_keys=tuple(str(key) for key in batch.stop_key),
        slit_transmitted_fraction=transmitted_fraction,
        slit_transmitted_current_pa=transmitted_current_pa,
        camera_recorded_fraction=camera_fraction,
        camera_recorded_current_pa=source_pa * camera_fraction,
        eels_transmitted_fraction=eels_fraction,
        eels_transmitted_current_pa=source_pa * eels_fraction,
        metrics=getattr(energy_filter, "_last_slit_metrics", None),
        eels_forward=eels_forward,
        eftem_image=eftem_image,
        source_ray_id=np.asarray([ray.source_ray_id for ray in rays], dtype=np.int64),
        source_fraction=weights.copy(),
        source_branch=tuple(ray.source_branch for ray in rays),
        source_path_index=np.asarray([ray.source_path_index for ray in rays], dtype=np.int64),
        entrance_time_s=entrance_times,
        timed_planes=_timed_filter_planes(energy_filter, batch, entrance_times),
        entrance_provenance=entrance_provenance,
        entrance_dependency_signature=entrance_dependency_signature,
    )
