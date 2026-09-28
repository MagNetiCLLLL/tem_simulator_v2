"""Conditional downstream responses for a non-wave projected STEM readout.

These quadrature paths start from executed incident ray coordinates. They are
an auxiliary intensity response, never an executed material state/checkpoint.
The caller supplies per-pixel direct/element probabilities and source survival
exactly once, then applies recording-plane absorption in physical order.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
import math
from threading import RLock

import numpy as np

from temsim.specimen.downstream_transport import (
    GeometricSpecimenExit, build_geometric_specimen_exit,
)
from temsim.specimen.elastic_transport import (
    ElasticTerminalBundle, ElasticTransportResult, incident_rays_from_simulation,
    rotate_direction_after_scatter, screened_rutherford_parameter,
    screened_rutherford_total_cross_section_cm2,
)
from temsim.specimen.projected_scattering import screened_angular_quadrature


DIRECT_RAY_LIMIT = 64
POLAR_BIN_COUNT = 32
AZIMUTH_SAMPLE_COUNT = 16
MODEL = "projected_conditional_real_column_response_v1"
RESPONSE_CACHE_BYTES = 64 * 1024**2
RESPONSE_CACHE_ENTRIES = 4
_RESPONSE_CACHE = OrderedDict()
_RESPONSE_CACHE_SIZE = 0
_CACHE_LOCK = RLock()


def _response_identity(state, incident, simulation, projection, distribution, stop, saved):
    """Reuse only a declared instrument inventory, never guessed fixture state."""
    from temsim.optics.model import State
    if not isinstance(state, State) or (distribution is not None and not is_dataclass(distribution)):
        return None
    from temsim.calculation_cache import calculation_signatures
    from temsim.physics.ray_identity import source_identity
    payload = {"model": MODEL, "downstream": calculation_signatures(state)["sample_downstream"],
               "inelastic": None if distribution is None else asdict(distribution),
               "stop": stop, "saved": saved,
               "quadrature": (DIRECT_RAY_LIMIT, POLAR_BIN_COUNT, AZIMUTH_SAMPLE_COUNT)}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    arrays = [np.array([(ray.source_ray_index, *ray.position_xy_nm, *ray.direction,
                        ray.kinetic_energy_ev, ray.weight) for ray in incident.rays])]
    arrays.extend(source_identity(simulation.incident))
    for element in projection.elements:
        arrays.extend((np.array([element.atomic_number]), element.energy_ev, element.cross_section_weights))
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(repr((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _store_responses(identity, responses):
    global _RESPONSE_CACHE_SIZE
    if identity is None:
        return
    size = 4096 + sum(len(repr(value.metrics).encode()) for value in responses.values())
    size += sum(item.nbytes for value in responses.values() for branch in value.branches
                for item in vars(branch).values() if isinstance(item, np.ndarray))
    if size > RESPONSE_CACHE_BYTES:
        return
    stored = deepcopy(responses)
    with _CACHE_LOCK:
        previous = _RESPONSE_CACHE.pop(identity, None)
        if previous is not None:
            _RESPONSE_CACHE_SIZE -= previous[1]
        _RESPONSE_CACHE[identity] = (stored, size)
        _RESPONSE_CACHE_SIZE += size
        while (_RESPONSE_CACHE_SIZE > RESPONSE_CACHE_BYTES
               or len(_RESPONSE_CACHE) > RESPONSE_CACHE_ENTRIES):
            _, (_, old_size) = _RESPONSE_CACHE.popitem(last=False)
            _RESPONSE_CACHE_SIZE -= old_size


def _systematic_rows(weights, count, *, phase=0.5):
    """Deterministic probability strata; no global random state or row crop."""
    weights = np.asarray(weights, dtype=float)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Projected response needs positive incident probability")
    cdf = np.cumsum(weights / total)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, (np.arange(count) + phase) / count, side="right")


def _direct_rows(rays):
    weights = np.array([ray.weight for ray in rays])
    if len(rays) <= DIRECT_RAY_LIMIT:
        return np.arange(len(rays)), weights
    chosen = _systematic_rows(weights, DIRECT_RAY_LIMIT)
    rows, counts = np.unique(chosen, return_counts=True)
    return rows, counts / float(DIRECT_RAY_LIMIT)


def _polar_edges(state=None, stop=None):
    """Logarithmic angle grid enriched at geometric collection boundaries.

    The hints are exact for a centred field-free detector. With column fields
    they merely add resolution; the propagated physical hit masks still own
    every collection decision.
    """
    hints = [math.pi/2]
    if state is not None:
        start = float(state.sample.z_mm)
        for plane in getattr(state, "recording_planes", ()):
            distance = float(plane.z_mm)-start
            if (distance <= 0 or (stop is not None and float(plane.z_mm) > stop)
                    or not bool(getattr(plane, "inserted", False))):
                continue
            for field in ("inner_diameter_mm", "outer_width_mm"):
                radius = float(getattr(plane, field, 0.0))*.5
                if math.isfinite(radius) and radius > 0:
                    hints.append(math.atan2(radius, distance))
    return np.unique(np.r_[0.0, np.geomspace(1.0e-6, math.pi, POLAR_BIN_COUNT), hints])


def _angular_rows(rays, element, edges=None):
    """512 stratified joint energy/direction/position samples per element.

    Each angular-bin mass uses the full executed energy mixture. Within that
    bin the actual incident ray is selected with weight w*sigma(E)*Pbin(E),
    retaining its energy, position and direction together. The polar node is
    that selected energy's conditional median, avoiding independent mixing of
    a ray's energy with a polar law belonging to a different electron.
    """
    edges = _polar_edges() if edges is None else np.asarray(edges)
    _, phi, weights = screened_angular_quadrature(
        element, edges, azimuth_samples=AZIMUTH_SAMPLE_COUNT)
    energies = np.array([ray.kinetic_energy_ev for ray in rays])
    incident_weights = np.array([ray.weight for ray in rays])
    cross_sections = np.array([
        screened_rutherford_total_cross_section_cm2(element.atomic_number, energy)
        for energy in energies])
    delta = np.array([screened_rutherford_parameter(element.atomic_number, energy)
                      for energy in energies])
    u = np.sin(edges * 0.5)**2
    rows, angles = [], []
    keep = np.zeros(len(weights), dtype=bool)
    for index in range(len(edges)-1):
        angular_slice = slice(index*AZIMUTH_SAMPLE_COUNT, (index+1)*AZIMUTH_SAMPLE_COUNT)
        if not np.any(weights[angular_slice] > 0.0):
            # A detector boundary can be only one float64 step from an
            # existing angle. A zero CDF mass has no conditional source to
            # sample; omit its nodes without changing any positive mass.
            continue
        keep[angular_slice] = True
        # Algebraic CDF difference avoids cancellation: the mixture may
        # retain a tiny positive bin even when each direct CDF subtraction
        # rounds to zero. Preserve that mass and its energy conditioning.
        lo, hi = u[index], u[index+1]
        probabilities = (1.0 + delta)*delta*(hi-lo)/((hi+delta)*(lo+delta))
        conditional = incident_weights * cross_sections * probabilities
        # A ring-dependent cyclic permutation prevents one source stratum
        # from always being assigned the same laboratory azimuth.
        picked = _systematic_rows(conditional, AZIMUTH_SAMPLE_COUNT)
        picked = np.roll(picked, (index * 7) % AZIMUTH_SAMPLE_COUNT)
        sine_squared = lo+(hi-lo)*(lo+delta[picked])/(lo+hi+2.0*delta[picked])
        rows.extend(picked)
        angles.extend(2.0 * np.arcsin(np.sqrt(np.clip(sine_squared, 0.0, 1.0))))
    return np.asarray(rows), np.asarray(angles), phi[keep], weights[keep]


def _terminal(rays, rows, weights, thickness_nm, *, theta=None, phi=None):
    positions = np.array([(*rays[index].position_xy_nm, 0.0) for index in rows])
    directions = np.array([rays[index].direction for index in rows])
    scattered = theta is not None
    if scattered:
        directions = np.array([rotate_direction_after_scatter(direction, polar, azimuth)
                               for direction, polar, azimuth in zip(directions, theta, phi, strict=True)])
    count = len(rows)
    return ElasticTerminalBundle(
        source_ray_index=np.array([rays[index].source_ray_index for index in rows]),
        position_nm=positions,
        direction=directions,
        kinetic_energy_ev=np.array([rays[index].kinetic_energy_ev for index in rows]),
        weight=np.asarray(weights),
        outcome=tuple("transmitted" if direction[2] > 1e-12 else "backscattered"
                      for direction in directions),
        event_count=np.full(count, int(scattered)),
        has_scattered=np.full(count, scattered),
        material_path_nm=np.full(count, thickness_nm),
        # The thin projection has no executed event depth/time. Unknown is
        # retained rather than advertising an invented material flight clock.
        reference_time_offset_s=None,
    )


def _conditional_simulation(simulation, incident):
    result = copy(simulation)
    branch = copy(simulation.incident)
    count = np.size(branch.alive)
    weights = np.zeros(count)
    alive = np.zeros(count, dtype=bool)
    for ray in incident.rays:
        weights[ray.source_ray_index] = ray.weight
        alive[ray.source_ray_index] = True
    branch.ray_weight, branch.alive = weights, alive
    result.incident = branch
    return result


def _observation_planes(state, stop_z_mm):
    start = float(state.sample.z_mm)
    planes = tuple(float(plane.z_mm) for plane in getattr(state, "recording_planes", ())
                   if bool(getattr(plane, "inserted", False)) and float(plane.z_mm) >= start)
    if stop_z_mm is None:
        detectors = tuple(float(plane.z_mm) for plane in getattr(state, "stem_detectors", ())
                          if bool(getattr(plane, "inserted", False)) and float(plane.z_mm) >= start)
        if not detectors:
            raise ValueError("Insert a downstream STEM detector or specify a projected-response stop plane")
        stop_z_mm = max(detectors)
    stop = float(stop_z_mm)
    if not math.isfinite(stop) or stop < start:
        raise ValueError("Projected-response stop must be finite and at or beyond the sample")
    energy_filter = getattr(state, "energy_filter", None)
    installed = bool(getattr(state, "energy_filter_installed", False))
    if installed or bool(getattr(energy_filter, "enabled", False)):
        entrance = float(getattr(energy_filter, "entrance_z_mm", math.nan))
        if not math.isfinite(entrance):
            raise ValueError("Installed energy filter has no finite entrance plane")
        if stop >= entrance:
            raise ValueError("Projected STEM responses cannot yet traverse the installed energy filter; "
                             "choose a detector before its entrance")
    apertures = tuple(float(part.z_mm) for part in getattr(state, "apertures", ())
                      if bool(getattr(part, "installed", True)) and bool(getattr(part, "enabled", True)))
    saved = tuple(sorted({start, stop, *(z for z in (*planes, *apertures) if start <= z <= stop)}))
    return stop, saved


def _compact_response(response, saved, key):
    branches = []
    for branch in response.branches:
        z = np.asarray(branch.z)
        keep = {0, len(z) - 1}
        for plane in saved:
            rows = np.flatnonzero(np.isclose(z, plane, rtol=0.0, atol=1e-12))
            if not rows.size:
                raise RuntimeError("Projected response lacks an exact requested detector/stop plane")
            keep.add(int(rows[-1]))
        # Retain the two bracketing samples of every physical intercept while
        # the original blocked_z/key/alive arrays retain its exact identity.
        for stop in np.asarray(branch.blocked_z)[np.isfinite(branch.blocked_z)]:
            hi = min(int(np.searchsorted(z, stop)), len(z) - 1)
            keep.update((max(hi - 1, 0), hi))
        rows = np.array(sorted(keep))
        changes = {name: np.asarray(getattr(branch, name))[rows].copy()
                   for name in ("z", "x", "y", "tx", "ty")}
        for name in ("flight_time_s", "kinetic_energy_ev"):
            value = getattr(branch, name, None)
            if value is not None:
                changes[name] = np.asarray(value)[rows].copy()
        branches.append(replace(branch, **changes))
    metrics = dict(response.metrics)
    metrics.update({
        "model": MODEL, "projected_response": key,
        "auxiliary_intensity_response": True, "checkpoint": False,
        "probability_reference": "unit probability conditional on current reaching the sample",
        "source_survival_applied": False, "projection_probability_applied": False,
        "recording_interceptions_deferred": True,
        "inelastic_geometry_scope": "representative nominal thickness of thin projected specimen",
        "flight_time_model": "unknown projected event depth; no material clock fabricated",
        "sample_downstream_signature": "",
        "direct_ray_limit": DIRECT_RAY_LIMIT, "polar_bins": POLAR_BIN_COUNT,
        "azimuth_samples": AZIMUTH_SAMPLE_COUNT,
        "polar_collection_boundary_hints": True,
        "numerical_domain_truncated_conditional_probability": math.fsum(
            float(branch.weight) * math.fsum(np.asarray(branch.ray_weight)[
                np.asarray(branch.blocked_key) == "projected_field_domain"])
            for branch in branches),
        "numerical_truncation_scope": "before per-pixel recording absorption; count only still-uncollected rays",
        "limitations": ("fixed reference probe; scan-dependent aperture/wall changes need retracing",
                        "bounded joint incident/angular quadrature; finite lateral correlations",
                        "thin projection with independent aggregate inelastic population",
                        "recording absorption is owned by the per-pixel physical-plane collector"),
    })
    return GeometricSpecimenExit(tuple(branches), metrics)


def build_projected_responses(state, simulation, projection, real_interactions=None, *,
                              stop_z_mm=None, progress_callback=None):
    """Return direct, vacuum and Z<number> real-column intensity responses.

    All conditional terminal distributions sum to one including backscatter.
    Returned branch sums may be smaller because backward/absorbed populations
    are kept in the probability ledger and never renormalized into detection.
    The bounded cache binds declared downstream hardware, actual incident
    rows, angular energy mixture and the full inelastic distribution. Returned
    compact responses are detached copies; a consumer cannot alter the cache.
    """
    stop, saved = _observation_planes(state, stop_z_mm)
    incident = incident_rays_from_simulation(state, simulation)
    rays = incident.rays
    identity = _response_identity(state, incident, simulation, projection, real_interactions, stop, saved)
    if identity is not None:
        with _CACHE_LOCK:
            hit = _RESPONSE_CACHE.get(identity)
            if hit is not None:
                _RESPONSE_CACHE.move_to_end(identity)
                result = deepcopy(hit[0])
                for value in result.values():
                    value.metrics["response_cache_hit"] = True
                if progress_callback is not None:
                    progress_callback(1, 1, "Reused projected downstream responses")
                return result
    conditional = _conditional_simulation(simulation, incident)
    thickness = float(state.sample.thickness_nm)
    if not math.isfinite(thickness) or thickness < 0:
        raise ValueError("Projected material thickness must be finite and nonnegative")
    direct_rows, direct_weights = _direct_rows(rays)
    terminals = [("direct", _terminal(rays, direct_rows, direct_weights, thickness)),
                 ("vacuum", _terminal(rays, direct_rows, direct_weights, 0.0))]
    edges = _polar_edges(state, stop)
    for element in projection.elements:
        rows, theta, phi, weights = _angular_rows(rays, element, edges)
        terminals.append((f"Z{element.atomic_number}",
                          _terminal(rays, rows, weights, thickness, theta=theta, phi=phi)))
    result = {}
    for index, (key, terminal) in enumerate(terminals):
        if progress_callback is not None:
            progress_callback(index, len(terminals), f"Propagating projected {key} response")
        response = build_geometric_specimen_exit(
            state, conditional,
            ElasticTransportResult(eds_tracks=(), trajectories=(), metrics={"model": MODEL},
                                   terminal_electrons=terminal),
            None if key == "vacuum" else real_interactions,
            save_z_mm=saved, stop_z_mm=stop,
            capture_sections=False, recording_interceptions=False,
        )
        result[key] = _compact_response(response, saved, key)
        result[key].metrics.update(response_cache_hit=False,
                                   response_cache_identity=identity or "",
                                   response_cache_scope="sample_downstream + actual incident + angular + inelastic")
    if progress_callback is not None:
        progress_callback(len(terminals), len(terminals), "Projected responses complete")
    _store_responses(identity, result)
    return result
