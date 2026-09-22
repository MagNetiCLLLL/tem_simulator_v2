"""Finite-specimen electron exit states for geometric downstream imaging.

The elastic Monte Carlo solver reports a conditional distribution for the
electrons that reached the specimen.  This adapter combines that finite-
geometry exit distribution with the existing exclusive inelastic populations,
then converts it to source-normalised ray branches for the ordinary downstream
column propagator.  It is deliberately excluded from multislice STEM, where
the wave solver already owns coherent elastic angular redistribution.
"""

from __future__ import annotations

from temsim.specimen.vector_field_transport import SpecimenFieldTransport

from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np

from temsim.physics.chromatic import (
    configured_objective_chromatic_focal_mm,
    objective_chromatic_kick_from_state,
)
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate
from temsim.physics.recording_clipping import clip_recording_planes
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.simulation import Branch, RAY_INTERACTION_COLOURS
from temsim.specimen.inelastic import real_inelastic_ray_branches
from temsim.specimen.axial_field_transport import (
    sample_axial_field_diagnostic,
)


ProgressCallback = Callable[[int, int, str], None]

_REQUIRED_SPECIMEN_EXIT_METRICS = (
    "tracked_downstream_source_probability",
    "inelastic_absorbed_source_probability",
)
_PROBABILITY_LEDGER_ABS_TOL = 5.0e-12
_PROBABILITY_LEDGER_REL_TOL = 1.0e-10


def _bounded_probability(value: float) -> float:
    """Bound only roundoff in an already validated probability calculation."""
    value = float(value)
    if (not math.isfinite(value) or value < 0.0
            or value > 1.0 + _PROBABILITY_LEDGER_ABS_TOL):
        raise ValueError("Calculated specimen-exit probability is outside [0, 1]")
    return min(1.0, value)


@dataclass(frozen=True, slots=True)
class GeometricSpecimenExit:
    """Source-normalised specimen-exit branches and their probability ledger."""

    branches: tuple[Branch, ...]
    metrics: dict[str, object]
    dependency_signature: str = ""
    checkpoints: tuple[object, ...] = ()
    segments: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        signature = str(self.dependency_signature)
        metrics = dict(self.metrics)
        metric_signature = str(
            metrics.get("sample_downstream_signature", "")
        )
        if signature and metric_signature and metric_signature != signature:
            raise ValueError(
                "Specimen-exit metrics cannot be relabelled with a different "
                "downstream dependency signature"
            )
        if not signature:
            signature = metric_signature
        metrics["sample_downstream_signature"] = signature
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "dependency_signature", signature)
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "segments", tuple(self.segments))


def validated_geometric_specimen_exit(
    value,
    expected_signature: str,
) -> GeometricSpecimenExit | None:
    """Return a downstream checkpoint only when its own provenance matches."""

    expected = str(expected_signature)
    if not expected or not isinstance(value, GeometricSpecimenExit):
        return None
    metric_signature = str(
        value.metrics.get("sample_downstream_signature", "")
    )
    if value.dependency_signature != expected or metric_signature != expected:
        return None
    probabilities: list[float] = []
    for key in _REQUIRED_SPECIMEN_EXIT_METRICS:
        try:
            metric = float(value.metrics[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            return None
        probabilities.append(metric)
    if math.fsum(probabilities) > 1.0 + _PROBABILITY_LEDGER_ABS_TOL:
        return None

    branch_weights: list[float] = []
    for branch in value.branches:
        try:
            weight = float(branch.weight)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(weight) or weight < 0.0:
            return None
        branch_weights.append(weight)
    try:
        branch_weight_sum = math.fsum(branch_weights)
    except OverflowError:
        return None
    if not math.isfinite(branch_weight_sum) or not math.isclose(
        branch_weight_sum,
        probabilities[0],
        rel_tol=_PROBABILITY_LEDGER_REL_TOL,
        abs_tol=_PROBABILITY_LEDGER_ABS_TOL,
    ):
        return None
    return value


def _post_sample_events(state) -> tuple[tuple[float, float, float], ...]:
    events: list[tuple[float, float, float]] = []
    for collection_name in ("deflectors", "corrector_elements"):
        for component in getattr(state, collection_name, ()):
            if not bool(getattr(component, "enabled", False)):
                continue
            if not hasattr(component, "kick_events"):
                continue
            try:
                component_events = component.kick_events(
                    time_s=float(getattr(state, "simulation_time_s", 0.0))
                )
            except TypeError:
                component_events = component.kick_events()
            events.extend(
                (float(z), float(dx), float(dy))
                for z, dx, dy in component_events
                if float(z) > float(state.sample.z_mm)
            )
    return tuple(sorted(events))


def _sample_incident_fraction(simulation) -> float:
    branch = simulation.incident
    alive = np.asarray(branch.alive, dtype=bool)
    if alive.ndim != 1:
        raise ValueError("Incident sample survival mask must be one-dimensional")
    ray_weight = getattr(branch, "ray_weight", None)
    if ray_weight is None:
        weights = np.full(alive.size, 1.0 / max(alive.size, 1), dtype=float)
    else:
        weights = np.asarray(ray_weight, dtype=float)
    if (weights.shape != alive.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)):
        raise ValueError("Incident sample ray weights do not match the ray bundle")
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Incident sample ray weights must have positive finite sum")
    return _bounded_probability(math.fsum(weights[alive]) / total)


def _inelastic_specs(distribution, source_ray_count: int):
    if distribution is None:
        return (
            ("zero_loss", 0.0, 0.0, 1.0, "real_zero_loss", 0.0),
        ), 1.0, 0.0
    branches = real_inelastic_ray_branches(
        distribution,
        ray_count=source_ray_count,
    )
    return (
        tuple(
            (
                branch.name,
                branch.kick_x_rad,
                branch.kick_y_rad,
                float(branch.probability),
                branch.interaction_kind,
                float(branch.energy_loss_ev),
            )
            for branch in branches
        ),
        float(distribution.tracked_probability),
        float(distribution.absorbed_probability),
    )


def _incident_reference_times(simulation, reference_z_mm, ray_count):
    """Read an executed reference-plane clock, never interpolate plot samples."""
    times = getattr(simulation.incident, "flight_time_s", None)
    z = np.asarray(getattr(simulation.incident, "z", ()), dtype=float)
    unknown = np.full(ray_count, np.nan, dtype=np.float64)
    if times is None or z.ndim != 1:
        return unknown
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (z.size, ray_count):
        raise ValueError("Incident flight-time history does not match its ray history")
    rows = np.flatnonzero(np.isclose(z, reference_z_mm, rtol=0., atol=1e-12))
    if not rows.size:
        return unknown
    result = times[rows[-1]].copy()
    result[~np.isfinite(result) | (result < 0.)] = np.nan
    return result


def _material_restart_identity(state, terminal, source_ids, source_azimuths,
                               terminal_times, specs, source_fraction):
    """Bind restart data to the executed material and its physical inputs."""
    from temsim.calculation_cache import calculation_signatures
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.particle_sections import _digest_arrays
    values = [getattr(terminal, key) for key in (
        "source_ray_index", "position_nm", "direction", "kinetic_energy_ev",
        "weight", "event_count", "has_scattered")]
    values.extend((np.asarray(terminal.outcome, dtype=str), source_ids,
                   source_azimuths, terminal_times, np.asarray(source_fraction)))
    for key in ("reference_time_offset_s", "material_path_nm"):
        value = getattr(terminal, key, None)
        values.append(np.asarray([] if value is None else value))
    for name, kx, ky, probability, kind, loss in specs:
        values.extend((np.asarray((name, kind), dtype=str), np.asarray(kx),
                       np.asarray(ky), np.asarray((probability, loss))))
    payload = ("executed-material-branch-restart-v1", solver_source_identity(),
               calculation_signatures(state)["elastic"], _digest_arrays(*values))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _material_previous_segment(value, count):
    """Reject incomplete historical/display arrays as continuation states."""
    from temsim.physics.particle_sections import ParticleSectionSegment
    if not isinstance(value, ParticleSectionSegment):
        return None
    try:
        cp, branch, plan = value.checkpoints, value.branch, value.plan
        z, indices = np.asarray(cp.z_mm), np.asarray(plan.checkpoint_index)
        if (not z.size or not np.array_equal(np.asarray(plan.z_mm)[indices], z)
                or branch.z[0] != plan.z_mm[0] or branch.z[-1] != plan.z_mm[-1]
                or z[-1] != branch.z[-1]):
            return None
        for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
            array = np.asarray(getattr(cp, key))
            if array.dtype != np.float64 or array.shape != (len(z), count):
                return None
            if key == "flight_time_s":
                if np.any(np.isinf(array)) or np.any(array < 0.):
                    return None
            elif not np.all(np.isfinite(array)):
                return None
        for key in ("x", "tx", "y", "ty", "flight_time_s"):
            if np.shape(getattr(branch, key)) != (len(branch.z), count):
                return None
        for key in ("source_ray_id", "source_azimuth_rad", "ray_weight", "energy_offset_ev",
                    "alive", "blocked_z"):
            if np.shape(getattr(branch, key)) != (count,):
                return None
        for key, cp_key in (("x", "x_m"), ("tx", "tx_rad"), ("y", "y_m"), ("ty", "ty_rad")):
            if not np.array_equal(getattr(branch, key)[-1], getattr(cp, cp_key)[-1]):
                return None
        return value
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def build_geometric_specimen_exit(
    state,
    simulation,
    elastic_transport,
    inelastic_distribution=None,
    *,
    save_z_mm: tuple[float, ...] = (),
    stop_z_mm: float | None = None,
    dependency_signature: str = "",
    progress_callback: ProgressCallback | None = None,
    existing_exit: GeometricSpecimenExit | None = None,
    tuning_component_keys: tuple[str, ...] = (),
) -> GeometricSpecimenExit:
    """Combine elastic exit phase space and exclusive inelastic populations.

    The coupling is an explicit independence approximation: the finite-envelope
    screened-Rutherford terminal distribution is multiplied by the aggregate
    Poisson energy-loss distribution.  This produces one conserved electron
    population; elastic and inelastic branch weights are never added together.
    """

    stop_z_mm = float(determine_tem_stop_z(state) if stop_z_mm is None else stop_z_mm)
    if not math.isfinite(stop_z_mm) or stop_z_mm < float(state.sample.z_mm):
        raise ValueError("Specimen downstream stop must be finite and at or after its reference plane")
    save_z_mm = tuple(float(value) for value in save_z_mm
                      if float(state.sample.z_mm) <= float(value) <= stop_z_mm)
    terminal = getattr(elastic_transport, "terminal_electrons", None)
    source_fraction = _sample_incident_fraction(simulation)
    empty_metrics = {
        "model": "finite_geometry_elastic_x_inelastic_tensor_product",
        "sample_incident_source_probability": source_fraction,
        "elastic_forward_conditional_probability": 0.0,
        "elastic_nontransmitted_conditional_probability": 1.0,
        "inelastic_tracked_conditional_probability": 1.0,
        "inelastic_absorbed_conditional_probability": 0.0,
        "tracked_downstream_source_probability": 0.0,
        "inelastic_absorbed_source_probability": 0.0,
        "elastic_nontransmitted_source_probability": source_fraction,
        "pre_sample_lost_source_probability": 1.0 - source_fraction,
        "downstream_forward_weight": 0.0,
        "downstream_source_ray_count": 0,
        "exit_plane_source_probability": 0.0,
        "exit_plane_weight": 0.0,
        "post_sample_reinjection_plane_z_mm": float(state.sample.z_mm),
        "material_terminal_z_collapsed_to_reference_plane": False,
        "terminal_state_projection_model": (
            "inverse shared-vector-field reference-plane matching"
        ),
        "terminal_state_projection_preserves_free_flight_line": False,
        "source_probability_conservation_error": 0.0,
        "source_probability_conserved": True,
        "geometric_reference_point_only": True,
        "pixel_resolved_specimen_contrast": False,
        "used_by_wave_multislice": False,
    }
    if terminal is None or len(terminal.outcome) == 0:
        if progress_callback is not None:
            progress_callback(1, 1, "No transmitted specimen-exit electrons")
        return GeometricSpecimenExit(
            (),
            empty_metrics,
            dependency_signature=dependency_signature,
        )

    positions_nm = np.asarray(terminal.position_nm, dtype=float)
    directions = np.asarray(terminal.direction, dtype=float)
    weights = np.asarray(terminal.weight, dtype=float)
    energies = np.asarray(terminal.kinetic_energy_ev, dtype=float)
    outcomes = np.asarray(terminal.outcome, dtype=object)
    scattered = np.asarray(terminal.has_scattered, dtype=bool)
    source_indices = np.asarray(terminal.source_ray_index, dtype=np.int64)
    count = len(outcomes)
    if (
        positions_nm.shape != (count, 3)
        or directions.shape != (count, 3)
        or weights.shape != (count,)
        or energies.shape != (count,)
        or scattered.shape != (count,)
        or source_indices.shape != (count,)
        or not np.all(np.isfinite(positions_nm))
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(energies))
        or np.any(weights < 0.0)
    ):
        raise ValueError("Elastic terminal bundle arrays are inconsistent")
    total_terminal_weight = math.fsum(weights)
    material_paths = getattr(terminal, "material_path_nm", None)
    if material_paths is None:
        raise ValueError("Elastic terminal bundle requires executed material_path_nm for every history")
    material_paths_known = True
    material_paths = np.asarray(material_paths, dtype=float)
    if material_paths.shape != (count,) or not np.all(np.isfinite(material_paths)) or np.any(material_paths < 0.0):
        raise ValueError("Terminal material paths must match all elastic histories")
    material_touched = material_paths > 0.0
    if not math.isclose(
        total_terminal_weight, 1.0, rel_tol=0.0, abs_tol=2.0e-12
    ):
        raise ValueError("Elastic terminal conditional weights must sum to one")
    # Correct only admitted floating-point normalisation error. The executed
    # terminal bundle stays unchanged; all derived groups share this local
    # conditional population, including the nontransmitted complement.
    weights = weights / total_terminal_weight

    source_ray_count = int(np.asarray(simulation.incident.alive).size)
    if (
        source_ray_count <= 0
        or np.any(source_indices < 0)
        or np.any(source_indices >= source_ray_count)
    ):
        raise ValueError("Elastic terminal source indices are outside the incident bundle")
    from temsim.physics.ray_identity import select_identity, source_identity

    source_ids, source_azimuths = source_identity(simulation.incident)
    incident_times = _incident_reference_times(
        simulation, float(state.sample.z_mm), source_ray_count)
    reference_offsets = getattr(terminal, "reference_time_offset_s", None)
    terminal_times = np.full(count, np.nan, dtype=np.float64)
    if reference_offsets is not None:
        reference_offsets = np.asarray(reference_offsets, dtype=np.float64)
        if reference_offsets.shape != (count,):
            raise ValueError("Specimen terminal timing must retain one value per history")
        terminal_times = incident_times[source_indices] + reference_offsets
        terminal_times[~np.isfinite(terminal_times) | (terminal_times < 0.)] = np.nan
    specs, inelastic_tracked, inelastic_absorbed = _inelastic_specs(
        inelastic_distribution,
        source_ray_count,
    )
    if not math.isclose(
        inelastic_tracked + inelastic_absorbed,
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        raise ValueError("Inelastic tracked and absorbed probabilities must sum to one")

    eligible = (outcomes == "transmitted") & (directions[:, 2] > 1.0e-12)
    material_forward_weight = _bounded_probability(math.fsum(weights[eligible & material_touched]))
    vacuum_forward_weight = _bounded_probability(math.fsum(weights[eligible & ~material_touched]))
    if vacuum_forward_weight > 0.0:
        specs = (*specs, ("vacuum_miss", 0.0, 0.0, 1.0, "vacuum", 0.0))
    positive_specs = sum(
        float(spec[3]) > 0.0
        for spec in specs
    )
    progress_total = max(1 + 2 * positive_specs, 1)
    progress_completed = 0
    if progress_callback is not None:
        progress_callback(
            0,
            progress_total,
            "Preparing specimen-exit phase space",
        )

    forward_weight = _bounded_probability(math.fsum(weights[eligible]))
    nontransmitted_weight = _bounded_probability(math.fsum(weights[~eligible]))
    nominal_energy_ev = float(state.beam_voltage_kv) * 1000.0
    if np.any(eligible) and np.any(float(state.sample.z_mm) + positions_nm[eligible, 2]*1e-6 > stop_z_mm + 1e-12):
        raise ValueError("Requested section lies inside finite specimen/support transport; choose a plane beyond the material exit")
    events = tuple(event for event in _post_sample_events(state)
                   if float(state.sample.z_mm) < float(event[0]) <= stop_z_mm)
    chromatic_focal_mm = configured_objective_chromatic_focal_mm(state)
    field_diagnostic = sample_axial_field_diagnostic(state)
    field_transport = SpecimenFieldTransport(state)
    branches: list[Branch] = []
    segments = []
    resume_records = []
    # Historical lightweight adapters and participating stochastic vacuum use
    # the original complete transport. Vacuum restart needs collision/RNG state.
    sectional = (callable(getattr(state, "to_dict", None))
                 and not state.vacuum_map.enabled)
    previous_segments = {}
    if sectional:
        from temsim.physics.particle_sections import _selection_boundary, _trace_segment
        boundary = _selection_boundary(state, tuple(tuning_component_keys))
        restart_identity = _material_restart_identity(
            state, terminal, source_ids, source_azimuths, terminal_times, specs, source_fraction)
        if isinstance(existing_exit, GeometricSpecimenExit):
            previous_segments = {segment.name: segment for segment in existing_exit.segments
                                 if hasattr(segment, "name")}
    exit_plane_source_contributions = []

    if np.any(eligible):
        local_positions = positions_nm[eligible]
        terminal_direction = directions[eligible]
        terminal_weights = weights[eligible]
        terminal_energies = energies[eligible]
        terminal_scattered = scattered[eligible]
        terminal_indices = source_indices[eligible]
        # Terminal Z is specimen-local.  Invert the same local objective-field
        # vector-field transport used inside the finite specimen to obtain a common
        # specimen-reference phase space for the ordinary downstream solver.
        reference_positions = np.empty_like(local_positions)
        reference_directions = np.empty_like(terminal_direction)
        matching_time_offsets = np.empty(len(local_positions), dtype=np.float64)
        for index, (position, direction, energy_ev) in enumerate(zip(
            local_positions,
            terminal_direction,
            terminal_energies,
            strict=True,
        )):
            (reference_positions[index], reference_directions[index],
             matching_time_offsets[index]) = (
                field_transport.to_plane(
                    position,
                    direction,
                    0.0,
                    energy_ev=float(energy_ev),
                    return_elapsed_time=True,
                )
            )
        reference_xy_nm = reference_positions[:, :2]
        elastic_slopes = (
            reference_directions[:, :2]
            / reference_directions[:, 2, None]
        )
        progress_completed = 1
        if progress_callback is not None:
            progress_callback(
                progress_completed,
                progress_total,
                "Projected elastic terminal states to the specimen plane",
            )

        for channel_name, kick_x, kick_y, channel_probability, kind, loss_ev in specs:
            if channel_probability <= 0.0:
                continue
            kick_x_values = np.asarray(kick_x, dtype=float)
            kick_y_values = np.asarray(kick_y, dtype=float)
            if kick_x_values.ndim == 0:
                kick_x_values = np.full(len(terminal_indices), float(kick_x_values))
                kick_y_values = np.full(len(terminal_indices), float(kick_y_values))
            else:
                kick_x_values = kick_x_values[terminal_indices]
                kick_y_values = kick_y_values[terminal_indices]
            outgoing_energy = terminal_energies - loss_ev
            if np.any(outgoing_energy <= 0.0):
                raise ValueError("Inelastic loss leaves a non-positive electron energy")
            energy_offset = outgoing_energy - nominal_energy_ev
            chromatic_x, chromatic_y = objective_chromatic_kick_from_state(
                state,
                reference_xy_nm[:, 0] * 1.0e-9,
                reference_xy_nm[:, 1] * 1.0e-9,
                energy_offset,
                resolved_focal_mm=chromatic_focal_mm,
            )
            start_tx = elastic_slopes[:, 0] + kick_x_values + chromatic_x
            start_ty = elastic_slopes[:, 1] + kick_y_values + chromatic_y
            # The signed inverse matching time initializes a virtual reference
            # coordinate. It is not extra physical travel. Times are published
            # only at/after the actual material terminal plane below.
            # Aggregate loss channels have no event depth, so their time of
            # flight is unknown even though their probability is well defined.
            initial_times = (terminal_times[eligible] + matching_time_offsets
                             if loss_ev == 0. else np.full(len(local_positions), np.nan))
            initial_times[~np.isfinite(initial_times) | (initial_times < 0.)] = np.nan

            for is_scattered, elastic_label in (
                (False, "primary"),
                (True, "elastic"),
            ):
                mask = (terminal_scattered == is_scattered) & (
                    ~material_touched[eligible] if kind == "vacuum" else material_touched[eligible])
                if not np.any(mask):
                    progress_completed += 1
                    if progress_callback is not None:
                        progress_callback(
                            progress_completed,
                            progress_total,
                            f"Propagated {channel_name} {elastic_label} electrons",
                        )
                    continue
                group_weights = terminal_weights[mask]
                group_weight = math.fsum(group_weights)
                absolute_probability = _bounded_probability(
                    source_fraction * group_weight * channel_probability
                )
                if absolute_probability <= 0.0:
                    progress_completed += 1
                    if progress_callback is not None:
                        progress_callback(
                            progress_completed,
                            progress_total,
                            f"Propagated {channel_name} {elastic_label} electrons",
                        )
                    continue
                # Terminal rows may repeat or reorder incident rays. Cache
                # identity follows their original lineage, never compact rows.
                group_source_ids, group_source_azimuths = select_identity(
                    source_ids, source_azimuths, terminal_indices[mask])
                branch_name = f"specimen_{elastic_label}:{channel_name}"
                medium_results = []
                traced_segment = None
                alive = np.ones(np.count_nonzero(mask), dtype=bool)
                blocked = np.full(alive.size, np.nan, dtype=float)
                blocked_keys = [""] * alive.size
                if sectional:
                    traced_segment, resume_z, reused = _trace_segment(
                        state, branch_name, float(state.sample.z_mm), stop_z_mm,
                        (reference_xy_nm[mask, 0] * 1e-9, start_tx[mask],
                         reference_xy_nm[mask, 1] * 1e-9, start_ty[mask], initial_times[mask]),
                        energy_offset[mask], group_weights / group_weight,
                        (group_source_ids, group_source_azimuths),
                        (alive, blocked, blocked_keys), events,
                        _material_previous_segment(previous_segments.get(branch_name), alive.size),
                        boundary, initial_kicks=False, save_z_mm=save_z_mm,
                        dependency_context=(restart_identity, branch_name, absolute_probability),
                    )
                    propagated = traced_segment.branch
                    z, x, tx, y, ty, flight_times = (getattr(propagated, key) for key in
                                                    ("z", "x", "tx", "y", "ty", "flight_time_s"))
                    alive, blocked, blocked_keys = (propagated.alive, propagated.blocked_z,
                                                     propagated.blocked_key)
                    resume_records.append({"branch": branch_name, "hit": bool(reused),
                                           "resume_z_mm": float(resume_z)})
                else:
                    z, x, tx, y, ty, flight_times = propagate(
                        state, float(state.sample.z_mm), stop_z_mm,
                        reference_xy_nm[mask, 0] * 1.0e-9, start_tx[mask],
                        reference_xy_nm[mask, 1] * 1.0e-9, start_ty[mask],
                        events, energy_offset[mask], save_z_mm=save_z_mm,
                        include_initial_plane_kicks=False,
                        particle_medium=state.vacuum_map.enabled, medium_output=medium_results,
                        medium_stream=100+progress_completed,
                        initial_time_s=initial_times[mask], return_flight_times=True,
                    )
                flight_times = np.asarray(flight_times, dtype=np.float64).copy()
                terminal_z_mm = float(state.sample.z_mm) + local_positions[mask, 2] * 1e-6
                flight_times[np.asarray(z)[:, None] < terminal_z_mm[None, :] - 1e-12] = np.nan
                if medium_results:
                    alive, blocked, blocked_keys = medium_results[0].merge_stops(alive, blocked, blocked_keys)
                if traced_segment is None:
                    alive, blocked, blocked_keys = clip_recording_planes(
                        state, z, x, y, alive, blocked, blocked_keys)
                    alive, blocked, blocked_keys = clip_column_wall(
                        state, z, x, y, alive, blocked, blocked_keys)
                flight_times[np.asarray(z)[:, None] > blocked[None, :] + 1e-12] = np.nan
                if save_z_mm:
                    boundary_z_mm = float(save_z_mm[0])
                    reaches_boundary = (
                        np.isnan(blocked)
                        | (blocked > boundary_z_mm + 1.0e-9)
                    )
                    exit_plane_source_contributions.append(absolute_probability *
                        _bounded_probability(math.fsum(group_weights[reaches_boundary]) / group_weight))
                interaction_kind = (
                    "sample_region_elastic"
                    if is_scattered
                    else "sample_region_primary"
                    if kind == "real_zero_loss"
                    else str(kind)
                )
                branches.append(
                    Branch(
                        name=branch_name,
                        colour=RAY_INTERACTION_COLOURS.get(
                            interaction_kind,
                            RAY_INTERACTION_COLOURS["unknown"],
                        ),
                        z=z,
                        x=x,
                        y=y,
                        tx=tx,
                        ty=ty,
                        alive=alive,
                        blocked_z=blocked,
                        blocked_key=blocked_keys,
                        weight=absolute_probability,
                        energy_offset_ev=energy_offset[mask],
                        ray_weight=group_weights / group_weight,
                        interaction_kind=interaction_kind,
                        vacuum_report=medium_results[0].report() if medium_results else None,
                        interaction_kick_x_rad=kick_x_values[mask],
                        interaction_kick_y_rad=kick_y_values[mask],
                        source_ray_id=group_source_ids,
                        source_azimuth_rad=group_source_azimuths,
                        flight_time_s=flight_times,
                    )
                )
                if traced_segment is not None:
                    # Public clocks remain unavailable before physical material
                    # exit and after absorption; solver checkpoint clocks retain
                    # the executed virtual matching coordinate for continuation.
                    segments.append(replace(traced_segment, branch=branches[-1]))
                progress_completed += 1
                if progress_callback is not None:
                    progress_callback(
                        progress_completed,
                        progress_total,
                        f"Propagated {channel_name} {elastic_label} electrons",
                    )

    if progress_callback is not None and progress_completed < progress_total:
        progress_callback(
            progress_total,
            progress_total,
            "Specimen-exit propagation complete",
        )

    tracked_source = _bounded_probability(source_fraction * math.fsum((
        material_forward_weight * inelastic_tracked, vacuum_forward_weight)))
    absorbed_source = _bounded_probability(source_fraction * material_forward_weight * inelastic_absorbed)
    elastic_nontransmitted_source = _bounded_probability(source_fraction * nontransmitted_weight)
    pre_sample_lost = 1.0 - source_fraction
    partition = math.fsum((tracked_source, absorbed_source,
                          elastic_nontransmitted_source, pre_sample_lost))
    conservation_error = partition - 1.0
    branch_weight_sum = math.fsum(branch.weight for branch in branches)
    exit_plane_source_probability = _bounded_probability(math.fsum(exit_plane_source_contributions))
    if not math.isclose(
        branch_weight_sum,
        tracked_source,
        rel_tol=0.0,
        abs_tol=5.0e-12,
    ):
        raise RuntimeError("Geometric specimen-exit branch weights are not conserved")
    if abs(conservation_error) > 5.0e-12:
        raise RuntimeError("Geometric specimen-exit source probability is not conserved")
    metrics = {
        "model": "finite_geometry_elastic_x_inelastic_tensor_product",
        "coupling_approximation": (
            "independent finite-geometry screened-Rutherford terminal state "
            "times exclusive aggregate Poisson energy-loss population"
        ),
        "sample_incident_source_probability": source_fraction,
        "elastic_forward_conditional_probability": forward_weight,
        "elastic_nontransmitted_conditional_probability": nontransmitted_weight,
        "inelastic_tracked_conditional_probability": inelastic_tracked,
        "inelastic_absorbed_conditional_probability": inelastic_absorbed,
        "inelastic_material_path_known": material_paths_known,
        "inelastic_geometry_scope": ("executed_material_paths" if material_paths_known else "historical_nominal_material_approximation"),
        "vacuum_miss_forward_conditional_probability": vacuum_forward_weight,
        "tracked_downstream_source_probability": tracked_source,
        "inelastic_absorbed_source_probability": absorbed_source,
        "elastic_nontransmitted_source_probability": elastic_nontransmitted_source,
        "pre_sample_lost_source_probability": pre_sample_lost,
        "downstream_branch_weight_sum": branch_weight_sum,
        "downstream_forward_weight": forward_weight,
        "downstream_source_ray_count": int(np.count_nonzero(eligible)),
        "exit_plane_source_probability": (
            exit_plane_source_probability if save_z_mm else tracked_source
        ),
        "exit_plane_weight": (
            _bounded_probability((exit_plane_source_probability if save_z_mm else tracked_source)
                                 / source_fraction)
            if source_fraction > 0.0
            else 0.0
        ),
        "post_sample_reinjection_plane_z_mm": float(state.sample.z_mm),
        "material_terminal_z_collapsed_to_reference_plane": False,
        "terminal_state_projection_model": (
            "inverse shared-vector-field reference-plane matching"
        ),
        "source_probability_conservation_error": conservation_error,
        "source_probability_conserved": abs(conservation_error) <= 5.0e-12,
        "terminal_state_projected_to_sample_reference_plane": True,
        "terminal_state_projection_preserves_free_flight_line": False,
        "sample_axial_field_t": field_diagnostic.total_field_t,
        "sample_objective_field_t": field_diagnostic.objective_field_t,
        "geometric_reference_point_only": True,
        "pixel_resolved_specimen_contrast": False,
        "used_by_wave_multislice": False,
        "downstream_stop_z_mm": stop_z_mm,
        "flight_time_model": "executed elastic flights with signed reference-plane matching",
        "flight_time_missing_inelastic_event_depth": material_forward_weight > 0.0 and any(float(spec[5]) > 0. for spec in specs),
        "flight_time_terminal_rows_known": int(np.count_nonzero(np.isfinite(terminal_times))),
        "flight_time_virtual_reference_rows_unavailable": True,
        "material_section_reused_prefix": any(record["hit"] for record in resume_records),
        "material_section_resume": tuple(resume_records),
        "material_section_vacuum_restart": "recomputed" if state.vacuum_map.enabled else "not_participating",
    }
    return GeometricSpecimenExit(
        tuple(branches),
        metrics,
        dependency_signature=dependency_signature,
        checkpoints=tuple(segment.checkpoints for segment in segments),
        segments=tuple(segments),
    )
