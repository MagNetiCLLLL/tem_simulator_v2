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
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class GeometricSpecimenExit:
    """Source-normalised specimen-exit branches and their probability ledger."""

    branches: tuple[Branch, ...]
    metrics: dict[str, object]
    dependency_signature: str = ""

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
    if weights.shape != alive.shape or np.any(weights < 0.0):
        raise ValueError("Incident sample ray weights do not match the ray bundle")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Incident sample ray weights must have positive finite sum")
    return float(np.sum(weights[alive]) / total)


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


def build_geometric_specimen_exit(
    state,
    simulation,
    elastic_transport,
    inelastic_distribution=None,
    *,
    save_z_mm: tuple[float, ...] = (),
    dependency_signature: str = "",
    progress_callback: ProgressCallback | None = None,
) -> GeometricSpecimenExit:
    """Combine elastic exit phase space and exclusive inelastic populations.

    The coupling is an explicit independence approximation: the finite-envelope
    screened-Rutherford terminal distribution is multiplied by the aggregate
    Poisson energy-loss distribution.  This produces one conserved electron
    population; elastic and inelastic branch weights are never added together.
    """

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
    total_terminal_weight = float(np.sum(weights))
    if not math.isclose(
        total_terminal_weight, 1.0, rel_tol=0.0, abs_tol=2.0e-12
    ):
        raise ValueError("Elastic terminal conditional weights must sum to one")

    source_ray_count = int(np.asarray(simulation.incident.alive).size)
    if (
        source_ray_count <= 0
        or np.any(source_indices < 0)
        or np.any(source_indices >= source_ray_count)
    ):
        raise ValueError("Elastic terminal source indices are outside the incident bundle")
    from temsim.physics.ray_identity import select_identity, source_identity

    source_ids, source_azimuths = source_identity(
        simulation.incident, getattr(simulation, "gun_trace", None)
    )
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

    eligible = (outcomes == "transmitted") & (directions[:, 2] > 1.0e-12)
    forward_weight = float(np.sum(weights[eligible]))
    nominal_energy_ev = float(state.beam_voltage_kv) * 1000.0
    stop_z_mm = float(determine_tem_stop_z(state))
    events = _post_sample_events(state)
    chromatic_focal_mm = configured_objective_chromatic_focal_mm(state)
    field_diagnostic = sample_axial_field_diagnostic(state)
    field_transport = SpecimenFieldTransport(state)
    branches: list[Branch] = []
    exit_plane_source_probability = 0.0

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
        for index, (position, direction, energy_ev) in enumerate(zip(
            local_positions,
            terminal_direction,
            terminal_energies,
            strict=True,
        )):
            reference_positions[index], reference_directions[index] = (
                field_transport.to_plane(
                    position,
                    direction,
                    0.0,
                    energy_ev=float(energy_ev),
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

            for is_scattered, elastic_label in (
                (False, "primary"),
                (True, "elastic"),
            ):
                mask = terminal_scattered == is_scattered
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
                group_weight = float(np.sum(group_weights))
                absolute_probability = (
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
                z, x, tx, y, ty = propagate(
                    state,
                    float(state.sample.z_mm),
                    stop_z_mm,
                    reference_xy_nm[mask, 0] * 1.0e-9,
                    start_tx[mask],
                    reference_xy_nm[mask, 1] * 1.0e-9,
                    start_ty[mask],
                    events,
                    energy_offset[mask],
                    save_z_mm=tuple(float(value) for value in save_z_mm),
                    include_initial_plane_kicks=False,
                )
                alive = np.ones(np.count_nonzero(mask), dtype=bool)
                blocked = np.full(alive.size, np.nan, dtype=float)
                blocked_keys = [""] * alive.size
                alive, blocked, blocked_keys = clip_recording_planes(
                    state, z, x, y, alive, blocked, blocked_keys
                )
                alive, blocked, blocked_keys = clip_column_wall(
                    state, z, x, y, alive, blocked, blocked_keys
                )
                if save_z_mm:
                    boundary_z_mm = float(save_z_mm[0])
                    reaches_boundary = (
                        np.isnan(blocked)
                        | (blocked > boundary_z_mm + 1.0e-9)
                    )
                    exit_plane_source_probability += absolute_probability * float(
                        np.sum((group_weights / group_weight)[reaches_boundary])
                    )
                interaction_kind = (
                    "sample_region_elastic"
                    if is_scattered
                    else "sample_region_primary"
                    if kind == "real_zero_loss"
                    else str(kind)
                )
                # Terminal/group rows may be reordered, repeated, or sparse.
                # Retain their original gun lineage, never the compact row ID.
                group_source_ids, group_source_azimuths = select_identity(
                    source_ids, source_azimuths, terminal_indices[mask]
                )
                branches.append(
                    Branch(
                        name=f"specimen_{elastic_label}:{channel_name}",
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
                        interaction_kick_x_rad=kick_x_values[mask],
                        interaction_kick_y_rad=kick_y_values[mask],
                        source_ray_id=group_source_ids,
                        source_azimuth_rad=group_source_azimuths,
                    )
                )
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

    tracked_source = source_fraction * forward_weight * inelastic_tracked
    absorbed_source = source_fraction * forward_weight * inelastic_absorbed
    elastic_nontransmitted_source = source_fraction * (1.0 - forward_weight)
    pre_sample_lost = 1.0 - source_fraction
    partition = (
        tracked_source
        + absorbed_source
        + elastic_nontransmitted_source
        + pre_sample_lost
    )
    conservation_error = partition - 1.0
    branch_weight_sum = float(sum(branch.weight for branch in branches))
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
        "elastic_nontransmitted_conditional_probability": 1.0 - forward_weight,
        "inelastic_tracked_conditional_probability": inelastic_tracked,
        "inelastic_absorbed_conditional_probability": inelastic_absorbed,
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
            (exit_plane_source_probability if save_z_mm else tracked_source)
            / source_fraction
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
    }
    return GeometricSpecimenExit(
        tuple(branches),
        metrics,
        dependency_signature=dependency_signature,
    )
