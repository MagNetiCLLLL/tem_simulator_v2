"""Executed, tip-origin optical-preview sections; never downstream sources.

Material scattering is omitted exactly as in the existing optical tuning path.
Restart states are float64 solver checkpoints, with original identity and clocks.
The straight-column coordinate ends at the enabled energy-filter entrance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from time import perf_counter

import numpy as np

from temsim.physics.core import (
    FIELD_SIGMA_CUTOFF, build_propagation_plan, execute_propagation_plan,
)

SECTION_SCHEMA = "executed-optical-particle-section-v2-kinetic-energy"
MATERIAL_SECTION_SCHEMA = "executed-material-particle-section-v2-kinetic-energy"


def particle_section_downstream_signature(state, target_z_mm: float) -> str:
    """Identity of a bounded material exit, independently verified by readers."""
    import hashlib
    from temsim.calculation_cache import calculation_signatures
    original = calculation_signatures(state)["sample_downstream"]
    return hashlib.sha256(repr((MATERIAL_SECTION_SCHEMA, "downstream", original,
                               float(target_z_mm))).encode()).hexdigest()


@dataclass(frozen=True)
class ParticleSectionSegment:
    name: str
    branch: object
    plan: object
    checkpoints: object
    initial_dependency_signature: str


@dataclass(frozen=True)
class ParticleSectionCheckpoint:
    schema: str
    gun_dependency_signature: str
    gun_trace: object
    segments: tuple[ParticleSectionSegment, ...]


@dataclass(frozen=True)
class MaterialSectionCache:
    """Executed specimen terminal state; never an independent source."""
    signatures: dict[str, str]
    incident_digest: str
    point_xy_nm: tuple[float, float]
    elastic_transport: object
    inelastic_distribution: object
    specimen_exit: object | None = None
    target_z_mm: float | None = None
    schema: str = MATERIAL_SECTION_SCHEMA
    eds_spectrum: object | None = None


def section_limits(state):
    """Allowed straight-column observation coordinates in millimetres."""
    from temsim.physics.recording_stop import determine_tem_stop_z
    lower = float(state.electron_gun.exit_plane_z_mm)
    energy_filter = getattr(state, "energy_filter", None)
    upper = (float(energy_filter.entrance_z_mm)
             if energy_filter is not None and bool(energy_filter.enabled)
             else float(determine_tem_stop_z(state)))
    return lower, upper


def _events(state):
    from temsim.physics.instrument_magnetic import active_column_events
    return active_column_events(state)


def _selection_boundary(state, keys):
    """Conservative earliest action, never a magnetic component's centre."""
    start, _ = section_limits(state)
    if not keys:
        return math.inf
    gun = state.electron_gun
    gun_keys = {str(getattr(c, "key", "")) for c in getattr(gun, "components", ())}
    gun_keys.update(str(getattr(getattr(gun, name, None), "key", ""))
                    for name in ("emitter", "dpa_aperture", "c1_aperture"))
    gun_keys.update(("electron_gun", "gun", "source", "emitter", "tip", str(getattr(gun, "type_key", ""))))
    components = {}
    for collection in ("lenses", "deflectors", "stigmators", "corrector_elements",
                       "apertures", "recording_planes"):
        components.update((str(c.key), c) for c in getattr(state, collection, ()))
    components[str(getattr(state.sample, "key", "sample"))] = state.sample
    bounds = []
    from temsim.physics.instrument_magnetic import column_dipole_fields
    dipoles = column_dipole_fields(state)
    for key in keys:
        if key in gun_keys:
            bounds.append(start)
            continue
        if key not in components:
            raise ValueError(f"Unknown section tuning component: {key}")
        component = components[key]
        dipole_support = [coil.lower_m*1e3 for coil in dipoles
                          if key in coil.drive_keys or coil.key.rsplit(":", 1)[0] == key]
        if dipole_support:
            bounds.append(min(dipole_support))
            continue
        from temsim.component_keys import CONDENSER_LENS_KEYS
        if key in CONDENSER_LENS_KEYS:
            component = state.condenser_system[key]
        # Mapped and shared nonlinear circuits may have a wider support than
        # an individual native element. Exact plan comparison remains required.
        if getattr(state, "lens_field_map_descriptors", {}):
            bounds.append(start)
            continue
        support = getattr(component, "field_support_mm", None)
        if callable(support):
            try:
                lower = support(FIELD_SIGMA_CUTOFF)[0]
            except TypeError:
                lower = support()[0]
            bounds.append(float(lower))
        elif hasattr(component, "kick_events"):
            try:
                events = component.kick_events(time_s=float(getattr(state, "simulation_time_s", 0.)))
            except TypeError:
                events = component.kick_events()
            bounds.append(min((float(e[0]) for e in events), default=start))
        elif key in {str(c.key) for c in getattr(state, "apertures", ())} | {
                str(c.key) for c in getattr(state, "recording_planes", ())}:
            bounds.append(float(component.z_mm))
        else:
            # Missing support, shared controls and arbitrary parameter owners
            # cannot prove that the earlier beam is independent of the edit.
            bounds.append(start)
    return min(bounds)


def gun_dependency_signature(state):
    """Bind persisted executed gun data to its actual source/model inputs."""
    from temsim.vacuum import bind_gun_environment
    from temsim.calculation_manifest import solver_source_identity
    from temsim.optics.electron_gun.tracing import GUN_FLIGHT_TIME_SCHEMA
    gun = state.electron_gun.validate()
    bind_gun_environment(state)
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    with instrument_gun_field_context(state):
        cache_key = getattr(gun, "_cache_key", None)
        if not callable(cache_key):
            return ""  # Unsupported source codecs execute their ordinary source path.
        identity = getattr(gun, "_instrument_magnetic_identity", None)
        if not isinstance(identity, str) or not identity:
            return ""
        payload = (SECTION_SCHEMA, solver_source_identity(), GUN_FLIGHT_TIME_SCHEMA,
                   cache_key(None))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _digest_arrays(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(repr((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_section_checkpoint(checkpoint):
    """Reject incomplete restart data; a readable display result is not a seed."""
    try:
        return _validate_section_checkpoint(checkpoint)
    except (AttributeError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed section checkpoint arrays or metadata") from exc


def _validate_section_checkpoint(checkpoint):
    from temsim.physics.simulation import (
        _validate_gun_flight_times, validate_flight_time_array, validate_kinetic_energy_array,
    )
    from temsim.physics.particle_energy import energy_survival_mask
    if not isinstance(checkpoint, ParticleSectionCheckpoint) or checkpoint.schema != SECTION_SCHEMA:
        raise ValueError("Unsupported optical section checkpoint schema")
    if not checkpoint.gun_dependency_signature or not _validate_gun_flight_times(checkpoint.gun_trace):
        raise ValueError("Section checkpoint lacks an executed gun identity or clock")
    count = len(checkpoint.gun_trace.exit_bundle.x_m)
    if not checkpoint.segments or tuple(s.name for s in checkpoint.segments) not in {("incident",), ("incident", "000")}:
        raise ValueError("Invalid section segment order")
    for segment in checkpoint.segments:
        if not isinstance(segment, ParticleSectionSegment) or not segment.initial_dependency_signature:
            raise ValueError("Invalid section segment identity")
        branch, cp, plan = segment.branch, segment.checkpoints, segment.plan
        z = np.asarray(branch.z)
        if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError("Section history planes are invalid")
        for key in ("x", "y", "tx", "ty"):
            if np.shape(getattr(branch, key)) != (len(z), count):
                raise ValueError("Section history population is inconsistent")
        for key in ("alive", "blocked_z", "energy_offset_ev", "ray_weight", "source_ray_id", "source_azimuth_rad"):
            if np.shape(getattr(branch, key)) != (count,):
                raise ValueError("Section source ancestry or survival state is incomplete")
        if len(branch.blocked_key) != count:
            raise ValueError("Section physical-stop labels do not match rays")
        if not validate_flight_time_array(branch.flight_time_s, (len(z), count), "Section"):
            raise ValueError("Section restart requires retained original flight times")
        if (np.ndim(cp.z_mm) != 1 or not len(cp.z_mm)
                or np.asarray(cp.z_mm).dtype != np.dtype(np.float64)
                or any(np.shape(getattr(cp, key)) != (len(cp.z_mm), count)
                       or np.asarray(getattr(cp, key)).dtype != np.dtype(np.float64)
                       for key in ("x_m", "y_m", "tx_rad", "ty_rad"))):
            raise ValueError("Section full-precision checkpoints are incomplete")
        if not validate_flight_time_array(cp.flight_time_s, np.shape(cp.x_m), "Section checkpoint"):
            raise ValueError("Section restart requires checkpoint flight times")
        if not validate_kinetic_energy_array(getattr(cp, "kinetic_energy_ev", None),
                np.shape(cp.x_m), "Section checkpoint", allow_unknown=True,
                required=energy_survival_mask(branch, cp.z_mm)):
            raise ValueError("Section restart requires executed checkpoint kinetic energies")
        if not validate_kinetic_energy_array(getattr(branch, "kinetic_energy_ev", None),
                np.shape(branch.x), "Section", allow_unknown=True,
                required=energy_survival_mask(branch, branch.z)):
            raise ValueError("Section restart requires executed energy history")
        plan_z, indices = np.asarray(plan.z_mm), np.asarray(plan.checkpoint_index)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= len(plan_z))):
            raise ValueError("Section checkpoint plan indices are invalid")
        if not np.array_equal(plan_z[indices], cp.z_mm):
            raise ValueError("Section checkpoints do not match the executed plan")
        if z[0] != plan.z_mm[0] or z[-1] != plan.z_mm[-1] or cp.z_mm[-1] != z[-1]:
            raise ValueError("Section endpoint does not match its executed plan")
        for key, cp_key in (("x", "x_m"), ("y", "y_m"), ("tx", "tx_rad"), ("ty", "ty_rad"), ("flight_time_s", "flight_time_s"), ("kinetic_energy_ev", "kinetic_energy_ev")):
            if not np.array_equal(np.asarray(getattr(branch, key))[-1], np.asarray(getattr(cp, cp_key))[-1], equal_nan=True):
                raise ValueError("Section terminal phase space differs from the retained checkpoint")
    return checkpoint


def _prefix_matches(old, new, old_index, new_index):
    """Compare every consumed node, RK midpoint, action and mapped field."""
    if any((getattr(plan, "electric_field", None) is not None
            or getattr(plan, "electric_reference_invariant_ev", None) is not None)
           and getattr(plan, "electric_field_identity", None) is None for plan in (old, new)):
        return False
    if old_index != new_index or old.solver_signature != new.solver_signature:
        return False
    count = new_index + 1
    for name in old.__dataclass_fields__:
        if name in {"signature", "solver_signature", "mapped_fields", "save_index", "checkpoint_index", "electric_field"}:
            continue
        if name in {"reference_momentum_kg_m_s", "electric_field_identity", "electric_reference_invariant_ev"}:
            if getattr(old, name) != getattr(new, name):
                return False
            continue
        size = new_index if name == "step_m" or name.startswith("midpoint_") else count
        if name in {"dipole_bx_t", "dipole_by_t"}:
            size = 3*new_index
        if not np.array_equal(np.asarray(getattr(old, name))[:size],
                              np.asarray(getattr(new, name))[:size]):
            return False
    old_maps = {item.lens_key: item for item in old.mapped_fields}
    new_maps = {item.lens_key: item for item in new.mapped_fields}
    boundary = float(new.z_mm[new_index])
    for key in old_maps.keys() | new_maps.keys():
        before, after = old_maps.get(key), new_maps.get(key)
        if before is not None and after is not None and before.fingerprint == after.fingerprint:
            continue
        if any(item is not None and item.field_map.field_support_mm[0] <= boundary
               for item in (before, after)):
            return False
    return True


def _trace_segment(state, name, start, stop, initial, energy, weights, identity,
                   survival, events, previous, boundary, *, initial_kicks,
                   save_z_mm=(), dependency_context="", initial_kinetic_energy_ev=None):
    from temsim.physics.simulation import Branch, _column_checkpoint_planes, _merge_checkpoints
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.column_wall import clip_column_wall
    from temsim.physics.residual_medium import ColumnMediumTransport
    x0, tx0, y0, ty0, time0 = initial
    kinetic0 = np.asarray(state.beam_voltage_kv*1000+energy if initial_kinetic_energy_ev is None
                          else initial_kinetic_energy_ev, dtype=np.float64)
    alive0, blocked0, keys0 = survival
    dependency = _digest_arrays(*initial, energy, kinetic0, weights, alive0, blocked0,
                                *identity, np.asarray(keys0, dtype=str),
                                np.asarray(str(dependency_context)))
    compatible = (isinstance(previous, ParticleSectionSegment)
                  and previous.initial_dependency_signature == dependency
                  and getattr(previous.checkpoints, "kinetic_energy_ev", None) is not None
                  and getattr(previous.branch, "kinetic_energy_ev", None) is not None
                  and not state.vacuum_map.enabled)
    checkpoints = tuple(_column_checkpoint_planes(start, stop, len(x0)))
    retained = (tuple(float(z) for z in previous.checkpoints.z_mm if start <= z <= stop)
                if compatible else ())
    safe_boundary = (float(np.nextafter(boundary, -math.inf)),) if start < boundary < stop else ()
    requested = tuple(float(z) for z in save_z_mm if start <= float(z) <= stop)
    planes = tuple(sorted(set((start, stop, *checkpoints, *retained, *safe_boundary, *requested))))
    # Keep the canonical step phase when extending a saved exact observation.
    # All checkpoints are actual integration planes, never nearest display rows.
    from temsim.physics.instrument_magnetic import events_overlapping_interval
    events = events_overlapping_interval(state, events, start, stop)
    plan = build_propagation_plan(state, start, stop, events,
        save_z_mm=planes, checkpoint_z_mm=planes,
        particle_medium=state.vacuum_map.enabled,
        medium_energy_ev=kinetic0)
    resume = None
    if compatible:
        for row in range(len(previous.checkpoints.z_mm)-1, -1, -1):
            z = float(previous.checkpoints.z_mm[row])
            if not start <= z <= stop or not z < boundary:
                continue
            oi = np.flatnonzero(np.asarray(previous.plan.z_mm) == z)
            ni = np.flatnonzero(np.asarray(plan.z_mm) == z)
            if oi.size and ni.size and _prefix_matches(previous.plan, plan, int(oi[0]), int(ni[0])):
                resume = (row, int(ni[0]), z)
                break
    transport = (ColumnMediumTransport(state, plan, len(x0), kinetic0,
                 alive=alive0, stream=1 if name == "incident" else 2)
                 if state.vacuum_map.enabled else None)
    index, resume_z = 0, start
    if resume is not None:
        row, index, resume_z = resume
        cp = previous.checkpoints
        x0, tx0, y0, ty0, time0 = (np.asarray(getattr(cp, key)[row]) for key in
                                  ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
        kinetic0 = np.asarray(cp.kinetic_energy_ev[row])
    energy_alive = alive0 if resume is None else previous.branch.alive
    unknown_stopped = ~np.isfinite(kinetic0) & ~np.asarray(energy_alive, dtype=bool)
    if np.any(unknown_stopped):
        x0, tx0, y0, ty0 = (np.where(unknown_stopped, np.nan, value)
                            for value in (x0, tx0, y0, ty0))
    energy_history = []
    output = execute_propagation_plan(state, plan, x0, tx0, y0, ty0, energy,
        start_index=index, include_initial_plane_kicks=initial_kicks if resume is None else False,
        defer_nonfinite_until_clipping=True, medium_transport=transport,
        initial_time_s=time0, return_flight_times=True,
        initial_kinetic_energy_ev=kinetic0, energy_output=energy_history)
    z, x, tx, y, ty, times, cp = output
    kinetic = energy_history[0]
    if resume is not None:
        old = previous.branch
        keep = np.asarray(old.z) < resume_z
        z = np.r_[np.asarray(old.z)[keep], z]
        x, tx, y, ty, times = tuple(np.vstack((np.asarray(getattr(old, key))[keep], value))
            for key, value in zip(("x", "tx", "y", "ty", "flight_time_s"), (x, tx, y, ty, times)))
        cp = _merge_checkpoints(previous.checkpoints, cp, resume_z)
        kinetic = np.vstack((np.asarray(old.kinetic_energy_ev)[keep], kinetic))
    # Publish exact checkpoint precision, including segment boundary rows.
    # A displayed material/vacuum interface must retain the same phase state
    # used by the next segment, without an extra float32 round trip.
    x, tx, y, ty = tuple(np.asarray(v, dtype=np.float64).copy() for v in (x, tx, y, ty))
    rows = np.searchsorted(z, cp.z_mm)
    matches = (rows < len(z)) & (z[np.minimum(rows, len(z)-1)] == cp.z_mm)
    for values, key in zip((x, tx, y, ty), ("x_m", "tx_rad", "y_m", "ty_rad")):
        values[rows[matches]] = getattr(cp, key)[matches]
    alive, blocked, labels = alive0.copy(), blocked0.copy(), list(keys0)
    if transport is not None:
        alive, blocked, labels = transport.merge_stops(alive, blocked, labels)
    alive, blocked, labels = clip_segment(state, z, x, y, alive, blocked, labels)
    alive, blocked, labels = clip_recording_planes(state, z, x, y, alive, blocked, labels)
    alive, blocked, labels = clip_column_wall(state, z, x, y, alive, blocked, labels)
    visible = ~np.isfinite(blocked)[None, :] | (z[:, None] <= blocked[None, :])
    if any(np.any(~np.isfinite(v) & visible) for v in (x, tx, y, ty)):
        raise ValueError("Non-finite section trajectory before a physical stop")
    branch = Branch(name, (.22, .74, .97), z, x, y, tx, ty, alive, blocked, labels,
                    1., np.asarray(kinetic[-1])-state.beam_voltage_kv*1000, weights.copy(),
                    interaction_kind="incident" if name == "incident" else "optical_reference",
                    source_ray_id=identity[0], source_azimuth_rad=identity[1],
                    vacuum_report=transport.report() if transport is not None else None,
                    flight_time_s=np.asarray(times, dtype=np.float64), kinetic_energy_ev=kinetic)
    return ParticleSectionSegment(name, branch, plan, cp, dependency), float(resume_z), resume is not None


def run_particle_section(state, *, observation_stop_z_mm, tuning_component_keys=(),
                         existing_simulation=None, resolved_layout=None):
    from temsim.physics.simulation import Simulation, Branch, _validate_gun_flight_times
    from temsim.physics.optical_tuning import check_tuning_cancelled, tuning_metrics
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.ray_identity import emitted_source_identity
    from temsim.physics.chromatic import objective_chromatic_kick_from_state, configured_objective_chromatic_focal_mm
    require_physical_gun_source(state.electron_gun)
    check_tuning_cancelled(state)
    if resolved_layout is None:
        from temsim.column.state_layout import apply_physical_layout_to_state
        resolved_layout = apply_physical_layout_to_state(state)
    from temsim.physics.backend_execution import classical_backend_preflight
    classical_backend_preflight(state)
    start, maximum = section_limits(state)
    target = float(observation_stop_z_mm)
    if not math.isfinite(target) or not start <= target <= maximum:
        raise ValueError(f"Optical section must lie between gun exit {start:g} mm and axial boundary {maximum:g} mm; bent energy-filter paths require their own transport")
    keys = tuple(str(key) for key in tuning_component_keys)
    boundary = _selection_boundary(state, keys)
    signature = gun_dependency_signature(state)
    old = getattr(existing_simulation, "section_checkpoint", None)
    reusable = (isinstance(old, ParticleSectionCheckpoint) and old.schema == SECTION_SCHEMA
                and signature and signature == old.gun_dependency_signature)
    reuse_reason = ("no_prior_checkpoint" if old is None else
                    "upstream_inputs_or_model_changed" if not reusable else "compatible_gun")
    gun = None
    if reusable:
        try:
            validate_section_checkpoint(old)
            if _validate_gun_flight_times(old.gun_trace):
                gun = old.gun_trace
        except (ValueError, TypeError, AttributeError):
            reusable = False
            reuse_reason = "invalid_saved_state"
    gun_started = perf_counter()
    if gun is None:
        gun = trace_source_to_exit(state)
        _validate_gun_flight_times(gun)
        gun_reused = False
    else:
        gun_reused = True
    gun_seconds = perf_counter() - gun_started
    check_tuning_cancelled(state)
    emitted = gun.exit_bundle
    identity = emitted_source_identity(gun)
    energy, weights = np.asarray(emitted.energy_offset_ev), np.asarray(emitted.weight)
    initial = tuple(np.asarray(getattr(emitted, name)) for name in
                    ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    survival = (np.asarray(emitted.alive), np.asarray(gun.blocked_z_mm), gun.blocked_key)
    previous = {segment.name: segment for segment in old.segments} if reusable else {}
    sample_z = float(state.sample.z_mm)
    events = _events(state)
    incident, resume_z, hit = _trace_segment(state, "incident", start, min(target, sample_z),
        initial, energy, weights, identity, survival,
        events,
        previous.get("incident"), boundary, initial_kicks=True)
    segments = [incident]
    branch = incident.branch
    # Keep the original gun history for rendering, but not in the segment's
    # column checkpoint codec; the physical gun result is retained separately.
    keep = np.asarray(gun.z_mm) < start
    incident_display = Branch(**{**vars(branch),
        "z": np.r_[np.asarray(gun.z_mm)[keep], branch.z],
        "kinetic_energy_ev": np.vstack((np.full(np.shape(gun.x_m), np.nan)[keep],
                                         branch.kinetic_energy_ev)),
        **{key: np.vstack((np.asarray(getattr(gun, gun_key))[keep], getattr(branch, key)))
           for key, gun_key in (("x", "x_m"), ("tx", "tx_rad"), ("y", "y_m"),
                                ("ty", "ty_rad"), ("flight_time_s", "flight_time_s"))}})
    branches = {}
    if target > sample_z:
        sample_energy = np.asarray(branch.kinetic_energy_ev[-1])
        cx, cy = objective_chromatic_kick_from_state(state, branch.x[-1], branch.y[-1],
                     sample_energy - state.beam_voltage_kv*1000,
                     resolved_focal_mm=configured_objective_chromatic_focal_mm(state))
        outgoing = (branch.x[-1], branch.tx[-1]+cx, branch.y[-1], branch.ty[-1]+cy,
                    np.where(branch.alive, branch.flight_time_s[-1], np.nan))
        post, post_resume, post_hit = _trace_segment(state, "000", sample_z, target,
            outgoing, energy, weights, identity, (branch.alive, branch.blocked_z, branch.blocked_key),
            events, previous.get("000"),
            boundary, initial_kicks=False, initial_kinetic_energy_ev=sample_energy)
        segments.append(post)
        branches["000"] = post.branch
        if post_hit:
            resume_z = post_resume
        hit = hit or post_hit
    metrics = tuning_metrics(state, incident_display)
    endpoint_metrics = tuning_metrics(state, segments[-1].branch)
    observation_keys = tuple(key for key in metrics if key.startswith("sample_")
                             and key not in {"sample_scattering_applied", "sample_scattering_model", "sample_inserted"})
    for key in observation_keys:
        metrics["section_" + key.removeprefix("sample_")] = endpoint_metrics[key]
    if target < sample_z:
        # The final incident row is this requested section, not the specimen.
        # Existing consumers must not interpret its focus as a sample result.
        for key in observation_keys:
            metrics[key] = ("NOT_REACHED" if key == "sample_statistics_status" else
                            0 if key in {"sample_beam_surviving_rays", "sample_support_probe_survivors"}
                            else float("nan"))
    metrics["sample_illumination_evaluated"] = target >= sample_z
    metrics.update(section_target_z_mm=target, section_component_keys=keys,
                   section_resume_z_mm=resume_z, section_schema=SECTION_SCHEMA,
                   section_reused_prefix=bool(hit), section_gun_reused=gun_reused,
                   section_resumable_through_z_mm=target,
                   section_gun_seconds=gun_seconds,
                   section_reuse_reason=("vacuum_restart_required" if state.vacuum_map.enabled else
                       "compatible_executed_prefix" if hit else
                       "no_compatible_column_prefix" if gun_reused else reuse_reason),
                   section_vacuum_restart="recomputed" if state.vacuum_map.enabled else "not_participating",
                   section_sample_reference_reached=target >= sample_z,
                   column_segment_cache={"mode": "checkpoint" if hit else "none", "hit": bool(hit), "resume_z_mm": resume_z})
    result = Simulation(incident_display, branches, metrics, gun_trace=gun,
                        incident_plan=incident.plan, incident_checkpoints=incident.checkpoints)
    result.section_checkpoint = ParticleSectionCheckpoint(SECTION_SCHEMA, signature, gun, tuple(segments))
    check_tuning_cancelled(state)
    return result
