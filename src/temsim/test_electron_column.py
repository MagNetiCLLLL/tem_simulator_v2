"""Virtual-particle adapter to the production forward column propagator.

There is no second downstream source. An executed gun endpoint (or explicitly
edited diagnostic initial state) is passed to the same plan and integrator as
the particle beam. The instrument inputs are captured, and only the diagnostic
numerical sampling is adjustable. Material and detector interactions remain
excluded; scene-owned apertures and mechanical contacts remain active.
"""
from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin
from time import monotonic

import numpy as np

from temsim import input_io
from temsim.physics.relativistic_lorentz import (
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
    ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
)


def _potential(scene, points):
    provider = scene.electric_provider
    query = getattr(provider, "potential_rise_v_at_global_positions", None)
    if query is None:
        query = provider.potential_v_at_global_positions
    return np.asarray(query(np.asarray(points)), dtype=float)


def trace_instrument_electron(scene, settings, cancelled, progress, interval, use_compiled):
    """Use the common gun step then the production column's declared model."""
    from temsim.magnetic_test_particle import (
        _trace_electromagnetic_electron, _electromagnetic_result,
        electron_momentum_and_speed,
    )
    from temsim.instrument_snapshot import decode_instrument

    handoff = float(scene._column_handoff_z_m)
    theta, azimuth = radians(settings.polar_angle_deg), radians(settings.azimuth_angle_deg)
    if cancelled() or not scene.diagnostic_position_is_valid(np.asarray(settings.position_m)):
        return _trace_electromagnetic_electron(scene, settings, cancelled, progress, interval, use_compiled)
    # The production axial model is forward-only. Preserve the existing full
    # time-domain diagnostic for a deliberately backward launch, explicitly
    # outside the scope in which a production-column comparison is meaningful.
    if settings.position_m[2] >= handoff and cos(theta) <= 0.:
        result = _trace_electromagnetic_electron(scene, settings, cancelled, progress, interval, use_compiled)
        return replace(result, notes=(*result.notes,
            "Backward diagnostic: full time-domain field transport; the production forward-column approximation does not admit this initial direction."))

    prefix = None
    if settings.position_m[2] < handoff < scene.diagnostic_bounds_m[1, 2]:
        gun_scene = replace(scene, _unsupported_stops=(*scene._unsupported_stops,
                                                       (handoff, "column_handoff")))
        prefix = _trace_electromagnetic_electron(gun_scene, settings, cancelled, progress, interval, use_compiled)
        if prefix.reason != "column_handoff":
            return prefix
    elif settings.position_m[2] < handoff:
        return _trace_electromagnetic_electron(scene, settings, cancelled, progress, interval, use_compiled)

    if prefix is None:
        momentum, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
        direction = np.array((sin(theta)*cos(azimuth), sin(theta)*sin(azimuth), cos(theta)))
        positions = [np.array(settings.position_m)]
        momenta = [momentum*direction/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)]
        times, distances = [0.], [0.]
        potentials = [float(_potential(scene, [settings.position_m])[0])]
    else:
        positions = list(prefix.positions_m.copy())
        momenta = list(prefix.momentum_kg_m_per_s/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S))
        times, distances = list(prefix.time_s), list(prefix.path_length_m)
        potentials = list(prefix.electrostatic_potential_v)

    state = decode_instrument(scene._column_input_graph)
    state._tuning_cancelled = cancelled
    # Normal diagnostics follow the captured toolbar policy, even for a
    # single electron. Only the explicit validation/reference option may
    # select uncompiled CPU execution instead of that captured preference.
    if not use_compiled:
        state.acceleration_enabled = False
        state.acceleration_backend = "CPU"
    state._active_backends_used = set()
    state._optical_tuning = True
    state.step_mm = min(float(state.step_mm), settings.step_m*1000.)
    state.history_step_mm = state.step_mm

    reason = "domain_exit"
    column_executed = False
    next_progress = monotonic()+interval
    from temsim.test_electron_intercepts import prepare_compiled_intercepts, compiled_intercept
    packed_contacts = prepare_compiled_intercepts(scene) if use_compiled else None

    def contact_between(start, stop):
        if packed_contacts is None:
            return scene.diagnostic_segment_stop(start, stop)
        data, reasons = packed_contacts
        fraction, index = compiled_intercept(start, stop, data)
        return (fraction, reasons[index]) if index >= 0 else None

    def result(reason):
        output = _electromagnetic_result(settings, positions, momenta, times, distances, potentials, reason)
        from temsim.simulation_modes import is_ideal
        note = ("Shared production column: forward paraxial propagation, captured electric potential, "
                "finite magnetic optics and the same enabled thin optical actions; no material/detector interactions.")
        extras = (note,
            "Gun and column use their common production segment methods; this is not an independent full-Lorentz column reference.",
            "Forward column step is capped by both instrument and diagnostic step sizes. Relative error tolerance controls the adaptive gun segment; column accuracy is checked by step refinement.",
            "Column travelled path is reconstructed by second-order speed integration between executed checkpoints; contact positions and times are interpolated within the stopping interval.")
        extras += (f"Column compute policy: {state.acceleration_backend}; acceleration "
                   f"{'enabled' if state.acceleration_enabled else 'disabled'}; executed backend: "
                   f"{state.active_backend if column_executed else 'not executed'}.",)
        if not use_compiled:
            extras += ("Explicit uncompiled CPU reference requested for this diagnostic; the captured instrument compute policy was not changed.",)
        if is_ideal(state):
            extras += ("Ideal optics retains the production chromatic-free trajectory approximation; reported kinetic energy and flight time use the actual electron energy.",)
        notes = tuple(note for note in output.notes
                      if not note.startswith(("Second-order discrete-gradient", "Time and travelled path")))
        return replace(output, notes=(*notes, *extras))

    with input_io.input_scope(state):
        from temsim.physics.core import build_propagation_plan, execute_propagation_plan
        from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval
        lower = float(positions[-1][2])*1000.
        remaining_path = settings.max_path_length_m-distances[-1]
        if remaining_path <= 0.:
            return result("path_limit")
        # Keep one extra interval for locating the requested path stop. The
        # second-order speed integral can slightly undershoot axial distance;
        # treating the path estimate itself as a new domain boundary would
        # otherwise misreport a short requested trace as a domain exit.
        end = min(float(scene.diagnostic_bounds_m[1, 2])*1000.,
                  lower+remaining_path*1000.+state.step_mm)
        if end <= lower:
            return result("domain_exit")
        sample = float(state.sample.z_mm)
        boundaries = [lower, *((sample,) if lower < sample < end else ()), end]
        for section, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if cancelled():
                return result("cancelled")
            events = events_overlapping_interval(state, active_column_events(state), start, stop)
            plan = build_propagation_plan(state, start, stop, events, maximum_step_mm=settings.step_m*1000.)
            # Retain the exact executed states at all integration nodes for
            # contacts and trajectory display, never float32 drawing history.
            indices = np.arange(len(plan.z_mm), dtype=np.int64)
            indices.setflags(write=False)
            plan = replace(plan, checkpoint_index=indices)
            initial_p = np.asarray(momenta[-1])
            if initial_p[2] <= 0.:
                return result("unsupported_field:forward_column_direction")
            energy = result("in_progress").kinetic_energy_ev[-1]
            try:
                values = execute_propagation_plan(state, plan,
                    np.array([positions[-1][0]]), np.array([initial_p[0]/initial_p[2]]),
                    np.array([positions[-1][1]]), np.array([initial_p[1]/initial_p[2]]),
                    np.array([energy-state.beam_voltage_kv*1000.]),
                    initial_kinetic_energy_ev=np.array([energy]),
                    initial_time_s=np.array([times[-1]]), return_flight_times=True,
                    include_initial_plane_kicks=(section == 0),
                    defer_nonfinite_until_clipping=True)
            except RuntimeError:
                if cancelled():
                    return result("cancelled")
                raise
            column_executed = True
            checkpoints = values[-1]
            # Reconstruct the executed single-particle history in one batch.
            # Repeating NumPy allocation and scalar field interpolation at
            # every display node costs much more than the shared integrator.
            xyz = np.column_stack((checkpoints.x_m[:, 0], checkpoints.y_m[:, 0],
                                   checkpoints.z_mm*1e-3))
            slope = np.column_stack((checkpoints.tx_rad[:, 0], checkpoints.ty_rad[:, 0],
                                     np.ones(len(xyz))))
            energies = checkpoints.kinetic_energy_ev[:, 0]
            clock = checkpoints.flight_time_s[:, 0]
            valid = (np.isfinite(xyz).all(axis=1) & np.isfinite(slope).all(axis=1)
                     & np.isfinite(energies) & (energies > 0.) & np.isfinite(clock))
            failed = np.flatnonzero(~valid)
            finite_stop = int(failed[0]) if len(failed) else len(xyz)
            particle_p = momentum_from_kinetic_energy_ev(energies[:finite_stop], slope[:finite_stop])
            speeds = np.linalg.norm(velocity_from_momentum_m_per_s(particle_p), axis=1)
            voltage = _potential(scene, xyz[:finite_stop]) if finite_stop else np.empty(0)
            travel = .5*(speeds[:-1]+speeds[1:])*np.diff(clock[:finite_stop])
            for row in range(1, len(checkpoints.z_mm)):
                if cancelled():
                    return result("cancelled")
                if len(times)-1 >= settings.max_steps:
                    return result("step_limit")
                if row >= finite_stop:
                    return result("numerical_limit")
                position, direction, p = xyz[row], slope[row], particle_p[row]
                kinetic, elapsed, travelled = float(energies[row]), float(clock[row]), float(travel[row-1])
                fraction, stop_reason = 1., None
                contact = contact_between(positions[-1], position)
                if contact is not None:
                    fraction, stop_reason = contact
                remaining = settings.max_path_length_m-distances[-1]
                if travelled > remaining and remaining/travelled < fraction:
                    fraction, stop_reason = max(0., remaining/travelled), "path_limit"
                if fraction < 1.:
                    position = positions[-1]+fraction*(position-positions[-1])
                    # Contact is located on this executed interval. Its energy
                    # follows the same conserved potential, never a nominal reset.
                    potential = float(_potential(scene, [position])[0])
                    invariant = kinetic-float(voltage[row])
                    kinetic = invariant+potential
                    previous_p = np.asarray(momenta[-1])*(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)
                    old_direction = previous_p/previous_p[2]
                    direction = old_direction+fraction*(direction-old_direction)
                    p = momentum_from_kinetic_energy_ev(np.array([kinetic]), direction[None, :])[0]
                    elapsed = times[-1]+fraction*(elapsed-times[-1])
                else:
                    potential = float(voltage[row])
                positions.append(position)
                momenta.append(p/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S))
                times.append(elapsed)
                distances.append(distances[-1]+fraction*travelled)
                potentials.append(potential)
                if progress is not None and monotonic() >= next_progress:
                    progress(result("in_progress"))
                    next_progress = monotonic()+interval
                if stop_reason is not None:
                    return result(stop_reason)
    if settings.max_path_length_m-distances[-1] <= max(settings.position_tolerance_m, settings.max_path_length_m*2e-12):
        reason = "path_limit"
    return result(reason)
