"""Relativistic tip-to-exit tracing with stops at their physical planes."""

from __future__ import annotations

import math

import numpy as np

ANALYTIC_ENERGY_SCHEMA = "launch-potential-handoff-v2"
ANALYTIC_STEP_SCHEMA = "all-active-field-step-doubled-boris-compiled-v2"
ANALYTIC_MAXIMUM_RELATIVE_IMPULSE = .025
GUN_FLIGHT_TIME_SCHEMA = "tip-clock-first-crossing-v1"

from temsim.component_keys import FEG_MONOCHROMATOR_SLIT
from temsim.optics.electron_gun.base import (
    GunEqualTimeHistory,
    GunExitBundle,
    GunPlaneArrival,
    GunTraceResult,
)
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    boris_step,
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


def trace_feg_to_exit(gun, count=None, *, cancelled=None) -> GunTraceResult:
    from temsim.vacuum import ensure_standalone_gun_environment
    ensure_standalone_gun_environment(gun)
    emitted = gun.emit(count)
    from temsim.physics.ray_identity import emission_reference
    launch_reference = emission_reference(emitted, getattr(gun.emitter, "surface_model", None))
    surface_model = getattr(gun.emitter, "surface_model", None)
    geometry_field = bool(getattr(gun, "uses_geometry_electric_field", surface_model is not None))
    electric_provider = gun.electric_field
    electric_base = getattr(electric_provider, "base_field", electric_provider)
    grounded_liner = getattr(electric_base, "request", {}).get("grounded_liner", ())
    tip_geometry = getattr(electric_base, "geometry", None) if geometry_field else None
    tip_material_mask = getattr(electric_base, "tip_material_mask", None) if geometry_field else None
    trace_potential = (getattr(electric_provider, "potential_rise_v_at_global_positions",
                               electric_provider.potential_v_at_global_positions)
                       if geometry_field else electric_provider.potential_v_at_global_positions)
    if cancelled is not None and cancelled():
        raise RuntimeError("Superseded optical tuning request")
    magnetic_provider = gun.magnetic_field
    n = emitted.x_m.size
    from temsim.physics.residual_medium import MediumTransport, region_rate_bound
    from temsim.physics.relativistic_lorentz import kinetic_energy_ev_from_momentum
    medium = MediumTransport(getattr(gun, "_vacuum_regions", ()), n,
                             getattr(gun, "_vacuum_seed", 914), stream=0,
                             max_step_tau=getattr(gun, "_vacuum_max_step_tau", .02))
    direction = np.column_stack((
        emitted.tx_rad,
        emitted.ty_rad,
        np.ones(n, dtype=float),
    ))
    launch_energy = (
        float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev
    )
    if surface_model is not None:
        launch_energy = emitted.surface_energy_ev
    if np.any(~np.isfinite(launch_energy)) or np.any(launch_energy <= 0.0):
        raise ValueError(
            f"{gun.display_name} emitter produced non-positive launch "
            "kinetic energy. Check its energy-spread parameters."
        )
    position = np.column_stack((
        emitted.x_m,
        emitted.y_m,
        np.zeros(n, dtype=float),
    ))
    surface_mesh_error = None
    surface_field = None
    if surface_model is not None:
        from temsim.physics.grounded_tip_field import grounded_field
        surface_field = grounded_field(gun)
        position, surface_mesh_error = surface_field.surface_mesh_positions(emitted.surface_position_m)
        direction = emitted.surface_direction
        launch_energy = emitted.surface_energy_ev
    elif hasattr(emitted, "surface_normal"):
        position = emitted.surface_position_m.copy()
        direction = emitted.surface_direction
    # K - e*phi is conserved. Use a stable voltage gauge for solved fields;
    # the emitted positions, directions and local kinetic energies are untouched.
    invariant_energy = launch_energy - trace_potential(position)
    launch_boundary_report = (electric_base.launch_boundary_report(position)
                              if hasattr(electric_base, "launch_boundary_report") else None)
    momentum = momentum_from_kinetic_energy_ev(launch_energy, direction)
    phase = RelativisticPhaseSpace(position, momentum)
    alive = np.ones(n, dtype=bool)
    completed = np.zeros(n, dtype=bool)
    dpa_passed = np.zeros(n, dtype=bool)
    c1_passed = np.zeros(n, dtype=bool)
    blocked_z = np.full(n, np.nan, dtype=float)
    blocked_time_s = np.full(n, np.nan, dtype=float)
    blocked_key = [""] * n
    exit_position = np.full((n, 3), np.nan, dtype=float)
    exit_momentum = np.full((n, 3), np.nan, dtype=float)
    dpa_arrival_time = np.full(n, np.nan, dtype=float)
    dpa_arrival_x = np.full(n, np.nan, dtype=float)
    dpa_arrival_y = np.full(n, np.nan, dtype=float)
    c1_arrival_time = np.full(n, np.nan, dtype=float)
    c1_arrival_x = np.full(n, np.nan, dtype=float)
    c1_arrival_y = np.full(n, np.nan, dtype=float)
    exit_arrival_time = np.full(n, np.nan, dtype=float)
    exit_arrival_x = np.full(n, np.nan, dtype=float)
    exit_arrival_y = np.full(n, np.nan, dtype=float)

    history_position = [position.copy()]
    history_momentum = [momentum.copy()]
    history_time = [0.0]
    history_alive = [alive.copy()]
    history_completed = [completed.copy()]
    history_stride = max(
        1, int(round(float(gun.history_step_mm) / gun.trace_step_mm))
    )
    exit_z_m = gun.exit_plane_z_mm * 1e-3
    dpa_z_m = gun.dpa_aperture.z_mm * 1e-3
    monochromator_installed = bool(
        getattr(gun, "monochromator_installed", False)
    )
    slit = (
        gun.monochromator.slit
        if monochromator_installed
        else None
    )
    slit_plane = gun.c1_aperture if slit is not None else None
    # Adaptive substeps are not counted in units of the requested spatial
    # cap. The old fixed-step allowance prematurely rejected slower bundles
    # (notably thermionic and monochromated guns).
    maximum_steps = max(int(np.ceil(gun.exit_plane_z_mm / gun.trace_step_mm)) * 8, 100000)

    from temsim.physics.analytic_particle_step import (
        prepare_analytic_execution, try_analytic_time_step,
    )
    analytic_execution = (prepare_analytic_execution(
        gun, magnetic_provider, electric_provider, n,
    ) if not geometry_field else None)

    from temsim.physics.analytic_particle_batch import prepare_analytic_batch
    analytic_batch = prepare_analytic_batch(gun, analytic_execution, cancelled)
    step_index = 0
    while step_index < maximum_steps:
        if cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        active = alive & ~completed
        if not np.any(active):
            break
        # A batch stops at every possible physical event and at the next
        # existing history step. The final accepted segment is processed below
        # by the unchanged Python boundary, arrival-time and history routines.
        until_history = history_stride - ((step_index - 1) % history_stride)
        batched = (analytic_batch.advance(
            phase, active, invariant_energy, dpa_passed, c1_passed,
            maximum_steps=min(until_history, maximum_steps-step_index),
            impulse=ANALYTIC_MAXIMUM_RELATIVE_IMPULSE,
        ) if analytic_batch is not None else None)
        if batched is not None:
            phase, previous_position, previous_momentum, previous_time_s, dt, batch_count = batched
            step_index += batch_count-1
        else:
            velocity = velocity_from_momentum_m_per_s(
                phase.momentum_kg_m_per_s[active]
            )
            forward = velocity[:, 2] > 0.0
            if not geometry_field and not np.all(forward):
                indices = np.flatnonzero(active)[~forward]
                alive[indices] = False
                blocked_time_s[indices] = float(phase.time_s)
                for index in indices:
                    blocked_z[index] = phase.position_m[index, 2] * 1000.0
                    blocked_key[index] = (
                        "feg_backstream"
                        if gun.type_key == "cold_feg"
                        else f"{gun.type_key}_backstream"
                    )
                active = alive & ~completed
                if not np.any(active):
                    break
                velocity = velocity_from_momentum_m_per_s(
                    phase.momentum_kg_m_per_s[active]
                )
            # This workspace belongs to one executed trace. begin_step checks the
            # current providers/inputs before sharing preparation across its two
            # kernels; unsupported or changed providers retain the reference path.
            step_execution = (analytic_execution if analytic_execution is not None
                              and analytic_execution.begin_step(phase, active) else None)
            active_z_mm = phase.position_m[active, 2] * 1000.0
            step_m = gun.integration_step_mm_at(active_z_mm) * 1e-3
            dt = step_m / max(float(np.max(velocity[:, 2])), 1.0)
            # The prepared step returns an independently owned phase. Boundary
            # routines only read the previous arrays, and histories own copies.
            previous_position = (phase.position_m if step_execution is not None
                                 else phase.position_m.copy())
            previous_momentum = (phase.momentum_kg_m_per_s if step_execution is not None
                                 else phase.momentum_kg_m_per_s.copy())
            previous_time_s = float(phase.time_s)
            if medium.regions:
                energies = kinetic_energy_ev_from_momentum(phase.momentum_kg_m_per_s[active])
                speed = np.linalg.norm(velocity, axis=1)
                for region in medium.regions:
                    local = (phase.position_m[active, 2]*1000 >= region.start_z_mm) & (phase.position_m[active, 2]*1000 < region.end_z_mm)
                    if np.any(local):
                        rate = region_rate_bound(region, energies[local])*speed[local]
                        if np.max(rate, initial=0) > 0:
                            dt = min(dt, medium.max_step_tau/(2*np.max(rate)))
            if geometry_field:
                # Resolve local extraction impulse for the same unaccelerated
                # source. Only an actual curved conductor needs the apex-scale cap.
                from temsim.physics.relativistic_lorentz import ELEMENTARY_CHARGE_C
                if tip_geometry is not None:
                    nearest = max(0., float(np.min(phase.position_m[active, 2])))
                    spatial_step = min(step_m, .025*(tip_geometry.apex_radius_nm*1e-9+nearest))
                    dt = min(dt, spatial_step/max(np.max(np.linalg.norm(velocity, axis=1)), 1.))
                field_strength = np.linalg.norm(electric_provider.field_at_global_positions_v_per_m(phase.position_m[active]), axis=1)
                momentum_size = np.linalg.norm(phase.momentum_kg_m_per_s[active], axis=1)
                if np.any(field_strength > 0):
                    dt = min(dt, float(np.min(.025*momentum_size[field_strength>0]/(ELEMENTARY_CHARGE_C*field_strength[field_strength>0]))))
                advanced, dt = _surface_step(phase, dt, active, magnetic_provider, electric_provider)
            else:
                # A spatial cap alone is unsafe at emission: a sub-eV electron can
                # gain keV within that step. Resolve the local Lorentz impulse
                # before applying Boris; energy projection cannot repair a wrong
                # direction or a skipped extraction trajectory.
                if step_execution is not None:
                    bounded_dt = step_execution.time_step(dt, ANALYTIC_MAXIMUM_RELATIVE_IMPULSE)
                elif getattr(gun, "compiled_particle_steps", True):
                    bounded_dt = try_analytic_time_step(
                        phase, dt, active, magnetic_provider, electric_provider,
                        ANALYTIC_MAXIMUM_RELATIVE_IMPULSE,
                    )
                else:
                    bounded_dt = None
                if bounded_dt is not None:
                    dt = bounded_dt
                else:
                    from temsim.physics.relativistic_lorentz import lorentz_derivative
                    _, force = lorentz_derivative(
                        phase.position_m[active], phase.momentum_kg_m_per_s[active],
                        magnetic_provider, electric_field=electric_provider,
                    )
                    # Include the same predicted midpoint as Boris. At the compact
                    # extractor-field entrance the current force can be exactly zero.
                    _, midpoint_force = lorentz_derivative(
                        phase.position_m[active] + .5 * dt * velocity,
                        phase.momentum_kg_m_per_s[active], magnetic_provider,
                        electric_field=electric_provider,
                    )
                    force_size = np.maximum(np.linalg.norm(force, axis=1),
                                            np.linalg.norm(midpoint_force, axis=1))
                    nonzero = force_size > 0
                    if np.any(nonzero):
                        momentum_size = np.linalg.norm(phase.momentum_kg_m_per_s[active], axis=1)
                        dt = min(dt, float(np.min(ANALYTIC_MAXIMUM_RELATIVE_IMPULSE
                                                 * momentum_size[nonzero] / force_size[nonzero])))
                advanced, dt = _analytic_step(
                    gun, phase, dt, active, magnetic_provider, electric_provider,
                    invariant_energy[active], execution=step_execution)
            if step_execution is not None:
                # Preserve the existing outer projection, including its validation
                # and floating-point ordering, without recopying the entire phase.
                phase = advanced
                active_position = phase.position_m[active]
                active_momentum = phase.momentum_kg_m_per_s[active]
                active_energy = invariant_energy[active]
                projected = step_execution.project_momentum(
                    active_position, active_momentum, active_energy)
                if projected is None:
                    projected = _enforce_static_field_energy(
                        gun, active_position, active_momentum, active_energy)
                phase.momentum_kg_m_per_s[active] = projected
            else:
                new_position = phase.position_m.copy()
                new_momentum = phase.momentum_kg_m_per_s.copy()
                new_position[active] = advanced.position_m[active]
                if geometry_field:
                    # A solved field evolves momentum without energy projection.
                    new_momentum[active] = advanced.momentum_kg_m_per_s[active]
                else:
                    new_momentum[active] = _enforce_static_field_energy(
                        gun, advanced.position_m[active], advanced.momentum_kg_m_per_s[active], invariant_energy[active])
                phase = RelativisticPhaseSpace(
                    new_position, new_momentum, phase.time_s + dt
                )

        if tip_material_mask is not None:
            returned = np.flatnonzero(alive & ~completed & tip_material_mask(phase.position_m))
            alive[returned] = False
            blocked_z[returned] = phase.position_m[returned, 2]*1000
            for index in returned:
                blocked_key[index] = "feg_tip_reabsorbed"

        _clip_body_bores(
            gun, previous_position, phase.position_m,
            alive, completed, blocked_z, blocked_key,
        )
        if grounded_liner:
            _clip_grounded_liner(
                grounded_liner, previous_position, phase.position_m,
                active & ~completed, alive, blocked_z, blocked_key,
            )
        _resolve_aperture_crossing(
            gun.dpa_aperture,
            dpa_z_m,
            previous_position,
            previous_momentum,
            phase.position_m,
            phase.momentum_kg_m_per_s,
            alive,
            completed,
            blocked_z,
            blocked_key,
            passed=dpa_passed,
            previous_time_s=previous_time_s,
            new_time_s=float(phase.time_s),
            arrival_time_s=dpa_arrival_time,
            arrival_x_m=dpa_arrival_x,
            arrival_y_m=dpa_arrival_y,
            **({"electric": electric_provider, "magnetic": magnetic_provider}
               if geometry_field else {}),
        )
        extra = {"electric": electric_provider, "magnetic": magnetic_provider} if geometry_field else {}
        # C1 owns either its circular opening or the bound monochromator slit
        # and mechanical bore. Apply it once, where the component is installed.
        # The column handoff is a propagation plane, not another aperture.
        _resolve_aperture_crossing(
            gun.c1_aperture,
            gun.c1_aperture.z_mm * 1.0e-3,
            previous_position,
            previous_momentum,
            phase.position_m,
            phase.momentum_kg_m_per_s,
            alive,
            completed,
            blocked_z,
            blocked_key,
            passed=c1_passed,
            previous_time_s=previous_time_s,
            new_time_s=float(phase.time_s),
            arrival_time_s=c1_arrival_time,
            arrival_x_m=c1_arrival_x,
            arrival_y_m=c1_arrival_y,
            **extra,
        )
        exit_resolver = _resolve_surface_exit_crossing if geometry_field else _resolve_exit_crossing
        exit_resolver(
            exit_z_m,
            previous_position,
            previous_momentum,
            phase.position_m,
            phase.momentum_kg_m_per_s,
            alive,
            completed,
            exit_position,
            exit_momentum,
            previous_time_s=previous_time_s,
            new_time_s=float(phase.time_s),
            arrival_time_s=exit_arrival_time,
            arrival_x_m=exit_arrival_x,
            arrival_y_m=exit_arrival_y,
            **extra,
        )
        if medium.regions:
            # Account only for the executed segment up to the earliest real
            # stop or gun exit. Rotate momentum without changing its magnitude.
            end = phase.position_m.copy()
            stopped = active & np.isfinite(blocked_z)
            dz = end[:, 2]-previous_position[:, 2]
            f = np.divide(blocked_z*1e-3-previous_position[:, 2], dz,
                          out=np.ones(n), where=np.abs(dz) > 1e-30)
            end[stopped] = previous_position[stopped]+np.clip(f[stopped], 0, 1)[:, None]*(end[stopped]-previous_position[stopped])
            end[completed & active] = exit_position[completed & active]
            momenta = phase.momentum_kg_m_per_s.copy()
            momenta[completed & active] = exit_momentum[completed & active]
            energy = kinetic_energy_ev_from_momentum(momenta)
            medium.alive = active.copy()
            directions = medium.advance(previous_position, end, momenta, energy)
            rotated = directions/np.linalg.norm(directions, axis=1, keepdims=True)*np.linalg.norm(momenta, axis=1, keepdims=True)
            momenta[active] = rotated[active]
            new_exit = active & completed & medium.alive
            exit_momentum[new_exit] = rotated[new_exit]
            phase = RelativisticPhaseSpace(phase.position_m, momenta, phase.time_s)
            alive, blocked_z, blocked_key = medium.merge_stops(alive, blocked_z, blocked_key)
            medium_stopped = np.array([key.startswith("medium_removal:") for key in blocked_key])
            dpa_passed[medium_stopped & (blocked_z <= gun.dpa_aperture.z_mm)] = False
            missed_dpa = medium_stopped & (blocked_z <= gun.dpa_aperture.z_mm)
            for values in (dpa_arrival_time, dpa_arrival_x, dpa_arrival_y):
                values[missed_dpa] = np.nan
            missed_exit = medium_stopped & (blocked_z <= gun.exit_plane_z_mm)
            for values in (exit_arrival_time, exit_arrival_x, exit_arrival_y):
                values[missed_exit] = np.nan
            missed_c1 = medium_stopped & (blocked_z <= gun.c1_aperture.z_mm)
            c1_passed[missed_c1] = False
            for values in (c1_arrival_time, c1_arrival_x, c1_arrival_y):
                values[missed_c1] = np.nan
            completed &= alive
        newly_stopped = active & ~alive & ~np.isfinite(blocked_time_s)
        if np.any(newly_stopped):
            # Reuse the accepted segment and the already-resolved physical
            # stop. A body or tip reabsorption is detected at the step end;
            # plane and medium stops retain their position within that step.
            indices = np.flatnonzero(newly_stopped)
            dz = phase.position_m[indices, 2] - previous_position[indices, 2]
            fraction = np.divide(
                blocked_z[indices] * 1e-3 - previous_position[indices, 2], dz,
                out=np.ones(indices.size), where=dz != 0.,
            )
            blocked_time_s[indices] = previous_time_s + np.clip(fraction, 0., 1.) * dt
            # A removal position stores only Z. If that segment has zero axial
            # travel, its sampled removal time cannot be recovered from Z.
            unknown = [index for index, delta in zip(indices, dz)
                       if delta == 0. and blocked_key[index].startswith("medium_removal:")]
            blocked_time_s[unknown] = np.nan
            for aperture, arrival in ((gun.dpa_aperture, dpa_arrival_time),
                                      (gun.c1_aperture, c1_arrival_time)):
                at_plane = newly_stopped & np.isfinite(arrival) & (blocked_z == aperture.z_mm)
                blocked_time_s[at_plane] = arrival[at_plane]
        if (
            step_index % history_stride == 0
            or not np.any(alive & ~completed)
        ):
            snapshot_position = phase.position_m.copy()
            snapshot_momentum = phase.momentum_kg_m_per_s.copy()
            if np.any(completed):
                snapshot_position[completed] = exit_position[completed]
                snapshot_momentum[completed] = exit_momentum[completed]
            history_position.append(snapshot_position)
            history_momentum.append(snapshot_momentum)
            history_time.append(float(phase.time_s))
            history_alive.append(alive.copy())
            history_completed.append(completed.copy())
        step_index += 1
    else:
        raise RuntimeError(
            f"{gun.display_name} trace did not reach its exit plane "
            "within the step limit."
        )

    passed_exit = alive & completed
    if not geometry_field and np.any(passed_exit):
        exit_momentum[passed_exit] = _enforce_static_field_energy(
            gun,
            exit_position[passed_exit],
            exit_momentum[passed_exit],
            invariant_energy[passed_exit],
        )
    pz = exit_momentum[:, 2]
    tx = np.divide(
        exit_momentum[:, 0], pz,
        out=np.zeros(n, dtype=float), where=passed_exit,
    )
    ty = np.divide(
        exit_momentum[:, 1], pz,
        out=np.zeros(n, dtype=float), where=passed_exit,
    )
    equal_time_history = _finalize_gun_history(
        history_position, history_momentum, history_time, history_alive,
        history_completed, cancelled=cancelled,
    )
    common_z, path_x, path_y, path_tx, path_ty = _resample_gun_paths(
        gun,
        None,
        equal_time_history.tx_rad,
        equal_time_history.ty_rad,
        exit_position,
        tx,
        ty,
        passed_exit,
        blocked_z,
        slit_plane,
        equal_time_history=equal_time_history,
        cancelled=cancelled,
    )
    plane_arrivals = [
        GunPlaneArrival(
            key=str(gun.dpa_aperture.key),
            name=str(gun.dpa_aperture.name),
            z_mm=float(gun.dpa_aperture.z_mm),
            time_s=dpa_arrival_time,
            x_m=dpa_arrival_x,
            y_m=dpa_arrival_y,
            reached=np.isfinite(dpa_arrival_time),
            transmitted=dpa_passed.copy(),
        )
    ]
    plane_arrivals.append(GunPlaneArrival(
        key=str(gun.c1_aperture.key),
        name=str(gun.c1_aperture.label),
        z_mm=float(gun.c1_aperture.z_mm),
        time_s=c1_arrival_time,
        x_m=c1_arrival_x,
        y_m=c1_arrival_y,
        reached=np.isfinite(c1_arrival_time),
        transmitted=c1_passed.copy(),
    ))
    slit_passed = c1_passed
    slit_reached = np.isfinite(c1_arrival_time)
    slit_x_m = c1_arrival_x
    if slit is not None:
        plane_arrivals.append(GunPlaneArrival(
            key=FEG_MONOCHROMATOR_SLIT,
            name=str(slit.name),
            z_mm=float(slit_plane.z_mm),
            time_s=c1_arrival_time,
            x_m=slit_x_m.copy(),
            y_m=c1_arrival_y,
            reached=slit_reached.copy(),
            transmitted=slit_passed.copy(),
        ))
    plane_arrivals.append(GunPlaneArrival(
        key=f"{gun.c1_aperture.key}:exit",
        name=f"{gun.display_name} Exit",
        z_mm=float(gun.exit_plane_z_mm),
        time_s=exit_arrival_time,
        x_m=exit_arrival_x,
        y_m=exit_arrival_y,
        reached=np.isfinite(exit_arrival_time),
        transmitted=passed_exit.copy(),
    ))
    terminal_z_mm = np.where(passed_exit, float(gun.exit_plane_z_mm), blocked_z)
    terminal_time_s = np.where(passed_exit, exit_arrival_time, blocked_time_s)
    flight_time_s = _resample_gun_flight_times(
        common_z, equal_time_history, plane_arrivals=plane_arrivals,
        terminal_z_mm=terminal_z_mm, terminal_time_s=terminal_time_s,
        path_outputs=(path_x, path_y, path_tx, path_ty),
        terminal_path_values=(exit_position[:, 0], exit_position[:, 1], tx, ty),
        terminal_path_valid=passed_exit,
        cancelled=cancelled,
    )
    # The downstream energy is referenced to nominal HT, not to the launch
    # mean. Preserve K - e*phi across the same physical endpoints used by the
    # executed trajectory. Copying launch offsets is only valid when launch
    # potential is zero and the analytic ramp reaches its nominal exit value.
    energy_offset = emitted.energy_offset_ev.copy()
    if geometry_field:
        # Carry the actually integrated energy through the handoff. A finite
        # grounded outlet can retain residual potential at the selected plane.
        energy_offset[passed_exit] = (
            kinetic_energy_ev_from_momentum(exit_momentum[passed_exit])
            - gun.nominal_exit_energy_ev
        )
    elif np.any(passed_exit):
        nominal_potential_rise = gun.nominal_exit_energy_ev - float(gun.emitter.emission_energy_ev)
        exit_potential = electric_provider.potential_v_at_global_positions(exit_position[passed_exit])
        launch_potential = launch_energy - invariant_energy
        # Preserve the original small offsets exactly for the historical
        # field-free endpoints; only add the actual potential correction.
        energy_offset[passed_exit] += (exit_potential - nominal_potential_rise
                                       - launch_potential[passed_exit])
    current = gun.emitted_current_a
    monochromator_current = None
    slit_dispersion = None
    if slit is not None:
        monochromator_current = (
            current * float(np.sum(emitted.weight[slit_passed]))
        )
        indices = np.flatnonzero(slit_reached)
        if (
            indices.size >= 2
            and np.ptp(energy_offset[indices]) > 0.0
        ):
            slit_dispersion = float(
                np.polyfit(
                    energy_offset[indices],
                    slit_x_m[indices] * 1.0e6,
                    1,
                )[0]
            )
    output_fwhm = _weighted_fwhm_from_standard_deviation(
        energy_offset[passed_exit],
        emitted.weight[passed_exit],
    )
    result = GunTraceResult(
        z_mm=common_z,
        x_m=path_x,
        y_m=path_y,
        tx_rad=path_tx,
        ty_rad=path_ty,
        exit_bundle=GunExitBundle(
            x_m=np.nan_to_num(exit_position[:, 0]),
            y_m=np.nan_to_num(exit_position[:, 1]),
            tx_rad=tx,
            ty_rad=ty,
            energy_offset_ev=energy_offset,
            weight=emitted.weight,
            ray_id=emitted.ray_id,
            alive=passed_exit,
            flight_time_s=exit_arrival_time.copy(),
        ),
        blocked_z_mm=blocked_z,
        blocked_key=tuple(blocked_key),
        emitted_current_a=current,
        dpa_transmitted_current_a=(
            current * float(np.sum(emitted.weight[dpa_passed]))
        ),
        c1_transmitted_current_a=(
            current * float(np.sum(emitted.weight[c1_passed]))
        ),
        monochromator_transmitted_current_a=monochromator_current,
        output_energy_fwhm_ev=output_fwhm,
        slit_dispersion_um_per_ev=slit_dispersion,
        slit_x_m=(slit_x_m if slit is not None else None),
        slit_reached=(slit_reached if slit is not None else None),
        equal_time_history=equal_time_history,
        plane_arrivals=tuple(plane_arrivals),
        vacuum_report=medium.report() if medium.regions else None,
        emission_reference=launch_reference,
        flight_time_s=flight_time_s,
    )
    if geometry_field:
        kinetic = kinetic_energy_ev_from_momentum(exit_momentum[passed_exit])
        exit_potential = trace_potential(exit_position[passed_exit])
        expected = invariant_energy[passed_exit]+exit_potential
        energy_error = float(np.max(np.abs(kinetic-expected), initial=0))
        if energy_error > 1e-3:
            raise ValueError(f"Electrode-field trace exit energy error {energy_error:.6g} eV exceeds 0.001 eV; result not accepted")
        report = {
            "model": (surface_model.schema if surface_model is not None
                      else electric_base.request["schema"]),
            "surface_mesh_displacement_m": surface_mesh_error,
            "potential_reference": "work_relative_to_executed_emission_positions",
            "exit_energy_status": "verified" if kinetic.size else "no_transmitted_particles",
            "transmitted_particle_count": int(kinetic.size),
            "minimum_exit_kinetic_energy_ev": float(kinetic.min()) if kinetic.size else None,
            "maximum_exit_kinetic_energy_ev": float(kinetic.max()) if kinetic.size else None,
            "minimum_expected_exit_kinetic_energy_ev": float(expected.min()) if kinetic.size else None,
            "maximum_expected_exit_kinetic_energy_ev": float(expected.max()) if kinetic.size else None,
            "maximum_exit_energy_error_ev": energy_error if kinetic.size else None,
            "exit_energy_error_budget_ev": 1e-3,
            "accelerating_voltage_kv": float(gun.accelerator.high_tension_kv),
            "extraction_voltage_kv": float(gun.extractor.voltage_kv),
            "electrostatic_field": dict(electric_base.report),
            "launch_boundary": launch_boundary_report,
            "scope": "classical prescribed outgoing flux; no coherent phase or tunnelling prediction",
        }
        object.__setattr__(result, "electrostatic_model_report", report)
        if surface_model is not None:
            object.__setattr__(result, "surface_model_report", report)
    return result


def _analytic_step(gun, phase, dt, active, magnetic, electric, invariant_energy, *, execution=None):
    """Bound transverse trajectory error as well as the static energy error.

    Energy projection is part of each compared step, not an error estimator.
    The two-half-step solution is accepted only after comparison to one full
    step. Stopped rays cannot constrain the live population's time step.
    """
    from temsim.physics.analytic_particle_step import try_analytic_step
    compiled = (execution.step(dt, invariant_energy) if execution is not None else
                try_analytic_step(gun, phase, dt, active, magnetic, electric, invariant_energy))
    if compiled is not None:
        return compiled
    local = RelativisticPhaseSpace(phase.position_m[active],
                                  phase.momentum_kg_m_per_s[active], phase.time_s)

    def advance(start, duration):
        result = boris_step(start, duration, magnetic, electric_field=electric)
        momentum = _enforce_static_field_energy(gun, result.position_m,
                                               result.momentum_kg_m_per_s, invariant_energy)
        return RelativisticPhaseSpace(result.position_m, momentum, result.time_s)

    for _ in range(32):
        full = advance(local, dt)
        half = advance(local, .5 * dt)
        fine = advance(half, .5 * dt)
        p = fine.momentum_kg_m_per_s
        pscale = np.maximum(np.linalg.norm(p, axis=1), 1e-30)
        delta_p = fine.momentum_kg_m_per_s - full.momentum_kg_m_per_s
        transverse_scale = np.maximum(np.linalg.norm(p[:, :2], axis=1), pscale * 1e-5)
        transverse_error = np.linalg.norm(delta_p[:, :2], axis=1) / transverse_scale
        transverse_position_error = np.linalg.norm(
            fine.position_m[:, :2] - full.position_m[:, :2], axis=1)
        position_budget = 1e-13 + 1e-6 * np.linalg.norm(fine.position_m[:, :2], axis=1)
        if (np.all(transverse_error <= 1e-5)
                and np.all(np.linalg.norm(delta_p, axis=1) / pscale <= 1e-6)
                and np.all(transverse_position_error <= position_budget)):
            positions, momenta = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
            positions[active], momenta[active] = fine.position_m, p
            return RelativisticPhaseSpace(positions, momenta, phase.time_s + dt), dt
        dt *= .5
    raise ValueError("Analytic gun integration did not meet the transverse error budget")


def _surface_step(phase, dt, active, magnetic, electric):
    """Step-doubled discrete-gradient Lorentz update, without energy projection."""
    from temsim.physics.static_energy_lorentz import static_energy_step
    # Evaluate only live trajectories: stopped rays must not constrain later
    # time steps or sample fields outside the physical domain.
    local = RelativisticPhaseSpace(phase.position_m[active], phase.momentum_kg_m_per_s[active], phase.time_s)
    for _ in range(32):
        try:
            full = static_energy_step(local, dt, magnetic, electric)
            half = static_energy_step(local, .5*dt, magnetic, electric)
            fine = static_energy_step(half, .5*dt, magnetic, electric)
        except ValueError as error:
            if "iteration did not converge" not in str(error):
                raise
            dt *= .5
            continue
        pscale = np.maximum(np.linalg.norm(fine.momentum_kg_m_per_s, axis=1), 1e-30)
        perr = np.max(np.linalg.norm(fine.momentum_kg_m_per_s-full.momentum_kg_m_per_s, axis=1)/pscale)
        xerr = np.max(np.linalg.norm(fine.position_m-full.position_m, axis=1))
        if perr <= 1e-6 and xerr <= max(1e-13, np.max(np.linalg.norm(fine.position_m-local.position_m, axis=1))*1e-5):
            positions, momenta = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
            positions[active], momenta[active] = fine.position_m, fine.momentum_kg_m_per_s
            return RelativisticPhaseSpace(positions, momenta, phase.time_s+dt), dt
        dt *= .5
    raise ValueError("Surface extraction integration did not meet the local error budget")


def _surface_plane_crossing(plane_z_m, position, momentum, previous_time_s,
                            duration, electric, magnetic):
    """Locate a plane on the same two-half-step path as the accepted update."""
    from scipy.optimize import brentq
    from temsim.physics.static_energy_lorentz import static_energy_step
    start = RelativisticPhaseSpace(position[None, :], momentum[None, :], previous_time_s)
    def at_fraction(fraction):
        if fraction == 0:
            return start
        half = static_energy_step(start, .5*duration*fraction, magnetic, electric)
        return static_energy_step(half, .5*duration*fraction, magnetic, electric)
    def residual(fraction):
        return float(at_fraction(fraction).position_m[0, 2]-plane_z_m)
    fraction = brentq(residual, 0., 1., xtol=1e-12)
    return at_fraction(fraction)


def _resolve_surface_exit_crossing(exit_z_m, previous_position, previous_momentum,
        new_position, new_momentum, alive, completed, exit_position, exit_momentum,
        *, previous_time_s, new_time_s, arrival_time_s, arrival_x_m, arrival_y_m,
        electric, magnetic):
    """Complete transport at the handoff without imposing a second C1 mask.

    Linear momentum interpolation loses work at the end of the accelerating
    field, so retain the existing energy-conserving event integration.
    """
    indices = np.flatnonzero(alive & ~completed & (previous_position[:, 2] < exit_z_m)
                             & (new_position[:, 2] >= exit_z_m))
    duration = new_time_s-previous_time_s
    for index in indices:
        crossing = _surface_plane_crossing(exit_z_m, previous_position[index],
            previous_momentum[index], previous_time_s, duration, electric, magnetic)
        position = crossing.position_m[0]
        arrival_time_s[index] = crossing.time_s
        arrival_x_m[index], arrival_y_m[index] = position[:2]
        exit_position[index] = position
        exit_momentum[index] = crossing.momentum_kg_m_per_s[0]
        completed[index] = True


def _finalize_gun_history(positions, momenta, times, alive, completed, *, cancelled=None):
    """Consume saved rows into the unchanged float64 equal-time schema.

    Convert momenta first, releasing each row instead of building a second full
    momentum history. Coordinates own only their declared arrays, so X/Y views
    do not retain a hidden three-vector base alongside the public Z array.
    """
    shape = (len(positions), len(positions[0]))
    tx, ty = np.empty(shape), np.empty(shape)
    for row, momentum in enumerate(momenta):
        if row % 64 == 0 and cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        pz = momentum[:, 2]
        nonzero = np.abs(pz) > 0.
        tx[row] = np.divide(momentum[:, 0], pz, out=np.zeros(shape[1]), where=nonzero)
        ty[row] = np.divide(momentum[:, 1], pz, out=np.zeros(shape[1]), where=nonzero)
        momenta[row] = None
    x, y, z = np.empty(shape), np.empty(shape), np.empty(shape)
    for row, position in enumerate(positions):
        if row % 64 == 0 and cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        x[row], y[row], z[row] = position[:, 0], position[:, 1], position[:, 2] * 1000.
        positions[row] = None
    return GunEqualTimeHistory(
        time_s=np.asarray(times, dtype=float), z_mm=z, x_m=x, y_m=y,
        tx_rad=tx, ty_rad=ty, alive=np.asarray(alive, dtype=bool),
        completed=np.asarray(completed, dtype=bool),
    )


def _retained_axial_indices(values):
    """Original chronological record test and its 1e-12 mm threshold."""
    indices = np.empty(len(values), dtype=np.int64)
    count, previous = 0, -math.inf
    for index in range(len(values)):
        value = values[index]
        if math.isfinite(value) and value > previous + 1e-12:
            indices[count] = index
            count += 1
            previous = value
    return indices[:count]


from temsim.physics.analytic_gun_field import njit as _history_njit
_history_indices = (_history_njit(cache=True, nogil=True)(_retained_axial_indices)
                    if _history_njit is not None else _retained_axial_indices)


def _resample_gun_paths(
    gun,
    equal_time_position,
    equal_time_tx,
    equal_time_ty,
    exit_position,
    exit_tx,
    exit_ty,
    passed_exit,
    blocked_z_mm,
    slit_plane,
    *,
    equal_time_history=None,
    cancelled=None,
):
    """Return ray paths on one strict, shared axial grid.

    The Boris solver advances all electrons at common laboratory times, so
    their instantaneous Z coordinates differ. Those equal-time snapshots must
    not be plotted against one scalar Z value. Each ray is instead interpolated
    independently onto this shared plane grid; the original equal-time data is
    retained separately on ``GunTraceResult``.
    """

    exit_z = float(gun.exit_plane_z_mm)
    history_step = float(gun.history_step_mm)
    axis_values = list(np.arange(0.0, exit_z, history_step, dtype=float))
    axis_values.extend((0.0, exit_z, float(gun.dpa_aperture.z_mm)))
    if slit_plane is not None:
        axis_values.append(float(slit_plane.z_mm))
    for component in getattr(gun, "components", ()):
        for attribute in ("z_mm", "upper_z_mm", "lower_z_mm"):
            value = getattr(component, attribute, None)
            if value is not None and 0.0 <= float(value) <= exit_z:
                axis_values.append(float(value))
    common_z = np.unique(np.asarray(axis_values, dtype=float))

    if equal_time_history is None:
        raw_z = np.asarray(equal_time_position[..., 2], dtype=float) * 1000.0
        raw_values = (np.asarray(equal_time_position[..., 0], dtype=float),
                      np.asarray(equal_time_position[..., 1], dtype=float),
                      np.asarray(equal_time_tx, dtype=float), np.asarray(equal_time_ty, dtype=float))
    else:
        raw_z = equal_time_history.z_mm
        raw_values = (equal_time_history.x_m, equal_time_history.y_m,
                      equal_time_history.tx_rad, equal_time_history.ty_rad)
    ray_count = raw_z.shape[1]
    outputs = [
        np.empty((common_z.size, ray_count), dtype=float)
        for _ in raw_values
    ]

    for ray in range(ray_count):
        if ray % 64 == 0 and cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        if bool(passed_exit[ray]):
            limit = exit_z
            endpoint = (
                float(exit_position[ray, 0]),
                float(exit_position[ray, 1]),
                float(exit_tx[ray]),
                float(exit_ty[ray]),
            )
        else:
            blocked = float(blocked_z_mm[ray])
            limit = blocked if math.isfinite(blocked) else float(np.nanmax(raw_z[:, ray]))
            limit = min(max(limit, 0.0), exit_z)
            endpoint = None

        keep = _history_indices(raw_z[:, ray])
        if not keep.size:
            raise RuntimeError("Electron-gun trace contains no finite path samples")

        monotonic_z = raw_z[keep, ray]
        if endpoint is None:
            endpoint = tuple(
                float(np.interp(limit, monotonic_z, values[keep, ray]))
                for values in raw_values
            )
        before = monotonic_z < limit - 1.0e-12
        path_z = monotonic_z[before]
        if path_z.size == 0 or path_z[0] > 1.0e-12:
            path_z = np.r_[0.0, path_z]
            start_values = tuple(float(values[keep[0], ray]) for values in raw_values)
        else:
            start_values = None
        path_z = np.r_[path_z, limit]

        for output, values, end_value, start_value in zip(
            outputs,
            raw_values,
            endpoint,
            start_values or (None,) * len(raw_values),
        ):
            path_values = values[keep, ray][before]
            if start_value is not None:
                path_values = np.r_[start_value, path_values]
            path_values = np.r_[path_values, end_value]
            output[:, ray] = np.interp(
                common_z, path_z, path_values,
                left=path_values[0], right=end_value,
            )
    return common_z, *outputs


def _resample_gun_flight_times(
    common_z_mm,
    equal_time_history,
    *,
    plane_arrivals=(),
    terminal_z_mm=None,
    terminal_time_s=None,
    path_outputs=None,
    terminal_path_values=None,
    terminal_path_valid=None,
    cancelled=None,
):
    """First arrival at each plane, from the executed laboratory-time path.

    Interpolation uses float64 equal-time trajectories, with resolved plane
    events inserted at their actual crossing times. The terminal event replaces
    an overshooting/frozen history endpoint. A turning path is searched in time
    order, not sorted by Z: earlier crossings survive a later return. No time
    is extrapolated before emission or to an unvisited axial plane.

    Optional existing X/Y/slope display arrays are updated in place using the
    same chronological crossing segments. Their untimed readability tails are
    retained. Intermediate coordinates remain interpolated display data; only
    recorded plane X/Y values and supplied terminal states are exact events.
    """
    planes = np.asarray(common_z_mm, dtype=np.float64)
    times = np.asarray(equal_time_history.time_s, dtype=np.float64)
    z_history = np.asarray(equal_time_history.z_mm, dtype=np.float64)
    if (planes.ndim != 1 or not np.all(np.isfinite(planes))
            or times.ndim != 1 or not times.size
            or not np.all(np.isfinite(times)) or times[0] != 0.
            or np.any(np.diff(times) <= 0.) or z_history.ndim != 2
            or z_history.shape[0] != times.size):
        raise ValueError("Gun timing requires finite planes and increasing tip-origin history times")
    ray_count = z_history.shape[1]
    active = (np.asarray(equal_time_history.alive, dtype=bool)
              & ~np.asarray(equal_time_history.completed, dtype=bool))
    if active.shape != z_history.shape:
        raise ValueError("Gun timing activity must match the time-by-ray history")
    terminal_z = (np.full(ray_count, np.nan) if terminal_z_mm is None
                  else np.asarray(terminal_z_mm, dtype=np.float64))
    terminal_time = (np.full(ray_count, np.nan) if terminal_time_s is None
                     else np.asarray(terminal_time_s, dtype=np.float64))
    if terminal_z.shape != (ray_count,) or terminal_time.shape != (ray_count,):
        raise ValueError("Gun timing terminal events must contain one value per ray")
    history_values = None
    if path_outputs is not None:
        if (len(path_outputs) != 4
                or any(np.shape(output) != (planes.size, ray_count) for output in path_outputs)):
            raise ValueError("Gun display outputs must contain four plane-by-ray arrays")
        history_values = tuple(np.asarray(values, dtype=np.float64) for values in (
            equal_time_history.x_m, equal_time_history.y_m,
            equal_time_history.tx_rad, equal_time_history.ty_rad,
        ))
        if any(values.shape != z_history.shape for values in history_values):
            raise ValueError("Gun display history must match the time-by-ray positions")
    if terminal_path_values is not None:
        terminal_path_values = tuple(np.asarray(values, dtype=np.float64)
                                     for values in terminal_path_values)
        if (len(terminal_path_values) != 4
                or any(values.shape != (ray_count,) for values in terminal_path_values)):
            raise ValueError("Gun terminal states must contain four values per ray")
    terminal_path_valid = (np.zeros(ray_count, dtype=bool) if terminal_path_valid is None
                           else np.asarray(terminal_path_valid, dtype=bool))
    if terminal_path_valid.shape != (ray_count,):
        raise ValueError("Gun terminal state validity must contain one flag per ray")
    arrivals = []
    for arrival in plane_arrivals:
        crossing_time = np.asarray(arrival.time_s, dtype=np.float64)
        reached = np.asarray(arrival.reached, dtype=bool)
        if crossing_time.shape != (ray_count,) or reached.shape != (ray_count,):
            raise ValueError("Gun plane arrival times must contain one value per ray")
        arrivals.append((float(arrival.z_mm), crossing_time, reached, arrival.x_m, arrival.y_m))
    result = np.full((planes.size, ray_count), np.nan, dtype=np.float64)

    for ray in range(ray_count):
        if ray % 64 == 0 and cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        stop = float(terminal_time[ray])
        known_stop = np.isfinite(stop) and np.isfinite(terminal_z[ray])
        if known_stop:
            if stop < 0.:
                raise ValueError("Gun terminal time cannot precede tip emission")
            count = int(np.searchsorted(times, stop, side="left"))
        else:
            # Without a crossing clock, a frozen or overshooting inactive row
            # cannot supply its own arrival time. Preserve only live samples.
            inactive = np.flatnonzero(~active[:, ray])
            count = int(inactive[0]) if inactive.size else times.size
        # Retained times are already strictly ordered. Keep them as NumPy
        # views; only the few exact terminal/aperture events need sorting.
        path_time = times[:count]
        path_z = z_history[:count, ray]
        extra_times, extra_z = [], []
        # Only one ray's display values are gathered at a time. The four full
        # histories remain views and the common-Z outputs are reused in place.
        path_values = (None if history_values is None else
                       np.stack([values[:count, ray] for values in history_values]))
        extra_values = []

        def event_values(plane_z, event_time):
            # Locate the retained chronological segment. Interpolate by Z so a
            # normal monotone path retains its original geometric interpolation,
            # even when a resolved event refines its crossing clock.
            upper = min(int(np.searchsorted(times, event_time)), times.size - 1)
            lower = max(upper - 1, 0)
            dz = z_history[upper, ray] - z_history[lower, ray]
            if dz != 0.:
                fraction = (plane_z - z_history[lower, ray]) / dz
            elif times[upper] != times[lower]:
                fraction = (event_time - times[lower]) / (times[upper] - times[lower])
            else:
                fraction = 0.
            fraction = np.clip(fraction, 0., 1.)
            return [values[lower, ray] + fraction * (values[upper, ray] - values[lower, ray])
                    for values in history_values]

        if known_stop:
            extra_times.append(stop)
            extra_z.append(float(terminal_z[ray]))
            if path_values is not None:
                endpoint = (tuple(values[ray] for values in terminal_path_values)
                            if terminal_path_values is not None and terminal_path_valid[ray]
                            else event_values(float(terminal_z[ray]), stop))
                extra_values.append(endpoint)
        exact = []
        for plane_z, event_time, reached, event_x, event_y in arrivals:
            time = float(event_time[ray])
            if reached[ray] and np.isfinite(time) and 0. <= time <= (stop if known_stop else times[-1]):
                extra_times.append(time)
                extra_z.append(plane_z)
                exact.append((plane_z, time, event_x[ray], event_y[ray]))
                if path_values is not None:
                    values_at_event = event_values(plane_z, time)
                    values_at_event[:2] = (event_x[ray], event_y[ray])
                    if (known_stop and time == stop and plane_z == terminal_z[ray]
                            and terminal_path_values is not None and terminal_path_valid[ray]):
                        values_at_event[2:] = (terminal_path_values[2][ray], terminal_path_values[3][ray])
                    extra_values.append(values_at_event)
        if not path_time.size and not extra_times:
            continue
        if extra_times:
            order = np.argsort(extra_times, kind="stable")
            events_t = np.asarray(extra_times, dtype=np.float64)[order]
            insert = np.searchsorted(path_time, events_t, side="right")
            t = np.insert(path_time, insert, events_t)
            z = np.insert(path_z, insert, np.asarray(extra_z, dtype=np.float64)[order])
            if path_values is not None:
                path_values = np.insert(path_values, insert, np.asarray(extra_values).T[:, order], axis=1)
        else:
            t, z = path_time, path_z
        finite = np.isfinite(z)
        t, z = t[finite], z[finite]
        values = None if path_values is None else path_values[:, finite]
        if not t.size:
            continue
        # Explicit physical crossings were appended last and own a coincident
        # time sample instead of a retained step-end position.
        unique = np.r_[t[1:] != t[:-1], True]
        t, z = t[unique], z[unique]
        if values is not None:
            values = values[:, unique]
        result[planes == z[0], ray] = t[0]
        if values is not None:
            for output, coordinate in zip(path_outputs, values):
                output[planes == z[0], ray] = coordinate[0]
        for sign in (1., -1.):
            along = sign * z
            # A new axial maximum/minimum is first reached on the immediately
            # preceding temporal segment, which can start after a reversal.
            records = np.flatnonzero(along[1:] > np.maximum.accumulate(along)[:-1]) + 1
            query_rows = np.flatnonzero(sign * planes > along[0])
            if not records.size or not query_rows.size:
                continue
            queries = sign * planes[query_rows]
            lookup = np.searchsorted(along[records], queries, side="left")
            valid = lookup < records.size
            rows = query_rows[valid]
            upper = records[lookup[valid]]
            fraction = ((sign * planes[rows] - along[upper - 1])
                        / (along[upper] - along[upper - 1]))
            result[rows, ray] = t[upper - 1] + fraction * (t[upper] - t[upper - 1])
            if values is not None:
                for output, coordinate in zip(path_outputs, values):
                    output[rows, ray] = coordinate[upper - 1] + fraction * (coordinate[upper] - coordinate[upper - 1])
        for plane_z, time, event_x, event_y in exact:
            result[planes == plane_z, ray] = time
            if path_outputs is not None:
                path_outputs[0][planes == plane_z, ray] = event_x
                path_outputs[1][planes == plane_z, ray] = event_y
    return result


def _weighted_fwhm_from_standard_deviation(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if values.size < 2 or total <= 0.0:
        return 0.0
    mean = float(np.sum(values * weights) / total)
    variance = float(
        np.sum(weights * (values - mean) ** 2) / total
    )
    return 2.354820045 * math.sqrt(max(0.0, variance))


def _enforce_static_field_energy(gun, position, momentum, launch_energy_ev):
    """Preserve K - e*phi for an electron in the static gun fields."""

    potential_v = gun.electric_field.potential_v_at_global_positions(position)
    target_energy_ev = np.maximum(
        np.asarray(launch_energy_ev, dtype=float) + potential_v,
        1.0e-6,
    )
    direction = np.asarray(momentum, dtype=float)
    return momentum_from_kinetic_energy_ev(target_energy_ev, direction)


def _crossing_fraction(old_z, new_z, plane_z):
    denominator = new_z - old_z
    return np.divide(
        plane_z - old_z,
        denominator,
        out=np.zeros_like(old_z),
        where=np.abs(denominator) > 0.0,
    )


def _resolve_aperture_crossing(
    aperture,
    plane_z_m,
    previous_position,
    previous_momentum,
    new_position,
    new_momentum,
    alive,
    completed,
    blocked_z,
    blocked_key,
    *,
    passed,
    previous_time_s,
    new_time_s,
    arrival_time_s,
    arrival_x_m,
    arrival_y_m,
    electric=None,
    magnetic=None,
):
    candidates = (
        alive
        & ~completed
        & ~passed
        & (previous_position[:, 2] <= plane_z_m)
        & (new_position[:, 2] >= plane_z_m)
    )
    indices = np.flatnonzero(candidates)
    if not indices.size:
        return
    fraction = _crossing_fraction(
        previous_position[indices, 2],
        new_position[indices, 2],
        plane_z_m,
    )
    xy = previous_position[indices, :2] + fraction[:, None] * (
        new_position[indices, :2] - previous_position[indices, :2]
    )
    times = previous_time_s + fraction * (new_time_s - previous_time_s)
    if electric is not None:
        for local, index in enumerate(indices):
            crossing = _surface_plane_crossing(plane_z_m, previous_position[index],
                previous_momentum[index], previous_time_s, new_time_s-previous_time_s,
                electric, magnetic)
            xy[local] = crossing.position_m[0, :2]
            times[local] = crossing.time_s
    arrival_x_m[indices] = xy[:, 0]
    arrival_y_m[indices] = xy[:, 1]
    arrival_time_s[indices] = times
    transmitted = aperture.transmission_mask(
        xy[:, 0] * 1000.0, xy[:, 1] * 1000.0
    )
    passed[indices[transmitted]] = True
    rejected = indices[~transmitted]
    alive[rejected] = False
    for index in rejected:
        blocked_z[index] = aperture.z_mm
        blocked_key[index] = aperture.key


def _resolve_exit_crossing(
    plane_z_m,
    previous_position,
    previous_momentum,
    new_position,
    new_momentum,
    alive,
    completed,
    exit_position,
    exit_momentum,
    *,
    previous_time_s,
    new_time_s,
    arrival_time_s,
    arrival_x_m,
    arrival_y_m,
):
    candidates = (
        alive
        & ~completed
        & (previous_position[:, 2] <= plane_z_m)
        & (new_position[:, 2] >= plane_z_m)
    )
    indices = np.flatnonzero(candidates)
    if not indices.size:
        return
    fraction = _crossing_fraction(
        previous_position[indices, 2],
        new_position[indices, 2],
        plane_z_m,
    )
    positions = previous_position[indices] + fraction[:, None] * (
        new_position[indices] - previous_position[indices]
    )
    momenta = previous_momentum[indices] + fraction[:, None] * (
        new_momentum[indices] - previous_momentum[indices]
    )
    arrival_time_s[indices] = previous_time_s + fraction * (
        new_time_s - previous_time_s
    )
    arrival_x_m[indices] = positions[:, 0]
    arrival_y_m[indices] = positions[:, 1]
    exit_position[indices] = positions
    exit_momentum[indices] = momenta
    completed[indices] = True


def _clip_body_bores(
    gun,
    previous_position,
    new_position,
    alive,
    completed,
    blocked_z,
    blocked_key,
):
    active = np.flatnonzero(alive & ~completed)
    if not active.size:
        return
    z_mm = new_position[active, 2] * 1000.0
    radius_mm = np.hypot(
        new_position[active, 0], new_position[active, 1]
    ) * 1000.0
    for component in gun.bore_components:
        half_length = 0.5 * component.mechanical_length_mm
        inside = np.abs(
            z_mm - component.mechanical_center_from_tip_mm
        ) <= half_length
        outside = radius_mm > (
            0.5 * component.mechanical_clear_bore_diameter_mm
        )
        rejected_local = np.flatnonzero(inside & outside)
        for local_index in rejected_local:
            index = active[local_index]
            if alive[index]:
                alive[index] = False
                blocked_z[index] = z_mm[local_index]
                blocked_key[index] = component.key


def _clip_grounded_liner(rows, previous, current, eligible, alive, blocked_z, blocked_key):
    """Intercept the accepted segment at a grounded tube wall or bore-step face.

    Coordinates and radii are metres. This event interpolation uses physical
    liner rows, not the finite numerical field boundary or a display outline.
    A preceding body stop is retained if it occurs earlier on the same segment.
    """
    indices = np.flatnonzero(eligible)
    if not indices.size:
        return
    start, delta = previous[indices], current[indices]-previous[indices]
    low_z = np.minimum(start[:, 2], current[indices, 2])
    high_z = np.maximum(start[:, 2], current[indices, 2])
    for row in rows:
        z0, z1, radius = row["start_m"], row["stop_m"], row["inner_m"]
        selected = np.flatnonzero((low_z <= z1) & (high_z >= z0))
        if not selected.size:
            continue
        p, d = start[selected], delta[selected]
        first = np.divide(z0-p[:, 2], d[:, 2], out=np.zeros(len(p)), where=d[:, 2] != 0.)
        last = np.divide(z1-p[:, 2], d[:, 2], out=np.ones(len(p)), where=d[:, 2] != 0.)
        low, high = np.maximum(0., np.minimum(first, last)), np.minimum(1., np.maximum(first, last))
        xy = p[:, :2]+low[:, None]*d[:, :2]
        offset = np.sum(xy*xy, axis=1)-radius*radius
        fraction = np.where(offset >= 0., low, np.inf)
        a = np.sum(d[:, :2]**2, axis=1)
        moving = (offset < 0.) & (a > 0.)
        if np.any(moving):
            b = 2.*np.sum(xy[moving]*d[moving, :2], axis=1)
            root = np.sqrt(np.maximum(0., b*b-4.*a[moving]*offset[moving]))
            step = np.empty_like(b)
            positive = b >= 0.
            step[positive] = -2.*offset[moving][positive]/(b[positive]+root[positive])
            step[~positive] = (-b[~positive]+root[~positive])/(2.*a[moving][~positive])
            fraction[moving] = low[moving]+step
        actual = indices[selected]
        preceding = np.isfinite(blocked_z[actual])
        previous_stop = np.full(len(actual), np.inf)
        previous_stop[preceding] = np.clip(np.divide(
            blocked_z[actual][preceding]*1e-3-p[preceding, 2], d[preceding, 2],
            out=np.ones(np.sum(preceding)), where=d[preceding, 2] != 0.), 0., 1.)
        hit = (fraction <= high) & (fraction < previous_stop)
        stopped = actual[hit]
        alive[stopped] = False
        blocked_z[stopped] = (p[hit, 2]+fraction[hit]*d[hit, 2])*1000.
        for index in stopped:
            blocked_key[index] = row["key"]


# Batch admission must not bypass instrumentation or custom event/step hooks.
_BATCH_ORIGINAL_FUNCTIONS = {name: globals()[name] for name in (
    "velocity_from_momentum_m_per_s", "_analytic_step", "_enforce_static_field_energy",
    "_clip_body_bores", "_resolve_aperture_crossing", "_resolve_exit_crossing",
    "_crossing_fraction",
)}
