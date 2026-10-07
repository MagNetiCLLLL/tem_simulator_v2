"""Distributed stationary scalar column operator acting on an executed beam.

Internal stage function, not a configurable source. Public tip-to-detector
requests execute the gun first. Every quadratic field, nonlinear multipole,
physical kick, aperture and sampled vacuum bore is owned by the shared plan.
"""
from dataclasses import asdict, fields as dataclass_fields, is_dataclass, replace
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextvars import copy_context
import math
from threading import Event, Lock

import numpy as np
from scipy.linalg import expm

from temsim.cpu_resources import initialize_numerical_thread, numerical_thread_budget
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.core import build_propagation_plan, electron
from temsim.physics.column_wall import _vacuum_segments, _expanded_profile_axis
from temsim.physics.multiplane_wave import propagate_plane_wave, reselect_phase_carrier
from temsim.physics.multipole_wave import apply_multipole_phase
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError, apply_resolved_operator
from temsim.physics.wave_device import array_module, device_scope, normalise_backend, to_host


def _component_events(state, start, stop, *, arrival_time=None):
    events, owners, seen = [], [], set()
    from temsim.physics.instrument_magnetic import column_dipole_fields
    for component in (*state.deflectors, *getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ())):
        if not getattr(component, "enabled", False):
            continue
        if component.key in seen:
            raise ValueError(f"Duplicate column component {component.key}")
        seen.add(component.key)
        if (getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False)) and hasattr(component, "validate"):
            component.validate()
    for coil in column_dipole_fields(state):
        if not (coil.lower_m < stop*1e-3 and coil.upper_m > start*1e-3):
            continue
        z, x, y = coil.event_z_mm, coil.event_dx_rad, coil.event_dy_rad
        time = None
        if coil.dynamic and arrival_time is not None:
            time = float(arrival_time(z))
            actual = next((item for item in column_dipole_fields(state, time_s=time)
                           if item.key == coil.key), None)
            if actual is None or (actual.lower_m, actual.upper_m, actual.event_z_mm) != (coil.lower_m, coil.upper_m, z):
                raise ValueError("Dynamic coil geometry changed with time")
            x, y = actual.event_dx_rad, actual.event_dy_rad
        events.append((z, x, y))
        owners.append({"component": coil.key.rsplit(":", 1)[0], "drive_keys": coil.drive_keys,
                       "z_mm": z, "kick_rad": (x, y), "dynamic": coil.dynamic,
                       "arrival_time_s": time, "finite_field": True,
                       "effective_thickness_mm": (coil.upper_m-coil.lower_m)*1e3})
    return events, owners


def _prepare_column(state, start, stop, maximum_step_mm):
    if not math.isfinite(stop) or stop <= start:
        raise ValueError("Column wave output must follow the executed input plane")
    if not math.isfinite(maximum_step_mm) or maximum_step_mm <= 0:
        raise ValueError("Column step must be finite and positive")
    if math.ceil((stop-start)/min(float(state.step_mm), maximum_step_mm)) > 1_000_000:
        raise ValueError("Column wave grid exceeds one million steps; choose a feasible numerical budget")
    if bool(getattr(state, "equivalent_image_lenses_enabled", False)):
        raise ValueError("Tip wave transport needs distributed image optics; only executed checkpoints can replace them")
    nano = getattr(state, "nanopulser", None)
    if nano is not None and nano.installed:
        raise ValueError("The installed electrostatic beam blanker needs time-energy wavepacket transport; it cannot be omitted")
    apertures = [a for a in state.apertures if all(bool(getattr(a, n, True))
        for n in ("installed", "enabled", "inserted")) and start < a.z_mm <= stop]
    if len({a.key for a in apertures}) != len(apertures):
        raise ValueError("Duplicate physical column aperture")
    walls = _vacuum_segments(state, np.array((start, stop)))
    boundaries = [z for w in walls for z in (w.start_z_mm, w.end_z_mm) if start < z < stop]
    events, owners = _component_events(state, start, stop)
    plan = build_propagation_plan(state, start, stop, events,
        save_z_mm=[a.z_mm for a in apertures]+boundaries, maximum_step_mm=maximum_step_mm)
    from temsim.physics.wave_field_admission import require_supported_column_wave_fields
    require_supported_column_wave_fields(state, start, stop, plan)
    if plan.mapped_fields:
        raise ValueError("Displaced/tilted magnetic lenses and 3-D field maps currently require particle/ray transport. "
                         "The coherent column-wave solver does not yet include their spatial field Hamiltonian.")
    if np.any(plan.thin_power_m1) or np.any(plan.thin_rotation_rad):
        raise ValueError("Unexecuted equivalent image maps are not admitted")
    axis, _, radii = _expanded_profile_axis(plan.z_mm, walls)
    if not np.array_equal(axis, plan.z_mm):
        raise ValueError("Column wave plan is missing a mechanical boundary")
    stops = {}
    for aperture in apertures:
        index = int(np.argmin(abs(plan.z_mm-aperture.z_mm)))
        if abs(plan.z_mm[index]-aperture.z_mm) > 1e-9:
            raise ValueError("Column wave plan is missing an aperture plane")
        stops.setdefault(index, []).append(aperture)
    return plan, radii, stops, owners


def _linear_factor(path, generator, distance, depth=0, *, force=None):
    """Lift a known constant Hamiltonian; bisect without changing its map."""
    try:
        if force is None or not np.any(force):
            path.append(expm(generator*distance))
        else:
            # Integrate the driven centre and its central Weyl action in the
            # same matrix exponential. a'=1/2(F_p.q-F_q.p), not a phase fitted
            # to the final ray displacement. A uniform force then has the
            # independently known a=F^2 L^3/12 in a unit drift.
            extended = np.zeros((6, 6))
            extended[:4, :4] = generator
            extended[:4, 5] = force
            extended[4, :4] = .5*np.r_[force[2:], -force[:2]]
            result = expm(extended*distance)
            path.append(result[:4, :4], result[:4, 5], action_m=float(result[4, 5]))
    except ValueError as error:
        if "Canonical phase path is undersampled" not in str(error) or depth >= 24:
            raise
        _linear_factor(path, generator, distance/2, depth+1, force=force)
        _linear_factor(path, generator, distance/2, depth+1, force=force)


def _electric_coefficients(plan):
    field = getattr(plan, "electric_field", None)
    if field is None:
        count = len(plan.step_m)
        return np.zeros(count+1), np.zeros(count), np.zeros((count, 2)), np.zeros((count, 2, 2))
    from temsim.physics.wave_field_admission import sample_wave_electric
    nodes = sample_wave_electric(field, plan.z_mm)[0]
    phi, gradient, hessian = sample_wave_electric(field, .5*(plan.z_mm[:-1]+plan.z_mm[1:]))
    return nodes, phi, gradient, hessian


def _column_transports(plan, energy_kev, *, electric=None, dipoles=None):
    """Symplectic maps in fixed entrance-p canonical units during acceleration.

    H/p0=(P+A)^2/(2 p(z) p0)-p(phi(x,y,z))/p0 plus the shared quadrupole
    potentials. Expanding the scalar p(phi) to second order includes both
    electrostatic curvature and the negative relativistic second derivative.
    Lens/stigmator coefficients were defined at instrument reference momentum;
    convert that force to this mode's fixed canonical units rather than changing
    the captured magnetic field when energy changes.
    """
    from scipy.constants import e, m_e
    from temsim.physics.tip_gun_wave import _momentum_velocity
    nodes, phi, gradient, hessian = _electric_coefficients(plan) if electric is None else electric
    entrance_p, _ = _momentum_velocity(energy_kev*1000.)
    momentum, velocity = _momentum_velocity(energy_kev*1000.+phi-nodes[0])
    reference_p = float(getattr(plan, "reference_momentum_kg_m_s", entrance_p))
    scale = reference_p/entrance_p
    strength = (e*hessian/velocity[:, None, None]
        -(m_e*m_e*e*e/momentum**3)[:, None, None]*gradient[:, :, None]*gradient[:, None, :])/entrance_p
    bx = np.asarray(getattr(plan, "dipole_bx_t", np.zeros(3*len(plan.step_m))))[1::3]
    by = np.asarray(getattr(plan, "dipole_by_t", np.zeros(3*len(plan.step_m))))[1::3]
    if dipoles is not None:
        bx, by = dipoles
    for i, dz in enumerate(plan.step_m):
        b = -e*plan.midpoint_magnetic_t[i]/2
        g = b/momentum[i]
        rotation = np.array(((0., g), (-g, 0.)))
        stiffness = np.diag((scale*plan.midpoint_sx_m2[i]+b*b/(entrance_p*momentum[i]),
                             scale*plan.midpoint_sy_m2[i]+b*b/(entrance_p*momentum[i])))
        stiffness[0, 1] = stiffness[1, 0] = scale*plan.midpoint_sxy_m2[i]
        generator = np.block([[rotation, np.eye(2)*entrance_p/momentum[i]],
                              [strength[i]-stiffness, rotation]])
        force = np.r_[np.zeros(2), e*(gradient[i]/velocity[i]+np.array((by[i], -bx[i])))/entrance_p]
        path = CanonicalPath(1e-3)
        if not np.any(rotation) and not np.any(strength[i]-stiffness) and not np.any(force):
            matrix = np.eye(4)
            matrix[:2, 2:] = np.eye(2)*float(dz)*entrance_p/momentum[i]
            path.append(matrix)
        else:
            _linear_factor(path, generator, float(dz), force=force)
        yield path


def _mode_dipoles(plan, owners, mode_owners, state):
    """Re-evaluate finite driven coils without adding a second centre kick."""
    from scipy.constants import e
    count = len(plan.step_m)
    bx = np.asarray(getattr(plan, "dipole_bx_t", np.zeros(3*count)))[1::3].copy()
    by = np.asarray(getattr(plan, "dipole_by_t", np.zeros(3*count)))[1::3].copy()
    reference_p = electron(state)[1]
    z = .5*(plan.z_mm[:-1]+plan.z_mm[1:])
    updated = {(row["component"], row["z_mm"]): row for row in mode_owners}
    for old in owners:
        new = updated[(old["component"], old["z_mm"])]
        if not old.get("finite_field", False):
            continue
        length = old["effective_thickness_mm"]
        active = abs(z-old["z_mm"]) < .5*length
        dx, dy = np.subtract(new["kick_rad"], old["kick_rad"])
        bx[active] -= reference_p*dy/(e*length*1e-3)
        by[active] += reference_p*dx/(e*length*1e-3)
    return bx, by


def _inside_bore(shape, basis_m, origin_m, radius_mm):
    """Prove every sample of an affine grid is strictly inside a circular bore."""
    if not math.isfinite(radius_mm):
        return True  # The existing wall operator is inactive for nonfinite radii.
    ny, nx = shape
    corners = np.array(((-(nx//2), -(ny//2)), (nx-1-nx//2, -(ny//2)),
                        (nx-1-nx//2, ny-1-ny//2), (-(nx//2), ny-1-ny//2)), dtype=float)
    corner_xy = (origin_m[:, None]+basis_m@corners.T)*1e3
    coordinate_bound = (abs(origin_m)+abs(basis_m)@np.max(abs(corners), axis=0))*1e3
    margin = 16*np.finfo(float).eps*np.hypot(*coordinate_bound)
    return np.nextafter(np.max(np.hypot(*corner_xy))+margin, np.inf) < radius_mm


def _linear_run(wave, wavelength, paths, first, plan, radii, stops, kick_x, kick_y):
    """Compose resolved quadratic steps only while every intermediate grid is safe.

    This changes the numerical carrier schedule, never the integrated fields.
    No event or possibly active wall is crossed. All tests use the actual input
    lattice after carrier selection, not a Gaussian fit or a ray envelope.
    """
    from temsim.physics.canonical_action import principal_reference_phase
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    inverse_basis = np.linalg.inv(wave.basis_m)
    basis_condition = np.linalg.cond(wave.basis_m)
    theta = wavelength*np.linalg.norm(inverse_basis, ord=2)
    extent = np.linalg.norm(wave.basis_m, ord=2)*min(wave.amplitude.shape)
    nxny = np.array(wave.amplitude.shape[::-1])
    combined = CanonicalPath(1e-3)
    accepted, last = None, first
    for i in range(first, len(paths)):
        j = i+1
        if (plan.midpoint_hex_normal_m3[i] or plan.midpoint_hex_skew_m3[i]
                or plan.cs_kick_m3[j] or kick_x[j] or kick_y[j] or stops.get(j)):
            break
        child = paths[i]
        # A child with an internally lifted full turn cannot be reconstructed
        # from its endpoint. Keep the stepwise operator for that case.
        if abs(child.reference_phase_rad-principal_reference_phase(
                child.matrix, child.reference_length_m)) > 1e-10:
            break
        try:
            combined.append(child.matrix, child.offset, action_m=child.action_m)
        except ValueError as error:
            if "Canonical phase path is undersampled" not in str(error):
                raise
            break
        a, b = combined.matrix[:2, :2], combined.matrix[:2, 2:]
        effective = a+b@curvature
        condition = np.linalg.cond(effective) if np.isfinite(effective).all() else np.inf
        if condition >= 1e8:
            break
        drift = np.linalg.solve(effective, b)
        # Retain the existing angular-spectrum bound, plus a bound in each
        # sampled direction for skew or highly anisotropic lattices.
        index_drift = wavelength*inverse_basis@drift@inverse_basis.T
        # Do not admit a boundary case merely because inverses/products rounded
        # inward. Ill-conditioned lattices conservatively keep the step path.
        slack = 64*np.finfo(float).eps*max(1., basis_condition, condition)
        if (np.linalg.norm(drift, ord=2)*theta >= (1-slack)*.25*extent
                or np.any(.5*np.sum(abs(index_drift), axis=1) >= (1-slack)*.25*nxny)):
            break
        basis = effective@wave.basis_m
        origin = a@wave.origin_m+b@tilt+combined.offset[:2]
        if not _inside_bore(wave.amplitude.shape, basis, origin, radii[j]):
            break
        # append mutates its owner, so the last accepted path needs a copy.
        accepted = CanonicalPath(combined.reference_length_m)
        accepted.matrix, accepted.offset = combined.matrix.copy(), combined.offset.copy()
        accepted.action_m, accepted.reference_phase_rad = combined.action_m, combined.reference_phase_rad
        last = j
    return last, accepted


def _clip(wave, radius_mm, apertures, z_mm, prior):
    xp = array_module(wave.amplitude)
    xy = None
    rows = []
    if math.isfinite(radius_mm):
        inside_bore = _inside_bore(wave.amplitude.shape, wave.basis_m, wave.origin_m, radius_mm)
        if not inside_bore:
            xy = wave.coordinates_m()*1e3
            before = wave.probability
            mask = xp.hypot(xy[0], xy[1]) < radius_mm
            if not bool(xp.all(mask)):
                wave = replace(wave, amplitude=xp.where(mask, wave.amplitude, 0j))
            if wave.probability < before:
                rows.append({"component": "column_wall", "z_mm": z_mm,
                             "lost_weight": prior*(before-wave.probability)})
    for a in apertures:
        if xy is None:
            xy = wave.coordinates_m()*1e3
        if float(getattr(a, "radius_mm", 1.)) <= 0:
            mask = xp.zeros_like(xy[0], dtype=bool)
        elif hasattr(a, "transmission_mask"):
            # Runtime apertures can supply non-circular/slit geometry. Keep
            # their authoritative mask; only transfer geometry at this event.
            mask = xp.asarray(a.transmission_mask(to_host(xy[0]), to_host(xy[1])), dtype=bool)
        else:
            mask = xp.hypot(xy[0]-a.offset_x_mm, xy[1]-a.offset_y_mm) <= a.radius_mm
        if mask.shape != wave.amplitude.shape:
            raise ValueError("Column aperture mask has the wrong shape")
        before = wave.probability
        wave = replace(wave, amplitude=xp.where(mask, wave.amplitude, 0j))
        rows.append({"component": a.key, "z_mm": z_mm,
            "input_weight": prior*before, "output_weight": prior*wave.probability,
            "lost_weight": prior*(before-wave.probability)})
    return wave, rows


def _maximum_mode_pixels(checkpoint):
    modes = checkpoint.beam.modes
    if hasattr(modes, "rows"):
        # This is only a resource estimate. The normal mode reader still
        # verifies checksums and validates every array before execution.
        return max((math.prod(row["arrays"]["amplitude"]["shape"]) for row in modes.rows), default=0)
    return max((mode.plane.amplitude.size for mode in modes), default=0)


def _propagate_column(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                      grid_numerics=WaveGridNumerics(), retained_bytes=0,
                      tip_time_s=None, _prepared=None, _combine_linear=True,
                      cancelled=lambda: False, progress_callback=None):
    """Execute a segment with resident device fields and host checkpoints."""
    from temsim.physics.compute_backend import gpu_failure_category, gpu_retry_reason
    prepared = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                if _prepared is None else _prepared)
    largest = _maximum_mode_pixels(checkpoint)
    arguments = dict(maximum_step_mm=maximum_step_mm, grid_numerics=grid_numerics,
        retained_bytes=retained_bytes, tip_time_s=tip_time_s, _prepared=prepared,
        _combine_linear=_combine_linear, cancelled=cancelled, progress_callback=progress_callback)
    try:
        with device_scope(grid_numerics.compute_backend,
                acceleration_enabled=grid_numerics.acceleration_enabled,
                maximum_working_bytes=grid_numerics.maximum_device_working_bytes,
                work_items=largest*len(prepared[0].step_m), required_bytes=256*largest) as device:
            return _propagate_column_impl(state, checkpoint, stop_z_mm, _device=device, **arguments)
    except Exception as error:
        # Physics, grid and cancellation failures never become CPU successes.
        if gpu_failure_category(error) is None:
            raise
        reason = gpu_retry_reason(error, normalise_backend(grid_numerics.compute_backend),
                                  stage="coherent_column")
        error.__traceback__ = None
        if cancelled():
            raise InterruptedError("Column wave propagation cancelled") from None
    if progress_callback:
        progress_callback(0, len(prepared[0].step_m), "GPU memory unavailable; retrying the unchanged column segment on CPU")
    with device_scope("CPU") as device:
        device.fallback_reason = reason
        return _propagate_column_impl(state, checkpoint, stop_z_mm, _device=device, **arguments)


def _propagate_column_impl(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                      grid_numerics=WaveGridNumerics(), retained_bytes=0,
                      tip_time_s=None, _prepared=None, _combine_linear=True, _device,
                      cancelled=lambda: False, progress_callback=None):
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    plan, radii, stops, owners = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                                 if _prepared is None else _prepared)
    outputs, records, maps = [], [], {}
    electric = _electric_coefficients(plan)
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    retained_bytes += resident_wave_bytes(checkpoint.beam)
    map_bytes = 512*len(plan.step_m)
    retained_bytes += map_bytes
    for mode in checkpoint.beam.modes:
        grid_numerics.check(mode.plane.amplitude.shape, retained_bytes=retained_bytes)
    for index, mode in enumerate(checkpoint.beam.modes):
        if cancelled():
            raise InterruptedError("Column wave propagation cancelled")
        from temsim.physics.tip_gun_wave import _momentum_velocity
        entrance_p, entrance_v = _momentum_velocity(mode.energy_kev*1000)
        nodes, phi, _, _ = electric
        energies = mode.energy_kev*1000+phi-nodes[0]
        momentum, velocity = _momentum_velocity(energies)
        exit_energy = mode.energy_kev*1000+nodes[-1]-nodes[0]
        exit_p, _ = _momentum_velocity(exit_energy)
        elapsed = np.r_[0., np.cumsum(plan.step_m/velocity)]
        flight_increment = float(elapsed[-1])
        action_increment = float(np.sum(plan.step_m*momentum))
        dynamic_actions = any(row.get("dynamic", False) for row in owners)
        mode_owners, kick_x, kick_y = owners, plan.kick_x_rad, plan.kick_y_rad
        if dynamic_actions:
            if mode.axial_reference is None:
                raise ValueError("Dynamic coil transport requires the executed flight time from the tip")
            epoch = float(getattr(state, "simulation_time_s", 0.) if tip_time_s is None else tip_time_s)
            if not math.isfinite(epoch):
                raise ValueError("Tip emission clock must be finite")
            events, mode_owners = _component_events(state, checkpoint.plane_z_mm, stop_z_mm,
                arrival_time=lambda z: epoch+mode.axial_reference.flight_time_s+
                (float(np.interp(z, plan.z_mm, elapsed)) if plan.z_mm[0] <= z <= plan.z_mm[-1]
                 else (z-checkpoint.plane_z_mm)*1e-3/entrance_v))
            kick_x, kick_y = np.zeros(len(plan.z_mm)), np.zeros(len(plan.z_mm))
            for (z, x, y), owner in zip(events, mode_owners):
                if owner.get("finite_field", False):
                    continue
                node = int(np.argmin(abs(plan.z_mm-z)))
                if abs(plan.z_mm[node]-z) > 1e-9:
                    raise ValueError("Dynamic coil is missing from the physical integration grid")
                kick_x[node] += x; kick_y[node] += y
        if mode.energy_kev not in maps or dynamic_actions:
            maps.clear()  # only one energy's current segment, never all past paths
            maps[mode.energy_kev] = tuple(_column_transports(plan, mode.energy_kev, electric=electric,
                dipoles=_mode_dipoles(plan, owners, mode_owners, state) if dynamic_actions else None))
        wave, losses, refinements, linear_runs = mode.plane, [], [], []
        if _device.backend == "cupy":
            wave = replace(wave, amplitude=_device.xp.asarray(wave.amplitude))
        wavelength = float(wavelength_m(mode.energy_kev*1000))
        def multipole(wave, z, **strengths):
            if not any(strengths.values()):
                return wave
            try:
                result, rows = apply_resolved_operator(wave,
                    lambda value: apply_multipole_phase(value, wavelength, **strengths),
                    numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
            except ValueError as error:
                raise ValueError(f"Column z={z:.9g} mm, mode {mode.mode_id}, strengths={strengths}: {error}") from error
            for row in rows:
                row.update(z_mm=float(z), strengths=strengths)
            refinements.extend(rows)
            if rows and progress_callback:
                progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m),
                                  f"Column wave refined {rows[0]['from_shape']} -> {rows[-1]['to_shape']} at {z:.9g} mm")
            return result
        def linear(wave, path, z, *, input_z, operator="distributed_linear", selected_gauge=None):
            output_rows = []
            try:
                if selected_gauge is None:
                    wave, gauge_rows = reselect_phase_carrier(wave, wavelength,
                        grid_numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
                else:
                    wave, gauge_rows = selected_gauge
                result, rows = apply_resolved_operator(wave,
                    lambda value: propagate_plane_wave(value, path.matrix, path.offset,
                        wavelength, **path.phase_kwargs(), grid_numerics=grid_numerics,
                        retained_bytes=retained_bytes, cancelled=cancelled,
                        sampling_records=output_rows),
                    numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
            except ValueError as error:
                raise ValueError(f"Column linear z={z:.9g} mm, mode {mode.mode_id}: {error}") from error
            for row in gauge_rows:
                row.update(z_mm=float(input_z), input_z_mm=float(input_z), output_z_mm=float(z),
                           mode_id=mode.mode_id, operator="phase_carrier_representation")
            rows = gauge_rows+rows+output_rows
            for row in rows[len(gauge_rows):]:
                row.update(z_mm=float(z), input_z_mm=float(input_z), output_z_mm=float(z),
                           mode_id=mode.mode_id, operator=operator)
            refinements.extend(rows)
            if rows and progress_callback:
                changed = rows[0]['from_shape'] != rows[-1]['to_shape']
                message = (f"Column wave grid {rows[0]['from_shape']} -> {rows[-1]['to_shape']}" if changed
                           else "Column wave phase representation checked")
                progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m),
                    f"{message} at {z:.9g} mm")
            return result
        i = 0
        paths = maps[mode.energy_kev]
        while i < len(paths):
            dz, path = plan.step_m[i], paths[i]
            if i % 64 == 0:
                if cancelled():
                    raise InterruptedError("Column wave propagation cancelled")
                if progress_callback:
                    progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m), "Distributed column wave")
            if wave.probability == 0:
                break
            multipole_scale = float(getattr(plan, "reference_momentum_kg_m_s", entrance_p))/entrance_p
            hn, hs = plan.midpoint_hex_normal_m3[i]*dz*multipole_scale/2, plan.midpoint_hex_skew_m3[i]*dz*multipole_scale/2
            wave = multipole(wave, plan.z_mm[i], normal_m2=hn, skew_m2=hs)
            selected = None
            if _combine_linear and not hn and not hs:
                selected = reselect_phase_carrier(wave, wavelength,
                    grid_numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
                last, combined = _linear_run(selected[0], wavelength, paths, i, plan, radii, stops, kick_x, kick_y)
                if last > i+1:
                    wave = linear(wave, combined, plan.z_mm[last], input_z=plan.z_mm[i], selected_gauge=selected)
                    linear_runs.append({"first_step": i, "last_step_exclusive": last})
                    i = last
                    continue
            wave = linear(wave, path, plan.z_mm[i+1], input_z=plan.z_mm[i], selected_gauge=selected)
            wave = multipole(wave, plan.z_mm[i+1], normal_m2=hn, skew_m2=hs)
            j = i+1
            if kick_x[j] or kick_y[j]:
                kick = CanonicalPath(1e-3)
                local_p, _ = _momentum_velocity(mode.energy_kev*1000+nodes[j]-nodes[0])
                kick.append(np.eye(4), np.array((0., 0., kick_x[j], kick_y[j]))*local_p/entrance_p)
                wave = linear(wave, kick, plan.z_mm[j], input_z=plan.z_mm[j], operator="physical_impulse")
            local_p, _ = _momentum_velocity(mode.energy_kev*1000+nodes[j]-nodes[0])
            wave = multipole(wave, plan.z_mm[j], spherical_m3=plan.cs_kick_m3[j]*local_p/entrance_p)
            wave, rows = _clip(wave, radii[j], stops.get(j, ()), float(plan.z_mm[j]), mode.weight_per_reference_electron)
            losses.extend(rows)
            i += 1
        norm = wave.probability
        reference = None if mode.axial_reference is None else mode.axial_reference.advance(flight_increment, action_increment)
        ratio = float(entrance_p/exit_p)
        amplitude = wave.amplitude/math.sqrt(norm) if norm else wave.amplitude
        output = replace(mode, plane=replace(wave, amplitude=to_host(amplitude),
                         curvature_m1=None if wave.curvature_m1 is None else wave.curvature_m1*ratio,
                         tilt_rad=None if wave.tilt_rad is None else wave.tilt_rad*ratio),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         energy_kev=float(exit_energy)*1e-3,
                         axial_reference=reference)
        outputs.append(output)
        retained_bytes += output.plane.amplitude.nbytes
        records.append({"mode_id": mode.mode_id, "energy_kev": mode.energy_kev, **_device.evidence(),
                        "input_weight": mode.weight_per_reference_electron,
                        "output_energy_kev": output.energy_kev,
                        "reference_flight_time_increment_s": flight_increment,
                        "reference_longitudinal_action_increment_j_s": action_increment,
                        "axial_reference": None if reference is None else asdict(reference),
                        "deflector_actions": mode_owners,
                        "output_weight": output.weight_per_reference_electron, "losses": losses,
                        "grid_refinements": refinements, "quadratic_runs": linear_runs, "output_shape": wave.amplitude.shape})
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), stop_z_mm,
        checkpoint.reference_current_a, {"schema": "executed-column-wave-v1", "upstream_digest": checkpoint.digest,
        "upstream": checkpoint.record, "field_plan_signature": plan.signature, "step_count": len(plan.step_m),
        "maximum_step_mm": maximum_step_mm, "deflector_actions": owners, "modes": records,
        "grid_numerics": grid_numerics.column_identity(),
        "integrator": "second-order midpoint affine quadratic Hamiltonian with variable longitudinal momentum / symmetric cubic split / discrete Cs",
        "coordinate_basis": "laboratory canonical normalized to each mode entrance momentum; output phase carriers converted to exit momentum; continuous symmetric axial-field gauge",
        "electric_model": "captured static scalar field, near-axis second-order transverse expansion; axis energy, axial action and flight time carried",
        "dipole_model": "shared finite field supports; driven coils frozen at per-mode axial arrival time, no longitudinal pulse envelope",
        "time_model": "actual coil laws at per-mode axial arrival times; frozen transverse slices, no longitudinal pulse wavepacket",
        "validation_status": "DEVELOPMENT"})


def _slice_prepared(prepared, first, last):
    """Slice existing integration nodes; no new steps or repeated entrance kick."""
    from types import SimpleNamespace
    from temsim.immutable_json import json_digest
    plan, radii, stops, owners = prepared
    values = {name: getattr(plan, name)[first:last] for name in (
        "step_m", "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2",
        "midpoint_sxy_m2",
        "midpoint_hex_normal_m3", "midpoint_hex_skew_m3")}
    values.update({name: getattr(plan, name)[first:last+1] for name in ("z_mm", "kick_x_rad", "kick_y_rad", "cs_kick_m3")})
    values.update({name: getattr(plan, name)[3*first:3*last] for name in ("dipole_bx_t", "dipole_by_t")
                   if hasattr(plan, name)})
    for name in ("electric_field", "electric_field_identity", "reference_momentum_kg_m_s"):
        if hasattr(plan, name):
            values[name] = getattr(plan, name)
    values["signature"] = json_digest((plan.signature, first, last))
    return (SimpleNamespace(**values), radii[first:last+1],
            {i-first: value for i, value in stops.items() if first < i <= last},
            [r for r in owners if (
                r["z_mm"]+.5*r.get("effective_thickness_mm", 0.) > plan.z_mm[first]
                and r["z_mm"]-.5*r.get("effective_thickness_mm", 0.) < plan.z_mm[last])
             if r.get("finite_field", False)] +
            [r for r in owners if not r.get("finite_field", False)
             and plan.z_mm[first] < r["z_mm"] <= plan.z_mm[last]])


def _column_mode_workers(checkpoint, requested=None):
    """Capture the owning numerical job's budget before starting threads."""
    return min(8, len(checkpoint.beam.modes), numerical_thread_budget(requested))


def _column_shared_bytes(checkpoint, prepared):
    """Resident inputs shared by workers, without loading stored amplitudes."""
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    resident = resident_wave_bytes(checkpoint.beam)
    if not hasattr(checkpoint.beam, "content_identity"):
        resident += sum(value.nbytes for value in checkpoint.auxiliary_arrays.values())
    plan = prepared[0]
    values = ((getattr(plan, item.name) for item in dataclass_fields(plan))
              if is_dataclass(plan) else vars(plan).values())
    arrays = [value for value in values if isinstance(value, np.ndarray)]
    arrays.append(prepared[1])
    return resident+sum(value.nbytes for value in arrays)


def _memory_budget_cause(error):
    """Follow explicit contextual wrappers; never infer from error strings."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, WaveMemoryBudgetError):
            return error
        seen.add(id(error))
        error = error.__cause__
    return None


def _write_column_modes(state, checkpoint, stop_z_mm, writer, *, prepared, maximum_step_mm,
                        grid_numerics, workers, shared_bytes, tip_time_s, cancelled,
                        progress_callback, parent_digest):
    """Bounded slots include in-flight inputs, scratch and pending outputs.

    Each batch has at most one mode per slot. Finished modes retain their slot
    until the main thread writes them in source order; no additional work is
    submitted while those outputs are pending. The 256-byte-per-pixel working
    estimate therefore covers both active scratch and a completed slot's
    immutable output. Shared inputs and plan arrays are reserved separately.
    """
    stop = Event()
    progress_lock = Lock()
    remaining = grid_numerics.maximum_working_bytes-shared_bytes
    if remaining < workers:
        raise WaveMemoryBudgetError("Resident column inputs and plan leave no working-memory partition")
    worker_bytes = remaining//workers
    local_numerics = replace(grid_numerics, maximum_working_bytes=worker_bytes)
    records = []

    def report(*args):
        if progress_callback is not None:
            with progress_lock:
                progress_callback(*args)

    def execute(index, *, threaded=False):
        if threaded:
            # BLAS is already serial in the owning numerical job. Numba's
            # thread-local mask must also be one in each mode worker.
            initialize_numerical_thread(1)
        if cancelled() or stop.is_set():
            raise InterruptedError("Column mode propagation cancelled")
        mode = checkpoint.beam.modes[index]
        local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
            checkpoint.reference_current_a, {"executed_parent": parent_digest})
        return _propagate_column(state, local, stop_z_mm, maximum_step_mm=maximum_step_mm,
            grid_numerics=local_numerics, tip_time_s=tip_time_s, _prepared=prepared,
            cancelled=lambda: cancelled() or stop.is_set(), progress_callback=report)

    def append(transported):
        if cancelled():
            raise InterruptedError("Column modes cancelled before writing checkpoint")
        writer.append(transported.beam.modes[0])
        records.extend(transported.record["modes"])

    if workers == 1:
        for index in range(len(checkpoint.beam.modes)):
            transported = execute(index)
            append(transported)
            del transported
        return records, worker_bytes

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="temsim-wave-mode")
    futures = []
    errors = []
    try:
        for first in range(0, len(checkpoint.beam.modes), workers):
            futures = [pool.submit(copy_context().run, execute, index, threaded=True)
                       for index in range(first, min(first+workers, len(checkpoint.beam.modes)))]
            errors = []
            positions = {future: index for index, future in enumerate(futures)}
            completed, next_write = set(), 0
            for future in as_completed(futures):
                try:
                    future.result()
                except CancelledError:
                    pass
                except BaseException as error:
                    errors.append(error)
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                else:
                    completed.add(positions[future])
                    while not errors and next_write in completed:
                        transported = futures[next_write].result()
                        append(transported)
                        del transported
                        next_write += 1
            if cancelled():
                raise InterruptedError("Column mode propagation cancelled")
            if errors:
                # A genuine physical/grid error must not be hidden by a
                # concurrent partition failure or its cooperative cancellation.
                genuine = [error for error in errors
                           if _memory_budget_cause(error) is None and not isinstance(error, InterruptedError)]
                if genuine:
                    raise genuine[0]
                memory = next((_memory_budget_cause(error) for error in errors
                               if _memory_budget_cause(error) is not None), None)
                if memory is not None:
                    message = str(memory)
                    # Release failed-worker frames before the serial retry;
                    # tracebacks otherwise retain their large intermediate grids.
                    for error in errors:
                        while error is not None:
                            error.__traceback__ = None
                            error = error.__cause__
                    raise WaveMemoryBudgetError(message) from None
                raise errors[0]
            positions.clear()
            futures.clear()
            del future
        return records, worker_bytes
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        futures.clear()
        errors.clear()


def _propagate_column_segmented(state, checkpoint, stop_z_mm, *, store, segment_steps,
                               maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=None,
                               cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True,
                               checkpoint_callback=None, _prepared=None):
    from temsim.immutable_json import json_digest
    prepared = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                if _prepared is None else _prepared)
    plan = prepared[0]
    if plan.z_mm[0] != checkpoint.plane_z_mm or plan.z_mm[-1] != stop_z_mm:
        raise ValueError("Prepared column plan must match the exact executed interval")
    result, hit = checkpoint, False
    for first in range(0, len(plan.step_m), segment_steps):
        if cancelled():
            raise InterruptedError("Segmented column propagation cancelled")
        last = min(first+segment_steps, len(plan.step_m))
        segment = _slice_prepared(prepared, first, last)
        time_key = tip_time_s if any(row["dynamic"] for row in segment[3]) else None
        parent_digest = result.digest
        key = store.key("column-segment", parent_digest, segment[0].signature, grid_numerics.column_identity(), time_key)
        cached = store.get(key) if use_cache else None
        if cached is not None:
            result, hit = cached, True
            if checkpoint_callback is not None:
                checkpoint_callback(result)
            continue
        shared_bytes = _column_shared_bytes(result, prepared)
        workers = _column_mode_workers(result)
        # The device pool belongs to one wave mode at a time. CPU mode workers
        # must not each independently claim the complete GPU memory budget.
        from temsim.physics.compute_backend import choose_wave_backend, WAVE_BACKEND_CUPY
        largest = _maximum_mode_pixels(result)
        selected, _ = choose_wave_backend(normalise_backend(grid_numerics.compute_backend),
            acceleration_enabled=grid_numerics.acceleration_enabled, work_items=largest*len(segment[0].step_m))
        if selected == WAVE_BACKEND_CUPY:
            workers = 1
        retry_serial = False
        while True:
            writer = store.writer(key, result.beam.reference_plane)
            try:
                records, worker_bytes = _write_column_modes(state, result, float(plan.z_mm[last]), writer,
                    prepared=segment, maximum_step_mm=maximum_step_mm, grid_numerics=grid_numerics,
                    workers=workers, shared_bytes=shared_bytes, tip_time_s=tip_time_s,
                    cancelled=cancelled, progress_callback=progress_callback, parent_digest=parent_digest)
                verify()
                if cancelled():
                    raise InterruptedError("Segment cancelled before checkpoint commit")
                result = writer.finish(float(plan.z_mm[last]), result.reference_current_a,
                    {"schema": "executed-column-segment-v1", "upstream_digest": parent_digest,
                     "upstream": result.record, "parent_plan_signature": plan.signature,
                     "node_span": (first, last), "modes": records,
                     "mode_execution": {"workers": workers, "retry_serial_for_memory": retry_serial,
                         "maximum_working_bytes": grid_numerics.maximum_working_bytes,
                         "shared_retained_bytes": shared_bytes, "worker_working_bytes": worker_bytes},
                     "memory_policy": "bounded mode slots; shared inputs reserved; ordered streaming commit"})
                break
            except WaveMemoryBudgetError as error:
                if workers == 1 or cancelled():
                    raise
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
                # The pool has joined and no partial checkpoint is published.
                # Retry the same modes/operators with the full remaining budget.
                workers, retry_serial = 1, True
                if progress_callback:
                    progress_callback(first, len(plan.step_m),
                        "Column mode memory partition too small; continuing serially")
            finally:
                writer.abort()
        if checkpoint_callback is not None:
            checkpoint_callback(result)
        if progress_callback:
            progress_callback(last, len(plan.step_m), "Column segment saved; previous wave buffers released")
    return result, hit
