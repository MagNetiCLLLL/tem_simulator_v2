"""Distributed stationary scalar column operator acting on an executed beam.

Internal stage function, not a configurable source. Public tip-to-detector
requests execute the gun first. Every quadratic field, nonlinear multipole,
physical kick, aperture and sampled vacuum bore is owned by the shared plan.
"""
from dataclasses import asdict, replace
import math

import numpy as np
from scipy.linalg import expm

from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.core import build_propagation_plan, electron
from temsim.physics.column_wall import _vacuum_segments, _expanded_profile_axis
from temsim.physics.multiplane_wave import propagate_plane_wave
from temsim.physics.multipole_wave import apply_multipole_phase
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics, apply_resolved_operator


def _component_events(state, start, stop, *, arrival_time=None):
    events, owners, seen = [], [], set()
    for component in (*state.deflectors, *getattr(state, "corrector_elements", ())):
        if not getattr(component, "enabled", False):
            continue
        if component.key in seen:
            raise ValueError(f"Duplicate column component {component.key}")
        seen.add(component.key)
        if not hasattr(component, "kick_events"):
            continue
        if (getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False)) and hasattr(component, "validate"):
            component.validate()
        try:
            rows = component.kick_events(time_s=float(getattr(state, "simulation_time_s", 0.)))
        except TypeError:
            rows = component.kick_events()
        for row_index, (z, x, y) in enumerate(rows):
            if start < z <= stop:
                dynamic = bool(getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False))
                time = None
                if dynamic and arrival_time is not None:
                    time = float(arrival_time(z))
                    actual = component.kick_events(time_s=time)
                    if len(actual) != len(rows) or actual[row_index][0] != z:
                        raise ValueError("Dynamic coil geometry changed with time")
                    _, x, y = actual[row_index]
                events.append((z, x, y))
                owners.append({"component": component.key, "z_mm": z, "kick_rad": (x, y),
                               "dynamic": dynamic, "arrival_time_s": time})
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
        raise ValueError("The installed nanopulser needs time-energy wavepacket transport; it cannot be omitted")
    apertures = [a for a in state.apertures if all(bool(getattr(a, n, True))
        for n in ("installed", "enabled", "inserted")) and start < a.z_mm <= stop]
    if len({a.key for a in apertures}) != len(apertures):
        raise ValueError("Duplicate physical column aperture")
    walls = _vacuum_segments(state, np.array((start, stop)))
    boundaries = [z for w in walls for z in (w.start_z_mm, w.end_z_mm) if start < z < stop]
    events, owners = _component_events(state, start, stop)
    plan = build_propagation_plan(state, start, stop, events,
        save_z_mm=[a.z_mm for a in apertures]+boundaries, maximum_step_mm=maximum_step_mm)
    if plan.mapped_fields:
        raise ValueError("Installed 3-D field maps need their non-polynomial wave Hamiltonian; no linear substitute is allowed")
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


def _linear_factor(path, generator, distance, depth=0):
    """Lift a known constant Hamiltonian; bisect without changing its map."""
    try:
        path.append(expm(generator*distance))
    except ValueError as error:
        if "Canonical phase path is undersampled" not in str(error) or depth >= 24:
            raise
        _linear_factor(path, generator, distance/2, depth+1)
        _linear_factor(path, generator, distance/2, depth+1)


def _column_transports(plan, energy_kev):
    from types import SimpleNamespace
    q, momentum, _ = electron(SimpleNamespace(beam_voltage_kv=energy_kev))
    for i, dz in enumerate(plan.step_m):
        g = q*plan.midpoint_magnetic_t[i]/(2*momentum)
        rotation = np.array(((0., g), (-g, 0.)))
        stiffness = np.diag((plan.midpoint_sx_m2[i]+g*g, plan.midpoint_sy_m2[i]+g*g))
        generator = np.block([[rotation, np.eye(2)], [-stiffness, rotation]])
        path = CanonicalPath(1e-3)
        _linear_factor(path, generator, float(dz))
        yield path


def _clip(wave, radius_mm, apertures, z_mm, prior):
    xy = wave.coordinates_m()*1e3
    rows = []
    if math.isfinite(radius_mm):
        before = wave.probability
        mask = np.hypot(xy[0], xy[1]) < radius_mm
        if not np.all(mask):
            wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
        if wave.probability < before:
            rows.append({"component": "column_wall", "z_mm": z_mm,
                         "lost_weight": prior*(before-wave.probability)})
    for a in apertures:
        if float(getattr(a, "radius_mm", 1.)) <= 0:
            mask = np.zeros_like(xy[0], dtype=bool)
        elif hasattr(a, "transmission_mask"):
            mask = np.asarray(a.transmission_mask(xy[0], xy[1]), dtype=bool)
        else:
            mask = np.hypot(xy[0]-a.offset_x_mm, xy[1]-a.offset_y_mm) <= a.radius_mm
        if mask.shape != wave.amplitude.shape:
            raise ValueError("Column aperture mask has the wrong shape")
        before = wave.probability
        wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
        rows.append({"component": a.key, "z_mm": z_mm,
            "input_weight": prior*before, "output_weight": prior*wave.probability,
            "lost_weight": prior*(before-wave.probability)})
    return wave, rows


def _propagate_column(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                      grid_numerics=WaveGridNumerics(), retained_bytes=0,
                      tip_time_s=None, _prepared=None,
                      cancelled=lambda: False, progress_callback=None):
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.simulation_modes import is_ideal
    plan, radii, stops, owners = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                                 if _prepared is None else _prepared)
    outputs, records, maps = [], [], {}
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    retained_bytes += resident_wave_bytes(checkpoint.beam)
    map_bytes = 512*len(plan.step_m)
    retained_bytes += map_bytes
    for mode in checkpoint.beam.modes:
        grid_numerics.check(mode.plane.amplitude.shape, retained_bytes=retained_bytes)
    for index, mode in enumerate(checkpoint.beam.modes):
        if cancelled():
            raise InterruptedError("Column wave propagation cancelled")
        optical_energy = state.beam_voltage_kv if is_ideal(state) else mode.energy_kev
        if optical_energy not in maps:
            maps.clear()  # only one energy's current segment, never all past paths
            maps[optical_energy] = tuple(_column_transports(plan, optical_energy))
        from temsim.physics.tip_gun_wave import _momentum_velocity
        momentum, velocity = _momentum_velocity(mode.energy_kev*1000)
        dynamic_actions = any(row.get("dynamic", False) for row in owners)
        mode_owners, kick_x, kick_y = owners, plan.kick_x_rad, plan.kick_y_rad
        if dynamic_actions:
            if mode.axial_reference is None:
                raise ValueError("Dynamic coil transport requires the executed flight time from the tip")
            epoch = float(getattr(state, "simulation_time_s", 0.) if tip_time_s is None else tip_time_s)
            if not math.isfinite(epoch):
                raise ValueError("Tip emission clock must be finite")
            events, mode_owners = _component_events(state, checkpoint.plane_z_mm, stop_z_mm,
                arrival_time=lambda z: epoch+mode.axial_reference.flight_time_s+(z-checkpoint.plane_z_mm)*1e-3/velocity)
            kick_x, kick_y = np.zeros(len(plan.z_mm)), np.zeros(len(plan.z_mm))
            for z, x, y in events:
                node = int(np.argmin(abs(plan.z_mm-z)))
                if abs(plan.z_mm[node]-z) > 1e-9:
                    raise ValueError("Dynamic coil is missing from the physical integration grid")
                kick_x[node] += x; kick_y[node] += y
        wave, losses, refinements = mode.plane, [], []
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
        for i, (dz, path) in enumerate(zip(plan.step_m, maps[optical_energy])):
            if i % 64 == 0:
                if cancelled():
                    raise InterruptedError("Column wave propagation cancelled")
                if progress_callback:
                    progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m), "Distributed column wave")
            if wave.probability == 0:
                break
            hn, hs = plan.midpoint_hex_normal_m3[i]*dz/2, plan.midpoint_hex_skew_m3[i]*dz/2
            wave = multipole(wave, plan.z_mm[i], normal_m2=hn, skew_m2=hs)
            wave = propagate_plane_wave(wave, path.matrix, path.offset, wavelength, **path.phase_kwargs())
            wave = multipole(wave, plan.z_mm[i+1], normal_m2=hn, skew_m2=hs)
            j = i+1
            if kick_x[j] or kick_y[j]:
                kick = CanonicalPath(1e-3)
                kick.append(np.eye(4), np.array((0., 0., kick_x[j], kick_y[j])))
                wave = propagate_plane_wave(wave, kick.matrix, kick.offset, wavelength, **kick.phase_kwargs())
            wave = multipole(wave, plan.z_mm[j], spherical_m3=plan.cs_kick_m3[j])
            wave, rows = _clip(wave, radii[j], stops.get(j, ()), float(plan.z_mm[j]), mode.weight_per_reference_electron)
            losses.extend(rows)
        norm = wave.probability
        length = (stop_z_mm-checkpoint.plane_z_mm)*1e-3
        reference = None if mode.axial_reference is None else mode.axial_reference.advance(float(length/velocity), float(length*momentum))
        output = replace(mode, plane=replace(wave, amplitude=wave.amplitude/math.sqrt(norm) if norm else wave.amplitude),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         axial_reference=reference)
        outputs.append(output)
        retained_bytes += output.plane.amplitude.nbytes
        records.append({"mode_id": mode.mode_id, "energy_kev": mode.energy_kev,
                        "input_weight": mode.weight_per_reference_electron,
                        "reference_flight_time_increment_s": float(length/velocity),
                        "reference_longitudinal_action_increment_j_s": float(length*momentum),
                        "axial_reference": None if reference is None else asdict(reference),
                        "deflector_actions": mode_owners,
                        "output_weight": output.weight_per_reference_electron, "losses": losses,
                        "grid_refinements": refinements, "output_shape": wave.amplitude.shape})
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), stop_z_mm,
        checkpoint.reference_current_a, {"schema": "executed-column-wave-v1", "upstream_digest": checkpoint.digest,
        "upstream": checkpoint.record, "field_plan_signature": plan.signature, "step_count": len(plan.step_m),
        "maximum_step_mm": maximum_step_mm, "deflector_actions": owners, "modes": records,
        "grid_numerics": asdict(grid_numerics),
        "integrator": "second-order midpoint quadratic Hamiltonian / symmetric cubic split / discrete Cs",
        "coordinate_basis": "laboratory normalized canonical; continuous symmetric axial-field gauge",
        "time_model": "actual coil laws at per-mode axial arrival times; frozen transverse slices, no longitudinal pulse wavepacket",
        "validation_status": "DEVELOPMENT"})


def _slice_prepared(prepared, first, last):
    """Slice existing integration nodes; no new steps or repeated entrance kick."""
    from types import SimpleNamespace
    from temsim.immutable_json import json_digest
    plan, radii, stops, owners = prepared
    values = {name: getattr(plan, name)[first:last] for name in (
        "step_m", "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2",
        "midpoint_hex_normal_m3", "midpoint_hex_skew_m3")}
    values.update({name: getattr(plan, name)[first:last+1] for name in ("z_mm", "kick_x_rad", "kick_y_rad", "cs_kick_m3")})
    values["signature"] = json_digest((plan.signature, first, last))
    return (SimpleNamespace(**values), radii[first:last+1],
            {i-first: value for i, value in stops.items() if first < i <= last},
            [r for r in owners if plan.z_mm[first] < r["z_mm"] <= plan.z_mm[last]])


def _propagate_column_segmented(state, checkpoint, stop_z_mm, *, store, segment_steps,
                               maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=None,
                               cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True):
    from temsim.immutable_json import json_digest
    prepared = _prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
    plan = prepared[0]
    result, hit = checkpoint, False
    for first in range(0, len(plan.step_m), segment_steps):
        if cancelled():
            raise InterruptedError("Segmented column propagation cancelled")
        last = min(first+segment_steps, len(plan.step_m))
        segment = _slice_prepared(prepared, first, last)
        time_key = tip_time_s if any(row["dynamic"] for row in segment[3]) else None
        key = store.key("column-segment", result.digest, segment[0].signature, asdict(grid_numerics), time_key)
        cached = store.get(key) if use_cache else None
        if cached is not None:
            result, hit = cached, True
            continue
        writer = store.writer(key, result.beam.reference_plane)
        records = []
        try:
            for mode in result.beam.modes:
                local = TipGunCheckpoint(BeamState((mode,), result.beam.reference_plane), result.plane_z_mm,
                    result.reference_current_a, {"executed_parent": result.digest})
                transported = _propagate_column(state, local, float(plan.z_mm[last]), maximum_step_mm=maximum_step_mm,
                    grid_numerics=grid_numerics, tip_time_s=tip_time_s, _prepared=segment,
                    cancelled=cancelled, progress_callback=progress_callback)
                writer.append(transported.beam.modes[0])
                records.extend(transported.record["modes"])
                del mode, local, transported
            verify()
            if cancelled():
                raise InterruptedError("Segment cancelled before checkpoint commit")
            result = writer.finish(float(plan.z_mm[last]), result.reference_current_a,
                {"schema": "executed-column-segment-v1", "upstream_digest": result.digest,
                 "upstream": result.record, "parent_plan_signature": plan.signature,
                 "node_span": (first, last), "modes": records,
                 "memory_policy": "stream one mode; previous segment arrays released after commit"})
        finally:
            writer.abort()
        if progress_callback:
            progress_callback(last, len(plan.step_m), "Column segment saved; previous wave buffers released")
    return result, hit
