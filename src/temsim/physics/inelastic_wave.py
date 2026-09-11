"""Conditional wave trajectories of the existing material transport model.

This is an explicit local, Markov, random momentum-transfer instrument, not
an atomic transition-potential or energy-differential dielectric calculation.
Each slice uses all existing Poisson channels, including its plural-event
approximation. Conditioned waves continue through the remaining specimen and
column at their new energy. Independent histories never interfere.
"""
from dataclasses import asdict, replace
from copy import copy
import math
from types import SimpleNamespace

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.multiplane_wave import propagate_plane_wave
from temsim.physics.tip_gun_wave import _momentum_velocity, TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_execution import InelasticWaveNumerics
from temsim.optics.electron_gun.tip_coherence import wavelength_m


def _trajectory_rng(seed, *identity):
    # Independent deterministic counters, unaffected by cached/skipped slices.
    digest = bytes.fromhex(json_digest((seed, identity)))
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint32))


def _collide_slice(mode, inside, distribution, rng, z_mm):
    inside = np.asarray(inside, bool)
    if inside.shape != mode.plane.amplitude.shape:
        raise ValueError("Inelastic material occupancy must match the wave")
    density = abs(mode.plane.amplitude)**2
    norm = float(density.sum())
    if norm == 0 or mode.weight_per_reference_electron == 0:
        return mode, {"kind": "already_absorbed", "absorbed_weight": 0.}
    material = float(density[inside].sum()/norm)
    channels = distribution.channels
    probabilities = [(1-material)+material*c.probability if c.key == "real_zero_loss"
                     else material*c.probability for c in channels]
    probabilities.append(material*distribution.absorbed_probability)
    if not math.isclose(sum(probabilities), 1., rel_tol=0, abs_tol=2e-12):
        raise ValueError("Inelastic Kraus outcomes do not preserve trace")
    u = float(rng.random())
    chosen = min(int(np.searchsorted(np.cumsum(probabilities), u, side="right")), len(probabilities)-1)
    event = {"z_mm": float(z_mm), "input_energy_kev": mode.energy_kev,
             "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference),
             "outcome_probability": probabilities[chosen], "random_draw": u,
             "conditional_phase_reference": "within this environmental history only; inter-history phase undefined"}
    if chosen == len(channels):
        event.update(kind="effective_absorption", absorbed_weight=mode.weight_per_reference_electron)
        return replace(mode, weight_per_reference_electron=0.,
                       plane=replace(mode.plane, amplitude=np.zeros_like(mode.plane.amplitude)),
                       scattering_history=(*mode.scattering_history, event)), event
    channel = channels[chosen]
    envelope = np.where(inside, math.sqrt(channel.probability), 1. if channel.key == "real_zero_loss" else 0.)
    amplitude = mode.plane.amplitude*envelope
    conditioned_norm = float(np.sum(abs(amplitude)**2))
    if conditioned_norm <= 0:
        raise ValueError("A zero-probability inelastic outcome was selected")
    wave = replace(mode.plane, amplitude=amplitude/math.sqrt(conditioned_norm))
    event.update(kind=channel.key, energy_loss_ev=channel.energy_loss_ev, mean_events=channel.mean_events,
                 approximation=channel.approximation, absorbed_weight=0.)
    if channel.key == "real_zero_loss":
        return replace(mode, plane=wave), event
    new_energy = mode.energy_kev-channel.energy_loss_ev*1e-3
    if new_energy <= 0:
        raise ValueError("Material representative loss exceeds electron energy; low-energy transport model required")
    old_p, _ = _momentum_velocity(mode.energy_kev*1000)
    new_p, _ = _momentum_velocity(new_energy*1000)
    ratio = float(old_p/new_p)
    # Same accumulated phase before the collision, now expressed with p_out.
    wave = replace(wave, curvature_m1=None if wave.curvature_m1 is None else wave.curvature_m1*ratio,
                   tilt_rad=None if wave.tilt_rad is None else wave.tilt_rad*ratio)
    azimuth = float(rng.uniform(0, 2*np.pi))
    angle = channel.characteristic_angle_mrad*1e-3
    kick = np.array((angle*np.cos(azimuth), angle*np.sin(azimuth)))
    path = CanonicalPath(1e-3)
    path.append(np.eye(4), np.r_[0., 0., kick])
    wave = propagate_plane_wave(wave, path.matrix, path.offset, float(wavelength_m(new_energy*1000)), **path.phase_kwargs())
    event.update(output_energy_kev=new_energy, kick_rad=kick.tolist(), azimuth_rad=azimuth)
    return replace(mode, plane=wave, energy_kev=new_energy,
                   scattering_history=(*mode.scattering_history, event)), event


def _propagate_inelastic_specimen(state, checkpoint, *, numerics, store, maximum_step_mm,
        grid_numerics, tip_time_s, cancelled, progress_callback, verify, use_cache):
    from temsim.physics.specimen_wave_transport import (_prepare_material_grid, _regrid_mode, _slice_phase, _bandlimit_mode)
    from temsim.physics.column_wave import _propagate_column
    from temsim.physics.wave_imaging import interaction_constant_rad_per_v_angstrom
    from temsim.specimen.inelastic import real_inelastic_distribution
    numerics.validate()
    start = checkpoint.plane_z_mm
    stop = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    parent_id = checkpoint.digest
    key = store.key("inelastic-specimen", parent_id, asdict(numerics), maximum_step_mm, asdict(grid_numerics), tip_time_s)
    cached = store.get(key) if use_cache else None
    if cached is not None:
        return cached
    scene, prepared, x, y, dzs = _prepare_material_grid(state, checkpoint)
    configs = prepared.potential_configurations_v_angstrom
    retained_potential_bytes = sum(p.nbytes for p in configs)
    grid_numerics.check((len(y), len(x)), retained_bytes=retained_potential_bytes+48*len(x)*len(y))
    writer = store.writer(key, checkpoint.beam.reference_plane)
    records = []
    total = len(checkpoint.beam.modes)*len(configs)*numerics.trajectories_per_mode
    try:
        for source in checkpoint.beam.modes:
            base, regrid = _regrid_mode(source, x, y)
            for ci, potential in enumerate(configs):
                for ti in range(numerics.trajectories_per_mode):
                    if cancelled():
                        raise InterruptedError("Inelastic trajectory cancelled")
                    mode = replace(base, mode_id=base.mode_id+f"/phonon:{ci}/trajectory:{ti}",
                        weight_per_reference_electron=base.weight_per_reference_electron/(len(configs)*numerics.trajectories_per_mode))
                    initial = mode.weight_per_reference_electron
                    mode, band_loss = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction)
                    current_z, absorption, column_loss, steps = start, 0., 0., []
                    for si, dz in enumerate(dzs):
                        if cancelled():
                            raise InterruptedError("Inelastic trajectory slice cancelled")
                        next_z = stop if si == len(dzs)-1 else current_z+float(dz)*1e-7
                        step_key = store.key("trajectory-slice", key, source.mode_id, ci, ti, si)
                        cached_slice = store.get(step_key) if use_cache else None
                        if cached_slice is not None:
                            mode = cached_slice.beam.modes[0]
                            row = cached_slice.record
                            band_loss, absorption, column_loss = row["band_loss"], row["absorption"], row["column_loss"]
                            steps = list(row["steps"])
                            current_z = next_z
                            continue
                        projected = potential[si] if potential.ndim == 3 else potential*dz/sum(dzs)
                        sigma = interaction_constant_rad_per_v_angstrom(mode.energy_kev)
                        if mode.weight_per_reference_electron:
                            mode = _slice_phase(mode, projected, x, y, sigma, .5)
                            mode, lost = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction); band_loss += lost
                        local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), current_z,
                            checkpoint.reference_current_a, {"specimen_parent": parent_id})
                        propagated = _propagate_column(state, local, next_z, maximum_step_mm=maximum_step_mm,
                            grid_numerics=grid_numerics, tip_time_s=tip_time_s, cancelled=cancelled,
                            retained_bytes=retained_potential_bytes+base.plane.amplitude.nbytes+source.plane.amplitude.nbytes)
                        column_loss += mode.weight_per_reference_electron-propagated.beam.total_weight
                        mode = propagated.beam.modes[0]
                        event = {"kind": "already_absorbed", "absorbed_weight": 0.}
                        if mode.weight_per_reference_electron:
                            mode = _slice_phase(mode, projected, x, y, sigma, .5)
                            sample = copy(state.sample)
                            sample.thickness_nm = float(dz)*.1
                            distribution = real_inelastic_distribution(SimpleNamespace(sample=sample, beam_voltage_kv=mode.energy_kev))
                            xy = mode.plane.coordinates_m()*1e9
                            inside = scene.sample_contains_xy(xy[0], xy[1])
                            mode, event = _collide_slice(mode, inside, distribution,
                                _trajectory_rng(numerics.seed, source.mode_id, ci, ti, si), next_z)
                            absorption += event["absorbed_weight"]
                            mode, lost = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction); band_loss += lost
                        steps.append({"slice": si, "event": event, "column": propagated.record["modes"]})
                        verify()
                        saved = store.put(step_key, TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), next_z,
                            checkpoint.reference_current_a, {"steps": steps, "band_loss": band_loss,
                                "absorption": absorption, "column_loss": column_loss, "parent": parent_id}))
                        del local, propagated, saved
                        current_z = next_z
                    accounted = mode.weight_per_reference_electron+band_loss+absorption+column_loss
                    if not math.isclose(accounted, initial, rel_tol=1e-9, abs_tol=1e-13):
                        raise ValueError("Quantum trajectory probability accounting failed")
                    writer.append(mode)
                    records.append({"mode_id": mode.mode_id, "input_weight": initial, "output_weight": mode.weight_per_reference_electron,
                        "energy_kev": mode.energy_kev, "regrid": regrid, "steps": steps,
                        "band_loss": band_loss, "absorption": absorption, "column_loss": column_loss,
                        "probability_residual": initial-accounted})
                    del mode
                    if progress_callback:
                        progress_callback(len(records), total, "Conditional inelastic wave saved; trajectory buffers released")
            del source, base
        verify()
        if cancelled():
            raise InterruptedError("Inelastic checkpoint cancelled before commit")
        return writer.finish(stop, checkpoint.reference_current_a,
            {"schema": "executed-inelastic-trajectories-v1", "upstream_digest": parent_id, "upstream": checkpoint.record,
             "potential": prepared.metrics, "modes": records, "numerics": asdict(numerics),
             "model": "local Markov momentum-transfer Kraus instrument of existing material Poisson channels",
             "phase": "conditional within each trajectory; no phase between environmental outcomes",
             "statistics": "independent trajectories; converge count and seed, errors scale as N^-1/2",
             "slice_model": "collision at slice exit; converge specimen slice thickness independently",
             "limitations": "representative losses/angles, including existing plural approximation; no atomic transition potentials or resolved EELS edges",
             "validation_status": "DEVELOPMENT"})
    finally:
        writer.abort()
