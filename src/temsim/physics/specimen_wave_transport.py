"""Finite-specimen wave coupling with the installed column still present.

Source modes and frozen-phonon configurations form an incoherent product
ensemble. A real potential supplies elastic phase; existing material IMFPs
remove probability from the coherent zero-loss branch and record that loss.
No phase is fabricated for the separate stochastic inelastic channels.
"""
from dataclasses import replace
import math

import numpy as np
from scipy.ndimage import map_coordinates

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.column_wave import _propagate_column
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_checkpoint_store import resident_wave_bytes
from temsim.physics.specimen_wave_channels import _attenuate_zero_loss


def _covering_domain(beam):
    from temsim.physics.wave_checkpoint_store import _StoredModes
    bounds = []
    geometries = beam.modes.geometries() if isinstance(beam.modes, _StoredModes) else (
        (m.plane.amplitude.shape, m.plane.basis_m, m.plane.origin_m) for m in beam.modes)
    for (ny, nx), basis, origin in geometries:
        corners = np.array([[-nx//2-.5, -ny//2-.5], [nx-1-nx//2+.5, -ny//2-.5],
                            [-nx//2-.5, ny-1-ny//2+.5], [nx-1-nx//2+.5, ny-1-ny//2+.5]])
        bounds.extend(origin+corners@basis.T)
    bounds = np.array(bounds)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    return (lower+upper)/2, float(max(upper-lower)*1.02)


def _regrid_mode(mode, x_m, y_m, tolerance=1e-4):
    """Lossless coordinate change, checked before a specimen can remove flux.

Interpolate the complex envelope, evaluate analytic carriers at the requested
coordinates, and test independent cubic/quintic estimates. No crop or source
fit is allowed. Small quadrature normalization is recorded separately from
physical flux and cannot restore any upstream aperture loss.
"""
    wave = mode.plane
    xx, yy = np.meshgrid(x_m, y_m)
    xy = np.stack((xx, yy))-wave.origin_m[:, None, None]
    index = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m), xy)
    ny, nx = wave.amplitude.shape
    coordinates = np.stack((index[1]+ny//2, index[0]+nx//2))
    # Require the target cell rectangle to contain the full source lattice.
    dx, dy = x_m[1]-x_m[0], y_m[1]-y_m[0]
    points = wave.coordinates_m()
    if (points[0].min() < x_m[0]-dx/2 or points[0].max() > x_m[-1]+dx/2
            or points[1].min() < y_m[0]-dy/2 or points[1].max() > y_m[-1]+dy/2):
        raise ValueError("Specimen grid would crop the incoming wave; enlarge the domain")
    scale = math.sqrt(abs(dx*dy/np.linalg.det(wave.basis_m)))
    a3, a5 = [map_coordinates(wave.amplitude, coordinates, order=order, mode="constant", cval=0)*scale for order in (3, 5)]
    norm = float(np.sum(abs(a5)**2))
    error = float(np.linalg.norm(a3-a5)/max(np.linalg.norm(a5), 1e-300))
    if abs(norm-wave.probability) > tolerance or error > tolerance:
        raise ValueError(f"Complex wave regridding is unresolved (norm error={norm-wave.probability:.3g}, interpolation difference={error:.3g}); refine the upstream wave grid")
    correction = math.sqrt(wave.probability/norm) if norm else 1.
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    # Change the carrier origin algebraically; do not fit phase to samples.
    origin = np.array((x_m[len(x_m)//2], y_m[len(y_m)//2]))
    delta = origin-wave.origin_m
    lam = float(wavelength_m(mode.energy_kev*1000))
    constant = (delta@q@delta/2+tilt@delta)*2*np.pi/lam
    plane = PlaneWave(a5*correction*np.exp(1j*constant), np.diag((dx, dy)), origin, q, tilt+q@delta)
    return replace(mode, plane=plane), {"interpolation_difference": error,
        "pre_correction_norm": norm, "quadrature_amplitude_correction": correction,
        "physical_weight_unchanged": mode.weight_per_reference_electron}


def _slice_phase(mode, potential, x_m, y_m, sigma, fraction):
    wave = mode.plane
    xy = wave.coordinates_m()
    coords = np.stack(((xy[1]-y_m[0])/(y_m[1]-y_m[0]), (xy[0]-x_m[0])/(x_m[1]-x_m[0])))
    nearest = np.rint(coords)
    coords = np.where(abs(coords-nearest) < 1e-10, nearest, coords)
    sampled = map_coordinates(np.asarray(potential, dtype=np.float64), coords, order=1, mode="constant", cval=0.)
    phase = sigma*fraction*sampled
    # Expanding an unresolved carrier would fold diffraction into false bands.
    full = wave.full_amplitude(float(wavelength_m(mode.energy_kev*1000)))
    occupied = abs(full) > abs(full).max()*1e-8
    for axis in (0, 1):
        low = [slice(None), slice(None)]; high = low.copy()
        low[axis] = slice(None, -1); high[axis] = slice(1, None)
        low, high = tuple(low), tuple(high)
        # Wave zeros/vortices can have a pi phase jump even in a resolved
        # complex field. Only the known added operator has an unambiguous
        # unwrapped phase here; the full field is bandwidth-filtered below.
        difference = np.diff(phase, axis=axis)
        if np.any(abs(difference)[occupied[low] & occupied[high]] >= np.pi):
            raise ValueError("Specimen phase is undersampled; refine the wave and potential grids")
    return replace(mode, plane=replace(wave, amplitude=wave.amplitude*np.exp(1j*phase)))


def _bandlimit_mode(mode, fraction):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Specimen bandwidth fraction must be in (0, 1]")
    wave = mode.plane
    full = wave.full_amplitude(float(wavelength_m(mode.energy_kev*1000)))
    fy, fx = np.meshgrid(np.fft.fftfreq(full.shape[0]), np.fft.fftfreq(full.shape[1]), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    cutoff = .5*fraction/np.linalg.svd(wave.basis_m, compute_uv=False).max()
    filtered = np.fft.ifft2(np.fft.fft2(full)*(np.hypot(frequency[0], frequency[1]) <= cutoff))
    norm = float(np.sum(abs(filtered)**2))
    before = mode.weight_per_reference_electron
    result = replace(mode, plane=PlaneWave(filtered/math.sqrt(norm) if norm else filtered, wave.basis_m, wave.origin_m),
                     weight_per_reference_electron=before*norm)
    return result, before-result.weight_per_reference_electron


def _prepare_material_grid(state, checkpoint):
    from temsim.physics.wave_imaging import prepare_specimen_potentials, interaction_constant_rad_per_v_angstrom
    from temsim.specimen.presets import load_specimen_preset
    from temsim.specimen.scene import SpecimenScene
    scene = SpecimenScene.from_state(state)
    start = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    stop = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    if not math.isclose(checkpoint.plane_z_mm, start, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Specimen interaction needs the wave at the actual upper face")
    if not scene.structure_available:
        raise ValueError("The installed specimen has no supported potential; it cannot be replaced by vacuum")
    if not state.sample.wave_multislice_enabled:
        raise ValueError("This finite-thickness tip wave path requires multislice; the selected phase-object model remains available in its existing workflow")
    if bool(getattr(state.sample, "real_high_angle_tail_enabled", False)):
        raise ValueError("The selected high-angle tail requires its wave scattering channel; it cannot be omitted")
    centre, fov = _covering_domain(checkpoint.beam)
    prepared = prepare_specimen_potentials(state, load_specimen_preset(scene.wave_template_key),
        field_of_view_angstrom_override=fov*1e10, calculation_roi_centre_nm=tuple(centre*1e9),
        calculation_roi_bounds_nm=tuple(np.r_[centre[0]-fov/2, centre[0]+fov/2, centre[1]-fov/2, centre[1]+fov/2]*1e9))
    if prepared.metrics.get("atomistic_fallback_reason"):
        raise ValueError("Selected specimen potential was not computed: "+str(prepared.metrics["atomistic_fallback_reason"]))
    x = prepared.x_angstrom*1e-10+centre[0]
    y = prepared.y_angstrom*1e-10+centre[1]
    configs = prepared.potential_configurations_v_angstrom
    dzs = prepared.slice_thicknesses_angstrom
    if dzs is None:
        n = max(1, math.ceil(state.sample.thickness_nm*10/state.sample.wave_slice_thickness_angstrom))
        dzs = np.full(n, state.sample.thickness_nm*10/n)
    if not math.isclose(float(sum(dzs)), state.sample.thickness_nm*10, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Specimen slice thicknesses do not span its physical faces")
    return scene, prepared, x, y, dzs


def _propagate_specimen(state, checkpoint, *, maximum_step_mm=.5, maximum_checkpoint_bytes=8*1024**3,
                        grid_numerics=WaveGridNumerics(), tip_time_s=None,
                        cancelled=lambda: False, progress_callback=None):
    from temsim.physics.wave_imaging import interaction_constant_rad_per_v_angstrom
    from temsim.specimen.scene import SpecimenScene
    from temsim.specimen.inelastic import real_inelastic_distribution
    start = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    stop = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    if not math.isclose(checkpoint.plane_z_mm, start, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Specimen interaction needs the wave at the actual upper face")
    if stop == start:
        return checkpoint
    if SpecimenScene.from_state(state).is_vacuum:
        return _propagate_column(state, checkpoint, stop, maximum_step_mm=maximum_step_mm,
            grid_numerics=grid_numerics, tip_time_s=tip_time_s, cancelled=cancelled, progress_callback=progress_callback)
    scene, prepared, x, y, dzs = _prepare_material_grid(state, checkpoint)
    configs = prepared.potential_configurations_v_angstrom
    required = len(x)*len(y)*(16*len(checkpoint.beam.modes)*len(configs)+256)
    if required > maximum_checkpoint_bytes:
        raise ValueError(f"Specimen wave modes and working buffers need approximately {required} bytes, above maximum_checkpoint_bytes={maximum_checkpoint_bytes}")
    outputs, records = [], []
    for source_mode in checkpoint.beam.modes:
        base, grid_record = _regrid_mode(source_mode, x, y)
        from types import SimpleNamespace
        energy_state = SimpleNamespace(sample=state.sample, beam_voltage_kv=source_mode.energy_kev)
        distribution = real_inelastic_distribution(energy_state)
        rate = sum(0. if not math.isfinite(value) else 1/value for value in
                   (distribution.total_inelastic_mean_free_path_nm, distribution.absorption_mean_free_path_nm))
        for config_index, potential in enumerate(configs):
            if cancelled():
                raise InterruptedError("Specimen wave propagation cancelled")
            if progress_callback:
                progress_callback(len(outputs), len(checkpoint.beam.modes)*len(configs), "Finite specimen wave / frozen-phonon branch")
            mode = replace(base, mode_id=base.mode_id+f"/phonon:{config_index}",
                           weight_per_reference_electron=base.weight_per_reference_electron/len(configs))
            initial_weight, current_z, slice_records = mode.weight_per_reference_electron, start, []
            first_events = {key: 0. for key in ("real_plasmon", "real_ionisation", "real_other_inelastic", "effective_absorption")}
            column_loss = 0.
            mode, band_loss = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction)
            sigma = interaction_constant_rad_per_v_angstrom(mode.energy_kev)
            for i, dz in enumerate(dzs):
                projected = potential[i] if potential.ndim == 3 else potential*dz/sum(dzs)
                mode = _slice_phase(mode, projected, x, y, sigma, .5)
                mode, lost = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction)
                band_loss += lost
                local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), current_z,
                    checkpoint.reference_current_a, {"specimen_parent": checkpoint.digest, "configuration": config_index, "slice": i})
                next_z = stop if i == len(dzs)-1 else current_z+float(dz)*1e-7
                transported = _propagate_column(state, local, next_z, maximum_step_mm=maximum_step_mm,
                    grid_numerics=grid_numerics, tip_time_s=tip_time_s,
                    retained_bytes=resident_wave_bytes(checkpoint.beam)
                        +sum(m.plane.amplitude.nbytes for m in outputs)+sum(p.nbytes for p in configs),
                    cancelled=cancelled)
                column_loss += mode.weight_per_reference_electron-transported.beam.total_weight
                mode = _slice_phase(transported.beam.modes[0], projected, x, y, sigma, .5)
                # Zero-loss survival applies only where the finite specimen has
                # matter. Lost weight stays in the ledger, not in a fake phase.
                xy_nm = mode.plane.coordinates_m()*1e9
                inside = scene.sample_contains_xy(xy_nm[0], xy_nm[1])
                mode, channel_record = _attenuate_zero_loss(mode, inside, float(dz)*.1, distribution)
                for event in channel_record["events"]:
                    first_events[event["kind"]] += event["weight"]
                mode, lost = _bandlimit_mode(mode, state.sample.wave_bandwidth_fraction)
                band_loss += lost
                slice_records.append({"slice": i, "z_mm": next_z, "column": transported.record["modes"],
                                      "inelastic_absorption_removed_weight": channel_record["removed_weight"],
                                      "first_event_ledger": channel_record})
                current_z = next_z
            outputs.append(mode)
            accounted = mode.weight_per_reference_electron+band_loss+column_loss+sum(first_events.values())
            if not math.isclose(accounted, initial_weight, rel_tol=1e-10, abs_tol=1e-13):
                raise ValueError("Specimen wave, physical losses, first events and numerical bandwidth did not conserve probability")
            records.append({"mode_id": mode.mode_id, "input_weight": initial_weight,
                "output_weight": mode.weight_per_reference_electron, "regrid": grid_record,
                "inelastic_material": {"key": distribution.material_key, "model": distribution.model,
                    "energy_kev": distribution.beam_energy_kev, "warnings": distribution.warnings,
                    "zero_loss_removal_rate_per_nm": rate,
                    "channels": [{"key": c.key, "energy_loss_ev": c.energy_loss_ev,
                                  "characteristic_angle_mrad": c.characteristic_angle_mrad,
                                  "mean_events": c.mean_events, "full_thickness_probability": c.probability,
                                  "inverse_mfp_per_nm": 0. if not math.isfinite(c.mean_free_path_nm) else 1/c.mean_free_path_nm}
                                 for c in distribution.channels]},
                "slices": slice_records, "numerical_band_loss": band_loss,
                "probability_balance": {"coherent_zero_loss": mode.weight_per_reference_electron,
                    "column_absorbed": column_loss, "numerical_band_loss": band_loss,
                    "first_events": first_events, "accounted_weight": accounted,
                    "residual": initial_weight-accounted}})
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), stop, checkpoint.reference_current_a,
        {"schema": "executed-specimen-wave-v1", "upstream_digest": checkpoint.digest, "upstream": checkpoint.record,
         "potential": prepared.metrics, "modes": records, "branch": "coherent zero-loss with material-IMFP attenuation",
         "inelastic_phase": "UNDEFINED; removed probability recorded; energy-loss branches remain in the existing particle workflow",
         "integrator": "symmetric specimen phase / distributed column Hamiltonian / specimen phase",
         "bandwidth_policy": "full-field filter at entry and each half-slice phase; operator phase Nyquist checked",
         "validation_status": "DEVELOPMENT"})
