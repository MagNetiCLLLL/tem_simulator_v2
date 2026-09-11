"""Historical exit-source mathematics only; this model cannot run the instrument."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.effective_source import (
    EffectiveGunSource, gun_binding_digest,
    _reconstruct_historical_emission as generate_gun_emission, wavelength_m)


def historical_parameters(gun, parameters):
    """Reconstruct archived inputs for isolated mathematics, never activation."""
    return replace(parameters, bound_gun_digest=gun_binding_digest(gun))


@pytest.fixture
def state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def test_historical_source_reconstruction_does_not_change_the_physical_gun(state):
    before = capture_instrument_snapshot(state).digest
    parameters = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9))
    source = generate_gun_emission(state.electron_gun, parameters)
    assert source.plane_z_mm == state.electron_gun.exit_plane_z_mm
    assert source.reference_current_a == 1e-9
    assert capture_instrument_snapshot(state).digest == before
    assert source.record["physical_calibration"] == "PHENOMENOLOGICAL_NOT_MEASURED"
    assert len(list(source.modes())) == 3
    assert sum(m.weight_per_reference_electron for m in source.modes()) == pytest.approx(1.)


def test_quantum_limited_mode_covariance_and_fourier_angle(state):
    p = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9, energy_fwhm_ev=0))
    source = generate_gun_emission(state.electron_gun, p)
    mode, = source.modes()
    field = mode.plane.full_amplitude(wavelength_m(source.energies_ev[0]))
    xy = mode.plane.coordinates_m()
    probability = abs(field)**2
    variance_x = float(np.sum(probability*xy[0]**2))
    spectrum = abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho")))**2
    angle = np.fft.fftshift(np.fft.fftfreq(field.shape[1], mode.plane.basis_m[0, 0]))*wavelength_m(source.energies_ev[0])
    variance_angle = float(np.sum(spectrum*angle[None, :]**2))
    assert variance_x == pytest.approx(source.covariance_by_energy[0, 0, 0], rel=1e-10)
    assert variance_angle == pytest.approx(source.covariance_by_energy[0, 2, 2], rel=1e-10)
    assert math.sqrt(variance_x*variance_angle) == pytest.approx(wavelength_m(source.energies_ev[0])/(4*math.pi), rel=1e-10)


def test_mixed_modes_match_analytic_gaussian_and_omit_not_renormalise(state):
    p = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9, energy_fwhm_ev=0,
        incoherent_angle_rms_mrad=.1, mode_tail_tolerance=1e-7, grid_pixels=256))
    source = generate_gun_emission(state.electron_gun, p)
    modes = list(source.modes())
    weight = sum(mode.weight_per_reference_electron for mode in modes)
    assert 0 < 1-weight <= p.mode_tail_tolerance
    assert 1-weight == pytest.approx(source.record["weighted_omitted_mode_probability"], abs=2e-15)
    variance_x = variance_angle = 0.
    for mode in modes:
        wave = mode.plane.amplitude
        xy = mode.plane.coordinates_m()
        variance_x += mode.weight_per_reference_electron*float(np.sum(abs(wave)**2*xy[0]**2))
        spectrum = abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave), norm="ortho")))**2
        angle = np.fft.fftshift(np.fft.fftfreq(wave.shape[1], mode.plane.basis_m[0, 0]))*wavelength_m(source.energies_ev[0])
        variance_angle += mode.weight_per_reference_electron*float(np.sum(spectrum*angle[None, :]**2))
    # Truncated high-order modes contribute disproportionately to second moments.
    assert variance_x == pytest.approx(source.covariance_by_energy[0, 0, 0], rel=3e-6)
    assert variance_angle == pytest.approx(source.covariance_by_energy[0, 2, 2], rel=3e-6)


def test_correlated_energy_samples_are_deterministic_and_readonly(state):
    p = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9,
        dispersion_nm_per_ev=40., angular_dispersion_mrad_per_ev=.2))
    source = generate_gun_emission(state.electron_gun, p)
    first = source.particle_samples(16384)
    second = source.particle_samples(16384)
    np.testing.assert_array_equal(first.x_m, second.x_m)
    assert not first.x_m.flags.writeable
    slope = np.cov(first.x_m, first.energy_offset_ev)[0, 1]/np.var(first.energy_offset_ev, ddof=1)
    assert slope == pytest.approx(40e-9, rel=.005)
    for mode in source.modes():
        delta = mode.energy_kev*1000 - source.record["mean_energy_ev"]
        assert mode.plane.origin_m[0] == pytest.approx(40e-9*delta, abs=1e-17)


def test_historical_source_identity_rejects_changed_gun_controls(state):
    p = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9))
    state.electron_gun.emitter.ray_count += 100
    state.electron_gun.emitter._tuning_boundary_probes = 33
    generate_gun_emission(state.electron_gun, p)
    state.electron_gun.dpa_aperture.radius_mm *= .9
    with pytest.raises(ValueError, match="stale"):
        generate_gun_emission(state.electron_gun, p)




def test_bound_source_shelf_roundtrips_without_enabling_it(state):
    gun = state.electron_gun
    gun.effective_source = historical_parameters(gun, EffectiveGunSource(1e-9))
    snapshot = capture_instrument_snapshot(state)
    restored = snapshot.restore()
    assert restored.electron_gun.source_representation == "classical_particles"
    assert generate_gun_emission(restored.electron_gun).digest == generate_gun_emission(gun).digest
    assert capture_instrument_snapshot(restored).digest == snapshot.digest


def test_budget_and_invalid_source_parameters_fail_before_allocation(state):
    with pytest.raises(ValueError):
        EffectiveGunSource(-1.)
    with pytest.raises(ValueError):
        EffectiveGunSource(1., energy_nodes=1)
    p = historical_parameters(state.electron_gun, EffectiveGunSource(1e-9, incoherent_angle_rms_mrad=1., maximum_modes=1))
    with pytest.raises(ValueError, match="above limit"):
        generate_gun_emission(state.electron_gun, p)


@pytest.mark.parametrize("tolerance", [1e-20, 1e-100, np.nextafter(0., 1.)])
def test_tiny_mode_tail_tolerances_do_not_cancel_to_zero(tolerance):
    from temsim.optics.electron_gun.effective_source import _mode_parameters
    parameters = EffectiveGunSource(1e-9, energy_fwhm_ev=0,
        incoherent_angle_rms_mrad=.01, mode_tail_tolerance=tolerance,
        maximum_modes=65536)  # Extreme tolerance deliberately exceeds the default mode budget.
    _, _, _, ratio, count, tail = _mode_parameters(parameters, 300000.)
    assert 0 < ratio < 1
    assert count > 1
    assert 0 <= tail <= tolerance
    # Independently check minimality in log space, without a cancellation-
    # prone 1-(1-q**n)**2 reference formula.
    previous_log_r = (count-1)*math.log(ratio)
    assert previous_log_r + math.log(2-math.exp(previous_log_r)) > math.log(tolerance)


@pytest.mark.parametrize("size,angle,match", [
    (1e-320, 0., "representable SI"),
    (1e100, 1., "above limit"),
    (1e200, 0., "covariance"),
    (1e-160, 0., "covariance"),
    (5., 1e308, "covariance"),
])
def test_extreme_finite_source_inputs_fail_explicitly(size, angle, match):
    from temsim.optics.electron_gun.effective_source import _mode_parameters
    parameters = EffectiveGunSource(1e-9, source_fwhm_nm=size,
        incoherent_angle_rms_mrad=angle)
    with pytest.raises(ValueError, match=match):
        _mode_parameters(parameters, 300000.)
