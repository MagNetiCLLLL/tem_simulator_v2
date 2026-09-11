"""New effective-source mathematics; NOT a complete column validation."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.effective_source import (
    EffectiveGunSource, bind_effective_source, generate_gun_emission, wavelength_m)


@pytest.fixture
def state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def test_new_source_binding_and_generation_do_not_change_legacy_gun(state):
    before = capture_instrument_snapshot(state).digest
    parameters = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9))
    source = generate_gun_emission(state.electron_gun, parameters)
    assert source.plane_z_mm == state.electron_gun.exit_plane_z_mm
    assert source.reference_current_a == 1e-9
    assert capture_instrument_snapshot(state).digest == before
    assert source.record["physical_calibration"] == "PHENOMENOLOGICAL_NOT_MEASURED"
    assert len(list(source.modes())) == 3
    assert sum(m.weight_per_reference_electron for m in source.modes()) == pytest.approx(1.)


def test_quantum_limited_mode_covariance_and_fourier_angle(state):
    p = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9, energy_fwhm_ev=0))
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
    p = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9, energy_fwhm_ev=0,
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
    p = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9,
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


def test_binding_rejects_gun_changes_but_not_ray_count(state):
    p = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9))
    state.electron_gun.emitter.ray_count += 100
    state.electron_gun.emitter._tuning_boundary_probes = 33
    generate_gun_emission(state.electron_gun, p)
    state.electron_gun.dpa_aperture.radius_mm *= .9
    with pytest.raises(ValueError, match="stale"):
        generate_gun_emission(state.electron_gun, p)


def test_alignment_uses_same_selected_gun_not_legacy_emitter(state, monkeypatch):
    from temsim.optics.direct_alignment import _CondenserMeasurementModel
    from temsim.optics.electron_gun.source import trace_source_to_exit
    gun = state.electron_gun
    gun.emitter.ray_count = 49
    gun.effective_source = bind_effective_source(gun, EffectiveGunSource(1e-9))
    gun.source_representation = "effective_gaussian_schell"
    expected = trace_source_to_exit(state)
    def forbidden(*args, **kwargs):
        raise AssertionError("The selected effective source must not call the legacy gun emitter")
    monkeypatch.setattr(type(gun), "trace_to_exit", forbidden)
    model = _CondenserMeasurementModel(state, step_mm=.2)
    np.testing.assert_array_equal(model.source_rays[0], expected.exit_bundle.x_m)
    np.testing.assert_array_equal(model.source_rays[2], expected.exit_bundle.tx_rad)


def test_bound_source_shelf_roundtrips_without_enabling_it(state):
    gun = state.electron_gun
    gun.effective_source = bind_effective_source(gun, EffectiveGunSource(1e-9))
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
    p = bind_effective_source(state.electron_gun, EffectiveGunSource(1e-9, incoherent_angle_rms_mrad=1., maximum_modes=1))
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


def test_particle_exit_is_the_same_canonical_source_without_fabricated_internal_rays(state):
    from temsim.optics.electron_gun.effective_source import trace_effective_source
    from temsim.physics.simulation import _gun_traces_match
    gun = state.electron_gun
    gun.effective_source = bind_effective_source(gun, EffectiveGunSource(1e-9, energy_fwhm_ev=0))
    gun.source_representation = "effective_gaussian_schell"
    before = capture_instrument_snapshot(state).digest
    trace = trace_effective_source(state, 4096)
    assert trace.z_mm.tolist() == [gun.exit_plane_z_mm]
    assert trace.dpa_transmitted_current_a is None
    assert trace.c1_transmitted_current_a is None
    assert trace.emitted_current_a == 1e-9
    assert _gun_traces_match(trace, trace_effective_source(state, 4096))
    assert capture_instrument_snapshot(state).digest == before
    with pytest.raises(TypeError):
        trace.source_record["source_id"] = "forged"
    with pytest.raises(ValueError, match="installed gun exit"):
        gun.emit()
    source = generate_gun_emission(gun)
    canonical = source.particle_samples(4096)
    from temsim.physics.core import electron
    q, momentum, _ = electron(state)
    g = q*trace.source_record["exit_bz_t"]/(2*momentum)
    np.testing.assert_allclose(trace.tx_rad[0]-g*trace.y_m[0], canonical.tx_rad, atol=1e-18)


def test_source_dialog_requires_explicit_current_and_preserves_old_shelf(state, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun)
    qtbot.addWidget(dialog)
    dialog.representation.setCurrentIndex(1)
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert dialog.error.text()
    dialog.inputs["reference_current_a"].setText("1e-9")
    dialog.accept()
    representation, parameters = dialog.value()
    assert representation == "effective_gaussian_schell"
    assert capture_instrument_snapshot(state).digest == before
    gun = state.electron_gun
    gun.source_representation, gun.effective_source = representation, parameters
    dialog2 = GunSourceDialog(gun)
    qtbot.addWidget(dialog2)
    dialog2.representation.setCurrentIndex(0)
    dialog2.accept()
    assert dialog2.value() == ("classical_particles", parameters)


def test_profile_retains_versioned_source_and_legacy_parameters(state, tmp_path):
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.instrument_snapshot import encode_instrument
    gun = state.electron_gun
    legacy = encode_instrument(gun.emitter)
    gun.effective_source = bind_effective_source(gun, EffectiveGunSource(2e-9))
    gun.source_representation = "effective_gaussian_schell"
    path = tmp_path / "source.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    restored = capture_instrument_snapshot(state).restore()
    restored.electron_gun.source_representation = "classical_particles"
    apply_profile_values(restored, values)
    assert restored.electron_gun.effective_source == gun.effective_source
    assert restored.electron_gun.source_representation == "effective_gaussian_schell"
    assert encode_instrument(restored.electron_gun.emitter) == legacy
    assert generate_gun_emission(restored.electron_gun).digest == generate_gun_emission(gun).digest


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_all_request_qualities_preserve_the_explicit_gun_binding(state, quality):
    from threading import Event
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.calculation_controller import CalculationController
    gun = state.electron_gun
    gun.effective_source = bind_effective_source(gun, EffectiveGunSource(1e-9))
    gun.source_representation = "effective_gaussian_schell"
    expected = generate_gun_emission(gun).digest
    captured = CapturedCalculationRequest.capture(state, quality, 49, .25)
    snapshot = captured.prepare(Event()).snapshot
    synchronous = CalculationController._calculation_snapshot(state, quality, 49, .25)
    for prepared in (snapshot, synchronous):
        assert prepared.electron_gun.source_representation == "effective_gaussian_schell"
        assert generate_gun_emission(prepared.electron_gun).digest == expected
