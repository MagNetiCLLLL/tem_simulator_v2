"""Physical tip definition and its independent quantum/statistical checks."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.constants import hbar

from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_coherence import (
    TipCoherence, TipWaveNumerics, generate_tip_emission, tip_covariance, wavelength_m,
)


@pytest.fixture
def gun():
    result = FieldEmissionGun()
    result.emitter.coherence = TipCoherence()
    result.emitter.energy_spread_fwhm_ev = 0.
    return result


def test_tip_is_explicit_and_never_uses_exit_voltage(gun):
    numerics = TipWaveNumerics(energy_samples=1)
    emitted = generate_tip_emission(gun, numerics)
    gun.accelerator.high_tension_kv = 200.
    assert generate_tip_emission(gun, numerics).digest == emitted.digest
    mode, = tuple(emitted.modes())
    assert mode.energy_kev == pytest.approx(.0003)
    assert emitted.record["plane_z_mm"] == 0.
    assert emitted.reference_current_a == gun.emitter.emitted_current_a
    gun.emitter.coherence = None
    with pytest.raises(ValueError, match="explicitly"):
        generate_tip_emission(gun, numerics)


@pytest.mark.parametrize("incoherent", [0., 45.])
def test_modes_reconstruct_full_quantum_covariance_and_preserve_tail(gun, incoherent):
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=incoherent,
        curvature_x_m1=2e6, curvature_xy_m1=1e6, curvature_y_m1=-1e6)
    emission = generate_tip_emission(gun, TipWaveNumerics(grid_pixels=256, energy_samples=1,
                                                        mode_tail_tolerance=1e-9, maximum_modes=1024))
    modes = tuple(emission.modes())
    wavelength = float(wavelength_m(gun.emitter.emission_energy_ev))
    covariance = sum(m.weight_per_reference_electron*m.plane.canonical_covariance(wavelength) for m in modes)
    expected = tip_covariance(gun.emitter, gun.emitter.emission_energy_ev)
    scale = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(covariance/np.outer(scale, scale), expected/np.outer(scale, scale), atol=2e-7)
    assert sum(m.weight_per_reference_electron for m in modes) == pytest.approx(
        1-emission.record["omitted_probability"], abs=1e-13)
    action = math.sqrt(expected[0, 0]*expected[2, 2]-expected[0, 2]**2)*(2*np.pi*hbar/wavelength)
    assert action >= hbar/2*(1-1e-14)
    assert not modes[0].plane.amplitude.flags.writeable


def test_wigner_particles_use_the_same_tip_momentum_distribution(gun):
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=20., offset_x_nm=3.,
        tilt_y_mrad=4., curvature_xy_m1=1e6)
    count = 65536
    bundle = gun.emit(count)
    direction = np.array((bundle.tx_rad, bundle.ty_rad))
    transverse = direction/np.sqrt(1+np.sum(direction**2, axis=0))
    values = np.vstack((bundle.x_m, bundle.y_m, transverse))
    expected = tip_covariance(gun.emitter, gun.emitter.emission_energy_ev)
    covariance = np.cov(values, bias=True)
    scale = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(covariance/np.outer(scale, scale), expected/np.outer(scale, scale), atol=.002)
    assert values[0].mean() == pytest.approx(3e-9, abs=2e-12)
    assert values[3].mean() == pytest.approx(.004, abs=5e-5)


def test_existing_energy_spread_and_current_are_retained(gun):
    gun.emitter.energy_spread_fwhm_ev = .3
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=1.)
    source = generate_tip_emission(gun, TipWaveNumerics(energy_samples=33))
    energies = np.array([row["energy_ev"] for row in source.record["energy_modes"]])
    assert energies.mean() == pytest.approx(gun.emitter.emission_energy_ev, abs=1e-10)
    assert energies.std() == pytest.approx(.3/2.354820045, abs=1e-10)
    assert energies.min() >= gun.emitter.minimum_kinetic_energy_ev
    bundle = gun.emit(33)
    np.testing.assert_array_equal(energies, bundle.energy_offset_ev+gun.emitter.emission_energy_ev)
    assert source.reference_current_a == gun.emitted_current_a
    with pytest.raises(ValueError, match="discard"):
        generate_tip_emission(gun, TipWaveNumerics(energy_samples=1))


def test_no_silent_mode_budget_or_sampling_fallback(gun):
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=1000.)
    with pytest.raises(ValueError, match="budget|maximum_modes"):
        generate_tip_emission(gun, TipWaveNumerics(energy_samples=1, maximum_modes=1))
    gun.emitter.coherence = TipCoherence()
    emission = generate_tip_emission(gun, TipWaveNumerics(energy_samples=1, grid_pixels=32))
    with pytest.raises(ValueError, match="undersampled"):
        tuple(emission.modes())


@pytest.mark.parametrize("parameters", [TipCoherence(incoherent_angle_rms_mrad=-1),
    TipCoherence(curvature_x_m1=float("nan")), TipCoherence(tilt_x_mrad=float("inf"))])
def test_invalid_tip_coherence_is_rejected(gun, parameters):
    gun.emitter.coherence = parameters
    with pytest.raises(ValueError):
        gun.validate()


def test_profile_and_exact_snapshot_preserve_tip_model(gun):
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.optics.column import default_state
    gun.emitter.coherence = TipCoherence(curvature_xy_m1=21., tilt_y_mrad=.5)
    restored = field_emission_gun_from_dict(gun.to_dict())
    assert restored.emitter.coherence == gun.emitter.coherence
    state = default_state()
    state.electron_gun = gun
    exact = decode_instrument(encode_instrument(state)).electron_gun
    assert exact.emitter.coherence == gun.emitter.coherence
    np.testing.assert_array_equal(exact.emit(9).x_m, gun.emit(9).x_m)


def test_tip_dialog_explicitly_selects_coherence_without_mutating_live_gun(gun, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun.emitter.coherence = None
    before = gun.to_dict()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    assert not dialog.coherence_enabled.isChecked()
    dialog.coherence_enabled.setChecked(True)
    assert not dialog.inputs["angular_cutoff_mrad"].isEnabled()
    dialog.coherence_inputs["incoherent_angle_rms_mrad"].setText("2.5")
    dialog.accept()
    assert dialog.value()["coherence"] == TipCoherence(incoherent_angle_rms_mrad=2.5)
    assert gun.to_dict() == before


def test_disabled_coherence_retains_the_old_truncated_emission(gun):
    gun.emitter.coherence = None
    before = gun.emit(33)
    gun.emitter.coherence = TipCoherence()
    assert np.max(np.abs(gun.emit(33).tx_rad)) > .01
    gun.emitter.coherence = None
    after = gun.emit(33)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_unselected_tip_model_preserves_the_archived_classical_field_schema():
    from dataclasses import fields
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.optics.column import default_state
    state = default_state()
    before = encode_instrument(state)
    names = {field.name for field in fields(state.electron_gun.emitter)}
    assert "coherence" not in names
    restored = decode_instrument(before)
    assert encode_instrument(restored) == before
    restored.electron_gun.emitter.coherence = TipCoherence()
    assert encode_instrument(restored) != before
    restored.electron_gun.emitter.coherence = None
    assert encode_instrument(restored) == before
    assert "coherence" not in restored.electron_gun.to_dict()["components"][restored.electron_gun.emitter.key]


def test_saved_operating_profile_retains_and_can_clear_tip_coherence(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    state.electron_gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=3., curvature_x_m1=12.)
    path = tmp_path/"coherent-tip.toml"
    save_profile(path, state, selection)
    target = default_state()
    selected, values = read_profile(path)
    catalog.apply(target, selected)
    assert apply_profile_values(target, values) == []
    assert target.electron_gun.emitter.coherence == state.electron_gun.emitter.coherence
    state.electron_gun.emitter.coherence = None
    save_profile(path, state, selection)
    _, values = read_profile(path)
    assert apply_profile_values(target, values) == []
    assert target.electron_gun.emitter.coherence is None


def test_medium_tuning_keeps_zero_current_guides_for_the_selected_tip_law(gun):
    gun.emitter.coherence = TipCoherence(offset_x_nm=3., tilt_y_mrad=4.)
    gun.emitter._tuning_boundary_probes = 33
    bundle = gun.emit(193)
    assert np.count_nonzero(bundle.weight == 0) == 33
    assert bundle.weight.sum() == pytest.approx(1.)
    assert bundle.x_m[-1] == pytest.approx(3e-9)
    assert bundle.ty_rad[-1] == pytest.approx(.004/np.sqrt(1-.004**2))
    assert np.max(abs(bundle.tx_rad[-33:])) > .1


def test_configured_tip_does_not_enable_incomplete_tem_stem_images(gun):
    from temsim.optics.column import default_state
    from temsim.physics.source_admission import gun_phase_readiness, require_gun_wave_source, UnsupportedWaveSource
    state = default_state()
    state.electron_gun = gun
    readiness = gun_phase_readiness(state)
    assert readiness["coherent_source"] == "TIP_GAUSSIAN_SCHELL_CONFIGURED"
    assert readiness["validation_status"] == "NOT_RUN"
    with pytest.raises(UnsupportedWaveSource, match="not full image admission"):
        require_gun_wave_source(state, product="TEM")
