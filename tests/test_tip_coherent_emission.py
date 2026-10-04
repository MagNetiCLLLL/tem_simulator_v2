"""Physical tip definition and its independent quantum/statistical checks."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.constants import hbar

from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_coherence import (
    DRIVEN_GAUSSIAN_SCHELL, FORWARD_GAUSSIAN_SCHELL,
    TipCoherence, TipWaveNumerics, describe_physical_tip_source,
    generate_tip_boundary_emission, generate_tip_emission, tip_covariance, wavelength_m,
)


@pytest.fixture
def gun():
    result = FieldEmissionGun()
    result.emitter.coherence = TipCoherence()
    # Explicit 4 eV local tip mathematics fixture; no extraction or column claim.
    result.emitter.emission_energy_ev = 4.
    result.emitter.minimum_kinetic_energy_ev = 3.
    result.emitter.energy_spread_fwhm_ev = 0.
    return result


def test_tip_is_explicit_and_never_uses_exit_voltage(gun):
    numerics = TipWaveNumerics(energy_samples=1)
    emitted = generate_tip_emission(gun, numerics)
    gun.accelerator.high_tension_kv = 200.
    assert generate_tip_emission(gun, numerics).digest == emitted.digest
    mode, = tuple(emitted.modes())
    assert mode.energy_kev == pytest.approx(.004)
    assert emitted.record["plane_z_mm"] == 0.
    assert emitted.reference_current_a == gun.emitter.emitted_current_a
    gun.emitter.coherence = None
    with pytest.raises(ValueError, match="explicitly"):
        generate_tip_emission(gun, numerics)


def test_complex_tip_boundary_definition_does_not_admit_impossible_rays():
    from temsim.optics.electron_gun.tip_source_domain import (
        TipSourceDomainError, inspect_tip_source_domain, validate_tip_boundary,
    )
    source = FieldEmissionGun()
    source.emitter.coherence = TipCoherence()
    before = source.to_dict()
    boundary = validate_tip_boundary(source.emitter)
    report = inspect_tip_source_domain(source.emitter)
    assert boundary["physical_boundary_status"] == "DEFINED"
    assert report["support_radius_over_p"] > 1.
    with pytest.raises(TipSourceDomainError):
        generate_tip_emission(source)
    with pytest.raises(TipSourceDomainError):
        source.emit(33)
    emission = generate_tip_boundary_emission(source)
    assert emission.record["schema"] == "physical-tip-driven-gaussian-boundary-v1"
    assert emission.record["source_domain"] == report
    assert emission.record["injected_flux_status"] == "REQUIRES_BOUNDARY_SOLVER"
    assert source.to_dict() == before


def test_explicit_driven_boundary_keeps_every_mode_and_the_same_source(gun):
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=18.,
        curvature_x_m1=2e6, curvature_xy_m1=1e6, curvature_y_m1=-1e6,
        offset_x_nm=3., offset_y_nm=-2., tilt_x_mrad=4., tilt_y_mrad=-1.)
    numerics = TipWaveNumerics(grid_pixels=256, energy_samples=1,
                              mode_tail_tolerance=1e-9, maximum_modes=1024)
    before = gun.to_dict()
    paraxial = generate_tip_emission(gun, numerics)
    driven = generate_tip_boundary_emission(gun, numerics)
    assert driven.parameters == paraxial.parameters
    assert driven.record["energy_modes"] == paraxial.record["energy_modes"]
    assert driven.record["physical_source"] == describe_physical_tip_source(gun.emitter)
    for left, right in zip(paraxial.modes(), driven.modes(), strict=True):
        np.testing.assert_array_equal(left.plane.amplitude, right.plane.amplitude)
        np.testing.assert_array_equal(left.plane.basis_m, right.plane.basis_m)
        np.testing.assert_array_equal(left.plane.origin_m, right.plane.origin_m)
        np.testing.assert_array_equal(left.plane.curvature_m1, right.plane.curvature_m1)
        np.testing.assert_array_equal(left.plane.tilt_rad, right.plane.tilt_rad)
        assert left.energy_kev == right.energy_kev
        assert left.weight_per_reference_electron == right.weight_per_reference_electron
    assert gun.to_dict() == before


def test_physical_tip_record_exports_one_physical_width_and_phase(gun):
    gun.emitter.coherence = TipCoherence(curvature_xy_m1=12., offset_x_nm=3., tilt_y_mrad=.2)
    before = gun.to_dict()
    record = describe_physical_tip_source(gun.emitter)
    assert record["emission_fwhm_nm"] == gun.emitter.virtual_source_fwhm_nm
    assert "virtual_source_fwhm_nm" not in record
    assert "physical tip plane" in record["width_definition"]
    assert record["phase"]["curvature_xy_m1"] == 12.
    assert record["phase"]["offset_x_nm"] == 3.
    assert record["phase"]["tilt_y_mrad"] == .2
    assert record["current_na"] == pytest.approx(gun.emitter.emission_current_na)
    assert gun.to_dict() == before
    with pytest.raises(TypeError):
        record["emission_fwhm_nm"] = 100.


def test_driven_planar_boundary_cannot_replace_curved_tip_geometry(gun):
    gun.emitter.curvature_nm_inv = .001
    before = gun.to_dict()
    with pytest.raises(ValueError, match="Continuous curvature requires classical tip emission"):
        generate_tip_boundary_emission(gun)
    assert gun.to_dict() == before


def test_driven_boundary_does_not_invent_a_paraxial_bound_at_zero_energy_floor():
    source = FieldEmissionGun()
    source.emitter.coherence = TipCoherence()
    source.emitter.minimum_kinetic_energy_ev = 0.
    before = source.to_dict()
    emitted = generate_tip_boundary_emission(source)
    assert emitted.record["source_domain"]["status"] == "BOUND_UNAVAILABLE"
    assert all(row["energy_ev"] > 0. for row in emitted.record["energy_modes"])
    assert emitted.parameters["minimum_kinetic_energy_ev"] == 0.
    with pytest.raises(ValueError, match="finite and positive"):
        generate_tip_emission(source)
    assert source.to_dict() == before


@pytest.mark.parametrize("incoherent", [0., 18.])
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


def test_driven_boundary_particles_are_geometric_rays_of_the_same_tip_distribution():
    source = FieldEmissionGun()
    phase = TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL,
        incoherent_angle_rms_mrad=2., curvature_x_m1=2e6, curvature_xy_m1=1e6,
        curvature_y_m1=-1e6, offset_x_nm=3., offset_y_nm=-2.,
        tilt_x_mrad=4., tilt_y_mrad=-1.)
    source.emitter.coherence = phase
    bundle = source.emit(16384)
    slopes = np.array((bundle.tx_rad, bundle.ty_rad))
    canonical = slopes/np.sqrt(1+np.sum(slopes**2, axis=0))
    xy = np.array((bundle.x_m, bundle.y_m))
    centred = xy-np.array((3., -2.))[:, None]*1e-9
    residual = canonical-phase.curvature@centred-np.array((.004, -.001))[:, None]
    sigma = source.emitter.virtual_source_fwhm_nm*1e-9/math.sqrt(8*math.log(2))
    np.testing.assert_allclose(np.cov(centred, bias=True)/sigma**2, np.eye(2), atol=.003)
    # Independent moment check: no hbar/(2*sigma*p) term in geometric rays.
    np.testing.assert_allclose(np.cov(residual, bias=True)/(.002**2), np.eye(2), atol=.003)
    assert np.all(bundle.energy_offset_ev+source.emitter.emission_energy_ev > 0.)
    assert bundle.weight.sum() == pytest.approx(1.)
    np.testing.assert_allclose(xy.mean(axis=1), np.array((3., -2.))*1e-9, atol=1e-11)
    emission = generate_tip_emission(source)
    assert emission.record["schema"] == "physical-tip-driven-gaussian-boundary-v1"
    assert emission.parameters["coherence"]["boundary_model"] == DRIVEN_GAUSSIAN_SCHELL
    assert emission.record["physical_source"]["boundary_model"] == DRIVEN_GAUSSIAN_SCHELL
    assert "not an exact quantum Wigner" in emission.record["physical_source"]["particle_representation"]


def test_driven_particle_energy_samples_equal_wave_energy_samples_without_width_change():
    source = FieldEmissionGun()
    source.emitter.coherence = TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    bundle = source.emit(33)
    emission = generate_tip_emission(source, TipWaveNumerics(energy_samples=33))
    np.testing.assert_array_equal(
        np.array([row["energy_ev"] for row in emission.record["energy_modes"]]),
        bundle.energy_offset_ev+source.emitter.emission_energy_ev)
    assert np.all(bundle.tx_rad == 0.) and np.all(bundle.ty_rad == 0.)
    assert emission.parameters["virtual_source_fwhm_nm"] == 5.
    assert emission.parameters["emission_energy_ev"] == .3
    assert emission.parameters["energy_spread_fwhm_ev"] == .3
    assert emission.reference_current_a == source.emitted_current_a


@pytest.mark.parametrize("phase", [
    TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL, tilt_x_mrad=1000.),
    TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL, incoherent_angle_rms_mrad=200.),
    TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL, curvature_x_m1=1e9),
])
def test_driven_geometric_distribution_rejects_nonforward_phase_without_clipping(phase):
    source = FieldEmissionGun()
    source.emitter.coherence = phase
    before = source.to_dict()
    with pytest.raises(ValueError, match="geometric.*forward support"):
        source.emit(33)
    with pytest.raises(ValueError, match="geometric.*forward support"):
        generate_tip_emission(source)
    assert source.to_dict() == before


def test_historical_phase_record_stays_forward_and_explicit_driven_round_trips():
    source = FieldEmissionGun()
    source.emitter.coherence = TipCoherence()
    historical = source.to_dict()
    historical["components"][source.emitter.key]["coherence"].pop("boundary_model")
    old = field_emission_gun_from_dict(historical)
    assert old.emitter.coherence.boundary_model == FORWARD_GAUSSIAN_SCHELL
    with pytest.raises(ValueError, match="non-paraxial"):
        generate_tip_emission(old)
    source.emitter.coherence = TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    new = field_emission_gun_from_dict(source.to_dict())
    assert new.emitter.coherence.boundary_model == DRIVEN_GAUSSIAN_SCHELL
    assert generate_tip_emission(new).digest == generate_tip_emission(source).digest


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
    assert dialog.value()["coherence"] == TipCoherence(incoherent_angle_rms_mrad=2.5,
        boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    assert gun.to_dict() == before


def test_disabling_coherence_restores_the_selected_classical_emission(gun):
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
    assert apply_profile_values(target, values) is None
    assert target.electron_gun.emitter.coherence == state.electron_gun.emitter.coherence
    state.electron_gun.emitter.coherence = None
    save_profile(path, state, selection)
    _, values = read_profile(path)
    assert apply_profile_values(target, values) is None
    assert target.electron_gun.emitter.coherence is None


def test_medium_tuning_keeps_zero_current_guides_for_the_selected_tip_law(gun):
    gun.emitter.coherence = TipCoherence(offset_x_nm=3., tilt_y_mrad=4.)
    gun.emitter._tuning_boundary_probes = 33
    bundle = gun.emit(193)
    assert np.count_nonzero(bundle.weight == 0) == 33
    assert bundle.weight.sum() == pytest.approx(1.)
    assert bundle.x_m[-1] == pytest.approx(3e-9)
    assert bundle.ty_rad[-1] == pytest.approx(.004/np.sqrt(1-.004**2))
    assert np.max(abs(bundle.tx_rad[-33:])) > np.max(abs(bundle.tx_rad[:-33]))


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
