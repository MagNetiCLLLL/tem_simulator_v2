"""Tip editing/capture contracts, not full wave qualification."""
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import (
    DRIVEN_GAUSSIAN_SCHELL, FORWARD_GAUSSIAN_SCHELL,
    TipCoherence, TipWaveNumerics, generate_tip_emission, tip_covariance, wavelength_m,
)
from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence, SurfaceCoherence, load_tip_surface_reference
from temsim.physics.coherent_inputs import (
    TipEmissionSettings, candidate_tip_emission, idealised_diffraction_tip_settings,
    prepare_coherent_state, source_settings_from_state, wave_input_summary,
)
from temsim.physics.tip_wave_pipeline import TipWaveRequest


def _supported_settings(**updates):
    # Local emission fixture, not a realistic-source default or a claim about
    # the complete extraction/column chain.
    return replace(TipEmissionSettings(enabled=True, tip_fwhm_nm=5.,
        tip_mean_energy_ev=4., tip_minimum_energy_ev=3., tip_energy_spread_fwhm_ev=0.), **updates)


def _publish(state, settings=None):
    state.electron_gun = candidate_tip_emission(state, settings or _supported_settings())
    return state


def test_other_gun_family_is_rejected_without_feg_defaults():
    from temsim.optics.electron_gun.thermionic import ThermionicGun
    state = default_state()
    state.electron_gun = ThermionicGun()
    before = state.electron_gun.to_dict()
    with pytest.raises(ValueError, match="gun family"):
        source_settings_from_state(state)
    with pytest.raises(ValueError, match="gun family"):
        candidate_tip_emission(state, TipEmissionSettings(enabled=True))
    with pytest.raises(ValueError, match="gun family"):
        prepare_coherent_state(state)
    assert state.electron_gun.to_dict() == before


def test_historical_example_is_not_used_as_a_runtime_default():
    settings = idealised_diffraction_tip_settings()
    assert not settings.enabled
    assert settings.tip_fwhm_nm == 28390.10000542304
    assert settings.tip_mean_energy_ev == 30.
    assert settings.tip_energy_spread_fwhm_ev == 0.
    assert settings.validate() is settings
    with pytest.raises(FrozenInstanceError):
        settings.enabled = True
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    active = source_settings_from_state(state)
    assert active.tip_fwhm_nm == 5.
    assert active.tip_mean_energy_ev == .3
    assert active.tip_energy_spread_fwhm_ev == .3
    assert not active.enabled
    assert active.boundary_model == DRIVEN_GAUSSIAN_SCHELL
    assert capture_instrument_snapshot(state).digest == before


def test_editor_reads_published_scalars_and_phase_without_changing_them():
    state = default_state()
    emitter = state.electron_gun.emitter
    emitter.virtual_source_fwhm_nm = 75.
    emitter.emission_energy_ev = 2.
    emitter.energy_spread_fwhm_ev = .1
    emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=.2,
        curvature_x_m1=12., curvature_y_m1=-24., curvature_xy_m1=3.,
        offset_x_nm=.25, offset_y_nm=-.5, tilt_x_mrad=.001, tilt_y_mrad=-.002)
    before = capture_instrument_snapshot(state).digest
    assert source_settings_from_state(state) == TipEmissionSettings(enabled=True,
        boundary_model=FORWARD_GAUSSIAN_SCHELL,
        incoherent_angle_rms_mrad=.2, tip_fwhm_nm=75., tip_mean_energy_ev=2.,
        tip_minimum_energy_ev=.01, tip_energy_spread_fwhm_ev=.1,
        tip_curvature_x_m1=12., tip_curvature_y_m1=-24., tip_curvature_xy_m1=3.,
        tip_offset_x_nm=.25, tip_offset_y_nm=-.5, tip_tilt_x_mrad=.001, tip_tilt_y_mrad=-.002)
    assert capture_instrument_snapshot(state).digest == before


def test_wave_capture_never_applies_a_private_source_draft():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    for draft in (None, _supported_settings()):
        with pytest.raises(ValueError, match="Apply tip parameters"):
            prepare_coherent_state(state, draft)
    assert state.electron_gun.emitter.coherence is None
    assert capture_instrument_snapshot(state).digest == before
    _publish(state)
    published = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Apply tip parameters changes.*tip_mean_energy_ev"):
        prepare_coherent_state(state, TipEmissionSettings(enabled=True, tip_mean_energy_ev=5.))
    assert capture_instrument_snapshot(state).digest == published


def test_candidate_is_transactional_and_capture_preserves_every_component():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    candidate = candidate_tip_emission(state, _supported_settings(
        tip_curvature_xy_m1=-3., tip_offset_x_nm=.125, tip_tilt_y_mrad=.005))
    assert candidate is not state.electron_gun
    assert candidate.emitter is not state.electron_gun.emitter
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.coherence is None
    state.electron_gun = candidate
    working = prepare_coherent_state(state, source_settings_from_state(state))
    assert capture_instrument_snapshot(working).digest == capture_instrument_snapshot(state).digest
    assert working.electron_gun is not candidate
    assert working.electron_gun.emitter is not candidate.emitter
    for name in ("extractor", "accelerator", "electrostatic_lens", "c1_aperture"):
        assert asdict(getattr(working.electron_gun, name)) == asdict(getattr(candidate, name))
    for name in ("lenses", "apertures", "stigmators", "deflectors", "recording_planes", "sample"):
        assert getattr(working, name) == getattr(state, name)
        assert getattr(working, name) is not getattr(state, name)


def test_partial_edit_preserves_unspecified_phase_and_angular_spread():
    state = _publish(default_state(), _supported_settings(incoherent_angle_rms_mrad=2.,
        tip_curvature_x_m1=1e4, tip_curvature_xy_m1=20., tip_offset_y_nm=1., tip_tilt_x_mrad=.1))
    previous = state.electron_gun.emitter.coherence
    candidate = candidate_tip_emission(state,
        TipEmissionSettings(enabled=True, tip_mean_energy_ev=5.))
    assert candidate.emitter.coherence == previous
    assert candidate.emitter.emission_energy_ev == 5.
    assert state.electron_gun.emitter.emission_energy_ev == 4.
    assert prepare_coherent_state(state, TipEmissionSettings(enabled=True)).electron_gun.emitter.coherence == previous


def test_display_rounding_does_not_change_captured_bits_or_cache_identity():
    state = _publish(default_state(), _supported_settings(tip_offset_x_nm=.06696090871824144))
    actual = source_settings_from_state(state)
    captured = prepare_coherent_state(state,
        replace(actual, tip_offset_x_nm=round(actual.tip_offset_x_nm, 12)))
    assert captured.electron_gun.emitter.coherence.offset_x_nm == actual.tip_offset_x_nm
    assert capture_instrument_snapshot(captured).digest == capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Apply tip parameters changes"):
        prepare_coherent_state(state, replace(actual, tip_offset_x_nm=actual.tip_offset_x_nm+1e-6))


def test_explicit_disabling_keeps_scalar_edits_and_physical_components():
    state = _publish(default_state())
    candidate = candidate_tip_emission(state, TipEmissionSettings(enabled=False,
        tip_fwhm_nm=7., tip_mean_energy_ev=.5, tip_minimum_energy_ev=.01,
        tip_energy_spread_fwhm_ev=.3))
    assert candidate.emitter.coherence is None
    assert candidate.emitter.virtual_source_fwhm_nm == 7.
    assert candidate.emitter.emission_energy_ev == .5
    assert candidate.emitter.energy_spread_fwhm_ev == .3
    assert state.electron_gun.emitter.coherence is not None
    assert asdict(candidate.accelerator) == asdict(state.electron_gun.accelerator)
    assert len(candidate.emit(9).x_m) == 9


@pytest.mark.parametrize("settings", [
    TipEmissionSettings(enabled=True, tip_fwhm_nm=0.),
    TipEmissionSettings(enabled=True, tip_mean_energy_ev=-1.),
    TipEmissionSettings(enabled=True, tip_energy_spread_fwhm_ev=-1.),
    TipEmissionSettings(enabled=True, tip_curvature_x_m1=float("nan")),
    TipEmissionSettings(enabled=True, tip_tilt_y_mrad=True),
    TipEmissionSettings(enabled=False, tip_mean_energy_ev=.01, tip_minimum_energy_ev=.3),
])
def test_invalid_candidate_keeps_current_gun_and_snapshot(settings):
    state = default_state()
    original = state.electron_gun
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError):
        candidate_tip_emission(state, settings)
    assert state.electron_gun is original
    assert capture_instrument_snapshot(state).digest == before


def test_unsupported_candidate_cannot_break_working_particle_source():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Tip unchanged.*cannot represent"):
        candidate_tip_emission(state, replace(source_settings_from_state(state), enabled=True,
            boundary_model=FORWARD_GAUSSIAN_SCHELL))
    assert capture_instrument_snapshot(state).digest == before
    assert len(state.electron_gun.emit(9).x_m) == 9


def test_default_tip_can_explicitly_select_driven_boundary_without_changing_physical_inputs(monkeypatch):
    from temsim.optics.electron_gun.tip_coherence import TipEmission
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    draft = replace(source_settings_from_state(state), enabled=True)
    candidate = candidate_tip_emission(state, draft)
    assert capture_instrument_snapshot(state).digest == before
    for name in ("virtual_source_fwhm_nm", "emission_energy_ev", "minimum_kinetic_energy_ev",
                 "energy_spread_fwhm_ev", "emission_current_na", "tip_radius_nm", "curvature_nm_inv"):
        assert getattr(candidate.emitter, name) == getattr(state.electron_gun.emitter, name)
    assert candidate.emitter.coherence.boundary_model == DRIVEN_GAUSSIAN_SCHELL
    state.electron_gun = candidate
    assert source_settings_from_state(state) == draft
    captured = prepare_coherent_state(state, draft)
    assert capture_instrument_snapshot(captured).digest == capture_instrument_snapshot(state).digest
    monkeypatch.setattr(TipEmission, "modes", lambda self: pytest.fail("Preflight allocated wave fields"))
    report = wave_input_summary(captured, TipWaveRequest())
    assert report["status"] == "SOURCE_READY"
    assert report["mode_count"] == TipWaveRequest().source.energy_samples
    assert report["source_domain"]["support_radius_over_p"] > 1.
    assert report["particle_domain"]["support_radius_over_p"] == 0.
    assert "Non-paraxial" in report["transport_requirement"]
    assert "not an exact quantum Wigner" in report["particle_representation"]
    assert np.all(captured.electron_gun.emit(33).tx_rad == 0.)


def test_boundary_model_switch_is_explicit_and_invalidates_capture_identity():
    state = _publish(default_state())
    previous = capture_instrument_snapshot(state)
    active = source_settings_from_state(state)
    assert active.boundary_model == FORWARD_GAUSSIAN_SCHELL
    draft = replace(active, boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    with pytest.raises(ValueError, match="Apply tip parameters changes.*boundary_model"):
        prepare_coherent_state(state, draft)
    state.electron_gun = candidate_tip_emission(state, draft)
    assert capture_instrument_snapshot(state).digest != previous.digest
    assert source_settings_from_state(previous.restore()).boundary_model == FORWARD_GAUSSIAN_SCHELL
    assert source_settings_from_state(state).boundary_model == DRIVEN_GAUSSIAN_SCHELL
    partial = candidate_tip_emission(state, TipEmissionSettings(enabled=True, tip_mean_energy_ev=5.))
    assert partial.emitter.coherence.boundary_model == DRIVEN_GAUSSIAN_SCHELL


@pytest.mark.parametrize("boundary", ["", "automatic", 1, True])
def test_invalid_boundary_model_is_rejected(boundary):
    with pytest.raises(ValueError, match="boundary model"):
        TipEmissionSettings(boundary_model=boundary).validate()


def test_numerical_product_quadrature_is_not_silently_removed():
    from temsim.optics.electron_gun.emitter import EmissionQuadrature
    state = default_state()
    quadrature = EmissionQuadrature(3, 3, 3)
    state.electron_gun.emitter.quadrature = quadrature
    state.electron_gun.emitter.ray_count = quadrature.total
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Product quadrature"):
        candidate_tip_emission(state, _supported_settings())
    assert state.electron_gun.emitter.quadrature == quadrature
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("field", ["tip_offset_x_nm", "tip_offset_y_nm", "tip_curvature_xy_m1"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_and_boolean_phase_inputs_are_rejected(field, value):
    with pytest.raises(ValueError, match="must be finite"):
        TipEmissionSettings(enabled=True, **{field: value}).validate()


def test_gaussian_controls_cannot_replace_surface_geometry():
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(),
        coherence=SurfaceCoherence(mean_energy_ev=.3, energy_rms_ev=0.))
    with pytest.raises(ValueError, match="Gaussian tip edits"):
        candidate_tip_emission(state, TipEmissionSettings(enabled=True, tip_fwhm_nm=50.))


def test_curved_tip_cannot_be_replaced_with_flat_wave():
    state = default_state()
    state.electron_gun.emitter.curvature_nm_inv = .01
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="curved tip"):
        candidate_tip_emission(state, TipEmissionSettings(enabled=True))
    assert capture_instrument_snapshot(state).digest == before
    state.electron_gun.emitter.coherence = TipCoherence()
    with pytest.raises(ValueError, match="curved tip"):
        prepare_coherent_state(state)
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_UNSUPPORTED" and "curved tip" in report["reason"]


@pytest.mark.parametrize("rms_ev", [.2, 0.])
def test_explicit_surface_apply_shares_energy_without_replacing_tip_geometry(rms_ev):
    state = default_state()
    surface = load_tip_surface_reference()
    state.electron_gun.emitter.surface_model = surface
    before = capture_instrument_snapshot(state).digest
    actual = source_settings_from_state(state)
    assert not actual.enabled
    assert actual.surface_mean_energy_ev == pytest.approx(.3)
    assert actual.surface_energy_rms_ev == pytest.approx(np.hypot(.2, .1))
    settings = TipEmissionSettings(enabled=True, surface_mean_energy_ev=.6,
        surface_energy_rms_ev=rms_ev, surface_edge_phase_rad=0.)
    candidate = candidate_tip_emission(state, settings)
    shared = candidate.emitter.surface_model
    assert isinstance(shared.coherence, SharedSurfaceCoherence)
    assert shared.geometry == surface.geometry
    assert shared.field_numerics == surface.field_numerics
    assert shared.emission.cap_half_angle_deg == surface.emission.cap_half_angle_deg
    assert shared.current_na == surface.current_na
    assert shared.mean_energy_ev == .6 and shared.energy_sigma_ev == rms_ev
    assert set(asdict(shared.coherence)) == {"model", "edge_phase_rad"}
    assert shared.emission.energy_distribution == ("gamma" if rms_ev else "monoenergetic")
    assert shared.emission.flux_profile == "cosine_cap"
    assert shared.emission.maximum_angle_deg == 0.
    for name in ("extractor", "accelerator", "electrostatic_lens", "c1_aperture"):
        assert asdict(getattr(candidate, name)) == asdict(getattr(state.electron_gun, name))
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.surface_model is surface
    # Publishing is explicit; capture only copies the one applied energy law.
    state.electron_gun = candidate
    assert source_settings_from_state(state) == settings
    applied = capture_instrument_snapshot(state)
    working = prepare_coherent_state(state, TipEmissionSettings(enabled=True))
    assert capture_instrument_snapshot(working).digest == applied.digest
    assert working.electron_gun.emitter.surface_model.to_dict() == shared.to_dict()
    report = wave_input_summary(working, TipWaveRequest())
    assert report["status"] == "SOURCE_READY"
    assert report["inputs"]["emission"]["kinetic_mean_ev"] == .6
    assert report["inputs"]["emission"]["kinetic_sigma_ev"] == rms_ev
    assert "Geometric-optics" in report["particle_representation"]
    particles = candidate.emit(9)
    assert np.all(particles.surface_energy_ev > 0.)
    np.testing.assert_allclose(particles.energy_offset_ev + shared.mean_energy_ev,
                               particles.surface_energy_ev, atol=1e-16)
    if rms_ev == 0.:
        energies, _ = shared.energy_quadrature(1)
        np.testing.assert_array_equal(particles.surface_energy_ev, np.full(9, energies[0]))


def test_shared_surface_rejects_unimplemented_phase_gradient_atomically():
    state = default_state()
    state.electron_gun.emitter.surface_model = load_tip_surface_reference()
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="only constant surface phase"):
        candidate_tip_emission(state, TipEmissionSettings(enabled=True,
            surface_mean_energy_ev=.3, surface_energy_rms_ev=.1, surface_edge_phase_rad=.2))
    assert capture_instrument_snapshot(state).digest == before


def test_existing_surface_wave_remains_readable_without_a_false_particle_claim():
    state = default_state()
    phase = SurfaceCoherence(mean_energy_ev=.6, energy_rms_ev=.2, edge_phase_rad=.5)
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=phase)
    actual = source_settings_from_state(state)
    assert actual == TipEmissionSettings(enabled=True, surface_mean_energy_ev=.6,
        surface_energy_rms_ev=.2, surface_edge_phase_rad=.5)
    working = prepare_coherent_state(state, replace(actual, surface_edge_phase_rad=None))
    assert working.electron_gun.emitter.surface_model.coherence == phase
    with pytest.raises(ValueError, match="Apply tip parameters changes"):
        prepare_coherent_state(state, replace(actual, surface_energy_rms_ev=.15))
    before = capture_instrument_snapshot(state).digest
    unchanged = candidate_tip_emission(state, actual)
    assert unchanged.emitter.surface_model.coherence == phase
    with pytest.raises(ValueError, match="read-only.*explicitly replace"):
        candidate_tip_emission(state, replace(actual, surface_energy_rms_ev=.15))
    with pytest.raises(ValueError, match="read-only"):
        candidate_tip_emission(state, replace(actual, enabled=False))
    with pytest.raises(ValueError, match="propagated as a wave"):
        state.electron_gun.emit(9)
    assert state.electron_gun.emitter.surface_model.coherence == phase
    assert capture_instrument_snapshot(state).digest == before
    assert wave_input_summary(working, TipWaveRequest())["particle_representation"].startswith("UNAVAILABLE")


@pytest.mark.parametrize("settings", [
    TipEmissionSettings(enabled=1), TipEmissionSettings(incoherent_angle_rms_mrad=-1),
    TipEmissionSettings(incoherent_angle_rms_mrad=True),
    TipEmissionSettings(surface_mean_energy_ev=0), TipEmissionSettings(surface_energy_rms_ev=-1),
    TipEmissionSettings(surface_edge_phase_rad=float("nan")),
    TipEmissionSettings(surface_mean_energy_ev=float("inf")),
])
def test_invalid_physical_inputs_are_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_existing_low_energy_source_remains_readable_and_unsupported_without_rewrite(monkeypatch):
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence()
    state = prepare_coherent_state(state)
    before = capture_instrument_snapshot(state).digest
    from temsim.optics.electron_gun.tip_coherence import TipEmission
    monkeypatch.setattr(TipEmission, "modes", lambda self: pytest.fail("Preflight constructed wave modes"))
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_UNSUPPORTED"
    assert "paraxial" in report["reason"] or "non-paraxial" in report["reason"]
    assert report["source_domain"]["minimum_evaluated_energy_ev"] > 0
    assert report["mode_count"] is None
    assert capture_instrument_snapshot(state).digest == before


def test_preflight_counts_modes_without_allocating_modes_or_fields(monkeypatch):
    import temsim.physics.instrument_electric as electric
    import temsim.physics.instrument_magnetic as magnetic
    from temsim.optics.electron_gun.tip_coherence import TipEmission
    state = prepare_coherent_state(_publish(default_state()))
    before = capture_instrument_snapshot(state).digest

    def unexpected(*_args, **_kwargs):
        pytest.fail("Preflight constructed wave arrays or propagation fields")

    monkeypatch.setattr(TipEmission, "modes", unexpected)
    monkeypatch.setattr(electric, "capture_instrument_electric_field", unexpected)
    monkeypatch.setattr(magnetic, "capture_instrument_magnetic_field", unexpected)
    report = wave_input_summary(state, TipWaveRequest(source=TipWaveNumerics(energy_samples=1)))
    assert report["status"] == "SOURCE_READY" and report["mode_count"] == 1
    assert report["minimum_initial_wave_bytes"] == 128**2*16
    assert "not wave execution" in report["scope"]
    assert report["source_plane"] == "physical tip before extraction"
    assert report["inputs"]["emission_energy_ev"] == 4.
    assert capture_instrument_snapshot(state).digest == before
    with pytest.raises(TypeError):
        report["status"] = "QUALIFIED"


def test_surface_energy_samples_are_explicit_and_not_dropped():
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    request = TipWaveRequest()
    invalid = replace(request, surface=replace(request.surface, energy_samples=1))
    report = wave_input_summary(state, invalid)
    assert report["status"] == "SOURCE_UNSUPPORTED" and "two energy samples" in report["reason"]
    report = wave_input_summary(state, request)
    assert report["status"] == "SOURCE_READY" and report["mode_count"] == request.surface.energy_samples


@pytest.mark.parametrize("surface", [False, True])
def test_unconfigured_source_is_not_propagation(surface):
    state = default_state()
    if surface:
        state.electron_gun.emitter.surface_model = load_tip_surface_reference()
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_NOT_CONFIGURED" and report["mode_count"] is None
    assert report["minimum_initial_wave_bytes"] is None


def test_published_edits_change_both_energy_laws_and_source_identity_and_survive_snapshot():
    state = _publish(default_state())
    first = generate_tip_emission(state.electron_gun, TipWaveNumerics(energy_samples=1))
    initial = capture_instrument_snapshot(state)
    _publish(state, _supported_settings(tip_fwhm_nm=8., tip_mean_energy_ev=6.,
        tip_minimum_energy_ev=4., tip_energy_spread_fwhm_ev=.3,
        tip_curvature_x_m1=1e6, tip_curvature_xy_m1=-.5e6,
        tip_offset_x_nm=1., tip_offset_y_nm=-2., tip_tilt_y_mrad=.5))
    source = generate_tip_emission(prepare_coherent_state(state).electron_gun,
        TipWaveNumerics(energy_samples=33))
    assert source.digest != first.digest
    assert capture_instrument_snapshot(state).digest != initial.digest
    particles = state.electron_gun.emit(33)
    energies = np.array([row["energy_ev"] for row in source.record["energy_modes"]])
    np.testing.assert_array_equal(energies, particles.energy_offset_ev+6.)
    assert energies.mean() == pytest.approx(6., abs=1e-10)
    assert energies.std() == pytest.approx(.3/2.354820045, abs=1e-10)
    assert source.reference_current_a == state.electron_gun.emitted_current_a
    restored = capture_instrument_snapshot(state).restore()
    assert source_settings_from_state(restored) == source_settings_from_state(state)
    replay = restored.electron_gun.emit(33)
    for field in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight"):
        np.testing.assert_array_equal(getattr(replay, field), getattr(particles, field))


def test_published_phase_has_the_same_quantum_covariance_in_particles_and_modes():
    state = _publish(default_state(), _supported_settings(incoherent_angle_rms_mrad=2.,
        tip_curvature_x_m1=2e6, tip_curvature_xy_m1=1e6, tip_curvature_y_m1=-1e6,
        tip_offset_x_nm=3., tip_tilt_y_mrad=4.))
    source = generate_tip_emission(prepare_coherent_state(state).electron_gun,
        TipWaveNumerics(grid_pixels=256, energy_samples=1, mode_tail_tolerance=1e-9))
    waves = sum(mode.weight_per_reference_electron*mode.plane.canonical_covariance(float(wavelength_m(4.)))
                for mode in source.modes())
    expected = tip_covariance(state.electron_gun.emitter, 4.)
    scale = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(waves/np.outer(scale, scale), expected/np.outer(scale, scale), atol=2e-7)
    bundle = state.electron_gun.emit(8192)
    slopes = np.array((bundle.tx_rad, bundle.ty_rad))
    canonical = slopes/np.sqrt(1+np.sum(slopes**2, axis=0))
    samples = np.vstack((bundle.x_m, bundle.y_m, canonical))
    # Finite deterministic sampling is tested separately from mode quadrature;
    # dimensionless normalization prevents small SI units from hiding errors.
    np.testing.assert_allclose(np.cov(samples, bias=True)/np.outer(scale, scale),
        expected/np.outer(scale, scale), atol=.015)
    assert samples[0].mean() == pytest.approx(3e-9, abs=1e-11)
    assert samples[3].mean() == pytest.approx(.004, abs=1e-4)
