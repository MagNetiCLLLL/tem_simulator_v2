"""Tip-only configuration/preflight contracts, not full wave qualification."""
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.physics.coherent_inputs import (
    CoherentSourceSettings, default_gaussian_tip_settings, prepare_coherent_state, wave_input_summary,
)
from temsim.physics.tip_wave_pipeline import TipWaveRequest


def test_gaussian_default_records_prescribed_tip_inputs_without_implicit_opt_in():
    settings = default_gaussian_tip_settings()
    # Independent literals from the retained input design, not outputs from a
    # propagation or an assertion that the resulting diffraction is qualified.
    assert asdict(settings) == {
        "enabled": False,
        "incoherent_angle_rms_mrad": 0.,
        "surface_mean_energy_ev": None,
        "surface_energy_rms_ev": None,
        "surface_edge_phase_rad": None,
        "tip_fwhm_nm": 28390.10000542304,
        "tip_mean_energy_ev": 30.,
        "tip_minimum_energy_ev": .01,
        "tip_energy_spread_fwhm_ev": 0.,
        "tip_curvature_x_m1": 624.3690658100246,
        "tip_curvature_xy_m1": -9.348776164576383e-12,
        "tip_curvature_y_m1": 624.369065810008,
        "tip_offset_x_nm": -.06696090871824144,
        "tip_offset_y_nm": -.07266953387249751,
        "tip_tilt_x_mrad": -4.18060377176702e-5,
        "tip_tilt_y_mrad": -4.537013038384244e-5,
    }
    assert settings.validate() is settings
    with pytest.raises(FrozenInstanceError):
        settings.enabled = True
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Select a coherent tip"):
        prepare_coherent_state(state, settings)
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.coherence is None


def test_opted_in_gaussian_default_preflight_preserves_particles_and_upstream_components(monkeypatch):
    import temsim.physics.instrument_electric as electric
    import temsim.physics.instrument_magnetic as magnetic
    from temsim.optics.electron_gun.tip_coherence import TipEmission

    state = default_state()
    before = capture_instrument_snapshot(state).digest
    working = prepare_coherent_state(state,
        replace(default_gaussian_tip_settings(), enabled=True))
    emitter = working.electron_gun.emitter
    assert emitter.virtual_source_fwhm_nm == 28390.10000542304
    assert emitter.emission_energy_ev == 30.
    assert emitter.minimum_kinetic_energy_ev == .01
    assert emitter.energy_spread_fwhm_ev == 0.
    assert emitter.coherence == TipCoherence(
        curvature_x_m1=624.3690658100246, curvature_xy_m1=-9.348776164576383e-12,
        curvature_y_m1=624.369065810008,
        offset_x_nm=-.06696090871824144, offset_y_nm=-.07266953387249751,
        tilt_x_mrad=-4.18060377176702e-5, tilt_y_mrad=-4.537013038384244e-5,
        incoherent_angle_rms_mrad=0.)
    assert asdict(working.electron_gun.accelerator) == asdict(state.electron_gun.accelerator)
    assert asdict(working.electron_gun.extractor) == asdict(state.electron_gun.extractor)
    assert asdict(working.electron_gun.electrostatic_lens) == asdict(state.electron_gun.electrostatic_lens)
    for name in ("lenses", "apertures", "stigmators", "deflectors", "recording_planes", "sample"):
        assert getattr(working, name) == getattr(state, name)
        assert getattr(working, name) is not getattr(state, name)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Source preflight constructed wave arrays or captured propagation fields")

    monkeypatch.setattr(TipEmission, "modes", unexpected)
    monkeypatch.setattr(electric, "capture_instrument_electric_field", unexpected)
    monkeypatch.setattr(magnetic, "capture_instrument_magnetic_field", unexpected)
    summary = wave_input_summary(working,
        TipWaveRequest(source=TipWaveNumerics(energy_samples=1)))
    assert summary["status"] == "SOURCE_READY"
    assert summary["mode_count"] == 1
    assert summary["source_plane"] == "physical tip before extraction"
    assert summary["inputs"]["virtual_source_fwhm_nm"] == 28390.10000542304
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.coherence is None


def test_gaussian_default_factory_does_not_rewrite_explicit_source_settings():
    state = default_state()
    state.electron_gun.emitter.virtual_source_fwhm_nm = 75.
    state.electron_gun.emitter.emission_energy_ev = 2.
    state.electron_gun.emitter.energy_spread_fwhm_ev = .1
    previous = TipCoherence(curvature_x_m1=12., curvature_y_m1=-24.,
        curvature_xy_m1=3., offset_x_nm=.25, offset_y_nm=-.5,
        tilt_x_mrad=.001, tilt_y_mrad=-.002)
    state.electron_gun.emitter.coherence = previous
    before = capture_instrument_snapshot(state).digest
    default_gaussian_tip_settings()
    explicit = CoherentSourceSettings(enabled=True, tip_mean_energy_ev=5.)
    working = prepare_coherent_state(state, explicit)
    assert working.electron_gun.emitter.emission_energy_ev == 5.
    assert working.electron_gun.emitter.virtual_source_fwhm_nm == 75.
    assert working.electron_gun.emitter.energy_spread_fwhm_ev == .1
    assert working.electron_gun.emitter.coherence == previous
    assert explicit == CoherentSourceSettings(enabled=True, tip_mean_energy_ev=5.)
    assert capture_instrument_snapshot(state).digest == before


def test_selection_is_explicit_and_does_not_change_particle_instrument():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="Select a coherent tip"):
        prepare_coherent_state(state)
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True))
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.coherence is None
    assert working.electron_gun.emitter.coherence == TipCoherence()
    assert working.electron_gun is not state.electron_gun
    assert working.sample is not state.sample
    assert working.lenses[0] is not state.lenses[0]
    for name in ("virtual_source_fwhm_nm", "emission_energy_ev", "emission_current_na",
                 "energy_spread_fwhm_ev", "minimum_kinetic_energy_ev"):
        assert getattr(working.electron_gun.emitter, name) == getattr(state.electron_gun.emitter, name)
    assert asdict(working.electron_gun.accelerator) == asdict(state.electron_gun.accelerator)


def test_existing_tip_phase_and_tilt_are_retained():
    state = default_state()
    previous = TipCoherence(curvature_x_m1=1e4, curvature_xy_m1=20., offset_y_nm=1., tilt_x_mrad=.1)
    state.electron_gun.emitter.coherence = previous
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True, incoherent_angle_rms_mrad=2.))
    assert working.electron_gun.emitter.coherence == replace(previous, incoherent_angle_rms_mrad=2.)
    assert state.electron_gun.emitter.coherence == previous


def test_explicit_phase_and_centre_controls_replace_only_detached_tip_boundary():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    expected = TipCoherence(incoherent_angle_rms_mrad=.02,
        curvature_x_m1=12., curvature_xy_m1=-3., curvature_y_m1=24.,
        offset_x_nm=.125, offset_y_nm=-.375, tilt_x_mrad=.005, tilt_y_mrad=-.01)
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True,
        incoherent_angle_rms_mrad=expected.incoherent_angle_rms_mrad,
        tip_curvature_x_m1=expected.curvature_x_m1,
        tip_curvature_xy_m1=expected.curvature_xy_m1,
        tip_curvature_y_m1=expected.curvature_y_m1,
        tip_offset_x_nm=expected.offset_x_nm, tip_offset_y_nm=expected.offset_y_nm,
        tip_tilt_x_mrad=expected.tilt_x_mrad, tip_tilt_y_mrad=expected.tilt_y_mrad))
    assert working.electron_gun.emitter.coherence == expected
    assert state.electron_gun.emitter.coherence is None
    assert capture_instrument_snapshot(state).digest == before


def test_explicit_tip_edits_admit_forward_source_without_changing_live_instrument():
    state = default_state()
    original = capture_instrument_snapshot(state)
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True,
        tip_fwhm_nm=50., tip_mean_energy_ev=.3, tip_minimum_energy_ev=.01,
        tip_energy_spread_fwhm_ev=0., tip_curvature_x_m1=10., tip_curvature_y_m1=-20.,
        tip_tilt_x_mrad=.001))
    assert capture_instrument_snapshot(state).digest == original.digest
    assert working.electron_gun.emitter.virtual_source_fwhm_nm == 50.
    assert working.electron_gun.emitter.coherence.curvature_x_m1 == 10.
    assert asdict(working.electron_gun.accelerator) == asdict(state.electron_gun.accelerator)
    request = replace(TipWaveRequest(), source=TipWaveNumerics(energy_samples=1))
    summary = wave_input_summary(working, request)
    assert summary["status"] == "SOURCE_READY"
    assert summary["mode_count"] == 1
    assert summary["source_plane"] == "physical tip before extraction"
    assert capture_instrument_snapshot(working).digest != original.digest


@pytest.mark.parametrize("settings", [
    CoherentSourceSettings(enabled=True, tip_fwhm_nm=0.),
    CoherentSourceSettings(enabled=True, tip_mean_energy_ev=-1.),
    CoherentSourceSettings(enabled=True, tip_energy_spread_fwhm_ev=-1.),
    CoherentSourceSettings(enabled=True, tip_curvature_x_m1=float("nan")),
    CoherentSourceSettings(enabled=True, tip_tilt_y_mrad=True),
])
def test_tip_edits_require_finite_physical_inputs(settings):
    with pytest.raises(ValueError):
        prepare_coherent_state(default_state(), settings)


@pytest.mark.parametrize("field", ["tip_offset_x_nm", "tip_offset_y_nm", "tip_curvature_xy_m1"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_explicit_phase_and_centre_inputs_reject_nonfinite_or_boolean_values(field, value):
    with pytest.raises(ValueError, match="must be finite"):
        CoherentSourceSettings(enabled=True, **{field: value}).validate()


def test_gaussian_controls_cannot_replace_surface_geometry():
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(),
        coherence=SurfaceCoherence(mean_energy_ev=.3, energy_rms_ev=0.))
    with pytest.raises(ValueError, match="Gaussian tip edits"):
        prepare_coherent_state(state, CoherentSourceSettings(enabled=True, tip_fwhm_nm=50.))


def test_continuous_curved_tip_cannot_be_replaced_with_flat_wave():
    state = default_state()
    state.electron_gun.emitter.curvature_nm_inv = .01
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="curved tip"):
        prepare_coherent_state(state, CoherentSourceSettings(enabled=True))
    assert capture_instrument_snapshot(state).digest == before
    state.electron_gun.emitter.coherence = TipCoherence()
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_UNSUPPORTED" and "curved tip" in report["reason"]


def test_surface_classical_probabilities_do_not_define_complex_reservoir():
    state = default_state()
    surface = load_tip_surface_reference()
    state.electron_gun.emitter.surface_model = surface
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match="explicit tip mean energy and energy RMS"):
        prepare_coherent_state(state, CoherentSourceSettings(enabled=True))
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True,
        surface_mean_energy_ev=.3, surface_energy_rms_ev=.1, surface_edge_phase_rad=.2))
    assert working.electron_gun.emitter.surface_model.coherence == SurfaceCoherence(.3, .1, .2)
    assert working.electron_gun.emitter.surface_model.geometry == surface.geometry
    assert working.electron_gun.emitter.surface_model.emission == surface.emission
    assert capture_instrument_snapshot(state).digest == before


def test_existing_surface_boundary_is_preserved_unless_explicitly_edited():
    state = default_state()
    source = SurfaceCoherence(mean_energy_ev=.6, energy_rms_ev=.2, edge_phase_rad=.5)
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=source)
    working = prepare_coherent_state(state, CoherentSourceSettings(enabled=True))
    assert working.electron_gun.emitter.surface_model.coherence == source
    edited = prepare_coherent_state(state, CoherentSourceSettings(enabled=True, surface_energy_rms_ev=.15))
    assert edited.electron_gun.emitter.surface_model.coherence == replace(source, energy_rms_ev=.15)


@pytest.mark.parametrize("settings", [
    CoherentSourceSettings(enabled=1), CoherentSourceSettings(incoherent_angle_rms_mrad=-1),
    CoherentSourceSettings(incoherent_angle_rms_mrad=True),
    CoherentSourceSettings(surface_mean_energy_ev=0), CoherentSourceSettings(surface_energy_rms_ev=-1),
    CoherentSourceSettings(surface_edge_phase_rad=float("nan")),
    CoherentSourceSettings(surface_mean_energy_ev=float("inf")),
])
def test_invalid_physical_inputs_are_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_default_low_energy_tip_is_reported_unsupported_without_being_changed(monkeypatch):
    state = prepare_coherent_state(default_state(), CoherentSourceSettings(enabled=True))
    before = capture_instrument_snapshot(state).digest
    # This preflight must neither solve fields nor construct even one wave array.
    from temsim.optics.electron_gun.tip_coherence import TipEmission
    monkeypatch.setattr(TipEmission, "modes", lambda self: pytest.fail("Preflight constructed wave modes"))
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_UNSUPPORTED"
    assert "paraxial" in report["reason"] or "non-paraxial" in report["reason"]
    assert report["source_domain"]["minimum_evaluated_energy_ev"] > 0
    assert report["mode_count"] is None
    assert capture_instrument_snapshot(state).digest == before


def test_source_summary_counts_modes_without_allocating_them(monkeypatch):
    state = default_state()
    # An explicitly wider TIP is a source-domain fixture, not a changed default.
    state.electron_gun.emitter.virtual_source_fwhm_nm = 50.
    state.electron_gun.emitter.energy_spread_fwhm_ev = 0.
    state = prepare_coherent_state(state, CoherentSourceSettings(enabled=True))
    from temsim.optics.electron_gun.tip_coherence import TipEmission
    monkeypatch.setattr(TipEmission, "modes", lambda self: pytest.fail("Preflight constructed wave modes"))
    report = wave_input_summary(state, TipWaveRequest(source=TipWaveNumerics(energy_samples=1)))
    assert report["status"] == "SOURCE_READY" and report["mode_count"] == 1
    assert report["minimum_initial_wave_bytes"] == 128**2*16
    assert "not wave execution" in report["scope"]
    assert report["inputs"]["emission_energy_ev"] == .3
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
def test_unconfigured_source_cannot_be_mistaken_for_propagation(surface):
    state = default_state()
    if surface:
        state.electron_gun.emitter.surface_model = load_tip_surface_reference()
    report = wave_input_summary(state, TipWaveRequest())
    assert report["status"] == "SOURCE_NOT_CONFIGURED" and report["mode_count"] is None
    assert report["minimum_initial_wave_bytes"] is None
