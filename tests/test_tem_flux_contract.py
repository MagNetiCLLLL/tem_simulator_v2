"""Manufactured CPU benchmarks through the production TEM entry point.

Only the input specimen/illumination and analytically known column map are
fixtures. FFT transfer, aperture ownership, camera deposition and ensemble
reduction are production code. These are not independent material validation.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import wave_imaging as imaging


@pytest.fixture
def tem_benchmark(monkeypatch):
    from temsim.optics import direct_alignment
    from temsim.physics import camera_wave, multiplane_wave
    from temsim.physics.core import electron

    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "TEM"
    state.projector_mode = "image"
    state.sample.z_mm = 0.
    state.sample.wave_multislice_enabled = False
    for aperture in state.apertures:
        aperture.enabled = False
    for detector in state.stem_detectors:
        detector.inserted = False
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    state.camera.z_mm = 200.
    aperture = state.objective_aperture
    aperture.enabled = aperture.installed = aperture.inserted = True
    aperture.z_mm = 100.
    aperture.radius_mm = .003
    state.objective_lens.percent = 0.

    n = 32
    axis = np.arange(n, dtype=float) - n // 2
    potential = np.zeros((n, n))
    prepared = imaging.PreparedSpecimen(axis, axis, (potential,), potential, None,
                                       {"atomistic_applied": False, "frozen_phonon_applied": False})
    monkeypatch.setattr(imaging, "_weighted_ray_statistics", lambda *_: {"convergence_semiangle_rad": 0.})
    fixture = SimpleNamespace(state=state, prepared=prepared, axis=axis,
                              incident=np.ones((n, n), complex), scale=1.)
    monkeypatch.setattr(imaging, "prepare_specimen_potentials", lambda *args: fixture.prepared)
    monkeypatch.setattr(imaging, "_incident_wave", lambda *args: fixture.incident)

    def matrix():
        b = .01 * fixture.scale
        return np.block([[np.zeros((2, 2)), b * np.eye(2)],
                         [-np.eye(2) / b, np.zeros((2, 2))]])

    def transfer(*args):
        m = matrix()
        return SimpleNamespace(matrix=m, j_img=m[:2, :2], j_diff_m_per_rad=m[:2, 2:])

    monkeypatch.setattr(direct_alignment, "diffraction_transfer", transfer)
    monkeypatch.setattr(camera_wave, "_camera_affine_offset_m", lambda *args: np.zeros(2))
    monkeypatch.setattr(multiplane_wave, "_canonical_map_and_offset",
                        lambda *args: (matrix(), np.zeros(4)))
    fixture.order_wave = np.broadcast_to(np.exp(2j * np.pi * 4 * axis / n), (n, n))
    fixture.order_position_mm = .01 * electron(state)[2] * 1e-9 * 4 / (n * 1e-10) * 1e3
    fixture.run = lambda: imaging.simulate_wave_image(state, SimpleNamespace(incident=None))
    return fixture


def test_at06_known_aperture_transmission_is_not_renormalised(tem_benchmark):
    f = tem_benchmark
    f.incident = np.sqrt(.2) * f.incident + np.sqrt(.8) * f.order_wave
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(.2, abs=1e-10)


def test_at02_offset_aperture_selects_nonzero_diffraction_order(tem_benchmark):
    f = tem_benchmark
    f.incident = np.sqrt(.2) * f.incident + np.sqrt(.8) * f.order_wave
    f.state.objective_aperture.offset_x_mm = f.order_position_mm
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(.8, abs=1e-10)


@pytest.mark.parametrize("strategy", ["physical_plane", "equivalent_pupil"])
def test_at04_closed_aperture_is_valid_zero_result(tem_benchmark, strategy):
    f = tem_benchmark
    f.state.sample.wave_objective_aperture_strategy = strategy
    f.state.objective_aperture.radius_mm = 0.
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == 0.
    assert np.all(np.isfinite(result.image_intensity))


@pytest.mark.parametrize("flag", ["enabled", "installed", "inserted"])
def test_at04_inactive_aperture_does_not_clip(tem_benchmark, flag):
    f = tem_benchmark
    f.incident = f.order_wave
    setattr(f.state.objective_aperture, flag, False)
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(1., abs=1e-10)
    assert not result.metrics["intermediate_aperture_transmissions"]


def test_at01_at03_actual_map_and_one_execution_per_branch(tem_benchmark):
    f = tem_benchmark
    f.incident = f.order_wave
    f.state.objective_aperture.offset_x_mm = f.order_position_mm
    first = f.run()
    f.scale = 2.  # Mechanical aperture fixed; change the canonical column map.
    second = f.run()
    assert first.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(1.)
    assert second.metrics["camera_collected_zero_loss_relative_intensity"] < 1e-20
    rows = first.metrics["intermediate_aperture_transmissions"]
    assert len(rows) == 1
    assert rows[0]["physical_element_id"] == "aperture:" + f.state.objective_aperture.key


def test_at05_equivalent_fourier_pupil_matches_physical_plane(tem_benchmark):
    f = tem_benchmark
    f.incident = np.sqrt(.2) * f.incident + np.sqrt(.8) * f.order_wave
    f.state.objective_aperture.offset_x_mm = f.order_position_mm
    physical = f.run()
    f.state.sample.wave_objective_aperture_strategy = "equivalent_pupil"
    equivalent = f.run()
    np.testing.assert_allclose(equivalent.camera_intensity, physical.camera_intensity, rtol=1e-10, atol=1e-15)
    rows = equivalent.metrics["intermediate_aperture_transmissions"]
    assert len(rows) == 1 and rows[0]["strategy"] == "equivalent_pupil"


def test_at05_non_fourier_equivalent_pupil_fails_explicitly(tem_benchmark, monkeypatch):
    from temsim.physics import multiplane_wave
    f = tem_benchmark
    f.state.sample.wave_objective_aperture_strategy = "equivalent_pupil"
    monkeypatch.setattr(multiplane_wave, "_canonical_map_and_offset",
                        lambda *args: (np.eye(4), np.zeros(4)))
    with pytest.raises(ValueError, match="equivalent_pupil unsupported"):
        f.run()


def test_at08_real_phase_grating_ensemble_keeps_prior_and_loss(tem_benchmark):
    from dataclasses import replace
    f = tem_benchmark
    f.state.sample.wave_multislice_enabled = True
    f.state.sample.wave_bandwidth_fraction = 1.
    sigma = imaging.interaction_constant_rad_per_v_angstrom(f.state.beam_voltage_kv)
    # Alternating phase +/- phi gives zero-order intensity cos(phi)^2.
    # One genuine multislice grating per configuration, no mocked propagation.
    checker = np.broadcast_to((-1.)**np.arange(32), (32, 32))
    configs = tuple((checker * np.arccos(np.sqrt(t)) / sigma)[None, :, :] for t in (.2, .8))
    f.prepared = replace(f.prepared, potential_configurations_v_angstrom=configs,
                         slice_thicknesses_angstrom=np.array([1.]))
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(.5, abs=1e-10)
    np.testing.assert_allclose(result.metrics["configuration_recorded_weights"], [.1, .4], atol=1e-10)
    assert result.metrics["configuration_prior_weights"] == (.5, .5)
    assert len(result.metrics["intermediate_aperture_transmissions"]) == 2
    assert result.absolute_diffraction_probability.sum() == pytest.approx(1., abs=1e-10)


def test_replay_offset_expansion_strategy_matches_fresh_without_mutation(tem_benchmark):
    f = tem_benchmark
    f.incident = np.sqrt(.2) * f.incident + np.sqrt(.8) * f.order_wave
    original = f.run()
    source = original.projector_checkpoint.unapertured_wave_configurations[0].copy()
    for strategy, offset, radius in (("physical_plane", f.order_position_mm, .003),
                                      ("physical_plane", 0., .1),
                                      ("equivalent_pupil", f.order_position_mm, .003)):
        f.state.sample.wave_objective_aperture_strategy = strategy
        f.state.objective_aperture.offset_x_mm = offset
        f.state.objective_aperture.radius_mm = radius
        replay = imaging.reproject_wave_image(f.state, original)
        fresh = f.run()
        np.testing.assert_array_equal(replay.camera_intensity, fresh.camera_intensity)
    np.testing.assert_array_equal(original.projector_checkpoint.unapertured_wave_configurations[0], source)


def test_at07_explicit_weight_and_reference_current_count_scaling(tem_benchmark):
    from temsim.physics.wave_flux import WaveMode, expected_electron_counts
    f = tem_benchmark
    modes = [WaveMode.from_legacy(f.incident, f.axis, f.axis, reference_discrete_norm=1024.,
                                  prior_weight=w, energy_kev=300.) for w in (1., .25)]
    from temsim.physics.camera_wave import project_wave_to_recording_plane
    projections = [project_wave_to_recording_plane(f.state, mode.weighted_density_amplitude(),
                    f.axis, f.axis, .0197, input_convention="weighted_density_per_m") for mode in modes]
    pixel_area = ((projections[0].x_mm[1] - projections[0].x_mm[0]) * 1e-3)**2
    p = projections[0].intensity * pixel_area
    counts = expected_electron_counts(p, reference_current_a=1e-12, exposure_s=.1)
    np.testing.assert_allclose(projections[1].intensity, projections[0].intensity * .25, rtol=1e-12)
    np.testing.assert_allclose(expected_electron_counts(p*.25, reference_current_a=1e-12, exposure_s=.1), counts*.25)
    np.testing.assert_allclose(expected_electron_counts(p, reference_current_a=2e-12, exposure_s=.1), counts*2)
    np.testing.assert_allclose(expected_electron_counts(p, reference_current_a=1e-12, exposure_s=.2), counts*2)


def test_at09_lossless_guard_detects_corrupted_propagation(monkeypatch):
    from dataclasses import replace
    from temsim.physics import multiplane_wave as mp
    wave = mp.PlaneWave(np.ones((4, 4), complex)/4, np.eye(2)*1e-9, np.zeros(2))
    good = mp.propagate_plane_wave(wave, np.eye(4), np.zeros(4), 2e-12)
    assert good.probability == pytest.approx(1., abs=1e-12)
    monkeypatch.setattr(mp, "_propagate_plane_wave", lambda *args: replace(wave, amplitude=wave.amplitude*.9))
    with pytest.raises(ValueError, match="lossless wave norm changed"):
        mp.propagate_plane_wave(wave, np.eye(4), np.zeros(4), 2e-12)


def test_duplicate_physical_execution_rejected():
    from temsim.physics.wave_flux import FluxLedger
    ledger = FluxLedger(branch_id="test")
    ledger.record("first", 1., .5, "physical_stop", physical_element_id="aperture:1")
    with pytest.raises(ValueError, match="executed twice"):
        ledger.record("second", .5, .2, "physical_stop", physical_element_id="aperture:1")
    with pytest.raises(ValueError, match="Discontinuous electron flux"):
        ledger.record("camera", 1., .5, "missed_detector")


def test_numerical_bandwidth_loss_is_retained_and_not_physical_absorption(tem_benchmark):
    f = tem_benchmark
    f.state.sample.wave_multislice_enabled = True
    f.state.sample.wave_bandwidth_fraction = 2/3
    f.state.objective_aperture.enabled = False
    high_order = np.broadcast_to(np.exp(2j*np.pi*12*f.axis/32), (32, 32))
    f.incident = np.sqrt(.2)*f.incident + np.sqrt(.8)*high_order
    result = f.run()
    assert result.metrics["camera_collected_zero_loss_relative_intensity"] == pytest.approx(.2, abs=1e-10)
    rows = result.metrics["camera_flux_ledger"]
    bandwidth = next(r for r in rows if r["loss_category"] == "numerical_bandwidth")
    assert bandwidth["lost_weight"] == pytest.approx(.8, abs=1e-10)
    assert bandwidth["physical_element_id"] is None
    assert not any(r["loss_category"] == "physical_stop" for r in rows)


def test_export_preserves_raw_flux_and_execution_graph(tem_benchmark, tmp_path):
    import json
    from temsim.physics.wave_export import export_wave_image
    f = tem_benchmark
    f.incident = np.sqrt(.2) * f.incident + np.sqrt(.8) * f.order_wave
    result = f.run()
    path = export_wave_image(result, tmp_path / "raw.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["camera_pixel_probability"].sum() == pytest.approx(.2, abs=1e-10)
        np.testing.assert_array_equal(archive["camera_density_post_psf_per_m2"], result.camera_intensity)
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["execution"] == result.metrics["wave_execution_manifest"]
        assert metadata["probability_reference"] == "specimen_entrance_conditional_zero_loss"


def test_provenance_binding_does_not_initialise_or_change_hardware(tem_benchmark):
    f = tem_benchmark
    physical_inputs = lambda: tuple((a.key, a.z_mm, a.enabled, getattr(a, "inserted", True)) for a in f.state.apertures)
    before = physical_inputs()
    result = f.run()
    assert physical_inputs() == before
    # Isolate the binding contract from the submission serializer (which has
    # its own manifest tests). No serialization should occur in this solver.
    manifest = SimpleNamespace(digest="test-submission",
                               to_dict=lambda: {"digest": "test-submission"})
    bound = imaging.bind_wave_request_manifest(result, manifest)
    assert bound.request_manifest["digest"] == manifest.digest
    assert bound.metrics["wave_source_request_digest"] == manifest.digest
    assert result.request_manifest is None
    assert physical_inputs() == before


def test_single_precision_norm_tolerance_is_explicit_without_repair():
    from temsim.physics.wave_flux import check_lossless_norm, FLOAT32_FLUX_RTOL
    with pytest.raises(ValueError, match="norm changed"):
        check_lossless_norm(1., 1.+1e-7, context="CPU")
    check_lossless_norm(1., 1.+1e-7, context="float32", rtol=FLOAT32_FLUX_RTOL)
    with pytest.raises(ValueError, match="norm changed"):
        check_lossless_norm(1., .999, context="float32", rtol=FLOAT32_FLUX_RTOL)


def test_tem_projection_cache_version_does_not_expire_other_products(monkeypatch):
    from temsim import calculation_cache as cache
    s = default_state()
    before = cache.calculation_signatures(s)
    monkeypatch.setattr(cache, "_TEM_PROJECTION_SCHEMA", "test-new-contract")
    after = cache.calculation_signatures(s)
    assert {k for k in before if before[k] != after[k]} == {"request", "wave", "wave_source"}


def test_policy_reuses_source_but_mechanical_edits_keep_geometry_dependencies():
    from temsim.calculation_cache import calculation_signatures
    s = default_state()
    before = calculation_signatures(s)
    s.sample.wave_objective_aperture_strategy = "equivalent_pupil"
    after = calculation_signatures(s)
    assert before["wave_source"] == after["wave_source"]
    assert before["wave"] != after["wave"]
    s.objective_aperture.radius_mm *= 1.2
    s.objective_aperture.offset_x_mm += .01
    changed = calculation_signatures(s)
    assert changed["wave_source"] != after["wave_source"]
    assert changed["wave"] != after["wave"]
