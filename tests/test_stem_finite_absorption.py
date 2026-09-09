"""Finite-material absorption budgets; bounded waves, no full-column scan."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from temsim.detector import stem_signal
from temsim.optics.column import default_state
from temsim.physics import stem_wave_imaging as wave
from temsim.specimen.inelastic import real_inelastic_distribution


def _state():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.specimen_mode = "reference"
    state.sample.reference_sample_key = "si_110"
    state.sample.envelope_shape = "rectangle"
    state.sample.size_x_nm = state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 64
    state.sample.wave_field_of_view_angstrom = 40.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_frozen_phonon_enabled = False
    state.sample.stem_wave_enabled = True
    state.sample.real_high_angle_tail_enabled = False
    state.probe_corrector_installed = False
    state.objective_lens.cs_mm = state.objective_lens.cc_mm = 0.0
    return state


def _simulation(*, centre_nm=(0.0, 0.0), survival=1.0):
    # A local coherent probe fixture, not a column-transport approximation.
    tx = np.array((0.0, 0.03, -0.03, 0.0, 0.0))
    ty = np.array((0.0, 0.0, 0.0, 0.03, -0.03))
    return SimpleNamespace(incident=SimpleNamespace(
        alive=np.ones(5, dtype=bool), ray_weight=np.full(5, survival / 5),
        x=np.full((1, 5), centre_nm[0] * 1e-9),
        y=np.full((1, 5), centre_nm[1] * 1e-9),
        tx=tx[None, :], ty=ty[None, :], energy_offset_ev=np.zeros(5),
    ), branches={})


def _zero_raster(state, monkeypatch):
    state.step_mm = 5.0
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    state.descan_deflector.enabled = state.descan_deflector.scan_enabled = False
    state.fluorescent_screen.inserted = state.camera.inserted = False
    for lens in state.lenses:
        lens.percent = 0.0
    for aperture in state.apertures:
        aperture.inserted = False
    for detector in state.stem_detectors:
        detector.inserted = detector.readout_enabled = detector.key == "bf"
        if detector.key == "bf":
            detector.outer_width_mm = 10000.0
    monkeypatch.setattr(stem_signal, "calibrate_scan_system", lambda *_: None)
    monkeypatch.setattr(stem_signal, "paired_kick_response", lambda *_: np.zeros((2, 2)))


def _empty_grid(monkeypatch, *, pixels=256, fov_nm=4.0, centre_nm=(0.0, 0.0)):
    # A zero phase object isolates incident-probe overlap without building a
    # crystal; real production probe formation and detector FFTs still run.
    axis = (np.arange(pixels) - pixels // 2) * (10 * fov_nm / pixels)
    potential = np.zeros((pixels, pixels))
    prepared = SimpleNamespace(
        x_angstrom=axis, y_angstrom=axis,
        mean_projected_potential_v_angstrom=potential,
        potential_configurations_v_angstrom=(potential,),
        slice_thicknesses_angstrom=None,
        metrics={"calculation_roi_centre_nm": centre_nm,
                 "atomistic_applied": False, "frozen_phonon_applied": False},
    )
    preset = SimpleNamespace(key="empty_phase_fixture", pixels=pixels,
                             field_of_view_angstrom=fov_nm * 10)
    monkeypatch.setattr(wave, "_wave_grid", lambda *_: (preset, prepared))


def _wave(state, simulation, scan_nm):
    scan = np.asarray(scan_nm, dtype=float)[None, :] * 1e-3
    return wave.simulate_angle_resolved_stem(
        state, simulation, (wave.AngularDetector("all", 0.0, 100.0),),
        scan, np.zeros_like(scan), compute_sample_overlap=True,
    )


def test_actual_vacuum_window_has_no_absorption_and_keeps_detector_signal(monkeypatch):
    state = _state()
    state.sample.centre_x_nm = 1000.0
    _zero_raster(state, monkeypatch)
    simulation = _simulation()
    frames = []
    for mean_free_path in (0.0, state.sample.thickness_nm):
        state.sample.real_absorption_mean_free_path_nm = mean_free_path
        simulation.real_interactions = real_inelastic_distribution(state)
        frames.append(stem_signal.acquire_stem_scan(simulation, state))
    reference, absorbing = frames
    assert absorbing.metrics["specimen_potential_model"] == "finite_sample_vacuum_outside"
    assert absorbing.metrics["specimen_atom_count"] == 0
    np.testing.assert_array_equal(absorbing.absorbed_fraction, 0.0)
    np.testing.assert_allclose(absorbing.fractions["bf"], reference.fractions["bf"], rtol=1e-12)
    assert absorbing.metrics["finite_sample_absorption_overlap"]["maximum"] == 0.0
    np.testing.assert_allclose(
        absorbing.absorbed_fraction + absorbing.uncollected_fraction + absorbing.fractions["bf"], 1.0,
        atol=2e-14,
    )


@pytest.mark.parametrize("centre_nm", ((0.0, 0.0), (7.0, -3.0)))
def test_partial_overlap_follows_translated_gaussian_intensity_and_physical_edge(monkeypatch, centre_nm):
    state = _state()
    state.sample.wave_multislice_enabled = False
    state.sample.size_x_nm = 1.0
    state.sample.size_y_nm = 10.0
    state.sample.centre_x_nm, state.sample.centre_y_nm = centre_nm
    _empty_grid(monkeypatch, centre_nm=centre_nm)
    sigma_nm = 0.2
    def gaussian_spectrum(_state, _stats, frequencies_x, frequencies_y, _wavelength):
        fx, fy = np.meshgrid(frequencies_x, frequencies_y)
        # The inverse transform has intensity exp(-r^2 / (2 sigma^2)).
        return np.exp(-4 * np.pi**2 * (10 * sigma_nm)**2 * (fx**2 + fy**2)).astype(complex)
    monkeypatch.setattr(wave, "_probe_spectrum", gaussian_spectrum)
    positions = np.array((-1.0, -0.5, 0.0, 0.5, 1.0))
    result = _wave(state, _simulation(centre_nm=centre_nm), positions)
    expected = ndtr((0.5 - positions) / sigma_nm) - ndtr((-0.5 - positions) / sigma_nm)
    # Envelope edges are sampled on the actual wave grid (one-pixel error).
    np.testing.assert_allclose(result.sample_overlap_fraction[0], expected, atol=0.017, rtol=0)
    assert result.sample_overlap_fraction[0, 0] < 0.01
    assert 0.48 < result.sample_overlap_fraction[0, 1] < 0.52
    assert result.sample_overlap_fraction[0, 2] > 0.98


def test_overlap_includes_configured_coherent_defocus_not_geometric_ray_rms(monkeypatch):
    state = _state()
    state.sample.envelope_shape = "disk"
    state.sample.size_x_nm = state.sample.size_y_nm = 2.0
    state.sample.wave_multislice_enabled = False
    _empty_grid(monkeypatch, pixels=512, fov_nm=8.0)
    simulation = _simulation()  # All geometric positions are exactly zero.
    focused = _wave(state, simulation, (0.0,))
    state.sample.wave_defocus_nm = -100.0
    defocused = _wave(state, simulation, (0.0,))
    assert focused.sample_overlap_fraction[0, 0] > 0.95
    assert 0.03 < defocused.sample_overlap_fraction[0, 0] < 0.3
    assert defocused.metrics["probe_effective_defocus_nm"] == pytest.approx(-100.0)


def test_full_projected_coverage_recovers_bulk_survival(monkeypatch):
    state = _state()
    state.sample.wave_multislice_enabled = False
    _zero_raster(state, monkeypatch)
    _empty_grid(monkeypatch, pixels=64)
    state.sample.real_absorption_mean_free_path_nm = state.sample.thickness_nm
    simulation = _simulation()
    simulation.real_interactions = real_inelastic_distribution(state)
    frame = stem_signal.acquire_stem_scan(simulation, state)
    assert frame.metrics["finite_sample_absorption_overlap"]["minimum"] == 1.0
    np.testing.assert_allclose(frame.absorbed_fraction, 1 - np.exp(-1), rtol=1e-12)
    np.testing.assert_allclose(frame.fractions["bf"] + frame.uncollected_fraction, np.exp(-1), rtol=1e-12)


@pytest.mark.parametrize(("field", "value"), (("inserted", False), ("thickness_nm", 0.0)))
def test_inactive_material_overlap_is_zero(monkeypatch, field, value):
    state = _state()
    setattr(state.sample, field, value)
    state.sample.wave_multislice_enabled = False
    _empty_grid(monkeypatch, pixels=64)
    result = _wave(state, _simulation(), (-0.5, 0.0, 0.5))
    np.testing.assert_array_equal(result.sample_overlap_fraction, 0.0)


def test_full_survival_channel_sum_tolerates_roundoff(monkeypatch):
    state = _state()
    state.sample.wave_multislice_enabled = False
    _zero_raster(state, monkeypatch)
    _empty_grid(monkeypatch, pixels=64)
    simulation = _simulation()
    simulation.real_interactions = SimpleNamespace(tracked_probability=1.0 + 2e-16)
    frame = stem_signal.acquire_stem_scan(simulation, state)
    np.testing.assert_array_equal(frame.absorbed_fraction, 0.0)
    assert frame.metrics["tracked_probability_after_inelastic_absorption"] == 1.0


def test_resident_cuda_path_uses_the_same_incident_overlap_without_another_transport(monkeypatch):
    from temsim.physics.compute_backend import WAVE_BACKEND_CUPY
    from temsim.physics.stem_cuda_pipeline import ResidentStemCudaResult
    from temsim.physics.wave_fft import WaveFftDiagnostics
    state = _state()
    state.sample.size_x_nm = state.sample.size_y_nm = 1.0
    state.sample.wave_multislice_enabled = False
    _empty_grid(monkeypatch, pixels=128)
    positions = (-0.7, 0.0, 0.7)
    cpu = _wave(state, _simulation(), positions)
    monkeypatch.setattr(wave, "choose_wave_backend", lambda *_a, **_k: (WAVE_BACKEND_CUPY, None))
    calls = []
    def resident(**kwargs):
        calls.append(kwargs)
        size = len(kwargs["scan_x_angstrom"])
        return ResidentStemCudaResult(
            fractions_flat={"all": np.ones(size)}, uncollected_flat=np.zeros(size),
            detector_relative_standard_error={}, multislice_diagnostics=None,
            fft_diagnostics=WaveFftDiagnostics(WAVE_BACKEND_CUPY, "complex64 / float32", None),
            metrics={"cuda_resident_pipeline": True}, truncated_flat=np.zeros(size),
        )
    monkeypatch.setattr(wave, "run_resident_stem_cuda", resident)
    gpu = _wave(state, _simulation(), positions)
    assert len(calls) == 1
    assert gpu.metrics["cuda_resident_pipeline"]
    np.testing.assert_allclose(gpu.sample_overlap_fraction, cpu.sample_overlap_fraction, rtol=1e-12)


def test_partial_absorption_preserves_source_budget_and_material_conditioned_tail(monkeypatch):
    state = _state()
    _zero_raster(state, monkeypatch)
    state.sample.real_high_angle_tail_enabled = True
    state.sample.real_tail_material_source = state.sample.real_tail_screening_source = "manual"
    state.sample.real_tail_atomic_number = 14
    state.sample.real_tail_areal_density_atoms_nm2 = 250.0
    state.sample.real_tail_screening_angle_mrad = 5.0
    state.sample.real_tail_max_angle_mrad = 100.0
    simulation = _simulation()
    bulk_survival, incident_fraction = 0.3, 0.8
    simulation.real_interactions = SimpleNamespace(tracked_probability=bulk_survival)
    overlap = np.array(((0.0, 0.25), (0.75, 1.0)))
    def prescribed_wave(_state, _simulation, _detectors, x, y, **kwargs):
        assert kwargs["compute_sample_overlap"]
        return SimpleNamespace(
            scan_x_um=x, scan_y_um=y, fractions={"bf": np.full(x.shape, 0.6)},
            uncollected_fraction=np.full(x.shape, 0.4), truncated_fraction=np.full(x.shape, 0.1),
            maximum_isotropic_angle_mrad=50.0, metrics={}, fourdstem_artifact=None,
            sample_overlap_fraction=overlap,
        )
    monkeypatch.setattr(stem_signal, "simulate_angle_resolved_stem", prescribed_wave)
    monkeypatch.setattr(stem_signal, "measure_sample_current", lambda *_: SimpleNamespace(fraction=incident_fraction))
    actual_tail = stem_signal._real_high_angle_tail
    raw_tail = []
    def counted_tail(*args, **kwargs):
        result = actual_tail(*args, **kwargs)
        raw_tail.append(result)
        return result
    monkeypatch.setattr(stem_signal, "_real_high_angle_tail", counted_tail)
    frame = stem_signal.acquire_stem_scan(simulation, state)
    assert np.max(raw_tail[0][2]) > 0.0
    np.testing.assert_allclose(frame.high_angle_tail_fraction["bf"], raw_tail[0][0]["bf"] * bulk_survival)
    np.testing.assert_allclose(frame.absorbed_fraction, incident_fraction * overlap * (1 - bulk_survival))
    np.testing.assert_allclose(frame.absorbed_fraction + frame.uncollected_fraction + frame.fractions["bf"], 1.0,
                               atol=2e-14)
    assert frame.metrics["real_probability_conserved"]
    # Truncation remains a subset of uncollected probability, not another loss.
    assert np.all(frame.truncated_fraction <= frame.uncollected_fraction)
    assert "not a spatial absorptive potential" in frame.metrics["finite_sample_absorption_scope"]


def test_diffraction_recollection_preserves_local_absorption_in_source_units(monkeypatch):
    from temsim.physics.diffraction_memory import recollect_stem
    state = _state()
    bf = next(item for item in state.stem_detectors if item.key == "bf")
    bf.inserted = bf.readout_enabled = True
    scan = np.zeros((2, 2))
    incident = 0.8
    absorbed = np.array(((0.0, 0.2), (0.4, 0.6)))
    artifact = SimpleNamespace(data=np.ones((2, 2, 1, 1)), calibration=SimpleNamespace(scan_times_s=scan))
    frame = stem_signal.StemScanResult(
        scan, scan, {"bf": np.zeros_like(scan)}, {},
        metrics={"incident_sample_fraction": incident,
                 "tracked_probability_after_inelastic_absorption": 0.25},
        absorbed_fraction=absorbed, fourdstem_artifact=artifact,
    )
    monkeypatch.setattr("temsim.physics.record_plane.build_record_plane_plan", lambda *_a, **_k: "plan")
    monkeypatch.setattr("temsim.physics.fourdstem.integrate_runtime_recording_planes", lambda *_a, **_k:
                        SimpleNamespace(images={"bf": np.full((2, 2), 0.25)},
                                        surviving_weight=np.full((2, 2), 0.75), plan_fingerprint="test"))
    replay = recollect_stem(state, frame)
    np.testing.assert_allclose(replay.fractions["bf"], 0.25 * (incident - absorbed))
    np.testing.assert_array_equal(replay.absorbed_fraction, absorbed)
    np.testing.assert_array_equal(frame.fractions["bf"], 0.0)
    np.testing.assert_array_equal(frame.absorbed_fraction, absorbed)
    np.testing.assert_allclose(replay.fractions["bf"] + replay.absorbed_fraction + replay.uncollected_fraction, 1.0)


def _physical_replay(monkeypatch, *, angular_weights, incident=0.8, bulk_survival=1.0,
                     legacy_absorption=False, aperture_radius_mm=None):
    """Use production cube routing through a prescribed, signed linear map."""
    from temsim.physics.diffraction_memory import recollect_stem
    from temsim.physics.first_order import TransverseTransfer
    from temsim.physics.fourdstem import FourDSTEMArtifact, FourDSTEMCalibration, PixelatedDetectorResponse
    from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
    state = _state()
    for detector in state.stem_detectors:
        detector.inserted = detector.readout_enabled = detector.key == "bf"
    angles = np.arange(len(angular_weights), dtype=float) * 2.0
    calibration = FourDSTEMCalibration.from_rectilinear_axes(
        np.array((0.0, 0.001)), np.array((0.0, 0.001)), angles, np.array((0.0,)))
    data = np.broadcast_to(np.asarray(angular_weights), calibration.shape).copy()
    artifact = FourDSTEMArtifact(None, None, None, calibration, data,
        {"detector_response": PixelatedDetectorResponse().provenance()})
    planes = []
    if aperture_radius_mm is not None:
        planes.append(PlaneStop("stop", "Upstream aperture", 1.0, "aperture", "disk",
                                radius_mm=aperture_radius_mm))
    planes.append(PlaneStop("bf", "BF", 2.0, "detector", "disk", outer_width_mm=2.0,
                            readout_enabled=True))
    # At this camera length 1 mrad maps to 1 mm, independent of scan position.
    transfers = tuple(TransverseTransfer(0.0, plane.z_mm, np.zeros((2, 2)), np.eye(2),
                                         np.zeros((2, 2)), np.eye(2)) for plane in planes)
    plan = RecordPlanePlan(0.0, tuple(planes), transfers, "a"*64, "b"*64)
    zero = np.zeros((2, 2))
    frame = stem_signal.StemScanResult(zero, zero, {"bf": zero}, {},
        metrics={"incident_sample_fraction": incident,
                 "tracked_probability_after_inelastic_absorption": bulk_survival},
        absorbed_fraction=(None if legacy_absorption else np.full((2, 2), incident * (1 - bulk_survival))),
        fourdstem_artifact=artifact)
    monkeypatch.setattr("temsim.physics.record_plane.build_record_plane_plan", lambda *_a, **_k: plan)
    return frame, recollect_stem(state, frame)


def test_physical_replay_counts_upstream_loss_when_bf_collects_every_arriving_electron(monkeypatch):
    _, replay = _physical_replay(monkeypatch, angular_weights=(1.0,), incident=0.8)
    np.testing.assert_allclose(replay.fractions["bf"], 0.8)
    np.testing.assert_allclose(replay.uncollected_fraction, 0.2)
    np.testing.assert_array_equal(replay.absorbed_fraction, 0.0)
    np.testing.assert_array_equal(replay.truncated_fraction, 0.0)
    assert replay.metrics["post_recording_surviving_mean_source_fraction"] == 0.0
    assert replay.metrics["other_stop_loss_mean_source_fraction"] == 0.0
    assert replay.metrics["real_probability_conserved"]


@pytest.mark.parametrize("legacy_absorption", (False, True))
def test_physical_replay_keeps_bandwidth_and_other_stops_inside_uncollected_budget(monkeypatch, legacy_absorption):
    # Raw probability: 0 mrad -> BF, 2 mrad -> beyond BF, 4 mrad -> aperture.
    # Their total 0.6 leaves another 0.4 removed by the stored bandwidth mask.
    old, replay = _physical_replay(
        monkeypatch, angular_weights=(0.1, 0.2, 0.3), incident=0.8,
        bulk_survival=0.75, legacy_absorption=legacy_absorption, aperture_radius_mm=3.0,
    )
    np.testing.assert_allclose(replay.absorbed_fraction, 0.2)
    np.testing.assert_allclose(replay.fractions["bf"], 0.06)
    np.testing.assert_allclose(replay.truncated_fraction, 0.24)
    assert replay.metrics["post_recording_surviving_mean_source_fraction"] == pytest.approx(0.12)
    assert replay.metrics["other_stop_loss_mean_source_fraction"] == pytest.approx(0.18)
    # 0.2 upstream + 0.24 bandwidth + 0.18 aperture + 0.12 surviving beam.
    np.testing.assert_allclose(replay.uncollected_fraction, 0.74)
    np.testing.assert_allclose(replay.fractions["bf"] + replay.absorbed_fraction + replay.uncollected_fraction, 1.0)
    assert np.all(replay.truncated_fraction <= replay.uncollected_fraction)
    if legacy_absorption:
        assert old.absorbed_fraction is None
    else:
        np.testing.assert_allclose(old.absorbed_fraction, 0.2)
    assert replay.metrics["real_probability_conserved"]
