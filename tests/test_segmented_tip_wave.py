"""Exact segment/cache and physical clock checks, separate from image acceptance."""
from dataclasses import replace, FrozenInstanceError
from types import SimpleNamespace
import weakref
import gc

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.column_wave import _propagate_column, _propagate_column_segmented
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.wave_execution import ScanWaveNumerics, WaveExecutionOptions
from temsim.physics.tip_wave_scan import physical_scan_samples
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.detector.wave_readout import read_wave_detector, read_wave_detector_streamed, WaveReadoutOptions
from test_wave_detector_readout import checkpoint, detector


def quiet_state():
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for item in collection:
            item.enabled = False
    state.apertures = []
    return state


def test_executed_cache_loads_one_immutable_mode_and_rejects_corruption(tmp_path, monkeypatch):
    store = ExecutedWaveStore(tmp_path, "independent-test-dependency", 1<<28)
    original = checkpoint(.45, two=True)
    key = store.key("operator fixture")
    saved = store.put(key, original)
    assert saved.beam.total_weight == original.beam.total_weight
    calls, load = [], np.load
    def counted(*args, **kw):
        calls.append(args[0]); return load(*args, **kw)
    monkeypatch.setattr(np, "load", counted)
    restored = store.get(key)
    assert restored.digest == saved.digest
    assert not calls
    mode = restored.beam.modes[0]
    np.testing.assert_array_equal(mode.plane.amplitude, original.beam.modes[0].plane.amplitude)
    assert len(calls) == 5
    with pytest.raises(FrozenInstanceError):
        restored.beam.modes.rows = ()
    weak = weakref.ref(mode.plane.amplitude)
    del mode; gc.collect()
    assert weak() is None  # disk container has not retained the previous array
    with pytest.raises(ValueError, match="dependency"):
        ExecutedWaveStore(tmp_path, "different-physical-inputs", 1<<28).get(key)
    path = restored.beam.modes.root/restored.beam.modes.rows[0]["arrays"]["amplitude"]["file"]
    with path.open("r+b") as f:
        f.seek(-1, 2); f.write(b"X")
    with pytest.raises(ValueError, match="checksum"):
        store.get(key).beam.modes[0]


def test_failed_writer_does_not_publish_or_delete_committed_results(tmp_path):
    store = ExecutedWaveStore(tmp_path, "fixture", 1<<20)
    key = store.key("good")
    good = store.put(key, checkpoint())
    writer = store.writer(store.key("partial"), good.beam.reference_plane)
    writer.append(good.beam.modes[0]); writer.abort()
    assert store.get(store.key("partial")) is None
    assert store.get(key).digest == good.digest
    store.maximum_bytes = 1
    with pytest.raises(ValueError, match="budget"):
        store.put(store.key("too big"), checkpoint())
    assert store.get(key).digest == good.digest
    assert not list(tmp_path.glob("pending-*"))


def test_segmented_complex_wave_matches_monolithic_and_resumes_after_cancellation(tmp_path):
    state = quiet_state()
    aperture = default_state().objective_aperture
    aperture.enabled = aperture.installed = aperture.inserted = True
    aperture.z_mm, aperture.radius_mm = 2000.5, .005
    state.apertures = [aperture]
    original = checkpoint(.6)
    original = replace(original, beam=replace(original.beam, modes=(replace(original.beam.modes[0],
        axial_reference=AxialWaveReference(2e-8, 1e-22)),)))
    full = _propagate_column(state, original, 2001., maximum_step_mm=.1)
    store = ExecutedWaveStore(tmp_path, "same-physical-fixture", 1<<28)
    origin = store.put(store.key("fixture-upstream"), original)
    cancel = [False]
    def progress(done, total, label):
        if label.startswith("Column segment saved"):
            cancel[0] = True
    with pytest.raises(InterruptedError):
        _propagate_column_segmented(state, origin, 2001., store=store, segment_steps=3,
            maximum_step_mm=.1, cancelled=lambda: cancel[0], progress_callback=progress)
    streamed, reused = _propagate_column_segmented(state, origin, 2001., store=store, segment_steps=3, maximum_step_mm=.1)
    assert reused
    a, b = full.beam.modes[0], streamed.beam.modes[0]
    assert streamed.beam.total_weight == pytest.approx(full.beam.total_weight, rel=2e-12)
    for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(a.plane, field), getattr(b.plane, field), rtol=1e-10, atol=1e-15)
    assert b.axial_reference.flight_time_s == pytest.approx(a.axial_reference.flight_time_s, rel=1e-14)


def test_actual_scan_coils_use_distinct_energy_dependent_arrival_times():
    state = quiet_state()
    scan = state.ac_deflector
    scan.enabled = scan.scan_enabled = True
    scan.wobble_enabled = False
    scan.scan_frame_period_s = 1e-5
    scan.scan_lines = scan.scan_pixels_x = 4
    scan.set_scan_command_matrix_mrad(((.01, 0.), (0., .01)))
    start, stop = scan.upper_z_mm-.5, scan.lower_z_mm+.5
    c = checkpoint(two=True)
    modes = tuple(replace(m, energy_kev=e, axial_reference=AxialWaveReference(2e-8, 1e-22)) for m, e in zip(c.beam.modes, (100., 300.)))
    c = replace(c, beam=replace(c.beam, modes=modes), plane_z_mm=start)
    epoch = 2e-7
    result = _propagate_column(state, c, stop, maximum_step_mm=.5, tip_time_s=epoch)
    for before, after, record in zip(c.beam.modes, result.beam.modes, result.record["modes"]):
        _, velocity = _momentum_velocity(before.energy_kev*1000)
        expected = before.plane.origin_m+(stop-start)*1e-3*before.plane.tilt_rad
        for index, z in enumerate((scan.upper_z_mm, scan.lower_z_mm)):
            time = epoch+2e-8+(z-start)*1e-3/velocity
            kick = np.asarray(scan.kick_events(time_s=time)[index][1:])
            expected += (stop-z)*1e-3*kick
            assert record["deflector_actions"][index]["arrival_time_s"] == pytest.approx(time, rel=1e-14)
        np.testing.assert_allclose(after.plane.origin_m, expected, rtol=1e-10, atol=1e-18)
    assert result.record["modes"][0]["deflector_actions"][1]["arrival_time_s"] != result.record["modes"][1]["deflector_actions"][1]["arrival_time_s"]


def test_hardware_raster_dwell_sampling_does_not_invent_a_new_scan_size():
    state = quiet_state()
    scan = state.ac_deflector
    scan.enabled = scan.scan_enabled = True
    scan.wobble_enabled = False
    scan.scan_frame_period_s = 1.
    scan.scan_lines, scan.scan_pixels_x = 4, 8
    rows = list(physical_scan_samples(state, ScanWaveNumerics(stride=2, dwell_samples=2)))
    assert len(rows) == 16
    assert rows[0][:3] == (0, 0, 0)
    assert rows[0][3] == pytest.approx(.25/8/4)
    assert {r[0] for r in rows} == {0, 2}
    assert {r[1] for r in rows} == {0, 2, 4, 6}
    assert all(r[4] == .5 for r in rows)
    with pytest.raises(ValueError, match="positions"):
        list(physical_scan_samples(state, ScanWaveNumerics(maximum_positions=1)))


def test_streamed_detector_matches_full_readout_and_keeps_phase_lazy(tmp_path):
    c, d = checkpoint(.3, two=True), detector()
    options = WaveReadoutOptions(phase=True, complex_amplitude=True, covariance=True)
    expected = read_wave_detector(c, d, options)
    store = ExecutedWaveStore(tmp_path, "detector-fixture", 1<<28)
    saved = store.put(store.key("incident"), c)
    actual = read_wave_detector_streamed(saved, d, options, pixels=64)
    np.testing.assert_allclose(actual.optical_probability, expected.optical_probability, atol=1e-17)
    np.testing.assert_allclose(actual.detected_probability, expected.detected_probability, atol=1e-17)
    for a, b in zip(actual.modes, expected.modes):
        np.testing.assert_allclose(a.complex_cell_amplitude, b.complex_cell_amplitude, atol=1e-15)
        np.testing.assert_array_equal(a.phase_valid, b.phase_valid)
        np.testing.assert_allclose(a.phase_rad, b.phase_rad, equal_nan=True)


def test_live_memory_shortage_stops_before_refinement_allocation(monkeypatch):
    from temsim.physics.wave_grid import refine_plane_wave, WaveGridNumerics
    monkeypatch.setattr("temsim.physics.wave_execution.available_physical_memory", lambda: (2*1024**3, 96*1024**3))
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **kw: pytest.fail("must not allocate FFT when free RAM is insufficient"))
    with pytest.raises(MemoryError, match="available physical memory"):
        refine_plane_wave(checkpoint().beam.modes[0].plane, (2048, 2048), numerics=WaveGridNumerics())


def test_scan_accumulator_preserves_missing_positions_and_dwell_weights():
    from temsim.physics.tip_wave_scan import ScanWaveSample, ScanResponseAccumulator
    accumulated = ScanResponseAccumulator()
    def sample(r, c, i, value):
        return ScanWaveSample(r, c, i, i*.1, .5, SimpleNamespace(detector=SimpleNamespace(record={"response_weight": value})))
    accumulated.add(sample(0, 0, 0, .2))
    accumulated.add(sample(0, 0, 1, .4))
    accumulated.add(sample(2, 4, 0, .9))
    arrays = accumulated.arrays()
    np.testing.assert_array_equal(arrays["physical_rows"], [0, 2])
    np.testing.assert_array_equal(arrays["physical_columns"], [0, 4])
    assert arrays["response_per_tip_electron"][0, 0] == pytest.approx(.3)
    assert np.isnan(arrays["response_per_tip_electron"][1, 1])
    assert np.isnan(arrays["response_per_tip_electron"][0, 1])
    assert arrays["dwell_coverage"][1, 1] == .5
    with pytest.raises(ValueError, match="twice"):
        accumulated.add(sample(0, 0, 1, .4))


def test_scan_generator_passes_hardware_times_and_releases_previous_result(monkeypatch):
    import temsim.physics.tip_wave_scan as scanning
    state = default_state()
    scan = state.ac_deflector
    scan.enabled = scan.scan_enabled = True
    scan.wobble_enabled = False
    scan.scan_lines = scan.scan_pixels_x = 2
    clocks, refs = [], []
    class Result:
        pass
    def calculated(state, request, **kwargs):
        clocks.append(request.tip_time_s)
        result = Result()
        refs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(scanning, "simulate_tip_wave", calculated)
    expected = list(scanning.physical_scan_samples(state))
    generator = scanning.simulate_tip_scan(state)
    first = next(generator)
    del first
    second = next(generator)
    gc.collect()
    assert refs[0]() is None
    del second
    for value in generator:
        del value
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert clocks == [r[3] for r in expected]


def test_timed_segment_cache_reuses_static_prefix_but_not_new_coil_actions(tmp_path):
    state = quiet_state()
    scan = state.ac_deflector
    scan.enabled = scan.scan_enabled = True
    scan.wobble_enabled = False
    scan.scan_frame_period_s = 1e-5
    scan.scan_lines = scan.scan_pixels_x = 4
    scan.set_scan_command_matrix_mrad(((.01, 0.), (0., .01)))
    start, stop = scan.upper_z_mm-2., scan.lower_z_mm+1.
    original = checkpoint(.4)
    original = replace(original, plane_z_mm=start, beam=replace(original.beam, modes=(replace(original.beam.modes[0],
        axial_reference=AxialWaveReference(2e-8, 1e-22)),)))
    store = ExecutedWaveStore(tmp_path, "timed-coil-operator-fixture", 1<<29)
    inputs = store.put(store.key("computed-fixture"), original)
    kw = dict(store=store, segment_steps=2, maximum_step_mm=.5)
    first, _ = _propagate_column_segmented(state, inputs, stop, tip_time_s=2e-7, **kw)
    second, hit = _propagate_column_segmented(state, inputs, stop, tip_time_s=8e-7, **kw)
    expected = _propagate_column(state, original, stop, tip_time_s=8e-7, maximum_step_mm=.5)
    assert hit  # static prefix before the first physical coil
    assert first.digest != second.digest
    a, b = second.beam.modes[0], expected.beam.modes[0]
    for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(a.plane, field), getattr(b.plane, field), rtol=1e-9, atol=1e-14)


def test_specimen_domain_planning_uses_stored_geometry_without_reading_amplitudes(tmp_path, monkeypatch):
    from temsim.physics.specimen_wave_transport import _covering_domain
    original = checkpoint(two=True)
    store = ExecutedWaveStore(tmp_path, "domain-fixture", 1<<28)
    stored = store.put(store.key("incident"), original)
    load = np.load
    def geometry_only(path, **kwargs):
        assert "amplitude" not in str(path)
        return load(path, **kwargs)
    monkeypatch.setattr(np, "load", geometry_only)
    actual, expected = _covering_domain(stored.beam), _covering_domain(original.beam)
    np.testing.assert_allclose(actual[0], expected[0], atol=0)
    assert actual[1] == expected[1]
