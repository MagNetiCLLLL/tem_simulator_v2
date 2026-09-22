"""Detector/scan orchestration fixtures, NOT physical-tip image acceptance.

Gun, specimen and long-column propagation are instrumented stubs in the bank
tests. Detector masks, complex readouts, absorption and disk stores are real.
No fixture is exposed as a GUI/CLI source or a qualified operating point.
"""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.wave_readout import WaveReadoutOptions
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
from temsim.physics.tip_wave_scan import ScanResponseAccumulator, ScanWaveSample
from temsim.physics.wave_execution import WaveExecutionOptions


@pytest.fixture
def bank(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.inelastic_wave as inelastic
    from test_wave_detector_readout import checkpoint
    state = default_state()
    state.camera.inserted = state.fluorescent_screen.inserted = False
    for detector, inner, outer in zip(state.stem_detectors, (.018, .007, 0.), (.05, .018, .007)):
        detector.inner_diameter_mm = inner
        detector.outer_width_mm = outer
        detector.geometry = "annulus" if inner else "disk"
        detector.point_spread_model = "none"
        detector.inserted = detector.readout_enabled = True
    seed = replace(checkpoint(.4, two=True), plane_z_mm=state.electron_gun.exit_plane_z_mm)
    calls = Counter()
    arrivals = []

    def gun(*args, **kwargs):
        calls["gun"] += 1
        return seed

    def column(state, upstream, stop, **kwargs):
        arrivals.append((upstream.plane_z_mm, stop))
        return replace(upstream, plane_z_mm=stop)

    def segmented(state, upstream, stop, **kwargs):
        return column(state, upstream, stop), False

    def specimen(state, upstream, **kwargs):
        calls["specimen"] += 1
        return replace(upstream, plane_z_mm=state.sample.z_mm+state.sample.thickness_nm*.5e-6)

    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", gun)
    monkeypatch.setattr(pipeline, "_prepare_column", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_propagate_column", column)
    monkeypatch.setattr(pipeline, "_propagate_column_segmented", segmented)
    monkeypatch.setattr(inelastic, "_propagate_inelastic_specimen", specimen)
    request = TipWaveRequest(detector_keys=("bf", "haadf", "df"), detector_pixels=256,
        execution=WaveExecutionOptions(segmented=False, cache_directory=str(tmp_path/"bank")),
        readout=WaveReadoutOptions(phase=True, complex_amplitude=True))
    return state, request, seed, calls, arrivals


@pytest.mark.parametrize("segmented", [False, True])
def test_bank_executes_upstream_once_and_reads_in_physical_order(bank, segmented):
    state, request, seed, calls, arrivals = bank
    request = replace(request, execution=replace(request.execution, segmented=segmented))
    result = simulate_tip_wave(state, request, use_cache=False)
    assert calls == {"gun": 1, "specimen": 1}
    assert [r.record["detector"] for r in result.detector_readouts] == ["haadf", "df", "bf"]
    assert result.detector is result.detector_readouts[-1]
    assert all(b > a for a, b in arrivals)
    assert [b for a, b in arrivals][-3:] == [d.z_mm for d in state.stem_detectors]
    # Independent Boolean partition of the unchanged fixture lattice. A
    # front annulus absorbs its pixels before the next physical detector.
    available = np.ones(seed.beam.modes[0].plane.amplitude.shape, bool)
    for detector, readout in zip(state.stem_detectors, result.detector_readouts):
        xy = seed.beam.modes[0].plane.coordinates_m()*1e3
        mask = detector.hit_mask(xy[0], xy[1])
        incoming = sum(m.weight_per_reference_electron*np.sum(abs(m.plane.amplitude)**2*available)
                       for m in seed.beam.modes)
        assert readout.record["incident_weight"] == pytest.approx(incoming, abs=2e-14)
        for old, new in zip(seed.beam.modes, readout.modes):
            expected = old.plane.full_amplitude(float(wavelength_m(old.energy_kev*1000)))
            # Compare the unchanged analytic carriers and envelope phase;
            # no aggregate phase is defined for these incoherent modes.
            valid = new.phase_valid
            np.testing.assert_allclose(np.exp(1j*new.phase_rad[valid]),
                                       np.exp(1j*np.angle(expected[valid])), atol=2e-12)
            assert new.mode_id == old.mode_id
            assert "UNDEFINED" in readout.record["mixed_total_phase"]
        available &= ~mask
    # The finite outer detector does not receive the Gaussian's residual
    # tail. Include that physical escape instead of renormalising it away.
    escaped = sum(m.weight_per_reference_electron*np.sum(abs(m.plane.amplitude)**2*available)
                  for m in seed.beam.modes)
    assert sum(r.record["optical_received_weight"] for r in result.detector_readouts)+escaped == pytest.approx(
        seed.beam.total_weight, abs=1e-12)
    assert 0 < result.detector.record["incident_weight"] < seed.beam.total_weight


def test_unselected_or_readout_disabled_detector_still_absorbs(bank):
    state, request, _, calls, _ = bank
    complete = simulate_tip_wave(state, request, use_cache=False)
    subset_request = replace(request, detector_keys=("df", "bf"))
    subset = simulate_tip_wave(state, subset_request, use_cache=False)
    state.stem_detectors[0].readout_enabled = False
    disabled = simulate_tip_wave(state, subset_request, use_cache=False)
    for reference, a, b in zip(complete.detector_readouts[1:], subset.detector_readouts, disabled.detector_readouts):
        np.testing.assert_allclose(a.optical_probability, reference.optical_probability, atol=1e-16)
        np.testing.assert_allclose(b.optical_probability, reference.optical_probability, atol=1e-16)
    assert calls["specimen"] == 3  # one per independent request, not per channel


def test_observables_do_not_change_downstream_wave_and_warm_matches_cold(bank):
    state, request, _, calls, _ = bank
    first = simulate_tip_wave(state, request)
    phase_only = replace(request, readout=WaveReadoutOptions(intensity=False, phase=True))
    warm = simulate_tip_wave(state, phase_only)
    cold = simulate_tip_wave(state, phase_only, use_cache=False)
    assert warm.propagation_cache_hit
    assert calls["gun"] == 2
    assert first.checkpoint.digest == warm.checkpoint.digest == cold.checkpoint.digest
    for a, b in zip(warm.detector_readouts, cold.detector_readouts):
        assert a.detected_probability is None
        for ma, mb in zip(a.modes, b.modes):
            np.testing.assert_allclose(ma.phase_rad, mb.phase_rad, equal_nan=True)
            np.testing.assert_allclose(ma.plane.amplitude, mb.plane.amplitude)


@pytest.mark.parametrize("keys", [("missing",), ("haadf", "df", "missing")])
def test_missing_requested_channel_rejects_before_gun_execution(bank, keys):
    state, request, _, calls, _ = bank
    with pytest.raises(ValueError, match="missing.*retracted"):
        simulate_tip_wave(state, replace(request, detector_keys=keys))
    assert not calls


def test_retracted_disabled_and_coincident_channels_are_not_silently_replaced(bank):
    state, request, _, calls, _ = bank
    state.stem_detectors[1].inserted = False
    with pytest.raises(ValueError, match="retracted"):
        simulate_tip_wave(state, request)
    state.stem_detectors[1].inserted = True
    state.stem_detectors[1].readout_enabled = False
    with pytest.raises(ValueError, match="readout-disabled"):
        simulate_tip_wave(state, request)
    state.stem_detectors[1].readout_enabled = True
    state.stem_detectors[1].z_mm = state.stem_detectors[0].z_mm
    with pytest.raises(ValueError, match="Coincident"):
        simulate_tip_wave(state, request)
    assert not calls


def test_bank_path_cannot_bypass_filter_before_last_detector(bank):
    from temsim.assembly_catalog import AssemblyCatalog
    state, request, _, calls, _ = bank
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    state.camera.inserted = state.fluorescent_screen.inserted = False
    for detector in state.stem_detectors:
        detector.inserted = detector.readout_enabled = True
    assert state.energy_filter_installed and state.energy_filter.enabled
    state.energy_filter.entrance_z_mm = (state.stem_detectors[0].z_mm+state.stem_detectors[-1].z_mm)/2
    with pytest.raises(ValueError, match="cannot skip"):
        simulate_tip_wave(state, request)
    assert not calls


@pytest.mark.parametrize("changes", [
    {"detector_keys": ["bf"]}, {"detector_keys": ("bf", "bf")}, {"detector_keys": ("",)},
    {"detector_keys": ("bf",), "detector_key": "haadf"},
    {"detector_keys": ("bf",), "stop": "gun_exit"},
])
def test_bank_request_is_unambiguous(changes):
    with pytest.raises(ValueError):
        TipWaveRequest(**changes).validate()


def _sample(row, column, dwell, responses, weight=.5, identity="fixed"):
    readouts = tuple(SimpleNamespace(record={"detector": key, "response_weight": value})
                     for key, value in responses.items())
    return ScanWaveSample(row, column, dwell, dwell*.1, weight,
        SimpleNamespace(detector=readouts[-1], detector_readouts=readouts, instrument_digest=identity))


def test_dwell_aggregation_has_separate_channels_and_no_false_zero_pixels():
    accumulator = ScanResponseAccumulator()
    accumulator.add(_sample(0, 0, 0, {"haadf": .1, "df": .2, "bf": .5}))
    accumulator.add(_sample(0, 0, 1, {"haadf": .3, "df": .4, "bf": .1}))
    accumulator.add(_sample(2, 4, 0, {"haadf": .2, "df": None, "bf": 0.}))
    arrays = accumulator.arrays()
    assert arrays["detector_keys"].tolist() == ["haadf", "df", "bf"]
    np.testing.assert_allclose(arrays["detector_response_per_tip_electron"][:, 0, 0], [.2, .3, .3])
    assert np.isnan(arrays["detector_response_per_tip_electron"][:, 1, 1]).all()
    assert np.isnan(arrays["detector_response_per_tip_electron"][:, 0, 1]).all()
    np.testing.assert_array_equal(arrays["detector_dwell_coverage"][:, 1, 1], [.5, 0, .5])
    np.testing.assert_array_equal(arrays["response_per_tip_electron"], arrays["detector_response_per_tip_electron"][-1])


@pytest.mark.parametrize("bad", [
    _sample(0, 0, 1, {"haadf": .2, "bf": float("nan")}),
    _sample(0, 0, 1, {"haadf": .2, "bf": .3}, weight=-.5),
    _sample(0, 0, 1, {"haadf": .2, "bf": .3}, weight=.75),
    _sample(0, 0, 1, {"bf": .3, "haadf": .2}),
    _sample(0, 0, 1, {"haadf": .2, "bf": .3}, identity="changed"),
])
def test_bad_dwell_does_not_partially_commit_any_channel(bad):
    accumulator = ScanResponseAccumulator()
    accumulator.add(_sample(0, 0, 0, {"haadf": .2, "bf": .3}))
    before = accumulator.arrays()
    with pytest.raises(ValueError):
        accumulator.add(bad)
    after = accumulator.arrays()
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    # A failed dwell did not consume this sample ID.
    accumulator.add(_sample(0, 0, 1, {"haadf": .2, "bf": .3}))
    np.testing.assert_allclose(accumulator.arrays()["detector_response_per_tip_electron"][:, 0, 0], [.2, .3])


def test_two_by_two_scan_calls_one_pipeline_per_dwell_not_per_detector(bank):
    from temsim.physics.tip_wave_scan import simulate_tip_scan
    state, request, _, calls, _ = bank
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    accumulator = ScanResponseAccumulator()
    for sample in simulate_tip_scan(state, request, use_cache=False):
        accumulator.add(sample)
    assert calls == {"gun": 4, "specimen": 4}
    arrays = accumulator.arrays()
    assert arrays["detector_response_per_tip_electron"].shape == (3, 2, 2)
    assert np.isfinite(arrays["detector_response_per_tip_electron"]).all()
    assert (arrays["detector_dwell_coverage"] == 1).all()


def test_readout_budget_is_shared_across_channels(bank, monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, request, _, _, _ = bank
    original = pipeline.read_wave_detector
    budgets = []
    def read(*args, **kwargs):
        budgets.append(kwargs["maximum_bytes"])
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "read_wave_detector", read)
    request = replace(request, maximum_readout_bytes=30*1024**2)
    simulate_tip_wave(state, request, use_cache=False)
    assert budgets == [10*1024**2]*3


@pytest.mark.parametrize("coherent", [False, True])
def test_default_stem_admission_is_not_opened_by_detector_bank(coherent):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    from temsim.physics.source_admission import admit_requested_wave_products, UnsupportedWaveSource
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.sample.stem_wave_enabled = True
    if coherent:
        state.electron_gun.emitter.coherence = TipCoherence()
    with pytest.raises(UnsupportedWaveSource, match="full image qualification remains open"):
        admit_requested_wave_products(state)
    assert state.electron_gun.emitter.virtual_source_fwhm_nm == 5.
