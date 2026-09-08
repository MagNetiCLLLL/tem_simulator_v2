"""Shot-noise/dose checks on saved STEM fractions; no wave transport runs."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.detector.stem_signal as stem_signal
from temsim.detector.stem_signal import (
    DetectorSignal, ELEMENTARY_CHARGE_C, StemScanResult, reweight_stem_scan,
)


def _state(*, pixel_count=256, source_electrons_per_pixel=500.0, seed=123):
    dwell = 1.0e-6
    return SimpleNamespace(
        electron_gun=SimpleNamespace(
            emitted_current_a=source_electrons_per_pixel * ELEMENTARY_CHARGE_C / dwell,
            ray_count=49,
        ),
        column_current_limit_percent=100.0,
        ac_deflector=SimpleNamespace(scan_frame_period_s=pixel_count * dwell),
        sample=SimpleNamespace(stem_poisson_enabled=True, stem_poisson_seed=seed),
    )


def _frame(shape=(16, 16), fractions=None, metrics=None):
    fractions = ({"bf": np.full(shape, 0.4), "haadf": np.full(shape, 0.2)}
                 if fractions is None else fractions)
    yy, xx = np.indices(shape, dtype=float)
    signals = {
        key: DetectorSignal(
            key=key, name=key.upper(), fraction=float(np.mean(values)),
            simulated_electrons=49.0 * float(np.mean(values)),
            current_pa=0.0, electrons_per_second=0.0, collection_angle=None,
        )
        for key, values in fractions.items()
    }
    return StemScanResult(
        scan_x_um=xx * 1.0e-3, scan_y_um=yy * 1.0e-3,
        fractions=fractions, detector_signals=signals,
        metrics=metrics, uncollected_fraction=np.full(shape, 0.35),
        absorbed_fraction=np.full(shape, 0.05),
        truncated_fraction=np.full(shape, 0.02),
        high_angle_tail_fraction={"haadf": np.full(shape, 0.01)},
        probe_state=object(), fourdstem_artifact=object(),
    )


@pytest.mark.parametrize("mean", (0.2, 5.0, 80.0))
def test_large_uniform_poisson_ensemble_has_correct_mean_and_variance(mean):
    shape = (128, 512)
    population = math.prod(shape)
    state = _state(pixel_count=population)
    frame = _frame(shape, fractions={"bf": np.full(shape, mean / 500.0)})
    result = reweight_stem_scan(state, frame)
    counts = result.poisson_counts["bf"]
    assert np.issubdtype(counts.dtype, np.integer)
    assert counts.min() >= 0
    np.testing.assert_allclose(result.expected_electrons["bf"], mean, rtol=1.0e-14)
    # Analytic six-standard-error limits, not arbitrary image tolerances.
    # For Poisson: central fourth moment = mean + 3*mean**2.
    mean_error = math.sqrt(mean / population)
    variance_error = math.sqrt(
        (mean + 2.0 * mean**2 + 2.0 * mean**2 / (population - 1)) / population
    )
    assert abs(float(counts.mean()) - mean) < 6.0 * mean_error
    assert abs(float(counts.var(ddof=1)) - mean) < 6.0 * variance_error


def test_seed_changes_realization_without_changing_dose_or_saved_fractions():
    state = _state()
    original = _frame()
    first = reweight_stem_scan(state, original)
    repeated = reweight_stem_scan(state, original)
    state.sample.stem_poisson_seed += 1
    changed = reweight_stem_scan(state, original)
    for key in first.fractions:
        np.testing.assert_array_equal(first.poisson_counts[key], repeated.poisson_counts[key])
        assert not np.array_equal(first.poisson_counts[key], changed.poisson_counts[key])
        np.testing.assert_array_equal(first.expected_electrons[key], changed.expected_electrons[key])
        np.testing.assert_array_equal(first.current_pa[key], changed.current_pa[key])
        np.testing.assert_array_equal(first.fractions[key], original.fractions[key])
    assert first.metrics["source_electrons_per_pixel"] == changed.metrics["source_electrons_per_pixel"]
    assert original.poisson_counts is None and original.expected_electrons is None


@pytest.mark.parametrize("zero_source", (True, False))
def test_no_current_or_no_collected_fraction_produces_exact_zero_counts(zero_source):
    state = _state()
    frame = _frame()
    if zero_source:
        state.column_current_limit_percent = 0.0
    else:
        frame = replace(frame, fractions={key: np.zeros_like(values)
                                         for key, values in frame.fractions.items()})
    result = reweight_stem_scan(state, frame)
    for key in frame.fractions:
        assert not np.any(result.current_pa[key])
        assert not np.any(result.expected_electrons[key])
        assert not np.any(result.poisson_counts[key])


def test_dose_reweight_updates_readout_and_metadata_without_transport_or_mutation(monkeypatch):
    state = _state()
    frame = _frame(metrics={"detector_sampling": {}, "sampling_state_signature": "old"})
    monkeypatch.setattr(stem_signal, "calculation_signatures", lambda candidate: {"stem": "current"})
    baseline = reweight_stem_scan(state, frame)
    old_counts = {key: values.copy() for key, values in baseline.poisson_counts.items()}

    def no_transport(*_args, **_kwargs):
        pytest.fail("Changing readout dose must not re-run physical transport")

    for name in ("acquire_stem_scan", "_real_high_angle_tail", "probe_state_from_simulation"):
        monkeypatch.setattr(stem_signal, name, no_transport)
    state.electron_gun.emitted_current_a *= 3.0
    state.ac_deflector.scan_frame_period_s *= 2.0
    updated = reweight_stem_scan(state, baseline)
    source_pa = state.electron_gun.emitted_current_a * 1.0e12
    for key in baseline.fractions:
        np.testing.assert_allclose(updated.current_pa[key], baseline.current_pa[key] * 3.0)
        np.testing.assert_allclose(updated.expected_electrons[key], baseline.expected_electrons[key] * 6.0)
        np.testing.assert_array_equal(updated.fractions[key], baseline.fractions[key])
        np.testing.assert_array_equal(baseline.poisson_counts[key], old_counts[key])
        signal = updated.detector_signals[key]
        assert signal.current_pa == pytest.approx(source_pa * signal.fraction)
        assert signal.electrons_per_second == pytest.approx(signal.current_pa * 1.0e-12 / ELEMENTARY_CHARGE_C)
        assert updated.detector_signals[key].fraction == baseline.detector_signals[key].fraction
        assert updated.detector_signals[key].current_pa == pytest.approx(baseline.detector_signals[key].current_pa * 3.0)
    assert updated.dwell_time_s == pytest.approx(2.0 * baseline.dwell_time_s)
    assert updated.metrics["scan_frame_period_s"] == state.ac_deflector.scan_frame_period_s
    assert updated.metrics["source_current_pa"] == pytest.approx(source_pa)
    assert updated.metrics["source_electrons_per_pixel"] == pytest.approx(3000.0)
    assert updated.metrics["expected_electron_unit"] == "electrons / scan pixel"
    assert updated.metrics["poisson_count_unit"] == "detected electrons / scan pixel"
    assert updated.metrics["sampling_state_signature"] == "current"
    assert frame.metrics["sampling_state_signature"] == "old"
    assert baseline.metrics["scan_frame_period_s"] == pytest.approx(0.000256)
    for name in ("scan_x_um", "scan_y_um", "probe_state", "fourdstem_artifact",
                 "uncollected_fraction", "absorbed_fraction", "truncated_fraction",
                 "high_angle_tail_fraction"):
        assert getattr(updated, name) is getattr(baseline, name)


def test_ray_history_count_is_not_the_physical_electron_dose():
    state = _state()
    frame = _frame()
    baseline = reweight_stem_scan(state, frame)
    for history_count in (15_000, 1_000_000):
        state.electron_gun.ray_count = history_count
        changed = reweight_stem_scan(state, frame)
        for key in frame.fractions:
            np.testing.assert_array_equal(changed.expected_electrons[key], baseline.expected_electrons[key])
            np.testing.assert_array_equal(changed.poisson_counts[key], baseline.poisson_counts[key])


def test_fixed_frame_duration_divides_dose_over_scan_pixels():
    state = _state(pixel_count=16 * 16)
    coarse = reweight_stem_scan(state, _frame((16, 16)))
    fine = reweight_stem_scan(state, _frame((32, 32)))
    assert fine.dwell_time_s == pytest.approx(coarse.dwell_time_s / 4.0)
    for key in coarse.fractions:
        assert float(fine.expected_electrons[key].mean()) == pytest.approx(float(coarse.expected_electrons[key].mean()) / 4.0)
        assert float(fine.expected_electrons[key].sum()) == pytest.approx(float(coarse.expected_electrons[key].sum()))


def test_noise_toggle_retains_deterministic_observables_and_handles_no_metadata():
    state = _state()
    first = reweight_stem_scan(state, _frame(metrics=None))
    state.sample.stem_poisson_enabled = False
    deterministic = reweight_stem_scan(state, first)
    assert deterministic.poisson_counts is None
    assert deterministic.metrics["poisson_seed"] is None
    for key in first.fractions:
        np.testing.assert_array_equal(deterministic.expected_electrons[key], first.expected_electrons[key])
        np.testing.assert_array_equal(deterministic.current_pa[key], first.current_pa[key])
    state.sample.stem_poisson_enabled = True
    restored = reweight_stem_scan(state, deterministic)
    for key in first.fractions:
        np.testing.assert_array_equal(restored.poisson_counts[key], first.poisson_counts[key])
