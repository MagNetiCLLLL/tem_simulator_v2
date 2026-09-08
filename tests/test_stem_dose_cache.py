"""Count readout changes reuse STEM transport and retain actual count arrays."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.calculation_cache as cache
import temsim.simulation_pipeline as pipeline
from temsim.detector.stem_signal import (
    DetectorSignal, ELEMENTARY_CHARGE_C, StemScanResult, reweight_stem_scan,
)
from temsim.optics.column import default_state
from temsim.physics.beam_current import effective_source_current_pa


@pytest.mark.parametrize("wave_enabled", [False, True])
@pytest.mark.parametrize("field,value", [
    ("stem_poisson_enabled", True), ("stem_poisson_seed", 47),
])
def test_poisson_settings_invalidate_counts_but_not_transport(wave_enabled, field, value):
    state = default_state()
    state.sample.stem_wave_enabled = wave_enabled
    state.sample.stem_poisson_enabled = False
    state.sample.stem_poisson_seed = 0
    before = cache.calculation_signatures(state)
    setattr(state.sample, field, value)
    after = cache.calculation_signatures(state)
    assert {key for key in before if before[key] != after[key]} == {"request", "stem"}
    assert before["stem_transport"] == after["stem_transport"]
    assert before["fourdstem_cube"] == after["fourdstem_cube"]


def test_frame_period_retains_real_dynamic_transport_dependency():
    state = default_state()
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.descan_deflector.enabled = state.descan_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = True
    before = cache.calculation_signatures(state)
    state.ac_deflector.scan_frame_period_s *= 2
    after = cache.calculation_signatures(state)
    # Scan time enters real deflections and sequential stop masks. It cannot
    # be globally stripped as though every frame-period change were dose-only.
    assert after["stem"] != before["stem"]
    assert after["stem_transport"] != before["stem_transport"]


@dataclass
class _CachedSimulation:
    metrics: dict


def _frame(state):
    x, y = np.meshgrid(np.arange(3)*0.001, np.arange(2)*0.001)
    fractions = {"bf": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                 "df": np.full((2, 3), 0.02)}
    source = effective_source_current_pa(state)
    signals = {key: DetectorSignal(key, key.upper(), float(values.mean()),
        float(values.mean())*state.electron_gun.ray_count,
        float(values.mean())*source,
        float(values.mean())*source*1e-12/ELEMENTARY_CHARGE_C, None)
        for key, values in fractions.items()}
    return reweight_stem_scan(state, StemScanResult(
        x, y, fractions, signals,
        metrics={"model": "multislice_angle_resolved", "detector_sampling": {},
                 "scan_frame_period_s": state.ac_deflector.scan_frame_period_s},
        high_angle_tail_fraction={"bf": np.zeros_like(x)},
    ))


def _isolate_pipeline(monkeypatch):
    for name in ("ensure_recording_system", "ensure_energy_filter",
                 "ensure_corrector_structure", "normalise_component_names",
                 "apply_physical_layout_to_state"):
        monkeypatch.setattr(pipeline, name, lambda _state: None)
    monkeypatch.setattr(pipeline, "sample_illumination_absent", lambda *_: False)
    monkeypatch.setattr(pipeline, "tem_wave_imaging_enabled", lambda *_: False)
    monkeypatch.setattr(pipeline, "_eds_point_requested", lambda *_: False)
    monkeypatch.setattr(pipeline, "_geometric_specimen_transport_requested", lambda *_: False)
    monkeypatch.setattr(pipeline, "run_specimen_interactions", lambda *_a, **_k: None)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Dose-only update must not recalculate source, specimen, waves or scan transport")
    for name in ("run", "calculate_stem_scan_frame", "acquire_stem_scan",
                 "reproject_wave_image", "calculate_scan_geometry", "calculate_scan_ray_paths"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr(pipeline, "simulate_energy_filter", lambda *_a, **_k: None)
    monkeypatch.setattr("temsim.physics.stem_wave_imaging.simulate_angle_resolved_stem", forbidden)


@pytest.mark.parametrize("change", ["enable", "disable", "seed", "current"])
def test_pipeline_reweights_counts_without_repeating_transport(monkeypatch, change):
    state = default_state()
    state.sample.stem_wave_enabled = True
    state.sample.stem_poisson_enabled = change != "enable"
    state.sample.stem_poisson_seed = 7
    state.sample.eds_enabled = False
    state.energy_filter.enabled = False
    state.column_current_limit_percent = 1e-6
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_frame_period_s = 0.006
    state.ac_deflector.scan_pixels_x = 3
    state.ac_deflector.scan_lines = 2
    state.probe_aberrations = state.image_aberrations = {}
    old = _frame(state)
    old_expected = {key: value.copy() for key, value in old.expected_electrons.items()}
    old_counts = None if old.poisson_counts is None else {key: value.copy() for key,value in old.poisson_counts.items()}
    old_signatures = cache.calculation_signatures(state)
    existing = pipeline.CalculationResult(
        simulation=_CachedSimulation({"sample_beam_surviving_fraction": 1.0}),
        energy_filter=None, state_snapshot=state,
        scan_geometry=object(), scan_ray_paths=object(), stem_scan=old,
        signatures=old_signatures,
    )
    if change == "enable":
        state.sample.stem_poisson_enabled = True
    elif change == "disable":
        state.sample.stem_poisson_enabled = False
    elif change == "seed":
        state.sample.stem_poisson_seed = 47
    else:
        state.column_current_limit_percent *= 2
    _isolate_pipeline(monkeypatch)
    reweights = []
    def reweight(current_state, frame):
        reweights.append(frame)
        return reweight_stem_scan(current_state, frame)
    monkeypatch.setattr(pipeline, "reweight_stem_scan", reweight)
    result = pipeline.calculate(state, existing_result=existing)
    assert reweights == [old]
    assert "stem_transport" in result.reused_products
    assert "stem" in result.calculated_products
    assert "column" in result.reused_products
    assert result.stem_scan is not old
    for key in old.fractions:
        assert result.stem_scan.fractions[key] is old.fractions[key]
        np.testing.assert_array_equal(old.expected_electrons[key], old_expected[key])
        if old_counts is not None:
            np.testing.assert_array_equal(old.poisson_counts[key], old_counts[key])
        np.testing.assert_allclose(result.stem_scan.expected_electrons[key],
            old_expected[key]*(2 if change == "current" else 1))
    assert result.stem_scan.high_angle_tail_fraction is old.high_angle_tail_fraction
    assert result.stem_scan.metrics["sampling_state_signature"] == result.signatures["stem"]
    if state.sample.stem_poisson_enabled:
        rng = np.random.default_rng(state.sample.stem_poisson_seed)
        for key, expected in result.stem_scan.expected_electrons.items():
            np.testing.assert_array_equal(result.stem_scan.poisson_counts[key], rng.poisson(expected))
    else:
        assert result.stem_scan.poisson_counts is None
    source = effective_source_current_pa(state)
    for key, signal in result.stem_scan.detector_signals.items():
        assert signal.current_pa == pytest.approx(source*old.fractions[key].mean())
        assert signal.electrons_per_second == pytest.approx(signal.current_pa*1e-12/ELEMENTARY_CHARGE_C)


def test_bank_readout_retains_actual_completed_count_arrays(monkeypatch):
    import temsim.interactive_calculation as interactive
    state = default_state()
    state.sample.stem_poisson_enabled = True
    state.sample.stem_poisson_seed = 41
    state.column_current_limit_percent = 1e-6
    frame = _frame(state)
    lens = state.lenses[0]
    control = interactive.RangeControl("lens", lens.key, "percent", "Lens", "%", lens.percent, "optical")
    plan = interactive.InteractivePlan((interactive.CalculationRange(control, lens.percent-1, lens.percent+1, 3),), 1_000_000)
    completed = pipeline.CalculationResult(None, None, state_snapshot=state, stem_scan=frame)
    bank = interactive.InteractiveBank(plan, state, "unchanged",
        (interactive.BankPoint((lens.percent,), completed, ()),), 1)
    monkeypatch.setattr(interactive, "external_model_signature", lambda *_: "unchanged")
    monkeypatch.setattr(interactive, "calculate", lambda *_a, **_k: pytest.fail("Bank display must not recalculate"))
    readout = interactive.read_bank(bank, {control.identity:lens.percent})
    assert readout.stem is frame
    assert readout.stem.expected_electrons is frame.expected_electrons
    assert readout.stem.poisson_counts is frame.poisson_counts
    assert readout.state_snapshot.sample.stem_poisson_enabled is True
    assert readout.state_snapshot.sample.stem_poisson_seed == 41


def test_bank_changed_stop_recollection_refreshes_expected_and_poisson_counts(monkeypatch):
    import temsim.interactive_calculation as interactive
    state = default_state()
    state.sample.stem_poisson_enabled = True
    state.sample.stem_poisson_seed = 41
    state.column_current_limit_percent = 1e-6
    frame = _frame(state)
    artifact = SimpleNamespace(data=np.ones((2, 3, 1, 1)),
        calibration=SimpleNamespace(scan_times_s=np.zeros((2, 3))))
    frame = replace(frame, fourdstem_artifact=artifact, metrics={**frame.metrics,
        "incident_sample_fraction": 0.5, "tracked_probability_after_inelastic_absorption": 0.8})
    bf = next(detector for detector in state.stem_detectors if detector.key == "bf")
    bf.inserted = bf.readout_enabled = True
    width = bf.outer_width_mm
    control = interactive.RangeControl("detector", bf.key, "outer_width_mm", "BF width", "mm", width, "readout")
    plan = interactive.InteractivePlan((interactive.CalculationRange(control, width*.9, width*1.1, 3),), 1_000_000)
    completed = pipeline.CalculationResult(None, None, state_snapshot=state, stem_scan=frame)
    bank = interactive.InteractiveBank(plan, state, "unchanged",
        (interactive.BankPoint((), completed, ()),), 1)
    monkeypatch.setattr(interactive, "external_model_signature", lambda *_: "unchanged")
    monkeypatch.setattr(interactive, "calculate", lambda *_a, **_k: pytest.fail("Bank stop readout must not rerun propagation"))
    monkeypatch.setattr("temsim.physics.record_plane.build_record_plane_plan", lambda *_a, **_k: "test-plan")
    # Isolate already-verified geometric integration, retaining production bank
    # recollection, source-fraction scaling, and statistical readout creation.
    def integrate(given_artifact, response, given_plan):
        assert given_artifact is artifact and given_plan == "test-plan"
        return SimpleNamespace(images={bf.key: np.full((2, 3), 0.25)},
            surviving_weight=np.full((2, 3), 0.75), plan_fingerprint="changed-stop")
    monkeypatch.setattr("temsim.physics.fourdstem.integrate_runtime_recording_planes", integrate)
    readout = interactive.read_bank(bank, {control.identity:width*1.05})
    assert readout.stem.fourdstem_artifact is artifact
    np.testing.assert_allclose(readout.stem.fractions[bf.key], 0.25*0.5*0.8)
    expected = np.full((2, 3), 0.1*effective_source_current_pa(state)*1e-12*frame.dwell_time_s/ELEMENTARY_CHARGE_C)
    np.testing.assert_allclose(readout.stem.expected_electrons[bf.key], expected)
    rng = np.random.default_rng(41)
    np.testing.assert_array_equal(readout.stem.poisson_counts[bf.key], rng.poisson(expected))
    assert readout.stem.poisson_counts is not frame.poisson_counts
    assert frame.fractions[bf.key][0, 1] == 0.2
    assert bf.outer_width_mm == width
