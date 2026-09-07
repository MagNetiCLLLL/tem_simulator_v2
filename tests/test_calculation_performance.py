"""Deterministic timing/readout tests; no physical acquisition is run."""
from types import SimpleNamespace

import pytest

from temsim.calculation_performance import calculation_performance_lines
from temsim import simulation_pipeline as pipeline


def test_stage_timings_work_without_progress_callback(monkeypatch):
    ticks = iter((10.0, 11.25, 14.0))
    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(ticks))
    progress = pipeline._StageProgress(["Column", "STEM"], None)
    progress.advance()
    progress.advance()
    assert progress.timings == [
        {"stage": "Column", "seconds": 1.25},
        {"stage": "STEM", "seconds": 2.75},
    ]


def test_stage_timer_includes_preparation_and_does_not_reset_on_local_updates(monkeypatch):
    ticks = iter((5.0, 7.0))
    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(ticks))
    events = []
    progress = pipeline._StageProgress(["Setup", "EDS"], lambda *x: events.append(x), started_at=1.0)
    progress.report()
    progress.advance()
    progress.update(1, 2, "Elastic")
    progress.update(0, 2, "Photons")
    progress.advance()
    assert [item["seconds"] for item in progress.timings] == [4.0, 2.0]
    assert events[-1][-1] == "Complete"


def make_result(**overrides):
    values = dict(
        cache_hit=False,
        performance={"stages": ({"stage": "Calculating the STEM detector frame", "seconds": 3.0},)},
        calculated_products=frozenset({"stem"}), reused_products=frozenset(),
        stem_scan=SimpleNamespace(metrics={
            "wave_compute_backend": "CuPy CUDA",
            "specimen_preparation_seconds": 0.125,
            "specimen_prepared_specimen_cache_hit": True,
            "cuda_transmission_cache_hits": 8,
            "cuda_transmission_cache_builds": 1,
            "cuda_transmission_cache_bytes": 1024**2,
        }),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_current_wave_backend_and_nested_cache_times_are_reported():
    result = make_result()
    assert calculation_performance_lines(result) == (
        "Timing | STEM frame: 3.000 s",
        "STEM compute | CuPy CUDA | potential: cache hit (0.125 s)",
        "STEM GPU transmission | 8 hits | 1 builds | 1.0 MiB",
    )
    assert result.stem_scan.metrics["specimen_preparation_seconds"] == 0.125


def test_complete_cache_hit_does_not_repeat_original_acquisition_times():
    assert calculation_performance_lines(make_result(cache_hit=True)) == (
        "Performance | Complete result reused; no new physics stages.",
    )


@pytest.mark.parametrize("overrides", [
    {"reused_products": frozenset({"stem"})},
    {"calculated_products": frozenset()},
    {"performance": {"stages": ({"stage": "Updating STEM dose readout", "seconds": 0.01},)}},
    {"reused_products": frozenset({"fourdstem_cube"})},
])
def test_partial_reuse_does_not_relabel_old_preparation_as_new(overrides):
    lines = calculation_performance_lines(make_result(**overrides))
    assert not any("potential:" in line or "GPU transmission" in line for line in lines)


def test_fallback_is_visible_and_legacy_missing_metrics_are_safe():
    result = make_result()
    result.stem_scan.metrics["cuda_pipeline_fallback_reason"] = "Capture requires CPU"
    assert calculation_performance_lines(result)[-1] == "STEM fallback | Capture requires CPU"
    assert calculation_performance_lines(SimpleNamespace()) == ()
    result.performance["stages"] = ({"stage": "bad", "seconds": float("nan")}, None)
    assert calculation_performance_lines(result) == ()


def test_cached_diffraction_metrics_do_not_claim_new_propagation():
    result = make_result()
    result.stem_scan.metrics["interactive_cube_reused"] = True
    assert calculation_performance_lines(result) == ("Timing | STEM frame: 3.000 s",)
    result.performance["stages"] = ({"stage": "Recollecting cached STEM diffraction", "seconds": 0.012},)
    assert calculation_performance_lines(result) == ("Timing | STEM cached diffraction readout: 0.012 s",)
