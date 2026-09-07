"""Honest stage/local progress without running physical calculations."""

import pytest

from temsim.simulation_pipeline import _StageProgress, _QT_PROGRESS_SAFE_MAX


def reporter():
    events = []
    progress = _StageProgress(
        ["Column", "EDS", "STEM", "Diagnostics"],
        lambda *event: events.append(event),
    )
    return progress, events


@pytest.mark.parametrize("histories", [1, 15_000, 1_000_000, 4_000_000_000])
def test_history_count_never_weights_eds_more_than_other_stages(histories):
    progress, events = reporter()
    progress.advance()
    progress.update(histories, histories, "Elastic histories complete")
    completed, total, label = events[-1]
    assert completed / total < 0.5
    assert label == "Stage 2/4 | Elastic histories complete | stage progress 100.0%"
    assert total <= _QT_PROGRESS_SAFE_MAX
    progress.advance()
    assert events[-1] == (20_000, 40_000, "Stage 3/4 | STEM")


def test_new_local_step_reports_real_counts_without_outer_regression():
    progress, events = reporter()
    progress.advance()
    progress.update(15_000, 15_000, "Elastic histories")
    progress.update(0, 80, "X-ray collection 0/80")
    progress.update(20, 80, "X-ray collection 20/80")
    assert events[-2][2].endswith("X-ray collection 0/80 | stage progress 0.0%")
    assert events[-1][2].endswith("X-ray collection 20/80 | stage progress 25.0%")
    positions = [done / total for done, total, _ in events]
    assert positions == sorted(positions)
    assert positions[-1] < 0.5


def test_only_finished_pipeline_reports_complete():
    progress, events = reporter()
    progress.report()
    for _ in range(3):
        progress.advance()
    progress.update(1, 1, "Diagnostics substep")
    assert events[-1][0] < events[-1][1]
    assert events[-1][2].startswith("Stage 4/4")
    progress.advance()
    assert events[-1] == (40_000, 40_000, "Complete")


def test_invalid_nested_counts_are_bounded_and_unknown_totals_are_ignored():
    progress, events = reporter()
    progress.update(-10, 2, "Before work")
    assert events[-1] == (0, 40_000, "Stage 1/4 | Before work | stage progress 0.0%")
    progress.update(10, 2, "After work")
    assert events[-1] == (9_999, 40_000, "Stage 1/4 | After work | stage progress 100.0%")
    count = len(events)
    progress.update(0, 0, "Unknown")
    assert len(events) == count


def test_callback_exceptions_preserve_cancellation_contract():
    def cancel(*_args):
        raise RuntimeError("Cancelled")

    progress = _StageProgress(["EDS"], cancel)
    with pytest.raises(RuntimeError, match="Cancelled"):
        progress.update(1, 2, "Photons")


def test_no_callback_does_not_require_progress_consumers():
    progress = _StageProgress(["EDS"], None)
    progress.report()
    progress.update(1, 2, "Photons")
    progress.advance()


def test_bank_progress_names_point_and_stage_without_global_percentage(qtbot):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QProgressBar
    from temsim.gui.interactive_calculation import InteractiveCalculationPage

    bar = QProgressBar()
    bar.setRange(0, 1000)
    qtbot.addWidget(bar)
    page = SimpleNamespace(progress=bar)
    InteractiveCalculationPage._progress(
        page, 1, 5, 15_000, 30_000,
        "Stage 2/3 | EDS photons 20/80 | stage progress 50.0%",
    )
    assert bar.value() == 300
    assert bar.text().startswith("Point 2/5 | Stage 2/3")
    assert "stage progress 50.0%" in bar.text()
    assert "%p" not in bar.format()


@pytest.mark.parametrize("eds_cached", [False, True])
def test_pipeline_cached_elastic_and_eds_omit_completed_stages(monkeypatch, eds_cached):
    from types import SimpleNamespace
    import temsim.simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    from temsim.specimen.interaction_types import SpecimenObservable

    state = default_state()
    state.sample.wave_enabled = False
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    state.energy_filter.enabled = False
    signatures = pipeline.calculation_signatures(state)
    simulation = pipeline.Simulation(incident=SimpleNamespace(), branches={}, metrics={})
    interactions = SimpleNamespace(
        elastic_transport=object(), eds_spectrum=object() if eds_cached else None,
        inelastic_distribution=object(), metrics={}, completed_observables=frozenset(),
    )
    cached = {key: signatures[key] for key in ("incident", "column", "elastic", "energy_filter")}
    if eds_cached:
        cached["eds"] = signatures["eds"]
    existing = pipeline.CalculationResult(
        simulation=simulation, energy_filter=None, specimen_interactions=interactions,
        signatures=cached,
    )
    monkeypatch.setattr(pipeline, "_eds_point_requested", lambda _state: True)
    monkeypatch.setattr(pipeline, "_geometric_specimen_transport_requested", lambda _state: False)
    monkeypatch.setattr(pipeline, "sample_illumination_absent", lambda *_args: False)
    monkeypatch.setattr(pipeline, "retain_specimen_observables", lambda *_args: interactions)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Cached ray or energy-filter propagation must not run")

    monkeypatch.setattr(pipeline, "run", forbidden)
    monkeypatch.setattr(pipeline, "simulate_energy_filter", forbidden)
    observed = []

    def fake_interactions(_state, _simulation, request, **kwargs):
        if SpecimenObservable.CHARACTERISTIC_X_RAY in request.observables:
            assert not eds_cached
            assert kwargs["existing_result"].elastic_transport is interactions.elastic_transport
            observed.append("EDS")
            callback = kwargs["progress_callback"]
            callback(0, 2, "EDS photons 0/2")
            callback(1, 2, "EDS photons 1/2")
            callback(2, 2, "EDS photons 2/2")
        return interactions

    monkeypatch.setattr(pipeline, "run_specimen_interactions", fake_interactions)
    events = []
    result = pipeline.calculate(state, existing_result=existing,
                                progress_callback=lambda *event: events.append(event))
    assert observed == ([] if eds_cached else ["EDS"])
    assert "elastic" in result.reused_products
    if eds_cached:
        assert "eds" in result.reused_products
    stages = 2 if eds_cached else 3
    assert events[0][2].startswith(f"Stage 1/{stages}")
    assert all("Transporting electrons" not in label for _, _, label in events)
    fractions = [done / total for done, total, _ in events]
    assert fractions == sorted(fractions)
    assert events[-1] == (stages * 10_000, stages * 10_000, "Complete")
    timings = result.performance["stages"]
    assert len(timings) == stages
    assert sum(item["seconds"] for item in timings) == pytest.approx(result.performance["pipeline_seconds"])
    assert all(item["seconds"] >= 0 for item in timings)
    assert all("Tracing the electron column" != item["stage"] for item in timings)
