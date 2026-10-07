"""Explicit page requests, immutable capture and independent result cache keys."""
from threading import Event
from types import SimpleNamespace

import pytest

from temsim.calculation_cache import calculation_signatures
from temsim.calculation_workflow import admit_workflow, workflow_signatures
from temsim.gui.calculation_controller import CalculationController, CalculationWorker, PreparationWorker
from temsim.gui.calculation_request import CapturedCalculationRequest
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult


def test_page_capture_preserves_physical_settings_and_incident_identity():
    state = default_state()
    state.sample.wave_enabled = True  # Unrequested wave work must not block rays.
    state.sample.eds_enabled = True
    original = state.to_dict()
    captured = CapturedCalculationRequest.capture(state, "High accuracy", 25, 1., workflow="rays")
    prepared = captured.prepare(Event())
    raw = calculation_signatures(prepared.snapshot)
    assert prepared.snapshot.sample.wave_enabled
    assert prepared.snapshot.sample.eds_enabled
    assert state.to_dict() == original
    assert prepared.request_signatures["incident"] == raw["incident"]
    assert prepared.request_signatures["request"] != raw["request"]
    assert prepared.request_signatures["workflow"] == "rays"


def test_readout_intent_has_distinct_request_identity():
    raw = {"request": "physical", "incident": "executed-prefix", "column": "column"}
    signatures = [workflow_signatures(raw, key) for key in ("full", "rays", "sample", "eds", "stem", "receiver")]
    assert len({entry["request"] for entry in signatures}) == 6
    assert {entry["incident"] for entry in signatures} == {"executed-prefix"}
    assert raw == {"request": "physical", "incident": "executed-prefix", "column": "column"}


def test_wave_gate_is_preserved_only_for_requested_wave_product():
    from temsim.physics.source_admission import UnsupportedWaveSource
    state = default_state()
    state.sample.wave_enabled = True
    for workflow in ("rays", "sample", "eds", "sample_region", "energy_filter", "receiver"):
        admit_workflow(state, workflow)
    with pytest.raises(UnsupportedWaveSource):
        admit_workflow(state, "imaging")
    state.sample.stem_wave_enabled = True
    with pytest.raises(UnsupportedWaveSource):
        admit_workflow(state, "stem")


def test_explicit_seed_and_scope_survive_background_preparation(qapp, monkeypatch):
    state = default_state()
    controller = CalculationController(persistent_cache_enabled=False)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    seed = CalculationResult(simulation=SimpleNamespace(), energy_filter=None)
    controller.submit_background(state, "High accuracy", 25, 1., workflow="eds", existing_result=seed)
    preparation = workers.pop()
    assert isinstance(preparation, PreparationWorker)
    preparation.run()
    assert len(workers) == 1
    worker = workers.pop()
    assert isinstance(worker, CalculationWorker)
    assert worker.workflow == "eds" and worker.existing_result is seed
    assert worker.request_signatures["workflow"] == "eds"
    controller.invalidate_pending(include_explicit=True)


def test_worker_passes_scope_and_never_loads_unrelated_incident_cache(qapp, monkeypatch):
    from temsim.gui import calculation_controller as module
    state = default_state()
    seed = CalculationResult(simulation=None, energy_filter=None)
    calls, results, errors = [], [], []
    def compute(snapshot, **kwargs):
        calls.append(kwargs)
        return CalculationResult(simulation=None, energy_filter=None, state_snapshot=snapshot,
                                 signatures={"incident": "actual-prefix"})
    monkeypatch.setattr(module, "calculate", compute)
    worker = CalculationWorker(1, "High accuracy", state, workflow="eds", existing_result=seed,
                               request_signatures={"request": "eds-request", "workflow": "eds"})
    monkeypatch.setattr(worker, "_load_persistent_incident_seed", lambda *_: pytest.fail("Unrelated disk seed"))
    worker.signals.result.connect(lambda _g, _q, value, _t: results.append(value))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert not errors
    assert calls[0]["workflow"] == "eds" and calls[0]["existing_result"] is seed
    assert results[0].signatures == {"request": "eds-request", "workflow": "eds", "incident": "actual-prefix"}


def test_toolbar_and_live_section_are_separate_requests():
    from temsim.gui.main_window import MainWindow
    calls = []
    page = SimpleNamespace(segment_request=lambda: {"target_z_mm": 1600., "component_keys": ()})
    window = SimpleNamespace(_submit_high_accuracy=lambda **kw: calls.append(kw),
                             workspace=SimpleNamespace(interactive_calculation=page))
    MainWindow.run_high_accuracy(window)
    MainWindow.run_section_high_accuracy(window)
    assert calls == [{"workflow": "rays"}, {"section_request": page.segment_request()}]


def test_page_uses_displayed_high_accuracy_seed_and_reports_missing_beam():
    from temsim.gui.main_window import MainWindow
    calls, errors = [], []
    window = SimpleNamespace(_submit_high_accuracy=lambda **kw: calls.append(kw),
                             _show_error=errors.append, workspace=SimpleNamespace(_high_accuracy_result=None))
    MainWindow.run_page_calculation(window, "stem")
    assert not calls and "Ray Diagram" in errors[-1]
    seed = SimpleNamespace(simulation=object())
    window.workspace._high_accuracy_result = seed
    MainWindow.run_page_calculation(window, "stem")
    assert calls == [{"workflow": "stem", "existing_result": seed}]


def test_ray_request_draws_optical_reference_even_with_retained_readouts():
    from temsim.gui.beam_display_source import downstream_display_branches
    ray = object()
    result = SimpleNamespace(workflow="rays", simulation=SimpleNamespace(branches={"000": ray}),
                             specimen_exit=object(), sample_region=object())
    assert downstream_display_branches(result) == ((ray,), "Optical reference")
