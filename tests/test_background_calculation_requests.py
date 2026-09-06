"""Frozen tuning requests and generation-safe off-thread preparation.

Solvers are controlled here; no specimen or high-accuracy calculation is run.
"""

from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.gui import calculation_request as requests
from temsim.gui.calculation_controller import (
    CalculationController, CalculationWorker, PreparationWorker,
)
from temsim.gui.calculation_request import CapturedCalculationRequest, PreparationCancelled
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult


@pytest.fixture
def state():
    return default_state()


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
@pytest.mark.parametrize("change", ("none", "lens", "geometry", "time", "negative_zero_time"))
def test_prepared_request_matches_synchronous_identity_and_snapshot(state, quality, change):
    if change == "lens":
        state.objective_lens.percent = 61.234
    elif change == "geometry":
        # This non-authoritative runtime edit belongs to the original model tag
        # but must still be normalized by installed TOML in the solver snapshot.
        state.objective_lens.upper_b0_t += .001
    elif change == "time":
        state.simulation_time_s = 1.2345
    elif change == "negative_zero_time":
        state.simulation_time_s = -0.0
    model = state_model_signature(state)
    expected = CalculationController._calculation_snapshot(state, quality, 49, 1.)
    signatures = calculation_signatures(expected)
    request = CapturedCalculationRequest.capture(state, quality, 49, 1.)
    result = request.prepare(Event())
    assert result.model_signature == model
    assert result.request_signatures == signatures
    assert result.snapshot.to_dict() == expected.to_dict()
    assert result.snapshot is not state
    assert result.snapshot.objective_lens is not state.objective_lens
    assert result.snapshot.sample is not state.sample
    assert result.snapshot._resolved_assembly is state._resolved_assembly
    if change == "geometry":
        assert result.snapshot.objective_lens.upper_b0_t != state.objective_lens.upper_b0_t
        assert result.model_signature != state_model_signature(result.snapshot)


def test_capture_owns_nested_payload_and_original_geometry(state):
    state.probe_aberrations = {"test_nested": {"values": [1., 2.]}}
    state.lens_field_map_descriptors = {"test": {"settings": {"mu": [1., 2.]}}}
    state.electron_gun_profiles = {"test": {"values": [3., 4.]}}
    state.simulation_mode_profiles = {"test": {"settings": [5., 6.]}}
    state.simulation_time_s = 1.25
    captured = CapturedCalculationRequest.capture(state, "Preview", 49, 1.)
    original_percent = state.objective_lens.percent
    state.probe_aberrations["test_nested"]["values"][0] = 100.
    state.lens_field_map_descriptors["test"]["settings"]["mu"][0] = 100.
    state.electron_gun_profiles["test"]["values"][0] = 100.
    state.simulation_mode_profiles["test"]["settings"][0] = 100.
    state.objective_lens.percent += 1.
    state.objective_lens.upper_b0_t += 1.
    state.sample.thickness_nm = 999.
    state.simulation_time_s = 2.5
    frozen = captured._model_state
    payload = frozen.to_dict()
    assert payload["probe_aberrations"]["test_nested"]["values"] == [1., 2.]
    assert payload["lens_field_map_descriptors"]["test"]["settings"]["mu"] == [1., 2.]
    assert payload["electron_gun_profiles"]["test"]["values"] == [3., 4.]
    assert payload["simulation_mode_profiles"]["test"]["settings"] == [5., 6.]
    assert next(l for l in frozen.lenses if l.key == "objective_lens").percent == original_percent
    assert frozen.sample.thickness_nm != 999.
    assert frozen.simulation_time_s == 1.25
    # Signature normalization must never mutate the captured source payload.
    payload["probe_aberrations"]["test_nested"]["values"][0] = -1.
    assert frozen.to_dict()["probe_aberrations"]["test_nested"]["values"][0] == 1.


@pytest.mark.parametrize("phase", ("initial", "model", "snapshot", "signatures"))
def test_prepare_cancels_at_safe_phase_boundaries(state, monkeypatch, phase):
    request = CapturedCalculationRequest.capture(state, "Preview", 49, 1.)
    cancelled = Event()
    entered = []
    for name, label in (("state_model_signature", "model"),
                        ("reconstruct_calculation_state", "snapshot"),
                        ("calculation_signatures", "signatures")):
        original = getattr(requests, name)
        def call(*args, _fn=original, _label=label, **kwargs):
            entered.append(_label)
            result = _fn(*args, **kwargs)
            if phase == _label:
                cancelled.set()
            return result
        monkeypatch.setattr(requests, name, call)
    if phase == "initial":
        cancelled.set()
    with pytest.raises(PreparationCancelled):
        request.prepare(cancelled)
    expected = ["model", "snapshot", "signatures"]
    assert entered == ([] if phase == "initial" else expected[:expected.index(phase) + 1])


def _complete(controller, worker):
    result = CalculationResult(
        simulation=SimpleNamespace(payload=np.zeros(8)), energy_filter=None,
        state_snapshot=worker.state, model_signature=worker.model_signature,
        signatures=worker.request_signatures,
    )
    controller._accept_result(worker.generation, worker.quality, result, .1)
    controller._accept_finished(worker.generation, worker.quality)
    return result


def test_background_preparation_and_hashing_are_not_on_gui_thread(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    gui_thread = get_ident()
    reached, release = Event(), Event()
    threads = []
    original = CapturedCalculationRequest.prepare
    def blocked(request, cancelled):
        threads.append(get_ident())
        reached.set()
        assert release.wait(10)
        return original(request, cancelled)
    monkeypatch.setattr(CapturedCalculationRequest, "prepare", blocked)
    workers = []
    pool_start = controller.pool.start
    def start(worker):
        if isinstance(worker, PreparationWorker):
            pool_start(worker)
        else:
            workers.append(worker)
    monkeypatch.setattr(controller.pool, "start", start)
    events = []
    controller.started.connect(lambda *_: events.append("started"))
    controller.finished.connect(lambda *_: events.append("finished"))
    try:
        original_percent = state.objective_lens.percent
        controller.submit_background(state, "Preview", 49, 1.)
        assert controller.generation == 1
        assert events == ["started"]
        state.objective_lens.percent += 1.
        qtbot.waitUntil(reached.is_set)
        assert threads[0] != gui_thread
        assert events == ["started"]
        release.set()
        qtbot.waitUntil(lambda: len(workers) == 1, timeout=10000)
        assert workers[0].state.objective_lens.percent == original_percent
        assert workers[0].generation == 1
        _complete(controller, workers[0])
        assert events == ["started", "finished"]
    finally:
        release.set()
        controller.invalidate_pending()
        assert controller.pool.waitForDone(10000)


def test_background_exact_history_reuses_solver_output(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    workers, delivered, lifecycle = [], [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.started.connect(lambda *_: lifecycle.append("started"))
    controller.finished.connect(lambda *_: lifecycle.append("finished"))
    controller.result_ready.connect(lambda q, r, t: delivered.append(r))
    controller.submit(state, "Preview", 49, 1.)
    original = _complete(controller, workers.pop())
    lifecycle.clear()
    controller.submit_background(state, "Preview", 49, 1.)
    assert isinstance(workers[0], PreparationWorker)
    workers.pop().run()
    qtbot.waitUntil(lambda: len(delivered) == 2)
    assert not workers, "An exact cache hit must not start another solver"
    assert delivered[-1].cache_hit
    assert delivered[-1].simulation is original.simulation
    assert lifecycle == ["started", "finished"]
    assert controller.cache_statistics()["tuning_hits"] == 1


def test_obsolete_preparation_cannot_start_or_publish_a_solver(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    workers, delivered = [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.result_ready.connect(lambda *_: delivered.append("result"))
    controller.failed.connect(lambda *_: delivered.append("error"))
    controller.finished.connect(lambda *_: delivered.append("finished"))
    controller.submit_background(state, "Preview", 49, 1.)
    old = workers.pop()
    prepared_old = old.request.prepare(Event())
    state.objective_lens.percent += .25
    controller.submit_background(state, "Preview", 49, 1.)
    newest = workers.pop()
    assert old.cancel_event.is_set()
    old.run()  # cancellation before preparation
    controller._accept_prepared(old.generation, old.quality, prepared_old)
    controller._accept_error(old.generation, old.quality, "late error")
    controller._accept_finished(old.generation, old.quality)
    assert not workers and not delivered
    newest.run()
    assert len(workers) == 1 and isinstance(workers[0], CalculationWorker)
    _complete(controller, workers.pop())
    assert delivered == ["result", "finished"]


def test_preparation_error_finishes_once_without_clearing_results(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    workers, events = [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    retained = CalculationResult(
        simulation=SimpleNamespace(payload=np.zeros(8)), energy_filter=None,
        signatures={"request": "retained-high"},
    )
    controller._cache_result(retained)
    controller.started.connect(lambda *_: events.append("started"))
    controller.failed.connect(lambda *_: events.append("failed"))
    controller.finished.connect(lambda *_: events.append("finished"))
    def fail(*args):
        raise RuntimeError("controlled preparation failure")
    monkeypatch.setattr(CapturedCalculationRequest, "prepare", fail)
    controller.submit_background(state, "Preview", 49, 1.)
    workers.pop().run()
    assert events == ["started", "failed", "finished"]
    assert controller.completed_high_accuracy_results() == (retained,)


def test_capture_failure_does_not_start_or_cancel_current_job(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    events = []
    controller.started.connect(lambda *_: events.append("started"))
    previous_event = controller._cancel_event
    def fail(*args):
        raise TypeError("controlled invalid payload")
    monkeypatch.setattr(CapturedCalculationRequest, "capture", fail)
    with pytest.raises(ValueError, match="Could not capture"):
        controller.submit_background(state, "Preview", 49, 1.)
    assert controller.generation == 0
    assert not previous_event.is_set()
    assert events == []


def test_background_api_preserves_high_accuracy_submission_contract(monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    calls = []
    monkeypatch.setattr(controller, "submit", lambda *args: calls.append(args))
    controller.submit_background(state, "High accuracy", 15000, .1)
    assert calls == [(state, "High accuracy", 15000, .1)]


def test_cached_result_reentrant_submission_does_not_finish_new_job(qtbot, monkeypatch, state):
    controller = CalculationController(persistent_cache_enabled=False)
    workers, events = [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.submit(state, "Preview", 49, 1.)
    _complete(controller, workers.pop())
    def submit_on_result(*args):
        events.append("result")
        state.objective_lens.percent += .1
        controller.submit_background(state, "Preview", 49, 1.)
    controller.result_ready.connect(submit_on_result)
    controller.finished.connect(lambda *_: events.append("finished"))
    controller.submit_background(state, "Preview", 49, 1.)
    workers.pop().run()
    qtbot.waitUntil(lambda: events == ["result"])
    assert len(workers) == 1 and isinstance(workers[0], PreparationWorker)
    assert events == ["result"], "The old cache callback must not finish the new generation"
    controller.invalidate_pending()
