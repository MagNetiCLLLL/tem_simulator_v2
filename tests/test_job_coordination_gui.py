"""Captured High completion never replaces a newer live instrument/display."""
from types import SimpleNamespace
from threading import Event

from temsim.calculation_cache import state_model_signature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.working_point import WorkingPointCheckpoint
from test_working_point_restore_gui import window


def test_old_high_result_is_published_under_its_inputs_without_live_display_write(window, monkeypatch):
    snapshot = capture_instrument_snapshot(window.state)
    model = state_model_signature(window.state)
    point = WorkingPointCheckpoint(snapshot, {}, window.state.sample.z_mm, snapshot.physical_digest, {})
    received = []
    monkeypatch.setattr(WorkingPointCheckpoint, "from_result", lambda result, **kwargs: (received.append(kwargs), point)[1])
    monkeypatch.setattr(window.workspace, "display_result", lambda *args: received.append("displayed"))
    previous = getattr(window, "_active_working_checkpoint", None)
    window.state.sample.thickness_nm += 1
    changed = capture_instrument_snapshot(window.state)
    result = SimpleNamespace(calculation_manifest=object(), model_signature=model, working_point_parent_id="captured-parent")
    window._calculation_ready("High accuracy", result, .1)
    assert received == [{"parent_id": "captured-parent"}]
    assert window.working_points.selected is point
    assert getattr(window, "_active_working_checkpoint", None) is previous
    assert capture_instrument_snapshot(window.state).digest == changed.digest
    assert "earlier settings" in window.status_label.text()


def test_explicit_high_queues_during_experiment_and_cancel_is_scoped(window, monkeypatch):
    calls = []
    window.design_sweeps._running = True
    independent = Event()
    window.design_sweeps._cancel_event = independent
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args, **kwargs: calls.append(args))
    window.run_high_accuracy()
    assert len(calls) == 1 and calls[0][1] == "High accuracy"
    window._cancel_calculations()
    assert not independent.is_set()
    window.design_sweeps._running = False
