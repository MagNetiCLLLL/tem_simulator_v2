"""Controller ownership and terminal-state tests without a physical solver."""
from types import SimpleNamespace

import pytest

from temsim.gui import direct_alignment_controller as module


@pytest.fixture
def submitted(qtbot, monkeypatch):
    controller = module.DirectAlignmentController()
    request = SimpleNamespace(key="alignment", target=1., registry_digest="registry",
                              options=None, start_snapshot=SimpleNamespace(digest="captured"))
    monkeypatch.setattr(module.AlignmentRequest, "capture", lambda *args, **kwargs: request)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.submit(object(), "alignment", 1.)
    return controller, workers


def test_finished_alignment_cannot_publish_again_or_finish_twice(submitted):
    controller, workers = submitted
    generation = workers[0].generation
    results, failures, finished = [], [], []
    controller.result_ready.connect(lambda *args: results.append(args))
    controller.failed.connect(lambda *args: failures.append(args))
    controller.finished.connect(finished.append)
    controller._accept_result(generation, "alignment", "candidate", .1)
    controller._accept_finished(generation, "alignment")
    controller._accept_result(generation, "alignment", "late candidate", .2)
    controller._accept_error(generation, "alignment", "late failure")
    controller._accept_finished(generation, "alignment")
    assert results == [("alignment", "candidate", .1)]
    assert not failures
    assert finished == ["alignment"]


def test_finished_callback_can_submit_next_alignment(submitted):
    controller, workers = submitted
    generation = workers[0].generation
    finished = []
    controller.finished.connect(lambda key: (finished.append(key), controller.submit(object(), key, 2.)))
    controller._accept_finished(generation, "alignment")
    assert len(workers) == 2
    current = workers[1].generation
    received = []
    controller.result_ready.connect(lambda *args: received.append(args))
    controller._accept_result(generation, "alignment", "obsolete", .1)
    controller._accept_result(current, "alignment", "current", .1)
    assert received == [("alignment", "current", .1)]
    assert finished == ["alignment"]


def test_capture_failure_preserves_active_alignment(submitted, monkeypatch):
    controller, workers = submitted
    cancellation = controller._cancellation
    def fail(*args, **kwargs):
        raise ValueError("invalid target")
    monkeypatch.setattr(module.AlignmentRequest, "capture", fail)
    with pytest.raises(ValueError, match="invalid target"):
        controller.submit(object(), "alignment", 1.)
    assert not cancellation.is_set()
    received = []
    controller.result_ready.connect(lambda *args: received.append(args))
    controller._accept_result(workers[0].generation, "alignment", "candidate", .1)
    assert received == [("alignment", "candidate", .1)]
