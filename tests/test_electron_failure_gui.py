"""Synthetic lifecycle evidence only: no field solve or numerical child process."""
from threading import Event
from dataclasses import replace

import pytest

from temsim.optics.model import State
from temsim.test_electron_execution import ElectronExecutionCancelled, ElectronExecutionError
from test_magnetic_test_electron_gui import (
    FakeIsolatedExecution, controller, electron_record, synthetic_trajectory, wait_for_result,
)


def failure_report(*, stage="trace", metadata=None):
    return {
        "schema": "electron-worker-diagnostics-v1", "run_id": "synthetic-run", "sequence": 7,
        "stage": stage, "backend": "synthetic GUI fixture", "exception_type": "RuntimeError",
        "message": "Synthetic diagnostic failure", "traceback": "Traceback: retained synthetic frame",
        "reason": "synthetic_failure", "exit_code": 29,
        "physical_identity": "physical-fixture", "numerical_identity": "numerical-fixture",
        "parameter_revision": (metadata or {}).get("parameter_revision"),
        "request_metadata": dict(metadata or {}),
        "stderr_tail": "Synthetic stderr marker", "log_path": r"C:\Users\PrivateUser\run\stderr.log",
    }


class FailureBackend(FakeIsolatedExecution):
    """Inject terminal failures/cancellation through the real Qt worker adapter."""

    def __init__(self):
        super().__init__()
        self.prepare_failure = False
        self.trace_failure = None
        self.restart_count = 0
        self.close_count = 0
        self.prepare_metadata = []
        self.trace_metadata = []
        self.entered = Event()
        self.release = Event()
        self.block_next = False
        self.cancel_loses_fields = False
        self.ignore_cancel = False

    def prepare(self, state, magnetic_scene, *, z_limits_mm, cancelled, request_metadata=None):
        self.prepare_metadata.append(request_metadata)
        if self.prepare_failure:
            self.prepare_failure = False
            self.alive = False
            raise ElectronExecutionError("Synthetic preparation failed", fields_lost=True,
                diagnostic=failure_report(stage="prepare", metadata=request_metadata))
        return super().prepare(state, magnetic_scene, z_limits_mm=z_limits_mm,
                               cancelled=cancelled, request_metadata=request_metadata)

    def trace(self, scene, settings, *, cancelled, request_metadata=None, **_kwargs):
        self.traces.append((scene, settings))
        self.trace_metadata.append(request_metadata)
        if self.block_next:
            self.block_next = False
            self.entered.set()
            assert self.release.wait(5.), "GUI test did not release the synthetic worker"
            if cancelled() and not self.ignore_cancel:
                self.alive = not self.cancel_loses_fields
                raise ElectronExecutionCancelled("Synthetic cancellation timeout" if self.cancel_loses_fields
                    else "Synthetic cooperative cancellation", fields_lost=self.cancel_loses_fields,
                    diagnostic=failure_report(metadata=request_metadata) if self.cancel_loses_fields else None)
        if self.trace_failure is not None:
            lost, self.trace_failure = self.trace_failure, None
            self.alive = not lost
            raise ElectronExecutionError("Synthetic trace failed", fields_lost=lost,
                diagnostic=failure_report(metadata=request_metadata))
        return synthetic_trajectory(settings)

    def restart(self):
        self.restart_count += 1
        self.alive = False
        self.identity = None

    def close(self):
        self.close_count += 1
        self.alive = False


def install_backend(controller, monkeypatch):
    backend = FailureBackend()
    controller._execution_backend = backend

    def no_local_execution(*_args, **_kwargs):
        raise AssertionError("Captured State work must use the synthetic isolated owner")

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", no_local_execution)
    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", no_local_execution)
    controller.set_captured_scene(State.__new__(State), None, (0., 100.))
    return backend


def wait_for_failure(qtbot, controller):
    qtbot.waitUntil(lambda: controller._worker is None and controller._scene_worker is None
                   and bool(controller._scene_error or (controller.selected_record and controller.selected_record.error)),
                   timeout=10000)


def test_prepare_failure_has_copyable_details_and_explicit_bounded_retry(controller, qtbot, monkeypatch):
    backend = install_backend(controller, monkeypatch)
    backend.prepare_failure = True
    controller.set_active(True)
    wait_for_failure(qtbot, controller)
    qtbot.wait(200)
    assert len(backend.prepare_metadata) == 1 and not backend.traces
    assert controller.failure_details.isReadOnly()
    assert "retained synthetic frame" in controller.failure_details.toPlainText()
    assert "Traceback" not in controller.status_text
    assert "lost" in controller.status_text
    assert controller.retry_button.isEnabled() and controller.export_diagnostic_button.isEnabled()
    assert not controller._timer.isActive()
    controller.retry_button.click()
    wait_for_result(qtbot, controller)
    assert backend.restart_count == 1 and len(backend.prepare_metadata) == 2
    assert not controller._scene_error and not controller.retry_button.isEnabled()
    assert "retained synthetic frame" in controller.failure_details.toPlainText()
    assert all(item["generation"] == controller._generation for item in backend.prepare_metadata)


@pytest.mark.parametrize("fields_lost", [False, True])
def test_retry_retains_other_records_and_only_rebuilds_lost_fields(controller, qtbot, monkeypatch, fields_lost):
    backend = install_backend(controller, monkeypatch)
    controller.set_active(True)
    retained = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.duplicate_electron()
    controller.energy.setValue(.7)
    wait_for_result(qtbot, controller)
    previous_scene = controller._scene
    backend.trace_failure = fields_lost
    controller.energy.setValue(.9)
    failed_key = controller.selected_record.key
    wait_for_failure(qtbot, controller)
    before_traces = len(backend.traces)
    qtbot.wait(200)
    assert len(backend.traces) == before_traces
    assert electron_record(controller, first_key).trajectory is retained
    metadata = backend.trace_metadata[-1]
    assert metadata == {"generation": controller._generation, "record_id": failed_key,
                        "parameter_revision": controller.selected_record.revision}
    controller.retry_button.click()
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == .9
    assert electron_record(controller, first_key).trajectory is retained
    assert backend.restart_count == int(fields_lost)
    assert len(backend.preparations) == 1 + int(fields_lost)
    assert not controller.selected_record.error
    if fields_lost:
        assert controller._scene.process_identity != previous_scene.process_identity
        assert controller._scene.token != previous_scene.token
    else:
        assert controller._scene is previous_scene


@pytest.mark.parametrize("fields_lost", [False, True])
def test_cancel_preserves_fields_or_reports_the_forced_loss(controller, qtbot, monkeypatch, fields_lost):
    backend = install_backend(controller, monkeypatch)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    previous_scene = controller._scene
    backend.block_next = True
    backend.cancel_loses_fields = fields_lost
    controller.energy.setValue(.8)
    try:
        qtbot.waitUntil(backend.entered.is_set, timeout=10000)
        controller.set_active(False)
    finally:
        backend.release.set()
    qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
    assert controller.current_trajectory is None
    assert not controller._timer.isActive()
    assert backend.owns_scene(previous_scene) is not fields_lost
    if fields_lost:
        assert controller._fields_lost and "lost" in controller.status_text
        assert "Synthetic stderr marker" in controller.failure_details.toPlainText()
        assert controller.export_diagnostic_button.isEnabled()
    else:
        assert not controller._fields_lost and controller._diagnostic is None
    controller.set_active(True)  # Reopening is an existing explicit recovery action.
    wait_for_result(qtbot, controller)
    assert len(backend.preparations) == 1 + int(fields_lost)


def test_diagnostic_export_uses_captured_report_after_retry(controller, qtbot, monkeypatch, tmp_path):
    backend = install_backend(controller, monkeypatch)
    backend.prepare_failure = True
    controller.set_active(True)
    wait_for_failure(qtbot, controller)
    captured = controller._diagnostic
    controller.retry_button.click()
    wait_for_result(qtbot, controller)
    destination = tmp_path / "diagnostic.json"
    assert controller.export_diagnostic(destination) == destination
    text = destination.read_text(encoding="utf-8")
    assert "PrivateUser" not in text
    assert "synthetic-run" in text and "retained synthetic frame" in text
    assert controller._diagnostic == captured


@pytest.mark.parametrize("action", ["delete", "change_fields", "hide_view", "shutdown"])
def test_late_progress_and_terminal_result_do_not_revive_cancelled_work(controller, qtbot, monkeypatch, action):
    backend = install_backend(controller, monkeypatch)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    backend.block_next = True
    backend.ignore_cancel = True
    controller.energy.setValue(.8)
    try:
        qtbot.waitUntil(backend.entered.is_set, timeout=10000)
        key = controller._worker.key
        if action == "delete":
            controller.remove_electron()
        elif action == "change_fields":
            controller.set_captured_scene(State.__new__(State), None, (0., 50.))
            controller.set_active(False)
        elif action == "hide_view":
            controller.set_active(False)
        else:
            controller.shutdown()
            controller.set_active(True)  # Permanent close must not reopen the owner.
        controller._progressed(key, replace(synthetic_trajectory(key[1], reason="in_progress"), completed=False))
        assert controller.current_trajectory is None
    finally:
        backend.release.set()
    qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
    assert (key[0], key[1]) not in controller._cache
    assert all(record.progress_trajectory is None for record in controller.records)
    assert controller.current_trajectory is None and not controller._timer.isActive()
    if action == "delete":
        assert not controller.records
    if action == "shutdown":
        assert backend.close_count == 1 and not controller._active
        assert not controller.retry_failed_execution()


def test_shutdown_during_preparation_rejects_late_scene_and_cannot_reopen(controller, qtbot, monkeypatch):
    backend = install_backend(controller, monkeypatch)
    original_prepare = backend.prepare

    def delayed_prepare(*args, **kwargs):
        scene = original_prepare(*args, **kwargs)
        backend.entered.set()
        assert backend.release.wait(5.), "GUI test did not release the synthetic preparation"
        return scene

    monkeypatch.setattr(backend, "prepare", delayed_prepare)
    controller.set_active(True)
    try:
        qtbot.waitUntil(backend.entered.is_set, timeout=10000)
        controller.shutdown()
        controller.set_active(True)
    finally:
        backend.release.set()
    qtbot.waitUntil(lambda: controller._scene_worker is None, timeout=10000)
    assert controller._scene is None and not controller.records
    assert backend.close_count == 1 and not backend.traces
    assert not controller._active and not controller._timer.isActive()


def test_previous_worker_signals_cannot_finish_retry_with_identical_parameters(controller, qtbot, monkeypatch):
    backend = install_backend(controller, monkeypatch)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    backend.trace_failure = False
    backend.block_next = True
    controller.energy.setValue(.8)
    try:
        qtbot.waitUntil(backend.entered.is_set, timeout=10000)
        old_signals = controller._worker.signals
        old_key = controller._worker.key
    finally:
        backend.release.set()
    wait_for_failure(qtbot, controller)
    backend.block_next = True
    backend.entered.clear()
    backend.release.clear()
    controller.retry_button.click()
    try:
        qtbot.waitUntil(backend.entered.is_set, timeout=10000)
        current_worker = controller._worker
        assert current_worker.key == old_key  # Parameter identity really is unchanged.
        old_signals.progress.emit(old_key, replace(synthetic_trajectory(old_key[1], reason="in_progress"), completed=False))
        old_signals.finished.emit(old_key, synthetic_trajectory(old_key[1]), None)
        qtbot.wait(30)
        assert controller._worker is current_worker
        assert controller.current_trajectory is None and controller.selected_record.progress_trajectory is None
    finally:
        backend.release.set()
    assert wait_for_result(qtbot, controller).kinetic_energy_ev[0] == .8


def test_restart_cleanup_failure_remains_visible_without_a_retry_loop(controller, qtbot, monkeypatch):
    backend = install_backend(controller, monkeypatch)
    backend.prepare_failure = True
    controller.set_active(True)
    wait_for_failure(qtbot, controller)

    def fail_restart():
        raise ElectronExecutionError("Synthetic cleanup could not finish", fields_lost=True,
                                      diagnostic=failure_report())

    monkeypatch.setattr(backend, "restart", fail_restart)
    assert not controller.retry_failed_execution()
    assert "cleanup could not finish" in controller.status_text
    assert len(backend.prepare_metadata) == 1 and not controller._timer.isActive()
