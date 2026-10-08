"""Bounded real design calculation and GUI cancellation/state ownership."""
import os
from pathlib import Path

import numpy as np
import pytest

from temsim.gui.condenser_scan_design import CondenserScanDesignPanel, _DesignWorker
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


def _small_state():
    state = default_state()
    state.step_mm = 0.2
    state.acceleration_backend = "CPU"
    return state


@pytest.fixture
def panel(qtbot):
    widget = CondenserScanDesignPanel()
    qtbot.addWidget(widget)
    widget.set_state(_small_state())
    yield widget
    assert widget.shutdown()


def test_real_bounded_worker_populates_results_without_changing_live_state(panel, qtbot):
    # Five CM positions, both AC responses and three AC position candidates
    # per mode. Keep the product's default enabled search and backend policy.
    panel._state.acceleration_backend = "Auto"
    panel._state.acceleration_enabled = True
    assert panel.search_ac.isChecked()
    before = capture_instrument_snapshot(panel._state)
    panel._calculate()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=60000)
    assert panel._result is not None, panel.status.text()
    current, search, scans, identity = panel._result
    assert identity == before.physical_digest
    assert capture_instrument_snapshot(panel._state).digest == before.digest
    assert len(search.candidates) == panel.table.rowCount() == 5
    assert len(panel.plot.listDataItems()) == 4
    assert current.excitation_magnitude_percent == panel._state.mini_condenser.percent
    assert np.isfinite(search.best.common_input_correlation_mismatch_rad_per_m)
    assert len(scans) == 2
    for _, result, positions in scans:
        assert positions is not None and len(positions.candidates) == 3
        assert positions.best is not None
        assert positions.upper_z_range_mm == (panel.ac_min.value(), panel.ac_max.value())
        assert result.controllable
        assert result.response_model == 'finite_coil'
        assert np.isfinite(result.upper_kick_matrix_mrad).all()
        assert result.position_relative_residual < 1e-7
        assert result.angle_relative_residual < 1e-7
    assert "mrad / full-FOV raster factor" in panel.details.toPlainText()
    assert "static AC alignment and other captured affine drives are retained" in panel.details.toPlainText()
    assert "not certify" in panel.status.text()
    assert panel.calculate.isEnabled() and not panel.cancel_button.isEnabled()
    artifact = os.environ.get("TEMSIM_GUI_ARTIFACT")
    if artifact:
        path = Path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.resize(1500, 900)
        panel.show()
        qtbot.wait(30)
        assert panel.grab().save(str(path))
    retained = panel._result
    panel._state.mini_condenser.percent += 1
    panel.set_state(panel._state)
    assert panel._result is retained
    assert "inputs changed" in panel.status.text()


def test_worker_cancelled_before_entry_emits_only_finished(qapp):
    worker = _DesignWorker(capture_instrument_snapshot(_small_state()), "mechanical", None)
    results, errors, finished = [], [], []
    worker.signals.ready.connect(results.append)
    worker.signals.failed.connect(errors.append)
    worker.signals.finished.connect(lambda: finished.append(True))
    worker.cancel_event.set()
    worker.run()
    assert results == errors == []
    assert finished == [True]


def test_cancellation_during_real_cm_calculation_stops_before_scan(monkeypatch, qapp):
    import temsim.optics.condenser_objective_design as cm_design
    import temsim.optics.scan_coil_design as scan_design

    worker = _DesignWorker(capture_instrument_snapshot(_small_state()), "mechanical", None)
    real_evaluate = cm_design.evaluate_condenser_objective_design
    evaluated, results, errors, finished = [], [], [], []

    def cancel_after_current(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        evaluated.append(result)
        worker.cancel_event.set()
        return result

    def unexpected_scan(*args, **kwargs):
        pytest.fail("A cancelled CM study must not begin the AC study")

    monkeypatch.setattr(cm_design, "evaluate_condenser_objective_design", cancel_after_current)
    monkeypatch.setattr(scan_design, "evaluate_scan_coil_design", unexpected_scan)
    worker.signals.ready.connect(results.append)
    worker.signals.failed.connect(errors.append)
    worker.signals.finished.connect(lambda: finished.append(True))
    worker.run()
    assert len(evaluated) == 1
    assert results == errors == []
    assert finished == [True]


def test_panel_cancellation_drops_late_result_and_restores_controls(panel, monkeypatch):
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel._calculate()
    worker = queued[0]
    panel.cancel()
    assert worker.cancel_event.is_set()
    # A result already queued by a worker must not be rendered after Cancel.
    worker.signals.ready.emit(object())
    worker.signals.finished.emit()
    assert panel._result is None
    assert panel._worker is None
    assert panel.calculate.isEnabled() and not panel.cancel_button.isEnabled()
    assert "cancel" in panel.status.text().lower()


def test_changed_applied_inputs_invalidate_pending_design(panel, monkeypatch):
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel._calculate()
    worker = queued[0]
    panel._state.mini_condenser.percent += 1.0
    panel.set_state(panel._state)
    # The mutable live object is reused by ordinary hardware edits, so object
    # identity cannot establish that the captured calculation is still current.
    assert worker.cancel_event.is_set()
    worker.signals.ready.emit(object())
    worker.signals.finished.emit()
    assert panel._result is None
    assert panel.calculate.isEnabled()


def test_publish_rechecks_live_inputs_even_without_refresh(panel, monkeypatch):
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel._calculate()
    worker = queued[0]
    panel._state.step_mm *= 0.5
    worker.signals.ready.emit(object())
    worker.signals.finished.emit()
    assert panel._result is None
    assert "inputs changed" in panel.status.text()


def test_virtual_observation_cursor_does_not_cancel_physical_design(panel, monkeypatch):
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel._calculate()
    worker = queued[0]
    panel._state.virtual_observation_z_mm = panel._state.sample.z_mm + 1
    panel.set_state(panel._state)
    assert not worker.cancel_event.is_set()
    worker.signals.finished.emit()


def test_cancelling_queued_job_releases_controls_without_numerical_work(panel):
    # Dispatch has not reached the event loop yet. Cancel must remove this
    # real coordinator job without waiting for an unrelated calculation.
    panel._calculate()
    worker = panel._worker
    assert panel.pool.coordinator.has_owner(panel.pool)
    panel.cancel()
    assert worker.cancel_event.is_set()
    assert not panel.pool.coordinator.has_owner(panel.pool)
    assert panel._worker is None
    assert panel.calculate.isEnabled()
    assert panel._result is None


def test_shared_invalidation_before_assignment_cancels_queued_job(panel):
    panel._calculate()
    worker = panel._worker
    panel.invalidate_current()
    assert worker.cancel_event.is_set()
    assert panel._worker is None
    assert "inputs changed" in panel.status.text()
    assert panel.calculate.isEnabled()


def test_late_old_worker_signals_cannot_replace_new_worker(panel, monkeypatch):
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel._calculate()
    old_worker = queued[0]
    panel.cancel()
    old_worker.signals.finished.emit()
    panel._calculate()
    new_worker = queued[1]
    message = panel.status.text()
    old_worker.signals.progress.emit("obsolete progress")
    old_worker.signals.failed.emit("obsolete failure")
    old_worker.signals.ready.emit(object())
    old_worker.signals.finished.emit()
    assert panel._worker is new_worker
    assert panel._result is None
    assert panel.status.text() == message
    assert not panel.calculate.isEnabled()
    new_worker.signals.finished.emit()


def test_design_explorer_hosts_the_panel(qtbot):
    from temsim.gui.design_explorer import DesignExplorerPage

    page = DesignExplorerPage()
    qtbot.addWidget(page)
    assert page.study_tabs.count() == 2
    assert page.study_tabs.widget(1) is page.condenser_scan_design
    assert page.study_tabs.tabText(1) == "CM / objective / scan"
    assert page.shutdown()
