"""GUI-only selected-plane lifecycle tests; numerical traces are substituted.

Real Qt timers, shared worker admission and queued signals are exercised here.
These tests do not qualify electron optics or diffraction physics.
"""

from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThread, Qt

import temsim.gui.selected_plane_readout as readout_module
from temsim.gui.selected_plane_readout import SelectedPlaneReadout
from temsim.physics.selected_plane import SelectedPlaneDiagnostic


def diagnostic(z_mm, kind="mixed"):
    return SelectedPlaneDiagnostic(z_mm, kind, 0.02, 0.25, "GUI-only fixture")


@pytest.fixture
def panel(qtbot, monkeypatch):
    view = SelectedPlaneReadout()
    qtbot.addWidget(view)
    view.show()
    monkeypatch.setattr(readout_module, "plane_status_without_trace", lambda *_args: None)
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda _result, z_mm, **_kwargs: diagnostic(z_mm))
    view.set_result(SimpleNamespace(name="captured-first"))
    yield view
    assert view.shutdown()


def wait_kind(qtbot, panel, kind="mixed"):
    qtbot.waitUntil(lambda: panel._diagnostic is not None and panel._diagnostic.kind == kind)


def test_calculation_runs_in_coordinated_worker_and_displays_residuals(panel, qtbot, monkeypatch):
    calls = []

    def solve(result, z_mm, **_kwargs):
        calls.append((result, z_mm, QThread.currentThread() == panel.thread()))
        return diagnostic(z_mm)

    monkeypatch.setattr(readout_module, "calculate_selected_plane", solve)
    result = panel._result
    panel.select_z(12.0)
    wait_kind(qtbot, panel)
    assert calls == [(result, 12.0, False)]
    assert panel.pool.maxThreadCount() == 1
    assert "Mixed plane" in panel.label.text()
    assert "image ‖B‖ 0.02 m/rad" in panel.label.text()
    assert "diffraction ‖A‖ 0.25" in panel.label.text()
    assert "captured instrument settings" in panel.label.toolTip()
    assert "not calculated crystal diffraction intensity" in panel.label.toolTip()
    assert "r(z) = A(z)" in panel.label.toolTip()
    assert "Mixed plane" in panel.label.toolTip()
    assert "0.02" not in panel.label.toolTip()  # Keep equations symbolic.
    assert panel.label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert not panel.label.wordWrap()


def test_continuous_edit_debounces_to_only_latest_z(panel, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda _result, z_mm, **_kwargs: (calls.append(z_mm), diagnostic(z_mm))[1])
    assert panel.DEBOUNCE_MS == 180
    panel.select_z(11.0)
    qtbot.wait(60)
    panel.select_z(12.0)
    qtbot.wait(60)
    panel.select_z(13.0)
    qtbot.wait(60)
    assert calls == []
    wait_kind(qtbot, panel)
    assert calls == [13.0]
    assert panel._diagnostic.z_mm == 13.0


def test_exact_cache_is_immediate_and_duplicate_selection_does_not_trace(panel, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda _result, z_mm, **_kwargs: (calls.append(z_mm), diagnostic(z_mm))[1])
    panel.select_z(12.0)
    panel.select_z(12.0)
    wait_kind(qtbot, panel)
    panel.select_z(13.0)
    wait_kind(qtbot, panel)
    panel.select_z(12.0)
    assert panel._diagnostic.z_mm == 12.0
    assert not panel.timer.isActive()
    qtbot.wait(250)
    assert calls == [12.0, 13.0]


def test_cache_is_bounded_by_exact_float_z_and_republication_clears_it(panel):
    for index in range(65):
        z_mm = float(index + 11)
        panel.select_z(z_mm)
        panel.timer.stop()
        panel._solved(panel._generation, panel._result_generation, diagnostic(z_mm))
    assert len(panel._cache) == 64
    assert 11.0 not in panel._cache
    panel.select_z(12.0)
    assert not panel.timer.isActive()
    panel.select_z(76.0000000001)
    assert panel.timer.isActive()  # No interpolation or rounded-Z cache alias.
    result = panel._result
    panel.set_result(result)
    assert not panel._cache
    assert panel._diagnostic is None
    assert not panel.timer.isActive()


@pytest.mark.parametrize("kind", ["upstream", "specimen", "not_calculated", "unavailable"])
def test_untraced_status_never_starts_a_worker(panel, qtbot, monkeypatch, kind):
    monkeypatch.setattr(readout_module, "plane_status_without_trace",
                        lambda _result, z_mm: SelectedPlaneDiagnostic(z_mm, kind))
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda *_args, **_kwargs: pytest.fail("Invalid plane started a trace"))
    panel.select_z(4.0)
    qtbot.wait(250)
    assert panel._diagnostic.kind == kind
    assert panel._worker is None
    assert not panel.timer.isActive()


def test_stale_inputs_keep_cached_result_and_cancel_new_plane_work(panel, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda _result, z_mm, **_kwargs: (calls.append(z_mm), diagnostic(z_mm))[1])
    panel.select_z(12.0)
    wait_kind(qtbot, panel)
    panel.mark_stale()
    assert "Previous optics" in panel.label.text()
    assert "previous result (stale)" in panel.label.toolTip()
    assert "Mixed plane" in panel.label.text()
    panel.select_z(13.0)
    assert "inputs changed" in panel.label.text()
    assert "Model only" in panel.label.toolTip()
    assert "recalculate Ray Diagram" in panel.label.text()
    assert not panel.timer.isActive()
    panel.select_z(12.0)
    assert "Previous optics" in panel.label.text()
    assert "Mixed plane" in panel.label.text()
    qtbot.wait(250)
    assert calls == [12.0]


def test_invalid_captured_metadata_reports_error_without_ui_exception(panel, monkeypatch):
    def bad_metadata(*_args):
        raise ValueError("GUI-only invalid captured metadata")

    monkeypatch.setattr(readout_module, "plane_status_without_trace", bad_metadata)
    panel.select_z(12.0)
    assert panel._diagnostic.kind == "unavailable"
    assert "invalid captured metadata" in panel.label.toolTip()
    assert not panel.timer.isActive()


@pytest.mark.parametrize("z_mm", [float("nan"), float("inf"), "invalid", True])
def test_invalid_linked_z_is_explicit_and_never_traced(panel, z_mm):
    panel.select_z(z_mm)
    assert panel._diagnostic.kind == "unavailable"
    assert "Invalid axial Z" in panel.label.text()
    assert not panel.timer.isActive()


def test_late_success_and_failure_cannot_replace_new_z_or_result(panel, qtbot, monkeypatch):
    entered, release = Event(), Event()

    def solve(_result, z_mm, **_kwargs):
        if z_mm == 12.0:
            entered.set()
            release.wait(2.0)
        return diagnostic(z_mm)

    monkeypatch.setattr(readout_module, "calculate_selected_plane", solve)
    panel.select_z(12.0)
    qtbot.waitUntil(entered.is_set)
    old_generation, old_result_generation = panel._generation, panel._result_generation
    panel.set_result(SimpleNamespace(name="captured-second"))
    panel.select_z(13.0)
    try:
        panel._solved(old_generation, old_result_generation, diagnostic(12.0, "image"))
        panel._failed(old_generation, "obsolete failure")
        assert "13 mm" in panel.label.text()
        assert "obsolete failure" not in panel.label.toolTip()
        assert panel._diagnostic is None
    finally:
        release.set()
    wait_kind(qtbot, panel)
    assert panel._diagnostic.z_mm == 13.0
    assert 12.0 not in panel._cache


def test_running_old_z_does_not_delay_latest_request_after_completion(panel, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def solve(_result, z_mm, **_kwargs):
        calls.append(z_mm)
        if z_mm == 12.0:
            entered.set()
            release.wait(2.0)
        return diagnostic(z_mm)

    monkeypatch.setattr(readout_module, "calculate_selected_plane", solve)
    panel.select_z(12.0)
    qtbot.waitUntil(entered.is_set)
    panel.select_z(13.0)
    qtbot.wait(230)
    panel.select_z(14.0)
    qtbot.wait(230)
    assert calls == [12.0]
    release.set()
    wait_kind(qtbot, panel)
    assert calls == [12.0, 14.0]


def test_failed_calculation_is_visible_and_new_selection_recovers(panel, qtbot, monkeypatch):
    def solve(_result, z_mm, **_kwargs):
        if z_mm == 12.0:
            raise ValueError("GUI-only intentional failure")
        return diagnostic(z_mm, "image")

    monkeypatch.setattr(readout_module, "calculate_selected_plane", solve)
    panel.select_z(12.0)
    wait_kind(qtbot, panel, "unavailable")
    assert "intentional failure" in panel.label.toolTip()
    assert 12.0 not in panel._cache
    panel.select_z(13.0)
    wait_kind(qtbot, panel, "image")
    assert "Image plane" in panel.label.toolTip()
    assert "intentional failure" not in panel.label.toolTip()


def test_clear_and_shutdown_cancel_pending_work_and_reject_late_results(panel, qtbot, monkeypatch):
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda *_args, **_kwargs: pytest.fail("Cancelled debounce started work"))
    panel.select_z(12.0)
    generation, result_generation = panel._generation, panel._result_generation
    panel.clear()
    panel._solved(generation, result_generation, diagnostic(12.0))
    assert panel._result is None
    assert panel._diagnostic is None
    assert "choose an axial Z" in panel.label.text()
    panel.set_result(SimpleNamespace(name="another-captured-result"))
    panel.select_z(13.0)
    assert panel.shutdown()
    panel.select_z(14.0)
    qtbot.wait(250)
    assert panel._worker is None
    assert not panel.timer.isActive()


def test_shutdown_waits_for_cancelled_owned_worker(panel, qtbot, monkeypatch):
    entered = Event()

    def solve(_result, z_mm, *, cancelled):
        entered.set()
        while not cancelled():
            Event().wait(0.005)
        return diagnostic(z_mm)

    monkeypatch.setattr(readout_module, "calculate_selected_plane", solve)
    panel.select_z(12.0)
    qtbot.waitUntil(entered.is_set)
    assert panel.shutdown()
    assert panel._worker is None
    assert panel._diagnostic is None


def test_result_publication_after_shutdown_never_reopens_observer(panel, qtbot, monkeypatch):
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda *_args, **_kwargs: pytest.fail("Closed observer started a trace"))
    captured_result = panel._result
    panel.select_z(12.0)
    assert panel.shutdown()
    panel.set_result(SimpleNamespace(name="late-publication-during-close"))
    panel.select_z(13.0)
    qtbot.wait(250)
    assert panel._closed
    assert panel._result is captured_result
    assert panel._worker is None
    assert not panel.timer.isActive()


def test_workspace_cursor_publication_stale_and_clear_are_wired(qtbot, monkeypatch):
    from test_incremental_ray_scene import _result
    from temsim.gui.visualization import VisualizationWorkspace

    monkeypatch.setattr(readout_module, "plane_status_without_trace", lambda *_args: None)
    monkeypatch.setattr(readout_module, "calculate_selected_plane",
                        lambda _result, z_mm, **_kwargs: diagnostic(z_mm))
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
    view.display_result(_result(), "Preview")
    readout = view.selected_plane_readout
    try:
        view.jump_to_ray_position(12.0, activate_tab=False)
        wait_kind(qtbot, readout)
        view.axial_cursor_item.setValue(14.0)
        assert view._selected_z_mm == 12.0
        assert readout._selected_z_mm == 14.0
        view.display_result(_result(1.4), "Preview")
        assert readout._selected_z_mm == 14.0
        wait_kind(qtbot, readout)
        view.mark_ray_stale(SimpleNamespace(electron_gun=SimpleNamespace(
            type_key="fixture", display_name="Electron source"), apertures=()))
        assert "Previous optics" in readout.label.text()
        view.clear_result()
        assert readout._result is None
        assert not readout._cache
        assert "choose an axial Z" in readout.label.text()
    finally:
        assert readout.shutdown()
