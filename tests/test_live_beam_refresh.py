"""Latest-only Live wave refresh through real Qt controls; no physical solver."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings

from temsim.gui.coherent_beam import CoherentBeamPage
from temsim.gui.live_beam_refresh import LiveBeamRefresh
from temsim.optics.column import default_state


@pytest.fixture
def live(qtbot, monkeypatch):
    state = default_state()
    page = CoherentBeamPage(state_provider=lambda: state)
    qtbot.addWidget(page)
    page.set_state(state)
    # A prior accepted observation, without spending a propagation in a
    # scheduling regression. Physical GPU/CPU parity is tested separately.
    page.result = SimpleNamespace(checkpoint=SimpleNamespace(plane_z_mm=1600.))
    page._stale = False
    page._session_ready = True
    refresh = LiveBeamRefresh(page, page)
    refresh.SETTLE_MS = 20
    calls = []
    monkeypatch.setattr(page, "calculate", lambda: calls.append(
        (state.objective_lens.percent, page._target_z_mm,
         state.acceleration_backend, page.state_list.display_mode())))
    yield state, page, refresh, calls
    refresh.cancel()
    page.shutdown()


def edit(live, percent):
    state, page, refresh, _ = live
    state.objective_lens.percent = percent
    refresh.invalidate(lambda: page.set_state(state))


def test_waits_for_successful_particle_frame_and_uses_latest_inputs(live, qtbot):
    state, page, refresh, calls = live
    for value in (67., 67.5, 68.):
        edit(live, value)
    qtbot.wait(45)
    assert calls == []
    state.acceleration_backend = "Require GPU"
    page.set_selected_z(1605.)
    refresh.particle_ready()
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert calls == [(68., 1605., "Require GPU", "current")]
    assert not refresh.pending
    refresh.particle_ready()
    qtbot.wait(40)
    assert len(calls) == 1


def test_completed_ray_frame_does_not_bypass_edit_settle_interval(live, qtbot):
    _, _, refresh, calls = live
    edit(live, 67.)
    refresh.particle_ready()
    assert calls == []
    # A new edit invalidates the earlier particle frame's readiness.
    edit(live, 68.)
    qtbot.wait(45)
    assert calls == []
    refresh.particle_ready()
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert calls[0][0] == 68.


@pytest.mark.parametrize("action", ["draft", "cancel", "explicit", "mode", "close"])
def test_user_actions_supersede_pending_live_refresh(live, qtbot, action):
    _, page, refresh, calls = live
    edit(live, 68.)
    if action == "draft":
        page.tip_offset_x.setValue(.125)
    elif action == "cancel":
        page.cancel_button.setEnabled(True)
        page.cancel_button.click()
    elif action == "explicit":
        # Signal dispatch tests scheduling cancellation only. The original
        # explicit action is covered in coherent page tests.
        page.calculate_button.clicked.disconnect()
        page.calculate_button.clicked.connect(refresh.cancel)
        page.calculate_button.click()
    elif action == "mode":
        page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("overlay"))
    else:
        page.shutdown()
    refresh.particle_ready()
    qtbot.wait(45)
    assert calls == []
    assert not refresh.pending
    if action == "draft":
        assert page.tip_offset_x.value() == .125


def test_preexisting_unapplied_draft_never_arms_refresh(live, qtbot):
    _, page, refresh, calls = live
    page.tip_offset_x.setValue(.125)
    edit(live, 68.)
    refresh.particle_ready()
    qtbot.wait(45)
    assert not refresh.pending and calls == []
    assert page.tip_offset_x.value() == .125


def test_particle_only_use_does_not_start_a_first_coherent_calculation(live, qtbot):
    _, page, refresh, calls = live
    page.result = None
    page._session_ready = False
    edit(live, 68.)
    refresh.particle_ready()
    qtbot.wait(45)
    assert calls == [] and not refresh.pending


def test_active_saved_state_display_uses_existing_state_calculation_action(live, qtbot):
    _, page, refresh, calls = live
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("overlay"))
    page.state_set.active = True
    edit(live, 68.)
    refresh.particle_ready()
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert calls[0][-1] == "overlay"


def test_blocked_or_failed_work_cannot_trigger_a_wave(live, qtbot):
    _, _, refresh, calls = live
    edit(live, 68.)
    refresh.allowed = lambda: False
    refresh.particle_ready()
    qtbot.wait(45)
    assert calls == [] and not refresh.pending


def test_main_window_wires_live_invalidation_and_explicit_cancellation(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell
    from temsim.gui import interactive_calculation as gui
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window.preview_timer, "start", lambda *_: None)
    page = window.workspace.coherent_beam
    page.result = SimpleNamespace(checkpoint=SimpleNamespace(plane_z_mm=1600.))
    page._stale = False
    page._session_ready = True
    window.schedule_preview("interactive_tuning")
    assert window.live_beam_refresh.pending
    window._cancel_calculations()
    assert not window.live_beam_refresh.pending
    page._stale = False
    page._session_ready = True
    window.schedule_preview("interactive_tuning")
    assert window.live_beam_refresh.pending
    window.schedule_preview("sample_changed", automatic=False)
    assert not window.live_beam_refresh.pending
