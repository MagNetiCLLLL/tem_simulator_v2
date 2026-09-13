"""Offscreen UI checks with offline stream fakes, never a microscope."""
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QPushButton, QLineEdit, QDoubleSpinBox, QCheckBox, QFileDialog

from temsim.gui.instrument_recorder import InstrumentRecorderWindow
from temsim.recorder.backend import InstrumentRecorder
from test_instrument_recorder import FakeClient


def make_window(qtbot, tmp_path):
    fake = FakeClient()
    backend = InstrumentRecorder(lambda: fake)
    settings = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    settings.setValue("output", str(tmp_path))
    window = InstrumentRecorderWindow(backend=backend, settings=settings)
    qtbot.addWidget(window)
    window.show()
    return window, fake


def connect(qtbot, window):
    window._start("connect", ("offline-test-double.invalid", 7521))
    qtbot.waitUntil(lambda: not window.busy)


def close_window(qtbot, window):
    qtbot.waitUntil(lambda: not window.busy, timeout=10000)
    window.close()
    qtbot.waitUntil(lambda: window._shutdown, timeout=10000)


def test_only_three_buttons_no_manual_acquisition_or_sample_inputs(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    assert len(window.findChildren(QPushButton)) == 3
    assert not window.findChildren(QLineEdit)
    assert not window.findChildren(QDoubleSpinBox)
    assert not window.findChildren(QCheckBox)
    assert fake.calls == []
    assert all(not b.isEnabled() for b in window.capture_buttons.values())
    connect(qtbot, window)
    assert all(b.isEnabled() for b in window.capture_buttons.values())
    close_window(qtbot, window)


def test_click_ceta_saves_without_any_parameters(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    qtbot.mouseClick(window.capture_buttons["ceta"], Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert Path(window._last_path, "channel_001/image.npy").exists()
    np.testing.assert_array_equal(window.image_item.image, fake.data)
    assert not window.image_channel.isVisible()
    assert "recorded" in window.status.text()
    close_window(qtbot, window)


def test_all_stem_channels_share_one_preview_window(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    fake.optics.optical_mode = "Stem"
    qtbot.mouseClick(window.capture_buttons["stem_all"], Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.image_channel.count() == 3 and window.image_channel.isVisible()
    assert [window.image_channel.itemText(i) for i in range(3)] == ["HAADF", "DF", "BF"]
    window.image_channel.setCurrentIndex(2)
    np.testing.assert_array_equal(window.image_item.image, fake.data)
    close_window(qtbot, window)


def test_failure_preserves_last_successful_preview_and_identity(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    window._capture("ceta")
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    previous = window.preview_label.text()
    fake.fail_channels = {"BM-Ceta", "Flucam"}
    window._capture("flucam")
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert "failed" in window.status.text()
    assert window.preview_label.text() == previous
    np.testing.assert_array_equal(window.image_item.image, fake.data)
    close_window(qtbot, window)


def test_flucam_works_in_current_stem_mode(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    fake.optics.optical_mode = "Stem"
    window._capture("flucam")
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window._last_manifest["images"][0]["detector"] == "Flucam"
    assert "Stem" in window.identity_label.text()
    close_window(qtbot, window)


def test_worker_close_guard_preserves_running_read(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    fake.after_frame = lambda: time.sleep(0.15)
    window._capture("ceta")
    assert window.close() is False and window.thread.isRunning()
    window._stop()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    close_window(qtbot, window)


def test_window_geometry_is_retained(qtbot, tmp_path):
    window, _ = make_window(qtbot, tmp_path)
    window.resize(780, 680)
    qtbot.wait(50)
    expected = window.size()
    close_window(qtbot, window)
    reopened, _ = make_window(qtbot, tmp_path)
    qtbot.wait(50)
    assert reopened.size() == expected and not reopened.connected
    close_window(qtbot, reopened)


def test_disconnect_failure_does_not_close_window(qtbot, tmp_path):
    window, fake = make_window(qtbot, tmp_path)
    connect(qtbot, window)
    fake.disconnect_error = True
    assert not window.close()
    qtbot.waitUntil(lambda: not window.busy)
    assert window.connected and not window._shutdown
    assert "not confirmed" in window.status.text()
    fake.disconnect_error = False
    close_window(qtbot, window)


def test_output_folder_menu_keeps_three_button_layout(qtbot, tmp_path, monkeypatch):
    window, _ = make_window(qtbot, tmp_path)
    custom = tmp_path / "chosen"
    custom.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(custom))
    window._browse()
    assert window._output_directory == custom
    assert window.settings.value("output") == str(custom)
    assert len(window.findChildren(QPushButton)) == 3
    close_window(qtbot, window)


def test_top_menu_opens_separate_window_without_touching_simulation(qtbot, tmp_path, monkeypatch):
    from temsim.gui import main_window as shell
    from temsim.gui import instrument_recorder as recorder_gui
    from PySide6.QtGui import QAction

    settings = QSettings(str(tmp_path / "main.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    constructor = recorder_gui.InstrumentRecorderWindow
    fake = FakeClient()
    def isolated(parent):
        return constructor(parent, backend=InstrumentRecorder(lambda: fake),
                           settings=QSettings(str(tmp_path / "recorder.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(recorder_gui, "InstrumentRecorderWindow", isolated)
    main = shell.MainWindow()
    main.preview_timer.stop()
    qtbot.addWidget(main)
    before = main.state.to_dict()
    tab_count = main.workspace.tabs.count()
    monkeypatch.setattr(main.calculations.pool, "start", lambda *_: (_ for _ in ()).throw(
        AssertionError("The recorder must not request a simulation")))
    action = main.findChild(QAction, "instrumentRecorderAction")
    action.trigger()
    recorder = main._instrument_recorder
    assert recorder.isWindow() and recorder.isVisible()
    assert main.workspace.tabs.count() == tab_count
    assert main.state.to_dict() == before
    assert fake.calls == []
    action.trigger()
    assert main._instrument_recorder is recorder
    close_window(qtbot, recorder)
    main.close()
