"""Cache preferences affect retention, not physical state or calculation jobs."""
from dataclasses import replace

from PySide6.QtCore import QSettings

from temsim.cache_preferences import GIB, MIB, SETTINGS_ROOT, default_cache_preferences, save_cache_preferences


def test_window_applies_saved_limits_and_reuses_modeless_dialog(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller

    settings = QSettings(str(tmp_path / "cache_ui.ini"), QSettings.Format.IniFormat)
    preferences = replace(
        default_cache_preferences(16 * GIB), high_cache_budget_bytes=2 * GIB,
        tuning_cache_budget_bytes=128 * MIB, ray_display_cache_budget_bytes=32 * MIB,
        disk_cache_budget_bytes=4 * GIB,
    )
    save_cache_preferences(settings, preferences)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    stats = window.calculations.cache_statistics()
    assert stats["high_budget_bytes"] == 2 * GIB
    assert stats["tuning_budget_bytes"] == 128 * MIB
    assert stats["disk_budget_bytes"] == 4 * GIB
    assert window.workspace.ray_display_cache_info()["budget_bytes"] == 32 * MIB
    assert window.cache_settings_action in window.simulation_menu.actions()

    started = []
    window.calculations.started.connect(started.append)
    generation = window.calculations.generation
    state_before = window.state.to_dict()
    cached_plot = object()
    window.workspace._high_accuracy_result = cached_plot
    window.workspace._high_accuracy_current = False
    window.cache_settings_action.trigger()
    dialog = window._cache_settings_dialog
    assert dialog.isVisible() and not dialog.isModal()
    dialog.high_cache.setValue(3.)
    dialog.ray_display_cache.setValue(64.)
    assert dialog.apply_preferences()
    assert window.calculations.cache_statistics()["high_budget_bytes"] == 3 * GIB
    assert window.workspace.ray_display_cache_info()["budget_bytes"] == 64 * MIB
    assert settings.value(f"{SETTINGS_ROOT}/high_cache_budget_bytes") == 3 * GIB
    assert window.calculations.generation == generation
    assert window.state.to_dict() == state_before
    assert window.workspace._high_accuracy_result is cached_plot
    assert not window.preview_timer.isActive()
    assert started == []
    dialog.resize(640, 540)
    dialog.close()
    assert not dialog.isVisible()
    assert not settings.value(f"{SETTINGS_ROOT}/dialog_geometry").isEmpty()
    window.cache_settings_action.trigger()
    assert window._cache_settings_dialog is dialog
    assert dialog.high_cache.value() == 3.
    dialog.close()
