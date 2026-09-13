"""Default record location and user overrides; no SDK or hardware access."""
from pathlib import Path

import pytest

from temsim.recorder.paths import prepare_output_directory


@pytest.mark.parametrize("saved", [None, "", "  "])
def test_default_is_created_under_project_not_working_directory(tmp_path, monkeypatch, saved):
    project = tmp_path / "project"
    working = tmp_path / "elsewhere"
    project.mkdir()
    working.mkdir()
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(project))
    monkeypatch.chdir(working)
    path, error = prepare_output_directory(saved)
    assert path == project / "instrument_records"
    assert path.is_dir() and error is None
    assert not (working / "instrument_records").exists()


def test_existing_records_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path))
    path, _ = prepare_output_directory()
    record = path / "raw.npy"
    record.touch()
    assert prepare_output_directory(str(path)) == (path, None)
    assert record.is_file()


def test_custom_location_is_preserved_not_created_or_redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path / "project"))
    custom = tmp_path / "chosen-location"
    assert prepare_output_directory(str(custom)) == (custom, None)
    assert not custom.exists()
    assert not (tmp_path / "project" / "instrument_records").exists()


def test_creation_failure_is_reported_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path))
    blocked = tmp_path / "instrument_records"
    blocked.touch()
    path, error = prepare_output_directory()
    assert path == blocked
    assert "Cannot create default output folder" in error
    assert blocked.is_file()


def test_gui_initialises_default_and_remembers_manual_override(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from temsim.gui.instrument_recorder import InstrumentRecorderWindow

    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path))
    settings = QSettings(str(tmp_path / "recorder.ini"), QSettings.Format.IniFormat)
    settings.setValue("output", "")  # Empty value saved by the previous version.
    window = InstrumentRecorderWindow(settings=settings)
    qtbot.addWidget(window)
    assert window._output_directory == tmp_path / "instrument_records"
    assert window._output_directory.is_dir()
    custom = tmp_path / "custom"
    custom.mkdir()
    window._output_directory = custom
    window.close()
    assert window._shutdown
    reopened = InstrumentRecorderWindow(settings=settings)
    qtbot.addWidget(reopened)
    assert reopened._output_directory == custom
    reopened.close()
    assert reopened._shutdown
