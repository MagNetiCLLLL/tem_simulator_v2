"""A calculation status cannot certify obsolete dimensions shown in an editor."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "project_and_recording_system/EnergyFilter.toml"
COIL = "intermediate_lens_excitation_coil"
OD = ("parts", COIL, "mechanical_outer_diameter_mm")
ASSEMBLY = SimpleNamespace(selected_module_paths=(("recording", MODULE),))


@pytest.fixture
def editor(qtbot, tmp_path):
    root = tmp_path / "instruments"
    path = root / MODULE
    path.parent.mkdir(parents=True)
    shutil.copyfile(INSTRUMENT_CONFIG_ROOT / MODULE, path)
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.set_project_context(root, ASSEMBLY, None)
    assert page.open_path(path, selected_key=COIL)
    page.set_simulation_context("ideal", {})
    return page


def _diameter(page):
    return next(page.dimensions.item(row, 1) for row in range(page.dimensions.rowCount())
                if tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)) == OD)


def _external_edit(page):
    document = PartModelDocument(page.session.path)
    document.set_dimension(OD, 93.)
    document.save()


def test_clean_source_refresh_adopts_saved_dimensions_and_retains_selection(editor):
    editor.view.set_axial_view()
    _external_edit(editor)
    editor.set_project_context(editor._project_root, ASSEMBLY, None)
    editor.set_calculation_status("current", "Accepted calculation after catalog reload")
    assert float(_diameter(editor).text()) == 93.
    assert editor._selected_key == COIL and not editor.session.dirty
    assert "Result current" in editor.calculation_label.text()
    editor.session.assert_source_current()


@pytest.mark.parametrize("draft", ["94", "unfinished"])
def test_source_change_preserves_valid_and_invalid_drafts(editor, draft):
    _diameter(editor).setText(draft)
    staged_text = _diameter(editor).text()
    _external_edit(editor)
    editor.set_project_context(editor._project_root, ASSEMBLY, None)
    editor.set_calculation_status("current", "New saved model result")
    assert _diameter(editor).text() == staged_text
    assert "Source changed" in editor.calculation_label.text()
    assert "Unsaved draft" in editor.calculation_label.text()
    assert "Result current" not in editor.calculation_label.text()
    assert not editor.save()
    assert PartModelDocument(editor.session.path).part(COIL)[OD[-1]] == 93.


def test_unreloaded_or_missing_source_never_appears_current(editor):
    _external_edit(editor)
    editor.set_calculation_status("current", "Calculation completed elsewhere")
    assert float(_diameter(editor).text()) != 93.
    assert "Source changed" in editor.calculation_label.text()
    assert "Result current" not in editor.calculation_label.text()
    editor.session.path.unlink()
    editor.set_project_context(editor._project_root, ASSEMBLY, None)
    editor.set_calculation_status("current", "Old result")
    assert "unavailable" in editor.calculation_label.text()
    assert "Result current" not in editor.calculation_label.text()


def test_inactive_module_loses_runtime_opening_when_assembly_changes(editor):
    column = "column/C3_ProbeCorrector.toml"
    path = editor._project_root / column
    path.parent.mkdir(parents=True)
    shutil.copyfile(INSTRUMENT_CONFIG_ROOT / column, path)
    key = "condenser_aperture_2"
    active = SimpleNamespace(selected_module_paths=(("column", column),))
    values = {key: {"radius_mm": .05}}
    editor.set_project_context(editor._project_root, active, None, values)
    assert editor.open_path(path, selected_key=key)
    assert "working_opening" in editor._mesh_records[0]["face_groups"]
    editor.set_project_context(editor._project_root, ASSEMBLY, None, values)
    editor.set_simulation_context("ideal", {})
    assert editor._runtime_refresh_pending  # Hidden CAD refreshes on presentation.
    assert "working_opening" in editor._mesh_records[0]["face_groups"]
    editor.show()
    assert "working_opening" not in editor._mesh_records[0]["face_groups"]
    assert all(editor.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[0] != "runtime"
               for row in range(editor.dimensions.rowCount()))
    assert "Not linked" in editor.calculation_label.text()
    editor.set_project_context(editor._project_root, active, None, values)
    editor.set_simulation_context("ideal", {})
    assert "working_opening" in editor._mesh_records[0]["face_groups"]


def test_main_window_save_and_intermediate_results_keep_dimension_status_honest(qtbot, tmp_path, monkeypatch):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.gui import main_window
    from temsim.manifest_editor import ManifestEditor, ManifestTarget

    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat))
    schedule = main_window.MainWindow.schedule_preview
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    window = main_window.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window, "schedule_preview", schedule.__get__(window))
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(root / MODULE, selected_key=COIL)
    page.set_calculation_status("current", "Initial saved model")
    assert window._save_manifest_updates(ManifestTarget(MODULE, COIL), {OD: 93.}, report_error=False)
    window.preview_timer.stop()
    assert float(_diameter(page).text()) == 93.
    assert "Results out of date" in page.calculation_label.text()

    window._calculation_started("Preview")
    assert "Calculating" in page.calculation_label.text()
    # Exercise the real acceptance guard without performing another ray solve
    # or asking unrelated plots to interpret a synthetic result.
    result = SimpleNamespace(state_snapshot=window.state, simulation=SimpleNamespace(
        metrics={"mode": "diffraction"}, incident=SimpleNamespace(x=np.zeros((1, 1)))))
    monkeypatch.setattr(window.workspace, "display_result", lambda *_: None)
    monkeypatch.setattr(window.workspace.interactive_calculation, "display_tuning_status", lambda *_: None)
    monkeypatch.setattr(window.workspace.model_inspector, "display_result", lambda *_: None)
    monkeypatch.setattr(window.assembly_panel, "update_direct_alignment_metrics", lambda *_: None)
    monkeypatch.setattr(window, "_interactive_preview_in_flight", lambda: True)
    window._interactive_preview_pending = True
    window._calculation_ready("Preview", result, .1)
    assert "Result current" not in page.calculation_label.text()
    window._interactive_preview_pending = False
    monkeypatch.setattr(window, "_interactive_preview_in_flight", lambda: False)
    window._calculation_ready("Preview", result, .1)
    assert "Result current" in page.calculation_label.text()
    monkeypatch.setattr(window, "_show_error", lambda *_: None)
    window._calculation_failed("Preview", "Fixture failure")
    assert "Calculation failed" in page.calculation_label.text()
