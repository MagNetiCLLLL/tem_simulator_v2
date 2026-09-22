"""A calculation status cannot certify obsolete dimensions shown in an editor."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.shared_tip import copy_catalog_tree


MODULE = "project_and_recording_system/NoEnergyFilter.toml"
COIL = "intermediate_lens_excitation_coil"
OD = ("parts", COIL, "mechanical_outer_diameter_mm")
ASSEMBLY = SimpleNamespace(selected_module_paths=(("recording", MODULE),))


@pytest.fixture
def editor(qtbot, tmp_path):
    root = tmp_path / "instruments"
    # Recording modules reference shared mechanical subassemblies. Preserve
    # their relative paths so edits exercise the authoritative source files.
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    path = root / MODULE
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
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
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
    assert MODULE in dict(window.state._resolved_assembly.selected_module_paths).values()
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
        metrics={"mode": "diffraction", "section_target_z_mm": 1100.,
                 "section_resumable_through_z_mm": 1050.},
        incident=SimpleNamespace(x=np.zeros((1, 1))),
        section_checkpoint=SimpleNamespace(gun_trace=SimpleNamespace(z_mm=np.array([0., 450.])))))

    def publish_extent(result, quality):
        # Presentation-only checkpoint metadata; physical continuation is
        # validated separately. Reset freshness as publishing a new frame does.
        window.workspace._last_result = result
        window.workspace._last_quality = quality
        window.workspace._ray_extent_stale = False
        window.workspace._refresh_ray_calculation_extent()

    monkeypatch.setattr(window.workspace, "display_result", publish_extent)
    observed_sections = []
    monkeypatch.setattr(window.workspace, "jump_to_ray_position",
                        lambda z, **options: observed_sections.append((z, options)))
    monkeypatch.setattr(window.workspace.interactive_calculation, "display_tuning_status", lambda *_: None)
    monkeypatch.setattr(window.workspace.model_inspector, "display_result", lambda *_: None)
    monkeypatch.setattr(window.workspace.vacuum_map, "set_result", lambda *_: None)
    monkeypatch.setattr(window.assembly_panel, "update_direct_alignment_metrics", lambda *_: None)
    monkeypatch.setattr(window, "_interactive_preview_in_flight", lambda: True)
    window._interactive_preview_pending = True
    window._calculation_ready("Preview", result, .1)
    assert "Result current" not in page.calculation_label.text()
    extent = window.workspace.ray_calculation_extent
    assert extent.extent["stale"]
    assert extent.extent["completed_z_mm"] == 1100.
    assert extent.extent["resumable_z_mm"] == 1050.
    assert extent.label.text().startswith("Previous")
    assert observed_sections == []
    window._interactive_preview_pending = False
    monkeypatch.setattr(window, "_interactive_preview_in_flight", lambda: False)
    window._calculation_ready("Preview", result, .1)
    assert "Result current" in page.calculation_label.text()
    assert not extent.extent["stale"]
    assert extent.extent["completed_z_mm"] == 1100.
    assert observed_sections == [(1100., {"activate_tab": False})]
    monkeypatch.setattr(window, "_show_error", lambda *_: None)
    window._calculation_failed("Preview", "Fixture failure")
    assert "Calculation failed" in page.calculation_label.text()
