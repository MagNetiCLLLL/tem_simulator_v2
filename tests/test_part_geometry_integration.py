"""Component selection through a real, transactional geometry save."""

from copy import deepcopy
import shutil

import pytest
from PySide6.QtCore import QSettings, Qt

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui import main_window
from temsim.gui.parameter_panel import ParameterPanel
from temsim.manifest_editor import ManifestEditor, ManifestField, ManifestTarget
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "project_and_recording_system/EnergyFilter.toml"
KEY = "intermediate_lens_excitation_coil"
TARGET = ManifestTarget(MODULE, KEY)


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    source = INSTRUMENT_CONFIG_ROOT / MODULE
    original = source.read_bytes()
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(
        str(tmp_path / "window.ini"), QSettings.Format.IniFormat
    ))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    yield widget
    assert source.read_bytes() == original


def _document(window):
    return module_manifest.read_document(window.manifest_editor.root / MODULE)


def _part(document):
    return next(part for part in document["parts"] if part["key"] == KEY)


def _change(dialog, dimension, value):
    control = dialog.dimension_controls[dimension]
    control.setValue(value)
    control.editingFinished.emit()


def test_layout_selection_opens_editor_and_applies_only_material_diameter(window, qtbot):
    before = deepcopy(_document(window))
    window.state.objective_lens.percent = 67.89
    strengths = {lens.key: lens.percent for lens in window.state.lenses}
    window.workspace.component_selected.emit(KEY)
    panel = window.parameter_panel
    assert panel._manifest_target == TARGET
    assert not panel.geometry_box.isHidden()
    assert panel.geometry_edit_button.isEnabled()
    qtbot.mouseClick(panel.geometry_edit_button, Qt.MouseButton.LeftButton)
    dialog = window._part_geometry_dialog
    assert dialog is not None
    assert dialog.windowModality() == Qt.WindowModality.WindowModal
    assert window._edit_part_geometry(TARGET) is dialog
    assert {item.key for item in dialog._neighbours} >= {
        "intermediate_lens_housing", "intermediate_lens_yoke"
    }

    _change(dialog, "inner_diameter_mm", 70.0)
    assert _document(window) == before  # Draft changes never write TOML.
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    expected = deepcopy(before)
    _part(expected)["mechanical_inner_diameter_mm"] = 70.0
    assert _document(window) == expected
    assert dialog.error_label.text() == ""
    assert not dialog.apply_button.isEnabled()
    assert {lens.key: lens.percent for lens in window.state.lenses} == strengths
    assert window.assembly.part(KEY).data["mechanical_inner_diameter_mm"] == 70.0
    assert "ID 70 mm" in panel.geometry_summary.text()

    # A second Apply uses the new baseline and does not trip stale-file protection.
    _change(dialog, "outer_diameter_mm", 95.0)
    dialog.apply()
    _part(expected)["mechanical_outer_diameter_mm"] = 95.0
    assert _document(window) == expected
    dialog.reject()
    assert window._part_geometry_dialog is None


def test_collision_leaves_file_and_state_intact_and_keeps_draft(window):
    before = _document(window)
    state = window.state
    dialog = window._edit_part_geometry(TARGET)
    _change(dialog, "outer_diameter_mm", 110.0)
    dialog.apply()
    assert _document(window) == before
    assert window.state is state
    assert dialog.geometry.outer_diameter_mm == 110.0
    assert dialog.apply_button.isEnabled()
    assert "intermediate_lens_yoke" in dialog.error_label.text()
    _change(dialog, "outer_diameter_mm", 95.0)
    dialog.apply()
    assert dialog.error_label.text() == ""
    assert _part(_document(window))["mechanical_outer_diameter_mm"] == 95.0


def test_assembly_reload_failure_rolls_back_file_and_is_reported_inline(window, monkeypatch):
    path = window.manifest_editor.root / MODULE
    before = path.read_bytes()
    state = window.state
    dialog = window._edit_part_geometry(TARGET)
    _change(dialog, "inner_diameter_mm", 70.0)

    def fail(*args, **kwargs):
        raise ValueError("Synthetic assembly reload failure")

    monkeypatch.setattr(window.catalog, "apply", fail)
    monkeypatch.setattr(window, "_show_error", lambda message: pytest.fail(message))
    dialog.apply()
    assert "Synthetic assembly reload failure" in dialog.error_label.text()
    assert path.read_bytes() == before
    assert window.state is state
    assert dialog.geometry.inner_diameter_mm == 70.0


def test_stale_module_does_not_overwrite_an_external_edit(window):
    dialog = window._edit_part_geometry(TARGET)
    _change(dialog, "inner_diameter_mm", 70.0)
    path = window.manifest_editor.root / MODULE
    # A valid metadata change to another row is still a module revision.
    text = path.read_text(encoding="utf-8")
    assert 'name = "Intermediate Lens Housing"' in text
    path.write_text(text.replace('name = "Intermediate Lens Housing"',
                                 'name = "Updated Intermediate Lens Housing"'), encoding="utf-8")
    external = path.read_bytes()
    dialog.apply()
    assert "changed while the editor was open" in dialog.error_label.text()
    assert path.read_bytes() == external
    assert dialog.geometry.inner_diameter_mm == 70.0


def test_close_discards_draft_and_reopen_reads_saved_geometry(window):
    before = _document(window)
    dialog = window._edit_part_geometry(TARGET)
    _change(dialog, "inner_diameter_mm", 70.0)
    dialog.reject()
    assert _document(window) == before
    assert window._part_geometry_dialog is None
    reopened = window._edit_part_geometry(TARGET)
    assert reopened.geometry.inner_diameter_mm == _part(before)["mechanical_inner_diameter_mm"]


def test_apply_preserves_existing_toml_drafts_including_invalid_text(window):
    window.workspace.component_selected.emit(KEY)
    panel = window.parameter_panel
    items = {panel.manifest_table.item(row, 0).text(): panel.manifest_table.item(row, 1)
             for row in range(panel.manifest_table.rowCount())}
    pending = {"name": '"Unsaved coil name"', "vacuum_inner_diameter_mm": "unfinished",
               "mechanical_inner_diameter_mm": "77"}
    for name, text in pending.items():
        items[name].setText(text)
    before = _document(window)
    dialog = window._edit_part_geometry(TARGET)
    _change(dialog, "inner_diameter_mm", 70.0)
    dialog.apply()
    expected = deepcopy(before)
    _part(expected)["mechanical_inner_diameter_mm"] = 70.0
    assert _document(window) == expected
    assert dialog.error_label.text() == ""
    assert "ID 70 mm" in panel.geometry_summary.text()
    assert not panel.manifest_draft_notice.isHidden()
    assert "remain unsaved" in panel.manifest_draft_notice.text()
    assert panel.manifest_draft_texts(TARGET) == {
        ("parts", KEY, name): text for name, text in pending.items()
    }
    # Later Applies preserve the same draft without trying to parse it.
    _change(dialog, "outer_diameter_mm", 95.0)
    dialog.apply()
    assert dialog.error_label.text() == ""
    assert panel.manifest_draft_texts(TARGET)[("parts", KEY, "vacuum_inner_diameter_mm")] == "unfinished"


def test_shared_and_split_bodies_are_explained_before_opening(window, monkeypatch):
    for key, reason in (("objective_lens_excitation_coil", "split"),
                        ("objective_lens_yoke", "split"),
                        ("condenser_lens_1_excitation_coil", "shared")):
        window.workspace.component_selected.emit(key)
        panel = window.parameter_panel
        assert panel._manifest_target.part_key == key
        assert not panel.geometry_edit_button.isEnabled()
        assert reason in panel.geometry_summary.text()
        errors = []
        monkeypatch.setattr(window, "_show_error", errors.append)
        assert window._edit_part_geometry(panel._manifest_target) is None
        assert reason in errors[-1]


def test_saved_summary_and_vacuum_tooltip_distinguish_staged_toml_values(qtbot):
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    fields = ManifestEditor().fields(TARGET)
    panel.set_context("Coil", None, TARGET, fields, None)
    summary = panel.geometry_summary.text()
    items = {panel.manifest_table.item(row, 0).text(): panel.manifest_table.item(row, 1)
             for row in range(panel.manifest_table.rowCount())}
    assert "beam" in items["vacuum_inner_diameter_mm"].toolTip().lower()
    assert "Edit dimensions" in items["vacuum_inner_diameter_mm"].toolTip()
    items["mechanical_inner_diameter_mm"].setText("70")
    assert panel.geometry_summary.text() == summary
    assert "Uncommitted TOML table edits are not imported" in panel.geometry_edit_button.toolTip()


@pytest.mark.parametrize("profiled", [False, True])
def test_unsupported_or_profiled_part_does_not_offer_an_annular_editor(qtbot, profiled):
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    part = dict(_part(module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / MODULE)))
    if profiled:
        part["magnetic_radial_profile_mm"] = [[0.0, 40.0, 45.0], [180.0, 40.0, 45.0]]
    else:
        part["mechanical_profile"] = "magnetic_pole_piece"
    fields = tuple(ManifestField(("parts", KEY, name), name, value) for name, value in part.items())
    panel.set_context("Part", None, TARGET, fields, None)
    assert not panel.geometry_edit_button.isEnabled()
    if profiled:
        assert "profile" in panel.geometry_summary.text()
    else:
        assert panel.geometry_box.isHidden()
