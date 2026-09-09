"""Real component dialogs stage structural edits without writing source files."""

from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from temsim import module_manifest
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "project_and_recording_system/EnergyFilter.toml"
COLUMN = "column/C3_ProbeCorrector_ImageCorrector.toml"
COIL = "intermediate_lens_excitation_coil"


@pytest.fixture
def page(qtbot, tmp_path):
    catalog = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, catalog)
    document = module_manifest.read_document(catalog / MODULE)
    part = next(part for part in document["parts"] if part["key"] == COIL)
    origin = 500.0
    assembly = SimpleNamespace(selected_module_paths=(("project_and_recording_system", MODULE), ("column", COLUMN)),
        parts=(SimpleNamespace(source_file=MODULE, center_z_mm=part["local_center_z_mm"] + origin, data=part),))
    widget = PartModelEditorPage()
    qtbot.addWidget(widget)
    widget.set_project_context(catalog, assembly, None)
    assert widget.open_path(catalog / MODULE, selected_key=COIL)
    widget.resize(1280, 760)
    # Showing only the operation dialog avoids rendering the whole editor in
    # tests that inspect its actual meshes and controls rather than screenshots.
    yield widget
    if widget._component_dialog is not None:
        widget._component_dialog.reject()


def _dialog(qtbot, page, action):
    button = {"new": page.new_component_button, "place": page.place_component_button,
              "copy": page.copy_component_button}[action]
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    dialog = page._component_dialog
    assert dialog is not None
    qtbot.waitUntil(dialog.isVisible)
    return dialog


def _confirm(qtbot, page, dialog):
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert not dialog.error_label.text(), dialog.error_label.text()
    qtbot.waitUntil(lambda: page._component_dialog is None)


def _add(qtbot, page, *, key="custom_part", shape="tube", center=150):
    dialog = _dialog(qtbot, page, "new")
    dialog.key_edit.setText(key)
    dialog.name_edit.setText("Independent mechanical component")
    dialog.shape.setCurrentIndex(dialog.shape.findData(shape))
    dialog.center.setText(str(center))
    _confirm(qtbot, page, dialog)
    return page.session.part(key)


def _inactive_target(page):
    return next(page._project_root / path for path in page._catalog_paths if path not in page._project_paths)


@pytest.mark.parametrize("shape", ["tube", "box", "elliptic_cylinder"])
def test_new_component_is_real_draft_and_tree_undo_redo_tracks_added_part(page, qtbot, shape):
    path = page.session.path
    original = path.read_bytes()
    original_count = len(page.session.document["parts"])
    part = _add(qtbot, page, shape=shape)
    assert part["mechanical_only"] is True
    assert part["mechanical_part_role"] == "custom_mechanical"
    assert page.session.dirty
    assert page._selected_key == "custom_part"
    assert page.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "custom_part"
    assert "custom_part" in page._tree_nodes
    assert any(mesh["key"] == "custom_part" and len(mesh["faces"]) for mesh in page._mesh_records)
    assert page.view.selection[0] == "custom_part"
    assert len(page.session.document["parts"]) == original_count + 1
    assert path.read_bytes() == original
    if shape == "tube":
        fields = {tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)): row
                  for row in range(page.dimensions.rowCount())}
        assert ("parts", "custom_part", "vacuum_inner_diameter_mm") not in fields
        length_row = fields[("parts", "custom_part", "length_mm")]
        assert page.dimensions.item(length_row, 3).text() == "CAD only"
    qtbot.mouseClick(page.undo_button, Qt.MouseButton.LeftButton)
    assert "custom_part" not in page._tree_nodes
    assert len(page.session.document["parts"]) == original_count
    assert not page.session.dirty
    qtbot.mouseClick(page.redo_button, Qt.MouseButton.LeftButton)
    assert "custom_part" in page._tree_nodes
    assert page._selected_key == "custom_part"
    assert page.view.selection[0] == "custom_part"
    assert path.read_bytes() == original


def test_invalid_new_component_stays_in_dialog_with_input_and_draft_unchanged(page, qtbot):
    before = deepcopy(page.session.document)
    dialog = _dialog(qtbot, page, "new")
    dialog.key_edit.setText("kept_key")
    dialog.name_edit.setText("Kept user name")
    dialog.values["inner_diameter_mm"].setText("30")
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert dialog.isVisible() and dialog.error_label.isVisible()
    assert dialog.values["inner_diameter_mm"].text() == "30"
    assert dialog.name_edit.text() == "Kept user name"
    assert page.session.document == before
    dialog.values["inner_diameter_mm"].setText("5")
    dialog.center.setText("nan")
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert "finite" in dialog.error_label.text()
    assert page.session.document == before
    dialog.center.setText("150")
    # The same real dialog can be corrected without re-entering other fields.
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._component_dialog is None)
    assert page.session.part("kept_key")["name"] == "Kept user name"


def test_place_global_converts_to_resolved_local_and_keeps_length(page, qtbot):
    part = _add(qtbot, page, center=150)
    length = part["length_mm"]
    dialog = _dialog(qtbot, page, "place")
    assert "entrance" in dialog.placement_context.text()
    assert "Axial envelopes overlap" in dialog.summary.text()
    dialog.coordinates.setCurrentIndex(dialog.coordinates.findData("global"))
    assert float(dialog.center.text()) == 650
    dialog.center.selectAll()
    qtbot.keyClicks(dialog.center, "675.125")
    assert "local Z 175.125" in dialog.summary.text()
    _confirm(qtbot, page, dialog)
    moved = page.session.part("custom_part")
    assert moved["local_center_z_mm"] == 175.125
    assert moved["length_mm"] == length
    assert moved["local_start_z_mm"] == 175.125 - length / 2
    page.undo()
    assert page.session.part("custom_part")["local_center_z_mm"] == 150


def test_same_file_independent_copy_can_use_unsaved_source_and_undo(page, qtbot):
    part = _add(qtbot, page)
    original = deepcopy(part)
    path = page.session.path
    bytes_before = path.read_bytes()
    dialog = _dialog(qtbot, page, "copy")
    dialog.key_edit.setText("custom_copy")
    dialog.name_edit.setText("My separate copy")
    dialog.center.setText("190")
    _confirm(qtbot, page, dialog)
    assert page.session.path == path
    assert page.session.part("custom_part") == original
    copy = page.session.part("custom_copy")
    assert copy["name"] == "My separate copy"
    assert copy["local_center_z_mm"] == 190
    assert copy["mechanical_part_role"] == "custom_mechanical_copy"
    assert page._selected_key == "custom_copy"
    assert path.read_bytes() == bytes_before
    page.undo()
    assert "custom_copy" not in page._tree_nodes
    assert "custom_part" in page._tree_nodes
    page.redo()
    assert page._selected_key == "custom_copy"


def test_all_catalog_targets_available_but_unresolved_global_is_disabled(page, qtbot):
    target = _inactive_target(page)
    assert len(page._catalog_paths) > len(page._project_paths)
    dialog = _dialog(qtbot, page, "copy")
    index = dialog.target.findData(str(target))
    assert index >= 0
    dialog.target.setCurrentIndex(index)
    assert str(target) == dialog.target_label.text()
    assert "Global position unavailable" in dialog.coordinate_note.text()
    assert not dialog.coordinates.model().item(1).isEnabled()
    assert dialog.coordinates.currentData() == "local"


def test_cross_file_copy_preserves_dirty_source_and_keeps_dialog_input(page, qtbot):
    _add(qtbot, page)
    source = page.session
    before = deepcopy(source.document)
    target = _inactive_target(page)
    target_bytes = target.read_bytes()
    dialog = _dialog(qtbot, page, "copy")
    dialog.target.setCurrentIndex(dialog.target.findData(str(target)))
    dialog.key_edit.setText("kept_copy_key")
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert "Save or Revert the source draft" in dialog.error_label.text()
    assert dialog.isVisible()
    assert dialog.key_edit.text() == "kept_copy_key"
    assert page.session is source and source.document == before
    assert target.read_bytes() == target_bytes
    # A same-file copy remains available without silently saving the source.
    dialog.target.setCurrentIndex(dialog.target.findData(str(source.path)))
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._component_dialog is None)
    assert page._selected_key == "kept_copy_key"


def test_cross_file_copy_opens_target_draft_and_inactive_save_uses_project_callback(page, qtbot):
    source_path = page.session.path
    source_bytes = source_path.read_bytes()
    target = _inactive_target(page)
    target_bytes = target.read_bytes()
    selections = []
    page.component_selected.connect(selections.append)
    dialog = _dialog(qtbot, page, "copy")
    dialog.target.setCurrentIndex(dialog.target.findData(str(target)))
    dialog.key_edit.setText("copied_coil")
    dialog.center.setText("150")
    _confirm(qtbot, page, dialog)
    assert page.session.path == target
    assert page._selected_key == "copied_coil"
    assert page.session.dirty
    assert source_path.read_bytes() == source_bytes
    assert target.read_bytes() == target_bytes
    assert page._model_runtime_values() == {}
    assert selections == []
    calls = []
    def save_target(path, changes):
        calls.append((path, changes))
        text = module_manifest.stage_manifest_text(path.read_text(encoding="utf-8-sig"), changes)
        path.write_text(text, encoding="utf-8")
    page._project_save = save_target
    assert page.save(), page.status.text()
    assert len(calls) == 1 and calls[0][0] == target
    assert "copied_coil" in {part["key"] for part in module_manifest.read_document(target)["parts"]}
    assert source_path.read_bytes() == source_bytes
    assert not page.session.dirty


@pytest.mark.parametrize("action", ["new", "place", "copy"])
def test_cancel_component_operation_never_mutates_document_or_source(page, qtbot, action):
    original = deepcopy(page.session.document)
    source = page.session.path.read_bytes()
    dialog = _dialog(qtbot, page, action)
    dialog.center.setText("123.456")
    qtbot.mouseClick(dialog.buttons.button(dialog.buttons.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._component_dialog is None)
    assert page.session.document == original
    assert page.session.path.read_bytes() == source


@pytest.mark.parametrize("changed", ["source", "target", "global_origin", "upstream_source"])
def test_changed_input_or_coordinate_frame_blocks_confirmation_without_losing_values(page, qtbot, changed):
    original = deepcopy(page.session.document)
    dialog = _dialog(qtbot, page, "copy")
    if changed == "target":
        target = _inactive_target(page)
        dialog.target.setCurrentIndex(dialog.target.findData(str(target)))
        target.write_bytes(target.read_bytes() + b"\n# revised while reviewing\n")
    elif changed == "source":
        source = page.session.path
        source.write_bytes(source.read_bytes() + b"\n# revised while reviewing\n")
    else:
        dialog.coordinates.setCurrentIndex(dialog.coordinates.findData("global"))
        if changed == "global_origin":
            page._module_origins[page.session.path] += 1
        else:
            upstream = page._project_root / COLUMN
            upstream.write_bytes(upstream.read_bytes() + b"\n# upstream assembly revised\n")
    dialog.key_edit.setText("reviewed_copy")
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert dialog.isVisible() and dialog.error_label.isVisible()
    assert "changed" in dialog.error_label.text()
    assert dialog.key_edit.text() == "reviewed_copy"
    assert page.session.document == original


def test_component_dialog_long_paths_and_parent_names_fit_without_horizontal_scroll(page, qtbot):
    dialog = _dialog(qtbot, page, "new")
    dialog.resize(620, 680)
    qtbot.wait(20)
    scroll = dialog.findChild(QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert scroll.widget().width() <= scroll.viewport().width()
    assert dialog.width() <= 680
    assert dialog.confirm_button.isVisible()
    assert dialog.target_label.toPlainText() == str(page.session.path)
    assert dialog.target_label.horizontalScrollBar().maximum() == 0
