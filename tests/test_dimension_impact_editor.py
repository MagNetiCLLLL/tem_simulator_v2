"""Editor explanations track the active mode without altering drafts or physics."""

from copy import deepcopy
import shutil
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "project_and_recording_system/EnergyFilter.toml"
COLUMN = "column/C3_ProbeCorrector.toml"
COIL = "intermediate_lens_excitation_coil"
C2 = "condenser_aperture_2"
RECIPE = {"solver": "axisymmetric_linear_fem", "ampere_turns": 1000., "relative_permeability": 500.}


@pytest.fixture
def page(qtbot, tmp_path):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    widget = PartModelEditorPage()
    qtbot.addWidget(widget)
    widget.set_project_context(root, SimpleNamespace(selected_module_paths=(("column", COLUMN), ("recording", MODULE))),
                               None, runtime_values={C2: {"radius_mm": .05}})
    assert widget.open_path(root / MODULE, selected_key=COIL)
    widget.set_simulation_context("ideal", {})
    return widget


def _row(page, key, field):
    return next(row for row in range(page.dimensions.rowCount())
                if tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)) == ("parts", key, field))


def test_c2_parameter_meaning_source_and_impact_are_separate(page):
    assert page.open_path(page._project_root / COLUMN, selected_key=C2)
    for field, category in (("length_mm", "envelope"), ("mechanical_outer_diameter_mm", "envelope"),
                            ("mechanical_bore_diameter_mm", "physical"), ("plate_thickness_mm", "physical"),
                            ("vacuum_inner_diameter_mm", "vacuum")):
        row = _row(page, C2, field)
        value = page.dimensions.item(row, 1)
        meaning = value.data(Qt.ItemDataRole.UserRole + 3)
        assert meaning.category == category
        assert meaning.source_kind != "measured"
        page.dimensions.setCurrentCell(row, 0)
        assert meaning.category_label in page.parameter_detail.toPlainText()
        assert "Evidence:" in page.parameter_detail.toPlainText()
    row = _row(page, C2, "plate_thickness_mm")
    impact = page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole)
    assert impact.status == "inactive" and "runtime opening" in impact.detail
    assert not page.session.dirty


def test_mode_changes_explain_coil_field_usage_without_losing_invalid_draft(page):
    row = _row(page, COIL, "mechanical_outer_diameter_mm")
    assert page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole).status == "inactive"
    source = page.session.path.read_bytes()
    page.dimensions.item(row, 1).setText("invalid diameter")
    invalid = deepcopy(page._invalid_inputs)
    page.set_simulation_context("linear_geometry", {"intermediate_lens": RECIPE})
    assert page.dimensions.item(row, 1).text() == "invalid diameter"
    assert page._invalid_inputs == invalid
    assert page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole).status == "active"
    assert "Unsaved draft" in page.calculation_label.text()
    assert page.session.path.read_bytes() == source


def test_missing_recipe_is_setup_not_a_claim_of_geometry_field_solution(page):
    page.set_simulation_context("linear_geometry", {})
    row = _row(page, COIL, "mechanical_outer_diameter_mm")
    impact = page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole)
    assert impact.status == "configuration_required"
    assert page.dimensions.item(row, 3).text() == "Setup"
    page.set_calculation_status("current", "Accepted preview")
    assert "for active model" in page.calculation_label.text()
    assert "Field setup required" in page.calculation_label.text()
    assert "configure" in page.calculation_label.toolTip().lower()


def test_cad_parameters_remain_visibly_excluded_after_save_and_calculation(page):
    page.set_simulation_context("linear_geometry", {"intermediate_lens": RECIPE})
    original = deepcopy(page.session.part(COIL))
    page.session.set_model_3d(COIL, {"schema_version": 1, "base": {"kind": "existing"},
                                    "transform": {"scale_xy": [1.1, 1.]}})
    page._draft_changed()
    assert "CAD changes excluded from physics" in page.calculation_label.text()
    assert "model_3d.transform.scale_xy[0]" in page.calculation_label.toolTip()
    assert "Unsaved draft" in page.calculation_label.text()
    rows = [row for row in range(page.dimensions.rowCount())
            if page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[2] == "model_3d"]
    assert rows and all(page.dimensions.item(row, 3).text() == "CAD only" for row in rows)
    assert page.save(), page.status.text()
    page.set_calculation_status("current", "Accepted preview")
    assert "Unsaved draft" not in page.calculation_label.text()
    assert "CAD changes excluded" in page.calculation_label.text()
    for name in ("length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"):
        assert page.session.part(COIL)[name] == original[name]


def test_external_file_is_not_reported_as_active_instrument_geometry(page, tmp_path):
    copy = tmp_path / "external.toml"
    shutil.copyfile(page.session.path, copy)
    assert page.open_path(copy, selected_key=COIL)
    page.set_simulation_context("linear_geometry", {"intermediate_lens": RECIPE})
    row = _row(page, COIL, "mechanical_outer_diameter_mm")
    assert page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole).status == "unknown"
    assert "Not linked" in page.calculation_label.text()


def test_result_status_does_not_mark_unsaved_geometry_as_calculated(page):
    page.set_calculation_status("current", "Saved preview complete")
    row = _row(page, COIL, "mechanical_outer_diameter_mm")
    page.dimensions.item(row, 1).setText("92")
    page.set_calculation_status("running", "Calculating saved model")
    assert "Unsaved draft" in page.calculation_label.text()
    assert "Result current" not in page.calculation_label.text()
    page.revert()
    page.set_calculation_status("stale", "Geometry was saved")
    assert "Results out of date" in page.calculation_label.text()
    page.set_calculation_status("failed", "Mesh invalid")
    assert "Calculation failed" in page.calculation_label.text()
    assert "Mesh invalid" in page.calculation_label.toolTip()


def test_audit_filters_and_navigation_preserve_component_identity(page, qtbot):
    dialog = page.show_dimension_audit()
    qtbot.addWidget(dialog)
    assert dialog.audit.module_count == 11 and dialog.audit.part_count == 482
    dialog.search.setText(C2)
    assert dialog.table.rowCount()
    dialog.filter.setCurrentIndex(dialog.filter.findData("envelope"))
    assert dialog.table.rowCount()
    item = dialog.table.item(0, 0)
    source, key, path = item.data(Qt.ItemDataRole.UserRole)
    dialog._activate(0, 0)
    assert page.session.path.as_posix() == str(source).replace("\\", "/")
    assert page._selected_key == key == C2
    assert "plate_thickness_mm" not in path


def test_save_copy_removes_live_opening_and_active_usage_from_independent_file(page, tmp_path, monkeypatch):
    assert page.open_path(page._project_root / COLUMN, selected_key=C2)
    source = page.session.path
    original = source.read_bytes()
    page.dimensions.item(_row(page, C2, "plate_thickness_mm"), 1).setText("0.25")
    assert any(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[0] == "runtime"
               for row in range(page.dimensions.rowCount()))
    assert any("working_opening" in record["surfaces"] for record in page._mesh_records)
    destination = tmp_path / "independent-column.toml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(destination), ""))
    page.save_copy()
    assert page.session.path == destination.resolve()
    assert source.read_bytes() == original
    assert not page.session.dirty
    assert page.session.part(C2)["plate_thickness_mm"] == 0.25
    assert page._model_runtime_values() == {}
    assert "Not linked" in page.calculation_label.text()
    assert all(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[0] != "runtime"
               for row in range(page.dimensions.rowCount()))
    assert all(page.dimensions.item(row, 3).data(Qt.ItemDataRole.UserRole).status == "unknown"
               for row in range(page.dimensions.rowCount()))
    assert not any("working_opening" in record["face_groups"] for record in page._mesh_records)


@pytest.mark.parametrize("path", [("geometry", "length_mm"), ("ports", "entrance", "local_z_mm")])
def test_audit_module_paths_show_full_names_and_locate_all_parameters(page, qtbot, path):
    from temsim.parameter_semantics import describe_parameter

    dialog = page.show_dimension_audit()
    qtbot.addWidget(dialog)
    selected = []
    page.component_selected.connect(selected.append)
    row = next(row for row in range(dialog.table.rowCount())
               if dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole)[2] == path
               and str(dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole)[0]).replace("\\", "/").endswith(COLUMN))
    assert dialog.table.item(row, 1).text() == ".".join(path)
    dialog._activate(row, 0)
    assert page.parameter_tabs.currentWidget() is page.parameters
    current = page.parameters.currentItem()
    assert tuple(current.data(0, Qt.ItemDataRole.UserRole)) == path
    assert current.isSelected() and current.parent().isExpanded()
    assert describe_parameter(page.session.document, path).label in page.parameter_detail.toPlainText()
    assert selected == []


@pytest.mark.parametrize("field,exists", [("aperture_plate_form", True), ("radius_mm", False)])
def test_audit_nonnumeric_and_missing_fields_do_not_substitute_an_unrelated_dimension(page, qtbot, field, exists):
    dialog = page.show_dimension_audit()
    qtbot.addWidget(dialog)
    selected = []
    page.component_selected.connect(selected.append)
    path = ("parts", C2, field)
    row = next(row for row in range(dialog.table.rowCount())
               if dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole)[2] == path
               and str(dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole)[0]).replace("\\", "/").endswith(COLUMN))
    dialog._activate(row, 0)
    assert page._selected_key == C2
    assert page.parameter_tabs.currentWidget() is page.parameters
    if exists:
        assert tuple(page.parameters.currentItem().data(0, Qt.ItemDataRole.UserRole)) == path
    else:
        assert not page.parameters.selectedItems()
        assert field in page.status.text() and "not defined" in page.status.text()
        assert page.parameter_detail.toPlainText() == page.status.text()
    assert not page.session.dirty
    assert selected == [C2]


def test_audit_external_component_does_not_select_an_active_instrument_part(page, tmp_path):
    source = tmp_path / "external-audit.toml"
    shutil.copyfile(page._project_root / COLUMN, source)
    selected = []
    page.component_selected.connect(selected.append)
    path = ("parts", C2, "plate_thickness_mm")
    page._audit_parameter_requested(str(source), C2, path)
    assert page.session.path == source.resolve()
    assert page._selected_key == C2
    assert tuple(page.dimensions.item(page.dimensions.currentRow(), 1).data(Qt.ItemDataRole.UserRole)) == path
    assert selected == []
