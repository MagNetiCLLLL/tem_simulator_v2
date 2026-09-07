"""Unified meaning/impact explanations preserve the actual parameter editors."""

from copy import deepcopy
import shutil
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit

from temsim import module_manifest
from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.part_geometry_editor import GeometryEditorDialog
from temsim.manifest_editor import ManifestEditor, ManifestField, ManifestTarget
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.runtime_parameters import RuntimeTarget


COLUMN = "column/C3_ProbeCorrector_ImageCorrector.toml"
RECORDING = "project_and_recording_system/EnergyFilter.toml"
APERTURE = "condenser_aperture_2"
COIL = "intermediate_lens_excitation_coil"


@pytest.fixture
def sources(tmp_path):
    originals = {}
    for module in (COLUMN, RECORDING):
        source = INSTRUMENT_CONFIG_ROOT / module
        originals[source] = source.read_bytes()
        target = tmp_path / module
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    yield tmp_path
    assert all(source.read_bytes() == contents for source, contents in originals.items())


@pytest.fixture
def panel_factory(qtbot, sources):
    def create(key=APERTURE, *, runtime=None):
        module = COLUMN if key == APERTURE else RECORDING
        document = module_manifest.read_document(sources / module)
        by_key = {part["key"]: part for part in document["parts"]}
        target = ManifestTarget(module, key)
        panel = ParameterPanel()
        qtbot.addWidget(panel)
        panel.set_context(by_key[key]["name"], runtime, target,
                          ManifestEditor(sources).fields(target), None,
                          geometry_parent=by_key.get(by_key[key].get("parent_key")))
        panel.tabs.setCurrentIndex(1)
        panel.resize(380, 700)
        panel.show()
        return panel, by_key
    return create


def _select_manifest(panel, field):
    item = next(panel.manifest_table.item(row, 1)
                for row in range(panel.manifest_table.rowCount())
                if panel.manifest_table.item(row, 0).text() == field)
    panel.manifest_table.setCurrentItem(item)
    return item


def _table_texts(table):
    return [[table.item(row, col).text() for col in range(table.columnCount())]
            for row in range(table.rowCount())]


def test_manifest_fields_append_meaning_without_replacing_raw_keys_or_values(sources):
    legacy = ManifestField(("parts", "test", "length_mm"), "length_mm", 20.0)
    assert legacy.meaning is None and legacy.editable
    document = module_manifest.read_document(sources / COLUMN)
    part = next(part for part in document["parts"] if part["key"] == APERTURE)
    fields = ManifestEditor(sources).fields(ManifestTarget(COLUMN, APERTURE))
    assert {field.label: field.value for field in fields} == part
    assert all(field.path == ("parts", APERTURE, field.label) for field in fields)
    by_name = {field.label: field for field in fields}
    assert by_name["length_mm"].meaning.label == "Mechanism envelope length"
    assert by_name["plate_thickness_mm"].meaning.label == "Aperture plate thickness"
    assert by_name["mechanical_bore_diameter_mm"].meaning.label == "Carrier bore diameter"
    assert by_name["length_mm"].meaning.category != by_name["plate_thickness_mm"].meaning.category


def test_toml_details_show_meaning_source_and_disconnected_context(panel_factory):
    panel, by_key = panel_factory()
    for field in ("length_mm", "plate_thickness_mm", "mechanical_bore_diameter_mm"):
        item = _select_manifest(panel, field)
        path = tuple(item.data(Qt.ItemDataRole.UserRole))
        meaning = describe_parameter(by_key[APERTURE], path, by_key=by_key)
        text = panel.parameter_details.text()
        assert meaning.label in text and meaning.category_label in text
        assert meaning.source_label in text and meaning.source_note in text
        assert "Simulation mode: not connected" in text
        assert panel.manifest_table.item(item.row(), 0).text() == field
        assert meaning.description in item.toolTip()
        assert "TOML key: " + field in item.toolTip()
    QApplication.processEvents()
    assert panel.parameter_details.wordWrap()
    assert panel.parameter_details.width() <= panel.width()
    assert panel.minimumSizeHint().width() <= 380


@pytest.mark.parametrize("mode", [None, "ideal", "analytical", "linear_geometry", "nonlinear_material"])
def test_mode_context_updates_impact_without_editing_or_rebuilding_table(panel_factory, mode):
    panel, by_key = panel_factory(COIL)
    item = _select_manifest(panel, "mechanical_inner_diameter_mm")
    item.setText("invalid draft text")
    snapshot = _table_texts(panel.manifest_table)
    signals = []
    panel.runtime_changed.connect(lambda *args: signals.append(args))
    panel.manifest_save_requested.connect(lambda *args: signals.append(args))
    descriptors = {}
    panel.set_simulation_context(mode, descriptors, by_key)
    assert panel.manifest_table.item(item.row(), 1) is item
    assert _table_texts(panel.manifest_table) == snapshot
    assert panel.manifest_draft_texts(panel._manifest_target)[tuple(item.data(Qt.ItemDataRole.UserRole))] == "invalid draft text"
    expected = describe_parameter_impact(by_key[COIL], ("parts", COIL, "mechanical_inner_diameter_mm"),
        by_key=by_key, simulation_mode=mode, descriptors=descriptors)
    assert expected.label in panel.parameter_details.text()
    assert expected.detail in panel.parameter_details.text()
    assert signals == []


def test_context_refresh_preserves_uncommitted_keyboard_delegate(panel_factory, qtbot):
    panel, by_key = panel_factory()
    item = _select_manifest(panel, "plate_thickness_mm")
    table = panel.manifest_table
    table.scrollToItem(item)
    table.editItem(item)
    QApplication.processEvents()
    editor = next(widget for widget in table.findChildren(QLineEdit) if widget.isVisible())
    editor.selectAll()
    qtbot.keyClicks(editor, "0.37")
    cursor = editor.cursorPosition()
    panel.set_simulation_context("analytical", {}, by_key)
    assert editor.isVisible() and editor.text() == "0.37"
    assert editor.cursorPosition() == cursor
    assert item.text() == "0.2"
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: item.text() == "0.37")


def test_operating_details_use_runtime_path_and_leave_live_editing_intact(panel_factory):
    obj = SimpleNamespace(radius_mm=0.05, enabled=True, offset_x_mm=0.0, offset_y_mm=0.0)
    target = RuntimeTarget(APERTURE, "C2 aperture", obj)
    panel, by_key = panel_factory(runtime=target)
    panel.set_simulation_context("ideal", {}, by_key)
    panel.tabs.setCurrentIndex(0)
    item = next(panel.runtime_table.item(row, 1) for row in range(panel.runtime_table.rowCount())
                if panel.runtime_table.item(row, 0).text() == "radius_mm")
    panel.runtime_table.setCurrentItem(item)
    path = ("runtime", APERTURE, "radius_mm")
    meaning = describe_parameter(by_key[APERTURE], path, by_key=by_key)
    impact = describe_parameter_impact(by_key[APERTURE], path, by_key=by_key, simulation_mode="ideal")
    assert meaning.source_label in panel.parameter_details.text()
    assert impact.label in panel.parameter_details.text()
    assert impact.active is True
    item.setText("0.1")
    assert obj.radius_mm == 0.1
    assert panel.runtime_table.item(item.row(), 0).text() == "radius_mm"
    assert item.data(Qt.ItemDataRole.UserRole) == 0.1


def test_geometry_editor_uses_semantics_and_sources_without_resetting_numeric_input(sources, qtbot):
    document = module_manifest.read_document(sources / RECORDING)
    by_key = {part["key"]: part for part in document["parts"]}
    original = deepcopy(by_key[COIL])
    dialog = GeometryEditorDialog(original, lambda updates: None)
    qtbot.addWidget(dialog)
    dialog.show()
    QApplication.processEvents()
    assert "Simulation mode: not connected" in dialog.parameter_details.text()
    meaning = describe_parameter(original, ("parts", COIL, "length_mm"), by_key=by_key)
    assert meaning.source_label in dialog.dimension_source.text()
    assert dialog.parameters_scroll.viewport().rect().intersects(
        dialog.dimension_source.geometry().translated(dialog.dimension_source.parentWidget().mapTo(
            dialog.parameters_scroll.viewport(), dialog.dimension_source.parentWidget().rect().topLeft())))
    for dimension, path in (
        ("length_mm", ("parts", COIL, "length_mm")),
        ("inner_diameter_mm", ("parts", COIL, "mechanical_inner_diameter_mm")),
        ("outer_diameter_mm", ("parts", COIL, "mechanical_outer_diameter_mm")),
        ("thickness_mm", ("derived", COIL, "radial_thickness_mm")),
    ):
        meaning = describe_parameter(original, path, by_key=by_key)
        assert dialog.dimension_labels[dimension].text() == meaning.label
        assert dialog.dimension_labels[dimension].wordWrap()
        assert meaning.source_note in dialog.dimension_controls[dimension].toolTip()
    control = dialog.dimension_controls["inner_diameter_mm"]
    control.setFocus()
    control.lineEdit().selectAll()
    qtbot.keyClicks(control.lineEdit(), "70")
    text, cursor, draft = control.lineEdit().text(), control.lineEdit().cursorPosition(), dialog.geometry
    dialog.set_simulation_context("linear_geometry", {}, by_key)
    assert control.lineEdit().text() == text
    assert control.lineEdit().cursorPosition() == cursor
    assert dialog.geometry == draft
    assert "Simulation mode: linear_geometry" in dialog.parameter_details.text()
    assert original == by_key[COIL]
    dialog.dimension_controls["thickness_mm"].setFocus()
    QApplication.processEvents()
    meaning = describe_parameter(original, ("derived", COIL, "radial_thickness_mm"), by_key=by_key)
    assert meaning.source_note in dialog.parameter_details.text()
    assert "Derived" in meaning.source_note or "derived" in meaning.source_note


def test_physical_layout_selection_uses_same_semantics_without_plot_or_signal_changes(sources, qtbot):
    from temsim.gui.diagnostic_tabs import PhysicalLayoutView

    document = module_manifest.read_document(sources / COLUMN)
    by_key = {part["key"]: part for part in document["parts"]}
    data = by_key[APERTURE]
    part = SimpleNamespace(key=APERTURE, data=data)
    view = PhysicalLayoutView()
    qtbot.addWidget(view)
    view._record_by_key[APERTURE] = SimpleNamespace(
        key=APERTURE, name=data["name"], start_z_mm=data["local_start_z_mm"],
        center_z_mm=data["local_center_z_mm"], end_z_mm=data["local_end_z_mm"],
        profile=view.APERTURE_MECHANISM_PROFILE,
        outer_diameter_mm=data["mechanical_outer_diameter_mm"],
        mechanical_bore_diameter_mm=data["mechanical_bore_diameter_mm"],
        vacuum_inner_diameter_mm=data["vacuum_inner_diameter_mm"],
        optical_references_mm=(data["optical_reference_local_z_mm"],),
        active_length_mm=data["plate_thickness_mm"], bore_diameter_mm=0.1,
        excitation_enabled=True,
    )
    emitted = []
    view.component_selected.connect(lambda *args: emitted.append(args))
    view.component_activated.connect(lambda *args: emitted.append(args))
    view.focus_component(part)
    highlight = view._highlight
    limits = deepcopy(view.plot.getViewBox().viewRange())
    for field in ("length_mm", "mechanical_outer_diameter_mm", "mechanical_bore_diameter_mm", "plate_thickness_mm"):
        meaning = describe_parameter(data, ("parts", APERTURE, field), by_key=by_key)
        assert meaning.label in view.summary.text()
        assert meaning.category_label in view.summary.text()
        assert meaning.source_note in view.summary.toolTip()
    assert "Simulation mode: not connected" in view.summary.toolTip()
    view.set_parameter_semantics_context("ideal", {}, by_key)
    assert "Simulation mode: ideal" in view.summary.toolTip()
    assert view._highlight is highlight
    assert view.plot.getViewBox().viewRange() == limits
    assert emitted == []
