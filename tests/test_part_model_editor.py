"""The 3-D workspace stages real module edits and preserves instrument state."""

from copy import deepcopy
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui import main_window
from temsim.gui.diagnostic_tabs import _selectable_key_at_scene_position
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.manifest_editor import ManifestEditor
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.simulation import run
from temsim.simulation_pipeline import CalculationResult


MODULE = "project_and_recording_system/EnergyFilter.toml"
COIL = "intermediate_lens_excitation_coil"
HOUSING = "intermediate_lens_housing"
COLUMN = "column/C3_ProbeCorrector_ImageCorrector.toml"


@pytest.fixture
def root(tmp_path):
    originals = {path: path.read_bytes() for path in INSTRUMENT_CONFIG_ROOT.rglob("*.toml")}
    copied = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, copied)
    yield copied
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.fixture
def page(qtbot, root):
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.set_project_context(root, SimpleNamespace(selected_module_paths=(
        ("project_and_recording_system", MODULE), ("column", COLUMN))), None,
        runtime_values={"intermediate_lens": {"percent": 34.56, "cs_mm": .75}})
    assert page.open_path(root / MODULE, selected_key=COIL)
    return page


@pytest.fixture
def window(qtbot, root, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(
        str(tmp_path / "window.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    return widget


@pytest.fixture
def navigation_window(window, qtbot):
    """A small real preview supplies both geometry and linked axial markers."""
    state = window.state
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 12.0
    state.sample.diffraction_enabled = False
    resolved = apply_physical_layout_to_state(state)
    result = CalculationResult(simulation=run(state, resolved_layout=resolved),
                               energy_filter=None, state_snapshot=state,
                               layout=resolved, assembly=window.assembly)
    window.resize(1700, 1050)
    window.show()
    window.workspace.auto_zoom.setChecked(False)
    window.workspace.display_result(result, "Preview")
    layout = window.workspace.physical_layout
    window.workspace.tabs.setCurrentWidget(layout)
    layout.tabs.setCurrentWidget(layout.section_page)
    qtbot.waitUntil(lambda: COIL in layout._record_by_key)
    record = layout._record_by_key[COIL]
    layout.plot.setXRange(record.start_z_mm - 30, record.end_z_mm + 30, padding=0)
    layout.plot.setYRange(-100, 100, padding=0)
    qtbot.wait(30)
    return window


def _double_click_scene(qtbot, layout, scene_position):
    viewport = layout.plot.viewport()
    world_position = layout.plot.getViewBox().mapSceneToView(scene_position)
    point = layout.plot.mapFromScene(scene_position)
    assert viewport.rect().contains(point)
    qtbot.mouseMove(viewport, pos=point)
    # Include the first ordinary click of a real double-click sequence. It must
    # keep even Energy Filter components in Physical Layout until activation.
    qtbot.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
    QApplication.processEvents()
    assert layout.isVisible() and layout.tabs.currentWidget() is layout.section_page
    point = layout.plot.mapFromScene(layout.plot.getViewBox().mapViewToScene(world_position))
    qtbot.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=point)
    # PyQtGraph creates its MouseClickEvent on release of the double press.
    qtbot.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=point)
    QApplication.processEvents()


def _coil_scene_point(layout):
    rectangle = next(item for item in layout._lens_excitation_coil_items[COIL]
                     if item.rect().center().y() > 0)
    # Avoid the centre marker and packed annotation leaders through the centre.
    point = rectangle.rect().center()
    point.setX(point.x() + rectangle.rect().width() * 0.17)
    return rectangle.mapToScene(point)


def _pan_model(qtbot, view):
    start = QPoint(view.width() // 2, view.height() // 2)
    end = start + QPoint(95, 61)
    qtbot.mousePress(view, Qt.MouseButton.RightButton, pos=start)
    QApplication.sendEvent(view, QMouseEvent(
        QEvent.Type.MouseMove, QPointF(end), QPointF(view.mapToGlobal(end)),
        Qt.MouseButton.NoButton, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier,
    ))
    qtbot.mouseRelease(view, Qt.MouseButton.RightButton, pos=end)
    QApplication.processEvents()


def _dimension(page, key, field, *indices):
    path = ("parts", key, field, *indices)
    for row in range(page.dimensions.rowCount()):
        item = page.dimensions.item(row, 1)
        if tuple(item.data(Qt.ItemDataRole.UserRole)) == path:
            return item
    raise AssertionError(f"Dimension is missing: {path}")


def _edit(page, key, field, value, *indices):
    item = _dimension(page, key, field, *indices)
    assert item.flags() & Qt.ItemFlag.ItemIsEditable
    item.setText(str(value))


def _vertices(page, key, region="body"):
    return np.concatenate([record["vertices"] for record in page._mesh_records
                           if record["key"] == key and record["region"] == region])


def _working_opening_radii(page, key):
    vertices = []
    for record in page._mesh_records:
        if record["key"] == key:
            faces = record["faces"][record["face_groups"] == "working_opening"]
            vertices.extend(record["vertices"][np.unique(faces)].tolist())
    assert vertices, "The operating opening must have a real inner surface"
    return np.linalg.norm(np.asarray(vertices)[:, :2], axis=1)


def _group(page, label):
    for index in range(page.parameters.topLevelItemCount()):
        node = page.parameters.topLevelItem(index)
        if node.text(0) == label:
            return node
    raise AssertionError(f"Parameter group is missing: {label}")


def test_selected_part_exposes_full_source_parent_and_operating_parameters(page):
    assert page._selected_key == COIL
    assert page.selection_label.text() == page.session.part(COIL)["name"]
    selected = _group(page, "Selected component")
    assert {selected.child(i).text(0) for i in range(selected.childCount())} == set(page.session.part(COIL))
    parent = _group(page, "Parent assembly")
    assert {parent.child(i).text(0) for i in range(parent.childCount())} == set(page.session.part("intermediate_lens"))
    operating = _group(page, "Parent operating values (read-only)")
    values = {operating.child(i).text(0): operating.child(i).text(1) for i in range(operating.childCount())}
    assert float(values["percent"]) == pytest.approx(34.56)
    assert float(values["cs_mm"]) == pytest.approx(.75)
    assert not operating.child(0).flags() & Qt.ItemFlag.ItemIsEditable
    assert _group(page, "Module") and _group(page, "Geometry") and _group(page, "Ports")


@pytest.mark.parametrize("field,value", [("length_mm", 178), ("mechanical_inner_diameter_mm", 70),
                                         ("mechanical_outer_diameter_mm", 95)])
def test_numeric_dimension_changes_mesh_and_draft_before_any_file_write(page, field, value):
    original = page.session.path.read_bytes()
    vertices = _vertices(page, COIL).copy()
    center = page.session.part(COIL)["local_center_z_mm"]
    _edit(page, COIL, field, value)
    assert page.session.part(COIL)[field] == value
    assert page.session.part(COIL)["local_center_z_mm"] == center
    assert not np.array_equal(_vertices(page, COIL), vertices)
    assert page.session.path.read_bytes() == original
    assert page.session.dirty and page.save_button.isEnabled()


def test_aperture_labels_distinguish_plate_from_mechanism_and_remain_readable(page, root, qtbot):
    from temsim.part_model_3d import part_dimension_specs

    key = "condenser_aperture_2"
    assert page.open_path(root / COLUMN, selected_key=key)
    page.resize(1000, 650)
    page.show()
    page.splitter.setSizes([160, 490, 330])
    qtbot.wait(30)
    fields = {field.path[2]: field for field in part_dimension_specs(page.session.document, key)}
    for name, expected in {
        "length_mm": "Mechanism envelope length",
        "mechanical_outer_diameter_mm": "Mechanism envelope diameter",
        "mechanical_bore_diameter_mm": "Carrier bore diameter",
        "plate_thickness_mm": "Aperture plate thickness",
    }.items():
        value = _dimension(page, key, name)
        label = page.dimensions.item(value.row(), 0)
        assert label.text() == expected
        assert fields[name].reason
        assert label.toolTip() == value.toolTip()
        assert fields[name].reason in value.toolTip()
        assert "Evidence:" in value.toolTip() and "Use:" in value.toolTip()
        text_height = page.dimensions.fontMetrics().boundingRect(
            QRect(0, 0, page.dimensions.columnWidth(0) - 12, 10000),
            Qt.TextFlag.TextWordWrap, label.text(),
        ).height()
        assert page.dimensions.rowHeight(value.row()) >= text_height + 2
    # Existing generic magnetic component labels keep their familiar wording.
    assert page.open_path(root / MODULE, selected_key=COIL)
    for name, expected in (("mechanical_inner_diameter_mm", "Material inner diameter"),
                           ("mechanical_outer_diameter_mm", "Material outer diameter")):
        value = _dimension(page, COIL, name)
        assert page.dimensions.item(value.row(), 0).text() == expected


def test_real_c2_plate_thickness_changes_thin_mesh_without_resizing_20mm_mechanism(page, root):
    key = "condenser_aperture_2"
    assert page.open_path(root / COLUMN, selected_key=key)
    original = deepcopy(page.session.part(key))
    before = page.session.path.read_bytes()
    assert original["plate_thickness_mm"] == pytest.approx(0.2)
    assert original["length_mm"] == pytest.approx(20.0)
    assert np.ptp(_vertices(page, key)[:, 2]) == pytest.approx(0.2)
    for thickness in (0.35, 0.2):
        _edit(page, key, "plate_thickness_mm", thickness)
        assert not page._invalid_inputs, page.status.text()
        assert page.session.part(key) == {**original, "plate_thickness_mm": thickness}
        vertices = _vertices(page, key)
        assert np.ptp(vertices[:, 2]) == pytest.approx(thickness)
        assert (vertices[:, 2].min() + vertices[:, 2].max()) / 2 == pytest.approx(original["local_center_z_mm"])
        assert page.session.path.read_bytes() == before
        assert page.save(), page.status.text()
        saved = module_manifest.read_document(page.session.path)
        part = next(part for part in saved["parts"] if part["key"] == key)
        assert part == {**original, "plate_thickness_mm": thickness}
        assert np.ptp(_vertices(page, key)[:, 2]) == pytest.approx(thickness)
        before = page.session.path.read_bytes()


@pytest.mark.parametrize("scope", ["part", "module"])
def test_live_aperture_opening_updates_mesh_but_keeps_draft_and_selected_face(page, root, qtbot, scope):
    key = "condenser_aperture_2"
    page.set_runtime_values({key: {"radius_mm": 0.05}})
    assert page.open_path(root / COLUMN, selected_key=key)
    _edit(page, key, "plate_thickness_mm", 0.25)
    page.resize(1280, 760)
    page.show()
    page.view.set_axial_view()
    QApplication.processEvents()
    record = next(record for record in page._mesh_records if record["key"] == key)
    triangles = record["vertices"][record["faces"]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    center = triangles[np.argmax(normals[:, 2])].mean(axis=0)
    projected = page.view.project_points([center])[0]
    position = QPoint(round(projected[0]), round(projected[1]))
    qtbot.mouseClick(page.view, Qt.MouseButton.LeftButton, pos=position)
    assert page._topology_selection
    identity = [(hit["key"], hit["kind"], hit["id"]) for hit in page._topology_selection]
    page.scope.setCurrentIndex(page.scope.findData(scope))
    # Retain even invalid text while a separate runtime control is changing.
    _edit(page, key, "length_mm", "invalid")
    draft, invalid = deepcopy(page.session.document), dict(page._invalid_inputs)
    assert invalid
    rotation = page.view._rotation.copy()
    before = page.session.path.read_bytes()
    assert _working_opening_radii(page, key) == pytest.approx(0.05)
    page.set_runtime_values({key: {"radius_mm": 0.1}})
    assert page.session.document == draft
    assert page._invalid_inputs == invalid
    assert page.session.path.read_bytes() == before
    assert page.session.dirty
    assert page.view._rotation == pytest.approx(rotation)
    assert [(hit["key"], hit["kind"], hit["id"]) for hit in page._topology_selection] == identity
    assert _working_opening_radii(page, key) == pytest.approx(0.1)
    runtime_rows = [page.dimensions.item(row, 1) for row in range(page.dimensions.rowCount())
                    if page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[0] == "runtime"]
    assert len(runtime_rows) == 1
    value = runtime_rows[0]
    assert float(value.text()) == pytest.approx(0.2)
    assert page.dimensions.item(value.row(), 0).text() == "Working opening diameter"
    assert not value.flags() & Qt.ItemFlag.ItemIsEditable
    assert page.dimensions.item(value.row(), 2).text() == "mm"


@pytest.mark.parametrize("inside_project", [False, True])
def test_external_and_inactive_toml_do_not_borrow_live_aperture_opening(page, root, inside_project):
    key = "condenser_aperture_2"
    destination = (root if inside_project else root.parent) / "independent_column.toml"
    shutil.copyfile(root / COLUMN, destination)
    page.set_runtime_values({key: {"radius_mm": 0.05}})
    assert page.open_path(destination, selected_key=key)
    before = _vertices(page, key).copy()
    assert page._model_runtime_values() == {}
    page.set_runtime_values({key: {"radius_mm": 0.1}})
    assert _vertices(page, key) == pytest.approx(before)
    assert all(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)[0] != "runtime"
               for row in range(page.dimensions.rowCount()))
    assert all("operating values" not in page.parameters.topLevelItem(index).text(0)
               for index in range(page.parameters.topLevelItemCount()))
    assert not page.session.dirty


def test_pole_bore_22_changes_real_hole_and_paired_poles_can_be_saved(page):
    source = page.session.path.read_bytes()
    for side in ("upper", "lower"):
        key = f"intermediate_lens_{side}_pole"
        page.select_part(key)
        before = _vertices(page, key).copy()
        _edit(page, key, "mechanical_bore_diameter_mm", 22)
        assert not np.array_equal(before, _vertices(page, key))
        assert np.linalg.norm(_vertices(page, key)[:, :2], axis=1).min() == pytest.approx(11)
    assert page.session.path.read_bytes() == source
    assert page.save(), page.status.text()
    saved = module_manifest.read_document(page.session.path)
    assert all(next(part for part in saved["parts"] if part["key"] == f"intermediate_lens_{side}_pole")
               ["mechanical_bore_diameter_mm"] == 22 for side in ("upper", "lower"))


def test_part_selection_preserves_shared_draft_and_undo_redo_revert(page):
    original = deepcopy(page.session.document)
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    page.select_part(HOUSING)
    _edit(page, HOUSING, "mechanical_inner_diameter_mm", 176)
    assert page.session.part(COIL)["mechanical_inner_diameter_mm"] == 70
    page.undo()
    assert page._selected_key == HOUSING
    assert page.session.part(HOUSING)["mechanical_inner_diameter_mm"] == next(
        part for part in original["parts"] if part["key"] == HOUSING)["mechanical_inner_diameter_mm"]
    assert page.session.part(COIL)["mechanical_inner_diameter_mm"] == 70
    page.redo()
    assert page.session.part(HOUSING)["mechanical_inner_diameter_mm"] == 176
    page.select_part(COIL)
    assert float(_dimension(page, COIL, "mechanical_inner_diameter_mm").text()) == 70
    page.revert()
    assert page.session.document == original
    assert not page.session.dirty and not page.undo_button.isEnabled()
    assert not page.redo_button.isEnabled() and not page.save_button.isEnabled()


def test_invalid_dimensions_keep_editable_draft_until_corrected(page):
    source = page.session.path.read_bytes()
    _edit(page, COIL, "mechanical_inner_diameter_mm", 100)
    assert page.session.part(COIL)["mechanical_inner_diameter_mm"] == 100
    assert "valid dimensions" in page.model_note.text()
    assert _dimension(page, COIL, "mechanical_inner_diameter_mm").flags() & Qt.ItemFlag.ItemIsEditable
    assert not page.save()
    assert page.session.dirty and page.session.path.read_bytes() == source
    page.select_part(HOUSING)
    page.select_part(COIL)
    assert float(_dimension(page, COIL, "mechanical_inner_diameter_mm").text()) == 100
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    assert page.save(), page.status.text()
    assert not page.session.dirty


@pytest.mark.parametrize("text", ["unfinished", "NaN", "inf"])
def test_invalid_numeric_text_survives_selection_and_blocks_replacement_until_fixed(page, root, text):
    _edit(page, COIL, "mechanical_outer_diameter_mm", 95)
    _edit(page, COIL, "mechanical_inner_diameter_mm", text)
    session = page.session
    assert not page.save_button.isEnabled()
    assert not page.save()
    assert not page.open_path(root / COLUMN)
    page.select_part(HOUSING)
    page.select_part(COIL)
    assert _dimension(page, COIL, "mechanical_inner_diameter_mm").text() == text
    assert page.session is session
    assert page.session.part(COIL)["mechanical_outer_diameter_mm"] == 95
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    assert page.save_button.isEnabled()
    assert page.save(), page.status.text()


def test_invalid_text_undo_and_revert_preserve_valid_history(page):
    original = deepcopy(page.session.document)
    _edit(page, COIL, "mechanical_outer_diameter_mm", 95)
    _edit(page, COIL, "mechanical_inner_diameter_mm", "unfinished")
    page.undo()
    assert not page._invalid_inputs
    assert page.session.part(COIL)["mechanical_outer_diameter_mm"] == 95
    assert page.save_button.isEnabled()
    _edit(page, COIL, "mechanical_inner_diameter_mm", "unfinished")
    page.revert()
    assert page.session.document == original
    assert not page._invalid_inputs and not page.revert_button.isEnabled()


def test_dirty_source_cannot_be_replaced_or_lost_on_close_show(page, root):
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    session = page.session
    page.close()
    assert not page.open_path(root / COLUMN)
    assert page.session is session
    assert page.session.part(COIL)["mechanical_inner_diameter_mm"] == 70
    assert "current draft" in page.status.text()
    page.show()
    assert page.session is session and page.session.dirty


def test_external_file_change_blocks_save_and_preserves_both_external_file_and_draft(page):
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    external = page.session.path.read_bytes() + b"\n# An independent external change\n"
    page.session.path.write_bytes(external)
    assert not page.save()
    assert "changed outside" in page.status.text()
    assert page.session.path.read_bytes() == external
    assert page.session.part(COIL)["mechanical_inner_diameter_mm"] == 70 and page.session.dirty


def test_split_regions_expose_parent_dimensions_and_save_distinct_materials(page, root):
    assert page.open_path(root / COLUMN, selected_key="objective_lens_yoke")
    key = "objective_lens_yoke"
    length = _dimension(page, key, "length_mm")
    assert not length.flags() & Qt.ItemFlag.ItemIsEditable
    assert "physical sections" in length.toolTip()
    for side in ("upper", "lower"):
        for edge in ("start", "end"):
            assert _dimension(page, "objective_lens", f"{side}_yoke_{edge}_local_z_mm").flags() & Qt.ItemFlag.ItemIsEditable
    assert {page.region.itemData(i) for i in range(page.region.count())} == {"body", "upper", "lower"}
    source = page.session.path.read_bytes()
    expected = {"body": "femm_pure_iron", "upper": "aluminum", "lower": "nonmagnetic_stainless_steel"}
    for region, material in expected.items():
        page.region.setCurrentIndex(page.region.findData(region))
        page.material.setCurrentIndex(page.material.findData(material))
        page.assign_material()
        assert page.session.part(key)["material_regions"][region]["material_key"] == material
    assert page.session.path.read_bytes() == source
    assert page.save(), page.status.text()
    saved = next(part for part in module_manifest.read_document(page.session.path)["parts"] if part["key"] == key)
    assert {region: value["material_key"] for region, value in saved["material_regions"].items()} == expected
    assert saved["material_class"] == "soft_magnetic"


def test_edit_existing_aperture_array_selects_and_previews_that_hole(page, root):
    path = root / "aperture-preview.toml"
    path.write_text('''[[parts]]
key = "existing_aperture"
name = "Existing aperture plate"
mechanical_profile = "circular_aperture"
local_start_z_mm = 10.0
local_center_z_mm = 10.1
local_end_z_mm = 10.2
length_mm = 0.2
plate_thickness_mm = 0.2
mechanical_outer_diameter_mm = 8.0
aperture_hole_diameters_um = [100.0, 200.0]
''', encoding="utf-8")
    assert page.open_path(path)
    source = path.read_bytes()
    _edit(page, "existing_aperture", "aperture_hole_diameters_um", 300, 1)
    assert page.aperture.currentData() == 1
    assert np.linalg.norm(_vertices(page, "existing_aperture")[:, :2], axis=1).min() == pytest.approx(.15)
    assert "index=1" in page.model_note.text()
    page.aperture.setCurrentIndex(0)
    assert np.linalg.norm(_vertices(page, "existing_aperture")[:, :2], axis=1).min() == pytest.approx(.05)
    assert page.session.part("existing_aperture")["aperture_hole_diameters_um"] == [100, 300]
    assert path.read_bytes() == source


def test_mainwindow_keeps_original_2d_view_and_binds_model_selection_and_project_save(window, qtbot, monkeypatch):
    layout = window.workspace.physical_layout
    original_plot = layout.plot
    assert layout.tabs.count() == 3
    assert [layout.tabs.tabText(index) for index in range(3)] == ["2D", "3D Parts", "3D"]
    assert layout.tabs.widget(0) is layout.section_page
    assert layout.tabs.widget(1) is layout.model_editor
    assert layout.tabs.widget(2) is layout.assembly_3d
    assert layout.plot.parentWidget() is layout.section_page
    assert window.workspace.tabs.widget(window.workspace.tabs.indexOf(layout)) is layout
    page = layout.model_editor
    assert page._project_root == window.manifest_editor.root.resolve()
    assert page._project_save.__self__ is window

    window.state.objective_lens.percent = 67.89
    strengths = {lens.key: lens.percent for lens in window.state.lenses}
    window.assembly_panel.select_key(COIL)
    assert page._pending_part is not None and page.session is None
    window.workspace.tabs.setCurrentWidget(layout)
    layout.tabs.setCurrentWidget(page)
    window.show()
    qtbot.waitUntil(lambda: page.session is not None)
    window.preview_timer.stop()
    assert page._selected_key == COIL
    assert page.session.path == (window.manifest_editor.root / MODULE).resolve()
    assert _group(page, "Parent operating values (read-only)")
    original_source = page.session.path.read_bytes()
    original_assembly = window.assembly
    preserve_requests = []
    apply = window.catalog.apply

    def record_apply(state, selection, **kwargs):
        preserve_requests.append(kwargs.get("preserve_operating_parameters"))
        return apply(state, selection, **kwargs)

    monkeypatch.setattr(window.catalog, "apply", record_apply)
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    assert page.session.path.read_bytes() == original_source
    emitted = []
    page.saved.connect(emitted.append)
    assert page.save(), page.status.text()
    window.preview_timer.stop()
    assert preserve_requests == [True]
    assert emitted == [str(page.session.path)]
    assert window.assembly is not original_assembly
    assert window.assembly.part(COIL).data["mechanical_inner_diameter_mm"] == 70
    assert {lens.key: lens.percent for lens in window.state.lenses} == strengths
    assert not page.session.dirty
    assert window.workspace.physical_layout is layout and layout.plot is original_plot
    assert layout.tabs.widget(0) is layout.section_page
    page.select_part(HOUSING)
    assert window.parameter_panel._manifest_target.part_key == HOUSING
    assert page._selected_key == HOUSING and page.session.part(COIL)["mechanical_inner_diameter_mm"] == 70


def test_actual_coil_double_click_opens_3d_without_visiting_ray_diagram(
    navigation_window, qtbot, monkeypatch,
):
    window = navigation_window
    workspace, layout = window.workspace, window.workspace.physical_layout
    point = _coil_scene_point(layout)
    assert _selectable_key_at_scene_position(layout.plot.scene(), point, layout._selectable_item_keys) == COIL
    monkeypatch.setattr(layout, "component_key_at", lambda *args: pytest.fail("An exact coil hit must not use the nearest-part fallback"))
    monkeypatch.setattr(workspace, "show_ray_diagram", lambda: pytest.fail("Physical Layout activation must not open Ray Diagram"))
    activated, visited = [], []
    layout.component_activated.connect(lambda *values: activated.append(values))
    workspace.tabs.currentChanged.connect(lambda index: visited.append(workspace.tabs.widget(index)))
    original = (window.manifest_editor.root / MODULE).read_bytes()

    _double_click_scene(qtbot, layout, point)

    assert len(activated) == 1 and activated[0][0] == COIL
    page = layout.model_editor
    assert workspace.tabs.currentWidget() is layout
    assert layout.tabs.currentWidget() is page
    assert workspace.ray_page not in visited
    assert page._selected_key == COIL and page.view.selection[0] == COIL
    assert window._selected_component_key == COIL
    assert window.parameter_panel._manifest_target.part_key == COIL
    assert workspace._focused_part.key == COIL
    assert workspace._selected_z_mm == pytest.approx(activated[0][1])
    assert workspace.axial_cursor_item.value() == pytest.approx(activated[0][1])
    assert page.session.path == (window.manifest_editor.root / MODULE).resolve()
    assert page.session.path.read_bytes() == original and not page.session.dirty


def test_actual_blank_axis_double_click_uses_nearby_geometry_and_opens_3d(
    navigation_window, qtbot, monkeypatch,
):
    window = navigation_window
    workspace, layout = window.workspace, window.workspace.physical_layout
    z_mm = layout._record_by_key[COIL].center_z_mm + 13.0
    scene = layout.plot.getViewBox().mapViewToScene(QPointF(z_mm, 0.0))
    # The beam passage has no material item: this exercises real empty-space picking.
    assert _selectable_key_at_scene_position(layout.plot.scene(), scene, layout._selectable_item_keys) is None
    calls, activated = [], []
    nearest = layout.component_key_at

    def record_fallback(z, radius=0.0):
        calls.append((z, radius))
        return nearest(z, radius)

    monkeypatch.setattr(layout, "component_key_at", record_fallback)
    layout.component_activated.connect(lambda *values: activated.append(values))
    _double_click_scene(qtbot, layout, scene)

    assert len(calls) == len(activated) == 1
    assert calls[0] == pytest.approx((z_mm, 0.0), abs=0.5)
    selected, selected_z = activated[0]
    assert selected.startswith("intermediate_lens")
    assert layout.model_editor._selected_key == selected
    assert window.parameter_panel._manifest_target.part_key == selected
    assert workspace.tabs.currentWidget() is layout
    assert layout.tabs.currentWidget() is layout.model_editor
    assert workspace._selected_z_mm == pytest.approx(selected_z)


@pytest.mark.parametrize("key", ["sample", "energy_filter"])
def test_special_component_first_click_stays_in_physical_layout_and_syncs_pending_model(
    navigation_window, qtbot, key,
):
    window = navigation_window
    workspace, layout = window.workspace, window.workspace.physical_layout
    visited = []
    workspace.tabs.currentChanged.connect(lambda index: visited.append(workspace.tabs.widget(index)))
    layout.component_selected.emit(key)
    QApplication.processEvents()

    part = window.assembly.part(key)
    page = layout.model_editor
    assert workspace.tabs.currentWidget() is layout
    assert layout.tabs.currentWidget() is layout.section_page
    assert all(destination is layout for destination in visited)
    assert window.parameter_panel._manifest_target.part_key == key
    assert workspace._focused_part.key == key
    assert page._pending_part == (window.manifest_editor.root / part.source_file, key)
    # A subsequent manual model-tab switch consumes that exact pending part.
    layout.tabs.setCurrentWidget(page)
    QApplication.processEvents()
    assert page._selected_key == key and page._pending_part is None
    assert page.session.path == (window.manifest_editor.root / part.source_file).resolve()


def test_2d_3d_and_ray_selection_markers_survive_manual_tab_switches(navigation_window, qtbot):
    window = navigation_window
    workspace, layout = window.workspace, window.workspace.physical_layout
    _double_click_scene(qtbot, layout, _coil_scene_point(layout))
    part = window.assembly.part(COIL)
    expected_span = (part.start_z_mm, part.end_z_mm)
    selected_z = workspace._selected_z_mm

    for destination in ("2d", "ray", "3d", "ray", "2d", "3d"):
        if destination == "ray":
            workspace.tabs.setCurrentWidget(workspace.ray_page)
        else:
            workspace.tabs.setCurrentWidget(layout)
            layout.tabs.setCurrentWidget(layout.section_page if destination == "2d" else layout.model_editor)
        QApplication.processEvents()
        assert layout._highlight.getRegion() == pytest.approx(expected_span)
        assert workspace._ray_component_highlight.component_key == COIL
        assert workspace._ray_component_highlight.getRegion() == pytest.approx(expected_span)
        assert layout.model_editor._selected_key == COIL
        assert layout.model_editor.view.selection[0] == COIL
        assert workspace._selected_z_mm == pytest.approx(selected_z)
        assert workspace.axial_cursor_item.value() == pytest.approx(selected_z)


def test_repeated_project_reveal_refits_the_same_mesh_in_module_scope_without_rotating(page, qtbot):
    page.resize(1280, 760)
    page.show()
    page.scope.setCurrentIndex(page.scope.findData("module"))
    page.view.set_isometric_view()
    QApplication.processEvents()
    rotation = page.view.camera_rotation
    part = SimpleNamespace(key=COIL, source_file=MODULE)
    vertices = _vertices(page, COIL)
    count = len(page.view._meshes)
    assert count > 1

    for _ in range(2):
        _pan_model(qtbot, page.view)
        viewport_center = np.array([page.view.width() / 2, page.view.height() / 2])
        assert not np.allclose(page.view.project_points(vertices)[:, :2].mean(axis=0), viewport_center)
        page.reveal_project_part(part)
        QApplication.processEvents()
        screen = page.view.project_points(vertices)[:, :2]
        assert screen.mean(axis=0) == pytest.approx(viewport_center)
        assert screen.min(axis=0).min() >= 0
        assert screen[:, 0].max() < page.view.width()
        assert screen[:, 1].max() < page.view.height()
        assert np.array_equal(page.view.camera_rotation, rotation)
        assert len(page.view._meshes) == count and page.scope.currentData() == "module"
        assert page._selected_key == COIL and page._pending_part is None


def test_cross_file_activation_keeps_dirty_3d_document_and_invalid_text(navigation_window, qtbot):
    window = navigation_window
    layout = window.workspace.physical_layout
    _double_click_scene(qtbot, layout, _coil_scene_point(layout))
    page = layout.model_editor
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    _edit(page, COIL, "mechanical_outer_diameter_mm", "unfinished")
    session, document = page.session, deepcopy(page.session.document)
    source = page.session.path.read_bytes()
    other = window.assembly.part("objective_lens_excitation_coil")
    assert other.source_file != MODULE
    layout.tabs.setCurrentWidget(layout.section_page)
    layout.component_activated.emit(other.key, other.center_z_mm)
    QApplication.processEvents()

    assert layout.tabs.currentWidget() is page
    assert window.workspace.tabs.currentWidget() is layout
    assert page.session is session and page.session.document == document
    assert page.session.path.read_bytes() == source and page.session.dirty
    assert page._selected_key == COIL
    assert _dimension(page, COIL, "mechanical_outer_diameter_mm").text() == "unfinished"
    assert page.view.selection[0] == COIL
    assert "current draft" in page.status.text()
    assert page._pending_part is not None


def test_parent_reveal_fits_and_highlights_real_descendants_in_module_scope(page, qtbot):
    parent_key = "intermediate_lens"
    child_keys = {part["key"] for part in page.session.document["parts"]
                  if part.get("parent_key") == parent_key}
    page.resize(1280, 760)
    page.show()
    page.scope.setCurrentIndex(page.scope.findData("module"))
    page.view.set_isometric_view()
    QApplication.processEvents()
    _pan_model(qtbot, page.view)
    rotation = page.view.camera_rotation
    page.reveal_project_part(SimpleNamespace(key=parent_key, source_file=MODULE))
    QApplication.processEvents()

    assert page._selected_key == parent_key
    assert page.view.selection[0] == parent_key
    assert child_keys <= page.view._selection_keys
    # The optical parent envelope is intentionally absent; actual material
    # children provide its visible selection and fit instead of an empty target.
    assert parent_key not in {mesh.key for mesh in page.view._meshes}
    vertices = np.concatenate([mesh.vertices for mesh in page.view._meshes if mesh.key in child_keys])
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    assert page.view.project_points([center])[0, :2] == pytest.approx(
        [page.view.width() / 2, page.view.height() / 2])
    screen = page.view.project_points(vertices)[:, :2]
    assert screen.min() >= 0
    assert screen[:, 0].max() < page.view.width()
    assert screen[:, 1].max() < page.view.height()
    assert np.array_equal(page.view.camera_rotation, rotation)
    assert page.scope.currentData() == "module"


@pytest.mark.parametrize("resolve", ["save", "revert"])
def test_resolving_draft_resumes_pending_cross_file_reveal_and_emits_selection(page, root, qtbot, resolve):
    page.resize(1280, 760)
    page.show()
    QApplication.processEvents()
    original = (root / MODULE).read_bytes()
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    session = page.session
    requested = SimpleNamespace(key="objective_lens_excitation_coil", source_file=COLUMN)
    selected = []
    page.component_selected.connect(selected.append)
    page.reveal_project_part(requested)
    assert page.session is session and page.session.dirty
    assert page._pending_part is not None
    assert page._selected_key == COIL

    if resolve == "save":
        assert page.save(), page.status.text()
    else:
        qtbot.mouseClick(page.revert_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert page.session is not session and page.session.path == (root / COLUMN).resolve()
    assert page._selected_key == requested.key and page.view.selection[0] == requested.key
    assert page._pending_part is None and not page.session.dirty
    assert selected[-1] == requested.key  # Deferred completion refreshes linked views.
    if resolve == "revert":
        assert (root / MODULE).read_bytes() == original
    else:
        saved_coil = next(part for part in module_manifest.read_document(root / MODULE)["parts"]
                          if part["key"] == COIL)
        assert saved_coil["mechanical_inner_diameter_mm"] == 70


def test_reselecting_current_part_cancels_obsolete_pending_navigation(page, root, qtbot):
    page.resize(1280, 760)
    page.show()
    QApplication.processEvents()
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    session = page.session
    page.reveal_project_part(SimpleNamespace(key="objective_lens_excitation_coil", source_file=COLUMN))
    assert page._pending_part is not None

    # A normal 2D/tree re-selection can choose the still-open file again.
    page.focus_project_part(SimpleNamespace(key=COIL, source_file=MODULE))
    assert page._pending_part is None
    qtbot.mouseClick(page.revert_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert page.session is session and page.session.path == (root / MODULE).resolve()
    assert page._selected_key == COIL and not page.session.dirty
    assert page._pending_part is None


def test_source_selector_tracks_the_open_component_file_and_shared_module(page, root, qtbot):
    page.show()
    QApplication.processEvents()
    assert page.modules.currentData() == MODULE
    assert MODULE.rsplit("/", 1)[-1] in page.source_label.text()
    assert COIL in page.source_label.text()
    original_session = page.session
    page.select_part(HOUSING)
    assert page.session is original_session and page.modules.currentData() == MODULE
    assert HOUSING in page.source_label.text()
    target = SimpleNamespace(key="objective_lens_excitation_coil", source_file=COLUMN)
    page.focus_project_part(target)
    assert page.session.path == (root / COLUMN).resolve()
    assert page.modules.currentData() == COLUMN
    assert target.key in page.source_label.text()
    index = page.modules.findData(MODULE)
    page.modules.setCurrentIndex(index)
    page.modules.activated.emit(index)
    assert page.session.path == (root / MODULE).resolve()
    assert page.modules.currentData() == MODULE


def test_blocked_module_selector_restores_actual_source_and_preserves_draft(page, root, qtbot):
    page.show()
    _edit(page, COIL, "mechanical_inner_diameter_mm", 70)
    session, document = page.session, deepcopy(page.session.document)
    index = page.modules.findData(COLUMN)
    page.modules.setCurrentIndex(index)
    page.modules.activated.emit(index)
    assert page.session is session and page.session.document == document
    assert page.modules.currentData() == MODULE
    assert "EnergyFilter.toml" in page.source_label.text()
    assert "draft" in page.status.text()


def test_box_and_asymmetric_transform_controls_change_mesh_and_undo(page, qtbot):
    original = page.session.path.read_bytes()
    page.base_shape.setCurrentIndex(page.base_shape.findData("box"))
    qtbot.mouseClick(page.apply_shape_button, Qt.MouseButton.LeftButton)
    assert page.session.part(COIL)["model_3d"]["base"]["kind"] == "box"
    before = _vertices(page, COIL).copy()
    _edit(page, COIL, "model_3d", 1.4, "transform", "scale_xy", 0)
    after = _vertices(page, COIL)
    assert np.ptp(after[:, 0]) == pytest.approx(np.ptp(before[:, 0]) * 1.4)
    assert np.ptp(after[:, 1]) == pytest.approx(np.ptp(before[:, 1]))
    assert page.session.path.read_bytes() == original
    page.undo()
    assert _vertices(page, COIL) == pytest.approx(before)


def test_feature_dialog_adds_real_cut_and_remove_is_undoable(page, qtbot, monkeypatch):
    from temsim.gui import part_feature_dialog
    page.base_shape.setCurrentIndex(page.base_shape.findData("box"))
    page._apply_base_shape()
    uncut = _vertices(page, COIL).copy()
    feature = dict(id="hole_1", kind="hole", axis="z", center_mm=[10, 3, 0], diameter_mm=6, depth_mm=220)
    class AcceptedFeatureDialog:
        from PySide6.QtWidgets import QDialog
        DialogCode = QDialog.DialogCode
        def __init__(self, *_args):
            pass
        def exec(self):
            return self.DialogCode.Accepted
        def feature(self):
            return deepcopy(feature)
    monkeypatch.setattr(part_feature_dialog, "PartFeatureDialog", AcceptedFeatureDialog)
    page._edit_feature("hole")
    assert page.session.part(COIL)["model_3d"]["features"] == [feature]
    cut = _vertices(page, COIL).copy()
    assert len(cut) > len(uncut)
    assert any("feature:hole_1" in str(group) for record in page._mesh_records for group in record["face_groups"])
    assert page.feature_tree.currentItem().text(0) == "hole_1"
    page._remove_feature()
    assert page.session.part(COIL)["model_3d"]["features"] == []
    assert _vertices(page, COIL) == pytest.approx(uncut)
    page.undo()
    assert _vertices(page, COIL) == pytest.approx(cut)


@pytest.mark.parametrize("mode", ["face", "edge"])
def test_topology_pick_highlights_multiple_real_dimension_rows(page, qtbot, mode):
    page.resize(1280, 760)
    page.show()
    page.view.set_axial_view()
    page.selection_mode.setCurrentIndex(page.selection_mode.findData(mode))
    QApplication.processEvents()
    vertices = _vertices(page, COIL)
    radius = 0.5 * (np.linalg.norm(vertices[:, :2], axis=1).min() + np.linalg.norm(vertices[:, :2], axis=1).max())
    if mode == "edge":
        radius = np.linalg.norm(vertices[:, :2], axis=1).max()
    point = page.view.project_points([[radius, 0, vertices[:, 2].max()]])[0]
    position = QPoint(int(round(point[0])), int(round(point[1])))
    hit = page.view.pick_topology_at(position, mode=mode)
    assert hit is not None and hit["key"] == COIL
    qtbot.mouseClick(page.view, Qt.MouseButton.LeftButton, pos=position)
    assert page._topology_selection
    paths = {tuple(path) for item in page._topology_selection for path in item["parameter_paths"]}
    highlighted = {tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole))
                   for row in range(page.dimensions.rowCount())
                   if page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole + 1)}
    assert {("parts", COIL, "length_mm"), ("parts", COIL, "mechanical_outer_diameter_mm")} <= highlighted
    assert highlighted <= paths
    assert page.parameter_tabs.currentWidget() is page.dimensions
    assert page.view.topology_selection
    page.clear_topology_button.click()
    assert not page._topology_selection
    assert not any(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole + 1)
                   for row in range(page.dimensions.rowCount()))


@pytest.mark.parametrize("save_path", ["model", "dimensions"])
@pytest.mark.parametrize("fail_after_reload", [False, True])
def test_geometry_save_preserves_each_panels_target_and_uncommitted_draft(
        window, root, monkeypatch, save_path, fail_after_reload):
    """Exercise both save entry points; persistence/reload is an in-memory stub."""
    from PySide6.QtWidgets import QLineEdit
    from temsim.manifest_editor import ManifestTarget

    assert window.assembly_panel.select_key(HOUSING)
    assert window._select_energy_filter_component(
        "energy_filter_slit", activate_page=False, focus_editor=False)
    main_panel = window.parameter_panel
    energy_panel = window.workspace.energy_filter_parameters
    snapshots = []
    for panel, text in ((main_panel, "invalid main draft"), (energy_panel, "invalid EELS draft")):
        panel.tabs.setCurrentIndex(1)
        row = next(row for row, field in enumerate(panel._manifest_fields)
                   if field.editable and isinstance(field.value, (int, float)))
        item = panel.manifest_table.item(row, 1)
        original_text = item.text()
        panel.manifest_table.editItem(item)
        editor = next(editor for _index, editor in panel._manifest_delegate._editors.values())
        assert isinstance(editor, QLineEdit)
        editor.setText(text)
        # Typed text has not reached the table model, let alone disk or state.
        assert item.text() == original_text
        path = tuple(item.data(Qt.ItemDataRole.UserRole))
        snapshots.append((panel, panel._manifest_target, path, text))

    calls = []
    def save_stub(target, updates, **_kwargs):
        calls.append((target, updates))
        # Reproduce the context replacements performed after a catalog reload.
        energy_panel._load_manifest()
        window.assembly_panel.select_key(target.part_key)
        if fail_after_reload:
            raise ValueError("test reload failure")

    monkeypatch.setattr(window, "_save_manifest_updates", save_stub)
    target = ManifestTarget(MODULE, COIL)
    if save_path == "model":
        window.workspace.physical_layout.model_editor._selected_key = COIL
        if fail_after_reload:
            with pytest.raises(ValueError, match="test reload failure"):
                window._save_model_document(root / MODULE, {("parts", COIL, "length_mm"): 2.})
        else:
            window._save_model_document(root / MODULE, {("parts", COIL, "length_mm"): 2.})
    else:
        dialog = window._edit_part_geometry(target)
        assert dialog is not None
        control = dialog.dimension_controls["outer_diameter_mm"]
        control.setValue(control.value() + 1.)
        dialog.apply()
        assert ("test reload failure" in dialog.error_label.text()) == fail_after_reload
        dialog.reject()
    assert len(calls) == 1
    assert window._selected_component_key == HOUSING
    for panel, own_target, path, text in snapshots:
        assert panel._manifest_target == own_target
        assert panel.manifest_draft_texts(own_target)[path] == text
        assert panel.tabs.currentIndex() == 1
        assert not panel._manifest_delegate.pending_texts()
