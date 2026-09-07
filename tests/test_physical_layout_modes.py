"""Three presentation modes; no instrument solve, profiles or source writes."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from temsim.gui import assembly_model_page as gui
from temsim.gui import main_window as shell


def _assembly(shift=0.0):
    rows, parts = [], []
    for index in range(2):
        start = float(index * 20)
        row = dict(key=f"coil{index}", name=f"Coil {index}",
                   mechanical_profile="magnetic_excitation_coil",
                   local_start_z_mm=start, local_center_z_mm=start + 5,
                   local_end_z_mm=start + 10, length_mm=10.0,
                   mechanical_inner_diameter_mm=8.0, mechanical_outer_diameter_mm=12.0,
                   material_class="insulated_copper_winding")
        rows.append(SimpleNamespace(data=row))
        parts.append(SimpleNamespace(
            key=row["key"], name=row["name"], module_key="test", source_file="test.toml",
            data=row, parent_key=None, branch="main", length_mm=10.0,
            start_z_mm=start + shift, center_z_mm=start + 5 + shift,
            end_z_mm=start + 10 + shift,
        ))
    return SimpleNamespace(parts=tuple(parts), modules=(SimpleNamespace(key="test", parts=rows),),
                           vacuum_liner_segments=())


@pytest.fixture
def page(qtbot):
    page = gui.AssemblyModelPage()
    qtbot.addWidget(page)
    page.resize(1100, 650)
    page.set_assembly(_assembly())
    return page


def _show(page, qtbot):
    page.show()
    qtbot.waitUntil(lambda: page.mesh_builds > 0)


def _camera(view):
    return (view.camera_rotation, view._center.copy(), view._pan.copy(), view.zoom_factor)


def _assert_camera(view, expected):
    for actual, wanted in zip(_camera(view), expected):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=0)


def test_lazy_meshes_and_tab_return_keep_camera_and_items(page, qtbot):
    assert page.mesh_builds == 0 and not page.view._meshes
    _show(page, qtbot)
    assert len(page.view._meshes) == 2
    assert not page.view._selection_keys.intersection(("coil0", "coil1"))
    page.view.set_column_isometric_view()
    page.view._zoom = 2.3
    page.view._pan[:] = (21, -14)
    camera, meshes = _camera(page.view), page.view._meshes
    page.hide()
    page.set_assembly(_assembly(), {"objective_lens": {"percent": 62.0}})
    assert page.mesh_builds == 1
    page.show()
    qtbot.waitUntil(lambda: not page._pending)
    assert page.mesh_builds == 1 and page.view._meshes is meshes
    _assert_camera(page.view, camera)


def test_changed_geometry_refreshes_once_and_preserves_camera(page, qtbot):
    _show(page, qtbot)
    camera = _camera(page.view)
    page.hide()
    page.set_assembly(_assembly(100))
    page.set_assembly(_assembly(200))
    page.show()
    qtbot.waitUntil(lambda: page.mesh_builds == 2)
    assert min(mesh.vertices[:, 2].min() for mesh in page.view._meshes) == 200
    _assert_camera(page.view, camera)
    assert page._model.errors == ()


def test_selection_visibility_and_section_only_redraw_cached_geometry(page, qtbot, monkeypatch):
    _show(page, qtbot)
    monkeypatch.setattr(gui, "assembly_model_from_assembly", lambda *_a, **_k: pytest.fail("Unexpected mesh rebuild"))
    camera = _camera(page.view)
    selected, edits = [], []
    page.component_selected.connect(selected.append)
    page.edit_part_requested.connect(lambda *values: edits.append(values))
    page.focus_component("coil0")
    assert not selected  # Programmatic selection cannot recurse into the main tree.
    _assert_camera(page.view, camera)
    page.tree.setCurrentItem(page._tree_items["coil1"])
    assert selected == ["coil1"]
    page.edit_part.click()
    assert edits == [("coil1", 25.0)]
    page._tree_items["coil0"].setCheckState(0, Qt.CheckState.Unchecked)
    assert {mesh.key for mesh in page.view._meshes} == {"coil1"}
    page.section.setChecked(True)
    assert all((triangles[..., 1] <= 1e-12).all() for triangles in page.view._triangles)
    _assert_camera(page.view, camera)
    assert page.fit_current_selection()
    assert page.mesh_builds == 1


def test_bad_new_geometry_does_not_display_old_assembly(page, qtbot, monkeypatch):
    _show(page, qtbot)
    def invalid(*_args, **_kwargs):
        raise ValueError("Invalid test geometry")
    monkeypatch.setattr(gui, "assembly_model_fingerprint", invalid)
    page.set_assembly(_assembly(12))
    qtbot.waitUntil(lambda: "unavailable" in page.status.text())
    assert not page.view._meshes and page._model is None
    assert not page.edit_part.isEnabled()


def test_liner_context_can_be_selected_and_hidden_without_opening_a_part(page, qtbot):
    assembly = _assembly()
    assembly.vacuum_liner_segments = (SimpleNamespace(
        key="@liner:test", name="Vacuum liner", start_z_mm=0.0, end_z_mm=30.0,
        inner_diameter_mm=6.0, outer_diameter_mm=7.0,
    ),)
    page.set_assembly(assembly)
    _show(page, qtbot)
    selected = []
    page.component_selected.connect(selected.append)
    page._surface_selected("@liner:test", "body")
    assert "Vacuum liner" in page.selection_label.text()
    assert not page.edit_part.isEnabled() and not selected
    page._tree_items["@liner:test"].setCheckState(0, Qt.CheckState.Unchecked)
    assert "@liner:test" not in {mesh.key for mesh in page.view._meshes}


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "layout.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    widget = shell.MainWindow()
    widget.preview_timer.stop()
    monkeypatch.setattr(widget.calculations.pool, "start", lambda *_: pytest.fail("Viewing must not calculate"))
    qtbot.addWidget(widget)
    widget.resize(1700, 950)
    return widget


def test_three_modes_use_saved_assembly_without_calculation(window, qtbot):
    workspace, layout = window.workspace, window.workspace.physical_layout
    state = deepcopy(window.state.to_dict())
    high = object()
    workspace._high_accuracy_result = high
    assert [layout.tabs.tabText(i) for i in range(layout.tabs.count())] == ["2D", "3D Parts", "3D"]
    assert layout.assembly_3d.mesh_builds == 0
    workspace.tabs.setCurrentWidget(layout)
    layout.tabs.setCurrentWidget(layout.assembly_3d)
    window.show()
    qtbot.waitUntil(lambda: layout.assembly_3d.mesh_builds == 1, timeout=20000)
    assert layout.assembly_3d._model.meshes
    assert not layout.assembly_3d._model.errors
    meshes = layout.assembly_3d.view._meshes
    layout.tabs.setCurrentWidget(layout.section_page)
    layout.tabs.setCurrentWidget(layout.assembly_3d)
    qtbot.wait(30)
    assert layout.assembly_3d.view._meshes is meshes
    assert workspace._high_accuracy_result is high
    assert window.state.to_dict() == state
    assert not window.preview_timer.isActive()
    assert "assemblyModelSplitter" in window.workspace_layouts.splitters


@pytest.mark.parametrize("old,new", [("2D section", "2D"), ("3D model editor", "3D Parts"), ("3D", "3D")])
def test_layout_restore_migrates_mode_labels(window, old, new):
    manager = window.workspace_layouts
    data = manager._snapshot()
    data["tabs"]["physicalLayoutTabs"] = old
    manager._apply(data)
    tabs = window.workspace.physical_layout.tabs
    assert tabs.tabText(tabs.currentIndex()) == new
    assert not window.preview_timer.isActive()


def test_restored_3d_mode_builds_when_shown_despite_blocked_tab_signals(window, qtbot):
    manager, workspace = window.workspace_layouts, window.workspace
    data = manager._snapshot()
    data["tabs"][workspace.tabs.objectName()] = "Physical Layout"
    data["tabs"]["physicalLayoutTabs"] = "3D"
    manager._apply(data)
    page = workspace.physical_layout.assembly_3d
    assert page.mesh_builds == 0
    window.show()
    qtbot.waitUntil(lambda: page.mesh_builds == 1, timeout=20000)
    assert page.isVisible() and page._model.meshes
    assert not window.preview_timer.isActive()


def test_part_editor_source_and_camera_controls_fit_compact_width(qtbot):
    from temsim.gui.part_model_editor import PartModelEditorPage
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.resize(890, 650)
    page.show()
    qtbot.wait(20)
    assert page.minimumSizeHint().width() < 890
    assert page.modules.geometry().center().y() == page.load_module_button.geometry().center().y()
    assert page.scope.y() > page.modules.y()
    for button in (page.load_module_button, page.fit_button, page.iso_button, page.front_button):
        assert button.isVisible()
        assert button.x() + button.width() <= page.width()
