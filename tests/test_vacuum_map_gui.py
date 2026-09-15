from copy import deepcopy
import pytest
from temsim.gui.vacuum_map_page import VacuumMapPage
from temsim.vacuum import resolve_regions


@pytest.fixture
def page(qtbot):
    from temsim.optics.column import default_state
    page = VacuumMapPage()
    qtbot.addWidget(page)
    state = default_state()
    state.vacuum_map.enabled = True  # Editing an explicitly enabled map.
    page.set_state(state)
    return page


def test_selected_region_apply_and_invalid_edit(page):
    page.select_region("projection")
    assert float(page.pressure.text()) == 8e-7
    page.pressure.setText("2e-5")
    assert page.apply()
    assert page.state.vacuum_map.regions[-1].medium.pressure_mbar == 2e-5
    page.pressure.setText("-3")
    assert not page.apply()
    assert page.state.vacuum_map.regions[-1].medium.pressure_mbar == 2e-5


def test_cell_medium_and_subdivision(page):
    page.select_region("specimen_cell")
    page.cell_inserted.setChecked(True)
    page.formula.setText("He")
    page.pressure.setText("5")
    assert page.apply()
    assert page.state.vacuum_map.cell.medium.formula == "He"
    assert len(resolve_regions(page.state)) == 9  # Ambient + interior + two windows.
    page.select_region("column")
    page.split_region()
    assert len(page.state.vacuum_map.regions) == 7
    page.remove_region()
    assert len(page.state.vacuum_map.regions) == 6


def test_boundaries_leave_independent_gap_and_follow_selected_component(page):
    previous = deepcopy(page.state.vacuum_map.regions[3])
    page.select_region("column")
    page.end_anchor.setCurrentIndex(page.end_anchor.findData("condenser_lens_2.center"))
    page.end_offset.setValue(0)
    assert page.apply()
    assert page.state.vacuum_map.regions[3] == previous
    assert any(r.end_medium is not None for r in resolve_regions(page.state))


def test_initial_parameters_hidden_and_regions_use_actual_z(page, qtbot):
    from PySide6.QtCore import QPointF, Qt
    assert page.current_key == "" and page.editor.isHidden()
    page.resize(1250, 800)
    page.show()
    page.diagram.setXRange(0, 3300, padding=0)
    qtbot.wait(10)
    rectangles = dict((key, box) for box, key in page.diagram.boxes)
    for row in resolve_regions(page.state, include_cell=False):
        assert rectangles[row.key].left() == row.start_z_mm
        assert rectangles[row.key].width() == pytest.approx(row.end_z_mm-row.start_z_mm)
    assert rectangles["specimen"].width() < rectangles["column"].width()/100
    scene = page.diagram.getViewBox().mapViewToScene(QPointF(page.state.sample.z_mm, 1.35))
    qtbot.mouseClick(page.diagram.viewport(), Qt.MouseButton.LeftButton,
                     pos=page.diagram.mapFromScene(scene))
    assert page.current_key == "specimen" and not page.editor.isHidden()


def test_module_local_z_roundtrip_and_disabled_editing(page, tmp_path):
    from temsim.vacuum import VacuumMap, boundary_anchors, module_axial_ranges
    page.state.vacuum_map.enabled = False
    page.set_state(page.state)
    page.select_region("column")
    original = next(r for r in resolve_regions(page.state, include_disabled=True) if r.key == "column")
    page.position_mode.setCurrentIndex(1)
    modules = module_axial_ranges(page.state)
    column = next(m for m in modules if "column/" in m.source_file)
    anchor = f"module:{column.key}.origin"
    page.coordinate_frame.setCurrentIndex(page.coordinate_frame.findData(anchor))
    assert page.start_z.value()+column.origin_z_mm == pytest.approx(original.start_z_mm)
    assert page.end_z.value()+column.origin_z_mm == pytest.approx(original.end_z_mm)
    page.end_z.setValue(page.end_z.value()-2)
    page.pressure.setText("3e-5")
    assert page.apply(), page.status.text()
    region = next(r for r in page.state.vacuum_map.regions if r.key == "column")
    assert region.start_anchor == region.end_anchor == anchor
    assert region.medium.pressure_mbar == 3e-5
    path = tmp_path/"module-vacuum.toml"
    page.state.vacuum_map.save(path)
    loaded = VacuumMap.load(path)
    assert loaded == page.state.vacuum_map
    rows = resolve_regions(page.state, include_disabled=True)
    assert next(r for r in rows if r.key == "column").end_z_mm == pytest.approx(original.end_z_mm-2)
    transition = next(r for r in rows if r.end_medium is not None)
    page.select_region(transition.key)
    assert page.editor.isHidden() and "automatic linear" in page.selection_hint.text()
    page.select_region("column")
    page.split_region()
    assert len(page.state.vacuum_map.regions) == 7
    assert not page.state.vacuum_map.enabled
    assert boundary_anchors(page.state)[anchor] == column.origin_z_mm


def test_global_z_anchor_and_mode_switch_preserve_unsaved_end(page):
    page.select_region("column")
    page.position_mode.setCurrentIndex(1)
    page.end_z.setValue(1500)
    page.position_mode.setCurrentIndex(0)
    page.position_mode.setCurrentIndex(1)
    assert page.end_z.value() == 1500
    assert page.apply(), page.status.text()
    region = next(r for r in page.state.vacuum_map.regions if r.key == "column")
    assert region.start_anchor == region.end_anchor == "axis_origin"
    assert region.end_offset_mm == 1500


def test_z_range_shared_both_ways_without_selection_or_resize_fit(qtbot):
    from temsim.gui.visualization import VisualizationWorkspace
    from temsim.optics.column import default_state
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    state = default_state()
    page, physical = workspace.vacuum_map, workspace.physical_layout
    page.set_state(state)
    workspace.resize(1500, 900)
    workspace.show()
    workspace.tabs.setCurrentWidget(physical)
    physical.plot.setXRange(450, 850, padding=0)
    qtbot.wait(30)
    physical_width = physical.plot.getViewBox().width()
    assert page.diagram.getViewBox().viewRange()[0] == pytest.approx([450, 850])
    workspace.tabs.setCurrentWidget(page)
    qtbot.wait(30)
    assert page.diagram.getViewBox().width() == pytest.approx(physical_width, abs=1)
    page.focus_component(state._resolved_assembly.part("condenser_lens_2"))
    assert page.diagram.getViewBox().viewRange()[0] == pytest.approx([450, 850])
    page.pressure.setText("4e-7")
    assert page.apply()
    assert page.diagram.getViewBox().viewRange()[0] == pytest.approx([450, 850])
    page.diagram.setXRange(1000, 1400, padding=0)
    assert physical.plot.getViewBox().viewRange()[0] == pytest.approx([1000, 1400])
    workspace.resize(1250, 800)
    workspace.tabs.setCurrentWidget(physical)
    qtbot.wait(30)
    physical_width = physical.plot.getViewBox().width()
    workspace.tabs.setCurrentWidget(page)
    qtbot.wait(30)
    assert page.diagram.getViewBox().width() == pytest.approx(physical_width, abs=1)
    assert page.diagram.getViewBox().viewRange()[0] == pytest.approx([1000, 1400])


def test_module_anchors_follow_installed_translation(page):
    from dataclasses import replace
    from temsim.vacuum import boundary_anchors, module_axial_ranges
    assembly = page.state._resolved_assembly
    module = module_axial_ranges(page.state)[1]
    anchors = boundary_anchors(page.state)
    shifted = tuple(replace(p, start_z_mm=p.start_z_mm+7, center_z_mm=p.center_z_mm+7,
                            end_z_mm=p.end_z_mm+7) if p.module_key == module.key else p
                    for p in assembly.parts)
    page.state._resolved_assembly = replace(assembly, parts=shifted)
    updated = boundary_anchors(page.state)
    for suffix in ("origin", "start", "end"):
        key = f"module:{module.key}.{suffix}"
        assert updated[key] == pytest.approx(anchors[key]+7)
    assert updated["axis_origin"] == 0


def test_numeric_projection_range_retains_dpa_anchor(page):
    from temsim.vacuum import boundary_anchors
    page.select_region("projection")
    page.position_mode.setCurrentIndex(1)
    page.pressure.setText("7e-7")
    assert not page.start_z.isEnabled()
    assert page.apply(), page.status.text()
    region = next(r for r in page.state.vacuum_map.regions if r.key == "projection")
    assert region.start_anchor == "projection_dpa" and region.start_offset_mm == 0
    assert page.position_mode.currentData() == "positions"
    assert page.start_z.value() == pytest.approx(boundary_anchors(page.state)["projection_dpa"])
