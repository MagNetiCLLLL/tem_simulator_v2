"""Section controls declare requests; no physical solver or checkpoint IO is mocked as execution."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QSlider

from temsim.component_keys import CONDENSER_LENS_1, OBJECTIVE_LENS, SELECTED_AREA_APERTURE
from temsim.gui import interactive_calculation as gui
from temsim.optics.column import default_state
from temsim.physics.particle_sections import section_limits


@pytest.fixture
def section_page(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "section-ui.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    page = gui.InteractiveCalculationPage()
    qtbot.addWidget(page)
    yield page
    page.shutdown()


def add_range(page, key, field, low, high):
    index = next(index for index in range(page.choice.count())
                 if page.choice.itemData(index).key == key
                 and page.choice.itemData(index).field == field)
    page.choice.setCurrentIndex(index)
    page._add_range()
    row = page.ranges.rowCount() - 1
    page.ranges.cellWidget(row, 1).setText(str(low))
    page.ranges.cellWidget(row, 2).setText(str(high))


def component_item(page, key):
    return next(page.section_components.item(row)
                for row in range(page.section_components.count())
                if page.section_components.item(row).data(Qt.ItemDataRole.UserRole) == key)


def completed_section(page, **changes):
    # A controlled UI payload, not evidence of executed transport.
    request = page.segment_request()
    metrics = {"section_target_z_mm": request["target_z_mm"],
               "section_component_keys": request["component_keys"]}
    metrics.update(changes)
    return SimpleNamespace(simulation=SimpleNamespace(metrics=metrics))


def test_section_is_opt_in_and_named_planes_follow_backend_limits(section_page, qtbot):
    page = section_page
    assert page.segment_request() is None
    assert not page.section_group.isEnabled()
    state = default_state()
    page.set_source(state)
    low, high = section_limits(state)
    assert page.section_z.minimum() == pytest.approx(low, abs=1e-6)
    assert page.section_z.maximum() == pytest.approx(high, abs=1e-6)
    assert all(low <= page.section_plane.itemData(row) <= high
               for row in range(1, page.section_plane.count()))
    assert any("Specimen reference plane" in page.section_plane.itemText(row)
               for row in range(page.section_plane.count()))
    assert page.segment_request() is None
    with qtbot.waitSignal(page.section_changed):
        page.section_group.setChecked(True)
    assert page.segment_request() == {"target_z_mm": page.section_z.value(), "component_keys": ()}
    assert "inserted specimens scatter electrons" in page.section_scope.text()
    flags = page.section_status.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    assert flags & Qt.TextInteractionFlag.TextSelectableByKeyboard


def test_named_plane_and_custom_z_remain_in_sync_without_changing_state(section_page, qtbot):
    page = section_page
    state = default_state()
    page.set_source(state)
    original_sample_z = state.sample.z_mm
    page.section_group.setChecked(True)
    with qtbot.waitSignal(page.section_changed):
        page.section_plane.setCurrentIndex(1)
    assert page.segment_request()["target_z_mm"] == pytest.approx(page.section_plane.currentData(), abs=1e-6)
    with qtbot.waitSignal(page.section_changed):
        page.section_z.setValue(page.section_z.value() + .125)
    assert page.section_plane.currentData() is None
    assert state.sample.z_mm == original_sample_z


def test_components_are_deduplicated_and_post_specimen_apertures_participate(section_page):
    page = section_page
    state = default_state()
    state.selected_area_aperture.enabled = True
    page.set_source(state)
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    add_range(page, SELECTED_AREA_APERTURE, "offset_x_mm", -.01, .01)
    add_range(page, SELECTED_AREA_APERTURE, "offset_y_mm", -.01, .01)
    add_range(page, "df", "z_mm", 2790, 2810)
    assert page.section_components.count() == 2
    old_bank = page.plan()
    page.section_group.setChecked(True)
    component_item(page, OBJECTIVE_LENS).setCheckState(Qt.CheckState.Unchecked)
    request = page.segment_request()
    assert request["component_keys"] == (SELECTED_AREA_APERTURE,)
    selected_plan = page.plan(precompute=False, live_only=True)
    assert len(selected_plan.ranges) == 2
    assert {axis.control.group for axis in selected_plan.ranges} == {"aperture"}
    assert {axis.control.stage for axis in selected_plan.ranges} == {"readout"}
    assert page.plan() == old_bank


def test_unselected_range_does_not_block_section_live_controls(section_page):
    page = section_page
    page.set_source(default_state())
    add_range(page, CONDENSER_LENS_1, "percent", "", "")
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    component_item(page, CONDENSER_LENS_1).setCheckState(Qt.CheckState.Unchecked)
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()
    assert [axis.control.key for axis in page.live_plan.ranges] == [OBJECTIVE_LENS]
    assert page.controller.bank is None
    with pytest.raises(ValueError, match="minimum and maximum"):
        page.plan()


def test_deselecting_live_component_stops_its_edits_and_preserves_other_value(section_page, qtbot):
    page = section_page
    page.set_source(default_state())
    add_range(page, CONDENSER_LENS_1, "percent", 50, 53)
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    page.start_live_tuning()
    page.timer.stop()
    first, second = page.live_plan.ranges
    first_widget = page.live_widgets[first.control.identity]
    second_widget = page.live_widgets[second.control.identity]
    original = second_widget.value()
    component_item(page, CONDENSER_LENS_1).setCheckState(Qt.CheckState.Unchecked)
    assert not first_widget.isEnabled()
    first_row = page.live_table.cellWidget(page.live_table.control_row(first.control.identity), 1)
    assert not first_row.findChild(QSlider).isEnabled()
    assert second_widget.isEnabled() and second_widget.value() == original
    with qtbot.waitSignal(page.tuning_changed) as update:
        page._read()
    assert [axis.control.key for axis, _ in update.args[0]] == [OBJECTIVE_LENS]


def test_restarting_section_controls_preserves_existing_live_values(section_page):
    page = section_page
    page.set_source(default_state())
    add_range(page, CONDENSER_LENS_1, "percent", 50, 53)
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    component_item(page, CONDENSER_LENS_1).setCheckState(Qt.CheckState.Unchecked)
    page.start_live_tuning()
    page.timer.stop()
    identity = page.live_plan.ranges[0].control.identity
    page.live_widgets[identity].setValue(66.25)
    page.timer.stop()
    component_item(page, CONDENSER_LENS_1).setCheckState(Qt.CheckState.Checked)
    page.start_live_tuning()
    page.timer.stop()
    assert len(page.live_plan.ranges) == 2
    assert page.live_widgets[identity].value() == pytest.approx(66.25)


def test_save_requires_matching_completed_section_and_changes_invalidate_it(section_page, qtbot):
    page = section_page
    page.set_source(default_state())
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    assert not page.section_save.isEnabled()
    assert not page.set_section_result(completed_section(page, section_target_z_mm=page.section_z.value()+1))
    assert not page.set_section_result(completed_section(page, section_component_keys=()))
    result = completed_section(page)
    assert page.set_section_result(result)
    assert page.section_save.isEnabled()
    with qtbot.waitSignal(page.section_save_requested):
        page.section_save.click()
    with qtbot.waitSignal(page.section_load_requested):
        page.section_load.click()
    page.section_z.setValue(page.section_z.value()+1)
    assert not page.section_save.isEnabled()
    assert not page.set_section_result(result)
    page.set_section_result(completed_section(page))
    component_item(page, OBJECTIVE_LENS).setCheckState(Qt.CheckState.Unchecked)
    assert not page.section_save.isEnabled()
    page.set_section_result(completed_section(page))
    page.invalidate_section_result("Input settings changed")
    assert not page.section_save.isEnabled()
    assert page.section_status.text() == "Input settings changed"


def test_live_value_change_invalidates_save_and_busy_blocks_section_actions(section_page):
    page = section_page
    page.set_source(default_state())
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    page.start_live_tuning()
    page.timer.stop()
    assert page.set_section_result(completed_section(page))
    next(iter(page.live_widgets.values())).setValue(66.)
    page.timer.stop()
    assert not page.section_save.isEnabled()
    page.set_section_result(completed_section(page))
    page._busy(True)
    assert not page.section_group.isEnabled()
    assert not page.section_save.isEnabled()
    assert not page.section_load.isEnabled()
    page._busy(False)
    assert page.section_group.isEnabled()
    assert page.section_save.isEnabled()


def test_completed_component_order_is_irrelevant_and_actual_reuse_is_reported(section_page):
    page = section_page
    page.set_source(default_state())
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    add_range(page, CONDENSER_LENS_1, "percent", 50, 53)
    page.section_group.setChecked(True)
    result = completed_section(
        page, section_component_keys=tuple(reversed(page.segment_request()["component_keys"])),
        section_resume_z_mm=600., section_reused_prefix=True,
        section_gun_reused=True, section_vacuum_restart="recomputed",
    )
    assert page.set_section_result(result)
    assert page.section_save.isEnabled()
    assert "Column state reused through Z = 600 mm" in page.section_status.text()
    assert "Gun calculation reused" in page.section_status.text()
    assert "Vacuum transport recalculated" in page.section_status.text()
    fresh = completed_section(
        page, section_resume_z_mm=450., section_reused_prefix=False,
        section_gun_reused=False, section_vacuum_restart="not_participating",
    )
    assert page.set_section_result(fresh)
    assert "Column transport calculated from Z = 450 mm" in page.section_status.text()
    assert "Gun transport calculated from tip emission" in page.section_status.text()
    assert "Vacuum scattering disabled" in page.section_status.text()


def test_recapture_clears_participants_and_previous_save_availability(section_page):
    page = section_page
    page.set_source(default_state())
    add_range(page, OBJECTIVE_LENS, "percent", 65, 72)
    page.section_group.setChecked(True)
    page.set_section_result(completed_section(page))
    page.set_source(default_state())
    assert page.section_components.count() == 0
    assert not page.section_save.isEnabled()
    assert page.segment_request()["component_keys"] == ()


def test_deflector_and_stigmator_ranges_participate_in_section_live_tuning(section_page):
    page = section_page
    state = default_state()
    page.set_source(state)
    deflector = state.deflectors[0]
    stigmator = state.stigmators[0]
    add_range(page, deflector.key, "upper_x_mrad", -.1, .1)
    add_range(page, stigmator.key, "strength_x_percent", -10., 10.)
    page.section_group.setChecked(True)
    assert set(page.segment_request()["component_keys"]) == {deflector.key, stigmator.key}
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()
    assert {axis.control.group for axis in page.live_plan.ranges} == {"deflector", "stigmator"}
