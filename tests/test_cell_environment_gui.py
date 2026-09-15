import pytest

from temsim.gui.vacuum_map_page import VacuumMapPage
from temsim.gui.cell_environment_editor import CellGeometryDialog


@pytest.fixture
def page(qtbot):
    from temsim.optics.column import default_state
    page = VacuumMapPage()
    qtbot.addWidget(page)
    page.set_state(default_state())
    page.select_region('specimen_cell')
    return page


def test_cell_units_windows_liquid_and_mixture_edit(page, qtbot):
    original_z = page.state.sample.z_mm
    geometry = CellGeometryDialog(page.state)
    qtbot.addWidget(geometry)
    geometry.inserted.setChecked(True)
    geometry.fields['length_mm'].setValue(100)
    geometry.fields['diameter_mm'].setValue(2000)
    downstream = geometry.window_editors['downstream_window']
    downstream.material.setCurrentIndex(2)
    downstream._choose_preset(2)
    geometry.window_editors['upstream_window'].thickness.setValue(8)
    geometry.accept()
    assert geometry.result() == geometry.DialogCode.Accepted
    page.state.vacuum_map = geometry._value
    page.set_state(page.state)
    page.phase.setCurrentIndex(page.phase.findData('liquid'))
    assert page.pressure.isEnabled() and page.density.isEnabled()
    page.pressure.setText('2500')
    page.formula.setText('H2O')
    page.mixture.setText('{"H2O": 0.9, "C2H6O": 0.1}')
    assert not page.formula.isEnabled()  # Mixture is authoritative while present.
    assert page.apply(), page.status.text()
    cell = page.state.vacuum_map.cell
    assert cell.length_mm == pytest.approx(.0001)
    assert cell.diameter_mm == pytest.approx(.002)
    assert cell.upstream_window.thickness_nm == 8
    assert cell.downstream_window.thickness_nm == .3354
    assert cell.downstream_window.medium.formula == 'C'
    assert cell.medium.mixture_mole_fractions['H2O'] == .9
    assert cell.medium.pressure_mbar == 2500
    assert page.state.sample.z_mm == original_z
    assert not page.state.vacuum_map.enabled
    assert page.chamber.plot.getViewBox().state['yInverted']
    assert page.chamber.layers['interior'] == pytest.approx((-50, 50))


def test_invalid_cell_edit_keeps_committed_state(page, qtbot):
    original = page.state.vacuum_map.to_dict()
    geometry = CellGeometryDialog(page.state)
    qtbot.addWidget(geometry)
    geometry.inserted.setChecked(True)
    geometry.fields['length_mm'].setValue(1)
    geometry.accept()
    assert geometry.result() != geometry.DialogCode.Accepted
    assert 'Sample intersects' in geometry.status.text()
    assert page.state.vacuum_map.to_dict() == original


def test_sample_marker_follows_sample_without_moving_shared_z_view(page):
    page.diagram.setXRange(1500, 1700, padding=0)
    page.state.sample.z_mm += 1
    page.set_state(page.state)
    assert page.diagram.sample_marker.value() == page.state.sample.z_mm
    assert page.diagram.getViewBox().viewRange()[0] == pytest.approx([1500, 1700])
    assert 'Sample Z' in page.chamber.summary.text()
    page.select_region('cell_window_upstream')
    assert page.current_key == 'specimen_cell'


def test_tab_return_refreshes_sample_geometry_without_losing_cell_draft(page, qtbot):
    page.show()
    qtbot.wait(10)
    page.hide()
    page.pressure.setText('123')
    page.state.sample.z_mm += 2
    page.state.sample.thickness_nm = 20
    page.show()
    qtbot.wait(10)
    assert page.diagram.sample_marker.value() == page.state.sample.z_mm
    assert 'Thickness 20 nm' in page.chamber.details.text()
    assert page.pressure.text() == '123'


def test_uint32_seed_roundtrip(page):
    page.seed.setText('3000000000')
    assert page.apply(), page.status.text()
    assert page.state.vacuum_map.seed == 3000000000


def test_windowless_history_can_be_edited_without_added_windows(page):
    page.state.vacuum_map.cell.upstream_window.thickness_nm = 0
    page.state.vacuum_map.cell.downstream_window.thickness_nm = 0
    page.set_state(page.state)
    assert page.apply()
    assert page.state.vacuum_map.cell.upstream_window.thickness_nm == 0


def test_pressure_gradient_and_single_geometry_entry(page, qtbot):
    assert not hasattr(page, 'window_editors')
    with qtbot.waitSignal(page.geometry_requested):
        page.edit_geometry.click()
    page.pressure.setText('1')
    page.gradient.setChecked(True)
    page.end_pressure.setText('9')
    assert page.apply(), page.status.text()
    assert page.state.vacuum_map.cell.pressure_gradient_enabled
    assert page.state.vacuum_map.cell.end_pressure_mbar == 9
    page.phase.setCurrentIndex(page.phase.findData('liquid'))
    assert not page.gradient.isEnabled() and not page.gradient.isChecked()
    assert page.apply(), page.status.text()
    assert not page.state.vacuum_map.cell.pressure_gradient_enabled
