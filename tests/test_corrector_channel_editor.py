"""Non-material selections stay inspectable without offering material edits."""
from copy import deepcopy

import pytest
from PySide6.QtCore import Qt

from temsim.gui.part_model_editor import PartModelEditorPage


@pytest.fixture
def page(qtbot, tmp_path):
    path = tmp_path / "channels.toml"
    rows = []
    for key in ("probe_qph1_quadrupole", "probe_dp12_scan_deflector", "real_coil"):
        rows.append(f'''[[parts]]
key = "{key}"
name = "{key}"
local_start_z_mm = 10.0
local_center_z_mm = 12.0
local_end_z_mm = 14.0
length_mm = 4.0
mechanical_profile = "magnetic_excitation_coil"
mechanical_inner_diameter_mm = 20.0
mechanical_outer_diameter_mm = 75.0
effective_length_mm = 2.0
''')
    path.write_text("\n".join(rows), encoding="utf-8")
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(path, selected_key="probe_qph1_quadrupole")
    return page


def _dimension(page, field):
    for row in range(page.dimensions.rowCount()):
        item = page.dimensions.item(row, 1)
        if item.data(Qt.ItemDataRole.UserRole) == ("parts", page._selected_key, field):
            return item
    raise AssertionError(field)


@pytest.mark.parametrize("key", ["probe_qph1_quadrupole", "probe_dp12_scan_deflector"])
def test_channel_selection_explains_empty_view_and_material_actions_restore_for_hardware(page, key):
    page.select_part(key)
    assert not page._mesh_records
    assert "no independent material body" in page.view._empty_text
    assert "no independent material body" in page.material_current.text()
    controls = (page.assign_button, page.material, page.region, page.base_shape,
                page.apply_shape_button, page.add_hole_button, page.add_slot_button)
    assert all(not control.isEnabled() for control in controls)
    assert page.save_copy_button.isEnabled()
    assert page.new_component_button.isEnabled()
    before = deepcopy(page.session.document)
    page.assign_material()
    page._apply_base_shape()
    page._edit_feature("hole")
    page._reset_model_shape()
    assert page.session.document == before
    page.select_part("real_coil")
    assert page._mesh_records
    assert all(control.isEnabled() for control in controls)


def test_dimension_inspector_and_all_parameters_identify_stored_channel_values(page):
    item = _dimension(page, "mechanical_outer_diameter_mm")
    field = item.data(Qt.ItemDataRole.UserRole + 2)
    for passed_field in (field, None):
        meaning, _, text = page._parameter_information(field.path, passed_field)
        assert meaning.category_label == "Channel / reference metadata"
        assert "Material outer diameter" not in text
        assert "no independent material body" in text
    assert "Channel / reference metadata" in item.toolTip()
    assert "Material outer diameter" not in item.toolTip()


def test_optical_effective_length_remains_editable_for_channel(page):
    item = _dimension(page, "effective_length_mm")
    assert item.flags() & Qt.ItemFlag.ItemIsEditable
    item.setText("3.0")
    assert page.session.part(page._selected_key)["effective_length_mm"] == 3.
    assert page.session.dirty and page.save_button.isEnabled()
    assert not page._mesh_records and not page.apply_shape_button.isEnabled()
