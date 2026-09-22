"""Quaternion-derived sweep controls are discoverable; no execution."""
import pytest
from temsim.gui.design_explorer import DesignExplorerPage, _sweep_unit


@pytest.mark.parametrize("axis", "xyz")
def test_rotation_sweep_controls_use_current_derived_paths(qtbot, axis):
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    path = f"sample.orientation_euler_{axis}_deg"
    index = page.sweep_parameter.findData(path)
    assert index >= 0
    page.sweep_parameter.setCurrentIndex(index)
    assert page._selected_sweep_path() == path
    assert _sweep_unit(path) == "deg"
    assert page.sweep_parameter.currentText() == f"Sample rotation {axis.upper()} (deg)"
    assert all("specimen_rotation" not in str(page.sweep_parameter.itemData(i))
               for i in range(page.sweep_parameter.count()))
