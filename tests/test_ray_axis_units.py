"""Physical SI tick labels without changing the millimetre ray/view geometry."""
import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace


@pytest.fixture
def view(qtbot):
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("axis_name", ["bottom", "left"])
@pytest.mark.parametrize("span_mm,prefix", [
    (30e-6, "n"), (30e-3, "µ"), (30., "m"), (3000., ""),
])
def test_ray_ticks_use_one_si_prefix(view, axis_name, span_mm, prefix):
    axis = view.plot.getAxis(axis_name)
    axis.setRange(-span_mm, span_mm)
    assert axis.labelUnits == "m"
    assert axis.labelUnitPrefix == prefix
    assert axis.scale == pytest.approx(1e-3)
    scale = axis.autoSIPrefixScale * axis.scale
    expected_tick = 2. if not prefix else 20.
    tick_mm = span_mm * 2 / 3
    assert float(axis.tickStrings([tick_mm], scale, span_mm / 6)[0]) == pytest.approx(expected_tick)
    assert f"({prefix}m)" in axis.labelString()


def test_zoom_and_label_refresh_preserve_coordinates_and_selected_plane(view):
    x = np.array([0., 14e-6, 30e-6])
    y = np.array([0., 10e-6, 20e-6])
    item = view.plot.plot(x, y)
    view._selected_z_mm = 14e-6
    view.plot.disableAutoRange()
    for span_mm in (30e-6, 30., 30e-3, 30e-6):
        view.plot.setXRange(-span_mm, span_mm, padding=0)
        view.plot.setYRange(-span_mm, span_mm, padding=0)
        before = view.plot.getViewBox().viewRange()
        view._set_ray_axis_label("left", "Projected displacement")
        view._set_ray_axis_label("bottom", "Axial position")
        np.testing.assert_array_equal(view.plot.getViewBox().viewRange(), before)
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)
        assert view._selected_z_mm == 14e-6
    assert view.plot.getAxis("bottom").labelUnitPrefix == "n"
    assert view.plot.getAxis("left").labelUnitPrefix == "n"
