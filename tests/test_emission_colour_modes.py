"""Synthetic cached-ray UI checks, including reordered specimen descendants."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QColor

from test_beam_analysis_modes import make_result, add_detailed_exit, switch, view


def with_launch():
    result = make_result()
    n = result.simulation.incident.x.shape[1]
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.linspace(.02, 1.3, n)
    result.simulation.gun_trace = SimpleNamespace(emission_reference={
        "ray_id": np.arange(n),
        "direction": np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))),
        "normal": np.tile([0., 0., 1.], (n, 1)),
    })
    return result


@pytest.mark.parametrize("colour", ["emission_direction", "emission_angle"])
def test_emission_colours_survive_projection_plane_focus_and_descendants(view, colour, monkeypatch):
    result = with_launch()
    add_detailed_exit(result)
    original = result.simulation.incident.x.copy()
    view.display_result(result)
    view.colour_mode.setCurrentIndex(view.colour_mode.findData(colour))
    by_id = {row["source_ray_id"]: brush.color().rgba()
             for row, brush in zip(view._scatter.data["data"], view._scatter.data["brush"])}
    view._apply_centered_view_ranges(3, 3)
    saved = np.array(view.plot.viewRange())
    view.focus_z(.4)
    view.set_projection_angle(75)
    np.testing.assert_allclose(view.plot.viewRange(), saved)
    for mode in ("position", "angular", "phase_u", "phase_v"):
        switch(view, mode)
        view.focus_z(2.)
        for data, brush in zip(view._scatter.data["data"], view._scatter.data["brush"]):
            assert brush.color().rgba() == by_id[data["source_ray_id"]]
            assert "Launch azimuth" in data["emission"]
    np.testing.assert_array_equal(result.simulation.incident.x, original)
    assert view._result is result
    if colour == "emission_angle":
        assert "90°" in view.analysis.legend.text()
    else:
        assert "Emission direction" in view.initial_beam_heading.text()
        assert "before extraction" in view.angle_colour_wheel.toolTip()


@pytest.mark.parametrize("mode", ["position", "angular"])
def test_old_cache_does_not_invent_emission_angles_from_current_slopes(view, mode):
    view.display_result(make_result())
    switch(view, mode)
    for colour in ("emission_direction", "emission_angle"):
        view.colour_mode.setCurrentIndex(view.colour_mode.findData(colour))
        assert all(brush.color() == QColor("#94a3b8") for brush in view._scatter.data["brush"])
