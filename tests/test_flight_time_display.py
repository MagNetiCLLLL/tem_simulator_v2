"""Clock/identity display checks, not physical-column acceptance."""
from types import SimpleNamespace
import numpy as np
from PySide6.QtGui import QColor

from temsim.physics.flight_time import sample_flight_time
from test_transverse_source_tracking import recorded_result, preset, view
from test_beam_analysis_modes import add_detailed_exit
from temsim.gui.filter_plane_data import sample_filter_plane
from temsim.optics.energy_filter_raytrace import FilterPlaneArrival
from temsim.component_keys import ENERGY_FILTER_EFTEM_OUTPUT_PLANE


def test_saved_clocks_are_interpolated_without_using_display_path_lengths():
    b = SimpleNamespace(z=np.array([0., 2.]), x=np.zeros((2, 3)), blocked_z=np.array([np.nan, 1., np.nan]),
                        flight_time_s=np.array([[0., 1., np.nan], [4., 5., np.nan]]) * 1e-9)
    np.testing.assert_allclose(sample_flight_time(b, 1.), [2e-9, 3e-9, np.nan])
    np.testing.assert_allclose(sample_flight_time(b, 1.5), [3e-9, np.nan, np.nan])
    assert np.all(np.isnan(sample_flight_time(b, 3.)))
    b.flight_time_s = None
    assert np.all(np.isnan(sample_flight_time(b, 1.)))


def test_tof_keeps_per_descendant_time_and_same_upper_lower_colours(view):
    result = recorded_result()
    children = add_detailed_exit(result)
    for j, child in enumerate(children):
        n = child.x.shape[1]
        initial = np.full(n, 1e-8)
        child.flight_time_s = np.stack((initial, initial + (np.arange(n)+1+j*4)*1e-16))
    view.display_result(result)
    view.focus_z(2.)
    preset(view, "tof")
    points = view._scatter.data["data"]
    shared = [p for p in points if p["source_ray_id"] == 0]
    assert len(shared) == 2
    assert shared[0]["flight_time_s"] != shared[1]["flight_time_s"]
    assert view.analysis.tof.reference_s == children[0].flight_time_s[-1, 0]
    assert len(view.source_plot.scatter.data) == len(points)
    for lower, upper in zip(view._scatter.data, view.source_plot.scatter.data):
        assert lower["brush"].color() == upper["brush"].color()
        assert lower["data"]["flight_time_s"] == upper["data"]["flight_time_s"]
    assert "overlapping" in view.source_plot.summary.text()
    assert "Delay" in view.source_plot._hover_text(0., 0., view.source_plot.scatter.data["data"][0])
    preset(view, "source")
    assert len(view.source_plot.scatter.data) == 16  # Restore complete source context.
    assert "flight_time_s" not in view.source_plot.scatter.data["data"][0]


def test_tof_preserves_missing_times_and_does_not_rescale_to_display_sample(view):
    result = recorded_result(3001)
    inc = result.simulation.incident
    times = np.linspace(4e-9, 5e-9, 3001)
    times[2] = 1e-9  # Not in linspace display sample, still the time reference.
    times[0] = np.nan
    inc.flight_time_s = np.stack((times*.5, times))
    view.display_result(result)
    preset(view, "tof")
    assert 2 not in view._display_source_ids
    assert view.analysis.tof.reference_s == 1e-9
    assert view._scatter.data["brush"][0].color() == QColor("#94a3b8")
    assert "3,000/3,001 paths timed" in view.summary.text()
    preset(view, "advanced")
    view.analysis.mode_combo.setCurrentIndex(view.analysis.mode_combo.findData("angular"))
    assert view.analysis.mode == "angular"
    assert view.analysis.tracking_combo.currentData() == "advanced"
    assert view.colour_mode.currentData() == "tof"
    assert view.analysis.tof.reference_s == 1e-9
    view._manual_view_range_changed(None)
    bounds = np.array(view.plot.viewRange())
    view.focus_z(.5)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)
    np.testing.assert_allclose(view.analysis.tof.reference_s, .75e-9, rtol=0., atol=1e-24)


def test_tof_no_arrivals_clears_previous_source_and_clock(view):
    result = recorded_result()
    inc = result.simulation.incident
    inc.flight_time_s = np.tile([0., 1e-9], (16, 1)).T
    inc.blocked_z[:] = .4
    view.display_result(result, focus=("z", .2))
    preset(view, "tof")
    assert len(view._scatter.data) == 16
    view.focus_z(.6)
    assert len(view._scatter.data) == 0
    assert view.source_plot.scatter is None
    assert view.analysis.tof.reference_s is None


def filter_result():
    result = recorded_result()
    result.simulation.incident.flight_time_s = np.tile([0., 1e-9], (16, 1)).T
    result.state_snapshot = SimpleNamespace(
        energy_filter=SimpleNamespace(enabled=True, entrance_z_mm=.8))
    result.energy_filter = SimpleNamespace(source_ray_id=np.array([2, 2, 4]),
        source_fraction=np.array([.1, .2, .3]), timed_planes=(FilterPlaneArrival(
            ENERGY_FILTER_EFTEM_OUTPUT_PLANE, np.array([3e-9, 4e-9, np.nan]),
            np.array([1., 2., 3.])*1e-6, np.array([4., 5., 6.])*1e-6,
            np.zeros(3), np.zeros(3), np.array([True, True, False])),))
    return result


def test_filter_plane_requires_executed_crossing_and_keeps_ancestry():
    result = filter_result()
    assert sample_filter_plane(result, .5, "prefilter_camera") is None
    assert sample_filter_plane(result, .8, None) is None
    arbitrary = sample_filter_plane(result, 1., None)
    assert arbitrary.ray_count == 0 and arbitrary.status == "Unavailable"
    data = sample_filter_plane(result, 1., ENERGY_FILTER_EFTEM_OUTPUT_PLANE)
    np.testing.assert_array_equal(data.source_ray_id, [2, 2])
    np.testing.assert_array_equal(data.column_index, [0, 1])
    np.testing.assert_allclose(data.flight_time_s, [3e-9, 4e-9])
    assert data.total_column_count == 3
    assert abs(data.total_source_fraction - .3) < 1e-15
    assert data.coordinate_frame == "sector_exit_local"
    result.energy_filter.timed_planes = ()
    # A named filter output must not borrow a straight-column clock even if
    # its schematic projection falls before the entrance's global Z.
    assert sample_filter_plane(result, .5, ENERGY_FILTER_EFTEM_OUTPUT_PLANE).status == "Unavailable"


def test_filter_display_switches_to_local_frame_and_never_bypasses_filter(view):
    result = filter_result()
    view.display_result(result, focus=("z", .5))
    preset(view, "tof")
    assert len(view._scatter.data) == 16
    view.focus_component(SimpleNamespace(key=ENERGY_FILTER_EFTEM_OUTPUT_PLANE, center_z_mm=1.))
    assert len(view._scatter.data) == 2
    np.testing.assert_allclose(view._scatter.data["x"], [1., 2.])
    assert view.analysis.tof.reference_s == 3e-9
    assert "Dispersive" in view.plot.getAxis("bottom").labelText
    view.focus_z(1.)
    assert not len(view._scatter.data)
    assert "global Z" in view.analysis.readout.text()
    assert view.analysis.readout.isVisible()
    assert view.source_plot.scatter is None
    view.focus_z(.5)
    assert len(view._scatter.data) == 16
    assert "Dispersive" not in view.plot.getAxis("bottom").labelText


def test_nonmonotone_gun_first_crossings_are_not_invalidated():
    branch = SimpleNamespace(z=np.array([0., 2.]), x=np.zeros((2, 1)),
        blocked_z=np.array([np.nan]), flight_time_s=np.array([[2e-12], [0.]]))
    np.testing.assert_allclose(sample_flight_time(branch, 1.), [1e-12], atol=1e-25)
