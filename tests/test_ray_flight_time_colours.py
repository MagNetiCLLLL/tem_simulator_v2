"""Integrated display-only clock gradients from small synthetic ray records."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QColor

from temsim.gui.flight_time_colours import FlightTimeColourScale
from temsim.gui.ray_flight_time_colours import path_times
from temsim.gui.ray_scalar_colours import scalar_colour_indices, scalar_rgb
from temsim.gui.visualization import VisualizationWorkspace
from test_ray_source_colour_mode import _branch, _checkpoint, _result


def timed_result():
    result = _result()
    for j, branch in enumerate((result.simulation.incident, result.simulation.branches["000"])):
        branch.flight_time_s = np.tile(np.array([j, j + 1.])[:, None], (1, 3)) * 1e-9
    return result


@pytest.fixture
def workspace(qtbot, monkeypatch):
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    monkeypatch.setattr(view, "_sync_ray_static_layers", lambda _r: None)
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
    monkeypatch.setattr(view, "_update_sample_region_control_availability", lambda: None)
    monkeypatch.setattr(view.sample_interactions_3d, "set_sample_region_result", lambda _r: None)
    return view


def publish_synthetic(view, result):
    view._last_result = view._high_accuracy_result = result
    view._last_quality = "High accuracy"
    view._draw_ray_diagram(result, "High accuracy")
    view.transverse_beam.display_result(result, focus=("z", 12.))
    view.ray_colour_mode.setCurrentIndex(view.ray_colour_mode.findData("tof"))


def test_ray_and_plane_share_cumulative_palette_and_rotation_keeps_gradient(workspace):
    result = timed_result()
    publish_synthetic(workspace, result)
    scale = workspace.transverse_beam.analysis.tof.colour_scale(result)
    assert scale.maximum_s == 2e-9
    expected = QColor(*scalar_rgb(int(scalar_colour_indices([1.2e-9], scale.maximum_s)[0])))
    assert all(brush.color() == expected for brush in workspace.transverse_beam._scatter.data["brush"])
    initial = workspace._ray_flight_time_colours.groups()
    assert len(initial) == 128
    colours = {key: item.opts["pen"].color() for key, item in workspace._ray_items_by_group.items()}
    originals = {key: (x.copy(), y.copy()) for key, (x, y) in initial.items()}
    workspace.plot.setRange(xRange=(2., 18.), yRange=(-.3, .3), padding=0, disableAutoRange=True)
    assert workspace._ray_flight_time_colours.groups() is initial
    workspace._projection_angle_deg = 90.
    workspace._redraw_projection_items()
    assert workspace._ray_flight_time_colours.groups() is not initial
    assert colours == {key: item.opts["pen"].color() for key, item in workspace._ray_items_by_group.items()}
    for key, (x, y) in workspace._ray_flight_time_colours.groups().items():
        np.testing.assert_array_equal(x, originals[key][0])
    assert any(not np.array_equal(y, originals[key][1], equal_nan=True)
               for key, (_x, y) in workspace._ray_flight_time_colours.groups().items())


def test_path_times_interpolates_supplied_stop_and_keeps_missing_clock(workspace):
    result = timed_result()
    branch = result.simulation.incident
    branch.blocked_z[0] = 7.5
    workspace._last_result = result
    z, _y = workspace._display_bundle_lines(branch, np.array([0]))
    np.testing.assert_allclose(z, [0., 7.5, np.nan])
    np.testing.assert_allclose(path_times(workspace, branch, 0, z), [0., .75e-9, np.nan])
    branch.flight_time_s = None
    assert np.all(np.isnan(path_times(workspace, branch, 0, z)))


def test_turning_tip_prefix_uses_temporal_order_including_repeated_z(workspace):
    result = timed_result()
    branch = result.simulation.incident
    branch.flight_time_s[0] = .5e-9
    history = SimpleNamespace(
        z_mm=np.tile(np.array([-.3, -.1, -.2, -.1, .1])[:, None], (1, 3)),
        x_m=np.tile(np.linspace(0., 4e-6, 5)[:, None], (1, 3)),
        y_m=np.tile(np.linspace(0., 8e-6, 5)[:, None], (1, 3)),
        time_s=np.arange(5) * 1e-10,
        alive=np.ones((5, 3), dtype=bool), completed=np.zeros((5, 3), dtype=bool),
    )
    result.simulation.gun_trace = SimpleNamespace(equal_time_history=history)
    workspace._last_result = result
    z, _y = workspace._display_bundle_lines(branch, np.array([0]))
    np.testing.assert_allclose(z[:4], [-.3, -.1, -.2, -.1])
    np.testing.assert_allclose(path_times(workspace, branch, 0, z)[:4], np.arange(4) * 1e-10)
    # A similarly shaped downstream branch must never borrow the gun prefix.
    downstream = result.simulation.branches["000"]
    assert np.all(np.isnan(path_times(workspace, downstream, 0, z[:4])))


def test_scale_includes_full_population_but_excludes_uncalculated_stopped_tail():
    result = timed_result()
    branch = result.simulation.incident
    branch.blocked_z[2] = 5.
    branch.flight_time_s[-1, 2] = 20e-9
    # The between-row physical stop time is 10 ns; the following 20 ns row is not reached.
    assert FlightTimeColourScale.from_result(result).maximum_s == 10e-9
    branch.flight_time_s[-1, 1] = np.nan
    assert FlightTimeColourScale.from_result(result).maximum_s == 10e-9


def test_completed_negative_z_gun_stop_does_not_invent_an_arrival_clock(workspace):
    result = timed_result()
    result.simulation.branches = {}
    branch = result.simulation.incident
    branch.blocked_z[:] = -.1
    branch.flight_time_s[:] = np.nan
    history = SimpleNamespace(
        z_mm=np.tile(np.array([-.3, -.2, -.1, -.1])[:, None], (1, 3)),
        x_m=np.zeros((4, 3)), y_m=np.zeros((4, 3)),
        time_s=np.arange(4) * 1e-10,
        alive=np.tile(np.array([True, True, False, False])[:, None], (1, 3)),
        completed=np.tile(np.array([False, False, True, True])[:, None], (1, 3)),
    )
    result.simulation.gun_trace = SimpleNamespace(equal_time_history=history)
    workspace._last_result = result
    z, _y = workspace._display_bundle_lines(branch, np.array([0]))
    times = path_times(workspace, branch, 0, z)
    np.testing.assert_allclose(z, [-.3, -.2, -.1, np.nan])
    # Completed positions are held at the boundary. Their sampling timestamp
    # is not the exact crossing event, which may lie before that history row.
    # Without a separately recorded terminal event the endpoint must be grey,
    # including the first inactive row; this follows _resample_gun_flight_times.
    np.testing.assert_allclose(times, [0., 1e-10, np.nan, np.nan])
    assert FlightTimeColourScale.from_result(result).maximum_s == 1e-10


def test_tof_scan_offsets_do_not_fabricate_new_clock_geometry(workspace):
    result = timed_result()
    publish_synthetic(workspace, result)
    branch = result.simulation.incident
    before = workspace._display_bundle_lines(branch, np.array([0]))
    workspace._scan_ray_offsets_m[branch.name] = np.full((2, 2), 1e-3)
    after = workspace._display_bundle_lines(branch, np.array([0]))
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    workspace.ray_colour_mode.setCurrentIndex(workspace.ray_colour_mode.findData("source"))
    assert not np.array_equal(workspace._display_bundle_lines(branch, np.array([0]))[1], before[1], equal_nan=True)


@pytest.mark.parametrize("transverse_visible", [False, True])
def test_detailed_exit_publication_invalidates_shared_time_scale_before_redraw(
        workspace, monkeypatch, transverse_visible):
    result = timed_result()
    publish_synthetic(workspace, result)
    assert workspace.transverse_beam.analysis.tof.colour_scale(result).maximum_s == 2e-9
    branch = _branch("late_scattering", (10., 20.))
    branch.flight_time_s = np.tile(np.array([1., 4.])[:, None], (1, 3)) * 1e-9
    checkpoint = _checkpoint((branch,))
    region = SimpleNamespace(metrics={"sample_downstream_signature": "current"}, specimen_exit=checkpoint)
    monkeypatch.setattr(workspace.transverse_beam, "isVisible", lambda: transverse_visible)
    for panel in (workspace.physical_layout, workspace.magnetic_field):
        monkeypatch.setattr(panel, "isVisible", lambda: False)
    workspace._set_sample_region_result(region)
    assert workspace.transverse_beam.analysis.tof.colour_scale(result).maximum_s == 4e-9
    # The top palette bin must begin at t = 127/128 * 4 ns, rather than 2 ns.
    # The late branch runs from 1 ns at Z=10 to 4 ns at Z=20.
    final_group = workspace._ray_items_by_group[("tof", 127)].getData()[0]
    expected_start = 10. + ((127 / 128 * 4) - 1) / 3 * 10
    assert np.nanmin(final_group) == pytest.approx(expected_start)
