"""All transverse observables read the same saved physical filter crossing.

Deterministic display fixtures; these do not qualify a new filter calculation.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.component_keys import ENERGY_FILTER_EFTEM_OUTPUT_PLANE, ENERGY_FILTER_SLIT
from temsim.gui.filter_plane_data import sample_filter_plane
from test_beam_analysis_modes import add_detailed_exit, switch, view
from test_flight_time_display import filter_result
from test_transverse_source_tracking import colours, preset


def focus_output(view, result=None):
    result = filter_result() if result is None else result
    view.display_result(result, focus=("z", .5))
    view.focus_component(SimpleNamespace(key=ENERGY_FILTER_EFTEM_OUTPUT_PLANE, center_z_mm=1.))
    return result


@pytest.mark.parametrize("tracking", ["source", "emission_direction", "tof"])
def test_each_tracking_preset_uses_only_executed_filter_arrivals(view, tracking):
    result = filter_result()
    plane = result.energy_filter.timed_planes[0]
    result.energy_filter.timed_planes = (replace(plane,
        tx_rad=np.array([.01, .02, .03]), ty_rad=np.array([-.04, -.05, -.06])),)
    focus_output(view, result)
    preset(view, tracking)
    assert len(view._scatter.data) == 2
    np.testing.assert_array_equal(view._display_source_ids, [2, 2])
    expected_x = np.arctan([.01, .02])*1e3 if tracking == "emission_direction" else [1., 2.]
    expected_y = np.arctan([-.04, -.05])*1e3 if tracking == "emission_direction" else [4., 5.]
    np.testing.assert_allclose(view._scatter.data["x"], expected_x)
    np.testing.assert_allclose(view._scatter.data["y"], expected_y)
    assert "30% of source" in view.summary.text()
    assert "Dispersive" in view.plot.getAxis("bottom").labelText
    assert "Filter exit frame" in view.heading.text()
    if tracking != "tof":
        launch_colours = colours(view.source_plot.scatter)
        for row in view._scatter.data:
            assert row["brush"].color().rgba() == launch_colours[row["data"]["source_ray_id"]]


@pytest.mark.parametrize("mode", ["position", "angular", "phase_u", "phase_v",
                                 "intensity", "angle_histogram", "interactions"])
def test_all_advanced_views_share_filter_population_and_local_axes(view, mode):
    focus_output(view)
    preset(view, "advanced")
    switch(view, mode)
    data = view.analysis.plane_data()
    assert data.ray_count == 2
    assert data.total_source_fraction == pytest.approx(.3)
    assert "30% of source" in view.summary.text()
    if mode == "interactions":
        assert view.analysis.table.rowCount() == 1
        assert view.analysis.table.item(0, 1).text() == "30"
    elif mode in {"intensity", "angle_histogram"}:
        view.analysis.fit()
        assert np.sum(view.analysis._hover_payload[1]) == pytest.approx(60.)
        axis = view.plot.getAxis("bottom").labelText
        assert ("outgoing axis" in axis if mode == "angle_histogram" else "Dispersive" in axis)
    else:
        assert len(view._scatter.data) == 2
        assert "dispersive" in view.plot.getAxis("bottom").labelText.lower()


@pytest.mark.parametrize("tracking", ["source", "emission_direction", "tof"])
def test_global_z_does_not_resurrect_straight_column_after_filter(view, tracking):
    focus_output(view)
    preset(view, tracking)
    view.focus_z(1.)  # Same Z, different physical-plane identity.
    assert view._scatter is None or len(view._scatter.data) == 0
    assert view.analysis.plane_data().status == "Unavailable"
    assert "global Z" in view.analysis.readout.text()
    assert view.analysis.readout.isVisible()
    view.focus_z(.5)
    assert len(view._scatter.data) == 16
    assert "Dispersive" not in view.plot.getAxis("bottom").labelText


def test_distinct_filter_components_at_same_global_z_do_not_share_cache(view):
    result = filter_result()
    output = result.energy_filter.timed_planes[0]
    slit = replace(output, key=ENERGY_FILTER_SLIT,
                   reached=np.array([True, False, False]), x_m=np.array([7., 8., 9.])*1e-6)
    result.energy_filter.timed_planes = (output, slit)
    focus_output(view, result)
    first = view.analysis.plane_data()
    view.focus_component(SimpleNamespace(key=ENERGY_FILTER_SLIT, center_z_mm=1.))
    second = view.analysis.plane_data()
    assert second is not first
    assert second.ray_count == 1
    np.testing.assert_allclose(view._scatter.data["x"], [7.])
    assert "10% of source" in view.summary.text()


def test_position_filter_transition_retains_user_range(view):
    result = filter_result()
    view.display_result(result, focus=("z", .5))
    view._apply_centered_view_ranges(12., 15.)
    view._view_scale_initialized = True
    bounds = np.asarray(view.plot.viewRange())
    view.focus_component(SimpleNamespace(key=ENERGY_FILTER_EFTEM_OUTPUT_PLANE, center_z_mm=1.))
    np.testing.assert_allclose(view.plot.viewRange(), bounds)
    view.focus_z(.5)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)


@pytest.mark.parametrize("changes", [
    {"time_reference": "filter_entrance"}, {"time_s": None}, {"time_s": ["missing"]},
])
def test_missing_tip_clock_keeps_geometry_and_flux_without_fabricating_time(view, changes):
    result = filter_result()
    plane = result.energy_filter.timed_planes[0]
    result.energy_filter.timed_planes = (replace(plane, **changes),)
    focus_output(view, result)
    for tracking in ("source", "emission_direction", "tof"):
        preset(view, tracking)
        assert len(view._scatter.data) == 2
        assert "30% of source" in view.summary.text()
    assert view.analysis.tof.reference_s is None
    assert np.all(np.isnan(view.analysis.plane_data().flight_time_s))


@pytest.mark.parametrize("ids", [None, [2.9, 2., 4.], [2, 4]])
def test_missing_or_invalid_source_ids_stay_unknown_with_geometry_intact(view, ids):
    result = filter_result()
    result.energy_filter.source_ray_id = ids
    focus_output(view, result)
    np.testing.assert_array_equal(view._display_source_ids, [-1, -1])
    assert all(row["brush"].color().name() == "#94a3b8" for row in view._scatter.data)
    assert "30% of source" in view.summary.text()


def test_filter_interactions_follow_recorded_descendant_not_shared_source_id(view):
    result = filter_result()
    branches = add_detailed_exit(result)
    output = result.energy_filter
    output.source_ray_id = np.array([0, 0, 2])
    output.source_branch = (branches[0].name, branches[1].name, branches[1].name)
    output.source_path_index = np.array([0, 0, 1])
    output.entrance_provenance = "validated_specimen_exit"
    original = tuple(branch.x.copy() for branch in branches)
    fractions = output.source_fraction.copy()
    focus_output(view, result)
    preset(view, "advanced")
    view.colour_mode.setCurrentIndex(view.colour_mode.findData("interaction"))
    assert list(view._scatter.data["symbol"]) == ["o", "star"]
    switch(view, "interactions")
    table = view.analysis.table
    assert [table.item(i, 1).text() for i in range(2)] == ["10", "20"]
    assert "Elastic + plasmon" in table.item(1, 0).text()
    assert "Specimen exit" in view.summary.text()
    np.testing.assert_array_equal(output.source_fraction, fractions)
    for branch, before in zip(branches, original):
        np.testing.assert_array_equal(branch.x, before)
    # A mismatched saved path must never inherit another descendant's channel.
    output.source_path_index[1] = 1
    data = sample_filter_plane(result, 1., ENERGY_FILTER_EFTEM_OUTPUT_PLANE)
    assert data.interaction_key[1] == "filter"


@pytest.mark.parametrize("mode", ["intensity", "angle_histogram", "interactions"])
def test_unavailable_filter_crossing_clears_quantitative_views(view, mode):
    focus_output(view)
    preset(view, "advanced")
    switch(view, mode)
    view.focus_z(1.)
    assert view.analysis._hover_payload is None
    assert view.analysis.table.rowCount() == 0
    assert "Flux unavailable" in view.summary.text()
    assert "global Z" in view.analysis.readout.text()


def test_local_projection_and_display_limit_do_not_rescale_filter_weight(view):
    result = filter_result()
    focus_output(view, result)
    view.MAX_DISPLAY_RAYS = 1
    view.set_projection_angle(90.)
    np.testing.assert_allclose(view._scatter.data["x"], [4.])
    np.testing.assert_allclose(view._scatter.data["y"], [-1.])
    assert "rotated" in view.plot.getAxis("bottom").labelText
    assert "30% of source" in view.summary.text()
    preset(view, "tof")
    assert len(view._scatter.data) == 1
    assert view.analysis.tof.reference_s == 3e-9
    assert "30% of source" in view.summary.text()
