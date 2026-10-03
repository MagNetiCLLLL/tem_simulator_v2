"""Captured physical cross sections in a cached beam view; no particle solve.

Synthetic outlines test display units, projection and publication ownership.
They do not validate conductor geometry or numerical transport.
"""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt

import temsim.gui.plane_hardware_overlay as overlay_module
import temsim.gui.beam_analysis as analysis_module
from test_beam_analysis_modes import make_result, switch, view
from test_transverse_source_tracking import preset


def rectangle(x0, y0, x1, y1):
    points = np.asarray(((x0, y0), (x1, y0), (x1, y1),
                         (x0, y1), (x0, y0)), dtype=float)
    points.setflags(write=False)
    return points


def outline(key, name, points, *, z_mm=1.0, kind="aperture", role="opening"):
    return SimpleNamespace(
        key=key, name=name, kind=kind, role=role, z_mm=z_mm,
        polylines_mm=(points,), description=f"Captured {name} at Z {z_mm:g} mm",
    )


@pytest.fixture
def hardware_geometry(monkeypatch):
    calls = []
    rows = (
        outline("offset_aperture", "Offset aperture", rectangle(.1, -.4, .3, -.2)),
        outline("column_wall", "Column bore", rectangle(-2., -2., 2., 2.),
                kind="column", role="wall"),
        outline("pixel_detector", "Pixelated detector", rectangle(-.5, -.4, .5, .4),
                kind="detector", role="absorbing"),
    )

    def capture(result, z_mm, *, coordinate_frame="column", circle_segments=128):
        assert coordinate_frame == "column"
        assert 8 <= circle_segments <= 256
        calls.append((result, float(z_mm)))
        return rows

    monkeypatch.setattr(overlay_module, "projected_hardware_outlines", capture)
    return rows, calls


def item_coordinates(hardware):
    return [(np.asarray(item.getData()[0]), np.asarray(item.getData()[1]))
            for item in hardware.items]


def ray_snapshot(view):
    scatter = view._scatter
    if scatter is None:
        return None
    return (
        scatter.data["x"].copy(), scatter.data["y"].copy(),
        tuple(brush.color().rgba() for brush in scatter.data["brush"]),
        view._display_source_ids.copy(),
    )


def assert_same_rays(actual, expected):
    assert (actual is None) == (expected is None)
    if expected is not None:
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]
        np.testing.assert_array_equal(actual[3], expected[3])


def test_hardware_enabled_by_default_with_explicit_fit_and_captured_details(view, hardware_geometry):
    rows, calls = hardware_geometry
    result = make_result()
    view.display_result(result)
    hardware = view.hardware
    assert hardware.toggle.isChecked()
    assert hardware.toggle.text() == "Cutoff projection"
    assert hardware.fit_button.text() == "Fit cutoff"
    assert len(hardware.items) == len(rows)
    assert all(isinstance(item, pg.PlotDataItem) for item in hardware.items)
    assert {row.key for row in hardware.outlines} == {row.key for row in rows}
    assert calls == [(result, 1.)]
    combined = hardware.status.text() + "\n" + hardware.status.toolTip()
    assert "Offset aperture" in combined
    assert "Column bore" in combined


@pytest.mark.parametrize("mode", ["position", "tof", "intensity"])
def test_spatial_views_include_same_hardware_without_changing_coordinates(view, hardware_geometry, mode):
    rows, _calls = hardware_geometry
    result = make_result()
    result.simulation.incident.flight_time_s = np.tile([0., 1e-9], (12, 1)).T
    view.display_result(result)
    if mode == "tof":
        preset(view, "tof")
    elif mode != "position":
        switch(view, mode)
    assert len(view.hardware.items) == len(rows)
    for (x, y), row in zip(item_coordinates(view.hardware), rows):
        np.testing.assert_allclose(x, row.polylines_mm[0][:, 0] * 1000.)
        np.testing.assert_allclose(y, row.polylines_mm[0][:, 1] * 1000.)
    assert view.plot.getAxis("bottom").labelUnits == "µm"
    assert view.plot.getAxis("left").labelUnits == "µm"


def test_offsets_rotate_in_same_uv_basis_and_do_not_mutate_captured_geometry(view, hardware_geometry):
    rows, calls = hardware_geometry
    before = [row.polylines_mm[0].copy() for row in rows]
    view.display_result(make_result())
    view.set_projection_angle(90.)
    for (u, v), row, original in zip(item_coordinates(view.hardware), rows, before):
        np.testing.assert_allclose(u, original[:, 1] * 1000., atol=1e-10)
        np.testing.assert_allclose(v, -original[:, 0] * 1000., atol=1e-10)
        np.testing.assert_array_equal(row.polylines_mm[0], original)
    assert len(calls) == 1  # A display rotation is not new physical geometry.


@pytest.mark.parametrize("mode", ["position", "tof", "intensity"])
def test_empty_arriving_population_keeps_hardware_visible(view, hardware_geometry, mode):
    rows, _calls = hardware_geometry
    result = make_result()
    result.simulation.incident.blocked_z[:] = .4
    result.simulation.incident.flight_time_s = np.tile([0., 1e-9], (12, 1)).T
    view.display_result(result, focus=("z", .6))
    if mode == "tof":
        preset(view, "tof")
    elif mode != "position":
        switch(view, mode)
    assert view._scatter is None or len(view._scatter.data) == 0
    assert len(view.hardware.items) == len(rows)
    assert "Column bore" in view.hardware.status.text() + view.hardware.status.toolTip()


@pytest.mark.parametrize("mode", ["angular", "phase_u", "phase_v", "angle_histogram", "interactions"])
def test_nonspatial_modes_do_not_draw_length_geometry_on_other_axes(view, hardware_geometry, mode):
    _rows, calls = hardware_geometry
    view.display_result(make_result())
    switch(view, mode)
    assert not view.hardware.items
    assert len(calls) == 1
    switch(view, "position")
    assert view.hardware.items
    assert len(calls) == 1


def test_filter_local_frame_does_not_inherit_straight_column_outline(view, hardware_geometry):
    _rows, calls = hardware_geometry
    view.display_result(make_result())
    data = SimpleNamespace(coordinate_frame="sector_exit_local")
    view.hardware.redraw(data)
    assert not view.hardware.items
    assert len(calls) == 1
    assert "filter" in (view.hardware.status.text() + view.hardware.status.toolTip()).lower()
    view.hardware.redraw(SimpleNamespace(coordinate_frame="column"))
    assert view.hardware.items
    assert len(calls) == 1


def test_wave_view_has_no_unbound_particle_hardware_overlay(view, hardware_geometry):
    _rows, calls = hardware_geometry
    view.display_result(make_result())
    # An executed wave checkpoint needs its own captured geometry identity;
    # the previous particle result is not that identity.
    original_wave = view.analysis.wave
    try:
        view.analysis.wave = SimpleNamespace()
        view.hardware.redraw()
        assert not view.hardware.items
        assert len(calls) == 1
    finally:
        view.analysis.wave = original_wave


def test_toggle_reuses_captured_geometry_and_preserves_points_colours_and_scale(view, hardware_geometry, monkeypatch):
    _rows, calls = hardware_geometry
    result = make_result()
    view.display_result(result)
    view._apply_centered_view_ranges(50., 50.)
    view._view_scale_initialized = True
    bounds = np.asarray(view.plot.viewRange())
    points = ray_snapshot(view)
    raw_x = result.simulation.incident.x.copy()

    def unexpected(*_args, **_kwargs):
        pytest.fail("Hardware visibility invoked particle-plane sampling")

    monkeypatch.setattr(view.analysis, "plane_data", unexpected)
    monkeypatch.setattr(analysis_module, "sample_beam_plane", unexpected)
    monkeypatch.setattr(analysis_module, "sample_filter_plane", unexpected)
    view.hardware.toggle.setChecked(False)
    assert not view.hardware.items
    view.hardware.toggle.setChecked(True)
    assert view.hardware.items
    assert len(calls) == 1
    assert_same_rays(ray_snapshot(view), points)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)
    np.testing.assert_array_equal(result.simulation.incident.x, raw_x)


def test_plane_changes_and_rotation_keep_manual_scale_until_explicit_fit(view, hardware_geometry):
    view.display_result(make_result())
    view._apply_centered_view_ranges(37., 37.)
    view._view_scale_initialized = True
    bounds = np.asarray(view.plot.viewRange())
    view.focus_z(.2)
    view.set_projection_angle(90.)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)
    view._fit_beam_view()
    beam_bounds = np.asarray(view.plot.viewRange())
    assert np.max(np.abs(beam_bounds)) < 10.
    view.hardware.fit_button.click()
    hardware_bounds = np.asarray(view.plot.viewRange())
    assert np.max(np.abs(hardware_bounds)) >= 2000.
    assert np.max(np.abs(hardware_bounds)) > np.max(np.abs(beam_bounds))


def test_hardware_fit_includes_full_beam_and_preserves_intensity_mode_range(view, hardware_geometry):
    result = make_result()
    result.simulation.incident.x[-1, 0] = .003  # Beam outside the 2 mm bore.
    view.display_result(result)
    view.hardware.fit()
    bounds = np.asarray(view.plot.viewRange())
    assert bounds[0, 1] >= 3000.
    assert bounds[1, 1] >= 2000.
    switch(view, "intensity")
    view.hardware.fit()
    fitted = np.asarray(view.plot.viewRange())
    assert fitted[0, 1] >= 3000.
    assert fitted[1, 1] >= 2000.
    view.focus_z(.5)
    np.testing.assert_allclose(view.plot.viewRange(), fitted)


def test_publication_invalidates_geometry_even_when_result_object_is_reused(view, hardware_geometry):
    _rows, calls = hardware_geometry
    result = make_result()
    view.display_result(result)
    view._redraw()
    assert len(calls) == 1
    view.display_result(result)
    assert len(calls) == 2
    replacement = make_result()
    view.display_result(replacement)
    assert calls[-1][0] is replacement
    assert len(calls) == 3
    view.display_result(None)
    assert view._result is None
    assert not view.hardware.items
    assert not view.hardware.outlines


def test_fresh_view_can_clear_without_an_executed_plane(view, hardware_geometry):
    _rows, calls = hardware_geometry
    view.display_result(None)
    assert not view.hardware.items
    assert not view.hardware.outlines
    assert calls == []
    assert not view.hardware.fit_button.isEnabled()


def test_aperture_selection_uses_captured_optical_stop_instead_of_carrier_centre(view, hardware_geometry):
    result = make_result()
    result.aperture_stops = ({"key": "offset_aperture", "z_mm": .3},)
    view.display_result(result)
    view.focus_component(SimpleNamespace(key="offset_aperture", center_z_mm=.8))
    assert view._plane_z_mm == .3
    assert hardware_geometry[1][-1] == (result, .3)


def test_invalid_hardware_geometry_is_reported_without_losing_beam_points(view, monkeypatch):
    def invalid(*_args, **_kwargs):
        raise ValueError("Aperture opening diameter has invalid physical geometry")
    monkeypatch.setattr(overlay_module, "projected_hardware_outlines", invalid)
    view.display_result(make_result())
    assert len(view._scatter.data) == 12
    assert not view.hardware.items
    assert not view.hardware.fit_button.isEnabled()
    assert "unavailable" in view.hardware.status.text()
    assert "diameter" in view.hardware.status.toolTip()


def test_geometry_cache_reuses_display_edits_and_has_bounded_exact_z_history(view, hardware_geometry):
    _rows, calls = hardware_geometry
    view.display_result(make_result())
    view._redraw()
    view.set_projection_angle(35.)
    view.set_projection_angle(90.)
    assert len(calls) == 1
    # The intentionally small geometry cache cannot grow without limit while
    # a user drags through many unique planes. No numeric transport is run.
    planes = np.linspace(.1, .9, 257)
    for z_mm in planes:
        view.focus_z(float(z_mm))
    count = len(calls)
    view._redraw()
    assert len(calls) == count
    view.focus_z(float(planes[0]))
    assert len(calls) == count + 1
    view._redraw()
    assert len(calls) == count + 1
    view.focus_z(float(planes[-1]))
    assert len(calls) == count + 2  # A single-plane cache is deliberately bounded.


def test_named_contours_have_true_z_and_fit_buttons_share_one_row(view, hardware_geometry):
    rows, _calls = hardware_geometry
    view.display_result(make_result(), focus=("z", 2.))
    hardware = view.hardware
    assert len(hardware.labels) == len(rows)
    for label, row, item in zip(hardware.labels, rows, hardware.items):
        assert row.name in label.toPlainText()
        assert "Z 1 mm" in label.toPlainText()
        assert item.opts["pen"].style() == Qt.PenStyle.DashLine
    layout = view.section_beam_panel.layout()
    same_row = [layout.itemAt(i).layout() for i in range(layout.count())
                if layout.itemAt(i).layout() is not None]
    assert any(row.indexOf(view.fit_beam) >= 0 and row.indexOf(hardware.fit_button) >= 0
               for row in same_row)


def test_each_boundary_can_hide_and_reset_without_modifying_beam_or_view(view, hardware_geometry, monkeypatch):
    rows, calls = hardware_geometry
    view.display_result(make_result())
    points, bounds = ray_snapshot(view), np.asarray(view.plot.viewRange())
    hardware = view.hardware
    action = next(action for identity, action in hardware.visibility_actions.items()
                  if identity[2] == "offset_aperture")
    monkeypatch.setattr(analysis_module, "sample_beam_plane", lambda *_a: pytest.fail("Visibility resampled beam"))
    action.setChecked(False)
    assert len(hardware.items) == len(rows) - 1
    assert len(hardware.labels) == len(rows) - 1
    assert hardware.toggle.isChecked()
    hardware.reset_button.click()
    assert len(hardware.items) == len(rows)
    assert all(action.isChecked() for action in hardware.visibility_actions.values())
    assert len(calls) == 1
    assert_same_rays(ray_snapshot(view), points)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)


def test_annular_detector_inner_outer_visibility_is_independent(view, monkeypatch):
    outer = rectangle(-2., -2., 2., 2.)
    inner = rectangle(-1., -1., 1., 1.)
    row = outline("haadf", "HAADF detector", outer, kind="detector", role="absorbing")
    row.polylines_mm = (outer, inner)
    monkeypatch.setattr(overlay_module, "projected_hardware_outlines", lambda *_a, **_k: (row,))
    view.display_result(make_result())
    actions = view.hardware.visibility_actions
    outer_action = next(action for identity, action in actions.items() if identity[4] == 0)
    inner_action = next(action for identity, action in actions.items() if identity[4] == 1)
    assert "outer" in outer_action.text() and "inner" in inner_action.text()
    inner_action.setChecked(False)
    assert outer_action.isChecked() and len(view.hardware.items) == 1
    np.testing.assert_array_equal(view.hardware.items[0].getData()[0], outer[:, 0] * 1000.)


def stopped_result():
    result = make_result()
    branch = result.simulation.incident
    branch.blocked_key = np.full(12, "", dtype=object)
    branch.blocked_key[0] = "offset_aperture"
    branch.blocked_z[0] = .25
    result.aperture_stops = ({"key": "offset_aperture", "name": "Offset aperture"},)
    return result


def test_stop_crosses_use_actual_interception_xy_z_and_rotate_with_beam(view, hardware_geometry):
    result = stopped_result()
    branch = result.simulation.incident
    view.display_result(result)
    item, = view.hardware.stop_items
    expected_x = (.75 * branch.x[0, 0] + .25 * branch.x[1, 0]) * 1e6
    expected_y = (.75 * branch.y[0, 0] + .25 * branch.y[1, 0]) * 1e6
    np.testing.assert_allclose(item.data["x"], [expected_x])
    np.testing.assert_allclose(item.data["y"], [expected_y])
    assert "Offset aperture" in item.data["data"][0]
    assert "stop Z 0.25 mm" in item.data["data"][0]
    assert "source ray 0" in item.data["data"][0]
    view.set_projection_angle(90.)
    item, = view.hardware.stop_items
    np.testing.assert_allclose(item.data["x"], [expected_y], atol=1e-12)
    np.testing.assert_allclose(item.data["y"], [-expected_x], atol=1e-12)
    view.focus_z(.2)
    assert not view.hardware.stop_items  # Has not yet encountered this stop.


def test_stop_visibility_independent_of_outlines_and_no_particle_resolve(view, hardware_geometry, monkeypatch):
    view.display_result(stopped_result())
    hardware = view.hardware
    points, bounds = ray_snapshot(view), np.asarray(view.plot.viewRange())
    monkeypatch.setattr(view.analysis, "plane_data", lambda: pytest.fail("Visibility sampled arrivals"))
    hardware.toggle.setChecked(False)
    assert not hardware.items and hardware.stop_items
    hardware.stops_toggle.setChecked(False)
    assert not hardware.stop_items
    hardware.reset_visibility()
    assert hardware.items and hardware.stop_items
    assert_same_rays(ray_snapshot(view), points)
    np.testing.assert_allclose(view.plot.viewRange(), bounds)


def test_live_insertion_preview_detaches_inputs_and_keeps_executed_stops(view, monkeypatch):
    from temsim.optics.column import default_state
    state = default_state()
    result = stopped_result()
    result.state_snapshot = state
    # Use real detector geometry at the selected detector's physical location.
    result.aperture_stops = ()
    detector = state.dark_field_detector
    target = detector.z_mm
    view.display_result(result, focus=("z", target))
    assert any(row.key == detector.key and row.kind == "detector" for row in view.hardware.outlines)
    points = ray_snapshot(view)
    detector.inserted = False
    view.hardware.set_current_state(state)
    assert not any(row.key == detector.key and row.kind == "detector" for row in view.hardware.outlines)
    assert "Edited hardware preview" in view.hardware.status.text()
    assert "previous calculation" in view.hardware.status.toolTip()
    assert_same_rays(ray_snapshot(view), points)
    detector.inserted = True
    view.hardware.redraw()
    assert not any(row.key == detector.key and row.kind == "detector" for row in view.hardware.outlines)  # Detached copy.
    view.hardware.set_current_state(state)
    assert any(row.key == detector.key and row.kind == "detector" for row in view.hardware.outlines)
    view.display_result(result, focus=("z", target))
    assert "Edited hardware preview" not in view.hardware.status.text()


def test_coincident_wall_rendering_preserves_each_name_and_independent_visibility(view, monkeypatch):
    points = rectangle(-2., -2., 2., 2.)
    rows = (outline("wall_a", "Body A", points, kind="column", role="wall", z_mm=.2),
            outline("wall_b", "Body B", points, kind="column", role="wall", z_mm=.4))
    monkeypatch.setattr(overlay_module, "projected_hardware_outlines", lambda *_a, **_k: rows)
    view.display_result(make_result())
    hardware = view.hardware
    assert len(hardware.items) == 1
    assert "Body A | Z 0.2 mm" in hardware.items[0].toolTip()
    assert "Body B | Z 0.4 mm" in hardware.items[0].toolTip()
    label = hardware.labels[0].toPlainText()
    view.set_projection_angle(13.)
    assert hardware.labels[0].toPlainText() == label  # Physical diameter is invariant.
    action_a = next(action for identity, action in hardware.visibility_actions.items() if identity[2] == "wall_a")
    action_b = next(action for identity, action in hardware.visibility_actions.items() if identity[2] == "wall_b")
    action_a.setChecked(False)
    assert len(hardware.items) == 1
    assert "Body A" not in hardware.items[0].toolTip()
    assert "Body B" in hardware.items[0].toolTip()
    action_b.setChecked(False)
    assert not hardware.items
    hardware.reset_visibility()
    assert len(hardware.items) == 1 and len(hardware.visibility_actions) == 2


def test_stop_summary_immediately_names_component_and_true_z(view, hardware_geometry):
    view.display_result(stopped_result())
    text = view.hardware.status.text()
    assert "Offset aperture" in text and "shown Z 0.25 mm" in text
    assert "1/1 paths" in text


def test_clear_discards_old_cutoff_menu_and_range_callbacks_stay_safe(view, hardware_geometry):
    view.display_result(make_result())
    hardware = view.hardware
    identity, action = next(iter(hardware.visibility_actions.items()))
    action.setChecked(False)
    assert hardware.visibility_button.isEnabled()
    view.display_result(None)
    assert not hardware.visibility_button.isEnabled()
    assert not hardware.menu.actions() and not hardware.visibility_actions
    assert not hardware._annotations
    view.plot.setRange(xRange=(-40., 40.), yRange=(-40., 40.), padding=0.)
    hardware.reset_visibility()
    assert not hardware.menu.actions()
    view.display_result(make_result())
    assert hardware.visibility_button.isEnabled()
    assert identity in hardware.visibility_actions
