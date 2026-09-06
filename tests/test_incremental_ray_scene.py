"""Incremental Ray Diagram regressions with synthetic, already-computed rays.

No trajectory propagation, field solve, specimen model, or preset calculation is
performed here. Axial coordinates are millimetres; branch X/Y are metres.
"""

import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui.visualization import VisualizationWorkspace
from temsim.gui.ray_scene import StaticRayLayers


class _Assembly(SimpleNamespace):
    def part(self, key):
        for part in self.parts:
            if part.key == key:
                return part
        raise KeyError(key)


def _part(key, name, z, *, profile="component", **data):
    return SimpleNamespace(
        key=key, name=name, start_z_mm=z - 0.2, center_z_mm=z,
        end_z_mm=z + 0.2, length_mm=0.4,
        data={"mechanical_profile": profile, "local_center_z_mm": z, **data},
    )


def _branch(name, z, scale, *, count=9, stops=True):
    z = np.asarray(z, dtype=float)
    index = np.linspace(-1.0, 1.0, count)
    phase = (z - z[0]) / max(z[-1] - z[0], 1.0)
    x = scale * 1e-4 * ((1.0 - 0.7 * phase[:, None]) * index[None, :]
                        + 0.13 * np.sin(z[:, None] + index[None, :]))
    y = scale * 1e-4 * (0.35 * np.cos(2.0 * index[None, :])
                        + 0.2 * phase[:, None] * index[None, :])
    tx = np.broadcast_to(0.002 * index, x.shape).copy()
    ty = np.broadcast_to(0.001 * np.sin(index), x.shape).copy()
    blocked_z = np.full(count, np.nan)
    blocked_key = np.full(count, "", dtype=object)
    if stops and count:
        blocked_z[0] = z[0] + 0.625 * (z[-1] - z[0])
        blocked_key[0] = "test_aperture" if name == "incident" else "column_wall"
    return SimpleNamespace(
        name=name, z=z, x=x, y=y, tx=tx, ty=ty,
        blocked_z=blocked_z, blocked_key=blocked_key,
        ray_weight=np.ones(count) / max(count, 1),
        colour=(0.3, 0.8, 0.95) if name == "incident" else (0.3, 0.9, 0.4),
        interaction_kind="incident" if name == "incident" else "transmitted",
    )


def _result(scale=1.0, *, lens_z=3.0, bore_mm=2.0, aperture_mm=0.15,
            detector_z=18.0, remove_lens=False, extra_branch=False,
            medium=False, count=9, end_z=20.0, stops=True):
    parts = [
        _part("test_lens", "Test Lens", lens_z),
        _part("test_aperture", "Test Aperture", 7.0),
        _part("sample", "Sample", 10.0),
        _part("camera", "Camera", detector_z, profile="camera_sensor_plane"),
    ]
    if remove_lens:
        parts = [part for part in parts if part.key != "test_lens"]
    assembly = _Assembly(
        selected_module_paths=("synthetic-column",), parts=tuple(parts),
        vacuum_bore_segments=(SimpleNamespace(
            key="synthetic-bore", start_z_mm=0.0, end_z_mm=20.0,
            inner_diameter_mm=bore_mm,
        ),),
    )
    incident = _branch("incident", np.linspace(0, 10, 11), scale, count=count, stops=stops)
    branches = {"000": _branch("000", np.linspace(10, end_z, 9), scale,
                                count=count, stops=stops)}
    if extra_branch:
        extra = _branch("plasmon", np.linspace(10, end_z, 9), scale * 1.6,
                        count=count, stops=stops)
        extra.interaction_kind = "plasmon"
        extra.colour = (1.0, 0.65, 0.1)
        branches["plasmon"] = extra
    return SimpleNamespace(
        model_signature=f"synthetic-{scale}", assembly=assembly,
        state_snapshot=SimpleNamespace(
            lenses=(SimpleNamespace(key="test_lens", z_mm=lens_z, percent=50.0 * scale),),
            sample=SimpleNamespace(z_mm=10.0, inserted=True, specimen_mode="virtual"),
            recording_planes=(SimpleNamespace(
                key="camera", z_mm=detector_z, inserted=True, readout_enabled=True,
                outer_width_mm=1.0, inner_diameter_mm=0.0,
                centre_offset_x_mm=0.02, centre_offset_y_mm=-0.03,
            ),),
        ),
        simulation=SimpleNamespace(
            incident=incident, branches=branches,
            metrics={"optical_tuning": True, "tuning_quality": "Medium" if medium else "Preview"},
            gun_waist=None, c2c3_crossover=None, corrector_crossovers=(),
        ),
        lens_crossovers=({"z_mm": 4.0 + scale * 0.2, "rms_radius_mm": 0.001,
                          "name": "Synthetic crossover"},),
        aperture_stops=({"key": "test_aperture", "enabled": True,
                         "installed": True, "z_mm": 7.0, "diameter_mm": aperture_mm,
                         "offset_x_mm": 0.01, "offset_y_mm": -0.02},),
    )


@pytest.fixture
def make_workspace(qtbot, monkeypatch):
    def make():
        view = VisualizationWorkspace()
        qtbot.addWidget(view)
        view.resize(1400, 900)
        # The scene test does not require an independent physical interaction
        # budget calculation when moving its selected-plane cursor.
        monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
        return view
    return make


def _marker(view, name):
    return next(item for item in view.component_marker_items
                if getattr(item, "label", None) is not None
                and item.label.toPlainText() == name)


def _set_user_view(view):
    view.plot.setRange(xRange=(1.0, 19.0), yRange=(-0.7, 0.9),
                       padding=0.0, disableAutoRange=True)
    return np.array(view.plot.viewRange())


def _ray_data(view):
    arrays = [item.getData() for item, _payload in view._ray_bundle_records]
    # Group creation order is not an observable. Compare each exact grouped
    # polyline independent of dictionary insertion order.
    return sorted(arrays, key=lambda pair: (
        len(pair[0]), tuple(np.nan_to_num(pair[0], nan=-1.0)),
        tuple(np.nan_to_num(pair[1], nan=-1.0)),
    ))


def _assert_same_dynamic_data(actual, fresh):
    current, expected = _ray_data(actual), _ray_data(fresh)
    assert len(current) == len(expected)
    for pair, expected_pair in zip(current, expected):
        for values, expected_values in zip(pair, expected_pair):
            np.testing.assert_allclose(values, expected_values, rtol=0, atol=1e-12, equal_nan=True)
    actual_stops = {group: records for _item, group, records in actual._stop_projection_records}
    expected_stops = {group: records for _item, group, records in fresh._stop_projection_records}
    assert actual_stops == expected_stops
    assert [item.value() for item in actual.crossover_marker_items if isinstance(item, pg.InfiniteLine)] == (
        [item.value() for item in fresh.crossover_marker_items if isinstance(item, pg.InfiniteLine)]
    )


def test_same_geometry_keeps_static_items_cursor_and_user_view(make_workspace, monkeypatch):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    view.jump_to_ray_position(8.0, activate_tab=False)
    before_range = _set_user_view(view)
    lens = _marker(view, "Test Lens")
    walls, sample = tuple(view.column_wall_items), tuple(view.sample_marker_items)
    cursor = view.axial_cursor_item
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("A new ray result cleared the complete scene"))
    view.display_result(_result(1.6), "Preview")
    assert _marker(view, "Test Lens") is lens
    assert tuple(view.column_wall_items) == walls
    assert tuple(view.sample_marker_items) == sample
    assert view.axial_cursor_item is cursor
    assert cursor.value() == 8.0
    np.testing.assert_allclose(view.plot.viewRange(), before_range, rtol=0, atol=1e-12)


@pytest.mark.parametrize("change", ["lens", "bore", "aperture", "detector"])
def test_geometry_edit_invalidates_only_affected_layer(make_workspace, change):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    view.jump_to_ray_position(8.0, activate_tab=False)
    before_range = _set_user_view(view)
    lens, walls = _marker(view, "Test Lens"), tuple(view.column_wall_items)
    sample, cursor = tuple(view.sample_marker_items), view.axial_cursor_item
    aperture = tuple(view.aperture_marker_items)
    detectors = tuple(view.recording_surface_range_items)
    options = {"lens": {"lens_z": 4.0}, "bore": {"bore_mm": 3.0},
               "aperture": {"aperture_mm": 0.3}, "detector": {"detector_z": 17.0}}
    changed = _result(1.4, **options[change])
    view.display_result(changed, "Preview")
    if change != "lens":
        assert _marker(view, "Test Lens") is lens
    else:
        assert _marker(view, "Test Lens").value() == 4.0
    if change != "bore":
        assert tuple(view.column_wall_items) == walls
    else:
        assert max(view.column_wall_items[0].getData()[1]) == 1.5
    if change != "aperture":
        assert tuple(view.aperture_marker_items) == aperture
    else:
        assert view._aperture_span_records[0][-1] == 0.15
    if change != "detector":
        assert tuple(view.recording_surface_range_items) == detectors
    else:
        assert np.all(view.recording_surface_range_items[0].getData()[0] == 17.0)
    assert tuple(view.sample_marker_items) == sample
    assert view.axial_cursor_item is cursor
    np.testing.assert_allclose(view.plot.viewRange(), before_range, rtol=0, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 37.25, 90.0])
@pytest.mark.parametrize("medium", [False, True])
def test_new_rays_stops_and_crossovers_match_fresh_render(make_workspace, angle, medium):
    view, fresh = make_workspace(), make_workspace()
    view.display_result(_result(), "Preview")
    view._set_projection_angle(angle)
    fresh._set_projection_angle(angle)
    changed = _result(1.8, extra_branch=True, medium=medium, count=13)
    before = [branch.x.copy() for branch in (changed.simulation.incident, *changed.simulation.branches.values())]
    view.display_result(changed, "Medium" if medium else "Preview")
    fresh.display_result(changed, "Medium" if medium else "Preview")
    _assert_same_dynamic_data(view, fresh)
    assert len(view._tuning_envelopes) == len(fresh._tuning_envelopes)
    for (lower, upper, _), (reference_lower, reference_upper, _) in zip(view._tuning_envelopes, fresh._tuning_envelopes):
        np.testing.assert_allclose(lower.getData()[1], reference_lower.getData()[1], equal_nan=True)
        np.testing.assert_allclose(upper.getData()[1], reference_upper.getData()[1], equal_nan=True)
    for branch, snapshot in zip((changed.simulation.incident, *changed.simulation.branches.values()), before):
        np.testing.assert_array_equal(branch.x, snapshot)


def test_republishing_mutated_arrays_updates_without_discarding_static_scene(make_workspace):
    view, fresh = make_workspace(), make_workspace()
    result = _result()
    view.display_result(result, "Preview")
    lens = _marker(view, "Test Lens")
    result.simulation.incident.x *= 1.7
    result.simulation.branches["000"].y += 3e-5
    result.simulation.branches["000"].blocked_z[0] = 12.5
    view.display_result(result, "Preview")
    fresh.display_result(result, "Preview")
    assert _marker(view, "Test Lens") is lens
    _assert_same_dynamic_data(view, fresh)


def test_removed_component_and_dynamic_groups_leave_no_duplicate_graphics(make_workspace):
    view, fresh = make_workspace(), make_workspace()
    first = _result()
    view.display_result(first, "Preview")
    removed_marker = _marker(view, "Test Lens")
    old_arrays = [weakref.ref(first.simulation.incident.x)]
    del first
    for index in range(12):
        result = _result(1 + index * 0.05, remove_lens=True,
                         extra_branch=index % 2 == 0, medium=index % 3 == 0,
                         stops=index % 2 == 0)
        view.display_result(result, "Medium" if index % 3 == 0 else "Preview")
        old_arrays.append(weakref.ref(result.simulation.incident.x))
    fresh.display_result(result, "Preview")
    assert removed_marker not in view.plot.plotItem.items
    assert all(getattr(item, "label", None) is None or item.label.toPlainText() != "Test Lens"
               for item in view.component_marker_items)
    assert len(view.plot.plotItem.items) == len(fresh.plot.plotItem.items)
    assert len(view.plot.plotItem.legend.items) == len(fresh.plot.plotItem.legend.items)
    assert len(view._ray_label_items) == len(fresh._ray_label_items)
    _assert_same_dynamic_data(view, fresh)
    gc.collect()
    assert all(reference() is None for reference in old_arrays[:-1])


def test_changed_axial_extent_clamps_cursor_without_resetting_user_range(make_workspace):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    view.jump_to_ray_position(19.0, activate_tab=False)
    cursor = view.axial_cursor_item
    before = _set_user_view(view)
    view.display_result(_result(end_z=16.0), "Preview")
    assert view.axial_cursor_item is cursor
    assert cursor.value() == 16.0
    assert view._selected_z_mm == 16.0
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=0, atol=1e-12)


def test_rotated_aperture_and_detector_keep_exact_spans_after_republication(make_workspace):
    view, fresh = make_workspace(), make_workspace()
    view.display_result(_result(), "Preview")
    aperture = tuple(view.aperture_marker_items)
    detector = tuple(view.recording_surface_range_items)
    view._set_projection_angle(90.0)
    changed = _result(1.5)
    view.display_result(changed, "Preview")
    fresh._set_projection_angle(90.0)
    fresh.display_result(changed, "Preview")
    _set_user_view(view)
    _set_user_view(fresh)
    assert tuple(view.aperture_marker_items) == aperture
    assert tuple(view.recording_surface_range_items) == detector
    assert view._aperture_span_records[0][-2] == pytest.approx(-0.02)
    assert "Allowed Y opening" in aperture[0].toolTip()
    for actual, expected in zip(view._aperture_span_records, fresh._aperture_span_records):
        np.testing.assert_allclose(actual[0].span, expected[0].span, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[1].span, expected[1].span, rtol=0, atol=1e-12)
    np.testing.assert_allclose(detector[0].getData()[1], fresh.recording_surface_range_items[0].getData()[1],
                               rtol=0, atol=1e-12)
    assert detector[0].active_intervals_mm == fresh.recording_surface_range_items[0].active_intervals_mm


def test_sample_insertion_updates_its_layer_without_rebuilding_lens_or_walls(make_workspace):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    lens, walls = _marker(view, "Test Lens"), tuple(view.column_wall_items)
    changed = _result(1.5)
    changed.state_snapshot.sample.inserted = False
    view.display_result(changed, "Preview")
    assert _marker(view, "Test Lens") is lens
    assert tuple(view.column_wall_items) == walls
    assert "SAMPLE RETRACTED" in view.sample_marker_items[0].label.toPlainText()


@pytest.mark.parametrize("sample_context", ["zero_z", "no_assembly", "no_state"])
def test_sample_layer_handles_zero_position_and_unknown_context(make_workspace, sample_context):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    changed = _result(1.4)
    expected_z = 10.0
    if sample_context == "zero_z":
        sample = changed.assembly.part("sample")
        sample.start_z_mm, sample.center_z_mm, sample.end_z_mm = -0.2, 0.0, 0.2
        changed.state_snapshot.sample.z_mm = 0.0
        expected_z = 0.0
    elif sample_context == "no_assembly":
        changed.assembly = None
    else:
        changed.state_snapshot = None
    before = _set_user_view(view)
    view.display_result(changed, "Preview")
    assert view.sample_marker_items[0].value() == expected_z
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=0, atol=1e-12)


def test_result_arriving_before_rotation_timer_uses_latest_component_projection(make_workspace):
    view, fresh = make_workspace(), make_workspace()
    view.display_result(_result(), "Preview")
    detector = view.recording_surface_range_items[0]
    aperture = view.aperture_marker_items[0]
    old_detector_y = detector.getData()[1].copy()
    # GUI handlers can receive a completed background result before the next
    # coalesced rotation refresh. New publication cancels that pending timer.
    view._set_projection_angle(90.0, defer_redraw=True)
    assert view._projection_redraw_timer.isActive()
    np.testing.assert_array_equal(detector.getData()[1], old_detector_y)
    latest = _result(1.4)
    view.display_result(latest, "Preview")
    fresh._set_projection_angle(90.0)
    fresh.display_result(latest, "Preview")
    assert view.recording_surface_range_items[0] is detector
    assert view.aperture_marker_items[0] is aperture
    assert not view._projection_redraw_timer.isActive()
    assert not view._projection_finalize_timer.isActive()
    np.testing.assert_allclose(detector.getData()[1], fresh.recording_surface_range_items[0].getData()[1],
                               rtol=0, atol=1e-12)
    assert detector.active_intervals_mm == fresh.recording_surface_range_items[0].active_intervals_mm
    assert "Allowed Y opening" in aperture.toolTip()
    assert aperture.toolTip() == fresh.aperture_marker_items[0].toolTip()
    _assert_same_dynamic_data(view, fresh)


def test_new_result_preserves_detector_focus_after_an_earlier_cursor_selection(
    make_workspace, qtbot, monkeypatch,
):
    view = make_workspace()
    transverse = view.transverse_beam
    # The behavior under test is the selected detector/plane, not the detector
    # PSF numerical calculation (the synthetic state has no optical solver).
    monkeypatch.setattr(transverse, "_add_point_spread_response", lambda: None)
    view.show()
    qtbot.waitUntil(transverse.isVisible)
    first = _result()
    view.display_result(first, "Preview")
    view.jump_to_ray_position(8.0, activate_tab=False)
    view.focus_component(first.assembly.part("camera"))
    assert transverse._focused_component_key == "camera"
    assert transverse._plane_z_mm == 18.0
    latest = _result(1.4, detector_z=17.0)
    view.display_result(latest, "Preview")
    assert view._selected_z_mm == 8.0
    assert view.axial_cursor_item.value() == 8.0
    assert view._transverse_focus_request[0] == "component"
    assert transverse._result is latest
    assert transverse._focused_component_key == "camera"
    assert transverse._plane_z_mm == 17.0


def test_failed_static_layer_replacement_releases_partial_items_and_keeps_prior_layer(make_workspace):
    view = make_workspace()
    layers = StaticRayLayers()
    old = pg.InfiniteLine(pos=3.0)
    def build_old():
        view.plot.addItem(old)
        view.component_marker_items.append(old)
    layers.begin(view)
    layers.update(view, ("test",), (3.0,), build_old)
    layers.finish(view)
    previous = layers.layers[("test",)]
    previous_items = set(view.plot.plotItem.items)
    partial = pg.InfiniteLine(pos=4.0)
    def fail_replacement():
        view.plot.addItem(partial)
        view.component_marker_items.append(partial)
        raise RuntimeError("Synthetic display failure")
    layers.begin(view)
    with pytest.raises(RuntimeError, match="Synthetic display failure"):
        layers.update(view, ("test",), (4.0,), fail_replacement)
    assert layers.layers[("test",)] is previous
    assert set(view.plot.plotItem.items) == previous_items
    assert partial not in view.component_marker_items
    # A subsequent publication can reuse the complete old layer immediately.
    layers.begin(view)
    layers.update(view, ("test",), (3.0,), lambda: pytest.fail("Complete prior layer was lost"))
    layers.finish(view)
    assert view.component_marker_items == [old]


def test_unavailable_crossover_rms_reuses_unchanged_layer(make_workspace):
    view = make_workspace()
    first = _result()
    first.lens_crossovers = ({"z_mm": 4.2, "name": "Synthetic crossover"},)
    view.display_result(first, "Preview")
    markers = tuple(view.crossover_marker_items)
    assert markers
    latest = _result(1.4)
    latest.lens_crossovers = ({"z_mm": 4.2, "rms_radius_mm": float("nan"),
                               "name": "Synthetic crossover"},)
    view.display_result(latest, "Preview")
    assert tuple(view.crossover_marker_items) == markers
    assert "Unavailable" in markers[0].toolTip()


def test_result_during_cursor_drag_preserves_current_position(make_workspace):
    view = make_workspace()
    view.display_result(_result(), "Preview")
    view.jump_to_ray_position(8.0, activate_tab=False)
    cursor = view.axial_cursor_item
    # Position changes arrive during dragging; the last committed selection is
    # only updated on release. A new result must not reset that in-flight move.
    cursor.setValue(12.0)
    assert view._selected_z_mm == 8.0
    assert cursor.value() == 12.0
    view.display_result(_result(1.4), "Preview")
    assert view.axial_cursor_item is cursor
    assert cursor.value() == 12.0
    assert view._selected_z_mm == 12.0
    assert view._transverse_focus_request == ("z", 12.0)
