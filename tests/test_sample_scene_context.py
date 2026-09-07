"""Offline sample-renderer contracts; no atom generation or OpenGL execution."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui import sample_panel
from temsim.gui.sample_panel import SampleSceneView
from temsim.gui.sample_scene_labels import sample_scene_labels
from temsim.optics.column import default_state
from temsim.specimen.geometry import (
    build_sample_geometry_snapshot,
    quaternion_from_euler_xyz_deg,
    quaternion_from_zone_axes,
    quaternion_to_matrix,
    set_sample_orientation,
)


def _readonly(values, dtype=float):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


@pytest.fixture
def local_snapshot():
    sample = default_state().sample
    sample.specimen_mode = "atomic"
    sample.cif_path = ""
    sample.envelope_shape = "disk"
    sample.size_x_nm = sample.size_y_nm = 3_000_000.0
    sample.thickness_nm = 10.0
    sample.centre_x_nm, sample.centre_y_nm = 125.0, -240.0
    sample.zone_axis_uvw = (1, 1, 0)
    sample.in_plane_axis_uvw = (1, -1, 0)
    set_sample_orientation(sample, quaternion_from_zone_axes(
        np.eye(3), sample.zone_axis_uvw, sample.in_plane_axis_uvw,
    ))
    bounds = (127.0, 139.0, -248.0, -232.0)
    snapshot = build_sample_geometry_snapshot(
        sample, load_atoms=False, calculation_roi_bounds_nm_override=bounds,
    )
    # Positions are already laboratory-oriented and relative to the physical
    # sample centre, as delivered by the real geometry builder. These three
    # synthetic sites stand in for an explicitly capped local atom display.
    snapshot = replace(
        snapshot,
        atom_positions_nm=_readonly([[6.5, -0.3, -4.0], [7.0, 0.0, 0.0], [7.5, 0.4, 4.0]]),
        atomic_numbers=_readonly([14, 14, 14], int),
        atom_bond_pairs=_readonly([[0, 1], [1, 2]], int),
        atom_display_centre_nm=(132.0, -240.0, 0.0),
        atom_display_size_nm=(3.0, 4.0, 10.0),
        atom_display_capped=True,
    )
    return sample, snapshot


@pytest.fixture
def fallback_scene(qtbot, monkeypatch):
    monkeypatch.setattr(sample_panel, "gl", None)
    scene = SampleSceneView()
    qtbot.addWidget(scene)
    scene.resize(700, 700)
    scene.show()
    qtbot.waitUntil(lambda: scene.view.width() > 100)
    assert not scene.opengl_available
    return scene


def _named_curves(scene):
    return {
        item.name(): item for item in scene.view.listDataItems()
        if isinstance(item, pg.PlotDataItem) and item.name()
    }


def test_gl_uniform_colour_is_not_passed_as_a_short_vertex_buffer(monkeypatch):
    """A four-value ndarray makes real GL wireframes largely disappear."""
    from types import SimpleNamespace
    captured = []
    fake_gl = SimpleNamespace(GLLinePlotItem=lambda **kwargs: captured.append(kwargs) or kwargs)
    monkeypatch.setattr(sample_panel, "gl", fake_gl)
    scene = SimpleNamespace(view=SimpleNamespace(addItem=lambda item: None), _items=[])
    positions = np.zeros((200, 3))
    sample_panel.SampleSceneView._add_gl_line(scene, positions, (0.2, 0.7, 0.9, 0.95))
    assert isinstance(captured[-1]["color"], tuple)
    assert captured[-1]["color"] == (0.2, 0.7, 0.9, 0.95)
    per_vertex = np.tile((0.1, 0.2, 0.3, 1.0), (200, 1))
    sample_panel.SampleSceneView._add_gl_line(scene, positions, per_vertex)
    np.testing.assert_array_equal(captured[-1]["color"], per_vertex)
    with pytest.raises(ValueError, match="per vertex"):
        sample_panel.SampleSceneView._add_gl_line(scene, positions, np.ones((2, 4)))


def _xy_bounds(curve):
    return np.array([[min(curve.xData), max(curve.xData)],
                     [min(curve.yData), max(curve.yData)]])


def test_fallback_keeps_full_3mm_outline_with_nm_atom_subset(fallback_scene, local_snapshot):
    _, snapshot = local_snapshot
    fallback_scene.display_snapshot(snapshot)
    curves = _named_curves(fallback_scene)
    assert set(curves) == {"Full sample", "Local region", "Displayed subset"}
    cx, cy, _ = snapshot.centre_nm
    np.testing.assert_allclose(_xy_bounds(curves["Full sample"]),
                               [[cx - 1_500_000, cx + 1_500_000],
                                [cy - 1_500_000, cy + 1_500_000]], rtol=0, atol=1e-8)
    np.testing.assert_allclose(_xy_bounds(curves["Local region"]),
                               [[127, 139], [-248, -232]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(_xy_bounds(curves["Displayed subset"]),
                               [[130.5, 133.5], [-242, -238]], rtol=0, atol=1e-12)
    assert curves["Full sample"].opts["pen"].color() != curves["Local region"].opts["pen"].color()
    assert curves["Displayed subset"].opts["pen"].color() != curves["Local region"].opts["pen"].color()


def test_local_region_and_oriented_atoms_have_correct_offset_lab_positions(fallback_scene, local_snapshot):
    _, snapshot = local_snapshot
    assert snapshot.local_material_bounds_nm == (127.0, 139.0, -248.0, -232.0, -5.0, 5.0)
    fallback_scene.display_snapshot(snapshot)
    scatter, = [item for item in fallback_scene.view.getPlotItem().items
                if isinstance(item, pg.ScatterPlotItem)]
    x, y = scatter.getData()
    expected = snapshot.atom_positions_nm[:, :2] + snapshot.centre_nm[:2]
    np.testing.assert_allclose(np.column_stack((x, y)), expected, rtol=0, atol=1e-12)
    # A [110] lattice tilt must not tilt the physical clipping solid or move
    # the absolute local bounds a second time by the sample offset.
    bounds = _xy_bounds(_named_curves(fallback_scene)["Local region"])
    np.testing.assert_allclose(bounds, [[127, 139], [-248, -232]], rtol=0, atol=1e-12)


def test_fit_actions_choose_full_or_requested_local_not_display_subset(fallback_scene, local_snapshot):
    sample, snapshot = local_snapshot
    sample_before = deepcopy(vars(sample))
    original_positions = snapshot.atom_positions_nm.copy()
    fallback_scene.display_snapshot(snapshot)
    full_range = np.asarray(fallback_scene.view.viewRange())
    assert np.all(np.ptp(full_range, axis=1) >= 3_000_000)
    fallback_scene.fit_local_region()
    local_range = np.asarray(fallback_scene.view.viewRange())
    assert np.all(np.ptp(local_range, axis=1) < 100)
    assert local_range[0, 0] < 127 and local_range[0, 1] > 139
    assert local_range[1, 0] < -248 and local_range[1, 1] > -232
    np.testing.assert_allclose(local_range.mean(axis=1), [133, -240], rtol=0, atol=1e-10)
    fallback_scene.fit_full_sample()
    np.testing.assert_allclose(fallback_scene.view.viewRange(), full_range, rtol=0, atol=1e-8)
    assert fallback_scene._snapshot is snapshot
    assert vars(sample) == sample_before
    np.testing.assert_array_equal(snapshot.atom_positions_nm, original_positions)


def test_redraw_and_draft_orientation_preserve_user_view_ranges(fallback_scene, local_snapshot):
    _, snapshot = local_snapshot
    fallback_scene.display_snapshot(snapshot)
    fallback_scene.fit_local_region()
    fallback_scene.view.setRange(xRange=(128, 136), yRange=(-245, -235), padding=0,
                                 disableAutoRange=True)
    manual_range = np.asarray(fallback_scene.view.viewRange())
    atom_before = snapshot.atom_positions_nm.copy()
    geometry_before = (snapshot.centre_nm, snapshot.size_nm, snapshot.orientation_quaternion_wxyz)
    draft = quaternion_from_euler_xyz_deg((27, -11, 36))
    for orientation in (None, draft, snapshot.orientation_quaternion_wxyz):
        fallback_scene.display_snapshot(snapshot, draft_quaternion=orientation)
        np.testing.assert_allclose(fallback_scene.view.viewRange(), manual_range, rtol=0, atol=1e-10)
        assert fallback_scene.view.getViewBox().state["autoRange"] == [False, False]
        assert _xy_bounds(_named_curves(fallback_scene)["Local region"]) == pytest.approx(
            np.asarray([[127, 139], [-248, -232]])
        )
    np.testing.assert_array_equal(snapshot.atom_positions_nm, atom_before)
    assert (snapshot.centre_nm, snapshot.size_nm, snapshot.orientation_quaternion_wxyz) == geometry_before


def test_uncapped_display_does_not_draw_misleading_subset_outline(fallback_scene, local_snapshot):
    _, snapshot = local_snapshot
    fallback_scene.display_snapshot(replace(snapshot, atom_display_capped=False))
    assert set(_named_curves(fallback_scene)) == {"Full sample", "Local region"}


def test_no_local_material_fits_full_sample_without_regeneration(fallback_scene, local_snapshot):
    _, snapshot = local_snapshot
    empty = replace(snapshot, local_material_bounds_nm=None, atom_display_capped=False,
                    atom_positions_nm=_readonly(np.empty((0, 3))), atomic_numbers=_readonly([], int),
                    atom_bond_pairs=_readonly(np.empty((0, 2)), int))
    fallback_scene.display_snapshot(empty)
    before = np.asarray(fallback_scene.view.viewRange())
    fallback_scene.fit_local_region()
    assert set(_named_curves(fallback_scene)) == {"Full sample"}
    np.testing.assert_allclose(fallback_scene.view.viewRange(), before)
    assert fallback_scene._snapshot is empty


def test_gl_renderer_contract_keeps_finite_outline_in_lab_frame(fallback_scene, local_snapshot, monkeypatch):
    """Capture renderer inputs only; this is not a live OpenGL visual test."""
    _, snapshot = local_snapshot
    lines, atoms = [], []
    monkeypatch.setattr(fallback_scene, "_add_gl_line",
        lambda positions, colour, **_kw: lines.append((np.asarray(positions).copy(), colour)))
    monkeypatch.setattr(fallback_scene, "_add_gl_atoms",
        lambda positions, numbers: atoms.append((np.asarray(positions).copy(), np.asarray(numbers).copy())))
    draft = quaternion_to_matrix(quaternion_from_euler_xyz_deg((18, 42, -30)))
    fallback_scene._display_gl(snapshot, draft)
    full, local, subset = (entry[0] for entry in lines[:3])
    centre = np.asarray(snapshot.centre_nm)
    np.testing.assert_allclose(full.min(axis=0), centre - np.asarray(snapshot.size_nm) / 2, rtol=0, atol=1e-8)
    np.testing.assert_allclose(full.max(axis=0), centre + np.asarray(snapshot.size_nm) / 2, rtol=0, atol=1e-8)
    np.testing.assert_allclose(local.min(axis=0), [127, -248, -5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(local.max(axis=0), [139, -232, 5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(subset.max(axis=0) - subset.min(axis=0), snapshot.atom_display_size_nm)
    expected_atoms = snapshot.atom_positions_nm @ np.asarray(snapshot.orientation_matrix) @ draft.T
    expected_atoms[:, :2] += centre[:2]
    np.testing.assert_allclose(atoms[0][0], expected_atoms, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(atoms[0][1], snapshot.atomic_numbers)


def test_dimension_labels_separate_full_local_and_rendering_cap(local_snapshot):
    _, snapshot = local_snapshot
    full, local, detail = sample_scene_labels(snapshot, completed_region=True)
    assert "Diameter 3 mm" in full and "Thickness 10 nm" in full
    assert "Local calculation region" in local and "12 nm × 16 nm × 10 nm" in local
    assert "3 atoms shown" in detail and "Display subset: 3 nm × 4 nm × 10 nm" in detail
    assert all(len(label) < 140 for label in (full, local, detail))
    assert "Local structure preview" in sample_scene_labels(snapshot, completed_region=False)[1]
