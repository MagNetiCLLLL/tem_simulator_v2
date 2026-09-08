"""Offline retained-scene and hidden-page performance contracts (fake GL)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QVector3D

from temsim.gui import sample_panel
from temsim.optics.column import default_state
from temsim.specimen.geometry import (
    build_sample_geometry_snapshot,
    quaternion_from_euler_xyz_deg,
    quaternion_to_matrix,
)


def _readonly(values, dtype=float):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


@pytest.fixture
def snapshot():
    sample = default_state().sample
    sample.specimen_mode = "atomic"
    sample.cif_path = ""
    sample.size_x_nm = sample.size_y_nm = 10.
    sample.thickness_nm = 4.
    sample.centre_x_nm, sample.centre_y_nm = 3., -2.
    base = build_sample_geometry_snapshot(sample, load_atoms=False, current_probe_nm=(3., -2.))
    return replace(base, atom_positions_nm=_readonly([[-1., 0., 0.], [1., .5, 1.]]),
                   atomic_numbers=_readonly([14, 14], int),
                   atom_bond_pairs=_readonly([[0, 1]], int),
                   atom_display_centre_nm=(3., -2., 0.), atom_display_size_nm=(2., 2., 4.))


@pytest.fixture
def scene(qtbot, monkeypatch):
    monkeypatch.setattr(sample_panel, "gl", None)
    widget = sample_panel.SampleSceneView()
    qtbot.addWidget(widget)
    widget.resize(600, 500)
    widget.show()
    return widget


def test_identical_scene_keeps_objects_and_user_ranges(scene, snapshot):
    scene.display_snapshot(snapshot)
    items = tuple(scene.view.getPlotItem().items)
    scene.view.setRange(xRange=(-7., 8.), yRange=(-6., 9.), padding=0, disableAutoRange=True)
    bounds = np.asarray(scene.view.viewRange())
    scene.display_snapshot(replace(snapshot, warnings=("UI-only warning",)))
    scene.display_snapshot(replace(snapshot, current_probe_nm=(4., -1.)))
    assert tuple(scene.view.getPlotItem().items) == items
    assert scene.model_builds == 1
    assert scene.model_reuses == 2
    np.testing.assert_array_equal(scene.view.viewRange(), bounds)
    scene.fit_full_sample()
    scene.fit_local_region()
    assert scene.model_builds == 1


def test_fallback_draft_moves_cached_atoms_and_bonds_without_rebuilding(scene, snapshot):
    scene.display_snapshot(snapshot)
    atom, bond = scene._atom_item, scene._bond_item
    for angles in ((0., 0., 90.), (15., 30., -40.), (0., 0., 0.)):
        quaternion = quaternion_from_euler_xyz_deg(angles)
        scene.display_snapshot(snapshot, draft_quaternion=quaternion)
        expected = scene._oriented_atoms(snapshot, quaternion_to_matrix(quaternion))
        expected[:, :2] += snapshot.centre_nm[:2]
        np.testing.assert_allclose(np.column_stack(atom.getData()), expected[:, :2], atol=1e-12)
        np.testing.assert_allclose(np.column_stack(bond.getData()),
                                   expected[snapshot.atom_bond_pairs].reshape(-1, 3)[:, :2], atol=1e-12)
        assert scene._atom_item is atom and scene._bond_item is bond
    assert scene.model_builds == 1
    # Display transforms never mutate the frozen source positions.
    np.testing.assert_array_equal(snapshot.atom_positions_nm, [[-1., 0., 0.], [1., .5, 1.]])


def test_changed_geometry_rebuilds_but_empty_array_identity_does_not(scene, snapshot):
    scene.display_snapshot(snapshot)
    old = scene._atom_item
    scene.display_snapshot(replace(snapshot, size_nm=(20., 20., 4.)))
    assert scene._atom_item is not old
    assert scene.model_builds == 2
    scene.display_snapshot(replace(scene._snapshot, cell_vectors_nm=np.empty((0, 3))))
    assert scene.model_builds == 2


def test_gl_draft_uses_item_transform_and_probe_updates_in_place(scene, snapshot, monkeypatch):
    class FakeItem:
        def __init__(self, positions):
            self.positions = np.asarray(positions).copy()
            self.transform = None

        def setTransform(self, matrix):
            self.transform = matrix

        def setData(self, *, pos):
            self.positions = np.asarray(pos)

    scene.opengl_available = True
    scene.view = SimpleNamespace(addItem=lambda _: None, removeItem=lambda _: None, update=lambda: None)
    scene._has_fitted = True

    def add(positions, *_args, **_kwargs):
        item = FakeItem(positions)
        scene._items.append(item)
        return item

    monkeypatch.setattr(scene, "_add_gl_line", add)
    monkeypatch.setattr(scene, "_add_gl_atoms", add)
    first = quaternion_from_euler_xyz_deg((10., 20., 30.))
    scene.display_snapshot(snapshot, draft_quaternion=first)
    items = tuple(scene._items)
    atom, bond, beam, probe = scene._atom_item, scene._bond_item, scene._beam_item, scene._probe_item
    second = quaternion_from_euler_xyz_deg((-20., 15., 60.))
    moved = replace(snapshot, current_probe_nm=(4., -1.))
    scene.display_snapshot(moved, draft_quaternion=second)
    expected = scene._oriented_atoms(snapshot, quaternion_to_matrix(second))
    expected[:, :2] += snapshot.centre_nm[:2]
    transformed = np.asarray([
        tuple(atom.transform.map(QVector3D(*point)).toTuple()) for point in atom.positions
    ])
    np.testing.assert_allclose(transformed, expected, atol=5e-7)
    assert bond.transform is not None
    assert all(item.transform is None for item in items if item is not atom and item is not bond)
    np.testing.assert_allclose(beam.positions[:, :2], [[4., -1.], [4., -1.]])
    np.testing.assert_allclose(probe.positions[:, :2].mean(axis=0), [4., -1.])
    assert tuple(scene._items) == items and scene.model_builds == 1
    # Leave cleanup using the real Qt fallback view path out of the fake GL test.
    scene._items = []


def test_hidden_page_defers_and_coalesces_latest_captured_sample(qtbot, monkeypatch):
    calls = []
    original = sample_panel.build_sample_geometry_snapshot

    def tracked(sample, **kwargs):
        calls.append(sample.size_x_nm)
        return original(sample, **(kwargs | {"load_atoms": False}))

    monkeypatch.setattr(sample_panel, "build_sample_geometry_snapshot", tracked)
    page = sample_panel.SamplePage()
    qtbot.addWidget(page)
    state = default_state()
    page.set_state(state)

    def result(size):
        saved = type(state).from_dict(state.to_dict())
        saved.sample.size_x_nm = saved.sample.size_y_nm = size
        return SimpleNamespace(state_snapshot=saved, wave_imaging=SimpleNamespace(metrics={
            "specimen_wave_window_bounds_nm": (-1., 1., -1., 1.)}))

    first, second = result(10.), result(20.)
    page.display_result(first)
    page.display_result(second)
    assert calls == [] and page._snapshot is None
    assert page._pending_calculation_result is second
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert calls == [20.]
    assert page._snapshot.size_nm[:2] == (20., 20.)
    assert "TEM" in page.scene_status.text()
    assert page._result is second and not page._refresh_pending
    page.hide()
    third = result(30.)
    page.display_result(first)
    page.display_result(third)
    assert calls == [20.] and page._snapshot.size_nm[:2] == (20., 20.)
    page.show()
    qtbot.waitUntil(lambda: page._snapshot.size_nm[0] == 30.)
    assert calls == [20., 30.]
    assert state.sample.size_x_nm == 10.0
    assert page._result is third


def test_hidden_page_first_open_renders_without_a_completed_result(qtbot, monkeypatch):
    page = sample_panel.SamplePage()
    qtbot.addWidget(page)
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = ""
    page.set_state(state)
    assert page._snapshot is None and page._refresh_pending
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert page._snapshot.size_nm[0] == state.sample.size_x_nm
    assert "Structural preview" in page.scene_status.text()
    assert page._result is None


def test_independently_built_real_cif_snapshots_reuse_scene_model(scene, tmp_path):
    from temsim.specimen.atomistic import atomistic_capability
    from temsim.specimen.display_cache import configure_sample_display_cache, sample_display_cache_info
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF display backend unavailable")
    from ase.build import bulk
    from ase.io import write
    path = tmp_path / "retained-scene-si.cif"
    write(path, bulk("Si", "diamond", a=5.43, cubic=True))
    sample = default_state().sample
    sample.specimen_mode = "atomic"
    sample.cif_path = str(path)
    sample.size_x_nm = sample.size_y_nm = sample.thickness_nm = 1.2
    previous_budget = sample_display_cache_info()["budget_bytes"]
    configure_sample_display_cache(budget_bytes=128 * 1024**2)
    try:
        first = build_sample_geometry_snapshot(sample)
        second = build_sample_geometry_snapshot(sample)
        assert first is not second
        assert len(first.atomic_numbers) > 0
        assert first.atom_positions_nm is second.atom_positions_nm
        assert first.atom_bond_pairs is second.atom_bond_pairs
        scene.display_snapshot(first)
        atom_item = scene._atom_item
        scene.display_snapshot(second)
        assert scene.model_builds == 1 and scene.model_reuses == 1
        assert scene._atom_item is atom_item
    finally:
        configure_sample_display_cache(budget_bytes=previous_budget)


def test_visible_page_cif_removal_discards_cached_atoms_but_keeps_geometry(qtbot, monkeypatch, tmp_path):
    from temsim.specimen import geometry
    from temsim.specimen.display_cache import configure_sample_display_cache, sample_display_cache_info
    path = tmp_path / "removable-display.cif"
    path.write_text("synthetic display fixture", encoding="utf-8")
    builds = []

    def fake_builder(*_args, **_kwargs):
        builds.append(1)
        return (_readonly([[0., 0., 0.], [.1, .1, .1]]), _readonly([14, 14], int),
                _readonly([[0, 1]], int), _readonly(np.eye(3)),
                (0., 0., 0.), (1., 1., 1.), False, ())

    monkeypatch.setattr(geometry, "_read_cif_preview_uncached", fake_builder)
    previous = sample_display_cache_info()["budget_bytes"]
    configure_sample_display_cache(budget_bytes=128 * 1024**2)
    page = sample_panel.SamplePage()
    qtbot.addWidget(page)
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    state.sample.size_x_nm = state.sample.size_y_nm = state.sample.thickness_nm = 10.
    try:
        page.set_state(state)
        page.show()
        qtbot.waitUntil(lambda: page._snapshot is not None)
        assert len(page._snapshot.atomic_numbers) == 2
        model_builds = page.scene.model_builds
        page.refresh_snapshot()
        assert len(builds) == 1 and page.scene.model_builds == model_builds
        path.unlink()
        page.refresh_snapshot()
        assert page._snapshot.size_nm == (10., 10., 10.)
        assert not page._snapshot.atomic_numbers.size
        assert page.scene._atom_item is None
        assert "Geometry retained" in page.atom_display_label.text()
        assert "CIF file does not exist" in page.atom_display_label.toolTip()
        assert len(page.scene.view.listDataItems()) >= 2
    finally:
        configure_sample_display_cache(budget_bytes=previous)
