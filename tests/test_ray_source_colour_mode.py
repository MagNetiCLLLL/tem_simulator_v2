"""Display-only source tracking; all trajectories are synthetic cached arrays."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QColor

from temsim.gui.visualization import VisualizationWorkspace
from temsim.physics.ray_identity import source_identity, select_identity
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def _branch(name, z, count=3):
    x = np.tile(np.linspace(-1e-4, 1e-4, count), (2, 1))
    y = np.tile(np.linspace(1e-4, -2e-4, count), (2, 1))
    return SimpleNamespace(
        name=name, z=np.asarray(z, dtype=float), x=x, y=y,
        tx=np.zeros_like(x), ty=np.zeros_like(y),
        alive=np.ones(count, dtype=bool), blocked_z=np.full(count, np.nan),
        blocked_key=[""] * count, ray_weight=np.full(count, 1 / count),
        weight=1.0, colour=(0.3, 0.9, 0.4),
        interaction_kind="incident" if name == "incident" else "sample_region_elastic",
    )


def _result():
    incident = _branch("incident", (0, 10))
    incident.source_ray_id, incident.source_azimuth_rad = source_identity(incident)
    reference = _branch("000", (10, 20))
    simulation = SimpleNamespace(
        incident=incident, branches={"000": reference}, metrics={},
        gun_waist=None, c2c3_crossover=None, corrector_crossovers=(),
    )
    return SimpleNamespace(
        simulation=simulation, signatures={"sample_downstream": "current"},
        sample_region=None, specimen_exit=None, lens_crossovers=(), aperture_stops=(),
        assembly=SimpleNamespace(parts=(), vacuum_bore_segments=()),
    )


def _checkpoint(branches, signature="current"):
    return GeometricSpecimenExit(
        tuple(branches),
        {"tracked_downstream_source_probability": float(bool(branches)),
         "inelastic_absorbed_source_probability": 0.0},
        dependency_signature=signature,
    )


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_sync_ray_static_layers", lambda _result: None)
    monkeypatch.setattr(widget, "_update_interaction_detail", lambda: None)
    return widget


def test_source_hue_survives_reordered_scattering_and_is_bounded(view):
    result = _result()
    incident = result.simulation.incident
    scattered = _branch("scattered", (10, 20), 2)
    scattered.source_ray_id, scattered.source_azimuth_rad = select_identity(
        incident.source_ray_id, incident.source_azimuth_rad, np.asarray((2, 0))
    )
    groups = view._source_colour_groups(result.simulation, (incident, scattered))
    lookup = {
        (id(branch), int(index)): key[1]
        for key, segments in groups.items() for branch, indices in segments for index in indices
    }
    assert lookup[(id(incident), 2)] == lookup[(id(scattered), 0)]
    assert lookup[(id(incident), 0)] == lookup[(id(scattered), 1)]
    for index, angle in enumerate(incident.source_azimuth_rad):
        expected = (QColor.fromHsvF(angle / (2*np.pi), .88, 1)
                    if np.isfinite(angle) else QColor("#94a3b8"))
        assert lookup[(id(incident), index)] == (expected.red(), expected.green(), expected.blue())
    large = _branch("incident", (0, 10), 500)
    large.source_ray_id, large.source_azimuth_rad = source_identity(large)
    groups = view._source_colour_groups(SimpleNamespace(incident=large), (large,))
    assert sum(len(indices) for segments in groups.values() for _b, indices in segments) == view.MAX_DISPLAY_RAYS
    assert len(groups) <= view.MAX_DISPLAY_RAYS


def test_colour_switch_preserves_ranges_arrays_and_skips_calculation(view, monkeypatch):
    result = _result()
    view._last_result = result
    view._last_quality = "High accuracy"
    view._draw_ray_diagram(result, "High accuracy")
    originals = {id(branch): branch.x.copy() for branch in view._display_ray_bundles()}
    view.plot.setRange(xRange=(3, 17), yRange=(-.1, .15), padding=0, disableAutoRange=True)
    ranges = np.asarray(view.plot.viewRange())
    monkeypatch.setattr(view, "_draw_ray_diagram", lambda *_a, **_kw: pytest.fail("Style change rebuilt the scene"))
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: pytest.fail("Style change recomputed diagnostics"))
    for mode in ("interaction", "source", "interaction", "source"):
        view.ray_colour_mode.setCurrentIndex(view.ray_colour_mode.findData(mode))
        np.testing.assert_array_equal(view.plot.viewRange(), ranges)
        assert view._ray_items_by_group
        assert all((key[0] == "source") == (mode == "source") for key in view._ray_items_by_group)
        for branch in view._display_ray_bundles():
            np.testing.assert_array_equal(branch.x, originals[id(branch)])
    assert "fixed emitted-position" in view.hint.text()


def test_unknown_source_id_stays_neutral_even_with_a_finite_stored_angle(view):
    branch = _branch("unknown", (10, 20), 1)
    branch.source_ray_id = np.asarray((-1,))
    branch.source_azimuth_rad = np.asarray((0.3,))
    groups = view._source_colour_groups(_result().simulation, (branch,))
    neutral = QColor("#94a3b8")
    assert tuple(groups) == (("source", (neutral.red(), neutral.green(), neutral.blue())),)


@pytest.mark.parametrize("exit_state", ("detailed", "empty", "stale"))
def test_main_ray_plot_routes_only_current_detailed_exit(view, exit_state):
    result = _result()
    detailed = _branch("finite_elastic", (10, 20), 2)
    detailed.source_ray_id, detailed.source_azimuth_rad = select_identity(
        result.simulation.incident.source_ray_id,
        result.simulation.incident.source_azimuth_rad, np.asarray((2, 0)),
    )
    result.specimen_exit = _checkpoint(
        () if exit_state == "empty" else (detailed,),
        "old" if exit_state == "stale" else "current",
    )
    view._last_result = result
    view._last_quality = "High accuracy"
    view._draw_ray_diagram(result, "High accuracy")
    displayed = {id(branch) for _item, payload in view._ray_bundle_records for branch, _indices in payload}
    expected = {id(result.simulation.incident)}
    if exit_state == "detailed":
        expected.add(id(detailed))
    elif exit_state == "stale":
        expected.add(id(result.simulation.branches["000"]))
    assert displayed == expected
    assert ("Optical reference" if exit_state == "stale" else "Specimen exit") in view.heading.text()
    assert tuple(result.simulation.branches) == ("000",)
    assert view._simulation_x_limits() == (0.0, 20.0)


@pytest.mark.parametrize("simulation_state", ("missing", "none", "no_incident"))
def test_metadata_only_region_publication_does_not_draw_rays(view, monkeypatch, simulation_state):
    shared = SimpleNamespace(signatures={}, calculated_products=frozenset(), reused_products=frozenset())
    if simulation_state != "missing":
        shared.simulation = None if simulation_state == "none" else SimpleNamespace()
    view._last_result = view._high_accuracy_result = shared
    monkeypatch.setattr(view.sample_interactions_3d, "set_sample_region_result", lambda _r: None)
    monkeypatch.setattr(view, "_redraw_last_result", lambda: pytest.fail("Metadata-only publication drew rays"))
    checkpoint = _checkpoint(())
    region = SimpleNamespace(metrics={"sample_downstream_signature": "current"}, specimen_exit=checkpoint)
    view._set_sample_region_result(region)
    view.ray_colour_mode.setCurrentIndex(view.ray_colour_mode.findData("interaction"))
    assert view._display_ray_bundles() == ()
    assert shared.sample_region is region
    assert shared.specimen_exit is checkpoint
    assert view.transverse_beam not in view._pending_ray_panels


@pytest.mark.parametrize("initially_visible", (False, True))
def test_region_publication_refreshes_transverse_without_changing_plane_or_scale(
    view, monkeypatch, initially_visible,
):
    result = _result()
    view._last_result = view._high_accuracy_result = result
    view._last_quality = "High accuracy"
    view._selected_z_mm = 12.0
    view._transverse_focus_request = ("z", 12.0)
    view._draw_ray_diagram(result, "High accuracy")
    transverse = view.transverse_beam
    transverse.display_result(result, focus=("z", 12.0))
    transverse._apply_centered_view_ranges(.2, .3)
    initial_ids = transverse._display_source_ids.copy()
    ranges = np.asarray(transverse.plot.viewRange())
    ray_ranges = np.asarray(view.plot.viewRange())
    visible = [initially_visible]
    for panel in (view.physical_layout, view.magnetic_field):
        monkeypatch.setattr(panel, "isVisible", lambda: False)
    monkeypatch.setattr(transverse, "isVisible", lambda: visible[0])
    monkeypatch.setattr(view.sample_interactions_3d, "set_sample_region_result", lambda _r: None)
    monkeypatch.setattr(view, "_update_sample_region_control_availability", lambda: None)
    detailed = _branch("finite_elastic", (10, 20), 2)
    detailed.source_ray_id, detailed.source_azimuth_rad = select_identity(
        result.simulation.incident.source_ray_id,
        result.simulation.incident.source_azimuth_rad, np.asarray((2, 0)),
    )
    region = SimpleNamespace(
        metrics={"sample_downstream_signature": "current"}, specimen_exit=_checkpoint((detailed,)),
    )
    view._set_sample_region_result(region)
    if not initially_visible:
        assert view._pending_ray_panels[transverse] is result
        np.testing.assert_array_equal(transverse._display_source_ids, initial_ids)
        visible[0] = True
        view._refresh_visible_ray_panels()
    assert transverse not in view._pending_ray_panels
    np.testing.assert_array_equal(transverse._display_source_ids, detailed.source_ray_id)
    assert transverse._plane_z_mm == view._selected_z_mm == 12.0
    assert view._transverse_focus_request == ("z", 12.0)
    np.testing.assert_allclose(transverse.plot.viewRange(), ranges, rtol=0, atol=1e-12)
    np.testing.assert_allclose(view.plot.viewRange(), ray_ranges, rtol=0, atol=1e-12)
