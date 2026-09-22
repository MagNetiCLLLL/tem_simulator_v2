"""Exact navigation metadata reuse; synthetic completed rays, no solver runs."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def branch(name, start, stop, *, slope=.1):
    z = np.linspace(start, stop, 5)
    x = np.zeros((5, 3))
    tx = np.full_like(x, slope)
    tx[-1, 0] = 9.0  # Excluded by the physical stop, not by view sampling.
    return SimpleNamespace(
        name=name, z=z, x=x, y=x.copy(), tx=tx, ty=np.full_like(x, .3),
        blocked_z=np.array([z[-2], np.nan, np.nan]), weight=1.,
    )


def detailed_exit(*branches):
    return GeometricSpecimenExit(
        tuple(branches),
        {"tracked_downstream_source_probability": float(bool(branches)),
         "inelastic_absorbed_source_probability": float(not branches)},
        dependency_signature="executed",
    )


@pytest.fixture
def view(qtbot):
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    incident = branch("incident", 0., 10.)
    reference = branch("reference", 10., 20., slope=.2)
    widget._last_result = SimpleNamespace(
        simulation=SimpleNamespace(incident=incident, branches={"reference": reference}, metrics={}),
        signatures={"sample_downstream": "executed"}, specimen_exit=None,
    )
    # Fixed view geometry isolates callbacks from unrelated Qt range signals.
    bounds = [[0., 25.], [-1., 1.]]
    widget.plot = SimpleNamespace(getViewBox=lambda: SimpleNamespace(
        viewRange=lambda: bounds, width=lambda: 1000., height=lambda: 500.))
    return widget, bounds


def test_navigation_limits_reuse_arrays_and_preserve_empty_exit_reference(view, monkeypatch):
    widget, _ = view
    widget._last_result.specimen_exit = detailed_exit()
    assert widget._simulation_x_limits() == (0., 20.)
    with monkeypatch.context() as patch:
        patch.setattr(np, "nanmin", lambda *_a, **_k: pytest.fail("Unchanged Z was reduced again"))
        patch.setattr(np, "nanmax", lambda *_a, **_k: pytest.fail("Unchanged Z was reduced again"))
        assert widget._simulation_x_limits() == (0., 20.)
    widget._last_result.simulation.branches["reference"].z = np.linspace(10., 30., 5)
    assert widget._simulation_x_limits() == (0., 30.)


def test_detailed_exit_replacement_updates_bounds_and_visible_slopes(view):
    widget, _ = view
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    assert widget._simulation_x_limits() == (0., 20.)
    child = branch("material", 10., 25., slope=.8)
    widget._last_result.specimen_exit = detailed_exit(child)
    assert widget._maximum_visible_projection_slope() == pytest.approx(.8)
    assert widget._simulation_x_limits() == (0., 25.)
    widget._last_result.specimen_exit = detailed_exit()
    assert widget._maximum_visible_projection_slope() == pytest.approx(.1)
    assert widget._simulation_x_limits() == (0., 20.)


def test_same_axial_window_skips_reduction_when_only_transverse_range_changes(view, monkeypatch):
    widget, bounds = view
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    bounds[1] = [-20., 20.]
    with monkeypatch.context() as patch:
        patch.setattr(np, "max", lambda *_a, **_k: pytest.fail("Same axial window was reduced again"))
        assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    bounds[0] = [0., 9.]
    assert widget._maximum_visible_projection_slope() == pytest.approx(.1)
    widget._projection_angle_deg = 90.
    assert widget._maximum_visible_projection_slope() == pytest.approx(.3)
    widget._last_result.simulation.incident.ty = np.full((5, 3), .7)
    assert widget._maximum_visible_projection_slope() == pytest.approx(.7)


def test_changed_stop_and_sampling_budget_invalidate_visible_slope(view):
    widget, bounds = view
    bounds[0] = [0., 10.]
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    incident = widget._last_result.simulation.incident
    incident.blocked_z = np.full(3, np.nan)
    assert widget._maximum_visible_projection_slope() == pytest.approx(9.)
    incident.tx = np.tile([.1, 5., .2], (5, 1))
    assert widget._maximum_visible_projection_slope() == pytest.approx(5.)
    widget.MAX_RANGE_SAMPLE_RAYS = 2
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)


def test_navigation_caches_retain_only_latest_window_and_respect_zero_budget(view):
    widget, bounds = view
    widget._maximum_visible_projection_slope()
    count = widget.ray_display_cache_info()["entries"]
    for right in np.linspace(1., 24., 40):
        bounds[0] = [0., float(right)]
        widget._maximum_visible_projection_slope()
    assert widget.ray_display_cache_info()["entries"] == count
    widget.set_ray_display_cache_limit_bytes(0)
    assert widget._simulation_x_limits() == (0., 20.)
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    assert widget.ray_display_cache_info()["entries"] == 0


def test_scale_notice_skips_same_text_but_updates_scale_and_projection(view):
    widget, bounds = view
    texts = []
    widget.hint = SimpleNamespace(text=lambda: texts[-1] if texts else "", setText=texts.append)
    widget._update_scale_notice()
    assert len(texts) == 1
    assert "11.3°" in texts[-1]  # atan(.2), not .2 radians shown as degrees.
    widget._update_scale_notice()
    bounds[1] = [9., 11.]  # Pan only: the scale notice is unchanged.
    widget._update_scale_notice()
    assert len(texts) == 1
    bounds[1] = [-.01, .01]
    widget._update_scale_notice()
    assert len(texts) == 2
    widget._projection_angle_deg = 90.
    widget._update_scale_notice()
    assert len(texts) == 3
    assert "16.7°" in texts[-1]


def test_same_object_republication_invalidates_in_place_navigation_data(view, monkeypatch):
    widget, _ = view
    result = widget._last_result
    assert widget._simulation_x_limits() == (0., 20.)
    assert widget._maximum_visible_projection_slope() == pytest.approx(.2)
    result.simulation.branches["reference"].z[-1] = 24.
    result.simulation.branches["reference"].tx[:, 1:] = .9

    class PublicationReached(Exception):
        pass

    def inspect_publication(*_args, **_kwargs):
        assert widget.ray_display_cache_info()["entries"] == 0
        assert widget._simulation_x_limits() == (0., 24.)
        assert widget._maximum_visible_projection_slope() == pytest.approx(.9)
        raise PublicationReached

    # Stop at the public publication boundary before unrelated page rendering.
    monkeypatch.setattr(widget.result_readout, "publish", inspect_publication)
    with pytest.raises(PublicationReached):
        widget.display_result(result, "High accuracy")
