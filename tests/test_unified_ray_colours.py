"""Shared emission colours across cached rays and both transverse plots.

These deliberately bent, reordered display fixtures are not a transport model
or a qualification of lens or specimen physics. No numerical work is dispatched.
"""
import numpy as np
import pytest
from PySide6.QtGui import QColor

from test_beam_analysis_modes import add_detailed_exit
from test_transverse_source_tracking import recorded_result, preset
from temsim.gui.visualization import VisualizationWorkspace


SOURCE_QUANTITIES = ("source", "emission_direction", "emission_angle")


def _rgb(colour):
    return colour.red(), colour.green(), colour.blue()


def _scatter_colours(scatter):
    assert scatter is not None
    output = {}
    for data, brush in zip(scatter.data["data"], scatter.data["brush"]):
        source_id = int(data["source_ray_id"])
        colour = _rgb(brush.color())
        # Multiple scattered descendants of one source keep its launch label.
        assert output.setdefault(source_id, colour) == colour
    return output


def _ray_colours(workspace):
    output = {}
    displayed_branches = set()
    for item, payload in workspace._ray_bundle_records:
        colour = _rgb(item.opts["pen"].color())
        z, _transverse = item.getData()
        assert np.count_nonzero(np.isfinite(z)) >= 2
        for branch, indices in payload:
            displayed_branches.add(id(branch))
            for index in indices:
                source_id = int(branch.source_ray_id[index])
                assert output.setdefault(source_id, colour) == colour
    assert output
    assert displayed_branches == {id(branch) for branch in workspace._display_ray_bundles()}
    return output


def _assert_views_agree(workspace):
    rays = _ray_colours(workspace)
    source = _scatter_colours(workspace.transverse_beam.source_plot.scatter)
    selected = _scatter_colours(workspace.transverse_beam._scatter)
    assert rays == source
    assert selected
    assert all(colour == rays[source_id] for source_id, colour in selected.items())
    return rays


def _result():
    result = recorded_result()
    emission = result.simulation.gun_trace.emission_reference
    phi = np.arctan2(emission["direction"][:, 1], emission["direction"][:, 0])
    theta = np.linspace(.01, .7, len(phi))
    emission["direction"] = np.column_stack((
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta),
    ))
    children = add_detailed_exit(result)
    for branch in (result.simulation.incident, *children):
        branch.z = np.linspace(branch.z[0], branch.z[-1], 3)
        for key in ("x", "y", "tx", "ty"):
            values = getattr(branch, key)
            # A crossover and changing downstream slopes must not recolour a ray.
            setattr(branch, key, np.stack((values[0], -values[0] * 3, values[-1])))
        branch.source_azimuth_rad = np.full(branch.x.shape[1], np.pi)
    for branch in children:
        for key in ("x", "y", "tx", "ty"):
            setattr(branch, key, getattr(branch, key)[:, ::-1].copy())
        for key in ("source_ray_id", "source_azimuth_rad", "ray_weight", "blocked_z"):
            setattr(branch, key, getattr(branch, key)[::-1].copy())
    return result


@pytest.fixture
def workspace(qtbot, monkeypatch):
    from temsim.physics import core, simulation

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changing presentation must not dispatch transport or rebuild the scene")

    monkeypatch.setattr(simulation, "run", forbidden)
    monkeypatch.setattr(core, "propagate", forbidden)
    monkeypatch.setattr(core, "execute_propagation_plan", forbidden)
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    widget.resize(1500, 1000)
    widget.show()
    widget.transverse_beam_toggle.setChecked(True)
    monkeypatch.setattr(widget, "_draw_ray_diagram", forbidden)
    monkeypatch.setattr(widget, "_update_interaction_detail", forbidden)
    qtbot.wait(10)
    return widget


def _publish(workspace, result):
    workspace._last_result = result
    workspace._last_quality = "Cached fixture"
    workspace.transverse_beam.display_result(result, focus=("z", 1.5))
    workspace._sync_ray_curves(result.simulation, workspace._display_ray_bundles())


def _choose_transverse(workspace, quantity):
    transverse = workspace.transverse_beam
    if quantity == "emission_angle":
        preset(transverse, "advanced")
        transverse.colour_mode.setCurrentIndex(transverse.colour_mode.findData(quantity))
    else:
        preset(transverse, quantity)


@pytest.mark.parametrize("quantity", SOURCE_QUANTITIES)
def test_transverse_choice_changes_ray_colours_and_retains_identity_along_bent_paths(workspace, quantity):
    result = _result()
    _publish(workspace, result)
    workspace.plot.setRange(xRange=(.2, 1.8), yRange=(-.02, .03), padding=0, disableAutoRange=True)
    bounds = np.asarray(workspace.plot.viewRange())
    before = [(branch, branch.x.copy(), branch.tx.copy()) for branch in workspace._display_ray_bundles()]
    _choose_transverse(workspace, quantity)
    assert workspace.ray_colour_mode.currentData() == quantity
    assert workspace.transverse_beam.colour_mode.currentData() == quantity
    colours = _assert_views_agree(workspace)
    assert len(set(colours.values())) > 1
    if quantity == "source":
        # Stored branch azimuths deliberately disagree with actual tip positions.
        assert colours[0] == _rgb(QColor.fromHsvF(0., .88, 1.))
        assert colours[0] != _rgb(QColor.fromHsvF(.5, .88, 1.))
    for z in (.25, .75, 1.5, 2.):
        workspace.transverse_beam.focus_z(z)
        assert _assert_views_agree(workspace) == colours
    np.testing.assert_array_equal(workspace.plot.viewRange(), bounds)
    for branch, x, tx in before:
        np.testing.assert_array_equal(branch.x, x)
        np.testing.assert_array_equal(branch.tx, tx)


@pytest.mark.parametrize("quantity", SOURCE_QUANTITIES)
def test_ray_choice_updates_transverse_and_projection_only_rotates_coordinates(workspace, quantity):
    _publish(workspace, _result())
    workspace.ray_colour_mode.setCurrentIndex(workspace.ray_colour_mode.findData(quantity))
    transverse = workspace.transverse_beam
    assert transverse.colour_mode.currentData() == quantity
    assert transverse.analysis.tracking_combo.currentData() == (
        "advanced" if quantity == "emission_angle" else quantity)
    colours = _assert_views_agree(workspace)
    before = transverse.source_plot.scatter.data["x"].copy()
    for angle in (90., 237., 0.):
        workspace._set_projection_angle(angle)
        assert _assert_views_agree(workspace) == colours
        if angle == 90.:
            assert not np.allclose(transverse.source_plot.scatter.data["x"], before)


@pytest.mark.parametrize("quantity", ("emission_direction", "emission_angle"))
def test_missing_launch_directions_stay_grey_in_all_views(workspace, quantity):
    result = _result()
    result.simulation.gun_trace.emission_reference["direction"][0] = np.nan
    _publish(workspace, result)
    _choose_transverse(workspace, quantity)
    colours = _assert_views_agree(workspace)
    assert colours[0] == _rgb(QColor("#94a3b8"))
    assert colours[4] != colours[0]


@pytest.mark.parametrize("quantity", ("emission_direction", "emission_angle"))
def test_ray_choice_syncs_before_hidden_transverse_has_a_result(workspace, quantity, qtbot):
    workspace.transverse_beam_toggle.setChecked(False)
    result = _result()
    workspace._last_result = result
    workspace._last_quality = "Cached fixture"
    assert workspace.transverse_beam._result is None
    workspace.ray_colour_mode.setCurrentIndex(workspace.ray_colour_mode.findData(quantity))
    assert workspace.transverse_beam.colour_mode.currentData() == quantity
    workspace._pending_ray_panels[workspace.transverse_beam] = result
    workspace._transverse_focus_request = ("z", 1.5)
    workspace.transverse_beam_toggle.setChecked(True)
    qtbot.wait(10)
    assert workspace.transverse_beam._result is result
    assert workspace.transverse_beam.colour_mode.currentData() == quantity
    _assert_views_agree(workspace)


def test_colour_only_changes_keep_each_plot_zoom_and_cached_plane(workspace):
    _publish(workspace, _result())
    transverse = workspace.transverse_beam
    preset(transverse, "advanced")
    transverse.plot.setRange(xRange=(-6., 6.), yRange=(-5., 5.), padding=0, disableAutoRange=True)
    transverse.plot.getViewBox().sigRangeChangedManually.emit([True, True])
    transverse.source_plot.plot.setRange(xRange=(-8., 8.), yRange=(-9., 9.), padding=0, disableAutoRange=True)
    workspace.plot.setRange(xRange=(.1, 1.9), yRange=(-.04, .06), padding=0, disableAutoRange=True)
    plots = (workspace.plot, transverse.plot, transverse.source_plot.plot)
    before = [np.asarray(plot.viewRange()) for plot in plots]
    cached = transverse.analysis.plane_data()
    for quantity in ("emission_angle", "emission_direction", "source"):
        transverse.colour_mode.setCurrentIndex(transverse.colour_mode.findData(quantity))
        assert transverse.analysis.plane_data() is cached
        assert workspace.ray_colour_mode.currentData() == quantity
        for plot, bounds in zip(plots, before):
            np.testing.assert_allclose(plot.viewRange(), bounds, rtol=0, atol=1e-12)
