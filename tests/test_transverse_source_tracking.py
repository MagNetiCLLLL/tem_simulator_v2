"""Source/plane display integration using explicit cached particle fixtures.

No gun or column execution: these test data provenance and presentation, not
qualification of a physical transport chain.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox

from test_beam_analysis_modes import make_result, add_detailed_exit, switch, view
from temsim.gui.beam_analysis import EmissionSourceData


def recorded_result(n=16):
    result = make_result(n)
    # Four distinct directions at every one of four source positions. The
    # duplicated coordinates must remain coincident, without cosmetic jitter.
    position_phi = (np.arange(n) // 4 % 4) * np.pi / 2
    direction_phi = (np.arange(n) % 4) * np.pi / 2
    positions = np.column_stack((np.cos(position_phi)*2e-9,
                                 np.sin(position_phi)*2e-9, np.zeros(n)))
    directions = np.column_stack((np.sin(.03)*np.cos(direction_phi),
                                  np.sin(.03)*np.sin(direction_phi),
                                  np.full(n, np.cos(.03))))
    result.simulation.gun_trace = SimpleNamespace(emission_reference={
        "ray_id": np.arange(n), "position_m": positions,
        "direction": directions, "normal": np.tile([0., 0., 1.], (n, 1)),
    })
    return result


def preset(view, key):
    combo = view.analysis.tracking_combo
    combo.setCurrentIndex(combo.findData(key))


def colours(scatter):
    return {row["source_ray_id"]: brush.color().rgba()
            for row, brush in zip(scatter.data["data"], scatter.data["brush"])}


@pytest.mark.parametrize("tracking", ["source", "emission_direction"])
def test_presets_show_real_source_positions_and_shared_origin_colours(view, tracking):
    result = recorded_result()
    view.display_result(result)
    preset(view, tracking)
    top = view.source_plot.scatter
    positions = result.simulation.gun_trace.emission_reference["position_m"]
    np.testing.assert_allclose(top.data["x"], positions[:, 0]*1e9)
    np.testing.assert_allclose(top.data["y"], positions[:, 1]*1e9)
    assert colours(top) == colours(view._scatter)
    assert view.analysis.mode == ("angular" if tracking == "emission_direction" else "position")
    assert view.source_plot.plot.getAxis("bottom").labelUnits == "nm"
    first_position_colours = {colours(top)[i] for i in range(4)}
    assert len(first_position_colours) == (4 if tracking == "emission_direction" else 1)
    assert view.findChild(QComboBox, "beamTrackingCoordinates") is None
    assert view.analysis.mode_combo.isHidden()
    assert view.colour_mode.isHidden()


def test_source_context_survives_apertures_descendants_and_projection(view):
    result = recorded_result()
    result.simulation.incident.blocked_z[::2] = .5
    add_detailed_exit(result)
    view.display_result(result)
    preset(view, "emission_direction")
    top = view.source_plot.scatter
    initial = colours(top)
    view.focus_z(.2)
    view.focus_z(.8)
    assert view.source_plot.scatter is top
    assert len(view._scatter.data) == 8
    assert colours(top) == initial
    view.focus_z(2.)
    assert len(view._scatter.data) == 5  # Repeated descendants keep their ID.
    assert all(brush.color().rgba() == initial[row["source_ray_id"]]
               for row, brush in zip(view._scatter.data["data"], view._scatter.data["brush"]))
    view.set_projection_angle(90.)
    positions = result.simulation.gun_trace.emission_reference["position_m"]
    np.testing.assert_allclose(view.source_plot.scatter.data["x"], positions[:, 1]*1e9, atol=1e-12)
    np.testing.assert_allclose(view.source_plot.scatter.data["y"], -positions[:, 0]*1e9, atol=1e-12)
    assert colours(view.source_plot.scatter) == initial


def test_lookup_and_brushes_reused_until_next_publication(view, monkeypatch):
    calls = []
    build = EmissionSourceData.from_simulation.__func__
    def counted(cls, simulation):
        calls.append(simulation)
        return build(cls, simulation)
    monkeypatch.setattr(EmissionSourceData, "from_simulation", classmethod(counted))
    result = recorded_result()
    view.display_result(result)
    source = view.analysis.source_data()
    brushes = view.analysis.source_brushes(source.source_ids)
    preset(view, "emission_direction")
    for z in (.1, .3, .7):
        view.focus_z(z)
    switch(view, "position")
    view.set_projection_angle(23.)
    preset(view, "source")
    assert len(calls) == 1
    assert view.analysis.source_data() is source
    # Qt items may copy QBrush values; the controller must reuse the actual
    # colour objects rather than rebuilding them when changing display modes.
    assert all(a is b for a, b in zip(brushes, view.analysis.source_brushes(source.source_ids)))
    # A publication is the cache invalidation boundary, even for the same wrapper.
    view.display_result(result)
    assert len(calls) == 2
    assert view.analysis.source_data() is not source


@pytest.mark.parametrize("tracking, coordinate_mode, colour_mode, unit", [
    ("source", "position", "source", "µm"),
    ("emission_direction", "angular", "emission_direction", "mrad"),
    ("tof", "position", "tof", "µm"),
])
def test_presets_replace_both_coordinate_and_colour_choices(
        view, tracking, coordinate_mode, colour_mode, unit):
    result = recorded_result()
    result.simulation.incident.flight_time_s = np.stack(
        (np.zeros(16), np.linspace(1e-9, 2e-9, 16)))
    view.display_result(result)
    # Start from an independent diagnostic choice. A main plot selection must
    # replace the entire presentation, including a coordinate that may match.
    preset(view, "advanced")
    switch(view, "angular")
    view.colour_mode.setCurrentIndex(view.colour_mode.findData("interaction"))
    preset(view, tracking)
    assert view.analysis.mode == coordinate_mode
    assert view.colour_mode.currentData() == colour_mode
    assert view.analysis.tracking_combo.currentData() == tracking
    assert view.analysis.mode_combo.isHidden()
    assert view.colour_mode.isHidden()
    assert not view.analysis._advanced
    assert not hasattr(view.analysis, "coordinates_combo")
    assert view.findChild(QComboBox, "beamTrackingCoordinates") is None
    assert view.analysis.view_label.text() == (
        "Selected plane: Angular X-Y" if coordinate_mode == "angular"
        else "Selected plane: Position X-Y")
    assert view.source_plot.plot.getAxis("bottom").labelUnits == "nm"
    assert view.plot.getAxis("bottom").labelUnits == unit
    expected_x = (np.arctan(result.simulation.incident.tx[-1])*1e3
                  if coordinate_mode == "angular"
                  else result.simulation.incident.x[-1]*1e6)
    np.testing.assert_allclose(view._scatter.data["x"], expected_x)
    assert colours(view.source_plot.scatter) == colours(view._scatter)


@pytest.mark.parametrize("tracking, diagnostic_coordinate", [
    ("source", "angular"), ("emission_direction", "position"),
    ("tof", "angular"),
])
def test_independent_diagnostic_coordinates_are_never_labelled_as_a_standard_plot(
        view, tracking, diagnostic_coordinate):
    view.display_result(recorded_result())
    preset(view, tracking)
    switch(view, diagnostic_coordinate)
    assert view.analysis.tracking_combo.currentData() == "advanced"
    assert view.analysis._advanced
    assert not view.analysis.mode_combo.isHidden()
    assert not view.colour_mode.isHidden()
    assert view.colour_mode.currentData() == tracking
    # Returning to matching coordinates does not silently exit diagnostics.
    switch(view, "angular" if tracking == "emission_direction" else "position")
    assert view.analysis.tracking_combo.currentData() == "advanced"
    preset(view, tracking)
    assert view.analysis.tracking_combo.currentData() == tracking
    assert not view.analysis._advanced


def test_plot_transitions_reuse_completed_source_and_plane_data(view, monkeypatch):
    from temsim.gui import beam_analysis

    result = recorded_result()
    result.simulation.incident.flight_time_s = np.stack(
        (np.zeros(16), np.linspace(1e-9, 2e-9, 16)))
    original = {key: getattr(result.simulation.incident, key).copy()
                for key in ("x", "y", "tx", "ty", "flight_time_s")}
    view.display_result(result)
    source = view.analysis.source_data()
    cached_plane = view.analysis.plane_data()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changing plot presentation must reuse the completed cache")

    monkeypatch.setattr(beam_analysis, "sample_beam_plane", forbidden)
    monkeypatch.setattr(EmissionSourceData, "from_simulation", classmethod(forbidden))
    for tracking in ("emission_direction", "source", "tof", "source", "emission_direction"):
        preset(view, tracking)
        view.set_projection_angle(31.)
        assert view.analysis.source_data() is source
        assert view.analysis.plane_data() is cached_plane
        assert view._result is result
        assert colours(view.source_plot.scatter) == colours(view._scatter)
    for key, values in original.items():
        np.testing.assert_array_equal(getattr(result.simulation.incident, key), values)


def test_source_display_is_bounded_without_using_survivor_sample(view):
    result = recorded_result(3001)
    result.simulation.incident.blocked_z[::3] = .5
    view.display_result(result)
    expected = np.linspace(0, 3000, view.MAX_DISPLAY_RAYS, dtype=int)
    ids = [row["source_ray_id"] for row in view.source_plot.scatter.data["data"]]
    np.testing.assert_array_equal(ids, expected)
    top = view.source_plot.scatter
    view.focus_z(.1)
    view.focus_z(.9)
    assert view.source_plot.scatter is top
    assert len(view._scatter.data) < len(ids)


def test_missing_historical_launch_positions_are_explicit_without_invented_tip(view):
    view.display_result(recorded_result())
    view.display_result(make_result())
    assert view.source_plot.scatter is None
    assert "unavailable" in view.source_plot.summary.text().lower()
    preset(view, "emission_direction")
    assert all(brush.color() == QColor("#94a3b8") for brush in view._scatter.data["brush"])
    assert view.source_plot.scatter is None


def test_advanced_views_and_colour_legend_remain_available(view):
    view.display_result(recorded_result())
    assert view.angle_colour_wheel.isHidden()
    view.colour_legend_toggle.setChecked(True)
    assert not view.angle_colour_wheel.isHidden()
    preset(view, "advanced")
    assert not view.analysis.mode_combo.isHidden()
    assert not view.colour_mode.isHidden()
    switch(view, "angular")
    assert not view.colour_mode.isHidden()
    view.colour_mode.setCurrentIndex(view.colour_mode.findData("interaction"))
    assert view.source_plot.isHidden()
    preset(view, "source")
    assert not view.source_plot.isHidden()
    assert view.colour_mode.isHidden()


def test_incomplete_tof_is_unavailable_and_does_not_run_any_calculation(view):
    result = recorded_result()
    view.display_result(result)
    combo = view.analysis.tracking_combo
    index = combo.findData("tof")
    assert combo.model().item(index).isEnabled()
    combo.setCurrentIndex(index)
    assert combo.currentData() == "tof"
    assert view._result is result
    assert view.analysis.mode == "position"
    assert "unavailable" in view.analysis.legend.text().lower()
    assert all(brush.color() == QColor("#94a3b8") for brush in view._scatter.data["brush"])


def test_known_azimuth_is_preserved_when_historical_normal_is_missing(view):
    result = recorded_result()
    del result.simulation.gun_trace.emission_reference["normal"]
    view.display_result(result)
    preset(view, "emission_direction")
    assert len(set(colours(view._scatter).values())) == 4
    assert colours(view.source_plot.scatter) == colours(view._scatter)
    assert all("angle to local normal unavailable" in row["emission"]
               for row in view._scatter.data["data"])


@pytest.mark.parametrize("tracking, diagnostic_coordinate", [
    ("source", None), ("emission_direction", None), ("tof", None),
    ("source", "angular"), ("emission_direction", "position"),
])
def test_wave_checkpoint_reader_restores_particle_preset_and_controls(
        view, tracking, diagnostic_coordinate):
    from test_wave_beam_analysis import checkpoint
    result = recorded_result()
    view.display_result(result)
    preset(view, tracking)
    if diagnostic_coordinate is not None:
        switch(view, diagnostic_coordinate)
    expected_mode = view.analysis.mode
    expected_colour = view.colour_mode.currentData()
    expected_tracking = view.analysis.tracking_combo.currentData()
    view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
    assert view.initial_beam_panel.isHidden()
    assert not view.analysis.mode_combo.isHidden()
    assert view.findChild(QComboBox, "beamTrackingCoordinates") is None
    view.display_result(result, focus=("z", 1.))
    assert view.analysis.tracking_combo.currentData() == expected_tracking
    assert view.analysis.mode == expected_mode
    assert view.colour_mode.currentData() == expected_colour
    assert view.analysis.mode_combo.isHidden() == (diagnostic_coordinate is None)
    assert view.colour_mode.isHidden() == (diagnostic_coordinate is None)
    assert view.findChild(QComboBox, "beamTrackingCoordinates") is None
    assert not view.source_plot.isHidden()
