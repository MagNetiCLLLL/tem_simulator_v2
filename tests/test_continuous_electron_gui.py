"""Continuous trajectory publication and honest previous/provisional states.

Worker fixtures below publish synthetic integrated prefixes solely to test GUI
identity and scheduling; they do not qualify electron transport physics.
"""
import math
import re
from threading import Event
from time import monotonic

import numpy as np
import pytest
from PySide6.QtCore import Qt

from tests.test_magnetic_test_electron_gui import (
    UniformScene, controller, install_trace, wait_for_result,  # noqa: F401 - shared pytest fixture
)
from temsim.magnetic_test_particle import TestElectronTrajectory as ElectronTrajectory


def sampled_trajectory(settings, *, count=9, reason="path_limit"):
    # A distinct, deterministic curve for every energy/angle snapshot. Prefixes
    # use the same time-ordered samples as their synthetic completed result.
    fraction = np.arange(count, dtype=float) / 8.
    distance = fraction * .001
    start = np.asarray(settings.position_m)
    excursion = settings.kinetic_energy_ev * 1e-12 + settings.polar_angle_deg * 1e-7
    positions = start + np.column_stack((excursion * fraction**2, np.zeros(count), distance))
    return ElectronTrajectory(
        positions_m=positions,
        directions=np.broadcast_to((0., 0., 1.), (count, 3)).copy(),
        time_s=distance / 1e8,
        path_length_m=distance,
        energy_invariant_error_ev=0., energy_invariant_relative_error=0.,
        reason=reason, completed=reason == "path_limit", steps=count - 1,
        kinetic_energy_ev=np.full(count, settings.kinetic_energy_ev),
        electrostatic_potential_v=np.zeros(count), speed_m_per_s=np.full(count, 1e8),
        momentum_kg_m_per_s=np.broadcast_to((0., 0., 1e-22), (count, 3)).copy(),
        notes=("Synthetic GUI lifecycle fixture, not a physical trajectory.",))


def test_progress_replaces_only_its_prefix_and_keeps_previous_full_path_separate(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def trace(_scene, settings, *, progress=None, **_kwargs):
        calls.append(settings)
        if len(calls) > 1:
            assert callable(progress)
            progress(sampled_trajectory(settings, count=3, reason="in_progress"))
            entered.set()
            assert release.wait(5.)
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    previous = wait_for_result(qtbot, controller)
    key = controller.selected_record.key
    controller.energy.setValue(250000.)
    immediate = controller.visible_paths()
    assert len(immediate) == 1 and immediate[0].state == "previous"
    np.testing.assert_array_equal(immediate[0].positions_m, previous.positions_m)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: any(path.state == "in_progress" for path in controller.visible_paths()))
        paths = {path.key: path for path in controller.visible_paths()}
        assert set(paths) == {key, key + "/progress"}
        assert paths[key].state == "previous"
        assert paths[key + "/progress"].state == "in_progress"
        np.testing.assert_array_equal(paths[key].positions_m, previous.positions_m)
        np.testing.assert_array_equal(paths[key + "/progress"].positions_m,
                                      sampled_trajectory(calls[-1], count=3, reason="in_progress").positions_m)
        assert controller.current_trajectory is None
        assert len(controller._cache) == 1
        assert all(result.reason != "in_progress" for result in controller._cache.values())
    finally:
        release.set()
    final = wait_for_result(qtbot, controller)
    assert final.kinetic_energy_ev[0] == 250000.
    assert [(path.key, path.state) for path in controller.visible_paths()] == [(key, "current")]
    assert len(controller._cache) == 2 and len(calls) == 2


def test_dense_slider_events_publish_real_snapshots_while_held_without_blanking(controller, qtbot, monkeypatch):
    calls, held_frames, identity_errors = [], [], []
    release = Event()

    def trace(_scene, settings, *, progress=None, cancelled=None, **_kwargs):
        calls.append(settings)
        if len(calls) == 1:
            return sampled_trajectory(settings)
        for count in range(2, 10):
            if cancelled is not None and cancelled():
                return sampled_trajectory(settings, count=count, reason="cancelled")
            progress(sampled_trajectory(settings, count=count, reason="in_progress"))
            if release.wait(.02):
                break
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    slider = controller.sliders["energy"]

    def collect(paths):
        if slider.isSliderDown():
            held_frames.append(tuple((path.key, path.state, path.positions_m.tobytes()) for path in paths))
        current = controller.current_trajectory
        if current is not None and current.kinetic_energy_ev[0] != controller.settings().kinetic_energy_ev:
            identity_errors.append("A different parameter snapshot was presented as current")
        for (_generation, settings), result in controller._cache.items():
            if settings.kinetic_energy_ev != result.kinetic_energy_ev[0]:
                identity_errors.append("Completed data cached under different parameters")

    controller.paths_changed.connect(collect)
    slider.setSliderDown(True)
    try:
        # Every 15 ms remains shorter than the live dispatch cadence. A trailing
        # debounce would fail to produce any of these held-drag updates.
        for tick in range(1150, 1190):
            slider.setValue(tick)
            qtbot.wait(15)
        assert slider.isSliderDown() and len(calls) >= 2
        assert held_frames and all(frame for frame in held_frames)
        prefixes = {data for frame in held_frames for key, state, data in frame
                    if key.endswith("/progress") and state in {"previous", "in_progress"}}
        assert len(prefixes) >= 2
        assert not identity_errors
    finally:
        release.set()
        slider.setSliderDown(False)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == controller.settings().kinetic_energy_ev
    assert all(path.state == "current" for path in controller.visible_paths())
    assert not identity_errors


@pytest.mark.parametrize("startup_delay_s", [.09, .15])
def test_continuous_drag_does_not_starve_progress_with_worker_startup_and_message_latency(
        controller, qtbot, monkeypatch, startup_delay_s):
    calls, held_frames, identity_errors = [], [], []
    prefix_parameters = {}
    release = Event()

    def wait_for_delivery(seconds, cancelled):
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            if cancelled():
                return False
            if release.wait(min(.005, max(0., deadline - monotonic()))):
                return not cancelled()
        return not cancelled()

    def trace(_scene, settings, *, progress=None, cancelled=None, **_kwargs):
        calls.append(settings)
        if len(calls) == 1:
            return sampled_trajectory(settings)
        # Admission/IPC startup and one compute/message interval precede the
        # first deliverable prefix. This reproduces cancellation starvation
        # that an immediate synthetic callback cannot expose.
        if not wait_for_delivery(startup_delay_s, cancelled):
            return sampled_trajectory(settings, count=1, reason="cancelled")
        for count in (3, 5, 7):
            if not wait_for_delivery(.07, cancelled):
                return sampled_trajectory(settings, count=count, reason="cancelled")
            prefix = sampled_trajectory(settings, count=count, reason="in_progress")
            prefix_parameters[prefix.positions_m.tobytes()] = settings.kinetic_energy_ev
            progress(prefix)
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    slider = controller.sliders["energy"]

    def collect(paths):
        if slider.isSliderDown():
            held_frames.append(tuple((path.key, path.state, path.positions_m.tobytes()) for path in paths))
        current = controller.current_trajectory
        if current is not None and current.kinetic_energy_ev[0] != controller.settings().kinetic_energy_ev:
            identity_errors.append("An earlier delayed sample was reported as current")
        for (_generation, settings), result in controller._cache.items():
            if settings.kinetic_energy_ev != result.kinetic_energy_ev[0] or result.reason == "in_progress":
                identity_errors.append("A delayed or provisional sample has an incorrect cache identity")

    controller.paths_changed.connect(collect)
    slider.setSliderDown(True)
    try:
        for tick in range(1150, 1230):
            slider.setValue(tick)
            qtbot.wait(15)
        assert slider.isSliderDown()
        assert held_frames and all(frame for frame in held_frames)
        delivered_prefixes = {data for frame in held_frames for key, state, data in frame
                              if key.endswith("/progress") and state in {"previous", "in_progress"}}
        # More than one captured parameter set must reach the display while
        # still dragging; a single retained old path cannot satisfy this.
        assert len({prefix_parameters[data] for data in delivered_prefixes}) >= 2
        assert not identity_errors
    finally:
        release.set()
        slider.setSliderDown(False)
        # Settle the delayed fixture before pytest-qt closes its parent widget,
        # including when the non-starvation assertion deliberately fails.
        qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == controller.settings().kinetic_energy_ev
    assert all(path.state == "current" for path in controller.visible_paths())
    assert not identity_errors


@pytest.mark.parametrize("change", ["parameters", "scene", "remove", "hide"])
def test_cancelled_or_replaced_worker_cannot_publish_late_progress(controller, qtbot, monkeypatch, change):
    entered, release = Event(), Event()
    calls, obsolete_callbacks = [], []

    def trace(_scene, settings, *, progress=None, **_kwargs):
        calls.append(settings)
        if len(calls) == 2:
            obsolete_callbacks.append(progress)
            progress(sampled_trajectory(settings, count=3, reason="in_progress"))
            entered.set()
            assert release.wait(5.)
            # Simulate data already crossing the worker boundary when cancelled.
            progress(sampled_trajectory(settings, count=8, reason="in_progress"))
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    controller.energy.setValue(240000.)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        old_worker = controller._worker
        if change == "parameters":
            controller.energy.setValue(260000.)
        elif change == "scene":
            controller.set_scene(UniformScene((0., .02, 0.)))
            assert not controller.visible_paths()
        elif change == "remove":
            removed_key = controller.selected_record.key
            controller.remove_electron()
            assert not controller.visible_paths()
            assert controller.add_electron() != removed_key
        else:
            controller.set_active(False)
        assert old_worker.cancelled.is_set()
    finally:
        release.set()
    if change == "hide":
        qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
        controller.set_active(True)
    latest = wait_for_result(qtbot, controller)
    before = tuple((path.key, path.state, path.positions_m.copy()) for path in controller.visible_paths())
    cache = dict(controller._cache)
    obsolete_callbacks[0](sampled_trajectory(calls[1], count=8, reason="in_progress"))
    qtbot.wait(50)
    assert controller.current_trajectory is latest
    after = controller.visible_paths()
    assert [(path.key, path.state) for path in after] == [(key, state) for key, state, _ in before]
    for path, (_key, _state, positions) in zip(after, before):
        np.testing.assert_array_equal(path.positions_m, positions)
    assert set(controller._cache) == set(cache)
    assert all(controller._cache[key] is result for key, result in cache.items())


def test_progress_never_moves_backwards_within_one_request(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def trace(_scene, settings, *, progress=None, **_kwargs):
        calls.append(settings)
        if len(calls) > 1:
            progress(sampled_trajectory(settings, count=6, reason="in_progress"))
            progress(sampled_trajectory(settings, count=3, reason="in_progress"))
            entered.set()
            assert release.wait(5.)
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    controller.energy.setValue(250000.)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: any(path.key.endswith("/progress") for path in controller.visible_paths()))
        prefix = next(path for path in controller.visible_paths() if path.key.endswith("/progress"))
        assert len(prefix.positions_m) == 6
        np.testing.assert_array_equal(prefix.positions_m,
                                      sampled_trajectory(calls[-1], count=6, reason="in_progress").positions_m)
        assert controller.current_trajectory is None
    finally:
        release.set()
    wait_for_result(qtbot, controller)


def test_earlier_live_completion_cannot_replace_a_newer_exact_cached_result(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls, invalidated_current = [], []

    def trace(_scene, settings, *, progress=None, **_kwargs):
        calls.append(settings)
        if len(calls) > 1:
            progress(sampled_trajectory(settings, count=3, reason="in_progress"))
            entered.set()
            assert release.wait(5.)
        return sampled_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    slider = controller.sliders["energy"]
    slider.setSliderDown(True)
    controller.energy.setValue(230000.)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        # Return to an exact already-calculated state while the previous live
        # sample is still allowed to run. Its later completion must not displace
        # the newly restored current result, even for one presentation frame.
        controller.energy.setValue(200000.)
        assert controller.current_trajectory is original
        controller.paths_changed.connect(
            lambda _paths: invalidated_current.append(True)
            if controller.current_trajectory is not original else None)
        release.set()
        qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
        qtbot.wait(150)
        assert controller.current_trajectory is original
        assert not invalidated_current
        assert len(calls) == 2
    finally:
        release.set()
        slider.setSliderDown(False)


@pytest.mark.parametrize("mrad", [0., 1., 100., math.pi * 1000.])
def test_polar_mrad_input_and_degrees_label_share_one_physical_parameter(controller, mrad):
    controller.set_scene(UniformScene())
    controller.polar.setValue(mrad)
    assert controller.polar.suffix().strip() == "mrad"
    expected = min(180., math.degrees(controller.polar.value() / 1000.))
    assert controller.settings().polar_angle_deg == pytest.approx(expected, abs=1e-6)
    assert "°" in controller.polar_degrees.text() or "deg" in controller.polar_degrees.text()
    # The accompanying label must provide a numeric degree value, not a second
    # mutable copy of the same physical parameter.
    numeric = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", controller.polar_degrees.text())
    assert numeric is not None
    displayed = float(numeric.group())
    assert displayed == pytest.approx(expected, abs=1e-3)
    assert controller._worker is None


def test_degree_equivalent_updates_when_switching_records_and_resetting_tip(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.polar.setValue(10.)
    first = wait_for_result(qtbot, controller)
    first_text = controller.polar_degrees.text()
    controller.duplicate_electron()
    controller.polar.setValue(20.)
    wait_for_result(qtbot, controller)
    second_text = controller.polar_degrees.text()
    assert first_text != second_text
    controller.select_electron(first_key)
    assert controller.polar.value() == 10. and controller.polar_degrees.text() == first_text
    assert controller.current_trajectory is first
    controller.reset_to_tip.click()
    result = wait_for_result(qtbot, controller)
    assert controller.polar.value() == 0. and controller.settings().polar_angle_deg == 0.
    assert result is original
    assert len(calls) == 3  # The original zero-angle path is already cached.


@pytest.mark.parametrize("tick, expected_mrad", [(0, 0.), (1, .0025), (200, .5), (400, 1.), (2000, 5.)])
def test_polar_slider_finely_covers_zero_to_five_milliradians(controller, tick, expected_mrad):
    controller.set_scene(UniformScene())
    slider = controller.sliders["polar"]
    slider.setValue(tick)
    assert controller.polar.value() == pytest.approx(expected_mrad, abs=1e-9)
    assert controller.settings().polar_angle_deg == pytest.approx(math.degrees(expected_mrad/1000.), abs=1e-10)
    assert controller.polar.singleStep() == .01
    assert controller.polar.maximum() == pytest.approx(math.pi*1000., abs=1e-6)


def test_numeric_polar_step_is_fine_without_restricting_the_full_angular_range(controller):
    controller.set_scene(UniformScene())
    controller.polar.setValue(.5)
    controller.polar.stepBy(1)
    assert controller.polar.value() == pytest.approx(.51)
    assert controller.sliders["polar"].value() == 204
    controller.polar.setValue(math.pi*1000.)
    assert controller.settings().polar_angle_deg == 180.
    assert controller.sliders["polar"].value() == controller.sliders["polar"].maximum()


def test_selecting_existing_wide_angle_retains_it_until_user_moves_the_fine_slider(controller):
    controller.set_scene(UniformScene())
    controller.polar.setValue(1200.)
    wide_key = controller.selected_record.key
    wide_settings = controller.settings()
    wide_revision = controller.selected_record.revision
    controller.duplicate_electron()
    controller.polar.setValue(.5)
    controller.select_electron(wide_key)
    assert controller.settings() is wide_settings
    assert controller.polar.value() == 1200.
    assert controller.selected_record.revision == wide_revision
    assert controller.sliders["polar"].value() == 2000
    # A deliberate user movement enters the fine range; merely displaying or
    # switching records must never silently replace a large physical angle.
    controller.sliders["polar"].setValue(1999)
    assert controller.polar.value() == pytest.approx(4.9975)
    assert controller.settings().polar_angle_deg == pytest.approx(math.degrees(.0049975))


def test_polar_and_azimuth_controls_reach_the_documented_initial_directions(controller):
    from dataclasses import replace
    from temsim.magnetic_test_particle import trace_test_electron

    scene = UniformScene((0., 0., 0.))
    controller.set_scene(scene)
    def initial_direction():
        settings = replace(controller.settings(), step_m=.001, max_path_length_m=.001, max_steps=10)
        return trace_test_electron(scene, settings).directions[0]

    controller.polar.setValue(0.)
    for azimuth in (0., 90., 230.):
        controller.azimuth.setValue(azimuth)
        np.testing.assert_array_equal(initial_direction(), (0., 0., 1.))
    controller.polar.setValue(1.)
    controller.azimuth.setValue(0.)
    np.testing.assert_allclose(initial_direction(), (math.sin(.001), 0., math.cos(.001)), atol=1e-15)
    controller.azimuth.setValue(90.)
    np.testing.assert_allclose(initial_direction(), (0., math.sin(.001), math.cos(.001)), atol=1e-15)


@pytest.mark.parametrize("azimuth", [0., math.pi/2., math.pi])
def test_direction_readout_uses_integrated_three_dimensional_angles_not_ray_slope(controller, azimuth):
    from dataclasses import replace

    controller.set_scene(UniformScene())
    angles = np.array((.0005, .003, .001))
    directions = np.column_stack((np.sin(angles)*np.cos(azimuth),
                                  np.sin(angles)*np.sin(azimuth), np.cos(angles)))
    result = replace(sampled_trajectory(controller.settings(), count=3), directions=directions)
    controller.selected_record.trajectory = result
    controller._show_result_status(controller.selected_record)
    assert "Physical angle: 0.5 → 1 mrad" in controller.energy_status.text()
    assert "Initial → final angle from +Z" in controller.energy_status.toolTip()
    assert "Maximum angle from +Z on this calculated path: 3 mrad" in controller.energy_status.toolTip()
    # The synthetic polyline has a different slope and no Y displacement, so
    # neither its projected shape nor the azimuth can provide these values.
    assert "Along path:" in controller.energy_status.text()


def test_direction_readout_covers_backward_motion_and_reports_zero_as_undefined(controller):
    from dataclasses import replace

    controller.set_scene(UniformScene())
    result = replace(sampled_trajectory(controller.settings(), count=3),
                     directions=np.array(((0., 0., 0.), (0., 1., 0.), (0., 0., -1.))))
    controller.selected_record.trajectory = result
    controller._show_result_status(controller.selected_record)
    assert "Physical angle: undefined → 3141.59 mrad" in controller.energy_status.text()
    assert "Maximum angle from +Z on this calculated path: 3141.59 mrad" in controller.energy_status.toolTip()
    controller.energy_status.setText("Live preview energy: 200000 eV; integration incomplete")
    controller._show_direction_status(result, terminal=False)
    assert "Physical angle (live):" in controller.energy_status.text()
    assert "Initial → current angle from +Z" in controller.energy_status.toolTip()
    assert "final angle" not in controller.energy_status.text()
    controller._refresh_status()
    assert controller.energy_status.toolTip() == ""


def test_path_presentation_states_validate_without_changing_physical_samples():
    from temsim.gui.test_electron_types import ElectronPath
    positions = np.array(((0., 0., 0.), (.0001, 0., .001)))
    for state in ("current", "previous", "in_progress"):
        path = ElectronPath("electron-1", "Electron 1", "#48cae4", positions, state=state)
        assert path.state == state and not path.positions_m.flags.writeable
        np.testing.assert_array_equal(path.positions_m, positions)
    for state in ("stale", "", None):
        with pytest.raises(ValueError, match="state"):
            ElectronPath("electron-1", "Electron 1", "#48cae4", positions, state=state)
    np.testing.assert_array_equal(positions, ((0., 0., 0.), (.0001, 0., .001)))


def test_previous_path_uses_dashed_style_without_reprojecting_or_refitting_scene(qtbot, monkeypatch):
    from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas
    from temsim.gui.test_electron_types import ElectronPath

    canvas = MagneticFieldCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 400)
    canvas.set_geometry(
        np.array([[[0., 0., 0.], [0., 0., .001]]]), np.array([.01]),
        reference_t=.01, bounds_m=np.array([[-.001, -.001, 0.], [.001, .001, .001]]))
    canvas.set_electron_mode(True)
    canvas.set_view_range_mm((0., 1.), (-.2, .2))
    positions = np.array(((0., 0., 0.), (.0001, 0., .001)))
    canvas.set_electron_paths((ElectronPath("electron-1", "Electron 1", "#48cae4", positions),))
    original_projection = canvas._electron_projections["electron-1"]
    magnetic_paths = canvas._colour_paths
    view_range = canvas.view_range_mm()
    pens = []
    make_pen = canvas._pen

    def record_pen(colour, width=1., style=Qt.PenStyle.SolidLine):
        if colour == "#48cae4":
            pens.append(style)
        return make_pen(colour, width, style)

    monkeypatch.setattr(canvas, "_pen", record_pen)
    canvas.set_electron_paths((ElectronPath("electron-1", "Previous parameters", "#48cae4",
                                          positions, state="previous"),))
    canvas.show()
    qtbot.wait(20)
    canvas.grab()
    assert Qt.PenStyle.DashLine in pens
    assert canvas._electron_projections["electron-1"] is original_projection
    assert canvas._colour_paths is magnetic_paths
    np.testing.assert_array_equal(canvas.view_range_mm(), view_range)
    pens.clear()
    canvas.set_electron_paths((ElectronPath("electron-1/progress", "Live preview", "#48cae4",
                                          positions[:1], state="in_progress"),))
    canvas.grab()
    assert Qt.PenStyle.DashLine not in pens
    assert canvas._colour_paths is magnetic_paths
    np.testing.assert_array_equal(canvas.view_range_mm(), view_range)


@pytest.fixture
def raster_field_canvas(qtbot):
    from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas
    from temsim.gui.test_electron_types import ElectronPath

    canvas = MagneticFieldCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 400)
    canvas.set_geometry([[[.001, 0., 0.], [.001, 0., 1.]]], [1.],
                        reference_t=1., bounds_m=[[-.002, -.002, 0.], [.002, .002, 1.]])
    canvas.set_electron_mode(True)
    canvas.set_view_range_mm((0., 1000.), (-2., 2.))
    canvas.set_electron_paths((ElectronPath("electron-1", "Electron 1", "#48cae4",
                                           np.array([[0., 0., 0.], [.0001, 0., 1.]])),))
    canvas.show()
    qtbot.wait(10)
    canvas.grab()
    return canvas


def test_electron_prefix_frames_reuse_the_unchanged_rasterized_field(raster_field_canvas):
    from temsim.gui.test_electron_types import ElectronPath

    canvas = raster_field_canvas
    layer = canvas._field_layer_pixmap
    original_physical_field = canvas._segments_m.copy()
    view_range = canvas.view_range_mm()
    for count in (2, 4, 8):
        t = np.linspace(0., 1., count)
        positions = np.column_stack((.0005*t*t, np.zeros(count), t))
        canvas.set_electron_paths((ElectronPath("electron-1/progress", "Live preview", "#48cae4",
                                               positions, state="in_progress"),))
        canvas.grab()
        assert canvas._field_layer_pixmap is layer
        assert canvas.electron_point_count == count
        np.testing.assert_array_equal(canvas._segments_m, original_physical_field)
        assert canvas.view_range_mm() == view_range


@pytest.mark.parametrize("change", ["pan", "zoom", "projection", "geometry", "resize", "mode", "pixel_edges", "dpr"])
def test_field_raster_refreshes_for_camera_geometry_or_display_changes(raster_field_canvas, monkeypatch, change):
    canvas = raster_field_canvas
    previous = canvas._field_layer_pixmap
    previous_pixels = previous.toImage()
    physical_field = canvas._segments_m.copy()
    if change == "pan":
        canvas.set_view_range_mm((100., 1100.), (-1.5, 2.5))
    elif change == "zoom":
        canvas.set_view_range_mm((100., 900.), (-1.2, 1.2))
    elif change == "projection":
        canvas.set_projection_angle(60.)
    elif change == "geometry":
        canvas.set_geometry([[[.0005, 0., 0.], [.0005, 0., 1.]]], [.5],
                            reference_t=1., bounds_m=canvas._bounds_m)
    elif change == "resize":
        canvas.resize(950, 450)
    elif change == "mode":
        canvas.set_electron_mode(False)
    elif change == "pixel_edges":
        canvas.set_horizontal_plot_edges((110., 690.))
    else:
        monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: 2.)
    canvas.grab()
    layer = canvas._field_layer_pixmap
    assert layer is not previous
    assert layer.toImage() != previous_pixels
    if change != "geometry":
        np.testing.assert_array_equal(canvas._segments_m, physical_field)
    if change == "dpr":
        assert layer.devicePixelRatioF() == 2.
        assert layer.width() == 2*canvas.width() and layer.height() == 2*canvas.height()
    canvas.grab()
    assert canvas._field_layer_pixmap is layer


def test_hiding_field_background_removes_its_pixels_and_showing_restores_the_cached_layer(raster_field_canvas):
    canvas = raster_field_canvas
    layer = canvas._field_layer_pixmap
    point = canvas.physical_mm_to_screen(250., 1.)
    x, y = round(point.x()), round(point.y())

    def nearby_pixels():
        image = canvas.grab().toImage()
        return np.array([image.pixelColor(x, row).getRgb() for row in range(y-2, y+3)])

    visible = nearby_pixels()
    canvas.set_field_lines_visible(False)
    hidden = nearby_pixels()
    assert np.max(np.abs(visible.astype(float)-hidden)) > 40.
    assert canvas._field_layer_pixmap is layer
    assert canvas.electron_point_count == 2
    canvas.set_field_lines_visible(True)
    restored = nearby_pixels()
    np.testing.assert_array_equal(restored, visible)
    assert canvas._field_layer_pixmap is layer
