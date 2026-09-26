"""Independent virtual-electron controls and latest-only GUI publication.

Synthetic worker outputs exercise lifecycle contracts, not microscope physics.
The uniform-field cases execute the real diagnostic integrator.
"""
from threading import Event
from types import SimpleNamespace
import math

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow, QWidget

from temsim.gui.magnetic_test_electron import (
    TestElectronController as ElectronController,
)
from temsim.magnetic_test_particle import TestElectronTrajectory as ElectronTrajectory


class UniformScene:
    def __init__(self, field=(0., 0., .01), *, initial_energy_ev=200000., initial_position_m=(1e-6, 0., .05)):
        self.bounds_m = np.array(((-.01, -.01, 0.), (.01, .01, .1)))
        self.field = np.asarray(field, dtype=float)
        self.bounds_m.setflags(write=False)
        self.field.setflags(write=False)
        self.initial_energy_ev = initial_energy_ev
        self.initial_position_m = initial_position_m
        self.default_path_length_m = .05
        self.source_regions = ()
        self.source_categories = ("synthetic",)
        self.notes = ("Synthetic constant magnetic field for diagnostic tests.",)

    def contains(self, points):
        return ((points >= self.bounds_m[0]) & (points <= self.bounds_m[1])).all(axis=1)

    def field_at_global_positions_t(self, points):
        return np.broadcast_to(self.field, np.asarray(points).shape).copy()


def synthetic_trajectory(settings, *, reason="path_limit"):
    start = np.asarray(settings.position_m)
    length = min(settings.max_path_length_m, .001)
    return ElectronTrajectory(
        positions_m=np.stack((start, start + (0., 0., length))),
        directions=np.array(((0., 0., 1.), (0., 0., 1.))),
        time_s=np.array((0., length / 1e8)),
        path_length_m=np.array((0., length)),
        momentum_kg_m_per_s=np.array(((0., 0., 1e-22), (0., 0., 1e-22))),
        electrostatic_potential_v=np.zeros(2), energy_invariant_error_ev=0.,
        energy_invariant_relative_error=0.,
        reason=reason, completed=reason != "step_limit", steps=1,
        kinetic_energy_ev=np.full(2, settings.kinetic_energy_ev), speed_m_per_s=np.full(2, 1e8),
        notes=("Synthetic result used only for GUI lifecycle tests.",))


@pytest.fixture
def controller(qtbot):
    parent = QMainWindow()
    qtbot.addWidget(parent)
    result = ElectronController(parent)
    dock = QDockWidget("Virtual electrons", parent)
    dock.setWidget(result.panel)
    parent.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    parent.resize(900, 650)
    parent.show()
    dock.hide()
    result.controls_requested.connect(dock.show)
    result.controls_requested.connect(dock.raise_)
    yield result
    result.set_active(False)
    qtbot.waitUntil(lambda: result._worker is None and result._scene_worker is None, timeout=10000)


def install_trace(monkeypatch):
    calls = []

    def trace(scene, settings, **_kwargs):
        calls.append((scene, settings))
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    return calls


def install_scene_preparer(monkeypatch):
    calls = []

    def prepare(state, magnetic_scene, *, z_limits_mm=None):
        calls.append((state, magnetic_scene, z_limits_mm))
        return magnetic_scene

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", prepare)
    return calls


def wait_for_result(qtbot, controller):
    qtbot.waitUntil(lambda: controller.current_trajectory is not None and controller._worker is None,
                    timeout=10000)
    return controller.current_trajectory


def electron_record(controller, key):
    return next(record for record in controller.records if record.key == key)


def show_overlay(controller):
    controller.display_mode.setCurrentIndex(controller.display_mode.findData("overlay"))


def wait_for_records(qtbot, controller, keys):
    qtbot.waitUntil(lambda: all(controller._has_result(electron_record(controller, key)) for key in keys)
                    and controller._worker is None, timeout=10000)


def electron_item(controller, key):
    return next(controller.electron_list.topLevelItem(index)
                for index in range(controller.electron_list.topLevelItemCount())
                if controller.electron_list.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole) == key)


def test_controls_convert_explicit_units_and_do_not_mutate_captured_scene(controller, qtbot, monkeypatch):
    scene = UniformScene()
    before = (scene.bounds_m.copy(), scene.field.copy(), dict(vars(scene)))
    calls = install_trace(monkeypatch)
    controller.set_scene(scene)
    controller.energy.setValue(125000.)
    controller.x.setValue(2.5)
    controller.y.setValue(-3.)
    controller.z.setValue(25.)
    controller.polar.setValue(math.radians(120.) * 1000.)
    controller.azimuth.setValue(240.)
    controller.length.setValue(7.)
    controller.step.setValue(.025)
    settings = controller.settings()
    assert settings.kinetic_energy_ev == 125000.
    np.testing.assert_allclose(settings.position_m, (2.5e-6, -3e-6, .025))
    assert settings.polar_angle_deg == pytest.approx(120., abs=1e-6)
    assert settings.azimuth_angle_deg == 240.
    assert settings.max_path_length_m == .007 and settings.step_m == .000025
    qtbot.wait(180)
    assert calls == []
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    assert calls == [(scene, settings)]
    np.testing.assert_array_equal(scene.bounds_m, before[0])
    np.testing.assert_array_equal(scene.field, before[1])
    assert vars(scene).keys() == before[2].keys()
    assert controller._scene is scene


def test_rapid_parameter_edits_are_debounced_into_one_latest_request(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    for energy in (210000., 220000., 230000., 240000.):
        controller.energy.setValue(energy)
    wait_for_result(qtbot, controller)
    assert len(calls) == 1
    assert calls[0][1].kinetic_energy_ev == 240000.


def test_slow_typing_waits_for_enter_and_never_traces_numeric_prefixes(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    previous = wait_for_result(qtbot, controller)
    controller.show_controls()
    line = controller.energy.lineEdit()
    line.setFocus()
    line.selectAll()
    for character in '250000':
        qtbot.keyClicks(line, character)
        qtbot.wait(180)  # Deliberately longer than the live-update dispatch interval.
        assert controller.current_trajectory is None
        paths = controller.visible_paths()
        assert paths and all(path.state == "previous" for path in paths)
        np.testing.assert_array_equal(paths[0].positions_m, previous.positions_m)
    assert [settings.kinetic_energy_ev for _, settings in calls] == [200000.]
    assert controller.energy.value() == 250000. and controller.settings().kinetic_energy_ev == 250000.
    assert 'Editing' in controller.status_text
    qtbot.keyClick(line, Qt.Key.Key_Return)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == 250000.
    assert [settings.kinetic_energy_ev for _, settings in calls] == [200000., 250000.]
    assert not controller._editing_controls


def test_text_commit_on_focus_loss_uses_latest_value(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    controller.show_controls()
    line = controller.polar.lineEdit()
    line.setFocus()
    line.selectAll()
    qtbot.keyClicks(line, '12.5')
    qtbot.wait(180)
    assert len(calls) == 1 and controller.current_trajectory is None
    controller.name_editor.setFocus()
    wait_for_result(qtbot, controller)
    assert len(calls) == 2 and calls[-1][1].polar_angle_deg == pytest.approx(math.degrees(.0125))
    assert not controller._editing_controls


def test_slider_drag_updates_before_release_and_reuses_final_result(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    slider = controller.sliders['energy']
    slider.setSliderDown(True)
    for tick in (1200, 1250, 1300):
        slider.setValue(tick)
        assert controller.visible_paths()  # Keep the previous calculated curve during an update.
        latest = wait_for_result(qtbot, controller)
        assert slider.isSliderDown()
        assert latest.kinetic_energy_ev[0] == controller.energy.value()
    assert len(calls) == 4
    last_energy = controller.energy.value()
    slider.setSliderDown(False)
    wait_for_result(qtbot, controller)
    qtbot.wait(180)
    assert len(calls) == 4 and calls[-1][1].kinetic_energy_ev == last_energy
    assert not controller._editing_controls


def test_record_switch_during_text_edit_does_not_leave_record_blocked(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    first = controller.selected_record.key
    second = controller.duplicate_electron()
    controller.select_electron(first)
    controller.show_controls()
    line = controller.energy.lineEdit()
    line.selectAll()
    qtbot.keyClicks(line, '250000')
    qtbot.wait(180)
    assert len(calls) == 1
    controller.select_electron(second)
    assert controller.current_trajectory is original and not controller._editing_controls
    qtbot.wait(180)
    assert len(calls) == 1  # The dirty first electron remains hidden.
    controller.select_electron(first)
    result = wait_for_result(qtbot, controller)
    assert len(calls) == 2 and result.kinetic_energy_ev[0] == 250000.


@pytest.mark.parametrize('hide_view', [False, True])
def test_hiding_during_drag_finishes_edit_transaction(controller, qtbot, monkeypatch, hide_view):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    controller.show_controls()
    slider = controller.sliders['energy']
    slider.setSliderDown(True)
    slider.setValue(1300)
    retained = wait_for_result(qtbot, controller)
    assert len(calls) == 2
    if hide_view:
        controller.set_active(False)
    else:
        controller.panel.parentWidget().hide()
    assert not controller._editing_controls and not slider.isSliderDown()
    if hide_view:
        qtbot.wait(180)
        assert len(calls) == 2
        controller.set_active(True)
    assert wait_for_result(qtbot, controller) is retained
    assert len(calls) == 2


def test_typing_cancels_old_worker_and_commits_only_final_result(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []
    def trace(_scene, settings, **_kwargs):
        calls.append(settings.kinetic_energy_ev)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)
    monkeypatch.setattr('temsim.magnetic_test_particle.trace_test_electron', trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set)
        controller.show_controls()
        line = controller.energy.lineEdit()
        line.selectAll()
        qtbot.keyClicks(line, '250000')
        assert controller._worker.cancelled.is_set()
        release.set()
        qtbot.waitUntil(lambda: controller._worker is None)
        qtbot.wait(180)
        assert calls == [200000.] and controller.current_trajectory is None
        qtbot.keyClick(line, Qt.Key.Key_Return)
    finally:
        release.set()
    result = wait_for_result(qtbot, controller)
    assert calls == [200000., 250000.] and result.kinetic_energy_ev[0] == 250000.


def test_edits_keep_bounded_coalescing_after_immediate_activation_and_cached_reentry(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    controller.set_active(False)
    controller.set_active(True)
    qtbot.wait(30)
    assert controller.current_trajectory is original and len(calls) == 1
    # Reentering a cached mode may dispatch immediately, but rapid subsequent
    # edits still need one bounded coalescing interval instead of one job each.
    for energy in (225000., 250000., 275000.):
        controller.energy.setValue(energy)
        assert len(calls) == 1
    qtbot.wait(15)
    assert len(calls) == 1
    latest = wait_for_result(qtbot, controller)
    assert latest.kinetic_energy_ev[0] == 275000. and len(calls) == 2


def test_running_old_parameters_never_publish_and_latest_runs_once(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls, published = [], []

    def trace(scene, settings, **_kwargs):
        calls.append(settings)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.paths_changed.connect(
        lambda _paths: published.append(controller.current_trajectory)
        if controller.current_trajectory is not None else None)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        controller.energy.setValue(210000.)
        controller.energy.setValue(275000.)
        assert controller._worker.cancelled.is_set()
        assert controller.current_trajectory is None
    finally:
        release.set()
    wait_for_result(qtbot, controller)
    assert [settings.kinetic_energy_ev for settings in calls] == [200000., 275000.]
    assert [result.kinetic_energy_ev[0] for result in published] == [275000.]
    assert len(controller._cache) == 1


def test_new_snapshot_cancels_old_scene_and_invalidates_its_cache(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    first, second = UniformScene(), UniformScene((0., .02, 0.))
    calls = []

    def trace(scene, settings, **_kwargs):
        calls.append(scene)
        if scene is first:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(first)
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        controller.set_scene(second)
        assert controller.current_trajectory is None and not controller._cache
    finally:
        release.set()
    wait_for_result(qtbot, controller)
    assert calls == [first, second]
    assert controller._scene is second
    assert all(key[0] == controller._generation for key in controller._cache)


def test_hidden_controller_cancels_and_does_not_restart_until_visible(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def trace(scene, settings, **_kwargs):
        calls.append(settings)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        controller.show_controls()
        controller.set_active(False)
        assert controller.panel.isVisible()  # Dock visibility is independent of tracing.
        assert controller._worker.cancelled.is_set()
        controller.energy.setValue(250000.)
    finally:
        release.set()
    qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
    qtbot.wait(200)
    assert controller.current_trajectory is None and len(calls) == 1
    controller.set_active(True)
    assert wait_for_result(qtbot, controller).kinetic_energy_ev[0] == 250000.
    assert len(calls) == 2


def test_returning_to_cached_parameters_and_mode_reuses_exact_trajectory(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    controller.energy.setValue(250000.)
    wait_for_result(qtbot, controller)
    controller.energy.setValue(200000.)
    assert wait_for_result(qtbot, controller) is original
    controller.set_active(False)
    controller.set_active(True)
    assert wait_for_result(qtbot, controller) is original
    qtbot.wait(180)
    assert len(calls) == 2 and len(controller._cache) == 2


def test_cache_is_bounded_to_eight_parameter_states(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    for energy in range(200000, 210000, 1000):
        controller.energy.setValue(float(energy))
        wait_for_result(qtbot, controller)
    assert len(calls) == 10 and len(controller._cache) == 8
    assert {key[1].kinetic_energy_ev for key in controller._cache} == set(range(202000, 210000, 1000))


def test_error_is_visible_and_not_automatically_retried(controller, qtbot, monkeypatch):
    calls = []

    def trace(_scene, settings, **_kwargs):
        calls.append(settings)
        raise ValueError("Captured field values unavailable")

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    qtbot.waitUntil(lambda: "Captured field values unavailable" in controller.status_text, timeout=10000)
    qtbot.wait(350)
    assert len(calls) == 1 and controller._worker is None and not controller._timer.isActive()
    assert controller.current_trajectory is None
    controller.energy.setValue(230000.)
    qtbot.waitUntil(lambda: len(calls) == 2 and controller._worker is None, timeout=10000)


def test_real_uniform_field_domain_exit_and_budget_have_distinct_status(controller, qtbot):
    scene = UniformScene((0., 0., 0.))
    controller.set_scene(scene)
    controller.z.setValue(99.5)
    controller.length.setValue(2.)
    controller.set_active(True)
    result = wait_for_result(qtbot, controller)
    assert result.reason == "domain_exit"
    assert "Field validity boundary" in controller.status_text
    assert result.positions_m[-1, 2] <= scene.bounds_m[1, 2]
    controller.z.setValue(50.)
    controller.step.setValue(.00001)
    controller.max_steps.setValue(100)
    result = wait_for_result(qtbot, controller)
    assert result.reason == "step_limit"
    assert "truncated" in controller.status_text


def test_real_outside_start_is_visible_without_fabricated_motion(controller, qtbot):
    controller.set_scene(UniformScene())
    controller.z.setValue(150.)
    controller.set_active(True)
    result = wait_for_result(qtbot, controller)
    assert result.reason == "initial_outside_domain"
    assert "outside" in controller.status_text.lower()
    assert result.steps == 0 and len(result.positions_m) == 1
    assert result.path_length_m[-1] == 0.


def test_view_centre_and_slider_change_only_diagnostic_inputs(controller):
    controller.set_scene(UniformScene())
    scene = controller._scene
    controller.set_axial_range_mm(12., 30.)
    controller.start_at_view.click()
    assert controller.z.value() == 21.
    controller.sliders["energy"].setValue(1500)
    assert controller.energy.value() != 200000.
    assert controller._scene is scene and controller._worker is None


def test_actual_tip_defaults_keep_subev_energy_and_reset_all_initial_inputs(controller):
    scene = UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.))
    scene.default_path_length_m = 3.0264
    controller.set_scene(scene)
    assert controller.energy.suffix().strip() == "eV"
    assert controller.energy.value() == .3
    settings = controller.settings()
    assert settings.kinetic_energy_ev == .3
    assert settings.position_m == (0., 0., 0.)
    assert settings.max_path_length_m == pytest.approx(3.0264)
    controller.energy.setValue(1000.)
    controller.x.setValue(5.)
    controller.y.setValue(-3.)
    controller.z.setValue(50.)
    controller.polar.setValue(math.radians(50.) * 1000.)
    controller.azimuth.setValue(100.)
    controller.reset_to_tip.click()
    reset = controller.settings()
    assert reset.kinetic_energy_ev == .3 and reset.position_m == (0., 0., 0.)
    assert reset.polar_angle_deg == 0. and reset.azimuth_angle_deg == 0.
    assert controller._scene is scene


def test_electric_scene_is_lazy_and_reused_for_parameter_and_mode_changes(controller, qtbot, monkeypatch):
    prepared = install_scene_preparer(monkeypatch)
    traced = install_trace(monkeypatch)
    state = SimpleNamespace(source_energy_ev=.3, marker="unchanged")
    magnetic = UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.))
    controller.set_captured_scene(state, magnetic, (0., 100.))
    qtbot.wait(200)
    assert not prepared and not traced and controller._scene_worker is None
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    assert prepared == [(state, magnetic, (0., 100.))]
    assert traced[0][1].kinetic_energy_ev == .3
    controller.energy.setValue(.5)
    wait_for_result(qtbot, controller)
    controller.set_active(False)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    qtbot.wait(180)
    assert len(prepared) == 1 and len(traced) == 2
    assert vars(state) == {"source_energy_ev": .3, "marker": "unchanged"}


def test_stale_electric_preparation_never_traces_or_replaces_current_snapshot(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    old_state, latest_state = object(), object()
    old_scene, latest_scene = UniformScene(), UniformScene((0., .02, 0.))
    preparations = []
    traced = install_trace(monkeypatch)

    def prepare(state, magnetic_scene, *, z_limits_mm=None):
        preparations.append(state)
        if state is old_state:
            entered.set()
            assert release.wait(5.)
        return magnetic_scene

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", prepare)
    controller.set_captured_scene(old_state, old_scene, (0., 100.))
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert controller._worker is None
        controller.set_captured_scene(latest_state, latest_scene, (0., 100.))
        assert controller._scene_worker.cancelled.is_set()
        assert controller.current_trajectory is None
    finally:
        release.set()
    wait_for_result(qtbot, controller)
    assert preparations == [old_state, latest_state]
    assert [row[0] for row in traced] == [latest_scene]
    assert controller._scene is latest_scene


def test_hidden_electric_preparation_finishes_without_launching_particle(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    preparations = []
    traced = install_trace(monkeypatch)

    def prepare(state, magnetic_scene, *, z_limits_mm=None):
        preparations.append(state)
        if len(preparations) == 1:
            entered.set()
            assert release.wait(5.)
        return magnetic_scene

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", prepare)
    controller.set_captured_scene(object(), UniformScene(), (0., 100.))
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        controller.set_active(False)
        assert controller._scene_worker.cancelled.is_set()
    finally:
        release.set()
    qtbot.waitUntil(lambda: controller._scene_worker is None, timeout=10000)
    qtbot.wait(200)
    assert not traced and controller.current_trajectory is None
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    assert len(traced) == 1


def test_electric_preparation_error_is_visible_and_does_not_restart_forever(controller, qtbot, monkeypatch):
    calls = []
    traced = install_trace(monkeypatch)

    def prepare(*_args, **_kwargs):
        calls.append(True)
        raise ValueError("Captured electric field preparation failed")

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", prepare)
    controller.set_captured_scene(object(), UniformScene(), (0., 100.))
    controller.set_active(True)
    qtbot.waitUntil(lambda: "Captured electric field preparation failed" in controller.status_text, timeout=10000)
    qtbot.wait(350)
    assert len(calls) == 1 and not traced
    assert controller._scene_worker is None and controller._worker is None
    assert not controller._timer.isActive()


class FakeIsolatedExecution:
    """Exercise GUI process routing without computing or spawning a process."""

    def __init__(self):
        self.preparations = []
        self.traces = []
        self.alive = False
        self.identity = None
        self.fail_next_trace = False

    def prepare(self, state, magnetic_scene, *, z_limits_mm, cancelled):
        from temsim.test_electron_execution import RemoteElectronScene
        assert not cancelled()
        self.preparations.append((state, magnetic_scene, z_limits_mm))
        self.identity = f"synthetic-process-{len(self.preparations)}"
        self.alive = True
        bounds = ((-.01, -.01, 0.), (.01, .01, .1))
        return RemoteElectronScene(
            token=f"scene-{len(self.preparations)}", process_identity=self.identity,
            initial_position_m=(0., 0., 0.), initial_energy_ev=.3,
            default_path_length_m=.05, bounds_m=bounds, diagnostic_bounds_m=bounds,
            notes=("Synthetic metadata for GUI routing only.",))

    def owns_scene(self, scene):
        return self.alive and scene.process_identity == self.identity

    def trace(self, scene, settings, *, cancelled, **_kwargs):
        from temsim.test_electron_execution import ElectronExecutionError
        assert not cancelled()
        self.traces.append((scene, settings))
        if self.fail_next_trace:
            self.fail_next_trace = False
            self.alive = False
            raise ElectronExecutionError("Synthetic process exited")
        if not self.owns_scene(scene):
            raise ElectronExecutionError("Synthetic fields no longer owned")
        return synthetic_trajectory(settings)

    def close(self):
        self.alive = False


def install_fake_isolated_execution(controller, monkeypatch):
    from temsim.optics.model import State
    backend = FakeIsolatedExecution()
    controller._execution_backend = backend

    def unexpected_local_execution(*_args, **_kwargs):
        raise AssertionError("A captured State must use the isolated backend")

    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", unexpected_local_execution)
    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", unexpected_local_execution)
    # Only the real State type dispatch is under test. Its attributes must never
    # be read by the GUI because the fake backend owns this test's preparation.
    state = State.__new__(State)
    controller.set_captured_scene(state, None, (0., 100.))
    return backend, state


def test_captured_state_without_prepared_magnetic_scene_routes_to_isolated_backend(controller, qtbot, monkeypatch):
    backend, state = install_fake_isolated_execution(controller, monkeypatch)
    qtbot.wait(180)
    assert not backend.preparations and not backend.traces
    controller.set_active(True)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == .3
    assert backend.preparations == [(state, None, (0., 100.))]
    controller.energy.setValue(.7)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == .7
    assert len(backend.preparations) == 1 and len(backend.traces) == 2
    assert vars(state) == {}


def test_edit_after_isolated_process_loss_reprepares_captured_fields_and_retains_other_results(controller, qtbot, monkeypatch):
    backend, state = install_fake_isolated_execution(controller, monkeypatch)
    controller.set_active(True)
    retained = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.duplicate_electron()
    controller.energy.setValue(.7)
    wait_for_result(qtbot, controller)
    backend.alive = False
    controller.energy.setValue(.9)
    latest = wait_for_result(qtbot, controller)
    assert latest.kinetic_energy_ev[0] == .9
    assert backend.preparations == [(state, None, (0., 100.)), (state, None, (0., 100.))]
    assert len(backend.traces) == 3
    assert electron_record(controller, first_key).trajectory is retained
    show_overlay(controller)
    qtbot.wait(180)
    assert len(backend.traces) == 3 and len(controller.visible_paths()) == 2


def test_isolated_process_failure_does_not_retry_until_reactivation(controller, qtbot, monkeypatch):
    backend, _state = install_fake_isolated_execution(controller, monkeypatch)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    backend.fail_next_trace = True
    controller.energy.setValue(.7)
    qtbot.waitUntil(lambda: controller.selected_record.error is not None and controller._worker is None,
                    timeout=10000)
    qtbot.wait(250)
    assert "Synthetic process exited" in controller.status_text
    assert len(backend.preparations) == 1 and len(backend.traces) == 2
    assert not controller._timer.isActive()
    controller.set_active(False)
    controller.set_active(True)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[0] == .7 and not controller.selected_record.error
    assert len(backend.preparations) == 2 and len(backend.traces) == 3


def test_acceleration_result_shows_varying_energy_without_constant_energy_claim(controller, qtbot, monkeypatch):
    from dataclasses import replace

    def trace(_scene, settings, **_kwargs):
        return replace(synthetic_trajectory(settings),
            kinetic_energy_ev=np.array((.3, 120000.3)),
            electrostatic_potential_v=np.array((0., 120000.)),
            speed_m_per_s=np.array((324000., 1.76e8)),
            energy_invariant_error_ev=1e-6, energy_invariant_relative_error=1e-11)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.)))
    controller.set_active(True)
    result = wait_for_result(qtbot, controller)
    assert result.kinetic_energy_ev[-1] > result.kinetic_energy_ev[0]
    assert "120000.3" in controller.energy_status.text() and "energy" in controller.energy_status.text().lower()
    assert "120000.3" in controller.status_text
    labels = [label.text().lower() for label in controller.panel.findChildren(QLabel)]
    assert not any("energy stays constant" in value for value in labels)


def field_geometry(reference=1.):
    return SimpleNamespace(
        segments_m=np.array([[[0., 0., 0.], [0., 0., .1]]]),
        strengths_t=np.array([reference]),
        direction_segments_m=np.array([[[0., 0., .04], [0., 0., .06]]]),
        bounds_m=np.array([[-.001, -.001, 0.], [.001, .001, .1]]),
        reference_t=reference, line_count=1, seed_count=1, notes=("Synthetic field lines.",))


def test_added_electron_uses_tip_defaults_and_reuses_shared_prepared_field(controller, qtbot, monkeypatch):
    preparations = install_scene_preparer(monkeypatch)
    calls = install_trace(monkeypatch)
    scene = UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.))
    state = SimpleNamespace(marker="frozen capture")
    controller.set_captured_scene(state, scene, (0., 100.))
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.energy.setValue(.6)
    edited = wait_for_result(qtbot, controller)
    controller.x.setValue(.002)
    edited = wait_for_result(qtbot, controller)
    second_key = controller.add_electron()
    assert second_key and second_key != first_key
    assert wait_for_result(qtbot, controller) is original
    second = electron_record(controller, second_key)
    assert second.settings.kinetic_energy_ev == .3 and second.settings.position_m == (0., 0., 0.)
    first = electron_record(controller, first_key)
    assert first.settings.kinetic_energy_ev == .6 and first.trajectory is edited
    assert first.colour != second.colour
    assert controller._scene is scene and len(preparations) == 1 and len(calls) == 3
    assert vars(state) == {"marker": "frozen capture"}


def test_duplicate_reuses_exact_result_and_edits_only_its_own_record(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    first_result = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    second_key = controller.duplicate_electron()
    assert second_key != first_key
    assert wait_for_result(qtbot, controller) is first_result and len(calls) == 1
    controller.energy.setValue(275000.)
    second_result = wait_for_result(qtbot, controller)
    assert electron_record(controller, first_key).trajectory is first_result
    assert electron_record(controller, first_key).settings.kinetic_energy_ev == 200000.
    controller.select_electron(first_key)
    assert controller.current_trajectory is first_result and controller.energy.value() == 200000.
    controller.x.setValue(3.)
    updated_first = wait_for_result(qtbot, controller)
    assert updated_first is not first_result and len(calls) == 3
    assert electron_record(controller, second_key).trajectory is second_result
    assert electron_record(controller, second_key).settings.position_m == (1e-6, 0., .05)


def test_energy_edit_preserves_unedited_exact_tip_position_and_numerical_settings(controller):
    scene = UniformScene(initial_energy_ev=.3,
                         initial_position_m=(1.23456789123e-8, -2.34567891234e-8, .012345678912345))
    scene.default_path_length_m = .053456789123456
    controller.set_scene(scene)
    before = controller.settings()
    controller.energy.setValue(.6)
    after = controller.settings()
    assert after.kinetic_energy_ev == .6
    assert after.position_m == before.position_m
    assert after.max_path_length_m == before.max_path_length_m
    assert after.step_m == before.step_m and after.position_tolerance_m == before.position_tolerance_m
    duplicate_key = controller.duplicate_electron()
    controller.select_electron(controller.records[0].key)
    controller.select_electron(duplicate_key)
    assert controller.settings() == after


def test_selection_overlay_name_and_checkbox_change_presentation_without_retrace(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    second_key = controller.duplicate_electron()
    controller.energy.setValue(250000.)
    wait_for_result(qtbot, controller)
    identities = {r.key: (r.colour, r.revision, r.trajectory) for r in controller.records}
    assert [path.key for path in controller.visible_paths()] == [second_key]
    show_overlay(controller)
    assert {path.key for path in controller.visible_paths()} == {first_key, second_key}
    controller.select_electron(first_key)
    assert [path.key for path in controller.visible_paths() if path.selected] == [first_key]
    controller.name_editor.selectAll()
    qtbot.keyClicks(controller.name_editor, "Reference electron")
    assert electron_record(controller, first_key).label == "Reference electron"
    assert next(path for path in controller.visible_paths() if path.key == first_key).label == "Reference electron"
    electron_item(controller, second_key).setCheckState(0, Qt.CheckState.Unchecked)
    assert not electron_record(controller, second_key).checked
    assert [path.key for path in controller.visible_paths()] == [first_key]
    controller.display_mode.setCurrentIndex(controller.display_mode.findData("selected"))
    controller.select_electron(second_key)
    assert [path.key for path in controller.visible_paths()] == [second_key]
    qtbot.wait(200)
    assert len(calls) == 2
    for record in controller.records:
        colour, revision, result = identities[record.key]
        assert record.colour == colour and record.revision == revision and record.trajectory is result


def test_only_checked_overlay_records_are_queued_and_checking_one_reuses_others(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    first_key = controller.selected_record.key
    controller.energy.setValue(210000.)
    second_key = controller.add_electron()
    controller.energy.setValue(220000.)
    third_key = controller.add_electron()
    controller.energy.setValue(230000.)
    controller.set_checked(second_key, False)
    controller.select_electron(first_key)
    show_overlay(controller)
    controller.set_active(True)
    wait_for_records(qtbot, controller, (first_key, third_key))
    retained = {key: electron_record(controller, key).trajectory for key in (first_key, third_key)}
    assert sorted(settings.kinetic_energy_ev for _, settings in calls) == [210000., 230000.]
    assert electron_record(controller, second_key).trajectory is None
    controller.set_checked(second_key, True)
    wait_for_records(qtbot, controller, (first_key, second_key, third_key))
    assert len(calls) == 3 and calls[-1][1].kinetic_energy_ev == 220000.
    assert all(electron_record(controller, key).trajectory is result for key, result in retained.items())


def test_unchecking_running_overlay_record_cancels_it_and_continues_other_records(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls, published = [], []

    def trace(_scene, settings, **_kwargs):
        calls.append(settings.kinetic_energy_ev)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    first_key = controller.selected_record.key
    controller.energy.setValue(210000.)
    second_key = controller.add_electron()
    controller.energy.setValue(220000.)
    show_overlay(controller)
    controller.paths_changed.connect(lambda paths: published.extend(path.key for path in paths))
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        running_key = next(record.key for record in controller.records
                           if record.settings.kinetic_energy_ev == calls[0])
        remaining_key = next(key for key in (first_key, second_key) if key != running_key)
        controller.set_checked(running_key, False)
        assert controller._worker.cancelled.is_set()
    finally:
        release.set()
    wait_for_records(qtbot, controller, (remaining_key,))
    qtbot.wait(200)
    assert electron_record(controller, running_key).trajectory is None
    assert running_key not in published and len(calls) == 2
    assert [path.key for path in controller.visible_paths()] == [remaining_key]


def test_remove_running_last_record_rejects_stale_result_and_new_id_is_not_reused(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls, published = [], []

    def trace(_scene, settings, **_kwargs):
        result = synthetic_trajectory(settings)
        calls.append(result)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return result

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    removed_key = controller.selected_record.key
    controller.paths_changed.connect(lambda paths: published.extend(path.key for path in paths))
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        controller.remove_electron(removed_key)
        assert not controller.records and controller.selected_record is None
        assert controller.current_trajectory is None and not controller.visible_paths()
        assert controller._worker.cancelled.is_set()
        assert not controller.energy.isEnabled()
        new_key = controller.add_electron()
        assert new_key and new_key != removed_key
    finally:
        release.set()
    result = wait_for_result(qtbot, controller)
    assert len(calls) == 2 and result is calls[1] and result is not calls[0]
    assert removed_key not in published
    assert [record.key for record in controller.records] == [new_key]
    assert controller.energy.isEnabled()


def test_new_snapshot_invalidates_all_results_but_retains_electron_definitions(controller, qtbot, monkeypatch):
    preparations = install_scene_preparer(monkeypatch)
    calls = install_trace(monkeypatch)
    controller.set_captured_scene(object(), UniformScene(), (0., 100.))
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    second_key = controller.duplicate_electron()
    controller.energy.setValue(250000.)
    wait_for_result(qtbot, controller)
    definitions = {record.key: (record.settings, record.label, record.colour, record.trajectory)
                   for record in controller.records}
    show_overlay(controller)
    latest_scene = UniformScene((0., .02, 0.))
    controller.set_captured_scene(object(), latest_scene, (0., 100.))
    assert not controller._cache and not controller.visible_paths()
    assert all(record.trajectory is None for record in controller.records)
    wait_for_records(qtbot, controller, (first_key, second_key))
    assert len(preparations) == 2 and len(calls) == 4
    assert all(scene is latest_scene for scene, _settings in calls[-2:])
    for record in controller.records:
        settings, label, colour, result = definitions[record.key]
        assert (record.settings, record.label, record.colour) == (settings, label, colour)
        assert record.trajectory is not result


def test_record_error_does_not_block_overlay_queue_or_retry_until_that_record_changes(controller, qtbot, monkeypatch):
    calls = []

    def trace(_scene, settings, **_kwargs):
        calls.append(settings.kinetic_energy_ev)
        if settings.kinetic_energy_ev == 111000.:
            raise ValueError("Synthetic failure for this electron")
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    error_key = controller.selected_record.key
    controller.energy.setValue(111000.)
    good_key = controller.add_electron()
    controller.energy.setValue(222000.)
    show_overlay(controller)
    controller.set_active(True)
    qtbot.waitUntil(lambda: bool(electron_record(controller, error_key).error)
                    and electron_record(controller, good_key).trajectory is not None
                    and controller._worker is None, timeout=10000)
    original_good = electron_record(controller, good_key).trajectory
    qtbot.wait(350)
    assert sorted(calls) == [111000., 222000.] and not controller._timer.isActive()
    assert electron_record(controller, error_key).trajectory is None
    controller.select_electron(error_key)
    qtbot.wait(180)
    assert len(calls) == 2
    controller.energy.setValue(112000.)
    wait_for_records(qtbot, controller, (error_key, good_key))
    assert calls[-1] == 112000. and len(calls) == 3
    assert electron_record(controller, good_key).trajectory is original_good
    assert not electron_record(controller, error_key).error


def test_editing_other_record_does_not_cancel_running_job_or_bypass_its_coalescing_interval(controller, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def trace(_scene, settings, **_kwargs):
        calls.append(settings.kinetic_energy_ev)
        if settings.kinetic_energy_ev == 220000.:
            entered.set()
            assert release.wait(5.)
        return synthetic_trajectory(settings)

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    second_key = controller.duplicate_electron()
    show_overlay(controller)
    controller.energy.setValue(220000.)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        running_worker = controller._worker
        controller.select_electron(first_key)
        controller.energy.setValue(240000.)
        qtbot.wait(10)
        controller.energy.setValue(250000.)
        assert controller._worker is running_worker and not running_worker.cancelled.is_set()
        release.set()
        qtbot.wait(15)
        assert calls == [200000., 220000.]
        assert "220000" not in controller.energy_status.text()
        assert "220000" not in controller.status_text
    finally:
        release.set()
    wait_for_records(qtbot, controller, (first_key, second_key))
    assert calls == [200000., 220000., 250000.]
    assert electron_record(controller, second_key).trajectory.kinetic_energy_ev[0] == 220000.
    assert controller.current_trajectory.kinetic_energy_ev[0] == 250000.


def test_add_duplicate_and_remove_buttons_dispatch_the_selected_record(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.)))
    controller.set_active(True)
    initial = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.energy.setValue(.7)
    edited = wait_for_result(qtbot, controller)
    controller.duplicate_button.click()
    duplicate_key = controller.selected_record.key
    assert duplicate_key != first_key and controller.current_trajectory is edited
    controller.add_button.click()
    added_key = controller.selected_record.key
    assert added_key not in (first_key, duplicate_key)
    assert wait_for_result(qtbot, controller) is initial
    assert len(controller.records) == 3 and len(calls) == 2
    controller.remove_button.click()
    assert {record.key for record in controller.records} == {first_key, duplicate_key}
    assert controller.selected_record.key == duplicate_key
    assert controller.current_trajectory is edited


def test_saved_record_result_survives_shared_history_cache_eviction(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    original = wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    second_key = controller.duplicate_electron()
    for energy in range(201000, 211000, 1000):
        controller.energy.setValue(float(energy))
        wait_for_result(qtbot, controller)
    assert len(controller._cache) == 8
    assert not any(key[1].kinetic_energy_ev == 200000. for key in controller._cache)
    assert electron_record(controller, first_key).trajectory is original
    controller.select_electron(first_key)
    assert controller.current_trajectory is original
    qtbot.wait(200)
    assert len(calls) == 11
    assert electron_record(controller, second_key).trajectory.kinetic_energy_ev[0] == 210000.


def test_selected_display_does_not_calculate_dirty_hidden_records(controller, qtbot, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    first_key = controller.selected_record.key
    controller.energy.setValue(210000.)
    second_key = controller.add_electron()
    controller.energy.setValue(220000.)
    controller.select_electron(first_key)
    controller.set_active(True)
    first_result = wait_for_result(qtbot, controller)
    qtbot.wait(200)
    assert [settings.kinetic_energy_ev for _, settings in calls] == [210000.]
    assert electron_record(controller, second_key).trajectory is None
    controller.select_electron(second_key)
    second_result = wait_for_result(qtbot, controller)
    assert second_result.kinetic_energy_ev[0] == 220000. and len(calls) == 2
    controller.select_electron(first_key)
    assert controller.current_trajectory is first_result
    assert electron_record(controller, second_key).trajectory is second_result


def test_compact_electron_mode_reuses_snapshot_camera_and_field_settings(qtbot, monkeypatch):
    from temsim.gui.diagnostic_tabs import MagneticFieldView
    calls = install_trace(monkeypatch)
    prepared = install_scene_preparer(monkeypatch)
    built = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Electron changes must reuse the captured scene")

    def build(scene, **kwargs):
        built.append(scene)
        return field_geometry(kwargs["reference_t"])

    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", forbidden)
    monkeypatch.setattr("temsim.magnetic_field_lines.build_field_lines", build)
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.resize(900, 350)
    view.show()
    page = view.field_lines
    scene = UniformScene()
    state = SimpleNamespace(beam_voltage_kv=300., source_marker="unchanged")
    page.update_snapshot(state, (), (0., 100.), peak_t=.01, prepared_scene=scene)
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    try:
        original = wait_for_result(qtbot, page.electron)
        qtbot.waitUntil(lambda: page._worker is None and page._current_geometry is not None, timeout=10000)
        assert view.view_stack.currentWidget() is page
        assert page.electron.controls_button.isVisible() and not page.density.isVisible()
        assert page.canvas.height() >= view.height() * .8
        page.set_projection_angle(298.3)
        page.set_axial_range_mm(20., 80.)
        page.fit_button.click()
        page.density.setCurrentIndex(2)
        qtbot.waitUntil(lambda: len(built) == 2 and page._worker is None, timeout=10000)
        assert page.electron.current_trajectory is original and len(calls) == 1
        assert page.canvas.projection_angle_deg == 298.3
        np.testing.assert_allclose(page.canvas._axial_range_m, (.02, .08))
        page.electron.energy.setValue(250000.)
        wait_for_result(qtbot, page.electron)
        assert len(calls) == 2 and len(built) == 2
        assert len(prepared) == 1
        assert vars(state) == {"beam_voltage_kv": 300., "source_marker": "unchanged"}
        assert page.canvas.electron_point_count == 2
        view.display_mode.setCurrentIndex(view.display_mode.findData("3d"))
        assert not page.electron._active and page.canvas.electron_point_count == 0
        qtbot.wait(180)
        assert len(calls) == 2
        view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
        wait_for_result(qtbot, page.electron)
        assert page.canvas.electron_point_count == 2
        assert len(calls) == 2
        view.display_mode.setCurrentIndex(view.display_mode.findData("2d"))
        assert not page.electron._active and not page._active
        view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
        wait_for_result(qtbot, page.electron)
        qtbot.wait(180)
        assert len(calls) == 2 and len(built) == 2
        view.hide()
        assert not page.electron._active
    finally:
        page.set_active(False)
        qtbot.waitUntil(lambda: page._worker is None and page.electron._worker is None and page.electron._scene_worker is None, timeout=10000)


def test_pending_main_snapshot_clears_probe_and_preserves_user_parameters(qtbot, monkeypatch):
    from temsim.gui.diagnostic_tabs import MagneticFieldView
    calls = install_trace(monkeypatch)
    prepared = install_scene_preparer(monkeypatch)
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.show()
    page = view.field_lines
    page.electron.background.setChecked(False)
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=300.), (), (0., 100.),
                         prepared_scene=UniformScene())
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    try:
        wait_for_result(qtbot, page.electron)
        page.electron.energy.setValue(150000.)
        wait_for_result(qtbot, page.electron)
        view.mark_presentation_pending()
        assert page.electron._scene is None and page.electron.current_trajectory is None
        assert not page.electron._cache and page.electron.energy.value() == 150000.
        qtbot.wait(200)
        assert len(calls) == 2
        assert len(prepared) == 1
    finally:
        page.set_active(False)
        qtbot.waitUntil(lambda: page._worker is None and page.electron._worker is None and page.electron._scene_worker is None, timeout=10000)


def test_compact_view_overlays_checked_paths_and_restores_them_without_retrace(qtbot, monkeypatch):
    from temsim.gui.diagnostic_tabs import MagneticFieldView

    calls = install_trace(monkeypatch)
    preparations = install_scene_preparer(monkeypatch)
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.resize(900, 350)
    view.show()
    page = view.field_lines
    page.electron.background.setChecked(False)
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=300.), (), (0., 100.),
                         prepared_scene=UniformScene())
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    try:
        wait_for_result(qtbot, page.electron)
        first_key = page.electron.selected_record.key
        second_key = page.electron.duplicate_electron()
        page.electron.x.setValue(3.)
        wait_for_result(qtbot, page.electron)
        show_overlay(page.electron)
        assert page.canvas.electron_path_count == 2 and page.canvas.electron_point_count == 4
        assert page.canvas.height() >= view.height() * .8
        page.set_projection_angle(135.)
        page.set_axial_range_mm(20., 80.)
        page.electron.set_checked(second_key, False)
        assert page.canvas.electron_path_count == 1 and page.canvas.electron_point_count == 2
        assert [path.key for path in page.electron.visible_paths()] == [first_key]
        page.electron.set_checked(second_key, True)
        view.display_mode.setCurrentIndex(view.display_mode.findData("3d"))
        assert page.canvas.electron_path_count == 0
        view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
        qtbot.waitUntil(lambda: page.canvas.electron_path_count == 2, timeout=10000)
        qtbot.wait(200)
        assert len(calls) == 2 and len(preparations) == 1
        assert page.canvas.projection_angle_deg == 135.
        np.testing.assert_allclose(page.canvas._axial_range_m, (.02, .08))
    finally:
        page.set_active(False)
        qtbot.waitUntil(lambda: page._worker is None and page.electron._worker is None
                        and page.electron._scene_worker is None, timeout=10000)
