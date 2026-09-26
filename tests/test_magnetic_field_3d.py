"""Lazy GUI publication and camera reuse with synthetic field-line geometry."""
from types import SimpleNamespace
from threading import Event

import numpy as np

from temsim.gui.magnetic_field_3d import MagneticField3DPage


def geometry(reference=1.):
    return SimpleNamespace(
        segments_m=np.array([[[0., 0., 0.], [0., 0., .01]]]),
        strengths_t=np.array([reference]),
        direction_segments_m=np.array([[[0., 0., .004], [0., 0., .006]]]),
        bounds_m=np.array([[-.001, -.001, 0.], [.001, .001, .01]]),
        reference_t=reference, line_count=1, seed_count=1,
        notes=("Synthetic field for GUI tests.",))


def records():
    return (SimpleNamespace(key="lens", name="Magnetic lens", peak_t=1.),)


def install_builder(monkeypatch):
    calls = []

    def prepare(state, **kwargs):
        calls.append((state, kwargs))
        return state

    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", prepare)
    monkeypatch.setattr("temsim.magnetic_field_lines.build_field_lines",
                        lambda _scene, **kwargs: geometry(kwargs["reference_t"]))
    return calls


def test_3d_is_lazy_and_camera_and_mode_reuse_never_sample_fields(qtbot, monkeypatch):
    calls = install_builder(monkeypatch)
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.show()
    state = object()
    page.update_snapshot(state, records(), (0., 10.))
    qtbot.wait(120)
    assert not calls
    page.set_active(True)
    qtbot.waitUntil(lambda: page._current_geometry is not None, timeout=10000)
    assert len(calls) == 1
    page.set_projection_angle(298.3)
    page.set_axial_range_mm(2., 8.)
    page.fit_button.click()
    page.set_active(False)
    page.set_active(True)
    qtbot.wait(130)
    assert len(calls) == 1
    assert calls[0][1] == {"z_limits_mm": (0., 10.)}
    assert page.canvas.projection_angle_deg == 298.3
    assert not hasattr(page, "source") and not hasattr(page, "camera")
    assert page._current_geometry is not None


def test_reference_is_fixed_across_new_snapshot_and_old_scene_is_cleared(qtbot, monkeypatch):
    calls = install_builder(monkeypatch)
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.set_active(True)
    page.update_snapshot(object(), records(), (0., 10.))
    qtbot.waitUntil(lambda: page._current_geometry is not None, timeout=10000)
    page.update_snapshot(object(), (SimpleNamespace(key="lens", name="Lens", peak_t=2.),), (0., 10.))
    assert page._current_geometry is None
    assert page.reference.value() == 1.
    qtbot.waitUntil(lambda: len(calls) == 2 and page._current_geometry is not None, timeout=10000)
    page.match_reference.click()
    qtbot.waitUntil(lambda: page._current_geometry is not None and page._current_geometry.reference_t == 2., timeout=10000)
    page.invalidate()
    assert page._current_geometry is None and page._state is None
    assert "pending" in page.status.text()


def test_stale_inflight_result_never_replaces_latest_snapshot(qtbot, monkeypatch):
    entered, release = Event(), Event()
    first, latest = object(), object()
    calls = []

    def prepare(state, **kwargs):
        calls.append(state)
        if state is first:
            entered.set()
            assert release.wait(5.)
        return state

    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", prepare)
    monkeypatch.setattr("temsim.magnetic_field_lines.build_field_lines",
                        lambda _scene, **kwargs: geometry(kwargs["reference_t"]))
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.set_active(True)
    page.update_snapshot(first, records(), (0., 10.))
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        page.update_snapshot(latest, records(), (0., 10.))
        assert page._current_geometry is None
    finally:
        release.set()
    qtbot.waitUntil(lambda: page._worker is None and page._current_geometry is not None, timeout=10000)
    assert calls == [first, latest]
    assert all(key[0] == page._generation for key in page._cache)


def test_provider_error_is_visible_without_old_geometry(qtbot, monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("Captured magnetic map is unavailable")

    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", fail)
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.update_snapshot(object(), records(), (0., 10.))
    page.set_active(True)
    qtbot.waitUntil(lambda: "map is unavailable" in page.status.text(), timeout=10000)
    assert page._current_geometry is None


def test_panel_switch_preserves_2d_plot_and_hides_2d_only_controls(qtbot):
    from temsim.gui.diagnostic_tabs import MagneticFieldView
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.resize(1200, 700)
    view.show()
    plot = view.plot
    view.plot.plot([0., 1.], [2., 3.])
    view.plot.setRange(xRange=(.2, .6), yRange=(2.1, 2.5), padding=0)
    qtbot.wait(10)
    view.display_mode.setCurrentIndex(1)
    assert view.view_stack.currentWidget() is view.field_lines
    assert not hasattr(view, "show_individual")
    assert not view.field_map_import.isVisible()
    assert view.field_lines._active
    view.display_mode.setCurrentIndex(0)
    assert view.view_stack.currentWidget() is plot
    assert not view.field_map_import.isVisible()
    assert view.field_lines.density.isVisible() is False
    np.testing.assert_allclose(view.plot.viewRange(), ((.2, .6), (2.1, 2.5)))
    assert not view.field_lines._active


def test_ray_projection_and_axial_range_follow_in_hidden_and_visible_field(qtbot, monkeypatch):
    from temsim.gui.visualization import VisualizationWorkspace
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    workspace.resize(1200, 850)
    workspace.show()
    field = workspace.magnetic_field
    # A camera operation must not request a new captured field or particle run.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unexpected field rebuild")
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", forbidden)
    monkeypatch.setattr("temsim.magnetic_field_lines.build_field_lines", forbidden)
    workspace._set_projection_angle(298.3)
    assert field.field_lines.canvas.projection_angle_deg == 298.3
    workspace.magnetic_field_toggle.setChecked(True)
    field.display_mode.setCurrentIndex(1)
    workspace.projection_yz.click()
    assert field.field_lines.canvas.projection_angle_deg == 90.
    workspace.plot.setXRange(1200., 1800., padding=0.)
    qtbot.wait(30)
    np.testing.assert_allclose(field.field_lines.view_range_mm(), workspace.plot.viewRange())
    workspace.magnetic_field_toggle.setChecked(False)
    workspace._set_projection_angle(35.)
    assert field.field_lines.canvas.projection_angle_deg == 35.


def test_compact_field_controls_leave_most_height_for_plot(qtbot):
    from temsim.gui.diagnostic_tabs import MagneticFieldView
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.resize(900, 350)
    view.show()
    view.display_mode.setCurrentIndex(1)
    qtbot.wait(30)
    assert view.field_lines.canvas.height() >= view.height() * .8
    assert not view.field_map_import.isVisible()
    assert not view.field_lines.reference.isVisible()
    assert view.field_lines.density.isVisible()
    assert view.field_lines.fit_button.isVisible()
    assert view.display_mode.geometry().center().y() == view.field_lines.density.geometry().center().y()


def test_3d_reuses_already_prepared_combined_scene(qtbot, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Captured profile scene must be reused")
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", forbidden)
    scenes = []
    def build(scene, **kwargs):
        scenes.append(scene)
        return geometry(kwargs["reference_t"])
    monkeypatch.setattr("temsim.magnetic_field_lines.build_field_lines", build)
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    scene = SimpleNamespace(source_regions=(), source_categories=("lens", "stigmator", "deflector"))
    page.update_snapshot(object(), records(), (0., 10.), peak_t=2., prepared_scene=scene)
    page.set_active(True)
    qtbot.waitUntil(lambda: page._current_geometry is not None, timeout=10000)
    assert page.reference.value() == 2.
    page.density.setCurrentIndex(2)
    qtbot.waitUntil(lambda: len(scenes) == 2 and page._worker is None, timeout=10000)
    assert scenes == [scene, scene]
    page.invalidate()
    assert page._prepared_scene is None


def test_physical_navigation_page_interface_never_retraces_electron_or_field(qtbot, monkeypatch):
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.resize(850, 500)
    page.show()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Physical view navigation must not request a particle or field calculation")

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", forbidden)
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", forbidden)
    emitted = []
    page.view_range_changed.connect(lambda x, y: emitted.append((x, y)))
    ranges = ((200., 900.), (-.001, .002))
    page.set_view_range_mm(*ranges)
    assert page.view_range_mm() == ranges
    assert not emitted
    assert page.link_view.isChecked()
    page.link_view.setChecked(False)
    page.set_electron_mode(True)
    page.set_electron_mode(False)
    page.invalidate()
    assert page.view_range_mm() == ranges
    page.set_view_range_mm((250., 800.), (-.001, .003), emit=True)
    assert emitted == [page.view_range_mm()]
    page.fit_button.click()
    assert len(emitted) == 2
    assert page._worker is None


def test_navigation_keeps_view_centre_action_current_without_moving_electron(qtbot, monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    from test_magnetic_test_electron_gui import UniformScene

    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.resize(850, 500)
    page.show()
    page.electron.set_scene(UniformScene())
    original = page.electron.selected_record.settings

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Navigation must not trace or replace an electron")

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", forbidden)
    page.set_view_range_mm((10., 30.), (-.001, .002))
    assert page.electron.start_at_view.isEnabled()
    assert page.electron._view_limits_mm == (10., 30.)
    point = page.canvas.axis_rect("bottom").center() + QPointF(50., 0.)
    event = QWheelEvent(point, page.canvas.mapToGlobal(point), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(page.canvas, event)
    assert page.electron._view_limits_mm == page.view_range_mm()[0]
    assert page.electron._view_limits_mm != (10., 30.)
    page.fit_button.click()
    assert page.electron._view_limits_mm == page.view_range_mm()[0]
    assert page.electron.selected_record.settings is original
    page.set_view_range_mm((20., 40.), (-.001, .002))
    page.electron.start_at_view.click()
    np.testing.assert_allclose(page.electron.selected_record.settings.position_m,
                               (original.position_m[0], original.position_m[1], .03))
    assert page.electron._worker is None
