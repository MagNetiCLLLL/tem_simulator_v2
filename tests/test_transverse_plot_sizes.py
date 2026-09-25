"""Fixed picture geometry uses cached display fixtures, never transport."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QDialogButtonBox

from test_beam_analysis_modes import view
from test_transverse_source_tracking import recorded_result, preset
from temsim.gui.transverse_plot_layout import TransversePlotLayout


SIZES = {"beam": [560, 440], "legend": [260, 220]}


def assert_sizes(view, sizes=SIZES):
    for key, widget in (("beam", view.source_plot.plot), ("beam", view.plot),
                        ("legend", view.angle_colour_wheel)):
        assert [widget.width(), widget.height()] == sizes[key]
        assert widget.minimumSize() == widget.maximumSize()
    assert view.initial_beam_panel.width() == view.section_beam_panel.width()


def test_default_source_and_plane_pictures_have_equal_fixed_sizes(view, qtbot):
    qtbot.wait(10)
    assert_sizes(view, {key: list(value) for key, value in TransversePlotLayout.DEFAULTS.items()})
    assert view.source_plot.plot.getViewBox().size() == view.plot.getViewBox().size()


def test_both_bottom_axis_titles_fit_inside_the_fixed_picture(view, qtbot):
    view.display_result(recorded_result())
    view.set_projection_angle(30.)
    qtbot.wait(10)
    for plot in (view.source_plot.plot, view.plot):
        label = plot.getAxis("bottom").label
        assert label.mapRectToScene(label.boundingRect()).bottom() <= plot.viewport().height()


def test_wrapped_captions_never_overlap_fixed_plots_across_modes_and_sizes(view, qtbot):
    result = recorded_result(49)
    for branch in (result.simulation.incident, *result.simulation.branches.values()):
        branch.flight_time_s = branch.z[:, None] * np.linspace(1., 1.2, 49)[None, :] * 1e-9
    view.display_result(result)
    view.resize(350, 500)
    for size in ([400, 360], [460, 330]):
        view.set_plot_size_state({"beam": size})
        for mode in ("source", "emission_direction", "tof", "advanced"):
            preset(view, mode)
            if mode == "advanced":
                view.colour_mode.setCurrentIndex(view.colour_mode.findData("emission_angle"))
            qtbot.wait(20)
            for plot, caption in ((view.source_plot.plot, view.source_plot.summary), (view.plot, view.summary)):
                canvas = QRect(plot.mapToGlobal(QPoint()), plot.size())
                text = QRect(caption.mapToGlobal(QPoint()), caption.size())
                title = plot.getAxis("bottom").label
                scene_title = title.mapRectToScene(title.boundingRect()).toAlignedRect()
                title_rect = QRect(plot.viewport().mapToGlobal(scene_title.topLeft()), scene_title.size())
                assert text.top() > canvas.bottom()
                assert not text.intersects(title_rect)
                assert canvas.contains(title_rect)
            assert not view.plot_layout._height_timer.isActive()


def test_sizes_survive_window_changes_modes_and_new_results(view, qtbot):
    result = recorded_result()
    original = result.simulation.incident.x.copy()
    view.set_plot_size_state(SIZES)
    view.display_result(result)
    for size in ((350, 500), (1000, 1200)):
        view.resize(*size)
        for mode in ("source", "emission_direction", "tof", "advanced"):
            preset(view, mode)
            view.focus_z(1.5)
            qtbot.wait(10)
            assert_sizes(view)
            assert view.source_plot.plot.getViewBox().size() == view.plot.getViewBox().size()
    view.hide()
    view.display_result(recorded_result())
    view.show()
    qtbot.wait(10)
    assert_sizes(view)
    np.testing.assert_array_equal(result.simulation.incident.x, original)


def test_large_pictures_scroll_without_enlarging_window_or_hiding_controls(view, qtbot):
    view.set_plot_size_state(SIZES)
    view.resize(350, 500)
    qtbot.wait(20)
    assert view.width() == 350
    assert view.height() == 500
    scroll = view.plot_layout.scroll
    assert scroll.horizontalScrollBar().maximum() > 0
    assert scroll.verticalScrollBar().maximum() > 0
    assert view.plot_layout.button.isVisible()
    scroll.ensureWidgetVisible(view.plot)
    assert_sizes(view)


def test_dialog_applies_shared_size_and_cancel_preserves_applied_values(view, qtbot):
    qtbot.mouseClick(view.plot_layout.button, Qt.MouseButton.LeftButton)
    controller = view.plot_layout
    dialog = controller.dialog
    buttons = dialog.findChild(QDialogButtonBox)
    assert set(controller.editors) == {"beam", "legend"}
    for key, size in SIZES.items():
        for editor, value in zip(controller.editors[key], size):
            editor.setValue(value)
    with qtbot.waitSignal(view.plot_sizes_changed):
        qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Apply), Qt.MouseButton.LeftButton)
    assert view.plot_size_state() == SIZES
    assert_sizes(view)
    controller.editors["beam"][0].setValue(800)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert controller.dialog is None
    assert view.plot_size_state() == SIZES


def test_dialog_reset_requires_confirmation_and_state_is_detached(view, qtbot):
    view.set_plot_size_state(SIZES)
    state = view.plot_size_state()
    state["beam"][0] = 999
    assert view.plot_size_state() == SIZES
    view.plot_layout.open_dialog()
    buttons = view.plot_layout.dialog.findChild(QDialogButtonBox)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults), Qt.MouseButton.LeftButton)
    assert view.plot_size_state() == SIZES
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert view.plot_size_state() == {k: list(v) for k, v in TransversePlotLayout.DEFAULTS.items()}


@pytest.mark.parametrize("bad", [None, "bad", [1], [True, 200], [400, float("nan")], [400, 99999]])
def test_invalid_saved_entry_falls_back_without_losing_other_sizes(view, bad):
    view.set_plot_size_state({**SIZES, "beam": bad})
    expected = {**SIZES, "beam": [400, 360]}
    assert view.plot_size_state() == expected
    assert_sizes(view, expected)


def test_obsolete_independent_sizes_cannot_split_the_two_pictures(view):
    view.set_plot_size_state({"source": [560, 270], "plane": [720, 440], "legend": SIZES["legend"]})
    expected = {**SIZES, "beam": [400, 360]}
    assert view.plot_size_state() == expected
    assert_sizes(view, expected)


def test_legend_resizing_scales_drawing_and_preserves_circular_geometry(view):
    wheel = view.angle_colour_wheel
    wheel.show()
    view.set_plot_size_state({**SIZES, "legend": [368, 368]})
    doubled = wheel.grab().toImage()
    view.set_plot_size_state({**SIZES, "legend": [468, 368]})
    wide = wheel.grab().toImage()
    # The wider canvas adds centred padding instead of stretching angles.
    for x, y in ((276, 170), (100, 170), (185, 83)):
        assert wide.pixelColor(x + 50, y) == doubled.pixelColor(x, y)
    assert doubled.pixelColor(276, 170).saturationF() > 0.5
