"""Real Qt controls and mouse drags for the axisymmetric part editor."""

from copy import deepcopy

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from temsim.gui.part_geometry_editor import GeometryEditorDialog


@pytest.fixture
def part():
    return {
        "key": "condenser_lens_1_excitation_coil",
        "name": "C1 excitation coil",
        "mechanical_profile": "magnetic_excitation_coil",
        "local_start_z_mm": 20.0,
        "local_center_z_mm": 110.0,
        "local_end_z_mm": 200.0,
        "length_mm": 180.0,
        "mechanical_inner_diameter_mm": 60.0,
        "mechanical_outer_diameter_mm": 100.0,
        "vacuum_inner_diameter_mm": 20.0,
        "material_class": "copper_winding",
    }


@pytest.fixture
def dialog_factory(qtbot, part):
    def create(*, source=None, callback=None, neighbours=()):
        calls = []
        dialog = GeometryEditorDialog(
            part if source is None else source,
            calls.append if callback is None else callback,
            neighbours=neighbours,
        )
        qtbot.addWidget(dialog)
        dialog.show()
        QApplication.processEvents()
        qtbot.wait(30)
        dialog.fit_view()
        QApplication.processEvents()
        return dialog, calls

    return create


def _number(qtbot, dialog, name, value):
    control = dialog.dimension_controls[name]
    control.setFocus()
    control.lineEdit().selectAll()
    qtbot.keyClicks(control.lineEdit(), str(value))
    qtbot.keyClick(control.lineEdit(), Qt.Key.Key_Return)
    QApplication.processEvents()


def _drag(qtbot, dialog, handle_name, destination, during=None):
    handle = dialog.handles[handle_name]
    start = dialog.plot.mapFromScene(handle.mapToScene(QPointF(0, 0)))
    end = dialog.plot.mapFromScene(
        dialog.plot.getViewBox().mapViewToScene(QPointF(*destination))
    )
    viewport = dialog.plot.viewport()
    QApplication.sendEvent(viewport, QMouseEvent(
        QEvent.Type.MouseMove, QPointF(start),
        QPointF(viewport.mapToGlobal(start)), Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    ))
    QApplication.processEvents()
    qtbot.wait(30)
    qtbot.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    for fraction in (0.35, 0.7, 1.0):
        position = QPoint(
            round(start.x() + (end.x() - start.x()) * fraction),
            round(start.y() + (end.y() - start.y()) * fraction),
        )
        # Offscreen QTest mouseMove does not retain the pressed-button state.
        # Deliver the real Qt mouse event to the viewport with LeftButton held.
        QApplication.sendEvent(viewport, QMouseEvent(
            QEvent.Type.MouseMove, QPointF(position),
            QPointF(viewport.mapToGlobal(position)), Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        ))
        QApplication.processEvents()
        qtbot.wait(20)
        if during is not None:
            during()
    qtbot.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    QApplication.processEvents()


def test_initial_dimensions_and_read_only_metadata_preserve_coil_180(dialog_factory):
    dialog, calls = dialog_factory()
    assert dialog.geometry.length_mm == 180.0
    assert dialog.dimension_controls["length_mm"].value() == 180.0
    assert dialog.dimension_controls["thickness_mm"].value() == 20.0
    assert all(control.suffix() == " mm" for control in dialog.dimension_controls.values())
    assert "2D axisymmetric" in dialog.windowTitle()
    assert dialog.center_label.text() == "110 mm"
    assert dialog.material_label.text() == "copper_winding"
    assert dialog.vacuum_label.text() == "Ø 20 mm"
    assert calls == []
    assert not dialog.apply_button.isEnabled()


def test_numeric_length_updates_geometry_and_handles_without_rounding_other_dimensions(
    qtbot, dialog_factory, part
):
    part["mechanical_outer_diameter_mm"] = 91.98322635
    before = deepcopy(part)
    dialog, calls = dialog_factory()

    _number(qtbot, dialog, "length_mm", 225)

    assert dialog.geometry.length_mm == 225
    assert dialog.geometry.center_z_mm == 110
    assert dialog.geometry.start_z_mm == -2.5
    assert dialog.geometry.end_z_mm == 222.5
    assert dialog.handles["start"].pos().x() == -2.5
    assert dialog.handles["end"].pos().x() == 222.5
    assert dialog.geometry.outer_diameter_mm == 91.98322635
    assert part == before
    assert calls == []


def test_radial_thickness_links_diameters_with_both_anchor_choices(qtbot, dialog_factory):
    dialog, _ = dialog_factory()
    _number(qtbot, dialog, "thickness_mm", 25)
    assert dialog.geometry.inner_diameter_mm == 60
    assert dialog.geometry.outer_diameter_mm == 110
    assert dialog.handles["outer_top"].pos().y() == 55
    assert dialog.handles["outer_bottom"].pos().y() == -55

    dialog.thickness_anchor.setCurrentIndex(dialog.thickness_anchor.findData("outer"))
    _number(qtbot, dialog, "thickness_mm", 15)
    assert dialog.geometry.outer_diameter_mm == 110
    assert dialog.geometry.inner_diameter_mm == 80
    assert dialog.dimension_controls["inner_diameter_mm"].value() == 80
    assert dialog.handles["inner_top"].pos().y() == 40
    assert dialog.handles["inner_bottom"].pos().y() == -40
    assert dialog.geometry.vacuum_inner_diameter_mm == 20


@pytest.mark.parametrize(
    "handle,destination,dimension,expected",
    [
        ("start", (0, 0), "length_mm", 220),
        ("end", (220, 0), "length_mm", 220),
        ("outer_top", (110, 60), "outer_diameter_mm", 120),
        ("inner_bottom", (110, -40), "inner_diameter_mm", 80),
    ],
)
def test_real_mouse_drag_previews_dimensions_and_is_one_undo_step(
    qtbot, dialog_factory, handle, destination, dimension, expected
):
    dialog, calls = dialog_factory()
    original = dialog.geometry
    history_during = []
    pixel_mm = max(dialog.plot.getViewBox().viewPixelSize())

    _drag(qtbot, dialog, handle, destination, during=lambda: history_during.append(dialog._history_index))

    assert getattr(dialog.geometry, dimension) == pytest.approx(expected, abs=3 * pixel_mm)
    assert dialog.geometry.center_z_mm == original.center_z_mm
    assert dialog.geometry.center_fraction == original.center_fraction
    assert dialog.dimension_controls[dimension].value() == pytest.approx(getattr(dialog.geometry, dimension))
    assert dialog.handles["inner_top"].pos().y() == pytest.approx(-dialog.handles["inner_bottom"].pos().y())
    assert dialog.handles["outer_top"].pos().y() == pytest.approx(-dialog.handles["outer_bottom"].pos().y())
    assert history_during == [0, 0, 0]
    assert dialog._history_index == 1
    assert calls == []
    dragged = dialog.geometry
    qtbot.mouseClick(dialog.undo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == original
    qtbot.mouseClick(dialog.redo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == dragged


def test_apply_uses_changed_dimensions_and_refreshes_baseline(qtbot, dialog_factory, part):
    dialog, calls = dialog_factory()
    _number(qtbot, dialog, "length_mm", 200)
    assert calls == []
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert calls == [{
        ("parts", part["key"], "length_mm"): 200.0,
        ("parts", part["key"], "local_start_z_mm"): 10.0,
        ("parts", part["key"], "local_end_z_mm"): 210.0,
    }]
    assert dialog.isVisible()
    assert not dialog.apply_button.isEnabled()
    _number(qtbot, dialog, "outer_diameter_mm", 112)
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert calls[-1] == {("parts", part["key"], "mechanical_outer_diameter_mm"): 112.0}
    assert not any("vacuum" in path[-1] for update in calls for path in update)


def test_apply_failure_keeps_dialog_and_draft_for_retry(qtbot, dialog_factory):
    attempts = []

    def callback(updates):
        attempts.append(updates)
        if len(attempts) == 1:
            raise ValueError("Part overlaps its neighbouring yoke")

    dialog, _ = dialog_factory(callback=callback)
    _number(qtbot, dialog, "length_mm", 200)
    draft = dialog.geometry
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert dialog.isVisible()
    assert dialog.geometry == draft
    assert dialog.error_label.isVisible()
    assert "overlaps its neighbouring yoke" in dialog.error_label.text()
    assert dialog.apply_button.isEnabled()
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert attempts[1] == attempts[0]
    assert dialog.geometry == draft
    assert not dialog.error_label.isVisible()
    assert not dialog.apply_button.isEnabled()


def test_revert_restores_last_applied_geometry_and_close_discards_pending_changes(qtbot, dialog_factory):
    dialog, calls = dialog_factory()
    _number(qtbot, dialog, "length_mm", 200)
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    applied = dialog.geometry
    _number(qtbot, dialog, "outer_diameter_mm", 120)
    qtbot.mouseClick(dialog.revert_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == applied
    assert len(calls) == 1
    _number(qtbot, dialog, "length_mm", 220)
    qtbot.mouseClick(dialog.close_button, Qt.MouseButton.LeftButton)
    assert not dialog.isVisible()
    assert dialog.geometry == applied
    assert len(calls) == 1


def test_numeric_edit_history_supports_undo_redo(qtbot, dialog_factory):
    dialog, _ = dialog_factory()
    original = dialog.geometry
    _number(qtbot, dialog, "length_mm", 200)
    one = dialog.geometry
    _number(qtbot, dialog, "outer_diameter_mm", 120)
    two = dialog.geometry
    qtbot.mouseClick(dialog.undo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == one
    qtbot.mouseClick(dialog.undo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == original
    qtbot.mouseClick(dialog.redo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == one
    qtbot.mouseClick(dialog.redo_button, Qt.MouseButton.LeftButton)
    assert dialog.geometry == two


def test_invalid_dimension_is_inline_and_cannot_apply(qtbot, dialog_factory):
    dialog, calls = dialog_factory()
    original = dialog.geometry
    _number(qtbot, dialog, "inner_diameter_mm", 120)
    assert dialog.geometry == original
    assert dialog.error_label.isVisible()
    assert not dialog.apply_button.isEnabled()
    assert calls == []
    qtbot.mouseClick(dialog.revert_button, Qt.MouseButton.LeftButton)
    assert not dialog.error_label.isVisible()
    assert dialog.dimension_controls["inner_diameter_mm"].value() == 60


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_fixed_centre_endpoint_handle_is_disabled(qtbot, dialog_factory, part, fraction):
    part["local_center_z_mm"] = part["local_start_z_mm"] if fraction == 0 else part["local_end_z_mm"]
    dialog, _ = dialog_factory()
    fixed = "start" if fraction == 0 else "end"
    moving = "end" if fraction == 0 else "start"
    assert dialog.handles[fixed].movable is False
    assert dialog.handles[moving].movable is True
    _number(qtbot, dialog, "length_mm", 200)
    assert dialog.geometry.center_z_mm == part["local_center_z_mm"]
    assert dialog.handles[fixed].pos().x() == part["local_center_z_mm"]


def test_dialog_fits_minimum_size_and_ignores_unsupported_neighbour(dialog_factory):
    dialog, _ = dialog_factory(neighbours=[{"key": "unsupported"}])
    dialog.resize(900, 600)
    QApplication.processEvents()
    assert dialog.width() == 900
    assert dialog.height() == 600
    assert dialog.plot.width() >= 480
    assert dialog.apply_button.geometry().bottom() <= dialog.height()
