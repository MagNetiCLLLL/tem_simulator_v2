"""Replay a stored trajectory through real Qt navigation; never rerun physics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QFontDatabase, QWheelEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from temsim.gui.diagnostic_tabs import MagneticFieldView
from temsim.gui.test_electron_types import ElectronPath


def wheel(widget, position, delta=120):
    event = QWheelEvent(position, QPointF(widget.mapToGlobal(position.toPoint())),
                        QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    QApplication.processEvents()
    assert event.isAccepted()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="NPZ containing positions_m and time_s")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.trajectory, allow_pickle=False) as data:
        positions, times = data["positions_m"].copy(), data["time_s"].copy()
    original_digest = sha256(positions.tobytes() + times.tobytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font.is_file():
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families:
            app.setFont(QFont(families[0], 9))
    # Fail immediately if a presentation operation accidentally invokes physics.
    import temsim.magnetic_test_particle as particle
    import temsim.test_electron_scene as scene

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Trajectory replay must not run any field or particle calculation")

    particle.trace_test_electron = forbidden
    scene.prepare_test_electron_scene = forbidden
    window = QWidget()
    window.setWindowTitle("Stored electron trajectory — navigation verification")
    layout = QVBoxLayout(window)
    ray = pg.PlotWidget(background="#050816", title="Ray Diagram — stored trajectory replay")
    for name, title in (("bottom", "Axial position"), ("left", "Projected displacement")):
        ray.setLabel(name, title, units="m")
        ray.getAxis(name).setScale(1e-3)
    ray.getAxis("left").setWidth(112)
    ray.plot(positions[:, 2] * 1000., positions[:, 0] * 1000., pen=pg.mkPen("#f7ae79", width=2))
    transverse_extent_mm = max(float(np.max(np.abs(positions[:, 0]))) * 1100., 1e-6)
    ray.setRange(xRange=(0., float(positions[:, 2].max() * 1000.)),
                 yRange=(-transverse_extent_mm, transverse_extent_mm), padding=0)
    view = MagneticFieldView()
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    view.link_axial_axis(ray)
    canvas = view.field_lines.canvas
    canvas.set_field_lines_visible(False)
    path = ElectronPath("replay", "Stored electron", "#f7ae79", positions, times, True)
    layout.addWidget(ray, 1)
    layout.addWidget(view, 1)
    window.resize(1280, 1000)
    window.show()
    app.processEvents()
    # Show events refresh the live controller. Publish the detached replay
    # afterward, so this evidence actually contains the recorded trajectory.
    canvas.set_electron_paths((path,))
    view.field_lines.status.setText("Replay of a stored executed trajectory; no new calculation.")
    app.processEvents()
    assert len(canvas._electron_paths) == 1
    np.testing.assert_allclose(canvas.view_range_mm(), ray.viewRange())
    window.grab().save(str(args.output / "linked.png"))
    scale = 1.02 ** (120 * ray.getViewBox().state["wheelScaleFactor"])
    before = np.asarray(canvas.view_range_mm())
    wheel(canvas, canvas._plot_rect().center())
    np.testing.assert_allclose(np.diff(canvas.view_range_mm(), axis=1), np.diff(before, axis=1) * scale)
    np.testing.assert_allclose(canvas.view_range_mm(), ray.viewRange())
    plot = canvas._plot_rect()
    before = np.asarray(canvas.view_range_mm())
    wheel(canvas, QPointF(plot.center().x(), plot.bottom() + 10))
    after = np.asarray(canvas.view_range_mm())
    np.testing.assert_allclose(after[1], before[1])
    np.testing.assert_allclose(np.diff(after[0]), np.diff(before[0]) * scale)
    before = after
    wheel(canvas, QPointF(plot.left() - 10, plot.center().y()))
    after = np.asarray(canvas.view_range_mm())
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(np.diff(after[1]), np.diff(before[1]) * scale)
    view.field_lines.link_view.setChecked(False)
    source_range = np.asarray(ray.viewRange())
    wheel(canvas, plot.center(), -120)
    np.testing.assert_array_equal(ray.viewRange(), source_range)
    independent = canvas.view_range_mm()
    ray.setRange(xRange=(0., 100.), yRange=(-.002, .002), padding=0)
    np.testing.assert_array_equal(canvas.view_range_mm(), independent)
    window.resize(1000, 750)
    app.processEvents()
    np.testing.assert_array_equal(canvas.view_range_mm(), independent)
    window.grab().save(str(args.output / "independent.png"))
    view.field_lines.link_view.setChecked(True)
    np.testing.assert_allclose(canvas.view_range_mm(), ray.viewRange())
    after_digest = sha256(path.positions_m.tobytes() + path.time_s.tobytes()).hexdigest()
    assert len(canvas._electron_paths) == 1
    assert original_digest == after_digest
    evidence = {"success": True, "scope": "Qt replay of a stored executed trajectory; physics not rerun",
                "trajectory": str(args.trajectory.resolve()), "sample_count": len(positions),
                "physical_array_sha256": original_digest, "unchanged_physical_arrays": True,
                "plot_and_axis_wheels": True, "linked_independent_relinked": True,
                "resize_preserves_physical_range": True, "platform": app.platformName()}
    (args.output / "validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    window.close()
    app.processEvents()
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
