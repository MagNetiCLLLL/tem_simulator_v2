"""PyQtGraph views for resolved TEM mechanics and axial magnetic fields."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import (
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from temsim.component_keys import (
    CONDENSER_LENS_1_LOWER_POLE,
    CONDENSER_LENS_2_UPPER_POLE,
)
from temsim.diagnostics import (
    lens_field_records,
    physical_layout_records,
    vacuum_bore_plot_points,
)


BUTTON_STYLE = """
    QPushButton {
        min-height: 28px;
        padding: 4px 12px;
        border: 1px solid #64748b;
        border-radius: 6px;
        background: #f8fafc;
        color: #0f172a;
        font-size: 13px;
        font-weight: 600;
    }
    QPushButton:hover { background: #e2e8f0; border-color: #334155; }
    QPushButton:checked { background: #2563eb; color: white; }
"""


def _component_colour(record) -> str:
    text = f"{record.kind} {record.profile}".lower()
    if record.key == "sample":
        return "#fb7185"
    if "aperture" in text or "slit" in text:
        return "#fbbf24"
    if "deflector" in text:
        return "#2dd4bf"
    if "excitation_coil" in text:
        return "#f97316"
    if "lens_housing" in text:
        return "#64748b"
    if "lens_yoke" in text:
        return "#2563eb"
    if any(word in text for word in ("stigmator", "quadrupole", "hexapole")):
        return "#c084fc"
    if "lens" in text or "pole_piece" in text:
        return "#60a5fa"
    if any(word in text for word in ("detector", "camera", "screen")):
        return "#4ade80"
    return "#94a3b8"


class PhysicalLayoutView(QWidget):
    component_selected = Signal(str)
    axial_position_selected = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result = None
        self._records = ()
        self._record_by_key = {}
        self._highlight = None
        self._design_reference_items = []
        self._vacuum_liner_items = []
        self._c1_c2_pole_gap = None

        self.heading = QLabel("Resolved mechanical layout")
        self.summary = QLabel(
            "Hollow-cylinder projections and vacuum bores come from the active TOML assembly."
        )
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.fit_all = QPushButton("Fit all hardware")
        self.fit_bore = QPushButton("Fit vacuum bore")
        for button in (self.fit_all, self.fit_bore):
            button.setStyleSheet(BUTTON_STYLE)

        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.fit_bore)
        header.addWidget(self.fit_all)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("physicalLayoutPlot")
        self.plot.setLabel("bottom", "Axial position", units="mm")
        self.plot.setLabel("left", "Mechanical radius", units="mm")
        self.plot.showGrid(x=True, y=True, alpha=0.16)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)

        self.fit_all.clicked.connect(self.plot.autoRange)
        self.fit_bore.clicked.connect(self._fit_column_bore)
        self.plot.scene().sigMouseClicked.connect(
            self._plot_position_clicked
        )

    def _plot_position_clicked(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not event.double()
        ):
            return
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        position = view_box.mapSceneToView(event.scenePos())
        self.axial_position_selected.emit(float(position.x()))
        event.accept()

    def _fit_column_bore(self) -> None:
        if self._result is None or not self._records:
            return
        diameter = max(
            float(segment.inner_diameter_mm)
            for segment in self._result.assembly.vacuum_bore_segments
        )
        start = min(item.start_z_mm for item in self._records)
        end = max(item.end_z_mm for item in self._records)
        self.plot.setXRange(start, end, padding=0.0)
        self.plot.setYRange(-0.58 * diameter, 0.58 * diameter, padding=0.0)

    def _add_pole_piece_projection(self, record, colour) -> None:
        """Draw the axial projection of a hollow, tapered pole piece."""

        start = float(record.start_z_mm)
        end = float(record.end_z_mm)
        length = max(end - start, 0.001)
        bore = min(0.5 * record.bore_diameter_mm, 0.5 * record.outer_diameter_mm)
        outer = 0.5 * record.outer_diameter_mm
        tip = 0.5 * record.pole_tip_diameter_mm
        tip = min(max(tip, bore + 0.02 * (outer - bore)), outer)
        face_at_end = self._pole_face_at_end(record.key)
        taper_length = min(0.38 * length, max(length - 0.001, 0.0))
        if face_at_end:
            outside_z = start
            shoulder_z = end - taper_length
            face_z = end
        else:
            outside_z = end
            shoulder_z = start + taper_length
            face_z = start
        tooltip = (
            f"{record.name}\nHollow pole-piece projection\n"
            f"Z {start:.6g}–{end:.6g} mm\n"
            f"vacuum ID {record.vacuum_inner_diameter_mm:.6g} mm | "
            f"tip OD {2.0 * tip:.6g} mm"
        )
        rgb = pg.mkColor(colour)
        for sign in (-1.0, 1.0):
            points = QPolygonF([
                QPointF(outside_z, sign * bore),
                QPointF(outside_z, sign * outer),
                QPointF(shoulder_z, sign * outer),
                QPointF(face_z, sign * tip),
                QPointF(face_z, sign * bore),
            ])
            polygon = QGraphicsPolygonItem(points)
            polygon.setPen(pg.mkPen(colour, width=0.9))
            polygon.setBrush(pg.mkBrush(
                rgb.red(), rgb.green(), rgb.blue(),
                105 if record.excitation_enabled is not False else 38,
            ))
            polygon.setToolTip(tooltip)
            self.plot.addItem(polygon)

    @staticmethod
    def _pole_face_at_end(key: str) -> bool:
        """Return which axial end owns the tapered pole-gap face."""
        if key == CONDENSER_LENS_1_LOWER_POLE:
            return True
        if key == CONDENSER_LENS_2_UPPER_POLE:
            return False
        return "upper" in key

    def _add_c1_c2_pole_gap_reference(self) -> None:
        c1_pole = self._record_by_key.get(CONDENSER_LENS_1_LOWER_POLE)
        c2_pole = self._record_by_key.get(CONDENSER_LENS_2_UPPER_POLE)
        if c1_pole is None or c2_pole is None:
            return
        gap_start = float(c1_pole.end_z_mm)
        gap_end = float(c2_pole.start_z_mm)
        if gap_end <= gap_start:
            return
        midpoint = 0.5 * (gap_start + gap_end)
        self._c1_c2_pole_gap = (gap_start, gap_end, midpoint)
        line = pg.InfiniteLine(
            pos=midpoint,
            angle=90,
            pen=pg.mkPen(
                "#22d3ee", width=1.6, style=Qt.PenStyle.DotLine
            ),
            label="C1-C2 pole-gap centre",
            labelOpts={
                "position": 0.04,
                "color": "#a5f3fc",
                "rotateAxis": (1, 0),
            },
        )
        tooltip = (
            "C1-C2 inter-lens pole gap\n"
            f"Z = [{gap_start:.6g}, {gap_end:.6g}] mm\n"
            f"Mid-plane = {midpoint:.6g} mm\n"
            "Confirmed design target for the C1-C2 crossover. This marker "
            "does not alter lens excitation or force the traced rays."
        )
        line.setZValue(32)
        line.setToolTip(tooltip)
        line.label.setToolTip(tooltip)
        self.plot.addItem(line)
        self._design_reference_items.append(line)

    def display_result(self, result) -> None:
        self._result = result
        self._records = physical_layout_records(result)
        self._record_by_key = {item.key: item for item in self._records}
        self.plot.clear()
        self._highlight = None
        self._design_reference_items = []
        self._vacuum_liner_items = []
        self._c1_c2_pole_gap = None
        if not self._records:
            return

        segments = result.assembly.vacuum_bore_segments
        minimum_diameter = min(float(item.inner_diameter_mm) for item in segments)
        maximum_diameter = max(float(item.inner_diameter_mm) for item in segments)
        start = min(item.start_z_mm for item in self._records)
        end = max(item.end_z_mm for item in self._records)
        vacuum_z, vacuum_upper, vacuum_lower = vacuum_bore_plot_points(
            result.assembly
        )
        spots = []
        for record in self._records:
            colour = _component_colour(record)
            width = max(record.end_z_mm - record.start_z_mm, 0.25)
            outer_half = 0.5 * record.outer_diameter_mm
            bore_half = min(0.5 * record.bore_diameter_mm, outer_half)
            if (
                record.profile == "magnetic_pole_piece"
                and outer_half > bore_half
            ):
                self._add_pole_piece_projection(record, colour)
            elif record.profile == "magnetic_lens_assembly":
                # Optical parent only: independent children carry its material.
                pass
            elif record.profile == "reference_plane" or outer_half <= bore_half:
                line = pg.InfiniteLine(
                    record.center_z_mm,
                    angle=90,
                    pen=pg.mkPen(colour, width=1.2, style=Qt.PenStyle.DashLine),
                )
                line.setToolTip(f"{record.name}\nReference plane")
                self.plot.addItem(line)
            else:
                material_height = outer_half - bore_half
                for lower in (-outer_half, bore_half):
                    rect = QGraphicsRectItem(
                        record.start_z_mm,
                        lower,
                        width,
                        material_height,
                    )
                    rect.setPen(pg.mkPen(colour, width=0.8))
                    alpha = 105 if record.excitation_enabled is not False else 38
                    rect.setBrush(pg.mkBrush(pg.mkColor(colour).red(), pg.mkColor(colour).green(), pg.mkColor(colour).blue(), alpha))
                    rect.setToolTip(
                        f"{record.name}\nZ {record.start_z_mm:.6g}–{record.end_z_mm:.6g} mm\n"
                        f"OD {record.outer_diameter_mm:.6g} mm | hardware bore {record.bore_diameter_mm:.6g} mm | "
                        f"vacuum ID {record.vacuum_inner_diameter_mm:.6g} mm"
                    )
                    self.plot.addItem(rect)
            for reference in record.optical_references_mm:
                self.plot.plot(
                    [reference, reference],
                    [-max(bore_half, 0.3), max(bore_half, 0.3)],
                    pen=pg.mkPen("#fde047", width=1.0, style=Qt.PenStyle.DotLine),
                )
            spots.append({
                "pos": (record.center_z_mm, 0.0),
                "data": record.key,
                "brush": pg.mkBrush(colour),
                "pen": pg.mkPen("#ffffff", width=0.8),
                "size": 7,
            })

        for segment in result.assembly.vacuum_liner_segments:
            width = segment.end_z_mm - segment.start_z_mm
            inner = 0.5 * segment.inner_diameter_mm
            outer = 0.5 * segment.outer_diameter_mm
            for lower in (-outer, inner):
                rect = QGraphicsRectItem(
                    segment.start_z_mm,
                    lower,
                    width,
                    segment.wall_thickness_mm,
                )
                rect.setPen(pg.mkPen("#94a3b8", width=0.7))
                rect.setBrush(pg.mkBrush(148, 163, 184, 150))
                rect.setToolTip(
                    f"{segment.name}\nVacuum liner\n"
                    f"ID {segment.inner_diameter_mm:.6g} mm | "
                    f"OD {segment.outer_diameter_mm:.6g} mm"
                )
                self.plot.addItem(rect)
                self._vacuum_liner_items.append(rect)
        for y in (vacuum_upper, vacuum_lower):
            self.plot.plot(
                vacuum_z, y,
                pen=pg.mkPen("#f8fafc", width=2.0),
            )

        self._add_c1_c2_pole_gap_reference()

        centres = pg.ScatterPlotItem(spots=spots, pxMode=True)
        centres.setZValue(40)
        centres.setToolTip("Click a component centre to select it")
        centres.sigClicked.connect(self._centre_clicked)
        self.plot.addItem(centres)
        self.plot.autoRange()
        self.heading.setText(
            f"Resolved mechanical layout — {len(self._records)} components | "
            f"vacuum ID {minimum_diameter:.6g}–{maximum_diameter:.6g} mm"
        )

    def _centre_clicked(self, _item, points, _event=None) -> None:
        if points:
            self.component_selected.emit(str(points[0].data()))

    def focus_component(self, part) -> None:
        record = self._record_by_key.get(getattr(part, "key", ""))
        if record is None:
            return
        if self._highlight is not None:
            self.plot.removeItem(self._highlight)
        half_width = max(0.5 * (record.end_z_mm - record.start_z_mm), 0.5)
        self._highlight = pg.LinearRegionItem(
            values=(record.center_z_mm - half_width, record.center_z_mm + half_width),
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(250, 204, 21, 42),
            pen=pg.mkPen("#facc15", width=1.2),
        )
        self._highlight.setZValue(35)
        self.plot.addItem(self._highlight)
        self.summary.setText(
            f"Selected: {record.name} | centre {record.center_z_mm:.6g} mm | "
            f"OD {record.outer_diameter_mm:.6g} mm | hardware bore {record.bore_diameter_mm:.6g} mm | "
            f"vacuum ID {record.vacuum_inner_diameter_mm:.6g} mm | "
            f"optical references: {', '.join(f'{value:.6g}' for value in record.optical_references_mm) or 'none'} mm"
        )


class TransverseBeamView(QWidget):
    """X-Y beam slice that makes round-lens image rotation observable."""

    MAX_DISPLAY_RAYS = 2_000
    _RAY_COLOURS = ("#ff453a", "#34c759", "#0a84ff", "#ffb000")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result = None
        self._plane_z_mm = None
        self._scatter = None

        self.heading = QLabel("Transverse beam X-Y")
        self.summary = QLabel(
            "Select a component to inspect the beam at its centre plane."
        )
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.fit_beam = QPushButton("Fit beam")
        self.fit_beam.setStyleSheet(BUTTON_STYLE)

        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.fit_beam)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("transverseBeamPlot")
        self.plot.setLabel("bottom", "X displacement", units="mm")
        self.plot.setLabel("left", "Y displacement", units="mm")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setAspectLocked(True)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        self.fit_beam.clicked.connect(self.plot.autoRange)

    @staticmethod
    def _branch_at_plane(simulation, z_mm):
        if z_mm <= float(simulation.incident.z[-1]) + 1.0e-9:
            return simulation.incident
        return simulation.branches.get(
            "000", next(iter(simulation.branches.values()), simulation.incident)
        )

    @staticmethod
    def _interpolate(values, z_values, z_mm):
        upper = int(np.searchsorted(z_values, z_mm, side="left"))
        if upper <= 0:
            return np.asarray(values[0], dtype=float)
        if upper >= len(z_values):
            return np.asarray(values[-1], dtype=float)
        if abs(float(z_values[upper]) - z_mm) <= 1.0e-12:
            return np.asarray(values[upper], dtype=float)
        lower = upper - 1
        fraction = (z_mm - z_values[lower]) / (
            z_values[upper] - z_values[lower]
        )
        return (
            (1.0 - fraction) * np.asarray(values[lower], dtype=float)
            + fraction * np.asarray(values[upper], dtype=float)
        )

    def display_result(self, result) -> None:
        self._result = result
        if self._plane_z_mm is None:
            self._plane_z_mm = float(result.simulation.incident.z[-1])
        self._redraw()

    def focus_component(self, part) -> None:
        self._plane_z_mm = float(part.center_z_mm)
        if self._result is not None:
            self._redraw()

    def _redraw(self) -> None:
        self.plot.clear()
        self._scatter = None
        if self._result is None or self._plane_z_mm is None:
            return
        simulation = self._result.simulation
        branch = self._branch_at_plane(simulation, self._plane_z_mm)
        z_values = np.asarray(branch.z, dtype=float)
        plane = float(np.clip(self._plane_z_mm, z_values[0], z_values[-1]))
        x_m = self._interpolate(branch.x, z_values, plane)
        y_m = self._interpolate(branch.y, z_values, plane)
        blocked = np.asarray(branch.blocked_z, dtype=float)
        keep = np.isnan(blocked) | (blocked >= plane - 1.0e-9)
        indices = np.flatnonzero(keep)
        if indices.size > self.MAX_DISPLAY_RAYS:
            indices = indices[np.unique(np.linspace(
                0, indices.size - 1, self.MAX_DISPLAY_RAYS, dtype=int
            ))]
        if indices.size == 0:
            self.summary.setText(
                f"Z {plane:.6g} mm | no rays survive to this plane."
            )
            return

        start_x = np.asarray(branch.x[0], dtype=float)
        start_y = np.asarray(branch.y[0], dtype=float)
        initial_angle = np.mod(np.arctan2(start_y, start_x), 2.0 * math.pi)
        colour_index = np.floor(initial_angle / (0.5 * math.pi)).astype(int) % 4
        brushes = [
            pg.mkBrush(self._RAY_COLOURS[value])
            for value in colour_index[indices]
        ]
        self._scatter = pg.ScatterPlotItem(
            x=x_m[indices] * 1.0e3,
            y=y_m[indices] * 1.0e3,
            size=5,
            pen=pg.mkPen(None),
            brush=brushes,
            pxMode=True,
        )
        self.plot.addItem(self._scatter)
        self.plot.addLine(x=0.0, pen=pg.mkPen("#94a3b8", width=0.8))
        self.plot.addLine(y=0.0, pen=pg.mkPen("#94a3b8", width=0.8))
        self.plot.autoRange()

        start = (
            start_x[indices] - float(np.mean(start_x[indices]))
            + 1j * (start_y[indices] - float(np.mean(start_y[indices])))
        )
        current = (
            x_m[indices] - float(np.mean(x_m[indices]))
            + 1j * (y_m[indices] - float(np.mean(y_m[indices])))
        )
        correlation = np.sum(current * np.conjugate(start))
        relative_rotation = (
            float(np.degrees(np.angle(correlation)))
            if abs(correlation) > 1.0e-30 else float("nan")
        )
        rms_radius_mm = 1.0e3 * float(np.sqrt(np.mean(
            (x_m[indices] - np.mean(x_m[indices])) ** 2
            + (y_m[indices] - np.mean(y_m[indices])) ** 2
        )))
        rotation_text = (
            f"{relative_rotation:+.6g} deg"
            if math.isfinite(relative_rotation) else "unavailable"
        )
        self.heading.setText(f"Transverse beam X-Y at Z = {plane:.6g} mm")
        self.summary.setText(
            f"{branch.name} | {indices.size} surviving rays | "
            f"RMS radius {rms_radius_mm:.6g} mm | "
            f"orientation relative to bundle start {rotation_text} | "
            "colour identifies the initial X-Y quadrant"
        )


class MagneticFieldView(QWidget):
    component_selected = Signal(str)
    axial_position_selected = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records = ()
        self._curves = {}
        self._selected_key = None
        self._support_item = None
        self._formula_samples = []

        self.heading = QLabel("Axial magnetic field Bz")
        self.summary = QLabel("Recalculate to evaluate lens fields.")
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.show_individual = QPushButton("Individual lenses")
        self.show_individual.setCheckable(True)
        self.show_individual.setChecked(True)
        self.show_individual.setStyleSheet(BUTTON_STYLE)

        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.show_individual)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("magneticFieldPlot")
        self.plot.setLabel("bottom", "Axial position", units="mm")
        self.plot.setLabel("left", "Bz", units="T")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.legend = self.plot.addLegend(offset=(10, 10))

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        self.show_individual.toggled.connect(self._apply_curve_styles)
        self.plot.scene().sigMouseClicked.connect(
            self._plot_position_clicked
        )

    def _plot_position_clicked(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not event.double()
        ):
            return
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        position = view_box.mapSceneToView(event.scenePos())
        self.axial_position_selected.emit(float(position.x()))
        event.accept()

    @staticmethod
    def _simulation_limits(simulation):
        bundles = (simulation.incident, *simulation.branches.values())
        return (
            min(float(np.min(branch.z)) for branch in bundles),
            max(float(np.max(branch.z)) for branch in bundles),
        )

    def display_result(self, result) -> None:
        self.plot.clear()
        self._curves = {}
        self._support_item = None
        self._formula_samples = []
        self.legend.clear()
        state = getattr(result, "state_snapshot", None)
        if state is None:
            self._records = ()
            self.summary.setText("No calculation state snapshot is available.")
            return
        start, end = self._simulation_limits(result.simulation)
        z_mm = np.linspace(start, end, 3_000)
        total, self._records = lens_field_records(state, z_mm)
        total_curve = self.plot.plot(
            z_mm,
            total,
            pen=pg.mkPen("#f8fafc", width=2.6),
        )
        self.legend.addItem(total_curve, "Total solver Bz")
        for record in self._records:
            curve = self.plot.plot(
                z_mm,
                record.field_t,
                pen=pg.mkPen(record.formula_colour, width=1.2),
            )
            curve.setToolTip(
                f"{record.name}\nPeak |Bz| {record.peak_t:.6g} T\n"
                f"Excitation {record.excitation_percent:.6g}%\n"
                f"Formula: {record.formula_label}\n"
                f"{record.formula_expression}\n"
                f"Signed field integral {record.signed_field_integral_t_m:.6g} T m\n"
                f"Larmor rotation {record.larmor_rotation_deg:+.6g} deg"
            )
            try:
                curve.setCurveClickable(True, width=8)
                curve.sigClicked.connect(
                    lambda *_args, key=record.key: self.component_selected.emit(key)
                )
            except AttributeError:
                pass
            self._curves[record.key] = curve
        formula_records = {}
        for record in self._records:
            formula_records.setdefault(record.formula_key, record)
        for record in formula_records.values():
            sample = pg.PlotDataItem(
                [], [], pen=pg.mkPen(record.formula_colour, width=2.4)
            )
            self.plot.addItem(sample)
            self.legend.addItem(sample, record.formula_label)
            self._formula_samples.append(sample)
        self.plot.autoRange()
        peak = float(np.max(np.abs(total))) if total.size else 0.0
        total_rotation_deg = sum(
            record.larmor_rotation_deg for record in self._records
        )
        self.heading.setText(
            f"Axial magnetic field Bz — {len(self._records)} lenses | total peak {peak:.6g} T"
        )
        self.summary.setText(
            "Positive rotation follows the right-hand rule about +Z | "
            f"full-column signed Larmor rotation {total_rotation_deg:+.6g} deg"
        )
        self._apply_curve_styles()

    def _apply_curve_styles(self) -> None:
        show = self.show_individual.isChecked()
        for record in self._records:
            curve = self._curves.get(record.key)
            if curve is None:
                continue
            curve.setVisible(show or record.key == self._selected_key)
            width = 3.2 if record.key == self._selected_key else 1.15
            alpha = 255 if record.key == self._selected_key else 145
            curve.setPen(pg.mkPen(record.formula_colour, width=width))
            curve.setOpacity(alpha / 255.0)

    def focus_component(self, part) -> None:
        key = getattr(part, "key", "")
        record = next((item for item in self._records if item.key == key), None)
        if record is None:
            return
        self._selected_key = key
        self._apply_curve_styles()
        if self._support_item is not None:
            self.plot.removeItem(self._support_item)
        support_colour = pg.mkColor(record.formula_colour)
        support_colour.setAlpha(34)
        self._support_item = pg.LinearRegionItem(
            values=record.support_mm,
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(support_colour),
            pen=pg.mkPen(record.formula_colour, width=1.2),
        )
        self._support_item.setZValue(-5)
        self.plot.addItem(self._support_item)
        self.summary.setText(self.diagnostic_text(key))

    def diagnostic_text(self, key: str) -> str:
        record = next((item for item in self._records if item.key == key), None)
        if record is None:
            return "Recalculate to update field and focal diagnostics."
        focal = (
            f"{record.focal_length_mm:.6g} mm"
            if math.isfinite(record.focal_length_mm) else "unfocused"
        )
        cs_text = (
            f"{record.spherical_aberration_mm:.6g} mm"
            if record.spherical_aberration_mm is not None else "off"
        )
        return (
            f"Selected: {record.name} | excitation {record.excitation_percent:.6g}% | "
            f"formula {record.formula_label} [{record.formula_expression}] | "
            f"peak |Bz| {record.peak_t:.6g} T | focal length {focal} | "
            f"signed integral {record.signed_field_integral_t_m:.6g} T m | "
            f"Larmor rotation {record.larmor_rotation_deg:+.6g} deg | "
            f"Cs {cs_text} | "
            f"field support {record.support_mm[0]:.6g}–{record.support_mm[1]:.6g} mm"
        )
