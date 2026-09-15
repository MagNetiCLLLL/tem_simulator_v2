"""Cached-plane analysis controls for the right-hand beam panel.

Particle views are bounded display samples. Flux maps, angular histograms and
interaction tables instead use every weighted ray reaching the selected plane.
None of these views is a wave image or a detector readout.
"""

from __future__ import annotations

from html import escape
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem,
)

from temsim.gui.input_policy import WheelSafeComboBox
from temsim.physics.ray_identity import emission_colour_values
from temsim.gui.beam_plane_data import (
    sample_beam_plane, spatial_histogram, angular_histogram,
)
from temsim.gui.transverse_projection import (
    transverse_view_coordinates, projection_axis_name, orthogonal_axis_name,
)


class BeamAnalysisControls:
    """Presentation-only companion of TransverseBeamView; one plane cache."""

    MODES = (
        ("Position X-Y", "position"),
        ("Angular X-Y", "angular"),
        ("Angle distribution", "angle_histogram"),
        ("Beam intensity", "intensity"),
        ("Interaction breakdown", "interactions"),
        ("Phase space U / θU", "phase_u"),
        ("Phase space V / θV", "phase_v"),
    )
    POINT_MODES = {"position", "angular", "phase_u", "phase_v"}
    CENTRED_MODES = {"angular", "intensity", "phase_u", "phase_v"}
    EQUAL_UNIT_MODES = {"position", "angular", "intensity"}
    BINS = 64
    COLOUR_MODES = (("Source position", "source"),
                   ("Emission direction (azimuth)", "emission_direction"),
                   ("Emission angle to normal", "emission_angle"),
                   ("Interaction type", "interaction"))

    def __init__(self, owner):
        self.owner = owner
        self.mode = "position"
        self.wave = None
        self._cache = None
        self._cache_key = None
        self._ranges = {}
        self._hover_payload = None
        self._drawing = False
        self.mode_combo = WheelSafeComboBox()
        self.mode_combo.setObjectName("beamAnalysisMode")
        for label, key in self.MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setToolTip(
            "Analyse cached rays at the selected Z. Intensity and distributions "
            "use physical weights, not the displayed point count. No retracing."
        )
        self.colour_combo = WheelSafeComboBox()
        self.colour_combo.setObjectName("beamAnalysisColourBy")
        for label, key in self.COLOUR_MODES:
            self.colour_combo.addItem(label, key)
        self.colour_combo.setToolTip(
            "Source position and emission angles are fixed at actual launch, before extraction. "
            "Emission azimuth is about +Z; emission angle is relative to the local surface normal. "
            "Grey means unavailable launch data. Interaction colour follows the recorded channel."
        )
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.readout = QLabel()
        for label in (self.legend, self.readout):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            label.setStyleSheet("color: #94a3b8;")
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("beamInteractionBreakdown")
        self.table.setHorizontalHeaderLabels(("Interaction", "Source (%)", "Current (pA)"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(250)
        self.table.setMaximumHeight(360)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.hide()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("View"))
        controls.addWidget(self.mode_combo, 1)
        colours = QHBoxLayout()
        self.colour_label = QLabel("Colour by")
        colours.addWidget(self.colour_label)
        colours.addWidget(self.colour_combo, 1)
        layout = owner.section_beam_panel.layout()
        layout.insertLayout(1, controls)
        layout.insertLayout(2, colours)
        layout.insertWidget(4, self.table)
        layout.insertWidget(5, self.legend)
        layout.insertWidget(6, self.readout)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.colour_combo.currentIndexChanged.connect(self._colour_changed)
        owner.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self._resize_timer = QTimer(owner)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._refresh_view_geometry)
        owner.plot.getViewBox().sigResized.connect(self._queue_geometry_refresh)
        self._update_controls()

    def _queue_geometry_refresh(self, *_args):
        if not self._drawing and self.mode == "intensity" and self.owner.isVisible():
            self._resize_timer.start(0)

    def _refresh_view_geometry(self):
        if self.mode == "intensity" and self.owner._result is not None and self.owner.isVisible():
            # An equal-aspect plot can acquire a wider field when its panel
            # is resized. Rebin that visible field using the same plane cache.
            self._ranges[self.mode] = self.owner.plot.viewRange()
            self.redraw()

    def invalidate(self):
        self._cache_key = None
        self._cache = None

    def plane_data(self):
        key = (id(self.owner._result), self.owner._plane_z_mm)
        if key != self._cache_key:
            self._cache = sample_beam_plane(self.owner._result, self.owner._plane_z_mm)
            self._cache_key = key
        return self._cache

    def _mode_changed(self):
        if self.wave is not None:
            self.wave.selection_changed()
            return
        if self.mode != "position" and self.mode != "interactions":
            self._ranges[self.mode] = self.owner.plot.viewRange()
        self.mode = self.mode_combo.currentData()
        self._hover_payload = None
        self.readout.clear()
        self.owner.plot.setAspectLocked(self.mode in self.EQUAL_UNIT_MODES, ratio=1.0)
        self._update_controls()
        if self.mode == "position":
            self.owner._update_projection_labels()
        self.owner._redraw()

    def _colour_changed(self):
        if self.wave is not None:
            self.wave.selection_changed()
            return
        self._update_controls()
        self.owner._redraw()

    def _update_controls(self):
        points = self.mode in self.POINT_MODES
        self.colour_combo.setVisible(points)
        self.colour_label.setVisible(points)
        colour = self.colour_combo.currentData()
        self.owner.initial_beam_panel.setVisible(points and colour in {"source", "emission_direction"})
        self.owner.initial_beam_heading.setText(
            "Emission direction colour" if colour == "emission_direction" else "Source position colour")
        self.owner.angle_colour_wheel.set_colour_quantity(colour)
        self.owner.plot.setVisible(self.mode != "interactions")
        self.table.setVisible(self.mode == "interactions")
        self.owner.fit_beam.setEnabled(self.mode != "interactions")
        self.owner.fit_beam.setToolTip(
            "Fit this view to all cached reaching rays. Other view ranges are retained."
        )
        self.legend.clear()
        self.readout.clear()
        self.legend.setVisible(colour in {"interaction", "emission_angle"} or not points)
        if colour == "emission_angle" and points:
            self._emission_angle_legend()
        self.readout.setVisible(self.mode in {"intensity", "angle_histogram"})

    def update_labels(self):
        u = projection_axis_name(self.owner._projection_angle_deg)
        v = orthogonal_axis_name(self.owner._projection_angle_deg)
        if self.mode == "angular":
            labels = ((f"θ {u}", "mrad"), (f"θ {v}", "mrad"))
        elif self.mode in {"phase_u", "phase_v"}:
            axis = u if self.mode == "phase_u" else v
            labels = ((f"{axis} displacement", "µm"), (f"θ {axis}", "mrad"))
        elif self.mode == "intensity":
            labels = ((f"{u} displacement", "µm"), (f"{v} displacement", "µm"))
        else:
            labels = (("Polar angle to +Z", "mrad"), ("Flux per bin", ""))
        for axis, (text, unit) in zip(("bottom", "left"), labels):
            self.owner.plot.getAxis(axis).enableAutoSIPrefix(False)
            self.owner.plot.getAxis(axis).setLabel(text, units=unit, siPrefixEnableRanges=())
        self.owner.plot.setToolTip(
            "U/V follow the Ray Diagram projection. Angles are atan of the "
            "projected ray slopes relative to +Z; polar angle is atan(hypot(tx, ty)). "
            "Geometric ray transport, not a diffraction pattern or detector image."
        )

    def finish_position(self, styles):
        """Add interaction symbols without changing existing XY sampling."""
        self._hover_payload = None
        self._annotate_emission(self.owner._display_source_ids)
        if self.colour_combo.currentData() in {"emission_direction", "emission_angle"}:
            if self.owner._scatter is not None:
                self.owner._scatter.setBrush(self.source_brushes(self.owner._display_source_ids))
            self.owner.summary.setToolTip(
                "Colours follow original tip emission, not the current propagation angle. "
                "Extraction, focusing and scattering preserve this label. Grey: launch data unavailable.")
            if self.colour_combo.currentData() == "emission_angle":
                self._emission_angle_legend()
            return
        if self.colour_combo.currentData() != "interaction":
            self.legend.clear()
            return
        scatter = self.owner._scatter
        if scatter is not None and styles:
            scatter.setBrush([pg.mkBrush(*row[2]) for row in styles])
            scatter.setSymbol([row[3] for row in styles])
            scatter.setSize(8)
        self.owner.summary.setToolTip(
            "Colours and symbols identify recorded interaction channels; "
            "focusing does not reset them. Hover over a point for source lineage. "
            "The displayed point count is not a weighted intensity."
        )
        self._interaction_legend(styles)

    def _emission_angle_legend(self):
        self.legend.setText('0° <span style="color:#440154">■</span> · angle to normal · '
                            '<span style="color:#fde725">■</span> 90°')
        self.legend.setToolTip("Fixed 0–90 degree scale at emission; not rescaled to surviving rays. Grey: unavailable.")

    def source_brushes(self, ids, source_azimuth=None):
        mode = self.colour_combo.currentData()
        angles = (source_azimuth if mode == "source" else
                  emission_colour_values(self.owner._result.simulation, ids, mode))
        result = []
        cmap = pg.colormap.get("viridis") if mode == "emission_angle" else None
        for ray_id, angle in zip(ids, angles):
            if ray_id < 0 or not np.isfinite(angle):
                colour = QColor("#94a3b8")
            elif cmap is not None:
                colour = cmap.mapToQColor(float(np.clip(angle/(math.pi/2), 0., 1.)))
            else:
                colour = QColor.fromHsvF(float(angle % (2*math.pi))/(2*math.pi), .88, 1.)
            result.append(pg.mkBrush(colour))
        return result

    def _annotate_emission(self, ids):
        """Keep detailed launch values in hover text, not another text box."""
        scatter = self.owner._scatter
        if scatter is None or self.owner._result is None:
            return
        simulation = self.owner._result.simulation
        azimuth = emission_colour_values(simulation, ids, "emission_direction")
        polar = emission_colour_values(simulation, ids, "emission_angle")
        for spot, phi, theta in zip(scatter.points(), azimuth, polar):
            if np.isfinite(theta):
                phi_text = f"{math.degrees(phi):.5g}°" if np.isfinite(phi) else "undefined"
                spot.setData({**spot.data(), "emission":
                    f"Launch azimuth {phi_text} | angle to local normal {math.degrees(theta):.5g}°"})

    def _interaction_legend(self, styles):
        unique = {row[0]: row for row in styles}
        symbols = {"o": "●", "t": "▲", "d": "◆", "star": "★", "x": "×"}
        shown = list(unique.values())[:4]
        compact = " · ".join(
            f'<span style="color:rgb{tuple(map(int, rgb))}">{symbols.get(symbol, "●")}</span> {escape(label)}'
            for _key, label, rgb, symbol in shown
        )
        if len(unique) > len(shown):
            compact += f" · +{len(unique)-len(shown)} more (hover)"
        self.legend.setText(compact)
        self.legend.setToolTip(
            "\n".join(row[1] for row in unique.values())
            + "\nCategories come from recorded model channels, not final focus or angle. "
            "Mixed elastic + inelastic groups use the existing independent-channel "
            "coupling approximation. They do not identify unmodelled channeling events."
        )

    def capture_manual_range(self):
        if self._drawing or self.mode == "interactions":
            return
        bounds = self.owner.plot.viewRange()
        if self.mode in self.CENTRED_MODES:
            bounds = [(-abs(high-low)/2, abs(high-low)/2) for low, high in bounds]
        self._ranges[self.mode] = bounds
        self.redraw()

    def fit(self):
        self._ranges.pop(self.mode, None)
        self.redraw()

    def _set_ranges(self, bounds):
        self.owner.plot.setRange(xRange=bounds[0], yRange=bounds[1], padding=0, disableAutoRange=True)
        self._ranges[self.mode] = self.owner.plot.viewRange()

    @staticmethod
    def _point_bounds(x, y):
        def half(values):
            finite = np.asarray(values)[np.isfinite(values)]
            value = float(np.max(np.abs(finite))) * 1.08 if finite.size else 1.0
            return value if value > 1e-9 else 1.0
        return [(-half(x), half(x)), (-half(y), half(y))]

    @staticmethod
    def _flux_units(data):
        if data.source_current_pa is not None:
            return data.source_current_pa, "pA / bin"
        return 100.0, "% source / bin"

    def _summary(self, data, extra=""):
        fraction = data.total_source_fraction
        total = "Flux unavailable" if fraction is None else f"{100*fraction:.6g}% of source"
        if data.current_pa is not None:
            total += f" | {data.current_pa:.6g} pA"
        self.owner.summary.setText(f"{data.provenance} | {total}" + extra)
        self.owner.summary.setToolTip(
            "Uses all weighted rays crossing this plane, including an exact interception plane. "
            "This is incident flux at the plane before its detector response, not detected counts "
            "or coherent TEM/STEM image intensity. " + " ".join(data.diagnostics)
        )

    def redraw(self):
        owner = self.owner
        self._drawing = True
        try:
            owner.plot.clear()
            owner._scatter = None
            owner._point_spread_image = owner._point_spread_response = None
            owner._fit_coordinates = None
            owner._display_source_ids = np.empty(0, dtype=np.int64)
            self._hover_payload = None
            self.legend.clear()
            self.readout.clear()
            self.table.setRowCount(0)
            self.update_labels()
            label = dict((key, name) for name, key in self.MODES)[self.mode]
            if owner._result is None or owner._plane_z_mm is None:
                owner.heading.setText(label)
                owner.summary.setText("No cached beam result")
                return
            owner.heading.setText(f"{label} | Z {owner._plane_z_mm:.6g} mm")
            data = self.plane_data()
            if self.mode == "interactions":
                self._draw_interactions(data)
            elif self.mode in {"angular", "phase_u", "phase_v"}:
                self._draw_points(data)
            elif self.mode == "intensity":
                self._draw_intensity(data)
            elif self.mode == "angle_histogram":
                self._draw_angles(data)
        except ValueError as exc:
            owner.summary.setText("Analysis unavailable")
            owner.summary.setToolTip(str(exc))
            self.readout.setText("Cached ray data are incomplete; hover for details.")
            self.readout.setToolTip(str(exc))
        finally:
            self._drawing = False

    def _draw_points(self, data):
        owner = self.owner
        u, v = transverse_view_coordinates(data.x_m, data.y_m, owner._projection_angle_deg)
        tu, tv = transverse_view_coordinates(data.tx, data.ty, owner._projection_angle_deg)
        au, av = np.arctan(tu)*1e3, np.arctan(tv)*1e3
        x, y = ((au, av) if self.mode == "angular" else
                (u*1e6, au) if self.mode == "phase_u" else (v*1e6, av))
        pool = np.unique(np.linspace(0, data.total_column_count-1,
            min(data.total_column_count, owner.MAX_DISPLAY_RAYS), dtype=int))
        selected = np.isin(data.column_index, pool) & np.isfinite(x) & np.isfinite(y)
        brushes, symbols = [], []
        source_brushes = (self.source_brushes(data.source_ray_id[selected], data.source_azimuth_rad[selected])
                          if self.colour_combo.currentData() != "interaction" else None)
        for display_index, index in enumerate(np.flatnonzero(selected)):
            if self.colour_combo.currentData() == "interaction":
                brushes.append(pg.mkBrush(*map(int, data.interaction_rgb[index])))
                symbols.append(str(data.interaction_symbol[index]))
            else:
                brushes.append(source_brushes[display_index]); symbols.append("o")
        owner._display_source_ids = data.source_ray_id[selected]
        axis_x = owner.plot.getAxis("bottom").labelText
        axis_y = owner.plot.getAxis("left").labelText
        xunit = "mrad" if self.mode == "angular" else "µm"
        owner._scatter = pg.ScatterPlotItem(
            x=x[selected], y=y[selected], brush=brushes, symbol=symbols,
            size=8 if self.colour_combo.currentData() == "interaction" else 5,
            pen=pg.mkPen(None), hoverable=True,
            data=[{"source_ray_id": int(data.source_ray_id[index]),
                   "interaction": str(data.interaction_label[index])} for index in np.flatnonzero(selected)],
            tip=lambda px, py, info: f"Source {info['source_ray_id']} | {info['interaction']}\n"
                f"{axis_x} {px:.6g} {xunit} | {axis_y} {py:.6g} mrad"
                + ("\n"+info["emission"] if "emission" in info else ""),
        )
        self._annotate_emission(owner._display_source_ids)
        owner.plot.addItem(owner._scatter)
        owner.plot.addLine(x=0, pen=pg.mkPen("#94a3b8", width=.8))
        owner.plot.addLine(y=0, pen=pg.mkPen("#94a3b8", width=.8))
        self._set_ranges(self._ranges.get(self.mode, self._point_bounds(x, y)))
        if self.colour_combo.currentData() == "interaction":
            self._interaction_legend(list(zip(data.interaction_key, data.interaction_label,
                map(tuple, data.interaction_rgb), data.interaction_symbol)))
        elif self.colour_combo.currentData() == "emission_angle":
            self._emission_angle_legend()
        self._summary(data, f" | {np.count_nonzero(selected):,} shown")

    def _draw_intensity(self, data):
        u, v = transverse_view_coordinates(data.x_m, data.y_m, self.owner._projection_angle_deg)
        bounds = self._ranges.get(self.mode, self._point_bounds(u*1e6, v*1e6))
        self._set_ranges(bounds)
        bounds = self._ranges[self.mode]
        hist, xe, ye = spatial_histogram(data, bins=self.BINS,
            projection_angle_deg=self.owner._projection_angle_deg,
            range_m=tuple(tuple(value*1e-6 for value in pair) for pair in bounds))
        scale, unit = self._flux_units(data)
        values = hist*scale
        maximum = float(np.max(values)) if values.size else 0.
        image = pg.ImageItem(axisOrder="row-major")
        image.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
        image.setImage(values.T, autoLevels=False, levels=(0, maximum if maximum > 0 else 1.))
        image.setRect(QRectF(xe[0]*1e6, ye[0]*1e6, np.ptp(xe)*1e6, np.ptp(ye)*1e6))
        self.owner.plot.addItem(image)
        self._hover_payload = ("intensity", values, xe*1e6, ye*1e6, unit)
        colours = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
        self.legend.setText("0 " + "".join(f'<span style="color:{colour}">■</span>' for colour in colours)
            + f" {maximum:.6g} {unit}")
        self.legend.setToolTip(
            "Linear colour scale; maximum updates for the current plane. Compare values, "
            "not colours, across planes. Each value is flux in a spatial bin, not per unit area."
        )
        self.readout.setText("Hover for position and bin flux")
        self._summary(data, f" | visible {float(np.sum(hist))*100:.6g}% of source")

    def _draw_angles(self, data):
        if self.mode not in self._ranges:
            hist, edges = angular_histogram(data, bins=self.BINS)
            scale, unit = self._flux_units(data)
            upper = max(float(np.max(hist))*scale*1.1, 1e-12)
            self._set_ranges(((edges[0], edges[-1]), (0., upper)))
        bounds = self._ranges[self.mode]
        hist, edges = angular_histogram(data, bins=self.BINS, range_mrad=tuple(bounds[0]))
        scale, unit = self._flux_units(data)
        values = hist*scale
        self.owner.plot.plot(edges, values, stepMode="center", fillLevel=0,
            brush=pg.mkBrush(56, 189, 248, 65), pen=pg.mkPen("#38bdf8", width=1.5))
        self.owner.plot.getAxis("left").setLabel(unit, siPrefixEnableRanges=())
        self._set_ranges(bounds)
        self._hover_payload = ("angle", values, edges, None, unit)
        self.readout.setText("Hover for angle and bin flux")
        self._summary(data, f" | visible {float(np.sum(hist))*100:.6g}% of source")

    def _draw_interactions(self, data):
        keys = list(dict.fromkeys(data.interaction_key))
        self.table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            mask = data.interaction_key == key
            first = int(np.flatnonzero(mask)[0])
            label = str(data.interaction_label[first])
            fraction = float(np.sum(data.source_fraction[mask])) if data.weights_valid else None
            current = (fraction*data.source_current_pa
                if fraction is not None and data.source_current_pa is not None else None)
            for column, text in enumerate((label, "Unavailable" if fraction is None else f"{100*fraction:.6g}",
                                           "Unavailable" if current is None else f"{current:.6g}")):
                item = QTableWidgetItem(text)
                item.setToolTip(label + " | Weighted flux reaching the selected plane, not event counts.")
                if column == 0:
                    item.setForeground(QColor(*map(int, data.interaction_rgb[first])))
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()
        self.legend.setText("All reaching weighted rays · not collision counts")
        self._summary(data)

    def _mouse_moved(self, position):
        if self.wave is not None:
            self.wave.mouse_moved(position)
            return
        if self._hover_payload is None or not self.owner.plot.sceneBoundingRect().contains(position):
            return
        point = self.owner.plot.getViewBox().mapSceneToView(position)
        kind, values, xedges, yedges, unit = self._hover_payload
        ix = int(np.searchsorted(xedges, point.x(), side="right")-1)
        if not 0 <= ix < len(xedges)-1:
            self.readout.clear(); return
        if kind == "angle":
            self.readout.setText(f"θ {point.x():.6g} mrad | {values[ix]:.6g} {unit}")
        else:
            iy = int(np.searchsorted(yedges, point.y(), side="right")-1)
            if 0 <= iy < len(yedges)-1:
                self.readout.setText(f"U {point.x():.6g} µm | V {point.y():.6g} µm | {values[ix, iy]:.6g} {unit}")
            else:
                self.readout.clear()
