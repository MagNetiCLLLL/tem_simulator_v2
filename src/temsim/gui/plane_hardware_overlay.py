"""Upstream hardware projection and recorded interception positions.

An axial projection keeps each component's own Z. It is not an acceptance
mask for the arriving beam at the selected Z. Visibility and fitting are
display operations; neither retraces particles nor changes hardware.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMenu, QPushButton, QToolButton

from temsim.gui.plane_cutoff_events import PlaneCutoffEventsCache
from temsim.gui.plane_hardware_geometry import hardware_geometry_snapshot, projected_hardware_outlines
from temsim.gui.transverse_projection import transverse_view_coordinates


class PlaneHardwareOverlay:
    COLOURS = {"wall": "#cbd5e1", "opening": "#fbbf24", "absorbing": "#22d3ee"}
    PALETTE = ("#fbbf24", "#22d3ee", "#f472b6", "#a78bfa", "#4ade80", "#fb923c", "#60a5fa")
    SPATIAL_MODES = {"position", "intensity"}

    def __init__(self, owner):
        self.owner = owner
        self.items, self.labels, self.label_lines, self.stop_items = [], [], [], []
        self._annotations = []
        self.outlines = ()
        self._coordinates = ()
        self._cache_key = self._cached_result = None
        self._cached_outlines = ()
        self._preview_geometry = None
        self._events = PlaneCutoffEventsCache()
        self._hidden = set()
        self._menu_key = None
        self.visibility_actions = {}
        self.toggle = QCheckBox("Cutoff projection")
        self.toggle.setObjectName("selectedPlaneHardwareOutlines")
        self.toggle.setChecked(True)
        self.toggle.setToolTip("Look upstream from selected Z: show hardware at its own position, not a mask at selected Z.")
        self.stops_toggle = QCheckBox("Stops")
        self.stops_toggle.setChecked(True)
        self.stops_toggle.setToolTip("Show retained path representatives at their recorded interception X/Y and Z; hover for cause.")
        self.visibility_button = QToolButton()
        self.visibility_button.setText("Cutoffs…")
        self.visibility_button.setEnabled(False)
        self.visibility_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.visibility_button)
        self.visibility_button.setMenu(self.menu)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Restore outline and stop visibility. Does not change insertion or calculation settings.")
        self.fit_button = QPushButton("Fit cutoff")
        self.fit_button.setObjectName("fitSelectedPlaneHardware")
        self.fit_button.setEnabled(False)
        self.fit_button.setToolTip("Fit visible upstream outlines, recorded stops and arriving beam. Fit beam restores the beam scale.")
        layout = owner.section_beam_panel.layout()
        # Relocate the existing beam fit rather than creating a second control.
        for index in range(layout.count()):
            child = layout.itemAt(index).layout()
            if child is not None:
                child.removeWidget(owner.fit_beam)
        fit_row = QHBoxLayout()
        fit_row.addStretch(1)
        fit_row.addWidget(owner.fit_beam)
        fit_row.addWidget(self.fit_button)
        self.fit_button.setStyleSheet(owner.fit_beam.styleSheet())
        controls = QHBoxLayout()
        controls.addWidget(self.toggle)
        controls.addWidget(self.stops_toggle)
        controls.addStretch(1)
        controls.addWidget(self.visibility_button)
        controls.addWidget(self.reset_button)
        location = layout.indexOf(owner.plot)
        layout.insertLayout(location, fit_row)
        layout.insertLayout(location + 1, controls)
        self.status = QLabel()
        self.status.setObjectName("selectedPlaneHardwareStatus")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout.addWidget(self.status)
        self.toggle.toggled.connect(lambda _checked: self.redraw())
        self.stops_toggle.toggled.connect(lambda _checked: self.redraw())
        self.fit_button.clicked.connect(self.fit)
        self.reset_button.clicked.connect(self.reset_visibility)
        owner.plot.getViewBox().sigRangeChanged.connect(self._reposition_labels)

    def invalidate(self):
        """An accepted publication replaces captured geometry and stop history."""
        self._cache_key = self._cached_result = None
        self._cached_outlines = ()
        self._preview_geometry = None
        self._events.clear()

    def set_current_state(self, state, assembly=None):
        """Preview edited insertion/geometry; retain executed beam provenance."""
        geometry = hardware_geometry_snapshot(state, assembly)
        self.set_geometry_preview(geometry)
        return geometry

    def set_geometry_preview(self, geometry):
        """Reapply a detached preview after deferred captured-result display."""
        self._preview_geometry = geometry
        self._cache_key = None
        if self.owner._result is not None:
            self.redraw()

    def _remove_items(self):
        present = self.owner.plot.items()
        for item in (*self.items, *self.labels, *self.label_lines, *self.stop_items):
            if item in present:
                self.owner.plot.removeItem(item)
        self.items, self.labels, self.label_lines, self.stop_items = [], [], [], []
        self._annotations = []
        self._coordinates = ()

    def clear(self, message=""):
        self._remove_items()
        self.outlines = ()
        self.fit_button.setEnabled(False)
        self.visibility_button.setEnabled(False)
        self.menu.clear()
        self.visibility_actions = {}
        self._menu_key = None
        self.status.setText(message)
        self.status.setToolTip(message)

    @staticmethod
    def _boundary_id(outline, index):
        # One body can have several physical bore diameters; detector inner and
        # outer circles are also individually selectable.
        span = tuple(np.ptp(np.asarray(outline.polylines_mm[index]), axis=0)) if outline.kind == "column" else ()
        return ("outline", outline.kind, str(outline.key), outline.name, index, span)

    @staticmethod
    def _boundary_name(outline, index):
        suffix = ""
        if len(outline.polylines_mm) > 1:
            suffix = (" outer" if index == 0 else " inner") if outline.kind == "detector" else f" boundary {index + 1}"
        return f"{outline.name}{suffix} | Z {outline.z_mm:.6g} mm"

    @staticmethod
    def _stop_id(group):
        return ("stops", group.blocked_key, group.provenance, group.stop_kind)

    def _stop_name(self, key, z_mm=None):
        result = self.owner._result
        if key == "column_wall":
            segments = getattr(getattr(result, "assembly", None), "vacuum_bore_segments", ())
            matches = [row for row in segments if z_mm is not None
                       and row.start_z_mm - 1e-7 <= z_mm <= row.end_z_mm + 1e-7]
            if matches:
                row = min(matches, key=lambda part: part.inner_diameter_mm)
                name = str(getattr(row, "name", "Column vacuum wall"))
                return "Column vacuum wall" if ":" in name else f"Column vacuum wall — {name}"
            return "Column vacuum wall"
        state = getattr(result, "state_snapshot", None)
        gun = getattr(state, "electron_gun", None)
        for row in (*getattr(result, "aperture_stops", ()),
                    *getattr(state, "recording_planes", ()), *getattr(gun, "bore_components", ())):
            row_key = row.get("key") if isinstance(row, dict) else getattr(row, "key", None)
            if row_key == key:
                return str(row.get("name", key) if isinstance(row, dict) else getattr(row, "name", key))
        return {"projected_field_domain": "Field validity boundary",
                "feg_tip_reabsorbed": "Tip reabsorption"}.get(key, key)

    def _sync_menu(self, events):
        entries = [(self._boundary_id(row, index), self._boundary_name(row, index))
                   for row in self.outlines for index in range(len(row.polylines_mm))]
        entries += [(self._stop_id(group), f"Stops: {self._stop_name(group.blocked_key)} ({group.provenance})")
                    for group in events.groups]
        key = tuple(entries)
        if key == self._menu_key:
            return
        self._menu_key = key
        self.menu.clear()
        self.visibility_actions = {}
        for identity, name in entries:
            action = QAction(name, self.menu)
            action.setCheckable(True)
            action.setChecked(identity not in self._hidden)
            action.toggled.connect(lambda checked, identity=identity: self._set_visible(identity, checked))
            self.menu.addAction(action)
            self.visibility_actions[identity] = action
        # Bound view preferences while retaining hidden components during Z drags.
        if len(self._hidden) > 512:
            self._hidden.intersection_update(self.visibility_actions)

    def _set_visible(self, identity, checked):
        if checked:
            self._hidden.discard(identity)
        else:
            self._hidden.add(identity)
        self.redraw()

    def reset_visibility(self):
        self._hidden.clear()
        self._menu_key = None
        self.toggle.blockSignals(True)
        self.stops_toggle.blockSignals(True)
        self.toggle.setChecked(True)
        self.stops_toggle.setChecked(True)
        self.toggle.blockSignals(False)
        self.stops_toggle.blockSignals(False)
        self.redraw()

    def _draw_outline(self, row, index, ordinal, coincident=(), count=1):
        xy = np.asarray(row.polylines_mm[index], dtype=float)
        u, v = transverse_view_coordinates(xy[:, 0], xy[:, 1], self.owner._projection_angle_deg)
        u, v = u * 1e3, v * 1e3  # Captured millimetres to plot micrometres.
        colour = self.COLOURS["wall"] if row.role == "wall" else self.PALETTE[ordinal % len(self.PALETTE)]
        at_plane = abs(row.z_mm - self.owner._plane_z_mm) <= 1e-9
        pen = pg.mkPen(colour, width=1.2, cosmetic=True,
                       style=Qt.PenStyle.SolidLine if at_plane else Qt.PenStyle.DashLine)
        item = pg.PlotDataItem(u, v, pen=pen, skipFiniteCheck=True)
        item.setZValue(10.)
        description = "\n".join(self._boundary_name(part, boundary) + "\n" + part.description
                                 for part, boundary in coincident) if coincident else row.description
        item.setToolTip(description)
        self.owner.plot.addItem(item, ignoreBounds=True)
        self.items.append(item)
        # Distribute names along both sides with leaders to the true boundary,
        # so small nested apertures do not stack all labels over the beam.
        angle = np.deg2rad(145. if ordinal % 2 == 0 else 35.)
        centre_u, centre_v = .5 * (u.min() + u.max()), .5 * (v.min() + v.max())
        score = (u - centre_u) * np.cos(angle) + (v - centre_v) * np.sin(angle)
        anchor = int(np.argmax(score))
        name = (f"Clear bore Ø {np.ptp(xy[:, 0]):.6g} mm | {len(coincident)} parts"
                if len(coincident) > 1 else self._boundary_name(row, index))
        # Keep annotation labels short; the menu and tooltip retain full names.
        name = name.replace(" Detector", "").replace("Column vacuum wall — ", "Bore: ")
        name = name.replace("Projection-Chamber Differential-Pumping Aperture", "Projection chamber aperture")
        left = ordinal % 2 == 0
        label = pg.TextItem(name, color=colour,
                            anchor=(0. if left else 1., .5))
        font = label.textItem.font()
        font.setPointSize(8)
        label.setFont(font)
        label.setZValue(12.)
        label.setToolTip(description)
        self.owner.plot.addItem(label, ignoreBounds=True)
        self.labels.append(label)
        line = pg.PlotDataItem(pen=pg.mkPen(colour, width=.6, cosmetic=True,
                                           style=Qt.PenStyle.DotLine))
        line.setZValue(9.)
        self.owner.plot.addItem(line, ignoreBounds=True)
        self.label_lines.append(line)
        self._annotations.append((label, line, u, v, anchor, ordinal, count))
        return (u, v)

    def _reposition_labels(self, *_args):
        """Keep text inside the actual plot after fitting/resizing/panning."""
        bounds = self.owner.plot.viewRange()
        width, height = bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]
        for label, line, u, v, anchor, ordinal, count in self._annotations:
            left = ordinal % 2 == 0
            label_x = bounds[0][0] + .02 * width if left else bounds[0][1] - .02 * width
            label_y = bounds[1][1] - (.05 + .9 * (ordinal + .5) / count) * height
            label.setPos(label_x, label_y)
            line.setData([u[anchor], label_x], [v[anchor], label_y])
            # Outside beam-only zoom, a circle must not leave a detached label.
            visible = bool(np.any((u >= bounds[0][0]) & (u <= bounds[0][1])
                                 & (v >= bounds[1][0]) & (v <= bounds[1][1])))
            label.setVisible(visible)
            line.setVisible(visible)

    def _draw_stops(self, group):
        u, v = transverse_view_coordinates(group.x_m, group.y_m, self.owner._projection_angle_deg)
        u, v = u * 1e6, v * 1e6
        colour = "#fb7185" if group.stop_kind == "hardware" else "#94a3b8"
        descriptions = [f"{self._stop_name(group.blocked_key, float(z))}\n"
                        f"Recorded stop Z {z:.9g} mm | source ray {source}\n"
                        f"X/Y {x * 1e6:.9g}/{y * 1e6:.9g} µm | {group.provenance}\n"
                        f"{group.stop_kind}: retained path representative; not an arriving beam point."
                        for x, y, z, source in zip(group.x_m, group.y_m, group.stop_z_mm, group.source_ray_id)]
        item = pg.ScatterPlotItem(u, v, data=descriptions, symbol="x", size=8,
                                  pen=pg.mkPen(colour, width=1.4), brush=None,
                                  hoverable=True, tip=lambda _x, _y, data: data)
        item.setZValue(11.)
        self.owner.plot.addItem(item, ignoreBounds=True)
        self.stop_items.append(item)
        return (u, v)

    def redraw(self, data=None):
        owner = self.owner
        self._remove_items()
        if owner.analysis.wave is not None:
            self.clear("Cutoff projection unavailable for this wave checkpoint.")
            return
        if owner._result is None or owner._plane_z_mm is None:
            self.clear()
            self.invalidate()
            return
        if data is None:
            data = owner.analysis.filter_plane_data()
        frame = getattr(data, "coordinate_frame", "column")
        if frame != "column" or (getattr(data, "status", None) == "Unavailable"
                and any("global Z" in text for text in getattr(data, "diagnostics", ()))):
            self.clear("Cutoff projection unavailable in this filter frame.")
            return
        geometry = self._preview_geometry if self._preview_geometry is not None else owner._result
        key = (id(geometry), owner._plane_z_mm, frame)
        if key != self._cache_key:
            try:
                self._cached_outlines = projected_hardware_outlines(geometry, owner._plane_z_mm, coordinate_frame=frame)
            except ValueError as error:
                self.clear("Hardware geometry unavailable.")
                self.status.setToolTip(str(error))
                return
            self._cache_key, self._cached_result = key, geometry
        self.outlines = self._cached_outlines
        events = self._events.sample(owner._result, owner._plane_z_mm)
        self._sync_menu(events)
        spatial = owner.analysis.mode in self.SPATIAL_MODES
        self.fit_button.setEnabled(spatial and bool(self.outlines or events.displayed_count))
        self.visibility_button.setEnabled(spatial and bool(self.visibility_actions))
        descriptions = [row.description for row in self.outlines]
        preview = self._preview_geometry is not None
        self.status.setToolTip("Axial projection looking upstream; dashed contours show the upstream projection. Body ranges are listed below.\n"
            "Each restriction acts at its own Z. These outlines are not an effective mask at selected Z.\n"
            "Stop crosses use cached interception positions, not arrival coordinates or electron counts.\n"
            + ("Edited hardware preview; beam and recorded stops belong to the previous calculation.\n" if preview else "Captured hardware and recorded stops.\n")
            + "\n".join((*descriptions, *events.diagnostics)))
        if not spatial:
            self.status.setText("Cutoff projection: switch to position X-Y or beam intensity (spatial units).")
            return
        coordinates, stop_descriptions = [], []
        if self.toggle.isChecked():
            grouped = {}
            for row in self.outlines:
                for index in range(len(row.polylines_mm)):
                    if self._boundary_id(row, index) not in self._hidden:
                        # Identical bore silhouettes are one drawn circle, not
                        # dozens of overlaid lines/names. Each original body
                        # keeps its independent visibility and true-Z tooltip.
                        identity = ((row.role, tuple(np.asarray(row.polylines_mm[index]).ravel()))
                                    if row.role == "wall" else self._boundary_id(row, index))
                        grouped.setdefault(identity, []).append((row, index))
            for ordinal, members in enumerate(grouped.values()):
                row, index = members[0]
                coordinates.append(self._draw_outline(row, index, ordinal, members, len(grouped)))
        if self.stops_toggle.isChecked():
            for group in events.groups:
                if self._stop_id(group) not in self._hidden:
                    coordinates.append(self._draw_stops(group))
                    first, last = group.stop_z_mm.min(), group.stop_z_mm.max()
                    planes = f"{first:.6g}" if first == last else f"{first:.6g}–{last:.6g}"
                    stop_descriptions.append(f"× {self._stop_name(group.blocked_key)} | shown Z {planes} mm | "
                                             f"{group.displayed_count}/{group.observed_count} paths")
        self._coordinates = tuple(coordinates)
        self._reposition_labels()
        heading = "Edited hardware preview; previous beam/stops" if preview else "Upstream cutoff projection"
        self.status.setText(f"{heading} | {len(self.items)} boundaries | "
                            f"{sum(len(item.data) for item in self.stop_items)} / {events.total_recorded_count} recorded stops"
                            + ("\n" + "\n".join(stop_descriptions[:5]) if stop_descriptions else "")
                            + ("\nMore stop groups in Cutoffs…" if len(stop_descriptions) > 5 else ""))
        self.status.setTextFormat(Qt.TextFormat.PlainText)

    def fit(self):
        if not self.fit_button.isEnabled():
            return
        coordinates = list(self._coordinates)
        # Include every cached arrival, not only bounded display representatives.
        data = self.owner.analysis.plane_data()
        u, v = transverse_view_coordinates(data.x_m, data.y_m, self.owner._projection_angle_deg)
        coordinates.append((u * 1e6, v * 1e6))
        finite = [(u, v) for u, v in coordinates if len(u)]
        if not finite:
            return
        x, y = np.concatenate([p[0] for p in finite]), np.concatenate([p[1] for p in finite])
        bounds = self.owner.analysis._point_bounds(x, y)
        if self.owner.analysis.mode == "position" and self.owner.analysis.colour_combo.currentData() != "tof":
            self.owner._view_scale_initialized = True
            self.owner._apply_centered_view_ranges(bounds[0][1], bounds[1][1])
            self.redraw()
        else:
            self.owner.analysis._ranges[self.owner.analysis.mode] = bounds
            self.owner.analysis.redraw()
