"""PyQtGraph views for resolved TEM mechanics and axial magnetic fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from temsim.component_keys import (
    CAMERA,
    CONDENSER_LENS_1_LOWER_POLE,
    CONDENSER_LENS_2_UPPER_POLE,
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    ENERGY_FILTER_TAPERED_PRISM,
    FLUORESCENT_SCREEN,
    STEM_DETECTOR_KEYS,
)
from temsim.component_names import (
    APERTURE_SHORT_NAMES,
    DEFLECTOR_SHORT_NAMES,
    LENS_SHORT_NAMES,
    STIGMATOR_SHORT_NAMES,
)
from temsim.diagnostics import (
    image_plane_rotation_records,
    lens_field_records,
    optical_transfer_records,
    physical_layout_records,
    vacuum_bore_plot_points,
)
from temsim.physics.first_order import (
    linear_map_properties,
    relative_image_diffraction_orientation,
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


def _register_selectable_graphics_item(registry, item, component_key) -> None:
    """Associate a plotted graphics item with one canonical component key."""

    if item is None or not component_key:
        return
    registry[id(item)] = str(component_key)
    try:
        item.setCursor(Qt.CursorShape.PointingHandCursor)
    except (AttributeError, TypeError):
        pass


def _selectable_key_at_scene_position(scene, position, registry):
    """Resolve a click to the topmost registered plotted component."""

    matches = []
    for scene_index, candidate in enumerate(scene.items(position)):
        item = candidate
        while item is not None:
            key = registry.get(id(item))
            if key is not None:
                stacking_z = 0.0
                ancestor = item
                while ancestor is not None:
                    try:
                        stacking_z += float(ancestor.zValue())
                    except (AttributeError, TypeError, ValueError):
                        pass
                    parent_item = getattr(ancestor, "parentItem", None)
                    ancestor = (
                        parent_item() if callable(parent_item) else None
                    )
                matches.append((stacking_z, -scene_index, key))
                break
            parent_item = getattr(item, "parentItem", None)
            item = parent_item() if callable(parent_item) else None
    if not matches:
        return None
    return max(matches, key=lambda match: match[:2])[2]


@dataclass(frozen=True)
class _EnergyFilterLabelCallout:
    """One screen-packed Energy Filter label and its physical anchor."""

    key: str
    label: pg.TextItem
    leader: pg.PlotDataItem
    anchor_x_mm: float
    anchor_z_mm: float
    priority: int
    preferred_side: int
    component_key: str


class EnergyFilterView(QWidget):
    """Curvilinear Iliad public topology and non-OEM branch model."""

    component_selected = Signal(str)
    MAXIMUM_DISPLAY_RAYS = 80
    M12_COLOUR = "#c084fc"
    SECTOR_COLOUR = "#60a5fa"
    LABEL_EDGE_PADDING_PX = 10.0
    LABEL_ROW_GAP_PX = 5.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.heading = QLabel("Energy Filter internal branch")
        self.summary = QLabel(
            "The branch is drawn in its own curvilinear X-Z frame; "
            "public topology is separated from adjustable non-OEM geometry."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.fit_all = QPushButton("Fit Energy Filter")
        self.fit_all.setStyleSheet(BUTTON_STYLE)
        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.fit_all)
        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("energyFilterInternalPlot")
        self.plot.setLabel("bottom", "Global branch X", units="mm")
        self.plot.setLabel("left", "Global branch Z", units="mm")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        view_box = self.plot.getViewBox()
        view_box.setAspectLocked(False)
        view_box.setMouseEnabled(x=True, y=True)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        self.fit_all.clicked.connect(self.plot.autoRange)
        self._prism_clear_aperture_items = []
        self._multipole_housing_items = []
        self._device_body_items = []
        self._multipole_centres = None
        self._device_centres = None
        self._label_callouts = {}
        self._selectable_item_keys = {}
        view_box.sigRangeChanged.connect(self._layout_labels)
        self.plot.scene().sigMouseClicked.connect(
            self._component_item_clicked
        )

    def _component_clicked(self, _item, points, _event=None) -> None:
        if points:
            self.component_selected.emit(str(points[0].data()))

    def _component_item_clicked(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or event.double()
        ):
            return
        key = _selectable_key_at_scene_position(
            self.plot.scene(),
            event.scenePos(),
            self._selectable_item_keys,
        )
        if key is None:
            return
        self.component_selected.emit(key)
        event.accept()

    def _label(self, text, x, z, colour, anchor=(0.5, 1.0)) -> None:
        label = pg.TextItem(text=text, color=colour, anchor=anchor)
        label.setPos(float(x), float(z))
        self.plot.addItem(label)

    def _register_label_callout(
        self,
        *,
        key: str,
        text: str,
        anchor_x_mm: float,
        anchor_z_mm: float,
        colour: str,
        tooltip: str,
        priority: int,
        preferred_side: int,
        component_key: str,
    ) -> None:
        """Add a main-column-style label linked by a dashed leader."""

        if key in self._label_callouts:
            raise ValueError(f"Duplicate Energy Filter label key: {key}")
        label = pg.TextItem(
            text,
            color=colour,
            anchor=(0.5, 0.5),
            border=pg.mkPen(colour, width=0.8),
            fill=pg.mkBrush(5, 8, 22, 215),
        )
        label.setToolTip(tooltip)
        label.setZValue(44)
        label.hide()
        self.plot.addItem(label)
        leader_colour = pg.mkColor(colour)
        leader_colour.setAlpha(178)
        leader = pg.PlotDataItem(
            pen=pg.mkPen(
                leader_colour,
                width=0.9,
                style=Qt.PenStyle.DashLine,
            ),
            connect="all",
        )
        leader.setToolTip(tooltip)
        leader.setZValue(43)
        leader.hide()
        self.plot.addItem(leader)
        callout = _EnergyFilterLabelCallout(
            key=str(key),
            label=label,
            leader=leader,
            anchor_x_mm=float(anchor_x_mm),
            anchor_z_mm=float(anchor_z_mm),
            priority=int(priority),
            preferred_side=1 if int(preferred_side) >= 0 else -1,
            component_key=str(component_key),
        )
        self._label_callouts[callout.key] = callout
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            label,
            component_key,
        )

    def _layout_labels(self, *_args) -> None:
        """Pack branch labels into screen-space rows on both sides."""

        if not self._label_callouts:
            return
        view_box = self.plot.getViewBox()
        (x_min, x_max), (z_min, z_max) = view_box.viewRange()
        if x_max <= x_min or z_max <= z_min:
            return
        scene_bounds = view_box.sceneBoundingRect()
        if not scene_bounds.isValid() or scene_bounds.height() <= 0.0:
            return

        active = []
        dimensions = {}
        for callout in self._label_callouts.values():
            callout.label.hide()
            callout.leader.hide()
            if not (
                x_min <= callout.anchor_x_mm <= x_max
                and z_min <= callout.anchor_z_mm <= z_max
            ):
                continue
            callout.label.setPos(
                callout.anchor_x_mm,
                callout.anchor_z_mm,
            )
            callout.label.show()
            rectangle = callout.label.sceneBoundingRect()
            callout.label.hide()
            dimensions[callout.key] = (
                max(float(rectangle.width()), 24.0),
                max(float(rectangle.height()), 18.0),
            )
            active.append(callout)
        if not active:
            return

        for side in (-1, 1):
            candidates = [
                callout for callout in active
                if callout.preferred_side == side
            ]
            if not candidates:
                continue
            maximum_height = max(
                dimensions[callout.key][1] for callout in candidates
            )
            usable_height = max(
                1.0,
                float(scene_bounds.height())
                - 2.0 * self.LABEL_EDGE_PADDING_PX,
            )
            capacity = max(1, int(
                usable_height
                // (maximum_height + self.LABEL_ROW_GAP_PX)
            ))
            if len(candidates) > capacity:
                candidates = sorted(
                    candidates,
                    key=lambda item: (
                        item.priority,
                        item.anchor_z_mm,
                        item.key,
                    ),
                )[:capacity]
            candidates.sort(key=lambda item: float(
                view_box.mapViewToScene(QPointF(
                    item.anchor_x_mm,
                    item.anchor_z_mm,
                )).y()
            ))
            count = len(candidates)
            top = float(scene_bounds.top()) + self.LABEL_EDGE_PADDING_PX
            bottom = (
                float(scene_bounds.bottom()) - self.LABEL_EDGE_PADDING_PX
            )
            for row, callout in enumerate(candidates):
                label_width, label_height = dimensions[callout.key]
                if count == 1:
                    anchor_scene = view_box.mapViewToScene(QPointF(
                        callout.anchor_x_mm,
                        callout.anchor_z_mm,
                    ))
                    centre_y = min(
                        max(
                            float(anchor_scene.y()),
                            top + 0.5 * label_height,
                        ),
                        bottom - 0.5 * label_height,
                    )
                else:
                    centre_y = (
                        top
                        + 0.5 * label_height
                        + row
                        * max(
                            0.0,
                            bottom - top - label_height,
                        )
                        / (count - 1)
                    )
                if side < 0:
                    centre_x = (
                        float(scene_bounds.left())
                        + self.LABEL_EDGE_PADDING_PX
                        + 0.5 * label_width
                    )
                    label_edge_x = centre_x + 0.5 * label_width
                    elbow_x = label_edge_x + 8.0
                else:
                    centre_x = (
                        float(scene_bounds.right())
                        - self.LABEL_EDGE_PADDING_PX
                        - 0.5 * label_width
                    )
                    label_edge_x = centre_x - 0.5 * label_width
                    elbow_x = label_edge_x - 8.0
                label_position = view_box.mapSceneToView(QPointF(
                    centre_x,
                    centre_y,
                ))
                callout.label.setPos(label_position)
                callout.label.show()
                anchor_scene = view_box.mapViewToScene(QPointF(
                    callout.anchor_x_mm,
                    callout.anchor_z_mm,
                ))
                elbow_at_anchor = view_box.mapSceneToView(QPointF(
                    elbow_x,
                    float(anchor_scene.y()),
                ))
                elbow_at_label = view_box.mapSceneToView(QPointF(
                    elbow_x,
                    centre_y,
                ))
                label_edge = view_box.mapSceneToView(QPointF(
                    label_edge_x,
                    centre_y,
                ))
                callout.leader.setData(
                    [
                        callout.anchor_x_mm,
                        float(elbow_at_anchor.x()),
                        float(elbow_at_label.x()),
                        float(label_edge.x()),
                    ],
                    [
                        callout.anchor_z_mm,
                        float(elbow_at_anchor.y()),
                        float(elbow_at_label.y()),
                        float(label_edge.y()),
                    ],
                )
                callout.leader.show()

    def _add_branch_hollow_envelope(
        self,
        centre_m,
        tangent,
        transverse,
        *,
        length_mm,
        bore_diameter_mm,
        outer_diameter_mm,
        colour,
        tooltip,
        component_key,
    ) -> None:
        """Draw an oriented two-bank branch envelope without adding field."""

        centre = np.asarray(centre_m, dtype=float)
        tangent = np.asarray(tangent, dtype=float)
        transverse = np.asarray(transverse, dtype=float)
        half_length_m = 0.5 * float(length_mm) * 1.0e-3
        inner_m = 0.5 * float(bore_diameter_mm) * 1.0e-3
        outer_m = 0.5 * float(outer_diameter_mm) * 1.0e-3
        if not 0.0 <= inner_m < outer_m or half_length_m <= 0.0:
            return
        rgb = pg.mkColor(colour)
        for sign in (-1.0, 1.0):
            inner = sign * inner_m
            outer = sign * outer_m
            points_m = np.asarray((
                centre - tangent * half_length_m + transverse * outer,
                centre - tangent * half_length_m + transverse * inner,
                centre + tangent * half_length_m + transverse * inner,
                centre + tangent * half_length_m + transverse * outer,
            ))
            points_mm = points_m[:, (0, 2)] * 1.0e3
            polygon = QGraphicsPolygonItem(QPolygonF([
                QPointF(float(x_mm), float(z_mm))
                for x_mm, z_mm in points_mm
            ]))
            polygon.setPen(pg.mkPen(colour, width=0.8))
            polygon.setBrush(pg.mkBrush(
                rgb.red(), rgb.green(), rgb.blue(), 92
            ))
            polygon.setToolTip(tooltip)
            polygon.setZValue(3.0)
            self.plot.addItem(polygon)
            self._device_body_items.append(polygon)
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                polygon,
                component_key,
            )

    def _add_branch_plane(
        self,
        centre_m,
        transverse,
        *,
        width_mm,
        colour,
        tooltip,
        component_key,
        line_width=2.0,
    ) -> None:
        centre = np.asarray(centre_m, dtype=float)
        transverse = np.asarray(transverse, dtype=float)
        half_width_m = 0.5 * float(width_mm) * 1.0e-3
        endpoints = np.asarray((
            centre - transverse * half_width_m,
            centre + transverse * half_width_m,
        ))[:, (0, 2)] * 1.0e3
        item = self.plot.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            pen=pg.mkPen(colour, width=float(line_width)),
        )
        item.setToolTip(tooltip)
        item.setZValue(4.0)
        self._device_body_items.append(item)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            item,
            component_key,
        )

    def display_result(self, result) -> None:
        self.plot.clear()
        self._prism_clear_aperture_items = []
        self._multipole_housing_items = []
        self._device_body_items = []
        self._multipole_centres = None
        self._device_centres = None
        self._label_callouts = {}
        self._selectable_item_keys = {}
        state = getattr(result, "state_snapshot", None)
        energy_filter = getattr(state, "energy_filter", None)
        if energy_filter is None or not energy_filter.enabled:
            self._label(
                "No Energy Filter in the active assembly",
                0.0,
                0.0,
                "#94a3b8",
                anchor=(0.5, 0.5),
            )
            self.summary.setText("Energy Filter branch is not installed.")
            return
        from temsim.optics.energy_filter_sector import (
            multipole_housing_bank_polygons_xz_mm,
            sector_from_energy_filter,
            sector_radial_aperture_paths_xz_mm,
            sector_reference_path_xz_mm,
        )

        sector = sector_from_energy_filter(energy_filter)
        entrance_x = float(sector.entrance_point_m[0]) * 1.0e3
        reference_path = sector_reference_path_xz_mm(sector)
        entrance_tooltip = (
            "Iliad Spectrometer Entrance Aperture\n"
            f"current clear diameter {energy_filter.entrance_aperture_mm:g} "
            "mm\nThe 5 mm value is a public experimental condition, not a "
            "unique installed mechanism size."
        )
        self._add_branch_plane(
            np.zeros(3),
            sector.entrance_frame.rotation_local_to_global[:, 0],
            width_mm=energy_filter.entrance_aperture_mm,
            colour="#fbbf24",
            tooltip=entrance_tooltip,
            component_key=ENERGY_FILTER_ENTRANCE_APERTURE,
            line_width=4.0,
        )
        self._register_label_callout(
            key=f"device:{ENERGY_FILTER_ENTRANCE_APERTURE}",
            text="SPECTROMETER ENTRANCE APERTURE",
            anchor_x_mm=0.0,
            anchor_z_mm=0.0,
            colour="#fbbf24",
            tooltip=entrance_tooltip,
            priority=0,
            preferred_side=-1,
            component_key=ENERGY_FILTER_ENTRANCE_APERTURE,
        )
        reference_item = self.plot.plot(
            np.r_[0.0, entrance_x, reference_path[:, 0]],
            np.r_[0.0, 0.0, reference_path[:, 1]],
            pen=pg.mkPen(
                "#fde047",
                width=1.0,
                style=Qt.PenStyle.DotLine,
            ),
        )
        reference_item.setToolTip(
            "Curvilinear optical reference axis; drawing only."
        )
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            reference_item,
            ENERGY_FILTER_TAPERED_PRISM,
        )
        sector_tooltip = (
            "Iliad large tapered-prism reference orbit and clear aperture\n"
            f"reference radius {float(energy_filter.prism_radius_mm):g} mm | "
            "radial half-width "
            f"{float(energy_filter.sector_radial_aperture_mm):g} mm in X-Z\n"
            f"non-dispersive Y pole gap {float(energy_filter.pole_gap_mm):g} "
            "mm\nOuter pole/yoke thickness is unresolved and is not drawn.\n"
            f"Geometry status: {energy_filter._prism_geometry_status}"
        )
        for aperture_path in sector_radial_aperture_paths_xz_mm(sector):
            item = self.plot.plot(
                aperture_path[:, 0],
                aperture_path[:, 1],
                pen=pg.mkPen(self.SECTOR_COLOUR, width=2.0),
            )
            item.setToolTip(sector_tooltip)
            self._prism_clear_aperture_items.append(item)
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                item,
                ENERGY_FILTER_TAPERED_PRISM,
            )

        sector_label_index = len(reference_path) // 2
        sector_label_point = reference_path[sector_label_index]
        self._register_label_callout(
            key="sector",
            text="LARGE TAPERED PRISM",
            anchor_x_mm=sector_label_point[0],
            anchor_z_mm=sector_label_point[1],
            colour=self.SECTOR_COLOUR,
            tooltip=sector_tooltip,
            priority=-1,
            preferred_side=1,
            component_key=ENERGY_FILTER_TAPERED_PRISM,
        )

        exit_point = sector.exit_point_m
        tangent = sector.exit_tangent
        transverse = sector.exit_frame.rotation_local_to_global[:, 0]
        detector_distance = float(energy_filter.zebra_detector_d_mm)
        branch_end = exit_point + tangent * detector_distance * 1.0e-3
        downstream_reference = self.plot.plot(
            [exit_point[0] * 1.0e3, branch_end[0] * 1.0e3],
            [exit_point[2] * 1.0e3, branch_end[2] * 1.0e3],
            pen=pg.mkPen(
                "#fde047",
                width=1.0,
                style=Qt.PenStyle.DotLine,
            ),
        )
        downstream_reference.setToolTip(
            "Curvilinear optical reference axis; drawing only."
        )
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            downstream_reference,
            "energy_filter",
        )

        spots = []
        m12_rgb = pg.mkColor(self.M12_COLOUR)
        for index, element in enumerate(energy_filter.multipoles, start=1):
            origin = np.asarray(element.frame.origin_m) * 1.0e3
            tooltip = (
                f"{element.name}\n"
                f"TOML mechanical envelope: length "
                f"{float(element.housing_length_m) * 1.0e3:g} mm, outer "
                f"diameter {2.0 * float(element.outer_radius_m) * 1.0e3:g} "
                f"mm, clear bore diameter "
                f"{2.0 * float(element.bore_radius_m) * 1.0e3:g} mm\n"
                f"Magnetic support length: {float(element.length_m) * 1.0e3:g} "
                "mm\nPublic topology: ten multipoles, most dodecapoles; "
                "this individual pole assignment is not public.\n"
                "Geometry status: "
                f"{getattr(element, '_mechanical_geometry_status', 'unknown')}"
            )
            for polygon_points in multipole_housing_bank_polygons_xz_mm(
                element
            ):
                polygon = QGraphicsPolygonItem(QPolygonF([
                    QPointF(float(x_mm), float(z_mm))
                    for x_mm, z_mm in polygon_points
                ]))
                polygon.setPen(pg.mkPen(self.M12_COLOUR, width=0.8))
                polygon.setBrush(pg.mkBrush(
                    m12_rgb.red(),
                    m12_rgb.green(),
                    m12_rgb.blue(),
                    105 if element.enabled else 38,
                ))
                polygon.setToolTip(tooltip)
                polygon.setZValue(3.0)
                self.plot.addItem(polygon)
                self._multipole_housing_items.append(polygon)
                _register_selectable_graphics_item(
                    self._selectable_item_keys,
                    polygon,
                    element.key,
                )
            spots.append({
                "pos": (float(origin[0]), float(origin[2])),
                "data": element.key,
                "symbol": "o",
                "size": 7,
                "pen": pg.mkPen("#ffffff", width=0.8),
                "brush": pg.mkBrush(self.M12_COLOUR),
            })
            self._register_label_callout(
                key=f"multipole:{element.key}",
                text=f"M{index:02d}",
                anchor_x_mm=origin[0],
                anchor_z_mm=origin[2],
                colour=self.M12_COLOUR,
                tooltip=(
                    tooltip
                    + "\nThe dashed leader terminates at this carrier."
                ),
                priority=0,
                preferred_side=-1 if index % 2 else 1,
                component_key=element.key,
            )
        carrier_scatter = pg.ScatterPlotItem(spots=spots)
        carrier_scatter.setZValue(40)
        carrier_scatter.setToolTip(
            "Ten independently powered multipole centres; click to edit"
        )
        carrier_scatter.sigClicked.connect(self._component_clicked)
        self.plot.addItem(carrier_scatter)
        self._multipole_centres = carrier_scatter

        slit = energy_filter.energy_slit
        bias = energy_filter.bias_tube
        shutter = energy_filter.fast_shutter
        camera_deflector = energy_filter.camera_deflector
        zebra = energy_filter.zebra_detector
        device_rows = (
            {
                "distance": float(energy_filter.slit_d_mm),
                "key": slit.key,
                "label": "XO / optional EFTEM slit",
                "colour": "#fbbf24",
                "drawing": "plane",
                "width": float(slit.maximum_gap_m) * 1.0e3,
                "status": getattr(
                    slit, "_mechanical_geometry_status", "unresolved"
                ),
                "detail": (
                    f"maximum mechanical gap {slit.maximum_gap_m * 1.0e3:g} "
                    "mm; this is the energy-selecting stop, not the shutter"
                ),
            },
            {
                "distance": float(
                    energy_filter.dynamic_focus_quadrupole_d_mm
                ),
                "key": ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
                "label": "Dynamic-focus electrostatic quadrupole",
                "colour": "#c084fc",
                "drawing": "hollow",
                "length": float(
                    energy_filter.dynamic_focus_quadrupole_length_mm
                ),
                "bore": float(
                    energy_filter.dynamic_focus_quadrupole_bore_mm
                ),
                "outer": float(
                    energy_filter.dynamic_focus_quadrupole_outer_mm
                ),
                "status": str(
                    energy_filter.dynamic_focus_quadrupole_geometry_status
                ),
                "detail": str(
                    energy_filter.dynamic_focus_quadrupole_model_status
                ),
            },
            {
                "distance": float(energy_filter.bias_tube_d_mm),
                "key": bias.key,
                "label": "MultiEELS bias tube",
                "colour": "#94a3b8",
                "drawing": "hollow",
                "length": float(bias.housing_length_mm),
                "bore": float(bias.clear_bore_diameter_mm),
                "outer": float(bias.mechanical_outer_diameter_mm),
                "status": getattr(
                    bias, "_mechanical_geometry_status", "unresolved"
                ),
                "detail": "fast kinetic-energy offset element",
            },
            {
                "distance": float(energy_filter.fast_shutter_d_mm),
                "key": shutter.key,
                "label": "Fast electrostatic shutter",
                "colour": "#fb7185",
                "drawing": "hollow",
                "length": float(shutter.electrode_length_mm),
                "bore": float(shutter.electrode_gap_mm),
                "outer": float(shutter.mechanical_outer_diameter_mm),
                "status": getattr(
                    shutter, "_mechanical_geometry_status", "unresolved"
                ),
                "detail": "fast beam gate; it does not select energy",
            },
            {
                "distance": float(energy_filter.camera_deflector_d_mm),
                "key": camera_deflector.key,
                "label": "Zebra camera deflector",
                "colour": "#2dd4bf",
                "drawing": "hollow",
                "length": float(camera_deflector.electrode_length_mm),
                "bore": float(camera_deflector.electrode_gap_mm),
                "outer": float(
                    camera_deflector.mechanical_outer_diameter_mm
                ),
                "status": getattr(
                    camera_deflector,
                    "_mechanical_geometry_status",
                    "unresolved",
                ),
                "detail": "rapid selector for Zebra strips 1 through 5",
            },
            {
                "distance": float(energy_filter.output_detector_d_mm),
                "key": ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
                "label": "Optional EFTEM output plane",
                "colour": "#38bdf8",
                "drawing": "plane",
                "width": float(energy_filter.output_detector_width_mm),
                "status": str(energy_filter.output_plane_geometry_status),
                "detail": "provisional output reference plane",
            },
            {
                "distance": detector_distance,
                "key": zebra.key,
                "label": "Zebra 5 x 2048 detector",
                "colour": "#4ade80",
                "drawing": "plane",
                "width": float(zebra.spectral_width_mm),
                "status": str(zebra.external_envelope_status),
                "detail": (
                    f"known active strip {zebra.spectral_width_mm:g} x "
                    f"{zebra.spectral_height_mm:g} mm; 2-D alignment area "
                    f"{zebra.alignment_width_mm:g} x "
                    f"{zebra.alignment_height_mm:g} mm; package not drawn"
                ),
            },
        )
        device_spots = []
        for lane_index, row in enumerate(device_rows):
            distance = float(row["distance"])
            key = str(row["key"])
            label_text = str(row["label"])
            colour = str(row["colour"])
            point = exit_point + tangent * distance * 1.0e-3
            x_mm = float(point[0] * 1.0e3)
            z_mm = float(point[2] * 1.0e3)
            tooltip = (
                f"{label_text}\n"
                f"Branch centre X {x_mm:.6g} mm | Z {z_mm:.6g} mm\n"
                f"{row['detail']}\nGeometry status: {row['status']}"
            )
            if row["drawing"] == "hollow":
                self._add_branch_hollow_envelope(
                    point,
                    tangent,
                    transverse,
                    length_mm=row["length"],
                    bore_diameter_mm=row["bore"],
                    outer_diameter_mm=row["outer"],
                    colour=colour,
                    tooltip=tooltip,
                    component_key=key,
                )
            else:
                self._add_branch_plane(
                    point,
                    transverse,
                    width_mm=row["width"],
                    colour=colour,
                    tooltip=tooltip,
                    component_key=key,
                    line_width=3.0 if key == zebra.key else 2.0,
                )
            device_spots.append({
                "pos": (x_mm, z_mm),
                "data": key,
                "symbol": "o",
                "size": 7,
                "pen": pg.mkPen("#ffffff", width=0.8),
                "brush": pg.mkBrush(colour),
            })
            self._register_label_callout(
                key=f"device:{key}",
                text=label_text.upper(),
                anchor_x_mm=x_mm,
                anchor_z_mm=z_mm,
                colour=colour,
                tooltip=tooltip,
                priority=1,
                preferred_side=-1 if lane_index % 2 == 0 else 1,
                component_key=key,
            )
        device_scatter = pg.ScatterPlotItem(spots=device_spots)
        device_scatter.setZValue(40)
        device_scatter.setToolTip(
            "Click an Energy Filter device centre to select it"
        )
        device_scatter.sigClicked.connect(self._component_clicked)
        self.plot.addItem(device_scatter)
        self._device_centres = device_scatter

        branch_result = getattr(result, "energy_filter", None)
        if branch_result is not None and branch_result.paths_u_mm:
            path_count = len(branch_result.paths_u_mm)
            indices = np.unique(np.linspace(
                0,
                path_count - 1,
                min(path_count, self.MAXIMUM_DISPLAY_RAYS),
                dtype=int,
            ))
            for index in indices:
                colour = branch_result.colours[index]
                self.plot.plot(
                    branch_result.paths_u_mm[index],
                    branch_result.paths_v_mm[index],
                    pen=pg.mkPen(colour, width=0.8),
                )

        mode = str(energy_filter.operating_mode).upper()
        metrics = getattr(energy_filter, "_last_slit_metrics", None)
        metric_text = (
            f" | dispersion {metrics.dispersion_um_per_ev:.4g} um/eV | "
            f"non-iso RMS {metrics.non_isochromaticity_ev_rms:.4g} eV"
            if metrics is not None
            else ""
        )
        result_text = (
            f" | {branch_result.status}"
            if branch_result is not None
            else " | Preview shows mechanics; High accuracy traces branch rays"
        )
        self.heading.setText(
            f"Energy Filter physical layout - {mode}"
        )
        entrance_carrier = energy_filter.multipoles[0]
        exit_carrier = energy_filter.multipoles[3]
        self.summary.setText(
            "Public topology: one large tapered prism and ten independently "
            "powered multipoles (most publicly described as dodecapoles; "
            "M01-M10 are simulator indices, not published product labels). "
            "TOML mechanics: M01-M03 "
            f"L {entrance_carrier.housing_length_m * 1.0e3:g} mm, "
            "M04-M10 "
            f"L {exit_carrier.housing_length_m * 1.0e3:g} mm; outer diameter "
            f"{2.0 * entrance_carrier.outer_radius_m * 1.0e3:g} mm and clear "
            f"bore {2.0 * entrance_carrier.bore_radius_m * 1.0e3:g} mm. "
            "All carrier sizes and coordinates are adjustable non-OEM "
            "envelopes. The layout now separates the XO/EFTEM slit from the "
            "fast shutter and includes the confirmed dynamic-focus "
            "electrostatic quadrupole, bias tube, camera deflector, optional "
            "EFTEM output plane, and Zebra active plane. Zebra strip active "
            f"area is {energy_filter.zebra_detector.spectral_width_mm:g} x "
            f"{energy_filter.zebra_detector.spectral_height_mm:g} mm; strip "
            "pitch and external package remain unknown. "
            "Names use the main Physical Layout callout style; X and Z can "
            "be zoomed independently."
            + metric_text
            + result_text
        )
        self.plot.autoRange()
        self._layout_labels()


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


@dataclass(frozen=True)
class _PhysicalLayoutLabelCallout:
    """One screen-packed label and its mechanical anchor."""

    key: str
    label: pg.TextItem
    leader: pg.PlotDataItem
    anchor_z_mm: float
    anchor_radius_mm: float
    priority: int
    preferred_side: int
    component_key: str | None = None


class PhysicalLayoutView(QWidget):
    component_selected = Signal(str)
    axial_position_selected = Signal(float)
    RECORDING_SURFACE_PROFILES = frozenset({
        "retractable_detector_plane",
        "camera_sensor_plane",
    })
    LABEL_MIN_ROWS_PER_SIDE = 6
    LABEL_MAX_ROWS_PER_SIDE = 18
    LABEL_ROW_GAP_PX = 5.0
    LABEL_EDGE_PADDING_PX = 8.0
    LABEL_HORIZONTAL_OFFSETS = (
        0.0,
        0.65,
        -0.65,
        1.3,
        -1.3,
        2.0,
        -2.0,
    )
    COMPONENT_LABEL_EXCLUDED_KEYS = frozenset({
        "objective_lens",
        "objective_upper_pole",
        "objective_lower_pole",
        "sample_stage",
        "sample",
        CAMERA,
        FLUORESCENT_SCREEN,
        *STEM_DETECTOR_KEYS,
    })

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result = None
        self._records = ()
        self._record_by_key = {}
        self._highlight = None
        self._design_reference_items = []
        self._vacuum_liner_items = []
        self._c1_c2_pole_gap = None
        self._objective_lens_half_items = []
        self._objective_lens_labels = []
        self._sample_stage_items = []
        self._sample_holder_items = []
        self._sample_plane_items = []
        self._sample_plane_labels = []
        self._recording_device_items = {}
        self._recording_device_labels = []
        self._part_by_key = {}
        self._component_label_items = {}
        self._component_label_leader_items = {}
        self._component_label_records = ()
        self._visible_component_label_keys = ()
        self._label_callouts = {}
        self._label_rows_per_side = 0
        self._selectable_item_keys = {}

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
        view_box = self.plot.getViewBox()
        view_box.setAspectLocked(False)
        view_box.setMouseEnabled(x=True, y=True)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)

        self.fit_all.clicked.connect(self.plot.autoRange)
        self.fit_bore.clicked.connect(self._fit_column_bore)
        self.plot.scene().sigMouseClicked.connect(
            self._plot_position_clicked
        )
        self.plot.scene().sigMouseClicked.connect(
            self._component_item_clicked
        )
        self.plot.getViewBox().sigRangeChanged.connect(
            self._layout_component_labels
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

    def _component_item_clicked(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or event.double()
        ):
            return
        key = _selectable_key_at_scene_position(
            self.plot.scene(),
            event.scenePos(),
            self._selectable_item_keys,
        )
        if key is None:
            return
        self.component_selected.emit(key)
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
        configured_nose = float(record.pole_nose_axial_length_mm)
        taper_length = min(
            configured_nose if configured_nose > 0.0 else 0.38 * length,
            max(length - 0.001, 0.0),
        )
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
        if configured_nose > 0.0:
            tooltip += f"\nNose axial length {configured_nose:.6g} mm"
        if record.pole_cone_angle_to_axis_deg > 0.0:
            tooltip += (
                "\nNominal cone angle metadata "
                f"{record.pole_cone_angle_to_axis_deg:.6g} deg to axis"
            )
        if record.pole_face_land_axial_thickness_mm > 0.0:
            tooltip += (
                "\nPole-face land axial thickness "
                f"{record.pole_face_land_axial_thickness_mm:.6g} mm"
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
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                polygon,
                record.key,
            )

    @staticmethod
    def _is_objective_lens_layer(record) -> bool:
        return (
            record.key.startswith("objective_lens_")
            and record.profile in {
                "magnetic_lens_housing",
                "magnetic_lens_yoke",
                "magnetic_excitation_coil",
            }
        )

    def _objective_pole_gap(self):
        upper = self._record_by_key.get("objective_upper_pole")
        lower = self._record_by_key.get("objective_lower_pole")
        if upper is None or lower is None:
            return None
        gap_start = float(upper.end_z_mm)
        gap_end = float(lower.start_z_mm)
        if gap_end <= gap_start:
            return None
        return gap_start, gap_end

    def _add_split_objective_lens_layer(self, record, colour) -> None:
        """Draw one Objective layer as separate upper and lower bodies."""

        gap = self._objective_pole_gap()
        if gap is None:
            return
        gap_start, gap_end = gap
        intervals = (
            (
                "Upper Objective Lens",
                float(record.start_z_mm),
                min(float(record.end_z_mm), gap_start),
            ),
            (
                "Lower Objective Lens",
                max(float(record.start_z_mm), gap_end),
                float(record.end_z_mm),
            ),
        )
        outer_half = 0.5 * float(record.outer_diameter_mm)
        bore_half = min(0.5 * float(record.bore_diameter_mm), outer_half)
        material_height = outer_half - bore_half
        rgb = pg.mkColor(colour)
        alpha = 105 if record.excitation_enabled is not False else 38
        for half_name, start, end in intervals:
            if end <= start or material_height <= 0.0:
                continue
            tooltip = (
                f"{half_name} / {record.name}\n"
                f"Z {start:.6g}-{end:.6g} mm\n"
                f"OD {record.outer_diameter_mm:.6g} mm | "
                f"hardware bore {record.bore_diameter_mm:.6g} mm\n"
                "The upper and lower Objective Lens mechanics are separated "
                "at the physical pole-piece gap."
            )
            for lower_y in (-outer_half, bore_half):
                rect = QGraphicsRectItem(
                    start,
                    lower_y,
                    end - start,
                    material_height,
                )
                rect.setPen(pg.mkPen(colour, width=0.8))
                rect.setBrush(pg.mkBrush(
                    rgb.red(), rgb.green(), rgb.blue(), alpha
                ))
                rect.setToolTip(tooltip)
                self.plot.addItem(rect)
                self._objective_lens_half_items.append(rect)
                _register_selectable_graphics_item(
                    self._selectable_item_keys,
                    rect,
                    record.key,
                )

    def _register_label_callout(
        self,
        *,
        key: str,
        label: pg.TextItem,
        anchor_z_mm: float,
        anchor_radius_mm: float,
        colour: str,
        priority: int,
        preferred_side: int,
        component_key: str | None = None,
    ) -> None:
        """Link one label to its physical anchor with a dashed leader."""

        if key in self._label_callouts:
            raise ValueError(f"Duplicate Physical Layout label key: {key}")
        leader_colour = pg.mkColor(colour)
        leader_colour.setAlpha(178)
        leader = pg.PlotDataItem(
            pen=pg.mkPen(
                leader_colour,
                width=0.9,
                style=Qt.PenStyle.DashLine,
            ),
            connect="all",
        )
        leader.setZValue(43)
        leader.setToolTip(label.toolTip())
        leader.hide()
        self.plot.addItem(leader)
        label.hide()
        callout = _PhysicalLayoutLabelCallout(
            key=str(key),
            label=label,
            leader=leader,
            anchor_z_mm=float(anchor_z_mm),
            anchor_radius_mm=max(abs(float(anchor_radius_mm)), 0.25),
            priority=int(priority),
            preferred_side=1 if int(preferred_side) >= 0 else -1,
            component_key=component_key,
        )
        self._label_callouts[callout.key] = callout
        if component_key is not None:
            # Keep this legacy index limited to the ordinary component
            # callouts.  Special schematics (sample holder, objective halves,
            # recording devices) are selectable too, but have their own item
            # collections and must not change the meaning of this mapping.
            if key.startswith("component:"):
                self._component_label_leader_items[component_key] = leader
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                label,
                component_key,
            )

    def _add_objective_lens_labels(self) -> None:
        objective = self._record_by_key.get("objective_lens")
        gap = self._objective_pole_gap()
        if objective is None or gap is None:
            return
        gap_start, gap_end = gap
        outer_half = 0.5 * float(objective.outer_diameter_mm)
        labels = (
            (
                "UPPER OBJECTIVE LENS",
                0.5 * (float(objective.start_z_mm) + gap_start),
            ),
            (
                "LOWER OBJECTIVE LENS",
                0.5 * (gap_end + float(objective.end_z_mm)),
            ),
        )
        for label_text, center_z in labels:
            label = pg.TextItem(
                text=label_text,
                color="#dbeafe",
                anchor=(0.5, 0.5),
                border=pg.mkPen("#60a5fa", width=0.8),
                fill=pg.mkBrush(5, 8, 22, 205),
            )
            label.setPos(center_z, outer_half + 4.0)
            label.setZValue(45)
            label.setToolTip(
                "Mechanical Objective Lens half; the optical Objective Lens "
                "model remains one coupled upper/lower field system."
            )
            self.plot.addItem(label)
            self._objective_lens_labels.append(label)
            callout_key = (
                "objective:upper"
                if label_text.startswith("UPPER")
                else "objective:lower"
            )
            self._register_label_callout(
                key=callout_key,
                label=label,
                anchor_z_mm=center_z,
                anchor_radius_mm=outer_half,
                colour="#60a5fa",
                priority=-3,
                preferred_side=1,
                component_key="objective_lens",
            )

    def _add_sample_stage_and_holder_schematic(self) -> None:
        """Show the transverse stage and holder without adding optical parts."""

        stage = self._record_by_key.get("sample_stage")
        sample = self._record_by_key.get("sample")
        objective = self._record_by_key.get("objective_lens")
        gap = self._objective_pole_gap()
        if stage is None or sample is None or objective is None or gap is None:
            return
        gap_start, gap_end = gap
        gap_width = gap_end - gap_start
        sample_z = float(sample.center_z_mm)
        outer_radius = 0.5 * float(objective.outer_diameter_mm)
        pole_tip_radius = max(
            0.5 * float(
                self._record_by_key["objective_upper_pole"].pole_tip_diameter_mm
            ),
            0.5 * float(
                self._record_by_key["objective_lower_pole"].pole_tip_diameter_mm
            ),
            2.5,
        )
        stage_half_width = 0.46 * gap_width
        stage_body_start = 0.88 * outer_radius
        stage_body_end = 1.24 * outer_radius
        stage_tooltip = (
            "Sample Stage / Goniometer (schematic)\n"
            "The transverse stage is shown as the outer guide and support. "
            "It occupies the side-access path through the Objective pole gap "
            "and does not add an axial optical element."
        )

        sleeve = QGraphicsRectItem(
            sample_z - stage_half_width,
            pole_tip_radius,
            2.0 * stage_half_width,
            stage_body_start - pole_tip_radius,
        )
        sleeve.setPen(pg.mkPen("#94a3b8", width=1.5))
        sleeve.setBrush(pg.mkBrush(100, 116, 139, 38))
        sleeve.setToolTip(stage_tooltip)
        sleeve.setZValue(25)
        self.plot.addItem(sleeve)
        self._sample_stage_items.append(sleeve)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            sleeve,
            "sample_stage",
        )

        body = QGraphicsRectItem(
            sample_z - stage_half_width,
            stage_body_start,
            2.0 * stage_half_width,
            stage_body_end - stage_body_start,
        )
        body.setPen(pg.mkPen("#cbd5e1", width=1.2))
        body.setBrush(pg.mkBrush(100, 116, 139, 150))
        body.setToolTip(stage_tooltip)
        body.setZValue(26)
        self.plot.addItem(body)
        self._sample_stage_items.append(body)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            body,
            "sample_stage",
        )

        holder_half_width = min(0.55, 0.16 * gap_width)
        holder_end = 1.14 * outer_radius
        holder_tooltip = (
            "Sample Holder (schematic)\n"
            "Inserted through the stage from the positive-radius side. "
            f"The holder tip terminates at the current sample plane: "
            f"Z = {sample_z:.6g} mm."
        )
        shaft = QGraphicsRectItem(
            sample_z - holder_half_width,
            0.0,
            2.0 * holder_half_width,
            holder_end,
        )
        shaft.setPen(pg.mkPen("#f59e0b", width=1.0))
        shaft.setBrush(pg.mkBrush(245, 158, 11, 205))
        shaft.setToolTip(holder_tooltip)
        shaft.setZValue(29)
        self.plot.addItem(shaft)
        self._sample_holder_items.append(shaft)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            shaft,
            "sample_stage",
        )

        tip_half_width = min(0.9, 0.3 * gap_width)
        tip = QGraphicsPolygonItem(QPolygonF([
            QPointF(sample_z, 0.0),
            QPointF(sample_z - tip_half_width, pole_tip_radius),
            QPointF(sample_z + tip_half_width, pole_tip_radius),
        ]))
        tip.setPen(pg.mkPen("#fbbf24", width=1.0))
        tip.setBrush(pg.mkBrush(251, 191, 36, 225))
        tip.setToolTip(holder_tooltip)
        tip.setZValue(30)
        self.plot.addItem(tip)
        self._sample_holder_items.append(tip)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            tip,
            "sample_stage",
        )

        grip = QGraphicsRectItem(
            sample_z - 0.36 * gap_width,
            holder_end,
            0.72 * gap_width,
            0.07 * outer_radius,
        )
        grip.setPen(pg.mkPen("#fbbf24", width=1.0))
        grip.setBrush(pg.mkBrush(180, 83, 9, 220))
        grip.setToolTip(holder_tooltip)
        grip.setZValue(30)
        self.plot.addItem(grip)
        self._sample_holder_items.append(grip)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            grip,
            "sample_stage",
        )

        sample_radius = max(0.5 * float(sample.outer_diameter_mm), 1.5)
        sample_line = self.plot.plot(
            [sample_z, sample_z],
            [-sample_radius, sample_radius],
            pen=pg.mkPen("#fb7185", width=3.0),
        )
        sample_line.setZValue(34)
        sample_line.setToolTip(
            f"Sample plane / holder tip\nZ = {sample_z:.6g} mm"
        )
        sample_marker = QGraphicsEllipseItem(
            sample_z - 0.32,
            -0.32,
            0.64,
            0.64,
        )
        sample_marker.setPen(pg.mkPen("#fecdd3", width=1.0))
        sample_marker.setBrush(pg.mkBrush(251, 113, 133, 235))
        sample_marker.setToolTip(
            f"Sample at the end of the holder\nZ = {sample_z:.6g} mm"
        )
        sample_marker.setZValue(35)
        self.plot.addItem(sample_marker)
        self._sample_plane_items.extend((sample_line, sample_marker))
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            sample_line,
            "sample",
        )
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            sample_marker,
            "sample",
        )

        sample_label = pg.TextItem(
            "SAMPLE / SPECIMEN",
            color="#fecdd3",
            anchor=(0.5, 0.5),
            border=pg.mkPen("#fb7185", width=0.8),
            fill=pg.mkBrush(5, 8, 22, 215),
        )
        sample_label.setZValue(46)
        sample_label.setToolTip(
            f"Sample / specimen at the holder tip\nZ = {sample_z:.6g} mm"
        )
        self.plot.addItem(sample_label)
        self._sample_plane_labels.append(sample_label)
        self._register_label_callout(
            key="sample:specimen",
            label=sample_label,
            anchor_z_mm=sample_z,
            anchor_radius_mm=sample_radius,
            colour="#fb7185",
            priority=-4,
            preferred_side=-1,
            component_key="sample",
        )

        stage_label = pg.TextItem(
            "STAGE (schematic)",
            color="#e2e8f0",
            anchor=(0.5, 0.5),
            border=pg.mkPen("#94a3b8", width=0.8),
            fill=pg.mkBrush(5, 8, 22, 205),
        )
        stage_label.setPos(gap_end + 0.8, stage_body_end)
        stage_label.setZValue(46)
        stage_label.setToolTip(stage_tooltip)
        self.plot.addItem(stage_label)
        self._sample_stage_items.append(stage_label)
        self._register_label_callout(
            key="sample:stage",
            label=stage_label,
            anchor_z_mm=sample_z,
            anchor_radius_mm=stage_body_end,
            colour="#94a3b8",
            priority=-3,
            preferred_side=1,
            component_key="sample_stage",
        )

        holder_label = pg.TextItem(
            "SAMPLE HOLDER",
            color="#fde68a",
            anchor=(0.5, 0.5),
            border=pg.mkPen("#f59e0b", width=0.8),
            fill=pg.mkBrush(5, 8, 22, 205),
        )
        holder_label.setPos(gap_end + 0.8, 0.58 * outer_radius)
        holder_label.setZValue(46)
        holder_label.setToolTip(holder_tooltip)
        self.plot.addItem(holder_label)
        self._sample_holder_items.append(holder_label)
        self._register_label_callout(
            key="sample:holder",
            label=holder_label,
            anchor_z_mm=sample_z,
            anchor_radius_mm=0.58 * outer_radius,
            colour="#f59e0b",
            priority=-3,
            preferred_side=1,
            component_key="sample_stage",
        )

    def _recording_plane_component(self, key: str):
        state = getattr(self._result, "state_snapshot", None)
        for component in getattr(state, "recording_planes", ()):
            if component.key == key:
                return component
        return None

    def _remember_recording_item(self, key: str, item) -> None:
        self._recording_device_items.setdefault(key, []).append(item)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            item,
            key,
        )

    def _add_recording_label(
        self,
        record,
        text: str,
        z_mm: float,
        radius_mm: float,
        colour: str,
        tooltip: str,
    ) -> None:
        label = pg.TextItem(
            text,
            color=colour,
            anchor=(0.5, 0.5),
            border=pg.mkPen(colour, width=0.8),
            fill=pg.mkBrush(5, 8, 22, 215),
        )
        label.setPos(z_mm, radius_mm)
        label.setZValue(46)
        label.setToolTip(tooltip)
        self.plot.addItem(label)
        self._remember_recording_item(record.key, label)
        self._recording_device_labels.append(label)
        self._register_label_callout(
            key=f"recording:{record.key}",
            label=label,
            anchor_z_mm=record.center_z_mm,
            anchor_radius_mm=max(
                0.5 * float(record.outer_diameter_mm),
                0.5 * float(record.bore_diameter_mm),
            ),
            colour=colour,
            priority=-2,
            preferred_side=1 if radius_mm >= 0.0 else -1,
            component_key=record.key,
        )

    def _add_retractable_probe_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        inserted = bool(getattr(component, "inserted", True))
        center_z = float(record.center_z_mm)
        outer_radius = 0.5 * float(record.outer_diameter_mm)
        inner_radius = min(
            0.5 * float(record.bore_diameter_mm), outer_radius
        )
        vacuum_radius = 0.5 * float(record.vacuum_inner_diameter_mm)
        housing_start = max(vacuum_radius + 7.0, outer_radius + 5.0)
        housing_depth = max(24.0, 2.0 * outer_radius + 10.0)
        housing_end = housing_start + housing_depth
        plane_width = max(float(record.end_z_mm - record.start_z_mm), 0.8)
        head_center_y = 0.0 if inserted else housing_start + outer_radius + 2.0
        status = "INSERTED" if inserted else "RETRACTED"
        alpha = 220 if inserted else 72
        pen_style = (
            Qt.PenStyle.SolidLine if inserted else Qt.PenStyle.DashLine
        )
        tooltip = (
            f"{record.name} / retractable probe ({status.lower()})\n"
            f"Active plane Z = {center_z:.6g} mm\n"
            f"active OD {record.outer_diameter_mm:.6g} mm | "
            f"central ID {record.bore_diameter_mm:.6g} mm\n"
            "The side actuator and housing are a Physical Layout schematic; "
            "only the thin active plane participates in ray interception."
        )
        rgb = pg.mkColor(colour)

        head_intervals = (
            (head_center_y - outer_radius, head_center_y - inner_radius),
            (head_center_y + inner_radius, head_center_y + outer_radius),
        )
        for start_y, end_y in head_intervals:
            if end_y <= start_y:
                continue
            head = QGraphicsRectItem(
                center_z - 0.5 * plane_width,
                start_y,
                plane_width,
                end_y - start_y,
            )
            head.setPen(pg.mkPen(colour, width=1.2, style=pen_style))
            head.setBrush(pg.mkBrush(
                rgb.red(), rgb.green(), rgb.blue(), alpha
            ))
            head.setToolTip(tooltip)
            head.setZValue(31)
            self.plot.addItem(head)
            self._remember_recording_item(record.key, head)

        arm_start = (
            outer_radius
            if inserted
            else head_center_y + outer_radius
        )
        arm_end = housing_end - 3.0
        if arm_end > arm_start:
            arm = QGraphicsRectItem(
                center_z - 0.9,
                arm_start,
                1.8,
                arm_end - arm_start,
            )
            arm.setPen(pg.mkPen("#86efac", width=0.9))
            arm.setBrush(pg.mkBrush(74, 222, 128, 145))
            arm.setToolTip(tooltip)
            arm.setZValue(28)
            self.plot.addItem(arm)
            self._remember_recording_item(record.key, arm)

        housing = QGraphicsRectItem(
            center_z - 6.0,
            housing_start,
            12.0,
            housing_depth,
        )
        housing.setPen(pg.mkPen("#94a3b8", width=1.2))
        housing.setBrush(pg.mkBrush(71, 85, 105, 82))
        housing.setToolTip(tooltip)
        housing.setZValue(27)
        self.plot.addItem(housing)
        self._remember_recording_item(record.key, housing)

        self._add_recording_label(
            record,
            f"{record.name.upper()}  [{status}]",
            center_z + 7.0,
            housing_end,
            colour,
            tooltip,
        )

    def _add_fluorescent_screen_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        inserted = bool(getattr(component, "inserted", True))
        center_z = float(record.center_z_mm)
        screen_radius = 0.5 * float(record.outer_diameter_mm)
        vacuum_radius = 0.5 * float(record.vacuum_inner_diameter_mm)
        housing_start = max(vacuum_radius + 8.0, screen_radius + 5.0)
        housing_end = housing_start + 22.0
        tilt = min(8.0, max(2.0, 0.2 * screen_radius))
        half_thickness = 0.65
        status = "INSERTED" if inserted else "RETRACTED"
        rgb = pg.mkColor(colour)
        tooltip = (
            f"{record.name} / hinged viewing screen ({status.lower()})\n"
            f"Interaction plane Z = {center_z:.6g} mm\n"
            f"screen diameter {record.outer_diameter_mm:.6g} mm\n"
            "The hinge, arm and parked position are schematic and remain "
            "outside the axial mechanical model."
        )
        if inserted:
            points = QPolygonF([
                QPointF(center_z - tilt - half_thickness, -screen_radius),
                QPointF(center_z - tilt + half_thickness, -screen_radius),
                QPointF(center_z + tilt + half_thickness, screen_radius),
                QPointF(center_z + tilt - half_thickness, screen_radius),
            ])
            pivot_z = center_z + tilt
            pivot_y = screen_radius
        else:
            parked_center_y = housing_start + 0.5 * screen_radius
            points = QPolygonF([
                QPointF(center_z - half_thickness, parked_center_y - screen_radius),
                QPointF(center_z + half_thickness, parked_center_y - screen_radius),
                QPointF(center_z + half_thickness, parked_center_y + screen_radius),
                QPointF(center_z - half_thickness, parked_center_y + screen_radius),
            ])
            pivot_z = center_z
            pivot_y = parked_center_y + screen_radius
        screen = QGraphicsPolygonItem(points)
        screen.setPen(pg.mkPen(
            colour,
            width=1.3,
            style=(
                Qt.PenStyle.SolidLine
                if inserted else Qt.PenStyle.DashLine
            ),
        ))
        screen.setBrush(pg.mkBrush(
            rgb.red(), rgb.green(), rgb.blue(), 155 if inserted else 48
        ))
        screen.setToolTip(tooltip)
        screen.setZValue(31)
        self.plot.addItem(screen)
        self._remember_recording_item(record.key, screen)

        arm_end = max(pivot_y, housing_start)
        arm = QGraphicsRectItem(
            pivot_z - 0.9,
            min(pivot_y, housing_end),
            1.8,
            abs(housing_end - pivot_y),
        )
        arm.setPen(pg.mkPen("#86efac", width=0.9))
        arm.setBrush(pg.mkBrush(74, 222, 128, 135))
        arm.setToolTip(tooltip)
        arm.setZValue(28)
        self.plot.addItem(arm)
        self._remember_recording_item(record.key, arm)

        housing = QGraphicsRectItem(
            center_z - 6.0,
            housing_start,
            12.0,
            housing_end - housing_start,
        )
        housing.setPen(pg.mkPen("#94a3b8", width=1.2))
        housing.setBrush(pg.mkBrush(71, 85, 105, 82))
        housing.setToolTip(tooltip)
        housing.setZValue(27)
        self.plot.addItem(housing)
        self._remember_recording_item(record.key, housing)

        self._add_recording_label(
            record,
            f"FLUORESCENT SCREEN  [{status}]",
            center_z + 7.0,
            max(housing_end, arm_end),
            colour,
            tooltip,
        )

    def _add_camera_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        active = bool(getattr(component, "inserted", True))
        center_z = float(record.center_z_mm)
        sensor_half = 0.5 * float(record.outer_diameter_mm)
        body_half = max(sensor_half + 9.0, 38.0)
        body_length = max(40.0, 0.72 * float(record.outer_diameter_mm))
        body_start = center_z - 4.0
        body_end = center_z + body_length
        status = "ACTIVE" if active else "INACTIVE"
        rgb = pg.mkColor(colour)
        tooltip = (
            f"{record.name} / fixed on-axis camera ({status.lower()})\n"
            f"Sensor plane Z = {center_z:.6g} mm\n"
            f"sensor width {record.outer_diameter_mm:.6g} mm\n"
            "The sensor plane participates in recording. The downstream "
            "camera body is a schematic external envelope only."
        )
        sensor = QGraphicsRectItem(
            center_z - 0.7,
            -sensor_half,
            1.4,
            2.0 * sensor_half,
        )
        sensor.setPen(pg.mkPen(colour, width=1.4))
        sensor.setBrush(pg.mkBrush(
            rgb.red(), rgb.green(), rgb.blue(), 220 if active else 62
        ))
        sensor.setToolTip(tooltip)
        sensor.setZValue(32)
        self.plot.addItem(sensor)
        self._remember_recording_item(record.key, sensor)

        for lower_y in (-body_half, sensor_half + 3.0):
            shell = QGraphicsRectItem(
                body_start,
                lower_y,
                body_end - body_start,
                body_half - sensor_half - 3.0,
            )
            shell.setPen(pg.mkPen("#64748b", width=1.0))
            shell.setBrush(pg.mkBrush(71, 85, 105, 112))
            shell.setToolTip(tooltip)
            shell.setZValue(27)
            self.plot.addItem(shell)
            self._remember_recording_item(record.key, shell)
        back = QGraphicsRectItem(
            body_end - 4.0,
            -sensor_half - 3.0,
            4.0,
            2.0 * sensor_half + 6.0,
        )
        back.setPen(pg.mkPen("#64748b", width=1.0))
        back.setBrush(pg.mkBrush(71, 85, 105, 145))
        back.setToolTip(tooltip)
        back.setZValue(27)
        self.plot.addItem(back)
        self._remember_recording_item(record.key, back)

        self._add_recording_label(
            record,
            f"CAMERA / SENSOR  [{status}]",
            center_z + 2.0,
            body_half + 4.0,
            colour,
            tooltip,
        )

    def _add_recording_device_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        colour = str(getattr(component, "colour", colour))
        if record.key == CAMERA:
            self._add_camera_schematic(record, colour)
        elif record.key == FLUORESCENT_SCREEN:
            self._add_fluorescent_screen_schematic(record, colour)
        elif record.key in STEM_DETECTOR_KEYS:
            self._add_retractable_probe_schematic(record, colour)

    @staticmethod
    def _component_label_text(record) -> str:
        for names in (
            LENS_SHORT_NAMES,
            APERTURE_SHORT_NAMES,
            STIGMATOR_SHORT_NAMES,
            DEFLECTOR_SHORT_NAMES,
        ):
            if record.key in names:
                return str(names[record.key])
        text = str(record.name)
        for old, new in (
            ("Condenser", "Cond"),
            ("Objective", "Obj"),
            ("Diffraction", "Diff"),
            ("Corrector", "Corr"),
            ("Deflector", "Def"),
            ("Stigmator", "Stig"),
            ("Aperture", "Apt"),
            ("Assembly", ""),
            ("Mechanism", "Mech"),
        ):
            text = text.replace(old, new)
        text = " ".join(text.split())
        return text if len(text) <= 24 else f"{text[:21]}..."

    def _component_label_priority(self, record) -> int:
        if (
            record.key in LENS_SHORT_NAMES
            or record.key in APERTURE_SHORT_NAMES
            or record.profile == "magnetic_lens_assembly"
            or "aperture" in record.kind.lower()
        ):
            return 0
        if (
            record.key in STIGMATOR_SHORT_NAMES
            or record.key in DEFLECTOR_SHORT_NAMES
            or any(token in record.profile.lower() for token in (
                "hexapole", "quadrupole", "deflector", "stigmator",
            ))
        ):
            return 1
        return 2

    def _component_requires_label(self, record) -> bool:
        if record.key in self.COMPONENT_LABEL_EXCLUDED_KEYS:
            return False
        part = self._part_by_key.get(record.key)
        if part is None:
            return False
        role = str(part.data.get("mechanical_part_role", ""))
        if bool(part.data.get("mechanical_only", False)):
            return role in {"slit_blade_carrier", "branch_interface"}
        if record.profile in {
            "magnetic_pole_piece",
            "magnetic_lens_housing",
            "magnetic_lens_yoke",
            "magnetic_excitation_coil",
        }:
            return False
        return True

    def _add_component_labels(self) -> None:
        labelled = []
        for record in self._records:
            if not self._component_requires_label(record):
                continue
            colour = _component_colour(record)
            short_name = self._component_label_text(record)
            label = pg.TextItem(
                short_name,
                color=colour,
                anchor=(0.5, 0.5),
                border=pg.mkPen(colour, width=0.8),
                fill=pg.mkBrush(5, 8, 22, 215),
            )
            label.setZValue(44)
            label.setToolTip(
                f"{record.name}\n"
                f"Z = {record.center_z_mm:.6g} mm\n"
                "The dashed leader terminates at this component. Labels are "
                "packed into multiple screen-space rows and relaid out when "
                "the view is zoomed."
            )
            label.hide()
            self.plot.addItem(label)
            self._component_label_items[record.key] = label
            priority = self._component_label_priority(record)
            self._register_label_callout(
                key=f"component:{record.key}",
                label=label,
                anchor_z_mm=record.center_z_mm,
                anchor_radius_mm=max(
                    0.5 * float(record.outer_diameter_mm),
                    0.5 * float(record.bore_diameter_mm),
                ),
                colour=colour,
                priority=priority,
                preferred_side=1 if len(labelled) % 2 == 0 else -1,
                component_key=record.key,
            )
            labelled.append((priority, record))
        self._component_label_records = tuple(sorted(
            labelled,
            key=lambda item: (item[0], item[1].center_z_mm, item[1].key),
        ))

    def _special_label_items(self):
        items = [
            *self._objective_lens_labels,
            *self._recording_device_labels,
            *self._sample_plane_labels,
        ]
        items.extend(
            item for item in (
                *self._sample_stage_items,
                *self._sample_holder_items,
            )
            if isinstance(item, pg.TextItem)
        )
        for line in self._design_reference_items:
            label = getattr(line, "label", None)
            if label is not None:
                items.append(label)
        return tuple(items)

    def _layout_component_labels(self, *_args) -> None:
        """Pack linked labels into dynamic screen-space rows."""

        if not self._label_callouts:
            return
        view_box = self.plot.getViewBox()
        (x_min, x_max), (y_min, y_max) = view_box.viewRange()
        x_span = float(x_max - x_min)
        y_span = float(y_max - y_min)
        if x_span <= 0.0 or y_span <= 0.0:
            return

        callouts = tuple(sorted(
            self._label_callouts.values(),
            key=lambda item: (
                item.priority,
                item.anchor_z_mm,
                item.key,
            ),
        ))
        for callout in callouts:
            callout.label.hide()
            callout.leader.hide()

        scene_bounds = view_box.sceneBoundingRect()
        if not scene_bounds.isValid() or scene_bounds.height() <= 0.0:
            return

        label_heights = []
        for callout in callouts:
            callout.label.setPos(callout.anchor_z_mm, 0.0)
            callout.label.show()
            rectangle = callout.label.sceneBoundingRect()
            if rectangle.isValid() and rectangle.height() > 0.0:
                label_heights.append(float(rectangle.height()))
            callout.label.hide()
        label_height = max(label_heights, default=18.0)
        usable_half_height = max(
            1.0,
            0.46 * float(scene_bounds.height())
            - self.LABEL_EDGE_PADDING_PX,
        )
        calculated_rows = int(
            usable_half_height
            // (label_height + self.LABEL_ROW_GAP_PX)
        )
        rows_per_side = min(
            self.LABEL_MAX_ROWS_PER_SIDE,
            max(self.LABEL_MIN_ROWS_PER_SIDE, calculated_rows),
        )
        self._label_rows_per_side = rows_per_side
        if rows_per_side == 1:
            row_offsets = (0.0,)
        else:
            maximum_offset = max(
                0.0,
                usable_half_height - 0.5 * label_height,
            )
            row_offsets = tuple(
                row * maximum_offset / (rows_per_side - 1)
                for row in range(rows_per_side)
            )

        occupied = []
        managed_label_ids = {
            id(callout.label) for callout in callouts
        }
        for label in self._special_label_items():
            if id(label) in managed_label_ids or not label.isVisible():
                continue
            rectangle = label.sceneBoundingRect()
            if rectangle.isValid():
                occupied.append(rectangle.adjusted(-4.0, -2.0, 4.0, 2.0))

        visible_keys = []
        horizontal_padding = self.LABEL_EDGE_PADDING_PX
        for callout in callouts:
            if (
                callout.anchor_z_mm < x_min
                or callout.anchor_z_mm > x_max
            ):
                continue

            anchor_scene = view_box.mapViewToScene(QPointF(
                callout.anchor_z_mm,
                0.0,
            ))
            callout.label.setPos(callout.anchor_z_mm, 0.0)
            callout.label.show()
            measured = callout.label.sceneBoundingRect()
            callout.label.hide()
            label_width = max(float(measured.width()), 24.0)
            left_limit = (
                float(scene_bounds.left())
                + horizontal_padding
                + 0.5 * label_width
            )
            right_limit = (
                float(scene_bounds.right())
                - horizontal_padding
                - 0.5 * label_width
            )
            if right_limit < left_limit:
                continue
            base_scene_x = min(
                max(float(anchor_scene.x()), left_limit),
                right_limit,
            )
            placed = False
            side_order = (
                callout.preferred_side,
                -callout.preferred_side,
            )
            for _row, row_offset in enumerate(row_offsets):
                if placed:
                    break
                for side in side_order:
                    if side > 0:
                        scene_y = (
                            float(scene_bounds.top())
                            + self.LABEL_EDGE_PADDING_PX
                            + 0.5 * label_height
                            + row_offset
                        )
                    else:
                        scene_y = (
                            float(scene_bounds.bottom())
                            - self.LABEL_EDGE_PADDING_PX
                            - 0.5 * label_height
                            - row_offset
                        )
                    for offset_multiplier in self.LABEL_HORIZONTAL_OFFSETS:
                        scene_x = min(
                            max(
                                base_scene_x
                                + offset_multiplier
                                * (label_width + 8.0),
                                left_limit,
                            ),
                            right_limit,
                        )
                        position = view_box.mapSceneToView(QPointF(
                            scene_x,
                            scene_y,
                        ))
                        callout.label.setPos(position)
                        callout.label.show()
                        rectangle = (
                            callout.label.sceneBoundingRect().adjusted(
                                -4.0,
                                -2.0,
                                4.0,
                                2.0,
                            )
                        )
                        if any(
                            rectangle.intersects(other)
                            for other in occupied
                        ):
                            callout.label.hide()
                            continue
                        occupied.append(rectangle)
                        source_y = side * callout.anchor_radius_mm
                        elbow_y = float(position.y()) - side * 0.035 * y_span
                        callout.leader.setData(
                            [
                                callout.anchor_z_mm,
                                callout.anchor_z_mm,
                                float(position.x()),
                            ],
                            [
                                source_y,
                                elbow_y,
                                float(position.y()),
                            ],
                        )
                        callout.leader.show()
                        if (
                            callout.component_key is not None
                            and callout.key.startswith("component:")
                        ):
                            visible_keys.append(callout.component_key)
                        placed = True
                        break
                    if placed:
                        break
            if not placed:
                callout.label.hide()
                callout.leader.hide()
        self._visible_component_label_keys = tuple(visible_keys)

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
        self._part_by_key = {
            part.key: part for part in getattr(result.assembly, "parts", ())
        }
        self.plot.clear()
        self._highlight = None
        self._design_reference_items = []
        self._vacuum_liner_items = []
        self._c1_c2_pole_gap = None
        self._objective_lens_half_items = []
        self._objective_lens_labels = []
        self._sample_stage_items = []
        self._sample_holder_items = []
        self._sample_plane_items = []
        self._sample_plane_labels = []
        self._recording_device_items = {}
        self._recording_device_labels = []
        self._component_label_items = {}
        self._component_label_leader_items = {}
        self._component_label_records = ()
        self._visible_component_label_keys = ()
        self._label_callouts = {}
        self._label_rows_per_side = 0
        self._selectable_item_keys = {}
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
            elif self._is_objective_lens_layer(record):
                self._add_split_objective_lens_layer(record, colour)
            elif record.profile == "magnetic_lens_assembly":
                # Optical parent only: independent children carry its material.
                pass
            elif record.key in {"sample_stage", "sample"}:
                # Their insertion direction is transverse. Drawing an axial
                # annulus here would imply the wrong mechanical topology.
                pass
            elif record.profile in self.RECORDING_SURFACE_PROFILES:
                self._add_recording_device_schematic(record, colour)
            elif record.profile == "reference_plane" or outer_half <= bore_half:
                line = pg.InfiniteLine(
                    record.center_z_mm,
                    angle=90,
                    pen=pg.mkPen(colour, width=1.2, style=Qt.PenStyle.DashLine),
                )
                line.setToolTip(f"{record.name}\nReference plane")
                self.plot.addItem(line)
                _register_selectable_graphics_item(
                    self._selectable_item_keys,
                    line,
                    record.key,
                )
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
                    _register_selectable_graphics_item(
                        self._selectable_item_keys,
                        rect,
                        record.key,
                    )
            for reference in record.optical_references_mm:
                reference_item = self.plot.plot(
                    [reference, reference],
                    [-max(bore_half, 0.3), max(bore_half, 0.3)],
                    pen=pg.mkPen("#fde047", width=1.0, style=Qt.PenStyle.DotLine),
                )
                _register_selectable_graphics_item(
                    self._selectable_item_keys,
                    reference_item,
                    record.key,
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

        self._add_objective_lens_labels()
        self._add_sample_stage_and_holder_schematic()
        self._add_c1_c2_pole_gap_reference()
        self._add_component_labels()

        centres = pg.ScatterPlotItem(spots=spots, pxMode=True)
        centres.setZValue(40)
        centres.setToolTip("Click a component centre to select it")
        centres.sigClicked.connect(self._centre_clicked)
        self.plot.addItem(centres)
        self.plot.autoRange()
        self._layout_component_labels()
        self.heading.setText(
            f"Resolved mechanical layout — {len(self._records)} components | "
            f"vacuum ID {minimum_diameter:.6g}–{maximum_diameter:.6g} mm"
        )

        self.summary.setText(
            "Objective mechanics are split into upper/lower lenses at the "
            "pole gap. The transverse stage and nested sample holder are "
            "schematic; the holder tip marks the current sample plane. "
            "Recording-device actuators and housings are schematic while "
            "their thin active planes retain the calculated coordinates. "
            "Names are packed into multiple screen-space rows; dashed leaders "
            "connect each visible name to its physical component and relayout "
            "automatically while zooming."
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


@dataclass(frozen=True)
class _CapturedOpticalTransfer:
    mode: str
    record: object
    wavelength_m: float
    assembly_signature: tuple
    provisional_field_polarity: bool


class OpticalTransferView(QWidget):
    """Inspect and pair signed sample-to-plane J_img/J_diff matrices."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records = ()
        self._current_mode = ""
        self._current_wavelength_m = math.nan
        self._current_signature = ()
        self._last_signature = None
        self._provisional_field_polarity = True
        self._captures = {}

        self.heading = QLabel("Signed first-order optical transfer")
        self.summary = QLabel("Recalculate to evaluate J_img and J_diff.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")

        self.target_plane = QComboBox()
        self.target_plane.setObjectName("opticalTransferPlane")
        self.target_plane.setMinimumWidth(260)
        self.capture_current = QPushButton("Capture current mode")
        self.capture_current.setObjectName("captureOpticalTransfer")
        self.capture_current.setStyleSheet(BUTTON_STYLE)
        self.clear_pair = QPushButton("Clear pair")
        self.clear_pair.setObjectName("clearOpticalTransferPair")
        self.clear_pair.setStyleSheet(BUTTON_STYLE)

        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(QLabel("Target plane"))
        header.addWidget(self.target_plane)
        header.addWidget(self.capture_current)
        header.addWidget(self.clear_pair)

        self.matrix_text = QPlainTextEdit()
        self.matrix_text.setObjectName("opticalTransferMatrixText")
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.matrix_text.setStyleSheet(
            "QPlainTextEdit { background: #050816; color: #e2e8f0; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 13px; "
            "border: 1px solid #334155; }"
        )
        self.matrix_text.setMinimumHeight(260)

        self.pair_summary = QLabel(
            "Capture one Image state and one Diffraction state at the same "
            "plane to calculate their relative orientation."
        )
        self.pair_summary.setObjectName("opticalTransferPairSummary")
        self.pair_summary.setWordWrap(True)
        self.pair_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.pair_summary.setStyleSheet(
            "color: #f8fafc; background: #111827; border: 1px solid #334155; "
            "padding: 8px;"
        )

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.summary)
        layout.addWidget(self.matrix_text, 1)
        layout.addWidget(self.pair_summary)

        self.target_plane.currentIndexChanged.connect(
            self._display_selected_record
        )
        self.capture_current.clicked.connect(self._capture_current_mode)
        self.clear_pair.clicked.connect(self._clear_captures)

    @staticmethod
    def _matrix_text(matrix) -> str:
        values = np.asarray(matrix, dtype=float)
        return "\n".join((
            f"[ [{values[0, 0]:+.9e}, {values[0, 1]:+.9e}],",
            f"  [{values[1, 0]:+.9e}, {values[1, 1]:+.9e}] ]",
        ))

    @staticmethod
    def _orientation_text(properties) -> str:
        angle = (
            f"{properties.orientation_deg:+.6g} deg"
            if properties.orientation_deg is not None
            else "unavailable (rank deficient)"
        )
        handedness = "mirrored" if properties.mirrored else "preserved"
        anisotropy = (
            f"{properties.anisotropy_ratio:.6g}"
            if math.isfinite(properties.anisotropy_ratio)
            else "infinite"
        )
        return (
            f"orientation {angle} | handedness {handedness} | "
            f"anisotropy {anisotropy} | rank {properties.rank}"
        )

    def _selected_record(self):
        key = self.target_plane.currentData()
        return next(
            (record for record in self._records if record.key == key),
            None,
        )

    def display_result(self, result) -> None:
        state = getattr(result, "state_snapshot", None)
        if state is None:
            self._records = ()
            self.matrix_text.setPlainText("")
            self.summary.setText("No calculation state snapshot is available.")
            return
        records = tuple(
            getattr(result.simulation, "optical_transfers", ()) or ()
        )
        self._records = records or optical_transfer_records(state)
        self._current_mode = str(getattr(state, "projector_mode", ""))
        self._current_wavelength_m = (
            float(result.simulation.metrics.get("lambda_nm", math.nan))
            * 1.0e-9
        )
        assembly = getattr(result, "assembly", None)
        self._current_signature = tuple(
            getattr(assembly, "selected_module_paths", ()) or ()
        )
        if (
            self._last_signature is not None
            and self._current_signature != self._last_signature
        ):
            self._captures.clear()
        self._last_signature = self._current_signature
        self._provisional_field_polarity = any(
            str(getattr(lens, "field_polarity_status", ""))
            == "provisional_model_assumption"
            for lens in getattr(state, "lenses", ())
        )

        selected_key = self.target_plane.currentData()
        self.target_plane.blockSignals(True)
        self.target_plane.clear()
        for record in self._records:
            insertion = (
                ""
                if record.inserted is None
                else " [inserted]" if record.inserted else " [retracted/virtual]"
            )
            self.target_plane.addItem(
                f"{record.name} at {record.z_mm:.6g} mm{insertion}",
                record.key,
            )
        index = self.target_plane.findData(selected_key)
        self.target_plane.setCurrentIndex(index if index >= 0 else 0)
        self.target_plane.blockSignals(False)
        self.capture_current.setEnabled(
            self._current_mode in {"image", "diffraction"}
            and bool(self._records)
        )
        self.capture_current.setText(
            "Capture Image J_img"
            if self._current_mode == "image"
            else "Capture Diffraction J_diff"
            if self._current_mode == "diffraction"
            else "Capture current mode"
        )
        self._display_selected_record()
        self._refresh_pair_summary()

    def _display_selected_record(self, *_args) -> None:
        record = self._selected_record()
        if record is None:
            self.matrix_text.setPlainText("")
            return
        transfer = record.transfer
        image_detector_map = (
            record.detector_frame.column_to_detector @ transfer.j_img
        )
        diffraction_detector_map = (
            record.detector_frame.column_to_detector
            @ transfer.j_diff_m_per_rad
        )
        image_detector_properties = linear_map_properties(image_detector_map)
        diffraction_detector_properties = linear_map_properties(
            diffraction_detector_map
        )
        active = "J_img" if self._current_mode == "image" else "J_diff"
        insertion = (
            "reference plane"
            if record.inserted is None
            else "inserted" if record.inserted else "retracted / virtual"
        )
        self.heading.setText(
            f"Signed first-order optical transfer - {self._current_mode or 'unknown'} mode"
        )
        self.summary.setText(
            f"{record.name} | Z {record.z_mm:.6g} mm | {insertion} | "
            f"active conjugate map {active}. Straight-column paraxial "
            "Jacobian; spherical aberration, hexapole nonlinearity and the "
            "curved Energy Filter branch are outside this matrix."
        )
        detector = record.detector_frame
        calibration = (
            "calibrated" if detector.is_calibrated else "UNCALIBRATED placeholder"
        )
        self.matrix_text.setPlainText(
            "Coordinate convention\n"
            "  state = (x, y, theta_x, theta_y); electrons travel along +Z\n"
            "  r_plane = J_img @ r_sample + J_diff @ theta_sample\n\n"
            "J_img (dimensionless, column X-Y)\n"
            f"{self._matrix_text(transfer.j_img)}\n"
            f"  {self._orientation_text(record.image_properties)}\n"
            f"  equivalent magnification {record.image_properties.isotropic_scale:.9g}\n\n"
            "J_diff (m/rad; numerically identical in mm/mrad, column X-Y)\n"
            f"{self._matrix_text(transfer.j_diff_m_per_rad)}\n"
            f"  {self._orientation_text(record.diffraction_properties)}\n"
            "  equivalent camera length "
            f"{record.diffraction_properties.isotropic_scale:.9g} m\n\n"
            "Conjugacy residuals at this plane\n"
            "  image residual ||J_diff||2 = "
            f"{np.linalg.norm(transfer.j_diff_m_per_rad, ord=2):.9g} m/rad\n"
            "  diffraction residual ||J_img||2 = "
            f"{np.linalg.norm(transfer.j_img, ord=2):.9g}\n\n"
            "Detector/display frame\n"
            f"  status {detector.status} ({calibration})\n"
            f"  +U axis angle {detector.axis_rotation_deg:+.9g} deg | "
            f"flip U {detector.flip_x} | flip V {detector.flip_y}\n"
            f"  stated uncertainty {detector.uncertainty_deg:.9g} deg\n"
            f"  source {detector.source}\n"
            "  detector-frame J_img: "
            f"{self._orientation_text(image_detector_properties)}\n"
            "  detector-frame J_diff: "
            f"{self._orientation_text(diffraction_detector_properties)}"
        )

    def _capture_current_mode(self) -> None:
        record = self._selected_record()
        if (
            record is None
            or self._current_mode not in {"image", "diffraction"}
            or not math.isfinite(self._current_wavelength_m)
        ):
            return
        self._captures[self._current_mode] = _CapturedOpticalTransfer(
            mode=self._current_mode,
            record=record,
            wavelength_m=self._current_wavelength_m,
            assembly_signature=self._current_signature,
            provisional_field_polarity=self._provisional_field_polarity,
        )
        self._refresh_pair_summary()

    def _clear_captures(self) -> None:
        self._captures.clear()
        self._refresh_pair_summary()

    def _refresh_pair_summary(self) -> None:
        image_capture = self._captures.get("image")
        diffraction_capture = self._captures.get("diffraction")
        if image_capture is None or diffraction_capture is None:
            captured = ", ".join(sorted(self._captures)) or "none"
            self.pair_summary.setText(
                "Capture one Image state and one Diffraction state at the "
                "same plane. Captured modes: " + captured + "."
            )
            return
        if (
            image_capture.assembly_signature
            != diffraction_capture.assembly_signature
        ):
            self.pair_summary.setText(
                "The captured states use different assemblies; clear the pair "
                "and capture both modes again."
            )
            return
        if image_capture.record.key != diffraction_capture.record.key:
            self.pair_summary.setText(
                "The captured states use different target planes "
                f"({image_capture.record.name} and "
                f"{diffraction_capture.record.name}); capture both at one plane."
            )
            return
        try:
            relation = relative_image_diffraction_orientation(
                image_capture.record.transfer,
                diffraction_capture.record.transfer,
                diffraction_capture.wavelength_m,
                image_detector=image_capture.record.detector_frame,
                diffraction_detector=(
                    diffraction_capture.record.detector_frame
                ),
            )
        except ValueError as exc:
            self.pair_summary.setText(
                f"Image/diffraction relation unavailable: {exc}."
            )
            return
        properties = relation.properties
        angle = (
            f"{properties.orientation_deg:+.6g} deg"
            if properties.orientation_deg is not None
            else "unavailable"
        )
        handedness = "mirrored" if properties.mirrored else "preserved"
        uncertainty = (
            f" +/- {relation.detector_uncertainty_deg:.6g} deg detector-axis uncertainty"
            if relation.detector_uncertainty_deg is not None
            else ""
        )
        calibration_warning = (
            " Absolute hardware orientation is NOT calibrated; edit the "
            "Camera TOML axis rotation, flips, uncertainty and provenance "
            "after a measured calibration."
            if relation.calibration_status == "uncalibrated_detector_axes"
            else " Detector axes are calibration-backed."
        )
        polarity_warning = (
            " Lens field polarities remain provisional model assumptions."
            if (
                image_capture.provisional_field_polarity
                or diffraction_capture.provisional_field_polarity
            )
            else ""
        )
        self.pair_summary.setText(
            f"Captured pair at {image_capture.record.name}. Normalised "
            "diffraction-vector -> image-direction map:\n"
            f"{self._matrix_text(relation.normalized_direction_map)}\n"
            f"rotation {angle}{uncertainty} | handedness {handedness} | "
            f"anisotropy {properties.anisotropy_ratio:.6g}. "
            "This maps the reciprocal-vector g direction, which is normal "
            "to lattice planes; it is not a direct-lattice length map."
            + calibration_warning
            + polarity_warning
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
        self._rotation_items = []
        self._plane_records = ()

        self.heading = QLabel("Axial magnetic field Bz")
        self.summary = QLabel("Recalculate to evaluate lens fields.")
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.show_individual = QPushButton("Individual lenses")
        self.show_individual.setCheckable(True)
        self.show_individual.setChecked(True)
        self.show_individual.setStyleSheet(BUTTON_STYLE)
        self.show_rotation_labels = QPushButton("Rotation labels")
        self.show_rotation_labels.setCheckable(True)
        self.show_rotation_labels.setChecked(True)
        self.show_rotation_labels.setStyleSheet(BUTTON_STYLE)

        header = QHBoxLayout()
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.show_rotation_labels)
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
        self.show_rotation_labels.toggled.connect(
            self._apply_rotation_marker_visibility
        )
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
        self._rotation_items = []
        self._plane_records = ()
        self.legend.clear()
        state = getattr(result, "state_snapshot", None)
        if state is None:
            self._records = ()
            self.summary.setText("No calculation state snapshot is available.")
            return
        start, end = self._simulation_limits(result.simulation)
        z_mm = np.linspace(start, end, 3_000)
        total, self._records = lens_field_records(state, z_mm)
        self._plane_records = image_plane_rotation_records(state)
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
                f"Field direction {'+Z' if record.polarity > 0 else '-Z'}\n"
                f"Polarity status {record.field_polarity_status}\n"
                f"Polarity source {record.field_polarity_source}\n"
                f"Lens Larmor rotation {record.larmor_rotation_deg:+.6g} deg\n"
                f"Cumulative column rotation "
                f"{record.cumulative_column_rotation_deg:+.6g} deg"
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
        self._add_rotation_markers(total)
        self.plot.autoRange()
        peak = float(np.max(np.abs(total))) if total.size else 0.0
        total_rotation_deg = sum(
            record.larmor_rotation_deg for record in self._records
        )
        self.heading.setText(
            f"Axial magnetic field Bz — {len(self._records)} lenses | total peak {peak:.6g} T"
        )
        plane_text = "; ".join(
            f"{record.name} θsample "
            f"{record.image_rotation_from_sample_deg:+.4g}°"
            for record in self._plane_records
        )
        self.summary.setText(
            "Positive rotation follows the right-hand rule about +Z | "
            f"full-column signed Larmor rotation {total_rotation_deg:+.6g} deg"
            + (f" | {plane_text}" if plane_text else "")
        )
        self._apply_curve_styles()

    def _add_rotation_markers(self, total_field_t) -> None:
        peak = max(
            float(np.max(np.abs(total_field_t)))
            if np.size(total_field_t) else 0.0,
            1.0e-6,
        )
        for index, record in enumerate(sorted(
            self._records, key=lambda item: item.center_z_mm
        )):
            field_index = int(np.argmin(np.abs(record.z_mm - record.center_z_mm)))
            field_value = float(record.field_t[field_index])
            offset = (0.055 + 0.025 * (index % 3)) * peak
            y_value = field_value + offset if field_value >= 0.0 else field_value - offset
            anchor_y = 1.0 if field_value >= 0.0 else 0.0
            label = pg.TextItem(
                text=(
                    f"{record.key}\n"
                    f"ΔφL {record.larmor_rotation_deg:+.3g}°"
                ),
                color="#cbd5e1",
                anchor=(0.5, anchor_y),
                fill=pg.mkBrush(5, 8, 22, 185),
                border=pg.mkPen(record.formula_colour, width=0.8),
            )
            label.setPos(record.center_z_mm, y_value)
            label.setToolTip(
                f"{record.name}\n"
                f"single-lens ΔφL {record.larmor_rotation_deg:+.6g} deg\n"
                f"column cumulative ΣφL "
                f"{record.cumulative_column_rotation_deg:+.6g} deg"
            )
            self.plot.addItem(label)
            self._rotation_items.append(label)

        plane_colour = "#fbbf24"
        for index, record in enumerate(self._plane_records):
            line = pg.InfiniteLine(
                pos=record.z_mm,
                angle=90,
                movable=False,
                pen=pg.mkPen(
                    plane_colour,
                    width=1.1,
                    style=Qt.PenStyle.DashLine,
                ),
            )
            line.setToolTip(
                f"{record.name}\n"
                f"image orientation from sample "
                f"{record.image_rotation_from_sample_deg:+.6g} deg\n"
                f"sample-to-plane Larmor integral "
                f"{record.larmor_rotation_from_sample_deg:+.6g} deg\n"
                f"|A| {record.magnification:.6g} | "
                f"||B|| {record.conjugacy_error_m:.6g} m/rad | "
                f"anisotropy {record.anisotropy_ratio:.6g}"
            )
            self.plot.addItem(line)
            self._rotation_items.append(line)
            y_value = peak * (0.92 - 0.13 * (index % 4))
            label = pg.TextItem(
                text=(
                    f"{record.name}\n"
                    f"θsample {record.image_rotation_from_sample_deg:+.3g}°"
                ),
                color=plane_colour,
                anchor=(0.5, 0.0),
                fill=pg.mkBrush(5, 8, 22, 210),
                border=pg.mkPen(plane_colour, width=0.9),
            )
            label.setPos(record.z_mm, y_value)
            label.setToolTip(line.toolTip())
            self.plot.addItem(label)
            self._rotation_items.append(label)
        self._apply_rotation_marker_visibility()

    def _apply_rotation_marker_visibility(self) -> None:
        visible = self.show_rotation_labels.isChecked()
        for item in self._rotation_items:
            item.setVisible(visible)

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
            f"field direction {'+Z' if record.polarity > 0 else '-Z'} | "
            f"lens Larmor rotation ΔφL {record.larmor_rotation_deg:+.6g} deg | "
            f"column cumulative ΣφL "
            f"{record.cumulative_column_rotation_deg:+.6g} deg | "
            f"Cs {cs_text} | "
            f"field support {record.support_mm[0]:.6g}–{record.support_mm[1]:.6g} mm"
        )
