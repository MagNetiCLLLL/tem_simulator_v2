"""PyQtGraph views for resolved TEM mechanics and axial magnetic fields."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
)

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPolygonF, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
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
    EDS_DETECTOR_SYSTEM,
    FLUORESCENT_SCREEN,
    POST_PROJECTOR_DETECTOR_CHAMBER,
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
from temsim.detector.plane_image import detector_response_image
from temsim.detector.eds_geometry import (
    EDSDetectorArrayGeometry,
    assess_axisymmetric_pole_centerline,
)
from temsim.mechanical_profiles import (
    FIXED_DIFFERENTIAL_PUMPING_APERTURE,
    POST_PROJECTOR_DETECTOR_CHAMBER as POST_PROJECTOR_DETECTOR_CHAMBER_PROFILE,
    TRANSVERSE_EDS_DETECTOR_ARRAY,
)
from temsim.physics.first_order import (
    linear_map_properties,
    relative_image_diffraction_orientation,
)
from temsim.gui.transverse_projection import (
    format_projection_angle,
    orthogonal_axis_name,
    projection_axis_name,
    transverse_view_coordinates,
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
        self.heading = QLabel(
            "Energy Filter physical layout and ray diagram"
        )
        self.summary = QLabel("Curvilinear Energy Filter branch")
        self.summary.setToolTip(
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

        physical_page = QWidget()
        physical_layout = QVBoxLayout(physical_page)
        physical_layout.setContentsMargins(0, 0, 0, 0)
        physical_layout.addWidget(self.plot, 1)

        self.spectrum_plot = pg.PlotWidget(background="#050816")
        self.spectrum_plot.setObjectName("energyFilterSpectrumPlot")
        self.spectrum_plot.setLabel("bottom", "Energy loss", units="eV")
        self.spectrum_plot.setLabel("left", "Detected counts")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.18)
        self.spectrum_curve = self.spectrum_plot.plot(
            [], [], pen=pg.mkPen("#67e8f9", width=1.8)
        )
        self.spectrum_cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#f8fafc", width=0.8, style=Qt.PenStyle.DashLine
            ),
        )
        self.spectrum_cursor.hide()
        self.spectrum_plot.addItem(self.spectrum_cursor)
        self.spectrum_point = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen("#f8fafc", width=1.0),
            brush=pg.mkBrush("#22d3ee"),
        )
        self.spectrum_point.hide()
        self.spectrum_plot.addItem(self.spectrum_point)
        self.spectrum_status = QLabel(
            "No cached High-accuracy EELS spectrum."
        )
        self.spectrum_status.setObjectName("energyFilterSpectrumStatus")
        self.spectrum_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.spectrum_status.setStyleSheet("color: #94a3b8;")
        spectrum_page = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_page)
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_layout.addWidget(self.spectrum_plot, 1)
        spectrum_layout.addWidget(self.spectrum_status)
        self._spectrum_energy_ev = np.asarray((), dtype=float)
        self._spectrum_counts = np.asarray((), dtype=float)
        self.spectrum_plot.scene().sigMouseMoved.connect(
            self._spectrum_mouse_moved
        )

        self.eftem_plot = pg.PlotWidget(background="#050816")
        self.eftem_plot.setObjectName("energyFilterEFTEMPlot")
        self.eftem_plot.setAspectLocked(True)
        self.eftem_plot.invertY(True)
        self.eftem_image_item = pg.ImageItem(axisOrder="row-major")
        self.eftem_image_item.hide()
        self.eftem_plot.addItem(self.eftem_image_item)
        self.eftem_status = QLabel(
            "No cached High-accuracy EFTEM image."
        )
        self.eftem_status.setObjectName("energyFilterEFTEMStatus")
        self.eftem_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.eftem_status.setStyleSheet("color: #94a3b8;")
        eftem_page = QWidget()
        eftem_layout = QVBoxLayout(eftem_page)
        eftem_layout.setContentsMargins(0, 0, 0, 0)
        eftem_layout.addWidget(self.eftem_plot, 1)
        eftem_layout.addWidget(self.eftem_status)

        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("energyFilterOutputTabs")
        self.output_tabs.addTab(physical_page, "Physical + rays")
        self.output_tabs.addTab(spectrum_page, "EELS spectrum")
        self.output_tabs.addTab(eftem_page, "EFTEM image")
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.output_tabs, 1)
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

    def _spectrum_mouse_moved(self, scene_position) -> None:
        """Read the nearest cached spectrum bin without recalculation."""

        if self._spectrum_energy_ev.size == 0:
            return
        view_box = self.spectrum_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return
        point = view_box.mapSceneToView(scene_position)
        index = int(np.argmin(np.abs(
            self._spectrum_energy_ev - float(point.x())
        )))
        energy_ev = float(self._spectrum_energy_ev[index])
        counts = float(self._spectrum_counts[index])
        self.spectrum_cursor.setPos(energy_ev)
        self.spectrum_cursor.show()
        self.spectrum_point.setData((energy_ev,), (counts,))
        self.spectrum_point.show()
        self.spectrum_status.setText(
            f"Energy {energy_ev:.6g} eV | counts {counts:.6g}"
        )

    def _display_scientific_outputs(self, branch_result, mode: str) -> None:
        """Render only data already attached to the completed result."""

        forward = getattr(branch_result, "eels_forward", None)
        if forward is None:
            self._spectrum_energy_ev = np.asarray((), dtype=float)
            self._spectrum_counts = np.asarray((), dtype=float)
            self.spectrum_curve.setData([], [])
            self.spectrum_cursor.hide()
            self.spectrum_point.hide()
            self.spectrum_status.setText(
                "No cached High-accuracy EELS spectrum."
            )
        else:
            energy = np.asarray(forward.energy_loss_ev, dtype=float)
            sampled = getattr(forward, "detected_sampled_counts", None)
            counts = np.asarray(
                sampled
                if sampled is not None
                else forward.detected_expected_counts,
                dtype=float,
            )
            if (
                energy.ndim != 1
                or counts.shape != energy.shape
                or not np.all(np.isfinite(energy))
                or not np.all(np.isfinite(counts))
            ):
                self._spectrum_energy_ev = np.asarray((), dtype=float)
                self._spectrum_counts = np.asarray((), dtype=float)
                self.spectrum_curve.setData([], [])
                self.spectrum_status.setText("Cached EELS spectrum is invalid.")
            else:
                self._spectrum_energy_ev = energy
                self._spectrum_counts = counts
                self.spectrum_curve.setData(energy, counts)
                self.spectrum_plot.autoRange()
                kind = "sampled" if sampled is not None else "expected"
                self.spectrum_status.setText(
                    f"{energy.size:,} bins | {kind} total counts "
                    f"{float(np.sum(counts)):.6g} | hover for bin values"
                )

        eftem_image = getattr(branch_result, "eftem_image", None)
        if str(mode).lower() != "eftem":
            self.eftem_image_item.hide()
            self.eftem_status.setText(
                "EFTEM image is available when acquisition mode is EFTEM."
            )
        elif eftem_image is None:
            self.eftem_image_item.hide()
            self.eftem_status.setText(
                "No cached High-accuracy EFTEM image."
            )
        else:
            image = np.asarray(eftem_image, dtype=float)
            if (
                image.ndim != 2
                or not np.all(np.isfinite(image))
                or np.any(image < 0.0)
            ):
                self.eftem_image_item.hide()
                self.eftem_status.setText("Cached EFTEM image is invalid.")
            else:
                self.eftem_image_item.setImage(image, autoLevels=True)
                self.eftem_image_item.show()
                self.eftem_plot.autoRange()
                self.eftem_status.setText(
                    f"Cached EFTEM image | {image.shape[1]} × "
                    f"{image.shape[0]} px"
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
        self._display_scientific_outputs(None, "")
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
        mode = str(energy_filter.operating_mode).upper()
        self._display_scientific_outputs(branch_result, mode)
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
        detail_text = (
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
        self.summary.setText(
            "TOML layout: 1 prism + 10 multipoles | "
            f"M01-M03 {entrance_carrier.housing_length_m * 1.0e3:g} mm | "
            f"M04-M10 {exit_carrier.housing_length_m * 1.0e3:g} mm"
            + metric_text
            + result_text
        )
        self.summary.setToolTip(detail_text)
        self.plot.autoRange()
        self._layout_labels()

    def mark_result_stale(self) -> None:
        """Keep the last complete branch trace visible after input changes."""

        self.summary.setText(
            "Previous Energy Filter trace retained | inputs changed"
        )
        self.summary.setToolTip(
            "Run High accuracy to update the branch ray trace for the current state."
        )
        if self._spectrum_energy_ev.size:
            self.spectrum_status.setText(
                "Previous complete EELS spectrum retained | inputs changed"
            )
        if self.eftem_image_item.isVisible():
            self.eftem_status.setText(
                "Previous complete EFTEM image retained | inputs changed"
            )


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
    component_activated = Signal(str, float)
    axial_position_selected = Signal(float)
    RECORDING_SURFACE_PROFILES = frozenset({
        "retractable_detector_plane",
        "camera_sensor_plane",
    })
    APERTURE_MECHANISM_PROFILE = "adjustable_circular_aperture"
    APERTURE_PLATE_COLOUR = "#e2e8f0"
    APERTURE_ROD_COLOUR = "#f59e0b"
    APERTURE_SCREW_COLOUR = "#f8fafc"
    APERTURE_COLUMN_WALL_PROFILES = frozenset({
        "accelerator_stack",
        "magnetic_lens_housing",
    })
    APERTURE_COLUMN_WALL_SEARCH_DISTANCE_MM = 25.0
    APERTURE_ROD_OVERHANG_MM = 5.0
    ACCELERATOR_STACK_PROFILE = "accelerator_stack"
    ACCELERATOR_STAGE_COLOURS = ("#cbd5e1", "#94a3b8")
    ACCELERATOR_SEPARATOR_COLOUR = "#f59e0b"
    EDS_DETECTOR_ARRAY_PROFILE = TRANSVERSE_EDS_DETECTOR_ARRAY
    EDS_ACTIVE_FACE_COLOUR = "#22d3ee"
    EDS_HOUSING_COLOUR = "#0e7490"
    EDS_ACCEPTANCE_COLOUR = "#67e8f9"
    # Display-only separation for unpublished EDS head dimensions. It is
    # deliberately not stored as product geometry in TOML.
    EDS_POLE_DISPLAY_CLEARANCE_MM = 1.0
    DETECTOR_CHAMBER_PROFILE = POST_PROJECTOR_DETECTOR_CHAMBER_PROFILE
    FIXED_DPA_PROFILE = FIXED_DIFFERENTIAL_PUMPING_APERTURE
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
        EDS_DETECTOR_SYSTEM,
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
        self._c1_c2_pole_piece_cartridge_items = []
        self._c1_c2_pole_gap = None
        self._objective_lens_half_items = []
        self._objective_lens_labels = []
        self._lens_excitation_coil_items = {}
        self._sample_stage_items = []
        self._sample_holder_items = []
        self._sample_plane_items = []
        self._sample_plane_labels = []
        self._eds_detector_items = {}
        self._eds_detector_labels = []
        self._detector_chamber_items = []
        self._accelerator_stack_items = {}
        self._aperture_mechanism_items = {}
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
        self._parameter_semantics_by_key = {}
        self._parameter_semantics_mode = None
        self._parameter_semantics_descriptors = None
        self._semantic_selected_summary = None

        self.heading = QLabel("Resolved mechanical layout")
        self.summary = QLabel(
            "Hollow-cylinder projections and vacuum bores come from the active TOML assembly."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.aperture_legend = QLabel(
            "Apertures: "
            "<span style='color:#64748b'>Pt strip</span> · "
            "<span style='color:#334155'>screw</span> · "
            "<span style='color:#b45309'>rear rod</span>"
        )
        self.aperture_legend.setToolTip(
            "The Pt perforated strip, screw joint and rear connecting rod are "
            "drawn separately. The rod extends 5 mm beyond the locally shown "
            "column wall; dimensions without a calibrated reference remain "
            "schematic."
        )
        self.aperture_legend.setTextFormat(Qt.TextFormat.RichText)
        self.aperture_legend.setWordWrap(True)
        self.aperture_legend.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.aperture_legend.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        self.accelerator_legend = QLabel(
            "Accelerator: "
            "<span style='color:#64748b'>electrostatic ring stages</span>"
        )
        self.accelerator_legend.setToolTip(
            "Repeated metal rings represent electrostatic accelerator stages, "
            "not magnetic coils. Ring thickness and separators are schematic."
        )
        self.accelerator_legend.setTextFormat(Qt.TextFormat.RichText)
        self.accelerator_legend.setWordWrap(True)
        self.accelerator_legend.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.accelerator_legend.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        self.eds_legend = QLabel(
            "EDS: six sample-facing segments · 4.04 sr with holder"
        )
        self.eds_legend.setToolTip(
            "The two visible heads are azimuthal projections of a six-segment "
            "windowless detector array. The >=4.45 sr unshadowed and 4.04 sr "
            "holder-conditioned acceptances are physical metadata. Public "
            "crystal and package dimensions are unavailable, so head sizes are "
            "schematic."
        )
        self.eds_legend.setWordWrap(True)
        self.eds_legend.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.eds_legend.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        self.fit_all = QPushButton("Fit all hardware")
        self.fit_bore = QPushButton("Fit vacuum bore")
        for button in (self.fit_all, self.fit_bore):
            button.setStyleSheet(BUTTON_STYLE)

        heading_row = QHBoxLayout()
        heading_row.addWidget(self.heading)
        heading_row.addStretch(1)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.fit_bore)
        action_row.addWidget(self.fit_all)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("physicalLayoutPlot")
        self.plot.setToolTip(
            "Click to select a component. Double-click to locate it in the 3D model editor."
        )
        self.plot.setLabel("bottom", "Axial position", units="mm")
        self.plot.setLabel("left", "Mechanical radius", units="mm")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        view_box = self.plot.getViewBox()
        view_box.setAspectLocked(False)
        view_box.setMouseEnabled(x=True, y=True)

        self.section_page = QWidget()
        layout = QVBoxLayout(self.section_page)
        layout.addLayout(heading_row)
        layout.addLayout(action_row)
        layout.addWidget(self.aperture_legend)
        layout.addWidget(self.accelerator_legend)
        layout.addWidget(self.eds_legend)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)

        from temsim.gui.part_model_editor import PartModelEditorPage
        self.model_editor = PartModelEditorPage()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("physicalLayoutTabs")
        self.tabs.addTab(self.section_page, "2D section")
        self.tabs.addTab(self.model_editor, "3D model editor")
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.tabs)

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
        self.plot.getViewBox().sigResized.connect(self._layout_component_labels)

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
        key = _selectable_key_at_scene_position(
            self.plot.scene(), event.scenePos(), self._selectable_item_keys
        )
        if key is None:
            key = self.component_key_at(float(position.x()), float(position.y()))
        if key is not None:
            self.component_activated.emit(key, float(position.x()))
        self.axial_position_selected.emit(float(position.x()))
        event.accept()

    def component_key_at(self, z_mm: float, radius_mm: float = 0.0) -> str | None:
        """Resolve empty section space to the nearest component envelope.

        Actual graphics and labels take precedence in the click handler. For
        gaps, prefer the nearest axial interval, then the nearest radial wall.
        """
        if not self._records or not np.isfinite((z_mm, radius_mm)).all():
            return None
        radius = abs(radius_mm)

        def distance(record):
            lower, upper = sorted((record.start_z_mm, record.end_z_mm))
            inner = 0.5 * record.mechanical_bore_diameter_mm
            outer = max(inner, 0.5 * record.outer_diameter_mm)
            return (
                max(lower - z_mm, z_mm - upper, 0.0),
                max(inner - radius, radius - outer, 0.0),
                abs(record.center_z_mm - z_mm),
                upper - lower,
                record.key,
            )

        return min(self._records, key=distance).key

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
        """Draw one hollow pole piece without treating its bore as a tube."""

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
        style = str(record.pole_piece_geometry_style)
        if style == "objective_vertical_back_inserted_shank_tapered_nose":
            shape_name = (
                "Objective pole piece: inserted mounting shank / shoulder / "
                "tapered nose"
            )
        elif style == "objective_mushroom_bore_stem":
            shape_name = "Objective mushroom/arrow pole-piece projection"
        elif style == "embedded_hourglass_bore":
            shape_name = "Embedded condenser hourglass pole-piece projection"
        else:
            shape_name = "Hollow tapered pole-piece projection"
        tooltip = (
            f"{record.name}\n{shape_name}\n"
            f"Z {start:.6g}–{end:.6g} mm\n"
            f"beam-path bore ID {record.bore_diameter_mm:.6g} mm | "
            f"tip OD {2.0 * tip:.6g} mm"
            "\nThe bore is open vacuum space; a separately drawn thin "
            "sleeve is the non-magnetic vacuum liner tube."
        )
        if configured_nose > 0.0:
            tooltip += f"\nNose axial length {configured_nose:.6g} mm"
        if record.pole_mounting_shank_axial_length_mm > 0.0:
            tooltip += (
                "\nMounting shank "
                f"ID {record.pole_mounting_shank_inner_diameter_mm:.6g} mm | "
                f"OD {record.pole_stem_outer_diameter_mm:.6g} mm | "
                "axial insertion "
                f"{record.pole_mounting_shank_axial_length_mm:.6g} mm"
            )
        if record.pole_vacuum_connector_axial_length_mm > 0.0:
            tooltip += (
                "\nVacuum connector tail "
                f"ID {record.bore_diameter_mm:.6g} mm | "
                f"OD {record.pole_vacuum_connector_outer_diameter_mm:.6g} "
                "mm | axial length "
                f"{record.pole_vacuum_connector_axial_length_mm:.6g} mm"
            )
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
            if style == "objective_vertical_back_inserted_shank_tapered_nose":
                shank_outer = 0.5 * float(
                    record.pole_stem_outer_diameter_mm
                )
                shank_inner = 0.5 * float(
                    record.pole_mounting_shank_inner_diameter_mm
                )
                insertion = min(
                    float(record.pole_mounting_shank_axial_length_mm),
                    max(length - taper_length - 0.001, 0.0),
                )
                direction = 1.0 if face_at_end else -1.0
                mounting_end_z = outside_z + direction * insertion
                connector_outer = min(
                    0.5 * float(
                        record.pole_vacuum_connector_outer_diameter_mm
                    ),
                    shank_outer,
                )
                connector_length = min(
                    float(record.pole_vacuum_connector_axial_length_mm),
                    max(insertion - 0.001, 0.0),
                )
                connector_end_z = outside_z + direction * connector_length
                if connector_outer <= shank_inner or connector_length <= 0.0:
                    connector_outer = shank_outer
                    connector_end_z = outside_z
                points = QPolygonF([
                    QPointF(outside_z, sign * shank_inner),
                    QPointF(outside_z, sign * connector_outer),
                    QPointF(connector_end_z, sign * connector_outer),
                    QPointF(connector_end_z, sign * shank_outer),
                    QPointF(mounting_end_z, sign * shank_outer),
                    QPointF(mounting_end_z, sign * outer),
                    QPointF(shoulder_z, sign * outer),
                    QPointF(face_z, sign * tip),
                    QPointF(face_z, sign * bore),
                    QPointF(mounting_end_z, sign * bore),
                    QPointF(mounting_end_z, sign * shank_inner),
                    QPointF(connector_end_z, sign * shank_inner),
                ])
            elif style == "objective_mushroom_bore_stem":
                stem = 0.5 * float(record.pole_stem_outer_diameter_mm)
                stem = min(max(stem, bore), outer)
                direction = 1.0 if face_at_end else -1.0
                neck_z = outside_z + direction * 0.46 * length
                head_z = outside_z + direction * 0.56 * length
                points = QPolygonF([
                    QPointF(outside_z, sign * bore),
                    QPointF(outside_z, sign * stem),
                    QPointF(neck_z, sign * stem),
                    QPointF(head_z, sign * outer),
                    QPointF(shoulder_z, sign * outer),
                    QPointF(face_z, sign * tip),
                    QPointF(face_z, sign * bore),
                ])
            else:
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

    def _add_magnetic_radial_profile(self, record, colour, profile) -> None:
        """Draw the same piecewise-linear radii used by the material solver."""
        z = record.start_z_mm + profile[:, 0]
        for sign in (-1, 1):
            points = list(zip(z, sign*profile[:, 2])) + list(zip(z[::-1], sign*profile[::-1, 1]))
            polygon = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
            polygon.setPen(pg.mkPen(colour, width=0.9))
            rgb = pg.mkColor(colour)
            polygon.setBrush(pg.mkBrush(rgb.red(), rgb.green(), rgb.blue(), 105))
            polygon.setToolTip(f"{record.name}\nExplicit magnetic radial profile (mm); shared with field geometry")
            self.plot.addItem(polygon)
            _register_selectable_graphics_item(self._selectable_item_keys, polygon, record.key)

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

    def _objective_lens_active_intervals(self, record):
        """Use the same resolved material intervals as the field solver."""
        from temsim.magnetic_geometry import objective_layer_intervals_mm
        objective = self._record_by_key.get("objective_lens")
        part = self._part_by_key.get("objective_lens")
        if objective is None or part is None:
            return ()
        intervals = objective_layer_intervals_mm(part.data, objective.start_z_mm, record.profile)
        return tuple((name, start, end) for name, (start, end) in
                     zip(("Upper Objective Lens", "Lower Objective Lens"), intervals))

    def _add_split_objective_lens_layer(self, record, colour) -> None:
        """Draw one Objective layer as separate upper and lower bodies."""

        intervals = self._objective_lens_active_intervals(record)
        if not intervals:
            return
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
                "This active upper/lower body interval comes from the "
                "separate Objective yoke ranges in the column TOML; the "
                "central region is reserved for pole pieces, stage, holder, "
                "cold trap and analytical-detector access."
            )
            if record.profile == "magnetic_excitation_coil":
                radial_thickness = 0.5 * (
                    float(record.outer_diameter_mm)
                    - float(record.bore_diameter_mm)
                )
                tooltip += (
                    f"\nAxial winding length "
                    f"{end - start:.6g} mm; radial "
                    f"winding thickness {radial_thickness:.6g} mm."
                    "\nProvisional non-OEM geometry: each Objective coil "
                    "half is inset from its yoke body; the thick radial "
                    "winding is an explicit Thermo/FEI-directed engineering "
                    "reconstruction, not an OEM measurement."
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
                if record.profile == "magnetic_excitation_coil":
                    self._lens_excitation_coil_items.setdefault(
                        record.key, []
                    ).append(rect)
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
        state_snapshot = getattr(self._result, "state_snapshot", None)
        sample_state = getattr(state_snapshot, "sample", None)
        inserted = bool(getattr(sample_state, "inserted", True))
        sample_status = "INSERTED" if inserted else "RETRACTED"
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
        sleeve.setZValue(47)
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
        body.setZValue(47)
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
            f"Sample Holder (schematic; {sample_status.lower()})\n"
            "The holder enters through the positive-radius side. "
            + (
                "Its tip terminates on the optical sample reference plane. "
                if inserted
                else "Its tip is parked outside the Objective pole gap. "
            )
            + f"Reference-plane Z = {sample_z:.6g} mm."
        )
        holder_tip_y = 0.0 if inserted else stage_body_start
        shaft = QGraphicsRectItem(
            sample_z - holder_half_width,
            holder_tip_y,
            2.0 * holder_half_width,
            max(holder_end - holder_tip_y, 0.5),
        )
        holder_alpha = 205 if inserted else 92
        holder_style = (
            Qt.PenStyle.SolidLine if inserted else Qt.PenStyle.DashLine
        )
        shaft.setPen(
            pg.mkPen("#f59e0b", width=1.0, style=holder_style)
        )
        shaft.setBrush(pg.mkBrush(245, 158, 11, holder_alpha))
        shaft.setToolTip(holder_tooltip)
        shaft.setZValue(48)
        self.plot.addItem(shaft)
        self._sample_holder_items.append(shaft)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            shaft,
            "sample_stage",
        )

        tip_half_width = min(0.9, 0.3 * gap_width)
        tip_depth = min(
            pole_tip_radius,
            max(holder_end - holder_tip_y, 0.5),
        )
        tip = QGraphicsPolygonItem(QPolygonF([
            QPointF(sample_z, holder_tip_y),
            QPointF(sample_z - tip_half_width, holder_tip_y + tip_depth),
            QPointF(sample_z + tip_half_width, holder_tip_y + tip_depth),
        ]))
        tip.setPen(pg.mkPen("#fbbf24", width=1.0, style=holder_style))
        tip.setBrush(
            pg.mkBrush(251, 191, 36, 225 if inserted else 105)
        )
        tip.setToolTip(holder_tooltip)
        tip.setZValue(49)
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
        grip.setZValue(49)
        self.plot.addItem(grip)
        self._sample_holder_items.append(grip)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            grip,
            "sample_stage",
        )

        sample_radius = max(0.5 * float(sample.outer_diameter_mm), 1.5)
        sample_colour = "#fb7185" if inserted else "#94a3b8"
        sample_pen_style = (
            Qt.PenStyle.SolidLine if inserted else Qt.PenStyle.DashLine
        )
        sample_line = self.plot.plot(
            [sample_z, sample_z],
            [-sample_radius, sample_radius],
            pen=pg.mkPen(
                sample_colour,
                width=3.0 if inserted else 1.5,
                style=sample_pen_style,
            ),
        )
        sample_line.setZValue(34)
        sample_line.setToolTip(
            f"Sample {sample_status.lower()} / optical reference plane\n"
            f"Z = {sample_z:.6g} mm"
        )
        sample_marker = QGraphicsEllipseItem(
            sample_z - 0.32,
            -0.32,
            0.64,
            0.64,
        )
        sample_marker.setPen(pg.mkPen(sample_colour, width=1.2))
        sample_marker.setBrush(
            pg.mkBrush(251, 113, 133, 235)
            if inserted
            else pg.mkBrush(0, 0, 0, 0)
        )
        sample_marker.setToolTip(
            f"Sample {sample_status.lower()}\n"
            f"Optical reference-plane Z = {sample_z:.6g} mm"
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
            (
                "SAMPLE / SPECIMEN"
                if inserted
                else "SAMPLE RETRACTED / REFERENCE PLANE"
            ),
            color="#fecdd3" if inserted else "#cbd5e1",
            anchor=(0.5, 0.5),
            border=pg.mkPen(sample_colour, width=0.8),
            fill=pg.mkBrush(5, 8, 22, 215),
        )
        sample_label.setZValue(46)
        sample_label.setToolTip(
            f"Sample {sample_status.lower()}\n"
            f"Optical reference-plane Z = {sample_z:.6g} mm"
        )
        self.plot.addItem(sample_label)
        self._sample_plane_labels.append(sample_label)
        self._register_label_callout(
            key="sample:specimen",
            label=sample_label,
            anchor_z_mm=sample_z,
            anchor_radius_mm=sample_radius,
            colour=sample_colour,
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

    def _remember_eds_detector_item(self, role: str, item) -> None:
        self._eds_detector_items.setdefault(role, []).append(item)
        # PlotDataItem hit shapes can span the whole line bounding rectangle,
        # so registering acceptance/centre lines would steal clicks from the
        # sample stage.  The exact-shape housing polygons and label remain
        # selectable entry points for the EDS aggregate.
        if role == "housing":
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                item,
                EDS_DETECTOR_SYSTEM,
            )

    def _eds_pole_centerline_assessment(self, geometry):
        upper = self._record_by_key.get("objective_upper_pole")
        lower = self._record_by_key.get("objective_lower_pole")
        if upper is None or lower is None:
            return None
        noses = []
        for pole in (upper, lower):
            configured = float(pole.pole_nose_axial_length_mm)
            noses.append(
                configured
                if configured > 0.0
                else 0.38 * max(float(pole.end_z_mm - pole.start_z_mm), 0.0)
            )
        return assess_axisymmetric_pole_centerline(
            takeoff_angle_deg=geometry.takeoff_angle_deg,
            pole_gap_mm=float(lower.start_z_mm - upper.end_z_mm),
            pole_bore_diameter_mm=max(
                float(upper.bore_diameter_mm),
                float(lower.bore_diameter_mm),
            ),
            pole_tip_diameter_mm=max(
                float(upper.pole_tip_diameter_mm),
                float(lower.pole_tip_diameter_mm),
            ),
            pole_outer_diameter_mm=max(
                float(upper.outer_diameter_mm),
                float(lower.outer_diameter_mm),
            ),
            pole_nose_axial_length_mm=min(noses),
        )

    def _add_eds_detector_array_schematic(self) -> None:
        """Draw angular acceptance without claiming unpublished head sizes."""

        record = self._record_by_key.get(EDS_DETECTOR_SYSTEM)
        part = self._part_by_key.get(EDS_DETECTOR_SYSTEM)
        objective = self._record_by_key.get("objective_lens")
        upper_pole = self._record_by_key.get("objective_upper_pole")
        lower_pole = self._record_by_key.get("objective_lower_pole")
        if any(
            item is None
            for item in (record, part, objective, upper_pole, lower_pole)
        ):
            return

        geometry = EDSDetectorArrayGeometry.from_part_data(part.data)
        sample_z = float(record.center_z_mm)
        takeoff_rad = math.radians(geometry.takeoff_angle_deg)
        cone_half_angle_deg = (
            geometry.equivalent_circular_cone_half_angle_deg()
        )
        holder_cone_half_angle_deg = (
            geometry.equivalent_circular_cone_half_angle_deg(
                analytical_holder=True
            )
        )
        pole_assessment = self._eds_pole_centerline_assessment(geometry)
        pole_tip_radius = max(
            0.5 * float(upper_pole.pole_tip_diameter_mm),
            0.5 * float(lower_pole.pole_tip_diameter_mm),
            2.5,
        )
        objective_radius = 0.5 * float(objective.outer_diameter_mm)

        # Display-only placement. Public reference material does not provide
        # head/package dimensions, so keep a small fixed schematic head and
        # solve its distance so every solid vertex clears the largest resolved
        # upper/lower pole-piece radius.  The configured centre line is
        # checked against the pole profile below.  Equivalent-acceptance
        # boundaries may still cross the axisymmetric projection because they
        # are aggregate angular metadata, not literal solid apertures.
        face_half_width = max(
            0.45 * pole_tip_radius,
            0.025 * objective_radius,
        )
        housing_length = max(
            0.8 * pole_tip_radius,
            0.05 * objective_radius,
        )
        inner_half_width = 1.25 * face_half_width
        outer_half_width = 0.82 * face_half_width
        pole_outer_radius = max(
            0.5 * float(upper_pole.outer_diameter_mm),
            0.5 * float(lower_pole.outer_diameter_mm),
        )
        drawing_distance = (
            pole_outer_radius
            + self.EDS_POLE_DISPLAY_CLEARANCE_MM
            + inner_half_width * math.sin(takeoff_rad)
        ) / math.cos(takeoff_rad)
        if pole_assessment is None:
            pole_clearance_text = ""
        else:
            pole_clearance_text = (
                "Configured non-OEM pole profile: reference centre-ray "
                f"minimum radial clearance {pole_assessment.minimum_radial_clearance_mm:.4g} "
                "mm in the axisymmetric meridional model ("
                f"{pole_assessment.face_radial_clearance_mm:.4g} mm at the "
                "flat face; "
                f"{pole_assessment.shoulder_radial_clearance_mm:.4g} mm at "
                "the cone shoulder). At the retained gap and pole OD, zero "
                "clearance requires tip OD <= "
                f"{pole_assessment.maximum_tip_diameter_mm_for_margin:.4g} "
                "mm and nose length >= "
                f"{pole_assessment.minimum_nose_axial_length_mm_for_margin:.4g} "
                "mm. Translating a detector along the same take-off ray "
                "does not change these intersections.\n"
            )
        tooltip = (
            "EDS detector array\n"
            f"{geometry.segment_count} windowless SDD segments around the "
            "sample; this meridional view projects two opposing azimuths.\n"
            f"Reference take-off angle {geometry.takeoff_angle_deg:.4g} deg "
            "(single user dataset; not a universal OEM value).\n"
            f"Unshadowed solid angle >= "
            f"{geometry.minimum_unshadowed_solid_angle_sr:.4g} sr total / "
            f">= {geometry.minimum_unshadowed_solid_angle_per_segment_sr:.4g} "
            "sr per equal-segment summary.\n"
            f"Analytical double-tilt holder: "
            f"{geometry.analytical_holder_solid_angle_sr:.4g} sr total.\n"
            f"Equivalent circular-cone half-angle: "
            f"{cone_half_angle_deg:.3g} deg unshadowed / "
            f"{holder_cone_half_angle_deg:.3g} deg with holder.\n"
            f"The configured segment axes are only "
            f"{geometry.minimum_axis_separation_deg:.3g} deg apart, less "
            f"than the {2.0 * cone_half_angle_deg:.3g} deg equivalent-cone "
            "diameter, so six such circular cones overlap. They summarize "
            "aggregate angular acceptance and are not literal, disjoint "
            "sensor or collimator apertures.\n"
            + pole_clearance_text
            + "Active "
            "area, sensor distance, crystal shape and external package "
            "dimensions are not public; the drawn heads and distance are "
            "explicitly non-dimensional schematics pending user cross-sections.\n"
            f"Solid head polygons use a {self.EDS_POLE_DISPLAY_CLEARANCE_MM:g} mm "
            "display-only clearance outside the resolved pole-piece OD. This "
            "prevents a false 2D material overlap but is not an OEM product "
            "clearance or a validated 3D collimator/shadowing model.\n"
            "This off-axis X-ray array does not intercept the axial electron "
            "beam and is not an electron recording plane."
        )

        for sign in (-1.0, 1.0):
            direction_z = -math.sin(takeoff_rad)
            direction_y = sign * math.cos(takeoff_rad)
            perpendicular_z = math.cos(takeoff_rad)
            perpendicular_y = sign * math.sin(takeoff_rad)
            face_center_z = sample_z + drawing_distance * direction_z
            face_center_y = drawing_distance * direction_y

            centerline = self.plot.plot(
                [sample_z, face_center_z],
                [0.0, face_center_y],
                pen=pg.mkPen(
                    self.EDS_ACTIVE_FACE_COLOUR,
                    width=1.1,
                    style=Qt.PenStyle.DashDotLine,
                ),
            )
            centerline.setZValue(51)
            centerline.setToolTip(tooltip)
            self._remember_eds_detector_item("centerline", centerline)

            for boundary_angle_deg in (
                max(0.1, geometry.takeoff_angle_deg - cone_half_angle_deg),
                min(89.9, geometry.takeoff_angle_deg + cone_half_angle_deg),
            ):
                boundary_angle = math.radians(boundary_angle_deg)
                boundary = self.plot.plot(
                    [sample_z, sample_z - drawing_distance * math.sin(
                        boundary_angle
                    )],
                    [0.0, sign * drawing_distance * math.cos(
                        boundary_angle
                    )],
                    pen=pg.mkPen(
                        self.EDS_ACCEPTANCE_COLOUR,
                        width=0.8,
                        style=Qt.PenStyle.DotLine,
                    ),
                )
                boundary.setZValue(50)
                boundary.setOpacity(0.58)
                boundary.setToolTip(tooltip)
                self._remember_eds_detector_item("acceptance", boundary)

            face_z = (
                face_center_z - face_half_width * perpendicular_z,
                face_center_z + face_half_width * perpendicular_z,
            )
            face_y = (
                face_center_y - face_half_width * perpendicular_y,
                face_center_y + face_half_width * perpendicular_y,
            )
            face = self.plot.plot(
                face_z,
                face_y,
                pen=pg.mkPen(self.EDS_ACTIVE_FACE_COLOUR, width=4.0),
            )
            face.setZValue(53)
            face.setToolTip(tooltip)
            self._remember_eds_detector_item("active_face", face)

            outer_center_z = face_center_z + housing_length * direction_z
            outer_center_y = face_center_y + housing_length * direction_y
            housing = QGraphicsPolygonItem(QPolygonF([
                QPointF(
                    face_center_z - inner_half_width * perpendicular_z,
                    face_center_y - inner_half_width * perpendicular_y,
                ),
                QPointF(
                    face_center_z + inner_half_width * perpendicular_z,
                    face_center_y + inner_half_width * perpendicular_y,
                ),
                QPointF(
                    outer_center_z + outer_half_width * perpendicular_z,
                    outer_center_y + outer_half_width * perpendicular_y,
                ),
                QPointF(
                    outer_center_z - outer_half_width * perpendicular_z,
                    outer_center_y - outer_half_width * perpendicular_y,
                ),
            ]))
            housing_colour = pg.mkColor(self.EDS_HOUSING_COLOUR)
            housing.setPen(pg.mkPen(self.EDS_ACTIVE_FACE_COLOUR, width=0.9))
            housing.setBrush(pg.mkBrush(
                housing_colour.red(),
                housing_colour.green(),
                housing_colour.blue(),
                150,
            ))
            housing.setZValue(52)
            housing.setToolTip(tooltip)
            self.plot.addItem(housing)
            self._remember_eds_detector_item("housing", housing)

        label = pg.TextItem(
            "ULTRA-X EDS\n6 segments; 2 projected",
            color="#cffafe",
            anchor=(0.5, 0.5),
            border=pg.mkPen(self.EDS_ACTIVE_FACE_COLOUR, width=0.8),
            fill=pg.mkBrush(5, 8, 22, 220),
        )
        # Use the same stacking level as the existing component callouts so a
        # packed EDS label cannot steal clicks from visible stage mechanics.
        label.setZValue(46)
        label.setToolTip(tooltip)
        self.plot.addItem(label)
        self._eds_detector_labels.append(label)
        self._register_label_callout(
            key="eds:detector_array",
            label=label,
            anchor_z_mm=sample_z + drawing_distance * (-math.sin(takeoff_rad)),
            anchor_radius_mm=drawing_distance * math.cos(takeoff_rad),
            colour=self.EDS_ACTIVE_FACE_COLOUR,
            priority=-5,
            preferred_side=-1,
            component_key=EDS_DETECTOR_SYSTEM,
        )

    def _add_post_projector_detector_chamber_schematic(
        self, record, colour
    ) -> None:
        """Draw the non-OEM Titan-topology viewing/detector chamber walls."""

        part = self._part_by_key.get(record.key)
        if part is None:
            return
        width = float(record.end_z_mm - record.start_z_mm)
        inner_radius = 0.5 * float(record.bore_diameter_mm)
        outer_radius = 0.5 * float(record.outer_diameter_mm)
        if width <= 0.0 or outer_radius <= inner_radius:
            return
        p2 = self._record_by_key.get("projector_lens_2")
        haadf = self._record_by_key.get("haadf")
        gap_text = "unknown"
        if p2 is not None and haadf is not None:
            gap_text = (
                f"{self._recording_signal_z(haadf) - p2.end_z_mm:.6g} mm"
            )
        source = str(part.data.get("mechanical_geometry_source", "")).strip()
        tooltip = (
            f"{record.name}\n"
            f"Z = [{record.start_z_mm:.6g}, {record.end_z_mm:.6g}] mm\n"
            f"schematic chamber ID/OD = "
            f"{record.bore_diameter_mm:.6g}/{record.outer_diameter_mm:.6g} mm\n"
            f"P2-end to HAADF active-plane gap = {gap_text}\n"
            "Titan public diagrams support the post-P2 viewing/detector "
            "chamber topology and detector order, not these absolute "
            "dimensions. The chamber is mechanical-only; all active detector "
            "planes retain their existing TOML coordinates and optical "
            "behaviour."
            + (f"\nSource/status: {source}" if source else "")
        )
        rgb = pg.mkColor(colour)
        for lower_radius in (-outer_radius, inner_radius):
            wall = QGraphicsRectItem(
                record.start_z_mm,
                lower_radius,
                width,
                outer_radius - inner_radius,
            )
            wall.setPen(pg.mkPen(colour, width=1.0))
            wall.setBrush(pg.mkBrush(
                rgb.red(), rgb.green(), rgb.blue(), 58
            ))
            wall.setToolTip(tooltip)
            wall.setZValue(8)
            self.plot.addItem(wall)
            self._detector_chamber_items.append(wall)
            _register_selectable_graphics_item(
                self._selectable_item_keys,
                wall,
                POST_PROJECTOR_DETECTOR_CHAMBER,
            )

    def _remember_accelerator_item(self, key: str, role: str, item) -> None:
        roles = self._accelerator_stack_items.setdefault(key, {})
        roles.setdefault(role, []).append(item)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            item,
            key,
        )

    @staticmethod
    def _accelerator_stage_drawing_width(record, index: int) -> float:
        """Return an explicitly schematic width derived from stage spacing."""

        centers = tuple(record.accelerator_stage_centers_mm)
        center = float(centers[index])
        limits = [
            2.0 * max(center - float(record.start_z_mm), 0.001),
            2.0 * max(float(record.end_z_mm) - center, 0.001),
        ]
        if index > 0:
            limits.append(0.46 * (center - float(centers[index - 1])))
        if index + 1 < len(centers):
            limits.append(0.46 * (float(centers[index + 1]) - center))
        return max(min(limits), 0.25)

    def _add_accelerator_stack_schematic(self, record) -> None:
        """Draw photo-informed repeated electrostatic electrode stages."""

        centers = tuple(float(value) for value in (
            record.accelerator_stage_centers_mm
        ))
        if len(centers) < 2:
            raise ValueError(
                f"Accelerator stack {record.key} requires at least two stages"
            )
        outer_radius = 0.5 * float(record.outer_diameter_mm)
        bore_radius = min(
            0.5 * float(record.mechanical_bore_diameter_mm),
            outer_radius,
        )
        stage_outer_radius = max(
            bore_radius,
            0.91 * outer_radius,
        )
        source = (
            record.accelerator_electrode_stack_evidence_source
            or "Accelerator ring-stack source is not declared."
        )
        common = (
            f"{record.name}\n"
            f"{len(centers)} configured electrostatic accelerator stages\n"
            f"Evidence: {source}\n"
            "The repeated rings are accelerator electrodes, not magnetic "
            "excitation coils."
        )

        envelope = QGraphicsRectItem(
            record.start_z_mm,
            -outer_radius,
            max(float(record.end_z_mm - record.start_z_mm), 0.001),
            2.0 * outer_radius,
        )
        envelope.setPen(pg.mkPen(
            "#64748b", width=0.9, style=Qt.PenStyle.DotLine
        ))
        envelope.setBrush(pg.mkBrush(0, 0, 0, 0))
        envelope.setToolTip(
            common
            + f"\nDashed TOML envelope OD {record.outer_diameter_mm:.6g} "
            f"mm | clear bore {record.mechanical_bore_diameter_mm:.6g} mm."
            " The envelope is not a solid cylinder."
        )
        envelope.setZValue(19)
        self.plot.addItem(envelope)
        self._remember_accelerator_item(record.key, "envelope", envelope)

        for index, center in enumerate(centers):
            drawing_width = self._accelerator_stage_drawing_width(
                record, index
            )
            colour = self.ACCELERATOR_STAGE_COLOURS[index % 2]
            rgb = pg.mkColor(colour)
            tooltip = (
                common
                + f"\nElectrostatic electrode stage {index + 1}/"
                f"{len(centers)} | configured center Z = {center:.6g} mm."
                f"\nDisplayed axial ring thickness {drawing_width:.6g} mm "
                "is derived only from stage spacing for visibility; it is "
                "not an OEM electrode thickness. Alternating shades separate "
                "stages visually and do not encode material or voltage."
            )
            for lower_y in (-stage_outer_radius, bore_radius):
                stage = QGraphicsRectItem(
                    center - 0.5 * drawing_width,
                    lower_y,
                    drawing_width,
                    stage_outer_radius - bore_radius,
                )
                stage.setPen(pg.mkPen("#e2e8f0", width=0.9))
                stage.setBrush(pg.mkBrush(
                    rgb.red(), rgb.green(), rgb.blue(), 172
                ))
                stage.setToolTip(tooltip)
                stage.setZValue(26)
                self.plot.addItem(stage)
                self._remember_accelerator_item(
                    record.key, "stage", stage
                )

        separator_height = max(0.08 * stage_outer_radius, 1.0)
        for upstream, downstream in zip(centers, centers[1:]):
            midpoint = 0.5 * (upstream + downstream)
            gap = downstream - upstream
            separator_width = min(1.6, 0.06 * gap)
            tooltip = (
                common
                + f"\nStage-separation marker at Z = {midpoint:.6g} mm."
                "\nThe narrow amber outer collars reproduce the visible "
                "stack rhythm only; their dimensions and material are not "
                "identified by the unscaled photograph."
            )
            for lower_y in (
                -stage_outer_radius,
                stage_outer_radius - separator_height,
            ):
                separator = QGraphicsRectItem(
                    midpoint - 0.5 * separator_width,
                    lower_y,
                    separator_width,
                    separator_height,
                )
                separator.setPen(pg.mkPen(
                    self.ACCELERATOR_SEPARATOR_COLOUR, width=0.8
                ))
                separator.setBrush(pg.mkBrush(245, 158, 11, 130))
                separator.setToolTip(tooltip)
                separator.setZValue(27)
                self.plot.addItem(separator)
                self._remember_accelerator_item(
                    record.key, "separator", separator
                )

    def _remember_aperture_item(self, key: str, role: str, item) -> None:
        roles = self._aperture_mechanism_items.setdefault(key, {})
        roles.setdefault(role, []).append(item)
        _register_selectable_graphics_item(
            self._selectable_item_keys,
            item,
            key,
        )

    @staticmethod
    def _aperture_plane_z(record) -> float:
        if len(record.optical_references_mm) != 1:
            raise ValueError(
                f"Aperture mechanism {record.key} requires one optical plane"
            )
        return float(record.optical_references_mm[0])

    @staticmethod
    def _axial_distance_to_record(z_mm: float, record) -> float:
        if record.start_z_mm <= z_mm <= record.end_z_mm:
            return 0.0
        return min(
            abs(z_mm - float(record.start_z_mm)),
            abs(z_mm - float(record.end_z_mm)),
        )

    def _aperture_column_wall_reference(self, record):
        """Resolve the local drawn column wall around one aperture.

        Nearby accelerator or magnetic-lens housings are explicit outer-wall
        geometry.  Where no such structure is locally adjacent (currently the
        Energy Filter entrance), the aperture mechanism's own TOML envelope
        remains the conservative local wall reference.
        """

        plate_z = self._aperture_plane_z(record)
        envelope_radius = max(
            0.5 * float(record.outer_diameter_mm),
            0.5 * float(record.vacuum_inner_diameter_mm) + 1.0,
        )
        candidates = []
        for candidate in self._records:
            if candidate.profile not in self.APERTURE_COLUMN_WALL_PROFILES:
                continue
            distance = self._axial_distance_to_record(plate_z, candidate)
            if distance > self.APERTURE_COLUMN_WALL_SEARCH_DISTANCE_MM:
                continue
            radius = 0.5 * float(candidate.outer_diameter_mm)
            if radius < envelope_radius:
                continue
            candidates.append((
                distance,
                -radius,
                str(candidate.key),
                candidate,
            ))
        if not candidates:
            return envelope_radius, record, 0.0
        distance, negative_radius, _key, source = min(candidates)
        return -negative_radius, source, distance

    def _add_aperture_mechanism_schematic(self, record) -> None:
        """Separate the real thin plate from its schematic radial carrier."""

        plate_z = self._aperture_plane_z(record)
        plate_thickness = float(record.active_length_mm)
        if plate_thickness <= 0.0:
            raise ValueError(
                f"Aperture mechanism {record.key} requires positive plate "
                "thickness metadata"
            )
        inserted = record.excitation_enabled is not False
        status = "INSERTED" if inserted else "RETRACTED"
        pen_style = (
            Qt.PenStyle.SolidLine
            if inserted else Qt.PenStyle.DashLine
        )
        plate_alpha = 225 if inserted else 76
        rod_alpha = 185 if inserted else 64

        opening_radius = 0.5 * float(record.bore_diameter_mm)
        hardware_bore_radius = 0.5 * float(
            record.mechanical_bore_diameter_mm
        )
        vacuum_radius = 0.5 * float(record.vacuum_inner_diameter_mm)
        envelope_radius = max(
            0.5 * float(record.outer_diameter_mm),
            vacuum_radius + 1.0,
        )
        margin = max(0.75, 0.15 * max(vacuum_radius, 1.0))
        negative_reach = max(
            vacuum_radius + margin,
            hardware_bore_radius + margin,
        )
        joint_local_y = max(
            negative_reach,
            min(0.60 * envelope_radius, vacuum_radius + 12.0),
        )
        plate_center_y = (
            0.0
            if inserted
            else vacuum_radius + negative_reach + margin
        )
        joint_y = plate_center_y + joint_local_y
        wall_radius, wall_source, wall_distance = (
            self._aperture_column_wall_reference(record)
        )
        rod_end_y = max(
            joint_y + 6.0,
            wall_radius + self.APERTURE_ROD_OVERHANG_MM,
        )

        axial_envelope = max(
            float(record.end_z_mm - record.start_z_mm), 0.001
        )
        rod_width = max(1.0, min(3.0, 0.18 * axial_envelope))
        screw_width = max(1.2, 1.25 * rod_width)
        screw_height = max(1.5, min(3.5, 1.35 * rod_width))
        source = (
            record.aperture_mechanism_evidence_source
            or "Aperture topology source is not declared."
        )
        common = (
            f"{record.name} [{status}]\n"
            f"Optical aperture plane Z = {plate_z:.6g} mm\n"
            f"Evidence: {source}"
        )

        envelope = QGraphicsRectItem(
            record.start_z_mm,
            -envelope_radius,
            axial_envelope,
            2.0 * envelope_radius,
        )
        envelope.setPen(pg.mkPen(
            "#64748b", width=0.8, style=Qt.PenStyle.DotLine
        ))
        envelope.setBrush(pg.mkBrush(0, 0, 0, 0))
        envelope.setToolTip(
            common
            + "\nDashed unfilled outline: TOML mechanism/cartridge envelope."
            " It is not a solid aperture plate."
        )
        envelope.setZValue(19)
        self.plot.addItem(envelope)
        self._remember_aperture_item(record.key, "envelope", envelope)

        plate_tooltip = (
            common
            + "\nPt perforated aperture strip (material user-identified; "
            "not independently verified)."
            f"\nConfigured TOML plate thickness {plate_thickness:.6g} mm | "
            "current "
            f"hard-edge opening {record.bore_diameter_mm:.6g} mm | carrier "
            f"bore {record.mechanical_bore_diameter_mm:.6g} mm."
            "\nThe transverse strip extent is schematic because the photo "
            "has no calibrated OEM scale. The photograph's additional holes "
            "are not assigned invented diameters or spacing; the displayed "
            "opening remains the simulator's continuous operating value. "
            "Only this thin plane clips rays."
        )
        plate_colour = pg.mkColor(self.APERTURE_PLATE_COLOUR)
        plate_intervals = (
            (
                plate_center_y - negative_reach,
                plate_center_y - opening_radius,
            ),
            (
                plate_center_y + opening_radius,
                joint_y,
            ),
        )
        for lower_y, upper_y in plate_intervals:
            if upper_y <= lower_y:
                continue
            plate = QGraphicsRectItem(
                plate_z - 0.5 * plate_thickness,
                lower_y,
                plate_thickness,
                upper_y - lower_y,
            )
            plate.setPen(pg.mkPen(
                self.APERTURE_PLATE_COLOUR,
                width=1.25,
                style=pen_style,
            ))
            plate.setBrush(pg.mkBrush(
                plate_colour.red(),
                plate_colour.green(),
                plate_colour.blue(),
                plate_alpha,
            ))
            plate.setToolTip(plate_tooltip)
            plate.setZValue(31)
            self.plot.addItem(plate)
            self._remember_aperture_item(record.key, "plate", plate)

        rod = QGraphicsRectItem(
            plate_z - 0.5 * rod_width,
            joint_y,
            rod_width,
            rod_end_y - joint_y,
        )
        rod.setPen(pg.mkPen(
            self.APERTURE_ROD_COLOUR,
            width=1.1,
            style=pen_style,
        ))
        rod_colour = pg.mkColor(self.APERTURE_ROD_COLOUR)
        rod.setBrush(pg.mkBrush(
            rod_colour.red(),
            rod_colour.green(),
            rod_colour.blue(),
            rod_alpha,
        ))
        rod.setToolTip(
            common
            + "\nSingle rear connecting/insertion rod; its material is "
            "unspecified."
            "\nThe rod is drawn on the positive mechanical-radius side and "
            f"ends {rod_end_y - wall_radius:.6g} mm beyond the local shown "
            f"column wall (radius {wall_radius:.6g} mm), resolved from "
            f"{wall_source.name} at an axial separation of "
            f"{wall_distance:.6g} mm."
            "\nIts width and reach are a photo-informed schematic, not an "
            "OEM rod-length measurement, and do not participate in ray "
            "clipping."
        )
        rod.setZValue(28)
        self.plot.addItem(rod)
        self._remember_aperture_item(record.key, "rod", rod)

        screw = QGraphicsEllipseItem(
            plate_z - 0.5 * screw_width,
            joint_y - 0.5 * screw_height,
            screw_width,
            screw_height,
        )
        screw.setPen(pg.mkPen(
            self.APERTURE_SCREW_COLOUR,
            width=1.2,
            style=pen_style,
        ))
        screw.setBrush(pg.mkBrush(248, 250, 252, 205 if inserted else 70))
        screw.setToolTip(
            common
            + "\nScrew joint between the Pt perforated strip and the "
            "single connecting rod. Screw dimensions are schematic."
        )
        screw.setZValue(33)
        self.plot.addItem(screw)
        self._remember_aperture_item(record.key, "screw", screw)

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

    @staticmethod
    def _recording_signal_z(record) -> float:
        """Return the detector's single TOML signal-reference plane."""

        if len(record.optical_references_mm) != 1:
            raise ValueError(
                f"Recording device {record.key} requires one signal plane"
            )
        return float(record.optical_references_mm[0])

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
            anchor_z_mm=float(z_mm),
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
        signal_z = self._recording_signal_z(record)
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
            f"Active detection plane Z = {signal_z:.6g} mm\n"
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
                signal_z - 0.5 * plane_width,
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
                signal_z - 0.9,
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
            signal_z - 6.0,
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
            signal_z + 7.0,
            housing_end,
            colour,
            tooltip,
        )

    def _add_fluorescent_screen_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        inserted = bool(getattr(component, "inserted", True))
        signal_z = self._recording_signal_z(record)
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
            f"Active detection plane Z = {signal_z:.6g} mm\n"
            f"screen diameter {record.outer_diameter_mm:.6g} mm\n"
            "The hinge, arm and parked position are schematic and remain "
            "outside the axial mechanical model."
        )
        if inserted:
            points = QPolygonF([
                QPointF(signal_z - tilt - half_thickness, -screen_radius),
                QPointF(signal_z - tilt + half_thickness, -screen_radius),
                QPointF(signal_z + tilt + half_thickness, screen_radius),
                QPointF(signal_z + tilt - half_thickness, screen_radius),
            ])
            pivot_z = signal_z + tilt
            pivot_y = screen_radius
        else:
            parked_center_y = housing_start + 0.5 * screen_radius
            points = QPolygonF([
                QPointF(signal_z - half_thickness, parked_center_y - screen_radius),
                QPointF(signal_z + half_thickness, parked_center_y - screen_radius),
                QPointF(signal_z + half_thickness, parked_center_y + screen_radius),
                QPointF(signal_z - half_thickness, parked_center_y + screen_radius),
            ])
            pivot_z = signal_z
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
            signal_z - 6.0,
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
            signal_z + 7.0,
            max(housing_end, arm_end),
            colour,
            tooltip,
        )

    def _add_camera_schematic(self, record, colour) -> None:
        component = self._recording_plane_component(record.key)
        active = bool(getattr(component, "inserted", True))
        signal_z = self._recording_signal_z(record)
        sensor_half = 0.5 * float(record.outer_diameter_mm)
        body_half = max(sensor_half + 9.0, 38.0)
        body_length = max(40.0, 0.72 * float(record.outer_diameter_mm))
        body_start = signal_z - 4.0
        body_end = signal_z + body_length
        status = "ACTIVE" if active else "INACTIVE"
        rgb = pg.mkColor(colour)
        tooltip = (
            f"{record.name} / fixed on-axis camera ({status.lower()})\n"
            f"Active detection plane Z = {signal_z:.6g} mm\n"
            f"sensor width {record.outer_diameter_mm:.6g} mm\n"
            "The sensor plane participates in recording. The downstream "
            "camera body is a schematic external envelope only."
        )
        sensor = QGraphicsRectItem(
            signal_z - 0.7,
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
            signal_z + 2.0,
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
            return role in {
                "slit_blade_carrier",
                "branch_interface",
                "pole_piece_cartridge",
                "detector_chamber_housing",
                "fixed_vacuum_restriction",
            }
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
        # A lazily activated page can receive its range/resize signal before
        # pyqtgraph's next paint. Measure labels in the current transform, not
        # the previous hidden viewport, or their first-frame rows overlap.
        view_box.updateMatrix()
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
        self._c1_c2_pole_piece_cartridge_items = []
        self._c1_c2_pole_gap = None
        self._objective_lens_half_items = []
        self._objective_lens_labels = []
        self._lens_excitation_coil_items = {}
        self._sample_stage_items = []
        self._sample_holder_items = []
        self._sample_plane_items = []
        self._sample_plane_labels = []
        self._eds_detector_items = {}
        self._eds_detector_labels = []
        self._detector_chamber_items = []
        self._accelerator_stack_items = {}
        self._aperture_mechanism_items = {}
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
            from temsim.magnetic_circuits import radial_profile_mm
            part = self._part_by_key.get(record.key)
            radial_profile = radial_profile_mm(part.data, part.length_mm) if part is not None else None
            if radial_profile is not None:
                self._add_magnetic_radial_profile(record, colour, radial_profile)
            elif (
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
            elif record.profile == self.EDS_DETECTOR_ARRAY_PROFILE:
                # The array is transverse and is drawn after the Objective
                # and specimen schematics so it is not mistaken for an axial
                # electron-intercepting annulus.
                pass
            elif record.profile == self.DETECTOR_CHAMBER_PROFILE:
                self._add_post_projector_detector_chamber_schematic(
                    record, colour
                )
            elif record.profile == self.ACCELERATOR_STACK_PROFILE:
                self._add_accelerator_stack_schematic(record)
            elif record.profile == self.APERTURE_MECHANISM_PROFILE:
                self._add_aperture_mechanism_schematic(record)
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
                        + (
                            f"\nAxial winding length "
                            f"{record.end_z_mm - record.start_z_mm:.6g} mm; "
                            f"radial winding thickness "
                            f"{0.5 * (record.outer_diameter_mm - record.bore_diameter_mm):.6g} mm."
                            "\nProvisional non-OEM geometry: axial length is "
                            "90% of the smaller lens envelope/column OD; "
                            "radial thickness uses an explicit per-lens "
                            "mechanical reconstruction when declared, with "
                            "the model design-peak-field rule as fallback."
                            if record.profile == "magnetic_excitation_coil"
                            else ""
                        )
                        + (
                            "\nAlways-inserted differential-pumping stop."
                            "\nSize/position adjustable; clips rays without "
                            "automatic preset recalculation."
                            if record.profile == self.FIXED_DPA_PROFILE
                            else ""
                        )
                    )
                    self.plot.addItem(rect)
                    if record.profile == "c1_c2_pole_piece_cartridge":
                        self._c1_c2_pole_piece_cartridge_items.append(rect)
                    if record.profile == "magnetic_excitation_coil":
                        self._lens_excitation_coil_items.setdefault(
                            record.key, []
                        ).append(rect)
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
            marker_z = (
                self._recording_signal_z(record)
                if record.profile in self.RECORDING_SURFACE_PROFILES
                else float(record.center_z_mm)
            )
            if record.profile != self.EDS_DETECTOR_ARRAY_PROFILE:
                spots.append({
                    "pos": (marker_z, 0.0),
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
                    f"{segment.name}\nNon-magnetic vacuum liner tube\n"
                    f"ID {segment.inner_diameter_mm:.6g} mm | "
                    f"OD {segment.outer_diameter_mm:.6g} mm"
                    "\nThis is a physical sleeve around the open beam-path "
                    "bore, not the bore itself."
                    + (
                        "\nPhoto-scaled continuous tube; it terminates at "
                        "the upper Objective pole connector tail. The metal "
                        "vacuum seal is user-identified as iridium."
                        if segment.key
                        == "@vacuum_liner:c1_c2_to_upper_objective"
                        else ""
                    )
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
        self._add_eds_detector_array_schematic()
        self._add_c1_c2_pole_gap_reference()
        self._add_component_labels()

        centres = pg.ScatterPlotItem(spots=spots, pxMode=True)
        centres.setZValue(40)
        centres.setToolTip(
            "Click a component marker; detector markers use their active "
            "detection plane"
        )
        centres.sigClicked.connect(self._centre_clicked)
        self.plot.addItem(centres)
        self.plot.autoRange()
        self._layout_component_labels()
        self.heading.setText(
            f"Resolved mechanical layout — {len(self._records)} components | "
            f"vacuum ID {minimum_diameter:.6g}–{maximum_diameter:.6g} mm"
        )

        lens_housing_diameters = {
            float(record.outer_diameter_mm)
            for record in self._records
            if record.profile == "magnetic_lens_housing"
            and not str(
                self._part_by_key[
                    str(self._part_by_key[record.key].parent_key)
                ].data.get("nested_lens_parent_key", "")
            )
        }
        column_od_text = (
            f"{next(iter(lens_housing_diameters)):.6g} mm"
            if len(lens_housing_diameters) == 1
            else "nonuniform"
        )
        layout_detail = (
            "External magnetic-lens housings use the provisional "
            f"{column_od_text} column OD; the Mini Condenser is an explicit "
            "radially nested internal-lens exception. Objective pole pieces "
            "use a thin straight mounting shank whose outer boundary "
            "coincides with the Objective coil bore, followed by a wider "
            "shoulder and tapered nose; condenser pole pieces form an embedded "
            "hourglass around the beam-path bore. The white bore boundary is "
            "open vacuum space; the thin grey sleeve is the separate "
            "non-magnetic vacuum liner tube. Objective mechanics are split "
            "into upper/lower lenses at the pole gap. The transverse stage "
            "and nested sample holder are "
            "schematic; the holder tip marks the current sample plane. "
            "Apertures separate the silver Pt perforated strip, white screw "
            "joint and amber rear connecting rod. Plate thickness and the "
            "optical plane are drawn at their TOML values; the transverse "
            "strip extent and screw size are photo-informed schematics. Each "
            "rod now extends 5 mm beyond the locally resolved shown column "
            "wall; this reach remains schematic because "
            "the supplied top view has no calibrated OEM scale. Other holes "
            "seen on the strip are not given invented spacing or diameters; "
            "the current continuous opening is shown. Retracted "
            "plates are parked on the positive-radius side and do not cross "
            "the beam. "
            "The accelerator is drawn as its configured sequence of annular "
            "electrostatic electrode stages, matching the repeated-ring "
            "appearance in the supplied side view. These are not magnetic "
            "coils; individual ring thickness, outer collars and material "
            "remain schematic because the photograph has no OEM scale. "
            "EDS is shown as a transverse six-segment X-ray array aimed "
            "at the sample. Its >=4.45 sr / 4.04 sr angular acceptance is "
            "physical metadata; the two projected detector heads are "
            "schematic because active area, sensor distance and package "
            "dimensions are not public. Their solid polygons are placed with "
            "a 1 mm display-only clearance outside the resolved Objective "
            "pole-piece OD; this removes a false 2D overlap but does not claim "
            "an OEM installation clearance or validate 3D shadowing. "
            "The constraint-derived 8.0 mm tip / 27.5584 mm taper clears only "
            "the 32.06 deg reference centre line by about 0.3109 mm; the "
            "overlapping equivalent-cone boundaries do not prove full "
            "4.45 sr mechanical clearance. It is not an axial electron stop. "
            "The translucent post-P2 enclosure is "
            "a non-OEM Titan-topology viewing/STEM-detector chamber. It makes "
            "the HAADF-first detector section explicit without moving any "
            "active plane; its absolute dimensions and the 7.25 mm P2-to-HAADF "
            "gap remain provisional. "
            "Recording-device actuators and housings are schematic while "
            "their thin active planes retain the calculated coordinates. "
            "Names are packed into multiple screen-space rows; dashed leaders "
            "connect each visible name to its physical component and relayout "
            "automatically while zooming."
        )
        self.summary.setText(
            f"TOML mechanical layout | {len(self._records)} components | "
            f"column OD {column_od_text} | select or hover for details"
        )
        self.summary.setToolTip(layout_detail)
        self._semantic_selected_summary = None

    def set_parameter_semantics_context(self, mode=None, descriptors=None, by_key=None):
        """Update selected-parameter explanations without changing the plot."""
        self._parameter_semantics_mode = mode
        self._parameter_semantics_descriptors = descriptors
        self._parameter_semantics_by_key = dict(by_key or {})
        self._refresh_selected_parameter_semantics()

    def _set_selected_summary(self, part, text, tooltip):
        self._semantic_selected_summary = (part, text, tooltip)
        self._refresh_selected_parameter_semantics()

    def _refresh_selected_parameter_semantics(self):
        if self._semantic_selected_summary is None:
            return
        from numbers import Real
        from temsim.parameter_semantics import describe_parameter
        from temsim.parameter_impact import describe_parameter_impact

        part, text, tooltip = self._semantic_selected_summary
        key = getattr(part, "key", "")
        data = self._parameter_semantics_by_key.get(key, getattr(part, "data", {}))
        if hasattr(data, "data"):
            data = data.data
        data = {**data, "key": key}
        mode = getattr(self._parameter_semantics_mode, "value", self._parameter_semantics_mode)
        compact, details = [], []
        groups = (("length_mm",), ("mechanical_outer_diameter_mm", "outer_diameter_mm"),
                  ("mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm", "bore_diameter_mm"),
                  ("plate_thickness_mm", "active_length_mm"))
        for group in groups:
            field = next((name for name in group if isinstance(data.get(name), Real)
                          and not isinstance(data[name], bool)), None)
            if field is None:
                continue
            path = ("parts", key, field)
            meaning = describe_parameter(data, path, by_key=self._parameter_semantics_by_key)
            impact = describe_parameter_impact(data, path, by_key=self._parameter_semantics_by_key,
                simulation_mode=self._parameter_semantics_mode, descriptors=self._parameter_semantics_descriptors)
            compact.append(f"{meaning.label}: {meaning.category_label}")
            details.append(f"{meaning.label}: {data[field]:g} {meaning.unit}\n"
                           f"Category: {meaning.category_label}\n"
                           f"Source: {meaning.source_label}. {meaning.source_note}\n{meaning.description}\n"
                           f"Simulation mode: {mode if mode is not None else 'not connected'}. "
                           f"Impact: {impact.label}. {impact.detail}")
        self.summary.setText(text + ("\n" + " · ".join(compact) if compact else ""))
        self.summary.setToolTip(tooltip + ("\n\n" + "\n\n".join(details) if details else ""))

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
        if record.profile == self.EDS_DETECTOR_ARRAY_PROFILE:
            geometry = EDSDetectorArrayGeometry.from_part_data(part.data)
            cone_half_angle = (
                geometry.equivalent_circular_cone_half_angle_deg()
            )
            pole_assessment = self._eds_pole_centerline_assessment(geometry)
            clearance_summary = (
                ""
                if pole_assessment is None
                else " | provisional pole-profile centre-ray clearance "
                f"{pole_assessment.minimum_radial_clearance_mm:.4g} mm"
            )
            detail_text = (
                f"Selected: {record.name} | sample-plane aggregate at "
                f"{record.center_z_mm:.6g} mm | {geometry.segment_count} "
                "windowless SDD segments | reference take-off angle "
                f"{geometry.takeoff_angle_deg:.4g} deg (single dataset) | "
                f"solid angle >= "
                f"{geometry.minimum_unshadowed_solid_angle_sr:.4g} sr "
                "unshadowed and "
                f"{geometry.analytical_holder_solid_angle_sr:.4g} sr with "
                "the analytical holder | equal-segment equivalent-cone "
                f"half-angle {cone_half_angle:.3g} deg (overlapping angular "
                "summary, not a literal segment aperture)"
                + clearance_summary
                + ". Active area, "
                "sensor distance and mechanical envelope are not public; "
                "the displayed heads are non-dimensional schematics."
            )
            self._set_selected_summary(part,
                f"Selected: {record.name} | Z {record.center_z_mm:.6g} mm | "
                f"{geometry.segment_count} segments | "
                f"{geometry.analytical_holder_solid_angle_sr:.4g} sr with holder", detail_text,
            )
            return
        detail_text = (
            f"Selected: {record.name} | centre {record.center_z_mm:.6g} mm | "
            f"OD {record.outer_diameter_mm:.6g} mm | hardware bore "
            f"{record.mechanical_bore_diameter_mm:.6g} mm | "
            f"vacuum ID {record.vacuum_inner_diameter_mm:.6g} mm | "
            f"optical references: {', '.join(f'{value:.6g}' for value in record.optical_references_mm) or 'none'} mm"
            + (
                " | Pt perforated strip thickness: "
                f"{record.active_length_mm:.6g} mm | single screw-connected "
                f"rear rod: schematic, non-optical | current opening: "
                f"{record.bore_diameter_mm:.6g} mm | state: "
                f"{'inserted' if record.excitation_enabled is not False else 'retracted'}"
                if record.profile == self.APERTURE_MECHANISM_PROFILE
                else ""
            )
            + (
                " | electrostatic stages: "
                f"{len(record.accelerator_stage_centers_mm)} at configured "
                "TOML centers | repeated-ring thickness/separators: "
                "photo-informed schematic, not magnetic coils"
                if record.profile == self.ACCELERATOR_STACK_PROFILE
                else ""
            )
            + (
                " | active detection plane: "
                f"{self._recording_signal_z(record):.6g} mm"
                if record.profile in self.RECORDING_SURFACE_PROFILES
                else ""
            )
        )
        extra = ""
        if record.profile == self.APERTURE_MECHANISM_PROFILE:
            extra = (
                f" | opening {record.bore_diameter_mm:.6g} mm | "
                f"{'inserted' if record.excitation_enabled is not False else 'retracted'}"
            )
        elif record.profile == self.ACCELERATOR_STACK_PROFILE:
            extra = f" | {len(record.accelerator_stage_centers_mm)} stages"
        elif record.profile in self.RECORDING_SURFACE_PROFILES:
            extra = f" | active Z {self._recording_signal_z(record):.6g} mm"
        self._set_selected_summary(part,
            f"Selected: {record.name} | Z {record.center_z_mm:.6g} mm | "
            f"OD {record.outer_diameter_mm:.6g} mm | "
            f"bore {record.mechanical_bore_diameter_mm:.6g} mm"
            + extra, detail_text,
        )

    def reveal_component(self, part) -> bool:
        """Highlight and centre a component without enabling live auto-range."""

        self.focus_component(part)
        record = self._record_by_key.get(getattr(part, "key", ""))
        if record is None:
            return False
        span = max(
            abs(float(record.end_z_mm) - float(record.start_z_mm)),
            1.0,
        )
        half_window = max(45.0, min(260.0, 1.6 * span))
        radial_half_window = max(
            0.65 * float(record.outer_diameter_mm),
            0.65 * float(record.mechanical_bore_diameter_mm),
            5.0,
        )
        centre = float(record.center_z_mm)
        self.plot.disableAutoRange()
        self.plot.setXRange(
            centre - half_window,
            centre + half_window,
            padding=0.0,
        )
        self.plot.setYRange(
            -radial_half_window,
            radial_half_window,
            padding=0.0,
        )
        self._layout_component_labels()
        return True


class InitialDirectionColourWheel(QWidget):
    """DPC-style cyclic legend for initial transverse ray direction."""

    NEUTRAL_COLOUR = QColor("#94a3b8")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._projection_angle_deg = 0.0
        self.setObjectName("initialDirectionColourWheel")
        self.setFixedSize(184, 184)
        self.setAccessibleName("Initial ray direction colour wheel")
        self._update_description()

    def _update_description(self) -> None:
        primary_name = projection_axis_name(self._projection_angle_deg)
        orthogonal_name = orthogonal_axis_name(self._projection_angle_deg)
        if primary_name.startswith("U("):
            primary_angle = self._projection_angle_deg % 360.0
            orthogonal_angle = (primary_angle + 90.0) % 360.0
            basis_description = (
                f"+U is {format_projection_angle(primary_angle)} degrees "
                "counter-clockwise from physical +X and +V is "
                f"{format_projection_angle(orthogonal_angle)} degrees."
            )
        else:
            basis_description = (
                f"+U is physical {primary_name} and +V is physical "
                f"{orthogonal_name}."
            )
        description = (
            "Continuous colour = initial polar angle about the starting "
            "bundle centroid. +X is 0 degrees and the angle increases "
            "counter-clockwise toward +Y. Colour tracks direction only; "
            "it does not represent ray radius, energy, intensity or "
            "survival state. The displayed basis follows Ray Diagram: "
            f"{basis_description} At a selected detector, the greyscale "
            "underlay is the peak-normalized forward PSF response; "
            "coloured dots remain the original rays."
        )
        self.setAccessibleDescription(description)
        self.setToolTip(description)

    def set_projection_angle(self, angle_deg: float) -> None:
        """Rotate the displayed U/V basis without changing physical hues."""

        angle = float(np.clip(angle_deg, 0.0, 360.0))
        if np.isclose(angle, self._projection_angle_deg, atol=1.0e-12):
            return
        self._projection_angle_deg = angle
        self._update_description()
        self.update()

    @staticmethod
    def _reference_label(angle_deg: float) -> str:
        name = projection_axis_name(angle_deg)
        if name.startswith("U("):
            rounded = int(np.floor((angle_deg % 360.0) + 0.5)) % 360
            return f"{rounded}°"
        return name

    @staticmethod
    def colour_for_angle(angle_rad: float) -> QColor:
        """Return the cyclic display colour for one CCW angle from +X."""

        hue = float(np.mod(float(angle_rad), 2.0 * math.pi) / (2.0 * math.pi))
        return QColor.fromHsvF(hue, 0.88, 1.0)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        wheel_rect = QRectF(31.0, 31.0, 122.0, 122.0)
        painter.setPen(Qt.PenStyle.NoPen)
        for degree in range(360):
            physical_degree = degree + self._projection_angle_deg
            painter.setBrush(
                self.colour_for_angle(math.radians(physical_degree))
            )
            # The one-unit overlap avoids hairline gaps after rasterisation.
            painter.drawPie(wheel_rect, degree * 16, 17)

        inner_rect = wheel_rect.adjusted(32.0, 32.0, -32.0, -32.0)
        painter.setBrush(QColor("#050816"))
        painter.drawEllipse(inner_rect)
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(
            inner_rect,
            Qt.AlignmentFlag.AlignCenter,
            "initial\nangle",
        )

        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(
            QRectF(151.0, 70.0, 33.0, 42.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "+U\n" + self._reference_label(self._projection_angle_deg),
        )
        painter.drawText(
            QRectF(66.0, 0.0, 52.0, 31.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "+V\n" + self._reference_label(
                self._projection_angle_deg + 90.0
            ),
        )
        painter.drawText(
            QRectF(0.0, 70.0, 31.0, 42.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            "-U\n" + self._reference_label(
                self._projection_angle_deg + 180.0
            ),
        )
        painter.drawText(
            QRectF(55.0, 153.0, 74.0, 31.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "-V\n" + self._reference_label(
                self._projection_angle_deg + 270.0
            ),
        )
        painter.end()


class TransverseBeamView(QWidget):
    """X-Y beam slice that makes round-lens image rotation observable."""

    MAX_DISPLAY_RAYS = 2_000
    CENTRE_DIRECTION_TOLERANCE_M = 1.0e-15
    DEFAULT_HALF_RANGE_DISPLAY = 1.0
    FIT_PADDING_FACTOR = 1.08
    MIN_HALF_RANGE_DISPLAY = 1.0e-9
    DISPLAY_UNIT = "µm"
    METRES_TO_DISPLAY = 1.0e6
    MILLIMETRES_TO_DISPLAY = 1.0e3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result = None
        self._plane_z_mm = None
        self._scatter = None
        self._focused_component_key = None
        self._point_spread_image = None
        self._point_spread_response = None
        self._fit_coordinates = None
        self._projection_angle_deg = 0.0
        self._view_scale_initialized = False
        self._view_change_guard = False
        self._view_ranges = (
            (-self.DEFAULT_HALF_RANGE_DISPLAY, self.DEFAULT_HALF_RANGE_DISPLAY),
            (-self.DEFAULT_HALF_RANGE_DISPLAY, self.DEFAULT_HALF_RANGE_DISPLAY),
        )

        self.heading = QLabel("Transverse beam X-Y")
        self.summary = QLabel(
            "Select a component to inspect the beam at its centre plane."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.fit_beam = QPushButton("Fit beam")
        self.fit_beam.setStyleSheet(BUTTON_STYLE)
        self.fit_beam.setToolTip(
            "Fit all surviving rays while keeping beam (0, 0) at the plot "
            "centre. This is the only automatic scale reset."
        )

        heading_row = QHBoxLayout()
        heading_row.addWidget(self.heading)
        heading_row.addStretch(1)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("transverseBeamPlot")
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
        self._update_projection_labels()
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setAspectLocked(True, ratio=1.0)
        self.plot.setMenuEnabled(False)
        self.plot.setMinimumHeight(250)
        self.plot.setMaximumHeight(360)
        self.plot.getViewBox().disableAutoRange()
        self._apply_centered_view_ranges(
            self.DEFAULT_HALF_RANGE_DISPLAY,
            self.DEFAULT_HALF_RANGE_DISPLAY,
        )
        self.plot.getViewBox().sigRangeChangedManually.connect(
            self._manual_view_range_changed
        )

        self.angle_colour_wheel = InitialDirectionColourWheel()
        self.initial_beam_heading = QLabel("Initial beam direction")
        self.initial_beam_heading.setStyleSheet(
            "color: #e2e8f0; font-weight: 700;"
        )
        self.initial_beam_panel = QWidget()
        self.initial_beam_panel.setObjectName("transverseInitialBeamPanel")
        self.initial_beam_panel.setMaximumWidth(520)
        self.initial_beam_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        initial_beam_layout = QVBoxLayout(self.initial_beam_panel)
        initial_beam_layout.setContentsMargins(4, 4, 4, 4)
        initial_beam_layout.addWidget(self.initial_beam_heading)
        initial_beam_layout.addWidget(
            self.angle_colour_wheel,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )

        self.section_beam_panel = QWidget()
        self.section_beam_panel.setObjectName("transverseSectionBeamPanel")
        self.section_beam_panel.setMaximumWidth(520)
        self.section_beam_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        section_beam_layout = QVBoxLayout(self.section_beam_panel)
        section_beam_layout.setContentsMargins(4, 4, 4, 4)
        heading_row.addWidget(self.fit_beam)
        section_beam_layout.addLayout(heading_row)
        section_beam_layout.addWidget(self.plot, 1)
        section_beam_layout.addWidget(self.summary)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            self.initial_beam_panel,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(
            self.section_beam_panel,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addStretch(1)
        self.fit_beam.clicked.connect(self._fit_beam_view)

    def _update_projection_labels(self) -> None:
        primary_name = projection_axis_name(self._projection_angle_deg)
        orthogonal_name = orthogonal_axis_name(self._projection_angle_deg)
        self.plot.getAxis("bottom").setLabel(
            f"{primary_name} displacement",
            units=self.DISPLAY_UNIT,
        )
        self.plot.getAxis("left").setLabel(
            f"{orthogonal_name} displacement",
            units=self.DISPLAY_UNIT,
        )
        self.plot.setToolTip(
            "Display coordinates follow Ray Diagram: "
            "U = X cos(angle) + Y sin(angle), "
            "V = -X sin(angle) + Y cos(angle)."
        )

    def set_projection_angle(self, angle_deg: float, *, redraw: bool = True) -> None:
        """Match the Ray Diagram basis; hidden owners can defer presentation."""

        angle = float(np.clip(angle_deg, 0.0, 360.0))
        changed = not np.isclose(
            angle, self._projection_angle_deg, atol=1.0e-12
        )
        self._projection_angle_deg = angle
        self.angle_colour_wheel.set_projection_angle(angle)
        self._update_projection_labels()
        if redraw and changed and self._result is not None:
            self._redraw()

    def _apply_centered_view_ranges(
        self,
        x_half_range: float,
        y_half_range: float,
    ) -> None:
        """Apply a finite origin-centred X-Y view in display micrometres."""

        x_half = max(float(x_half_range), self.MIN_HALF_RANGE_DISPLAY)
        y_half = max(float(y_half_range), self.MIN_HALF_RANGE_DISPLAY)
        if not (math.isfinite(x_half) and math.isfinite(y_half)):
            return
        self._view_change_guard = True
        try:
            self.plot.getViewBox().setRange(
                xRange=(-x_half, x_half),
                yRange=(-y_half, y_half),
                padding=0.0,
                disableAutoRange=True,
            )
        finally:
            self._view_change_guard = False
        ranges = self.plot.getViewBox().viewRange()
        self._view_ranges = (
            (float(ranges[0][0]), float(ranges[0][1])),
            (float(ranges[1][0]), float(ranges[1][1])),
        )

    def _restore_centered_view_ranges(self) -> None:
        x_range, y_range = self._view_ranges
        self._apply_centered_view_ranges(
            0.5 * (x_range[1] - x_range[0]),
            0.5 * (y_range[1] - y_range[0]),
        )

    def _manual_view_range_changed(self, _axis_mask) -> None:
        """Keep a user-selected scale but reject transverse panning."""

        if self._view_change_guard:
            return
        x_range, y_range = self.plot.getViewBox().viewRange()
        self._view_scale_initialized = True
        self._apply_centered_view_ranges(
            0.5 * float(x_range[1] - x_range[0]),
            0.5 * float(y_range[1] - y_range[0]),
        )

    def _fit_beam_view(self) -> None:
        """Fit current surviving rays symmetrically about physical zero."""

        if self._fit_coordinates is None:
            return
        x_values, y_values = self._fit_coordinates
        finite_x = np.asarray(x_values, dtype=float)
        finite_y = np.asarray(y_values, dtype=float)
        finite_x = finite_x[np.isfinite(finite_x)]
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_x.size == 0 or finite_y.size == 0:
            return
        x_half = max(
            float(np.max(np.abs(finite_x))) * self.FIT_PADDING_FACTOR,
            self.MIN_HALF_RANGE_DISPLAY,
        )
        y_half = max(
            float(np.max(np.abs(finite_y))) * self.FIT_PADDING_FACTOR,
            self.MIN_HALF_RANGE_DISPLAY,
        )
        if x_half <= self.MIN_HALF_RANGE_DISPLAY:
            x_half = self.DEFAULT_HALF_RANGE_DISPLAY
        if y_half <= self.MIN_HALF_RANGE_DISPLAY:
            y_half = self.DEFAULT_HALF_RANGE_DISPLAY
        self._view_scale_initialized = True
        self._apply_centered_view_ranges(x_half, y_half)

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

    def display_result(self, result, *, focus=None) -> None:
        self._result = result
        if focus is not None:
            kind, value = focus
            if kind == "component" and value is not None:
                self.focus_component(value, redraw=False)
            elif kind == "component":
                self._focused_component_key = None
            else:
                self.focus_z(value, redraw=False)
        if self._plane_z_mm is None:
            self._plane_z_mm = float(result.simulation.incident.z[-1])
        self._redraw()

    def focus_component(self, part, *, redraw: bool = True) -> None:
        key = str(part.key)
        changed_focus = self._focused_component_key != key
        self._focused_component_key = key
        plane_z = float(part.center_z_mm)
        state = getattr(self._result, "state_snapshot", None)
        detector = next(
            (
                item for item in getattr(state, "recording_planes", ())
                if str(item.key) == key
            ),
            None,
        )
        if detector is not None:
            plane_z = float(detector.z_mm)
        self._set_plane_z(plane_z, redraw=redraw, force_redraw=changed_focus)

    def focus_z(self, z_mm: float, *, redraw: bool = True) -> None:
        """Display an arbitrary finite axial plane without retracing rays."""

        changed_focus = self._focused_component_key is not None
        self._focused_component_key = None
        self._set_plane_z(z_mm, redraw=redraw, force_redraw=changed_focus)

    def _set_plane_z(self, z_mm: float, *, redraw: bool = True, force_redraw=False) -> None:
        if not math.isfinite(float(z_mm)):
            return
        changed = self._plane_z_mm != float(z_mm)
        self._plane_z_mm = float(z_mm)
        if redraw and (changed or force_redraw) and self._result is not None:
            self._redraw()

    def _add_point_spread_response(self):
        """Overlay detector response only for a selected recording device."""

        state = getattr(self._result, "state_snapshot", None)
        if state is None or self._focused_component_key is None:
            return None
        plane_keys = {
            str(item.key) for item in getattr(state, "recording_planes", ())
        }
        if self._focused_component_key not in plane_keys:
            return None
        response = detector_response_image(
            self._result.simulation,
            state,
            self._focused_component_key,
        )
        self._point_spread_response = response
        if not np.any(response.intensity > 0.0):
            return response
        x0, x1, y0, y1 = response.extent
        extent_scale = self.MILLIMETRES_TO_DISPLAY
        image = pg.ImageItem()
        image.setImage(
            response.intensity.T,
            autoLevels=False,
            levels=(0.0, 1.0),
        )
        image.setRect(QRectF(
            x0 * extent_scale,
            y0 * extent_scale,
            (x1 - x0) * extent_scale,
            (y1 - y0) * extent_scale,
        ))
        angle_rad = math.radians(self._projection_angle_deg)
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        physical_to_display = QTransform(
            cosine,
            -sine,
            sine,
            cosine,
            0.0,
            0.0,
        )
        image.setTransform(image.transform() * physical_to_display)
        image.setOpacity(0.62)
        image.setZValue(-20.0)
        image.setToolTip(
            "Peak-normalized forward detector PSF response; original rays "
            "remain as the coloured point overlay."
        )
        self.plot.addItem(image)
        self._point_spread_image = image
        return response

    def _redraw(self) -> None:
        self.plot.clear()
        self._scatter = None
        self._point_spread_image = None
        self._point_spread_response = None
        self._fit_coordinates = None
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
            self._restore_centered_view_ranges()
            return

        point_spread_response = self._add_point_spread_response()

        start_x = np.asarray(branch.x[0], dtype=float)
        start_y = np.asarray(branch.y[0], dtype=float)
        branch_start = float(z_values[0])
        reference_mask = np.isnan(blocked) | (
            blocked >= branch_start - 1.0e-9
        )
        reference_x = start_x[reference_mask]
        reference_y = start_y[reference_mask]
        centre_x = float(np.mean(reference_x)) if reference_x.size else 0.0
        centre_y = float(np.mean(reference_y)) if reference_y.size else 0.0
        relative_x = start_x - centre_x
        relative_y = start_y - centre_y
        initial_angle = np.mod(
            np.arctan2(relative_y, relative_x), 2.0 * math.pi
        )
        initial_radius = np.hypot(relative_x, relative_y)
        brushes = []
        for index in indices:
            colour = (
                InitialDirectionColourWheel.NEUTRAL_COLOUR
                if initial_radius[index] <= self.CENTRE_DIRECTION_TOLERANCE_M
                else InitialDirectionColourWheel.colour_for_angle(
                    initial_angle[index]
                )
            )
            brushes.append(pg.mkBrush(colour))
        display_x_m, display_y_m = transverse_view_coordinates(
            x_m[indices],
            y_m[indices],
            self._projection_angle_deg,
        )
        display_x = display_x_m * self.METRES_TO_DISPLAY
        display_y = display_y_m * self.METRES_TO_DISPLAY
        self._fit_coordinates = (display_x, display_y)
        self._scatter = pg.ScatterPlotItem(
            x=display_x,
            y=display_y,
            size=5,
            pen=pg.mkPen(None),
            brush=brushes,
            pxMode=True,
        )
        self.plot.addItem(self._scatter)
        self.plot.addLine(x=0.0, pen=pg.mkPen("#94a3b8", width=0.8))
        self.plot.addLine(y=0.0, pen=pg.mkPen("#94a3b8", width=0.8))
        if self._view_scale_initialized:
            self._restore_centered_view_ranges()
        else:
            self._fit_beam_view()

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
        rms_radius_display = self.METRES_TO_DISPLAY * float(np.sqrt(np.mean(
            (x_m[indices] - np.mean(x_m[indices])) ** 2
            + (y_m[indices] - np.mean(y_m[indices])) ** 2
        )))
        rotation_text = (
            f"{relative_rotation:+.6g} deg"
            if math.isfinite(relative_rotation) else "unavailable"
        )
        point_spread_text = ""
        point_spread_compact = ""
        if point_spread_response is not None:
            retained_text = (
                f"{100.0 * point_spread_response.retained_fraction:.6g}%"
                if math.isfinite(point_spread_response.retained_fraction)
                else "unavailable (no accepted hits)"
            )
            point_spread = point_spread_response.point_spread
            point_spread_text = (
                f" | detector PSF {point_spread.model}, sigma(X,Y)="
                f"({point_spread.sigma_x_mm:.6g}, "
                f"{point_spread.sigma_y_mm:.6g}) mm, "
                f"rotation {point_spread.rotation_deg:.6g} deg, "
                f"finite-area retained weight {retained_text} "
                f"[{point_spread.status}]"
            )
            point_spread_compact = f" | PSF retained {retained_text}"
        primary_name = projection_axis_name(self._projection_angle_deg)
        orthogonal_name = orthogonal_axis_name(self._projection_angle_deg)
        self.heading.setText(
            f"Transverse beam {primary_name}/{orthogonal_name} at "
            f"Z = {plane:.6g} mm"
        )
        detail_text = (
            f"{branch.name} | {indices.size} surviving rays | "
            f"RMS radius {rms_radius_display:.6g} {self.DISPLAY_UNIT} | "
            f"orientation relative to bundle start {rotation_text} | "
            f"display basis U={primary_name}, V={orthogonal_name} | "
            "continuous colour identifies initial direction about the "
            "bundle centroid"
            + point_spread_text
        )
        self.summary.setText(
            f"{branch.name} | {indices.size} rays | "
            f"RMS {rms_radius_display:.6g} {self.DISPLAY_UNIT} | "
            f"rotation {rotation_text}"
            + point_spread_compact
        )
        self.summary.setToolTip(detail_text)


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

        heading_row = QHBoxLayout()
        heading_row.addWidget(self.heading)
        heading_row.addStretch(1)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target plane"))
        target_row.addWidget(self.target_plane, 1)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.capture_current)
        action_row.addWidget(self.clear_pair)

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
        layout.addLayout(heading_row)
        layout.addLayout(target_row)
        layout.addLayout(action_row)
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
        detail_text = (
            f"{record.name} | Z {record.z_mm:.6g} mm | {insertion} | "
            f"active conjugate map {active}. Straight-column paraxial "
            "Jacobian; spherical aberration, hexapole nonlinearity and the "
            "curved Energy Filter branch are outside this matrix."
        )
        self.summary.setText(
            f"{record.name} | Z {record.z_mm:.6g} mm | {insertion} | {active}"
        )
        self.summary.setToolTip(detail_text)
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
    field_map_import_requested = Signal(str, str, str, float, int)
    field_map_clear_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records = ()
        self._curves = {}
        self._total_curve = None
        self._formula_sample_by_key = {}
        self._legend_key = None
        self._rotation_by_key = {}
        self._has_field_scene = False
        self._selected_key = None
        self._support_item = None
        self._formula_samples = []
        self._rotation_items = []
        self._sample_field_items = []
        self._plane_records = ()
        self._state_snapshot = None
        self._presentation_pending = False

        self.heading = QLabel("Axial magnetic field Bz")
        self.summary = QLabel("Recalculate to evaluate lens fields.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.show_individual = QPushButton("Individual lenses")
        self.show_individual.setCheckable(True)
        self.show_individual.setChecked(True)
        self.show_individual.setStyleSheet(BUTTON_STYLE)
        self.show_rotation_labels = QPushButton("Rotation labels")
        self.show_rotation_labels.setCheckable(True)
        self.show_rotation_labels.setChecked(True)
        self.show_rotation_labels.setStyleSheet(BUTTON_STYLE)

        self.field_map_lens = QComboBox()
        self.field_map_lens.setObjectName("magneticFieldMapLens")
        self.field_map_lens.setMinimumContentsLength(18)
        self.field_map_provenance = QComboBox()
        self.field_map_provenance.setObjectName(
            "magneticFieldMapProvenance"
        )
        self.field_map_provenance.addItem("Select source", None)
        self.field_map_provenance.addItem("Measured", "measured")
        self.field_map_provenance.addItem("FEM", "fem")
        self.field_map_reference = QDoubleSpinBox()
        self.field_map_reference.setObjectName(
            "magneticFieldMapReferenceExcitation"
        )
        self.field_map_reference.setRange(0.0, 1_000_000.0)
        self.field_map_reference.setDecimals(6)
        self.field_map_reference.setSuffix(" %")
        self.field_map_reference.setSpecialValueText("Reference required")
        self.field_map_reference.setValue(0.0)
        self.field_map_polarity = QComboBox()
        self.field_map_polarity.setObjectName(
            "magneticFieldMapReferencePolarity"
        )
        self.field_map_polarity.addItem("Select polarity", None)
        self.field_map_polarity.addItem("+Z", 1)
        self.field_map_polarity.addItem("-Z", -1)
        self.field_map_import = QPushButton("Import SI map")
        self.field_map_import.setObjectName("magneticFieldMapImport")
        self.field_map_import.setToolTip(
            "Import NPZ/CSV with coordinates in metres and magnetic field in "
            "tesla. No unit inference or conversion is performed. NPZ may "
            "carry explicit registration metadata; tidy CSV coordinates must "
            "already use the global column frame."
        )
        self.field_map_clear = QPushButton("Clear map")
        self.field_map_clear.setObjectName("magneticFieldMapClear")
        self.field_map_status = QLabel(
            "Select a lens to inspect its geometry-bound field provider."
        )
        self.field_map_status.setObjectName("magneticFieldMapStatus")
        self.field_map_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.field_map_status.setStyleSheet("color: #94a3b8;")
        self.digital_twin_status = QLabel(
            "Digital-twin fit: no sourced axial Bz dataset attached."
        )
        self.digital_twin_status.setObjectName(
            "magneticFieldCalibrationStatus"
        )
        self.digital_twin_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.digital_twin_status.setStyleSheet("color: #64748b;")

        field_map_row = QHBoxLayout()
        field_map_row.addWidget(QLabel("Field map"))
        field_map_row.addWidget(self.field_map_lens, 1)
        field_map_row.addWidget(self.field_map_provenance)
        field_map_row.addWidget(self.field_map_reference)
        field_map_row.addWidget(self.field_map_polarity)
        field_map_row.addWidget(self.field_map_import)
        field_map_row.addWidget(self.field_map_clear)

        heading_row = QHBoxLayout()
        heading_row.addWidget(self.heading)
        heading_row.addStretch(1)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.show_rotation_labels)
        action_row.addWidget(self.show_individual)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("magneticFieldPlot")
        self.plot.setLabel("bottom", "Axial position", units="mm")
        self.plot.setLabel("left", "Bz", units="T")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.legend = self.plot.addLegend(offset=(10, 10))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(heading_row)
        layout.addLayout(action_row)
        layout.addLayout(field_map_row)
        layout.addWidget(self.field_map_status)
        layout.addWidget(self.digital_twin_status)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        self.show_individual.toggled.connect(self._apply_curve_styles)
        self.show_rotation_labels.toggled.connect(
            self._apply_rotation_marker_visibility
        )
        self.plot.scene().sigMouseClicked.connect(
            self._plot_position_clicked
        )
        self.field_map_lens.currentIndexChanged.connect(
            self._refresh_field_map_status
        )
        self.field_map_provenance.currentIndexChanged.connect(
            self._update_field_map_import_enabled
        )
        self.field_map_reference.valueChanged.connect(
            self._update_field_map_import_enabled
        )
        self.field_map_polarity.currentIndexChanged.connect(
            self._update_field_map_import_enabled
        )
        self.field_map_import.clicked.connect(self._choose_field_map)
        self.field_map_clear.clicked.connect(self._clear_field_map)
        self._update_field_map_import_enabled()

    def _update_field_map_import_enabled(self, *_args) -> None:
        self.field_map_import.setEnabled(bool(
            self.field_map_lens.currentData()
            and self.field_map_provenance.currentData() in {"measured", "fem"}
            and self.field_map_reference.value() > 0.0
            and self.field_map_polarity.currentData() in {-1, 1}
        ))

    def _choose_field_map(self) -> None:
        lens_key = self.field_map_lens.currentData()
        provenance = self.field_map_provenance.currentData()
        polarity = self.field_map_polarity.currentData()
        reference = float(self.field_map_reference.value())
        if (
            not lens_key
            or provenance not in {"measured", "fem"}
            or polarity not in {-1, 1}
            or reference <= 0.0
        ):
            self.set_field_map_operation_status(
                "Select source, actual reference excitation and polarity.",
                error=True,
            )
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import geometry-bound magnetic field map",
            "",
            "SI magnetic field maps (*.npz *.csv)",
        )
        if not path:
            return
        self.set_field_map_operation_status("Validating selected map...")
        self.field_map_import_requested.emit(
            str(lens_key),
            str(path),
            str(provenance),
            reference,
            int(polarity),
        )

    def _clear_field_map(self) -> None:
        lens_key = self.field_map_lens.currentData()
        if lens_key:
            self.field_map_clear_requested.emit(str(lens_key))

    def set_field_map_operation_status(
        self, text: str, *, error: bool = False
    ) -> None:
        self.field_map_status.setText(str(text))
        self.field_map_status.setStyleSheet(
            "color: #f87171;" if error else "color: #94a3b8;"
        )

    def _populate_field_map_lenses(self) -> None:
        selected = self.field_map_lens.currentData()
        rows = tuple((record.key, record.name) for record in self._records)
        if tuple(
            (self.field_map_lens.itemData(index),
             self.field_map_lens.itemText(index))
            for index in range(self.field_map_lens.count())
        ) != rows:
            self.field_map_lens.blockSignals(True)
            self.field_map_lens.clear()
            for key, name in rows:
                self.field_map_lens.addItem(str(name), str(key))
            index = self.field_map_lens.findData(selected)
            self.field_map_lens.setCurrentIndex(max(index, 0))
            self.field_map_lens.blockSignals(False)
        self._refresh_field_map_status()
        self._update_field_map_import_enabled()

    def _refresh_field_map_status(self, *_args) -> None:
        state = self._state_snapshot
        key = self.field_map_lens.currentData()
        if state is None or not key:
            self.field_map_clear.setEnabled(False)
            self.set_field_map_operation_status(
                "Select a lens to inspect its geometry-bound field provider."
            )
            return
        descriptor = dict(
            getattr(state, "lens_field_map_descriptors", {}).get(
                str(key), {}
            )
            or {}
        )
        diagnostic = dict(
            getattr(state, "_field_provider_diagnostics", {}).get(
                str(key), {}
            )
            or {}
        )
        self.field_map_clear.setEnabled(bool(descriptor))
        mode = str(diagnostic.get("mode", ""))
        if mode == "imported_field_map":
            source = Path(str(descriptor.get("source_path", ""))).name
            kind = str(descriptor.get("provenance_kind", "map")).upper()
            divergence = (
                "divergence check passed"
                if bool(diagnostic.get("divergence_within_tolerance", False))
                else "divergence warning"
            )
            self.set_field_map_operation_status(
                f"{kind} map matched | {source} | {divergence}"
            )
        elif descriptor:
            reason = str(
                diagnostic.get("reason", "saved_map_unavailable_or_stale")
            )
            self.set_field_map_operation_status(
                f"Saved map rejected ({reason}) | provisional TOML Gaussian",
                error=True,
            )
        else:
            self.set_field_map_operation_status(
                "Provisional TOML Gaussian | no matching measured/FEM map"
            )
        detail = "\n".join(
            str(value) for value in (
                descriptor.get("source_path", ""),
                descriptor.get("source_sha256", ""),
                diagnostic.get("geometry_fingerprint", ""),
                diagnostic.get("descriptor_error", ""),
            )
            if value
        )
        self.field_map_status.setToolTip(detail)
        source_note = str(descriptor.get("source_note", ""))
        if "axial Bz calibration" in source_note:
            self.digital_twin_status.setText(
                "Digital-twin fit: calibration provenance attached."
            )
            self.digital_twin_status.setToolTip(source_note)
        else:
            self.digital_twin_status.setText(
                "Digital-twin fit: not run; core API requires sourced axial "
                "Bz points."
            )
            self.digital_twin_status.setToolTip(
                "fit_axial_field_map_calibration returns scale, axial "
                "registration and residuals without modifying source data."
            )

    def link_axial_axis(self, source_plot) -> None:
        """Share the Ray Diagram's axial range and plotting boundaries."""

        self.plot.getAxis("left").setWidth(
            source_plot.getAxis("left").minimumWidth()
        )
        self.plot.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)
        self.plot.setXLink(source_plot)

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

    def mark_presentation_pending(self) -> None:
        """Do not report the old field's numbers as current while hidden."""
        self._presentation_pending = True

    def _clear_field_graphics(self) -> None:
        """Discard only owned graphics when there is no calculation snapshot."""
        items = [self._total_curve, self._support_item, *self._curves.values(),
                 *self._formula_samples, *self._rotation_items, *self._sample_field_items]
        for item in items:
            if item is not None:
                self.plot.removeItem(item)
        self.legend.clear()
        self._curves = {}
        self._total_curve = None
        self._support_item = None
        self._formula_samples = []
        self._formula_sample_by_key = {}
        self._legend_key = None
        self._rotation_items = []
        self._rotation_by_key = {}
        self._sample_field_items = []
        self._has_field_scene = False

    def _update_formula_legend(self) -> None:
        formula_records = {}
        for record in self._records:
            formula_records.setdefault(record.formula_key, record)
        for key in self._formula_sample_by_key.keys() - formula_records.keys():
            self.plot.removeItem(self._formula_sample_by_key.pop(key))
        for key, record in formula_records.items():
            sample = self._formula_sample_by_key.get(key)
            if sample is None:
                sample = pg.PlotDataItem([], [])
                self.plot.addItem(sample)
                self._formula_sample_by_key[key] = sample
            sample.setPen(pg.mkPen(record.formula_colour, width=2.4))
        legend_key = tuple((record.formula_key, record.formula_label)
                           for record in formula_records.values())
        if legend_key != self._legend_key:
            self.legend.clear()
            self.legend.addItem(self._total_curve, "Total solver Bz")
            for key, record in formula_records.items():
                self.legend.addItem(self._formula_sample_by_key[key], record.formula_label)
            self._legend_key = legend_key
        self._formula_samples = [self._formula_sample_by_key[key] for key in formula_records]
        self.legend.update()

    def display_result(self, result) -> None:
        self._presentation_pending = False
        state = getattr(result, "state_snapshot", None)
        self._state_snapshot = state
        if state is None:
            self._clear_field_graphics()
            self._records = ()
            self._plane_records = ()
            self.heading.setText("Axial magnetic field Bz")
            self.summary.setText("No calculation state snapshot is available.")
            self.summary.setToolTip("")
            self._populate_field_map_lenses()
            return
        view_box = self.plot.getViewBox()
        previous_y = tuple(view_box.viewRange()[1]) if self._has_field_scene else None
        view_box.disableAutoRange(axis=pg.ViewBox.YAxis)
        start, end = self._simulation_limits(result.simulation)
        z_mm = np.linspace(start, end, 3_000)
        # Evaluate the authoritative providers for every new snapshot. Reusing
        # Qt items must not turn partial field dependencies into a physics cache.
        total, self._records = lens_field_records(state, z_mm)
        self._populate_field_map_lenses()
        self._plane_records = image_plane_rotation_records(state)
        if self._total_curve is None:
            self._total_curve = self.plot.plot(pen=pg.mkPen("#f8fafc", width=2.6))
        self._total_curve.setData(z_mm, total)
        sample_z_mm = float(state.sample.z_mm)
        sample_field_t = float(sum(
            record.field_at_sample_t for record in self._records
        ))
        if not self._sample_field_items:
            sample_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen("#f97316", width=1.4, style=Qt.PenStyle.DashLine),
            )
            self.plot.addItem(sample_line)
            self._sample_field_items.append(sample_line)
        sample_line = self._sample_field_items[0]
        sample_line.setValue(sample_z_mm)
        sample_line.setToolTip(
            f"Specimen plane Z {sample_z_mm:.6g} mm\n"
            f"Total solver Bz {sample_field_t:+.6g} T\n"
            "The specimen-local electron model uses this field; X-rays remain "
            "undeflected."
        )
        record_keys = {record.key for record in self._records}
        for key in self._curves.keys() - record_keys:
            self.plot.removeItem(self._curves.pop(key))
        if self._selected_key not in record_keys:
            self._selected_key = None
            if self._support_item is not None:
                self.plot.removeItem(self._support_item)
                self._support_item = None
        for record in self._records:
            curve = self._curves.get(record.key)
            if curve is None:
                curve = self.plot.plot()
                self._curves[record.key] = curve
                try:
                    curve.setCurveClickable(True, width=8)
                    curve.sigClicked.connect(
                        lambda *_args, key=record.key: self.component_selected.emit(key)
                    )
                except AttributeError:
                    pass
            curve.setData(z_mm, record.field_t)
            curve.setToolTip(
                f"{record.name}\nPeak |Bz| {record.peak_t:.6g} T\n"
                f"Excitation {record.excitation_percent:.6g}%\n"
                f"Formula: {record.formula_label}\n"
                f"{record.formula_expression}\n"
                f"At specimen {record.field_at_sample_t:+.6g} T\n"
                f"Numerical support: {record.support_definition}\n"
                f"Model status: {record.field_model_status}\n"
                f"Geometry/material coupling: {record.geometry_material_coupling}\n"
                f"Signed field integral {record.signed_field_integral_t_m:.6g} T m\n"
                f"Field direction {'+Z' if record.polarity > 0 else '-Z'}\n"
                f"Polarity status {record.field_polarity_status}\n"
                f"Polarity source {record.field_polarity_source}\n"
                f"Lens Larmor rotation {record.larmor_rotation_deg:+.6g} deg\n"
                f"Cumulative column rotation "
                f"{record.cumulative_column_rotation_deg:+.6g} deg"
            )
        self._update_formula_legend()
        self._add_rotation_markers(total)
        if previous_y is None:
            view_box.disableAutoRange()
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            view_box.updateAutoRange()
            view_box.disableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            view_box.setYRange(*previous_y, padding=0)
        self._has_field_scene = True
        peak = float(np.max(np.abs(total))) if total.size else 0.0
        total_rotation_deg = sum(
            record.larmor_rotation_deg for record in self._records
        )
        self.heading.setText(
            f"Axial magnetic field Bz — {len(self._records)} lenses | "
            f"total peak {peak:.6g} T | sample {sample_field_t:+.6g} T"
        )
        plane_text = "; ".join(
            f"{record.name} θsample "
            f"{record.image_rotation_from_sample_deg:+.4g}°"
            for record in self._plane_records
        )
        detail_text = (
            "Positive rotation follows the right-hand rule about +Z | "
            f"full-column signed Larmor rotation {total_rotation_deg:+.6g} deg | "
            "Gaussian fields use numerical 7σ tails; field maps use finite grid "
            "support. Inspect each curve for geometry/material coupling. "
            "Joint B-H fields are counted once, not decomposed per lens."
            + (f" | {plane_text}" if plane_text else "")
        )
        self.summary.setText(
            f"Signed Larmor rotation {total_rotation_deg:+.6g}° | "
            f"{len(self._plane_records)} reference plane(s) | Provider-defined support"
        )
        self.summary.setToolTip(detail_text)
        self._apply_curve_styles()
        if self._selected_key is not None:
            self.focus_component(next(record for record in self._records
                                      if record.key == self._selected_key))

    def _add_rotation_markers(self, total_field_t) -> None:
        """Refresh keyed annotations without accumulating new graphics items."""
        peak = max(
            float(np.max(np.abs(total_field_t)))
            if np.size(total_field_t) else 0.0,
            1.0e-6,
        )
        items = {}
        for index, record in enumerate(sorted(
            self._records, key=lambda item: item.center_z_mm
        )):
            field_index = int(np.argmin(np.abs(record.z_mm - record.center_z_mm)))
            field_value = float(record.field_t[field_index])
            offset = (0.055 + 0.025 * (index % 3)) * peak
            y_value = field_value + offset if field_value >= 0.0 else field_value - offset
            anchor_y = 1.0 if field_value >= 0.0 else 0.0
            key = ("lens", record.key)
            label = self._rotation_by_key.get(key)
            if label is None:
                label = pg.TextItem(color="#cbd5e1", fill=pg.mkBrush(5, 8, 22, 185))
                self.plot.addItem(label)
            label.setText(
                f"{record.key}\nΔφL {record.larmor_rotation_deg:+.3g}°"
            )
            label.setAnchor((0.5, anchor_y))
            label.border = pg.mkPen(record.formula_colour, width=0.8)
            label.setPos(record.center_z_mm, y_value)
            label.setToolTip(
                f"{record.name}\n"
                f"single-lens ΔφL {record.larmor_rotation_deg:+.6g} deg\n"
                f"column cumulative ΣφL "
                f"{record.cumulative_column_rotation_deg:+.6g} deg"
            )
            label.update()
            items[key] = label

        plane_colour = "#fbbf24"
        for index, record in enumerate(self._plane_records):
            key = ("plane_line", record.key)
            line = self._rotation_by_key.get(key)
            if line is None:
                line = pg.InfiniteLine(
                    angle=90, movable=False,
                    pen=pg.mkPen(plane_colour, width=1.1, style=Qt.PenStyle.DashLine),
                )
                self.plot.addItem(line)
            line.setValue(record.z_mm)
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
            items[key] = line
            y_value = peak * (0.92 - 0.13 * (index % 4))
            key = ("plane_label", record.key)
            label = self._rotation_by_key.get(key)
            if label is None:
                label = pg.TextItem(
                    color=plane_colour, anchor=(0.5, 0.0),
                    fill=pg.mkBrush(5, 8, 22, 210),
                    border=pg.mkPen(plane_colour, width=0.9),
                )
                self.plot.addItem(label)
            label.setText(
                f"{record.name}\nθsample {record.image_rotation_from_sample_deg:+.3g}°"
            )
            label.setPos(record.z_mm, y_value)
            label.setToolTip(line.toolTip())
            items[key] = label
        for key in self._rotation_by_key.keys() - items.keys():
            self.plot.removeItem(self._rotation_by_key[key])
        self._rotation_by_key = items
        self._rotation_items = list(items.values())
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
        support_colour = pg.mkColor(record.formula_colour)
        support_colour.setAlpha(34)
        if self._support_item is None:
            self._support_item = pg.LinearRegionItem(
                values=record.support_mm, orientation="vertical", movable=False,
                brush=pg.mkBrush(support_colour),
                pen=pg.mkPen(record.formula_colour, width=1.2),
            )
            self._support_item.setZValue(-5)
            self.plot.addItem(self._support_item)
        else:
            self._support_item.setRegion(record.support_mm)
            self._support_item.setBrush(pg.mkBrush(support_colour))
            for line in self._support_item.lines:
                line.setPen(pg.mkPen(record.formula_colour, width=1.2))
        detail_text = self.diagnostic_text(key)
        focal = (
            f"{record.focal_length_mm:.6g} mm"
            if math.isfinite(record.focal_length_mm) else "unfocused"
        )
        self.summary.setText(
            f"Selected: {record.name} | {record.excitation_percent:.6g}% | "
            f"peak |Bz| {record.peak_t:.6g} T | focal length {focal} | "
            f"ΔφL {record.larmor_rotation_deg:+.6g}°"
        )
        self.summary.setToolTip(detail_text)

    def diagnostic_text(self, key: str) -> str:
        if self._presentation_pending:
            return "Field view pending | show Magnetic field to update diagnostics."
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
            f"field at sample {record.field_at_sample_t:+.6g} T | "
            f"numerical support {record.support_mm[0]:.6g}–"
            f"{record.support_mm[1]:.6g} mm ({record.support_definition}) | "
            f"model status {record.field_model_status} | "
            f"geometry/material coupling {record.geometry_material_coupling}"
        )
