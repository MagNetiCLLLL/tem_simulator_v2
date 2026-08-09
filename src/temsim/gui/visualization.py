"""Central ray-path visualization workspace."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.diagnostics import ray_stop_records, vacuum_bore_plot_points
from temsim.gui.diagnostic_tabs import (
    EnergyFilterView,
    MagneticFieldView,
    OpticalTransferView,
    PhysicalLayoutView,
    TransverseBeamView,
)


class WaveImagingView(QWidget):
    """Display the optional one-shot TEM wave image and diffraction pattern."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.summary = QLabel(
            "Enable TEM wave imaging on the Sample and run High accuracy."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self.image = pg.ImageView()
        self.image.setObjectName("waveImageView")
        self.diffraction = pg.ImageView()
        self.diffraction.setObjectName("waveDiffractionView")
        for view in (self.image, self.diffraction):
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
        panels = QHBoxLayout()
        panels.addWidget(self.image, 1)
        panels.addWidget(self.diffraction, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(panels, 1)

    def display_result(self, wave_result) -> None:
        if wave_result is None:
            self.image.clear()
            self.diffraction.clear()
            self.summary.setToolTip("")
            self.summary.setText(
                "No wave image in this result. Enable TEM wave imaging on "
                "the Sample and run High accuracy."
            )
            return
        self.image.setImage(
            np.asarray(wave_result.image_intensity, dtype=float).T,
            autoRange=True,
            autoLevels=True,
        )
        self.diffraction.setImage(
            np.asarray(wave_result.diffraction_intensity, dtype=float).T,
            autoRange=True,
            autoLevels=True,
        )
        metrics = wave_result.metrics
        model = str(metrics.get("specimen_model", "unknown"))
        slices = int(metrics.get("specimen_slice_count", 0))
        potential_model = str(
            metrics.get("specimen_potential_model", "unknown potential")
        )
        configurations = int(
            metrics.get("specimen_configuration_count", 1)
        )
        warnings = []
        if bool(metrics.get("wave_sampling_truncates_illumination", False)):
            warnings.append("illumination exceeds wave bandwidth")
        if not bool(
            metrics.get(
                "wave_intensity_conservation_within_0_1_percent", True
            )
        ):
            warnings.append("intensity conservation check failed")
        if metrics.get("fft_fallback_reason"):
            warnings.append("wave CUDA fell back to CPU")
        atomistic_fallback = metrics.get(
            "specimen_atomistic_fallback_reason"
        )
        if atomistic_fallback and wave_result.preset_key != "vacuum":
            warnings.append("atomistic potential fell back")
        if (
            bool(metrics.get("specimen_frozen_phonon_applied", False))
            and configurations < 4
        ):
            warnings.append("frozen-phonon ensemble may be under-converged")
        relative_standard_error = float(
            metrics.get(
                "image_configuration_relative_standard_error", 0.0
            )
        )
        if relative_standard_error > 0.1:
            warnings.append("frozen-phonon image standard error exceeds 10%")
        warning_text = f" | WARNING: {', '.join(warnings)}" if warnings else ""
        backend = str(metrics.get("wave_compute_backend", "NumPy CPU"))
        thermal_text = ""
        if bool(metrics.get("specimen_frozen_phonon_applied", False)):
            thermal_text = (
                ", sigma "
                f"{float(metrics['specimen_thermal_sigma_angstrom']):.4g} Å"
            )
        self.summary.setText(
            f"{wave_result.preset_name} | "
            f"{model}, {slices} slices | "
            f"{potential_model}, {configurations} configuration(s)"
            f"{thermal_text} | "
            f"compute {backend} | "
            f"FOV {float(metrics['field_of_view_angstrom']):.5g} Å | "
            f"pixel {float(metrics['pixel_size_angstrom']):.5g} Å | "
            f"surviving rays {int(metrics['surviving_rays'])}"
            f"{warning_text}"
        )
        details = [
            "Intensity treatment: "
            f"{metrics.get('displayed_intensity_average', 'unknown')}",
            "Potential builder: "
            f"{metrics.get('specimen_potential_builder_backend', 'unknown')}",
        ]
        realised_extent = metrics.get(
            "specimen_realised_lateral_extent_angstrom"
        )
        if realised_extent is not None:
            details.append(
                "Realised periodic cell: "
                f"{float(realised_extent[0]):.5g} x "
                f"{float(realised_extent[1]):.5g} Å; "
                "realised thickness "
                f"{float(metrics.get('specimen_total_thickness_angstrom', 0.0)):.5g} Å."
            )
        if atomistic_fallback:
            details.append(f"Atomistic fallback: {atomistic_fallback}")
        if bool(metrics.get("specimen_frozen_phonon_applied", False)):
            details.append(
                "Independent isotropic Gaussian displacements; correlated "
                "phonons are not included."
            )
            details.append(
                "Image relative standard error: "
                f"{relative_standard_error:.3g}"
            )
        self.summary.setToolTip("\n".join(details))


class VisualizationWorkspace(QWidget):
    component_selected = Signal(str)
    MAX_DISPLAY_RAYS = 48
    MAX_RANGE_SAMPLE_RAYS = 256
    RAY_LABEL_BASE_PT = 10
    RAY_LABEL_MAX_PT = 14
    RAY_AXIS_TICK_PT = 10
    RAY_AXIS_LABEL_PT = 11
    RAY_LEGEND_PT = 10
    OPTION_BUTTON_STYLE = """
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
        QPushButton:hover {
            background: #e2e8f0;
            border-color: #334155;
        }
        QPushButton:checked {
            background: #2563eb;
            border-color: #1d4ed8;
            color: white;
        }
        QPushButton:checked:hover {
            background: #1d4ed8;
        }
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("visualizationWorkspace")

        self.heading = QLabel("Electron ray paths")
        font = self.heading.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self.heading.setFont(font)

        self._projection_angle_deg = 0.0
        self._projection_syncing = False
        self.projection_label = QLabel("View angle")
        self.projection_label.setToolTip(
            "Rotate the displayed transverse axis continuously about Z; "
            "this reprojects the existing X/Y result without retracing rays"
        )
        self.projection_xz = QPushButton("X-Z")
        self.projection_xz.setObjectName("projectionXZButton")
        self.projection_xz.setCheckable(True)
        self.projection_xz.setChecked(True)
        self.projection_yz = QPushButton("Y-Z")
        self.projection_yz.setObjectName("projectionYZButton")
        self.projection_yz.setCheckable(True)
        self.projection_slider = QSlider(Qt.Orientation.Horizontal)
        self.projection_slider.setObjectName("projectionAngleSlider")
        self.projection_slider.setRange(0, 3600)
        self.projection_slider.setSingleStep(1)
        self.projection_slider.setPageStep(50)
        self.projection_slider.setMinimumWidth(120)
        self.projection_slider.setMaximumWidth(180)
        self.projection_slider.setToolTip(
            "Transverse projection angle, in tenths of a degree"
        )
        self.projection_angle = QDoubleSpinBox()
        self.projection_angle.setObjectName("projectionAngleSpin")
        self.projection_angle.setRange(0.0, 360.0)
        self.projection_angle.setDecimals(2)
        self.projection_angle.setSingleStep(1.0)
        self.projection_angle.setSuffix(" deg")
        self.projection_angle.setWrapping(True)
        self.projection_angle.setToolTip(
            "0 deg = X-Z, 90 deg = Y-Z; intermediate angles show "
            "U = X cos(angle) + Y sin(angle)"
        )

        self.component_centres = QPushButton("Component centres")
        self.component_centres.setObjectName("componentCentresToggle")
        self.component_centres.setCheckable(True)
        self.component_centres.setChecked(True)
        self.crossovers = QPushButton("Crossovers")
        self.crossovers.setObjectName("crossoversToggle")
        self.crossovers.setCheckable(True)
        self.crossovers.setChecked(True)
        self.auto_zoom = QPushButton("Auto zoom")
        self.auto_zoom.setObjectName("autoZoomToggle")
        self.auto_zoom.setCheckable(True)
        self.auto_zoom.setToolTip(
            "Automatically zoom to the selected assembly component"
        )
        self.auto_zoom.setChecked(False)
        self.column_walls = QPushButton("Vacuum walls")
        self.column_walls.setObjectName("columnWallsToggle")
        self.column_walls.setCheckable(True)
        self.column_walls.setChecked(True)
        self.column_walls.setToolTip(
            "Show the position-dependent circular vacuum inner diameter"
        )
        self.fit_column = QPushButton("Fit vacuum")
        self.fit_column.setObjectName("fitColumnButton")
        self.fit_column.setToolTip(
            "Fit the complete axial range and column inner diameter"
        )
        self.axial_position = QDoubleSpinBox()
        self.axial_position.setObjectName("rayDiagramAxialPosition")
        self.axial_position.setRange(-1.0e6, 1.0e6)
        self.axial_position.setDecimals(3)
        self.axial_position.setSingleStep(1.0)
        self.axial_position.setSuffix(" mm")
        self.axial_position.setKeyboardTracking(False)
        self.axial_position.setToolTip(
            "Exact axial Z position to open in the Ray Diagram"
        )
        self.jump_to_position = QPushButton("Go to Z")
        self.jump_to_position.setObjectName("rayDiagramGoToPosition")
        self.jump_to_position.setToolTip(
            "Centre the Ray Diagram on the entered axial position"
        )
        for option_button in (
            self.projection_xz,
            self.projection_yz,
            self.auto_zoom,
            self.component_centres,
            self.crossovers,
            self.column_walls,
            self.fit_column,
            self.jump_to_position,
        ):
            option_button.setStyleSheet(self.OPTION_BUTTON_STYLE)

        title_row = QHBoxLayout()
        title_row.addWidget(self.heading)
        title_row.addStretch(1)
        title_row.addWidget(self.projection_label)
        title_row.addWidget(self.projection_xz)
        title_row.addWidget(self.projection_yz)
        title_row.addWidget(self.projection_slider)
        title_row.addWidget(self.projection_angle)
        title_row.addWidget(self.auto_zoom)
        title_row.addWidget(self.fit_column)
        title_row.addWidget(self.column_walls)
        title_row.addWidget(self.component_centres)
        title_row.addWidget(self.crossovers)

        navigation_row = QHBoxLayout()
        navigation_hint = QLabel(
            "Double-click an axial position in Ray Diagram, Physical Layout, "
            "or Magnetic Field to open the same Z here"
        )
        navigation_hint.setStyleSheet("color: #64748b; font-weight: 600;")
        navigation_row.addWidget(navigation_hint)
        navigation_row.addStretch(1)
        navigation_row.addWidget(QLabel("Axial Z"))
        navigation_row.addWidget(self.axial_position)
        navigation_row.addWidget(self.jump_to_position)

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("rayPlot")
        self._set_ray_axis_label("bottom", "Axial position")
        self._set_ray_axis_label("left", "Projected displacement")
        self._style_ray_axes()
        # Keep the ViewBox geometry invariant while the projection angle
        # changes.  Otherwise differently sized X/Y/U axis titles move the
        # on-screen Z origin even when the numeric Z range is restored.
        self.plot.getAxis("left").setWidth(112)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setMenuEnabled(True)
        self._style_ray_legend(self.plot.addLegend(offset=(10, 10)))
        self.component_marker_items = []
        self._component_labels = []
        self._ray_label_items = []
        self._ray_label_font_pt = None
        self.sample_marker_items = []
        self.aperture_marker_items = []
        self.aperture_optical_plane_items = []
        self.aperture_stop_segment_items = []
        self._aperture_span_records = []
        self._aperture_stops_by_key = {}
        self.deflector_pair_items = []
        self.crossover_marker_items = []
        self.column_wall_items = []
        self.stop_marker_items = []
        self.axial_cursor_item = None
        self._selected_z_mm = None
        self._last_result = None
        self._last_quality = ""
        self._focused_part = None
        self._show_notice("Waiting for the first calculation")

        self.stop_detail = QLabel(
            "Click a stop marker to inspect the first physical intercept"
        )
        self.stop_detail.setStyleSheet("color: #fbbf24; font-weight: 600;")
        self.hint = QLabel("Angle and display-scale diagnostics appear here")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.hint.setStyleSheet("color: #64748b; font-weight: 600;")
        self.hint.setToolTip(
            "Hover centre lines for details | Mouse wheel: zoom | "
            "Drag: pan | Right click: plot menu"
        )

        ray_page = QWidget()
        ray_layout = QVBoxLayout(ray_page)
        ray_layout.addLayout(title_row)
        ray_layout.addLayout(navigation_row)
        ray_layout.addWidget(self.plot, 1)
        ray_layout.addWidget(self.stop_detail)
        ray_layout.addWidget(self.hint)

        self.physical_layout = PhysicalLayoutView()
        self.magnetic_field = MagneticFieldView()
        self.optical_transfer = OpticalTransferView()
        self.energy_filter = EnergyFilterView()
        self.transverse_beam = TransverseBeamView()
        self.wave_imaging = WaveImagingView()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("visualizationTabs")
        self.tabs.addTab(ray_page, "Ray Diagram")
        self.tabs.addTab(self.physical_layout, "Physical Layout")
        self.tabs.addTab(self.magnetic_field, "Magnetic Field")
        self.tabs.addTab(self.optical_transfer, "Optical Transfer")
        self.tabs.addTab(self.energy_filter, "Energy Filter")
        self.tabs.addTab(self.transverse_beam, "Transverse X-Y")
        self.tabs.addTab(self.wave_imaging, "TEM Wave Image")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self.component_centres.toggled.connect(self._redraw_last_result)
        self.crossovers.toggled.connect(self._redraw_last_result)
        self.column_walls.toggled.connect(self._redraw_last_result)
        self.fit_column.clicked.connect(self._fit_column_view)
        self.auto_zoom.toggled.connect(self._auto_zoom_toggled)
        self.jump_to_position.clicked.connect(
            self._jump_to_position_input
        )
        self.axial_position.lineEdit().returnPressed.connect(
            self._jump_to_position_input
        )
        self.projection_xz.clicked.connect(
            lambda: self._set_projection_angle(0.0)
        )
        self.projection_yz.clicked.connect(
            lambda: self._set_projection_angle(90.0)
        )
        self.projection_slider.valueChanged.connect(
            self._projection_slider_changed
        )
        self.projection_angle.valueChanged.connect(
            self._projection_spin_changed
        )
        self.plot.getViewBox().sigXRangeChanged.connect(
            self._update_component_label_visibility
        )
        self.plot.getViewBox().sigRangeChanged.connect(
            self._update_scale_notice
        )
        self.plot.getViewBox().sigYRangeChanged.connect(
            self._update_aperture_spans
        )
        self.plot.scene().sigMouseClicked.connect(
            self._ray_plot_position_clicked
        )
        self.physical_layout.component_selected.connect(
            self.component_selected.emit
        )
        self.magnetic_field.component_selected.connect(
            self.component_selected.emit
        )
        self.energy_filter.component_selected.connect(
            self.component_selected.emit
        )
        self.physical_layout.axial_position_selected.connect(
            self.jump_to_ray_position
        )
        self.magnetic_field.axial_position_selected.connect(
            self.jump_to_ray_position
        )

    def _show_notice(self, text: str) -> None:
        self.plot.clear()
        notice = pg.TextItem(text, color="#94a3b8", anchor=(0.5, 0.5))
        notice.setFont(self._marker_font(self.RAY_AXIS_LABEL_PT))
        notice.setPos(0.5, 0.5)
        self.plot.addItem(notice)
        self.plot.setXRange(0.0, 1.0, padding=0.0)
        self.plot.setYRange(0.0, 1.0, padding=0.0)

    def _set_ray_axis_label(self, axis: str, text: str) -> None:
        self.plot.setLabel(
            axis,
            text,
            units="mm",
            **{
                "color": "#e2e8f0",
                "font-size": f"{self.RAY_AXIS_LABEL_PT}pt",
                "font-weight": "600",
            },
        )

    def _style_ray_axes(self) -> None:
        tick_font = self._marker_font(self.RAY_AXIS_TICK_PT)
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.setTickFont(tick_font)
            axis.setStyle(tickTextOffset=7)

    def _style_ray_legend(self, legend) -> None:
        legend.setLabelTextSize(f"{self.RAY_LEGEND_PT}pt")
        legend.setLabelTextColor("#e2e8f0")

    @staticmethod
    def _bundle_lines(
        z, values, count: int, blocked_z=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build visible ray segments ending at their first blocking plane."""
        z = np.asarray(z, dtype=float)
        values = np.asarray(values, dtype=float)
        ray_count = values.shape[1]
        indices = np.unique(np.linspace(0, ray_count - 1, count, dtype=int))
        x_segments = []
        y_segments = []
        for index in indices:
            ray_z = z
            ray_value = values[:, index]
            if blocked_z is not None and np.isfinite(blocked_z[index]):
                stop_z = float(blocked_z[index])
                stop_index = int(np.searchsorted(z, stop_z, side="right"))
                if stop_index == 0:
                    continue
                ray_z = z[:stop_index]
                ray_value = values[:stop_index, index]
                if stop_index < z.size and ray_z[-1] < stop_z:
                    left_z = float(z[stop_index - 1])
                    right_z = float(z[stop_index])
                    fraction = (stop_z - left_z) / (right_z - left_z)
                    stop_value = ray_value[-1] + fraction * (
                        values[stop_index, index] - ray_value[-1]
                    )
                    ray_z = np.append(ray_z, stop_z)
                    ray_value = np.append(ray_value, stop_value)
            x_segments.extend((ray_z, np.array([np.nan])))
            y_segments.extend((ray_value * 1.0e3, np.array([np.nan])))
        if not x_segments:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(x_segments), np.concatenate(y_segments)

    def _redraw_last_result(self) -> None:
        if self._last_result is not None:
            self._draw_ray_diagram(
                self._last_result,
                self._last_quality,
                preserve_view=True,
            )

    @staticmethod
    def _ray_geometry_signature(result) -> tuple | None:
        """Identify geometry changes that require a fresh column fit."""

        assembly = getattr(result, "assembly", None)
        if assembly is None:
            return None
        return (
            tuple(getattr(assembly, "selected_module_paths", ())),
            tuple(
                (
                    str(part.key),
                    float(part.start_z_mm),
                    float(part.center_z_mm),
                    float(part.end_z_mm),
                )
                for part in assembly.parts
                if not bool(part.data.get("branch_path_only", False))
            ),
        )

    @staticmethod
    def _project_transverse_values(x, y, angle_deg: float) -> np.ndarray:
        """Project X/Y values onto a transverse axis rotated about Z."""
        angle_rad = np.deg2rad(float(angle_deg))
        return (
            np.asarray(x, dtype=float) * np.cos(angle_rad)
            + np.asarray(y, dtype=float) * np.sin(angle_rad)
        )

    def _project_transverse(self, x, y) -> np.ndarray:
        return self._project_transverse_values(
            x, y, self._projection_angle_deg
        )

    def _projection_axis_name(self) -> str:
        angle = self._projection_angle_deg % 360.0
        cardinal = (
            (0.0, "X"),
            (90.0, "Y"),
            (180.0, "-X"),
            (270.0, "-Y"),
        )
        for cardinal_angle, name in cardinal:
            if np.isclose(angle, cardinal_angle, atol=0.05):
                return name
        return f"U({self._format_angle(angle)} deg)"

    @staticmethod
    def _format_angle(angle_deg: float) -> str:
        return f"{float(angle_deg):.2f}".rstrip("0").rstrip(".")

    def _projection_slider_changed(self, value: int) -> None:
        if not self._projection_syncing:
            self._set_projection_angle(float(value) / 10.0)

    def _projection_spin_changed(self, value: float) -> None:
        if not self._projection_syncing:
            self._set_projection_angle(value)

    def _set_projection_angle(self, angle_deg: float) -> None:
        angle = float(np.clip(angle_deg, 0.0, 360.0))
        changed = not np.isclose(
            angle, self._projection_angle_deg, atol=1.0e-12
        )
        self._projection_angle_deg = angle
        self._projection_syncing = True
        try:
            self.projection_slider.setValue(
                int(np.floor(angle * 10.0 + 0.5))
            )
            self.projection_angle.setValue(angle)
            normalized = angle % 360.0
            self.projection_xz.setChecked(
                bool(np.isclose(normalized, 0.0, atol=0.05))
            )
            self.projection_yz.setChecked(
                bool(np.isclose(normalized, 90.0, atol=0.05))
            )
        finally:
            self._projection_syncing = False
        if changed and self._last_result is not None:
            self._draw_ray_diagram(
                self._last_result,
                self._last_quality,
                preserve_view=True,
            )

    def _auto_zoom_toggled(self, checked: bool) -> None:
        if checked and self._focused_part is not None:
            self._apply_component_zoom(self._focused_part)

    def focus_component(self, part) -> None:
        """Remember the selected part and optionally focus its optical region."""
        self._focused_part = part
        self.physical_layout.focus_component(part)
        self.magnetic_field.focus_component(part)
        self.transverse_beam.focus_component(part)
        if self.auto_zoom.isChecked() and self._last_result is not None:
            self._apply_component_zoom(part)

    @staticmethod
    def _component_x_range(part) -> tuple[float, float]:
        mechanical_span = max(
            abs(float(part.end_z_mm) - float(part.start_z_mm)),
            abs(float(part.length_mm)),
            1.0,
        )
        half_window = max(45.0, min(260.0, 1.6 * mechanical_span))
        centre = float(part.center_z_mm)
        return centre - half_window, centre + half_window

    def _simulation_x_limits(self) -> tuple[float, float] | None:
        if self._last_result is None:
            return None
        branches = [self._last_result.simulation.incident]
        branches.extend(self._last_result.simulation.branches.values())
        minima = [float(np.nanmin(branch.z)) for branch in branches]
        maxima = [float(np.nanmax(branch.z)) for branch in branches]
        return min(minima), max(maxima)

    def _jump_to_position_input(self) -> None:
        self.jump_to_ray_position(self.axial_position.value())

    def _ray_plot_position_clicked(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not event.double()
        ):
            return
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        position = view_box.mapSceneToView(event.scenePos())
        self.jump_to_ray_position(float(position.x()))
        event.accept()

    def _add_axial_position_cursor(self) -> None:
        if self._selected_z_mm is None:
            return
        limits = self._simulation_x_limits()
        cursor = pg.InfiniteLine(
            pos=self._selected_z_mm,
            angle=90,
            movable=True,
            pen=pg.mkPen("#22d3ee", width=2.0),
            hoverPen=pg.mkPen("#a5f3fc", width=3.0),
            label="Selected Z",
            labelOpts={
                "position": 0.96,
                "color": "#a5f3fc",
                "rotateAxis": (1, 0),
            },
        )
        if limits is not None:
            cursor.setBounds(limits)
        cursor.setZValue(45)
        cursor.setToolTip(
            "Selected axial position; drag to another Z or double-click "
            "an axial plot"
        )
        self._register_ray_label(cursor.label)
        cursor.sigPositionChangeFinished.connect(
            self._axial_cursor_move_finished
        )
        self.plot.addItem(cursor)
        self.axial_cursor_item = cursor

    def _axial_cursor_move_finished(self, cursor) -> None:
        self.jump_to_ray_position(float(cursor.value()))

    def jump_to_ray_position(
        self,
        z_mm: float,
        window_mm: float | None = None,
        activate_tab: bool = True,
    ) -> None:
        """Open an axial Z location in the Ray Diagram without retracing."""
        limits = self._simulation_x_limits()
        if limits is None or not np.isfinite(z_mm):
            return
        lower_limit, upper_limit = limits
        selected = float(np.clip(z_mm, lower_limit, upper_limit))
        self._selected_z_mm = selected
        self.axial_position.setRange(lower_limit, upper_limit)
        self.axial_position.setValue(selected)
        if activate_tab:
            self.tabs.setCurrentIndex(0)

        if self.axial_cursor_item is None:
            self._add_axial_position_cursor()
        else:
            self.axial_cursor_item.setValue(selected)

        current_range = self.plot.getViewBox().viewRange()[0]
        current_span = max(float(current_range[1] - current_range[0]), 1.0)
        requested_span = current_span if window_mm is None else float(window_mm)
        full_span = max(upper_limit - lower_limit, 1.0)
        view_span = min(max(requested_span, 10.0), 260.0, full_span)
        x_min = selected - 0.5 * view_span
        x_max = selected + 0.5 * view_span
        x_min, x_max = self._clamp_focus_range(x_min, x_max)
        # The initial full-column fit is one-shot. Keeping AutoRange enabled
        # lets aperture span updates pull a manually zoomed view back out.
        self.plot.disableAutoRange()
        self.plot.setXRange(x_min, x_max, padding=0.0)
        y_range = self._local_y_range(x_min, x_max)
        if y_range is not None:
            self.plot.setYRange(*y_range, padding=0.0)
        self._update_component_label_visibility()
        self.stop_detail.setText(
            f"Selected axial position: Z {selected:.9g} mm | "
            "drag the cyan cursor or double-click another axial plot"
        )

    def _column_radius_mm(self) -> float | None:
        assembly = getattr(self._last_result, "assembly", None)
        segments = getattr(assembly, "vacuum_bore_segments", ())
        if not segments:
            return None
        return 0.5 * max(
            float(segment.inner_diameter_mm) for segment in segments
        )

    def _fit_column_view(self) -> None:
        limits = self._simulation_x_limits()
        radius_mm = self._column_radius_mm()
        if limits is None or radius_mm is None:
            return
        self.plot.disableAutoRange()
        self.plot.setXRange(*limits, padding=0.0)
        margin = max(0.05 * radius_mm, 0.01)
        self.plot.setYRange(
            -radius_mm - margin,
            radius_mm + margin,
            padding=0.0,
        )
        self._update_component_label_visibility()

    def _add_column_walls(self, result) -> None:
        if not self.column_walls.isChecked():
            return
        assembly = getattr(result, "assembly", None)
        segments = getattr(assembly, "vacuum_bore_segments", ())
        if not segments:
            return
        z, upper, lower = vacuum_bore_plot_points(assembly)
        minimum = min(float(item.inner_diameter_mm) for item in segments)
        maximum = max(float(item.inner_diameter_mm) for item in segments)
        tooltip = (
            "Position-dependent circular vacuum bore\n"
            f"Inner diameter range = {minimum:.6g}–{maximum:.6g} mm\n"
            "Each axial section is owned by its TOML component; a ray stops "
            "at its first X/Y radial contact"
        )
        for index, displacement in enumerate((upper, lower)):
            item = self.plot.plot(
                z,
                displacement,
                pen=pg.mkPen("#f8fafc", width=2.3),
                name=(
                    f"Vacuum wall (ID {minimum:.6g}–{maximum:.6g} mm)"
                    if index == 0 else None
                ),
            )
            item.setZValue(2)
            item.setToolTip(tooltip)
            self.column_wall_items.append(item)

    def _add_stop_markers(self, simulation) -> None:
        records = ray_stop_records(simulation, maximum_records=600)
        groups = {}
        for record in records:
            if record.key == "column_wall":
                group = "Column wall"
                colour = "#ff453a"
            elif "aperture" in record.key:
                group = "Aperture stop"
                colour = "#ffb000"
            else:
                group = "Recording/device stop"
                colour = "#a78bfa"
            groups.setdefault((group, colour), []).append(record)
        for (group, colour), group_records in groups.items():
            projected_mm = self._project_transverse(
                [record.x_mm for record in group_records],
                [record.y_mm for record in group_records],
            )
            item = pg.ScatterPlotItem(
                x=[record.z_mm for record in group_records],
                y=projected_mm,
                data=group_records,
                symbol="x",
                size=9,
                pen=pg.mkPen(colour, width=1.8),
                name=group,
            )
            item.setZValue(30)
            item.setToolTip(
                f"{group}: projected on {self._projection_axis_name()}; "
                "click a marker for exact X/Y diagnostics"
            )
            item.sigClicked.connect(self._stop_marker_clicked)
            self.plot.addItem(item)
            self.stop_marker_items.append(item)

    def _stop_marker_clicked(self, _item, points, _event=None) -> None:
        if not points:
            return
        record = points[0].data()
        projected_mm = float(
            self._project_transverse(record.x_mm, record.y_mm)
        )
        self.stop_detail.setText(
            f"First intercept: {record.key} | {record.bundle} ray {record.ray_index} | "
            f"Z {record.z_mm:.9g} mm | {self._projection_axis_name()} "
            f"{projected_mm:.9g} mm | X {record.x_mm:.9g} mm | "
            f"Y {record.y_mm:.9g} mm | radius {record.radial_mm:.9g} mm"
        )

    @staticmethod
    def _column_wall_stop_count(simulation) -> int:
        incident_keys = np.asarray(simulation.incident.blocked_key, dtype=object)
        count = int(np.count_nonzero(incident_keys == "column_wall"))
        sample_z = float(simulation.incident.z[-1])
        for branch in simulation.branches.values():
            keys = np.asarray(branch.blocked_key, dtype=object)
            blocked_z = np.asarray(branch.blocked_z, dtype=float)
            count += int(np.count_nonzero(
                (keys == "column_wall") & (blocked_z > sample_z + 1.0e-9)
            ))
        return count

    def _clamp_focus_range(
        self, x_min: float, x_max: float
    ) -> tuple[float, float]:
        limits = self._simulation_x_limits()
        if limits is None:
            return x_min, x_max
        lower, upper = limits
        requested_span = x_max - x_min
        if requested_span >= upper - lower:
            return lower, upper
        if x_min < lower:
            return lower, lower + requested_span
        if x_max > upper:
            return upper - requested_span, upper
        return x_min, x_max

    def _local_y_range(
        self, x_min: float, x_max: float
    ) -> tuple[float, float] | None:
        if self._last_result is None:
            return None
        simulation = self._last_result.simulation
        branches = [simulation.incident, *simulation.branches.values()]
        local_values = []
        for branch in branches:
            z_values = np.asarray(branch.z, dtype=float)
            in_window = (z_values >= x_min) & (z_values <= x_max)
            if not np.any(in_window):
                continue
            z_indices = np.flatnonzero(in_window)
            ray_count = branch.x.shape[1]
            ray_indices = np.unique(
                np.linspace(
                    0,
                    ray_count - 1,
                    min(ray_count, self.MAX_RANGE_SAMPLE_RAYS),
                    dtype=int,
                )
            )
            projected_values = (
                self._project_transverse(
                    branch.x[np.ix_(z_indices, ray_indices)],
                    branch.y[np.ix_(z_indices, ray_indices)],
                )
                * 1.0e3
            )
            blocked_z = np.asarray(branch.blocked_z, dtype=float)[ray_indices]
            valid_to_stop = (
                np.isnan(blocked_z)[None, :]
                | (z_values[z_indices, None] <= blocked_z[None, :])
            )
            finite = projected_values[
                np.isfinite(projected_values) & valid_to_stop
            ]
            if finite.size:
                local_values.append(finite)
        if not local_values:
            return None
        values = np.concatenate(local_values)
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        span = max(y_max - y_min, 0.02)
        midpoint = 0.5 * (y_min + y_max)
        margin = max(0.12 * span, 0.01)
        return midpoint - span / 2.0 - margin, midpoint + span / 2.0 + margin

    def _maximum_visible_projection_slope(self) -> float:
        if self._last_result is None:
            return 0.0
        x_min, x_max = self.plot.getViewBox().viewRange()[0]
        simulation = self._last_result.simulation
        branches = [simulation.incident, *simulation.branches.values()]
        maximum = 0.0
        for branch in branches:
            z_values = np.asarray(branch.z, dtype=float)
            in_window = (z_values >= x_min) & (z_values <= x_max)
            if not np.any(in_window):
                continue
            z_indices = np.flatnonzero(in_window)
            ray_count = branch.tx.shape[1]
            ray_indices = np.unique(
                np.linspace(
                    0,
                    ray_count - 1,
                    min(ray_count, self.MAX_RANGE_SAMPLE_RAYS),
                    dtype=int,
                )
            )
            slopes = self._project_transverse(
                branch.tx[np.ix_(z_indices, ray_indices)],
                branch.ty[np.ix_(z_indices, ray_indices)],
            )
            blocked_z = np.asarray(branch.blocked_z, dtype=float)[ray_indices]
            valid_to_stop = (
                np.isnan(blocked_z)[None, :]
                | (z_values[z_indices, None] <= blocked_z[None, :])
            )
            valid = np.abs(slopes[np.isfinite(slopes) & valid_to_stop])
            if valid.size:
                maximum = max(maximum, float(np.max(valid)))
        return maximum

    def _update_scale_notice(self, *_args) -> None:
        if self._last_result is None:
            return
        view_box = self.plot.getViewBox()
        (x_min, x_max), (y_min, y_max) = view_box.viewRange()
        x_span = max(float(x_max - x_min), np.finfo(float).eps)
        y_span = max(float(y_max - y_min), np.finfo(float).eps)
        pixels_per_x_mm = max(float(view_box.width()), 1.0) / x_span
        pixels_per_y_mm = max(float(view_box.height()), 1.0) / y_span
        transverse_magnification = pixels_per_y_mm / pixels_per_x_mm
        maximum_angle_deg = float(
            np.degrees(
                np.arctan(self._maximum_visible_projection_slope())
            )
        )
        if transverse_magnification >= 10.0:
            magnification_text = f"{transverse_magnification:.0f}×"
        else:
            magnification_text = f"{transverse_magnification:.2f}×"
        self.hint.setText(
            f"Max physical {self._projection_axis_name()} angle: "
            f"{maximum_angle_deg:.3g}° | "
            f"Transverse display: {magnification_text} (angles not to scale) | "
            "Blocked rays stop at first intercept | Column wall uses radial X/Y"
        )

    def _apply_component_zoom(self, part) -> None:
        if self._last_result is None:
            return
        x_min, x_max = self._clamp_focus_range(*self._component_x_range(part))
        self.plot.disableAutoRange()
        self.plot.setXRange(x_min, x_max, padding=0.0)
        y_range = self._local_y_range(x_min, x_max)
        if y_range is not None:
            self.plot.setYRange(*y_range, padding=0.0)
        self._update_component_label_visibility()

    @staticmethod
    def _all_crossovers(result) -> list[dict[str, object]]:
        """Combine lens and gun waists without drawing duplicate positions."""
        records = [dict(item) for item in result.lens_crossovers]
        gun_waist = result.simulation.gun_waist
        if gun_waist is not None:
            records.insert(
                0,
                {
                    **gun_waist,
                    "name": "Gun crossover",
                    "source_lens_name": "Electrostatic Gun Lens",
                },
            )

        supplemental = []
        if result.simulation.c2c3_crossover is not None:
            supplemental.append(result.simulation.c2c3_crossover)
        supplemental.extend(result.simulation.corrector_crossovers or [])
        for item in supplemental:
            z_mm = float(item["z_mm"])
            if not any(abs(float(old["z_mm"]) - z_mm) <= 1.0 for old in records):
                records.append(dict(item))
        return sorted(records, key=lambda item: float(item["z_mm"]))

    @staticmethod
    def _marker_font(point_size: int = RAY_LABEL_BASE_PT) -> QFont:
        font = QFont()
        font.setPointSize(int(point_size))
        font.setWeight(QFont.Weight.DemiBold)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return font

    def _ray_label_point_size(self) -> int:
        """Return a readable, bounded label size for the current axial zoom."""

        limits = self._simulation_x_limits()
        if limits is None:
            return self.RAY_LABEL_BASE_PT
        x_min, x_max = self.plot.getViewBox().viewRange()[0]
        visible_span = max(float(x_max - x_min), np.finfo(float).eps)
        full_span = max(float(limits[1] - limits[0]), visible_span)
        zoom_ratio = max(full_span / visible_span, 1.0)
        zoom_steps = int(np.floor(np.log2(zoom_ratio) / 2.0))
        return int(
            np.clip(
                self.RAY_LABEL_BASE_PT + zoom_steps,
                self.RAY_LABEL_BASE_PT,
                self.RAY_LABEL_MAX_PT,
            )
        )

    def _register_ray_label(self, label) -> None:
        self._ray_label_items.append(label)
        label.setFont(self._marker_font(self._ray_label_point_size()))

    def _update_ray_label_fonts(self) -> int:
        point_size = self._ray_label_point_size()
        if self._ray_label_font_pt == point_size:
            return point_size
        font = self._marker_font(point_size)
        for label in self._ray_label_items:
            label.setFont(font)
        self._ray_label_font_pt = point_size
        return point_size

    @staticmethod
    def _deflector_planes(part) -> tuple[float, ...]:
        local_planes = part.data.get("interaction_centers_local_z_mm", ())
        if len(local_planes) != 2:
            return ()
        local_center = float(
            part.data.get("local_center_z_mm", part.center_z_mm)
        )
        module_origin = float(part.center_z_mm) - local_center
        return tuple(module_origin + float(value) for value in local_planes)

    @staticmethod
    def _aperture_optical_plane(part) -> float:
        local_center = float(
            part.data.get("local_center_z_mm", part.center_z_mm)
        )
        local_optical = float(
            part.data.get("optical_reference_local_z_mm", local_center)
        )
        module_origin = float(part.center_z_mm) - local_center
        return module_origin + local_optical

    def _update_aperture_spans(self, *_args) -> None:
        if not self._aperture_span_records:
            return
        y_min, y_max = self.plot.getViewBox().viewRange()[1]
        y_span = max(float(y_max - y_min), np.finfo(float).eps)
        for lower, upper, centre_u_mm, radius_mm in self._aperture_span_records:
            opening_lower = centre_u_mm - radius_mm
            opening_upper = centre_u_mm + radius_mm
            lower_fraction = float(
                np.clip((opening_lower - y_min) / y_span, 0.0, 1.0)
            )
            upper_fraction = float(
                np.clip((opening_upper - y_min) / y_span, 0.0, 1.0)
            )
            lower.setSpan(0.0, lower_fraction)
            upper.setSpan(upper_fraction, 1.0)

    def _add_aperture_stop(
        self, part, optical_z_mm: float, record
    ):
        available = record is not None
        enabled = bool(
            available
            and record.get("enabled", True)
            and record.get("installed", True)
        )
        if not enabled:
            status = "RETRACTED" if available else "REFERENCE ONLY"
            line = pg.InfiniteLine(
                pos=optical_z_mm,
                angle=90,
                pen=pg.mkPen(
                    "#94a3b8", width=1.1, style=Qt.PenStyle.DotLine
                ),
                label=f"{part.name} [OPTICAL STOP {status}]",
                labelOpts={
                    "position": 0.62,
                    "color": "#cbd5e1",
                    "rotateAxis": (1, 0),
                },
            )
            tooltip = (
                f"{part.name}\nOptical reference Z = {optical_z_mm:.6g} mm\n"
                "No active hard-edge blocking is applied."
            )
            lines = (line,)
            representative = line
        else:
            radius_mm = max(0.0, float(record["radius_mm"]))
            offset_x_mm = float(record["offset_x_mm"])
            offset_y_mm = float(record["offset_y_mm"])
            centre_u_mm = float(
                self._project_transverse(offset_x_mm, offset_y_mm)
            )
            lower_edge = centre_u_mm - radius_mm
            upper_edge = centre_u_mm + radius_mm
            tooltip = (
                f"{part.name}\nEffective optical stop Z = "
                f"{optical_z_mm:.6g} mm\nAllowed "
                f"{self._projection_axis_name()} opening = "
                f"[{lower_edge:.6g}, {upper_edge:.6g}] mm\n"
                f"Circular radius = {radius_mm:.6g} mm\n"
                f"X/Y offset = {offset_x_mm:.6g} / "
                f"{offset_y_mm:.6g} mm\n"
                "The blank gap is the circular opening projected onto the "
                "selected transverse axis; solid segments block."
            )
            lower = pg.InfiniteLine(
                pos=optical_z_mm,
                angle=90,
                span=(0.0, 0.45),
                pen=pg.mkPen("#ffb000", width=2.4),
            )
            upper = pg.InfiniteLine(
                pos=optical_z_mm,
                angle=90,
                span=(0.55, 1.0),
                pen=pg.mkPen("#ffb000", width=2.4),
                label=f"{part.name} [OPTICAL STOP]",
                labelOpts={
                    "position": 0.62,
                    "color": "#ffe29a",
                    "rotateAxis": (1, 0),
                },
            )
            self._aperture_span_records.append(
                (lower, upper, centre_u_mm, radius_mm)
            )
            lines = (lower, upper)
            representative = upper

        for line in lines:
            line.setZValue(14)
            line.setToolTip(tooltip)
            self.plot.addItem(line)
            self.aperture_stop_segment_items.append(line)
        self._register_ray_label(representative.label)
        representative.label.setToolTip(tooltip)
        self.aperture_marker_items.append(representative)
        self.aperture_optical_plane_items.append(representative)
        return representative

    def _add_aperture_component(self, part, index: int) -> None:
        record = self._aperture_stops_by_key.get(part.key)
        optical_z_mm = (
            float(record["z_mm"])
            if record is not None
            else self._aperture_optical_plane(part)
        )
        separate_plane = (
            abs(optical_z_mm - float(part.center_z_mm)) > 1.0e-9
        )
        if separate_plane:
            body = pg.InfiniteLine(
                pos=part.center_z_mm,
                angle=90,
                pen=pg.mkPen(
                    "#fb923c", width=1.2, style=Qt.PenStyle.DashLine
                ),
                label=f"{part.name} [BODY CENTRE]",
                labelOpts={
                    "position": 0.76 + 0.07 * (index % 4),
                    "color": "#fb923c",
                    "rotateAxis": (1, 0),
                },
            )
            tooltip = (
                f"{part.name}\nMechanical body centre Z = "
                f"{part.center_z_mm:.6g} mm\nEffective optical stop Z = "
                f"{optical_z_mm:.6g} mm"
            )
            body.setZValue(12)
            body.setToolTip(tooltip)
            self._register_ray_label(body.label)
            body.label.setToolTip(tooltip)
            self.plot.addItem(body)
            self.component_marker_items.append(body)
            self._component_labels.append(
                (body.label, float(part.center_z_mm), True)
            )

        stop = self._add_aperture_stop(part, optical_z_mm, record)
        if not separate_plane:
            self.component_marker_items.append(stop)
        self._component_labels.append((stop.label, optical_z_mm, True))

    def _add_deflector_pair(self, part, planes: tuple[float, ...]) -> None:
        upper_z, lower_z = planes
        coincident = abs(upper_z - lower_z) <= 1.0e-9
        if not coincident:
            region = pg.LinearRegionItem(
                values=(upper_z, lower_z),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(16, 255, 170, 28),
                pen=pg.mkPen(None),
            )
            region.setZValue(3)
            region.setToolTip(f"{part.name}\nPaired deflector envelope")
            self.plot.addItem(region)
            self.deflector_pair_items.append(region)

        plane_values = (upper_z,) if coincident else (upper_z, lower_z)
        plane_names = ("U/L",) if coincident else ("U", "L")
        for index, (z_mm, plane_name) in enumerate(zip(plane_values, plane_names)):
            if coincident:
                label = f"{part.name} U/L coincident"
                label_position = 0.72
            elif index == 0:
                label = f"{part.name} U"
                label_position = 0.70
            else:
                label = "L"
                label_position = 0.52
            line = pg.InfiniteLine(
                pos=z_mm,
                angle=90,
                pen=pg.mkPen("#2cffad", width=2.0),
                label=label,
                labelOpts={
                    "position": label_position,
                    "color": "#b7ffdf",
                    "rotateAxis": (1, 0),
                },
            )
            line.setZValue(11)
            tooltip = (
                f"{part.name}\n{plane_name} deflection plane Z = "
                f"{z_mm:.6g} mm"
            )
            line.setToolTip(tooltip)
            self._register_ray_label(line.label)
            line.label.setToolTip(tooltip)
            self.plot.addItem(line)
            self.deflector_pair_items.append(line)

    def _add_component_markers(self, assembly) -> None:
        if assembly is None or not self.component_centres.isChecked():
            return
        for index, part in enumerate(assembly.parts):
            if bool(part.data.get("branch_path_only", False)):
                continue
            # The specimen plane is drawn independently and remains visible
            # when generic component-centre markers are hidden.
            if part.key == "sample":
                continue
            is_lens = "lens" in part.key
            is_aperture = "aperture" in part.key
            if is_aperture:
                self._add_aperture_component(part, index)
                continue
            planes = self._deflector_planes(part)
            is_deflector = bool(planes)
            if is_deflector:
                centre = pg.ScatterPlotItem(
                    x=[part.center_z_mm],
                    y=[0.0],
                    symbol="+",
                    size=13,
                    pen=pg.mkPen("#c3ffe3", width=2.0),
                )
                centre.setZValue(12)
                centre.setToolTip(
                    f"{part.name}\nPair centre Z = "
                    f"{part.center_z_mm:.6g} mm"
                )
                self.plot.addItem(centre)
                self.component_marker_items.append(centre)
                self._add_deflector_pair(part, planes)
                continue
            if is_lens:
                colour = "#22d3ee"
                width = 1.3
                style = Qt.PenStyle.DashLine
                label = part.name
            else:
                colour = "#94a3b8"
                width = 0.7
                style = Qt.PenStyle.DashLine
                label = part.name
            line = pg.InfiniteLine(
                pos=part.center_z_mm,
                angle=90,
                pen=pg.mkPen(
                    colour,
                    width=width,
                    style=style,
                ),
                label=label,
                labelOpts={
                    "position": 0.76 + 0.07 * (index % 4),
                    "color": colour,
                    "rotateAxis": (1, 0),
                },
            )
            line.setZValue(6)
            tooltip = f"{part.name}\nCentre Z = {part.center_z_mm:.6g} mm"
            line.setToolTip(tooltip)
            self._register_ray_label(line.label)
            line.label.setToolTip(tooltip)
            self.plot.addItem(line)
            self.component_marker_items.append(line)
            self._component_labels.append(
                (
                    line.label,
                    float(part.center_z_mm),
                    is_lens,
                )
            )

    def _add_sample_marker(self, result) -> None:
        """Draw the incident/post-specimen boundary above rays and lenses."""

        assembly = getattr(result, "assembly", None)
        if assembly is not None:
            try:
                sample_part = assembly.part("sample")
            except KeyError:
                sample_part = None
        else:
            sample_part = None
        sample_z_mm = (
            float(sample_part.center_z_mm)
            if sample_part is not None
            else float(result.simulation.incident.z[-1])
        )
        label = f"SAMPLE / SPECIMEN  Z={sample_z_mm:.6g} mm"
        tooltip = (
            "Sample / specimen plane\n"
            f"Exact axial position Z = {sample_z_mm:.9g} mm\n"
            "The blue incident bundle terminates here and every "
            "post-specimen branch starts here. Their overlap at this plane "
            "is the continuous ray boundary, not a second optical element."
        )
        line = pg.InfiniteLine(
            pos=sample_z_mm,
            angle=90,
            pen=pg.mkPen("#ffffff", width=2.6),
            label=label,
            labelOpts={
                "position": 0.94,
                "color": "#ffffff",
                "rotateAxis": (1, 0),
            },
        )
        line.setZValue(40)
        line.setToolTip(tooltip)
        self._register_ray_label(line.label)
        line.label.setToolTip(tooltip)
        self.plot.addItem(line)

        axis_marker = pg.ScatterPlotItem(
            x=[sample_z_mm],
            y=[0.0],
            symbol="s",
            size=11,
            pen=pg.mkPen("#ffffff", width=2.0),
            brush=pg.mkBrush("#ef4444"),
        )
        axis_marker.setZValue(41)
        axis_marker.setToolTip(tooltip)
        self.plot.addItem(axis_marker)

        self.sample_marker_items.extend((line, axis_marker))
        # Preserve the established one-marker-per-assembly-part diagnostic.
        self.component_marker_items.append(line)
        self._component_labels.append((line.label, sample_z_mm, True))

    def _update_component_label_visibility(self, *_args) -> None:
        """Keep the full-column view legible and reveal labels while zooming."""
        point_size = self._update_ray_label_fonts()
        if not self._component_labels:
            return
        x_min, x_max = self.plot.getViewBox().viewRange()[0]
        span = max(float(x_max - x_min), np.finfo(float).eps)
        pixels_per_mm = max(self.plot.width(), 1) / span
        priority_pixels = [
            (z_mm - x_min) * pixels_per_mm
            for _label, z_mm, is_priority in self._component_labels
            if is_priority
        ]
        priority_clearance = max(32.0, 3.0 * point_size)
        regular_spacing = max(38.0, 3.5 * point_size)
        last_regular_pixel = -float("inf")
        for label, z_mm, is_priority in sorted(
            self._component_labels, key=lambda item: item[1]
        ):
            pixel = (z_mm - x_min) * pixels_per_mm
            if is_priority:
                label.setVisible(True)
                continue
            separated_from_priority = all(
                abs(pixel - priority_pixel) >= priority_clearance
                for priority_pixel in priority_pixels
            )
            visible = (
                separated_from_priority
                and pixel - last_regular_pixel >= regular_spacing
            )
            label.setVisible(visible)
            if visible:
                last_regular_pixel = pixel

    def _add_crossover_markers(self, result) -> None:
        if not self.crossovers.isChecked():
            return
        records = self._all_crossovers(result)
        for index, record in enumerate(records):
            z_mm = float(record["z_mm"])
            rms_radius = float(record.get("rms_radius_mm", float("nan")))
            name = str(record.get("name", "Crossover"))
            line = pg.InfiniteLine(
                pos=z_mm,
                angle=90,
                pen=pg.mkPen(
                    "#ff4d8d",
                    width=1.8,
                    style=Qt.PenStyle.DotLine,
                ),
                label=name,
                labelOpts={
                    "position": 0.08 + 0.07 * (index % 3),
                    "color": "#ffc1d8",
                    "rotateAxis": (1, 0),
                },
            )
            line.setZValue(15)
            radius_text = (
                f"{rms_radius:.6g} mm" if np.isfinite(rms_radius) else "Unavailable"
            )
            tooltip = (
                f"{name}\nZ = {z_mm:.6g} mm\nRMS radius = {radius_text}"
            )
            line.setToolTip(tooltip)
            self._register_ray_label(line.label)
            line.label.setToolTip(tooltip)
            self.plot.addItem(line)
            self.crossover_marker_items.append(line)

        if records:
            marker = pg.ScatterPlotItem(
                x=[float(item["z_mm"]) for item in records],
                y=[0.0] * len(records),
                symbol="d",
                size=9,
                pen=pg.mkPen("#fff1f6", width=1.5),
                brush=pg.mkBrush("#ff2f7d"),
                name="Crossover",
            )
            marker.setToolTip("Detected beam crossovers (see labelled lines)")
            self.plot.addItem(marker)
            self.crossover_marker_items.append(marker)

    def _draw_ray_diagram(
        self, result, quality: str, preserve_view: bool = False
    ) -> None:
        preserved_range = (
            self.plot.getViewBox().viewRange() if preserve_view else None
        )
        simulation = result.simulation
        self.plot.clear()
        self._set_ray_axis_label("left", "Projected displacement")
        self._style_ray_axes()
        self.component_marker_items = []
        self._component_labels = []
        self._ray_label_items = []
        self._ray_label_font_pt = None
        self.sample_marker_items = []
        self.aperture_marker_items = []
        self.aperture_optical_plane_items = []
        self.aperture_stop_segment_items = []
        self._aperture_span_records = []
        self._aperture_stops_by_key = {
            str(record["key"]): dict(record)
            for record in getattr(result, "aperture_stops", ())
        }
        self.deflector_pair_items = []
        self.crossover_marker_items = []
        self.column_wall_items = []
        self.stop_marker_items = []
        self.axial_cursor_item = None
        limits = self._simulation_x_limits()
        if limits is not None:
            self.axial_position.setRange(*limits)
        legend = self.plot.addLegend(offset=(10, 10))
        self._style_ray_legend(legend)

        bundles = [(simulation.incident, "Incident")]
        bundles.extend(
            (branch, branch.name) for branch in simulation.branches.values()
        )
        for branch, label in bundles:
            display_count = min(self.MAX_DISPLAY_RAYS, branch.x.shape[1])
            projected = self._project_transverse(branch.x, branch.y)
            z, transverse = self._bundle_lines(
                branch.z,
                projected,
                display_count,
                branch.blocked_z,
            )
            colour = (
                "#7dd3fc"
                if label == "Incident"
                else tuple(max(64, int(255 * value)) for value in branch.colour)
            )
            if z.size:
                self.plot.plot(
                    z,
                    transverse,
                    pen=pg.mkPen(colour, width=1.35),
                    name=label,
                    connect="finite",
                )

        self.plot.plot(
            [], [], pen=pg.mkPen("#ffb000", width=2.0), name="Aperture"
        )
        self.plot.plot(
            [], [], pen=pg.mkPen("#2cffad", width=2.0), name="Deflector U/L"
        )
        self.plot.plot(
            [], [], pen=pg.mkPen("#ffffff", width=2.6), name="Sample plane"
        )

        self._add_column_walls(result)
        self._add_stop_markers(simulation)
        self._add_component_markers(result.assembly)
        self._add_sample_marker(result)
        self._add_crossover_markers(result)
        self._add_axial_position_cursor()

        if preserved_range is not None:
            self.plot.disableAutoRange()
            (x_min, x_max), (y_min, y_max) = preserved_range
            self.plot.setXRange(x_min, x_max, padding=0.0)
            self.plot.setYRange(y_min, y_max, padding=0.0)
            self._update_component_label_visibility()
        else:
            self.plot.enableAutoRange()
            if self.auto_zoom.isChecked() and self._focused_part is not None:
                self._apply_component_zoom(self._focused_part)
            elif self._selected_z_mm is not None:
                self.jump_to_ray_position(
                    self._selected_z_mm, activate_tab=False
                )
            else:
                # Resolve the initial data bounds synchronously, then freeze
                # them. Aperture openings are view-dependent graphics, so an
                # always-live AutoRange creates a zoom/span feedback loop.
                self.plot.getViewBox().autoRange()
                self.plot.disableAutoRange()
                self._update_component_label_visibility()
        self._update_aperture_spans()
        self._update_scale_notice()
        if self._selected_z_mm is None:
            self.stop_detail.setText(
                f"Projection: {self._projection_axis_name()} = "
                "X cos(angle) + Y sin(angle) | "
                "click a stop marker for exact X/Y diagnostics"
            )
        crossover_count = len(self._all_crossovers(result))
        wall_stop_count = self._column_wall_stop_count(simulation)
        self.heading.setText(
            f"Electron ray paths — {quality} | "
            f"{self._projection_axis_name()} projection at "
            f"{self._format_angle(self._projection_angle_deg)}° | "
            f"{crossover_count} crossovers | "
            f"{wall_stop_count} column-wall stops"
        )

    def display_result(self, result, quality: str) -> None:
        preserve_ray_view = (
            self._last_result is not None
            and self._ray_geometry_signature(self._last_result)
            == self._ray_geometry_signature(result)
        )
        self._last_result = result
        self._last_quality = quality
        self._draw_ray_diagram(
            result,
            quality,
            preserve_view=preserve_ray_view,
        )
        self.physical_layout.display_result(result)
        self.magnetic_field.display_result(result)
        self.optical_transfer.display_result(result)
        self.energy_filter.display_result(result)
        self.transverse_beam.display_result(result)
        self.wave_imaging.display_result(getattr(result, "wave_imaging", None))
        if self._focused_part is not None:
            self.physical_layout.focus_component(self._focused_part)
            self.magnetic_field.focus_component(self._focused_part)
            self.transverse_beam.focus_component(self._focused_part)
