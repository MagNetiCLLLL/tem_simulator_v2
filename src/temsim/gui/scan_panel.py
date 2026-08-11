"""STEM scan/descan geometry, detector images and acquisition controls."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.component_keys import STEM_DETECTOR_KEYS
from temsim.physics.scan_geometry import (
    SHARED_RASTER_FIELDS,
    calibrate_scan_system,
)


class ScanControlView(QWidget):
    """Edit STEM scan drives and display geometry or detector images."""

    parameters_changed = Signal(str)
    error = Signal(str)
    playback_time_changed = Signal(float)
    playback_active_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("scanControlView")
        self._state = None
        self._result = None
        self._stem_frame = None
        self._stem_auto_range_pending = True
        self._updating = False
        self._playback_started_s = 0.0
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(33)
        self._playback_timer.timeout.connect(self._playback_tick)

        self.summary = QLabel(
            "Enable AC Scan or Descan to calculate a raster trajectory."
        )
        self.summary.setObjectName("scanGeometrySummary")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #0f172a; font-weight: 600;")
        scope = QLabel(
            "Scan is available in TEM and STEM with both microprobe and "
            "nanoprobe illumination. AC and Descan expose the same raster and "
            "foil controls. AC is coupled for zero first-order angle at the "
            "sample; Descan uses the opposite raster command and is coupled "
            "at the Selected Area Aperture image-reference station. Calculated "
            "first image/diffraction planes and the two physical aperture "
            "stations are classified independently as image, diffraction, or "
            "mixed for the current lens state. HAADF / DF / BF frames contain "
            "the detector fraction calculated at every probe position; "
            "Preview uses the geometric approximation."
        )
        scope.setObjectName("scanScopeNotice")
        scope.setWordWrap(True)
        scope.setStyleSheet("color: #475569;")

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        self.component_fov_labels = {}
        self.ac_controls = self._add_component_controls(
            controls_layout,
            title="AC Scan Foils",
            prefix="ac",
        )
        self.descan_controls = self._add_component_controls(
            controls_layout,
            title="Descan Foils",
            prefix="descan",
        )
        self.wave_scan_enabled = QCheckBox(
            "Use wave / multislice detector signal in High accuracy"
        )
        self.wave_scan_enabled.setObjectName("stemWaveScanEnabled")
        self.wave_scan_enabled.setToolTip(
            "Preview always uses the fast geometric detector approximation; "
            "High accuracy uses the specimen wave model when this is enabled."
        )
        self.wave_scan_enabled.toggled.connect(
            self._wave_scan_model_changed
        )
        controls_layout.addWidget(self.wave_scan_enabled)
        statistics = QGroupBox("STEM image statistics")
        statistics_form = QFormLayout(statistics)
        self.poisson_enabled = QCheckBox("Generate seeded Poisson counts")
        self.poisson_enabled.setObjectName("stemPoissonEnabled")
        self.poisson_seed = QSpinBox()
        self.poisson_seed.setObjectName("stemPoissonSeed")
        self.poisson_seed.setRange(0, 2_147_483_647)
        self.poisson_seed.setKeyboardTracking(False)
        statistics_form.addRow(self.poisson_enabled)
        statistics_form.addRow("Poisson seed", self.poisson_seed)
        self.poisson_enabled.toggled.connect(self._poisson_changed)
        self.poisson_seed.valueChanged.connect(self._poisson_seed_changed)
        controls_layout.addWidget(statistics)
        pivot_help = QLabel(
            "Both foil pairs use one signed upper gain and a derived lower-foil "
            "2 x 2 coupling. The raster clock, pixel count, pixel size and FOV "
            "are shared. Active magnetic lenses can rotate and remap both pairs, "
            "so every calculation re-solves the two couplings."
        )
        pivot_help.setObjectName("scanPivotHelp")
        pivot_help.setWordWrap(True)
        pivot_help.setStyleSheet("color: #475569;")
        controls_layout.addWidget(pivot_help)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("scanControlsScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(280)
        controls_scroll.setWidget(controls_widget)

        self.sample_plot = self._create_plot(
            "Sample scan position",
            "scanSamplePlot",
        )
        self.plane_selector = QComboBox()
        self.plane_selector.setObjectName("scanObservationPlane")
        self.plane_selector.setToolTip(
            "Choose a downstream recording plane for the combined "
            "AC Scan + Descan trajectory"
        )
        self.plane_selector.currentIndexChanged.connect(
            self._redraw_downstream
        )
        downstream_header = QHBoxLayout()
        downstream_header.addWidget(QLabel("Downstream recording plane"))
        downstream_header.addWidget(self.plane_selector, 1)
        self.downstream_plot = self._create_plot(
            "Combined downstream position",
            "scanDownstreamPlot",
        )

        sample_panel = QWidget()
        sample_layout = QVBoxLayout(sample_panel)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(QLabel("Sample plane"))
        sample_layout.addWidget(self.sample_plot, 1)
        downstream_panel = QWidget()
        downstream_layout = QVBoxLayout(downstream_panel)
        downstream_layout.setContentsMargins(0, 0, 0, 0)
        downstream_layout.addLayout(downstream_header)
        downstream_layout.addWidget(self.downstream_plot, 1)

        plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        plot_splitter.setObjectName("scanPlotSplitter")
        plot_splitter.addWidget(sample_panel)
        plot_splitter.addWidget(downstream_panel)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 1)

        geometry_page = QWidget()
        geometry_layout = QVBoxLayout(geometry_page)
        geometry_layout.setContentsMargins(0, 0, 0, 0)
        geometry_layout.addWidget(plot_splitter, 1)

        detector_page = QWidget()
        detector_layout = QVBoxLayout(detector_page)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        self.detector_playback_summary = QLabel(
            "Enable AC Scan to calculate one HAADF / DF / BF frame."
        )
        self.detector_playback_summary.setObjectName(
            "stemScanPlaybackSummary"
        )
        self.detector_playback_summary.setWordWrap(True)
        detector_layout.addWidget(self.detector_playback_summary)
        self.image_model_notice = QLabel(
            "Run a STEM calculation to identify whether the images are a "
            "geometry preview, virtual specimen signal, or CIF multislice signal."
        )
        self.image_model_notice.setObjectName("stemImageModelNotice")
        self.image_model_notice.setWordWrap(True)
        self.image_model_notice.setStyleSheet(
            "color: #92400e; background: #fffbeb; border: 1px solid #f59e0b; "
            "padding: 6px;"
        )
        detector_layout.addWidget(self.image_model_notice)
        interpretation = QLabel(
            "BF records transmitted/low-angle electrons; DF records its configured "
            "scattered-angle band; HAADF records the configured high-angle band. "
            "The exact angular ranges are listed above each image."
        )
        interpretation.setObjectName("stemImageInterpretation")
        interpretation.setWordWrap(True)
        interpretation.setStyleSheet("color: #475569;")
        detector_layout.addWidget(interpretation)
        detector_images = QHBoxLayout()
        self.detector_image_views = {}
        self.detector_image_items = {}
        self.detector_geometry_labels = {}
        for key in STEM_DETECTOR_KEYS:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(key.upper())
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            geometry_label = QLabel(
                "Detector geometry and collection angle available after "
                "calculation"
            )
            geometry_label.setObjectName(f"{key}StemDetectorGeometry")
            geometry_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            geometry_label.setWordWrap(True)
            geometry_label.setStyleSheet("color: #64748b; font-size: 9pt;")
            view = pg.PlotWidget(background="#050816")
            view.setObjectName(f"{key}StemScanImage")
            view.setMinimumSize(100, 140)
            view.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Expanding,
            )
            view.showGrid(x=True, y=True, alpha=0.15)
            view.setLabel("bottom", "scan X", units="um")
            view.setLabel("left", "scan Y", units="um")
            view.setToolTip(
                "Laboratory X/Y use the same physical scale. Mouse-wheel zoom "
                "and drag remain available; right-click can restore auto range."
            )
            view_box = view.getViewBox()
            view_box.setAspectLocked(True, ratio=1.0)
            view_box.setMouseEnabled(x=True, y=True)
            image_item = pg.ImageItem()
            view.addItem(image_item)
            panel_layout.addWidget(label)
            panel_layout.addWidget(geometry_label)
            panel_layout.addWidget(view, 1)
            detector_images.addWidget(panel, 1)
            self.detector_image_views[key] = view
            self.detector_image_items[key] = image_item
            self.detector_geometry_labels[key] = geometry_label
        detector_layout.addLayout(detector_images, 1)

        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("stemResultTabs")
        self.result_tabs.addTab(geometry_page, "Geometry")
        self.result_tabs.addTab(detector_page, "Images")

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setObjectName("scanContentSplitter")
        content_splitter.addWidget(controls_scroll)
        content_splitter.addWidget(self.result_tabs)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([390, 900])

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(scope)
        layout.addWidget(content_splitter, 1)

    def _add_component_controls(
        self,
        parent_layout: QVBoxLayout,
        *,
        title: str,
        prefix: str,
    ) -> dict[str, QWidget]:
        group = QGroupBox(title)
        group.setObjectName(f"{prefix}ScanGroup")
        form = QFormLayout(group)
        widgets: dict[str, QWidget] = {}

        enabled = QCheckBox("Enable raster drive")
        enabled.setObjectName(f"{prefix}ScanEnabled")
        enabled.setToolTip(
            "Enable this physical foil pair. Descan follows the shared raster "
            "with the command matrix opposite to AC Scan."
        )
        form.addRow(enabled)
        widgets["scan_enabled"] = enabled

        def add_float(
            field: str,
            label: str,
            *,
            minimum: float,
            maximum: float,
            decimals: int = 6,
            step: float = 0.01,
            suffix: str = "",
        ) -> None:
            control = QDoubleSpinBox()
            control.setObjectName(
                f"{prefix}{''.join(part.title() for part in field.split('_'))}"
            )
            control.setDecimals(decimals)
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setKeyboardTracking(False)
            control.setSuffix(suffix)
            form.addRow(label, control)
            widgets[field] = control

        add_float(
            "scan_pixel_size_nm",
            "Pixel size",
            minimum=1.0e-3,
            maximum=1.0e6,
            decimals=6,
            step=0.1,
            suffix=" nm",
        )
        widgets["scan_pixel_size_nm"].setToolTip(
            "Shared specimen-plane square pixel pitch. Supported input range: "
            "0.001 nm (1 pm) to 1 mm. The active column optics derive the AC "
            "command and its opposite Descan command."
        )
        fov_x = QLabel()
        fov_x.setObjectName(f"{prefix}ScanFieldOfViewX")
        fov_y = QLabel()
        fov_y.setObjectName(f"{prefix}ScanFieldOfViewY")
        for label in (fov_x, fov_y):
            label.setStyleSheet("color: #0f766e; font-weight: 600;")
            label.setToolTip(
                "Derived field of view = pixel count x pixel size. "
                "Reported width includes the full pixel footprints."
            )
        form.addRow("Field of view X", fov_x)
        form.addRow("Field of view Y", fov_y)
        self.component_fov_labels[prefix] = (fov_x, fov_y)
        if prefix == "ac":
            self.ac_fov_x, self.ac_fov_y = fov_x, fov_y
        else:
            self.descan_fov_x, self.descan_fov_y = fov_x, fov_y
        add_float(
            "scan_frame_period_s",
            "Frame period",
            minimum=1.0e-6,
            maximum=1.0e6,
            step=0.1,
            suffix=" s",
        )

        for field, label in (
            ("scan_pixels_x", "Pixels X"),
            ("scan_lines", "Lines Y"),
        ):
            control = QSpinBox()
            control.setObjectName(
                f"{prefix}{''.join(part.title() for part in field.split('_'))}"
            )
            control.setRange(2, 4096)
            control.setSingleStep(1)
            control.setKeyboardTracking(False)
            form.addRow(label, control)
            widgets[field] = control

        add_float(
            "upper_coil_gain",
            "Upper foil / pair gain",
            minimum=-1000.0,
            maximum=1000.0,
            step=0.05,
        )
        add_float(
            "lower_coil_gain",
            "Lower foil gain (derived coupling)",
            minimum=-1000.0,
            maximum=1000.0,
            step=0.05,
        )
        widgets["lower_coil_gain"].setEnabled(False)

        enabled.toggled.connect(
            lambda value: self._control_changed(
                prefix, "scan_enabled", bool(value)
            )
        )
        for field, control in widgets.items():
            if field == "scan_enabled":
                continue
            control.valueChanged.connect(
                lambda value, component_prefix=prefix, name=field: (
                    self._control_changed(component_prefix, name, value)
                )
            )
        parent_layout.addWidget(group)
        return widgets

    @staticmethod
    def _create_plot(title: str, object_name: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#050816")
        plot.setObjectName(object_name)
        plot.setTitle(title, color="#e2e8f0", size="11pt")
        plot.setLabel("bottom", "X", units="um")
        plot.setLabel("left", "Y", units="um")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.getViewBox().setAspectLocked(True)
        return plot

    @staticmethod
    def _component_for_prefix(state, prefix: str):
        return (
            state.ac_deflector
            if prefix == "ac"
            else state.descan_deflector
        )

    def set_state(self, state) -> None:
        """Bind all controls to the current live microscope state."""

        if state is not self._state:
            self._playback_timer.stop()
            self._stem_frame = None
            self._stem_auto_range_pending = True
            for item in self.detector_image_items.values():
                item.clear()
            self._update_image_model_notice(None)
        self._state = state
        self._updating = True
        try:
            self._sync_controls(
                state.ac_deflector,
                self.ac_controls,
            )
            self._sync_controls(
                state.descan_deflector,
                self.descan_controls,
            )
            self._update_fov_labels()
            self.wave_scan_enabled.setChecked(
                bool(getattr(state.sample, "stem_wave_enabled", False))
            )
            self.poisson_enabled.setChecked(
                bool(getattr(state.sample, "stem_poisson_enabled", False))
            )
            self.poisson_seed.setValue(
                int(getattr(state.sample, "stem_poisson_seed", 0))
            )
            self._update_detector_geometry_labels(None)
        finally:
            self._updating = False

    def _wave_scan_model_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_wave_enabled = bool(enabled)
        self.parameters_changed.emit("sample.stem_wave_enabled")

    def _poisson_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_poisson_enabled = bool(enabled)
        self.parameters_changed.emit("sample.stem_poisson_enabled")

    def _poisson_seed_changed(self, seed: int) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_poisson_seed = int(seed)
        self.parameters_changed.emit("sample.stem_poisson_seed")

    @staticmethod
    def _sync_controls(component, widgets: dict[str, QWidget]) -> None:
        maximum = abs(float(component.maximum_kick_mrad))
        for field, widget in widgets.items():
            value = getattr(component, field)
            if field.startswith("scan_amplitude_"):
                widget.setRange(-maximum, maximum)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            else:
                widget.setValue(float(value))

    @staticmethod
    def _format_length_nm(value_nm: float) -> str:
        value_nm = float(value_nm)
        magnitude = abs(value_nm)
        if magnitude < 1.0:
            return f"{value_nm * 1.0e3:.6g} pm"
        if magnitude < 1.0e3:
            return f"{value_nm:.6g} nm"
        if magnitude < 1.0e6:
            return f"{value_nm * 1.0e-3:.6g} um"
        return f"{value_nm * 1.0e-6:.6g} mm"

    def _update_fov_labels(self) -> None:
        if self._state is None:
            return
        for prefix, labels in self.component_fov_labels.items():
            component = self._component_for_prefix(self._state, prefix)
            labels[0].setText(
                self._format_length_nm(component.scan_field_of_view_x_nm)
            )
            labels[1].setText(
                self._format_length_nm(component.scan_field_of_view_y_nm)
            )

    def _control_changed(self, prefix: str, field: str, value) -> None:
        if self._updating or self._state is None:
            return
        component = self._component_for_prefix(self._state, prefix)
        components = (
            self._state.ac_deflector,
            self._state.descan_deflector,
        )
        snapshots = tuple(dict(item.__dict__) for item in components)
        try:
            old_value = getattr(component, field)
            converted = (
                bool(value)
                if isinstance(old_value, bool)
                else int(value)
                if isinstance(old_value, int)
                else float(value)
            )
            setattr(component, field, converted)
            if field in SHARED_RASTER_FIELDS:
                other_prefix = "descan" if prefix == "ac" else "ac"
                other = self._component_for_prefix(
                    self._state,
                    other_prefix,
                )
                setattr(other, field, converted)
            if prefix == "ac" and field == "scan_enabled" and converted:
                component.wobble_enabled = False
            calibrate_scan_system(self._state)
        except Exception as exc:
            for item, snapshot in zip(components, snapshots):
                for name, original in snapshot.items():
                    object.__setattr__(item, name, original)
            self._updating = True
            try:
                self._sync_controls(
                    self._state.ac_deflector,
                    self.ac_controls,
                )
                self._sync_controls(
                    self._state.descan_deflector,
                    self.descan_controls,
                )
                self._update_fov_labels()
            finally:
                self._updating = False
            self.error.emit(str(exc))
            return
        self._updating = True
        try:
            self._sync_controls(
                self._state.ac_deflector,
                self.ac_controls,
            )
            self._sync_controls(
                self._state.descan_deflector,
                self.descan_controls,
            )
            self._update_fov_labels()
        finally:
            self._updating = False
        if prefix == "ac" and field == "scan_enabled":
            if converted:
                self._playback_timer.stop()
                self.detector_playback_summary.setText(
                    "Calculating one HAADF / DF / BF detector-signal frame..."
                )
            else:
                self._set_playback_active(False)
        self.parameters_changed.emit(f"{prefix}.{field}")

    def display_result(self, result, stem_frame=None) -> None:
        """Display scan geometry and one reusable detector-signal frame."""

        self._result = result
        if stem_frame is not None:
            self._set_stem_frame(stem_frame)
        live_ac = (
            getattr(self._state, "ac_deflector", None)
            if self._state is not None
            else None
        )
        scanning = bool(
            result is not None
            and result.ac_enabled
            and live_ac is not None
            and live_ac.enabled
            and live_ac.scan_enabled
        )
        self._set_playback_active(scanning and self._stem_frame is not None)
        previous_key = self.plane_selector.currentData()
        self.plane_selector.blockSignals(True)
        self.plane_selector.clear()
        if result is not None:
            for key, name in result.plane_names.items():
                role = result.plane_roles.get(key, "unclassified")
                self.plane_selector.addItem(f"{name} [{role}]", key)
        previous_index = self.plane_selector.findData(previous_key)
        target_index = (
            self.plane_selector.findData(result.descan_target_key)
            if result is not None and result.descan_target_key is not None
            else -1
        )
        self.plane_selector.setCurrentIndex(
            previous_index
            if previous_index >= 0
            else target_index
            if target_index >= 0
            else 0
        )
        self.plane_selector.blockSignals(False)

        if result is None:
            self.sample_plot.clear()
            self.downstream_plot.clear()
            self.summary.setText(
                "AC Scan and Descan are both off. Enable either raster drive "
                "to calculate first-order scan geometry."
            )
            return

        self._draw_trace(
            self.sample_plot,
            result.sample_x_um,
            result.sample_y_um,
            colour="#22d3ee",
        )
        self._redraw_downstream()
        sample_span_x = float(np.ptp(result.sample_x_um))
        sample_span_y = float(np.ptp(result.sample_y_um))
        preview_y, preview_x = result.times_s.shape
        ac_pivot = self._format_pivot(result.ac_drift_pivot_z_mm)
        descan_pivot = self._format_pivot(result.descan_drift_pivot_z_mm)
        coupling_text = ""
        if result.ac_lower_from_upper is not None:
            coupling = np.asarray(result.ac_lower_from_upper, dtype=float)
            residual = float(result.ac_angular_residual or 0.0)
            coupling_text = (
                " | AC pure-shift lower <- upper "
                f"[[{coupling[0, 0]:.5g}, {coupling[0, 1]:.5g}], "
                f"[{coupling[1, 0]:.5g}, {coupling[1, 1]:.5g}]], "
                f"angular residual {residual:.3g}"
            )
        descan_text = ""
        if result.descan_target_z_mm is not None:
            descan_coupling = np.asarray(
                result.descan_lower_from_upper,
                dtype=float,
            )
            descan_text = (
                f" | Descan target {result.descan_target_name} at "
                f"Z={result.descan_target_z_mm:.6g} mm: "
                f"{result.descan_target_plane_kind}; opposite command, "
                "lower <- upper "
                f"[[{descan_coupling[0, 0]:.5g}, "
                f"{descan_coupling[0, 1]:.5g}], "
                f"[{descan_coupling[1, 0]:.5g}, "
                f"{descan_coupling[1, 1]:.5g}]], response residual "
                f"{float(result.descan_compensation_residual or 0.0):.3g}, "
                "image conjugacy ||J_diff|| "
                f"{float(result.descan_target_conjugacy_residual_m_per_rad or 0.0):.3g} m/rad"
            )
        if result.pixel_size_nm is None:
            scale_text = "AC pixel scale inactive"
        else:
            scale_text = (
                "pixel size "
                f"{self._format_length_nm(result.pixel_size_nm)} | FOV "
                f"{self._format_length_nm(result.field_of_view_x_nm)} x "
                f"{self._format_length_nm(result.field_of_view_y_nm)}"
            )
        symmetry_text = ""
        if result.scan_pair_symmetry_error_mm is not None:
            symmetry_text = (
                " | foil-pair centres: AC "
                f"{result.ac_distance_above_sample_mm:.6g} mm above, "
                f"Descan {result.descan_distance_below_sample_mm:.6g} mm "
                "below sample; mirror error "
                f"{result.scan_pair_symmetry_error_mm:.3g} mm"
            )
        self.summary.setText(
            f"AC Scan: {'ON' if result.ac_enabled else 'OFF'} | "
            f"Descan: {'ON' if result.descan_enabled else 'OFF'} | "
            f"requested raster {result.requested_pixels_x} x "
            f"{result.requested_pixels_y}; preview {preview_x} x "
            f"{preview_y} | {scale_text} | "
            f"sample-centre span {sample_span_x:.6g} x "
            f"{sample_span_y:.6g} um | drift-only pivot: "
            f"AC {ac_pivot}, Descan {descan_pivot}"
            f"{symmetry_text}{coupling_text}{descan_text}"
        )

    def _set_stem_frame(self, frame) -> None:
        shape = np.asarray(frame.scan_x_um, dtype=float).shape
        if len(shape) != 2 or not all(value > 0 for value in shape):
            raise ValueError("STEM scan frame must use a non-empty 2-D raster.")
        for key, values in frame.fractions.items():
            array = np.asarray(values, dtype=float)
            if array.shape != shape:
                raise ValueError(
                    f"{key}: detector image does not match the scan raster."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{key}: detector image must be finite.")
        self._stem_frame = frame
        self._update_detector_geometry_labels(frame)
        self._update_image_model_notice(frame)
        self._stem_auto_range_pending = True
        self._render_stem_rows(shape[0])

    def _update_image_model_notice(self, frame) -> None:
        if frame is None:
            self.image_model_notice.setText(
                "No STEM frame is loaded. Run Preview for scan/detector geometry "
                "or High accuracy with wave/multislice enabled for specimen-dependent contrast."
            )
            return
        metrics = getattr(frame, "metrics", None) or {}
        model = str(metrics.get("model", "unknown"))
        pixel_nm = metrics.get("scan_pixel_size_nm")
        fov_x_nm = metrics.get("scan_field_of_view_x_nm")
        fov_y_nm = metrics.get("scan_field_of_view_y_nm")
        scale = ""
        if pixel_nm is not None and fov_x_nm is not None and fov_y_nm is not None:
            scale = (
                f" Sampling: {self._format_length_nm(float(pixel_nm))} per pixel; "
                f"FOV {self._format_length_nm(float(fov_x_nm))} x "
                f"{self._format_length_nm(float(fov_y_nm))}."
            )
        scale += self._sample_scale_warning(
            pixel_nm=pixel_nm,
            fov_x_nm=fov_x_nm,
            fov_y_nm=fov_y_nm,
        )
        if model == "geometric_detector_interception":
            cif_path = str(
                getattr(getattr(self._state, "sample", None), "cif_path", "")
            ).strip()
            cif_note = (
                f" The selected {Path(cif_path).name} structure is not used by this preview."
                if cif_path
                else ""
            )
            text = (
                "Preview geometry only — not a specimen STEM image. The polygons "
                "and sharp wedges are detector-clipping boundaries produced by "
                "scan/descan ray interception; they are not atoms or diffraction "
                f"contrast.{cif_note} Enable wave/multislice and run High accuracy "
                f"to calculate CIF-dependent elastic contrast.{scale}"
            )
            colour = (
                "color: #92400e; background: #fffbeb; border: 1px solid #f59e0b;"
            )
        elif model in {"multislice_angle_resolved", "thin_phase_angle_resolved"}:
            potential = str(metrics.get("specimen_potential_model", "specimen potential"))
            text = (
                f"Specimen-dependent {model.replace('_', ' ')} image using "
                f"{potential}. Detector values are fractions of emitted source "
                f"current integrated over the physical detector masks.{scale}"
            )
            colour = (
                "color: #166534; background: #f0fdf4; border: 1px solid #22c55e;"
            )
        elif model == "finite_virtual_absolute_probability":
            text = (
                "Virtual-sample image from the configured absolute interaction "
                f"probabilities and finite density regions.{scale}"
            )
            colour = (
                "color: #1e40af; background: #eff6ff; border: 1px solid #60a5fa;"
            )
        else:
            limitation = str(metrics.get("model_limitation", "")).strip()
            text = f"Image model: {model}. {limitation}{scale}".strip()
            colour = (
                "color: #334155; background: #f8fafc; border: 1px solid #94a3b8;"
            )
        self.image_model_notice.setText(text)
        self.image_model_notice.setStyleSheet(f"{colour} padding: 6px;")

    def _sample_scale_warning(self, *, pixel_nm, fov_x_nm, fov_y_nm) -> str:
        sample = getattr(self._state, "sample", None)
        if sample is None:
            return ""
        warnings = []
        if fov_x_nm is not None and fov_y_nm is not None:
            size_x = float(getattr(sample, "size_x_nm", float("inf")))
            size_y = float(getattr(sample, "size_y_nm", float("inf")))
            if float(fov_x_nm) > size_x or float(fov_y_nm) > size_y:
                warnings.append(
                    "the scan FOV extends outside the finite sample, so those pixels are vacuum"
                )
        cif_path = str(getattr(sample, "cif_path", "")).strip()
        if cif_path and pixel_nm is not None:
            try:
                from ase.io import read

                atoms = read(Path(cif_path).expanduser())
                distances = np.asarray(
                    atoms.get_all_distances(mic=True),
                    dtype=float,
                )
                positive = distances[distances > 1.0e-8]
                nearest_nm = float(np.min(positive)) * 0.1
                if float(pixel_nm) > 0.5 * nearest_nm:
                    warnings.append(
                        f"pixel pitch {self._format_length_nm(float(pixel_nm))} "
                        f"is coarser than half the shortest CIF atom spacing "
                        f"({nearest_nm:.6g} nm), so atomic columns are undersampled"
                    )
            except Exception:
                # CIF validity is reported by the Sample page/calculation.  A
                # sampling hint must never make an otherwise valid frame fail.
                pass
        return (
            " Sampling warning: " + "; ".join(warnings) + "."
            if warnings
            else ""
        )

    def _update_detector_geometry_labels(self, frame) -> None:
        detectors = {
            str(detector.key): detector
            for detector in getattr(self._state, "stem_detectors", ())
        }
        signals = getattr(frame, "detector_signals", {}) if frame else {}
        for key, label in self.detector_geometry_labels.items():
            detector = detectors.get(key)
            signal = signals.get(key)
            if detector is None:
                label.setText("Detector not installed")
                continue
            geometry = (
                f"Z {float(detector.z_mm):.6g} mm | "
                f"OD {float(detector.outer_width_mm):.6g} mm | "
                f"ID {float(detector.inner_diameter_mm):.6g} mm"
            )
            angle = getattr(signal, "collection_angle", None)
            if angle is None:
                angle_text = "collection angle pending"
            elif not (
                np.isfinite(angle.inner_mrad)
                and np.isfinite(angle.outer_mrad)
            ):
                angle_text = "collection angle unavailable (singular transfer)"
            elif angle.anisotropic:
                angle_text = (
                    f"collection {angle.inner_mrad:.6g} to "
                    f"{angle.outer_mrad:.6g} mrad; anisotropic ranges "
                    f"ID {angle.inner_range_mrad[0]:.6g} to "
                    f"{angle.inner_range_mrad[1]:.6g}, OD "
                    f"{angle.outer_range_mrad[0]:.6g} to "
                    f"{angle.outer_range_mrad[1]:.6g} mrad"
                )
            else:
                angle_text = (
                    f"collection {angle.inner_mrad:.6g} to "
                    f"{angle.outer_mrad:.6g} mrad"
                )
            label.setText(f"{geometry}\n{angle_text}")
            label.setToolTip(
                "Detector position and active inner/outer dimensions come "
                "from the selected instrument TOML. Collection angle is "
                "derived from the active sample-to-detector first-order "
                "transfer, so lens rotation and anisotropy are retained."
            )

    def _frame_period_s(self) -> float:
        metrics = getattr(self._stem_frame, "metrics", None) or {}
        value = metrics.get("scan_frame_period_s")
        if value is None and self._state is not None:
            value = self._state.ac_deflector.scan_frame_period_s
        return max(float(value or 1.0), 1.0e-6)

    def _set_playback_active(self, active: bool) -> None:
        if active and self._stem_frame is not None:
            self._playback_started_s = perf_counter()
            if not self._playback_timer.isActive():
                self._playback_timer.start()
            self.playback_active_changed.emit(True)
            self._playback_tick()
            return
        self._playback_timer.stop()
        self.playback_active_changed.emit(False)
        if self._stem_frame is not None:
            rows = np.asarray(self._stem_frame.scan_x_um).shape[0]
            self._render_stem_rows(rows)
            self.detector_playback_summary.setText(
                self._stem_frame_summary("Stopped; last frame retained")
            )
        else:
            self.detector_playback_summary.setText(
                "Enable AC Scan to calculate one HAADF / DF / BF frame."
            )

    def _playback_tick(self) -> None:
        if self._stem_frame is None:
            self._playback_timer.stop()
            return
        rows = np.asarray(self._stem_frame.scan_x_um).shape[0]
        period_s = self._frame_period_s()
        elapsed_s = max(perf_counter() - self._playback_started_s, 0.0)
        frame_time_s = elapsed_s % period_s
        if period_s <= 2.0 * self._playback_timer.interval() * 1.0e-3:
            completed_rows = rows
            frame_number = int(elapsed_s / period_s) + 1
        else:
            frame_number = int(elapsed_s / period_s) + 1
            phase = frame_time_s / period_s
            completed_rows = min(rows, max(1, int(phase * rows) + 1))
        self.playback_time_changed.emit(frame_time_s)
        self._render_stem_rows(completed_rows)
        self.detector_playback_summary.setText(
            self._stem_frame_summary(
                f"Scanning continuously; frame {frame_number}, "
                f"line {completed_rows}/{rows}"
            )
        )

    def _stem_frame_summary(self, playback: str) -> str:
        frame = self._stem_frame
        metrics = getattr(frame, "metrics", None) or {}
        model = str(metrics.get("model", "detector signal"))
        values = []
        for key in STEM_DETECTOR_KEYS:
            signal = frame.detector_signals.get(key)
            if signal is not None:
                values.append(
                    f"{key.upper()} mean {float(signal.fraction):.5g} "
                    f"({float(signal.current_pa):.5g} pA)"
                )
        detail = " | ".join(values) if values else "no inserted detector"
        return f"{playback} | {model} | {detail}"

    @staticmethod
    def _coordinate_edges(values, count: int, fallback_step_um: float):
        values = np.asarray(values, dtype=float)
        lower = float(np.min(values))
        upper = float(np.max(values))
        if count > 1 and upper > lower:
            step = (upper - lower) / float(count - 1)
        else:
            step = max(float(fallback_step_um), 1.0e-12)
        return lower - 0.5 * step, upper + 0.5 * step

    def _stem_image_rect(self) -> QRectF:
        frame = self._stem_frame
        scan_x = np.asarray(frame.scan_x_um, dtype=float)
        scan_y = np.asarray(frame.scan_y_um, dtype=float)
        rows, columns = scan_x.shape
        metrics = getattr(frame, "metrics", None) or {}
        fallback_step_um = float(metrics.get("scan_pixel_size_nm", 1.0)) * 1.0e-3
        x0, x1 = self._coordinate_edges(scan_x, columns, fallback_step_um)
        y0, y1 = self._coordinate_edges(scan_y, rows, fallback_step_um)
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def _render_stem_rows(self, completed_rows: int) -> None:
        if self._stem_frame is None:
            return
        auto_range = self._stem_auto_range_pending
        image_rect = self._stem_image_rect()
        for key, view in self.detector_image_views.items():
            image_item = self.detector_image_items[key]
            values = self._stem_frame.fractions.get(key)
            if values is None:
                image_item.clear()
                continue
            full = np.asarray(values, dtype=float)
            shown = full.copy()
            shown[max(0, int(completed_rows)):, :] = np.nan
            finite = full[np.isfinite(full)]
            if finite.size:
                low = float(np.min(finite))
                high = float(np.max(finite))
                if high <= low:
                    high = low + max(abs(low) * 1.0e-6, 1.0e-12)
                levels = (low, high)
            else:
                levels = (0.0, 1.0)
            image_item.setImage(
                shown.T,
                autoLevels=False,
                levels=levels,
            )
            image_item.setRect(image_rect)
            if auto_range:
                view.getViewBox().autoRange()
        self._stem_auto_range_pending = False

    @staticmethod
    def _format_pivot(value: float | None) -> str:
        if value is None:
            return "none (zero net angle)"
        return f"Z={float(value):.6g} mm"

    @staticmethod
    def _draw_trace(
        plot: pg.PlotWidget,
        x_values,
        y_values,
        *,
        colour: str,
    ) -> None:
        x = np.asarray(x_values, dtype=float).ravel()
        y = np.asarray(y_values, dtype=float).ravel()
        plot.clear()
        plot.addLine(x=0.0, pen=pg.mkPen("#334155", width=1))
        plot.addLine(y=0.0, pen=pg.mkPen("#334155", width=1))
        if x.size == 0:
            return
        symbol = "o" if x.size <= 1024 else None
        plot.plot(
            x,
            y,
            pen=pg.mkPen(colour, width=1.2),
            symbol=symbol,
            symbolSize=3,
            symbolBrush=colour,
            symbolPen=None,
        )
        span = max(float(np.ptp(x)), float(np.ptp(y)))
        if span <= 1.0e-12:
            centre_x = float(x[0])
            centre_y = float(y[0])
            plot.setXRange(centre_x - 1.0, centre_x + 1.0, padding=0.0)
            plot.setYRange(centre_y - 1.0, centre_y + 1.0, padding=0.0)
        else:
            plot.enableAutoRange()

    def _redraw_downstream(self, _index: int = -1) -> None:
        result = self._result
        key = self.plane_selector.currentData()
        if result is None or key not in result.plane_positions_um:
            self.downstream_plot.clear()
            return
        x_values, y_values = result.plane_positions_um[str(key)]
        self._draw_trace(
            self.downstream_plot,
            x_values,
            y_values,
            colour="#f472b6",
        )
