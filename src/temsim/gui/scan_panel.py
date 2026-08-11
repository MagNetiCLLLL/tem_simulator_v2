"""Independent AC scan and descan controls with geometric previews."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
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


class ScanControlView(QWidget):
    """Edit independent scan drives and display first-order trajectories."""

    parameters_changed = Signal(str)
    error = Signal(str)

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
            "nanoprobe illumination. AC upper/lower foils are automatically "
            "coupled for zero first-order angle at the sample. HAADF / DF / "
            "BF frames contain the detector fraction calculated at every "
            "probe position; Preview uses the geometric approximation."
        )
        scope.setObjectName("scanScopeNotice")
        scope.setWordWrap(True)
        scope.setStyleSheet("color: #475569;")

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(6, 6, 6, 6)
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
        pivot_help = QLabel(
            "Upper/lower gain is signed. Their ratio controls the double-coil "
            "pivot in the drift approximation; their common scale controls "
            "scan magnitude. Magnetic lenses can rotate and remap this "
            "relationship, so the plots use the active column optics."
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
        detector_images = QHBoxLayout()
        self.detector_image_views = {}
        self.detector_image_items = {}
        for key in STEM_DETECTOR_KEYS:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(key.upper())
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view = pg.PlotWidget(background="#050816")
            view.setObjectName(f"{key}StemScanImage")
            view.setMinimumSize(100, 140)
            view.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Expanding,
            )
            view.showGrid(x=True, y=True, alpha=0.15)
            view.setLabel("bottom", "scan X", units="pixel")
            view.setLabel("left", "scan Y", units="line")
            image_item = pg.ImageItem()
            view.addItem(image_item)
            panel_layout.addWidget(label)
            panel_layout.addWidget(view, 1)
            detector_images.addWidget(panel, 1)
            self.detector_image_views[key] = view
            self.detector_image_items[key] = image_item
        detector_layout.addLayout(detector_images, 1)

        result_tabs = QTabWidget()
        result_tabs.setObjectName("scanResultTabs")
        result_tabs.addTab(geometry_page, "Geometry")
        result_tabs.addTab(detector_page, "HAADF / DF / BF Images")

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setObjectName("scanContentSplitter")
        content_splitter.addWidget(controls_scroll)
        content_splitter.addWidget(result_tabs)
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
            "Enable this raster independently of the other deflector pair"
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
            "scan_amplitude_x_mrad",
            "X half-amplitude",
            minimum=-100.0,
            maximum=100.0,
            suffix=" mrad",
        )
        add_float(
            "scan_amplitude_y_mrad",
            "Y half-amplitude",
            minimum=-100.0,
            maximum=100.0,
            suffix=" mrad",
        )
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
            "Upper foil / pair gain" if prefix == "ac" else "Upper foil gain",
            minimum=-1000.0,
            maximum=1000.0,
            step=0.05,
        )
        add_float(
            "lower_coil_gain",
            (
                "Lower foil gain (pure-shift coupled)"
                if prefix == "ac"
                else "Lower foil gain"
            ),
            minimum=-1000.0,
            maximum=1000.0,
            step=0.05,
        )
        if prefix == "ac":
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
            self.wave_scan_enabled.setChecked(
                bool(getattr(state.sample, "stem_wave_enabled", False))
            )
        finally:
            self._updating = False

    def _wave_scan_model_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_wave_enabled = bool(enabled)
        self.parameters_changed.emit("sample.stem_wave_enabled")

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

    def _control_changed(self, prefix: str, field: str, value) -> None:
        if self._updating or self._state is None:
            return
        component = self._component_for_prefix(self._state, prefix)
        snapshot = {
            name: getattr(component, name)
            for name in (
                "scan_enabled",
                "scan_amplitude_x_mrad",
                "scan_amplitude_y_mrad",
                "scan_frame_period_s",
                "scan_pixels_x",
                "scan_lines",
                "upper_coil_gain",
                "lower_coil_gain",
            )
        }
        if prefix == "ac":
            snapshot["wobble_enabled"] = component.wobble_enabled
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
            if prefix == "ac" and field == "scan_enabled" and converted:
                component.wobble_enabled = False
            component.validate()
        except Exception as exc:
            for name, original in snapshot.items():
                setattr(component, name, original)
            self._updating = True
            try:
                controls = (
                    self.ac_controls if prefix == "ac"
                    else self.descan_controls
                )
                self._sync_controls(component, controls)
            finally:
                self._updating = False
            self.error.emit(str(exc))
            return
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
                self.plane_selector.addItem(name, key)
        previous_index = self.plane_selector.findData(previous_key)
        self.plane_selector.setCurrentIndex(
            previous_index if previous_index >= 0 else 0
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
                " | pure-shift lower←upper "
                f"[[{coupling[0, 0]:.5g}, {coupling[0, 1]:.5g}], "
                f"[{coupling[1, 0]:.5g}, {coupling[1, 1]:.5g}]], "
                f"angular residual {residual:.3g}"
            )
        self.summary.setText(
            f"AC Scan: {'ON' if result.ac_enabled else 'OFF'} | "
            f"Descan: {'ON' if result.descan_enabled else 'OFF'} | "
            f"requested raster {result.requested_pixels_x} x "
            f"{result.requested_pixels_y}; preview {preview_x} x "
            f"{preview_y} | sample span {sample_span_x:.6g} x "
            f"{sample_span_y:.6g} um | drift-only pivot: "
            f"AC {ac_pivot}, Descan {descan_pivot}"
            f"{coupling_text}"
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
        self._stem_auto_range_pending = True
        self._render_stem_rows(shape[0])

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
            self._playback_tick()
            return
        self._playback_timer.stop()
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
        if period_s <= 2.0 * self._playback_timer.interval() * 1.0e-3:
            completed_rows = rows
            frame_number = int(
                (perf_counter() - self._playback_started_s) / period_s
            ) + 1
        else:
            elapsed_s = max(perf_counter() - self._playback_started_s, 0.0)
            frame_number = int(elapsed_s / period_s) + 1
            phase = (elapsed_s % period_s) / period_s
            completed_rows = min(rows, max(1, int(phase * rows) + 1))
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

    def _render_stem_rows(self, completed_rows: int) -> None:
        if self._stem_frame is None:
            return
        auto_range = self._stem_auto_range_pending
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
