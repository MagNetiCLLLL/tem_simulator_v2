"""Central ray-path visualization workspace."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeComboBox as QComboBox,
)

from html import escape
from collections import OrderedDict
from time import perf_counter
import weakref

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from temsim.diagnostics import ray_stop_records, vacuum_bore_plot_points
from temsim.design_explorer import summarise_calculation_result
from temsim.gui.diagnostic_tabs import (
    EnergyFilterView,
    MagneticFieldView,
    OpticalTransferView,
    PhysicalLayoutView,
    TransverseBeamView,
)
from temsim.gui.scan_panel import ScanControlView
from temsim.gui.sample_panel import SamplePage
from temsim.gui.sample_interactions_3d import SampleInteractions3DPage
from temsim.gui.eds_panel import EDSPage
from temsim.gui.aberration_view import AberrationComparisonView
from temsim.gui.ray_scene import StaticRayLayers
from temsim.gui.design_explorer import DesignExplorerPage
from temsim.gui.interactive_calculation import InteractiveCalculationPage
from temsim.gui.model_inspector import ModelInspectorPage
from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.transverse_projection import (
    format_projection_angle,
    project_transverse_values,
    projection_axis_name,
)
from temsim.specimen.source import specimen_structure_available
from temsim.physics.recording_stop import active_tem_recording_plane
from temsim.physics.beam_current import sample_illumination_absent


class WaveImagingView(QWidget):
    """Display the optional one-shot TEM wave image and diffraction pattern."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_wave_presentation = (None, None, "", False)
        self._current_wave_stale = False
        self._bank_readout = None
        self._bank_readout_pending = ""
        self._display_source = "current"
        self._source_view_ranges = {}
        self.image_source = QComboBox()
        self.image_source.setObjectName("temImageSource")
        self.image_source.addItem("Current calculation", "current")
        self.image_source.addItem("Advanced bank", "bank")
        self.image_source.setToolTip("Choose a stored result to display. Current instrument parameters are not changed.")
        self.image_source_status = QLabel("Current calculation")
        self.image_source_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.image_source_status.setWordWrap(True)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Result source"))
        source_row.addWidget(self.image_source)
        source_row.addWidget(self.image_source_status, 1)
        self.summary = QLabel(
            "No TEM wave image | enable it on Sample and run High accuracy"
        )
        self.summary.setToolTip(
            "Enable TEM wave imaging on Sample, then run High accuracy."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self.image_plot = pg.PlotItem()
        self.image = pg.ImageView(view=self.image_plot)
        self.image.setObjectName("waveImageView")
        self.diffraction_plot = pg.PlotItem()
        self.diffraction = pg.ImageView(view=self.diffraction_plot)
        self.diffraction.setObjectName("waveDiffractionView")
        for view in (self.image, self.diffraction):
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.getView().setAspectLocked(True)
        self.image.getView().setTitle(
            "Physical recording-plane image (display-normalised)"
        )
        self.image.getView().setLabel("bottom", "recording x", units="mm")
        self.image.getView().setLabel("left", "recording y", units="mm")
        self.diffraction.getView().setTitle(
            "Specimen exit-wave diffraction reference (log display)"
        )
        self.diffraction.getView().setLabel("bottom", "qₓ", units="Å⁻¹")
        self.diffraction.getView().setLabel("left", "qᵧ", units="Å⁻¹")
        panels = QHBoxLayout()
        panels.addWidget(self.image, 1)
        panels.addWidget(self.diffraction, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(source_row)
        layout.addWidget(self.summary)
        layout.addLayout(panels, 1)
        self.image_source.currentIndexChanged.connect(self._source_changed)

    def _remember_source_ranges(self) -> None:
        if self.image.image is not None or self.diffraction.image is not None:
            self._source_view_ranges[self._display_source] = (
                self.image.getView().viewRange(), self.diffraction.getView().viewRange(),
            )

    def _source_changed(self, *_args) -> None:
        self._remember_source_ranges()
        self._display_source = str(self.image_source.currentData())
        self._refresh_source()

    def _refresh_source(self) -> None:
        if self._display_source == "bank":
            readout = self._bank_readout
            wave = getattr(readout, "wave", None)
            if wave is None:
                self.image.clear()
                self.diffraction.clear()
                self.summary.setText("No TEM image in this Advanced bank readout")
                self.summary.setToolTip("The selected bank must already contain a TEM result. Selecting a source does not calculate it.")
            else:
                self._display_wave_result(wave, getattr(readout, "state_snapshot", None), "High accuracy")
        else:
            wave, state, quality, no_illumination = self._current_wave_presentation
            self._display_wave_result(wave, state, quality, no_illumination=no_illumination)
            if self._current_wave_stale:
                self._show_current_stale_notice()
        ranges = self._source_view_ranges.get(self._display_source)
        if ranges is not None:
            for plot, bounds in zip((self.image, self.diffraction), ranges):
                plot.getView().setRange(xRange=bounds[0], yRange=bounds[1], padding=0, disableAutoRange=True)
        self._update_source_status()

    def _update_source_status(self) -> None:
        if self._display_source == "bank":
            if self._bank_readout is None:
                status = "No Advanced bank readout"
            elif self._bank_readout_pending:
                status = "Advanced bank | previous readout retained"
            else:
                status = "Advanced bank | captured settings"
            details = ["Display only. Sample and Image Aberrations controls still edit the current instrument."]
            if self._bank_readout_pending:
                details.append(self._bank_readout_pending)
            details.extend(getattr(self._bank_readout, "notes", ()))
            details.extend(f"{key}: {value:.9g}" for key, value in getattr(self._bank_readout, "coordinates", {}).items())
        else:
            status = "Current calculation | inputs changed" if self._current_wave_stale else "Current calculation"
            details = ["Displays the main calculation result. A completed Advanced bank is stored separately."]
        self.image_source_status.setText(status)
        self.image_source_status.setToolTip("\n".join(details))

    def set_bank_readout(self, readout) -> None:
        """Publish a detached bank result without replacing the main image."""
        self._bank_readout = readout
        self._bank_readout_pending = ""
        if self._display_source == "bank":
            self._remember_source_ranges()
            self._refresh_source()

    def mark_bank_readout_pending(self, message: str) -> None:
        self._bank_readout_pending = str(message)
        self._update_source_status()

    @staticmethod
    def _axis_transform(x_axis, y_axis):
        x_values = np.asarray(x_axis, dtype=float)
        y_values = np.asarray(y_axis, dtype=float)
        if x_values.size < 2 or y_values.size < 2:
            raise ValueError("Wave-image axes need at least two samples.")
        step_x = float(x_values[1] - x_values[0])
        step_y = float(y_values[1] - y_values[0])
        if (
            not np.isfinite(step_x)
            or not np.isfinite(step_y)
            or step_x <= 0.0
            or step_y <= 0.0
        ):
            raise ValueError("Wave-image axes must be finite and increasing.")
        return (
            (
                float(x_values[0] - 0.5 * step_x),
                float(y_values[0] - 0.5 * step_y),
            ),
            (step_x, step_y),
        )

    def display_result(
        self, wave_result, state=None, quality: str = "", *, no_illumination=False,
    ) -> None:
        self._current_wave_presentation = (wave_result, state, quality, no_illumination)
        self._current_wave_stale = False
        if self._display_source == "current":
            self._display_wave_result(wave_result, state, quality, no_illumination=no_illumination)
            self._update_source_status()

    def _display_wave_result(
        self, wave_result, state=None, quality: str = "", *, no_illumination=False,
    ) -> None:
        if wave_result is None:
            self.image.clear()
            self.diffraction.clear()
            self.summary.setToolTip("")
            if no_illumination:
                self.summary.setText("No incident current at the specimen | no TEM wave image")
                return
            illumination = str(
                getattr(state, "illumination_mode", "") if state is not None else ""
            ).upper()
            specimen_mode = str(
                getattr(getattr(state, "sample", None), "specimen_mode", "")
                if state is not None
                else ""
            ).lower()
            requested = bool(
                getattr(getattr(state, "sample", None), "wave_enabled", False)
                if state is not None
                else False
            )
            try:
                active_tem_recording_plane(state)
                recording_available = True
                recording_error = ""
            except (AttributeError, StopIteration, ValueError) as exc:
                recording_available = False
                recording_error = str(exc)
            if quality == "Preview":
                message = (
                    "Preview omits TEM wave imaging. Run High accuracy to "
                    "calculate the physical recording-plane result."
                )
            elif requested and illumination != "TEM":
                message = (
                    "TEM wave imaging is inactive in Nanoprobe (STEM) mode; "
                    "use STEM detector imaging or switch to Microprobe (TEM)."
                )
            elif requested and not specimen_structure_available(state.sample):
                message = (
                    "TEM wave imaging requires an imported CIF/MCIF in Real "
                    "mode or a TOML reference specimen in Virtual mode."
                )
            elif requested and not recording_available:
                message = recording_error or (
                    "Insert the Fluorescent Screen or Camera, then run High "
                    "accuracy."
                )
            else:
                message = (
                    "No TEM wave image in this result. Enable TEM image / "
                    "diffraction on the Sample and run High accuracy."
                )
            self.summary.setText(message)
            return
        image_x = np.asarray(
            getattr(wave_result, "camera_x_mm", wave_result.x_angstrom),
            dtype=float,
        )
        image_y = np.asarray(
            getattr(wave_result, "camera_y_mm", wave_result.y_angstrom),
            dtype=float,
        )
        image_pos, image_scale = self._axis_transform(image_x, image_y)
        self.image.setImage(
            np.asarray(wave_result.image_intensity, dtype=float).T,
            autoRange=True,
            autoLevels=True,
            pos=image_pos,
            scale=image_scale,
        )
        frequency_x = np.asarray(
            wave_result.spatial_frequency_inv_angstrom,
            dtype=float,
        )
        frequency_y = np.asarray(
            getattr(
                wave_result,
                "spatial_frequency_y_inv_angstrom",
                frequency_x,
            ),
            dtype=float,
        )
        diffraction_pos, diffraction_scale = self._axis_transform(
            frequency_x,
            frequency_y,
        )
        self.diffraction.setImage(
            np.asarray(wave_result.diffraction_intensity, dtype=float).T,
            autoRange=True,
            autoLevels=True,
            pos=diffraction_pos,
            scale=diffraction_scale,
        )
        metrics = wave_result.metrics
        plane_name = str(
            metrics.get("recording_plane_name", "Camera")
        )
        observable = (
            "Diffraction pattern"
            if metrics.get("recording_plane_observable")
            == "diffraction_pattern"
            else "Image"
        )
        self.image.getView().setTitle(
            f"{plane_name}: {observable} (display-normalised)"
        )
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
            f"{configurations} configuration(s){thermal_text} | "
            f"{plane_name} {observable.lower()} | "
            f"{backend}"
            f"{warning_text}"
        )
        details = [
            "Scope: gun-conditioned illumination, specimen interaction, "
            "complete projector transfer and physical recording response.",
            "Recording propagation: "
            f"{metrics.get('camera_wave_propagation_method', 'unknown')}",
            "Recording sampling: "
            f"{metrics.get('camera_calculation_pixels_xy', 'unknown')} pixels; "
            f"binning {metrics.get('camera_binning_xy', 'unknown')}.",
            "Energy-loss scope: "
            f"{metrics.get('wave_energy_loss_scope', 'not reported')}",
            "Zero-loss probability per sample-incident electron: "
            f"{float(metrics.get('zero_loss_probability_per_sample_incident', 1.0)):.6g}",
            "Intensity treatment: "
            f"{metrics.get('displayed_intensity_average', 'unknown')}",
            "Image display: "
            f"{metrics.get('image_display_scaling', 'unknown')}",
            "Diffraction display: "
            f"{metrics.get('diffraction_display_scaling', 'unknown')}",
            "Potential builder: "
            f"{metrics.get('specimen_potential_builder_backend', 'unknown')}",
            f"Potential model: {potential_model}",
            "Projector magnification: "
            f"{float(metrics.get('projector_magnification', 0.0)):.5g}x",
            f"Surviving rays: {int(metrics['surviving_rays'])}",
            f"Output: {plane_name} {observable.lower()}, display-normalised.",
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

    def mark_result_stale(self) -> None:
        """Retain the last complete image while preventing a false cache hit."""

        self._current_wave_stale = True
        if self._display_source == "current":
            self._show_current_stale_notice()
            self._update_source_status()

    def _show_current_stale_notice(self) -> None:
        if self.image.image is None and self.diffraction.image is None:
            return
        self.summary.setText(
            "Previous High accuracy image retained | inputs changed"
        )
        self.summary.setToolTip(
            "The displayed complete frame belongs to the previous microscope "
            "state. Run High accuracy to update it."
        )


class VisualizationWorkspace(QWidget):
    ray_layout_changing = Signal()
    ray_layout_changed = Signal()
    component_selected = Signal(str)
    scan_parameters_changed = Signal(str)
    scan_error = Signal(str)
    calculation_artifacts_changed = Signal(object)
    MAX_DISPLAY_RAYS = 48
    MAX_RANGE_SAMPLE_RAYS = 256
    RAY_LABEL_BASE_PT = 10
    RAY_LABEL_MAX_PT = 14
    RAY_AXIS_TICK_PT = 10
    RAY_AXIS_LABEL_PT = 11
    RAY_LEGEND_PT = 10
    CONVERGENCE_SHADE_BINS = 5
    INTERACTION_LABELS = {
        "incident": "Incident (pre-sample)",
        "vacuum": "Vacuum continuation",
        "real_sample_reference": "Real sample: reference ray only",
        "real_zero_loss": "Real: zero loss / elastic coherent",
        "real_plasmon": "Real: plasmon / low loss",
        "real_ionisation": "Real: core ionisation",
        "real_other_inelastic": "Real: other inelastic",
        "real_plural_inelastic": "Real: plural inelastic",
        "virtual_interactions_disabled": "Virtual interactions disabled",
        "transmitted": "Transmitted / direct",
        "diffraction_spots": "Diffraction spots",
        "diffuse_ring": "Diffuse ring",
        "gaussian_diffuse": "Gaussian diffuse",
        "arbitrary_angular": "Arbitrary angular",
        "user_screened_power_law": "User screened power law",
        "physical_rutherford": "Physical Rutherford approximation",
        "unknown": "Unknown interaction",
    }
    OPTION_BUTTON_STYLE = """
        QPushButton {
            min-height: 24px;
            padding: 2px 7px;
            border: 1px solid #64748b;
            border-radius: 6px;
            background: #f8fafc;
            color: #0f172a;
            font-size: 12px;
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
        # Exact display-only data, never full solver histories or signals.
        self._ray_display_cache = OrderedDict()
        self._ray_display_cache_budget = 512 * 1024**2
        self._ray_display_cache_bytes = 0
        self._ray_display_cache_hits = 0
        self._ray_display_cache_misses = 0
        self._ray_display_cache_result = None
        self._ray_static_layers = StaticRayLayers()
        self._ray_scene_initialized = False
        self._ray_scene_updates = 0
        self._ray_scene_last_update_ms = 0.0
        self._ray_items_by_group = {}
        self._ray_legend_items = {}
        self._stop_items_by_group = {}
        self._support_items_by_branch = {}
        self._tuning_envelopes = []
        self._scan_ray_paths = None
        self._scan_ray_offsets_m: dict[str, np.ndarray] = {}
        self._scan_playback_active = False
        self._scan_playback_time_s: float | None = None
        self._convergence_colour_reference_mrad = 0.0
        self._projection_redraw_timer = QTimer(self)
        self._projection_redraw_timer.setSingleShot(True)
        self._projection_redraw_timer.setInterval(16)
        self._projection_redraw_timer.timeout.connect(
            self._redraw_projection_items
        )
        self._projection_finalize_timer = QTimer(self)
        self._projection_finalize_timer.setSingleShot(True)
        self._projection_finalize_timer.setInterval(150)
        self._projection_finalize_timer.timeout.connect(
            self._finalize_projection
        )
        self.projection_label = QLabel("Angle")
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
        for projection_button in (self.projection_xz, self.projection_yz):
            projection_button.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
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
        self.component_centres = QPushButton("Centres")
        self.component_centres.setObjectName("componentCentresToggle")
        self.component_centres.setCheckable(True)
        self.component_centres.setChecked(True)
        self.component_centres.setToolTip("Show component centre markers")
        self.crossovers = QPushButton("Crossovers")
        self.crossovers.setObjectName("crossoversToggle")
        self.crossovers.setCheckable(True)
        self.crossovers.setChecked(True)
        self.crossovers.setToolTip("Show detected ray crossovers")
        self.auto_zoom = QPushButton("Auto")
        self.auto_zoom.setObjectName("autoZoomToggle")
        self.auto_zoom.setCheckable(True)
        self.auto_zoom.setToolTip(
            "Automatically zoom to the selected assembly component"
        )
        self.auto_zoom.setChecked(False)
        self.column_walls = QPushButton("Walls")
        self.column_walls.setObjectName("columnWallsToggle")
        self.column_walls.setCheckable(True)
        self.column_walls.setChecked(True)
        self.column_walls.setToolTip(
            "Show the position-dependent circular vacuum inner diameter"
        )
        self.fit_column = QPushButton("Fit")
        self.fit_column.setObjectName("fitColumnButton")
        self.fit_column.setToolTip(
            "Fit the complete axial range and column inner diameter"
        )
        self.axial_position = QDoubleSpinBox()
        self.axial_position.setObjectName("rayDiagramAxialPosition")
        self.axial_position.setRange(-1.0e6, 1.0e6)
        self.axial_position.setDecimals(6)
        self.axial_position.setSingleStep(0.000001)
        self.axial_position.setSuffix(" mm")
        self.axial_position.setKeyboardTracking(False)
        self.axial_position.setToolTip(
            "Axial Z in mm; six decimal places and a 1 nm step resolve "
            "probe defocus near the sample plane"
        )
        self.jump_to_position = QPushButton("Go to Z")
        self.jump_to_position.setObjectName("rayDiagramGoToPosition")
        self.jump_to_position.setToolTip(
            "Centre the Ray Diagram on the entered axial position"
        )
        self.magnetic_field_toggle = QPushButton("Magnetic field")
        self.magnetic_field_toggle.setObjectName("rayMagneticFieldToggle")
        self.magnetic_field_toggle.setCheckable(True)
        self.magnetic_field_toggle.setChecked(False)
        self.magnetic_field_toggle.setToolTip(
            "Show or hide the axial magnetic-field panel below the ray diagram"
        )
        self.transverse_beam_toggle = QPushButton("Transverse beam")
        self.transverse_beam_toggle.setObjectName("rayTransverseBeamToggle")
        self.transverse_beam_toggle.setCheckable(True)
        self.transverse_beam_toggle.setChecked(True)
        self.transverse_beam_toggle.setToolTip(
            "Show or hide the origin-centred Transverse X-Y panel on the "
            "right"
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
            self.magnetic_field_toggle,
            self.transverse_beam_toggle,
        ):
            option_button.setStyleSheet(self.OPTION_BUTTON_STYLE)

        # Keep the view controls responsive without allowing overlay buttons to
        # expand into oversized grid cells or wrap across multiple rows.
        heading_row = QHBoxLayout()
        heading_row.addWidget(self.heading)
        heading_row.addWidget(self.magnetic_field_toggle)
        heading_row.addWidget(self.transverse_beam_toggle)
        self.live_tuning_toggle = QToolButton()
        self.live_tuning_toggle.setObjectName("rayLiveTuningToggle")
        self.live_tuning_toggle.setText("Live tuning")
        self.live_tuning_toggle.setToolTip("Show or hide the Live tuning dock")
        heading_row.addWidget(self.live_tuning_toggle)
        heading_row.addStretch(1)

        self.view_controls_panel = QWidget()
        self.view_controls_panel.setObjectName("rayDiagramControlRow")
        view_controls = QHBoxLayout(self.view_controls_panel)
        view_controls.setContentsMargins(0, 0, 0, 0)
        view_controls.setSpacing(3)
        view_controls.setAlignment(Qt.AlignmentFlag.AlignLeft)
        view_controls.addWidget(self.projection_label)
        view_controls.addWidget(self.projection_xz)
        view_controls.addWidget(self.projection_yz)
        view_controls.addWidget(self.projection_slider)
        option_buttons = (
            self.auto_zoom,
            self.fit_column,
            self.column_walls,
            self.component_centres,
            self.crossovers,
        )
        for button in option_buttons:
            button.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            view_controls.addWidget(button)
        view_controls.addStretch(1)
        self.view_controls_panel.adjustSize()
        self.view_controls_scroll = QScrollArea()
        self.view_controls_scroll.setObjectName("rayDiagramControlRowScroll")
        self.view_controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.view_controls_scroll.setWidgetResizable(False)
        self.view_controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.view_controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.view_controls_scroll.setWidget(self.view_controls_panel)
        self.view_controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.view_controls_scroll.setFixedHeight(
            self.view_controls_panel.sizeHint().height()
            + self.view_controls_scroll.horizontalScrollBar().sizeHint().height()
            + 2
        )
        self.view_controls_scroll.setStyleSheet(
            "QScrollArea#rayDiagramControlRowScroll {"
            " background: transparent; border: none; }"
            "QScrollArea#rayDiagramControlRowScroll > QWidget > QWidget {"
            " background: transparent; }"
            "QScrollBar:horizontal { height: 12px; }"
        )

        navigation_hint = QLabel(
            "Double-click any axial plot to update Transverse X-Y"
        )
        navigation_hint.setToolTip(
            "Double-click an axial position in Ray Diagram, Physical Layout, "
            "or Magnetic Field to update the Transverse X-Y panel on the right."
        )
        navigation_hint.setWordWrap(True)
        navigation_hint.setStyleSheet("color: #64748b; font-weight: 600;")
        navigation_controls = QHBoxLayout()
        navigation_controls.addStretch(1)
        navigation_controls.addWidget(QLabel("Axial Z"))
        navigation_controls.addWidget(self.axial_position)
        navigation_controls.addWidget(self.jump_to_position)

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
        self.recording_surface_range_items = []
        self._aperture_span_records = []
        self._aperture_projection_records = []
        self._aperture_stops_by_key = {}
        self.deflector_pair_items = []
        self.crossover_marker_items = []
        self.column_wall_items = []
        self.stop_marker_items = []
        self._stop_projection_records = []
        self._ray_bundle_records = []
        self._crossover_count = 0
        self._wall_stop_count = 0
        self.axial_cursor_item = None
        self._selected_z_mm = None
        self._last_result = None
        self._last_quality = ""
        self._preview_result = None
        self._high_accuracy_result = None
        self._high_accuracy_current = False
        self._sample_region_result = None
        self._focused_part = None
        self._ray_component_highlight = None
        self._show_notice("Waiting for the first calculation")

        self.stop_detail = QLabel(
            "Click a stop marker to inspect the first physical intercept"
        )
        self.stop_detail.setWordWrap(True)
        self.stop_detail.setStyleSheet("color: #fbbf24; font-weight: 600;")
        self.interaction_detail = QTextBrowser()
        self.interaction_detail.setObjectName(
            "rayPlaneInteractionSummary"
        )
        self.interaction_detail.setMinimumHeight(64)
        self.interaction_detail.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.interaction_detail.setOpenExternalLinks(False)
        self.interaction_detail.setStyleSheet(
            "QTextBrowser { background: #0b1020; color: #cbd5e1; "
            "border: 1px solid #334155; border-radius: 5px; padding: 4px; }"
        )
        self.interaction_detail.setHtml(
            "<b>Selected-plane interaction budget</b><br>"
            "Choose an axial Z position to calculate source-normalised "
            "interaction fractions."
        )
        self.hint = QLabel("Angle and display-scale diagnostics appear here")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.hint.setStyleSheet("color: #64748b; font-weight: 600;")
        self.hint.setToolTip(
            "Hover centre lines for details | Mouse wheel: zoom | "
            "Drag: pan | Right click: plot menu"
        )

        self.magnetic_field = MagneticFieldView()
        self.magnetic_field.setObjectName("embeddedMagneticFieldView")
        self.magnetic_field.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.magnetic_field.link_axial_axis(self.plot)

        self.ray_primary_panel = QWidget()
        self.ray_primary_panel.setObjectName("rayDiagramPrimaryPanel")
        ray_primary_layout = QVBoxLayout(self.ray_primary_panel)
        ray_primary_layout.setContentsMargins(0, 0, 0, 0)
        ray_primary_layout.addLayout(heading_row)
        ray_primary_layout.addWidget(self.view_controls_scroll)
        ray_primary_layout.addWidget(navigation_hint)
        ray_primary_layout.addLayout(navigation_controls)
        ray_primary_layout.addWidget(self.plot, 1)
        ray_primary_layout.addWidget(self.stop_detail)
        ray_primary_layout.addWidget(self.hint)

        self.ray_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.ray_vertical_splitter.setObjectName(
            "rayDiagramVerticalSplitter"
        )
        self.ray_vertical_splitter.setChildrenCollapsible(False)
        self.ray_vertical_splitter.setHandleWidth(7)
        self.ray_vertical_splitter.setStyleSheet(
            "QSplitter#rayDiagramVerticalSplitter::handle:vertical {"
            " background: #334155; border-top: 1px solid #64748b;"
            " border-bottom: 1px solid #0f172a; }"
            "QSplitter#rayDiagramVerticalSplitter::handle:vertical:hover {"
            " background: #2563eb; border-top-color: #60a5fa; }"
        )
        self.ray_vertical_splitter.setToolTip(
            "Drag the horizontal separators to resize the Ray Diagram, "
            "selected-plane interaction budget, and Magnetic Field panels."
        )
        self.ray_vertical_splitter.addWidget(self.ray_primary_panel)
        self.ray_vertical_splitter.addWidget(self.interaction_detail)
        self.ray_vertical_splitter.addWidget(self.magnetic_field)
        self.ray_vertical_splitter.setStretchFactor(0, 6)
        self.ray_vertical_splitter.setStretchFactor(1, 1)
        self.ray_vertical_splitter.setStretchFactor(2, 3)
        # Keep the electron-ray display dominant on first use.  These are
        # relative weights; the user can freely drag both splitter handles.
        self.ray_vertical_splitter.setSizes((650, 110, 260))
        self.magnetic_field.setVisible(False)

        self.ray_page = QWidget()
        ray_layout = QVBoxLayout(self.ray_page)
        ray_layout.setContentsMargins(0, 0, 0, 0)

        self.physical_layout = PhysicalLayoutView()
        self.probe_aberrations = AberrationComparisonView(
            fixed_system="probe"
        )
        self.image_aberrations = AberrationComparisonView(
            fixed_system="image"
        )
        # Compatibility alias for callers that previously inspected the
        # single switchable aberration page.
        self.aberrations = self.probe_aberrations
        self.optical_transfer = OpticalTransferView()
        self.energy_filter = EnergyFilterView()
        self.energy_filter_parameters = ParameterPanel()
        self.energy_filter_parameters.tabs.setObjectName("energyFilterEditorTabs")
        self.energy_filter_parameters.setObjectName(
            "energyFilterParameterPanel"
        )
        self.energy_filter_parameters.setMinimumWidth(0)
        self.energy_filter_component_selector = QComboBox()
        self.energy_filter_component_selector.setObjectName(
            "energyFilterComponentSelector"
        )
        self.energy_filter_component_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.energy_filter_component_selector.setMinimumContentsLength(24)
        energy_filter_selector_row = QHBoxLayout()
        energy_filter_selector_row.addWidget(QLabel("Component"))
        energy_filter_selector_row.addWidget(
            self.energy_filter_component_selector, 1
        )
        energy_filter_parameter_page = QWidget()
        energy_filter_parameter_layout = QVBoxLayout(
            energy_filter_parameter_page
        )
        energy_filter_parameter_layout.setContentsMargins(0, 0, 0, 0)
        energy_filter_parameter_layout.addLayout(
            energy_filter_selector_row
        )
        energy_filter_parameter_layout.addWidget(
            self.energy_filter_parameters, 1
        )
        self.energy_filter_parameter_tabs = QTabWidget()
        self.energy_filter_parameter_tabs.setObjectName(
            "energyFilterParameterTabs"
        )
        self.energy_filter_parameter_tabs.addTab(
            energy_filter_parameter_page, "Parameters"
        )
        self.energy_filter_parameter_tabs.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.energy_filter_parameter_tabs.setMinimumWidth(320)
        self.energy_filter.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.energy_filter.setMinimumWidth(360)
        self.energy_filter_page = QSplitter(Qt.Orientation.Horizontal)
        self.energy_filter_page.setObjectName("energyFilterSplitter")
        self.energy_filter_page.setChildrenCollapsible(False)
        self.energy_filter_page.addWidget(
            self.energy_filter_parameter_tabs
        )
        self.energy_filter_page.addWidget(self.energy_filter)
        self.energy_filter_page.setStretchFactor(0, 0)
        self.energy_filter_page.setStretchFactor(1, 1)
        self.energy_filter_page.setSizes((420, 1000))
        self.transverse_beam = TransverseBeamView()
        self.transverse_beam.setMinimumWidth(340)
        self.transverse_beam.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        # Preserve the compact ray-toolbar labels even at the workspace's
        # minimum width; 540 px is still small enough to keep the complete
        # two-column page below the existing 900 px shell threshold.
        self.ray_vertical_splitter.setMinimumWidth(540)
        self.ray_workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.ray_workspace_splitter.setObjectName(
            "rayDiagramWorkspaceSplitter"
        )
        self.ray_workspace_splitter.setChildrenCollapsible(False)
        self.ray_workspace_splitter.setHandleWidth(7)
        self.ray_workspace_splitter.setStyleSheet(
            "QSplitter#rayDiagramWorkspaceSplitter::handle:horizontal {"
            " background: #334155; border-left: 1px solid #64748b;"
            " border-right: 1px solid #0f172a; }"
            "QSplitter#rayDiagramWorkspaceSplitter::handle:horizontal:hover {"
            " background: #2563eb; border-left-color: #60a5fa; }"
        )
        self.ray_workspace_splitter.setToolTip(
            "Drag the vertical separator to resize Ray Diagram and "
            "Transverse X-Y."
        )
        self.ray_workspace_splitter.addWidget(self.ray_vertical_splitter)
        self.ray_workspace_splitter.addWidget(self.transverse_beam)
        self.ray_workspace_splitter.setStretchFactor(0, 3)
        self.ray_workspace_splitter.setStretchFactor(1, 1)
        self.ray_workspace_splitter.setSizes((1350, 450))
        self.scan_control = ScanControlView()
        self.sample_page = SamplePage()
        self.sample_interactions_3d = SampleInteractions3DPage()
        self.eds_page = EDSPage()
        self.sample_interactions_3d.set_parameters_widget(
            self.eds_page.settings_panel
        )
        self.wave_imaging = WaveImagingView()
        scanning_parameters, scanning_results = (
            self.scan_control.take_workspace_panels()
        )
        self.scanning_controls_tabs = QTabWidget()
        self.scanning_controls_tabs.setObjectName("scanningControlTabs")
        self.scanning_controls_tabs.addTab(
            scanning_parameters, "Scanning Parameters"
        )
        self.scanning_controls_tabs.addTab(
            self.probe_aberrations, "Probe Aberrations"
        )
        self.scanning_results_tabs = scanning_results
        self.scanning_controls_tabs.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.scanning_controls_tabs.setMinimumWidth(260)
        self.scanning_controls_tabs.tabBar().setExpanding(False)
        self.scanning_controls_tabs.tabBar().setUsesScrollButtons(True)
        self.scanning_controls_tabs.tabBar().setElideMode(
            Qt.TextElideMode.ElideRight
        )
        self.scanning_results_tabs.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.scanning_results_tabs.setMinimumWidth(360)
        self.scanning_page = QSplitter(Qt.Orientation.Horizontal)
        self.scanning_page.setObjectName("scanningImageSplitter")
        self.scanning_page.addWidget(self.scanning_controls_tabs)
        self.scanning_page.addWidget(self.scanning_results_tabs)
        self.scanning_page.setStretchFactor(0, 0)
        self.scanning_page.setStretchFactor(1, 1)
        self.scanning_page.setSizes((420, 1000))
        self.illuminating_page = QTabWidget()
        self.illuminating_page.setObjectName("illuminatingImageTabs")
        self.illuminating_page.addTab(
            self.wave_imaging, "Illuminating Image"
        )
        self.illuminating_page.addTab(
            self.image_aberrations, "Image Aberrations"
        )
        self.design_explorer = DesignExplorerPage()
        self.interactive_calculation = InteractiveCalculationPage(self)
        self.interactive_calculation.hide()
        self.ray_result_tabs = QTabWidget()
        self.ray_result_tabs.setObjectName("rayResultTabs")
        self.ray_result_tabs.addTab(self.ray_workspace_splitter, "Rays")
        self.ray_result_tabs.addTab(self.interactive_calculation.readout_panel, "Cached signals")
        self.ray_result_tabs.setTabToolTip(1, "Detached Advanced-bank readout, separate from current live settings")
        ray_layout.addWidget(self.ray_result_tabs, 1)
        self.interactive_calculation.rays_requested.connect(self.show_ray_diagram)
        self.interactive_calculation.readout_updated.connect(self.scan_control.set_bank_readout)
        self.interactive_calculation.readout_updated.connect(self.wave_imaging.set_bank_readout)
        self.interactive_calculation.readout_status_changed.connect(self.scan_control.mark_bank_readout_pending)
        self.interactive_calculation.readout_status_changed.connect(self.wave_imaging.mark_bank_readout_pending)
        self.model_inspector = ModelInspectorPage()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("visualizationTabs")
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.addTab(self.ray_page, "Ray Diagram")
        self.tabs.addTab(self.physical_layout, "Physical Layout")
        self.tabs.addTab(self.energy_filter_page, "Energy Filter")
        self.tabs.addTab(self.sample_page, "Sample")
        self.tabs.addTab(self.sample_interactions_3d, "Sample Interactions 3D")
        self.tabs.addTab(self.eds_page, "EDS")
        self.tabs.addTab(self.scanning_page, "Scanning Image")
        self.tabs.addTab(self.illuminating_page, "Illuminating Image")
        self.tabs.addTab(self.optical_transfer, "Optical Transfer")
        self.tabs.addTab(self.model_inspector, "Model Inspector")
        self.tabs.addTab(self.design_explorer, "Design Explorer")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self.component_centres.toggled.connect(self._redraw_last_result)
        self.crossovers.toggled.connect(self._redraw_last_result)
        self.column_walls.toggled.connect(self._redraw_last_result)
        self.magnetic_field_toggle.toggled.connect(
            lambda visible: self._set_ray_panel_visible(self.magnetic_field, visible)
        )
        self.transverse_beam_toggle.toggled.connect(
            lambda visible: self._set_ray_panel_visible(self.transverse_beam, visible)
        )
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
        self.energy_filter_component_selector.currentIndexChanged.connect(
            self._energy_filter_component_changed
        )
        self.scan_control.parameters_changed.connect(
            self.scan_parameters_changed.emit
        )
        self.scan_control.error.connect(self.scan_error.emit)
        self.sample_page.parameters_changed.connect(
            self.scan_parameters_changed.emit
        )
        self.sample_page.error.connect(self.scan_error.emit)
        self.eds_page.parameters_changed.connect(
            self.scan_parameters_changed.emit
        )
        self.eds_page.error.connect(self.scan_error.emit)
        self.eds_page.sample_region_result_ready.connect(
            self._set_sample_region_result
        )
        self.eds_page.specimen_interactions_updated.connect(
            self._set_specimen_interactions
        )
        self.sample_interactions_3d.sample_region_requested.connect(
            self._ensure_sample_region_result
        )
        self.scan_control.playback_time_changed.connect(
            self._scan_playback_time_changed
        )
        self.scan_control.playback_active_changed.connect(
            self._scan_playback_active_changed
        )
        self.physical_layout.axial_position_selected.connect(
            self.jump_to_ray_position
        )
        self.magnetic_field.axial_position_selected.connect(
            self.jump_to_ray_position
        )

        # Presentation-only queues: one newest result per optional panel, not
        # another history cache. Scientific result publication below is eager.
        self._pending_ray_panels = {}
        self._pending_ray_focus = set()
        self._presented_ray_panels = set()
        self._transverse_focus_request = None
        self._ray_panel_refresh_timer = QTimer(self)
        self._ray_panel_refresh_timer.setSingleShot(True)
        self._ray_panel_refresh_timer.timeout.connect(self._refresh_visible_ray_panels)
        for panel in self._optional_ray_panels():
            panel.installEventFilter(self)
        self.tabs.currentChanged.connect(self._schedule_visible_ray_panels)
        self.ray_result_tabs.currentChanged.connect(self._schedule_visible_ray_panels)

    def _optional_ray_panels(self):
        return (self.physical_layout, self.magnetic_field, self.transverse_beam)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Show and watched in self._optional_ray_panels():
            self._schedule_visible_ray_panels()
        return super().eventFilter(watched, event)

    def _schedule_visible_ray_panels(self, *_args) -> None:
        # Show events may arrive while Qt is still changing parent visibility.
        if not self._ray_panel_refresh_timer.isActive():
            self._ray_panel_refresh_timer.start(0)

    @staticmethod
    def _current_presentation_part(result, part):
        """Resolve remembered selection against the newly published assembly."""
        if part is None:
            return None
        parts = getattr(getattr(result, "assembly", None), "parts", None)
        if parts is None:
            return part
        key = str(getattr(part, "key", ""))
        return next((item for item in parts if str(item.key) == key), None)

    def _refresh_visible_ray_panels(self, *_args) -> None:
        for panel in self._optional_ray_panels():
            if not panel.isVisible():
                continue
            if panel not in self._pending_ray_panels:
                if panel in self._pending_ray_focus and self._focused_part is not None:
                    panel.focus_component(self._focused_part)
                    self._pending_ray_focus.discard(panel)
                continue
            result = self._pending_ray_panels.pop(panel)
            previous_range = (panel.plot.getViewBox().viewRange()
                              if panel in self._presented_ray_panels else None)
            # Magnetic Field shares Ray Diagram's axial axis. A hidden view
            # may still have stale layout-dependent linked bounds of its own.
            ray_x_range = (self.plot.getViewBox().viewRange()[0]
                           if panel is self.magnetic_field else None)
            if panel is self.transverse_beam:
                focus = self._transverse_focus_request
                if focus is not None and focus[0] == "component":
                    focus = (focus[0], self._current_presentation_part(result, focus[1]))
                panel.display_result(result, focus=focus)
            else:
                panel.display_result(result)
                part = self._current_presentation_part(result, self._focused_part)
                if part is not None:
                    panel.focus_component(part)
                if ray_x_range is not None:
                    panel.plot.setRange(
                        xRange=ray_x_range,
                        yRange=previous_range[1] if previous_range is not None else None,
                        padding=0.0, disableAutoRange=True,
                    )
                elif previous_range is not None:
                    # Reopening a dirty panel is not an implicit Fit command.
                    panel.plot.setRange(xRange=previous_range[0], yRange=previous_range[1],
                                        padding=0.0, disableAutoRange=True)
            self._presented_ray_panels.add(panel)
            self._pending_ray_focus.discard(panel)

    def _publish_optional_ray_panels(self, result, *, refresh=True) -> None:
        for panel in self._optional_ray_panels():
            self._pending_ray_panels[panel] = result
        self.magnetic_field.mark_presentation_pending()
        if refresh:
            self._refresh_visible_ray_panels()

    def _focus_transverse(self, kind: str, value) -> None:
        self._transverse_focus_request = (kind, value)
        if not self.transverse_beam.isVisible():
            if self._last_result is not None:
                self._pending_ray_panels[self.transverse_beam] = self._last_result
            return
        if self.transverse_beam in self._pending_ray_panels:
            self._refresh_visible_ray_panels()
        elif kind == "component":
            self.transverse_beam.focus_component(value)
        else:
            self.transverse_beam.focus_z(value)

    def show_sample_page(self) -> None:
        """Activate the central owner of all specimen parameters."""

        index = self.tabs.indexOf(self.sample_page)
        if index >= 0:
            self.tabs.setCurrentIndex(index)
            self.sample_page.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_energy_filter_components(
        self,
        components: tuple[tuple[str, str], ...],
        current_key: str | None = None,
    ) -> None:
        """Populate the Energy Filter-local component navigator."""

        blocker = QSignalBlocker(self.energy_filter_component_selector)
        self.energy_filter_component_selector.clear()
        for key, label in components:
            self.energy_filter_component_selector.addItem(label, key)
        if current_key is not None:
            index = self.energy_filter_component_selector.findData(
                str(current_key)
            )
            if index >= 0:
                self.energy_filter_component_selector.setCurrentIndex(index)
        del blocker

    def select_energy_filter_component(self, key: str) -> bool:
        """Select one Energy Filter-local component without recursion."""

        index = self.energy_filter_component_selector.findData(str(key))
        if index < 0:
            return False
        blocker = QSignalBlocker(self.energy_filter_component_selector)
        self.energy_filter_component_selector.setCurrentIndex(index)
        del blocker
        return True

    def show_energy_filter_page(self) -> None:
        """Activate the Energy Filter-owned controls and branch view."""

        index = self.tabs.indexOf(self.energy_filter_page)
        if index >= 0:
            self.tabs.setCurrentIndex(index)
            self.energy_filter_component_selector.setFocus(
                Qt.FocusReason.OtherFocusReason
            )

    def _energy_filter_component_changed(self, index: int) -> None:
        key = self.energy_filter_component_selector.itemData(index)
        if key is not None:
            self.component_selected.emit(str(key))

    def _show_notice(self, text: str) -> None:
        self.plot.clear()
        self._ray_component_highlight = None
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
        if ray_count <= 0 or int(count) <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        count = min(int(count), ray_count)
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

    def _display_ray_indices(self, branch) -> np.ndarray:
        ray_count = int(branch.x.shape[1])
        if ray_count <= 0:
            return np.array([], dtype=int)
        display_count = min(self.MAX_DISPLAY_RAYS, ray_count)
        return np.unique(
            np.linspace(0, ray_count - 1, display_count, dtype=int)
        )

    def _display_bundle_lines(
        self, branch, indices=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reproject exact clipped X/Y bases of the deterministic drawn subset.

        Published result arrays are read-only inputs here. A new display_result
        call invalidates the bases, even when the result object was reused.
        """

        if indices is None:
            indices = self._display_ray_indices(branch)
        else:
            indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        sources = (branch.z, branch.x, branch.y, branch.blocked_z)
        key = ("lines", tuple(id(value) for value in sources), indices.tobytes())
        basis = self._ray_display_cache_get(key, sources)
        if basis is None:
            blocked_z = np.asarray(branch.blocked_z, dtype=float)[indices]
            z, x_values = self._bundle_lines(
                branch.z, np.asarray(branch.x)[:, indices], len(indices), blocked_z
            )
            _, y_values = self._bundle_lines(
                branch.z, np.asarray(branch.y)[:, indices], len(indices), blocked_z
            )
            basis = (z, x_values, y_values)
            self._ray_display_cache_put(key, sources, basis)
        z, x_values, y_values = basis
        projected = self._project_transverse(x_values, y_values)
        scan_offset = self._scan_ray_offsets_m.get(
            str(getattr(branch, "name", ""))
        )
        if scan_offset is not None:
            scan_offset = np.asarray(scan_offset, dtype=float)
            if scan_offset.shape == (len(branch.z), 2):
                offsets_mm = self._project_transverse(
                    scan_offset[:, 0], scan_offset[:, 1]
                ) * 1.0e3
                # Interpolate scan displacement at the same terminal stop
                # planes as the original rays. NaN line separators survive.
                projected += np.interp(z, branch.z, offsets_mm)
        return z, projected

    def set_ray_display_cache_limit_bytes(self, limit: int) -> None:
        """Bound display-array retention; zero disables it, not ray drawing."""
        if isinstance(limit, bool) or int(limit) != limit or int(limit) < 0:
            raise ValueError("Ray display cache limit must be a non-negative integer")
        self._ray_display_cache_budget = int(limit)
        while self._ray_display_cache and self._ray_display_cache_bytes > int(limit):
            _, (_, _, size) = self._ray_display_cache.popitem(last=False)
            self._ray_display_cache_bytes -= size

    def ray_display_cache_info(self) -> dict[str, int]:
        return {
            "budget_bytes": self._ray_display_cache_budget,
            "used_bytes": self._ray_display_cache_bytes,
            "entries": len(self._ray_display_cache),
            "hits": self._ray_display_cache_hits,
            "misses": self._ray_display_cache_misses,
        }

    def _ray_display_cache_get(self, key, sources):
        entry = self._ray_display_cache.get(key)
        if entry is not None:
            references, arrays, size = entry
            if all(reference() is source for reference, source in zip(references, sources)):
                self._ray_display_cache.move_to_end(key)
                self._ray_display_cache_hits += 1
                return arrays
            self._ray_display_cache_bytes -= size
            del self._ray_display_cache[key]
        self._ray_display_cache_misses += 1
        return None

    def _ray_display_cache_put(self, key, sources, arrays) -> None:
        size = sum(array.nbytes for array in arrays)
        if size > self._ray_display_cache_budget or not self._ray_display_cache_budget:
            return
        # Weak references guard against Python id reuse without retaining
        # gigabytes of physical ray histories outside the solver cache budget.
        try:
            references = tuple(weakref.ref(source) for source in sources)
        except TypeError:
            return
        previous = self._ray_display_cache.pop(key, None)
        if previous is not None:
            self._ray_display_cache_bytes -= previous[2]
        while self._ray_display_cache and self._ray_display_cache_bytes + size > self._ray_display_cache_budget:
            _, (_, _, old_size) = self._ray_display_cache.popitem(last=False)
            self._ray_display_cache_bytes -= old_size
        for array in arrays:
            array.setflags(write=False)
        self._ray_display_cache[key] = (references, arrays, size)
        self._ray_display_cache_bytes += size

    @staticmethod
    def _canonical_interaction_kind(kind: str) -> str:
        aliases = {
            "diffraction_spot": "diffraction_spots",
            "isotropic_ring": "diffuse_ring",
        }
        value = str(kind or "unknown").strip().lower()
        return aliases.get(value, value)

    def _branch_interaction_kind(self, branch) -> str:
        kind = self._canonical_interaction_kind(
            getattr(branch, "interaction_kind", "unknown")
        )
        if kind != "unknown":
            return kind
        name = str(getattr(branch, "name", "")).strip().lower()
        if name == "incident":
            return "incident"
        if name == "000":
            return "transmitted"
        if name in {"+g", "-g", "virtual_+g", "virtual_-g"}:
            return "diffraction_spots"
        return "unknown"

    def _sample_convergence_semiangles_mrad(
        self, branch, indices=None
    ) -> np.ndarray:
        """Return each ray's 3-D angle to its branch chief ray at sample Z.

        The common interaction kick is removed by measuring every outgoing
        branch relative to its own weighted chief ray.  Hue can therefore
        encode interaction type independently of convergence brightness.
        """

        tx_history = np.asarray(getattr(branch, "tx", ()), dtype=float)
        ty_history = np.asarray(getattr(branch, "ty", ()), dtype=float)
        if (
            tx_history.ndim != 2
            or ty_history.shape != tx_history.shape
            or tx_history.shape[1] == 0
        ):
            size = 0 if indices is None else np.asarray(indices).size
            return np.full(size, np.nan, dtype=float)
        row = -1 if self._branch_interaction_kind(branch) == "incident" else 0
        tx = tx_history[row].copy()
        ty = ty_history[row].copy()
        # Real inelastic populations sample an azimuthal characteristic-angle
        # ring within one branch. Remove that recorded interaction kick before
        # encoding illumination convergence as brightness.
        if row == 0:
            interaction_kick_x = getattr(
                branch, "interaction_kick_x_rad", None
            )
            interaction_kick_y = getattr(
                branch, "interaction_kick_y_rad", None
            )
            if interaction_kick_x is not None:
                kick_x = np.asarray(interaction_kick_x, dtype=float)
                if kick_x.shape == tx.shape:
                    tx -= kick_x
            if interaction_kick_y is not None:
                kick_y = np.asarray(interaction_kick_y, dtype=float)
                if kick_y.shape == ty.shape:
                    ty -= kick_y
        directions = np.stack((tx, ty, np.ones_like(tx)), axis=1)
        norms = np.linalg.norm(directions, axis=1)
        valid = np.isfinite(directions).all(axis=1) & (norms > 0.0)
        z_history = np.asarray(getattr(branch, "z", ()), dtype=float)
        blocked_z = np.asarray(
            getattr(branch, "blocked_z", ()), dtype=float
        )
        if (
            z_history.ndim == 1
            and z_history.size
            and blocked_z.shape == tx.shape
        ):
            sample_z = float(z_history[row])
            # Rays stopped upstream have no sample-plane convergence. Rays
            # stopped later did reach the sample and remain valid here.
            valid &= np.isnan(blocked_z) | (
                blocked_z >= sample_z - 1.0e-9
            )
        result = np.full(tx.size, np.nan, dtype=float)
        if np.any(valid):
            directions = directions[valid] / norms[valid, None]
            raw_weights = np.asarray(
                getattr(branch, "ray_weight", np.ones(tx.size)),
                dtype=float,
            )
            if raw_weights.shape != tx.shape:
                raw_weights = np.ones(tx.size, dtype=float)
            weights = raw_weights[valid]
            weights = np.where(
                np.isfinite(weights) & (weights >= 0.0), weights, 0.0
            )
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones_like(weights)
            weights /= np.sum(weights)
            chief = np.sum(weights[:, None] * directions, axis=0)
            chief_norm = float(np.linalg.norm(chief))
            if chief_norm > 0.0 and np.isfinite(chief_norm):
                chief /= chief_norm
                dot = np.clip(directions @ chief, -1.0, 1.0)
                cross = np.linalg.norm(
                    np.cross(directions, chief[None, :]), axis=1
                )
                result[valid] = np.arctan2(cross, dot) * 1.0e3
        if indices is None:
            return result
        return result[np.asarray(indices, dtype=int)]

    def _convergence_reference_mrad(self, simulation) -> float:
        metric = float(
            getattr(simulation, "metrics", {}).get(
                "sample_convergence_99_mrad", float("nan")
            )
        )
        if np.isfinite(metric) and metric > 0.0:
            return metric
        angles = self._sample_convergence_semiangles_mrad(
            simulation.incident
        )
        finite = angles[np.isfinite(angles)]
        if finite.size == 0:
            return 0.0
        fallback = float(np.percentile(finite, 99.0))
        return fallback if np.isfinite(fallback) and fallback > 0.0 else 0.0

    @staticmethod
    def _shade_colour(base_colour, normalised_angle: float) -> tuple[int, int, int]:
        """Keep hue fixed while mapping low-to-high convergence dark-to-light."""

        channels = np.asarray(base_colour, dtype=float).reshape(-1)[:3]
        if channels.size != 3 or not np.all(np.isfinite(channels)):
            channels = np.asarray((0.89, 0.91, 0.94), dtype=float)
        if float(np.max(channels)) <= 1.0:
            channels = channels * 255.0
        level = float(np.clip(normalised_angle, 0.0, 1.0))
        brightness = 0.42 + 0.58 * level
        return tuple(
            int(round(value))
            for value in np.clip(channels * brightness, 0.0, 255.0)
        )

    def _ray_colour_groups(self, bundles, reference_mrad: float):
        """Group displayed rays by interaction hue and convergence shade."""

        groups = {}
        kind_order = []
        base_colours = {}
        bin_count = self.CONVERGENCE_SHADE_BINS
        for branch in bundles:
            indices = self._display_ray_indices(branch)
            if indices.size == 0:
                continue
            kind = self._branch_interaction_kind(branch)
            if kind not in base_colours:
                kind_order.append(kind)
                base_colours[kind] = getattr(
                    branch, "colour", (0.89, 0.91, 0.94)
                )
            angles = self._sample_convergence_semiangles_mrad(
                branch, indices
            )
            if reference_mrad > 0.0:
                normalised = np.clip(
                    np.nan_to_num(
                        angles / reference_mrad,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ),
                    0.0,
                    1.0,
                )
            else:
                normalised = np.zeros(indices.size, dtype=float)
            bins = np.minimum(
                np.floor(normalised * bin_count).astype(int),
                bin_count - 1,
            )
            for bin_index in np.unique(bins):
                selected = indices[bins == bin_index]
                groups.setdefault((kind, int(bin_index)), []).append(
                    (branch, selected)
                )
        return kind_order, base_colours, groups

    def _ray_record_lines(self, payload) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild one grouped plot item after projection or scan changes."""

        segments = (
            ((payload, None),)
            if hasattr(payload, "x")
            else tuple(payload)
        )
        z_parts = []
        transverse_parts = []
        for branch, indices in segments:
            z, transverse = self._display_bundle_lines(branch, indices)
            if z.size:
                z_parts.append(z)
                transverse_parts.append(transverse)
        if not z_parts:
            return np.array([], dtype=float), np.array([], dtype=float)
        if len(z_parts) == 1:
            return z_parts[0], transverse_parts[0]
        return np.concatenate(z_parts), np.concatenate(transverse_parts)

    def _redraw_last_result(self) -> None:
        self._projection_redraw_timer.stop()
        self._projection_finalize_timer.stop()
        if self._last_result is not None:
            self._draw_ray_diagram(
                self._last_result,
                self._last_quality,
                preserve_view=True,
            )

    def _set_sample_region_result(
        self, result, *, mark_calculated: bool = True
    ) -> None:
        self._sample_region_result = result
        if self._high_accuracy_result is not None:
            self._high_accuracy_result.sample_region = result
            calculated = set(
                getattr(self._high_accuracy_result, "calculated_products", ())
            )
            reused = set(
                getattr(self._high_accuracy_result, "reused_products", ())
            )
            if result is None:
                calculated.discard("sample_region")
                reused.discard("sample_region")
                if getattr(
                    self._high_accuracy_result,
                    "specimen_exit",
                    None,
                ) is None:
                    calculated.discard("sample_downstream")
                    reused.discard("sample_downstream")
            elif mark_calculated:
                metrics = getattr(result, "metrics", {})
                checkpoint = getattr(result, "specimen_exit", None)
                if checkpoint is not None:
                    self._high_accuracy_result.specimen_exit = checkpoint
                region_signature = str(
                    metrics.get("sample_region_signature", "")
                )
                downstream_signature = str(
                    metrics.get("sample_downstream_signature", "")
                )
                if region_signature or downstream_signature:
                    signatures = dict(
                        getattr(self._high_accuracy_result, "signatures", {})
                        or {}
                    )
                    if region_signature:
                        signatures["sample_region"] = region_signature
                    if downstream_signature:
                        signatures["sample_downstream"] = (
                            downstream_signature
                        )
                    self._high_accuracy_result.signatures = signatures
                calculated.add("sample_region")
                reused.discard("sample_region")
                if checkpoint is not None:
                    calculated.add("sample_downstream")
                    reused.discard("sample_downstream")
            self._high_accuracy_result.calculated_products = frozenset(
                calculated
            )
            self._high_accuracy_result.reused_products = frozenset(reused)
        self.sample_interactions_3d.set_sample_region_result(result)
        self._update_sample_region_control_availability()
        self.calculation_artifacts_changed.emit(self._high_accuracy_result)

    def _set_specimen_interactions(self, interactions) -> None:
        if self._high_accuracy_result is None:
            return
        self._high_accuracy_result.specimen_interactions = interactions
        metrics = getattr(interactions, "metrics", {}) or {}
        dependencies = metrics.get("dependency_signatures", {})
        signatures = dict(
            getattr(self._high_accuracy_result, "signatures", {}) or {}
        )
        calculated = set(
            getattr(self._high_accuracy_result, "calculated_products", ())
        )
        reused = set(
            getattr(self._high_accuracy_result, "reused_products", ())
        )
        observable_stages = {
            "elastic_transport": ("elastic",),
            "characteristic_x_ray": ("eds",),
            "coherent_elastic_wave": ("wave", "wave_source"),
        }
        calculated_observables = set(
            metrics.get("calculated_observables_this_call", ())
        )
        reused_observables = set(metrics.get("reused_observables", ()))
        for observable, stage_keys in observable_stages.items():
            for stage_key in stage_keys:
                signature = str(dependencies.get(stage_key, ""))
                if signature:
                    signatures[stage_key] = signature
                if observable in calculated_observables:
                    calculated.add(stage_key)
                    reused.discard(stage_key)
                elif observable in reused_observables:
                    reused.add(stage_key)
                    calculated.discard(stage_key)
        self._high_accuracy_result.signatures = signatures
        self._high_accuracy_result.calculated_products = frozenset(calculated)
        self._high_accuracy_result.reused_products = frozenset(reused)
        self.sample_interactions_3d.display_result(
            self._high_accuracy_result
        )
        self.calculation_artifacts_changed.emit(self._high_accuracy_result)

    def _update_sample_region_control_availability(self) -> None:
        """Keep the 3-D page's request bound to the shared EDS result."""
        available = self._sample_region_result is not None
        runnable = self.eds_page.sample_region_calculation_available()
        self.sample_interactions_3d.calculate_paths.setEnabled(available or runnable)

    def _ensure_sample_region_result(self) -> bool:
        if self._sample_region_result is not None:
            return True
        calculated = self.eds_page.calculate_sample_region()
        self._update_sample_region_control_availability()
        return bool(calculated and self._sample_region_result is not None)

    def _update_projection_text(self) -> None:
        scan_text = ""
        if self._scan_ray_paths is not None:
            if self._scan_playback_time_s is None:
                scan_text = " | scan frame cached"
            else:
                period_s = max(
                    float(self._scan_ray_paths.frame_period_s),
                    1.0e-12,
                )
                phase = (self._scan_playback_time_s / period_s) % 1.0
                line_position = phase * int(self._scan_ray_paths.pixels_y)
                line = min(
                    int(line_position),
                    int(self._scan_ray_paths.pixels_y) - 1,
                )
                column = min(
                    int(
                        (line_position - line)
                        * int(self._scan_ray_paths.pixels_x)
                    ),
                    int(self._scan_ray_paths.pixels_x) - 1,
                )
                status = "playing" if self._scan_playback_active else "held"
                scan_text = (
                    f" | scan {status} pixel {column + 1}, line {line + 1}"
                )
        simulation = getattr(self._last_result, "simulation", None)
        tuning = bool((getattr(simulation, "metrics", None) or {}).get("optical_tuning", False))
        tuning_text = " | optical tuning only; no specimen signals" if tuning else ""
        self.heading.setText(
            f"Electron ray paths — {self._last_quality} | "
            f"{self._projection_axis_name()} projection at "
            f"{self._format_angle(self._projection_angle_deg)}° | "
            f"{self._crossover_count} crossovers | "
            f"{self._wall_stop_count} column-wall stops"
            f"{scan_text}{tuning_text}"
        )
        if self._selected_z_mm is None:
            self.stop_detail.setText(
                f"Projection: {self._projection_axis_name()} = "
                "X cos(angle) + Y sin(angle) | "
                "click a stop marker for exact X/Y diagnostics"
            )

    def _redraw_projection_items(self) -> None:
        """Update angle-dependent graphics without rebuilding static markers."""

        if self._last_result is None:
            return
        for item, payload in self._ray_bundle_records:
            z, transverse = self._ray_record_lines(payload)
            item.setData(z, transverse, connect="finite")
        if getattr(self, "_tuning_envelopes", ()):
            from temsim.physics.optical_tuning import projected_support
            for lower, upper, branch in self._tuning_envelopes:
                lo, hi = projected_support(branch, self._projection_angle_deg)
                lower.setData(branch.z, lo, connect="finite")
                upper.setData(branch.z, hi, connect="finite")
        for item, group, records in self._stop_projection_records:
            projected_mm = self._project_transverse(
                [record.x_mm for record in records],
                [record.y_mm for record in records],
            )
            item.setData(
                x=[record.z_mm for record in records],
                y=projected_mm,
                data=records,
            )
            item.setToolTip(
                f"{group}: projected on {self._projection_axis_name()}; "
                "click a marker for exact X/Y diagnostics"
            )
        self._redraw_component_projection_items()
        self._update_projection_text()

    def _redraw_component_projection_items(self) -> None:
        """Refresh only small projected aperture/detector graphics."""
        updated_spans = []
        for lower, upper, offset_x_mm, offset_y_mm, radius_mm in (
            self._aperture_projection_records
        ):
            centre_u_mm = float(
                self._project_transverse(offset_x_mm, offset_y_mm)
            )
            updated_spans.append(
                (lower, upper, centre_u_mm, radius_mm)
            )
            # Keep the numeric opening tooltip in sync without replacing
            # the aperture or its user-visible label.
            tooltip = lower.toolTip().split("\nAllowed ")[0]
            tooltip += (
                f"\nAllowed {self._projection_axis_name()} opening = "
                f"[{centre_u_mm - radius_mm:.6g}, {centre_u_mm + radius_mm:.6g}] mm\n"
                f"Circular radius = {radius_mm:.6g} mm\n"
                f"X/Y offset = {offset_x_mm:.6g} / {offset_y_mm:.6g} mm\n"
                "The blank gap is the circular opening projected onto the "
                "selected transverse axis; solid segments block."
            )
            lower.setToolTip(tooltip)
            upper.setToolTip(tooltip)
            upper.label.setToolTip(tooltip)
        if updated_spans:
            self._aperture_span_records = updated_spans
            self._update_aperture_spans()
        for item in self.recording_surface_range_items:
            basis = getattr(item, "_projection_basis", None)
            if basis is None:
                continue
            z_values, radial_values, offset_x, offset_y, interval_starts = basis
            y_values = radial_values + float(self._project_transverse(offset_x, offset_y))
            item.setData(z_values, y_values, connect="finite")
            item.active_intervals_mm = tuple(
                (float(y_values[start]), float(y_values[start + 1]))
                for start in interval_starts
            )

    def _finalize_projection(self) -> None:
        """Finish at the exact requested angle without clearing the scene."""
        if self._projection_redraw_timer.isActive():
            self._projection_redraw_timer.stop()
            self._redraw_projection_items()
        self._update_scale_notice()

    def _prepare_scan_ray_playback(self, result) -> None:
        self._scan_ray_paths = getattr(result, "scan_ray_paths", None)
        self._scan_ray_offsets_m = {}
        self._scan_playback_active = False
        self._scan_playback_time_s = None

    def _scan_playback_active_changed(self, active: bool) -> None:
        self._scan_playback_active = bool(active)
        self._update_projection_text()

    def _scan_playback_time_changed(self, time_s: float) -> None:
        paths = self._scan_ray_paths
        result = self._last_result
        if paths is None or result is None:
            return
        state = getattr(result, "state_snapshot", None)
        if state is None:
            return
        ac_command_mrad = np.asarray(
            state.ac_deflector.scan_kick_mrad(float(time_s)),
            dtype=float,
        )
        descan = state.descan_deflector
        descan_command_mrad = np.asarray(
            (
                descan.scan_kick_mrad(float(time_s))
                if bool(descan.enabled and descan.scan_enabled)
                else (0.0, 0.0)
            ),
            dtype=float,
        )
        delta_ac_rad = (
            ac_command_mrad
            - np.asarray(paths.baseline_ac_command_mrad, dtype=float)
        ) * 1.0e-3
        delta_descan_rad = (
            descan_command_mrad
            - np.asarray(paths.baseline_descan_command_mrad, dtype=float)
        ) * 1.0e-3
        self._scan_ray_offsets_m = {
            str(name): (
                np.einsum("zij,j->zi", ac_response, delta_ac_rad)
                + np.einsum(
                    "zij,j->zi",
                    descan_response,
                    delta_descan_rad,
                )
            )
            for name, (ac_response, descan_response) in (
                paths.responses_m_per_rad.items()
            )
        }
        self._scan_playback_time_s = float(time_s)
        for item, payload in self._ray_bundle_records:
            z_values, transverse = self._ray_record_lines(payload)
            item.setData(z_values, transverse, connect="finite")
        self._update_projection_text()

    @staticmethod
    def _project_transverse_values(x, y, angle_deg: float) -> np.ndarray:
        """Project X/Y values onto a transverse axis rotated about Z."""
        return project_transverse_values(x, y, angle_deg)

    def _project_transverse(self, x, y) -> np.ndarray:
        return self._project_transverse_values(
            x, y, self._projection_angle_deg
        )

    def _projection_axis_name(self) -> str:
        return projection_axis_name(self._projection_angle_deg)

    @staticmethod
    def _format_angle(angle_deg: float) -> str:
        return format_projection_angle(angle_deg)

    def _projection_slider_changed(self, value: int) -> None:
        if not self._projection_syncing:
            self._set_projection_angle(
                float(value) / 10.0,
                defer_redraw=True,
            )

    def _set_projection_angle(
        self, angle_deg: float, *, defer_redraw: bool = False
    ) -> None:
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
            normalized = angle % 360.0
            self.projection_xz.setChecked(
                bool(np.isclose(normalized, 0.0, atol=0.05))
            )
            self.projection_yz.setChecked(
                bool(np.isclose(normalized, 90.0, atol=0.05))
            )
        finally:
            self._projection_syncing = False
        self.transverse_beam.set_projection_angle(
            angle, redraw=(self.transverse_beam.isVisible()
                           and self.transverse_beam not in self._pending_ray_panels)
        )
        if changed and not self.transverse_beam.isVisible() and self._last_result is not None:
            self._pending_ray_panels[self.transverse_beam] = self._last_result
        if changed and self._last_result is not None:
            if defer_redraw:
                if not self._projection_redraw_timer.isActive():
                    self._projection_redraw_timer.start()
                self._projection_finalize_timer.start()
            else:
                self._projection_redraw_timer.stop()
                self._projection_finalize_timer.stop()
                self._redraw_projection_items()
                self._update_scale_notice()

    def _auto_zoom_toggled(self, checked: bool) -> None:
        if checked and self._focused_part is not None:
            self._apply_component_zoom(self._focused_part)

    def focus_component(self, part) -> None:
        """Remember the selected part and optionally focus its optical region."""
        self._focused_part = part
        self._pending_ray_focus.update((self.physical_layout, self.magnetic_field))
        self._focus_transverse("component", part)
        self._refresh_visible_ray_panels()
        if self._last_result is not None:
            self._highlight_ray_component(part)
        if self.auto_zoom.isChecked() and self._last_result is not None:
            self._apply_component_zoom(part)

    def _set_ray_panel_visible(self, panel, visible: bool) -> None:
        self.ray_layout_changing.emit()
        panel.setVisible(visible)
        self._refresh_visible_ray_panels()
        self.ray_layout_changed.emit()

    def show_ray_diagram(self) -> None:
        self.tabs.setCurrentWidget(self.ray_page)
        self.ray_result_tabs.setCurrentWidget(self.ray_workspace_splitter)
        self._refresh_visible_ray_panels()

    def reveal_component(self, part, *, preferred_view: str = "ray") -> str:
        """Show and centre a tree-activated component in a visual page."""

        current_page = self.tabs.currentWidget()
        use_physical = (
            preferred_view == "physical"
            or current_page is self.physical_layout
        )
        self.focus_component(part)
        if use_physical:
            self.tabs.setCurrentWidget(self.physical_layout)
            self._refresh_visible_ray_panels()
            self.physical_layout.reveal_component(part)
            return "Physical Layout"

        self.show_ray_diagram()
        self._highlight_ray_component(part)
        self._apply_component_zoom(part)
        return "Ray Diagram"

    def _highlight_ray_component(self, part) -> None:
        """Draw a persistent selected-component band independent of labels."""
        centre = float(part.center_z_mm)
        half_width = max(
            0.5 * abs(float(part.end_z_mm) - float(part.start_z_mm)),
            0.5,
        )
        highlight = self._ray_component_highlight
        if highlight is None:
            highlight = pg.LinearRegionItem(
                values=(centre - half_width, centre + half_width),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(250, 204, 21, 42),
                pen=pg.mkPen("#facc15", width=1.2),
            )
            self.plot.addItem(highlight)
        else:
            highlight.setRegion((centre - half_width, centre + half_width))
        highlight.setZValue(24)
        highlight.setToolTip(
            f"Selected: {part.name}\nCentre Z = {centre:.6g} mm"
        )
        highlight.component_key = str(part.key)
        self._ray_component_highlight = highlight

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
        cursor.sigPositionChanged.connect(self._axial_cursor_moved)
        self.plot.addItem(cursor)
        self.axial_cursor_item = cursor

    def _axial_cursor_moved(self, cursor) -> None:
        """Preview the transverse slice continuously while the cursor moves."""

        self._focus_transverse("z", float(cursor.value()))

    def _axial_cursor_move_finished(self, cursor) -> None:
        self.jump_to_ray_position(float(cursor.value()))

    @staticmethod
    def _interaction_percent(value: float) -> str:
        percentage = max(float(value), 0.0) * 100.0
        if percentage == 0.0:
            return "0"
        if percentage < 0.001:
            return f"{percentage:.3e}"
        return f"{percentage:.6g}"

    def _update_interaction_detail(self) -> None:
        if self._last_result is None or self._selected_z_mm is None:
            self.interaction_detail.setHtml(
                "<b>Selected-plane interaction budget</b><br>"
                "Choose an axial Z position to calculate source-normalised "
                "interaction fractions."
            )
            self.interaction_detail.setToolTip("")
            return
        try:
            from temsim.physics.interaction_budget import (
                plane_interaction_budget,
            )

            budget = plane_interaction_budget(
                self._last_result, self._selected_z_mm
            )
        except Exception as exc:
            self.interaction_detail.setHtml(
                "<b>Selected-plane interaction budget unavailable</b><br>"
                f"<span style='color:#fca5a5'>{escape(str(exc))}</span>"
            )
            self.interaction_detail.setToolTip(str(exc))
            return

        rows = []
        for channel in budget.channels:
            loss = channel.representative_loss_ev
            loss_text = (
                "—"
                if loss is None
                else "0 (zero loss)"
                if abs(float(loss)) <= 1.0e-15
                else f"{float(loss):.6g} eV"
            )
            rows.append(
                "<tr>"
                f"<td>{escape(channel.label)}</td>"
                f"<td align='right'>{self._interaction_percent(channel.probability_at_sample)}%</td>"
                f"<td align='right'>{self._interaction_percent(channel.source_fraction_at_plane)}%</td>"
                f"<td align='right'>{self._interaction_percent(channel.composition_at_plane)}%</td>"
                f"<td align='right'>{escape(loss_text)}</td>"
                "</tr>"
            )
        material = (
            f" | material {escape(budget.material_name)}"
            if budget.material_name else ""
        )
        physics = ""
        if budget.mean_inelastic_events is not None:
            mfp = budget.total_inelastic_mean_free_path_nm
            mfp_text = (
                f"{float(mfp):.6g} nm"
                if mfp is not None and np.isfinite(mfp)
                else "∞"
            )
            physics = (
                "<br><span style='color:#94a3b8'>"
                f"Mean inelastic events t/λ = {float(budget.mean_inelastic_events):.6g}; "
                f"combined inelastic λ = {mfp_text}. "
                "Elastic diffraction is a coexisting coherent multislice "
                "intensity, not another exclusive row.</span>"
            )
        elastic_wave = ""
        wave_result = getattr(self._last_result, "wave_imaging", None)
        if wave_result is not None and budget.z_mm >= budget.sample_z_mm - 1.0e-9:
            wave_metrics = wave_result.metrics
            outside = float(
                wave_metrics.get(
                    "elastic_exit_intensity_outside_incident_cone_fraction",
                    0.0,
                )
            )
            baseline = float(
                wave_metrics.get(
                    "elastic_incident_baseline_outside_cone_fraction",
                    0.0,
                )
            )
            cone = float(
                wave_metrics.get("elastic_incident_cone_mrad", 0.0)
            )
            elastic_wave = (
                "<br><span style='color:#7dd3fc'>Elastic wave observable "
                "(non-exclusive, conditional zero-loss): exit intensity "
                f"outside incident α99={cone:.6g} mrad cone "
                f"{self._interaction_percent(outside)}%; incident-wave "
                f"baseline {self._interaction_percent(baseline)}%; "
                f"redistribution Δ {(outside - baseline) * 100.0:+.6g}%.</span>"
            )
        warnings = ""
        if budget.warnings:
            warnings = (
                "<br><span style='color:#fbbf24'>Model note: "
                f"{escape(budget.warnings[0])}</span>"
            )
        self.interaction_detail.setHtml(
            "<b>Selected-plane interaction budget</b> — "
            f"Z {budget.z_mm:.9g} mm ({escape(budget.location)}){material}<br>"
            f"Current reaching Z: <b>{self._interaction_percent(budget.source_fraction_at_plane)}% of source</b> | "
            f"sample incident {self._interaction_percent(budget.sample_incident_source_fraction)}%<br>"
            "<table cellspacing='2' cellpadding='2' width='100%'>"
            "<tr style='color:#94a3b8'><th align='left'>Interaction state</th>"
            "<th>at sample<br>% incident</th><th>at Z<br>% source</th>"
            "<th>at Z<br>composition</th><th>representative loss</th></tr>"
            + "".join(rows)
            + "</table>"
            "<span style='color:#94a3b8'>Not reaching this Z: pre-sample stops "
            f"{self._interaction_percent(budget.pre_sample_stopped_source_fraction)}%, "
            f"sample absorption/removal {self._interaction_percent(budget.sample_absorbed_source_fraction)}%, "
            f"downstream stops {self._interaction_percent(budget.downstream_stopped_source_fraction)}%. "
            f"Conservation error {budget.conservation_error:.3e}.</span>"
            + physics
            + elastic_wave
            + warnings
        )
        tooltip = [f"Model: {budget.model}"]
        real = getattr(
            self._last_result.simulation, "real_interactions", None
        )
        if real is not None:
            if real.reference:
                tooltip.append(f"Reference: {real.reference}")
            if real.applicability:
                tooltip.append(f"Applicability: {real.applicability}")
            tooltip.extend(real.warnings)
        self.interaction_detail.setToolTip("\n".join(tooltip))

    def jump_to_ray_position(
        self,
        z_mm: float,
        activate_tab: bool = True,
    ) -> None:
        """Select an axial Z without changing a user-defined plot range.

        Cursor, spin-box, and linked-view Z changes preserve both Ray Diagram
        axes exactly. Fit and component Auto zoom remain explicit view tools.
        """
        limits = self._simulation_x_limits()
        if limits is None or not np.isfinite(z_mm):
            return
        preserved_range = self.plot.getViewBox().viewRange()
        self.plot.disableAutoRange()
        lower_limit, upper_limit = limits
        selected = float(np.clip(z_mm, lower_limit, upper_limit))
        self._selected_z_mm = selected
        self._focus_transverse("z", selected)
        self.axial_position.setRange(lower_limit, upper_limit)
        self.axial_position.setValue(selected)
        if activate_tab:
            self.show_ray_diagram()

        if self.axial_cursor_item is None:
            self._add_axial_position_cursor()
        else:
            self.axial_cursor_item.setValue(selected)

        self.plot.getViewBox().setRange(
            xRange=tuple(float(value) for value in preserved_range[0]),
            yRange=tuple(float(value) for value in preserved_range[1]),
            padding=0.0,
            disableAutoRange=True,
        )
        self._update_component_label_visibility()
        self.stop_detail.setText(
            f"Selected axial position: Z {selected:.9g} mm | "
            "drag the cyan cursor or double-click another axial plot"
        )
        self._update_interaction_detail()

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
        """Replace intercept data, reusing each surviving scatter group."""
        self.stop_marker_items = []
        self._stop_projection_records = []
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
            key = (group, colour)
            item = self._stop_items_by_group.get(key)
            if item is None:
                item = pg.ScatterPlotItem(
                    symbol="x", size=9,
                    pen=pg.mkPen(colour, width=1.8), name=group,
                )
                item.sigClicked.connect(self._stop_marker_clicked)
                self.plot.addItem(item)
                self._stop_items_by_group[key] = item
            item.setData(
                x=[record.z_mm for record in group_records],
                y=projected_mm, data=group_records,
            )
            item.setZValue(30)
            item.setToolTip(
                f"{group}: projected on {self._projection_axis_name()}; "
                "click a marker for exact X/Y diagnostics"
            )
            self.stop_marker_items.append(item)
            self._stop_projection_records.append(
                (item, group, tuple(group_records))
            )
        for key in tuple(self._stop_items_by_group):
            if key not in groups:
                self.plot.removeItem(self._stop_items_by_group.pop(key))

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
            first = int(np.searchsorted(z_values, x_min, side="left"))
            last = int(np.searchsorted(z_values, x_max, side="right"))
            if first >= last or branch.tx.shape[1] == 0:
                continue
            sources = (branch.z, branch.tx, branch.ty, branch.blocked_z)
            source_key = tuple(id(value) for value in sources)
            angle_key = ("slope-max", source_key, self.MAX_RANGE_SAMPLE_RAYS,
                         float(self._projection_angle_deg) % 360.0)
            maxima = self._ray_display_cache_get(angle_key, sources)
            if maxima is None:
                basis_key = ("slopes", source_key, self.MAX_RANGE_SAMPLE_RAYS)
                basis = self._ray_display_cache_get(basis_key, sources)
                if basis is None:
                    ray_count = branch.tx.shape[1]
                    ray_indices = np.unique(np.linspace(
                        0, ray_count - 1,
                        min(ray_count, self.MAX_RANGE_SAMPLE_RAYS), dtype=int,
                    ))
                    tx = np.asarray(branch.tx[:, ray_indices], dtype=float).copy()
                    ty = np.asarray(branch.ty[:, ray_indices], dtype=float).copy()
                    blocked_z = np.asarray(branch.blocked_z, dtype=float)[ray_indices]
                    valid_to_stop = (
                        np.isnan(blocked_z)[None, :]
                        | (z_values[:, None] <= blocked_z[None, :])
                    )
                    tx[~valid_to_stop] = np.nan
                    ty[~valid_to_stop] = np.nan
                    basis = (tx, ty)
                    self._ray_display_cache_put(basis_key, sources, basis)
                slopes = self._project_transverse(*basis)
                slopes = np.abs(slopes)
                slopes[~np.isfinite(slopes)] = 0.0
                maxima = (np.max(slopes, axis=1),)
                self._ray_display_cache_put(angle_key, sources, maxima)
            maximum = max(maximum, float(np.max(maxima[0][first:last])))
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
        if self._convergence_colour_reference_mrad > 0.0:
            colour_text = (
                "Hue = interaction type | shade = sample convergence "
                "(dark α≈0 → bright at/above α99 "
                f"{self._convergence_colour_reference_mrad:.4g} mrad)"
            )
        else:
            colour_text = (
                "Hue = interaction type | convergence shade unavailable"
            )
        self.hint.setText(
            f"Max physical {self._projection_axis_name()} angle: "
            f"{maximum_angle_deg:.3g}° | "
            f"Transverse display: {magnification_text} (angles not to scale) | "
            f"{colour_text} | Blocked rays stop at first intercept | "
            "Column wall uses radial X/Y"
        )

    def _apply_component_zoom(self, part) -> None:
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
            if "diameter_mm" in record:
                radius_mm = 0.5 * max(
                    0.0, float(record["diameter_mm"])
                )
            else:
                # Compatibility with pre-diameter calculation records.
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
            self._aperture_projection_records.append(
                (
                    lower,
                    upper,
                    offset_x_mm,
                    offset_y_mm,
                    radius_mm,
                )
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

    @staticmethod
    def _recording_plane_component(result, key: str):
        state = getattr(result, "state_snapshot", None)
        return next(
            (
                component
                for component in getattr(state, "recording_planes", ())
                if str(component.key) == str(key)
            ),
            None,
        )

    def _add_recording_surface_range(self, result, part, index: int) -> None:
        """Draw the finite detector acceptance in its physical Z plane."""

        component = self._recording_plane_component(result, part.key)
        signal_z_mm = (
            float(component.z_mm)
            if component is not None
            else self._aperture_optical_plane(part)
        )
        inserted = bool(getattr(component, "inserted", True))
        readout_enabled = getattr(component, "readout_enabled", None)
        outer_radius_mm = 0.5 * float(
            getattr(
                component,
                "outer_width_mm",
                part.data.get(
                    "outer_width_mm",
                    part.data.get("mechanical_outer_diameter_mm", 0.0),
                ),
            )
        )
        inner_radius_mm = 0.5 * float(
            getattr(
                component,
                "inner_diameter_mm",
                part.data.get("inner_diameter_mm", 0.0),
            )
        )
        centre_u_mm = float(
            self._project_transverse(
                getattr(component, "centre_offset_x_mm", 0.0),
                getattr(component, "centre_offset_y_mm", 0.0),
            )
        )
        if inner_radius_mm > 0.0:
            x_values = np.array(
                [signal_z_mm, signal_z_mm, np.nan, signal_z_mm, signal_z_mm]
            )
            y_values = np.array(
                [
                    centre_u_mm - outer_radius_mm,
                    centre_u_mm - inner_radius_mm,
                    np.nan,
                    centre_u_mm + inner_radius_mm,
                    centre_u_mm + outer_radius_mm,
                ]
            )
            radial_text = (
                f"{inner_radius_mm:.6g} to {outer_radius_mm:.6g} mm "
                "on both sides of the detector axis"
            )
        else:
            x_values = np.array([signal_z_mm, signal_z_mm])
            y_values = np.array(
                [
                    centre_u_mm - outer_radius_mm,
                    centre_u_mm + outer_radius_mm,
                ]
            )
            radial_text = (
                f"0 to {outer_radius_mm:.6g} mm about the detector axis"
            )

        colour = str(getattr(component, "colour", "#facc15"))
        range_pen = pg.mkPen(
            colour,
            width=3.0 if inserted else 1.5,
            style=(
                Qt.PenStyle.SolidLine
                if inserted
                else Qt.PenStyle.DashLine
            ),
        )
        status = "INSERTED" if inserted else "RETRACTED"
        if readout_enabled is not None:
            status += " / READOUT ON" if readout_enabled else " / READOUT OFF"
        range_kind = "Active range" if inserted else "Parked reference range"
        tooltip = (
            f"{part.name} ({status.lower()})\n"
            f"Detection plane Z = {signal_z_mm:.6g} mm\n"
            f"{range_kind}: {radial_text}"
        )
        range_item = self.plot.plot(
            x_values,
            y_values,
            pen=range_pen,
            connect="finite",
        )
        range_item.setZValue(16 if inserted else 7)
        range_item.setToolTip(tooltip)
        # These attributes make the physical drawing semantics inspectable in
        # GUI regression tests without coupling them to pyqtgraph internals.
        range_item.recording_plane_key = str(part.key)
        range_item.recording_plane_inserted = inserted
        range_item.active_intervals_mm = tuple(
            (float(y_values[start]), float(y_values[start + 1]))
            for start in (
                (0, 3) if inner_radius_mm > 0.0 else (0,)
            )
        )
        range_item._projection_basis = (
            x_values, y_values - centre_u_mm,
            float(getattr(component, "centre_offset_x_mm", 0.0)),
            float(getattr(component, "centre_offset_y_mm", 0.0)),
            (0, 3) if inner_radius_mm > 0.0 else (0,),
        )
        self.recording_surface_range_items.append(range_item)

        # Use an invisible axial carrier for a view-anchored label. The only
        # visible detector geometry is the finite solid/dashed range above.
        label_anchor = pg.InfiniteLine(
            pos=signal_z_mm,
            angle=90,
            pen=pg.mkPen(None),
            label=f"{part.name} [{status}]",
            labelOpts={
                "position": 0.76 + 0.07 * (index % 4),
                "color": colour,
                "rotateAxis": (1, 0),
            },
        )
        label_anchor.setZValue(17)
        label_anchor.setToolTip(tooltip)
        self._register_ray_label(label_anchor.label)
        label_anchor.label.setToolTip(tooltip)
        self.plot.addItem(label_anchor)
        self.component_marker_items.append(label_anchor)
        self._component_labels.append(
            (label_anchor.label, signal_z_mm, False)
        )

    def _add_component_marker(self, result, part, index: int) -> None:
        """Build one independently invalidated component layer."""
        if bool(part.data.get("branch_path_only", False)):
            return
        # The specimen plane is drawn independently and remains visible
        # when generic component-centre markers are hidden.
        if part.key == "sample":
            return
        is_lens = "lens" in part.key
        # Only modeled optical stops receive adjustable opening markers.
        is_aperture = (
            "aperture" in part.key
            and not bool(part.data.get("mechanical_only", False))
        )
        if is_aperture:
            self._add_aperture_component(part, index)
            return
        is_recording_surface = part.data.get("mechanical_profile") in {
            "retractable_detector_plane",
            "camera_sensor_plane",
        }
        if is_recording_surface:
            self._add_recording_surface_range(result, part, index)
            return
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
            return
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
        state_snapshot = getattr(result, "state_snapshot", None)
        sample_state = getattr(state_snapshot, "sample", None)
        inserted = bool(getattr(sample_state, "inserted", True))
        specimen_mode = str(
            getattr(sample_state, "specimen_mode", "atomic")
        ).strip().lower()
        if inserted:
            label = f"SAMPLE / SPECIMEN  Z={sample_z_mm:.6g} mm"
            if specimen_mode == "virtual":
                tooltip = (
                    "Virtual sample plane (inserted)\n"
                    f"Exact axial position Z = {sample_z_mm:.9g} mm\n"
                    "The selected TOML reference supplies ideal wave/EDS "
                    "material structure. Explicit user-defined interaction "
                    "channels start here in the Ray Diagram. "
                    "Ray hue identifies interaction type; brightness encodes "
                    "the ray's convergence semi-angle relative to that "
                    "branch's chief ray. Their common start is a continuous "
                    "ray boundary, not a second optical element."
                )
            else:
                tooltip = (
                    "Real sample plane (inserted)\n"
                    f"Exact axial position Z = {sample_z_mm:.9g} mm\n"
                    "Only the imported CIF/MCIF supplies specimen structure. "
                    "Ray Diagram adds no artificial +g/-g or diffuse "
                    "diffraction branches. Coherent elastic scattering is "
                    "calculated by wave/multislice; coloured energy-loss "
                    "paths are material-IMFP/Poisson quadrature for plasmon, "
                    "ionisation and plural inelastic transport. Brightness "
                    "still encodes convergence semi-angle. Their common "
                    "start remains a continuous ray boundary."
                )
            colour = "#ffffff"
            marker_brush = pg.mkBrush("#ef4444")
            line_style = Qt.PenStyle.SolidLine
        else:
            label = (
                "SAMPLE RETRACTED / REFERENCE PLANE  "
                f"Z={sample_z_mm:.6g} mm"
            )
            tooltip = (
                "Sample holder retracted\n"
                f"Optical probe-reference position Z = {sample_z_mm:.9g} mm\n"
                "No specimen diffraction, diffuse scattering, or atomistic "
                "potential is applied. The incident/outgoing split is only a "
                "continuous computational boundary."
            )
            colour = "#94a3b8"
            marker_brush = pg.mkBrush(0, 0, 0, 0)
            line_style = Qt.PenStyle.DashLine
        line = pg.InfiniteLine(
            pos=sample_z_mm,
            angle=90,
            pen=pg.mkPen(
                colour,
                width=2.6 if inserted else 1.6,
                style=line_style,
            ),
            label=label,
            labelOpts={
                "position": 0.94,
                "color": colour,
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
            pen=pg.mkPen(colour, width=2.0),
            brush=marker_brush,
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

    def ray_scene_info(self) -> dict[str, int | float]:
        """Rendering diagnostics; timings exclude solving and deferred painting."""
        return {
            "initial_builds": int(self._ray_scene_initialized),
            "updates": self._ray_scene_updates,
            "layers_rebuilt": self._ray_static_layers.rebuilt,
            "layers_reused": self._ray_static_layers.reused,
            "static_layers": len(self._ray_static_layers.layers),
            "ray_groups": len(self._ray_items_by_group),
            "last_update_ms": self._ray_scene_last_update_ms,
        }

    def _component_drawing_signature(self, result, part, index: int) -> tuple:
        """Snapshot only values used by a component's drawing, not its optics."""
        data = part.data
        signature = (
            str(part.key), str(part.name), index,
            float(part.start_z_mm), float(part.center_z_mm), float(part.end_z_mm),
            bool(data.get("mechanical_only", False)), data.get("mechanical_profile"),
            self._deflector_planes(part), self._aperture_optical_plane(part),
        )
        if "aperture" in part.key and not data.get("mechanical_only", False):
            record = self._aperture_stops_by_key.get(part.key)
            signature += (None if record is None else tuple(
                record.get(key) for key in (
                    "z_mm", "enabled", "installed", "diameter_mm", "radius_mm",
                    "offset_x_mm", "offset_y_mm",
                )
            ),)
        if data.get("mechanical_profile") in {
            "retractable_detector_plane", "camera_sensor_plane",
        }:
            component = self._recording_plane_component(result, part.key)
            signature += (
                tuple(getattr(component, key, None) for key in (
                    "z_mm", "inserted", "readout_enabled", "outer_width_mm",
                    "inner_diameter_mm", "centre_offset_x_mm",
                    "centre_offset_y_mm", "colour",
                )),
                data.get("outer_width_mm"), data.get("mechanical_outer_diameter_mm"),
                data.get("inner_diameter_mm"),
            )
        return signature

    def _sync_ray_static_layers(self, result) -> None:
        layers = self._ray_static_layers
        layers.begin(self)
        assembly = getattr(result, "assembly", None)
        segments = getattr(assembly, "vacuum_bore_segments", ())
        if self.column_walls.isChecked() and segments:
            signature = tuple(
                (float(item.start_z_mm), float(item.end_z_mm),
                 float(item.inner_diameter_mm)) for item in segments
            )
            layers.update(self, ("walls",), signature, lambda: self._add_column_walls(result))
        parts = getattr(assembly, "parts", ())
        if self.component_centres.isChecked():
            for index, part in enumerate(parts):
                if part.key == "sample" or part.data.get("branch_path_only", False):
                    continue
                layers.update(
                    self, ("component", str(part.key)),
                    self._component_drawing_signature(result, part, index),
                    lambda part=part, index=index: self._add_component_marker(result, part, index),
                )
        sample = getattr(getattr(result, "state_snapshot", None), "sample", None)
        sample_z = next(
            (float(part.center_z_mm) for part in parts if part.key == "sample"),
            float(result.simulation.incident.z[-1]),
        )
        layers.update(
            self, ("sample",),
            (sample_z, bool(getattr(sample, "inserted", True)),
             str(getattr(sample, "specimen_mode", "atomic")).strip().lower()),
            lambda: self._add_sample_marker(result),
        )
        if self.crossovers.isChecked():
            signature = tuple(
                (float(record["z_mm"]), float(record.get("rms_radius_mm", float("nan"))).hex(),
                 str(record.get("name", "Crossover")))
                for record in self._all_crossovers(result)
            )
            layers.update(
                self, ("crossovers",), signature,
                lambda: self._add_crossover_markers(result),
            )
        layers.finish(self)
        # A new result can arrive before the deferred angle timer fires.
        # Bring reused apparatus graphics to that angle, without redrawing rays.
        self._redraw_component_projection_items()
        if self.axial_cursor_item is not None:
            self._ray_label_items.append(self.axial_cursor_item.label)

    def _sync_tuning_envelopes(self, simulation, bundles) -> None:
        active = set()
        self._tuning_envelopes = []
        if (getattr(simulation, "metrics", None) or {}).get("tuning_quality") == "Medium":
            from temsim.physics.optical_tuning import projected_support
            for index, branch in enumerate(bundles):
                key = (index, str(branch.name))
                active.add(key)
                items = self._support_items_by_branch.get(key)
                if items is None:
                    lower = self.plot.plot([], [], pen=pg.mkPen("#b4e55f", width=.7))
                    upper = self.plot.plot([], [], pen=pg.mkPen("#b4e55f", width=.7))
                    fill = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(140, 200, 100, 20))
                    fill.setZValue(-5)
                    fill.setToolTip("Sampled support guide only; not density, signal or a guaranteed outer boundary.")
                    self.plot.addItem(fill)
                    items = (lower, upper, fill)
                    self._support_items_by_branch[key] = items
                lower, upper, _fill = items
                lo, hi = projected_support(branch, self._projection_angle_deg)
                lower.setData(branch.z, lo, connect="finite")
                upper.setData(branch.z, hi, connect="finite")
                self._tuning_envelopes.append((lower, upper, branch))
        for key in tuple(self._support_items_by_branch):
            if key not in active:
                lower, upper, fill = self._support_items_by_branch.pop(key)
                # Disconnect the fill's curve observers before releasing curves.
                lower.sigPlotChanged.disconnect(fill.curveChanged)
                upper.sigPlotChanged.disconnect(fill.curveChanged)
                for item in (fill, lower, upper):
                    self.plot.removeItem(item)

    def _sync_ray_curves(self, simulation, bundles) -> None:
        self._convergence_colour_reference_mrad = self._convergence_reference_mrad(simulation)
        kind_order, base_colours, colour_groups = self._ray_colour_groups(
            bundles, self._convergence_colour_reference_mrad,
        )
        for kind in kind_order:
            pen = pg.mkPen(self._shade_colour(base_colours[kind], 1.0), width=1.8)
            if kind not in self._ray_legend_items:
                label = self.INTERACTION_LABELS.get(kind, kind.replace("_", " ").title())
                self._ray_legend_items[kind] = self.plot.plot([], [], pen=pen, name=label)
            else:
                if self._ray_legend_items[kind].opts["pen"] != pen:
                    self._ray_legend_items[kind].setPen(pen)
        for kind in tuple(self._ray_legend_items):
            if kind not in kind_order:
                self.plot.removeItem(self._ray_legend_items.pop(kind))
        active = set()
        self._ray_bundle_records = []
        for key, segments in colour_groups.items():
            kind, bin_index = key
            payload = tuple(segments)
            z, transverse = self._ray_record_lines(payload)
            if not z.size:
                continue
            active.add(key)
            shade = bin_index / max(self.CONVERGENCE_SHADE_BINS - 1, 1)
            pen = pg.mkPen(self._shade_colour(base_colours[kind], shade), width=1.35)
            item = self._ray_items_by_group.get(key)
            if item is None:
                item = self.plot.plot([], [], pen=pen, connect="finite")
                self._ray_items_by_group[key] = item
            elif item.opts["pen"] != pen:
                item.setPen(pen)
            item.setData(z, transverse, connect="finite")
            label = self.INTERACTION_LABELS.get(kind, kind.replace("_", " ").title())
            if self._convergence_colour_reference_mrad > 0.0:
                detail = (
                    f"Convergence shade {bin_index + 1}/{self.CONVERGENCE_SHADE_BINS}; "
                    f"brightness saturates at α99 = {self._convergence_colour_reference_mrad:.6g} mrad"
                )
            else:
                detail = "Convergence shade unavailable"
            item.setToolTip(
                f"Interaction: {label}\n{detail}\n"
                "Semi-angle is measured relative to this branch's "
                "weighted chief ray at the sample plane."
            )
            self._ray_bundle_records.append((item, payload))
        for key in tuple(self._ray_items_by_group):
            if key not in active:
                self.plot.removeItem(self._ray_items_by_group.pop(key))

    def _draw_ray_diagram(
        self, result, quality: str, preserve_view: bool = False
    ) -> None:
        """Publish changed layers without clearing unchanged geometry or rays."""
        started = perf_counter()
        self._projection_redraw_timer.stop()
        self._projection_finalize_timer.stop()
        if result is not self._ray_display_cache_result:
            self._ray_display_cache.clear()
            self._ray_display_cache_bytes = 0
            self._ray_display_cache_result = result
        preserved_range = self.plot.getViewBox().viewRange() if preserve_view else None
        self.plot.disableAutoRange()
        if not self._ray_scene_initialized:
            self.plot.clear()  # Remove the initial waiting notice, once only.
            self._set_ray_axis_label("left", "Projected displacement")
            self._style_ray_axes()
            self._style_ray_legend(self.plot.addLegend(offset=(10, 10)))
            for label, colour, width in (
                ("Aperture", "#ffb000", 2.0),
                ("Deflector U/L", "#2cffad", 2.0),
                ("Sample plane", "#ffffff", 2.6),
            ):
                self.plot.plot([], [], pen=pg.mkPen(colour, width=width), name=label)
            self._ray_scene_initialized = True
        else:
            self._ray_scene_updates += 1
        self._aperture_stops_by_key = {
            str(record["key"]): dict(record)
            for record in getattr(result, "aperture_stops", ())
        }
        simulation = result.simulation
        bundles = [simulation.incident, *simulation.branches.values()]
        self._sync_tuning_envelopes(simulation, bundles)
        self._sync_ray_curves(simulation, bundles)
        self._add_stop_markers(simulation)
        self._sync_ray_static_layers(result)
        limits = self._simulation_x_limits()
        if limits is not None:
            self.axial_position.setRange(*limits)
            if self._selected_z_mm is not None:
                # A worker may publish between cursor-drag events and release.
                # Keep the displayed cursor position, not its last committed Z.
                cursor_z = (self.axial_cursor_item.value()
                            if self.axial_cursor_item is not None else self._selected_z_mm)
                self._selected_z_mm = float(np.clip(cursor_z, *limits))
                self.axial_position.setValue(self._selected_z_mm)
                if self.axial_cursor_item is None:
                    self._add_axial_position_cursor()
                else:
                    with QSignalBlocker(self.axial_cursor_item):
                        self.axial_cursor_item.setBounds(limits)
                        self.axial_cursor_item.setValue(self._selected_z_mm)
                if (self._transverse_focus_request is None
                        or self._transverse_focus_request[0] == "z"):
                    self._transverse_focus_request = ("z", self._selected_z_mm)
        if self._focused_part is not None:
            parts = getattr(getattr(result, "assembly", None), "parts", ())
            self._focused_part = next(
                (part for part in parts if part.key == self._focused_part.key), None
            )
            if self._focused_part is not None:
                self._highlight_ray_component(self._focused_part)
            elif self._ray_component_highlight is not None:
                self.plot.removeItem(self._ray_component_highlight)
                self._ray_component_highlight = None
        if preserved_range is not None:
            (x_min, x_max), (y_min, y_max) = preserved_range
            self.plot.getViewBox().setRange(
                xRange=(x_min, x_max), yRange=(y_min, y_max),
                padding=0.0, disableAutoRange=True,
            )
        elif self.auto_zoom.isChecked() and self._focused_part is not None:
            self._apply_component_zoom(self._focused_part)
        else:
            self.plot.getViewBox().autoRange()
            self.plot.disableAutoRange()
        self._update_component_label_visibility()
        self._update_aperture_spans()
        self._update_scale_notice()
        self._crossover_count = len(self._all_crossovers(result))
        self._wall_stop_count = self._column_wall_stop_count(simulation)
        self._update_projection_text()
        self._update_interaction_detail()
        self._ray_scene_last_update_ms = (perf_counter() - started) * 1000.0

    def display_result(self, result, quality: str) -> None:
        # Explicit republication is also the invalidation boundary for callers
        # that updated an existing result/array in place before handing it back.
        self._ray_display_cache.clear()
        self._ray_display_cache_bytes = 0
        self._ray_display_cache_result = result
        is_preview = str(quality).strip().lower().startswith("preview") or quality == "Medium"
        optical_tuning = bool((getattr(result.simulation, "metrics", None) or {}).get("optical_tuning", False))
        if is_preview:
            self._preview_result = result
            current_high = bool(
                self._high_accuracy_result is not None
                and getattr(self._high_accuracy_result, "model_signature", "")
                and getattr(self._high_accuracy_result, "model_signature", "")
                == getattr(result, "model_signature", "")
            )
            if not current_high:
                self.mark_high_accuracy_stale()
        else:
            self._high_accuracy_result = result
            self._high_accuracy_current = True
        # Geometry changes also preserve the user's view; Fit is explicit.
        preserve_ray_view = self._last_result is not None
        self._last_result = result
        self._last_quality = quality
        no_illumination = sample_illumination_absent(
            getattr(result, "simulation", None),
            getattr(result, "state_snapshot", None),
        )
        if not optical_tuning and (not is_preview or no_illumination):
            self._sample_region_result = getattr(result, "sample_region", None)
        self._prepare_scan_ray_playback(result)
        # Queue before drawing the axial cursor: its focus notifications must
        # never interpolate the previous result while a new one is arriving.
        self._publish_optional_ray_panels(result, refresh=False)
        self._draw_ray_diagram(
            result,
            quality,
            preserve_view=preserve_ray_view,
        )
        self._refresh_visible_ray_panels()
        if optical_tuning:
            # Low-count tuning can miss a tiny aperture. It must never erase
            # completed spectra/images or replace them with synthetic frames.
            self._update_projection_text()
            self.heading.setToolTip("Optical tuning only: specimen scattering and image/spectrum calculations are deferred. Medium uses interior rays plus zero-current support probes; no density is inferred from the outline.")
            return
        self.probe_aberrations.display_result(result)
        self.image_aberrations.display_result(result)
        self.optical_transfer.display_result(result)
        if not is_preview or no_illumination:
            self.energy_filter.display_result(result)
            if no_illumination:
                self.energy_filter.summary.setText(
                    "No incident current at the specimen | no transmitted beam"
                )
        if not is_preview or no_illumination or self._high_accuracy_result is None:
            self.scan_control.display_result(
                getattr(result, "scan_geometry", None),
                getattr(result, "stem_scan", None),
                complete=not is_preview or no_illumination,
            )
            if no_illumination:
                self.scan_control.image_model_notice.setText(
                    "No incident current at the specimen | no STEM frame"
                )
                self.scan_control.image_model_notice.setToolTip("")
        self.sample_page.display_result(
            result,
            (
                getattr(result, "stem_scan", None)
                if not is_preview
                else None
            ),
        )
        if not is_preview or no_illumination:
            cached_sample_region = getattr(result, "sample_region", None)
            self.sample_interactions_3d.display_result(result)
            self.eds_page.display_result(result)
            if cached_sample_region is not None:
                self._set_sample_region_result(
                    cached_sample_region,
                    mark_calculated=False,
                )
            self._update_sample_region_control_availability()
            self.wave_imaging.display_result(
                getattr(result, "wave_imaging", None),
                getattr(result, "state_snapshot", None),
                quality,
                no_illumination=no_illumination,
            )

    def mark_high_accuracy_stale(self) -> None:
        """Keep completed displays visible but detach them from live inputs."""

        if self._high_accuracy_result is None or not self._high_accuracy_current:
            return
        self._high_accuracy_current = False
        self.energy_filter.mark_result_stale()
        self.eds_page.mark_result_stale()
        self.sample_interactions_3d.mark_result_stale()
        self.scan_control.mark_stem_frame_stale()
        self.wave_imaging.mark_result_stale()

    def high_accuracy_result_summary(self):
        """Return detached metadata for the displayed complete calculation."""

        if self._high_accuracy_result is None:
            return None
        return summarise_calculation_result(self._high_accuracy_result)
