"""STEM scan/descan geometry, detector images and acquisition controls."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeSpinBox as QSpinBox,
)

from pathlib import Path
from copy import copy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from temsim.specimen.source import active_cif_path
from temsim.physics.stem_sampling import frame_sampling_report


class ScanControlView(QWidget):
    """Edit STEM scan drives and display geometry or detector images."""

    parameters_changed = Signal(str)
    error = Signal(str)
    playback_time_changed = Signal(float)
    playback_active_changed = Signal(bool)
    df_geometry_requested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("scanControlView")
        self._state = None
        self._result = None
        self._stem_frame = None
        self._stem_frame_context = None
        self._stem_frame_stale = False
        self._paused_display_frame = None
        self._paused_display_context = None
        self._bank_readout = None
        self._bank_display_context = None
        self._wave_action_needed = True
        self._bank_pending_message = ""
        self._images_have_frame = False
        self._rendered_image_frame = None
        self._rendered_image_rows = 0
        self._fourdstem_artifact = None
        self._fourdstem_virtual_image = None
        self._fourdstem_physical_result = None
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
            "AC/Descan raster with physical HAADF, DF and BF readout."
        )
        scope.setToolTip(
            "Scan is available in TEM and STEM with both microprobe and "
            "nanoprobe illumination. AC and Descan expose the same raster and "
            "foil controls. AC is coupled for zero first-order angle at the "
            "sample; Descan uses the opposite raster command and is coupled "
            "at the Selected Area Aperture image-reference station. Calculated "
            "image/diffraction planes and physical aperture stations are "
            "classified independently for the current lens state. Preview "
            "uses the geometric detector approximation."
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
            "Calculate STEM detector images (High accuracy)"
        )
        self.wave_scan_enabled.setObjectName("stemWaveScanEnabled")
        self.wave_scan_enabled.setToolTip(
            "Use the specimen wave model for BF, DF and HAADF images during "
            "High accuracy. Shared multislice and wave-grid settings are on Sample. "
            "Preview remains geometric; this switch does not start a calculation."
        )
        self.wave_scan_enabled.toggled.connect(
            self._wave_scan_model_changed
        )
        controls_layout.addWidget(self.wave_scan_enabled)
        fourdstem_group = QGroupBox("4D-STEM diffraction cube")
        fourdstem_group.setObjectName("stemFourDSTEMGroup")
        fourdstem_form = QFormLayout(fourdstem_group)
        self.fourdstem_enabled = QCheckBox(
            "Save angle-resolved diffraction frames"
        )
        self.fourdstem_enabled.setObjectName("stemFourDSTEMEnabled")
        self.fourdstem_enabled.setToolTip(
            "Runs only during an explicit High accuracy wave-STEM scan."
        )
        fourdstem_form.addRow(self.fourdstem_enabled)
        self.fourdstem_path = QLineEdit()
        self.fourdstem_path.setObjectName("stemFourDSTEMOutputPath")
        self.fourdstem_path.setPlaceholderText("Choose a .npy output file")
        self.fourdstem_browse = QPushButton("Browse…")
        self.fourdstem_browse.setObjectName("stemFourDSTEMBrowse")
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.fourdstem_path, 1)
        path_layout.addWidget(self.fourdstem_browse)
        fourdstem_form.addRow("Output", path_row)
        self.fourdstem_overwrite = QCheckBox("Overwrite")
        self.fourdstem_overwrite.setObjectName("stemFourDSTEMOverwrite")
        self.fourdstem_resume = QCheckBox("Resume partial")
        self.fourdstem_resume.setObjectName("stemFourDSTEMResume")
        policy_row = QWidget()
        policy_layout = QHBoxLayout(policy_row)
        policy_layout.setContentsMargins(0, 0, 0, 0)
        policy_layout.addWidget(self.fourdstem_overwrite)
        policy_layout.addWidget(self.fourdstem_resume)
        policy_layout.addStretch(1)
        fourdstem_form.addRow("Existing output", policy_row)
        self.fourdstem_response_mode = QComboBox()
        self.fourdstem_response_mode.setObjectName(
            "stemFourDSTEMResponseMode"
        )
        self.fourdstem_response_mode.addItem("Ideal", "ideal")
        self.fourdstem_response_mode.addItem("Adjustable", "adjustable")
        self.fourdstem_response_mode.setToolTip(
            "Ideal preserves expected electrons. Adjustable applies a user "
            "model; it is not an OEM detector calibration."
        )
        fourdstem_form.addRow("Pixel response", self.fourdstem_response_mode)
        self.fourdstem_response_controls = {}
        response_specs = (
            ("quantum_efficiency", "Quantum efficiency", 0.0, 1.0, 1.0, 4),
            ("charge_spread_sigma_px", "Charge spread sigma", 0.0, 100.0, 0.0, 4),
            ("dark_electrons_per_pixel", "Dark electrons / pixel", 0.0, 1.0e9, 0.0, 4),
            ("read_noise_electrons_rms", "Read noise RMS", 0.0, 1.0e9, 0.0, 4),
            ("saturation_electrons", "Saturation (0 = off)", 0.0, 1.0e15, 0.0, 4),
            ("gain_counts_per_electron", "Gain counts / electron", 1.0e-9, 1.0e9, 1.0, 6),
            ("offset_counts", "Offset counts", -1.0e12, 1.0e12, 0.0, 4),
        )
        for field, label, minimum, maximum, value, decimals in response_specs:
            control = QDoubleSpinBox()
            control.setObjectName(
                "stemFourDSTEM" + "".join(part.title() for part in field.split("_"))
            )
            control.setRange(minimum, maximum)
            control.setDecimals(decimals)
            control.setValue(value)
            control.setKeyboardTracking(False)
            control.valueChanged.connect(
                lambda changed, name=field: self._fourdstem_numeric_changed(
                    name, changed
                )
            )
            fourdstem_form.addRow(label, control)
            self.fourdstem_response_controls[field] = control
        self.fourdstem_response_poisson = QCheckBox("Poisson counting")
        self.fourdstem_response_poisson.setObjectName(
            "stemFourDSTEMPoissonEnabled"
        )
        self.fourdstem_response_seed = QSpinBox()
        self.fourdstem_response_seed.setObjectName("stemFourDSTEMSeed")
        self.fourdstem_response_seed.setRange(0, 2_147_483_647)
        self.fourdstem_response_seed.setKeyboardTracking(False)
        fourdstem_form.addRow(self.fourdstem_response_poisson)
        fourdstem_form.addRow("Random seed", self.fourdstem_response_seed)
        self.fourdstem_summary = QLabel("4D-STEM capture disabled.")
        self.fourdstem_summary.setObjectName("stemFourDSTEMSummary")
        self.fourdstem_summary.setWordWrap(True)
        self.fourdstem_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        fourdstem_form.addRow(self.fourdstem_summary)
        postprocess_label = QLabel("Post-process an existing cube")
        postprocess_label.setStyleSheet("font-weight: 600;")
        fourdstem_form.addRow(postprocess_label)
        self.fourdstem_virtual_inner_mrad = QDoubleSpinBox()
        self.fourdstem_virtual_inner_mrad.setObjectName(
            "stemFourDSTEMVirtualInnerMrad"
        )
        self.fourdstem_virtual_inner_mrad.setRange(0.0, 1.0e6)
        self.fourdstem_virtual_inner_mrad.setDecimals(4)
        self.fourdstem_virtual_inner_mrad.setValue(0.0)
        self.fourdstem_virtual_inner_mrad.setSuffix(" mrad")
        self.fourdstem_virtual_outer_mrad = QDoubleSpinBox()
        self.fourdstem_virtual_outer_mrad.setObjectName(
            "stemFourDSTEMVirtualOuterMrad"
        )
        self.fourdstem_virtual_outer_mrad.setRange(1.0e-6, 1.0e6)
        self.fourdstem_virtual_outer_mrad.setDecimals(4)
        self.fourdstem_virtual_outer_mrad.setValue(50.0)
        self.fourdstem_virtual_outer_mrad.setSuffix(" mrad")
        self.fourdstem_virtual_inner_mrad.setKeyboardTracking(False)
        self.fourdstem_virtual_outer_mrad.setKeyboardTracking(False)
        fourdstem_form.addRow(
            "Annular inner", self.fourdstem_virtual_inner_mrad
        )
        fourdstem_form.addRow(
            "Annular outer", self.fourdstem_virtual_outer_mrad
        )
        self.fourdstem_open_cube = QPushButton("Open cube…")
        self.fourdstem_open_cube.setObjectName("stemFourDSTEMOpenCube")
        self.fourdstem_integrate_virtual = QPushButton("Integrate annulus")
        self.fourdstem_integrate_virtual.setObjectName(
            "stemFourDSTEMIntegrateVirtual"
        )
        postprocess_row = QWidget()
        postprocess_layout = QHBoxLayout(postprocess_row)
        postprocess_layout.setContentsMargins(0, 0, 0, 0)
        postprocess_layout.addWidget(self.fourdstem_open_cube)
        postprocess_layout.addWidget(self.fourdstem_integrate_virtual)
        fourdstem_form.addRow(postprocess_row)
        self.fourdstem_rederive_physical = QPushButton(
            "Re-integrate current physical detectors"
        )
        self.fourdstem_rederive_physical.setObjectName(
            "stemFourDSTEMRederivePhysical"
        )
        self.fourdstem_rederive_physical.setToolTip(
            "Uses the current D/I/P lenses, apertures and detector geometry; "
            "the stored diffraction cube is not recalculated."
        )
        fourdstem_form.addRow(self.fourdstem_rederive_physical)
        physical_limit = QLabel(
            "Physical re-integration requires an Ideal cube; adjustable pixel "
            "response is already baked into stored counts."
        )
        physical_limit.setWordWrap(True)
        physical_limit.setStyleSheet("color: #64748b;")
        fourdstem_form.addRow(physical_limit)
        self.fourdstem_enabled.toggled.connect(
            self._fourdstem_enabled_changed
        )
        self.fourdstem_path.editingFinished.connect(
            self._fourdstem_path_changed
        )
        self.fourdstem_browse.clicked.connect(self._browse_fourdstem_path)
        self.fourdstem_overwrite.toggled.connect(
            lambda enabled: self._fourdstem_policy_changed(
                "overwrite", enabled
            )
        )
        self.fourdstem_resume.toggled.connect(
            lambda enabled: self._fourdstem_policy_changed("resume", enabled)
        )
        self.fourdstem_response_mode.currentIndexChanged.connect(
            self._fourdstem_response_mode_changed
        )
        self.fourdstem_response_poisson.toggled.connect(
            self._fourdstem_poisson_changed
        )
        self.fourdstem_response_seed.valueChanged.connect(
            self._fourdstem_seed_changed
        )
        self.fourdstem_open_cube.clicked.connect(self._open_fourdstem_cube)
        self.fourdstem_virtual_inner_mrad.valueChanged.connect(
            self._fourdstem_virtual_detector_changed
        )
        self.fourdstem_virtual_outer_mrad.valueChanged.connect(
            self._fourdstem_virtual_detector_changed
        )
        self.fourdstem_integrate_virtual.clicked.connect(
            self._integrate_fourdstem_annulus
        )
        self.fourdstem_rederive_physical.clicked.connect(
            self._rederive_fourdstem_physical_detectors
        )
        controls_layout.addWidget(fourdstem_group)
        statistics = QGroupBox("STEM image statistics")
        statistics_form = QFormLayout(statistics)
        self.poisson_enabled = QCheckBox("Generate seeded Poisson counts")
        self.poisson_enabled.setObjectName("stemPoissonEnabled")
        self.poisson_enabled.setToolTip(
            "Generate one reproducible electron-count array for each detector when "
            "calculating a frame. Checking this selects Poisson counts in Images; "
            "older frames without stored counts remain unavailable until recalculated."
        )
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
            "Shared raster; scan/descan foil coupling is solved automatically."
        )
        pivot_help.setToolTip(
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
        self.controls_scroll = controls_scroll

        self.parameters_page = QWidget()
        self.parameters_page.setObjectName("scanningParametersPage")
        parameters_layout = QVBoxLayout(self.parameters_page)
        parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_layout.addWidget(self.summary)
        parameters_layout.addWidget(scope)
        parameters_layout.addWidget(self.controls_scroll, 1)

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
        self.pause_image_refresh = QCheckBox(
            "Pause refresh (show previous complete frame)"
        )
        self.pause_image_refresh.setObjectName("stemPauseImageRefresh")
        self.pause_image_refresh.setToolTip(
            "Freeze HAADF / DF / BF image updates on the previous complete "
            "frame. The scan clock and Ray Diagram playback continue."
        )
        self.pause_image_refresh.toggled.connect(
            self._image_refresh_pause_changed
        )
        refresh_row = QHBoxLayout()
        refresh_row.addWidget(QLabel("Result source"))
        self.image_source = QComboBox()
        self.image_source.setObjectName("stemImageSource")
        self.image_source.addItem("Current calculation", "current")
        self.image_source.addItem("Advanced bank", "bank")
        self.image_source.setToolTip(
            "Switch cached STEM images only. Scan controls, geometry, 4D-STEM "
            "and Ray Diagram playback always use the current calculation. "
            "No calculation or parameter changes are made by this selector."
        )
        self.image_source.currentIndexChanged.connect(self._image_source_changed)
        refresh_row.addWidget(self.image_source)
        refresh_row.addWidget(self.pause_image_refresh)
        self.match_detector_sampling = QPushButton("Match detector sampling")
        self.match_detector_sampling.setObjectName("stemMatchDetectorSampling")
        self.match_detector_sampling.hide()
        self.match_detector_sampling.clicked.connect(self._match_detector_sampling)
        refresh_row.addWidget(self.match_detector_sampling)
        refresh_row.addStretch(1)
        detector_layout.addLayout(refresh_row)
        quantity_row = QHBoxLayout()
        quantity_row.addWidget(QLabel("Display"))
        self.image_display_quantity = QComboBox()
        self.image_display_quantity.setObjectName("stemImageDisplayQuantity")
        self.image_display_quantity.addItem("Ideal intensity", "ideal")
        self.image_display_quantity.addItem("Expected electrons", "expected")
        self.image_display_quantity.addItem("Poisson counts", "poisson")
        self.image_display_quantity.setToolTip(
            "Compare stored intensity fractions, expected electrons and sampled "
            "electron counts from the displayed frame. This selector never "
            "calculates a signal or generates new random samples."
        )
        self.image_display_quantity.currentIndexChanged.connect(self._image_quantity_changed)
        quantity_row.addWidget(self.image_display_quantity)
        self.image_quantity_notice = QLabel("Ideal intensity | No STEM frame")
        self.image_quantity_notice.setObjectName("stemImageQuantityNotice")
        self.image_quantity_notice.setWordWrap(True)
        self.image_quantity_notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        quantity_policy = self.image_quantity_notice.sizePolicy()
        quantity_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.image_quantity_notice.setSizePolicy(quantity_policy)
        quantity_row.addWidget(self.image_quantity_notice, 1)
        detector_layout.addLayout(quantity_row)
        self.detector_playback_summary = QLabel(
            "Enable AC Scan to calculate one HAADF / DF / BF frame."
        )
        self.detector_playback_summary.setObjectName(
            "stemScanPlaybackSummary"
        )
        self.detector_playback_summary.setWordWrap(True)
        self.detector_playback_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        detector_layout.addWidget(self.detector_playback_summary)
        self.image_model_notice = QLabel("No STEM frame loaded.")
        self.image_model_notice.setToolTip(
            "Run a STEM calculation to identify whether the images are a "
            "geometry/material-particle preview or CIF multislice signal."
        )
        self.image_model_notice.setObjectName("stemImageModelNotice")
        self.image_model_notice.setWordWrap(True)
        self.image_model_notice.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.image_model_notice.setStyleSheet(
            "color: #92400e; background: #fffbeb; border: 1px solid #f59e0b; "
            "padding: 6px;"
        )
        detector_layout.addWidget(self.image_model_notice)
        model_action_row = QHBoxLayout()
        self.enable_wave_images = QPushButton("Enable CIF wave imaging")
        self.enable_wave_images.setObjectName("stemEnableCifWaveImages")
        self.enable_wave_images.setToolTip(
            "Enable the existing STEM wave-image setting for the current calculation. "
            "Run High accuracy separately; the displayed frame remains unchanged."
        )
        self.enable_wave_images.clicked.connect(lambda: self.wave_scan_enabled.setChecked(True))
        self.wave_image_action_note = QLabel("Then run High accuracy to calculate specimen contrast.")
        self.wave_image_action_note.setWordWrap(True)
        model_action_row.addWidget(self.enable_wave_images)
        model_action_row.addWidget(self.wave_image_action_note, 1)
        detector_layout.addLayout(model_action_row)
        interpretation = QLabel(
            "BF: low angle · DF: selected band · HAADF: high angle"
        )
        interpretation.setToolTip(
            "BF records transmitted/low-angle electrons; DF records its "
            "configured scattered-angle band; HAADF records the configured "
            "high-angle band. Exact angular ranges appear above each image. "
            "Small-angle BF and low-angle DF can reverse atomic contrast with "
            "thickness, focus and detector angle. Geometry/particle previews "
            "use a fixed 0–1 fraction scale; uniform nonzero signals use "
            "mid-gray and an explicit constant-value label. Wave images use "
            "per-channel auto contrast. Larger signal is brighter. No inversion "
            "or forced BF/DF complement is applied. Expected electrons and "
            "Poisson counts share a zero-based count scale for each detector "
            "and frame; stored noise is not resampled during playback."
        )
        interpretation.setObjectName("stemImageInterpretation")
        interpretation.setWordWrap(True)
        interpretation.setStyleSheet("color: #94a3b8;")
        detector_layout.addWidget(interpretation)
        detector_images = QHBoxLayout()
        self.detector_image_views = {}
        self.detector_image_items = {}
        self.detector_geometry_labels = {}
        self.detector_sampling_labels = {}
        self.detector_contrast_labels = {}
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
            self._set_micrometre_axes(view, "scan X", "scan Y")
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
            sampling_label = QLabel()
            sampling_label.setObjectName(f"{key}StemSamplingStatus")
            sampling_label.setWordWrap(True)
            sampling_policy = sampling_label.sizePolicy()
            sampling_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            sampling_label.setSizePolicy(sampling_policy)
            sampling_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sampling_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            sampling_label.hide()
            panel_layout.addWidget(sampling_label)
            self.detector_sampling_labels[key] = sampling_label
            if key == "df":
                self.exclude_direct_beam = QPushButton("Exclude direct beam…")
                self.exclude_direct_beam.setObjectName("stemExcludeDirectBeam")
                self.exclude_direct_beam.setToolTip(
                    "Review DF inner and outer diameters fitted to the current "
                    "model's illumination disk, camera length, full signed optical "
                    "transfer and raster offsets. Saving changes the active module "
                    "TOML. Recheck after projector/camera-length or convergence changes. "
                    "Upstream HAADF interception remains physical and is not compensated."
                )
                self.exclude_direct_beam.clicked.connect(self._request_df_geometry)
                self.exclude_direct_beam.hide()
                panel_layout.addWidget(self.exclude_direct_beam)
            panel_layout.addWidget(view, 1)
            contrast_label = QLabel()
            contrast_label.setObjectName(f"{key}StemAutoContrast")
            contrast_label.setWordWrap(True)
            contrast_policy = contrast_label.sizePolicy()
            contrast_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            contrast_label.setSizePolicy(contrast_policy)
            contrast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            contrast_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            contrast_label.setStyleSheet("color: #94a3b8; font-size: 9pt;")
            contrast_label.hide()
            panel_layout.addWidget(contrast_label)
            self.detector_contrast_labels[key] = contrast_label
            detector_images.addWidget(panel, 1)
            self.detector_image_views[key] = view
            self.detector_image_items[key] = image_item
            self.detector_geometry_labels[key] = geometry_label
        detector_layout.addLayout(detector_images, 1)

        fourdstem_page = QWidget()
        self.fourdstem_page = fourdstem_page
        fourdstem_page_layout = QVBoxLayout(fourdstem_page)
        fourdstem_page_layout.setContentsMargins(0, 0, 0, 0)
        self.fourdstem_result_summary = QLabel(
            "Open or calculate a 4D-STEM cube, then integrate a virtual detector."
        )
        self.fourdstem_result_summary.setObjectName(
            "stemFourDSTEMResultSummary"
        )
        self.fourdstem_result_summary.setWordWrap(True)
        self.fourdstem_result_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        fourdstem_page_layout.addWidget(self.fourdstem_result_summary)
        self.fourdstem_image_view = pg.PlotWidget(background="#050816")
        self.fourdstem_image_view.setObjectName("stemFourDSTEMImage")
        self._set_micrometre_axes(self.fourdstem_image_view, "scan X", "scan Y")
        self.fourdstem_image_view.showGrid(x=True, y=True, alpha=0.15)
        self.fourdstem_image_view.getViewBox().setAspectLocked(True)
        self.fourdstem_image_item = pg.ImageItem()
        self.fourdstem_image_view.addItem(self.fourdstem_image_item)
        fourdstem_page_layout.addWidget(self.fourdstem_image_view, 1)

        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("stemResultTabs")
        self.result_tabs.addTab(geometry_page, "Geometry")
        self.result_tabs.addTab(detector_page, "Images")
        self.result_tabs.addTab(fourdstem_page, "4D-STEM")

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("scanContentSplitter")
        self.content_splitter.addWidget(self.parameters_page)
        self.content_splitter.addWidget(self.result_tabs)
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([390, 900])
        self._workspace_panels_taken = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content_splitter, 1)

    def take_workspace_panels(self) -> tuple[QWidget, QTabWidget]:
        """Detach the parameter and result panes for a composite workspace.

        Standalone ``ScanControlView`` instances retain their original split
        layout. The main Scanning Image page calls this once so it can place a
        Probe Aberrations tab beside the parameter pane without nesting the
        Geometry/Images result tabs inside Scanning Parameters.
        """

        if self._workspace_panels_taken:
            raise RuntimeError("Scanning workspace panels were already taken")
        self._workspace_panels_taken = True
        self.parameters_page.setParent(None)
        self.result_tabs.setParent(None)
        return self.parameters_page, self.result_tabs

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
    def _set_micrometre_axes(plot, x_label, y_label) -> None:
        # View and image coordinates remain micrometres. Convert tick values
        # to base metres before AxisItem chooses a single SI prefix.
        for name, label in (("bottom", x_label), ("left", y_label)):
            plot.setLabel(name, label, units="m")
            plot.getAxis(name).setScale(1.0e-6)

    @staticmethod
    def _create_plot(title: str, object_name: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#050816")
        plot.setObjectName(object_name)
        plot.setTitle(title, color="#e2e8f0", size="11pt")
        ScanControlView._set_micrometre_axes(plot, "X", "Y")
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
            self._stem_frame_context = None
            self._paused_display_frame = None
            self._paused_display_context = None
            self._fourdstem_artifact = None
            self._fourdstem_virtual_image = None
            self._fourdstem_physical_result = None
            self.fourdstem_image_item.clear()
            self._stem_auto_range_pending = True
            if not self._showing_bank_images():
                self._clear_detector_images()
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
            self.fourdstem_enabled.setChecked(bool(getattr(
                state.sample, "stem_fourdstem_enabled", False
            )))
            self.fourdstem_path.setText(str(getattr(
                state.sample, "stem_fourdstem_output_path", ""
            )))
            resume_fourdstem = bool(getattr(
                state.sample, "stem_fourdstem_resume", False
            ))
            overwrite_fourdstem = bool(getattr(
                state.sample, "stem_fourdstem_overwrite", False
            )) and not resume_fourdstem
            if resume_fourdstem:
                state.sample.stem_fourdstem_overwrite = False
            self.fourdstem_overwrite.setChecked(overwrite_fourdstem)
            self.fourdstem_resume.setChecked(resume_fourdstem)
            response_mode = str(getattr(
                state.sample, "stem_fourdstem_response_mode", "ideal"
            )).strip().lower()
            response_index = self.fourdstem_response_mode.findData(
                response_mode
            )
            self.fourdstem_response_mode.setCurrentIndex(
                max(response_index, 0)
            )
            response_defaults = {
                "quantum_efficiency": 1.0,
                "charge_spread_sigma_px": 0.0,
                "dark_electrons_per_pixel": 0.0,
                "read_noise_electrons_rms": 0.0,
                "saturation_electrons": 0.0,
                "gain_counts_per_electron": 1.0,
                "offset_counts": 0.0,
            }
            for field, control in self.fourdstem_response_controls.items():
                control.setValue(float(getattr(
                    state.sample,
                    f"stem_fourdstem_{field}",
                    response_defaults[field],
                )))
            self.fourdstem_response_poisson.setChecked(bool(getattr(
                state.sample, "stem_fourdstem_poisson_enabled", False
            )))
            self.fourdstem_response_seed.setValue(int(getattr(
                state.sample, "stem_fourdstem_seed", 0
            )))
            self.fourdstem_virtual_inner_mrad.setValue(float(getattr(
                state.sample, "stem_fourdstem_virtual_inner_mrad", 0.0
            )))
            self.fourdstem_virtual_outer_mrad.setValue(float(getattr(
                state.sample, "stem_fourdstem_virtual_outer_mrad", 50.0
            )))
            self.poisson_enabled.setChecked(
                bool(getattr(state.sample, "stem_poisson_enabled", False))
            )
            self.poisson_seed.setValue(
                int(getattr(state.sample, "stem_poisson_seed", 0))
            )
            if not self._showing_bank_images():
                self._update_detector_geometry_labels(self._paused_display_frame or self._stem_frame)
        finally:
            self._updating = False
        self._sync_fourdstem_control_state()
        self._update_wave_image_action()
        self._update_fourdstem_summary(self._stem_frame)

    def _wave_scan_model_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_wave_enabled = bool(enabled)
        self._update_wave_image_action()
        self.parameters_changed.emit("sample.stem_wave_enabled")

    def _update_wave_image_action(self) -> None:
        sample = getattr(self._state, "sample", None)
        enabled = bool(getattr(sample, "stem_wave_enabled", False))
        self.enable_wave_images.setVisible(self._wave_action_needed)
        self.wave_image_action_note.setVisible(self._wave_action_needed)
        self.enable_wave_images.setEnabled(sample is not None and not enabled)
        self.enable_wave_images.setText("CIF wave imaging enabled" if enabled else "Enable CIF wave imaging")
        note = "For current settings: run High accuracy to calculate new CIF wave images."
        if not enabled:
            note = "Enable the wave model, then run High accuracy for CIF atomic contrast."
        if self.pause_image_refresh.isChecked():
            note += " Resume refresh to show new frames."
        if self._showing_bank_images():
            note += " The bank image retains its captured settings."
        self.wave_image_action_note.setText(note)

    def _sync_fourdstem_control_state(self) -> None:
        capture_enabled = self.fourdstem_enabled.isChecked()
        for widget in (
            self.fourdstem_path,
            self.fourdstem_browse,
            self.fourdstem_overwrite,
            self.fourdstem_resume,
            self.fourdstem_response_mode,
        ):
            widget.setEnabled(capture_enabled)
        adjustable = bool(
            capture_enabled
            and self.fourdstem_response_mode.currentData() == "adjustable"
        )
        for widget in self.fourdstem_response_controls.values():
            widget.setEnabled(adjustable)
        self.fourdstem_response_poisson.setEnabled(adjustable)
        self.fourdstem_response_seed.setEnabled(adjustable)

    def _fourdstem_enabled_changed(self, enabled: bool) -> None:
        self._sync_fourdstem_control_state()
        if self._updating or self._state is None:
            return
        self._state.sample.stem_fourdstem_enabled = bool(enabled)
        self._update_fourdstem_summary(self._stem_frame)
        self.parameters_changed.emit("sample.stem_fourdstem_enabled")

    def _fourdstem_path_changed(self) -> None:
        if self._updating or self._state is None:
            return
        path = self.fourdstem_path.text().strip()
        self.fourdstem_path.setText(path)
        self._state.sample.stem_fourdstem_output_path = path
        self.parameters_changed.emit("sample.stem_fourdstem_output_path")

    def _browse_fourdstem_path(self) -> None:
        current = self.fourdstem_path.text().strip()
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save 4D-STEM diffraction cube",
            current,
            "NumPy array (*.npy)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".npy":
            path = path.with_suffix(".npy")
        self.fourdstem_path.setText(str(path))
        self._fourdstem_path_changed()

    def _fourdstem_policy_changed(self, policy: str, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        field = f"stem_fourdstem_{policy}"
        setattr(self._state.sample, field, bool(enabled))
        other_policy = "resume" if policy == "overwrite" else "overwrite"
        other = (
            self.fourdstem_resume
            if other_policy == "resume"
            else self.fourdstem_overwrite
        )
        other_field = f"stem_fourdstem_{other_policy}"
        other_changed = bool(enabled and getattr(
            self._state.sample, other_field, False
        ))
        if enabled:
            setattr(self._state.sample, other_field, False)
            self._updating = True
            try:
                other.setChecked(False)
            finally:
                self._updating = False
        self.parameters_changed.emit(f"sample.{field}")
        if other_changed:
            self.parameters_changed.emit(f"sample.{other_field}")

    def _fourdstem_response_mode_changed(self, _index: int) -> None:
        self._sync_fourdstem_control_state()
        if self._updating or self._state is None:
            return
        self._state.sample.stem_fourdstem_response_mode = str(
            self.fourdstem_response_mode.currentData()
        )
        self.parameters_changed.emit(
            "sample.stem_fourdstem_response_mode"
        )

    def _fourdstem_numeric_changed(self, field: str, value: float) -> None:
        if self._updating or self._state is None:
            return
        setattr(
            self._state.sample,
            f"stem_fourdstem_{field}",
            float(value),
        )
        self.parameters_changed.emit(f"sample.stem_fourdstem_{field}")

    def _fourdstem_poisson_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_fourdstem_poisson_enabled = bool(enabled)
        self.parameters_changed.emit(
            "sample.stem_fourdstem_poisson_enabled"
        )

    def _fourdstem_seed_changed(self, seed: int) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_fourdstem_seed = int(seed)
        self.parameters_changed.emit("sample.stem_fourdstem_seed")

    def _fourdstem_virtual_detector_changed(self, _value: float) -> None:
        """Persist a derived-mask edit without invalidating the raw cube."""

        if self._updating or self._state is None:
            return
        sample = self._state.sample
        sample.stem_fourdstem_virtual_inner_mrad = float(
            self.fourdstem_virtual_inner_mrad.value()
        )
        sample.stem_fourdstem_virtual_outer_mrad = float(
            self.fourdstem_virtual_outer_mrad.value()
        )
        self._fourdstem_virtual_image = None
        self.fourdstem_result_summary.setText(
            "Virtual-detector bounds changed; integrate the stored cube again."
        )

    def _update_fourdstem_summary(self, frame) -> None:
        frame_artifact = (
            getattr(frame, "fourdstem_artifact", None) if frame else None
        )
        if frame_artifact is not None:
            self._fourdstem_artifact = frame_artifact
        artifact = self._fourdstem_artifact
        if artifact is not None:
            path = str(Path(artifact.path).resolve())
            shape = " × ".join(str(int(value)) for value in artifact.data.shape)
            self.fourdstem_summary.setText(
                f"Saved: {path}\nShape: {shape} (scan Y × X × detector Y × X)"
            )
            self.fourdstem_summary.setToolTip(
                "Completed out-of-core diffraction cube."
            )
        elif self.fourdstem_enabled.isChecked():
            self.fourdstem_summary.setText(
                "Enabled; run High accuracy to save the diffraction cube."
            )
            self.fourdstem_summary.setToolTip(
                "Requires STEM illumination and wave / multislice detector signal."
            )
        else:
            self.fourdstem_summary.setText("4D-STEM capture disabled.")
            self.fourdstem_summary.setToolTip("")

    def _load_fourdstem_cube(self, path: str | Path):
        from temsim.physics.fourdstem import open_fourdstem

        artifact = open_fourdstem(path)
        self._fourdstem_artifact = artifact
        self._fourdstem_virtual_image = None
        self._fourdstem_physical_result = None
        self._update_fourdstem_summary(None)
        self.fourdstem_result_summary.setText(
            f"Opened {Path(artifact.path).resolve()} | shape "
            + " × ".join(str(int(value)) for value in artifact.data.shape)
        )
        return artifact

    def _open_fourdstem_cube(self) -> None:
        current = self.fourdstem_path.text().strip()
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open 4D-STEM diffraction cube",
            current,
            "NumPy array (*.npy)",
        )
        if not selected:
            return
        try:
            self._load_fourdstem_cube(selected)
        except Exception as exc:
            self.fourdstem_result_summary.setText(
                f"Could not open 4D-STEM cube: {exc}"
            )
            self.error.emit(str(exc))

    def _current_fourdstem_artifact(self):
        if self._fourdstem_artifact is not None:
            return self._fourdstem_artifact
        configured = self.fourdstem_path.text().strip()
        if not configured:
            raise ValueError("Open a completed 4D-STEM cube first.")
        return self._load_fourdstem_cube(configured)

    def _record_fourdstem_derived_product(
        self,
        stage_key: str,
        artifact,
        details: dict[str, object],
    ) -> None:
        """Attach small derived-product provenance to the owning STEM frame."""

        frame = self._stem_frame
        frame_artifact = getattr(frame, "fourdstem_artifact", None)
        metrics = getattr(frame, "metrics", None)
        if (
            self._state is None
            or frame_artifact is None
            or not isinstance(metrics, dict)
        ):
            return
        try:
            same_path = (
                Path(frame_artifact.path).resolve()
                == Path(artifact.path).resolve()
            )
        except (AttributeError, OSError, TypeError):
            same_path = frame_artifact is artifact
        if not same_path:
            return
        from temsim.calculation_cache import calculation_signatures

        signature = calculation_signatures(self._state)[stage_key]
        metrics[f"{stage_key}_signature"] = signature
        metrics[f"{stage_key}_provenance"] = {
            "artifact_path": str(Path(artifact.path).resolve()),
            **details,
        }

    def _fourdstem_virtual_response(self, artifact):
        """Return deferred pixel response and absolute dose when available."""

        metadata = getattr(artifact, "metadata", {}) or {}
        provenance = (
            metadata.get("provenance", {})
            if isinstance(metadata, dict) or hasattr(metadata, "get")
            else {}
        )
        raw_probability = bool(
            provenance.get("stored_frame_quantity")
            == "configuration-averaged diffraction probability"
        )
        if not raw_probability or self._state is None:
            return None, 1.0, "stored detector values"

        from temsim.detector.stem_signal import (
            ELEMENTARY_CHARGE_C,
            _configured_fourdstem_response,
            source_current_pa,
        )

        response = _configured_fourdstem_response(self._state.sample)
        frame = self._stem_frame
        frame_artifact = getattr(frame, "fourdstem_artifact", None)
        same_artifact = False
        if frame_artifact is not None:
            try:
                same_artifact = (
                    Path(frame_artifact.path).resolve()
                    == Path(artifact.path).resolve()
                )
            except (AttributeError, OSError, TypeError):
                same_artifact = frame_artifact is artifact
        if not same_artifact:
            return response, 1.0, "response-weighted probability"
        metrics = getattr(frame, "metrics", {}) or {}
        incident_fraction = float(metrics.get("incident_sample_fraction", 1.0))
        dwell_time_s = float(getattr(frame, "dwell_time_s", 0.0) or 0.0)
        if dwell_time_s <= 0.0:
            return response, 1.0, "response-weighted probability"
        electron_dose = (
            source_current_pa(self._state)
            * 1.0e-12
            * incident_fraction
            * dwell_time_s
            / ELEMENTARY_CHARGE_C
        )
        return response, electron_dose, "detector counts"

    def _display_fourdstem_image(self, image, calibration) -> None:
        values = np.asarray(image, dtype=float)
        if values.shape != calibration.scan_x_um.shape:
            raise ValueError("4D-STEM derived image does not match the scan grid.")
        scan_x = np.asarray(calibration.scan_x_um, dtype=float)
        scan_y = np.asarray(calibration.scan_y_um, dtype=float)
        x0, x1 = self._coordinate_edges(scan_x, values.shape[1], 1.0e-6)
        y0, y1 = self._coordinate_edges(scan_y, values.shape[0], 1.0e-6)
        self.fourdstem_image_item.setImage(values.T, autoLevels=True)
        self.fourdstem_image_item.setRect(QRectF(
            x0, y0, x1 - x0, y1 - y0
        ))
        self.fourdstem_image_view.autoRange()
        self.result_tabs.setCurrentIndex(
            self.result_tabs.indexOf(self.fourdstem_page)
        )

    def _integrate_fourdstem_annulus(self) -> None:
        try:
            from temsim.physics.fourdstem import (
                annular_virtual_detector,
                integrate_virtual_detectors,
            )

            artifact = self._current_fourdstem_artifact()
            inner = float(self.fourdstem_virtual_inner_mrad.value())
            outer = float(self.fourdstem_virtual_outer_mrad.value())
            detector = annular_virtual_detector(
                "user_annular", artifact.calibration, inner, outer
            )
            response, electron_dose, quantity = (
                self._fourdstem_virtual_response(artifact)
            )
            image = integrate_virtual_detectors(
                artifact,
                (detector,),
                chunk_scan_points=32,
                response=response,
                electrons_per_frame=electron_dose,
            )[detector.key]
            self._fourdstem_virtual_image = image
            self._record_fourdstem_derived_product(
                "fourdstem_virtual_detectors",
                artifact,
                {
                    "detector": "annular",
                    "inner_mrad": inner,
                    "outer_mrad": outer,
                    "quantity": quantity,
                    "detector_response": (
                        None if response is None else response.provenance()
                    ),
                    "recalculated_specimen": False,
                    "recalculated_multislice": False,
                },
            )
            self._display_fourdstem_image(image, artifact.calibration)
            self.fourdstem_result_summary.setText(
                f"Virtual annulus {inner:.6g}–{outer:.6g} mrad | integrated "
                f"as {quantity}; specimen and multislice were not rerun."
            )
        except Exception as exc:
            self.fourdstem_result_summary.setText(
                f"Virtual-detector integration failed: {exc}"
            )
            self.error.emit(str(exc))

    def _rederive_fourdstem_physical_detectors(self) -> None:
        try:
            from temsim.physics.fourdstem import (
                integrate_runtime_recording_planes,
            )
            from temsim.physics.record_plane import build_record_plane_plan

            if self._state is None:
                raise ValueError("No microscope state is loaded.")
            artifact = self._current_fourdstem_artifact()
            plan = build_record_plane_plan(
                self._state, scan_times_s=artifact.calibration.scan_times_s, recalibrate_scan=True,
            )
            result = integrate_runtime_recording_planes(
                artifact,
                None,
                plan,
                chunk_scan_points=16,
            )
            self._fourdstem_physical_result = result
            if not result.images:
                raise ValueError(
                    "The current runtime state has no inserted readout detector."
                )
            self._record_fourdstem_derived_product(
                "fourdstem_physical_recording",
                artifact,
                {
                    "record_plane_plan_fingerprint": result.plan_fingerprint,
                    "detector_keys": tuple(result.images),
                    "recalculated_cube": False,
                },
            )
            key = next(iter(result.images))
            self._display_fourdstem_image(
                result.images[key], artifact.calibration
            )
            detector_list = ", ".join(result.images)
            changed = (
                " | geometry changed since capture"
                if result.plan_changed_since_capture
                else ""
            )
            self.fourdstem_result_summary.setText(
                f"Physical detector {key} shown; derived for current runtime "
                f"plan ({detector_list}){changed}. The raw cube was not rerun."
            )
        except Exception as exc:
            detail = str(exc)
            if "detector response belongs after interception" in detail:
                detail = (
                    "Physical re-integration is unavailable because the pixel "
                    "detector response is baked into this cube. Capture an "
                    "Ideal cube for trajectory-based re-integration."
                )
            self.fourdstem_result_summary.setText(detail)
            self.error.emit(detail)

    def _poisson_changed(self, enabled: bool) -> None:
        if self._updating or self._state is None:
            return
        self._state.sample.stem_poisson_enabled = bool(enabled)
        if enabled:
            self.image_display_quantity.setCurrentIndex(
                self.image_display_quantity.findData("poisson")
            )
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
                if not self._showing_bank_images():
                    self.detector_playback_summary.setText(
                        "Calculating one HAADF / DF / BF detector-signal frame..."
                    )
            else:
                self._set_playback_active(False)
        self.parameters_changed.emit(f"{prefix}.{field}")

    def _showing_bank_images(self) -> bool:
        return self.image_source.currentData() == "bank"

    def _image_quantity_changed(self, _index: int = 0) -> None:
        """Change presentation of stored arrays without touching acquisition."""
        bank = self._showing_bank_images()
        frame = (getattr(self._bank_readout, "stem", None) if bank else
                 self._paused_display_frame if self.pause_image_refresh.isChecked()
                 and self._paused_display_frame is not None else self._stem_frame)
        if frame is None:
            self._clear_detector_images()
            return
        rows = (self._rendered_image_rows if frame is self._rendered_image_frame
                else np.asarray(frame.scan_x_um).shape[0])
        self._render_stem_rows(rows, frame=frame, bank=bank,
                               preserve_range=self._images_have_frame)

    @staticmethod
    def _stored_image_values(frame, key: str, quantity: str):
        field = {"ideal": "fractions", "expected": "expected_electrons",
                 "poisson": "poisson_counts"}[quantity]
        values = (getattr(frame, field, None) or {}).get(key)
        if values is None:
            return None, "not stored"
        try:
            array = np.asarray(values, dtype=float)
            if array.shape != np.asarray(frame.scan_x_um).shape:
                return None, "invalid array shape"
            if quantity != "ideal" and (
                not np.all(np.isfinite(array)) or np.any(array < 0)
                or (quantity == "poisson" and np.any(array != np.floor(array)))
            ):
                return None, "invalid count data"
        except (TypeError, ValueError, OverflowError):
            return None, "invalid stored data"
        return array, ""

    @classmethod
    def _shared_count_levels(cls, frame, key: str):
        # Both views use the same full-frame range, including any sampled peak
        # above the expectation. Never derive a range from only revealed rows.
        upper = 1.0
        for quantity in ("expected", "poisson"):
            values, _reason = cls._stored_image_values(frame, key, quantity)
            if values is not None and values.size:
                upper = max(upper, float(np.max(values)))
        return 0.0, upper

    def _update_image_quantity_notice(self, frame, unavailable=(), *, bank=False) -> None:
        quantity = self.image_display_quantity.currentData()
        name = self.image_display_quantity.currentText()
        if frame is None:
            text = f"{name} | Unavailable: no stored frame"
            if quantity != "ideal":
                text += ("\nSelect a bank frame containing these counts."
                         if self._showing_bank_images() else
                         "\nEnable Generate seeded Poisson counts and run High accuracy."
                         if quantity == "poisson" else "\nRun High accuracy to store expected electrons.")
            self.image_quantity_notice.setText(text)
            self.image_quantity_notice.setToolTip(
                "Run High accuracy to calculate a frame; for Poisson counts, "
                "enable Generate seeded Poisson counts before calculating. "
                "The display selector does not calculate or resample data."
            )
            return
        metrics = getattr(frame, "metrics", None) or {}
        detail = ["All displayed values belong to this stored frame, not live controls."]
        if quantity == "ideal":
            text = "Ideal intensity | fraction of emitted electrons"
        else:
            seed = metrics.get("poisson_seed")
            dwell = getattr(frame, "dwell_time_s", None)
            if dwell is None:
                dwell = metrics.get("dwell_time_s")
            try:
                dwell_text = (f"Dwell {float(dwell):.6g} s/pixel" if dwell is not None
                              and np.isfinite(float(dwell)) and float(dwell) >= 0
                              else "Dwell not recorded")
            except (TypeError, ValueError, OverflowError):
                dwell_text = "Dwell not recorded"
            text = f"{name} | electrons/pixel | {dwell_text}"
            if quantity == "poisson" or seed is not None:
                text += f" | Seed {seed}" if seed is not None else " | Seed not recorded"
            detail.append(
                "Expected = detector current × pixel dwell / electron charge. "
                "Poisson counts are the stored integer realization. Expected and "
                "Poisson views share each detector's full-frame zero-based scale, "
                "expanded to include the stored Poisson maximum. Playback only "
                "reveals these fixed arrays; it never draws fresh random values."
            )
            for field, label in (("source_current_pa", "Captured source current (pA)"),
                                 ("source_electrons_per_pixel", "Emitted electrons per pixel")):
                if metrics.get(field) is not None:
                    detail.append(f"{label}: {metrics[field]}")
        if unavailable:
            text += "\nUnavailable: " + "; ".join(f"{key.upper()} ({reason})" for key, reason in unavailable)
            if quantity != "ideal" and any("stored" in reason or "invalid" in reason
                                           for _key, reason in unavailable):
                action = ("Enable Generate seeded Poisson counts and run High accuracy."
                          if quantity == "poisson" else "Run High accuracy to store expected electrons.")
                if bank:
                    action += " Regenerate or select a bank frame containing these data."
                elif self.pause_image_refresh.isChecked():
                    action += " Resume refresh to display the new frame."
                text += "\n" + action
        self.image_quantity_notice.setText(text)
        self.image_quantity_notice.setToolTip("\n".join(detail))

    @staticmethod
    def _capture_image_context(state):
        if state is None:
            return None
        sample = copy(getattr(state, "sample", None))
        try:
            cif_path = active_cif_path(sample) if sample is not None else ""
        except ValueError:
            cif_path = ""
        return SimpleNamespace(
            sample=sample, cif_path=cif_path,
            stem_detectors=tuple(copy(item) for item in getattr(state, "stem_detectors", ())),
        )

    def _image_context(self, frame, *, bank=False):
        if bank:
            return self._bank_display_context
        if frame is None:
            return self._capture_image_context(self._state)
        if frame is self._paused_display_frame:
            return self._paused_display_context
        if frame is self._stem_frame:
            return self._stem_frame_context
        return None

    def set_bank_readout(self, readout) -> None:
        """Publish a detached bank product without altering current scan state."""
        frame = getattr(readout, "stem", None)
        if frame is not None:
            try:
                self._validate_stem_frame(frame)
            except (AttributeError, TypeError, ValueError) as exc:
                self.mark_bank_readout_pending(f"Bank STEM unavailable: {exc}")
                return
        self._bank_readout = readout
        self._bank_display_context = self._capture_image_context(getattr(readout, "state_snapshot", None))
        self._bank_pending_message = ""
        if self._showing_bank_images():
            self._display_bank_images()

    def mark_bank_readout_pending(self, message: str) -> None:
        """Retain the last bank frame while a different readout is pending."""
        self._bank_pending_message = str(message)
        if self._showing_bank_images():
            self._update_bank_image_status()

    def _image_source_changed(self, _index: int = 0) -> None:
        bank = self._showing_bank_images()
        self.pause_image_refresh.setEnabled(not bank)
        if bank:
            self._display_bank_images()
            return
        frame = (self._paused_display_frame if self.pause_image_refresh.isChecked()
                 and self._paused_display_frame is not None else self._stem_frame)
        self._update_detector_geometry_labels(frame)
        self._update_image_model_notice(frame, check_cif=False)
        if frame is None:
            self._clear_detector_images()
            self.detector_playback_summary.setText("Current calculation | no STEM frame")
            self.detector_playback_summary.setToolTip("")
            return
        self._render_stem_rows(np.asarray(frame.scan_x_um).shape[0], frame=frame,
                               preserve_range=self._images_have_frame)
        status = ("Image refresh paused; previous complete frame displayed"
                  if self.pause_image_refresh.isChecked() else "Current calculation; complete frame")
        self.detector_playback_summary.setText(self._stem_frame_summary(status, frame=frame))
        self.detector_playback_summary.setToolTip("")

    def _display_bank_images(self) -> None:
        frame = getattr(self._bank_readout, "stem", None)
        self._update_detector_geometry_labels(frame, bank=True)
        self._update_image_model_notice(frame, bank=True)
        if frame is None:
            self._clear_detector_images()
        else:
            self._render_stem_rows(np.asarray(frame.scan_x_um).shape[0], frame=frame,
                                   bank=True, preserve_range=self._images_have_frame)
        self._update_bank_image_status()

    def _update_bank_image_status(self) -> None:
        readout = self._bank_readout
        frame = getattr(readout, "stem", None)
        notes = tuple(getattr(readout, "notes", ()) or ())
        coordinates = getattr(readout, "coordinates", {}) or {}
        detail = ["Advanced bank images use captured settings, not the live scan controls."]
        detail.extend(str(note) for note in notes)
        detail.extend(f"{key}: {value}" for key, value in coordinates.items())
        if frame is not None:
            detail.append(self._stem_frame_summary("Completed bank frame", frame=frame))
        if self._bank_pending_message:
            status = ("Advanced bank | previous frame retained"
                      if frame is not None else "Advanced bank | no completed STEM frame")
            detail.append(self._bank_pending_message)
        else:
            status = ("Advanced bank | complete STEM frame | captured settings"
                      if frame is not None else "Advanced bank | STEM unavailable for this selection")
        self.detector_playback_summary.setText(status)
        self.detector_playback_summary.setToolTip("\n".join(detail))
        if frame is None:
            self.image_model_notice.setText("No bank STEM image | current calculation retained separately")
            self.image_model_notice.setToolTip("\n".join(detail))
        elif not self._bank_pending_message:
            if any("approximation" in str(note).lower() for note in notes if str(note).startswith("STEM")):
                self.image_model_notice.setText(
                    "Advanced bank | angular-routing approximation | " + self.image_model_notice.text()
                )
            self.image_model_notice.setToolTip(self.image_model_notice.toolTip() + "\n" + "\n".join(detail))

    def display_result(self, result, stem_frame=None, *, complete=False, state_snapshot=None) -> None:
        """Display scan geometry and one reusable detector-signal frame."""

        self._result = result
        if stem_frame is not None:
            self._set_stem_frame(stem_frame, state_snapshot=state_snapshot)
        elif complete:
            self._stem_frame = None
            self._stem_frame_context = None
            self._paused_display_frame = None
            self._paused_display_context = None
            if not self._showing_bank_images():
                self._clear_detector_images()
                self._update_image_model_notice(None)
            self._update_fourdstem_summary(None)
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

    def mark_stem_frame_stale(self) -> None:
        """Freeze the previous complete detector frame after input changes."""

        if self._stem_frame is None:
            return
        self._stem_frame_stale = True
        self.match_detector_sampling.setEnabled(False)
        self._set_playback_active(False)
        if self._showing_bank_images():
            return
        frame = self._paused_display_frame or self._stem_frame
        self._update_image_model_notice(frame, check_cif=False)

    @staticmethod
    def _validate_stem_frame(frame) -> None:
        shape = np.asarray(frame.scan_x_um, dtype=float).shape
        if len(shape) != 2 or not all(value > 0 for value in shape):
            raise ValueError("STEM scan frame must use a non-empty 2-D raster.")
        for name in ("scan_x_um", "scan_y_um"):
            coordinates = np.asarray(getattr(frame, name), dtype=float)
            if coordinates.shape != shape or not np.all(np.isfinite(coordinates)):
                raise ValueError("STEM scan coordinates must match the finite 2-D raster.")
        for key, values in frame.fractions.items():
            array = np.asarray(values, dtype=float)
            if array.shape != shape:
                raise ValueError(
                    f"{key}: detector image does not match the scan raster."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{key}: detector image must be finite.")

    def _set_stem_frame(self, frame, *, state_snapshot=None) -> None:
        self._validate_stem_frame(frame)
        previous_frame = self._stem_frame
        previous_context = self._stem_frame_context
        self._stem_frame = frame
        self._stem_frame_context = self._capture_image_context(
            self._state if state_snapshot is None else state_snapshot
        )
        self._stem_frame_stale = False
        if self.pause_image_refresh.isChecked():
            if self._paused_display_frame is None:
                self._paused_display_frame = previous_frame or frame
                self._paused_display_context = previous_context if previous_frame is not None else self._stem_frame_context
            display_frame = self._paused_display_frame
        else:
            display_frame = frame
        self._update_fourdstem_summary(display_frame)
        self._stem_auto_range_pending = True
        if self._showing_bank_images():
            return
        self._update_detector_geometry_labels(display_frame)
        self._update_image_model_notice(display_frame)
        display_rows = np.asarray(display_frame.scan_x_um).shape[0]
        self._render_stem_rows(display_rows, frame=display_frame)

    def _update_image_model_notice(self, frame, *, bank=False, check_cif=True) -> None:
        self._update_sampling_controls(frame, bank=bank)
        if frame is None:
            self._wave_action_needed = True
            self._update_wave_image_action()
            self.image_model_notice.setText(
                "No STEM frame | Preview: geometry · High accuracy + wave model: specimen contrast"
            )
            self.image_model_notice.setToolTip(
                "Run Preview for scan/detector geometry or High accuracy with "
                "wave/multislice enabled for specimen-dependent contrast."
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
        image_state = self._image_context(frame, bank=bank)
        sample = getattr(image_state, "sample", None)
        cif_path = getattr(image_state, "cif_path", "")
        sampling_warning = self._sample_scale_warning(
            pixel_nm=pixel_nm,
            fov_x_nm=fov_x_nm,
            fov_y_nm=fov_y_nm,
            sample=sample,
            check_cif=check_cif and not bank,
            cif_path=cif_path,
        )
        scale += sampling_warning
        compact_scale = ""
        if pixel_nm is not None and fov_x_nm is not None and fov_y_nm is not None:
            compact_scale = (
                f" | {self._format_length_nm(float(pixel_nm))}/px | "
                f"FOV {self._format_length_nm(float(fov_x_nm))} × "
                f"{self._format_length_nm(float(fov_y_nm))}"
            )
        if sampling_warning:
            compact_scale += " | sampling warning"
        if model == "geometric_detector_interception":
            material_transport = bool(
                metrics.get("finite_specimen_exit_used")
                or metrics.get("shared_specimen_interactions_used")
                or metrics.get("shared_specimen_exit_transport_used")
            )
            cif_note = (
                f" Captured structure: {Path(cif_path).name}."
                if cif_path
                else ""
            )
            detail_text = (
                "Preview geometry only — not a specimen STEM image with atomic contrast. The polygons "
                "and sharp wedges are detector-clipping boundaries produced by "
                "scan/descan ray interception; they are not atoms or diffraction "
                "contrast. "
                + ("CIF-derived material composition and density may enter finite-particle transport; "
                   "the CIF lattice is not propagated at each scan pixel. " if material_transport
                   else "The selected CIF structure is not used by this geometry preview. ")
                + f"{cif_note} Enable CIF wave imaging and run High accuracy "
                f"to calculate pixel-resolved atomic contrast.{scale}"
            )
            text = (
                ("Material particle preview | CIF atomic contrast not calculated" if material_transport
                 else "Geometry preview only | selected CIF not used")
                + compact_scale
            )
            colour = (
                "color: #92400e; background: #fffbeb; border: 1px solid #f59e0b;"
            )
        elif model in {"multislice_angle_resolved", "thin_phase_angle_resolved"}:
            potential = str(metrics.get("specimen_potential_model", "specimen potential"))
            detail_text = (
                f"Specimen-dependent {model.replace('_', ' ')} image using "
                f"{potential}. Underlying detector fractions reference emitted source "
                "current integrated over the physical detector masks. Selected count "
                f"views use the stored current and pixel dwell.{scale}"
            )
            text = (
                f"Specimen image | {model.replace('_', ' ')} | {potential}"
                + compact_scale
            )
            colour = (
                "color: #166534; background: #f0fdf4; border: 1px solid #22c55e;"
            )
        elif model == "finite_virtual_absolute_probability":
            detail_text = (
                "Virtual-sample image from the configured absolute interaction "
                f"probabilities and finite density regions.{scale}"
            )
            text = "Virtual-sample image | absolute probabilities" + compact_scale
            colour = (
                "color: #1e40af; background: #eff6ff; border: 1px solid #60a5fa;"
            )
        else:
            limitation = str(metrics.get("model_limitation", "")).strip()
            detail_text = f"Image model: {model}. {limitation}{scale}".strip()
            text = f"Image model: {model}" + compact_scale
            colour = (
                "color: #334155; background: #f8fafc; border: 1px solid #94a3b8;"
            )
        sampling = frame_sampling_report(metrics)
        if sampling is not None and not sampling["coverage_complete"]:
            text = "Limited angular coverage | diagnostic image only" + compact_scale
            if sampling.get("legacy_unchecked"):
                text = "Angular coverage unchecked | recalculate this older frame"
            detail_text += (
                " Angular sampling is incomplete or unchecked. Partial detector "
                "values are not full-band signals; an outside-grid zero is not "
                "a physical zero. Increase the wave Grid without changing FOV."
            )
            colour = "color: #92400e; background: #fffbeb; border: 1px solid #f59e0b;"
        if not bank and self._stem_frame_stale:
            text = "Previous frame | inputs changed\n" + text
            detail_text = "Retained completed frame; run High accuracy to update it. " + detail_text
        self._wave_action_needed = model == "geometric_detector_interception"
        self._update_wave_image_action()
        self.image_model_notice.setText(text)
        self.image_model_notice.setToolTip(detail_text)
        self.image_model_notice.setStyleSheet(f"{colour} padding: 6px;")

    def _update_sampling_controls(self, frame, *, bank=False) -> None:
        metrics = getattr(frame, "metrics", None) or {}
        report = frame_sampling_report(metrics)
        df_row = (report or {}).get("detectors", {}).get("df", {})
        overlap = bool(df_row.get("overlaps_illumination_disk"))
        self.exclude_direct_beam.setVisible(overlap)
        self.exclude_direct_beam.setEnabled(
            overlap and not bank and not self._stem_frame_stale
            and frame is self._stem_frame and self._state is not None
            and bool(metrics.get("sampling_state_signature"))
        )
        self.match_detector_sampling.setVisible(not bank and report is not None and not report["coverage_complete"])
        pixels = None if report is None else report.get("recommended_grid_pixels")
        storage = 0 if report is None else (report.get("estimated_potential_bytes") or 0)
        allowed = pixels is not None and pixels <= 8192 and storage <= 4 * 1024**3
        self.match_detector_sampling.setEnabled(not bank and allowed and not self._stem_frame_stale)
        if pixels is not None:
            tip = (f"Proposed wave Grid: {pixels}; grid area about "
                   f"{report['grid_area_factor']:.1f}x; potential storage about {storage / 1024**3:.2f} GiB "
                   "(not total peak memory). No calculation starts automatically.")
            if not allowed:
                tip += " Proposal exceeds the grid or potential-memory limit. Review active detectors, FOV and phonon count."
        else:
            tip = "No finite sampling proposal. Recalculate old results or inspect the angular transfer at the detector."
        self.match_detector_sampling.setToolTip(tip)
        for key, label in self.detector_sampling_labels.items():
            row = (report or {}).get("detectors", {}).get(key)
            label.setVisible(row is not None)
            label.setText("")
            if row is None:
                continue
            status = row["status"]
            if status == "outside":
                message = "Not simulated: outside wave grid"
                if metrics.get("rutherford_tail_enabled"):
                    message += " | approximate tail only"
            elif status == "partial":
                message = f"Partial band | grid limit {report['maximum_simulated_angle_mrad']:.3g} mrad"
            elif status == "unknown":
                message = "Angular coverage unknown (transfer)"
            else:
                message = "Band covered"
            if not report.get("illumination_covered", True):
                message += " | illumination undersampled"
            outer = row.get("required_outer_mrad")
            outer_text = "unbounded" if outer is None else f"{outer:.4g}"
            overlaps = bool(row.get("overlaps_illumination_disk"))
            message = f"Required: {row['required_inner_mrad']:.4g}\u2013{outer_text} mrad\n{message}"
            if overlaps:
                message += ("\nOverlaps illumination disk "
                            f"(probe {report['probe_semiangle_mrad']:.4g} mrad)")
            label.setText(message)
            label.setStyleSheet("color: #86efac;" if status == "full" and report.get("illumination_covered", True) and not overlaps
                                else "color: #fbbf24;")
            label.setToolTip(
                f"Conservative acceptance bound: {row['required_inner_mrad']:.4g} to {outer_text} mrad. "
                f"Probe semi-angle: {report['probe_semiangle_mrad']:.4g} mrad. "
                + ("Acceptance overlaps the illumination disk; do not assume incoherent dark-field contrast. "
                   if row.get("overlaps_illumination_disk") else "")
                + "Coverage includes raster shifts and anisotropy, before aperture/detector blocking. "
                "Full coverage alone does not establish numerical convergence."
            )

    def require_current_df_frame(self, frame) -> None:
        """Recheck proposal authority at request time and again before saving."""
        from temsim.calculation_cache import calculation_signatures

        shown = (self._paused_display_frame if self.pause_image_refresh.isChecked()
                 and self._paused_display_frame is not None else self._stem_frame)
        if (self._showing_bank_images() or self._stem_frame_stale or frame is None
                or frame is not self._stem_frame or frame is not shown or self._state is None):
            raise ValueError("DF dimensions require the current calculated frame. Resume refresh and run High accuracy.")
        metrics = getattr(frame, "metrics", None) or {}
        signature = metrics.get("sampling_state_signature")
        if not signature or signature != calculation_signatures(self._state)["stem"]:
            raise ValueError(
                "DF proposal belongs to an older camera length or microscope state. "
                "Run High accuracy to update it."
            )

    def _request_df_geometry(self) -> None:
        frame = self._stem_frame
        try:
            self.require_current_df_frame(frame)
        except ValueError as exc:
            self.error.emit(str(exc))
            return
        self.df_geometry_requested.emit(frame)

    def _match_detector_sampling(self) -> None:
        if self._showing_bank_images():
            return
        from temsim.calculation_cache import calculation_signatures

        frame = (self._paused_display_frame if self.pause_image_refresh.isChecked()
                 and self._paused_display_frame is not None else self._stem_frame)
        metrics = getattr(frame, "metrics", None) or {}
        report = frame_sampling_report(metrics)
        pixels = None if report is None else report.get("recommended_grid_pixels")
        if self._state is None or pixels is None or not self.match_detector_sampling.isEnabled():
            return
        if metrics.get("sampling_state_signature") != calculation_signatures(self._state)["stem"]:
            self.error.emit("Sampling proposal belongs to an older state. Run High accuracy to update it.")
            return
        answer = QMessageBox.question(
            self, "Update wave sampling",
            f"Set wave Grid to {pixels}? Grid area grows about {report['grid_area_factor']:.1f}x.\n"
            "FOV, scan pixels, lenses and detectors stay unchanged.\n"
            "Existing results are retained. Run High accuracy when ready.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._state.sample.wave_grid_pixels = int(pixels)
            self.mark_stem_frame_stale()
            self.parameters_changed.emit("sample.wave_grid_pixels")

    def _sample_scale_warning(self, *, pixel_nm, fov_x_nm, fov_y_nm,
                              sample=None, check_cif=True, cif_path=None) -> str:
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
        if cif_path is None:
            cif_path = active_cif_path(sample)
        if check_cif and cif_path and pixel_nm is not None:
            try:
                from temsim.specimen.cif_io import read_cif_atoms

                atoms = read_cif_atoms(Path(cif_path).expanduser())
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

    def _update_detector_geometry_labels(self, frame, *, bank=False) -> None:
        image_state = self._image_context(frame, bank=bank)
        detectors = {
            str(detector.key): detector
            for detector in getattr(image_state, "stem_detectors", ())
        }
        signals = getattr(frame, "detector_signals", {}) if frame else {}
        for key, label in self.detector_geometry_labels.items():
            detector = detectors.get(key)
            signal = signals.get(key)
            if detector is None:
                label.setText("Bank detector geometry unavailable" if bank and image_state is None
                              else "Detector not installed")
                label.setToolTip("")
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
                ("Advanced bank snapshot. " if bank else "Current calculation. ")
                + "Detector position and active inner/outer dimensions come "
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

    def _image_refresh_pause_changed(self, paused: bool) -> None:
        self._update_wave_image_action()
        if paused:
            self._paused_display_frame = self._stem_frame
            self._paused_display_context = self._stem_frame_context
            if self._showing_bank_images():
                return
            if self._paused_display_frame is not None:
                rows = np.asarray(
                    self._paused_display_frame.scan_x_um
                ).shape[0]
                self._render_stem_rows(
                    rows,
                    frame=self._paused_display_frame,
                )
                self.detector_playback_summary.setText(
                    self._stem_frame_summary(
                        "Image refresh paused; previous complete frame displayed",
                        frame=self._paused_display_frame,
                    )
                )
            return
        self._paused_display_frame = None
        self._paused_display_context = None
        if self._showing_bank_images():
            return
        if self._stem_frame is None:
            return
        self._update_detector_geometry_labels(self._stem_frame)
        self._update_image_model_notice(self._stem_frame)
        if self._playback_timer.isActive():
            self._playback_tick()
        else:
            rows = np.asarray(self._stem_frame.scan_x_um).shape[0]
            self._render_stem_rows(rows)
            self.detector_playback_summary.setText(
                self._stem_frame_summary("Stopped; last frame retained")
            )

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
        if self._showing_bank_images():
            return
        if self._stem_frame is not None:
            display_frame = (
                self._paused_display_frame
                if self.pause_image_refresh.isChecked()
                and self._paused_display_frame is not None
                else self._stem_frame
            )
            rows = np.asarray(display_frame.scan_x_um).shape[0]
            self._render_stem_rows(rows, frame=display_frame)
            status = (
                "Image refresh paused; previous complete frame displayed"
                if self.pause_image_refresh.isChecked()
                else "Stopped; last frame retained"
            )
            self.detector_playback_summary.setText(
                self._stem_frame_summary(status, frame=display_frame)
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
        if self._showing_bank_images():
            return
        paused = self.pause_image_refresh.isChecked()
        if not paused:
            self._render_stem_rows(completed_rows)
        playback = (
            "Image refresh paused; previous complete frame displayed; "
            f"scan continues at frame {frame_number}, line {completed_rows}/{rows}"
            if paused
            else (
                f"Scanning continuously; frame {frame_number}, "
                f"line {completed_rows}/{rows}"
            )
        )
        self.detector_playback_summary.setText(
            self._stem_frame_summary(
                playback,
                frame=(self._paused_display_frame if paused else None),
            )
        )

    def _stem_frame_summary(self, playback: str, *, frame=None) -> str:
        frame = self._stem_frame if frame is None else frame
        metrics = getattr(frame, "metrics", None) or {}
        model = str(metrics.get("model", "detector signal"))
        values = []
        report = frame_sampling_report(metrics)
        for key in STEM_DETECTOR_KEYS:
            signal = frame.detector_signals.get(key)
            if signal is not None:
                status = (report or {}).get("detectors", {}).get(key, {}).get("status")
                if report is not None and not report.get("illumination_covered", True):
                    values.append(f"{key.upper()} invalid (illumination undersampled)")
                    continue
                if status == "outside" and not metrics.get("rutherford_tail_enabled"):
                    values.append(f"{key.upper()} not simulated (outside grid)")
                    continue
                values.append(
                    f"{key.upper()} mean fraction {float(signal.fraction):.5g} "
                    f"({float(signal.current_pa):.5g} pA)"
                    + (f" [{status} band]" if status and status != "full" else "")
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

    def _stem_image_rect(self, frame=None) -> QRectF:
        frame = self._stem_frame if frame is None else frame
        scan_x = np.asarray(frame.scan_x_um, dtype=float)
        scan_y = np.asarray(frame.scan_y_um, dtype=float)
        rows, columns = scan_x.shape
        metrics = getattr(frame, "metrics", None) or {}
        fallback_step_um = float(metrics.get("scan_pixel_size_nm", 1.0)) * 1.0e-3
        x0, x1 = self._coordinate_edges(scan_x, columns, fallback_step_um)
        y0, y1 = self._coordinate_edges(scan_y, rows, fallback_step_um)
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def _clear_detector_images(self) -> None:
        self._rendered_image_frame = None
        self._rendered_image_rows = 0
        for key, item in self.detector_image_items.items():
            item.clear()
            self._update_detector_contrast(key, None)
        self._update_image_quantity_notice(None)

    def _update_detector_contrast(self, key: str, levels, *, signal_range=None,
                                  geometry_preview=False, quantity="ideal") -> None:
        label = self.detector_contrast_labels[key]
        if levels is None:
            label.clear()
            label.setToolTip("")
            label.hide()
            return
        low, high = levels
        constant = signal_range is not None and signal_range[0] == signal_range[1]
        if quantity != "ideal":
            kind = "Expected" if quantity == "expected" else "Counts"
            observed = (f"{signal_range[0]:.9g}–{signal_range[1]:.9g}"
                        if signal_range is not None else "unavailable")
            label.setText(f"Shared count scale: {low:.6g}–{high:.6g} e⁻/pixel\n{kind}: {observed}")
            detail = (
                "Expected-electron and Poisson-count views share this detector's "
                "full-frame range, including any stored Poisson peak. Zero is "
                "black. Count values are not normalized into intensity fractions. "
            )
        elif constant:
            value = signal_range[0]
            label.setText(f"Constant signal: {value:.12g}\n" + ("Zero shown black" if value == 0 else "Uniform mid-gray"))
            detail = "Uniform nonzero signals are shown mid-gray; zero is black. A uniform image does not imply zero signal. "
        elif geometry_preview and signal_range is not None:
            label.setText(f"Fixed fraction scale: 0–1\nSignal range: {signal_range[0]:.9g}–{signal_range[1]:.9g}")
            detail = (
                "Geometry/particle preview uses a fixed fraction-of-emitted-current scale. "
                "Small ray-count changes are not stretched to full black and white. "
            )
        else:
            label.setText(f"Auto contrast: {low:.6g}\u2013{high:.6g}")
            detail = "Each nonconstant wave-image channel uses its own full-frame range. "
        label.setToolTip(
            f"Black = minimum ({low:.9g}); white = maximum ({high:.9g}) of this "
            "display range. " + detail + "Stored values are unchanged. "
            "The range stays fixed during line playback."
        )
        label.show()

    def _render_stem_rows(self, completed_rows: int, *, frame=None,
                          bank=False, preserve_range=False) -> None:
        if bank != self._showing_bank_images():
            return
        frame = self._stem_frame if frame is None else frame
        if frame is None:
            self._clear_detector_images()
            return
        auto_range = not preserve_range and (not self._images_have_frame if bank
                                            else self._stem_auto_range_pending)
        image_rect = self._stem_image_rect(frame)
        metrics = getattr(frame, "metrics", None) or {}
        report = frame_sampling_report(metrics)
        geometry_preview = metrics.get("model") == "geometric_detector_interception"
        quantity = self.image_display_quantity.currentData()
        unavailable = []
        any_image = False
        for key, view in self.detector_image_views.items():
            image_item = self.detector_image_items[key]
            values, reason = self._stored_image_values(frame, key, quantity)
            status = (report or {}).get("detectors", {}).get(key, {}).get("status")
            if status == "outside" and not metrics.get("rutherford_tail_enabled"):
                values, reason = None, "outside simulated angular coverage"
            if report is not None and not report.get("illumination_covered", True):
                values, reason = None, "illumination undersampled"
            if values is None:
                image_item.clear()
                self._update_detector_contrast(key, None)
                unavailable.append((key, reason))
                if quantity != "ideal":
                    label = self.detector_contrast_labels[key]
                    label.setText(f"Unavailable: {reason}")
                    label.setToolTip("No replacement intensity or resampled counts are displayed.")
                    label.show()
                continue
            full = np.asarray(values, dtype=float)
            shown = full.copy()
            shown[max(0, int(completed_rows)):, :] = np.nan
            finite = full[np.isfinite(full)]
            signal_range = None
            if finite.size:
                low = float(np.min(finite))
                high = float(np.max(finite))
                signal_range = (low, high)
                if quantity != "ideal":
                    levels = self._shared_count_levels(frame, key)
                elif high == low:
                    levels = ((0.0, 1.0) if low == 0 else
                              (min(0.0, 2.0 * low), max(0.0, 2.0 * low)))
                elif geometry_preview:
                    levels = (0.0, 1.0)
                else:
                    levels = signal_range
            else:
                levels = (0.0, 1.0)
            image_item.setImage(
                shown.T,
                autoLevels=False,
                levels=levels,
            )
            self._update_detector_contrast(key, levels, signal_range=signal_range,
                                           geometry_preview=geometry_preview, quantity=quantity)
            image_item.setRect(image_rect)
            if auto_range:
                view.getViewBox().autoRange()
            any_image = True
        self._rendered_image_frame = frame
        self._rendered_image_rows = max(0, int(completed_rows))
        self._update_image_quantity_notice(frame, unavailable, bank=bank)
        if any_image:
            self._images_have_frame = True
            if not bank:
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
