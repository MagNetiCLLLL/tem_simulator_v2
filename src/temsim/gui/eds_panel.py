"""Explicit EDS acquisition controls and elastic-transport diagnostics."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeSpinBox as QSpinBox,
)

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.specimen.support import (
    available_support_materials,
    available_support_meshes,
)
from temsim.specimen.source import specimen_structure_available
from temsim.gui.transverse_projection import (
    format_projection_angle,
    orthogonal_axis_name,
    project_transverse_values,
    projection_axis_name,
)


class EDSPage(QWidget):
    """Own EDS settings, point acquisition and 3-D path projections."""

    parameters_changed = Signal(str)
    error = Signal(str)
    projection_angle_changed = Signal(float)
    sample_region_result_ready = Signal(object)
    specimen_interactions_updated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("edsPage")
        self._state = None
        self._result = None
        self._eds_result = None
        self._elastic_result = None
        self._specimen_interactions = None
        self._sample_region_result = None
        self._updating = False
        self._projection_angle_deg = 0.0
        self._projection_syncing = False
        self._spectrum_energy_kev = np.empty(0, dtype=float)
        self._spectrum_counts = np.empty(0, dtype=float)
        self._spectrum_count_label = "Counts"
        self._spectrum_cursor = None
        self._projection_redraw_timer = QTimer(self)
        self._projection_redraw_timer.setSingleShot(True)
        self._projection_redraw_timer.setInterval(16)
        self._projection_redraw_timer.timeout.connect(
            self._redraw_elastic_projection
        )

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(6, 6, 6, 6)

        eds = QGroupBox("EDS signal and specimen support")
        eds.setObjectName("edsControls")
        self.eds_group = eds
        eds_form = QFormLayout(eds)
        eds_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.eds_enabled = QCheckBox("Enable explicit EDS acquisition")
        self.eds_enabled.setObjectName("sampleEdsEnabled")
        self.eds_support_material = QComboBox()
        self.eds_support_material.setObjectName("sampleEdsSupportMaterial")
        for key, name in available_support_materials():
            self.eds_support_material.addItem(name, key)
        self.eds_support_mesh = QComboBox()
        self.eds_support_mesh.setObjectName("sampleEdsSupportMesh")
        for key, name in available_support_meshes():
            self.eds_support_mesh.addItem(name, key)
        self.eds_solid_angle = QComboBox()
        self.eds_solid_angle.setObjectName("sampleEdsSolidAngle")
        self.eds_solid_angle.addItem(
            "Installed holder-conditioned acceptance", "installed_holder"
        )
        self.eds_solid_angle.addItem(
            "Unshadowed reference acceptance", "unshadowed"
        )
        self.eds_transport = QComboBox()
        self.eds_transport.setObjectName("sampleEdsTransportMode")
        self.eds_transport.addItem(
            "Elastic Monte Carlo (calculated sample-plane rays)",
            "elastic_monte_carlo",
        )
        self.eds_transport.addItem(
            "Straight primary reference", "straight_primary"
        )
        self.eds_transport.setToolTip(
            "Elastic mode uses every upstream electron ray that reaches the "
            "sample plane. Position, incident direction/Larmor rotation, "
            "energy offset and current weight all enter the 3-D transport."
        )

        self.eds_scalar_controls = {
            "eds_support_offset_x_um": self._double_control(
                "sampleEdsSupportOffsetX", -1.0e6, 1.0e6, suffix=" um"
            ),
            "eds_support_offset_y_um": self._double_control(
                "sampleEdsSupportOffsetY", -1.0e6, 1.0e6, suffix=" um"
            ),
            "eds_support_rotation_deg": self._double_control(
                "sampleEdsSupportRotation", -360.0, 360.0, suffix=" deg"
            ),
            "eds_detector_efficiency": self._double_control(
                "sampleEdsDetectorEfficiency", 0.0, 1.0, decimals=5
            ),
            "eds_spectrum_max_energy_ev": self._double_control(
                "sampleEdsMaximumEnergy", 1.0, 1.0e7, suffix=" eV"
            ),
            "eds_spectrum_bin_width_ev": self._double_control(
                "sampleEdsBinWidth", 1.0e-3, 1.0e6, suffix=" eV"
            ),
            "eds_energy_resolution_fwhm_ev": self._double_control(
                "sampleEdsEnergyResolution",
                0.0,
                1.0e6,
                suffix=" eV FWHM",
            ),
        }
        self.eds_scalar_controls[
            "eds_energy_resolution_fwhm_ev"
        ].setSpecialValueText("Ideal line spectrum")
        self.eds_poisson_enabled = QCheckBox("Sample Poisson counts")
        self.eds_poisson_enabled.setObjectName("sampleEdsPoissonEnabled")
        self.eds_poisson_seed = self._integer_control(
            "sampleEdsPoissonSeed", 0, 2_147_483_647
        )
        self.eds_elastic_seed = self._integer_control(
            "sampleEdsElasticSeed", 0, 2_147_483_647
        )
        self.eds_elastic_max_events = self._integer_control(
            "sampleEdsElasticMaximumEvents", 1, 1_000_000
        )
        self.eds_elastic_max_events.setToolTip(
            "Safety guard only. A nonzero event-limit fraction means the "
            "transport result is truncated."
        )
        self.incident_summary = QLabel(
            "Run a column calculation to resolve the sample-plane ray bundle."
        )
        self.incident_summary.setObjectName("edsIncidentRaySummary")
        self.incident_summary.setWordWrap(True)
        self.incident_summary.setStyleSheet(
            "color: #0f766e; font-weight: 600;"
        )
        self.eds_acquire = QPushButton("Update point EDS")
        self.eds_acquire.setObjectName("sampleEdsAcquirePoint")
        self.eds_acquire.setToolTip(
            "Uses the shared High accuracy result and calculates only a "
            "missing or invalid EDS product."
        )
        self.eds_summary = QLabel(
            "No EDS point acquisition has been calculated."
        )
        self.eds_summary.setObjectName("sampleEdsSummary")
        self.eds_summary.setWordWrap(True)
        self.eds_summary.setStyleSheet("color: #64748b; font-weight: 600;")

        eds_form.addRow("Calculate", self.eds_enabled)
        eds_form.addRow("Support material", self.eds_support_material)
        eds_form.addRow("Grid mesh", self.eds_support_mesh)
        eds_form.addRow(
            "Grid offset X", self.eds_scalar_controls["eds_support_offset_x_um"]
        )
        eds_form.addRow(
            "Grid offset Y", self.eds_scalar_controls["eds_support_offset_y_um"]
        )
        eds_form.addRow(
            "Grid rotation", self.eds_scalar_controls["eds_support_rotation_deg"]
        )
        eds_form.addRow("Acceptance", self.eds_solid_angle)
        eds_form.addRow("Electron paths", self.eds_transport)
        eds_form.addRow("Incident histories", self.incident_summary)
        eds_form.addRow("Elastic seed", self.eds_elastic_seed)
        eds_form.addRow(
            "Maximum events / trajectory", self.eds_elastic_max_events
        )
        eds_form.addRow(
            "Ideal scalar efficiency",
            self.eds_scalar_controls["eds_detector_efficiency"],
        )
        eds_form.addRow(
            "Spectrum maximum",
            self.eds_scalar_controls["eds_spectrum_max_energy_ev"],
        )
        eds_form.addRow(
            "Bin width", self.eds_scalar_controls["eds_spectrum_bin_width_ev"]
        )
        eds_form.addRow(
            "Energy resolution",
            self.eds_scalar_controls["eds_energy_resolution_fwhm_ev"],
        )
        eds_form.addRow("Shot noise", self.eds_poisson_enabled)
        eds_form.addRow("Poisson seed", self.eds_poisson_seed)
        eds_form.addRow(self.eds_acquire)
        eds_form.addRow("Result", self.eds_summary)
        controls_layout.addWidget(eds)

        local = QGroupBox("Detailed sample region")
        local.setObjectName("sampleRegionControls")
        local_form = QFormLayout(local)
        local_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.sample_region_upstream = self._double_control(
            "sampleRegionUpstreamDistance", 0.001, 1.0e6, decimals=3, suffix=" um"
        )
        self.sample_region_downstream = self._double_control(
            "sampleRegionDownstreamDistance", 0.001, 1.0e6, decimals=3, suffix=" um"
        )
        self.sample_region_photons = self._integer_control(
            "sampleRegionPhotonPathCount", 0, 10_000
        )
        self.sample_region_seed = self._integer_control(
            "sampleRegionSeed", 0, 2_147_483_647
        )
        self.sample_region_run = QPushButton("Build detailed sample view")
        self.sample_region_run.setObjectName("sampleRegionRunHighAccuracy")
        self.sample_region_run.setToolTip(
            "Builds the detailed local view from shared specimen results and "
            "calculates only products that are still missing."
        )
        self.sample_region_summary = QLabel(
            "No bounded sample-region calculation has been run."
        )
        self.sample_region_summary.setObjectName("sampleRegionSummary")
        self.sample_region_summary.setWordWrap(True)
        self.sample_region_summary.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        local_note = QLabel(
            "Elastic MC uses Rutherford scattering; channeling uses multislice."
        )
        local_note.setToolTip(
            "Rutherford is treated as the active elastic approximation. "
            "Channeling remains a coherent multislice result, not a random "
            "particle label. X-ray representatives are sampled only for the "
            "manual specimen-region and EDS display."
        )
        local_note.setWordWrap(True)
        local_note.setStyleSheet("color: #64748b;")
        local_form.addRow("Entry plane before sample", self.sample_region_upstream)
        local_form.addRow("Exit plane after sample", self.sample_region_downstream)
        local_form.addRow("Displayed X-ray photons", self.sample_region_photons)
        local_form.addRow("Sampling seed", self.sample_region_seed)
        local_form.addRow(self.sample_region_run)
        local_form.addRow("Result", self.sample_region_summary)
        local_form.addRow(local_note)
        controls_layout.addWidget(local)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("edsControlsScrollArea")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(390)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setWidget(controls)

        self.trajectory_tabs = QTabWidget()
        self.trajectory_tabs.setObjectName("edsTrajectoryProjectionTabs")
        self.eds_trajectory_plot = self._trajectory_plot("X", "X-Z")
        self.eds_trajectory_yz_plot = self._trajectory_plot("Y", "Y-Z")
        self.trajectory_tabs.addTab(self.eds_trajectory_plot, "X-Z projection")
        self.trajectory_tabs.addTab(
            self.eds_trajectory_yz_plot, "Y-Z projection"
        )
        trajectory_page = QWidget()
        trajectory_layout = QVBoxLayout(trajectory_page)
        projection_controls = QHBoxLayout()
        self.projection_label = QLabel("Angle")
        self.projection_xz = QPushButton("X-Z")
        self.projection_xz.setObjectName("edsProjectionXZButton")
        self.projection_xz.setCheckable(True)
        self.projection_xz.setChecked(True)
        self.projection_yz = QPushButton("Y-Z")
        self.projection_yz.setObjectName("edsProjectionYZButton")
        self.projection_yz.setCheckable(True)
        self.projection_slider = QSlider(Qt.Orientation.Horizontal)
        self.projection_slider.setObjectName("edsProjectionAngleSlider")
        self.projection_slider.setRange(0, 3600)
        self.projection_slider.setSingleStep(1)
        self.projection_slider.setPageStep(50)
        self.projection_slider.setMinimumWidth(140)
        self.projection_slider.setMaximumWidth(220)
        self.projection_slider.setToolTip(
            "Shared Ray Diagram / EDS projection angle in tenths of a degree"
        )
        self.projection_value = QLabel("0°")
        self.projection_value.setObjectName("edsProjectionAngleValue")
        for control in (
            self.projection_label,
            self.projection_xz,
            self.projection_yz,
            self.projection_slider,
            self.projection_value,
        ):
            projection_controls.addWidget(control)
        projection_controls.addStretch(1)
        projection_note = QLabel(
            "Shared Ray/EDS rotation · elastic events in red · +Z downward"
        )
        projection_note.setToolTip(
            "The transport calculation is three-dimensional. The X-Z and Y-Z "
            "views use the same rotation angle as Ray Diagram. Rotating the "
            "view reprojects stored histories without rerunning transport."
        )
        projection_note.setWordWrap(True)
        projection_note.setStyleSheet("color: #64748b; font-weight: 600;")
        trajectory_layout.addLayout(projection_controls)
        trajectory_layout.addWidget(projection_note)
        trajectory_layout.addWidget(self.trajectory_tabs, 1)

        spectrum_page = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_page)
        self.spectrum_plot = pg.PlotWidget(background="#050816")
        self.spectrum_plot.setObjectName("edsSpectrumPlot")
        self.spectrum_plot.setLabel("bottom", "X-ray energy", units="keV")
        self.spectrum_plot.setLabel("left", "Expected counts")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.22)
        self.spectrum_hover_readout = QLabel(
            "Hover spectrum: energy and counts"
        )
        self.spectrum_hover_readout.setToolTip(
            "Move the pointer over the spectrum to read the nearest energy bin "
            "and its expected or sampled counts."
        )
        self.spectrum_hover_readout.setObjectName("edsSpectrumHoverReadout")
        self.spectrum_hover_readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.spectrum_hover_readout.setStyleSheet(
            "color: #cbd5e1; font-weight: 600;"
        )
        self.spectrum_plot.scene().sigMouseMoved.connect(
            self._spectrum_mouse_moved
        )
        self.eds_lines = self._table(
            ("Source", "Element", "Transition", "Energy / counts")
        )
        self.eds_lines.setObjectName("sampleEdsLineTable")
        spectrum_layout.addWidget(self.spectrum_hover_readout)
        spectrum_layout.addWidget(self.spectrum_plot, 2)
        spectrum_layout.addWidget(self.eds_lines, 1)

        results = QTabWidget()
        results.setObjectName("edsResultTabs")
        results.addTab(trajectory_page, "Elastic trajectories")
        results.addTab(spectrum_page, "EDS spectrum")
        self.result_tabs = results

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((420, 1000))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.eds_enabled.toggled.connect(
            lambda value: self._set_bool("eds_enabled", value)
        )
        self.eds_poisson_enabled.toggled.connect(
            lambda value: self._set_bool("eds_poisson_enabled", value)
        )
        for field, control in self.eds_scalar_controls.items():
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )
        for combo, field in (
            (self.eds_support_material, "eds_support_material_key"),
            (self.eds_support_mesh, "eds_support_mesh_key"),
            (self.eds_solid_angle, "eds_solid_angle_mode"),
            (self.eds_transport, "eds_transport_mode"),
        ):
            combo.currentIndexChanged.connect(
                lambda _index, control=combo, name=field: self._set_choice(
                    name, control.currentData()
                )
            )
        self.eds_poisson_seed.valueChanged.connect(
            lambda value: self._set_integer("eds_poisson_seed", value)
        )
        self.eds_elastic_seed.valueChanged.connect(
            lambda value: self._set_integer("eds_elastic_seed", value)
        )
        self.eds_elastic_max_events.valueChanged.connect(
            lambda value: self._set_integer("eds_elastic_max_events", value)
        )
        self.eds_acquire.clicked.connect(self._calculate_eds_point)
        for control, field in (
            (
                self.sample_region_upstream,
                "sample_region_upstream_distance_um",
            ),
            (
                self.sample_region_downstream,
                "sample_region_downstream_distance_um",
            ),
        ):
            control.valueChanged.connect(
                lambda value, name=field: self._set_sample_region_scalar(
                    name, value
                )
            )
        for control, field in (
            (self.sample_region_photons, "sample_region_photon_path_count"),
            (self.sample_region_seed, "sample_region_seed"),
        ):
            control.valueChanged.connect(
                lambda value, name=field: self._set_sample_region_integer(
                    name, value
                )
            )
        self.sample_region_run.clicked.connect(self.calculate_sample_region)
        self.projection_xz.clicked.connect(
            lambda: self.set_projection_angle(
                0.0, emit_signal=True, defer_redraw=False
            )
        )
        self.projection_yz.clicked.connect(
            lambda: self.set_projection_angle(
                90.0, emit_signal=True, defer_redraw=False
            )
        )
        self.projection_slider.valueChanged.connect(
            self._projection_slider_changed
        )

    @staticmethod
    def _double_control(
        object_name, minimum, maximum, *, decimals=6, suffix=""
    ):
        control = QDoubleSpinBox()
        control.setObjectName(object_name)
        control.setDecimals(decimals)
        control.setRange(minimum, maximum)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_control(object_name, minimum, maximum):
        control = QSpinBox()
        control.setObjectName(object_name)
        control.setRange(minimum, maximum)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return table

    @staticmethod
    def _trajectory_plot(axis_label, projection):
        plot = pg.PlotWidget(background="#050816")
        plot.setMinimumHeight(300)
        plot.setTitle(f"Elastic trajectories: {projection}")
        plot.setLabel("bottom", axis_label, units="nm")
        plot.setLabel("left", "Z", units="nm")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.getViewBox().invertY(True)
        return plot

    @staticmethod
    def _project_transverse_values(x, y, angle_deg: float) -> np.ndarray:
        """Project X/Y onto the same rotated transverse axis as Ray Diagram."""
        return project_transverse_values(x, y, angle_deg)

    @staticmethod
    def _format_angle(angle_deg: float) -> str:
        return format_projection_angle(angle_deg)

    @classmethod
    def _projection_axis_name(cls, angle_deg: float) -> str:
        return projection_axis_name(angle_deg)

    @classmethod
    def _orthogonal_axis_name(cls, angle_deg: float) -> str:
        return orthogonal_axis_name(angle_deg)

    def _projection_slider_changed(self, value: int) -> None:
        if not self._projection_syncing:
            self.set_projection_angle(
                float(value) / 10.0,
                emit_signal=True,
                defer_redraw=True,
            )

    def set_projection_angle(
        self,
        angle_deg: float,
        *,
        emit_signal: bool = False,
        defer_redraw: bool = False,
    ) -> None:
        """Set the shared transverse view angle without retracing electrons."""

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
        primary_name = self._projection_axis_name(angle)
        orthogonal_name = self._orthogonal_axis_name(angle)
        self.projection_value.setText(f"{self._format_angle(angle)}°")
        self.eds_trajectory_plot.setTitle(
            f"Elastic trajectories: {primary_name}-Z"
        )
        self.eds_trajectory_plot.setLabel(
            "bottom", primary_name, units="nm"
        )
        self.eds_trajectory_yz_plot.setTitle(
            f"Elastic trajectories: {orthogonal_name}-Z"
        )
        self.eds_trajectory_yz_plot.setLabel(
            "bottom", orthogonal_name, units="nm"
        )
        self.trajectory_tabs.setTabText(
            0, f"{primary_name}-Z projection"
        )
        self.trajectory_tabs.setTabText(
            1, f"{orthogonal_name}-Z projection"
        )
        if changed and self._elastic_result is not None:
            if defer_redraw:
                if not self._projection_redraw_timer.isActive():
                    self._projection_redraw_timer.start()
            else:
                self._projection_redraw_timer.stop()
                self._plot_elastic_trajectories(self._elastic_result)
        if changed and emit_signal:
            self.projection_angle_changed.emit(angle)

    def _redraw_elastic_projection(self) -> None:
        if self._elastic_result is not None:
            self._plot_elastic_trajectories(self._elastic_result)

    def set_state(self, state):
        self._state = state
        self._updating = True
        try:
            sample = state.sample
            self.eds_enabled.setChecked(bool(sample.eds_enabled))
            self.eds_poisson_enabled.setChecked(bool(sample.eds_poisson_enabled))
            self.eds_poisson_seed.setValue(int(sample.eds_poisson_seed))
            self.eds_elastic_seed.setValue(int(sample.eds_elastic_seed))
            self.eds_elastic_max_events.setValue(
                int(sample.eds_elastic_max_events)
            )
            self.sample_region_upstream.setValue(
                float(sample.sample_region_upstream_distance_um)
            )
            self.sample_region_downstream.setValue(
                float(sample.sample_region_downstream_distance_um)
            )
            self.sample_region_photons.setValue(
                int(sample.sample_region_photon_path_count)
            )
            self.sample_region_seed.setValue(int(sample.sample_region_seed))
            for field, control in self.eds_scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            for combo, value in (
                (self.eds_support_material, sample.eds_support_material_key),
                (self.eds_support_mesh, sample.eds_support_mesh_key),
                (self.eds_solid_angle, sample.eds_solid_angle_mode),
                (self.eds_transport, sample.eds_transport_mode),
            ):
                index = combo.findData(str(value))
                combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._updating = False
        self._update_controls()

    def display_result(self, result):
        changed = result is not self._result
        if changed:
            self._clear_result(
                "No EDS product is available in this shared result."
            )
        self._result = result
        if changed:
            self._specimen_interactions = getattr(
                result, "specimen_interactions", None
            )
            cached_spectrum = getattr(
                self._specimen_interactions, "eds_spectrum", None
            )
            if cached_spectrum is not None:
                self._eds_result = cached_spectrum
                self._elastic_result = cached_spectrum.elastic_transport
                if self._elastic_result is not None:
                    self._plot_elastic_trajectories(self._elastic_result)
                self._plot_spectrum(cached_spectrum)
                self._populate_lines(cached_spectrum)
                self.eds_summary.setText(
                    f"Cached EDS | {cached_spectrum.total_expected_counts:.6g} "
                    f"expected counts | {len(cached_spectrum.lines)} lines"
                )
            self._sample_region_result = getattr(
                result, "sample_region", None
            )
        if not self._has_incident_current():
            self.eds_summary.setText("No incident current at the specimen | no EDS signal")
            self.sample_region_summary.setText(
                "No incident current at the specimen | sample transport unavailable"
            )
        self._update_incident_summary()
        self._update_controls()

    def _has_incident_current(self):
        from temsim.physics.beam_current import sample_illumination_absent

        return self._result is not None and not sample_illumination_absent(
            getattr(self._result, "simulation", None),
            getattr(self._result, "state_snapshot", self._state),
        )

    def mark_result_stale(self) -> None:
        """Detach live calculation controls without erasing complete plots."""

        if self._result is None:
            return
        self._result = None
        self.eds_summary.setText(
            "Previous EDS result retained | inputs changed"
        )
        self.eds_summary.setToolTip(
            "The displayed spectrum belongs to the previous High accuracy "
            "state. Run High accuracy before calculating new EDS data."
        )
        self.sample_region_summary.setText(
            "Previous sample-region result retained | inputs changed"
        )
        self._update_controls()

    def _update_incident_summary(self):
        if self._state is None or self._result is None:
            return
        simulation = getattr(self._result, "simulation", None)
        if simulation is None:
            self.incident_summary.setText("No sample-plane ray bundle is available.")
            return
        try:
            from temsim.specimen.elastic_transport import (
                incident_rays_from_simulation,
            )

            bundle = incident_rays_from_simulation(
                self._state,
                simulation,
                target_x_nm=float(self._state.sample.scan_origin_x_nm),
                target_y_nm=float(self._state.sample.scan_origin_y_nm),
            )
        except Exception as exc:
            self.incident_summary.setText(str(exc))
            return
        detail_text = (
            f"{bundle.reaching_ray_count:,} of {bundle.emitted_ray_count:,} "
            f"calculated rays reach the sample plane "
            f"({100.0 * bundle.surviving_fraction:.6g}% source current). "
            f"All {bundle.reaching_ray_count:,} become MC histories; no "
            f"separate count is entered. Chief angle "
            f"({bundle.chief_angle_mrad[0]:+.6g}, "
            f"{bundle.chief_angle_mrad[1]:+.6g}) mrad; energy "
            f"{bundle.energy_range_ev[0] * 1.0e-3:.6g}–"
            f"{bundle.energy_range_ev[1] * 1.0e-3:.6g} keV."
        )
        self.incident_summary.setText(
            f"Sample input: {bundle.reaching_ray_count:,}/"
            f"{bundle.emitted_ray_count:,} rays | "
            f"{100.0 * bundle.surviving_fraction:.6g}% current | "
            f"chief angle ({bundle.chief_angle_mrad[0]:+.6g}, "
            f"{bundle.chief_angle_mrad[1]:+.6g}) mrad"
        )
        self.incident_summary.setToolTip(detail_text)

    def _set_bool(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, bool(value))
        self._update_controls()
        self._changed(name)

    def _set_scalar(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, float(value))
        self._changed(name)

    def _set_integer(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, int(value))
        self._changed(name)

    def _set_sample_region_scalar(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, float(value))
        self._clear_sample_region_result(
            "Sample-region display settings changed; rebuild the detailed view."
        )

    def _set_sample_region_integer(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, int(value))
        self._clear_sample_region_result(
            "Sample-region display settings changed; rebuild the detailed view."
        )

    def _set_choice(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, str(value))
        self._update_controls()
        self._changed(name)

    def _changed(self, name):
        if self._result is None:
            self.eds_summary.setText(
                "EDS inputs changed | run High accuracy to update the shared result"
            )
        else:
            self.mark_result_stale()
        self.parameters_changed.emit(f"sample.{name}")

    def _update_controls(self):
        enabled = self._state is not None and self.eds_enabled.isChecked()
        material_is_vacuum = (
            str(self.eds_support_material.currentData()) == "vacuum"
        )
        elastic = str(self.eds_transport.currentData()) == "elastic_monte_carlo"
        for control in (
            self.eds_support_material,
            self.eds_solid_angle,
            self.eds_transport,
            self.eds_poisson_enabled,
            self.sample_region_upstream,
            self.sample_region_downstream,
            self.sample_region_photons,
            self.sample_region_seed,
            *self.eds_scalar_controls.values(),
        ):
            control.setEnabled(enabled)
        self.eds_acquire.setEnabled(enabled and self._has_incident_current())
        self.sample_region_run.setEnabled(
            self.sample_region_calculation_available()
        )
        self.eds_support_mesh.setEnabled(enabled and not material_is_vacuum)
        self.eds_poisson_seed.setEnabled(
            enabled and self.eds_poisson_enabled.isChecked()
        )
        self.eds_elastic_seed.setEnabled(enabled and elastic)
        self.eds_elastic_max_events.setEnabled(enabled and elastic)

    def _clear_result(self, text):
        self._projection_redraw_timer.stop()
        self._eds_result = None
        self._elastic_result = None
        self._specimen_interactions = None
        self.eds_summary.setText(text)
        self.eds_lines.setRowCount(0)
        self.eds_trajectory_plot.clear()
        self.eds_trajectory_yz_plot.clear()
        self.spectrum_plot.clear()
        self._reset_spectrum_hover()
        self._clear_sample_region_result(
            "Sample-region result is stale; run the manual calculation again."
        )

    def _clear_sample_region_result(self, text: str) -> None:
        """Invalidate display geometry without discarding shared EDS physics."""

        had_sample_region = self._sample_region_result is not None
        self._sample_region_result = None
        if self._result is not None:
            self._result.sample_region = None
        self.sample_region_summary.setText(text)
        if had_sample_region:
            self.sample_region_result_ready.emit(None)

    def _store_specimen_interactions(self, interactions) -> None:
        """Share an enriched result with every view of this column result."""

        self._specimen_interactions = interactions
        if self._result is not None:
            self._result.specimen_interactions = interactions
        self.specimen_interactions_updated.emit(interactions)

    def sample_region_calculation_available(self) -> bool:
        """Return whether an explicit bounded specimen calculation can run."""

        if self._state is None or self._result is None:
            return False
        if not self._has_incident_current():
            return False
        sample = self._state.sample
        return bool(
            self.eds_enabled.isChecked()
            and str(self.eds_transport.currentData())
            == "elastic_monte_carlo"
            and getattr(sample, "inserted", False)
            and specimen_structure_available(sample)
        )

    def calculate_sample_region(self) -> bool:
        """Run the user-requested bounded specimen calculation once."""

        if self._state is None or self._result is None:
            self.error.emit(
                "Run High accuracy before building the detailed sample view."
            )
            return False
        if not self.sample_region_calculation_available():
            self.error.emit(
                "Sample-region transport requires an inserted Real CIF or "
                "Virtual reference sample, enabled EDS, and Elastic Monte "
                "Carlo transport."
            )
            return False
        self.sample_region_summary.setText(
            "Calculating the manually requested bounded sample region..."
        )
        try:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
            from temsim.specimen.sample_region import simulate_sample_region

            assembly = getattr(self._result, "assembly", None)
            if assembly is None:
                raise ValueError(
                    "The current result has no installed EDS geometry."
                )
            geometry = EDSDetectorArrayGeometry.from_part_data(
                assembly.part(EDS_DETECTOR_SYSTEM).data
            )
            result = simulate_sample_region(
                self._state,
                self._result,
                geometry,
                upstream_distance_um=self.sample_region_upstream.value(),
                downstream_distance_um=self.sample_region_downstream.value(),
                photon_path_count=self.sample_region_photons.value(),
                secondary_path_count=0,
                seed=self.sample_region_seed.value(),
                existing_interactions=self._specimen_interactions,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            self.sample_region_summary.setText(f"Calculation failed: {exc}")
            return False
        self._sample_region_result = result
        self._result.sample_region = result
        self._store_specimen_interactions(result.interactions)
        self._eds_result = result.spectrum
        self._elastic_result = result.spectrum.elastic_transport
        self._plot_elastic_trajectories(self._elastic_result)
        self._plot_spectrum(result.spectrum)
        self._populate_lines(result.spectrum)
        metrics = result.metrics
        forward = 100.0 * float(metrics.get("downstream_forward_weight", 0.0))
        exit_weight = 100.0 * float(metrics.get("exit_plane_weight", 0.0))
        shared_text = (
            " Reused the cached EDS and elastic specimen calculation."
            if not metrics.get(
                "specimen_observables_calculated_for_this_view", ()
            )
            else " Calculated the missing specimen signals once."
        )
        detail_text = (
            f"Entry {result.entry_z_mm:.6g} mm -> exit {result.exit_z_mm:.6g} mm; "
            f"{metrics['sample_ray_count']:,} specimen histories, "
            f"{len(result.photon_paths):,} isotropic X-ray representatives, "
            f"Forward elastic terminal weight {forward:.6g}%; "
            f"{exit_weight:.6g}% tracked electron weight reaches the selected "
            f"exit boundary after specimen losses.{shared_text}"
        )
        self.sample_region_summary.setText(
            f"{metrics['sample_ray_count']:,} histories | "
            f"{len(result.photon_paths):,} X-ray paths | "
            f"forward {forward:.6g}% | exit {exit_weight:.6g}%"
        )
        self.sample_region_summary.setToolTip(
            detail_text
            + "\n\n"
            + "\n".join(
                f"{key}: {value}"
                for key, value in metrics.items()
                if not str(key).startswith("secondary_")
            )
        )
        eds_detail = (
            f"Bounded EDS result: {result.spectrum.total_expected_counts:.6g} "
            "expected counts. X-ray lines are propagated isotropically for "
            "display; detected paths use the exact aggregate-solid-angle "
            "angular surrogate, not an invented sensor face."
        )
        self.eds_summary.setText(
            f"Bounded EDS | {result.spectrum.total_expected_counts:.6g} "
            f"expected counts | {len(result.spectrum.lines)} line contributions"
        )
        self.eds_summary.setToolTip(eds_detail)
        self.sample_region_result_ready.emit(result)
        return True

    def _calculate_eds_point(self):
        if self._state is None or self._result is None:
            self.error.emit(
                "Run a column calculation before the explicit EDS acquisition."
            )
            return
        if not self.eds_enabled.isChecked():
            self.eds_summary.setText("EDS acquisition is disabled.")
            return
        try:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
            from temsim.specimen.interaction_engine import (
                run_specimen_interactions,
            )
            from temsim.specimen.interaction_types import (
                SpecimenInteractionRequest,
            )

            assembly = getattr(self._result, "assembly", None)
            if assembly is None:
                raise ValueError("The current result has no installed EDS geometry.")
            geometry = EDSDetectorArrayGeometry.from_part_data(
                assembly.part(EDS_DETECTOR_SYSTEM).data
            )
            interactions = run_specimen_interactions(
                self._state,
                getattr(self._result, "simulation", None),
                SpecimenInteractionRequest.eds_point(),
                detector_geometry=geometry,
                existing_result=self._specimen_interactions,
            )
            spectrum = interactions.eds_spectrum
            if spectrum is None:
                raise RuntimeError(
                    "Specimen interaction engine returned no EDS spectrum"
                )
        except Exception as exc:
            self.error.emit(str(exc))
            self.eds_summary.setText(f"EDS calculation failed: {exc}")
            return
        self._store_specimen_interactions(interactions)
        self._eds_result = spectrum
        self._elastic_result = spectrum.elastic_transport
        self._plot_elastic_trajectories(self._elastic_result)
        self._plot_spectrum(spectrum)
        self._populate_lines(spectrum)
        sampled_text = (
            ""
            if spectrum.sampled_counts is None
            else f"; sampled {int(np.sum(spectrum.sampled_counts))} counts"
        )
        source_names = ", ".join(
            dict.fromkeys(line.source_key for line in spectrum.lines)
        ) or "vacuum only"
        if self._elastic_result is None:
            transport_text = "Straight-primary reference; no elastic MC."
        else:
            metrics = self._elastic_result.metrics
            transport_text = (
                f"Elastic MC used all {metrics['trajectory_count']:,} rays "
                f"reaching the sample plane; "
                f"{metrics['mean_elastic_events_per_trajectory']:.6g} "
                f"weighted mean events, "
                f"{100.0 * metrics['transmitted_fraction']:.5g}% forward, "
                f"{100.0 * metrics['backscattered_fraction']:.5g}% reverse. "
                f"{metrics['stored_trajectory_count']:,} histories are shown."
            )
            if metrics["rutherford_heavy_element_warning"]:
                transport_text += " Z>30 encountered; ELSEPA is recommended."
        reused_text = (
            " Cached specimen result reused."
            if "characteristic_x_ray" not in set(
                interactions.metrics.get(
                    "calculated_observables_this_call", ()
                )
            )
            else ""
        )
        detail_text = (
            f"EDS point: {spectrum.total_expected_counts:.6g} expected "
            f"counts{sampled_text}; {len(spectrum.lines)} characteristic "
            f"track-line contributions; sources: {source_names}. "
            f"{transport_text} Bremsstrahlung is not yet included.{reused_text}"
        )
        self.eds_summary.setText(
            f"EDS point | {spectrum.total_expected_counts:.6g} expected "
            f"counts{sampled_text} | {len(spectrum.lines)} line contributions | "
            f"{source_names}"
        )
        self.eds_summary.setToolTip(
            detail_text
            + "\n\n"
            + "\n".join(
                f"{key}: {value}" for key, value in spectrum.metrics.items()
            )
        )

    def _populate_lines(self, spectrum):
        from ase.data import chemical_symbols

        ordered = sorted(
            spectrum.lines,
            key=lambda line: line.expected_detected_counts,
            reverse=True,
        )[:100]
        self.eds_lines.setRowCount(len(ordered))
        for row, line in enumerate(ordered):
            values = (
                line.source_key,
                chemical_symbols[line.atomic_number],
                f"{line.subshell} / {line.transition}",
                f"{line.energy_ev * 1.0e-3:.6g} keV | "
                f"{line.expected_detected_counts:.6g}",
            )
            for column, value in enumerate(values):
                self.eds_lines.setItem(row, column, QTableWidgetItem(str(value)))

    def _plot_spectrum(self, spectrum):
        self.spectrum_plot.clear()
        values = (
            spectrum.sampled_counts
            if spectrum.sampled_counts is not None
            else spectrum.expected_counts
        )
        energy_kev = np.asarray(
            spectrum.energy_bin_centres_ev, dtype=float
        ) * 1.0e-3
        counts = np.asarray(values, dtype=float)
        if energy_kev.ndim != 1 or counts.shape != energy_kev.shape:
            raise ValueError(
                "EDS spectrum energies and counts must be matching 1-D arrays"
            )
        if not np.all(np.isfinite(energy_kev)) or not np.all(
            np.isfinite(counts)
        ):
            raise ValueError("EDS spectrum contains NaN or infinity")
        self._spectrum_energy_kev = energy_kev
        self._spectrum_counts = counts
        self._spectrum_count_label = (
            "Sampled counts"
            if spectrum.sampled_counts is not None
            else "Expected counts"
        )
        self.spectrum_plot.setLabel("left", self._spectrum_count_label)
        self.spectrum_plot.plot(
            energy_kev,
            counts,
            pen=pg.mkPen("#22d3ee", width=1.4),
        )
        self._spectrum_cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#f8fafc",
                width=1.0,
                style=Qt.PenStyle.DashLine,
            ),
        )
        self._spectrum_cursor.setVisible(False)
        self.spectrum_plot.addItem(
            self._spectrum_cursor,
            ignoreBounds=True,
        )
        self.spectrum_hover_readout.setText(
            "Hover spectrum: energy and counts"
        )
        self.spectrum_plot.enableAutoRange()

    def _reset_spectrum_hover(self) -> None:
        self._spectrum_energy_kev = np.empty(0, dtype=float)
        self._spectrum_counts = np.empty(0, dtype=float)
        self._spectrum_count_label = "Counts"
        self._spectrum_cursor = None
        self.spectrum_hover_readout.setText(
            "Hover spectrum: energy and counts"
        )

    def _spectrum_mouse_moved(self, scene_position) -> None:
        """Snap the hover readout to the nearest physical energy bin."""

        if isinstance(scene_position, (tuple, list)):
            if not scene_position:
                return
            scene_position = scene_position[0]
        view_box = self.spectrum_plot.getViewBox()
        if (
            self._spectrum_energy_kev.size == 0
            or not view_box.sceneBoundingRect().contains(scene_position)
        ):
            if self._spectrum_cursor is not None:
                self._spectrum_cursor.setVisible(False)
            return
        energy_at_pointer = float(
            view_box.mapSceneToView(scene_position).x()
        )
        insertion = int(
            np.searchsorted(self._spectrum_energy_kev, energy_at_pointer)
        )
        candidates = tuple(
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < self._spectrum_energy_kev.size
        )
        if not candidates:
            return
        index = min(
            candidates,
            key=lambda value: abs(
                float(self._spectrum_energy_kev[value]) - energy_at_pointer
            ),
        )
        energy_kev = float(self._spectrum_energy_kev[index])
        counts = float(self._spectrum_counts[index])
        self.spectrum_hover_readout.setText(
            f"Energy {energy_kev:.6g} keV | "
            f"{self._spectrum_count_label} {counts:.6g}"
        )
        if self._spectrum_cursor is not None:
            self._spectrum_cursor.setPos(energy_kev)
            self._spectrum_cursor.setVisible(True)

    def _plot_elastic_trajectories(self, transport):
        self.eds_trajectory_plot.clear()
        self.eds_trajectory_yz_plot.clear()
        if transport is None or not transport.trajectories:
            return
        colours = {
            "transmitted": (34, 197, 94, 150),
            "backscattered": (249, 115, 22, 180),
            "lateral_escape": (59, 130, 246, 170),
            "event_limit": (239, 68, 68, 200),
            "path_limit": (168, 85, 247, 200),
        }
        event_primary, event_orthogonal, event_z = [], [], []
        orthogonal_angle = (self._projection_angle_deg + 90.0) % 360.0
        for trajectory in transport.trajectories:
            points = trajectory.points_nm
            if points.shape[0] < 2:
                continue
            pen = pg.mkPen(
                colours.get(trajectory.outcome, (100, 116, 139, 150)),
                width=1.1,
            )
            primary = self._project_transverse_values(
                points[:, 0], points[:, 1], self._projection_angle_deg
            )
            orthogonal = self._project_transverse_values(
                points[:, 0], points[:, 1], orthogonal_angle
            )
            self.eds_trajectory_plot.plot(primary, points[:, 2], pen=pen)
            self.eds_trajectory_yz_plot.plot(
                orthogonal, points[:, 2], pen=pen
            )
            for event in trajectory.events:
                event_primary.append(
                    float(
                        self._project_transverse_values(
                            event.position_nm[0],
                            event.position_nm[1],
                            self._projection_angle_deg,
                        )
                    )
                )
                event_orthogonal.append(
                    float(
                        self._project_transverse_values(
                            event.position_nm[0],
                            event.position_nm[1],
                            orthogonal_angle,
                        )
                    )
                )
            event_z.extend(event.position_nm[2] for event in trajectory.events)
        for plot, positions in (
            (self.eds_trajectory_plot, event_primary),
            (self.eds_trajectory_yz_plot, event_orthogonal),
        ):
            if positions:
                plot.plot(
                    positions,
                    event_z,
                    pen=None,
                    symbol="o",
                    symbolSize=3.5,
                    symbolPen=None,
                    symbolBrush=pg.mkBrush(239, 68, 68, 155),
                )
            plot.enableAutoRange()
