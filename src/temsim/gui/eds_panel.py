"""Spectrum-only EDS page with shared acquisition settings hosted by the sample view."""

from __future__ import annotations

from collections.abc import Mapping

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeSpinBox as QSpinBox,
)

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from temsim.specimen.support import (
    available_support_materials,
    available_support_meshes,
)
from temsim.specimen.source import specimen_interactions_active
from temsim.gui.eds_peak_labels import EDSPeakLabels


class EDSPage(QWidget):
    """Show the spectrum; retain one set of settings for the shared sample workflow."""

    parameters_changed = Signal(str)
    error = Signal(str)
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
        self._spectrum_energy_kev = np.empty(0, dtype=float)
        self._spectrum_counts = np.empty(0, dtype=float)
        self._spectrum_count_label = "Counts"
        self._spectrum_cursor = None
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
        self.eds_overlap_sampling_enabled = QCheckBox("Weighted beam / sample overlap (EDS)")
        self.eds_overlap_sampling_enabled.setObjectName("sampleEdsOverlapSamplingEnabled")
        self.eds_overlap_sampling_enabled.setToolTip(
            "For sparse hits on a small, thin sample without a material support, "
            "estimate the continuous incident density from weighted rays and "
            "integrate additional material paths. This KDE approximation retains "
            "the small overlap current and does not change downstream electrons "
            "or coherent STEM. Unsupported cases retain the original ray estimate."
        )
        self.eds_overlap_sampling_points = self._integer_control(
            "sampleEdsOverlapSamplingPoints", 32, 4096
        )
        self.eds_overlap_sampling_points.setToolTip(
            "Additional EDS integration points; compare larger values for "
            "quadrature convergence. More points do not remove uncertainty "
            "in reconstructing the beam density from finite upstream rays."
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
        eds_form.addRow("Overlap sampling", self.eds_overlap_sampling_enabled)
        eds_form.addRow("Overlap integration points", self.eds_overlap_sampling_points)
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
        local_form.addRow("Result", self.sample_region_summary)
        local_form.addRow(local_note)
        controls_layout.addWidget(local)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea(self)
        self.settings_panel = controls_scroll
        controls_scroll.setObjectName("sampleInteractionSettingsScrollArea")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(280)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setWidget(controls)
        # The workspace reparents this one settings widget into the sample view.
        # Keep it hidden when the spectrum page is used on its own.
        controls_scroll.hide()

        spectrum_layout = QVBoxLayout(self)
        spectrum_layout.setContentsMargins(6, 6, 6, 6)
        self.spectrum_plot = pg.PlotWidget(background="#050816")
        self.spectrum_plot.setObjectName("edsSpectrumPlot")
        self.spectrum_plot.setLabel("bottom", "X-ray energy", units="keV")
        self.spectrum_plot.setLabel("left", "Expected counts")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.22)
        self.peak_labels = EDSPeakLabels(self.spectrum_plot, self)
        self.spectrum_hover_readout = QLabel(
            "Hover spectrum: energy and counts"
        )
        self.spectrum_hover_readout.setToolTip(
            "Move the pointer over the spectrum to read the nearest energy bin "
            "and its expected or sampled counts."
        )
        self.spectrum_hover_readout.setObjectName("edsSpectrumHoverReadout")
        self.peak_labels.annotations_changed.connect(self._clear_spectrum_readout)
        self.spectrum_hover_readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.spectrum_hover_readout.setStyleSheet(
            "color: #cbd5e1; font-weight: 600;"
        )
        self.spectrum_plot.scene().sigMouseMoved.connect(
            self._spectrum_mouse_moved
        )
        self.eds_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.signal_diagnostics = QLabel()
        self.signal_diagnostics.setObjectName("edsSignalDiagnostics")
        self.signal_diagnostics.setWordWrap(True)
        diagnostics_policy = self.signal_diagnostics.sizePolicy()
        diagnostics_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.signal_diagnostics.setSizePolicy(diagnostics_policy)
        self.signal_diagnostics.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.signal_diagnostics.hide()
        spectrum_layout.addWidget(self.eds_summary)
        spectrum_layout.addWidget(self.signal_diagnostics)
        spectrum_layout.addWidget(self.peak_labels)
        spectrum_layout.addWidget(self.spectrum_hover_readout)
        spectrum_layout.addWidget(self.spectrum_plot, 1)

        self.eds_enabled.toggled.connect(
            lambda value: self._set_bool("eds_enabled", value)
        )
        self.eds_poisson_enabled.toggled.connect(
            lambda value: self._set_bool("eds_poisson_enabled", value)
        )
        self.eds_overlap_sampling_enabled.toggled.connect(
            lambda value: self._set_bool("eds_overlap_sampling_enabled", value)
        )
        self.eds_overlap_sampling_points.valueChanged.connect(
            lambda value: self._set_integer("eds_overlap_sampling_points", value)
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

    def set_state(self, state):
        self._state = state
        self._updating = True
        try:
            sample = state.sample
            self.eds_enabled.setChecked(bool(sample.eds_enabled))
            self.eds_poisson_enabled.setChecked(bool(sample.eds_poisson_enabled))
            self.eds_overlap_sampling_enabled.setChecked(bool(sample.eds_overlap_sampling_enabled))
            self.eds_overlap_sampling_points.setValue(int(sample.eds_overlap_sampling_points))
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
        changed = result is not self._result or (result is None and self._eds_result is not None)
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
                self._plot_spectrum(cached_spectrum)
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

    def _result_calculation_state(self):
        """Return the immutable calculation context that produced the rays.

        Manual EDS enrichment must use the High-accuracy snapshot, including
        its overridden ray count and integration step.  Using the editable
        live state here can relabel a retained 15k-ray interaction as a 1k-ray
        product and make a later cache lookup reuse the wrong artifact.
        """

        if self._result is None:
            raise ValueError("No completed calculation result is available.")
        snapshot = getattr(self._result, "state_snapshot", None)
        if snapshot is not None:
            return snapshot
        # Lightweight legacy/test results predate calculation snapshots.  A
        # signed production result must never silently fall back to live state.
        if getattr(self._result, "signatures", None):
            raise ValueError(
                "The completed result has no calculation-state snapshot."
            )
        if self._state is None:
            raise ValueError("No microscope state is available.")
        return self._state

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
        self._update_signal_diagnostics(self._eds_result, stale=True)
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
        self.eds_support_mesh.setEnabled(enabled and not material_is_vacuum)
        self.eds_poisson_seed.setEnabled(
            enabled and self.eds_poisson_enabled.isChecked()
        )
        self.eds_elastic_seed.setEnabled(enabled and elastic)
        self.eds_elastic_max_events.setEnabled(enabled and elastic)
        self.eds_overlap_sampling_enabled.setEnabled(enabled and elastic)
        self.eds_overlap_sampling_points.setEnabled(
            enabled and elastic and self.eds_overlap_sampling_enabled.isChecked()
        )

    def _clear_result(self, text):
        self._eds_result = None
        self._elastic_result = None
        self._specimen_interactions = None
        self.eds_summary.setText(text)
        self.spectrum_plot.clear()
        self._reset_spectrum_hover()
        self._update_signal_diagnostics(None)
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

    def _store_specimen_interactions(self, interactions) -> bool:
        """Share enrichment only when its scoped cache identities match."""

        expected = getattr(self._result, "signatures", None) or {}
        metrics = getattr(interactions, "metrics", None) or {}
        actual = (
            metrics.get("dependency_signatures", {})
            if isinstance(metrics, Mapping)
            else {}
        )
        mismatched = tuple(
            key
            for key in ("elastic", "eds", "wave", "wave_source")
            if isinstance(expected, Mapping)
            and isinstance(actual, Mapping)
            and expected.get(key)
            and expected.get(key) != actual.get(key)
        )
        if mismatched:
            self.error.emit(
                "Manual specimen result does not match the completed "
                "High-accuracy calculation ("
                + ", ".join(mismatched)
                + "). Run High accuracy again."
            )
            return False

        self._specimen_interactions = interactions
        if self._result is not None:
            self._result.specimen_interactions = interactions
        self.specimen_interactions_updated.emit(interactions)
        return True

    def sample_region_calculation_available(self) -> bool:
        """Return whether an explicit bounded specimen calculation can run."""

        if self._state is None or self._result is None:
            return False
        if not self._has_incident_current():
            return False
        try:
            sample = self._result_calculation_state().sample
        except ValueError:
            return False
        return bool(
            self.eds_enabled.isChecked()
            and str(self.eds_transport.currentData())
            == "elastic_monte_carlo"
            and specimen_interactions_active(sample)
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
                self._result_calculation_state(),
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
        if not self._store_specimen_interactions(result.interactions):
            self.sample_region_summary.setText(
                "Calculation discarded because its cache identity did not "
                "match the High-accuracy result."
            )
            return False
        self._sample_region_result = result
        self._result.sample_region = result
        self._result.specimen_exit = result.specimen_exit
        result_signatures = dict(
            getattr(self._result, "signatures", None) or {}
        )
        for key in ("sample_region", "sample_downstream"):
            signature = str(result.metrics.get(f"{key}_signature", ""))
            if signature:
                result_signatures[key] = signature
        self._result.signatures = result_signatures
        self._result.calculated_products = frozenset(
            set(getattr(self._result, "calculated_products", ()) or ())
            | {"sample_region", "sample_downstream"}
        )
        self._eds_result = result.spectrum
        self._elastic_result = result.spectrum.elastic_transport
        self._plot_spectrum(result.spectrum)
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
                self._result_calculation_state(),
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
        if not self._store_specimen_interactions(interactions):
            self.eds_summary.setText(
                "EDS result discarded because its cache identity did not "
                "match the High-accuracy result."
            )
            return
        self._eds_result = spectrum
        self._elastic_result = spectrum.elastic_transport
        self._plot_spectrum(spectrum)
        sampled_text = (
            ""
            if spectrum.sampled_counts is None
            else f"; sampled {int(np.sum(spectrum.sampled_counts))} counts"
        )
        source_names = ", ".join(
            dict.fromkeys(line.source_key for line in spectrum.lines)
        ) or "no characteristic contributions"
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
                f"{metrics['stored_trajectory_count']:,} histories are available in Sample Interactions 3D."
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

    def _update_signal_diagnostics(self, spectrum, *, stale=False) -> None:
        """Explain the displayed estimate using its completed transport only."""
        if spectrum is None:
            self.signal_diagnostics.clear()
            self.signal_diagnostics.setToolTip("")
            self.signal_diagnostics.hide()
            return
        metrics = dict(getattr(getattr(spectrum, "elastic_transport", None), "metrics", None) or {})
        metrics.update(getattr(spectrum, "metrics", None) or {})
        expected = getattr(spectrum, "total_expected_counts", None)
        if expected is None:
            expected = float(np.sum(spectrum.expected_counts))
        sampled_values = getattr(spectrum, "sampled_counts", None)
        sampled = None if sampled_values is None else float(np.sum(sampled_values))
        counts = f"Expected: {expected:.6g} counts"
        if sampled is not None:
            counts += f" | Poisson sampled: {sampled:.6g} counts"
        rows = [counts]
        overlap_active = metrics.get("eds_overlap_sampling_status") == "active"
        if overlap_active:
            fraction = float(metrics.get("eds_overlap_material_weight_fraction", 0.0))
            points = metrics.get("eds_overlap_sampling_point_count", "?")
            rows.append(
                f"Weighted overlap (EDS only): {100.0 * fraction:.6g}% of sample-plane current; "
                f"{points} integration points. Continuous ray-density estimate (KDE approximation)."
            )
            rows.append("Original electron histories remain unchanged; overlap is not renormalised to full beam current.")
        elif metrics.get("eds_overlap_sampling_status") not in (None, "disabled"):
            rows.append("Overlap sampling: " + str(metrics.get(
                "eds_overlap_sampling_detail", metrics["eds_overlap_sampling_status"]
            )))
        status = metrics.get("material_sampling_status")
        warning = status == "no_sampled_material_hits" and not overlap_active
        if warning:
            rows.append(
                "No sampled ray crossed material; zero estimate does not establish zero physical signal. "
                "Check overlap / increase ray sampling."
            )
        elif status == "no_material":
            rows.append("Transport reports no material.")
        elif status == "no_sampled_material_hits" and overlap_active:
            rows.append("Original rays missed material; the EDS estimate uses the weighted overlap integral.")
        if sampled == 0 and expected > 0:
            warning = True
            rows.append("Poisson sample is zero despite a nonzero expected signal.")
        emitted = metrics.get("total_expected_emitted_photons")
        collected = metrics.get("photon_transport_expected_detected_counts")
        outside_energy = metrics.get("counts_outside_spectrum")
        if expected == 0 and (
            (collected is not None and collected > 0)
            or (outside_energy is not None and outside_energy > 0)
        ):
            warning = True
            rows.append("Collected X-rays lie outside the displayed energy range. Check the spectrum energy limits.")
            if outside_energy is not None:
                rows.append(f"Expected counts outside spectrum: {outside_energy:.6g}")
        elif emitted is not None and emitted > 0 and expected == 0 and collected == 0:
            warning = True
            rows.append("X-rays generated, none collected.")
            photon_counts = [f"Generated photons: {emitted:.6g}"]
            for key, name in (
                ("photon_transport_expected_unattenuated_counts", "Unattenuated expected counts"),
                ("photon_transport_expected_detected_counts", "Collected expected counts"),
            ):
                value = metrics.get(key)
                if value is not None:
                    photon_counts.append(f"{name}: {value:.6g}")
            rows.append(" | ".join(photon_counts))
            blocked = metrics.get("photon_transport_blocked_by_component_counts")
            if isinstance(blocked, Mapping):
                components = [f"{key} ({count:,})" for key, count in blocked.items() if count > 0]
                if components:
                    rows.append("Blocked photon paths: " + ", ".join(components))
            outside = metrics.get("photon_transport_outside_acceptance_count")
            if outside is not None:
                rows.append(f"Photon paths outside acceptance: {outside:,}")
        elif emitted is not None and emitted > 0 and expected == 0 and collected is None:
            warning = True
            rows.append("X-rays generated; collection diagnostics unavailable for this result.")
        positive_count = metrics.get("positive_weight_trajectory_count")
        hits = []
        for prefix, name in (("material", "Material"), ("sample", "Sample")):
            count = metrics.get(f"{prefix}_hit_trajectory_count")
            if count is None or positive_count is None:
                continue
            text = f"{name} hits: {count:,}/{positive_count:,} positive-weight rays"
            weight = metrics.get(f"{prefix}_hit_weight_fraction")
            if weight is not None:
                text += f" ({100.0 * weight:.6g}% beam weight)"
            hits.append(text)
        rows.append(" | ".join(hits) if hits else "Material-hit diagnostics unavailable for this result.")
        footprint = []
        rms_radius = metrics.get("incident_position_rms_radius_nm")
        if rms_radius is not None:
            footprint.append(f"Incident RMS radius: {rms_radius:.6g} nm")
        size = metrics.get("sample_size_xy_nm")
        thickness = metrics.get("sample_thickness_nm")
        if size is not None and len(size) == 2:
            dimensions = f"Sample size: {size[0]:.6g} \u00d7 {size[1]:.6g}"
            if thickness is not None:
                dimensions += f" \u00d7 {thickness:.6g}"
            footprint.append(dimensions + " nm")
        elif thickness is not None:
            footprint.append(f"Sample thickness: {thickness:.6g} nm")
        if footprint:
            rows.append(" | ".join(footprint))
        if stale:
            rows.insert(0, "Previous EDS result | inputs changed; recalculate to update these diagnostics.")
        self.signal_diagnostics.setText("\n".join(rows))
        self.signal_diagnostics.setToolTip(
            "Diagnostics describe the displayed completed spectrum, not live settings. "
            "Hits count positive-weight trajectories that actually crossed material; "
            "material includes the sample and support. Hit weights are conditional "
            "fractions of the traced incident beam weight, not emitted source current. "
            "Sparse or absent crossings do not establish a precise confidence interval. "
            "Expected and Poisson-sampled counts are distinct; these diagnostics do not change either array."
        )
        self.signal_diagnostics.setStyleSheet(
            "color: #fbbf24;" if warning or stale else "color: #cbd5e1;"
        )
        self.signal_diagnostics.show()

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
        self.peak_labels.set_spectrum(spectrum)
        self._update_signal_diagnostics(spectrum)

    def _clear_spectrum_readout(self) -> None:
        """Do not retain an old element identity after changing label scope."""
        if self._spectrum_cursor is not None:
            self._spectrum_cursor.setVisible(False)
        self.spectrum_hover_readout.setText("Hover spectrum: energy and counts")

    def _reset_spectrum_hover(self) -> None:
        self.peak_labels.clear()
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
            + (f" | {matches}" if (matches := self.peak_labels.nearby_text(energy_kev)) else "")
        )
        if self._spectrum_cursor is not None:
            self._spectrum_cursor.setPos(energy_kev)
            self._spectrum_cursor.setVisible(True)
