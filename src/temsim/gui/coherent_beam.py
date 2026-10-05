"""Explicit tip-origin wave development, independent of particle results.

The page owns captured sessions and exact observation-plane results. Moving Z
never defines a source or interpolates a complex field. Mixed modes contribute
intensities; their individual complex fields remain in the executed checkpoint.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import json
import math
from numbers import Real
from threading import Event
from time import monotonic

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, QRunnable, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.immutable_json import thaw_json
from temsim.physics.coherent_inputs import (
    TipEmissionSettings, candidate_tip_emission, prepare_coherent_state,
    source_settings_from_state, wave_input_summary,
)
from temsim.physics.tip_wave_pipeline import TipWaveObservationSession, TipWaveRequest, simulate_tip_wave

GIB = 1024**3


def _coherent_backend_text(checkpoint):
    """Describe executed column/material records, never the requested policy."""
    column_rows = None
    material_rows = None
    record = getattr(checkpoint, "record", None)
    seen = set()
    while isinstance(record, Mapping) and id(record) not in seen:
        seen.add(id(record))
        schema = record.get("schema")
        if column_rows is None and schema in ("executed-column-wave-v1", "executed-column-segment-v1"):
            # Only the current column stage supplies its device evidence. An
            # older upstream GPU stage must not relabel a CPU fallback.
            column_rows = record.get("modes", ())
        if material_rows is None and schema in ("executed-specimen-wave-v1", "executed-inelastic-trajectories-v1"):
            material_rows = []
            material_column = []
            for mode in record.get("modes", ()):
                steps = mode.get("steps", mode.get("slices", ()))
                for step in steps:
                    material_rows.extend(step[key] for key in ("phase_before", "phase_after") if key in step)
                if steps:
                    material_column.extend(steps[-1].get("column", ()))
            if column_rows is None:
                column_rows = material_column
        if column_rows is not None and material_rows is not None:
            break
        record = record.get("upstream")

    def describe(name, rows):
        rows = [row for row in (rows or ()) if isinstance(row, Mapping)]
        backends = sorted({str(row["compute_backend"]) for row in rows if row.get("compute_backend")})
        if not backends:
            return f"{name}: execution device not recorded"
        labels = {"cupy": "GPU (CuPy)", "numpy": "CPU (NumPy)"}
        text = f"{name}: " + ", ".join(labels.get(value, value) for value in backends)
        precision = sorted({str(row["numeric_precision"]) for row in rows if row.get("numeric_precision")})
        text += "; " + (", ".join(precision) if precision else "precision not recorded")
        reasons = sorted({str(row["fallback_reason"]) for row in rows if row.get("fallback_reason")})
        if reasons:
            text += "; fallback: " + "; ".join(reasons)
        return text

    return " | ".join((describe("Column", column_rows), describe("Material", material_rows), "Gun: CPU"))


def _wave_resource_claim(state, request):
    """Conservative stage peak plus bounded growth, independent of cache size.

    The cap reservoir keeps its radial gun load alive while solving the FEM
    boundary, so those two working caps overlap. Column and screen stages run
    afterward. Segmented execution releases completed wave buffers; at most
    one new observation checkpoint is retained by this page per request.
    Existing displayed/cache roots are inventoried separately and deduplicated
    by the coordinator, rather than added to this prospective reservation.
    """
    gun = request.gun.maximum_checkpoint_bytes
    column = request.wave_grid.maximum_working_bytes
    stage_peak = max(gun, column, request.maximum_readout_bytes)
    surface = getattr(state.electron_gun.emitter, "surface_model", None)
    if surface is not None and surface.coherence is not None:
        # tip_wave_pipeline sets the effective radial cap from the gun cap.
        stage_peak = max(stage_peak, request.surface.maximum_working_bytes+gun)
    output_cap = max(gun, column)
    possible_checkpoints = 1 if request.execution.segmented else 8
    cache_growth = min(request.execution.maximum_ram_cache_bytes,
                       possible_checkpoints*output_cap)
    if not request.execution.segmented:
        # The standalone gun cache has a distinct cap in this path.
        cache_growth += gun
    return ResourceClaim(stage_peak+cache_growth+256*1024**2)


@dataclass(frozen=True)
class _Preview:
    density: np.ndarray
    bounds_um: np.ndarray
    probability: float
    mode_count: int
    retained_bytes: int
    grid_shapes: tuple[tuple[int, int], ...] = ()
    pair_token: str | None = None
    comparison: object = None
    comparison_error: str | None = None
    axial_bz_t: float | None = None


def _intensity_preview(checkpoint, *, cancelled=lambda: False, bins=128):
    """Conservative affine-cell probability density, not detector counts.

    Values are probabilities per tip electron per square micrometre. Native
    cells from different mode grids are redistributed by area overlap, so a
    change of cell area does not change the displayed integrated probability.
    This is a display only; amplitudes and per-mode phase are untouched.
    """
    from temsim.physics.affine_cell_histogram import affine_cell_histogram

    bounds = np.array(((np.inf, -np.inf), (np.inf, -np.inf)))
    count = 0
    resident = 0
    grid_shapes = set()
    modes = checkpoint.beam.modes
    geometries = getattr(modes, "geometries", None)
    # Stored modes expose checked geometry without loading/checksumming each
    # large complex amplitude. The intensity pass below still reads every mode.
    geometry_rows = (geometries() if callable(geometries) else
                     ((mode.plane.amplitude.shape, mode.plane.basis_m,
                       mode.plane.origin_m) for mode in modes))
    for shape, basis, origin in geometry_rows:
        if cancelled():
            raise InterruptedError("Coherent beam display cancelled")
        ny, nx = shape
        grid_shapes.add((int(ny), int(nx)))
        indices = np.array(((-(nx//2)-.5, nx-1-nx//2+.5,
                             nx-1-nx//2+.5, -(nx//2)-.5),
                            (-(ny//2)-.5, -(ny//2)-.5,
                             ny-1-ny//2+.5, ny-1-ny//2+.5)))
        corners = (origin[:, None]+basis@indices)*1e6
        bounds[:, 0] = np.minimum(bounds[:, 0], corners.min(axis=1))
        bounds[:, 1] = np.maximum(bounds[:, 1], corners.max(axis=1))
        count += 1
    if not count:
        # A completely absorbed beam still has an honest zero intensity.
        bounds = np.array(((-.5, .5), (-.5, .5)))
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Executed wave has no finite observation-plane geometry")
    probability = np.zeros((bins, bins))
    for mode in modes:
        if cancelled():
            raise InterruptedError("Coherent beam display cancelled")
        weight = mode.weight_per_reference_electron*abs(mode.plane.amplitude)**2
        contribution, _ = affine_cell_histogram(weight, mode.plane.basis_m*1e6,
            mode.plane.origin_m*1e6, bounds, bins)
        probability += contribution
        resident += sum(getattr(mode.plane, name).nbytes for name in (
            "amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")
            if isinstance(getattr(mode.plane, name), np.ndarray))
    # A streamed checkpoint retains disk references, not all mode arrays.
    if hasattr(checkpoint.beam, "content_identity"):
        resident = 0
    cell_area = np.prod((bounds[:, 1]-bounds[:, 0])/bins)
    density = probability/cell_area
    density.setflags(write=False)
    bounds.setflags(write=False)
    return _Preview(density, bounds, float(probability.sum()), count,
                    resident+density.nbytes+bounds.nbytes, tuple(sorted(grid_shapes)))


class _Signals(QObject):
    solved = Signal(int, int, object, object)
    failed = Signal(int, str)
    progress = Signal(int, str)
    finished = Signal(int)


def _plane_axial_field(state, z_mm):
    """Capture diagnostic metadata from this execution's installed optics."""
    from temsim.input_io import input_scope
    from temsim.physics.core import bz
    with input_scope(state, inherit=False):
        return float(bz(np.array([z_mm]), state)[0])


class _WaveWorker(QRunnable):
    def __init__(self, generation, session, state, request, event, retained=(), *, observer=None,
                 pair_context=None):
        super().__init__()
        self.generation = generation
        self.session = session
        self.state = state
        self.request = request
        self.event = event
        self.observer = observer
        self.pair_context = pair_context
        # Inventory roots only: these executed displays are never supplied as
        # a replacement source to simulate_tip_wave.
        self.existing_result = (retained+((pair_context,) if pair_context is not None else ())
                                +((observer,) if observer is not None else ()))
        self._last_progress_time = 0.
        self.job_input_identity = f"coherent-session:{session}:Z:{request.observation_z_mm.hex()}"
        self.resource_claim = _wave_resource_claim(state, request)
        self.signals = _Signals()

    def _report_progress(self, done, total, text):
        now = monotonic()
        if not self.event.is_set() and (done == total or now-self._last_progress_time >= .05):
            self._last_progress_time = now
            self.signals.progress.emit(self.generation, f"{text} ({done}/{total})")

    def run(self):
        try:
            if self.pair_context is not None:
                from temsim.instrument_snapshot import capture_instrument_snapshot
                if capture_instrument_snapshot(self.state).digest != self.pair_context.instrument_identity:
                    raise ValueError("Captured coherent source or optics do not match this pair")
                self.pair_context.verify_particle()
            # CoordinatedPool holds the application's shared numerical lease
            # and its CPU budget. Do not create a nested independent pool.
            if self.observer is None:
                result = simulate_tip_wave(self.state, self.request, cancelled=self.event.is_set,
                    progress_callback=self._report_progress)
            else:
                result = self.observer.observe(self.request.observation_z_mm,
                    cancelled=self.event.is_set, progress_callback=self._report_progress)
            preview = _intensity_preview(result.checkpoint, cancelled=self.event.is_set)
            # Probability current needs the executed magnetic state, not the
            # live instrument which may have changed while this job ran.
            preview = replace(preview, axial_bz_t=_plane_axial_field(
                self.state, result.checkpoint.plane_z_mm))
            if self.pair_context is not None and not self.event.is_set():
                context = self.pair_context
                try:
                    if result.instrument_digest != context.instrument_identity:
                        raise ValueError("Executed wave identity does not match this captured pair")
                    from temsim.gui.beam_plane_data import sample_beam_plane
                    from temsim.physics.beam_comparison import compare_beam_planes
                    plane = sample_beam_plane(context.particle_result, result.checkpoint.plane_z_mm)
                    if (context.specimen_active and plane.z_mm > context.specimen_z_mm
                            and plane.provenance != "Specimen exit"):
                        raise ValueError("Executed specimen-exit particles are unavailable at this Z; the optical reference is not a specimen comparison")
                    comparison = compare_beam_planes(plane, result.checkpoint)
                    preview = replace(preview, pair_token=context.token, comparison=comparison)
                except (ValueError, TypeError) as error:
                    preview = replace(preview, pair_token=context.token,
                                      comparison_error=str(error))
            if not self.event.is_set():
                self.signals.solved.emit(self.generation, self.session, result, preview)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
        finally:
            # The exception variable has now released its numerical frames.
            # Drain retired device pools even after cancellation or failure.
            from temsim.physics.wave_device import release_device_memory
            try:
                release_device_memory()
            finally:
                self.signals.finished.emit(self.generation)


class _ObservableLabel(QLabel):
    text_changed = Signal(str)

    def setText(self, text):
        if text == self.text():
            return
        super().setText(text)
        self.text_changed.emit(text)


def _label(text="", *, observable=False):
    result = (_ObservableLabel if observable else QLabel)(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                   | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    result.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    result.setWordWrap(True)
    result.setMinimumWidth(0)
    return result


def _double(value, minimum, maximum, suffix="", decimals=5):
    control = QDoubleSpinBox()
    control.setDecimals(decimals)
    control.setRange(minimum, maximum)
    control.setValue(value)
    control.setSuffix(suffix)
    control.setKeyboardTracking(False)
    return control


def _integer(value, minimum, maximum):
    control = QSpinBox()
    control.setRange(minimum, maximum)
    control.setValue(value)
    return control


class CoherentIntensityView(QWidget):
    """A lazy display of an accepted preview; this widget never propagates waves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background="#080e19")
        self.plot.setLabel("bottom", "X", units="µm")
        self.plot.setLabel("left", "Y", units="µm")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.image = pg.ImageItem(axisOrder="col-major")
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.plot.addItem(self.image)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self._pending = None
        self._rendered = None
        self._bounds = None
        self.fit_button = QPushButton("Fit full")
        self.fit_button.setToolTip("Fit the displayed wave area once. Moving Z preserves your zoom and position.")
        self.fit_button.setEnabled(False)
        self.fit_button.clicked.connect(self.fit_full)

    @Slot(object, object, int)
    def set_observation(self, result, preview, scale):
        if preview is None:
            return
        self._pending = (result, preview, scale)
        if self.isVisible():
            self._render()

    def _render(self):
        if self._pending is None:
            return
        result, preview, scale = self._pending
        identity = (id(result), id(preview), scale)
        if identity == self._rendered:
            return
        # A changed wave domain is data geometry, not a request to move the
        # observer's viewport. Also stop persistent auto-range after a manual A.
        self.plot.disableAutoRange()
        density = preview.density
        maximum = max(float(density.max()), np.finfo(float).tiny)
        if scale == 1:
            display = np.log1p(1000.*(density/maximum))/np.log1p(1000.)
            self.image.setImage(display, autoLevels=False, levels=(0., 1.))
        else:
            self.image.setImage(density, autoLevels=False, levels=(0., maximum))
        bounds = preview.bounds_um
        first_image = self._bounds is None
        if self._bounds is None or not np.array_equal(bounds, self._bounds):
            self.image.setRect(QRectF(bounds[0, 0], bounds[1, 0],
                bounds[0, 1]-bounds[0, 0], bounds[1, 1]-bounds[1, 0]))
            self._bounds = bounds
        self.fit_button.setEnabled(True)
        if first_image:
            self.fit_full()
        label = ("log display of intensity density" if scale == 1
                 else "intensity density")
        # A long single-line title forces PlotItem's minimum width beyond the
        # narrow Ray sidebar and clips the right half of the physical screen.
        self.plot.setTitle(f"Exact Z {result.checkpoint.plane_z_mm:.9g} mm<br>{label}")
        self.plot.setToolTip("Intensity density / µm² per tip electron. "
                            "The logarithmic scale changes display only.")
        self._rendered = identity

    @Slot()
    def fit_full(self):
        """One-shot display fit; never propagate or modify a captured result."""
        if self._bounds is not None:
            self.plot.setRange(xRange=self._bounds[0], yRange=self._bounds[1], padding=.02)

    def showEvent(self, event):
        super().showEvent(event)
        self._render()


class CoherentBeamMirror(QWidget):
    """Ray Diagram readout of the page's one captured calculation session."""

    controls_requested = Signal()

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.setObjectName("rayCoherentBeamReadout")
        self.follow_ray = QCheckBox("Follow Ray Z")
        self.follow_ray.setChecked(owner.follow_ray.isChecked())
        self.follow_ray.setToolTip("Shares the Electron beam page setting. When unchecked, Ray Diagram Z does not change the observation plane.")
        self.follow_ray.toggled.connect(owner.follow_ray.setChecked)
        owner.follow_ray.toggled.connect(self.follow_ray.setChecked)
        self.intensity_scale = QComboBox()
        for index in range(owner.intensity_scale.count()):
            self.intensity_scale.addItem(owner.intensity_scale.itemText(index))
        self.intensity_scale.setCurrentIndex(owner.intensity_scale.currentIndex())
        self.intensity_scale.currentIndexChanged.connect(owner.intensity_scale.setCurrentIndex)
        owner.intensity_scale.currentIndexChanged.connect(self.intensity_scale.setCurrentIndex)
        self.controls_button = QPushButton("Controls…")
        self.controls_button.clicked.connect(self.controls_requested)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.follow_ray)
        toolbar.addWidget(self.intensity_scale)
        toolbar.addStretch()
        toolbar.addWidget(self.controls_button)
        self.status = _label(owner.status.text())
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.plane_positions = _label(owner.plane_positions.text())
        self.plane_positions.setToolTip(owner.plane_positions.toolTip())
        self.screen = CoherentIntensityView()
        toolbar.insertWidget(toolbar.count()-1, self.screen.fit_button)
        self.readout = _label(owner.readout.text())
        self.comparison_status = _label(owner.comparison_status.text())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(toolbar)
        layout.addWidget(self.plane_positions)
        layout.addWidget(self.status)
        layout.addWidget(self.screen, 1)
        layout.addWidget(self.readout)
        layout.addWidget(self.comparison_status)
        owner.status.text_changed.connect(self.status.setText)
        owner.plane_positions.text_changed.connect(self.plane_positions.setText)
        owner.readout.text_changed.connect(self.readout.setText)
        owner.comparison_status.text_changed.connect(self.comparison_status.setText)
        owner.observation_changed.connect(self.screen.set_observation)
        self.screen.set_observation(owner.result, owner.preview,
                                    owner.intensity_scale.currentIndex())


class CoherentBeamPage(QWidget):
    """Explicit calculation starts a session; subsequent Z queries reuse it."""

    QUERY_INTERVAL_MS = 40
    CACHE_LIMIT = 64
    observation_changed = Signal(object, object, int)
    source_applied = Signal(object)
    tip_editor_requested = Signal()
    paired_calculation_requested = Signal()
    pair_invalidated = Signal()
    _SCOPE = (
        "Development: physical tip → extraction → acceleration → installed column → exact Z. "
        "Ideal vacuum (no gas scattering/attenuation); fields and hardware absorption remain active. "
        "First stage supports observation after the gun exit; sample interiors and the energy-filter "
        "branch may be unsupported. Installed apertures and physical detector absorption remain active. "
        "An unsupported request is reported; it is not full TEM/STEM qualification."
    )

    def __init__(self, parent=None, *, state_provider=None):
        super().__init__(parent)
        self.setObjectName("coherentBeamPage")
        self.state_provider = state_provider
        self.source_applier = None
        self.paired_busy_provider = None
        self._surface_edge_phase_rad = None
        self._loaded_source_settings = None
        self._loaded_source_values = {}
        self._state = None
        self._captured = None
        self._request = None
        self._observation_session = None
        self._session = 0
        self._generation = 0
        self._worker = None
        self._pending_z = False
        self._closed = False
        self._stale = True
        self._session_ready = False
        self._cache = OrderedDict()
        self.result = None
        self.preview = None
        self._source_model = "gaussian"
        self._energy_samples_edited = False
        self._target_z_mm = 1.
        self._pair_context = None

        self.calculate_button = QPushButton("Calculate beam")
        self.calculate_button.setObjectName("calculateElectronBeam")
        self.calculate_button.setProperty("calculationAction", True)
        self.calculate_button.setStyleSheet(
            "QPushButton {background:#3ce878; color:#073519; border:1px solid #28b961; padding:5px 9px;}"
            "QPushButton:disabled {background:#29483a; color:#b5c5bc;}")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.compare_classical = QCheckBox("Compare classical rays")
        self.compare_classical.setObjectName("compareClassicalRays")
        self.compare_classical.setToolTip("Also calculate classical trajectories from the same applied Tip and optics. This approximation does not include interference. Applies to Current tip parameters; saved states retain individual wave results.")
        self.compare_classical.setEnabled(False)
        self.target_z = _double(1., -1e6, 1e6, " mm", 12)
        self.target_z.setObjectName("coherentObservationZ")
        self.follow_ray = QCheckBox("Follow Ray Z")
        self.follow_ray.setChecked(True)
        self.follow_ray.setToolTip("After an explicit successful calculation, query exact Ray Diagram Z in the same captured session.")
        self.intensity_scale = QComboBox()
        self.intensity_scale.addItems(("Linear intensity", "Log intensity"))
        self.intensity_scale.setToolTip("Display mapping only; executed intensity and complex fields are unchanged.")
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.calculate_button)
        toolbar.addWidget(self.cancel_button)
        toolbar.addWidget(QLabel("Z"))
        toolbar.addWidget(self.target_z)
        toolbar.addWidget(self.follow_ray)
        toolbar.addWidget(self.intensity_scale)
        toolbar.addStretch()

        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setObjectName("coherentZSlider")
        self.z_slider.setRange(0, 10000)
        self.z_slider.setTracking(True)
        self.z_slider.setToolTip("Browse exact observation planes continuously after Calculate. Intermediate queued positions are replaced by the latest Z.")
        self.z_range_min = _double(0., -1e6, 1e6, " mm", 6)
        self.z_range_max = _double(1., -1e6, 1e6, " mm", 6)
        for control in (self.z_range_min, self.z_range_max):
            control.setToolTip("Slider range only. Narrow the range for fine Z adjustment; this does not change the calculated optics.")
        self.z_range_min.setAccessibleName("Z slider range start")
        self.z_range_max.setAccessibleName("Z slider range end")
        browse = QHBoxLayout()
        browse.addWidget(QLabel("Z range"))
        self.z_range_start_label = QLabel("Start")
        self.z_range_end_label = QLabel("End")
        browse.addWidget(self.z_range_start_label)
        browse.addWidget(self.z_range_min)
        browse.addWidget(self.z_slider, 1)
        browse.addWidget(self.z_range_end_label)
        browse.addWidget(self.z_range_max)

        self.source_enabled = QCheckBox("Gaussian-Schell emission")
        self.tip_boundary = QComboBox()
        self.tip_boundary.setObjectName("tipBoundaryModel")
        self.tip_boundary.addItem("Driven emission · non-paraxial near tip", "driven_gaussian_schell")
        self.tip_boundary.addItem("Forward emission · paraxial", "forward_gaussian_schell")
        self.tip_boundary.setToolTip("Driven emission retains the low-energy near field and reflection. Rays are the geometric-optics comparison of the same Tip; wave diffraction is calculated separately. Apply tip parameters is required.")
        self.source_enabled.setObjectName("tipCoherenceModelEnabled")
        self.source_enabled.setToolTip("Select the Tip emission and coherence model. Apply tip parameters writes these physical inputs to the instrument; every calculation reads that Tip. Calculate never applies pending edits.")
        self.apply_source_button = QPushButton("Apply tip parameters")
        self.apply_source_button.setObjectName("applyTipParameters")
        self.apply_source_button.setToolTip("Apply these physical-tip inputs to the instrument and invalidate both results. Does not start a calculation or adjust lenses.")
        self.apply_source_button.setEnabled(False)
        self.edit_tip_button = QPushButton("Tip parameters…")
        self.edit_tip_button.setObjectName("tipParameterEditor")
        self.edit_tip_button.setToolTip("Edit the instrument's physical tip. A curved metal tip uses its actual surface and electrode fields for both rays and waves.")
        self.edit_tip_button.clicked.connect(self.tip_editor_requested)
        self.source_info = _label("No instrument inputs available.")
        self.source_assumption = _label()
        self.preflight_message = _label()
        self.preflight_message.setVisible(False)
        self.preflight_group = QGroupBox("Source check details")
        self.preflight_group.setCheckable(True)
        self.preflight_group.setChecked(False)
        self.preflight_details = QPlainTextEdit()
        self.preflight_details.setReadOnly(True)
        self.preflight_details.setMinimumHeight(150)
        self.preflight_details.setMaximumHeight(220)
        self.preflight_details.setToolTip("Read-only report of the last explicit calculation attempt. Select or copy text; viewing it never starts propagation.")
        self.preflight_details.setVisible(False)
        self.preflight_group.toggled.connect(self.preflight_details.setVisible)
        preflight_layout = QVBoxLayout(self.preflight_group)
        preflight_layout.addWidget(self.preflight_details)
        self.preflight_group.setVisible(False)
        self.angle_rms = _double(0., 0., 1e4, " mrad", 15)
        self.surface_mean = _double(.3, 1e-9, 1e6, " eV", 15)
        self.surface_rms = _double(.1, 0., 1e6, " eV", 15)
        self.angle_caption = QLabel("Incoherent angular RMS")
        self.mean_caption = QLabel("Tip mean kinetic energy")
        self.rms_caption = QLabel("Tip energy RMS")
        self.tip_fwhm = _double(5., 1e-6, 1e6, " nm", 12)
        self.tip_fwhm.setToolTip("Intensity FWHM of the prescribed planar Gaussian emission boundary at the physical tip. This is an emission width, not the metal apex radius or a downstream virtual source size.")
        self.tip_energy = _double(.3, 1e-9, 1e6, " eV", 15)
        self.tip_energy.setToolTip("Mean kinetic energy at the physical tip before extraction; this is not the energy spread.")
        self.tip_minimum_energy = _double(.01, 1e-9, 1e6, " eV", 15)
        self.tip_energy_spread = _double(.3, 0., 1e6, " eV", 15)
        self.tip_energy_spread.setToolTip("RMS-equivalent FWHM of the Tip's positive Young/Boersch energy law. The asymmetric distribution's literal half-maximum width can differ.")
        self.tip_offset_x = _double(0., -1e6, 1e6, " nm", 15)
        self.tip_offset_y = _double(0., -1e6, 1e6, " nm", 15)
        self.tip_curvature_x = _double(0., -1e12, 1e12, " m⁻¹", 15)
        self.tip_curvature_xy = _double(0., -1e12, 1e12, " m⁻¹", 15)
        self.tip_curvature_y = _double(0., -1e12, 1e12, " m⁻¹", 15)
        self.tip_tilt_x = _double(0., -1e4, 1e4, " mrad", 15)
        self.tip_tilt_y = _double(0., -1e4, 1e4, " mrad", 15)
        self.gaussian_inputs = QWidget()
        tip_form = QFormLayout(self.gaussian_inputs)
        tip_form.setContentsMargins(0, 0, 0, 0)
        for caption, control in (("Physical emission FWHM", self.tip_fwhm),
                ("Tip mean kinetic energy", self.tip_energy),
                ("Minimum kinetic energy", self.tip_minimum_energy),
                ("Energy spread (RMS-equiv. FWHM)", self.tip_energy_spread),
                ("Tip emission centre X", self.tip_offset_x),
                ("Tip emission centre Y", self.tip_offset_y),
                ("Wavefront curvature X", self.tip_curvature_x),
                ("Wavefront curvature XY", self.tip_curvature_xy),
                ("Wavefront curvature Y", self.tip_curvature_y),
                ("Wavefront tilt X", self.tip_tilt_x), ("Wavefront tilt Y", self.tip_tilt_y)):
            tip_form.addRow(caption, control)
        source_form = QFormLayout()
        source_form.addRow(self.angle_caption, self.angle_rms)
        source_form.addRow(self.mean_caption, self.surface_mean)
        source_form.addRow(self.rms_caption, self.surface_rms)

        self.advanced = QGroupBox("Advanced calculation options")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.advanced_body = QWidget()
        numerics = QFormLayout(self.advanced_body)
        numerics.addRow(self.compare_classical)
        default_request = TipWaveRequest()
        self.grid_pixels = _integer(128, 32, 4096)
        self.maximum_grid_pixels = _integer(default_request.wave_grid.maximum_pixels, 32, 65536)
        self.maximum_grid_pixels.setToolTip("Maximum cells per axis after automatic refinement. A budget failure never applies an unresolved operator or discards physical effects.")
        self.energy_samples = _integer(1, 1, 64)
        self.energy_samples.valueChanged.connect(self._mark_energy_samples_edited)
        self.tip_energy_spread.valueChanged.connect(self._refresh_energy_sample_default)
        self.material_trajectories = _integer(default_request.inelastic.trajectories_per_mode, 1, 1000000)
        self.material_trajectories.setToolTip("Independent material histories per spatial/energy mode. If every executed slice has exactly zero event probabilities, identical histories are represented once with their full weight. Any nonzero scattering or absorption probability keeps all requested histories; converge count and seed separately. Sample interactions remain active.")
        self.column_step = _double(.5, .000001, 100., " mm", 6)
        self.gun_step = _double(.05, .000001, 100., " mm", 6)
        self.gun_bore_step = _double(default_request.gun.bore_step_mm, .000001, 100., " mm", 6)
        self.gun_bore_step.setToolTip("Axial sampling of the gun body's absorbing bore projections. Apertures and interception remain active; refine independently when checking bore convergence.")
        self.gun_energy_step = _double(5., .005, 20., " %", 3)
        self.gun_energy_step.setToolTip("Maximum kinetic-energy change within a gun integration step. Lower values resolve rapid acceleration near the tip more finely.")
        self.segment_steps = _integer(128, 1, 100000)
        self.working_gib = _double(8., .25, 80., " GiB", 2)
        self.working_gib.setToolTip("Working limit for each numerical stage. Coupled tip/radial stages can overlap; shared admission also accounts for retained checkpoints.")
        self.gpu_working_gib = _double(24., .25, 1024., " GiB", 2)
        self.gpu_working_gib.setObjectName("coherentGpuWorkingLimit")
        self.gpu_working_gib.setToolTip("GPU working-memory limit for coherent column and material operators. The toolbar compute policy selects the requested device; actual device use and any CPU fallback appear with the result. Gun propagation remains on CPU. This limit is separate from the CPU stage and cache limits.")
        self.ram_cache_gib = _double(8., .25, 80., " GiB", 2)
        self.disk_cache_gib = _double(192., 1., 2048., " GiB", 2)
        for caption, control in (("Initial grid pixels", self.grid_pixels),
                ("Maximum refined grid pixels", self.maximum_grid_pixels),
                ("Energy samples", self.energy_samples),
                ("Material trajectories per mode", self.material_trajectories),
                ("Column step", self.column_step),
                ("Gun field step", self.gun_step), ("Gun bore sampling step", self.gun_bore_step),
                ("Gun energy change per step", self.gun_energy_step),
                ("Steps per segment", self.segment_steps),
                ("Working limit per stage", self.working_gib), ("GPU working limit", self.gpu_working_gib),
                ("RAM cache limit", self.ram_cache_gib),
                ("Disk cache limit", self.disk_cache_gib)):
            numerics.addRow(caption, control)
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.addWidget(self.advanced_body)
        self.advanced_body.setVisible(False)
        self.advanced.toggled.connect(self.advanced_body.setVisible)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        from temsim.gui.coherent_state_list import TipStateList
        self.state_list = TipStateList()
        controls_layout.addWidget(self.state_list)
        self.tip_parameters = QGroupBox("Tip parameters")
        tip_layout = QVBoxLayout(self.tip_parameters)
        tip_layout.addWidget(self.source_enabled)
        tip_layout.addWidget(self.tip_boundary)
        tip_layout.addWidget(self.edit_tip_button)
        tip_layout.addWidget(self.source_info)
        tip_layout.addWidget(self.source_assumption)
        tip_layout.addWidget(self.gaussian_inputs)
        tip_layout.addLayout(source_form)
        tip_layout.addWidget(self.apply_source_button)
        controls_layout.addWidget(self.tip_parameters)
        controls_layout.addWidget(self.preflight_message)
        controls_layout.addWidget(self.preflight_group)
        controls_layout.addWidget(self.advanced)
        controls_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(250)
        self.screen = CoherentIntensityView()
        from temsim.gui.electron_beam_observation import ElectronBeamObservation
        self.observation = ElectronBeamObservation(self.screen)
        self.plot, self.image = self.screen.plot, self.screen.image
        self.observation_changed.connect(self.observation.set_observation)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.observation)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 900])
        self.scope_label = _label(self._SCOPE)
        self.scope_label.setStyleSheet("color:#d3ab61;")
        self.status = _label("Apply the Tip parameters, then click Calculate beam. Choose an observation without recalculating propagation.", observable=True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.plane_positions = _label(observable=True)
        self.plane_positions.setObjectName("coherentPlanePositions")
        self.plane_positions.setToolTip("Requested Z is the newest input. Calculating Z is the submitted worker target, including any wait for shared resources. Displayed Z belongs to the last completed image; moving the cursor does not relabel it.")
        self._update_plane_positions()
        self.readout = _label("Screen intensity density per tip electron; no detector counts or aggregate phase.", observable=True)
        self.comparison_status = _label("Optional classical comparison: enable Compare classical rays under Advanced calculation options before Calculate beam.", observable=True)
        self.comparison_status.setObjectName("pairedBeamStatus")
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setObjectName("pairedBeamComparison")
        self.comparison_table.setHorizontalHeaderLabels(("Same-Z observable", "Particles", "Coherent modes"))
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison_table.setMaximumHeight(205)
        self.comparison_table.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(toolbar)
        layout.addLayout(browse)
        layout.addWidget(self.scope_label)
        layout.addWidget(self.plane_positions)
        layout.addWidget(self.status)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.readout)
        layout.addWidget(self.comparison_status)
        layout.addWidget(self.comparison_table)

        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.coordinator.register_retained(self, "_retained_wave_roots")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_query)
        self.calculate_button.clicked.connect(self.calculate)
        self.cancel_button.clicked.connect(self.cancel)
        self.target_z.valueChanged.connect(self._z_changed)
        self.z_slider.valueChanged.connect(self._slider_changed)
        self.z_range_min.valueChanged.connect(self._sync_z_slider)
        self.z_range_max.valueChanged.connect(self._sync_z_slider)
        self.source_enabled.toggled.connect(self._mark_source_draft)
        self.tip_boundary.currentIndexChanged.connect(self._mark_source_draft)
        self.apply_source_button.clicked.connect(self.apply_source)
        self.intensity_scale.currentIndexChanged.connect(self._refresh_intensity)
        for control in (self.angle_rms, self.surface_mean, self.surface_rms,
                        self.tip_fwhm, self.tip_energy, self.tip_minimum_energy, self.tip_energy_spread,
                        self.tip_offset_x, self.tip_offset_y,
                        self.tip_curvature_x, self.tip_curvature_xy, self.tip_curvature_y,
                        self.tip_tilt_x, self.tip_tilt_y):
            control.valueChanged.connect(self._mark_source_draft)
        for control in (self.grid_pixels, self.maximum_grid_pixels, self.energy_samples,
                        self.material_trajectories, self.column_step,
                        self.gun_step, self.gun_bore_step, self.gun_energy_step,
                        self.segment_steps, self.working_gib, self.gpu_working_gib,
                        self.ram_cache_gib, self.disk_cache_gib):
            control.valueChanged.connect(self.mark_inputs_stale)
        self._show_source_controls()
        from temsim.gui.coherent_state_controller import CoherentStateController
        self.state_set = CoherentStateController(self, self.state_list)

    def _mark_energy_samples_edited(self, *_args):
        self._energy_samples_edited = True

    def _set_default_energy_samples(self, count):
        if not self._energy_samples_edited:
            self.energy_samples.blockSignals(True)
            self.energy_samples.setValue(count)
            self.energy_samples.blockSignals(False)

    def _refresh_energy_sample_default(self, *_args):
        if self._source_model == "gaussian":
            self._set_default_energy_samples(1 if self.tip_energy_spread.value() == 0. else 9)

    def _show_source_controls(self):
        surface = self._source_model == "surface"
        self.source_enabled.setText("Constant-phase surface emission" if surface else "Gaussian-Schell emission")
        self.gaussian_inputs.setVisible(not surface)
        self.tip_boundary.setVisible(not surface)
        for control in (self.angle_caption, self.angle_rms):
            control.setVisible(not surface)
        for control in (self.mean_caption, self.surface_mean, self.rms_caption, self.surface_rms):
            control.setVisible(surface)

    def set_state(self, state):
        if self._closed:
            return
        self._state = state
        self.mark_inputs_stale()
        self.apply_source_button.setEnabled(False)
        self._loaded_source_settings = None
        self._loaded_source_values = None
        if state is None:
            self.source_info.setText("No instrument inputs available.")
            return
        emitter = getattr(getattr(state, "electron_gun", None), "emitter", None)
        if emitter is None:
            self.source_info.setText("No physical tip input fields are available for this source. Coherent calculation requires a supported physical tip.")
            return
        try:
            settings = source_settings_from_state(state)
        except ValueError as error:
            self.source_enabled.blockSignals(True)
            self.source_enabled.setChecked(False)
            self.source_enabled.blockSignals(False)
            self.source_enabled.setEnabled(False)
            self.gaussian_inputs.hide()
            for control in (self.angle_caption, self.angle_rms, self.mean_caption,
                            self.surface_mean, self.rms_caption, self.surface_rms):
                control.hide()
            self.source_info.setText(str(error))
            self.source_assumption.clear()
            return
        self.source_enabled.setEnabled(True)
        surface = emitter.surface_model
        self._source_model = "gaussian" if surface is None else "surface"
        self._show_source_controls()
        self._surface_edge_phase_rad = settings.surface_edge_phase_rad
        self.source_enabled.blockSignals(True)
        self.source_enabled.setChecked(settings.enabled)
        self.source_enabled.blockSignals(False)
        self.apply_source_button.setEnabled(True)
        if surface is None:
            self.tip_boundary.blockSignals(True)
            self.tip_boundary.setCurrentIndex(self.tip_boundary.findData(settings.boundary_model))
            self.tip_boundary.blockSignals(False)
            for control, field in (
                    (self.tip_fwhm, "tip_fwhm_nm"),
                    (self.tip_energy, "tip_mean_energy_ev"),
                    (self.tip_minimum_energy, "tip_minimum_energy_ev"),
                    (self.tip_energy_spread, "tip_energy_spread_fwhm_ev"),
                    (self.tip_offset_x, "tip_offset_x_nm"),
                    (self.tip_offset_y, "tip_offset_y_nm"),
                    (self.tip_curvature_x, "tip_curvature_x_m1"),
                    (self.tip_curvature_xy, "tip_curvature_xy_m1"),
                    (self.tip_curvature_y, "tip_curvature_y_m1"),
                    (self.tip_tilt_x, "tip_tilt_x_mrad"),
                    (self.tip_tilt_y, "tip_tilt_y_mrad"),
                    (self.angle_rms, "incoherent_angle_rms_mrad")):
                control.blockSignals(True)
                control.setValue(getattr(settings, field))
                control.blockSignals(False)
            self.source_info.setText(
                f"Planar Gaussian boundary | emission FWHM {emitter.virtual_source_fwhm_nm:g} nm | "
                f"mean kinetic energy {emitter.emission_energy_ev:g} eV | "
                f"energy spread {emitter.energy_spread_fwhm_ev:g} eV (RMS-equiv. FWHM) | "
                f"current {emitter.emission_current_na:g} nA.")
            self.source_assumption.setText(
                "These are the instrument's Tip parameters. Every calculation reads the applied values. "
                "Driven emission propagates the low-energy near field before continuing through the gun. "
                "Rays sample the same intensity, energy and local phase directions as a geometric-optics "
                "comparison; the wave additionally includes diffraction and reflection. Emission FWHM and wavefront "
                "curvature are not the metal tip radius or mechanical curvature. "
                "This boundary uses a planar cathode field, not the stored metal apex radius. "
                "Use Tip parameters to edit the physical geometry and emission model. "
                "Forward emission retains its separate paraxial support check. "
                "Neither model changes the source width or energy to pass a check.")
            self._refresh_energy_sample_default()
        else:
            for control, value in ((self.surface_mean, settings.surface_mean_energy_ev),
                                   (self.surface_rms, settings.surface_energy_rms_ev)):
                control.blockSignals(True)
                control.setEnabled(value is not None)
                if value is not None:
                    control.setValue(value)
                control.blockSignals(False)
            self.source_info.setText(f"Curved physical tip | apex radius {surface.geometry.apex_radius_nm:g} nm | cap {surface.emission.cap_half_angle_deg:g}° | reference current {surface.current_na:g} nA. The actual metal geometry is used by the electrode-field solver.")
            historical = surface.coherence is not None and not surface.shared_boundary
            self.surface_mean.setReadOnly(historical)
            self.surface_rms.setReadOnly(historical)
            self.source_enabled.setEnabled(not historical)
            self.apply_source_button.setEnabled(not historical)
            if historical:
                self.source_assumption.setText(
                    "Historical wave-only reservoir: no matching particle source. Inputs are read-only. "
                    "Use Tip parameters to explicitly replace this historical boundary.")
            else:
                self.source_assumption.setText(
                    "These are the instrument's Tip parameters. Constant-phase surface emission "
                    "selects a cosine-cap flux profile and positive total-energy law. "
                    "Rays follow the local surface normals as a geometric-optics "
                    "comparison; waves additionally retain diffraction and reflection. "
                    "Reference current is outward injected flux; net escaping current is a result. "
                    "This supported axisymmetric boundary has constant surface phase. "
                    "Mean energy and RMS are editable physical inputs, not fixed acceptance values.")
            self._set_default_energy_samples(1 if settings.surface_energy_rms_ev == 0. else 9)
        self._loaded_source_settings = settings
        self._loaded_source_values = asdict(self._draft_settings())
        if self.result is None:
            # Non-vacuum specimen interiors need a truncated material
            # operator. Choose the actual exit boundary for the first view.
            default_z = float(state.sample.z_mm)+float(getattr(state.sample, "thickness_nm", 0.))*.5e-6
            self.target_z.blockSignals(True)
            self.target_z.setValue(default_z)
            self.target_z.blockSignals(False)
            self._target_z_mm = default_z
            self._update_plane_positions()
            self.target_z.setToolTip(f"Initial target: specimen exit Z {default_z:.12g} mm. A finite sample interior is outside this development view.")
            start = float(getattr(state.electron_gun, "exit_plane_z_mm", 0.))
            positions = [default_z, start+1.]
            for device in getattr(state, "recording_planes", ()):
                positions.append(float(device.z_mm))
            self.z_range_min.setValue(start)
            self.z_range_max.setValue(max(positions))
            self._sync_z_slider()

    @Slot()
    def _mark_source_draft(self, *_args):
        if self._closed:
            return
        self.mark_inputs_stale()
        self.status.setText("Tip source draft changed. Click Apply tip parameters before Calculate beam."
                            + (" Previous optics remain displayed." if self.result is not None else ""))

    @Slot()
    def apply_source(self):
        """Publish a validated tip edit, never a propagation request."""
        if self._closed:
            return
        target = self._target_z_mm
        lower, upper = self.z_range_min.value(), self.z_range_max.value()
        try:
            state = self.state_provider() if self.state_provider is not None else self._state
            if state is None:
                raise ValueError("No current instrument inputs are available")
            settings = self._settings()
            previous = source_settings_from_state(state)
            generation = self._generation
            if self.source_applier is not None:
                state = self.source_applier(settings)
                if state is None:
                    raise ValueError("Tip application did not return the current instrument")
            else:
                candidate = candidate_tip_emission(state, settings)
                gun = state.electron_gun
                if candidate.to_dict() != gun.to_dict():
                    gun.emitter = candidate.emitter
                    gun.source_representation = candidate.source_representation
                    gun._trace_cache = gun._trace_cache_key = None
        except Exception as error:
            self.status.setStyleSheet("color:#ffb86b;")
            self.status.setText(f"Tip not applied; active source unchanged: {error}")
            return
        if (state is self._state and previous == source_settings_from_state(state)
                and generation == self._generation):
            self.status.setStyleSheet("")
            self.status.setText("Tip unchanged. Existing calculation and cache retained.")
            return
        self.set_state(state)
        # Source editing does not change the observation cursor or its range.
        self.target_z.blockSignals(True)
        self.target_z.setValue(target)
        self.target_z.blockSignals(False)
        self._target_z_mm = target
        self.z_range_min.setValue(lower)
        self.z_range_max.setValue(upper)
        self._sync_z_slider()
        self._update_plane_positions()
        surface = state.electron_gun.emitter.surface_model
        message = ("Saved surface wave source applied; its particle representation remains unavailable."
                   if surface is not None and surface.coherence is not None and not surface.shared_boundary else
                   "Tip applied. Particle and coherent results require recalculation.")
        self.status.setStyleSheet("")
        self.status.setText(message+" No calculation started.")
        self.source_applied.emit(state)

    @Slot()
    def _sync_z_slider(self, *_args):
        lower, upper = self.z_range_min.value(), self.z_range_max.value()
        self.z_slider.setEnabled(upper > lower)
        if upper > lower:
            fraction = min(1., max(0., (self._target_z_mm-lower)/(upper-lower)))
            value = round(fraction*self.z_slider.maximum())
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(value)
            self.z_slider.blockSignals(False)

    @Slot(int)
    def _slider_changed(self, value):
        lower, upper = self.z_range_min.value(), self.z_range_max.value()
        if upper <= lower:
            return
        z_mm = lower+(upper-lower)*value/self.z_slider.maximum()
        self.target_z.blockSignals(True)
        self.target_z.setValue(z_mm)
        self.target_z.blockSignals(False)
        self._z_changed(z_mm)

    def _cancel_request(self):
        self._generation += 1
        self.timer.stop()
        self._pending_z = False
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()
        self._worker = None
        self._update_cancel_enabled()
        self._update_plane_positions()

    def _update_cancel_enabled(self):
        states = getattr(self, "state_set", None)
        paired_busy = bool(self.paired_busy_provider and self.paired_busy_provider())
        self.cancel_button.setEnabled(self._worker is not None
            or (states is not None and states.worker is not None) or paired_busy)

    @Slot()
    def mark_inputs_stale(self, *_args):
        if self._closed:
            return
        if hasattr(self, "state_set"):
            self.state_set.inputs_changed()
        self.invalidate_pair("Pair invalidated by input changes; calculate a new pair to compare results.")
        self._cancel_request()
        self._stale = True
        self._session_ready = False
        self._cache.clear()
        self._observation_session = None
        self.status.setStyleSheet("")
        if self.preflight_details.toPlainText():
            self.preflight_message.setText("Previous source check; inputs changed. Click Calculate beam to check the current settings. The retained report describes the previous attempt.")
            self.preflight_message.setStyleSheet("color:#d3ab61;")
        self.status.setText("Inputs changed. Click Calculate beam to capture a new session."
                            + (" Previous optics remain displayed." if self.result is not None else ""))

    def _show_source_check(self, summary, settings, error=None):
        """Refresh the report while preserving the user's detail visibility."""
        data = None if summary is None else thaw_json(summary)
        report = {"source_summary": data,
                  "tip_inputs": None if settings is None else asdict(settings),
                  "observation_z_mm": self._target_z_mm, "error": error,
                  "scope": "Source checks only; not propagation or image qualification"}
        self.preflight_details.setPlainText(json.dumps(report, indent=2, allow_nan=False))
        self.preflight_group.setVisible(True)
        self.preflight_message.setVisible(True)
        ready = data is not None and data.get("status") == "SOURCE_READY"
        if ready and error is None:
            self.preflight_message.setText("Source check passed; this is not propagation or image qualification. Installed fields, apertures, material and sampling are checked during execution.")
            self.preflight_message.setStyleSheet("")
            return
        if ready:
            self.preflight_message.setText(f"Source check passed, but calculation setup rejected; propagation did not start: {error}. This is not propagation or image qualification.")
            self.preflight_message.setStyleSheet("color:#ffb86b;")
            return
        lines = ["Tip source check rejected; propagation did not start." if data is not None and not ready
                 else "Calculation setup rejected; propagation did not start."]
        if data is not None:
            inputs = data.get("inputs", {})
            for key, label, unit in (("virtual_source_fwhm_nm", "Physical emission FWHM", "nm"),
                    ("emission_energy_ev", "Tip mean kinetic energy", "eV"),
                    ("energy_spread_fwhm_ev", "Energy width (RMS-equivalent FWHM)", "eV")):
                if key in inputs:
                    lines.append(f"{label}: {inputs[key]:.9g} {unit}.")
            domain = data.get("source_domain", {})
            if domain:
                def finite_number(key):
                    value = domain.get(key)
                    return (float(value) if not isinstance(value, bool) and isinstance(value, Real)
                            and math.isfinite(value) else None)

                energy = finite_number("minimum_evaluated_energy_ev")
                if energy is not None:
                    lines.append(f"Evaluated at the physical tip before acceleration: {energy:.9g} eV.")
                bound = finite_number("support_radius_over_p")
                if bound is not None:
                    lines.append(f"Conservative transverse momentum / total momentum bound: {bound:.9g}.")
                budget = finite_number("paraxial_generator_relative_error_budget")
                tail = finite_number("tail_probability_budget")
                if budget is not None and 0 < budget < .5 and tail is not None and 0 < tail < 1:
                    # Invert the existing generator-error bound for display only.
                    # This does not change or replace physical admission.
                    limit = math.sqrt(4*budget*(1-budget))
                    lines.append(f"Current model limit: {limit:.6f} at {budget:.1%} generator error and tail probability {tail:g}.")
                else:
                    lines.append("Model limit unavailable in this source report; no physical values were substituted.")
                lines.append("Particle ray count or observation Z cannot change this source check. The bound is a probability-region estimate, not an individual electron angle.")
            lines.append("Review the tip model and its validity range. Inputs are not adjusted automatically; source admission alone does not guarantee propagation.")
        else:
            lines.append(str(error))
        self.preflight_message.setText("\n".join(lines))
        self.preflight_message.setStyleSheet("color:#ffb86b;")

    def _settings(self):
        draft = self._draft_settings()
        if self._loaded_source_settings is None:
            return draft
        # Spin boxes round for display. Unedited values retain all original
        # bits on Apply as well as Calculate, preserving source cache identity.
        unchanged = {name: getattr(self._loaded_source_settings, name)
                     for name, value in asdict(draft).items()
                     if name != "enabled" and value == self._loaded_source_values.get(name)}
        return replace(draft, **unchanged)

    def _draft_settings(self):
        if self._source_model == "surface":
            return TipEmissionSettings(enabled=self.source_enabled.isChecked(),
                surface_mean_energy_ev=self.surface_mean.value() if self.surface_mean.isEnabled() else None,
                surface_energy_rms_ev=self.surface_rms.value() if self.surface_rms.isEnabled() else None,
                surface_edge_phase_rad=self._surface_edge_phase_rad)
        return TipEmissionSettings(enabled=self.source_enabled.isChecked(),
            boundary_model=self.tip_boundary.currentData(),
            incoherent_angle_rms_mrad=self.angle_rms.value(),
            tip_fwhm_nm=self.tip_fwhm.value(), tip_mean_energy_ev=self.tip_energy.value(),
            tip_minimum_energy_ev=self.tip_minimum_energy.value(),
            tip_energy_spread_fwhm_ev=self.tip_energy_spread.value(),
            tip_offset_x_nm=self.tip_offset_x.value(), tip_offset_y_nm=self.tip_offset_y.value(),
            tip_curvature_x_m1=self.tip_curvature_x.value(), tip_curvature_xy_m1=self.tip_curvature_xy.value(),
            tip_curvature_y_m1=self.tip_curvature_y.value(),
            tip_tilt_x_mrad=self.tip_tilt_x.value(), tip_tilt_y_mrad=self.tip_tilt_y.value())

    def _make_request(self, state=None):
        default = TipWaveRequest()
        working = round(self.working_gib.value()*GIB)
        if state is None:
            state = self.state_provider() if self.state_provider is not None else self._state
        compute_backend = getattr(state, "acceleration_backend", "Auto")
        return replace(default, stop="plane", observation_z_mm=self._target_z_mm,
            source=replace(default.source, grid_pixels=self.grid_pixels.value(),
                           energy_samples=self.energy_samples.value()),
            surface=replace(default.surface, energy_samples=self.energy_samples.value(),
                            maximum_working_bytes=working),
            gun=replace(default.gun, field_step_mm=self.gun_step.value(),
                        bore_step_mm=self.gun_bore_step.value(),
                        maximum_fractional_energy_change=self.gun_energy_step.value()/100.,
                        maximum_checkpoint_bytes=working),
            radial_gun=replace(default.radial_gun, maximum_working_bytes=working),
            column_step_mm=self.column_step.value(),
            wave_grid=replace(default.wave_grid, maximum_pixels=self.maximum_grid_pixels.value(),
                              maximum_working_bytes=working, compute_backend=compute_backend,
                              acceleration_enabled=getattr(state, "acceleration_enabled", True),
                              maximum_device_working_bytes=round(self.gpu_working_gib.value()*GIB)),
            inelastic=replace(default.inelastic, trajectories_per_mode=self.material_trajectories.value()),
            maximum_readout_bytes=working,
            execution=replace(default.execution, segment_steps=self.segment_steps.value(),
                maximum_ram_cache_bytes=round(self.ram_cache_gib.value()*GIB),
                maximum_disk_cache_bytes=round(self.disk_cache_gib.value()*GIB))).validate()

    def capture_calculation_inputs(self):
        """Check drafts and freeze one source, optics and numerical request."""
        summary = None
        settings = None
        try:
            settings = self._settings()
            state = self.state_provider() if self.state_provider is not None else self._state
            if state is None:
                raise ValueError("No current instrument inputs are available")
            captured = prepare_coherent_state(state, settings)
            request = self._make_request(captured)
            summary = wave_input_summary(captured, request)
            if summary["status"] != "SOURCE_READY":
                raise ValueError(summary.get("reason", "Coherent tip inputs are not ready"))
            initial_bytes = summary.get("minimum_initial_wave_bytes") or 0
            if initial_bytes > request.wave_grid.maximum_working_bytes:
                raise MemoryError(f"Initial complete mode fields need at least {initial_bytes} bytes, "
                    f"above the selected working limit {request.wave_grid.maximum_working_bytes}. "
                    "Increase the numerical working budget; no source modes were removed.")
        except Exception as error:
            self._stale = True
            self._show_source_check(summary, settings, str(error))
            self.status.setStyleSheet("color:#ffb86b;")
            prefix = ("Tip source check rejected; propagation did not start" if summary is not None
                      and summary.get("status") != "SOURCE_READY" else "Coherent beam not calculated; propagation did not start")
            self.status.setText(f"{prefix}: {error}"
                                + (" Previous result remains displayed." if self.result is not None else ""))
            raise
        self._show_source_check(summary, settings)
        return captured, request, summary

    @Slot()
    def calculate(self):
        if self._closed:
            return
        if self.state_set.viewing_states():
            self.state_set.calculate()
            return
        # Closing the advanced group hides its controls; it must not change
        # the selected calculation. Only the explicit availability flag on
        # this checkbox decides whether a host supports classical pairing.
        if (self.compare_classical.isEnabledTo(self.advanced_body)
                and self.compare_classical.isChecked()):
            self.paired_calculation_requested.emit()
            return
        self.invalidate_pair("Independent coherent calculation; no matched particle result is claimed.")
        self._cancel_request()
        self._session_ready = False
        self._cache.clear()
        self._observation_session = None
        try:
            captured, request, summary = self.capture_calculation_inputs()
        except Exception:
            return
        self.start_captured_calculation(captured, request, summary)

    def set_projection_angle(self, angle):
        self.observation.set_projection_angle(angle)

    def start_captured_calculation(self, captured, request, summary, *, pair_context=None):
        """Run already validated captured inputs; never re-read live controls."""
        if self._closed:
            return
        self.state_set.cancel()
        self.state_list.mode.setCurrentIndex(self.state_list.mode.findData("current"))
        self._cancel_request()
        self._session += 1
        self._session_ready = False
        self._cache.clear()
        self._pair_context = pair_context
        if pair_context is not None:
            self.comparison_status.setText(
                f"Pair {pair_context.token[:8]} | input identity {pair_context.physical_identity[:12]} | "
                "particles completed; waiting for a same-Z coherent plane.")
        self.comparison_table.hide()
        self.status.setStyleSheet("")
        self._captured = captured
        self._request = request
        self._observation_session = TipWaveObservationSession(captured, request)
        self._stale = False
        self.status.setText(f"Source ready | {summary['model']} | {summary['mode_count']} modes. Queued exact-Z calculation; field support is checked during execution.")
        self._start_query()

    def invalidate_pair(self, message="Paired calculation cancelled; previous images are retained."):
        self._pair_context = None
        self.comparison_table.hide()
        self.comparison_table.setRowCount(0)
        self.comparison_status.setText(message)
        self.pair_invalidated.emit()

    def _display_comparison(self, preview):
        context = self._pair_context
        if context is None or preview.pair_token != context.token:
            self.comparison_table.hide()
            return
        if preview.comparison is None:
            self.comparison_table.hide()
            self.comparison_status.setText(f"Pair {context.token[:8]} | comparison unavailable: {preview.comparison_error or 'no matched data'}")
            return
        values = preview.comparison
        particle, wave = values["particle"], values["wave"]
        self.comparison_status.setText(
            f"Pair {context.token[:8]} | same captured inputs {context.physical_identity[:12]} | "
            f"compared Z {wave['z_mm']:.9g} mm | laboratory X/Y. "
            "Interference can change wave intensity and width; agreement is not assumed. "
            "Wave energy is the propagated mode reference. Phase belongs to individual wave modes only.")
        self.comparison_status.setToolTip(
            f"Particle energy: {particle['energy_definition']}\nWave energy: {wave['energy_definition']}\n"
            "Statistics use the complete retained populations, not the plotted points or image viewport.")
        rows = (("Mean energy (eV)", "mean_energy_ev", 1.),
                ("Energy RMS (eV)", "rms_energy_ev", 1.),
                ("Current (pA)", "current_a", 1e12),
                ("Tip reference fraction (%)", "source_fraction", 100.),
                ("Centre X, Y (µm)", "centroid_xy_m", 1e6),
                ("RMS X, Y (µm)", "rms_xy_m", 1e6),
                ("Radial RMS (µm)", "radial_rms_m", 1e6))
        self.comparison_table.setRowCount(len(rows))
        for row, (label, key, scale) in enumerate(rows):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(label))
            for col, data in ((1, particle), (2, wave)):
                value = data[key]
                text = ("Unavailable" if value is None else
                        ", ".join(f"{float(v)*scale:.6g}" for v in value)
                        if isinstance(value, (tuple, list)) else f"{float(value)*scale:.6g}")
                self.comparison_table.setItem(row, col, QTableWidgetItem(text))
        self.comparison_table.show()

    def _retained_wave_roots(self):
        """Keep idle wave products visible to all shared calculation owners."""
        return self.result, self.preview, self._cache, self._captured, self._pair_context, self._observation_session

    def _update_plane_positions(self):
        worker = self._worker
        if hasattr(self, "state_set") and self.state_set.worker is not None:
            worker = self.state_set.worker
        calculating = (f"{worker.request.observation_z_mm:.9g} mm"
                       if worker is not None else "—")
        displayed = (f"{self.result.checkpoint.plane_z_mm:.9g} mm"
                     if self.result is not None else "—")
        self.plane_positions.setText(
            f"Requested Z {self._target_z_mm:.9g} mm | "
            f"Calculating Z {calculating} | Displayed Z {displayed}")

    @Slot(float)
    def set_selected_z(self, z_mm):
        if self._closed or not self.follow_ray.isChecked():
            return
        if isinstance(z_mm, bool) or not isinstance(z_mm, Real):
            return
        z_mm = float(z_mm)
        if not math.isfinite(z_mm) or not self.target_z.minimum() <= z_mm <= self.target_z.maximum():
            return
        # Retain the exact linked float even if the spinbox display rounds it.
        if z_mm == self._target_z_mm:
            return
        self.target_z.blockSignals(True)
        self.target_z.setValue(z_mm)
        self.target_z.blockSignals(False)
        self._z_changed(z_mm)

    @Slot(float)
    def _z_changed(self, z_mm):
        if self._closed:
            return
        self._target_z_mm = float(z_mm)
        self._update_plane_positions()
        self._sync_z_slider()
        if self.state_set.viewing_states():
            self.state_set.set_z()
            return
        if self._stale or (not self._session_ready and self._worker is None):
            self.status.setText("Click Calculate beam to start a session at this Z."
                                + (" Previous optics remain displayed." if self.result is not None else ""))
            return
        cached = self._cache.get(float(z_mm))
        if cached is not None:
            self._pending_z = False
            self.timer.stop()
            self._cache.move_to_end(float(z_mm))
            self._display(*cached)
            return
        self._pending_z = True
        shown = (f" Showing completed Z {self.result.checkpoint.plane_z_mm:.9g} mm."
                 if self.result is not None else " First upstream calculation is still running.")
        self.status.setText(f"Exact Z {z_mm:.9g} mm requested.{shown}")
        # Throttle, never debounce or cancel on motion: a stream of input must
        # make progress. At most one running solve and one latest Z are owned.
        if self._worker is None and not self.timer.isActive():
            self.timer.start(self.QUERY_INTERVAL_MS)

    @Slot()
    def _start_query(self):
        if self._closed or self._stale or self._captured is None or self._worker is not None:
            return
        self.timer.stop()
        self._pending_z = False
        self._generation += 1
        request = replace(self._request, observation_z_mm=self._target_z_mm)
        retained = tuple(self._cache.values())+(self.result, self.preview)
        worker = _WaveWorker(self._generation, self._session, self._captured, request, Event(), retained,
                             observer=self._observation_session, pair_context=self._pair_context)
        worker.signals.solved.connect(self._solved)
        worker.signals.failed.connect(self._failed)
        worker.signals.progress.connect(self._progress)
        worker.signals.finished.connect(self._finished)
        self._worker = worker
        self._update_plane_positions()
        self.cancel_button.setEnabled(True)
        self.pool.start(worker)

    @Slot(int, int, object, object)
    def _solved(self, generation, session, result, preview):
        if self._closed or self._stale or generation != self._generation or session != self._session:
            return
        z_mm = float(result.checkpoint.plane_z_mm)
        submitted = (self._worker.request.observation_z_mm if self._worker is not None
                     else self._target_z_mm)
        if z_mm != submitted:
            self.status.setText("Rejected wave result: executed Z differs from the requested observation plane.")
            return
        self._session_ready = True
        self._cache[z_mm] = (result, preview)
        self._cache.move_to_end(z_mm)
        if self._target_z_mm in self._cache:
            self._cache.move_to_end(self._target_z_mm)
            self._pending_z = False
        budget = self._request.execution.maximum_ram_cache_bytes
        while len(self._cache) > self.CACHE_LIMIT or (
                len(self._cache) > 1 and sum(value[1].retained_bytes for value in self._cache.values()) > budget):
            self._cache.popitem(last=False)
        # A requested cache hit has priority over the completion of an older
        # in-flight query. Otherwise show useful progress with its true Z.
        current = self._cache.get(self._target_z_mm)
        self._display(*(current if current is not None else (result, preview)))

    def _display(self, result, preview):
        self.result, self.preview = result, preview
        self.state_set.single_display = (result, preview)
        self._display_comparison(preview)
        self._update_plane_positions()
        self._refresh_intensity()
        self.status.setStyleSheet("")
        self.status.setText(f"Completed exact plane Z {result.checkpoint.plane_z_mm:.9g} mm in captured optics. "
            + (f"Updating requested Z {self._target_z_mm:.9g} mm automatically."
               if result.checkpoint.plane_z_mm != self._target_z_mm else
               "Move Z for automatic readout; source or instrument changes require Calculate."))
        grids = "; ".join(f"{nx} × {ny}" for ny, nx in preview.grid_shapes)
        grid_text = (f"Displayed wave grid{'s' if len(preview.grid_shapes) > 1 else ''}: {grids} cells"
                     if grids else "Displayed wave grid: unavailable")
        self.readout.setText(f"{_coherent_backend_text(result.checkpoint)} | {grid_text} | Density / µm² per tip electron | {preview.mode_count} modes | displayed probability {preview.probability:.8g} per tip electron | "
            f"tip reference current {result.checkpoint.reference_current_a*1e9:.6g} nA | "
            f"{preview.density.shape[1]} × {preview.density.shape[0]} display bins; the executed wave grid is unchanged. Individual complex fields and phase references are preserved. Mixed modes add intensity, never a single aggregate phase.")

    def _refresh_intensity(self, *_args):
        if self.preview is None:
            return
        self.observation_changed.emit(self.result, self.preview,
                                      self.intensity_scale.currentIndex())

    @Slot(int, str)
    def _failed(self, generation, message):
        if self._closed or generation != self._generation:
            return
        # The cursor may already request a newer plane. Attribute this error
        # to the submitted worker, never to the latest requested position.
        failed_plane = (f" at Z {self._worker.request.observation_z_mm:.9g} mm"
                        if self._worker is not None else "")
        retained = (f" Previous complete plane Z {self.result.checkpoint.plane_z_mm:.9g} mm remains displayed."
                    if self.result is not None else "")
        self.status.setText(f"Coherent beam calculation failed{failed_plane}: {message}. "
                            f"No new image was produced.{retained}")

    @Slot(int, str)
    def _progress(self, generation, message):
        if not self._closed and generation == self._generation:
            shown = (f" | Displayed Z {self.result.checkpoint.plane_z_mm:.9g} mm"
                     if self.result is not None else "")
            self.status.setText(f"Requested Z {self._target_z_mm:.9g} mm{shown} | {message}")

    @Slot(int)
    def _finished(self, generation):
        if generation == self._generation:
            self._worker = None
            self._update_plane_positions()
            self._update_cancel_enabled()
            if self._pending_z and not self._closed and not self._stale:
                # The prior task is allowed to commit reusable upstream work.
                # No queue of obsolete planes is submitted to the coordinator.
                self.timer.start(0)

    @Slot()
    def cancel(self):
        if self._closed:
            return
        self.invalidate_pair()
        self.state_set.cancel()
        self._cancel_request()
        self.status.setText("Coherent beam request cancelled."
                            + (" Previous complete plane remains displayed." if self.result is not None else ""))

    def shutdown(self, msecs=3000):
        if not self._closed:
            self._cancel_request()
            self._closed = True
            self._observation_session = None
        states_finished = self.state_set.shutdown(msecs)
        observations_finished = self.observation.shutdown(msecs)
        return self.pool.waitForDone(msecs) and states_finished and observations_finished

    def closeEvent(self, event):
        if self.shutdown():
            event.accept()
        else:
            event.ignore()
