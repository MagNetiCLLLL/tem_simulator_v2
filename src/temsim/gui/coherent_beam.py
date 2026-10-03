"""Explicit tip-origin wave development, independent of particle results.

The page owns captured sessions and exact observation-plane results. Moving Z
never defines a source or interpolates a complex field. Mixed modes contribute
intensities; their individual complex fields remain in the executed checkpoint.
"""

from collections import OrderedDict
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
    QVBoxLayout, QWidget,
)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.immutable_json import thaw_json
from temsim.physics.coherent_inputs import (
    CoherentSourceSettings, default_gaussian_tip_settings, prepare_coherent_state, wave_input_summary,
)
from temsim.physics.tip_wave_pipeline import TipWaveObservationSession, TipWaveRequest, simulate_tip_wave

GIB = 1024**3


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


class _WaveWorker(QRunnable):
    def __init__(self, generation, session, state, request, event, retained=(), *, observer=None):
        super().__init__()
        self.generation = generation
        self.session = session
        self.state = state
        self.request = request
        self.event = event
        self.observer = observer
        # Inventory roots only: these executed displays are never supplied as
        # a replacement source to simulate_tip_wave.
        self.existing_result = retained+((observer,) if observer is not None else ())
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
            # CoordinatedPool holds the application's shared numerical lease
            # and its CPU budget. Do not create a nested independent pool.
            if self.observer is None:
                result = simulate_tip_wave(self.state, self.request, cancelled=self.event.is_set,
                    progress_callback=self._report_progress)
            else:
                result = self.observer.observe(self.request.observation_z_mm,
                    cancelled=self.event.is_set, progress_callback=self._report_progress)
            preview = _intensity_preview(result.checkpoint, cancelled=self.event.is_set)
            if not self.event.is_set():
                self.signals.solved.emit(self.generation, self.session, result, preview)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
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
        self.follow_ray.setToolTip("Shares the Coherent beam page setting. When unchecked, Ray Diagram Z does not change the coherent observation plane.")
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(toolbar)
        layout.addWidget(self.plane_positions)
        layout.addWidget(self.status)
        layout.addWidget(self.screen, 1)
        layout.addWidget(self.readout)
        owner.status.text_changed.connect(self.status.setText)
        owner.plane_positions.text_changed.connect(self.plane_positions.setText)
        owner.readout.text_changed.connect(self.readout.setText)
        owner.observation_changed.connect(self.screen.set_observation)
        self.screen.set_observation(owner.result, owner.preview,
                                    owner.intensity_scale.currentIndex())


class CoherentBeamPage(QWidget):
    """Explicit calculation starts a session; subsequent Z queries reuse it."""

    QUERY_INTERVAL_MS = 40
    CACHE_LIMIT = 64
    observation_changed = Signal(object, object, int)
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

        self.calculate_button = QPushButton("Calculate coherent beam")
        self.calculate_button.setObjectName("calculateCoherentBeam")
        self.calculate_button.setProperty("calculationAction", True)
        self.calculate_button.setStyleSheet(
            "QPushButton {background:#3ce878; color:#073519; border:1px solid #28b961; padding:5px 9px;}"
            "QPushButton:disabled {background:#29483a; color:#b5c5bc;}")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
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

        self.source_enabled = QCheckBox("Use a coherent boundary at the physical tip")
        self.source_enabled.setObjectName("coherentTipEnabled")
        self.source_enabled.setToolTip("Explicitly select a tip phase/mutual-intensity model for this captured wave session. Particle settings are unchanged.")
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
        self.surface_mean = _double(.3, 1e-9, 1e6, " eV", 6)
        self.surface_rms = _double(.1, 0., 1e6, " eV", 6)
        self.angle_caption = QLabel("Incoherent angular RMS")
        self.mean_caption = QLabel("Tip reservoir mean energy")
        self.rms_caption = QLabel("Tip reservoir energy RMS")
        tip_defaults = default_gaussian_tip_settings()
        self.tip_fwhm = _double(tip_defaults.tip_fwhm_nm, 1e-6, 1e6, " nm", 12)
        self.tip_fwhm.setToolTip("Intensity FWHM of the prescribed planar Gaussian emission boundary at the physical tip. This is an emission width, not the metal apex radius or a downstream virtual source size.")
        self.tip_energy = _double(tip_defaults.tip_mean_energy_ev, 1e-9, 1e6, " eV", 15)
        self.tip_minimum_energy = _double(tip_defaults.tip_minimum_energy_ev, 1e-9, 1e6, " eV", 15)
        self.tip_energy_spread = _double(tip_defaults.tip_energy_spread_fwhm_ev, 0., 1e6, " eV", 15)
        self.tip_offset_x = _double(tip_defaults.tip_offset_x_nm, -1e6, 1e6, " nm", 15)
        self.tip_offset_y = _double(tip_defaults.tip_offset_y_nm, -1e6, 1e6, " nm", 15)
        self.tip_curvature_x = _double(tip_defaults.tip_curvature_x_m1, -1e12, 1e12, " m⁻¹", 15)
        self.tip_curvature_xy = _double(tip_defaults.tip_curvature_xy_m1, -1e12, 1e12, " m⁻¹", 15)
        self.tip_curvature_y = _double(tip_defaults.tip_curvature_y_m1, -1e12, 1e12, " m⁻¹", 15)
        self.tip_tilt_x = _double(tip_defaults.tip_tilt_x_mrad, -1e4, 1e4, " mrad", 15)
        self.tip_tilt_y = _double(tip_defaults.tip_tilt_y_mrad, -1e4, 1e4, " mrad", 15)
        self.gaussian_inputs = QWidget()
        tip_form = QFormLayout(self.gaussian_inputs)
        tip_form.setContentsMargins(0, 0, 0, 0)
        for caption, control in (("Tip emission FWHM", self.tip_fwhm),
                ("Tip mean kinetic energy", self.tip_energy),
                ("Minimum kinetic energy", self.tip_minimum_energy),
                ("Energy spread FWHM", self.tip_energy_spread),
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

        self.advanced = QGroupBox("Advanced numerical budgets")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.advanced_body = QWidget()
        numerics = QFormLayout(self.advanced_body)
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
                ("Working limit per stage", self.working_gib), ("RAM cache limit", self.ram_cache_gib),
                ("Disk cache limit", self.disk_cache_gib)):
            numerics.addRow(caption, control)
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.addWidget(self.advanced_body)
        self.advanced_body.setVisible(False)
        self.advanced.toggled.connect(self.advanced_body.setVisible)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.source_enabled)
        controls_layout.addWidget(self.source_info)
        controls_layout.addWidget(self.source_assumption)
        controls_layout.addWidget(self.gaussian_inputs)
        controls_layout.addLayout(source_form)
        controls_layout.addWidget(self.preflight_message)
        controls_layout.addWidget(self.preflight_group)
        controls_layout.addWidget(self.advanced)
        controls_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(250)
        self.screen = CoherentIntensityView()
        toolbar.insertWidget(toolbar.count()-1, self.screen.fit_button)
        self.plot, self.image = self.screen.plot, self.screen.image
        self.observation_changed.connect(self.screen.set_observation)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.screen)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 900])
        self.scope_label = _label(self._SCOPE)
        self.scope_label.setStyleSheet("color:#d3ab61;")
        self.status = _label("Select a tip coherence model, then click Calculate coherent beam. No calculation starts when this tab opens.", observable=True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.plane_positions = _label(observable=True)
        self.plane_positions.setObjectName("coherentPlanePositions")
        self.plane_positions.setToolTip("Requested Z is the newest input. Calculating Z is the submitted worker target, including any wait for shared resources. Displayed Z belongs to the last completed image; moving the cursor does not relabel it.")
        self._update_plane_positions()
        self.readout = _label("Screen intensity density per tip electron; no detector counts or aggregate phase.", observable=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(toolbar)
        layout.addLayout(browse)
        layout.addWidget(self.scope_label)
        layout.addWidget(self.plane_positions)
        layout.addWidget(self.status)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.readout)

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
        self.source_enabled.toggled.connect(self.mark_inputs_stale)
        self.intensity_scale.currentIndexChanged.connect(self._refresh_intensity)
        for control in (self.angle_rms, self.surface_mean, self.surface_rms,
                        self.tip_fwhm, self.tip_energy, self.tip_minimum_energy, self.tip_energy_spread,
                        self.tip_offset_x, self.tip_offset_y,
                        self.tip_curvature_x, self.tip_curvature_xy, self.tip_curvature_y,
                        self.tip_tilt_x, self.tip_tilt_y,
                        self.grid_pixels, self.maximum_grid_pixels, self.energy_samples,
                        self.material_trajectories, self.column_step,
                        self.gun_step, self.gun_bore_step, self.gun_energy_step,
                        self.segment_steps, self.working_gib,
                        self.ram_cache_gib, self.disk_cache_gib):
            control.valueChanged.connect(self.mark_inputs_stale)
        self._show_source_controls()

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
        self.gaussian_inputs.setVisible(not surface)
        for control in (self.angle_caption, self.angle_rms):
            control.setVisible(not surface)
        for control in (self.mean_caption, self.surface_mean, self.rms_caption, self.surface_rms):
            control.setVisible(surface)

    def set_state(self, state):
        if self._closed:
            return
        self._state = state
        self.mark_inputs_stale()
        if state is None:
            self.source_info.setText("No instrument inputs available.")
            return
        emitter = getattr(getattr(state, "electron_gun", None), "emitter", None)
        if emitter is None:
            self.source_info.setText("No physical tip input fields are available for this source. Coherent calculation requires a supported physical tip.")
            return
        surface = emitter.surface_model
        self._source_model = "gaussian" if surface is None else "surface"
        self._show_source_controls()
        if surface is None:
            coherence = emitter.coherence
            use_defaults = coherence is None and float(getattr(emitter, "curvature_nm_inv", 0.)) == 0.
            defaults = default_gaussian_tip_settings()
            for control, setting, value in (
                    (self.tip_fwhm, "tip_fwhm_nm", emitter.virtual_source_fwhm_nm),
                    (self.tip_energy, "tip_mean_energy_ev", emitter.emission_energy_ev),
                    (self.tip_minimum_energy, "tip_minimum_energy_ev", emitter.minimum_kinetic_energy_ev),
                    (self.tip_energy_spread, "tip_energy_spread_fwhm_ev", emitter.energy_spread_fwhm_ev),
                    (self.tip_offset_x, "tip_offset_x_nm", 0. if coherence is None else coherence.offset_x_nm),
                    (self.tip_offset_y, "tip_offset_y_nm", 0. if coherence is None else coherence.offset_y_nm),
                    (self.tip_curvature_x, "tip_curvature_x_m1", 0. if coherence is None else coherence.curvature_x_m1),
                    (self.tip_curvature_xy, "tip_curvature_xy_m1", 0. if coherence is None else coherence.curvature_xy_m1),
                    (self.tip_curvature_y, "tip_curvature_y_m1", 0. if coherence is None else coherence.curvature_y_m1),
                    (self.tip_tilt_x, "tip_tilt_x_mrad", 0. if coherence is None else coherence.tilt_x_mrad),
                    (self.tip_tilt_y, "tip_tilt_y_mrad", 0. if coherence is None else coherence.tilt_y_mrad)):
                control.blockSignals(True)
                control.setValue(getattr(defaults, setting) if use_defaults else value)
                control.blockSignals(False)
            self.angle_rms.blockSignals(True)
            self.angle_rms.setValue(0. if coherence is None else coherence.incoherent_angle_rms_mrad)
            self.angle_rms.blockSignals(False)
            if use_defaults:
                self.source_info.setText(
                    f"Idealised diffraction example defaults | FWHM {defaults.tip_fwhm_nm:g} nm | "
                    f"emission energy {defaults.tip_mean_energy_ev:g} eV | energy spread 0 eV | "
                    f"current {emitter.emission_current_na:g} nA from the instrument.")
            else:
                self.source_info.setText(
                    f"Captured tip boundary | FWHM {emitter.virtual_source_fwhm_nm:g} nm | "
                    f"emission energy {emitter.emission_energy_ev:g} eV | energy spread "
                    f"{emitter.energy_spread_fwhm_ev:g} eV | current {emitter.emission_current_na:g} nA.")
            self.source_assumption.setText(
                ("Default inputs were designed for a centred 1.5 nm intensity FWHM at the specimen with the example optics. They are idealised planar-cathode inputs, not measured metal-tip properties. " if use_defaults else "") +
                "Edits below prescribe a planar Gaussian emission boundary at the physical tip of this captured wave session. Emission FWHM and wavefront curvature are not the metal tip radius or its mechanical curvature. Particle inputs and installed fields are unchanged. Source-domain checks still apply.")
            self._refresh_energy_sample_default()
        else:
            coherence = surface.coherence
            if coherence is not None:
                for control, value in ((self.surface_mean, coherence.mean_energy_ev),
                                       (self.surface_rms, coherence.energy_rms_ev)):
                    control.blockSignals(True)
                    control.setValue(value)
                    control.blockSignals(False)
            self.source_info.setText(f"Curved physical tip | apex radius {surface.geometry.apex_radius_nm:g} nm | current {surface.current_na:g} nA. Geometry and extraction settings come from the instrument.")
            self.source_assumption.setText(
                "Explicit quantum assumption: the mean energy and energy RMS below define a coherent cap reservoir, one axisymmetric spatial mode per energy. "
                "They are not inferred from classical emission probabilities. Enable the tip boundary and click Calculate to select this model.")
            self._set_default_energy_samples(9)
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
        self.cancel_button.setEnabled(False)
        self._update_plane_positions()

    @Slot()
    def mark_inputs_stale(self, *_args):
        if self._closed:
            return
        self._cancel_request()
        self._stale = True
        self._session_ready = False
        self._cache.clear()
        self._observation_session = None
        self.status.setStyleSheet("")
        if self.preflight_details.toPlainText():
            self.preflight_message.setText("Previous source check; inputs changed. Click Calculate coherent beam to check the current settings. The retained report describes the previous attempt.")
            self.preflight_message.setStyleSheet("color:#d3ab61;")
        self.status.setText("Inputs changed. Click Calculate coherent beam to capture a new session."
                            + (" Previous optics remain displayed." if self.result is not None else ""))

    def _show_source_check(self, summary, settings, error=None):
        """Show the executed preflight without changing inputs or its admission."""
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
            self.preflight_group.setChecked(True)
            return
        lines = ["Tip source check rejected; propagation did not start." if data is not None and not ready
                 else "Calculation setup rejected; propagation did not start."]
        if data is not None:
            inputs = data.get("inputs", {})
            for key, label, unit in (("virtual_source_fwhm_nm", "Tip emission FWHM", "nm"),
                    ("emission_energy_ev", "Tip mean kinetic energy", "eV"),
                    ("energy_spread_fwhm_ev", "Energy spread FWHM", "eV")):
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
            lines.append("Review the explicit tip width, energy law and coherence, or load the documented Si input example. Inputs are not adjusted automatically; source admission alone does not guarantee propagation.")
        else:
            lines.append(str(error))
        self.preflight_message.setText("\n".join(lines))
        self.preflight_message.setStyleSheet("color:#ffb86b;")
        self.preflight_group.setChecked(True)

    def _settings(self):
        if self._source_model == "surface":
            return CoherentSourceSettings(enabled=self.source_enabled.isChecked(),
                surface_mean_energy_ev=self.surface_mean.value(),
                surface_energy_rms_ev=self.surface_rms.value())
        return CoherentSourceSettings(enabled=self.source_enabled.isChecked(),
            incoherent_angle_rms_mrad=self.angle_rms.value(),
            tip_fwhm_nm=self.tip_fwhm.value(), tip_mean_energy_ev=self.tip_energy.value(),
            tip_minimum_energy_ev=self.tip_minimum_energy.value(),
            tip_energy_spread_fwhm_ev=self.tip_energy_spread.value(),
            tip_offset_x_nm=self.tip_offset_x.value(), tip_offset_y_nm=self.tip_offset_y.value(),
            tip_curvature_x_m1=self.tip_curvature_x.value(), tip_curvature_xy_m1=self.tip_curvature_xy.value(),
            tip_curvature_y_m1=self.tip_curvature_y.value(),
            tip_tilt_x_mrad=self.tip_tilt_x.value(), tip_tilt_y_mrad=self.tip_tilt_y.value())

    def _make_request(self):
        default = TipWaveRequest()
        working = round(self.working_gib.value()*GIB)
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
                              maximum_working_bytes=working),
            inelastic=replace(default.inelastic, trajectories_per_mode=self.material_trajectories.value()),
            maximum_readout_bytes=working,
            execution=replace(default.execution, segment_steps=self.segment_steps.value(),
                maximum_ram_cache_bytes=round(self.ram_cache_gib.value()*GIB),
                maximum_disk_cache_bytes=round(self.disk_cache_gib.value()*GIB))).validate()

    @Slot()
    def calculate(self):
        if self._closed:
            return
        self._cancel_request()
        self._session += 1
        self._session_ready = False
        self._cache.clear()
        self._observation_session = None
        summary = None
        settings = None
        try:
            settings = self._settings()
            state = self.state_provider() if self.state_provider is not None else self._state
            if state is None:
                raise ValueError("No current instrument inputs are available")
            captured = prepare_coherent_state(state, settings)
            request = self._make_request()
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
            return
        self._show_source_check(summary, settings)
        self.status.setStyleSheet("")
        self._captured = captured
        self._request = request
        self._observation_session = TipWaveObservationSession(captured, request)
        self._stale = False
        self.status.setText(f"Source ready | {summary['model']} | {summary['mode_count']} modes. Queued exact-Z calculation; field support is checked during execution.")
        self._start_query()

    def _retained_wave_roots(self):
        """Keep idle wave products visible to all shared calculation owners."""
        return self.result, self.preview, self._cache, self._captured, self._observation_session

    def _update_plane_positions(self):
        calculating = (f"{self._worker.request.observation_z_mm:.9g} mm"
                       if self._worker is not None else "—")
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
        if self._stale or (not self._session_ready and self._worker is None):
            self.status.setText("Click Calculate coherent beam to start a session at this Z."
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
                             observer=self._observation_session)
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
        self.readout.setText(f"{grid_text} | Density / µm² per tip electron | {preview.mode_count} modes | displayed probability {preview.probability:.8g} per tip electron | "
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
            self.cancel_button.setEnabled(False)
            if self._pending_z and not self._closed and not self._stale:
                # The prior task is allowed to commit reusable upstream work.
                # No queue of obsolete planes is submitted to the coordinator.
                self.timer.start(0)

    @Slot()
    def cancel(self):
        if self._closed:
            return
        self._cancel_request()
        self.status.setText("Coherent beam request cancelled."
                            + (" Previous complete plane remains displayed." if self.result is not None else ""))

    def shutdown(self, msecs=3000):
        if not self._closed:
            self._cancel_request()
            self._closed = True
            self._observation_session = None
        return self.pool.waitForDone(msecs)

    def closeEvent(self, event):
        if self.shutdown():
            event.accept()
        else:
            event.ignore()
