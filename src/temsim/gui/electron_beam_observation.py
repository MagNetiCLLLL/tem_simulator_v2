"""Different observations of one accepted beam; never a propagation entry point."""

from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QRectF, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.physics.electron_detection import MAX_EMITTED_ELECTRONS, sample_electron_detections


class _DetectionSignals(QObject):
    solved = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _DetectionWorker(QRunnable):
    def __init__(self, generation, preview, count, seed):
        super().__init__()
        self.generation, self.preview = generation, preview
        self.count, self.seed = count, seed
        self.event = Event()
        self.signals = _DetectionSignals()
        self.resource_claim = ResourceClaim(96*1024**2)
        self.job_input_identity = f"screen-events:{id(preview)}:{count}:{seed}"
        self.existing_result = (preview,)

    def run(self):
        try:
            if not self.event.is_set():
                result = sample_electron_detections(self.preview.density, self.preview.bounds_um,
                    emitted_electrons=self.count, seed=self.seed)
                if not self.event.is_set():
                    self.signals.solved.emit(self.generation, result)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
        finally:
            self.signals.finished.emit(self.generation)


class ElectronBeamObservation(QWidget):
    """Read-only intensity, virtual arrivals and per-state complex diagnostics.

    Count and seed describe a virtual exposure of the already calculated
    probability image. They never alter the Tip, optical settings or ray budget.
    Independent state overlays retain their members for phase/current readouts.
    """

    MODES = (("Intensity", "intensity"), ("Electron arrivals", "arrivals"),
             ("Beam current", "wave_current"), ("Phase", "wave_phase"),
             ("Probability flow", "wave_flow"), ("Angular distribution", "wave_angles"),
             ("Interactions", "wave_interactions"))

    def __init__(self, intensity_view, parent=None):
        super().__init__(parent)
        self.screen = intensity_view
        self.result = self.preview = self.detections = None
        self._diagnostic_identity = None
        self._bounds = None
        self._worker = None
        self._generation = 0
        self._closed = False
        self._detection_identity = None
        self._projection_angle_deg = 0.
        self.diagnostics = None
        self.pool = CoordinatedPool(self)
        self.pool.coordinator.register_retained(self, "retained_roots")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_detections)

        self.mode = QComboBox()
        self.mode.setObjectName("electronBeamObservation")
        for label, key in self.MODES:
            self.mode.addItem(label, key)
        self.count = QSpinBox()
        self.count.setObjectName("observationEmissionCount")
        self.count.setRange(0, MAX_EMITTED_ELECTRONS)
        self.count.setValue(3000)
        self.count.setKeyboardTracking(False)
        self.count.setToolTip("Virtual exposure: electrons emitted by the Tip. This changes sampled arrivals only, not the particle tracing budget. Lost electrons remain absent.")
        self.seed = QSpinBox()
        self.seed.setObjectName("observationRandomSeed")
        self.seed.setRange(0, 2**31-1)
        self.seed.setKeyboardTracking(False)
        self.count_label, self.seed_label = QLabel("Emitted electrons"), QLabel("Seed")
        self.member = QComboBox()
        self.member.setObjectName("observationTipState")
        self.member.setToolTip("Choose one executed Tip state for complex-field diagnostics. The intensity and arrivals views use the weighted overlay.")
        self.member_label = QLabel("Inspect state")
        self.fit = QPushButton("Fit view")
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Observe"))
        toolbar.addWidget(self.mode)
        for control in (self.count_label, self.count, self.seed_label, self.seed,
                        self.member_label, self.member):
            toolbar.addWidget(control)
        toolbar.addStretch(1)
        toolbar.addWidget(self.fit)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.screen)
        self.arrival_plot = pg.PlotWidget(background="#080e19")
        self.arrival_plot.setLabel("bottom", "X", units="µm")
        self.arrival_plot.setLabel("left", "Y", units="µm")
        self.arrival_plot.getAxis("left").setWidth(76)
        self.arrival_plot.getAxis("bottom").setHeight(46)
        self.arrival_plot.setAspectLocked(True)
        self.arrival_plot.disableAutoRange()
        self.arrival_plot.showGrid(x=True, y=True, alpha=.15)
        self.points = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush("#f5e76d"), size=3)
        self.arrival_image = pg.ImageItem(axisOrder="col-major")
        self.arrival_image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.arrival_plot.addItem(self.arrival_image)
        self.arrival_plot.addItem(self.points)
        self.stack.addWidget(self.arrival_plot)
        self.note = QLabel("Choose an observation of the calculated beam. No propagation is started by changing this view.")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.note)
        self.mode.currentIndexChanged.connect(self._view_changed)
        self.member.currentIndexChanged.connect(self._view_changed)
        self.count.valueChanged.connect(self._schedule_detections)
        self.seed.valueChanged.connect(self._schedule_detections)
        self.fit.clicked.connect(self.fit_view)
        self._view_changed()

    def retained_roots(self):
        return self.result, self.preview, self.detections

    def set_observation(self, result, preview, scale):
        if self._closed or preview is None:
            return
        changed = result is not self.result or preview is not self.preview
        self.result, self.preview = result, preview
        self.screen.set_observation(result, preview, scale)
        if changed:
            self._cancel_detections()
            self.detections = None
            self._detection_identity = None
            self.member.blockSignals(True)
            previous = self.member.currentIndex()
            self.member.clear()
            members = getattr(result, "members", (result,))
            for index, _ in enumerate(members):
                weight = getattr(result, "relative_weights", (1.,))[index]
                self.member.addItem(f"State {index+1} | weight {weight:.4g}", index)
            self.member.setCurrentIndex(min(max(previous, 0), len(members)-1))
            self.member.blockSignals(False)
        self._view_changed()

    def _view_changed(self, *_args):
        if self._closed:
            return
        mode = self.mode.currentData()
        arrivals = mode == "arrivals"
        diagnostic = str(mode).startswith("wave_")
        for widget in (self.count_label, self.count, self.seed_label, self.seed):
            widget.setVisible(arrivals)
        for widget in (self.member_label, self.member):
            widget.setVisible(diagnostic and self.member.count() > 1)
        if not arrivals:
            self._cancel_detections()
        if diagnostic:
            self._show_diagnostic(mode)
        elif arrivals:
            self.stack.setCurrentWidget(self.arrival_plot)
            if self.preview is None:
                self.note.setText("Calculate beam to obtain a screen probability distribution first.")
            elif (self.detections is not None and self._detection_identity ==
                  (id(self.preview), self.count.value(), self.seed.value())):
                self._show_detections()
            else:
                self._schedule_detections()
        else:
            self.stack.setCurrentWidget(self.screen)
            self.note.setText("Expected screen intensity from the retained complex modes. Purple is low intensity on this colour scale; it is not a material or vacuum background.")

    def _cancel_detections(self):
        self.timer.stop()
        self._generation += 1
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()
        self._worker = None

    def _schedule_detections(self, *_args):
        if self._closed or self.mode.currentData() != "arrivals" or self.preview is None:
            return
        self._cancel_detections()
        self.note.setText("Sampling virtual arrivals from the displayed plane; the propagated beam is unchanged.")
        self.timer.start(50)

    def _start_detections(self):
        if self._closed or self.preview is None or self.mode.currentData() != "arrivals":
            return
        key = (id(self.preview), self.count.value(), self.seed.value())
        if key == self._detection_identity and self.detections is not None:
            self._show_detections()
            return
        worker = _DetectionWorker(self._generation, self.preview, self.count.value(), self.seed.value())
        worker.signals.solved.connect(self._detections_ready)
        worker.signals.failed.connect(self._detections_failed)
        worker.signals.finished.connect(self._detections_finished)
        self._worker = worker
        self.pool.start(worker)

    def _detections_ready(self, generation, detections):
        if self._closed or generation != self._generation or self._worker is None:
            return
        self.detections = detections
        self._detection_identity = (id(self._worker.preview), self._worker.count, self._worker.seed)
        self._show_detections()

    def _show_detections(self):
        data = self.detections
        bounds = self.preview.bounds_um
        # Drawing a million markers freezes the GUI. Dense exposures show ALL
        # arrivals binned as counts, never a subsample passed off as the full dose.
        dense = data.detected_count > 50_000
        self.points.setVisible(not dense)
        self.arrival_image.setVisible(dense)
        if dense:
            self.arrival_image.setImage(data.counts, autoLevels=False,
                                       levels=(0, max(1, int(data.counts.max()))))
            self.arrival_image.setRect(QRectF(bounds[0, 0], bounds[1, 0], *np.diff(bounds, axis=1)[:, 0]))
        else:
            self.points.setData(data.points_um[:, 0], data.points_um[:, 1])
        first = self._bounds is None
        self._bounds = bounds
        if first:
            self.fit_view()
        self.arrival_plot.setTitle(f"Exact Z {self.result.checkpoint.plane_z_mm:.9g} mm<br>Simulated electron arrivals")
        self.note.setText(f"Emitted {data.emitted_count:,} | Arrived {data.detected_count:,} | Not reaching this screen {data.lost_count:,} | "
            f"Expected arrival fraction {data.detection_probability:.7g}. "
            + ("All arrivals shown as counts per bin. " if dense else "Dots are independent virtual screen events, not trajectories. ")
            + "Positions are sampled at the intensity display-bin resolution. No detector response or electron-electron interactions are added.")

    def _detections_failed(self, generation, error):
        if not self._closed and generation == self._generation:
            self.note.setText(f"Electron arrivals unavailable: {error}. The calculated beam is retained.")

    def _detections_finished(self, generation):
        if generation == self._generation:
            self._worker = None

    def _show_diagnostic(self, mode):
        if self.diagnostics is None:
            from temsim.gui.diagnostic_tabs import TransverseBeamView
            self.diagnostics = TransverseBeamView()
            self.diagnostics.plot_layout.use_available_space()
            self.diagnostics.initial_beam_panel.hide()
            self.diagnostics.fit_beam.hide()  # The shared Fit view action handles this view.
            hardware = self.diagnostics.hardware
            for widget in (hardware.toggle, hardware.stops_toggle, hardware.visibility_button,
                           hardware.reset_button, hardware.fit_button, hardware.status):
                widget.hide()
            self.stack.addWidget(self.diagnostics)
        self.stack.setCurrentWidget(self.diagnostics)
        if self.result is None:
            self.note.setText("Calculate beam before inspecting its complex-field observables.")
            return
        members = getattr(self.result, "members", (self.result,))
        result = members[max(0, self.member.currentIndex())]
        if not hasattr(result.checkpoint, "beam"):
            self.diagnostics.plot.clear()
            self.note.setText("This result has no retained complex modes for diagnostic observation.")
            return
        gauge = getattr(self.preview, "magnetic_gauge", None)
        axial_bz = getattr(self.preview, "axial_bz_t", None)
        identity = (id(result.checkpoint), None if gauge is None else gauge.fingerprint, axial_bz)
        if identity != self._diagnostic_identity:
            # All listed states share installed optics and exact Z. Their
            # fields remain individual; an overlay is never given one phase.
            self.diagnostics.display_wave_checkpoint(result.checkpoint,
                axial_bz_t=axial_bz, magnetic_gauge=gauge)
            self._diagnostic_identity = identity
            self.diagnostics.set_projection_angle(self._projection_angle_deg)
        combo = self.diagnostics.analysis.mode_combo
        combo.setCurrentIndex(combo.findData(mode))
        combo.hide()
        self.diagnostics.analysis.view_label.hide()
        self.note.setText("Readout of one executed Tip state; independent modes sum currents/intensities. Phase selects a single mode. Probability-flow arrows are local current directions, not measured electron trajectories.")

    def set_projection_angle(self, angle):
        self._projection_angle_deg = float(angle)
        if self.diagnostics is not None:
            self.diagnostics.set_projection_angle(angle)

    def fit_view(self):
        mode = self.mode.currentData()
        if mode == "intensity":
            self.screen.fit_full()
        elif mode == "arrivals" and self._bounds is not None:
            self.arrival_plot.setRange(xRange=self._bounds[0], yRange=self._bounds[1], padding=.02)
        elif self.diagnostics is not None:
            self.diagnostics.fit_beam.click()

    def shutdown(self, msecs=3000):
        self._closed = True
        self._cancel_detections()
        done = self.pool.waitForDone(msecs)
        if self.diagnostics is not None and self.diagnostics.analysis.wave is not None:
            wave = self.diagnostics.analysis.wave
            wave.deactivate()
            done = wave.pool.waitForDone(msecs) and done
        return done
