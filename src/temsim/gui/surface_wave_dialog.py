"""Non-blocking viewer for an explicit coherent-tip draft, separate from images."""
from dataclasses import replace

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog)

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave


class _SurfaceWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, snapshot, numerics, parent):
        super().__init__(parent)
        self.snapshot, self.numerics = snapshot, numerics

    def run(self):
        try:
            result = simulate_tip_wave(self.snapshot.restore(),
                TipWaveRequest(stop="tip_near_field", surface=self.numerics),
                cancelled=self.isInterruptionRequested,
                progress_callback=lambda n, total, label: self.progress.emit(f"{n}/{total} | {label}"))
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class SurfaceWaveDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        self.snapshot = capture_instrument_snapshot(state)
        self.worker = None
        self.calculation = None
        self._closing = False
        self.setWindowTitle("Coherent tip near field — development")
        self.resize(1000, 700)
        layout = QVBoxLayout(self)
        note = QLabel("Draft calculation only. Does not replace Ray Diagram, TEM or STEM results.")
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        row = QHBoxLayout()
        self.radial, self.axial, self.energies = QSpinBox(), QSpinBox(), QSpinBox()
        for widget, lo, hi, value, label in ((self.radial, 3, 4097, 97, "R nodes"),
                (self.axial, 3, 4097, 193, "Z nodes"), (self.energies, 1, 64, 3, "Energy samples")):
            widget.setRange(lo, hi); widget.setValue(value)
            row.addWidget(QLabel(label)); row.addWidget(widget)
        form.addRow("Wave mesh", row)
        row = QHBoxLayout()
        self.height, self.extent = QDoubleSpinBox(), QDoubleSpinBox()
        self.height.setDecimals(3); self.height.setRange(.001, 1000.); self.height.setValue(2.)
        self.extent.setDecimals(3); self.extent.setRange(1.001, 10.); self.extent.setValue(1.5)
        row.addWidget(QLabel("Top height (nm)")); row.addWidget(self.height)
        row.addWidget(QLabel("Radial extent / patch radius")); row.addWidget(self.extent)
        form.addRow("Numerical domain", row)
        layout.addLayout(form)
        self.controls = (self.radial, self.axial, self.energies, self.height, self.extent)
        self.status = QLabel("Local Robin boundaries are approximate. Refine mesh, field grid and domain before interpreting results.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.calculate = QPushButton("Calculate near field")
        self.cancel = QPushButton("Cancel calculation"); self.cancel.setEnabled(False)
        self.export = QPushButton("Export complex fields..."); self.export.setEnabled(False)
        self.mode = QComboBox()
        for widget in (self.calculate, self.cancel, self.export, self.mode):
            row.addWidget(widget)
        layout.addLayout(row)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.calculate.clicked.connect(self._start)
        self.cancel.clicked.connect(self._cancel)
        self.export.clicked.connect(self._export)
        self.mode.currentIndexChanged.connect(self._draw)
        for widget in self.controls:
            widget.valueChanged.connect(self._draft_changed)

    def _draft_changed(self):
        if self.calculation is not None:
            self.status.setText("Numerical inputs changed. Previous computed field retained; calculate to update.")

    def _start(self):
        if self.worker is not None:
            return
        numerics = replace(SurfaceWaveNumerics(), radial_nodes=self.radial.value(),
            axial_nodes=self.axial.value(), energy_samples=self.energies.value(),
            exit_height_nm=self.height.value(), outer_radius_factor=self.extent.value())
        self.calculate.setEnabled(False); self.cancel.setEnabled(True)
        self.export.setEnabled(False)
        for control in self.controls:
            control.setEnabled(False)
        self.worker = _SurfaceWorker(self.snapshot, numerics, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.result_ready.connect(self._result)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.status.setText("Calculating coherent tip near field...")
        self.worker.start()

    def _cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText("Cancelling after the current sparse solve. Previous result is retained.")

    def _failed(self, message):
        self.status.setText(message + (" | Previous field retained." if self.calculation is not None else ""))

    def _result(self, calculation):
        self.calculation = calculation
        result = calculation.checkpoint
        flux = {key: sum(m.weight*m.flux[key] for m in result.modes) for key in ("reflected", "top", "side")}
        self.status.setText(f"{'Cached' if calculation.propagation_cache_hit else 'Computed'} near field | "
            f"Reflected {flux['reflected']:.3%} | Top {flux['top']:.3%} | Side {flux['side']:.3%}\n"
            "Finite-domain development result. Not a complete gun or image calculation.")
        self.mode.blockSignals(True); self.mode.clear()
        for index, mode in enumerate(result.modes):
            self.mode.addItem(f"Phase: {mode.energy_ev:.5g} eV", index)
        self.mode.blockSignals(False)
        self._draw()

    def _finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.calculate.setEnabled(True); self.cancel.setEnabled(False)
        self.export.setEnabled(self.calculation is not None)
        for control in self.controls:
            control.setEnabled(True)
        if self._closing:
            super().reject()

    def _draw(self):
        if self.calculation is not None and self.mode.currentIndex() >= 0:
            from temsim.physics.surface_wave_export import draw_surface_wave
            draw_surface_wave(self.figure, self.calculation.checkpoint, self.mode.currentIndex())
            self.canvas.draw_idle()

    def _export(self):
        from temsim.physics.surface_wave_export import export_surface_wave
        path, _ = QFileDialog.getSaveFileName(self, "Export computed near field to a new file", "tip_near_field.npz", "NumPy archive (*.npz)")
        if not path:
            return
        try:
            export_surface_wave(self.calculation, path)
            self.status.setText(f"Saved executed near field: {path}")
        except (OSError, ValueError) as error:
            self.status.setText(str(error))

    def reject(self):
        if self.worker is not None:
            self._closing = True
            self._cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)
