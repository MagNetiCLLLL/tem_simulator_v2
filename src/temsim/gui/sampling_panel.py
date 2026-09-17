"""Detached two-run sampling checks; never installs a result into the column."""
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QPushButton, QPlainTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView)
from temsim.gui.input_policy import (WheelSafeComboBox as QComboBox,
    WheelSafeSpinBox as QSpinBox, WheelSafeDoubleSpinBox as QDoubleSpinBox)

from temsim.immutable_json import thaw_json
from temsim.sampling_convergence import (AXES, UNAVAILABLE_AXES, ConvergenceRequest,
    SamplingCancelled, run_convergence, write_evidence)


class _Signals(QObject):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _Worker(QRunnable):
    def __init__(self, request, cancellation):
        super().__init__()
        self.request, self.cancellation = request, cancellation
        from temsim.immutable_json import json_digest
        self.job_input_identity = json_digest({**vars(request), "checkpoint": request.checkpoint.digest})
        self.signals = _Signals()

    def run(self):
        try:
            report = run_convergence(self.request, cancelled=self.cancellation.is_set,
                                     progress=self.signals.progress.emit)
            self.signals.result.emit(report)
        except SamplingCancelled:
            pass
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class SamplingPanel(QWidget):
    evidence_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        from temsim.gui.job_coordinator import CoordinatedPool
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._cancel = Event()
        self._busy = False
        self._checkpoint = None
        self.report = None
        self._history = {}
        layout = QVBoxLayout(self)
        caption = QLabel("Sampling & Convergence | independent captured inputs | two runs through specimen entrance")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        self.summary = QLabel("Select a working point. N_eff measures weight concentration, not convergence.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.moments_toggle = QCheckBox("Show weighted phase-space moments")
        layout.addWidget(self.moments_toggle)
        self.moments = QTableWidget(5, 4)
        self.moments.setObjectName("weightedPhaseSpaceMoments")
        self.moments.setHorizontalHeaderLabels(("x (nm)", "theta X (mrad)", "y (nm)", "theta Y (mrad)"))
        self.moments.setVerticalHeaderLabels(("Mean", "Cov: x", "Cov: theta X", "Cov: y", "Cov: theta Y"))
        self.moments.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.moments.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.moments.setMaximumHeight(190)
        self.moments.setToolTip("Current-weighted population moments. Covariance entries have row-unit times column-unit. These are particle statistics, not a wave phase or convergence certificate.")
        self.moments.setVisible(False)
        self.moments_toggle.toggled.connect(self.moments.setVisible)
        layout.addWidget(self.moments)
        self.emittance = QLabel("RMS emittance: unavailable")
        self.emittance.setWordWrap(True)
        self.emittance.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.emittance)
        controls = QHBoxLayout()
        self.axis = QComboBox()
        for key, label in AXES.items():
            self.axis.addItem(label, key)
        self.axis.setToolTip("Only numerical settings change. Physical source support, current and installed optics stay fixed.")
        controls.addWidget(self.axis)
        controls.addWidget(QLabel("Maximum rays per run"))
        self.budget = QSpinBox()
        self.budget.setRange(9, 65536)
        self.budget.setValue(4096)
        controls.addWidget(self.budget)
        self.run_button = QPushButton("Run comparison")
        self.run_button.clicked.connect(self.start)
        self.run_button.setEnabled(False)
        controls.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export evidence...")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)
        factors = QHBoxLayout()
        self.factors = {}
        for key, label in (("spatial", "Tip positions"), ("directions", "Directions per position"), ("energies", "Energy samples per direction")):
            factors.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setRange(3, 65536)
            spin.setValue(9)
            self.factors[key] = spin
            factors.addWidget(spin)
        layout.addLayout(factors)
        factor_note = QLabel("Independent sampling axes use these explicit baseline factors; total rays are their product. Surface energy remains conditional on local direction. Existing joint sampling is retained for the total-count axis.")
        factor_note.setWordWrap(True)
        layout.addWidget(factor_note)
        topology = QHBoxLayout()
        self.topology = QCheckBox("Check crossover topology (extra tip-origin executions)")
        topology.addWidget(self.topology)
        topology.addWidget(QLabel("Bracket spacing (mm)"))
        self.bracket_spacing = QDoubleSpinBox()
        self.bracket_spacing.setDecimals(3)
        self.bracket_spacing.setRange(.01, 100.)
        self.bracket_spacing.setValue(2.)
        topology.addWidget(self.bracket_spacing)
        topology.addWidget(QLabel("Checkpoint budget (MiB)"))
        self.checkpoint_budget = QSpinBox()
        self.checkpoint_budget.setRange(1, 32768)
        self.checkpoint_budget.setValue(512)
        topology.addWidget(self.checkpoint_budget)
        layout.addLayout(topology)
        plans = QHBoxLayout()
        self.plan_button = QPushButton("Multi-level check...")
        self.resume_button = QPushButton("Resume check...")
        self.plan_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.plan_button.clicked.connect(self._new_plan)
        self.resume_button.clicked.connect(self._resume_plan)
        plans.addWidget(self.plan_button)
        plans.addWidget(self.resume_button)
        plans.addStretch()
        layout.addLayout(plans)
        self.reference = QLabel("Topology reference: no selection")
        self.reference.setWordWrap(True)
        layout.addWidget(self.reference)
        self.status = QLabel("No comparison run. Coherent wave development is paused.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText("Pending capabilities:\n" + "\n".join(f"{k}: {v}" for k, v in UNAVAILABLE_AXES.items()))
        layout.addWidget(self.details)

    def set_checkpoint(self, checkpoint, summary=None):
        self._checkpoint = checkpoint
        self.moments.clearContents()
        self.emittance.setText("RMS emittance: unavailable")
        self.summary.setToolTip("")
        self.run_button.setEnabled(checkpoint is not None and not self._busy)
        self.plan_button.setEnabled(checkpoint is not None and not self._busy)
        self.resume_button.setEnabled(checkpoint is not None and not self._busy)
        if checkpoint is None:
            self.summary.setText("No working point selected")
            self.reference.setText("Topology reference: no selection")
            return
        from temsim.topology_evidence import topology_reference
        try:
            reference = topology_reference(checkpoint.snapshot)
            expected = reference.get("reference", {})
            self.reference.setText(f"Topology reference: {reference['status']} | "
                f"mode {reference['mode']} | intermediate count {expected.get('count', 'unavailable')} | "
                "target only; not a current validation")
            import json
            self.reference.setToolTip(json.dumps(thaw_json(reference), indent=2))
        except (ValueError, OSError, KeyError) as exc:
            self.reference.setText(f"Topology reference unavailable: {exc}")
        from temsim.sampling_diagnostics import checkpoint_sampling_summary
        summary = summary or checkpoint_sampling_summary(checkpoint)
        def shown(key, factor=1.0):
            value = summary[key]
            return "Unavailable" if value is None else f"{value * factor:.6g}"
        self.summary.setText(f"Checkpoint {checkpoint.digest[:12]} | Z {checkpoint.plane_z_mm:g} mm\n"
            f"Emitted {summary['emitted_samples']} | transmitted {summary['transmitted_samples']} | "
            f"N_eff {shown('effective_samples')} | transmission {shown('transmission', 100)}%\n"
            f"Source current {shown('source_current_a', 1e12)} pA | plane current {shown('plane_current_a', 1e12)} pA | "
            f"{summary['status']}: {summary['reason']}\n"
            "Comparison recalculates from the tip to specimen entrance. Its plane can differ from this checkpoint.")
        import json
        self.summary.setToolTip(json.dumps(thaw_json(summary.get("phase_space", {})), indent=2))
        self._show_moments(summary.get("phase_space"))

    def _show_moments(self, stats):
        if stats is None:
            return
        # SI values remain authoritative in the evidence; scaling is display-only.
        import numpy as np
        scale = np.array([1e9, 1e3, 1e9, 1e3])
        rows = np.vstack((np.asarray(stats["mean"])*scale,
                          np.asarray(stats["covariance"])*scale[:, None]*scale[None, :]))
        for i, values in enumerate(rows):
            for j, value in enumerate(values):
                self.moments.setItem(i, j, QTableWidgetItem(f"{value:.6g}"))
        x, y = stats["rms_emittance_m_rad"]
        self.emittance.setText(f"Projected RMS emittance: X {x*1e12:.6g} | Y {y*1e12:.6g} nm mrad | geometric, not normalised")

    def start(self):
        if self._busy or self._checkpoint is None:
            return
        request = ConvergenceRequest(self._checkpoint, self.axis.currentData(), self.budget.value(),
            self.factors["spatial"].value(), self.factors["directions"].value(), self.factors["energies"].value(),
            self.topology.isChecked(), self.bracket_spacing.value(), self.checkpoint_budget.value()*1024**2,
            tuple(self._history.get(self._checkpoint.digest, {}).values()))
        self._cancel = Event()
        self._busy = True
        self.plan_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(f"Checking saved inputs for {self._checkpoint.digest[:12]}...")
        worker = _Worker(request, self._cancel)
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._result)
        worker.signals.error.connect(self._error)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    def _new_plan(self):
        if self._busy or self._checkpoint is None:
            return
        from temsim.gui.qualification_dialog import QualificationDialog
        dialog = QualificationDialog(self)
        if not dialog.exec():
            return
        try:
            plan = dialog.plan(self._checkpoint, maximum_rays=self.budget.value(),
                maximum_checkpoint_bytes=self.checkpoint_budget.value()*1024**2,
                spatial_samples=self.factors["spatial"].value(),
                direction_samples=self.factors["directions"].value(),
                energy_samples=self.factors["energies"].value(),
                check_topology=self.topology.isChecked(),
                checkpoint_spacing_mm=self.bracket_spacing.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save resumable scalar check", "qualification.json", "JSON (*.json)")
            if path:
                self._start_plan(plan, path)
        except Exception as exc:
            self._error(str(exc))

    def _resume_plan(self):
        if self._busy or self._checkpoint is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Resume matching numerical check", "", "JSON (*.json)")
        if not path:
            return
        try:
            from temsim.sampling_qualification import QualificationPlan, load_journal
            previous = load_journal(path)
            plan = QualificationPlan.from_manifest(self._checkpoint, previous["plan"])
            self._start_plan(plan, path, previous)
        except Exception as exc:
            self._error(str(exc))

    def _start_plan(self, plan, path, previous=None):
        from temsim.gui.qualification_dialog import QualificationWorker
        self._cancel = Event()
        self._busy = True
        for button in (self.run_button, self.plan_button, self.resume_button):
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("Checking declared inputs and budgets; no live settings are changed")
        worker = QualificationWorker(plan, self._cancel, path, previous)
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._plan_result)
        worker.signals.error.connect(self._error)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    def _plan_result(self, report):
        import json
        self.plan_report = report  # Keep completed receipts even after cancellation.
        for comparison in report["completed"]:
            self.remember_evidence(comparison)
            self.evidence_ready.emit(comparison)
        self.status.setText(f"Plan {report['plan_id'][:12]}: {report['status']} | {report['reason']}")
        self.details.setPlainText(json.dumps(thaw_json(report), indent=2))

    def _progress(self, message):
        if not self._cancel.is_set():
            self.status.setText(message)

    def _result(self, report):
        if self._cancel.is_set():
            return
        import json
        self.report = report
        self.remember_evidence(report)
        self.export_button.setEnabled(True)
        self.status.setText(f"Evidence for {report['checkpoint_id'][:12]}: {report['comparison']['status']} | "
                            "full numerical and physical qualification not established")
        self.details.setPlainText(json.dumps(thaw_json(report), indent=2))
        self.evidence_ready.emit(report)

    def remember_evidence(self, report):
        level = report.get("refinement_level", 0)
        key = report["axis"] if level == 0 else f"{report['axis']}/refinement-{level}"
        self._history.setdefault(report["checkpoint_id"], {})[key] = report

    def _error(self, message):
        if not self._cancel.is_set():
            self.status.setText(f"Comparison unavailable or failed: {message}. Previous evidence is preserved.")

    def _finished(self):
        self._busy = False
        if self._cancel.is_set():
            self.status.setText("Comparison cancelled. Previous complete evidence is preserved.")
        self.cancel_button.setEnabled(False)
        self.run_button.setEnabled(self._checkpoint is not None)
        self.plan_button.setEnabled(self._checkpoint is not None)
        self.resume_button.setEnabled(self._checkpoint is not None)

    def cancel(self):
        self._cancel.set()
        self.status.setText("Cancellation requested; waiting for the current solver boundary. Previous evidence is preserved.")

    def shutdown(self, timeout_ms=3000):
        self._cancel.set()
        return self.pool.waitForDone(timeout_ms)

    def _export(self):
        if self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export scalar convergence evidence", "sampling-evidence.json", "JSON (*.json)")
        if path:
            try:
                write_evidence(self.report, path)
            except Exception as exc:
                self._error(str(exc))
