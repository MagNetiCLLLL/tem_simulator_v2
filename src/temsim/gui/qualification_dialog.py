"""Explicit budgets for the existing Sampling & Convergence workflow."""
from PySide6.QtCore import QObject, QRunnable, Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QCheckBox, QLabel, QDialogButtonBox,
)
from temsim.gui.input_policy import WheelSafeSpinBox as QSpinBox
from temsim.sampling_qualification import PLAN_AXES, QualificationPlan, run_qualification


class QualificationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-level numerical check")
        layout = QVBoxLayout(self)
        note = QLabel("Select axes and total budgets. Uses the sampling factors, ray limit and topology settings in the parent panel.")
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        self.axes = {}
        for key, text in PLAN_AXES.items():
            control = QCheckBox(text)
            control.setChecked(key in {"gun_step", "column_step"})
            self.axes[key] = control
            layout.addWidget(control)
        form = QFormLayout()
        self.levels = QSpinBox()
        self.levels.setRange(2, 8)
        self.levels.setValue(2)
        form.addRow("Refinements per axis", self.levels)
        self.comparisons = QSpinBox()
        self.comparisons.setRange(1, 128)
        self.comparisons.setValue(12)
        form.addRow("Maximum comparisons (plus optional root solves)", self.comparisons)
        self.seconds = QSpinBox()
        self.seconds.setRange(1, 7200)
        self.seconds.setValue(120)
        form.addRow("Total time budget (s)", self.seconds)
        layout.addLayout(form)
        warning = QLabel("Insufficient evidence stops the plan. Time cancellation waits for a solver boundary. Field/domain axes without a supported solver remain unavailable.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def plan(self, checkpoint, **common):
        return QualificationPlan(checkpoint,
            tuple(key for key, control in self.axes.items() if control.isChecked()),
            refinements=self.levels.value(), maximum_comparisons=self.comparisons.value(),
            wall_seconds=self.seconds.value(), **common)


class PlanSignals(QObject):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class QualificationWorker(QRunnable):
    def __init__(self, request, cancellation, path, previous=None):
        super().__init__()
        self.request, self.cancellation, self.path = request, cancellation, path
        self.previous = previous
        self.job_input_identity = request.identity
        self.signals = PlanSignals()

    def run(self):
        try:
            report = run_qualification(self.request, previous=self.previous, path=self.path,
                cancelled=self.cancellation.is_set, progress=self.signals.progress.emit)
            self.signals.result.emit(report)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
