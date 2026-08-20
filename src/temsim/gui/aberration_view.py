"""System and component aberration diagnostics."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.optics.aberrations import (
    SYSTEM_COEFFICIENT_ROWS,
    effective_aberration_comparison,
)


class AberrationComparisonView(QWidget):
    """Display effective coefficients before and after active correction."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = None
        self._stale = False
        self.system = QComboBox()
        self.system.setObjectName("aberrationSystemSelector")
        self.system.addItem("Probe / specimen", "probe")
        self.system.addItem("Objective / image", "image")
        self.system.currentIndexChanged.connect(self._refresh)
        self.summary = QLabel(
            "Run a calculation to evaluate the active round-lens and "
            "multipole configuration."
        )
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #475569; font-weight: 600;")
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("aberrationComparisonTable")
        self.table.setHorizontalHeaderLabels(
            ("Term", "Meaning", "Uncorrected", "Corrected", "Difference", "Azimuth")
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("Effective system"))
        selector.addWidget(self.system)
        selector.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(selector)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

    def display_result(self, result) -> None:
        self._state = getattr(result, "state_snapshot", None)
        self._stale = True
        self.summary.setText(
            "Aberration state updated. Open this page to run the compact "
            "corrector comparison."
        )

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._stale:
            self._refresh()

    @staticmethod
    def _meaning(term: str) -> str:
        return {
            "C1": "defocus",
            "A1": "two-fold astigmatism",
            "B2": "axial coma",
            "A2": "three-fold astigmatism",
            "C3": "third-order spherical",
            "S3": "star aberration",
            "A3": "four-fold astigmatism",
            "C5": "fifth-order spherical",
            "Cc": "first-order chromatic",
        }[term]

    @staticmethod
    def _coefficient_text(value: float) -> str:
        return f"{float(value):+.6g} mm"

    def _refresh(self, *_args) -> None:
        if self._state is None:
            return
        self._stale = False
        try:
            before, after, diagnostics = effective_aberration_comparison(
                self._state,
                str(self.system.currentData()),
            )
        except Exception as exc:
            self.summary.setText(f"Aberration comparison unavailable: {exc}")
            self.table.setRowCount(0)
            return
        ratio = float(diagnostics["c3_residual_ratio"])
        rms_before = float(diagnostics["ray_error_rms_before"])
        rms_after = float(diagnostics["ray_error_rms_after"])
        self.summary.setText(
            f"Reference: {before.reference_plane}. C3 residual ratio "
            f"{ratio:+.4g}; transverse ray-error RMS "
            f"{rms_before:.4g} → {rms_after:.4g} rad. "
            f"{diagnostics['source']}. Values are a non-OEM principle model."
        )
        self.table.setRowCount(len(SYSTEM_COEFFICIENT_ROWS))
        for row, (term, value_name, angle_name) in enumerate(SYSTEM_COEFFICIENT_ROWS):
            value_before = float(getattr(before, value_name))
            value_after = float(getattr(after, value_name))
            values = (
                term,
                self._meaning(term),
                self._coefficient_text(value_before),
                self._coefficient_text(value_after),
                self._coefficient_text(value_after - value_before),
                (
                    "axisymmetric"
                    if angle_name is None
                    else f"{float(getattr(after, angle_name)):.4g}°"
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column >= 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
