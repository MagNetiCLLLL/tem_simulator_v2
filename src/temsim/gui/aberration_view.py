"""System and component aberration diagnostics."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
)

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
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
    _aberration_cache_signature,
)
from temsim.physics.beam_statistics import branch_sample_statistics


class AberrationComparisonView(QWidget):
    """Display effective coefficients before and after active correction."""

    def __init__(self, parent=None, *, fixed_system: str | None = None) -> None:
        super().__init__(parent)
        if fixed_system not in {None, "probe", "image"}:
            raise ValueError("fixed_system must be probe, image or None")
        self.fixed_system = fixed_system
        self._state = None
        self._probe_statistics = None
        self._probe_statistics_error = "Run a calculation to measure the sample probe."
        self._stale = False
        self.system = QComboBox()
        self.system.setObjectName("aberrationSystemSelector")
        self.system.addItem("Probe / specimen", "probe")
        self.system.addItem("Objective / image", "image")
        if fixed_system is not None:
            self.system.setCurrentIndex(self.system.findData(fixed_system))
            self.system.setEnabled(False)
        self.system.currentIndexChanged.connect(self._refresh)
        self.summary = QLabel(
            "Run a calculation to evaluate the active round-lens and "
            "multipole configuration."
        )
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setStyleSheet("font-weight: 600;")
        self.probe_summary = QLabel()
        self.probe_summary.setObjectName("sampleProbeShapeSummary")
        self.probe_summary.setWordWrap(True)
        self.probe_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.probe_summary.setToolTip(
            "Current-weighted geometric rays at the physical sample plane, using "
            "all surviving rays. D95 contains 95% of surviving current. Shape "
            "moments are |<z^n>|/<|z|^n> for centered z=x+iy, n=2 and 3, "
            "in [0, 1]. A three-lobed spot can have zero twofold moment. "
            "These are not fitted A1/A2 coefficients or wave intensity; small "
            "moments alone do not establish an ideal round probe. Finite-source "
            "sampling also contributes to the measured moments. Waist offset "
            "is a local linear extrapolation; positive means downstream."
        )
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("aberrationComparisonTable")
        self.table.setHorizontalHeaderLabels(
            ("Term", "Meaning", "Reference", "Active model", "Difference", "Azimuth", "Basis")
        )
        for column, width in enumerate((48, 180, 100, 110, 110, 100)):
            self.table.setColumnWidth(column, width)
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
        layout.addWidget(self.probe_summary)
        layout.addWidget(self.table, 1)
        self._refresh_probe_summary()

    def display_result(self, result) -> None:
        self._state = getattr(result, "state_snapshot", None)
        self._probe_statistics = None
        try:
            self._probe_statistics = branch_sample_statistics(result.simulation.incident)
        except (AttributeError, ValueError) as exc:
            self._probe_statistics_error = f"Sample probe unavailable: {exc}"
        self._refresh_probe_summary()
        self._stale = True
        self.summary.setText(
            "Aberration state updated. Open this page to inspect coefficients."
        )

    def _refresh_probe_summary(self) -> None:
        self.probe_summary.setVisible(
            str(self.fixed_system or self.system.currentData()) == "probe"
        )
        stats = self._probe_statistics
        if stats is None:
            self.probe_summary.setText(self._probe_statistics_error)
            return
        self.probe_summary.setText(
            f"Sample-plane geometric probe | RMS radius {stats.radius_rms_m * 1.0e9:.4g} nm "
            f"| D95 {2.0 * stats.radius_95_m * 1.0e9:.4g} nm\n"
            f"Twofold moment {stats.twofold_moment:.4f} | threefold moment "
            f"{stats.threefold_moment:.4f} | waist offset "
            f"{stats.waist_offset_m * 1.0e9:+.4g} nm | "
            f"{stats.surviving_rays} surviving rays"
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
            "A5": "six-fold astigmatism",
            "Cc": "first-order chromatic",
        }[term]

    @staticmethod
    def _coefficient_text(value: float) -> str:
        return f"{float(value):+.6g} mm"

    def _refresh(self, *_args) -> None:
        self._refresh_probe_summary()
        if self._state is None:
            return
        self._stale = False
        system = str(self.fixed_system or self.system.currentData())
        from temsim.simulation_modes import is_ideal
        ideal_mode = is_ideal(self._state)
        try:
            field_mode = not ideal_mode and getattr(self._state, f"{system}_aberrations", {}).get("mode") == "field_derived"
            if field_mode:
                entry = getattr(self._state, "_effective_aberration_cache", {}).get(system)
                if entry is None or entry[0] != _aberration_cache_signature(self._state, system):
                    raise ValueError("Run a calculation to fit field-derived coefficients")
            before, after, diagnostics = effective_aberration_comparison(
                self._state,
                system,
            )
        except Exception as exc:
            self.summary.setText(f"Aberration comparison unavailable: {exc}")
            self.table.setRowCount(0)
            return
        self.table.setColumnHidden(2, field_mode)
        self.table.setColumnHidden(4, field_mode)
        if ideal_mode:
            self.summary.setText("Ideal Optics | lens aberrations disabled | defocus retained")
            detail_text = diagnostics["diagnostic_scope"]
        elif field_mode:
            holdout = diagnostics.get("holdout_rms_m")
            validation = f" | holdout RMS {holdout * 1e9:.4g} nm" if holdout is not None else ""
            support = diagnostics.get("field_support_status", "support not assessed")
            self.summary.setText(
                f"{after.reference_plane} | Field-derived | fit RMS "
                f"{diagnostics['fit_rms_m'] * 1e9:.4g} nm{validation} | {support}\n"
                "correction comparison not calculated"
            )
            detail_text = (
                f"{diagnostics['source']}. {diagnostics['diagnostic_scope']}. "
                f"Fit condition number: {diagnostics['fit_condition_number']:.4g}. "
                "Fit residual is model approximation error, not a corrected beam size."
                f" Semi-angle: {diagnostics.get('fit_semiangle_mrad', 'unknown')} mrad; "
                f"step: {diagnostics.get('fit_step_mm', 'unknown')} mm. "
                f"Cc energy convergence: {diagnostics.get('chromatic_convergence', {}).get('status', 'NOT_ASSESSED')}. "
                f"Unmapped lenses: {', '.join(diagnostics.get('unmapped_round_lenses', ()))}. "
                f"Not implemented: {', '.join(diagnostics.get('unimplemented_terms', ()))}."
            )
        else:
            ratio = float(diagnostics["c3_residual_ratio"])
            rms_before = float(diagnostics["ray_error_rms_before"]) * 1.0e9
            rms_after = float(diagnostics["ray_error_rms_after"]) * 1.0e9
            detail_text = (
                f"Reference: {before.reference_plane}. C3 residual ratio "
                f"{ratio:+.4g}; transverse ray-error RMS "
                f"{rms_before:.4g} → {rms_after:.4g} nm. "
                f"{diagnostics['source']}. {diagnostics['diagnostic_scope']} "
                "Values are a non-OEM principle model."
            )
            self.summary.setText(
                f"{before.reference_plane} | compact-ring C3 residual {ratio:+.4g} | "
                f"ray-error RMS {rms_before:.4g} → {rms_after:.4g} nm. "
                "Other residual coefficients are not measured by this comparison."
            )
        self.summary.setToolTip(detail_text)
        self.table.setRowCount(len(SYSTEM_COEFFICIENT_ROWS))
        for row, (term, value_name, angle_name) in enumerate(SYSTEM_COEFFICIENT_ROWS):
            value_before = float(getattr(before, value_name))
            value_after = float(getattr(after, value_name))
            unmeasured = term in diagnostics["unmeasured_coefficients"]
            values = (
                term,
                self._meaning(term),
                "—" if unmeasured else self._coefficient_text(value_before),
                "—" if unmeasured else self._coefficient_text(value_after),
                "—" if unmeasured else self._coefficient_text(value_after - value_before),
                (
                    "—" if unmeasured else
                    "axisymmetric"
                    if angle_name is None
                    else f"{float(getattr(after, angle_name)):.4g}°"
                ),
                diagnostics["coefficient_status"][term],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if 2 <= column <= 5:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
