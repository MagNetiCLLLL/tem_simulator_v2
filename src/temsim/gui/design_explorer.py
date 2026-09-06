"""Compact dependency and A/B comparison page for microscope designs."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.design_explorer import (
    ArtifactState,
    DesignDifference,
    DesignSnapshot,
    EmptyContainer,
    MissingValue,
    ProductStageStatus,
    diff_design_snapshots,
)
from temsim.design_experiments import (
    DesignRecipe,
    MetricObservation,
    ParameterSweep,
    SensitivityEstimate,
    StateHistory,
    SweepAxis,
    ToleranceResult,
    ToleranceRule,
    estimate_sensitivities,
    evaluate_tolerances,
    plan_parameter_sweep,
    recipe_from_snapshot,
)
from temsim.design_sweep_execution import SWEEP_METRICS


_STATUS_PRESENTATION = {
    ArtifactState.CALCULATED: ("Calculated", "#38bdf8", "#082f49"),
    ArtifactState.REUSED: ("Reusable", "#4ade80", "#052e16"),
    ArtifactState.STALE: ("Stale", "#fbbf24", "#451a03"),
    ArtifactState.MISSING: ("Missing", "#fb7185", "#4c0519"),
}
_OFF_PRESENTATION = ("Off", "#94a3b8", "#1e293b")


class DesignExplorerPage(QWidget):
    """Show cache dependencies and compare two detached design captures."""

    capture_requested = Signal(str)
    sweep_requested = Signal(object, object, object)
    sweep_cancel_requested = Signal()
    sweep_error = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("designExplorerPage")
        self._snapshots: dict[str, DesignSnapshot] = {}
        self._history = StateHistory(limit=100)

        self.cache_summary = QLabel("No High accuracy cache")
        self.cache_summary.setObjectName("designCacheSummary")
        self.cache_summary.setStyleSheet(
            "color: #cbd5e1; font-weight: 600; padding: 2px;"
        )
        self.cache_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.cache_summary.setToolTip(
            "Status is based on calculation identities and cached artifacts. "
            "It does not certify physical accuracy."
        )

        self.capture_a = QPushButton("Capture A")
        self.capture_a.setObjectName("captureDesignA")
        self.capture_b = QPushButton("Capture B")
        self.capture_b.setObjectName("captureDesignB")
        self.clear_captures_button = QPushButton("Clear A/B")
        self.clear_captures_button.setObjectName("clearDesignCaptures")
        for button in (
            self.capture_a,
            self.capture_b,
            self.clear_captures_button,
        ):
            button.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
        self.capture_a.setToolTip(
            "Store an independent copy of the current settings in slot A"
        )
        self.capture_b.setToolTip(
            "Store an independent copy of the current settings in slot B"
        )
        self.clear_captures_button.setToolTip(
            "Remove both comparison captures; calculated results are untouched"
        )

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self.cache_summary, 1)
        toolbar.addWidget(self.capture_a)
        toolbar.addWidget(self.capture_b)
        toolbar.addWidget(self.clear_captures_button)

        self.stage_table = self._new_table(
            "designDependencyTable",
            ("Stage", "Depends on", "Status"),
        )
        self.stage_table.setToolTip(
            "Each row is evaluated from its own scoped dependency identity; "
            "a stale parent does not automatically invalidate its children."
        )
        stage_header = self.stage_table.horizontalHeader()
        stage_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        stage_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        stage_header.setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )

        self.compare_summary = QLabel("Capture A and B to compare")
        self.compare_summary.setObjectName("designComparisonSummary")
        self.compare_summary.setStyleSheet(
            "color: #94a3b8; font-weight: 600; padding: 2px;"
        )
        self.compare_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.compare_summary.setToolTip(
            "Captures contain settings and identities only; they never start "
            "a calculation or copy large result arrays."
        )
        self.diff_table = self._new_table(
            "designDifferenceTable",
            ("Parameter", "A", "B"),
        )
        diff_header = self.diff_table.horizontalHeader()
        diff_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        diff_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        diff_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        dependency_panel = QWidget()
        dependency_layout = QVBoxLayout(dependency_panel)
        dependency_layout.setContentsMargins(0, 0, 0, 0)
        dependency_label = QLabel("Calculation dependencies")
        dependency_label.setStyleSheet("font-weight: 700;")
        dependency_layout.addWidget(dependency_label)
        dependency_layout.addWidget(self.stage_table, 1)

        comparison_panel = QWidget()
        comparison_layout = QVBoxLayout(comparison_panel)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_label = QLabel("A/B settings")
        comparison_label.setStyleSheet("font-weight: 700;")
        comparison_layout.addWidget(comparison_label)
        comparison_layout.addWidget(self.compare_summary)
        comparison_layout.addWidget(self.diff_table, 1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("designExplorerSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(dependency_panel)
        self.splitter.addWidget(comparison_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes((520, 980))

        self.sweep_slot = QComboBox()
        self.sweep_slot.setObjectName("designSweepSourceSlot")
        self.sweep_slot.addItems(("A", "B"))
        self.sweep_parameter = QComboBox()
        self.sweep_parameter.setObjectName("designSweepParameter")
        self.sweep_parameter.setEditable(True)
        for label, path in (
            ("C1 strength (%)", "lenses[condenser_lens_1].percent"),
            ("C2 strength (%)", "lenses[condenser_lens_2].percent"),
            ("C3 strength (%)", "lenses[condenser_lens_3].percent"),
            ("Objective strength (%)", "lenses[objective_lens].percent"),
            ("Diffraction strength (%)", "lenses[diffraction_lens].percent"),
            ("Intermediate strength (%)", "lenses[intermediate_lens].percent"),
            ("P1 strength (%)", "lenses[projector_lens_1].percent"),
            ("P2 strength (%)", "lenses[projector_lens_2].percent"),
        ):
            self.sweep_parameter.addItem(label, path)
        self.sweep_parameter.setToolTip(
            "Choose a common control or type a stable recipe parameter path"
        )
        self.sweep_values = QLineEdit("40, 50, 60")
        self.sweep_values.setObjectName("designSweepValues")
        self.sweep_values.setPlaceholderText("Values, comma separated")
        self.sweep_values.setToolTip(
            "Two or more finite values; one sweep is limited to 64 points"
        )
        self.run_sweep_button = QPushButton("Run sweep")
        self.run_sweep_button.setObjectName("runDesignSweep")
        self.cancel_sweep_button = QPushButton("Cancel")
        self.cancel_sweep_button.setObjectName("cancelDesignSweep")
        self.cancel_sweep_button.setEnabled(False)
        sweep_controls = QHBoxLayout()
        sweep_controls.setContentsMargins(0, 0, 0, 0)
        sweep_controls.addWidget(QLabel("Sweep"))
        sweep_controls.addWidget(self.sweep_slot)
        sweep_controls.addWidget(self.sweep_parameter, 2)
        sweep_controls.addWidget(self.sweep_values, 1)
        sweep_controls.addWidget(self.run_sweep_button)
        sweep_controls.addWidget(self.cancel_sweep_button)
        self.additional_axes = QTableWidget(0, 2)
        self.additional_axes.setObjectName("designSweepAdditionalAxes")
        self.additional_axes.setHorizontalHeaderLabels(("Parameter path", "Values, comma separated"))
        self.additional_axes.horizontalHeader().setStretchLastSection(True)
        self.additional_axes.setMaximumHeight(140)
        self.additional_axes.setToolTip("Additional independent axes. The Cartesian product is limited to 64 points; no preset optimisation is run.")
        add_axis = QPushButton("Add parameter")
        remove_axis = QPushButton("Remove selected parameter")
        add_axis.clicked.connect(lambda: self.additional_axes.insertRow(self.additional_axes.rowCount()))
        remove_axis.clicked.connect(lambda: self.additional_axes.removeRow(self.additional_axes.currentRow()))
        axes_controls = QHBoxLayout()
        axes_controls.addWidget(add_axis)
        axes_controls.addWidget(remove_axis)
        axes_controls.addStretch(1)

        self.tolerance_metric = QComboBox()
        self.tolerance_metric.setObjectName("designSweepToleranceMetric")
        self.tolerance_metric.addItem("No tolerance rule", "")
        for definition in SWEEP_METRICS:
            self.tolerance_metric.addItem(
                f"{definition.label} ({definition.unit})",
                definition.key,
            )
        self.tolerance_metric.setToolTip(
            "Optional physical metric checked independently at every point"
        )
        self.tolerance_minimum = QLineEdit()
        self.tolerance_minimum.setObjectName("designSweepToleranceMinimum")
        self.tolerance_minimum.setPlaceholderText("Minimum")
        self.tolerance_maximum = QLineEdit()
        self.tolerance_maximum.setObjectName("designSweepToleranceMaximum")
        self.tolerance_maximum.setPlaceholderText("Maximum")
        for editor in (self.tolerance_minimum, self.tolerance_maximum):
            editor.setMaximumWidth(120)
            editor.setToolTip(
                "Leave one bound empty for a one-sided tolerance"
            )
        tolerance_controls = QHBoxLayout()
        tolerance_controls.setContentsMargins(0, 0, 0, 0)
        tolerance_controls.addWidget(QLabel("Tolerance"))
        tolerance_controls.addWidget(self.tolerance_metric, 1)
        tolerance_controls.addWidget(self.tolerance_minimum)
        tolerance_controls.addWidget(self.tolerance_maximum)
        tolerance_controls.addStretch(1)

        self.sweep_status = QLabel("Capture A or B, then run a bounded sweep")
        self.sweep_status.setObjectName("designSweepStatus")
        self.sweep_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.sweep_progress = QProgressBar()
        self.sweep_progress.setObjectName("designSweepProgress")
        self.sweep_progress.setRange(0, 1)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setFormat("Idle")

        self.sweep_result_table = self._new_table(
            "designSweepResultTable",
            ("Point", "Coordinates", "Metrics", "Reuse", "Tolerance"),
        )
        result_header = self.sweep_result_table.horizontalHeader()
        result_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        result_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        result_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        result_header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        result_header.setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.sensitivity_table = self._new_table(
            "designSweepSensitivityTable",
            ("Metric", "Parameter", "Slope", "R²"),
        )
        sensitivity_header = self.sensitivity_table.horizontalHeader()
        sensitivity_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        sensitivity_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        for column in (2, 3):
            sensitivity_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        sweep_tables = QSplitter(Qt.Orientation.Horizontal)
        sweep_tables.setObjectName("designSweepTablesSplitter")
        sweep_tables.setChildrenCollapsible(False)
        sweep_tables.addWidget(self.sweep_result_table)
        sweep_tables.addWidget(self.sensitivity_table)
        sweep_tables.setSizes((900, 600))
        sweep_panel = QWidget()
        sweep_layout = QVBoxLayout(sweep_panel)
        sweep_layout.setContentsMargins(0, 0, 0, 0)
        sweep_layout.addLayout(sweep_controls)
        sweep_layout.addLayout(axes_controls)
        sweep_layout.addWidget(self.additional_axes)
        sweep_layout.addLayout(tolerance_controls)
        sweep_info = QHBoxLayout()
        sweep_info.setContentsMargins(0, 0, 0, 0)
        sweep_info.addWidget(self.sweep_status, 1)
        sweep_info.addWidget(self.sweep_progress, 1)
        sweep_layout.addLayout(sweep_info)
        sweep_layout.addWidget(sweep_tables, 1)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setObjectName("designExplorerVerticalSplitter")
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.addWidget(self.splitter)
        self.vertical_splitter.addWidget(sweep_panel)
        self.vertical_splitter.setSizes((580, 340))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(toolbar)
        layout.addWidget(self.vertical_splitter, 1)

        self.capture_a.clicked.connect(
            lambda: self.capture_requested.emit("A")
        )
        self.capture_b.clicked.connect(
            lambda: self.capture_requested.emit("B")
        )
        self.clear_captures_button.clicked.connect(self.clear_captures)
        self.run_sweep_button.clicked.connect(self._request_sweep)
        self.cancel_sweep_button.clicked.connect(
            self.sweep_cancel_requested.emit
        )

    @staticmethod
    def _new_table(name: str, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setObjectName(name)
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().hide()
        table.setSortingEnabled(False)
        return table

    @property
    def snapshot_a(self) -> DesignSnapshot | None:
        return self._snapshots.get("A")

    @property
    def snapshot_b(self) -> DesignSnapshot | None:
        return self._snapshots.get("B")

    def set_product_statuses(
        self, statuses: Sequence[ProductStageStatus]
    ) -> None:
        """Replace the dependency rows without changing any calculation."""

        rows = tuple(statuses)
        labels_by_key = {
            evaluated.stage.key: evaluated.stage.label
            for evaluated in rows
        }
        self.stage_table.setRowCount(len(rows))
        counts = {status: 0 for status in ArtifactState}
        off_count = 0
        for row, evaluated in enumerate(rows):
            if evaluated.requested:
                counts[evaluated.artifact_state] += 1
                presentation = _STATUS_PRESENTATION[evaluated.artifact_state]
            else:
                off_count += 1
                presentation = _OFF_PRESENTATION
            stage = evaluated.stage
            dependency_text = ", ".join(
                labels_by_key.get(key, key) for key in stage.depends_on
            ) or "—"
            label, foreground, background = presentation
            stage_item = QTableWidgetItem(stage.label)
            dependency_item = QTableWidgetItem(dependency_text)
            status_item = QTableWidgetItem(label)
            status_item.setForeground(QColor(foreground))
            status_item.setBackground(QColor(background))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            for item in (stage_item, dependency_item, status_item):
                item.setToolTip(evaluated.reason)
            self.stage_table.setItem(row, 0, stage_item)
            self.stage_table.setItem(row, 1, dependency_item)
            self.stage_table.setItem(row, 2, status_item)

        calculated = counts[ArtifactState.CALCULATED]
        reusable = counts[ArtifactState.REUSED]
        stale = counts[ArtifactState.STALE]
        missing = counts[ArtifactState.MISSING]
        if stale or missing:
            parts = []
            if reusable:
                parts.append(f"{reusable} reusable")
            if stale:
                parts.append(f"{stale} stale")
            if missing:
                parts.append(f"{missing} missing")
            text = " · ".join(parts)
        elif calculated or reusable:
            parts = []
            if calculated:
                parts.append(f"{calculated} calculated")
            if reusable:
                parts.append(f"{reusable} reused")
            text = " · ".join(parts)
        else:
            text = (
                f"{off_count} stages off"
                if off_count
                else "No active High accuracy stages"
            )
        self.cache_summary.setText(text)
        self.cache_summary.setToolTip(
            "Status is based on calculation identities and cached artifacts. "
            "It does not certify physical accuracy."
        )

    def set_calculation_status(self, text: str) -> None:
        self.cache_summary.setText(str(text))

    def set_capture(self, snapshot: DesignSnapshot) -> None:
        """Assign a detached capture to its declared A/B slot."""

        self._snapshots[snapshot.slot] = snapshot
        self._history.record(recipe_from_snapshot(
            snapshot, name=f"Capture {snapshot.slot}"
        ))
        self._refresh_comparison()

    @property
    def state_history(self) -> StateHistory:
        """Expose capture history without retaining live states or results."""

        return self._history

    def recipe_for_slot(self, slot: str, *, name: str = "") -> DesignRecipe:
        key = str(slot).strip().upper()
        try:
            snapshot = self._snapshots[key]
        except KeyError as exc:
            raise ValueError(f"Design slot {key or slot!r} is empty") from exc
        return recipe_from_snapshot(
            snapshot, name=name.strip() or f"Capture {key}"
        )

    def plan_sweep(
        self,
        slot: str,
        axes: Sequence[SweepAxis],
        *,
        maximum_points: int = 256,
    ) -> ParameterSweep:
        """Plan a bounded sweep from an existing A/B capture."""

        return plan_parameter_sweep(
            self.recipe_for_slot(slot),
            axes,
            maximum_points=maximum_points,
        )

    @staticmethod
    def analyse_sweep(
        sweep: ParameterSweep,
        observations: Sequence[MetricObservation],
    ) -> tuple[SensitivityEstimate, ...]:
        return estimate_sensitivities(sweep, observations)

    @staticmethod
    def check_tolerances(
        metrics: dict[str, float],
        rules: Sequence[ToleranceRule],
    ) -> tuple[ToleranceResult, ...]:
        return evaluate_tolerances(metrics, rules)

    def clear_captures(self) -> None:
        self._snapshots.clear()
        self._refresh_comparison()

    def _selected_sweep_path(self) -> str:
        index = self.sweep_parameter.currentIndex()
        current_text = self.sweep_parameter.currentText().strip()
        if index >= 0 and current_text == self.sweep_parameter.itemText(index):
            return str(self.sweep_parameter.itemData(index))
        return current_text

    def _request_sweep(self) -> None:
        try:
            recipe = self.recipe_for_slot(self.sweep_slot.currentText())
            path = self._selected_sweep_path()
            values = tuple(
                float(value.strip())
                for value in self.sweep_values.text().split(",")
                if value.strip()
            )
            if len(values) < 2:
                raise ValueError("Enter at least two sweep values")
            axes = [SweepAxis(path, values, "%" if path.endswith(".percent") else "")]
            for row in range(self.additional_axes.rowCount()):
                cells = [self.additional_axes.item(row, column) for column in range(2)]
                if any(cell is None or not cell.text().strip() for cell in cells):
                    raise ValueError(f"Complete parameter row {row + 2} or remove it")
                extra_path = cells[0].text().strip()
                extra_values = tuple(float(value.strip()) for value in cells[1].text().split(","))
                if len(extra_values) < 2:
                    raise ValueError("Every parameter requires at least two values")
                axes.append(SweepAxis(extra_path, extra_values, "%" if extra_path.endswith(".percent") else ""))
            sweep = plan_parameter_sweep(
                recipe,
                tuple(axes),
                maximum_points=64,
            )
            metric = str(self.tolerance_metric.currentData() or "")
            minimum_text = self.tolerance_minimum.text().strip()
            maximum_text = self.tolerance_maximum.text().strip()
            if not metric and (minimum_text or maximum_text):
                raise ValueError("Choose a metric for the tolerance bounds")
            tolerance_rules = ()
            if metric:
                minimum = float(minimum_text) if minimum_text else None
                maximum = float(maximum_text) if maximum_text else None
                tolerance_rules = (ToleranceRule(
                    metric,
                    minimum=minimum,
                    maximum=maximum,
                ),)
        except (KeyError, TypeError, ValueError) as exc:
            self.sweep_error.emit(str(exc))
            return
        self.sweep_requested.emit(recipe, sweep, tolerance_rules)

    def set_sweep_running(self, point_count: int) -> None:
        self.run_sweep_button.setEnabled(False)
        self.cancel_sweep_button.setEnabled(True)
        self.sweep_progress.setRange(0, max(int(point_count), 1))
        self.sweep_progress.setValue(0)
        self.sweep_progress.setFormat(f"0/{int(point_count)}")
        self.sweep_status.setText("Sweep running")

    def set_sweep_progress(self, progress) -> None:
        total = max(int(progress.total_points), 1)
        completed = min(max(int(progress.completed_points), 0), total)
        self.sweep_progress.setRange(0, total)
        self.sweep_progress.setValue(completed)
        self.sweep_progress.setFormat(f"{completed}/{total}")
        self.sweep_status.setText(
            f"Point {int(progress.point_index) + 1}: {progress.stage}"
        )

    def set_sweep_result(self, result, duration_s: float) -> None:
        definitions = {
            row.key: row for row in result.metric_definitions
        }
        parameter_units = {
            axis.path: axis.unit for axis in result.sweep_axes
        }
        tolerances_by_point = {
            assessment.point_index: assessment.results
            for assessment in result.tolerances
        }
        self.sweep_result_table.setRowCount(len(result.point_results))
        for row, point in enumerate(result.point_results):
            coordinates = ", ".join(
                f"{key}={value:.6g}"
                f"{(' ' + parameter_units[key]) if parameter_units.get(key) else ''}"
                for key, value in point.coordinates.items()
            )
            metric_rows = [
                f"{definitions[key].label}: {value:.6g} "
                f"{definitions[key].unit}"
                for key, value in point.metrics.items()
                if key in definitions
            ]
            reuse = (
                "Complete"
                if point.complete_cache_hit
                else f"{len(point.reused_products)} products"
            )
            tolerance_results = tolerances_by_point.get(
                point.point_index, ()
            )
            failed_tolerances = sum(
                not tolerance.passed for tolerance in tolerance_results
            )
            tolerance_text = (
                "No rule"
                if not tolerance_results
                else "Pass"
                if failed_tolerances == 0
                else f"{failed_tolerances} failed"
            )
            display = (
                str(point.point_index + 1),
                coordinates,
                "; ".join(metric_rows[:3]) or "No finite metric",
                reuse,
                tolerance_text,
            )
            for column, text in enumerate(display):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setToolTip("\n".join(metric_rows))
                elif column == 3:
                    item.setToolTip(
                        "Reused: "
                        + (", ".join(point.reused_products) or "none")
                    )
                elif column == 4:
                    item.setToolTip("\n".join(
                        f"{tolerance.metric}: {tolerance.reason}"
                        for tolerance in tolerance_results
                    ))
                self.sweep_result_table.setItem(row, column, item)

        self.sensitivity_table.setRowCount(len(result.sensitivities))
        for row, estimate in enumerate(result.sensitivities):
            definition = definitions.get(estimate.metric)
            metric_unit = definition.unit if definition else ""
            parameter_unit = parameter_units.get(
                estimate.parameter_path, ""
            )
            slope_unit = "/".join(
                value for value in (metric_unit, parameter_unit) if value
            )
            values = (
                definition.label if definition else estimate.metric,
                estimate.parameter_path,
                f"{estimate.derivative:.6g}"
                f"{(' ' + slope_unit) if slope_unit else ''}",
                f"{estimate.r_squared:.5f}",
            )
            for column, value in enumerate(values):
                self.sensitivity_table.setItem(
                    row, column, QTableWidgetItem(value)
                )
        status = "Cancelled" if result.cancelled else "Complete"
        tolerance_failures = sum(
            not row.passed
            for assessment in result.tolerances
            for row in assessment.results
        )
        tolerance_summary = (
            f"; {tolerance_failures} tolerance failures"
            if any(assessment.results for assessment in result.tolerances)
            else "; no tolerance rule"
        )
        self.sweep_status.setText(
            f"{status}: {result.completed_points} points in {duration_s:.3f} s"
            f"{tolerance_summary}"
        )
        self.sweep_status.setToolTip("\n".join(result.analysis_notes))
        self.sweep_progress.setValue(result.completed_points)

    def set_sweep_error(self, message: str) -> None:
        self.sweep_status.setText(f"Sweep failed: {message}")
        self.sweep_status.setToolTip(str(message))

    def set_sweep_finished(self) -> None:
        self.run_sweep_button.setEnabled(True)
        self.cancel_sweep_button.setEnabled(False)

    def _refresh_comparison(self) -> None:
        snapshot_a = self.snapshot_a
        snapshot_b = self.snapshot_b
        if snapshot_a is None or snapshot_b is None:
            occupied = " / ".join(sorted(self._snapshots))
            self.compare_summary.setText(
                f"Captured {occupied}; capture the other slot"
                if occupied
                else "Capture A and B to compare"
            )
            self.diff_table.setRowCount(0)
            if self._snapshots:
                captured = next(iter(self._snapshots.values()))
                self.compare_summary.setToolTip(
                    f"{captured.slot} captured {captured.captured_at_utc}"
                )
            else:
                self.compare_summary.setToolTip(
                    "Captures contain settings and identities only; they "
                    "never start a calculation or copy large result arrays."
                )
            return

        differences = diff_design_snapshots(snapshot_a, snapshot_b)
        self.diff_table.setRowCount(len(differences))
        for row, difference in enumerate(differences):
            path = self._display_path(difference)
            value_a = self._display_value(difference.value_a)
            value_b = self._display_value(difference.value_b)
            display_values = (path, value_a, value_b)
            tooltip_values = (
                difference.path,
                str(difference.value_a),
                str(difference.value_b),
            )
            for column, value in enumerate(display_values):
                item = QTableWidgetItem(value)
                item.setToolTip(tooltip_values[column])
                self.diff_table.setItem(row, column, item)
        if differences:
            self.compare_summary.setText(
                f"{len(differences)} changed settings"
            )
        else:
            self.compare_summary.setText("A and B match")
        self.compare_summary.setToolTip(
            f"A: {snapshot_a.captured_at_utc}\n"
            f"B: {snapshot_b.captured_at_utc}"
        )

    @staticmethod
    def _display_path(difference: DesignDifference) -> str:
        if difference.kind == "external_identity":
            if difference.path == "identity.calculation_request":
                return "Calculation request identity"
            return "External model / data identity"
        path = difference.path
        if path.startswith("state."):
            path = path[len("state."):]
        path = path.replace(".", " › ").replace("[", " › ")
        return path.replace("]", "")

    @staticmethod
    def _display_value(value: object) -> str:
        if isinstance(value, MissingValue):
            return "—"
        if isinstance(value, EmptyContainer):
            return "{}" if value.kind == "mapping" else "[]"
        if isinstance(value, bool):
            return "On" if value else "Off"
        if isinstance(value, float):
            return f"{value:.9g}"
        text = str(value)
        if len(text) == 64 and all(
            character in "0123456789abcdef" for character in text.lower()
        ):
            return f"{text[:10]}…"
        return text


__all__ = ("DesignExplorerPage",)
