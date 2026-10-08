"""Bounded CM/OL and AC diagnostics over a captured instrument, never live edits."""
from __future__ import annotations

from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QPlainTextEdit, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim


def _label(text=''):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def _physical_graph_signature(graph):
    from temsim.immutable_json import thaw_json, json_digest
    payload = thaw_json(graph)
    payload['nodes'][payload['root']['ref']]['attributes'].pop('virtual_observation_z_mm', None)
    return json_digest(payload)


class _Signals(QObject):
    progress = Signal(str)
    ready = Signal(object)
    failed = Signal(str)
    finished = Signal()


class _DesignWorker(QRunnable):
    def __init__(self, snapshot, target, ac_range):
        super().__init__()
        self.snapshot, self.target, self.ac_range = snapshot, target, ac_range
        self.cancel_event = Event()
        self.signals = _Signals()
        self.resource_claim = ResourceClaim(512 * 1024**2)
        self.job_input_identity = snapshot.physical_digest
        self.input_signature = _physical_graph_signature(snapshot.graph)

    def run(self):
        try:
            from temsim.optics.condenser_objective_design import (
                evaluate_condenser_objective_design, solve_condenser_objective_design,
            )
            from temsim.optics.scan_coil_design import evaluate_scan_coil_design, search_scan_coil_positions
            state = self.snapshot.restore()
            if self.cancel_event.is_set():
                return
            self.signals.progress.emit('Comparing the current CM position at equal positive / negative excitation…')
            current = evaluate_condenser_objective_design(state)
            search = solve_condenser_objective_design(state, candidate_count=5,
                cancel_check=self.cancel_event.is_set,
                progress_callback=lambda done, total: self.signals.progress.emit(f'CM position candidates: {done}/{total}'))
            # AC is evaluated in the same candidate combined field shown by
            # the plot. This does not apply the geometry to the live column.
            state.mini_condenser.mechanical_center_from_tip_mm = search.best.center_z_mm
            scan_results = []
            for name, sign in (('Microprobe (+)', 1), ('Nanoprobe (−)', -1)):
                if self.cancel_event.is_set():
                    return
                state.mini_condenser.polarity = sign
                self.signals.progress.emit(f'{name}: solving AC position / angle response…')
                existing = evaluate_scan_coil_design(state, target=self.target,
                    response_model='finite_coil', maximum_step_mm=.2)
                candidates = None
                if self.ac_range is not None:
                    candidates = search_scan_coil_positions(state,
                        upper_z_range_mm=self.ac_range,
                        coil_gap_mm=state.ac_deflector.lower_z_mm - state.ac_deflector.upper_z_mm,
                        candidate_count=3, target=self.target, response_model='finite_coil', maximum_step_mm=.2,
                        cancel_check=self.cancel_event.is_set,
                        progress_callback=lambda done, total, label=name:
                            self.signals.progress.emit(f'{label}: AC position candidates {done}/{total}'))
                scan_results.append((name, existing, candidates))
            if not self.cancel_event.is_set():
                self.signals.ready.emit((current, search, tuple(scan_results), self.job_input_identity))
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class CondenserScanDesignPanel(QWidget):
    """First-order design study; hardware controls continue to have one owner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker = None
        self._result = None
        self._result_signature = None
        self._stale = False
        self.pool = CoordinatedPool(self)
        self.pool.coordinator.register_retained(self, 'retained_roots')
        self.summary = _label('Uses the current applied instrument. No design has been calculated.')
        self.status = _label()
        self.calculate = QPushButton('Calculate CM / scan design')
        self.calculate.setProperty('calculationAction', True)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.angle_target = WheelSafeComboBox()
        self.angle_target.addItem('Mechanical direction at sample (current scan convention)', 'mechanical')
        self.angle_target.addItem('Canonical direction at sample (includes vector potential)', 'canonical')
        self.angle_target.setToolTip('A design constraint only. Mechanical and canonical angles differ inside a magnetic field. Neither alone proves detector stationarity.')
        self.search_ac = QCheckBox('Search AC positions within this candidate interval')
        self.search_ac.setChecked(True)
        self.ac_min, self.ac_max = WheelSafeDoubleSpinBox(), WheelSafeDoubleSpinBox()
        for spin in (self.ac_min, self.ac_max):
            spin.setRange(0., 10000.)
            spin.setDecimals(4)
            spin.setSuffix(' mm')
            spin.setKeyboardTracking(False)
        self._range_initialized = False
        controls = QHBoxLayout()
        controls.addWidget(self.calculate)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        form = QFormLayout()
        form.addRow('Sample scan constraint', self.angle_target)
        interval = QHBoxLayout()
        interval.addWidget(self.search_ac)
        interval.addWidget(_label('Upper coil Z from'))
        interval.addWidget(self.ac_min)
        interval.addWidget(_label('to'))
        interval.addWidget(self.ac_max)
        form.addRow(interval)
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Axial Z', units='mm')
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.setLabel('left', 'Combined axial B', units='T')
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.plot.addLegend()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(('CM centre (mm)', 'CM / OL overlap', 'Input correlation mismatch (1/m)', 'Scope'))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Calculated AC positions, physical kick angles and full two-axis pivot checks will appear here.')
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.details, 2)
        split = QSplitter()
        split.addWidget(self.plot)
        split.addWidget(right)
        split.setSizes([650, 650])
        layout = QVBoxLayout(self)
        layout.addWidget(_label(
            'Equal CM excitation magnitude: + Microprobe / − Nanoprobe. All other source and lens inputs are held fixed. '
            'Position searches use the applied fields; changing a mode label alone does not establish the required probe.'))
        layout.addWidget(self.summary)
        layout.addLayout(controls)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)
        layout.addWidget(_label(
            'Diagnostic candidates only; the live column is unchanged. CM bounds check axial envelopes. '
            'AC search preserves coil separation and needs a separate mechanical-clearance check. '
            'Kick values are angles, not amperes: a coil field/current calibration is required for electrical drive. '
            'A scan pivot is not the waist of an individual probe.'))
        self.calculate.clicked.connect(self._calculate)
        self.cancel_button.clicked.connect(self.cancel)
        self.angle_target.currentIndexChanged.connect(self.invalidate_current)
        self.search_ac.toggled.connect(self.invalidate_current)
        self.ac_min.valueChanged.connect(self.invalidate_current)
        self.ac_max.valueChanged.connect(self.invalidate_current)

    def set_state(self, state):
        self._state = state
        cm, ac = state.mini_condenser, state.ac_deflector
        if not self._range_initialized:
            self.ac_min.setValue(ac.upper_z_mm - 1.)
            self.ac_max.setValue(ac.upper_z_mm + 1.)
            self._range_initialized = True
        self.summary.setText(
            f'Applied CM: {cm.polarity * cm.percent:+.6g}% at Z {cm.z_mm:.6g} mm | '
            f'AC upper / lower: {ac.upper_z_mm:.6g} / {ac.lower_z_mm:.6g} mm | '
            f'FOV: {ac.scan_field_of_view_x_nm:.6g} × {ac.scan_field_of_view_y_nm:.6g} nm. '
            'Results describe the captured inputs from the last Calculate click.')
        if self._worker is not None or self._result_signature is not None:
            signature = self._live_signature()
            if self._worker is not None and signature != self._worker.input_signature:
                self._stale = True
                self._worker.cancel_event.set()
                self.status.setText('Applied inputs changed; cancelling the captured design. Calculate again to update.')
            elif self._result_signature is not None and signature != self._result_signature:
                self._stale = True
                self.status.setText('Previous design: applied inputs changed. Calculate again to update.')

    def _live_signature(self):
        from temsim.instrument_snapshot import encode_instrument
        return _physical_graph_signature(encode_instrument(self._state))

    def retained_roots(self):
        return (self._result, self._worker.snapshot if self._worker is not None else None)

    def _calculate(self):
        if self._state is None or self._worker is not None:
            return
        try:
            from temsim.instrument_snapshot import capture_instrument_snapshot
            interval = (self.ac_min.value(), self.ac_max.value()) if self.search_ac.isChecked() else None
            if interval is not None and interval[0] >= interval[1]:
                raise ValueError('AC candidate interval must have increasing Z limits')
            worker = _DesignWorker(capture_instrument_snapshot(self._state), self.angle_target.currentData(), interval)
            worker.signals.progress.connect(lambda text, owner=worker: self._progress(owner, text))
            worker.signals.ready.connect(lambda result, owner=worker: self._receive(owner, result))
            worker.signals.failed.connect(lambda text, owner=worker: self._progress(owner, 'Design not calculated: ' + text))
            worker.signals.finished.connect(lambda owner=worker: self._finished(owner))
            self._worker = worker
            self._stale = False
            self.calculate.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.status.setText('Queued CM / scan design; previous results remain visible.')
            self.pool.start(worker)
        except Exception as exc:
            self.status.setText('Design not calculated: ' + str(exc))
            if self._worker is not None:
                self._finished()

    def _progress(self, owner, text):
        if owner is self._worker and not owner.cancel_event.is_set():
            self.status.setText(text)

    def _receive(self, owner, result):
        if owner is not self._worker:
            return
        self._show_result(result)

    @Slot(object)
    def _show_result(self, result):
        if self._worker is not None and self._worker.cancel_event.is_set():
            return
        if self._worker is not None and self._worker.input_signature != self._live_signature():
            self._stale = True
            self.status.setText('Completed design belongs to previous inputs; calculate again to update.')
            return
        self._result = result
        self._result_signature = self._live_signature() if self._state is not None else None
        current, search, scans, identity = result
        best = search.best
        self.plot.clear()
        for name, candidate, dash in (('Current', current, Qt.PenStyle.DashLine),
                                       ('Candidate', best, Qt.PenStyle.SolidLine)):
            for mode, colour in ((candidate.micro, '#38bdf8'), (candidate.nano, '#f472b6')):
                self.plot.plot(candidate.field_z_mm, mode.total_axial_field_t,
                    pen=pg.mkPen(colour, width=2, style=dash), name=f'{name}: {mode.name}')
        self.table.setRowCount(len(search.candidates))
        for row, candidate in enumerate(search.candidates):
            mismatch = candidate.common_input_correlation_mismatch_rad_per_m
            values = (f'{candidate.center_z_mm:.6g}', f'{100*candidate.cm_objective_normalized_overlap:.4g}%',
                'unresolved' if mismatch is None else f'{mismatch:.6g}', 'First-order diagnostic')
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        lines = [
            f'Captured input: {identity[:16]}',
            f'Fixed CM magnitude: {current.excitation_magnitude_percent:.6g}%',
            f'CM axial bounds: {search.bounds.minimum_center_z_mm:.6g}–{search.bounds.maximum_center_z_mm:.6g} mm',
            f'Lowest sampled mismatch: CM Z {best.center_z_mm:.6g} mm',
            'Ranking compares K in theta = K r needed for A + B K = 0 (nano) and C + D K = 0 (micro).',
            'Input / output coordinates for this comparison: mechanical (x, y, theta_x, theta_y).',
            'K is a direction–position correlation; it need not be a symmetric wave phase curvature.',
            'No finite-probe acceptance: an incident covariance / probe target has not been supplied.',
            '', f'AC diagnostics at candidate CM Z {best.center_z_mm:.6g} mm:',
        ]
        for name, existing, search_ac in scans:
            lines.extend(self._scan_text(name + ' / current AC position', existing))
            if search_ac is not None:
                if search_ac.best is None:
                    lines.append('No candidate satisfies the configured drive / conditioning limits.')
                else:
                    lines.extend(self._scan_text(name + ' / lowest sampled drive', search_ac.best))
        lines.append(best.qualification)
        self.details.setPlainText('\n'.join(lines))
        self.status.setText('Design calculated. Candidates are not applied and do not certify Microprobe / Nanoprobe performance.')

    @staticmethod
    def _scan_text(title, result):
        lines = ['', title,
            f'Upper / lower Z: {result.upper_z_mm:.6g} / {result.lower_z_mm:.6g} mm | target: {result.target}',
            f'Condition number: {result.condition_number:.5g} | drive within limits: {result.kick_limit_pass}',
            result.scope,
            result.linearization_scope,
            'Raster factors −1 to +1 span the full FOV on each axis.']
        if not result.controllable:
            return lines + [result.message]
        for name, matrix in (('Upper', result.upper_kick_matrix_mrad), ('Lower', result.lower_kick_matrix_mrad)):
            lines.append(f'{name} kick matrix (mrad / full-FOV raster factor): {np.array2string(np.asarray(matrix), precision=7)}')
        lines.append(f'Peak physical axis drive: {result.peak_kick_mrad:.7g} mrad | limit: {result.kick_limit_mrad:.7g} mrad')
        lines.append(f'Sample position residual: {result.position_residual_m:.4g} m | angle residual: {result.angle_residual_rad:.4g} rad')
        if result.pivot is not None:
            pivot = result.pivot
            kind = 'Common scan pivot candidate' if pivot.is_common_pivot else 'Closest approach only; not a common pivot'
            lines.append(f'{kind}: Z {pivot.z_mm:.7g} mm | relative 2-axis residual {pivot.relative_residual:.4g}')
        return lines

    @Slot()
    def _finished(self, owner=None):
        if owner is not None and owner is not self._worker:
            return
        cancelled = self._worker is not None and self._worker.cancel_event.is_set()
        self._worker = None
        self.calculate.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self._stale:
            self.status.setText('Previous design: applied inputs changed. Calculate again to update.')
        elif cancelled:
            self.status.setText('Design cancelled; previous results remain visible.')

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel_event.set()
            self.status.setText('Cancelling design calculation…')
            self.pool.clear()

    def invalidate_current(self, *_args):
        """Shared input invalidation can precede the actual state assignment."""
        if self._worker is not None or self._result is not None:
            self._stale = True
            if self._worker is not None:
                self._worker.cancel_event.set()
                self.pool.clear()
            self.status.setText('Previous design: inputs changed. Calculate again to update.')

    def shutdown(self):
        self.cancel()
        self.pool.clear()
        return self.pool.waitForDone(3000)
