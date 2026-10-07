"""Explicit, cached conjugate-plane searches for a captured Ray Diagram."""

from collections import OrderedDict
import math
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.gui.conjugate_plane_context import RecordedConjugateContext
from temsim.gui.plane_equations import plane_equation_tooltip
from temsim.physics.conjugate_planes import build_conjugate_atlas, find_conjugate_planes


class _Signals(QObject):
    atlas_ready = Signal(int, object)
    solved = Signal(int, int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _ConjugateWorker(QRunnable):
    def __init__(self, generation, result_generation, result, reference_z_mm, atlas):
        super().__init__()
        self.generation = generation
        self.result_generation = result_generation
        self.existing_result = result
        self.reference_z_mm = reference_z_mm
        self.atlas = atlas
        self.event = Event()
        self.job_input_identity = f"conjugate:{result_generation}:{reference_z_mm.hex()}"
        self.resource_claim = ResourceClaim(512 * 1024**2)
        self.signals = _Signals()

    def run(self):
        try:
            if self.event.is_set():
                return
            atlas = self.atlas
            if atlas is None:
                atlas = build_conjugate_atlas(self.existing_result, cancelled=self.event.is_set)
                if self.event.is_set():
                    return
                # The atlas is independent of the selected reference. Preserve
                # completed construction even when a subsequent search is cancelled.
                self.signals.atlas_ready.emit(self.result_generation, atlas)
            search = find_conjugate_planes(
                atlas, self.reference_z_mm, cancelled=self.event.is_set,
            )
            if not self.event.is_set():
                self.signals.solved.emit(self.generation, self.result_generation, search)
        except Exception as exc:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class ConjugatePlanePanel(QWidget):
    """Pin a reference explicitly; jumping to a result never changes that pin.

    Only the Find button starts work. Cursor changes and visibility changes do
    not trace the column. Each accepted result publication owns a fresh atlas;
    exact-reference searches reuse that atlas and a bounded display cache.
    """

    plane_selected = Signal(float)
    search_changed = Signal(object)
    CACHE_LIMIT = 32
    _SCOPE = (
        "First-order real-plane conjugacy in the captured fields, from the gun-exit "
        "validity boundary to the executed column / energy-filter boundary. "
        "Upstream results are reciprocal conjugate planes, not virtual images or "
        "backward electron propagation. Matrix conjugacy does not prove particles "
        "reach the plane: inspect particle rays and cutoff outlines separately. "
        "Line focus is not a complete two-dimensional image."
    )
    _KINDS = {"image": "Image", "approximate": "Approximate", "line_focus": "Line focus"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("conjugatePlanePanel")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._result = None
        self._result_generation = 0
        self._generation = 0
        self._selected_z_mm = None
        self._reference_z_mm = None
        self._atlas = None
        self._cache = OrderedDict()
        self._search = None
        self._worker = None
        self._pending = None
        self._stale = False
        self._closed = False
        self._recorded_context = RecordedConjugateContext()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)
        controls = QHBoxLayout()
        controls.setSpacing(4)
        self.reference = QLabel("Reference Z: not pinned")
        self._selectable(self.reference)
        self.reference.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.reference.setMinimumWidth(0)
        controls.addWidget(self.reference, 1)
        self.use_selected_button = QPushButton("Use selected Z")
        self.use_selected_button.setToolTip("Pin the current Ray Diagram Z as the object plane. Then click Find.")
        self.use_selected_button.clicked.connect(self.use_selected_z)
        layout.addLayout(controls)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        actions.addWidget(self.use_selected_button)
        self.find_button = QPushButton("Find conjugate planes")
        self.find_button.setProperty("calculationAction", True)
        self.find_button.setToolTip("Build the column transfer once, then reuse it for searches from other reference planes.")
        self.find_button.clicked.connect(self.find_planes)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.go_button = QPushButton("Go to plane")
        self.go_button.clicked.connect(self._go_to_current)
        actions.addWidget(self.find_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.go_button)
        actions.addStretch(1)
        self.details_toggle = QPushButton("Details")
        self.details_toggle.setObjectName("toggleConjugatePlaneDetails")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolTip("Show the selected plane details and limits of the conjugate search.")
        actions.addWidget(self.details_toggle)
        layout.addLayout(actions)

        self.status = QLabel("Choose a Ray Diagram Z, then click Find conjugate planes.")
        self._selectable(self.status)
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.status.setToolTip(self._SCOPE)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("conjugatePlaneTable")
        self.table.setHorizontalHeaderLabels((
            "Z (mm)", "ΔZ (mm)", "Type", "Scale 1", "Scale 2", "Rotation (°)", "‖B‖ (m/rad)",
        ))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(90)
        self.table.setToolTip(self._SCOPE)
        self.table.horizontalHeaderItem(1).setToolTip(
            "Relative to the pinned reference. Negative ΔZ means an upstream reciprocal conjugate, not a virtual image."
        )
        for column in (3, 4):
            self.table.horizontalHeaderItem(column).setToolTip(
                "Principal transverse position scales (singular values of A); not fixed laboratory X/Y magnifications."
            )
        self.table.horizontalHeaderItem(6).setToolTip(
            "Two-dimensional angular-to-position residual. Zero is ideal first-order image conjugacy."
        )
        self.table.currentCellChanged.connect(self._row_changed)
        self.table.cellClicked.connect(self._row_clicked)
        layout.addWidget(self.table, 1)
        self.details = QPlainTextEdit()
        self.details.setObjectName("conjugatePlaneDetails")
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(40)
        self.details.setMaximumHeight(65)
        self.details.setPlainText(self._SCOPE)
        self.details.setVisible(False)
        self.details_toggle.toggled.connect(self.details.setVisible)
        layout.addWidget(self.details)
        self._update_buttons()

    @staticmethod
    def _selectable(label):
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )

    def _cancel_request(self):
        self._generation += 1
        self._pending = None
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()

    def _update_buttons(self):
        available = self._result is not None and not self._stale and not self._closed
        self.use_selected_button.setEnabled(available and self._selected_z_mm is not None)
        self.find_button.setEnabled(available and (self._reference_z_mm is not None or self._selected_z_mm is not None))
        self.cancel_button.setEnabled(not self._closed and (self._worker is not None or self._pending is not None))
        self.go_button.setEnabled(not self._closed and not self._stale and self.table.currentRow() >= 0)

    def _show_reference(self):
        pin = "not pinned" if self._reference_z_mm is None else f"{self._reference_z_mm:.9g} mm"
        selected = "—" if self._selected_z_mm is None else f"{self._selected_z_mm:.9g} mm"
        self.reference.setText(f"Reference Z: {pin} | selected {selected}")
        self.reference.setToolTip(plane_equation_tooltip(
            "reference" if self._reference_z_mm is not None else "pending",
            reference="pinned reference plane", conjugate_search=True, stale=self._stale,
        ))

    def set_result(self, result):
        if self._closed:
            return
        self._cancel_request()
        self._result_generation += 1
        self._result = result
        self._recorded_context.set_result(result)
        self._atlas = None
        self._cache.clear()
        self._search = None
        self._reference_z_mm = None
        self._stale = False
        self.table.setRowCount(0)
        self.status.setText("Choose a Ray Diagram Z, then click Find conjugate planes.")
        self.details.setPlainText(self._SCOPE)
        self.search_changed.emit(None)
        self._show_reference()
        self._update_buttons()

    def select_z(self, z_mm):
        if self._closed:
            return
        try:
            if isinstance(z_mm, (bool, str, bytes)):
                raise ValueError
            z_mm = float(z_mm)
            if not math.isfinite(z_mm):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            z_mm = None
        self._selected_z_mm = z_mm
        self._show_reference()
        self._update_buttons()

    @Slot()
    def use_selected_z(self):
        if self._closed or self._stale or self._result is None or self._selected_z_mm is None:
            return
        if self._reference_z_mm == self._selected_z_mm:
            return
        self._cancel_request()
        self._reference_z_mm = self._selected_z_mm
        self._search = None
        self.table.setRowCount(0)
        self.search_changed.emit(None)
        self.status.setText("Reference pinned. Click Find conjugate planes.")
        self.details.setPlainText(self._SCOPE)
        self._show_reference()
        self._update_buttons()

    @Slot()
    def find_planes(self):
        if self._closed or self._stale or self._result is None:
            return
        if self._reference_z_mm is None:
            self.use_selected_z()
        if self._reference_z_mm is None:
            return
        self._cancel_request()
        if self._reference_z_mm in self._cache:
            self._cache.move_to_end(self._reference_z_mm)
            self._display(self._cache[self._reference_z_mm])
            self._update_buttons()
            return
        self._pending = (self._generation, self._result_generation, self._result, self._reference_z_mm)
        self.status.setText("Building captured column transfer…" if self._atlas is None else "Searching cached column transfer…")
        self._start_pending()
        self._update_buttons()

    def _start_pending(self):
        if self._closed or self._stale or self._pending is None or self._worker is not None:
            return
        generation, result_generation, result, reference = self._pending
        self._pending = None
        if generation != self._generation or result_generation != self._result_generation:
            return
        worker = _ConjugateWorker(generation, result_generation, result, reference, self._atlas)
        self._worker = worker
        worker.signals.atlas_ready.connect(self._atlas_ready)
        worker.signals.solved.connect(self._solved)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    @Slot(int, object)
    def _atlas_ready(self, result_generation, atlas):
        if self._closed or self._stale or result_generation != self._result_generation:
            return
        self._atlas = atlas

    @Slot(int, int, object)
    def _solved(self, generation, result_generation, search):
        if (self._closed or self._stale or generation != self._generation
                or result_generation != self._result_generation
                or float(search.reference_z_mm) != self._reference_z_mm):
            return
        self._cache[self._reference_z_mm] = search
        self._cache.move_to_end(self._reference_z_mm)
        while len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        self._display(search)

    @Slot(int, str)
    def _failed(self, generation, message):
        if self._closed or self._stale or generation != self._generation:
            return
        self.status.setText("Conjugate search unavailable: " + str(message))
        self.details.setPlainText(str(message) + "\n\n" + self._SCOPE)

    @Slot(int)
    def _finished(self, generation):
        if self._worker is not None and self._worker.generation == generation:
            self._worker = None
        self._start_pending()
        self._update_buttons()

    def _display(self, search):
        self._search = search
        self.table.setRowCount(len(search.candidates))
        for row, candidate in enumerate(search.candidates):
            scales = candidate.magnifications
            values = (
                f"{candidate.z_mm:.9g}", f"{candidate.z_mm-search.reference_z_mm:+.9g}",
                self._KINDS.get(candidate.kind, str(candidate.kind)),
                self._number(scales[0]), self._number(scales[1]),
                self._number(candidate.rotation_deg), self._number(candidate.residual_m_per_rad),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, float(candidate.z_mm))
                item.setToolTip(plane_equation_tooltip(
                    candidate.kind, reference="pinned reference plane", conjugate_search=True,
                ))
                self.table.setItem(row, column, item)
        self.status.setText(
            f"Reference {search.reference_z_mm:.9g} mm | {len(search.candidates)} candidates | "
            f"searched {search.lower_z_mm:.9g}–{search.upper_z_mm:.9g} mm"
        )
        self.details.setPlainText(str(search.detail) + "\n\n" + self._SCOPE)
        self.search_changed.emit(search)
        self._update_buttons()

    @staticmethod
    def _number(value):
        return f"{float(value):.6g}" if value is not None and math.isfinite(float(value)) else "—"

    def _candidate_detail(self, candidate):
        upstream = "Upstream reciprocal conjugate; not a virtual image. " if candidate.z_mm < self._reference_z_mm else ""
        mirrored = "Mirrored position map. " if candidate.mirrored else ""
        return upstream + mirrored + str(candidate.detail) + "\n" + self._SCOPE

    def _row_changed(self, row, _column, _previous_row, _previous_column):
        if self._search is not None and 0 <= row < len(self._search.candidates):
            candidate = self._search.candidates[row]
            detail = self._candidate_detail(candidate)
            if self._stale:
                detail = "Previous optics — inputs changed.\n" + detail
            else:
                detail += "\n\n" + self._recorded_context.at(candidate.z_mm)
            self.details.setPlainText(detail)
        self._update_buttons()

    def _row_clicked(self, row, _column):
        self._jump(row)

    def _go_to_current(self):
        self._jump(self.table.currentRow())

    def _jump(self, row):
        if self._closed or self._stale or self._search is None or not 0 <= row < len(self._search.candidates):
            return
        self.plane_selected.emit(float(self._search.candidates[row].z_mm))

    @Slot()
    def cancel(self):
        self._cancel_request()
        self.status.setText("Search cancelled. Previous complete candidates remain displayed.")
        self._update_buttons()

    def mark_stale(self):
        if self._closed:
            return
        self._cancel_request()
        self._stale = True
        self._atlas = None
        self._cache.clear()
        self._recorded_context.set_result(None)
        self.status.setText("Previous optics — inputs changed. Recalculate Ray Diagram before searching.")
        self._show_reference()
        if self._search is not None:
            for row, candidate in enumerate(self._search.candidates):
                tooltip = plane_equation_tooltip(
                    candidate.kind, reference="pinned reference plane", conjugate_search=True, stale=True,
                )
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    if item is not None:
                        item.setToolTip(tooltip)
        self.search_changed.emit(None)
        self._update_buttons()

    def clear(self):
        self.select_z(None)
        self.set_result(None)

    def shutdown(self):
        self._closed = True
        self._cancel_request()
        self._update_buttons()
        return self.pool.waitForDone(3_000)
