"""Read-only cache browser. Restore/fork are explicit main-window transactions."""
import json
from threading import Event

from PySide6.QtCore import Qt, Signal, Slot, QThreadPool
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QSplitter,
    QLabel, QPushButton, QFileDialog, QAbstractItemView, QInputDialog)

from temsim.immutable_json import thaw_json
from temsim.working_point import (WorkingPointCheckpoint, WorkingPointArchiveIndex,
    OBSERVABLE_DEFINITIONS, snapshot_changes, migrate_working_point_inputs)
from temsim.calculation_manifest import solver_source_identity
from temsim.sampling_diagnostics import checkpoint_sampling_summary, working_point_description
from temsim.gui.sampling_panel import SamplingPanel


class WorkingPointPanel(QWidget):
    restore_requested = Signal(object, bool)
    illumination_requested = Signal(object)
    undo_requested = Signal()
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workingPointPanel")
        self._points = []
        self._descriptions = {}
        self._summaries = {}
        self._evidence = {}
        self._pins = {}
        from temsim.gui.job_coordinator import CoordinatedPool
        self._archive_pool = CoordinatedPool(self)
        self._archive_pool.setMaxThreadCount(1)
        self._archive_cancel = Event()
        self._archive_loading = False
        self._archive_worker = None
        self._imported_evidence = {}
        self._implementation = solver_source_identity()
        self.current_snapshot = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Selected checkpoint | read-only | no working point selected")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bar = QHBoxLayout()
        for label, callback in (("Import...", self._import), ("Export...", self._export),
                                ("Load retained data", self._load_selected),
                                ("Make portable input copy", self._make_portable),
                                ("Save input candidate", self._save_inputs)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
        layout.addLayout(bar)
        bar = QHBoxLayout()
        for label, callback in (("Compare with current", self._compare), ("Apply illumination...", self._apply_illumination),
                                ("Restore working point", lambda: self._restore(False)),
                                ("Fork compatible point", lambda: self._restore(True)),
                                ("Migrate inputs", self._migrate), ("Undo last apply", self.undo_requested.emit)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
        layout.addLayout(bar)
        compare_bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter source, assembly, mode, status, date or identity")
        self.filter.textChanged.connect(self._filter_rows)
        compare_bar.addWidget(self.filter)
        for label, callback in (("Pin A", lambda: self._pin("A")), ("Pin B", lambda: self._pin("B")),
                                ("Compare A / B", self._compare_pins)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            compare_bar.addWidget(button)
        layout.addLayout(compare_bar)
        self.points = QTableWidget(0, 12)
        self.points.setObjectName("workingPointIndex")
        self.points.setHorizontalHeaderLabels(["Record", "Source", "Assembly", "Mode", "Current (pA)",
            "D95 (um)", "Alpha95 (mrad)", "Z (mm)", "Numerical evidence", "Date (UTC)", "Compatibility", "Execution"])
        self.points.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.points.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.points.setSortingEnabled(True)
        for column, width in enumerate((210, 150, 210, 160, 125, 110, 140, 110, 250, 150, 260, 150)):
            self.points.setColumnWidth(column, width)
        layout.addWidget(self.points, 1)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workingPointBrowserSplitter")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Component", "Model"])
        self.tree.setColumnWidth(0, 220)
        self.values = QTableWidget(0, 2)
        self.values.setHorizontalHeaderLabels(["Captured parameter", "Value"])
        self.values.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.values.horizontalHeader().setStretchLastSection(True)
        self.values.setColumnWidth(0, 240)
        self.observables = QTableWidget(0, 4)
        self.observables.setHorizontalHeaderLabels(["Observable", "Value", "Unit", "Status"])
        self.observables.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.observables.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((245, 180, 90)):
            self.observables.setColumnWidth(column, width)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("workingPointDetailsSplitter")
        right.addWidget(self.values)
        right.addWidget(self.observables)
        for widget in (self.tree, right):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([250, 570])
        tabs = QTabWidget()
        tabs.addTab(self.splitter, "Captured state and readouts")
        self.sampling = SamplingPanel(self)
        tabs.addTab(self.sampling, "Sampling and Convergence")
        self.sampling.evidence_ready.connect(self._accept_evidence)
        layout.addWidget(tabs, 2)
        self.points.currentCellChanged.connect(lambda row, *_: self._select(row))
        self.tree.currentItemChanged.connect(self._component)

    @property
    def selected(self):
        item = self.points.item(self.points.currentRow(), 0)
        identity = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return next((point for point in self._points if point.digest == identity), None)

    def add_checkpoint(self, checkpoint, *, label="Completed", select=True):
        previous_selection = self.selected
        for row in range(self.points.rowCount()):
            if self.points.item(row, 0).data(Qt.ItemDataRole.UserRole) == checkpoint.digest:
                old = next(point for point in self._points if point.digest == checkpoint.digest)
                if not isinstance(old, WorkingPointArchiveIndex) or isinstance(checkpoint, WorkingPointArchiveIndex):
                    if select:
                        self.points.setCurrentCell(row, 0)
                    return
                self._points.remove(old)
                self.points.removeRow(row)
                break
        description = working_point_description(checkpoint, current_implementation=self._implementation)
        summary = checkpoint_sampling_summary(checkpoint)
        self._descriptions[checkpoint.digest] = description
        self._summaries[checkpoint.digest] = summary
        self._points.append(checkpoint)
        if isinstance(checkpoint, WorkingPointArchiveIndex):
            self._imported_evidence[checkpoint.digest] = checkpoint.evidence
        self.points.setSortingEnabled(False)
        row = self.points.rowCount()
        self.points.insertRow(row)
        def scaled(key, factor):
            value = summary[key]
            return "Unavailable" if value is None else value * factor
        values = (f"{label} | {checkpoint.digest[:12]}", description["source"], description["assembly"],
            f"{description['mode']} / {description['simulation_mode']}", scaled("plane_current_a", 1e12), scaled("diameter95_m", 1e6),
            scaled("alpha95_rad", 1e3), checkpoint.plane_z_mm, "NOT_RUN", description["date"],
            description["compatibility"], description["execution"])
        for column, value in enumerate(values):
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, value)
            item.setData(Qt.ItemDataRole.UserRole, checkpoint.digest)
            item.setToolTip(str(value))
            self.points.setItem(row, column, item)
        selected_item = self.points.item(row, 0)
        self.points.setSortingEnabled(True)
        if select or (previous_selection is not None and previous_selection.digest == checkpoint.digest):
            self.points.setCurrentItem(selected_item)
        elif previous_selection is not None:
            for row in range(self.points.rowCount()):
                if self.points.item(row, 0).data(Qt.ItemDataRole.UserRole) == previous_selection.digest:
                    self.points.setCurrentCell(row, 0)
                    break
        for report in self._imported_evidence.get(checkpoint.digest, ()):
            self._accept_imported_evidence(checkpoint, report)
        self._filter_rows()

    def _select(self, index):
        self.tree.clear()
        cp = self.selected
        if cp is None:
            self.values.setRowCount(0)
            self.observables.setRowCount(0)
            self.sampling.set_checkpoint(None)
            return
        description = self._descriptions[cp.digest]
        self.status.setText(f"Selected checkpoint {cp.digest[:12]} | Z {cp.plane_z_mm:g} mm | "
                            f"{cp.metadata.get('source_representation', 'historical')} | read-only\n"
                            f"Preset: {description['quality']} | physics validation: {description['physical_validation']} | "
                            f"{description['compatibility']}\n{description['wave_capability']}")
        if cp.is_input_design:
            self.status.setText(f"Input design {cp.digest[:12]} | Restore loads settings only; all results require recalculation")
            if "archived_inputs" in cp.snapshot.graph:
                self.status.setText(self.status.text() + " | Archived inputs; compatible solver required; structure is read-only")
        if cp.is_metadata_only:
            self.status.setText(f"Metadata summary {cp.digest[:12]} | Read-only; input assets and retained results are absent. Restore and recalculation are unavailable.")
        # Include every graph node, including non-component configuration and
        # shared references, rather than filtering out disabled hardware.
        for index, node in enumerate(cp.snapshot.graph["nodes"]):
            attrs = node.get("attributes", {})
            name = str(attrs.get("name", attrs.get("key", f"Object {index}")))
            item = QTreeWidgetItem([name, node["type"].split(":")[-1]])
            item.setData(0, Qt.ItemDataRole.UserRole, thaw_json(attrs))
            self.tree.addTopLevelItem(item)
        summary = self._summaries[cp.digest]
        self.sampling.set_checkpoint(None if cp.is_metadata_only else cp, summary)
        if cp.is_metadata_only or isinstance(cp, WorkingPointArchiveIndex):
            self.observables.setRowCount(1)
            values = (("Numeric products", "Absent; historical scalar summary only", "", "METADATA_ONLY")
                      if cp.is_metadata_only else ("Numeric products", "Load retained data to verify", "", "INDEX_ONLY"))
            for column, value in enumerate(values):
                self.observables.setItem(0, column, QTableWidgetItem(value))
            if self.tree.topLevelItemCount():
                self.tree.setCurrentItem(self.tree.topLevelItem(0))
            return
        self.observables.setRowCount(len(OBSERVABLE_DEFINITIONS) + len(summary))
        reader = cp.observables
        for row, key in enumerate(OBSERVABLE_DEFINITIONS):
            try:
                record = reader.get(key)
            except ValueError as exc:
                from temsim.working_point import ObservableRecord
                record = ObservableRecord(key, None, "", "unavailable", cp.plane_id,
                                          cp.digest, "UNAVAILABLE", reason=str(exc))
            for column, value in enumerate((key, record.value, record.unit, record.status)):
                item = QTableWidgetItem("—" if value is None else str(value))
                item.setToolTip(f"{record.definition_id}\n{record.reason}\n{record.plane_id}")
                self.observables.setItem(row, column, item)
        for row, (key, value) in enumerate(summary.items(), start=len(OBSERVABLE_DEFINITIONS)):
            for column, text in enumerate((key, value, "see field suffix", summary["status"])):
                self.observables.setItem(row, column, QTableWidgetItem("Unavailable" if text is None else str(text)))
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _component(self, item, previous=None):
        if item is None:
            return
        self._show_values((key, json.dumps(value, ensure_ascii=True)) for key, value in
                          item.data(0, Qt.ItemDataRole.UserRole).items())

    def _show_values(self, rows):
        rows = list(rows)
        self.values.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.values.setItem(row, column, item)

    def _compare(self):
        if self.selected is None or self.current_snapshot is None:
            return
        try:
            current = self.current_snapshot()
            changes = snapshot_changes(self.selected.snapshot, current)
            rows = [(path, json.dumps({"captured": old, "current": new})) for path, old, new in changes]
            from temsim.parameter_registry import dependency_plan
            try:
                plan = dependency_plan(self.selected.snapshot, current)
                rows.extend(("Reuse / " + stage, ("Reuse: " if row["reusable"] else "Recompute: ") + "; ".join(row["reasons"]))
                            for stage, row in plan["stages"].items())
                rows.append(("Field support", json.dumps(thaw_json(plan["incident_field_support"]))))
            except (ValueError, TypeError, KeyError) as exc:
                rows.append(("Reuse plan unavailable", str(exc)))
            self._show_values(rows)
            self.status.setText(f"Selected checkpoint → current | {len(changes)} exact differences | no changes applied")
        except Exception as exc:
            self.error.emit(str(exc))

    def _apply_illumination(self):
        if self.selected is None:
            self.status.setText("Select a compatible working point to preview illumination controls")
            return
        if self.selected.is_metadata_only:
            self.status.setText("Metadata-only records cannot apply illumination controls")
            return
        self.illumination_requested.emit(self.selected)

    def _filter_rows(self, *_):
        query = self.filter.text().casefold().strip()
        for row in range(self.points.rowCount()):
            text = " ".join(self.points.item(row, col).text() for col in range(self.points.columnCount()))
            text += " " + self.points.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.points.setRowHidden(row, query not in text.casefold())

    def _pin(self, name):
        if self.selected is not None:
            self._pins[name] = self.selected
            self.status.setText(f"Pinned {name}: {self.selected.digest[:12]} | captured state remains unchanged")

    def _compare_pins(self):
        if set(self._pins) != {"A", "B"}:
            self.status.setText("Pin both A and B to compare captured inputs and readouts")
            return
        a, b = self._pins["A"], self._pins["B"]
        da, db = self._descriptions[a.digest], self._descriptions[b.digest]
        compatible = a.plane_id == b.plane_id and all(da[k] == db[k] for k in ("source", "assembly", "mode", "simulation_mode"))
        rows = [("Checkpoint A", a.digest), ("Checkpoint B", b.digest),
                ("Comparison context", "Matching labels and plane; inspect full input differences" if compatible else "Different source, assembly, mode or plane")]
        rows += [("Readout / " + key, json.dumps(thaw_json({"A": self._summaries[a.digest].get(key), "B": self._summaries[b.digest].get(key)})))
                 for key in sorted(self._summaries[a.digest].keys() | self._summaries[b.digest].keys())]
        rows += [(path, json.dumps({"A": old, "B": new})) for path, old, new in snapshot_changes(a.snapshot, b.snapshot)]
        self._show_values(rows)
        self.status.setText(f"Pinned A {a.digest[:12]} / B {b.digest[:12]} | read-only; no qualification inferred")

    def _accept_evidence(self, report):
        identity = report["checkpoint_id"]
        cp = next((point for point in self._points if point.digest == identity), None)
        if (cp is None or cp.snapshot.digest != report["snapshot_id"] or
                cp.snapshot.implementation != report["implementation"] or report["implementation"] != self._implementation):
            self.error.emit("Evidence identity mismatch; no current numerical status was changed")
            return
        self._evidence[identity] = report
        self.sampling.remember_evidence(report)
        reports = list(self._imported_evidence.get(identity, ()))
        if not any(old.get("digest") == report.get("digest") for old in reports):
            reports.append(report)
        self._imported_evidence[identity] = tuple(reports)
        # Sorting can move a row as its status changes; use its immutable ID.
        for row in range(self.points.rowCount()):
            if self.points.item(row, 0).data(Qt.ItemDataRole.UserRole) == identity:
                self.points.item(row, 8).setText(f"{report['axis']}: {report['comparison']['status']}")
                break
        self._filter_rows()

    def _accept_imported_evidence(self, checkpoint, report):
        from temsim.working_point_evidence import assess_evidence
        try:
            status, reason = assess_evidence(checkpoint, report, implementation=self._implementation)
            if status == "MATCHING_NUMERICAL_COMPARISON":
                self._accept_evidence(report)
            else:
                for row in range(self.points.rowCount()):
                    if self.points.item(row, 0).data(Qt.ItemDataRole.UserRole) == checkpoint.digest:
                        self.points.item(row, 8).setText(status)
                        self.points.item(row, 8).setToolTip(reason)
                        break
        except Exception as exc:
            self.error.emit(str(exc))

    def _load_selected(self):
        point = self.selected
        if point is not None and point.is_metadata_only:
            self.status.setText("This metadata-only record contains no numeric payload to load")
            return
        if not isinstance(point, WorkingPointArchiveIndex):
            self.status.setText("Selected numeric products are already loaded, or no record is selected")
            return
        if self._archive_loading:
            self.status.setText("An archive is already loading; the current record remains readable")
            return
        from temsim.gui.working_point_loader import ArchiveLoader
        self._archive_loading = True
        self.status.setText(f"Verifying retained numeric products for {point.digest[:12]}...")
        worker = ArchiveLoader(point, self._archive_cancel)
        self._archive_worker = worker
        worker.signals.ready.connect(self._accept_archive, Qt.ConnectionType.QueuedConnection)
        worker.signals.failed.connect(self.error.emit)
        worker.signals.finished.connect(self._load_finished)
        self._archive_pool.start(worker)

    @Slot(object)
    def _accept_archive(self, loaded):
        label = "Portable inputs" if self._archive_worker.portable_inputs else "Verified archive"
        self.add_checkpoint(loaded, label=label, select=False)

    @Slot()
    def _load_finished(self):
        self._archive_loading = False
        self._archive_worker = None

    def _make_portable(self):
        point = self.selected
        if point is None or point.is_metadata_only:
            self.status.setText("Select complete, compatible inputs to capture a portable copy")
            return
        if self._archive_loading:
            self.status.setText("Archive work is already running; the selected record remains readable")
            return
        from temsim.gui.working_point_loader import ArchiveLoader
        self._archive_loading = True
        self.status.setText("Capturing a portable input copy; existing results retain their original identity...")
        worker = ArchiveLoader(point, self._archive_cancel, portable_inputs=True)
        self._archive_worker = worker
        worker.signals.ready.connect(self._accept_archive, Qt.ConnectionType.QueuedConnection)
        worker.signals.failed.connect(self.error.emit)
        worker.signals.finished.connect(self._load_finished)
        self._archive_pool.start(worker)

    def shutdown(self, timeout_ms=3000):
        self._archive_cancel.set()
        archive_done = self._archive_pool.waitForDone(timeout_ms)
        sampling_done = self.sampling.shutdown(timeout_ms)
        return archive_done and sampling_done

    def _migrate(self):
        if self.selected is None:
            self.status.setText("Select a historical record to create a new input candidate")
            return
        if self.selected.is_metadata_only:
            self.status.setText("Metadata-only records cannot migrate absent input assets")
            return
        try:
            migrated = migrate_working_point_inputs(self.selected)
            self.add_checkpoint(migrated, label="Migrated inputs")
            self.status.setText("New input-only identity created. Historical results remain with the original record; live controls unchanged.")
        except Exception as exc:
            self.error.emit(f"Input migration unavailable: {exc}")

    def _save_inputs(self):
        if self.current_snapshot is None:
            return
        try:
            from datetime import datetime, timezone
            snapshot = self.current_snapshot()
            nodes = snapshot.graph["nodes"]
            root = nodes[snapshot.graph["root"]["ref"]]["attributes"]
            sample = nodes[root["sample"]["ref"]]["attributes"]
            cp = WorkingPointCheckpoint(snapshot, {}, float(sample["z_mm"]), snapshot.physical_digest,
                {"package_kind": "INSTRUMENT_INPUTS_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "validation_status": "NOT_RUN", "source_representation": "captured-inputs-only"})
            self.add_checkpoint(cp, label="Input candidate")
        except Exception as exc:
            self.error.emit(str(exc))

    def _restore(self, fork):
        if self.selected is not None and self.selected.is_metadata_only:
            self.status.setText("Metadata-only records cannot Restore or Fork; import a package containing complete inputs")
            return
        if isinstance(self.selected, WorkingPointArchiveIndex):
            self.status.setText("Load retained data before exact Restore or Fork; archive products must be verified first")
            return
        if self.selected is not None:
            self.restore_requested.emit(self.selected, fork)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import working point", "", "Working point (*.temwp)")
        if path:
            try:
                self.add_checkpoint(WorkingPointArchiveIndex.read(path), label="Indexed archive")
            except Exception as exc:
                self.error.emit(str(exc))

    def _export(self):
        point = self.selected
        if point is None:
            return
        choices = ["Complete inputs and retained results", "Complete inputs; recalculate results", "Metadata only; cannot restore"]
        default = 2 if point.is_metadata_only else 0 if point.has_retained_payload else 1
        choice, accepted = QInputDialog.getItem(self, "Export content",
            "Included content (use Make portable input copy to embed all configuration dependencies)", choices, default, False)
        if not accepted:
            return
        mode = ("inputs_and_results", "inputs", "metadata")[choices.index(choice)]
        if isinstance(point, WorkingPointArchiveIndex) and mode == "inputs_and_results":
            self.status.setText("Load retained data before export to verify the original archive")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export working point", "working-point.temwp", "Working point (*.temwp)")
        if path:
            try:
                reports = list(self._imported_evidence.get(point.digest, ()))
                current = self._evidence.get(point.digest)
                if current is not None and current not in reports:
                    reports.append(current)
                point.write_package(path, overwrite=True, evidence=reports, mode=mode)  # Native dialog confirms an existing target.
                self.status.setText(f"Exported {choice.lower()}; no live controls changed")
            except Exception as exc:
                self.error.emit(str(exc))
