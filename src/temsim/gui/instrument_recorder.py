"""Three signal-collection buttons and a last-signal preview; no acquisition presets."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.recorder.backend import InstrumentRecorder
from temsim.recorder.records import CaptureRequest
from temsim.recorder.paths import prepare_output_directory


def label(text):
    item = QLabel(text)
    item.setWordWrap(True)
    item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return item


class RecorderWorker(QObject):
    completed = Signal(str, object)
    failed = Signal(str, str, bool)
    progress = Signal(str)

    def __init__(self, backend, cancel):
        super().__init__()
        self.backend = backend
        self.cancel = cancel

    @Slot(str, object)
    def perform(self, operation, payload):
        try:
            if operation == "connect":
                result = self.backend.connect(*payload)
            elif operation == "disconnect":
                result = self.backend.disconnect()
            elif operation == "inspect":
                result = self.backend.inspect()
            elif operation == "capture":
                result = self.backend.capture(payload, self.cancel, self.progress.emit)
            else:
                raise ValueError(f"Unknown recorder operation: {operation}")
            self.completed.emit(operation, result)
        except Exception as exc:
            self.failed.emit(operation, f"{type(exc).__name__}: {exc}", self.backend.client is not None)


class InstrumentRecorderWindow(QMainWindow):
    operation_requested = Signal(str, object)

    def __init__(self, parent=None, *, backend=None, settings=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Instrument Recorder")
        self.setObjectName("instrumentRecorderWindow")
        self.settings = settings or QSettings("TEM Simulator", "Instrument Recorder")
        self.backend = backend or InstrumentRecorder()
        self.cancel_event = Event()
        self.busy = self.connected = self._closing = self._shutdown = False
        self._last_path = self._last_manifest = None
        self._previews = []
        self._devices = {"camera": [], "scanning": []}
        self._output_directory, output_error = prepare_output_directory(self.settings.value("output", ""))
        self._build_ui()
        if output_error:
            self.status.setText(output_error)
        self.thread = QThread(self)
        self.worker = RecorderWorker(self.backend, self.cancel_event)
        self.worker.moveToThread(self.thread)
        self.operation_requested.connect(self.worker.perform)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.resize(1100, 800)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._refresh_buttons()

    def _build_ui(self):
        connection = self.menuBar().addMenu("Connection")
        self.connect_action = connection.addAction("Connect...")
        self.connect_action.triggered.connect(self._connect_dialog)
        self.disconnect_action = connection.addAction("Disconnect")
        self.disconnect_action.triggered.connect(lambda: self._start("disconnect"))
        self.inspect_action = connection.addAction("Refresh connection")
        self.inspect_action.triggered.connect(lambda: self._start("inspect"))
        records = self.menuBar().addMenu("Records")
        self.output_action = records.addAction("Choose output folder...")
        self.output_action.triggered.connect(self._browse)
        records.addAction("Open output folder", self._open_output)
        self.last_record_action = records.addAction("Last record details...")
        self.last_record_action.triggered.connect(self._show_record)
        self.stop_action = records.addAction("Cancel collection")
        self.stop_action.triggered.connect(self._stop)
        self.menuBar().addAction("Help", self._help)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        row = QHBoxLayout()
        self.capture_buttons = {}
        for route, title in (("flucam", "Collect Flucam"), ("ceta", "Collect Ceta"),
                             ("stem_all", "Collect all STEM detectors")):
            button = QPushButton(title)
            button.setObjectName(f"recorderCollect_{route}")
            button.clicked.connect(lambda checked=False, r=route: self._capture(r))
            row.addWidget(button)
            self.capture_buttons[route] = button
        layout.addLayout(row)
        self.identity_label = label("Not connected · Connection → Connect")
        layout.addWidget(self.identity_label)
        self.image_channel = QComboBox()
        self.image_channel.currentIndexChanged.connect(self._show_channel)
        self.image_channel.setVisible(False)
        layout.addWidget(self.image_channel)
        self.preview_label = label("No signal collected")
        layout.addWidget(self.preview_label)
        self.image_plot = pg.PlotWidget()
        self.image_plot.setLabel("bottom", "Column", units="px")
        self.image_plot.setLabel("left", "Row", units="px")
        for axis in ("bottom", "left"):
            self.image_plot.getAxis(axis).enableAutoSIPrefix(False)
        self.image_plot.getViewBox().setAspectLocked(True)
        self.image_plot.getViewBox().invertY(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.image_item)
        layout.addWidget(self.image_plot, 1)
        self.status = label("Uses current microscope signals. No acquisition parameters are set here.")
        layout.addWidget(self.status)

    def _refresh_buttons(self):
        ready = self.connected and not self.busy and not self._shutdown
        # Backend uses fresh device and mode readbacks; a stale UI enumeration must not block a mode change.
        for button in self.capture_buttons.values():
            button.setEnabled(ready)
        self.connect_action.setEnabled(not self.connected and not self.busy and not self._shutdown)
        self.disconnect_action.setEnabled(ready)
        self.inspect_action.setEnabled(ready)
        self.output_action.setEnabled(not self.busy)
        self.stop_action.setEnabled(self.busy and not self.cancel_event.is_set())
        self.last_record_action.setEnabled(self._last_manifest is not None)

    def _connect_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Microscope connection")
        form = QFormLayout(dialog)
        host = QLineEdit(str(self.settings.value("host", "")))
        host.setPlaceholderText("Microscope AutoScript host")
        port = QSpinBox()
        port.setRange(1, 65535)
        port.setValue(int(self.settings.value("port", 7521)))
        form.addRow("Host", host)
        form.addRow("Port", port)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.setValue("host", host.text())
            self.settings.setValue("port", port.value())
            self._start("connect", (host.text(), port.value()))

    def _start(self, operation, payload=None):
        if self.busy or self._shutdown:
            return
        self.busy = True
        self.cancel_event.clear()
        self._refresh_buttons()
        self.status.setText(f"{operation.title()} in progress...")
        self.operation_requested.emit(operation, payload)

    def _capture(self, route):
        request = CaptureRequest(output=str(self._output_directory), route=route)
        try:
            request.validate()
            if not self._output_directory.is_dir():
                raise ValueError("Output folder is unavailable. Use Records → Choose output folder.")
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._start("capture", request)

    @Slot(str)
    def _progress(self, message):
        prefix = "Stop requested; finishing safely. " if self.cancel_event.is_set() else ""
        self.status.setText(prefix + message)

    @Slot(str, object)
    def _completed(self, operation, result):
        self.busy = False
        if operation in ("connect", "inspect"):
            self.connected = True
            self._devices = result["detectors"]
            identity = result["identity"]
            self.identity_label.setText(f"{identity['name']} · {identity['serial_number']} · "
                                        f"{result['mode']} / {result['projector_mode']}")
            self.status.setText("Connected. Start the signal on the microscope, then collect it here.")
            self.status.setToolTip(json.dumps(result["errors"], indent=2) if result["errors"] else "")
        elif operation == "disconnect":
            self.connected = False
            self.identity_label.setText("Disconnected")
            self.status.setText("Disconnected. Existing microscope acquisitions were not stopped.")
        elif operation == "capture":
            manifest = result["manifest"]
            self._last_path, self._last_manifest = result["path"], manifest
            if result["previews"]:
                self._previews = result["previews"]
                self.image_channel.clear()
                self.image_channel.addItems([item["label"] for item in self._previews])
                self.image_channel.setVisible(len(self._previews) > 1)
                self.preview_label.setText(f"Last collected signal · {manifest['record_id'][:12]}")
            else:
                # A failed request must not replace or relabel the last successfully retrieved signal.
                self.status.setToolTip("No new signal. The previous signal remains displayed.")
            self.status.setText(f"{manifest['status'].replace('_', ' ')} · "
                                f"{len(manifest['images'])} signal(s) saved")
            details = "\n".join([result["path"], manifest.get("error", ""), *manifest.get("warnings", [])])
            self.status.setToolTip(details)
            context = manifest.get("preflight", manifest["acquisition_context"])
            self.identity_label.setText(f"{manifest['instrument']['name']} · "
                                        f"{context['optical_mode']} / {context['projector_mode']}")
        self._refresh_buttons()
        if operation == "disconnect" and self._closing:
            self.close()

    def _show_channel(self, index):
        if 0 <= index < len(self._previews):
            self.image_item.setImage(self._previews[index]["pixels"], autoLevels=True)
            self.image_plot.autoRange()

    @Slot(str, str, bool)
    def _failed(self, operation, message, connection_retained):
        self.busy = False
        self.connected = connection_retained
        self._closing = False
        self.status.setText(f"{operation.title()} failed: {message}")
        self._refresh_buttons()

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Record output folder", str(self._output_directory))
        if path:
            self._output_directory = Path(path)
            self.settings.setValue("output", str(self._output_directory))

    def _open_output(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_directory)))

    def _show_record(self):
        if self._last_manifest is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Last record")
        box.setText(f"{self._last_manifest['status']}\n{self._last_path}")
        box.setDetailedText(json.dumps(self._last_manifest, indent=2, ensure_ascii=False))
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def _stop(self):
        self.cancel_event.set()
        self.status.setText("Stopping collection after the current SDK read. The microscope stream is not stopped.")
        self._refresh_buttons()

    def _help(self):
        box = QMessageBox(self)
        box.setWindowTitle("Signal collection")
        box.setText("Connect, start the desired signal on the microscope, then click its collect button.\n"
                    "Images and system readbacks are named and saved automatically.")
        box.setDetailedText(
            "No sample type, particle, pixel size, exposure or dwell is assumed.\n"
            "Flucam accepts any optical mode; Ceta requires TEM; STEM requires STEM.\n"
            "Only existing streams exposed by AutoScript 1.18 can be read. Vendor GUI streams "
            "are not guaranteed to be visible. No stream is started, stopped or reconfigured.\n"
            "The SDK reads the next available buffered frame (10-second timeout per read). "
            "Reading may consume a buffered frame; coordinate with other acquisition clients.\n"
            "Image metadata is preserved separately from system readbacks. Unknown or old frame "
            "timing is flagged; timestamps do not establish synchronised clocks. STEM channels "
            "may come from different frames.\n"
            "Original EMD import and automatic microscope magnification/rotation fitting are not implemented.")
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def closeEvent(self, event):
        if self.busy:
            self.status.setText("Collection is running. Use Records → Cancel collection, then close.")
            event.ignore()
            return
        if self.connected:
            self._closing = True
            self._start("disconnect")
            event.ignore()
            return
        self.thread.quit()
        if not self.thread.wait(1500):
            self.status.setText("Waiting for the recorder worker to finish; try closing again.")
            event.ignore()
            return
        self._shutdown = True
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("output", str(self._output_directory))
        super().closeEvent(event)
