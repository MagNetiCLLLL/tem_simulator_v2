"""Independent electron records with selected or overlaid electromagnetic paths.

These inputs never enter an instrument state, source, or transport checkpoint.
Captured field inputs and cached field solutions are reused without launching
an instrument population or creating a new source/checkpoint.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox
from temsim.gui.input_policy import WheelSafeSpinBox as QSpinBox
from temsim.gui.test_electron_types import ElectronPath
from temsim.gui.virtual_electron_panel import VirtualElectronPanel
from temsim.test_electron_execution import ElectronExecutionBackend, RemoteElectronScene

if TYPE_CHECKING:
    from temsim.magnetic_test_particle import (
        TestElectronSettings,
        TestElectronTrajectory,
    )


class _Signals(QObject):
    finished = Signal(object, object, object)
    progress = Signal(object, object)


class _SceneWorker(QRunnable):
    """Prepare electric fields once, separately from adjustable electron inputs."""

    def __init__(self, generation, state, magnetic_scene, z_limits_mm, *, backend=None):
        super().__init__()
        self.generation = generation
        self.state, self.magnetic_scene, self.z_limits_mm = state, magnetic_scene, z_limits_mm
        self.backend = backend
        self.cancelled = Event()
        self.signals = _Signals()

    def run(self):
        scene, error = None, None
        try:
            from temsim.magnetic_field_scene import MagneticSceneField
            from temsim.optics.model import State
            from temsim.cpu_resources import numerical_job
            from temsim.test_electron_scene import prepare_test_electron_scene
            if (self.backend is not None and isinstance(self.state, State)
                    and (self.magnetic_scene is None or isinstance(self.magnetic_scene, MagneticSceneField))):
                scene = self.backend.prepare(self.state, self.magnetic_scene,
                    z_limits_mm=self.z_limits_mm, cancelled=self.cancelled.is_set)
            else:
                # Standalone injected field fixtures have no captured instrument
                # graph. The application's captured fields always use isolation.
                with numerical_job(1, cancelled=self.cancelled.is_set):
                    if not self.cancelled.is_set():
                        scene = prepare_test_electron_scene(
                            self.state, self.magnetic_scene, z_limits_mm=self.z_limits_mm)
        except Exception as exc:  # noqa: BLE001 - report failures across the Qt worker boundary
            error = str(exc)
        if self.cancelled.is_set():
            scene, error = None, None
        self.signals.finished.emit(self.generation, scene, error)


class _ElectronWorker(QRunnable):
    def __init__(self, key, scene, *, backend=None):
        super().__init__()
        self.key, self.scene = key, scene
        self.backend = backend
        self.cancelled = Event()
        self.signals = _Signals()

    def _publish_progress(self, result):
        if not self.cancelled.is_set():
            self.signals.progress.emit(self.key, result)

    def run(self):
        result, error = None, None
        try:
            from temsim.cpu_resources import numerical_job
            from temsim.magnetic_test_particle import trace_test_electron
            if isinstance(self.scene, RemoteElectronScene):
                if self.backend is None:
                    raise RuntimeError("The prepared diagnostic fields have no execution owner")
                result = self.backend.trace(self.scene, self.key[1], cancelled=self.cancelled.is_set,
                                            progress=self._publish_progress)
            else:
                with numerical_job(1, cancelled=self.cancelled.is_set):
                    if not self.cancelled.is_set():
                        result = trace_test_electron(
                            self.scene, self.key[1], cancelled=self.cancelled.is_set,
                            progress=self._publish_progress)
        except Exception as exc:  # noqa: BLE001 - report failures across the Qt worker boundary
            error = str(exc)
        if self.cancelled.is_set():
            result, error = None, None
        self.signals.finished.emit(self.key, result, error)


@dataclass
class ElectronRecord:
    key: str
    label: str
    colour: str
    settings: TestElectronSettings
    checked: bool = True
    trajectory: TestElectronTrajectory | None = None
    revision: int = 0
    attempted: bool = False
    error: str | None = None
    ready_at: float = 0.
    trajectory_generation: int = -1
    trajectory_settings: TestElectronSettings | None = None
    progress_trajectory: TestElectronTrajectory | None = None
    progress_key: tuple | None = None
    _path_signature: object | None = None
    _paths: dict = field(default_factory=dict)


class TestElectronController(QObject):
    """Independent records sharing one frozen scene and one numerical worker."""

    paths_changed = Signal(object)
    status_changed = Signal(str)
    background_changed = Signal(bool)
    controls_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self._execution_backend = ElectronExecutionBackend()
        self.destroyed.connect(self._execution_backend.close)
        self._scene = None
        self._scene_request = None
        self._scene_worker = None
        self._scene_error = None
        self._generation = 0
        self._active = False
        self._created_initial_record = False
        self._loading_editor = True
        self._worker = None
        self._cache = OrderedDict()
        self._records = OrderedDict()
        self._selected_key = None
        self._next_id = 1
        self._items = {}
        self._icons = {}
        self._editing_controls = {}
        self._view_limits_mm = None
        self.status_text = "Select captured fields to trace a virtual electron from the tip."
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._request)

        self.controls_button = QPushButton("Electrons…", parent)
        self.controls_button.setObjectName("magneticTestElectronControls")
        self.controls_button.setToolTip("Manage independent virtual electrons in the captured electric and magnetic fields.")
        self.controls_button.clicked.connect(self.show_controls)
        self.controls_button.hide()
        self.panel = VirtualElectronPanel(parent)
        self.panel.installEventFilter(self)
        self.panel.hide()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add electron")
        self.add_button.setToolTip("Add an independent electron with the captured tip position and emission energy.")
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setToolTip("Duplicate these parameters and reuse the identical calculated path until parameters change.")
        self.remove_button = QPushButton("Remove")
        self.add_button.clicked.connect(self.add_electron)
        self.duplicate_button.clicked.connect(self.duplicate_electron)
        self.remove_button.clicked.connect(lambda: self.remove_electron())
        for button in (self.add_button, self.duplicate_button, self.remove_button):
            buttons.addWidget(button)
        left_layout.addLayout(buttons)
        self.display_mode = QComboBox()
        self.display_mode.setObjectName("testElectronDisplayMode")
        self.display_mode.addItem("Selected electron", "selected")
        self.display_mode.addItem("Overlay checked electrons", "overlay")
        self.display_mode.currentIndexChanged.connect(self._visibility_changed)
        left_layout.addWidget(self.display_mode)
        self.electron_list = QTreeWidget()
        self.electron_list.setObjectName("magneticTestElectronList")
        self.electron_list.setHeaderLabels(("Electron", "State", "End Z (mm)"))
        self.electron_list.setRootIsDecorated(False)
        self.electron_list.setAlternatingRowColors(True)
        self.electron_list.setColumnWidth(0, 145)
        self.electron_list.setColumnWidth(1, 85)
        self.electron_list.setColumnWidth(2, 95)
        self.electron_list.currentItemChanged.connect(self._list_selection_changed)
        self.electron_list.itemChanged.connect(self._list_item_changed)
        left_layout.addWidget(self.electron_list, 1)
        independent = QLabel("Each electron is an independent test path. Check rows to include them in an overlay; electrons do not interact with one another. Identical parameters can produce overlapping paths.")
        independent.setWordWrap(True)
        independent.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(independent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        self.scope_label = QLabel(
            "Virtual electron in captured electric and magnetic fields. Starts at the tip by default; "
            "kinetic energy changes during extraction and acceleration. "
            "Hardware interception is included; specimen and detector interactions are excluded.")
        self.scope_label.setWordWrap(True)
        self.scope_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.scope_label)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self.name_editor = QLineEdit()
        self.name_editor.setObjectName("testElectronName")
        self.name_editor.textEdited.connect(self._name_changed)
        form.addRow("Name", self.name_editor)
        self.sliders = {}
        self.energy = self._input(form, "Initial energy", "energy", .001, 1e7, .3, 4, " eV", .1, log=True)
        self.energy.setToolTip("Kinetic energy at the initial position. Tip emission is usually sub-eV; this is not the final accelerating energy.")
        self.x = self._input(form, "Initial X", "x", -10000., 10000., 0., 6, " µm", .001, slider_range=(-.01, .01))
        self.y = self._input(form, "Initial Y", "y", -10000., 10000., 0., 6, " µm", .001, slider_range=(-.01, .01))
        self.z = self._input(form, "Initial Z", "z", -10000., 10000., 0., 9, " mm", .000001, slider_range=(0., 3100.))
        self.polar_degrees = QLabel("0°")
        self.polar_degrees.setObjectName("testElectronPolarDegrees")
        self.polar_degrees.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.polar = self._input(form, "Polar angle\nslider 0–5 mrad", "polar", 0., math.pi*1000., 0., 6, " mrad", .01,
                                 slider_range=(0., 5.), companion=self.polar_degrees)
        self.polar.setToolTip(
            "Initial angle from +Z in milliradians. The adjacent value is degrees: "
            "0° forward, 90° transverse, 180° backward. The slider finely adjusts 0–5 mrad; "
            "the numerical input retains the full 0–3141.592654 mrad range.")
        self.sliders["polar"].setToolTip(
            "Fine polar angle: 0–5 mrad, in 0.0025 mrad increments. Measured from +Z. "
            "Values outside this slider range remain unchanged in the numerical input "
            "until you move the slider.")
        self.polar.valueChanged.connect(lambda value: self.polar_degrees.setText(f"{math.degrees(value/1000.):.6g}°"))
        self.azimuth = self._input(form, "Azimuth", "azimuth", 0., 360., 0., 3, " °", 1.)
        self.azimuth.setToolTip(
            "Orientation of the initial transverse direction: measured from +X towards +Y "
            "(0° = +X, 90° = +Y). At polar angle 0, the electron points along +Z; "
            "changing azimuth does not change its initial direction. This changes the electron's "
            "initial direction, not the Ray Diagram projection rotation.")
        self.sliders["azimuth"].setToolTip(self.azimuth.toolTip())
        self.length = self._input(form, "Maximum path", "length", .001, 10000., 50., 3, " mm", 1., log=True)
        self.start_at_view = QPushButton("Start at view centre")
        self.start_at_view.setEnabled(False)
        self.start_at_view.clicked.connect(self._use_view_centre)
        form.addRow("", self.start_at_view)
        self.background = QCheckBox("Show magnetic field lines in background")
        self.background.setChecked(True)
        self.background.toggled.connect(self.background_changed)
        layout.addWidget(self.background)
        self.advanced_toggle = QCheckBox("Advanced numerical settings")
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        self.step = QDoubleSpinBox()
        self.step.setObjectName("magneticTestElectronStep")
        self.step.setRange(.00001, 10.)
        self.step.setDecimals(5)
        self.step.setValue(1.)
        self.step.setSuffix(" mm")
        self.step.setToolTip("Maximum spatial step. Actual steps also resolve electric acceleration, field maps and hardware boundaries.")
        self.max_steps = QSpinBox()
        self.max_steps.setRange(100, 100000)
        self.max_steps.setValue(20000)
        self.max_steps.setSingleStep(1000)
        self.relative_tolerance = QDoubleSpinBox()
        self.relative_tolerance.setDecimals(8)
        self.relative_tolerance.setRange(1e-8, 1e-2)
        self.relative_tolerance.setValue(1e-4)
        self.relative_tolerance.setSingleStep(1e-4)
        self.position_tolerance = QDoubleSpinBox()
        self.position_tolerance.setDecimals(6)
        self.position_tolerance.setRange(.000001, 1000.)
        self.position_tolerance.setValue(.001)
        self.position_tolerance.setSingleStep(.001)
        self.position_tolerance.setSuffix(" nm")
        for name, control in (("step", self.step), ("max_steps", self.max_steps),
                              ("relative_tolerance", self.relative_tolerance),
                              ("position_tolerance", self.position_tolerance)):
            self._connect_text_commit(control, name)
        advanced_form.addRow("Maximum step", self.step)
        advanced_form.addRow("Step budget", self.max_steps)
        advanced_form.addRow("Relative tolerance", self.relative_tolerance)
        advanced_form.addRow("Position tolerance", self.position_tolerance)
        self.step.valueChanged.connect(lambda _value: self._changed("step"))
        self.max_steps.valueChanged.connect(lambda _value: self._changed("max_steps"))
        self.relative_tolerance.valueChanged.connect(lambda _value: self._changed("relative_tolerance"))
        self.position_tolerance.valueChanged.connect(lambda _value: self._changed("position_tolerance"))
        layout.addWidget(self.advanced)
        self.advanced.hide()
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.domain_label = QLabel("Field validity range pending a captured scene.")
        self.domain_label.setWordWrap(True)
        self.domain_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.domain_label)
        self.reset_to_tip = QPushButton("Reset to tip emission")
        self.reset_to_tip.setObjectName("testElectronResetToTip")
        self.reset_to_tip.clicked.connect(self._reset_to_tip)
        layout.addWidget(self.reset_to_tip)
        self.energy_status = QLabel("Energy along the trajectory: pending")
        self.energy_status.setObjectName("testElectronEnergyReadout")
        self.energy_status.setWordWrap(True)
        self.energy_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.energy_status)
        self.status_label = QLabel(self.status_text)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        scroll.setWidget(content)
        self.panel.install_columns(left, scroll)
        # Return commits the numerical editor without activating an action.
        for button in self.panel.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self._loading_editor = False
        self._set_inputs_enabled(False)
        self.add_button.setEnabled(False)
        self.duplicate_button.setEnabled(False)
        self.remove_button.setEnabled(False)

    def _set_inputs_enabled(self, enabled):
        for control in (self.name_editor, self.energy, self.x, self.y, self.z, self.polar, self.azimuth,
                        self.length, self.step, self.max_steps, self.relative_tolerance,
                        self.position_tolerance, self.reset_to_tip, *self.sliders.values()):
            control.setEnabled(enabled)
        self.start_at_view.setEnabled(enabled and self._view_limits_mm is not None)

    def _input(self, form, label, name, lower, upper, value, decimals, suffix, step,
               *, log=False, slider_range=None, companion=None):
        spin = QDoubleSpinBox()
        spin.setObjectName("magneticTestElectron" + name.title())
        spin.setDecimals(decimals)
        spin.setRange(lower, upper)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.setMinimumWidth(145)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 2000)
        self.sliders[name] = slider
        self._connect_text_commit(spin, name)
        slider.sliderPressed.connect(lambda: self._editing_started(("slider", name)))
        slider.sliderReleased.connect(lambda: self._editing_finished(("slider", name)))
        low, high = slider_range or (lower, upper)
        if log:
            low, high = math.log(low), math.log(high)

        def spin_changed(number):
            fraction = ((math.log(number) if log else number) - low) / (high - low)
            slider.blockSignals(True)
            slider.setValue(round(np.clip(fraction, 0., 1.) * 2000))
            slider.blockSignals(False)
            self._changed(name)

        def slider_changed(tick):
            number = low + (high - low) * tick / 2000
            spin.setValue(math.exp(number) if log else number)

        spin.valueChanged.connect(spin_changed)
        slider.valueChanged.connect(slider_changed)
        row = QHBoxLayout()
        row.addWidget(spin)
        row.addWidget(slider, 1)
        if companion is not None:
            row.addWidget(companion)
        form.addRow(label, row)
        spin_changed(value)
        return spin

    def _connect_text_commit(self, control, name):
        # Keep the current value visible while typing, but do not trace numeric
        # prefixes when a user pauses longer than the ordinary debounce delay.
        control.lineEdit().textEdited.connect(lambda _text: self._editing_started(("text", name)))
        control.editingFinished.connect(lambda: self._editing_finished(("text", name)))

    def _is_editing(self, record):
        return record is not None and record.key in self._editing_controls.values()

    def _is_typing(self, record):
        return record is not None and any(kind == "text" and key == record.key
            for (kind, _name), key in self._editing_controls.items())

    def _is_sliding(self, record):
        return record is not None and any(kind == "slider" and key == record.key
            for (kind, _name), key in self._editing_controls.items())

    def _editing_started(self, control):
        if self._loading_editor or self.selected_record is None:
            return
        key = self.selected_record.key
        if self._editing_controls.get(control) == key:
            return
        self._editing_controls[control] = key
        self._cancel_unwanted_worker()
        self._refresh()
        self._schedule()

    def _editing_finished(self, control):
        key = self._editing_controls.pop(control, None)
        if key is None or self._loading_editor:
            return
        record = self._records.get(key)
        if record is not None:
            record.ready_at = monotonic()
            self._reuse_result(record)
        self._refresh()
        self._schedule()

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() == QEvent.Type.Hide and self._editing_controls:
            keys = set(self._editing_controls.values())
            self._editing_controls.clear()
            for slider in self.sliders.values():
                if slider.isSliderDown():
                    slider.setSliderDown(False)
            for key in keys:
                record = self._records.get(key)
                if record is not None:
                    self._reuse_result(record)
            self._refresh()
            self._schedule()
        return super().eventFilter(watched, event)

    @Slot()
    def show_controls(self):
        self.controls_requested.emit()


    @property
    def records(self):
        return tuple(self._records.values())

    @property
    def selected_record(self):
        return self._records.get(self._selected_key)

    @property
    def current_trajectory(self):
        record = self.selected_record
        return record.trajectory if record is not None and self._has_result(record) else None

    def settings(self):
        record = self.selected_record
        return None if record is None else record.settings

    def _has_result(self, record):
        return (not self._is_typing(record) and record.trajectory is not None and record.trajectory_generation == self._generation
                and record.trajectory_settings == record.settings)

    def _work_key(self, record):
        return self._generation, record.settings, record.key, record.revision

    def _wanted_records(self):
        if self.display_mode.currentData() == "selected":
            record = self.selected_record
            return () if record is None else (record,)
        checked = [record for record in self.records if record.checked]
        checked.sort(key=lambda record: record.key != self._selected_key)
        return tuple(checked)

    def visible_paths(self):
        paths = []
        for record in self._wanted_records():
            current = self._has_result(record)
            progress_matches = (record.progress_key is not None and record.progress_key[1] == record.settings
                                and not self._is_typing(record))
            signature = (self._generation, id(record.trajectory), id(record.progress_trajectory),
                         progress_matches, current, record.label, record.colour)
            if signature != record._path_signature:
                record._paths.clear()
                record._path_signature = signature
            selected = record.key == self._selected_key
            if selected not in record._paths:
                shown = []
                if record.trajectory is not None and record.trajectory_generation == self._generation:
                    result = record.trajectory
                    shown.append(ElectronPath(
                        key=record.key, label=record.label if current else record.label+" · previous",
                        colour=record.colour, positions_m=result.positions_m, time_s=result.time_s,
                        selected=selected, state="current" if current else "previous"))
                if not current and record.progress_trajectory is not None and record.progress_key[0] == self._generation:
                    result = record.progress_trajectory
                    shown.append(ElectronPath(
                        key=record.key+"/progress", label=record.label+" · live preview"+("" if progress_matches else " · earlier settings"),
                        colour=record.colour, positions_m=result.positions_m, time_s=result.time_s,
                        selected=selected, state="in_progress" if progress_matches else "previous"))
                record._paths[selected] = tuple(shown)
            paths.extend(record._paths[selected])
        return tuple(paths)

    def _tip_settings(self):
        from temsim.magnetic_test_particle import TestElectronSettings
        return TestElectronSettings(
            kinetic_energy_ev=float(self._scene.initial_energy_ev),
            position_m=tuple(float(v) for v in self._scene.initial_position_m),
            polar_angle_deg=0., azimuth_angle_deg=0.,
            max_path_length_m=float(self._scene.default_path_length_m),
            step_m=.001, max_steps=20000,
            relative_tolerance=1e-4, position_tolerance_m=1e-12)

    def _new_record(self, settings):
        palette = ("#ffd166", "#48cae4", "#ef7fae", "#90e06f",
                   "#c39cff", "#ff995e", "#69e6c3", "#83a9ff")
        index = self._next_id
        self._next_id += 1
        record = ElectronRecord(
            key=f"electron-{index}", label=f"Electron {index}",
            colour=palette[(index-1) % len(palette)], settings=settings,
            ready_at=monotonic()+.150)
        self._records[record.key] = record
        self._selected_key = record.key
        return record

    def add_electron(self, *_args):
        if self._scene is None:
            return None
        record = self._new_record(self._tip_settings())
        self._reuse_result(record)
        self._load_editor()
        self._refresh()
        self._schedule()
        return record.key

    def duplicate_electron(self, *_args):
        original = self.selected_record
        if original is None or self._scene is None:
            return None
        record = self._new_record(original.settings)
        if not self._reuse_result(record) and original.attempted and original.error:
            record.attempted = True
            record.error = original.error
        self._load_editor()
        self._refresh()
        self._schedule()
        return record.key

    def remove_electron(self, key=None):
        key = self._selected_key if key is None else key
        if key not in self._records:
            return False
        keys = list(self._records)
        index = keys.index(key)
        self._records.pop(key)
        if self._worker is not None and self._worker.key[2] == key:
            self._worker.cancelled.set()
        if self._selected_key == key:
            remaining = list(self._records)
            self._selected_key = remaining[min(index, len(remaining)-1)] if remaining else None
        self._load_editor()
        self._refresh()
        self._schedule()
        return True

    def select_electron(self, key):
        if key not in self._records:
            return False
        if key != self._selected_key:
            self._selected_key = key
            self._load_editor()
        self._refresh()
        self._schedule()
        return True

    def set_checked(self, key, checked):
        record = self._records.get(key)
        if record is None:
            return False
        record.checked = bool(checked)
        self._refresh()
        self._schedule()
        return True

    def _list_selection_changed(self, current, _previous=None):
        if current is not None:
            self.select_electron(current.data(0, Qt.ItemDataRole.UserRole))

    def _list_item_changed(self, item, column):
        if column == 0:
            self.set_checked(item.data(0, Qt.ItemDataRole.UserRole),
                             item.checkState(0) == Qt.CheckState.Checked)

    def _visibility_changed(self, *_args):
        if self._loading_editor:
            return
        self._refresh()
        self._schedule()

    def _name_changed(self, text):
        if self._loading_editor or self.selected_record is None:
            return
        record = self.selected_record
        label = str(text).strip()
        if label and label != record.label:
            record.label = label
            self._refresh()

    def _load_editor(self):
        record = self.selected_record
        # A programmatic record switch abandons the text editor's unfinished
        # representation; the prior record retains its last valid input value.
        self._editing_controls.clear()
        self._loading_editor = True
        try:
            self.name_editor.setText("" if record is None else record.label)
            if record is not None:
                settings = record.settings
                self.energy.setValue(settings.kinetic_energy_ev)
                self.x.setValue(settings.position_m[0]*1e6)
                self.y.setValue(settings.position_m[1]*1e6)
                self.z.setValue(settings.position_m[2]*1000.)
                self.polar.setValue(math.radians(settings.polar_angle_deg)*1000.)
                self.azimuth.setValue(settings.azimuth_angle_deg)
                self.length.setValue(settings.max_path_length_m*1000.)
                self.step.setValue(settings.step_m*1000.)
                self.max_steps.setValue(settings.max_steps)
                self.relative_tolerance.setValue(settings.relative_tolerance)
                self.position_tolerance.setValue(settings.position_tolerance_m*1e9)
        finally:
            self._loading_editor = False
        self._set_inputs_enabled(record is not None and self._scene is not None)

    def _clear_result(self, record, *, retain_display=False):
        if not retain_display:
            record.trajectory = None
            record.trajectory_generation = -1
            record.trajectory_settings = None
            record.progress_trajectory = None
            record.progress_key = None
            record._path_signature = None
            record._paths.clear()
        record.attempted = False
        record.error = None

    def _changed(self, name):
        if self._loading_editor or self.selected_record is None:
            return
        record = self.selected_record
        was_pending = not self._has_result(record) and not record.attempted
        # Editing one control must not round-trip untouched values through
        # spin boxes. Captured tip coordinates and numerical tolerances retain
        # their exact precision and cache identity until explicitly changed.
        if name in {"x", "y", "z"}:
            point = list(record.settings.position_m)
            index, control, scale = {
                "x": (0, self.x, 1e6), "y": (1, self.y, 1e6), "z": (2, self.z, 1000.),
            }[name]
            point[index] = control.value()/scale
            settings = replace(record.settings, position_m=tuple(point))
        else:
            field_name, value = {
                "energy": ("kinetic_energy_ev", self.energy.value()),
                "polar": ("polar_angle_deg", min(180., math.degrees(self.polar.value()/1000.))),
                "azimuth": ("azimuth_angle_deg", self.azimuth.value()),
                "length": ("max_path_length_m", self.length.value()/1000.),
                "step": ("step_m", self.step.value()/1000.),
                "max_steps": ("max_steps", self.max_steps.value()),
                "relative_tolerance": ("relative_tolerance", self.relative_tolerance.value()),
                "position_tolerance": ("position_tolerance_m", self.position_tolerance.value()/1e9),
            }[name]
            settings = replace(record.settings, **{field_name: value})
        if settings == record.settings:
            return
        record.settings = settings
        record.revision += 1
        if not was_pending:
            record.ready_at = monotonic()+.060
        self._clear_result(record, retain_display=True)
        self._reuse_result(record)
        self._restore_lost_execution()
        self._refresh()
        self._schedule()

    def _reset_to_tip(self):
        record = self.selected_record
        if self._scene is None or record is None:
            return
        default = self._tip_settings()
        updated = replace(record.settings,
            kinetic_energy_ev=default.kinetic_energy_ev, position_m=default.position_m,
            polar_angle_deg=default.polar_angle_deg, azimuth_angle_deg=default.azimuth_angle_deg)
        if updated == record.settings:
            return
        record.settings = updated
        record.revision += 1
        record.ready_at = monotonic()+.060
        self._clear_result(record, retain_display=True)
        if self._worker is not None and self._worker.key[2] == record.key:
            self._worker.cancelled.set()
        self._reuse_result(record)
        self._load_editor()
        self._refresh()
        self._schedule()

    def set_captured_scene(self, state, magnetic_scene, z_limits_mm):
        """Invalidate results, retaining every electron's parameters and identity."""
        self.invalidate()
        if state is not None:
            self._scene_request = (state, magnetic_scene, z_limits_mm)
            self._refresh()
            self._schedule()

    def set_scene(self, scene):
        """Install an already prepared electromagnetic scene."""
        self.invalidate()
        self._install_scene(scene)

    def _install_scene(self, scene):
        self._scene = scene
        if scene is None:
            return
        if not self._created_initial_record:
            self._created_initial_record = True
            self._new_record(self._tip_settings())
        if hasattr(scene, "bounds_m"):
            bounds = np.asarray(scene.bounds_m)
            diagnostic_bounds = getattr(scene, "diagnostic_bounds_m", None)
            domain = bounds if diagnostic_bounds is None else np.asarray(diagnostic_bounds)
            self.domain_label.setText(
                f"Diagnostic range: Z {domain[0, 2]*1000.:.6g}–{domain[1, 2]*1000.:.6g} mm. "
                "Extraction, gun lens and acceleration fields are included. "
                "Stops distinguish physical interception from field-model limits.")
            self.domain_label.setToolTip("\n".join(getattr(scene, "notes", ())))
        self._load_editor()
        self._refresh()
        self._schedule()

    def invalidate(self):
        self._generation += 1
        self._scene = None
        self._scene_request = None
        self._scene_error = None
        self._cache.clear()
        self._timer.stop()
        self._editing_controls.clear()
        for worker in (self._worker, self._scene_worker):
            if worker is not None:
                worker.cancelled.set()
        for record in self.records:
            self._clear_result(record)
        self.domain_label.setText("Field validity range pending a captured scene.")
        self.domain_label.setToolTip("")
        self._set_inputs_enabled(False)
        self._refresh()

    def set_active(self, active):
        if active and not self._active:
            self._restore_lost_execution()
        self._active = bool(active)
        if not self._active:
            self._timer.stop()
            self._editing_controls.clear()
            for slider in self.sliders.values():
                if slider.isSliderDown():
                    slider.setSliderDown(False)
            for worker in (self._worker, self._scene_worker):
                if worker is not None:
                    worker.cancelled.set()
        self._refresh()
        self._schedule()

    def _restore_lost_execution(self):
        """An explicit edit/reopen may reprepare after child loss, never loop."""
        if (isinstance(self._scene, RemoteElectronScene) and self._scene_request is not None
                and not self._execution_backend.owns_scene(self._scene)):
            self._scene = None
            self._scene_error = None
            for record in self.records:
                if record.error:
                    record.error = None
                    record.attempted = False

    def shutdown(self):
        """Cancel owned work before the application waits for its thread pool."""
        self._active = False
        self._timer.stop()
        self._editing_controls.clear()
        for worker in (self._worker, self._scene_worker):
            if worker is not None:
                worker.cancelled.set()
        self._execution_backend.close()

    def set_axial_range_mm(self, lower, upper):
        self._view_limits_mm = (float(lower), float(upper))
        self.start_at_view.setEnabled(self._scene is not None and self.selected_record is not None)

    def _use_view_centre(self):
        if self._view_limits_mm is not None and self.selected_record is not None:
            self.z.setValue(sum(self._view_limits_mm)/2.)

    def _reuse_result(self, record):
        if self._scene is None or self._has_result(record) or self._is_typing(record):
            return False
        key = self._generation, record.settings
        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
        else:
            result = next((other.trajectory for other in self.records
                           if other.key != record.key and other.settings == record.settings
                           and self._has_result(other)), None)
        if result is None:
            return False
        self._store_result(record, result)
        return True

    def _store_result(self, record, result, *, settings=None):
        settings = record.settings if settings is None else settings
        key = self._generation, settings
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        # A sampled request may finish after the user returns to an exact
        # cached state. Keep that current display; only archive the older sample.
        if settings != record.settings and self._has_result(record):
            return
        record.trajectory = result
        record.trajectory_generation = self._generation
        record.trajectory_settings = settings
        record.attempted = settings == record.settings
        record.progress_trajectory = None
        record.progress_key = None
        record.error = None
        record._paths.clear()
        record._path_signature = None

    def _cancel_unwanted_worker(self):
        if self._worker is None:
            return
        wanted = {record.key for record in self._wanted_records()} if self._active else set()
        key = self._worker.key
        record = self._records.get(key[2])
        if record is None or record.key not in wanted or key[0] != self._generation or self._is_typing(record):
            self._worker.cancelled.set()
        elif self._work_key(record) != key:
            if self._has_result(record):
                self._worker.cancelled.set()
                return
            # Give a sampled live request time to publish a real prefix. A
            # trailing debounce or cancelling every mouse event starves updates.
            # Include IPC/admission and a GUI paint in the sampling window,
            # otherwise a fast solver can still be cancelled before any prefix
            # reaches the interface during a sustained drag.
            remaining = self._worker.started_at+.350-monotonic() if self._is_sliding(record) else 0.
            if remaining > 0:
                self._timer.start(max(1, math.ceil(remaining*1000.)))
            else:
                self._worker.cancelled.set()

    def _schedule(self):
        self._cancel_unwanted_worker()
        if not self._active:
            return
        if self._worker is not None or self._scene_worker is not None:
            return
        if self._scene is None:
            if self._scene_request is not None and self._scene_error is None:
                self._timer.start(0)
            return
        pending = [record for record in self._wanted_records()
                   if not self._has_result(record) and not record.attempted and not self._is_typing(record)]
        if pending:
            earliest = min(record.ready_at for record in pending)
            self._timer.start(max(0, math.ceil((earliest-monotonic())*1000.)))
        else:
            self._timer.stop()

    def _status(self, text):
        self.status_text = text
        self.status_label.setText(text)
        self.status_changed.emit(text)

    def _row_status(self, record):
        if self._is_typing(record):
            return "Editing"
        if self._has_result(record):
            return "Ready" if record.trajectory.completed else "Limited"
        if record.error:
            return "Failed"
        if self._worker is not None and self._worker.key == self._work_key(record):
            return "Stopping" if self._worker.cancelled.is_set() else "Tracing"
        if self._scene is None:
            return "No field"
        return "Pending" if record in self._wanted_records() else "Hidden"

    def _refresh_list(self):
        was_blocked = self.electron_list.blockSignals(True)
        try:
            for key in tuple(self._items):
                if key not in self._records:
                    item = self._items.pop(key)
                    self.electron_list.takeTopLevelItem(self.electron_list.indexOfTopLevelItem(item))
            for record in self.records:
                if record.key not in self._items:
                    item = QTreeWidgetItem(self.electron_list)
                    item.setData(0, Qt.ItemDataRole.UserRole, record.key)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    self._items[record.key] = item
                item = self._items[record.key]
                item.setText(0, record.label)
                item.setText(1, self._row_status(record))
                item.setText(2, (f"{record.trajectory.positions_m[-1, 2]*1000.:.6g}"
                                if self._has_result(record) and len(record.trajectory.positions_m) else "—"))
                item.setCheckState(0, Qt.CheckState.Checked if record.checked else Qt.CheckState.Unchecked)
                item.setToolTip(0, f"{record.label}\nInclude this electron in the overlay.")
                item.setToolTip(1, record.error or (record.trajectory.reason if self._has_result(record) else ""))
                if record.colour not in self._icons:
                    pixmap = QPixmap(12, 12)
                    pixmap.fill(QColor(record.colour))
                    self._icons[record.colour] = QIcon(pixmap)
                item.setIcon(0, self._icons[record.colour])
            self.electron_list.setCurrentItem(self._items.get(self._selected_key))
        finally:
            self.electron_list.blockSignals(was_blocked)

    def _refresh(self):
        self.add_button.setEnabled(self._scene is not None)
        self.duplicate_button.setEnabled(self._scene is not None and self.selected_record is not None)
        self.remove_button.setEnabled(self.selected_record is not None)
        self._refresh_list()
        self._refresh_status()
        self.paths_changed.emit(self.visible_paths())

    def _refresh_status(self):
        record = self.selected_record
        self.status_label.setToolTip("")
        self.energy_status.setToolTip("")
        if record is not None and self._has_result(record):
            self._show_result_status(record)
            return
        self.energy_status.setText("Energy along the trajectory: pending" if record is not None else "No electron selected.")
        if self._scene_error:
            self._status(f"Electric/magnetic fields unavailable: {self._scene_error}")
        elif self._scene_worker is not None and not self._scene_worker.cancelled.is_set():
            self._status("Preparing electric and magnetic fields; reusing matching field caches…")
        elif self._scene is None:
            self._status("Virtual electrons pending captured electric and magnetic fields.")
        elif record is None:
            self._status("No electrons. Add an electron to trace from the tip.")
        elif record.error:
            self._status(f"{record.label} unavailable: {record.error}. Any previous path is retained for comparison.")
        elif self._is_typing(record):
            self._status(f"Editing {record.label}; previous path retained. Press Enter or leave the input to calculate.")
        elif record.progress_trajectory is not None:
            result = record.progress_trajectory
            same = record.progress_key[1] == record.settings
            self.energy_status.setText(f"Live preview energy: {result.kinetic_energy_ev[-1]:.9g} eV; integration incomplete")
            self._show_direction_status(result, terminal=False)
            self._status(f"{record.label} · Live preview{' · earlier settings' if not same else ''} | "
                         f"Calculated to Z {result.positions_m[-1, 2]*1000.:.6g} mm | Updating…")
        elif self._is_sliding(record):
            self._status(f"Updating {record.label} while dragging; previous path retained until new points arrive.")
        elif self._worker is not None and self._worker.key == self._work_key(record) and not self._worker.cancelled.is_set():
            self._status(f"Tracing {record.label} in electric and magnetic fields…")
        elif not self._active:
            self._status(f"{record.label} ready; select the trajectory view.")
        elif self.display_mode.currentData() == "overlay" and not record.checked:
            self._status(f"{record.label} is unchecked; check it to calculate and show its path.")
        else:
            self._status(f"{record.label} awaiting calculation.")

    @Slot()
    def _request(self):
        if not self._active:
            return
        self._cancel_unwanted_worker()
        if self._scene is None:
            self._prepare_scene()
            return
        reused = False
        for record in self._wanted_records():
            reused = self._reuse_result(record) or reused
        if reused:
            self._refresh()
        if self._worker is not None or self._scene_worker is not None:
            return
        now = monotonic()
        pending = [record for record in self._wanted_records()
                   if not self._has_result(record) and not record.attempted and not self._is_typing(record)]
        record = next((record for record in pending if record.ready_at <= now), None)
        if record is None:
            self._schedule()
            return
        record.attempted = True
        worker = _ElectronWorker(self._work_key(record), self._scene, backend=self._execution_backend)
        worker.started_at = monotonic()
        self._worker = worker
        worker.destruction_connection = self.destroyed.connect(
            lambda *_args, event=worker.cancelled: event.set())
        worker.signals.finished.connect(self._finished)
        worker.signals.progress.connect(self._progressed)
        self._refresh()
        QThreadPool.globalInstance().start(worker)

    def _prepare_scene(self):
        if (self._scene_request is None or self._scene_worker is not None
                or self._worker is not None or self._scene_error is not None):
            return
        worker = _SceneWorker(self._generation, *self._scene_request, backend=self._execution_backend)
        self._scene_worker = worker
        worker.destruction_connection = self.destroyed.connect(
            lambda *_args, event=worker.cancelled: event.set())
        worker.signals.finished.connect(self._scene_prepared)
        self._refresh()
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object, object)
    def _scene_prepared(self, generation, scene, error):
        worker = self._scene_worker
        if worker is None or worker.generation != generation:
            return
        QObject.disconnect(worker.destruction_connection)
        self._scene_worker = None
        if generation == self._generation and not worker.cancelled.is_set():
            if error:
                self._scene_error = str(error)
            elif scene is not None:
                self._install_scene(scene)
            else:
                self._scene_error = "Field preparation returned no scene."
        self._refresh()
        self._schedule()

    @Slot(object, object)
    def _progressed(self, key, result):
        worker = self._worker
        record = self._records.get(key[2])
        if (worker is None or worker.key != key or worker.cancelled.is_set() or not self._active
                or key[0] != self._generation or record is None or record not in self._wanted_records()
                or self._is_typing(record) or result.reason != "in_progress" or result.completed
                or not len(result.positions_m)):
            return
        if (record.progress_key == key and record.progress_trajectory is not None
                and result.steps <= record.progress_trajectory.steps):
            return
        record.progress_key = key
        record.progress_trajectory = result
        self._refresh()

    @Slot(object, object, object)
    def _finished(self, key, result, error):
        worker = self._worker
        if worker is None or worker.key != key:
            return
        QObject.disconnect(worker.destruction_connection)
        self._worker = None
        record = self._records.get(key[2])
        matches = (record is not None and key == self._work_key(record)
                   and self._scene is not None)
        if (not matches and record is not None and key[0] == self._generation
                and not worker.cancelled.is_set() and result is not None and error is None
                and result.reason not in {"cancelled", "in_progress"}):
            # This complete live sample may predate the latest slider value.
            # Its cache and display stay bound to its own executed parameters.
            self._store_result(record, result, settings=key[1])
        if matches:
            if worker.cancelled.is_set() or (result is not None and result.reason == "cancelled"):
                record.attempted = False
            elif error:
                record.attempted = True
                record.error = str(error)
            elif result is not None and result.reason != "in_progress":
                self._store_result(record, result)
            else:
                record.attempted = True
                record.error = "Calculation returned no trajectory."
        self._refresh()
        self._schedule()

    def _show_direction_status(self, result, *, terminal=True):
        """Report integrated 3D directions, independent of display-axis scaling."""
        directions = np.asarray(result.directions, dtype=float)
        angles = np.full(len(directions), np.nan)
        if directions.ndim == 2 and directions.shape[1] == 3:
            transverse = np.hypot(directions[:, 0], directions[:, 1])
            valid = np.isfinite(directions).all(axis=1) & (np.hypot(transverse, directions[:, 2]) > 0.)
            angles[valid] = np.arctan2(transverse[valid], directions[valid, 2])*1000.

        def formatted(value):
            return f"{value:.6g}" if np.isfinite(value) else "undefined"

        first, last = (angles[0], angles[-1]) if len(angles) else (np.nan, np.nan)
        endpoint = "final" if terminal else "current"
        heading = "Physical angle" if terminal else "Physical angle (live)"
        self.energy_status.setText(
            self.energy_status.text() +
            f"\n{heading}: {formatted(first)} → {formatted(last)} mrad")
        maximum = np.max(angles[np.isfinite(angles)]) if np.isfinite(angles).any() else np.nan
        self.energy_status.setToolTip(
            f"Initial → {endpoint} angle from +Z, in milliradians.\n"
            f"Maximum angle from +Z on this calculated path: {formatted(maximum)} mrad.\n"
            "Angles use the integrated three-dimensional direction. The diagram may enlarge "
            "the transverse axis, so its apparent slope is not the physical angle. "
            "A zero direction has no defined angle.")

    def _show_result_status(self, record):
        result = record.trajectory
        energies = result.kinetic_energy_ev
        reason = {
            "path_limit": "Requested path reached", "domain_exit": "Field validity boundary",
            "initial_outside_domain": "Initial position outside field validity",
            "step_limit": "Step budget reached; path truncated", "cancelled": "Cancelled",
            "tip_return": "Returned to tip",
            "numerical_limit": "Numerical accuracy limit; path incomplete",
        }.get(result.reason, result.reason)
        if result.reason.startswith("aperture:"):
            reason = "Aperture interception: "+result.reason.split(":", 1)[1].replace("_", " ")
        elif result.reason.startswith("hardware:"):
            reason = "Hardware interception: "+result.reason.split(":", 1)[1].replace("_", " ")
        elif result.reason.startswith("unsupported_field:"):
            reason = "Field model unavailable beyond "+result.reason.split(":", 1)[1].replace("_", " ")
        self._status(
            f"{record.label} · {reason} | K {energies[0]:.5g} → {energies[-1]:.8g} eV"
            f" | End Z {result.positions_m[-1, 2]*1000.:.6g} mm"
            f" | Path {result.path_length_m[-1]*1000.:.5g} mm"
            f" | TOF {result.time_s[-1]*1e12:.5g} ps | {result.steps} steps")
        self.energy_status.setText(
            f"Kinetic energy: {energies[0]:.9g} → {energies[-1]:.9g} eV\n"
            f"Along path: {np.min(energies):.9g}–{np.max(energies):.9g} eV")
        self._show_direction_status(result)
        self.status_label.setToolTip(
            f"Electrostatic total-energy drift (K − eφ): {result.energy_invariant_error_ev:.6g} eV\n"
            "Electric fields change kinetic energy; magnetic fields change direction. "
            "Captured hardware fields stay fixed for every independent diagnostic electron.")


TestElectronController.__test__ = False
