"""Complete TEM assembly selector."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.assembly_catalog import AssemblySelection
from temsim.component_keys import ENERGY_FILTER_INTERNAL_KEYS
from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
from temsim.gui.instrument_tree import InstrumentTree, OPTICAL_FILTERS
from temsim.operating_modes import (
    compatible_modes,
    load_operating_mode_catalog,
    mode_by_key,
)


class AssemblyPanel(QWidget):
    selection_requested = Signal(object)
    operating_mode_requested = Signal(str, str)
    direct_alignment_requested = Signal(str, float)
    component_selected = Signal(object)

    def __init__(self, catalog, selection: AssemblySelection, parent=None) -> None:
        super().__init__(parent)
        self.catalog = catalog

        box = QGroupBox("Assembly modules")
        form = QFormLayout(box)
        self.gun = QComboBox()
        self.column = QComboBox()
        self.recording = QComboBox()
        self.gun.addItems([option.name for option in catalog.guns])
        self.column.addItems([option.name for option in catalog.columns])
        self.recording.addItems(
            [option.name for option in catalog.recording_systems]
        )
        self.set_selection(selection)
        form.addRow("Gun", self.gun)
        form.addRow("Column", self.column)
        form.addRow("Recording", self.recording)
        apply_button = QPushButton("Load assembly")
        apply_button.setObjectName("loadAssemblyButton")
        apply_button.clicked.connect(self._request_selection)
        form.addRow(apply_button)

        self.operating_mode_catalog = load_operating_mode_catalog()
        mode_box = QGroupBox("Optical operating preset")
        mode_form = QFormLayout(mode_box)
        self.probe_mode = QComboBox()
        self.probe_mode.setObjectName("probeModeSelector")
        self.projector_mode = QComboBox()
        self.projector_mode.setObjectName("projectorModeSelector")
        mode_form.addRow("Probe / illumination", self.probe_mode)
        mode_form.addRow("Projection", self.projector_mode)
        self.apply_operating_mode_button = QPushButton(
            "Apply calculated lens preset"
        )
        self.apply_operating_mode_button.setObjectName(
            "applyOperatingModeButton"
        )
        mode_form.addRow(self.apply_operating_mode_button)
        self.operating_mode_status = QLabel()
        self.operating_mode_status.setObjectName("operatingModeStatus")
        self.operating_mode_status.setWordWrap(True)
        self.operating_mode_status.setStyleSheet(
            "color: #475569; font-weight: 600;"
        )
        mode_form.addRow(self.operating_mode_status)
        self.load_operating_modes(
            selection, "nano_probe", "diffraction"
        )

        self._assembly = None
        self._runtime_targets = {}
        self._suppress_forward_selection = False

        optical_page = QWidget()
        optical_layout = QVBoxLayout(optical_page)
        optical_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.optical_filter = QComboBox()
        self.optical_filter.setObjectName("opticalComponentFilter")
        for key, label in OPTICAL_FILTERS:
            self.optical_filter.addItem(label, key)
        self.optical_filter.setToolTip(
            "Show only active components of one optical function; "
            "corrector elements remain in their own group"
        )
        filter_row.addWidget(self.optical_filter, 1)
        optical_layout.addLayout(filter_row)
        self.tree = InstrumentTree()
        optical_layout.addWidget(self.tree, 1)

        mechanical_page = QWidget()
        mechanical_layout = QVBoxLayout(mechanical_page)
        mechanical_layout.setContentsMargins(0, 0, 0, 0)
        mechanical_hint = QLabel(
            "TOML/layout parts without an independent optical runtime object"
        )
        mechanical_hint.setWordWrap(True)
        mechanical_hint.setStyleSheet("color: #64748b; font-weight: 600;")
        mechanical_layout.addWidget(mechanical_hint)
        self.mechanical_tree = InstrumentTree()
        self.mechanical_tree.setObjectName("mechanicalInstrumentTree")
        mechanical_layout.addWidget(self.mechanical_tree, 1)

        self.component_pages = QTabWidget()
        self.component_pages.setObjectName("componentNavigationPages")
        self.component_pages.addTab(optical_page, "Optical")
        self.component_pages.addTab(mechanical_page, "Mechanical")
        self.direct_alignment_panel = DirectAlignmentPanel(
            self.operating_mode_catalog, self
        )
        self.direct_alignment = self.direct_alignment_panel
        self.component_pages.addTab(
            self.direct_alignment_panel, "Direct Alignment"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        layout.addWidget(mode_box)
        layout.addWidget(self.component_pages, 1)

        self.optical_filter.currentIndexChanged.connect(
            self._reload_optical_tree
        )
        self.component_pages.currentChanged.connect(self._page_changed)
        self.probe_mode.currentIndexChanged.connect(
            self._update_operating_mode_description
        )
        self.projector_mode.currentIndexChanged.connect(
            self._update_operating_mode_description
        )
        self.apply_operating_mode_button.clicked.connect(
            self._request_operating_mode
        )
        self.tree.component_selected.connect(
            lambda selection: self._forward_selection(0, selection)
        )
        self.mechanical_tree.component_selected.connect(
            lambda selection: self._forward_selection(1, selection)
        )
        self.direct_alignment_panel.adjustment_requested.connect(
            self.direct_alignment_requested.emit
        )

    def current_selection(self) -> AssemblySelection:
        return AssemblySelection(
            gun=self.gun.currentText(),
            column=self.column.currentText(),
            recording=self.recording.currentText(),
        )

    def set_selection(self, selection: AssemblySelection) -> None:
        self.gun.setCurrentText(selection.gun)
        self.column.setCurrentText(selection.column)
        self.recording.setCurrentText(selection.recording)

    def reload_catalog(self, catalog, selection: AssemblySelection) -> None:
        self.catalog = catalog
        for combo, options in (
            (self.gun, catalog.guns),
            (self.column, catalog.columns),
            (self.recording, catalog.recording_systems),
        ):
            combo.clear()
            combo.addItems([option.name for option in options])
        self.set_selection(selection)
        self.operating_mode_catalog = load_operating_mode_catalog()
        self.direct_alignment_panel.set_catalog(self.operating_mode_catalog)
        self.load_operating_modes(selection)

    def load_operating_modes(
        self,
        selection: AssemblySelection,
        condenser_key: str | None = None,
        projector_key: str | None = None,
    ) -> None:
        """Populate only modes compatible with the selected assembly."""

        condenser_key = condenser_key or self.probe_mode.currentData()
        projector_key = projector_key or self.projector_mode.currentData()
        condenser_modes = compatible_modes(
            "condenser",
            selection.column,
            selection.recording,
            self.operating_mode_catalog,
        )
        projector_modes = compatible_modes(
            "projector",
            selection.column,
            selection.recording,
            self.operating_mode_catalog,
        )
        for combo, modes, preferred in (
            (self.probe_mode, condenser_modes, condenser_key),
            (self.projector_mode, projector_modes, projector_key),
        ):
            blocked = combo.blockSignals(True)
            combo.clear()
            for mode in modes:
                combo.addItem(mode.name, mode.key)
            index = combo.findData(preferred)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(blocked)
        self.apply_operating_mode_button.setEnabled(
            self.probe_mode.count() > 0 and self.projector_mode.count() > 0
        )
        self._update_operating_mode_description()

    def set_operating_mode_keys(
        self, condenser_key: str, projector_key: str
    ) -> None:
        for combo, key in (
            (self.probe_mode, condenser_key),
            (self.projector_mode, projector_key),
        ):
            index = combo.findData(key)
            if index >= 0:
                combo.setCurrentIndex(index)

    def set_operating_mode_status(self, text: str) -> None:
        self.operating_mode_status.setText(text)

    def set_direct_alignment_state(self, state) -> None:
        available_modes = {
            str(combo.itemData(index))
            for combo in (self.probe_mode, self.projector_mode)
            for index in range(combo.count())
            if combo.itemData(index) is not None
        }
        self.direct_alignment_panel.set_state(state, available_modes)

    def update_direct_alignment_metrics(self, metrics) -> None:
        self.direct_alignment_panel.update_metrics(metrics)

    def show_direct_alignment_result(self, result) -> None:
        self.direct_alignment_panel.show_result(result)

    def set_direct_alignment_busy(self, key: str | None) -> None:
        self.direct_alignment_panel.set_busy(key)

    def set_direct_alignment_message(
        self, message: str, *, error: bool = False
    ) -> None:
        self.direct_alignment_panel.show_status_message(
            message, error=error
        )

    def _update_operating_mode_description(self, _index: int = -1) -> None:
        condenser_key = self.probe_mode.currentData()
        projector_key = self.projector_mode.currentData()
        if condenser_key is None or projector_key is None:
            self.operating_mode_status.setText(
                "No calibrated preset is available for this assembly."
            )
            return
        condenser = mode_by_key(
            condenser_key, self.operating_mode_catalog
        )
        projector = mode_by_key(
            projector_key, self.operating_mode_catalog
        )
        angle = condenser.targets.get(
            "achieved_convergence_sem_angle_mrad"
        )
        plane = projector.targets.get("conjugate_plane", "")
        plane_label = {
            "objective_image_plane": "objective image plane",
            "objective_back_focal_plane": "objective back focal plane",
        }.get(str(plane), str(plane))
        if condenser.key == "nano_probe":
            illumination_note = (
                "Convergence: C2 + C3 + C2 aperture; "
                "STEM focus: Objective lens."
            )
        else:
            illumination_note = (
                "Microprobe illumination is calibrated quasi-parallel at "
                "the sample."
            )
        devices = condenser.devices
        c2 = float(devices["condenser_lens_2"]["percent"])
        c3 = float(devices["condenser_lens_3"]["percent"])
        objective = float(devices["objective_lens"]["percent"])
        aperture_um = float(
            condenser.apertures["condenser_aperture_2"]["radius_mm"]
        ) * 1000.0
        self.operating_mode_status.setText(
            f"Preset reference sample semi-angle: {float(angle):.3f} mrad. "
            f"C2 {c2:.2f}%, C3 {c3:.2f}%, C2 aperture "
            f"{aperture_um:.0f} µm, Objective focus {objective:.1f}%. "
            f"{illumination_note} Projection conjugate: {plane_label}. "
            "Apply the preset, use Direct Alignment for coupled user-level "
            "adjustments, or edit individual values under Optical > "
            "Lenses/Correctors."
        )

    def _request_operating_mode(self) -> None:
        condenser_key = self.probe_mode.currentData()
        projector_key = self.projector_mode.currentData()
        if condenser_key is not None and projector_key is not None:
            self.operating_mode_requested.emit(
                str(condenser_key), str(projector_key)
            )

    def load_assembly(self, assembly, runtime_targets=None) -> None:
        self._assembly = assembly
        self._runtime_targets = dict(runtime_targets or {})
        self.tree.load_optical(
            assembly,
            self._runtime_targets,
            self.optical_filter.currentData() or "all",
            select_first=True,
        )
        self.mechanical_tree.load_mechanical(
            assembly,
            self._runtime_targets,
            select_first=False,
        )

    def _reload_optical_tree(self, _index: int = -1) -> None:
        if self._assembly is None:
            return
        previous_key = self.tree.current_key()
        self.tree.load_optical(
            self._assembly,
            self._runtime_targets,
            self.optical_filter.currentData() or "all",
            select_first=False,
        )
        if previous_key is None or not self.tree.select_key(previous_key):
            self.tree.select_first()

    def _page_changed(self, index: int) -> None:
        if self._suppress_forward_selection:
            return
        if index == 2:
            return
        if index not in (0, 1):
            return
        tree = self.tree if index == 0 else self.mechanical_tree
        if tree.currentItem() is None:
            tree.select_first()
        else:
            selection = tree.currentItem().data(
                0, Qt.ItemDataRole.UserRole
            )
            if selection is not None:
                self.component_selected.emit(selection)

    def _forward_selection(self, page_index: int, selection) -> None:
        if (
            not self._suppress_forward_selection
            and self.component_pages.currentIndex() == page_index
        ):
            self.component_selected.emit(selection)

    def _optical_category_for_key(self, key: str) -> str | None:
        if key == "energy_filter" or key in ENERGY_FILTER_INTERNAL_KEYS:
            return "energy_filter"
        if key in {"simulation", "electron_gun"}:
            return "other"
        if self._assembly is None:
            return None
        part = next(
            (
                candidate for candidate in self._assembly.parts
                if candidate.key == key
            ),
            None,
        )
        target = self._runtime_targets.get(key)
        if (
            part is None
            or target is None
            or not InstrumentTree._is_optical_part(
                part, self._runtime_targets
            )
        ):
            return None
        return InstrumentTree.optical_category(part, target)

    def _emit_current_tree_selection(self, tree: InstrumentTree) -> None:
        current = tree.currentItem()
        if current is None:
            return
        selection = current.data(0, Qt.ItemDataRole.UserRole)
        if selection is not None:
            self.component_selected.emit(selection)

    def select_key(self, key: str) -> bool:
        """Open, filter and emit the component selected from a plot."""

        key = str(key)
        selected_tree = None
        selected_page = -1
        self._suppress_forward_selection = True
        try:
            category = self._optical_category_for_key(key)
            if category is not None:
                index = self.optical_filter.findData(category)
                if (
                    index >= 0
                    and self.optical_filter.currentIndex() != index
                ):
                    self.optical_filter.setCurrentIndex(index)
                if self.tree.select_key(key):
                    selected_tree = self.tree
                    selected_page = 0
            if selected_tree is None and self.tree.select_key(key):
                selected_tree = self.tree
                selected_page = 0
            if (
                selected_tree is None
                and self.optical_filter.currentData() != "all"
            ):
                all_index = self.optical_filter.findData("all")
                if all_index >= 0:
                    self.optical_filter.setCurrentIndex(all_index)
                if self.tree.select_key(key):
                    selected_tree = self.tree
                    selected_page = 0
            if (
                selected_tree is None
                and self.mechanical_tree.select_key(key)
            ):
                selected_tree = self.mechanical_tree
                selected_page = 1
            if selected_tree is not None:
                self.component_pages.setCurrentIndex(selected_page)
        finally:
            self._suppress_forward_selection = False
        if selected_tree is None:
            return False
        self._emit_current_tree_selection(selected_tree)
        return True

    def _request_selection(self) -> None:
        self.selection_requested.emit(self.current_selection())
