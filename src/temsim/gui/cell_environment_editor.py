"""Window controls and a specimen-relative cell section (+Z down)."""
from copy import deepcopy
import tomllib

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QLineEdit, QDoubleSpinBox, QLabel, QGraphicsRectItem)

from temsim.paths import CONFIG_ROOT
from temsim.vacuum import CellWindow


def readable_label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class WindowEditor(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self._window = CellWindow()
        with (CONFIG_ROOT / "environments" / "cell_window_materials.toml").open("rb") as stream:
            self.presets = tomllib.load(stream)["materials"]
        form = QFormLayout(self)
        self.material = QComboBox()
        self.material.addItem("Custom material", None)
        for preset in self.presets:
            self.material.addItem(preset["name"], preset)
        self.formula = QLineEdit()
        self.density, self.thickness = QDoubleSpinBox(), QDoubleSpinBox()
        self.density.setRange(.000001, 1e6)
        self.density.setDecimals(6)
        self.thickness.setRange(0, 1e9)
        self.thickness.setDecimals(6)
        self.thickness.setSingleStep(1)
        self.thickness.setToolTip("Thickness along Z. Zero explicitly means no window (also used by historical windowless maps).")
        self.density.setToolTip("User-supplied mass density. Presets are illustrative; graphene uses a graphite-equivalent areal density, not a crystalline wave model.")
        for title, widget in (("Material preset", self.material), ("Formula", self.formula),
                              ("Density (kg/m³)", self.density), ("Thickness (nm)", self.thickness)):
            form.addRow(title, widget)
        self.material.activated.connect(self._choose_preset)

    def _choose_preset(self, index):
        preset = self.material.itemData(index)
        if preset:
            self.formula.setText(preset["formula"])
            self.density.setValue(preset["density_kg_m3"])
            self.thickness.setValue(preset["thickness_nm"])
            self._window.material = preset["name"]
            self._window.reference = preset["reference"]
            self._window.medium.mixture_mole_fractions = {}
            self._window.medium.removal_cross_section_m2 = 0
            self._window.medium.removal_reference = ""
        else:
            self._window.material = "Custom material"
            self._window.reference = "User-defined material"

    def set_window(self, window):
        self._window = deepcopy(window)
        self.formula.setText(window.medium.formula)
        self.density.setValue(window.medium.density_kg_m3)
        self.thickness.setValue(window.thickness_nm)
        self.material.setCurrentIndex(max(0, self.material.findText(window.material)))
        self.setToolTip(window.reference)

    def window(self):
        window = deepcopy(self._window)
        formula = self.formula.text().strip()
        if formula != window.medium.formula or self.density.value() != window.medium.density_kg_m3:
            window.material = f"Custom {formula}"
            window.reference = "User-defined material"
        if formula != window.medium.formula:
            window.medium.mixture_mole_fractions = {}
        window.medium.formula = formula
        window.medium.density_kg_m3 = self.density.value()
        window.thickness_nm = self.thickness.value()
        return window.validate()


class CellChamberView(QWidget):
    """Physical X/Z positions, independent axis scales; never edits Sample."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = readable_label()
        layout.addWidget(self.summary)
        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setMinimumHeight(240)
        self.plot.setLabel("left", "Z from sample centre (+Z down)", units="nm")
        self.plot.setLabel("bottom", "X", units="nm")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getViewBox().invertY(True)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        layout.addWidget(self.plot)
        self.details = readable_label()
        layout.addWidget(self.details)
        self.layers = {}

    def set_state(self, state):
        cell, sample = state.vacuum_map.cell, state.sample
        self.plot.clear()
        self.layers = {}
        half = cell.length_mm*5e5
        centre = cell.offset_z_mm*1e6
        a, b = centre-half, centre+half
        cx, radius = cell.offset_x_mm*1e6, cell.diameter_mm*5e5
        up, down = cell.upstream_window.thickness_nm, cell.downstream_window.thickness_nm
        for key, lo, hi, colour in (("interior", a, b, "#164e63"),
                                  ("upstream", a-up, a, "#67e8f9"),
                                  ("downstream", b, b+down, "#a78bfa")):
            self.layers[key] = (lo, hi)
            rect = QGraphicsRectItem(QRectF(cx-radius, lo, 2*radius, hi-lo))
            rect.setPen(pg.mkPen(colour))
            rect.setBrush(pg.mkBrush(colour))
            rect.setToolTip(f"{key}: Z {lo:g} to {hi:g} nm from Sample centre")
            self.plot.addItem(rect)
        from temsim.specimen.source import specimen_is_vacuum
        present = sample.inserted and not specimen_is_vacuum(sample)
        x = sample.centre_x_nm
        width, thickness = sample.size_x_nm, sample.thickness_nm
        # X-Z projection of the complete finite envelope, not the cropped atoms.
        if present:
            rect = QGraphicsRectItem(QRectF(x-width/2, -thickness/2, width, thickness))
            rect.setPen(pg.mkPen("#fbbf24", width=2))
            rect.setBrush(pg.mkBrush(251, 191, 36, 100))
            self.plot.addItem(rect)
        self.sample_marker = pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#fbbf24", width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.sample_marker)
        self.plot.disableAutoRange()
        span_lo, span_hi = min(a-up, -thickness/2), max(b+down, thickness/2)
        self.plot.setXRange(cx-1.1*radius, cx+1.1*radius, padding=0)
        self.plot.setYRange(span_lo, span_hi, padding=.1)
        self.summary.setText(f"Sample chamber · Sample Z {sample.z_mm:.9g} mm · "
                             f"{'Cell inserted' if cell.inserted else 'Cell draft (not inserted)'} · "
                             f"{'Transport on' if state.vacuum_map.enabled else 'Transport off'}")
        self.details.setText(f"Cyan: upstream {up:g} nm | Purple: downstream {down:g} nm | Gap {2*half:g} nm\n"
            f"Gold: Sample {'envelope' if present else 'reference plane (vacuum / retracted)'} · "
            f"X {sample.centre_x_nm:g}, Y {sample.centre_y_nm:g} nm · Thickness {thickness:g} nm\n"
            f"Cell centre relative to Sample: X {cx-sample.centre_x_nm:g}, "
            f"Y {cell.offset_y_mm*1e6-sample.centre_y_nm:g}, Z {centre:g} nm. X/Z scales are independent.")
        self.details.setToolTip("Sample geometry comes from Sample, not from this editor. The view shows the cell aperture; a larger or displaced sample can extend outside it. Windows are flat parallel membranes; frames, seals and pressure bulging are not modeled.")
