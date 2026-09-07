"""Display-only EDS line annotations backed by the shared offline line library."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from temsim.detector.eds_line_library import (
    aggregate_simulated_lines,
    element_lines,
    library_elements,
    library_provenance,
)
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox


class EDSPeakLabels(QWidget):
    """A small toolbar and overlay; changing it never requests a calculation.

    Simulated annotations come exclusively from the displayed result's positive
    detected line contributions, not from the currently edited sample. Library
    references are explicitly labelled as references, never as detections.
    """

    annotations_changed = Signal()

    def __init__(self, plot, parent=None):
        super().__init__(parent)
        self.setObjectName("edsPeakLabels")
        self.plot = plot
        self._generated = ()
        self._candidates = ()
        self._items = []
        self._energy_kev = np.empty(0)
        self._resolution_ev = 0.0
        self._response_known = False
        self._bin_ev = 0.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._draw)

        self.enabled = QCheckBox("Peak labels")
        self.enabled.setObjectName("edsShowPeakLabels")
        self.enabled.setChecked(True)
        self.element = QComboBox()
        self.element.setObjectName("edsPeakLibraryElement")
        self.element.addItem("Simulated lines", 0)
        for z, symbol in library_elements():
            self.element.addItem(f"Library: {symbol} (Z={z})", z)
        self.element.setToolTip(
            "Simulated lines identify positive contributions in the displayed result. "
            "Library entries are reference energies, not detected elements."
        )
        self.status = QLabel("No spectrum")
        self.status.setObjectName("edsPeakLibraryStatus")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setToolTip(library_provenance())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enabled)
        layout.addWidget(self.element)
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.enabled.toggled.connect(self._annotations_changed)
        self.element.currentIndexChanged.connect(self._select)
        plot.getViewBox().sigRangeChanged.connect(self._schedule)

    def _schedule(self, *_args):
        self._timer.start(16)

    def _annotations_changed(self, *_args):
        self.annotations_changed.emit()
        self._schedule()

    def set_spectrum(self, spectrum):
        self._remove_items()
        self._generated = aggregate_simulated_lines(getattr(spectrum, "lines", ()))
        self._energy_kev = np.asarray(spectrum.energy_bin_centres_ev, dtype=float) * 1e-3
        metrics = getattr(spectrum, "metrics", {}) or {}
        self._resolution_ev = max(0.0, float(metrics.get("energy_resolution_fwhm_ev", 0.0)))
        self._response_known = "energy_resolution_fwhm_ev" in metrics
        self._bin_ev = (float(np.median(np.diff(self._energy_kev))) * 1000
                        if self._energy_kev.size > 1 else 0.0)
        self._select()

    def clear(self):
        self._timer.stop()
        self._remove_items()
        self._generated = self._candidates = ()
        self._energy_kev = np.empty(0)
        self._resolution_ev = self._bin_ev = 0.0
        self._response_known = False
        self.status.setText("No spectrum")

    def _select(self, *_args):
        z = self.element.currentData()
        self._candidates = element_lines(z) if z else self._generated
        response = (f"{self._resolution_ev:g} eV FWHM" if self._resolution_ev > 0 else "ideal response")
        if not self._response_known:
            response = "Resolution unavailable"
        if z:
            detail = response if self._energy_kev.size else "No spectrum"
            if not self._candidates:
                detail = "No tabulated lines"
            self.status.setText(f"Reference only | {detail}")
        elif self._energy_kev.size:
            self.status.setText(f"{len(self._generated)} simulated lines | {response}")
        else:
            self.status.setText("No spectrum")
        self._annotations_changed()

    def _remove_items(self):
        for item in self._items:
            self.plot.removeItem(item)
        self._items.clear()

    def nearby_text(self, energy_kev):
        """Describe nearby known lines, not an automatic composition fit."""
        if not self.enabled.isChecked():
            return ""
        tolerance_ev = max(5.0, self._bin_ev * 0.5, self._resolution_ev * 0.5)
        nearby = sorted(
            (line for line in self._candidates
             if abs(line.energy_ev - 1000 * energy_kev) <= tolerance_ev),
            key=lambda line: abs(line.energy_ev - 1000 * energy_kev),
        )
        if not nearby:
            return ""
        prefix = "Reference" if self.element.currentData() else "Simulated"
        description = "; ".join(f"{line.label} {line.energy_ev / 1000:.5g} keV" for line in nearby[:3])
        if len(nearby) > 3:
            description += f"; +{len(nearby) - 3} lines"
        return f"{prefix}: {description}"

    def _draw(self):
        self._remove_items()
        if not self.enabled.isChecked():
            return
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or x1 <= x0 or y1 <= y0:
            return
        reference = bool(self.element.currentData())
        score = (lambda line: line.relative_rate) if reference else (lambda line: line.expected_counts)
        visible = [line for line in self._candidates if x0 <= line.energy_ev / 1000 <= x1]
        if not visible:
            return
        maximum = max(score(line) for line in visible)
        # Bound visual clutter only; weak lines remain available in hover text.
        visible = sorted((line for line in visible if score(line) >= maximum * 0.01),
                         key=score, reverse=True)
        chosen = []
        separation_kev = max(self._bin_ev, self._resolution_ev * 0.5, 1.0) * 1e-3
        for line in visible:
            if any(line.atomic_number == old.atomic_number
                   and abs(line.energy_ev - old.energy_ev) * 1e-3 < separation_kev for old in chosen):
                continue
            chosen.append(line)
            if len(chosen) == 16:
                break
        colour = "#fbbf24" if reference else "#a7f3d0"
        label_rows = [[], [], []]
        for index, line in enumerate(sorted(chosen, key=lambda item: item.energy_ev)):
            x = line.energy_ev * 1e-3
            marker = pg.InfiniteLine(x, angle=90, movable=False,
                                     pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DotLine))
            marker.setOpacity(0.45)
            self.plot.addItem(marker, ignoreBounds=True)
            self._items.append(marker)
            # IUPAC transition stays in hover/tooltip; the short label fits the
            # plot without adding a permanent, verbose explanation panel.
            short = line.label.split(" (")[0]
            text = pg.TextItem(("Ref " if reference else "") + short,
                               color=colour, anchor=(0.5, 0), fill="#111827")
            # Keep edge labels readable without moving their line energy or
            # expanding the user's plot range.
            width = max(1.0, self.plot.getViewBox().width())
            half_label = 0.5 * text.boundingRect().width() + 4.0
            pixel_x = (x - x0) / (x1 - x0) * width
            if pixel_x < half_label:
                text.setAnchor((0, 0))
            elif width - pixel_x < half_label:
                text.setAnchor((1, 0))
            left = pixel_x - text.anchor.x() * text.boundingRect().width()
            right = left + text.boundingRect().width()
            for offset in range(3):
                row = (index + offset) % 3
                if all(right + 6 < start or left - 6 > end for start, end in label_rows[row]):
                    label_rows[row].append((left, right))
                    break
            else:
                # Dense clusters retain their marker and hover information.
                continue
            y = y1 - (0.06 + 0.12 * row) * (y1 - y0)
            tooltip = f"{line.label}\n{line.energy_ev / 1000:.6g} keV\n"
            if reference:
                tooltip += "Library reference; not evidence of a detected element."
            else:
                tooltip += (f"Expected contribution: {line.expected_counts:.6g} counts\n"
                            f"Sources: {', '.join(line.source_keys)}")
            text.setToolTip(tooltip)
            text.setPos(x, y)
            self.plot.addItem(text, ignoreBounds=True)
            self._items.append(text)
