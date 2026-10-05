"""Display-only markers for a captured conjugate-plane search, in axial mm."""
from __future__ import annotations

from PySide6.QtCore import Qt
import pyqtgraph as pg
from temsim.gui.plane_equations import plane_equation_tooltip


class ConjugatePlaneOverlay:
    """Keep theoretical conjugates separate from physical ray crossovers.

    Labels do not participate in plot fitting and never initiate calculations.
    Amber markers deliberately distinguish approximate and line-focus minima.
    """

    def __init__(self, plot):
        self.plot = plot
        self.search = None
        self.items = []
        self._visible = False

    def set_search(self, search):
        self.clear()
        self.search = search
        if search is None:
            return
        self._add(search.reference_z_mm, "Reference", "#a78bfa",
                  plane_equation_tooltip("reference", reference="pinned reference plane", conjugate_search=True))
        for index, candidate in enumerate(search.candidates, 1):
            kind = candidate.kind
            tooltip = plane_equation_tooltip(kind, reference="pinned reference plane", conjugate_search=True)
            label = {"image": "Conj", "line_focus": "Line", "approximate": "Approx"}.get(kind, kind)
            self._add(candidate.z_mm, f"{label} {index}",
                      "#4ade80" if kind == "image" else "#fbbf24", tooltip)
        self.refresh()

    def _add(self, z_mm, label, colour, tooltip):
        line = pg.InfiniteLine(
            pos=z_mm, angle=90, movable=False,
            pen=pg.mkPen(colour, width=1.3, style=Qt.PenStyle.DashLine),
            label=label, labelOpts={"position": 0.84, "color": colour,
                                    "rotateAxis": (1, 0)},
        )
        line.setZValue(39)
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        line.setAcceptHoverEvents(True)
        line.setToolTip(tooltip)
        line.label.setToolTip(tooltip)
        self.items.append(line)

    def refresh(self):
        """Reattach markers after a normal Ray Diagram scene reset."""
        for item in self.items:
            if item.scene() is None:
                self.plot.addItem(item, ignoreBounds=True)
            item.setVisible(self._visible)

    def setVisible(self, visible):
        self._visible = bool(visible)
        self.refresh()

    def clear(self):
        for item in self.items:
            self.plot.removeItem(item)
        self.items.clear()
        self.search = None
