"""Vacuum intervals on Physical Layout's shared, millimetre-valued Z axis."""
from dataclasses import dataclass
import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainterPath
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem


@dataclass
class _IntervalLabel:
    text: pg.TextItem
    rectangle: QGraphicsRectItem
    leader: QGraphicsPathItem
    base_rect: QRectF
    band: str
    key: str | None


class VacuumAxialView(pg.PlotWidget):
    selected = Signal(str)

    def __init__(self):
        super().__init__(background="#050816")
        self.rows, self.modules, self.current = (), (), ""
        self.boxes = []
        self._items, self._labels = [], []
        self._physical_plot = None
        self._syncing = False
        self._laying_out = False
        self._label_timer = QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.timeout.connect(self._layout_labels)
        self.component = None
        self.sample_z_mm = None
        self.sample_marker = None
        self.setMinimumHeight(210)
        self.setLabel("bottom", "Axial position", units="mm")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").setTicks([[(2.45, "Assembly"), (1.35, "Vacuum"), (.35, "Cell")]])
        self.getAxis("left").setWidth(70)
        self.showGrid(x=True, y=False, alpha=.16)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.disableAutoRange()
        self.setYRange(0, 3, padding=0)
        self.setMouseEnabled(x=True, y=False)
        self.scene().sigMouseClicked.connect(self._clicked)
        self.getViewBox().sigXRangeChanged.connect(self._vacuum_range_changed)
        # Finish the graphics layout's resize before adjusting annotation space.
        self.getViewBox().sigResized.connect(lambda *_: self._label_timer.start(0))
        self.setToolTip("Z range and scale are shared with Physical Layout. Click a vacuum region to edit its pressure; drag or scroll to change the shared Z view.")

    def bind_physical_plot(self, plot):
        self._physical_plot = plot
        # Fixed identical gutters keep millimetres per screen pixel identical.
        plot.getAxis("left").setWidth(70)
        plot.getViewBox().sigXRangeChanged.connect(self._physical_range_changed)
        self._physical_range_changed()

    def _physical_range_changed(self, *_):
        if self._syncing or self._physical_plot is None:
            return
        self._syncing = True
        try:
            self.setXRange(*self._physical_plot.getViewBox().viewRange()[0], padding=0)
        finally:
            self._syncing = False
        self._layout_labels()

    def _vacuum_range_changed(self, *_):
        if not self._syncing and self._physical_plot is not None:
            self._syncing = True
            try:
                self._physical_plot.setXRange(*self.getViewBox().viewRange()[0], padding=0)
            finally:
                self._syncing = False
        self._layout_labels()

    def showEvent(self, event):
        super().showEvent(event)
        self._physical_range_changed()

    def set_regions(self, rows, modules, current=""):
        self.rows, self.modules, self.current = tuple(rows), tuple(modules), current
        self.redraw()

    def _rectangle(self, start, end, bottom, height, colour, label, key=None, dashed=False, band="assembly"):
        box = QRectF(start, bottom, end-start, height)
        item = QGraphicsRectItem(box)
        selected = key == self.current
        item.setPen(pg.mkPen("#fbbf24" if selected else "#64748b", width=2 if selected else 1,
                            style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
        item.setBrush(pg.mkBrush(colour))
        item.setToolTip(f"{label}\nZ {start:.9g} to {end:.9g} mm")
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.addItem(item, ignoreBounds=True)
        self._items.append(item)
        if key:
            self.boxes.append((box, key))
        text = pg.TextItem(label, color="#f8fafc", anchor=(.5, .5))
        text.setZValue(20)
        text.setToolTip(item.toolTip())
        text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        text.setPos((start+end)/2, bottom+height/2)
        self.addItem(text, ignoreBounds=True)
        self._items.append(text)
        leader = QGraphicsPathItem()
        leader.setPen(pg.mkPen("#fbbf24" if selected else "#94a3b8", width=1,
                              style=Qt.PenStyle.DashLine))
        leader.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        leader.setZValue(10)
        self.addItem(leader, ignoreBounds=True)
        self._items.append(leader)
        self._labels.append(_IntervalLabel(text, item, leader, box, band, key))

    def redraw(self):
        for item in self._items:
            self.removeItem(item)
        self._items, self._labels, self.boxes = [], [], []
        lanes = []
        for i, module in enumerate(self.modules):
            # Functional extents can overlap in Z; stack them, never shorten or
            # re-scale a section to suggest separate pressure compartments.
            lane = next((n for n, end in enumerate(lanes) if end <= module.start_z_mm), len(lanes))
            if lane == len(lanes):
                lanes.append(module.end_z_mm)
            else:
                lanes[lane] = module.end_z_mm
            self._rectangle(module.start_z_mm, module.end_z_mm, 2.1+.8*lane, .7,
                            "#23364a" if i % 2 == 0 else "#35425a",
                            module.name)
        for row in self.rows:
            cell = row.radius_mm is not None
            transition = row.end_medium is not None
            self._rectangle(row.start_z_mm, row.end_z_mm, .1 if cell else 1.0,
                            .5 if cell else .7,
                            "#475569" if transition else ("#785420" if cell else "#155e75"),
                            ("Cell pressure gradient" if cell else "Linear transition") if transition else row.name,
                            row.key, transition, band="cell" if cell else "vacuum")
        if self.sample_z_mm is not None:
            self.sample_marker = pg.InfiniteLine(self.sample_z_mm, pen=pg.mkPen("#fbbf24", width=2))
            self.sample_marker.setToolTip(f"Sample centre Z {self.sample_z_mm:.9g} mm; geometry is defined in Sample.")
            self.addItem(self.sample_marker, ignoreBounds=True)
            self._items.append(self.sample_marker)
            label = pg.TextItem(f"Sample · {self.sample_z_mm:.9g} mm", color="#fbbf24", anchor=(.5, 0))
            label.setPos(self.sample_z_mm, .95)
            self.addItem(label, ignoreBounds=True)
            self._items.append(label)
        if self.component is not None:
            part = self.component
            band = pg.LinearRegionItem((part.start_z_mm, part.end_z_mm), movable=False,
                                       brush=pg.mkBrush(251, 191, 36, 35), pen=pg.mkPen(None))
            band.setZValue(5)
            self.addItem(band, ignoreBounds=True)
            self._items.append(band)
            line = pg.InfiniteLine(part.center_z_mm, pen=pg.mkPen("#fbbf24", width=1))
            line.setZValue(6)
            self.addItem(line, ignoreBounds=True)
            self._items.append(line)
        self._layout_labels()

    def _layout_labels(self, *_):
        if self._laying_out:
            return
        self._laying_out = True
        try:
            self._place_labels()
        finally:
            self._laying_out = False

    def _place_labels(self):
        """Pack callouts in screen space; never expand or fit the shared Z range.

        Each band gets its own annotation rows, so leaders do not run through
        a different band. Only the nonphysical vertical spacing can grow.
        """
        vb = self.getViewBox()
        viewport = vb.sceneBoundingRect()
        left, right = viewport.left()+6, viewport.right()-6
        if right-left < 20:
            return
        z_min, z_max = vb.viewRange()[0]
        # These are annotation units, not physical distances. A minimum height
        # guarantees that text rows stay readable when the splitter is reduced.
        pixels_per_row_unit = 40
        callouts = {band: [] for band in ("cell", "vacuum", "assembly")}
        inline = []
        for entry in self._labels:
            label, rect = entry.text, entry.base_rect
            entry.leader.hide()
            visible = rect.right() >= z_min and rect.left() <= z_max
            label.setVisible(visible)
            if not visible:
                continue
            label.setTextWidth(-1)
            x1 = max(left, vb.mapViewToScene(pg.Point(rect.left(), 0)).x())
            x2 = min(right, vb.mapViewToScene(pg.Point(rect.right(), 0)).x())
            target = (x1+x2)/2
            if label.boundingRect().width()+10 <= x2-x1:
                label.fill = pg.mkBrush(None)
                label.setColor("#fbbf24" if entry.key == self.current else "#f8fafc")
                inline.append((entry, target))
                continue
            if label.boundingRect().width() > min(280, right-left):
                label.setTextWidth(min(280, right-left))
            width, height = label.boundingRect().width(), label.boundingRect().height()
            callouts[entry.band].append((target, width, height, entry))

        packed, heights = {}, {}
        for band, entries in callouts.items():
            rows = []
            for target, width, height, entry in sorted(entries, key=lambda item: item[0]):
                desired = max(left, min(target-width/2, right-width))
                for row in rows:
                    x = max(desired, row["right"]+8)
                    if x+width <= right:
                        break
                else:
                    row = {"right": left-8, "height": 0, "entries": []}
                    rows.append(row)
                    x = desired
                row["right"] = x+width
                row["height"] = max(row["height"], height)
                row["entries"].append((entry, x+width/2, target, height))
            offset = 8 if rows else 0
            packed[band] = []
            for row in rows:
                for entry, x, target, height in row["entries"]:
                    packed[band].append((entry, x, target, (offset+row["height"]/2)/pixels_per_row_unit, height))
                offset += row["height"]+6
            heights[band] = offset/pixels_per_row_unit

        assembly_top = max((e.base_rect.bottom() for e in self._labels if e.band == "assembly"), default=2.8)
        assembly_top += heights["vacuum"]
        y_min = min(0, .1-heights["cell"]-.15)
        y_max = max(3, assembly_top+heights["assembly"]+.2)
        self.setMinimumHeight(max(210, math.ceil((y_max-y_min)*pixels_per_row_unit+45)))
        self.setYRange(y_min, y_max, padding=0)
        self.getAxis("left").setTicks([[(2.45+heights["vacuum"], "Assembly"), (1.35, "Vacuum"), (.35, "Cell")]])
        for entry in self._labels:
            rect = QRectF(entry.base_rect)
            if entry.band == "assembly":
                rect.translate(0, heights["vacuum"])
            entry.rectangle.setRect(rect)
        for entry, x in inline:
            entry.text.setPos(vb.mapSceneToView(pg.Point(x, 0)).x(), entry.rectangle.rect().center().y())
        for band, entries in packed.items():
            base = {"cell": .1, "vacuum": 1.7, "assembly": assembly_top}[band]
            direction = -1 if band == "cell" else 1
            for entry, x, target, offset, height in entries:
                label = entry.text
                colour = "#fbbf24" if entry.key == self.current else {"cell": "#fde68a", "vacuum": "#67e8f9", "assembly": "#cbd5e1"}[band]
                label.fill = pg.mkBrush("#050816")
                label.setColor(colour)
                label.setPos(vb.mapSceneToView(pg.Point(x, 0)).x(), base+direction*offset)
                rect = entry.rectangle.rect()
                start = pg.Point(vb.mapSceneToView(pg.Point(target, 0)).x(), rect.top() if band == "cell" else rect.bottom())
                # Stop at the text boundary, using the final view transform.
                text_centre = vb.mapViewToScene(label.pos())
                end = vb.mapSceneToView(pg.Point(text_centre.x(), text_centre.y()+direction*height/2))
                path = QPainterPath(start)
                path.lineTo(end)
                entry.leader.setPath(path)
                entry.leader.show()
                label.update()

    def region_at(self, z, y=1.35):
        for box, key in reversed(self.boxes):
            if box.contains(z, y):
                return key
        return None

    def _clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        for entry in reversed(self._labels):
            if entry.key and entry.text.isVisible() and entry.text.sceneBoundingRect().contains(event.scenePos()):
                self.selected.emit(entry.key)
                event.accept()
                return
        position = vb.mapSceneToView(event.scenePos())
        key = self.region_at(position.x(), 1.35 if position.y() >= 2.1 else position.y())
        if key:
            self.selected.emit(key)
            event.accept()
