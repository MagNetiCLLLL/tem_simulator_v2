"""Vacuum intervals on Physical Layout's shared, millimetre-valued Z axis."""
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QGraphicsRectItem


class VacuumAxialView(pg.PlotWidget):
    selected = Signal(str)

    def __init__(self):
        super().__init__(background="#050816")
        self.rows, self.modules, self.current = (), (), ""
        self.boxes = []
        self._items, self._labels = [], []
        self._physical_plot = None
        self._syncing = False
        self.component = None
        self.sample_z_mm = None
        self.sample_marker = None
        self.setMinimumHeight(210)
        self.setLabel("bottom", "Axial position", units="mm")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").setTicks([[(2.45, "Modules"), (1.35, "Vacuum"), (.35, "Cell")]])
        self.getAxis("left").setWidth(70)
        self.showGrid(x=True, y=False, alpha=.16)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.disableAutoRange()
        self.setYRange(0, 3, padding=0)
        self.setMouseEnabled(x=True, y=False)
        self.scene().sigMouseClicked.connect(self._clicked)
        self.getViewBox().sigXRangeChanged.connect(self._vacuum_range_changed)
        self.getViewBox().sigResized.connect(self._layout_labels)
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

    def _rectangle(self, start, end, bottom, height, colour, label, key=None, dashed=False):
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
        text.setPos((start+end)/2, bottom+height/2)
        self.addItem(text, ignoreBounds=True)
        self._items.append(text)
        self._labels.append((text, start, end))

    def redraw(self):
        for item in self._items:
            self.removeItem(item)
        self._items, self._labels, self.boxes = [], [], []
        for i, module in enumerate(self.modules):
            self._rectangle(module.start_z_mm, module.end_z_mm, 2.1, .7,
                            "#23364a" if i % 2 == 0 else "#35425a",
                            Path(module.source_file).stem.replace("_", " "))
        for row in self.rows:
            cell = row.radius_mm is not None
            transition = row.end_medium is not None
            self._rectangle(row.start_z_mm, row.end_z_mm, .1 if cell else 1.0,
                            .5 if cell else .7,
                            "#475569" if transition else ("#785420" if cell else "#155e75"),
                            "Linear transition" if transition else row.name,
                            row.key, transition)
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
        vb = self.getViewBox()
        for label, start, end in self._labels:
            x1 = vb.mapViewToScene(pg.Point(start, 0)).x()
            x2 = vb.mapViewToScene(pg.Point(end, 0)).x()
            label.setVisible(abs(x2-x1) > label.boundingRect().width()+10)

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
        position = vb.mapSceneToView(event.scenePos())
        key = self.region_at(position.x(), 1.35 if position.y() >= 2.1 else position.y())
        if key:
            self.selected.emit(key)
            event.accept()
