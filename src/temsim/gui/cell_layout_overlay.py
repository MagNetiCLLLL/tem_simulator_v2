"""Exact applied X-Z cell projection; selection markers never change geometry."""
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsRectItem

from temsim.cell_geometry import CellPhysicalContext, COLOURS, CELL_SAMPLE_KEY


class CellLayoutOverlay:
    def __init__(self, plot):
        self.plot, self.items = plot, []
        self.context = CellPhysicalContext()

    def render(self, mapping):
        for item in self.items:
            mapping.pop(id(item), None)
            self.plot.removeItem(item)
        self.items = []
        for r in self.context.layers:
            colour = tuple(round(v*255) for v in COLOURS[r.key])
            item = QGraphicsRectItem(r.start_z_mm, r.center_x_mm-r.radius_mm,
                                    r.end_z_mm-r.start_z_mm, 2*r.radius_mm)
            item.setPen(pg.mkPen(colour[:3], width=1))
            item.setBrush(pg.mkBrush(colour))
            item.setZValue(18)
            if r.medium.phase == "vacuum":
                medium_text = "Ideal vacuum"
            elif r.medium.phase == "gas":
                pressure = f"{r.medium.pressure_mbar:g}"
                if r.end_medium is not None:
                    pressure += f" → {r.end_medium.pressure_mbar:g}"
                medium_text = f"{r.medium.formula} · {pressure} mbar · {r.medium.temperature_k:g} K"
            else:
                medium_text = f"{r.medium.formula} · {r.medium.density_kg_m3:g} kg/m³"
            item.setToolTip(f"{r.name}\n{medium_text}\n"
                f"Z {r.start_z_mm:.12g} to {r.end_z_mm:.12g} mm\n"
                f"Thickness {(r.end_z_mm-r.start_z_mm)*1e6:g} nm; diameter {2*r.radius_mm*1e6:g} nm\n"
                f"Centre X {r.center_x_mm*1e6:g}, Y {r.center_y_mm*1e6:g} nm. X-Z projection.")
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.plot.addItem(item, ignoreBounds=True)
            self.items.append(item)
            mapping[id(item)] = r.key
        if self.context.layers:
            cell = next(r for r in self.context.layers if r.key == "specimen_cell")
            marker = pg.ScatterPlotItem([.5*(cell.start_z_mm+cell.end_z_mm)], [cell.center_x_mm],
                                       symbol="s", size=9, pen="#67e8f9", brush=(22, 78, 99, 100))
            marker.setToolTip("Cell location marker (not to scale). Use Fit cell to inspect the actual membranes.")
            marker.setZValue(25)
            self.plot.addItem(marker, ignoreBounds=True)
            self.items.append(marker)
            mapping[id(marker)] = "specimen_cell"
            line = pg.PlotCurveItem([self.context.sample_z_mm]*2,
                [self.context.sample_x_mm-cell.radius_mm, self.context.sample_x_mm+cell.radius_mm],
                pen=pg.mkPen("#fbbf24", width=2, style=Qt.PenStyle.DashLine))
            line.setToolTip("Sample reference from Sample; not additional material")
            line.setZValue(24)
            self.plot.addItem(line, ignoreBounds=True)
            self.items.append(line)
            mapping[id(line)] = CELL_SAMPLE_KEY

    def fit(self):
        if not self.context.layers:
            return False
        z0 = min(self.context.sample_z_mm, *(r.start_z_mm for r in self.context.layers))
        z1 = max(self.context.sample_z_mm, *(r.end_z_mm for r in self.context.layers))
        x0 = min(r.center_x_mm-r.radius_mm for r in self.context.layers)
        x1 = max(r.center_x_mm+r.radius_mm for r in self.context.layers)
        self.plot.setRange(xRange=(z0, z1), yRange=(x0, x1), padding=.12, disableAutoRange=True)
        return True
