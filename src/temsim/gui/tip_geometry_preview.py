"""Equal-scale apex cross-section; a geometry view, not a ray calculation."""
import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from temsim.optics.electron_gun.tip_patch import patch_dimensions


class TipGeometryPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self.setMinimumHeight(160)
        self.setMaximumHeight(210)
        self.setToolTip(
            "Equal-scale X-Z apex detail; +Z points downstream. The highlighted arc is the emitting cap. "
            "Arrows are local surface normals, not propagated electron trajectories. The shank is cropped.")

    def set_model(self, model):
        if model is not None:
            patch_dimensions(model.geometry, model.emission.cap_half_angle_deg)
        self._model = model
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(12, 20, "Tip cross-section (apex detail)")
        if self._model is None:
            painter.drawText(12, 44, "Enter valid tip geometry")
            return
        geometry = self._model.geometry
        radius = geometry.apex_radius_nm
        scale = min((self.width() - 44) / (3 * radius), (self.height() - 46) / (2.5 * radius))
        origin = QPointF(self.width() / 2, 32 + 1.6 * radius * scale)

        def point(x, z):
            return QPointF(origin.x() + x * scale, origin.y() + z * scale)

        z = np.linspace(-1.6 * radius, 0, 120)
        r = geometry.radius_m(z * 1e-9) * 1e9
        path = QPainterPath(point(-r[0], z[0]))
        for xx, zz in zip(-r, z):
            path.lineTo(point(xx, zz))
        for xx, zz in zip(r[::-1], z[::-1]):
            path.lineTo(point(xx, zz))
        path.closeSubpath()
        painter.fillPath(path, QColor("#26364c"))
        painter.setPen(QPen(QColor("#9ca3af"), 1.4))
        painter.drawPath(path)
        angle = math.radians(self._model.emission.cap_half_angle_deg)
        arc = QPainterPath()
        for index, theta in enumerate(np.linspace(-angle, angle, 100)):
            position = point(radius * math.sin(theta), -2 * radius * math.sin(theta / 2)**2)
            if index == 0:
                arc.moveTo(position)
            else:
                arc.lineTo(position)
        painter.setPen(QPen(QColor("#fbbf24"), 3))
        painter.drawPath(arc)
        painter.setPen(QPen(QColor("#34d399"), 1.3))
        for theta in (-angle, 0, angle):
            x, zz = radius * math.sin(theta), -2 * radius * math.sin(theta / 2)**2
            length = .38 * radius
            end = point(x + length * math.sin(theta), zz + length * math.cos(theta))
            painter.drawLine(point(x, zz), end)
            for sign in (-1, 1):
                painter.drawLine(end, point(x + .78 * length * math.sin(theta) + sign * .1 * length * math.cos(theta),
                                           zz + .78 * length * math.cos(theta) - sign * .1 * length * math.sin(theta)))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(12, self.height() - 12, "Emitting surface")
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(self.width() - 42, self.height() - 12, "+Z ↓")
