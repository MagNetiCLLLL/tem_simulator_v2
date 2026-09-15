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
        self._continuous = None
        self.setMinimumHeight(160)
        self.setMaximumHeight(210)
        self.setToolTip(
            "Equal-scale X-Z apex detail; +Z points downstream. The highlighted arc is the emitting cap. "
            "Arrows are local surface normals, not propagated electron trajectories. The shank is cropped.")

    def set_model(self, model):
        if model is not None:
            patch_dimensions(model.geometry, model.emission.cap_half_angle_deg)
        self._model = model
        self._continuous = None
        self.update()

    def set_continuous_tip(self, emitter):
        """Render the operating emitting patch, not the archived metal CAD."""
        from temsim.optics.electron_gun.tip_curvature import ANGLE_ONLY_MODEL, validate_curvature, support_radius_nm
        validate_curvature(emitter)
        self._model = None
        self._continuous = (emitter.curvature_nm_inv, support_radius_nm(emitter), emitter.curvature_model)
        geometry = ("Historical angle-only model: all launch positions stay at Z = 0. "
                    if emitter.curvature_model == ANGLE_ONLY_MODEL else
                    "The tip centre stays at Z = 0; off-axis emission points have negative Z. ")
        self.setToolTip("Equal-scale X-Z emitting section; +Z points downstream. " + geometry +
                        "Arrows show local emission axes, not trajectories. "
                        "Analytic gun-field approximation, not a recomputed metal electrode.")
        self.update()

    def _paint_continuous(self, painter):
        from temsim.optics.electron_gun.tip_curvature import ANGLE_ONLY_MODEL, LEGACY_MODEL
        k, support, model = self._continuous
        angle_only = model == ANGLE_ONLY_MODEL
        title = ("Historical launch plane" if angle_only else
                 "Historical emitting surface" if model == LEGACY_MODEL else "Emitting surface · centre Z = 0")
        painter.drawText(12, 20, title)
        extent = max(support, 1e-6)
        scale = min((self.width()-44)/(2.6*extent), (self.height()-68)/(1.5*extent))
        origin = QPointF(self.width()/2, self.height()*.52)
        def point(x, z):
            return QPointF(origin.x()+x*scale, origin.y()+z*scale)
        def sag(x):
            return 0.0 if angle_only else -k*x*x/(1+np.sqrt(1-(k*x)**2))
        x = np.linspace(-support, support, 101)
        arc = QPainterPath(point(x[0], sag(x[0])))
        for xx in x[1:]:
            arc.lineTo(point(xx, sag(xx)))
        painter.setPen(QPen(QColor("#fbbf24"), 3))
        painter.drawPath(arc)
        painter.setPen(QPen(QColor("#34d399"), 1.3))
        for xx in (-support, 0, support):
            nz = np.sqrt(1-(k*xx)**2)
            length = .45*extent
            painter.drawLine(point(xx, sag(xx)), point(xx+length*k*xx, sag(xx)+length*nz))
        painter.setPen(QColor("#e5e7eb"))
        painter.drawEllipse(point(0, 0), 3, 3)
        painter.drawText(point(0, 0) + QPointF(7, -7), "Z = 0")
        radius = f"R {1/k:.6g} nm" if k else "Flat · R infinite"
        painter.drawText(12, self.height()-30, f"{radius} | support diameter {2*support:.6g} nm")
        painter.drawText(12, self.height()-12,
                         "Launch Z = 0 · angle only" if angle_only else f"Edge Z {sag(support):.6g} nm")
        painter.drawText(self.width()-42, self.height()-12, "+Z ↓")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        painter.setPen(QColor("#e5e7eb"))
        if self._continuous is not None:
            self._paint_continuous(painter)
            return
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
