"""A spatial execution extent aligned with the ray plot's physical Z axis.

This widget only displays supplied result metadata. A requested cutoff never
becomes completed transport, and no numerical calculation is performed here.
"""

from __future__ import annotations

import math
import weakref

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class RayCalculationExtentBar(QWidget):
    """Narrow physical-Z track plus a selectable, plain-text result summary."""

    TRACK_HEIGHT = 8.0
    TRACK_TOP = 8.0
    COLORS = {
        "uncomputed": "#3b4656",
        "completed": "#24855c",
        "stale": "#b7791f",
        "completed_marker": "#5eead4",
        "stale_marker": "#fcd34d",
        "resumable": "#60a5fa",
        "requested": "#c084fc",
    }
    SPATIAL_HINT = (
        "Physical Z position in millimetres, aligned with the ray plot. "
        "This is a spatial calculation range, not elapsed time or a time percentage. "
        "A requested cutoff is a target; only an executed result fills the completed range. "
        "Calculated through Z describes the executed interval; individual electrons can stop earlier. "
        "The start follows the retained display history, not each electron's emission position. "
        "A resumable plane contains retained restart state; it does not mean a file has been saved, "
        "and its upstream inputs must still match. "
        "Edge arrows mean the actual position lies outside the visible plot. "
        "Green: calculated; grey: uncalculated; purple: cutoff; cyan: calculated end; "
        "blue: resumable plane; amber: previous result."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rayCalculationExtentBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(46)
        self.start_z_mm = None
        self.completed_z_mm = None
        self.resumable_z_mm = None
        self.requested_z_mm = None
        self.stale = False
        self.quality = ""
        self._plot_ref = None
        self._view_box = None
        self._connections = []
        self._watched = []
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._sync_view)

        self.label = QLabel(self)
        self.label.setObjectName("rayCalculationExtentSummary")
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.label.setMinimumWidth(0)
        self.label.setWordWrap(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(0)
        layout.addSpacing(24)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self._sync_view()

    def sizeHint(self):
        return QSize(360, 46)

    @staticmethod
    def _finite(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    @property
    def extent(self):
        """Detached scalar display metadata, with missing values kept explicit."""
        return {"start_z_mm": self.start_z_mm,
                "completed_z_mm": self.completed_z_mm,
                "resumable_z_mm": self.resumable_z_mm,
                "requested_z_mm": self.requested_z_mm,
                "stale": self.stale, "quality": self.quality}

    def set_extent(self, start_z_mm=None, completed_z_mm=None,
                   resumable_z_mm=None, requested_z_mm=None, stale=False, quality=""):
        self.start_z_mm = self._finite(start_z_mm)
        self.completed_z_mm = self._finite(completed_z_mm)
        self.resumable_z_mm = self._finite(resumable_z_mm)
        self.requested_z_mm = self._finite(requested_z_mm)
        self.stale = bool(stale)
        self.quality = str(quality or "")
        self._sync_view()

    def bind_plot(self, plotwidget):
        """Follow one PlotWidget's ViewBox range, viewport and layout geometry."""
        for signal, callback in self._connections:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        for watched in self._watched:
            try:
                watched.removeEventFilter(self)
            except RuntimeError:
                pass
        self._connections.clear()
        self._watched.clear()
        self._plot_ref = None
        self._view_box = None
        if plotwidget is not None:
            view_box = plotwidget.getViewBox()
            self._plot_ref = weakref.ref(plotwidget)
            self._view_box = view_box
            for signal in (view_box.sigRangeChanged, view_box.sigResized):
                signal.connect(self._schedule_sync)
                self._connections.append((signal, self._schedule_sync))
            plotwidget.destroyed.connect(self._plot_destroyed)
            self._connections.append((plotwidget.destroyed, self._plot_destroyed))
            for watched in (plotwidget, plotwidget.viewport()):
                watched.installEventFilter(self)
                self._watched.append(watched)
        self._sync_view()
        self._schedule_sync()

    def _plot_destroyed(self, *_):
        self._plot_ref = None
        self._view_box = None
        self._connections.clear()
        self._watched.clear()
        self._schedule_sync()

    def _schedule_sync(self, *_):
        if not self._refresh_timer.isActive():
            self._refresh_timer.start(0)

    def _plot(self):
        return self._plot_ref() if self._plot_ref is not None else None

    def _view_range(self):
        if self._view_box is None or self._plot() is None:
            return None
        try:
            ranges = self._view_box.viewRange()
            left, right = map(float, ranges[0])
            if not all(math.isfinite(v) for v in (left, right)) or left == right:
                return None
            return min(left, right), max(left, right), .5*sum(ranges[1])
        except RuntimeError:
            return None

    def map_z_to_x(self, z_mm):
        """Map physical Z to this widget's X; deliberately never clamp it."""
        z_mm = self._finite(z_mm)
        ranges, plot = self._view_range(), self._plot()
        if z_mm is None or ranges is None or plot is None:
            return None
        try:
            scene = self._view_box.mapViewToScene(QPointF(z_mm, ranges[2]))
            viewport = plot.viewportTransform().map(scene)
            origin = self.mapFromGlobal(plot.viewport().mapToGlobal(QPoint(0, 0)))
            return float(origin.x()) + float(viewport.x())
        except RuntimeError:
            return None

    def display_geometry(self):
        """Current unrounded track/marker positions for painting and testing."""
        ranges = self._view_range()
        result = {"view_z_mm": None, "track_rect": None,
                  "completed_span_px": None, "markers": {},
                  "completed_color": self.COLORS["stale" if self.stale else "completed"]}
        if ranges is None:
            return result
        x0, x1 = self.map_z_to_x(ranges[0]), self.map_z_to_x(ranges[1])
        if x0 is None or x1 is None or x0 == x1:
            return result
        left, right = min(x0, x1), max(x0, x1)
        track = QRectF(left, self.TRACK_TOP, right-left, self.TRACK_HEIGHT)
        result.update(view_z_mm=ranges[:2], track_rect=track)
        if (self.start_z_mm is not None and self.completed_z_mm is not None
                and self.completed_z_mm >= self.start_z_mm):
            start = self.map_z_to_x(self.start_z_mm)
            completed = self.map_z_to_x(self.completed_z_mm)
            a, b = max(left, min(start, completed)), min(right, max(start, completed))
            if b > a:
                result["completed_span_px"] = (a, b)
        for name, z in (("completed", self.completed_z_mm),
                        ("resumable", self.resumable_z_mm),
                        ("requested", self.requested_z_mm)):
            if z is None:
                continue
            if name == "resumable" and z == self.completed_z_mm:
                continue
            x = self.map_z_to_x(z)
            visible = ranges[0] <= z <= ranges[1]
            result["markers"][name] = {
                "z_mm": z, "x_px": x, "in_view": visible,
                "offscreen": None if visible else "left" if x < left else "right",
            }
        return result

    def _sync_view(self, *_):
        geometry = self.display_geometry()
        markers = geometry["markers"]
        parts = []
        if self.completed_z_mm is None:
            if self.quality:
                parts.append("Calculated range unavailable")
            else:
                parts.append("No completed particle calculation")
            if self.requested_z_mm is not None or self.resumable_z_mm is not None:
                parts.append("Z (mm)")
        else:
            if self.stale:
                parts.append("Previous")
            parts.extend(("Z (mm)", self._position_text(
                "Calculated", self.completed_z_mm, markers.get("completed"))))
        if self.requested_z_mm is not None:
            parts.append(self._position_text("Cutoff", self.requested_z_mm, markers.get("requested")))
        if self.resumable_z_mm is not None and self.resumable_z_mm != self.completed_z_mm:
            parts.append(self._position_text("Resume", self.resumable_z_mm, markers.get("resumable")))
        self.label.setText("  |  ".join(parts))
        details = [self.SPATIAL_HINT, self.label.text()]
        if self.quality:
            details.append(f"Quality: {self.quality}")
        if self.stale:
            details.append("Parameters changed; the displayed extent belongs to the previous calculation.")
        if self.start_z_mm is not None:
            details.append(f"Displayed calculation start: Z = {self.start_z_mm:.12g} mm")
        for name, value in (("Calculated through", self.completed_z_mm),
                            ("Resumable plane", self.resumable_z_mm),
                            ("Requested cutoff", self.requested_z_mm)):
            if value is not None:
                details.append(f"{name}: Z = {value:.12g} mm")
        tooltip = "\n".join(details)
        self.setToolTip(tooltip)
        self.label.setToolTip(tooltip)
        self.update()

    @staticmethod
    def _position_text(name, value, marker):
        outside = "" if marker is None or marker["in_view"] else f" ({marker['offscreen']} of view)"
        return f"{name} {value:.12g}{outside}"

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move,
                            QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self._schedule_sync()
        # Observing layout must not consume scrolling, dragging or zooming.
        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_sync()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_sync()

    def paintEvent(self, event):
        super().paintEvent(event)
        geometry = self.display_geometry()
        track = geometry["track_rect"]
        if track is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.COLORS["uncomputed"]))
        painter.drawRoundedRect(track, 3., 3.)
        span = geometry["completed_span_px"]
        if span is not None:
            painter.setBrush(QColor(geometry["completed_color"]))
            painter.drawRect(QRectF(span[0], track.top(), span[1]-span[0], track.height()))
        for name, marker in geometry["markers"].items():
            color = self.COLORS[("stale_marker" if self.stale else "completed_marker")
                               if name == "completed" else name]
            painter.setPen(QPen(QColor(color), 2.))
            painter.setBrush(QColor(color))
            if marker["in_view"]:
                x = marker["x_px"]
                # Coincident requested/completed positions still show both
                # facts: the target occupies the top, execution the lower tick.
                top, bottom = {"completed": (6., 21.), "requested": (1., 10.),
                               "resumable": (13., 23.)}[name]
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            else:
                # An arrow is explicitly an off-view indicator, never a tick
                # claiming that the real endpoint is at the viewport edge.
                left = marker["offscreen"] == "left"
                x = track.left() if left else track.right()
                inward = 5. if left else -5.
                y = {"completed": 5., "requested": 12., "resumable": 19.}[name]
                painter.drawPolygon(QPolygonF((QPointF(x, y),
                    QPointF(x+inward, y-3.), QPointF(x+inward, y+3.))))
        painter.end()
