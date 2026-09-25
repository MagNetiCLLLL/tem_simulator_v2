"""Cached accelerator annotations from a displayed calculation's snapshot.

All axial coordinates are millimetres in the Ray Diagram's resolved column
frame. This module never constructs a field provider or propagates particles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import Qt
import pyqtgraph as pg


ACCELERATOR_GAP_TOOLTIP = (
    "Amber marks show the displayed calculation's accelerator stages. "
    "For the analytic field, bands span each potential transition "
    "(field centre plus or minus soft edge); lines mark its centre. "
    "For a solved curved-tip field, lines mark electrode positions only; "
    "they do not define the electric-field extent. Hover for stage coordinates."
)


@dataclass(frozen=True, slots=True)
class AcceleratorGapRecord:
    """Detached geometry for one stage; absent bounds mean electrode only."""

    stage_number: int
    center_z_mm: float
    start_z_mm: float | None = None
    end_z_mm: float | None = None

    @property
    def tooltip(self) -> str:
        if self.start_z_mm is None:
            return (
                f"Accelerator electrode {self.stage_number}\n"
                f"Electrode centre Z = {self.center_z_mm:.9g} mm\n"
                "Solved curved-tip field: this is an electrode position, "
                "not a field boundary."
            )
        return (
            f"Accelerator gap {self.stage_number}\n"
            f"Field centre Z = {self.center_z_mm:.9g} mm\n"
            f"Analytic potential transition: {self.start_z_mm:.9g} "
            f"to {self.end_z_mm:.9g} mm\n"
            "Band follows the calculated field-centre offset and soft edge."
        )


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def accelerator_gap_records(result) -> tuple[AcceleratorGapRecord, ...]:
    """Read captured geometry only; incomplete historical records draw nothing.

    Snapshot stage coordinates have already received the resolved assembly
    translation. Applying another origin shift would misplace the annotations.
    A surface-model gun uses a solved field, whose support cannot be inferred
    from the analytic soft-edge settings retained in its compatibility record.
    """
    snapshot = getattr(result, "state_snapshot", None)
    gun = getattr(snapshot, "electron_gun", None)
    accelerator = getattr(gun, "accelerator", None)
    if accelerator is None:
        return ()
    solved = (
        getattr(gun, "type_key", None) == "cold_feg"
        and getattr(getattr(gun, "emitter", None), "surface_model", None) is not None
    )
    offset = 0.0 if solved else _finite_float(
        getattr(accelerator, "field_center_offset_mm", None)
    )
    if offset is None:
        return ()
    records = []
    for number, stage in enumerate(getattr(accelerator, "stages", ()) or (), 1):
        center = _finite_float(getattr(stage, "center_from_tip_mm", None))
        if center is None:
            continue
        center += offset
        if solved:
            records.append(AcceleratorGapRecord(number, center))
            continue
        edge = _finite_float(getattr(stage, "soft_edge_mm", None))
        if edge is None or edge <= 0.0:
            continue
        start, end = center - edge, center + edge
        if all(math.isfinite(value) for value in (center, start, end)):
            records.append(AcceleratorGapRecord(number, center, start, end))
    return tuple(records)


class AcceleratorGapOverlay:
    """Own lightweight pyqtgraph items without affecting rays or auto-ranging.

    Call ``sync(result, visible=...)`` when the displayed result changes and
    ``setVisible(...)`` for the toggle. Neither operation requests a retrace.
    Unchanged stage graphics survive publications and visibility toggles.
    """

    def __init__(self, plot) -> None:
        self.plot = plot
        self.records: tuple[AcceleratorGapRecord, ...] = ()
        self._items_by_record: dict[AcceleratorGapRecord, tuple] = {}
        self._visible = True

    @property
    def items(self) -> tuple:
        return tuple(
            item for record in self.records for item in self._items_by_record[record]
        )

    def sync(self, result, *, visible: bool = True) -> None:
        records = accelerator_gap_records(result)
        wanted = set(records)
        for record in tuple(self._items_by_record):
            if record not in wanted:
                for item in self._items_by_record.pop(record):
                    self.plot.removeItem(item)
        for record in records:
            if record not in self._items_by_record:
                self._items_by_record[record] = self._build(record)
        self.records = records
        # PlotWidget.clear() can detach cached graphics during scene reset.
        # Reattach those same objects when the result is published again.
        for item in self.items:
            if item.scene() is None:
                self.plot.addItem(item, ignoreBounds=True)
        self.setVisible(visible)

    def _build(self, record: AcceleratorGapRecord) -> tuple:
        items = []
        if record.start_z_mm is not None:
            band = pg.LinearRegionItem(
                values=(record.start_z_mm, record.end_z_mm),
                orientation="vertical", movable=False,
                brush=pg.mkBrush(245, 180, 60, 24),
                pen=pg.mkPen(None), hoverBrush=pg.mkBrush(245, 180, 60, 24),
                hoverPen=pg.mkPen(None),
            )
            band.setZValue(-15)
            for edge in band.lines:
                edge.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                edge.setToolTip(record.tooltip)
            items.append(band)
        line = pg.InfiniteLine(
            pos=record.center_z_mm, angle=90, movable=False,
            pen=pg.mkPen(245, 180, 60, 145, width=1,
                         style=Qt.PenStyle.DashLine),
        )
        line.setZValue(-14)
        items.append(line)
        for item in items:
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setAcceptHoverEvents(True)
            item.setToolTip(record.tooltip)
            self.plot.addItem(item, ignoreBounds=True)
        return tuple(items)

    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for item in self.items:
            item.setVisible(self._visible)

    def clear(self) -> None:
        for item in self.items:
            self.plot.removeItem(item)
        self._items_by_record.clear()
        self.records = ()
