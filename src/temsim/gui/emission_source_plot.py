"""Read-only view of recorded emission positions, independent of downstream loss.

The caller supplies a bounded display sample and its original colours. Positions
are SI metres and angles are radians; this widget only rotates the display basis.
It never samples a source, propagates particles, or offsets overlapping points.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from temsim.gui.transverse_projection import (
    orthogonal_axis_name,
    projection_axis_name,
    transverse_view_coordinates,
)


class EmissionSourcePlot(QWidget):
    """Source X/Y scatter in nanometres, with fixed per-particle identity.

    ``set_source`` accepts IDs, colours and two angle arrays of length N, plus
    positions of shape (N, 3). The caller selects the display sample; all inputs
    are copied. Unknown angles may be NaN. Invalid positions are omitted with an
    explicit readout. Recolouring the same IDs and positions preserves pan/zoom.
    """

    METRES_TO_NM = 1.0e9

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ids = np.empty(0, dtype=np.int64)
        self._positions = np.empty((0, 3), dtype=float)
        self._brushes = ()
        self._azimuth = np.empty(0, dtype=float)
        self._angle_to_normal = np.empty(0, dtype=float)
        self._flight_time = None
        self._time_reference = None
        self._projection_angle_deg = 0.0
        self._overlap_sample_count = 0
        self._overlap_position_count = 0
        self.scatter = None

        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setObjectName("emissionSourcePlot")
        self.plot.setFixedHeight(196)
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setAspectLocked(True, ratio=1.0)
        self.plot.getViewBox().disableAutoRange()
        self.plot.getAxis("left").setWidth(66)
        self.plot.getAxis("bottom").setHeight(42)
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setSIPrefixEnableRanges(())
            axis.enableAutoSIPrefix(False)
        self._update_labels()
        self.plot.setToolTip(
            "Actual recorded emission positions before extraction, in nanometres. "
            "Colours identify the same source particles in the downstream view. "
            "Overlapping positions are not displaced. This source sample does not "
            "change when downstream particles are stopped."
        )

        self.summary = QLabel()
        self.summary.setObjectName("emissionSourceSummary")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #94a3b8;")
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.fit_button = QPushButton("Fit source")
        self.fit_button.setObjectName("fitEmissionSource")
        self.fit_button.setToolTip("Fit the recorded source positions. No recalculation.")
        self.fit_button.clicked.connect(self.fit_to_source)
        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.addWidget(self.summary, 1)
        summary_row.addWidget(self.fit_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.plot)
        layout.addLayout(summary_row)
        self.clear("Recorded emission positions are unavailable.")

    def set_source(
        self, *, ids, position_m, brushes, azimuth_rad, angle_to_normal_rad,
        status: str = "",
        flight_time_s=None, reference_time_s=None,
    ) -> None:
        """Display original launch records without changing their coordinates.

        Shape/type mismatches raise ``ValueError`` before replacing the current
        view. All colours are supplied by the caller, using the downstream map.
        """
        ids = np.asarray(ids)
        positions = np.asarray(position_m, dtype=float)
        azimuth = np.asarray(azimuth_rad, dtype=float)
        polar = np.asarray(angle_to_normal_rad, dtype=float)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Emission source IDs must be a one-dimensional integer array.")
        count = ids.size
        if (positions.shape != (count, 3) or azimuth.shape != (count,)
                or polar.shape != (count,)):
            raise ValueError("Emission source positions and angles must match the source IDs.")
        brushes = tuple(brushes)
        flight_times = None if flight_time_s is None else np.asarray(flight_time_s, dtype=float)
        if flight_times is not None and flight_times.shape != (count,):
            raise ValueError("Source path times must match displayed paths.")
        if len(brushes) != count:
            raise ValueError("Emission source colours must match the source IDs.")
        same_positions = (
            self.scatter is not None
            and np.array_equal(self._ids, ids)
            and np.array_equal(self._positions, positions, equal_nan=True)
        )
        self._ids = ids.copy()
        self._positions = positions.copy()
        self._azimuth = azimuth.copy()
        self._angle_to_normal = polar.copy()
        self._flight_time = None if flight_times is None else flight_times.copy()
        self._time_reference = reference_time_s
        self.plot.setToolTip(
            "Actual tip emission positions in nanometres, without position jitter. "
            + ("Each arriving path uses its own downstream delay colour. The displayed paths "
               "can change with the observation plane; coincident colours overlap."
               if flight_times is not None else
               "Colours track original emission. The source sample remains fixed when "
               "downstream particles are stopped; coincident colours overlap.")
        )
        self._brushes = tuple(pg.mkBrush(brush) for brush in brushes)
        finite = np.all(np.isfinite(positions), axis=1)
        shown = int(np.count_nonzero(finite))
        if shown == 0:
            self.clear(status or "No finite recorded emission positions are available.")
            return
        self._draw(fit=not same_positions)
        self.fit_button.setEnabled(True)
        text = (f"{shown:,} arriving paths at emission positions" if flight_times is not None
                else f"{shown:,} recorded emission samples")
        if self._overlap_sample_count:
            place = "position" if self._overlap_position_count == 1 else "positions"
            text += (
                f" | {self._overlap_sample_count:,} overlapping samples at "
                f"{self._overlap_position_count:,} source {place}; hover for "
                + ("path times" if flight_times is not None else "directions")
            )
        if shown != count:
            text += f" | {count - shown:,} positions unavailable"
        if status:
            text += f" | {status}"
        self.summary.setText(text)

    def clear(self, message: str) -> None:
        """Remove stale source points and explain why the source is unavailable."""
        self._ids = np.empty(0, dtype=np.int64)
        self._positions = np.empty((0, 3), dtype=float)
        self._brushes = ()
        self._azimuth = np.empty(0, dtype=float)
        self._angle_to_normal = np.empty(0, dtype=float)
        self._flight_time = None
        self._time_reference = None
        self._overlap_sample_count = 0
        self._overlap_position_count = 0
        self.scatter = None
        self.plot.clear()
        self.summary.setText(message)
        self.fit_button.setEnabled(False)

    def fit_to_source(self) -> None:
        """Restore the view bounds using only the recorded display sample."""
        if self.scatter is not None:
            self._draw(fit=True)

    def set_projection_angle(self, angle_deg: float) -> None:
        """Rotate X/Y into the shared U/V display basis while retaining zoom."""
        angle = float(angle_deg)
        if not math.isfinite(angle):
            raise ValueError("The source display projection angle must be finite.")
        angle %= 360.0
        if math.isclose(angle, self._projection_angle_deg, rel_tol=0.0, abs_tol=1.0e-12):
            return
        self._projection_angle_deg = angle
        self._update_labels()
        if self.scatter is not None:
            self._draw(fit=False)

    def _update_labels(self) -> None:
        for name, label in (
            ("bottom", projection_axis_name(self._projection_angle_deg)),
            ("left", orthogonal_axis_name(self._projection_angle_deg)),
        ):
            self.plot.getAxis(name).setLabel(
                f"Source {label}", units="nm", siPrefixEnableRanges=(),
            )

    @staticmethod
    def _hover_text(_x, _y, record) -> str:
        x, y, z = record["position_m"]

        def degrees(value):
            return f"{math.degrees(value):.6g}°" if math.isfinite(value) else "unavailable"

        text = (
            f"Source ray {record['source_ray_id']}\n"
            f"Original X {x * 1e9:.6g} nm | Y {y * 1e9:.6g} nm | Z {z * 1e9:.6g} nm"
        )
        def time_text(sample):
            if "flight_time_s" not in sample:
                return ""
            time = sample["flight_time_s"]
            if not np.isfinite(time):
                return " | Flight time unavailable"
            reference = sample.get("reference_time_s")
            delay = "" if reference is None else f" | Delay {(time-reference)*1e15:.8g} fs"
            return f" | Flight time {time*1e9:.12g} ns" + delay
        text += time_text(record)
        coincident = record.get("coincident_samples", (record,))
        if len(coincident) == 1:
            text += (
                f"\nLaunch azimuth {degrees(record['azimuth_rad'])}\n"
                f"Angle to local normal {degrees(record['angle_to_normal_rad'])}"
            )
        text += (
            f"\nAt this recorded X/Y: {len(coincident):,} displayed samples"
            "\nSample count, not intensity or probability."
        )
        if len(coincident) > 1:
            # Include the hovered path among the eight reported samples. A
            # source particle may have several descendants with distinct times.
            displayed = [record] + [
                sample for sample in coincident
                if sample is not record
            ][:7]
            for sample in displayed:
                text += (
                    f"\nSource ray {sample['source_ray_id']} | "
                    f"Z {sample['position_m'][2] * 1e9:.6g} nm | "
                    f"Launch azimuth {degrees(sample['azimuth_rad'])} | "
                    f"angle to local normal {degrees(sample['angle_to_normal_rad'])}"
                    + time_text(sample)
                )
            if len(coincident) > 8:
                text += f"\nPlus {len(coincident) - 8:,} more coincident samples."
        return text

    def _draw(self, *, fit: bool) -> None:
        bounds = self.plot.viewRange()
        finite = np.all(np.isfinite(self._positions), axis=1)
        rows = np.flatnonzero(finite)
        positions = self._positions[finite]
        u, v = transverse_view_coordinates(
            positions[:, 0], positions[:, 1], self._projection_angle_deg,
        )
        u, v = u * self.METRES_TO_NM, v * self.METRES_TO_NM
        records = [
            {"source_ray_id": int(self._ids[row]),
             "position_m": tuple(self._positions[row]),
             "azimuth_rad": float(self._azimuth[row]),
             "angle_to_normal_rad": float(self._angle_to_normal[row])}
            for row in rows
        ]
        if self._flight_time is not None:
            for record, row in zip(records, rows):
                record.update(flight_time_s=float(self._flight_time[row]),
                              reference_time_s=self._time_reference)
        groups = {}
        for record in records:
            groups.setdefault(record["position_m"][:2], []).append(record)
        groups = {position: tuple(samples) for position, samples in groups.items()}
        overlapping = [samples for samples in groups.values() if len(samples) > 1]
        self._overlap_sample_count = sum(map(len, overlapping))
        self._overlap_position_count = len(overlapping)
        # Share each original record with its group so hover can exclude only
        # that path by identity, retaining other paths with the same source ID.
        # Shared tuples avoid N-squared copies at a point source. Grouping does
        # not introduce coordinate offsets or averaged colours.
        for record in records:
            record["coincident_samples"] = groups[record["position_m"][:2]]
        self.plot.clear()
        self.scatter = pg.ScatterPlotItem(
            x=u, y=v, brush=[self._brushes[row] for row in rows],
            pen=pg.mkPen(None), size=5, data=records, hoverable=True,
            tip=self._hover_text, pxMode=True,
        )
        self.plot.addItem(self.scatter)
        self.plot.addLine(x=0.0, pen=pg.mkPen("#64748b", width=0.6))
        self.plot.addLine(y=0.0, pen=pg.mkPen("#64748b", width=0.6))
        if fit:
            bounds = []
            for values in (u, v):
                centre = (float(np.min(values)) + float(np.max(values))) / 2.0
                half = max(float(np.ptp(values)) * 0.54, 0.5)
                bounds.append((centre - half, centre + half))
        self.plot.setRange(
            xRange=bounds[0], yRange=bounds[1], padding=0.0, disableAutoRange=True,
        )
