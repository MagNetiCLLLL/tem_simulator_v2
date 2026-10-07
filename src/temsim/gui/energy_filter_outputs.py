"""Cached scientific readouts for the standalone Energy Filter page."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget


class EnergyFilterOutputsView(QWidget):
    """Display completed EELS/EFTEM data without building a branch diagram."""

    result_displayed = Signal(object)
    result_stale = Signal()
    OUTPUT_GUIDANCE = (
        "Structure: Physical Layout → Energy Filter. "
        "Particle trajectories: Ray Diagram → Energy Filter rays."
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result = None
        self._outputs_stale = False
        self.heading = QLabel("Energy Filter outputs")
        self.summary = QLabel(
            "Click Calculate energy filter to display EELS or EFTEM results."
        )
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary.setStyleSheet("color: #64748b; font-weight: 600;")
        self.summary.setToolTip(self.OUTPUT_GUIDANCE)
        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self._create_output_tabs(), 1)
        layout.addWidget(self.summary)

    def _create_output_tabs(self):
        self.spectrum_plot = pg.PlotWidget(background="#050816")
        self.spectrum_plot.setObjectName("energyFilterSpectrumPlot")
        self.spectrum_plot.setLabel("bottom", "Energy loss", units="eV")
        self.spectrum_plot.setLabel("left", "Detected counts")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.18)
        self.spectrum_curve = self.spectrum_plot.plot(
            [], [], pen=pg.mkPen("#67e8f9", width=1.8)
        )
        self.spectrum_cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#f8fafc", width=0.8, style=Qt.PenStyle.DashLine
            ),
        )
        self.spectrum_cursor.hide()
        self.spectrum_plot.addItem(self.spectrum_cursor)
        self.spectrum_point = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen("#f8fafc", width=1.0),
            brush=pg.mkBrush("#22d3ee"),
        )
        self.spectrum_point.hide()
        self.spectrum_plot.addItem(self.spectrum_point)
        self.spectrum_status = QLabel("No cached High-accuracy EELS spectrum.")
        self.spectrum_status.setObjectName("energyFilterSpectrumStatus")
        self.spectrum_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.spectrum_status.setStyleSheet("color: #94a3b8;")
        spectrum_page = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_page)
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_layout.addWidget(self.spectrum_plot, 1)
        spectrum_layout.addWidget(self.spectrum_status)
        self._spectrum_energy_ev = np.asarray((), dtype=float)
        self._spectrum_counts = np.asarray((), dtype=float)
        self.spectrum_plot.scene().sigMouseMoved.connect(
            self._spectrum_mouse_moved
        )

        self.eftem_plot = pg.PlotWidget(background="#050816")
        self.eftem_plot.setObjectName("energyFilterEFTEMPlot")
        self.eftem_plot.setAspectLocked(True)
        self.eftem_plot.invertY(True)
        self.eftem_image_item = pg.ImageItem(axisOrder="row-major")
        self.eftem_image_item.hide()
        self.eftem_plot.addItem(self.eftem_image_item)
        self.eftem_status = QLabel("No cached High-accuracy EFTEM image.")
        self.eftem_status.setObjectName("energyFilterEFTEMStatus")
        self.eftem_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.eftem_status.setStyleSheet("color: #94a3b8;")
        eftem_page = QWidget()
        eftem_layout = QVBoxLayout(eftem_page)
        eftem_layout.setContentsMargins(0, 0, 0, 0)
        eftem_layout.addWidget(self.eftem_plot, 1)
        eftem_layout.addWidget(self.eftem_status)

        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("energyFilterOutputTabs")
        self.output_tabs.addTab(spectrum_page, "EELS spectrum")
        self.output_tabs.addTab(eftem_page, "EFTEM image")
        return self.output_tabs

    def _spectrum_mouse_moved(self, scene_position) -> None:
        """Read the nearest cached spectrum bin without recalculation."""

        if self._spectrum_energy_ev.size == 0:
            return
        view_box = self.spectrum_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return
        point = view_box.mapSceneToView(scene_position)
        index = int(np.argmin(np.abs(
            self._spectrum_energy_ev - float(point.x())
        )))
        energy_ev = float(self._spectrum_energy_ev[index])
        counts = float(self._spectrum_counts[index])
        self.spectrum_cursor.setPos(energy_ev)
        self.spectrum_cursor.show()
        self.spectrum_point.setData((energy_ev,), (counts,))
        self.spectrum_point.show()
        stale = (
            "Previous complete EELS spectrum | inputs changed | "
            if self._outputs_stale else ""
        )
        self.spectrum_status.setText(
            f"{stale}Energy {energy_ev:.6g} eV | counts {counts:.6g}"
        )

    def _display_scientific_outputs(self, branch_result, mode: str) -> None:
        """Render only data already attached to the completed result."""

        self._spectrum_energy_ev = np.asarray((), dtype=float)
        self._spectrum_counts = np.asarray((), dtype=float)
        self.spectrum_curve.setData([], [])
        self.spectrum_cursor.hide()
        self.spectrum_point.hide()
        forward = getattr(branch_result, "eels_forward", None)
        if forward is None:
            self.spectrum_status.setText(
                "No cached High-accuracy EELS spectrum."
            )
        else:
            energy = np.asarray(forward.energy_loss_ev, dtype=float)
            sampled = getattr(forward, "detected_sampled_counts", None)
            counts = np.asarray(
                sampled
                if sampled is not None
                else forward.detected_expected_counts,
                dtype=float,
            )
            if (
                energy.ndim != 1
                or counts.shape != energy.shape
                or not np.all(np.isfinite(energy))
                or not np.all(np.isfinite(counts))
            ):
                self.spectrum_status.setText("Cached EELS spectrum is invalid.")
            else:
                self._spectrum_energy_ev = energy
                self._spectrum_counts = counts
                self.spectrum_curve.setData(energy, counts)
                self.spectrum_plot.autoRange()
                kind = "sampled" if sampled is not None else "expected"
                self.spectrum_status.setText(
                    f"{energy.size:,} bins | {kind} total counts "
                    f"{float(np.sum(counts)):.6g} | hover for bin values"
                )

        self.eftem_image_item.clear()
        self.eftem_image_item.hide()
        eftem_image = getattr(branch_result, "eftem_image", None)
        if str(mode).lower() != "eftem":
            self.eftem_status.setText(
                "EFTEM image is available when acquisition mode is EFTEM."
            )
        elif eftem_image is None:
            self.eftem_status.setText(
                "No cached High-accuracy EFTEM image."
            )
        else:
            image = np.asarray(eftem_image, dtype=float)
            if (
                image.ndim != 2
                or not np.all(np.isfinite(image))
                or np.any(image < 0.0)
            ):
                self.eftem_status.setText("Cached EFTEM image is invalid.")
            else:
                self.eftem_image_item.setImage(image, autoLevels=True)
                self.eftem_image_item.show()
                self.eftem_plot.autoRange()
                self.eftem_status.setText(
                    f"Cached EFTEM image | {image.shape[1]} × "
                    f"{image.shape[0]} px"
                )

    def display_result(self, result) -> None:
        """Publish cached outputs and forward the same result to ray views."""

        self._result = result
        self._outputs_stale = False
        self.summary.setToolTip(self.OUTPUT_GUIDANCE)
        state = getattr(result, "state_snapshot", None)
        energy_filter = getattr(state, "energy_filter", None)
        enabled = energy_filter is not None and energy_filter.enabled
        branch_result = getattr(result, "energy_filter", None) if enabled else None
        mode = str(getattr(energy_filter, "operating_mode", ""))
        self._display_scientific_outputs(branch_result, mode)
        self.heading.setText(
            f"Energy Filter outputs - {mode.upper()}"
            if enabled else "Energy Filter outputs"
        )
        if not enabled:
            self.summary.setText("Energy Filter is not installed or enabled.")
        elif branch_result is None:
            self.summary.setText(
                "No Energy Filter result. Click Calculate energy filter."
            )
        else:
            status = str(getattr(branch_result, "status", ""))
            metrics = getattr(energy_filter, "_last_slit_metrics", None)
            metric_text = (
                f" | dispersion {metrics.dispersion_um_per_ev:.4g} um/eV | "
                f"non-iso RMS {metrics.non_isochromaticity_ev_rms:.4g} eV"
                if metrics is not None else ""
            )
            self.summary.setText(
                (f"{mode.upper()} outputs | {status}" if status
                 else f"Cached {mode.upper()} outputs") + metric_text
            )
        self.result_displayed.emit(result)

    def mark_result_stale(self) -> None:
        """Keep previous completed readouts visible after parameter changes."""

        self._outputs_stale = True
        self.summary.setText(
            "Previous Energy Filter outputs retained | inputs changed | "
            "Click Calculate energy filter to update."
            if self._result is not None
            else "Inputs changed | Click Calculate energy filter."
        )
        if self._spectrum_energy_ev.size:
            self.spectrum_status.setText(
                "Previous complete EELS spectrum retained | inputs changed"
            )
        if self.eftem_image_item.isVisible():
            self.eftem_status.setText(
                "Previous complete EFTEM image retained | inputs changed"
            )
        self.result_stale.emit()
