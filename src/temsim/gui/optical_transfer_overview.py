"""Display cached first-order responses, separately from beam intensities."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from temsim.gui.plane_equations import plane_equation_tooltip

from temsim.physics.scan_geometry import (
    DIFFRACTION_CONJUGACY_TOLERANCE,
    IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD,
    classify_sample_plane_transfer,
)


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class _ResponsePanel(QWidget):
    """Map a unit circle and its positive basis directions into column X/Y."""

    def __init__(self, *, angular: bool, parent=None) -> None:
        super().__init__(parent)
        self.angular = angular
        self.unit = "µm" if angular else "nm"
        heading = _label(
            "Canonical angle → target position" if angular else "Position → target position"
        )
        heading.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc;")
        self.metrics = _label("No calculated response")
        self.metrics.setMinimumHeight(48)
        self.metrics.setStyleSheet("color: #e2e8f0;")
        self.plot = pg.PlotWidget(background="#050816")
        self.plot.setMinimumSize(220, 180)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", "ΔX at target", units=self.unit)
        self.plot.setLabel("left", "ΔY at target", units=self.unit)
        for name in ("bottom", "left"):
            self.plot.getAxis(name).enableAutoSIPrefix(False)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#334155")))
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#334155")))
        self.outline = self.plot.plot(pen=pg.mkPen("#c4b5fd", width=2))
        self.basis_lines = []
        self.basis_labels = []
        for axis, colour in (("X", "#38bdf8"), ("Y", "#fbbf24")):
            self.basis_lines.append(self.plot.plot(
                pen=pg.mkPen(colour, width=2), symbol="o", symbolSize=5,
                symbolBrush=colour, symbolPen=None,
            ))
            label = pg.TextItem(
                (f"+η{axis}" if angular else f"+{axis}"), color=colour,
                anchor=(0, 1),
            )
            self.plot.addItem(label)
            self.basis_labels.append(label)
        self.origin = self.plot.plot([0], [0], pen=None, symbol="o", symbolSize=6,
                                     symbolBrush="#e2e8f0", symbolPen=None)
        self.status = _label()
        self.status.setMinimumHeight(42)
        self.status.setStyleSheet("color: #94a3b8;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(_label(
            "1 mrad canonical-angle circle; sample position held fixed."
            if angular else "1 nm position circle; canonical momentum held fixed."
        ))
        layout.addWidget(self.metrics)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.status)
        self.clear()

    def show_map(self, matrix, properties) -> None:
        # J_img: nm/nm. J_diff: m/rad * 0.001 rad * 1e6 µm/m.
        response = np.asarray(matrix, dtype=float) * (1000.0 if self.angular else 1.0)
        angle = np.linspace(0.0, 2.0 * np.pi, 129)
        outline = response @ np.stack((np.cos(angle), np.sin(angle)))
        self.outline.setData(outline[0], outline[1])
        self.origin.show()
        for index, (line, label) in enumerate(zip(self.basis_lines, self.basis_labels)):
            vector = response[:, index]
            line.setData([0.0, vector[0]], [0.0, vector[1]])
            label.setPos(*vector)
            label.setVisible(bool(np.linalg.norm(vector) > 1.0e-15))
        extent = float(np.max(np.abs(outline)))
        limit = 1.25 * extent if extent > 1.0e-15 else 1.0
        self.plot.setRange(xRange=(-limit, limit), yRange=(-limit, limit), padding=0)
        scale = properties.isotropic_scale
        metric = (
            f"Equivalent camera length {scale * 1000.0:.6g} mm"
            if self.angular else f"Equivalent magnification {scale:.6g}×"
        )
        if properties.rank == 2:
            orientation = f"Rotation {properties.orientation_deg:+.3f}°"
            shape = f"anisotropy {properties.anisotropy_ratio:.4g}"
            handedness = "mirrored" if properties.mirrored else "handedness preserved"
        else:
            orientation = "Rotation unavailable"
            shape = f"rank {properties.rank}/2"
            handedness = "handedness undefined"
        self.metrics.setText(f"{metric}\n{orientation} · {shape} · {handedness}")
        self.status.setText(
            "No first-order response: the circle collapses to the origin."
            if not np.any(response) else
            "Response below the numerical rank threshold; rotation is unavailable."
            if properties.rank == 0 else
            "Nearly collapsed to a line (numerical rank 1); rotation is unavailable."
            if properties.rank == 1 else
            "Purple: mapped circle. Cyan / amber: positive input X / Y directions."
        )

    def clear(self) -> None:
        self.outline.setData([], [])
        self.origin.hide()
        for line, label in zip(self.basis_lines, self.basis_labels):
            line.setData([], [])
            label.hide()
        self.metrics.setText("No calculated response")
        self.status.setText("")


class OpticalTransferOverview(QWidget):
    """Human-readable view of the existing cached Jacobian and its limits."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plane_kind = _label("No calculated target plane")
        self.plane_kind.setObjectName("opticalTransferPlaneKind")
        self.plane_kind.setToolTip(plane_equation_tooltip("pending"))
        self.interpretation = _label()
        self.calibration = _label()
        self.calibration.setStyleSheet("color: #94a3b8;")
        self.position_response = _ResponsePanel(angular=False)
        self.angle_response = _ResponsePanel(angular=True)
        responses = QHBoxLayout()
        responses.setSpacing(18)
        responses.addWidget(self.position_response, 1)
        responses.addWidget(self.angle_response, 1)
        self.conjugacy = _label()
        self.conjugacy.setStyleSheet("color: #cbd5e1;")
        footer = _label(
            "Linear response illustrations, not beam intensity or diffraction spots. "
            "Independent plot scales. Coordinates are relative to the reference ray, "
            "in column X/Y. Canonical angle η is transverse canonical momentum divided "
            "by reference momentum p0; it can differ from ray slope in a magnetic field. "
            "First-order, near-axis model; nonlinear aberrations, "
            "aperture clipping and the curved Energy Filter path are not shown."
        )
        footer.setStyleSheet("color: #94a3b8;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.plane_kind)
        layout.addWidget(self.interpretation)
        layout.addWidget(self.calibration)
        layout.addLayout(responses, 1)
        layout.addWidget(self.conjugacy)
        layout.addWidget(footer)

    def show_record(self, record, *, provisional_polarity: bool = False) -> None:
        transfer = record.transfer
        kind, image_residual, diffraction_residual = classify_sample_plane_transfer(transfer)
        titles = {
            "image": "Image plane · canonical-angle contribution is near zero",
            "diffraction": "Diffraction plane · position contribution is near zero",
            "mixed": "Mixed plane · position and canonical angle both contribute",
            "degenerate": "Degenerate response · both contributions are near zero",
        }
        colour = "#fbbf24" if kind in {"mixed", "degenerate"} else "#4ade80"
        self.plane_kind.setText(titles[kind])
        self.plane_kind.setToolTip(plane_equation_tooltip(kind))
        self.plane_kind.setStyleSheet(f"color: {colour}; font-size: 18px; font-weight: 600;")
        explanations = {
            "image": "Within the model tolerance, a sample point maps to one target position regardless of its small canonical-angle variation.",
            "diffraction": "Within the model tolerance, canonical angle determines target position with little dependence on the sample position.",
            "mixed": "This target is not a pure sample-image or diffraction plane under the current optics, regardless of its name or selected mode.",
            "degenerate": "A collapsed mapping cannot define a useful image or diffraction orientation.",
        }
        self.interpretation.setText(explanations[kind])
        frame = record.detector_frame
        frame_note = (
            "Column X/Y simulation reference; not a measured detector-axis calibration."
            if frame.status == "column_coordinates" else
            "Plots use column X/Y; calibrated detector-axis conversion is in Numeric details."
            if frame.is_calibrated else
            "Plots use column X/Y; absolute detector orientation is uncalibrated."
        )
        if provisional_polarity:
            frame_note += " Lens field polarities include provisional model assumptions."
        self.calibration.setText(frame_note)
        self.position_response.show_map(transfer.j_img, record.image_properties)
        self.angle_response.show_map(transfer.j_diff_m_per_rad, record.diffraction_properties)
        self.conjugacy.setText(
            f"Image-plane test: canonical-angle response {image_residual:.4g} m/rad "
            f"(≤ {IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD:g})   |   "
            f"Diffraction-plane test: position response {diffraction_residual:.4g} "
            f"(≤ {DIFFRACTION_CONJUGACY_TOLERANCE:g}). "
            "These use different units and are not directly comparable."
        )
        self.conjugacy.setToolTip(plane_equation_tooltip(kind))

    def clear(self) -> None:
        self.plane_kind.setText("No calculated target plane")
        self.plane_kind.setToolTip(plane_equation_tooltip("pending"))
        self.interpretation.clear()
        self.calibration.clear()
        self.conjugacy.clear()
        self.conjugacy.setToolTip(plane_equation_tooltip("pending"))
        self.position_response.clear()
        self.angle_response.clear()
