"""User-level coupled lens adjustments for the Direct Alignment page."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from temsim.operating_modes import (
    DirectAlignmentDefinition,
    OperatingModeCatalog,
    load_operating_mode_catalog,
)


_CONTROL_NAMES = {
    "nanoprobe_convergence": (
        "nanoprobeConvergenceTarget",
        "applyNanoprobeConvergence",
    ),
    "microprobe_illumination": (
        "microprobeIlluminationTarget",
        "applyMicroprobeIllumination",
    ),
    "image_magnification": (
        "imageMagnificationTarget",
        "applyImageMagnification",
    ),
    "diffraction_camera_length": (
        "cameraLengthTarget",
        "applyCameraLength",
    ),
}

_CURRENT_METRICS = {
    "nanoprobe_convergence": "sample_convergence_95_mrad",
    "microprobe_illumination": "sample_illumination_diameter_95_um",
    "image_magnification": "magnification",
    "diffraction_camera_length": "effective_camera_length_m",
}

_MODE_LABELS = {
    "nano_probe": "Nanoprobe (STEM)",
    "micro_probe": "Microprobe (TEM)",
    "imaging": "Image",
    "diffraction": "Diffraction",
}


@dataclass(slots=True)
class _AlignmentControl:
    definition: DirectAlignmentDefinition
    group: QGroupBox
    target: QDoubleSpinBox
    apply_button: QPushButton
    availability: QLabel
    current: QLabel
    result: QLabel


class DirectAlignmentPanel(QWidget):
    """Four mode-gated controls backed by the operating-mode catalog."""

    adjustment_requested = Signal(str, float)

    def __init__(
        self,
        catalog: OperatingModeCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("directAlignmentPanel")
        self._catalog = catalog or load_operating_mode_catalog()
        self._state = None
        self._available_mode_keys: set[str] | None = None
        self._metrics: dict[str, object] = {}
        self._controls: dict[str, _AlignmentControl] = {}
        self._busy_key: str | None = None

        introduction = QLabel(
            "Direct Alignment changes user-facing optical values while solving "
            "the listed lenses together. Nanoprobe convergence is the "
            "95%-current semi-angle relative to the chief ray, so Larmor "
            "rotation does not change its definition. Requested ranges may be "
            "unreachable with the current non-OEM field limits and conjugate "
            "constraints; an unsuccessful solve restores the previous lens "
            "values."
        )
        introduction.setObjectName("directAlignmentDescription")
        introduction.setWordWrap(True)
        introduction.setStyleSheet("color: #475569; font-weight: 600;")

        self.result_status = QLabel("Select an active operating mode and target.")
        self.result_status.setObjectName("directAlignmentStatus")
        self.result_status.setWordWrap(True)
        self.result_status.setStyleSheet("color: #475569; font-weight: 600;")
        # A short alias is convenient for callers and tests without creating a
        # second, potentially inconsistent status widget.
        self.status = self.result_status

        self._control_host = QWidget()
        self._control_layout = QVBoxLayout(self._control_host)
        self._control_layout.setContentsMargins(0, 0, 0, 0)
        self._control_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setObjectName("directAlignmentScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._control_host)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(introduction)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.result_status)

        self.set_catalog(self._catalog)

    @property
    def controls(self) -> Mapping[str, _AlignmentControl]:
        """Read-only view used by the AssemblyPanel and GUI tests."""

        return self._controls

    def set_catalog(self, catalog: OperatingModeCatalog) -> None:
        """Rebuild controls from the currently loaded TOML-backed catalog."""

        self._catalog = catalog
        while self._control_layout.count():
            item = self._control_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._controls = {}

        for definition in catalog.direct_alignments:
            self._add_control(definition)
        self._control_layout.addStretch(1)
        self._update_mode_gating()
        self.update_metrics(self._metrics)

    def _add_control(self, definition: DirectAlignmentDefinition) -> None:
        try:
            target_object_name, button_object_name = _CONTROL_NAMES[
                definition.key
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Direct Alignment control {definition.key!r}"
            ) from exc

        group = QGroupBox(definition.name)
        group.setObjectName(f"{definition.key}Group")
        group.setToolTip(
            f"{definition.calibration_status}\n"
            f"{definition.calibration_reference}\n"
            f"Coupled devices: {', '.join(definition.devices)}"
        )
        form = QFormLayout(group)

        target = QDoubleSpinBox()
        target.setObjectName(target_object_name)
        target.setDecimals(4 if definition.unit == "m" else 3)
        target.setRange(definition.minimum, definition.maximum)
        target.setValue(definition.default_value)
        target.setSuffix(f" {definition.unit}")
        target.setKeyboardTracking(False)
        target.setAccelerated(True)
        target.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )
        target.setToolTip(
            f"Requested range: {definition.minimum:g} to "
            f"{definition.maximum:g} {definition.unit}. The requested range "
            "may be unreachable with the current field limits."
        )

        apply_button = QPushButton("Apply coupled adjustment")
        apply_button.setObjectName(button_object_name)

        availability = QLabel()
        availability.setWordWrap(True)
        availability.setStyleSheet("color: #64748b; font-weight: 600;")

        current = QLabel(self._unavailable_current_text(definition.key))
        current.setWordWrap(True)
        current.setStyleSheet("color: #334155;")

        result = QLabel()
        result.setWordWrap(True)
        result.setStyleSheet("color: #475569;")

        form.addRow("Target", target)
        form.addRow(apply_button)
        form.addRow(availability)
        form.addRow(current)
        form.addRow(result)

        control = _AlignmentControl(
            definition=definition,
            group=group,
            target=target,
            apply_button=apply_button,
            availability=availability,
            current=current,
            result=result,
        )
        self._controls[definition.key] = control
        apply_button.clicked.connect(
            lambda _checked=False, key=definition.key: self._request(key)
        )
        self._control_layout.addWidget(group)

        # Public attribute names mirror the alignment keys and make the live
        # widgets easy to address without relying on QObject tree searches.
        setattr(self, f"{definition.key}_target", target)
        setattr(self, f"apply_{definition.key}", apply_button)

    def _request(self, key: str) -> None:
        control = self._controls[key]
        if not control.apply_button.isEnabled():
            return
        value = float(control.target.value())
        self.result_status.setStyleSheet(
            "color: #475569; font-weight: 600;"
        )
        self.result_status.setText(
            f"Solving {control.definition.name}: {value:g} "
            f"{control.definition.unit}..."
        )
        self.adjustment_requested.emit(key, value)

    def set_state(self, state, available_mode_keys=None) -> None:
        """Update mode gating from the applied microscope state."""

        self._state = state
        self._available_mode_keys = (
            None
            if available_mode_keys is None
            else {str(value) for value in available_mode_keys}
        )
        self._update_mode_gating()

    @staticmethod
    def _applied_mode_keys(state) -> set[str]:
        if state is None:
            return set()

        probe_value = str(getattr(state, "probe_mode", "")).lower()
        if probe_value not in {"nano_probe", "micro_probe"}:
            illumination = str(
                getattr(state, "illumination_mode", "")
            ).upper()
            probe_value = {
                "STEM": "nano_probe",
                "TEM": "micro_probe",
            }.get(illumination, "")

        projector_value = str(
            getattr(state, "projector_mode", "")
        ).lower()
        projector_value = {
            "image": "imaging",
            "imaging": "imaging",
            "diffraction": "diffraction",
        }.get(projector_value, "")
        return {value for value in (probe_value, projector_value) if value}

    def _update_mode_gating(self) -> None:
        active_modes = self._applied_mode_keys(self._state)
        lenses = {
            str(getattr(lens, "key", "")): lens
            for lens in getattr(self._state, "lenses", ())
        }
        for control in self._controls.values():
            required = control.definition.mode_key
            mode_available = (
                self._available_mode_keys is None
                or required in self._available_mode_keys
            )
            devices_ready = all(
                key in lenses
                and bool(getattr(lenses[key], "enabled", True))
                for key in control.definition.devices
            )
            active = (
                required in active_modes
                and mode_available
                and devices_ready
                and self._busy_key is None
            )
            control.target.setEnabled(active)
            control.apply_button.setEnabled(active)
            mode_label = _MODE_LABELS.get(required, required)
            if self._busy_key is not None:
                control.availability.setText(
                    "A coupled Direct Alignment solve is running."
                )
                control.availability.setStyleSheet(
                    "color: #0369a1; font-weight: 600;"
                )
            elif active:
                control.availability.setText(f"Active in {mode_label} mode.")
                control.availability.setStyleSheet(
                    "color: #15803d; font-weight: 600;"
                )
            elif not mode_available:
                control.availability.setText(
                    f"No calibrated {mode_label} preset is available for "
                    "the selected assembly."
                )
                control.availability.setStyleSheet(
                    "color: #64748b; font-weight: 600;"
                )
            elif not devices_ready:
                control.availability.setText(
                    "The selected assembly does not contain every coupled "
                    "optical device required by this adjustment."
                )
                control.availability.setStyleSheet(
                    "color: #64748b; font-weight: 600;"
                )
            else:
                control.availability.setText(
                    f"Available only in {mode_label} mode; apply that "
                    "operating preset first."
                )
                control.availability.setStyleSheet(
                    "color: #64748b; font-weight: 600;"
                )

    def set_busy(self, key: str | None) -> None:
        """Disable all requests while one background solve is active."""

        self._busy_key = None if key is None else str(key)
        self._update_mode_gating()

    def show_status_message(self, message: str, *, error: bool = False) -> None:
        """Replace a pending status after cancellation or worker failure."""

        colour = "#b91c1c" if error else "#475569"
        self.result_status.setStyleSheet(
            f"color: {colour}; font-weight: 600;"
        )
        self.result_status.setText(str(message))

    @staticmethod
    def _metric_mapping(metrics) -> dict[str, object]:
        if metrics is None:
            return {}
        if hasattr(metrics, "simulation"):
            metrics = metrics.simulation.metrics
        elif hasattr(metrics, "metrics"):
            metrics = metrics.metrics
        return dict(metrics)

    @staticmethod
    def _finite_value(metrics: Mapping[str, object], key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _unavailable_current_text(key: str) -> str:
        return {
            "nanoprobe_convergence": (
                "Current 95%-current semi-angle: unavailable."
            ),
            "microprobe_illumination": (
                "Current 95%-current illuminated diameter: unavailable."
            ),
            "image_magnification": (
                "Current physical image magnification: unavailable."
            ),
            "diffraction_camera_length": (
                "Current effective camera length: unavailable."
            ),
        }[key]

    def update_metrics(self, metrics) -> None:
        """Show current values measured by the latest accepted calculation."""

        self._metrics = self._metric_mapping(metrics)
        for key, control in self._controls.items():
            value = self._finite_value(self._metrics, _CURRENT_METRICS[key])
            if value is None:
                control.current.setText(self._unavailable_current_text(key))
                continue
            if key == "nanoprobe_convergence":
                waist = self._finite_value(
                    self._metrics, "sample_waist_offset_mm"
                )
                suffix = (
                    f"; sample waist offset {waist:.5g} mm"
                    if waist is not None else ""
                )
                text = (
                    f"Current 95%-current semi-angle: {value:.6g} mrad"
                    f"{suffix}."
                )
            elif key == "microprobe_illumination":
                angle = self._finite_value(
                    self._metrics, "sample_convergence_95_mrad"
                )
                curvature = self._finite_value(
                    self._metrics, "sample_wavefront_curvature_per_m"
                )
                details = []
                if angle is not None:
                    details.append(
                        f"95%-current semi-angle {angle:.6g} mrad"
                    )
                if curvature is not None:
                    details.append(
                        f"wavefront curvature {curvature:.6g} 1/m"
                    )
                suffix = f"; {', '.join(details)}" if details else ""
                text = (
                    "Current 95%-current illuminated diameter: "
                    f"{value:.6g} um{suffix}."
                )
            elif key == "image_magnification":
                text = (
                    "Current physical image magnification: "
                    f"{value:.6g} x."
                )
            else:
                text = f"Current effective camera length: {value:.6g} m."
            control.current.setText(text)

    @staticmethod
    def _result_attribute(result, name: str, default=None):
        if isinstance(result, Mapping):
            return result.get(name, default)
        return getattr(result, name, default)

    def show_result(self, result) -> None:
        """Display a completed coupled solve without assuming it succeeded."""

        key = str(self._result_attribute(result, "key", ""))
        control = self._controls.get(key)
        if control is None:
            self.result_status.setStyleSheet(
                "color: #b91c1c; font-weight: 600;"
            )
            self.result_status.setText(
                f"Unknown Direct Alignment result: {key or 'missing key'}."
            )
            return

        success = bool(self._result_attribute(result, "success", False))
        requested = float(
            self._result_attribute(result, "requested", control.target.value())
        )
        achieved = float(self._result_attribute(result, "achieved", math.nan))
        unit = str(
            self._result_attribute(
                result, "unit", control.definition.unit
            )
        )
        message = str(self._result_attribute(result, "message", "")).strip()
        strengths = self._result_attribute(result, "strengths", {}) or {}
        strength_text = ", ".join(
            f"{lens_key}={float(value):.5g}%"
            for lens_key, value in dict(strengths).items()
        )
        heading = "Applied" if success else "Not applied"
        summary = (
            f"{heading}: requested {requested:.6g} {unit}; achieved "
            f"{achieved:.6g} {unit}."
        )
        if message:
            summary += f" {message}"
        control.result.setText(summary)
        control.result.setToolTip(strength_text)
        colour = "#15803d" if success else "#b91c1c"
        control.result.setStyleSheet(f"color: {colour}; font-weight: 600;")
        self.result_status.setStyleSheet(
            f"color: {colour}; font-weight: 600;"
        )
        self.result_status.setText(summary)

        if success and math.isfinite(achieved):
            current_text = {
                "nanoprobe_convergence": (
                    f"Solved 95%-current semi-angle: {achieved:.6g} {unit}."
                ),
                "microprobe_illumination": (
                    "Solved 95%-current illuminated diameter: "
                    f"{achieved:.6g} {unit}."
                ),
                "image_magnification": (
                    "Solved physical image magnification: "
                    f"{achieved:.6g} {unit}."
                ),
                "diffraction_camera_length": (
                    f"Solved effective camera length: {achieved:.6g} {unit}."
                ),
            }[key]
            control.current.setText(current_text)
