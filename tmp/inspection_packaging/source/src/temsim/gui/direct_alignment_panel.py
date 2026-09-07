"""User-level coupled lens adjustments for the Direct Alignment page."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox
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
    "nano_probe": "Nanoprobe",
    "micro_probe": "Microprobe",
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
    """Catalog-backed, mode-gated user alignment controls."""

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
        self._selected_mode_keys: set[str] | None = None
        self._metrics: dict[str, object] = {}
        self._controls: dict[str, _AlignmentControl] = {}
        self._busy_key: str | None = None

        introduction = QLabel(
            "Adjust sample illumination and image / diffraction projection."
        )
        introduction.setToolTip(
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
        introduction.setStyleSheet("font-weight: 600;")

        self.mode_status = QLabel()
        self.mode_status.setObjectName("directAlignmentAppliedModes")
        self.mode_status.setWordWrap(True)

        target_group = self.projector_calibration_group = QGroupBox(
            "Projector diffraction calibration"
        )
        target_form = QFormLayout(target_group)
        target_notice = QLabel(
            "Camera length is measured at the active projection reference plane."
        )
        target_notice.setToolTip(
            "Camera length is one independent D/I/P1/P2 projector setting. "
            "The active reference plane is reported with its measured value. "
            "HAADF, DF and BF retain distinct Z positions and collection-angle "
            "transfers; detector insertion and readout do not select a "
            "projector preset."
        )
        target_notice.setWordWrap(True)
        target_form.addRow(target_notice)
        self.projector_field_calibration = QLabel()
        self.projector_field_calibration.setObjectName(
            "projectorFieldCalibrationSummary"
        )
        self.projector_field_calibration.setWordWrap(True)
        self.projector_field_calibration.setStyleSheet(
            "color: #92400e; background: #fffbeb; "
            "border: 1px solid #f59e0b; padding: 5px;"
        )
        target_form.addRow(self.projector_field_calibration)

        self.result_status = QLabel("Select an active operating mode and target.")
        self.result_status.setObjectName("directAlignmentStatus")
        self.result_status.setWordWrap(True)
        self.result_status.setStyleSheet("font-weight: 600;")
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
        layout.addWidget(self.mode_status)
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
            if widget is not None and widget is not self.projector_calibration_group:
                widget.deleteLater()
        self._controls = {}

        for definition in catalog.direct_alignments:
            # Retire this control even when opening an older external catalog.
            if definition.key == "spot_size_current_limit":
                continue
            self._add_control(definition)
        self._control_layout.addWidget(self.projector_calibration_group)
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
        coupled_description = (
            f"Coupled devices: {', '.join(definition.devices)}"
            if definition.devices
            else "No lens field is changed by this ideal current control."
        )
        group.setToolTip(
            f"{definition.calibration_status}\n"
            f"{definition.calibration_reference}\n"
            f"{coupled_description}"
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
        availability.setStyleSheet("font-weight: 600;")

        current = QLabel(self._unavailable_current_text(definition.key))
        current.setWordWrap(True)

        result = QLabel()
        result.setWordWrap(True)

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
        self._update_mode_gating()
        if control.group.isHidden() or not control.apply_button.isEnabled():
            return
        value = float(control.target.value())
        self.result_status.setStyleSheet(
            "font-weight: 600;"
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
        from temsim.optics.direct_alignment import (
            projector_field_calibration_rows,
        )
        rows = projector_field_calibration_rows(state)
        calibration_detail = "\n".join(
                f"{row['key']} {row['maximum_peak_field_t']:.4g} T, "
                f"half-width {row['field_half_width_mm']:.4g} mm, "
                f"limit {row['maximum_excitation_percent']:.4g}%, "
                f"{row['status']}"
                for row in rows
        )
        provisional_count = sum(
            "provisional" in str(row["status"]).lower() for row in rows
        )
        self.projector_field_calibration.setText(
            f"Projector field limits: {len(rows)} lenses | "
            f"{provisional_count} provisional"
        )
        self.projector_field_calibration.setToolTip(
            calibration_detail
            + "\n\n"
            + "\n".join(f"{row['key']}: {row['source']}" for row in rows)
        )
        self._update_mode_gating()

    def set_selected_mode_keys(
        self, condenser_key: str | None, projector_key: str | None
    ) -> None:
        """Keep pending selector choices separate from the applied optics."""

        self._selected_mode_keys = {
            str(key) for key in (condenser_key, projector_key) if key
        }
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
        pending_mode_change = (
            self._selected_mode_keys is not None
            and not self._selected_mode_keys.issubset(active_modes)
        )
        active_probe = next(
            (key for key in ("micro_probe", "nano_probe") if key in active_modes),
            None,
        )
        active_projector = next(
            (key for key in ("imaging", "diffraction") if key in active_modes),
            None,
        )
        if active_probe and active_projector:
            mode_text = (
                f"Applied: {_MODE_LABELS[active_probe]} + "
                f"{_MODE_LABELS[active_projector]}."
            )
        else:
            mode_text = "Apply an optical preset to show its alignment controls."
        if pending_mode_change:
            mode_text += (
                " Selection changed: apply the selected optical preset "
                "before adjusting Direct Alignment."
            )
        self.mode_status.setText(mode_text)
        self.projector_calibration_group.setVisible("diffraction" in active_modes)
        lenses = {
            str(getattr(lens, "key", "")): lens
            for lens in getattr(self._state, "lenses", ())
        }
        for control in self._controls.values():
            required_modes = set(control.definition.active_mode_keys)
            mode_active = bool(required_modes & active_modes)
            control.group.setVisible(mode_active)
            mode_available = (
                self._available_mode_keys is None
                or bool(required_modes & self._available_mode_keys)
            )
            devices_ready = all(
                key in lenses
                and bool(getattr(lenses[key], "enabled", True))
                for key in control.definition.devices
            )
            active = (
                mode_active
                and mode_available
                and devices_ready
                and self._busy_key is None
                and not pending_mode_change
            )
            control.target.setEnabled(active)
            control.apply_button.setEnabled(active)
            mode_label = " or ".join(
                _MODE_LABELS.get(required, required)
                for required in control.definition.active_mode_keys
            )
            if self._busy_key is not None:
                control.availability.setText(
                    "A coupled Direct Alignment solve is running."
                )
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )
            elif pending_mode_change:
                control.availability.setText(
                    "Apply the selected optical preset first."
                )
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )
            elif active:
                control.availability.setText(f"Active in {mode_label} mode.")
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )
            elif not mode_available:
                control.availability.setText(
                    f"No calibrated {mode_label} preset is available for "
                    "the selected assembly."
                )
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )
            elif not devices_ready:
                control.availability.setText(
                    "The selected assembly does not contain every coupled "
                    "optical device required by this adjustment."
                )
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )
            else:
                control.availability.setText(
                    f"Available only in {mode_label} mode; apply that "
                    "operating preset first."
                )
                control.availability.setStyleSheet(
                    "font-weight: 600;"
                )

    def set_busy(self, key: str | None) -> None:
        """Disable all requests while one background solve is active."""

        self._busy_key = None if key is None else str(key)
        self._update_mode_gating()

    def show_status_message(self, message: str, *, error: bool = False) -> None:
        """Replace a pending status after cancellation or worker failure."""

        self.result_status.setStyleSheet(
            "color: #991b1b; background: #fef2f2; font-weight: 600;"
            if error else "font-weight: 600;"
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
                angle_99 = self._finite_value(
                    self._metrics, "sample_convergence_99_mrad"
                )
                curvature = self._finite_value(
                    self._metrics, "sample_wavefront_curvature_per_m"
                )
                details = []
                if angle is not None:
                    details.append(
                        f"95%-current semi-angle {angle:.6g} mrad"
                    )
                if angle_99 is not None:
                    details.append(
                        f"99%-current semi-angle {angle_99:.6g} mrad"
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
                residual = self._finite_value(
                    self._metrics, "diffraction_conjugacy_residual"
                )
                depth = self._finite_value(
                    self._metrics, "diffraction_focus_depth_mm"
                )
                target_key = self._metrics.get(
                    "transfer_analysis_plane_key"
                )
                details = []
                if target_key:
                    details.append(f"target {str(target_key).upper()}")
                if residual is not None:
                    details.append(f"||A||2 {residual:.6g}")
                if depth is not None:
                    details.append(
                        f"local conjugacy depth {depth:.6g} mm"
                    )
                suffix = f"; {', '.join(details)}" if details else ""
                text = (
                    f"Current effective camera length: {value:.6g} m"
                    f"{suffix}."
                )
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
                "color: #991b1b; background: #fef2f2; font-weight: 600;"
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
        status_style = (
            "color: #166534; background: #f0fdf4; font-weight: 600;"
            if success else
            "color: #991b1b; background: #fef2f2; font-weight: 600;"
        )
        control.result.setStyleSheet(status_style)
        self.result_status.setStyleSheet(status_style)
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
