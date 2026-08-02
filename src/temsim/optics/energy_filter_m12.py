"""Independent entrance and exit M12 units for the Energy Filter."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from temsim.component_keys import (
    ENERGY_FILTER_ENTRANCE_M12,
    ENERGY_FILTER_EXIT_M12,
)
from temsim.optics.twelve_pole_element import (
    LocalCoordinateFrame,
    TwelvePoleElement,
)
from temsim.physics.finite_multipole_field import (
    FiniteMultipoleField,
    SoftEdgeEnvelope,
)
from temsim.physics.multipole_field import MultipoleField
from temsim.physics.relativistic_lorentz import (
    ELEMENTARY_CHARGE_C,
    momentum_from_kinetic_energy_ev,
)


MINIMUM_MATCH_VOLTAGE_KV = 30.0
MAXIMUM_MATCH_VOLTAGE_KV = 300.0
DEFAULT_REFERENCE_VOLTAGE_KV = 300.0


def magnetic_rigidity_t_m(voltage_kv):
    """Return relativistic electron magnetic rigidity for an HT setting."""

    voltage = float(voltage_kv)
    if not math.isfinite(voltage) or voltage <= 0.0:
        raise ValueError(
            "Accelerating voltage must be finite and positive."
        )
    momentum = momentum_from_kinetic_energy_ev(
        voltage * 1000.0,
        [0.0, 0.0, 1.0],
    )
    return float(np.linalg.norm(momentum)) / ELEMENTARY_CHARGE_C


def rigidity_scale(target_voltage_kv, reference_voltage_kv):
    return (
        magnetic_rigidity_t_m(target_voltage_kv)
        / magnetic_rigidity_t_m(reference_voltage_kv)
    )


def _coefficient_array(values, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (MultipoleField.ORDER_COUNT,):
        raise ValueError(f"{name} must contain six coefficients.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} coefficients must all be finite.")
    return result.copy()


@dataclass
class M12VoltageCalibration:
    """Independent reference coefficients and fixed trims for one M12."""

    calibration_id: str
    reference_voltage_kv: float = DEFAULT_REFERENCE_VOLTAGE_KV
    reference_normal_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros(
            MultipoleField.ORDER_COUNT,
            dtype=float,
        )
    )
    reference_skew_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros(
            MultipoleField.ORDER_COUNT,
            dtype=float,
        )
    )
    normal_trim_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros(
            MultipoleField.ORDER_COUNT,
            dtype=float,
        )
    )
    skew_trim_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros(
            MultipoleField.ORDER_COUNT,
            dtype=float,
        )
    )

    def __post_init__(self):
        if (
            not isinstance(self.calibration_id, str)
            or not self.calibration_id.strip()
        ):
            raise ValueError("M12 calibration ID must not be empty.")
        reference_voltage = float(self.reference_voltage_kv)
        if not math.isfinite(reference_voltage) or reference_voltage <= 0.0:
            raise ValueError(
                "M12 reference voltage must be finite and positive."
            )
        self.reference_voltage_kv = reference_voltage
        self.reference_normal_coefficients = _coefficient_array(
            self.reference_normal_coefficients,
            "Reference normal",
        )
        self.reference_skew_coefficients = _coefficient_array(
            self.reference_skew_coefficients,
            "Reference skew",
        )
        self.normal_trim_coefficients = _coefficient_array(
            self.normal_trim_coefficients,
            "Normal trim",
        )
        self.skew_trim_coefficients = _coefficient_array(
            self.skew_trim_coefficients,
            "Skew trim",
        )

    def set_reference_from_field(self, multipole_field, voltage_kv):
        if not isinstance(multipole_field, MultipoleField):
            raise TypeError(
                "M12 reference capture requires a MultipoleField."
            )
        voltage = float(voltage_kv)
        magnetic_rigidity_t_m(voltage)
        self.reference_voltage_kv = voltage
        self.reference_normal_coefficients = (
            multipole_field.normal_coefficients
            - self.normal_trim_coefficients
        )
        self.reference_skew_coefficients = (
            multipole_field.skew_coefficients
            - self.skew_trim_coefficients
        )
        return self

    def coefficients_for_voltage(self, voltage_kv):
        scale = rigidity_scale(
            voltage_kv,
            self.reference_voltage_kv,
        )
        return (
            self.reference_normal_coefficients * scale
            + self.normal_trim_coefficients,
            self.reference_skew_coefficients * scale
            + self.skew_trim_coefficients,
            scale,
        )


@dataclass
class EnergyFilterM12Component(TwelvePoleElement):
    """One Energy Filter M12 with its own voltage calibration."""

    role: str = "entrance"
    calibration: M12VoltageCalibration = field(
        default_factory=lambda: M12VoltageCalibration(
            "unassigned_m12"
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if self.role not in {"entrance", "exit"}:
            raise ValueError("M12 role must be 'entrance' or 'exit'.")
        expected_key = (
            ENERGY_FILTER_ENTRANCE_M12
            if self.role == "entrance"
            else ENERGY_FILTER_EXIT_M12
        )
        if self.key != expected_key:
            raise ValueError(
                f"{self.name} key does not match its {self.role} role."
            )
        if not isinstance(self.field_backend, FiniteMultipoleField):
            raise TypeError(
                "Analytic Energy Filter M12 requires "
                "FiniteMultipoleField."
            )
        if not isinstance(self.calibration, M12VoltageCalibration):
            raise TypeError(
                "Energy Filter M12 requires M12VoltageCalibration."
            )

    @property
    def multipole_field(self):
        return self.field_backend.multipole_field

    def apply_voltage_match(self, voltage_kv):
        normal, skew, scale = (
            self.calibration.coefficients_for_voltage(voltage_kv)
        )
        for order in range(1, MultipoleField.MAX_ORDER + 1):
            self.multipole_field.set_component(
                order,
                normal=normal[order - 1],
                skew=skew[order - 1],
            )
        return scale


def _create_m12(role, reference_voltage_kv):
    if role == "entrance":
        key = ENERGY_FILTER_ENTRANCE_M12
        name = "Entrance M12"
    elif role == "exit":
        key = ENERGY_FILTER_EXIT_M12
        name = "Exit M12"
    else:
        raise ValueError("M12 role must be 'entrance' or 'exit'.")
    reference_voltage = float(reference_voltage_kv)
    magnetic_rigidity_t_m(reference_voltage)
    return EnergyFilterM12Component(
        name=name,
        key=key,
        field_backend=FiniteMultipoleField(
            MultipoleField(),
            SoftEdgeEnvelope(
                length_m=20.0e-3,
                entrance_soft_edge_m=4.0e-3,
                exit_soft_edge_m=4.0e-3,
            ),
        ),
        frame=LocalCoordinateFrame(),
        bore_radius_m=7.5e-3,
        outer_radius_m=35.0e-3,
        pole_zero_angle_rad=0.0,
        enabled=True,
        role=role,
        calibration=M12VoltageCalibration(
            calibration_id=f"{key}_calibration",
            reference_voltage_kv=reference_voltage,
        ),
    )


def create_entrance_m12(
    reference_voltage_kv=DEFAULT_REFERENCE_VOLTAGE_KV,
):
    return _create_m12("entrance", reference_voltage_kv)


def create_exit_m12(
    reference_voltage_kv=DEFAULT_REFERENCE_VOLTAGE_KV,
):
    return _create_m12("exit", reference_voltage_kv)


def serialise_energy_filter_m12(component):
    if not isinstance(component, EnergyFilterM12Component):
        raise TypeError("Expected an EnergyFilterM12Component.")
    backend = component.field_backend
    calibration = component.calibration
    return {
        "name": component.name,
        "key": component.key,
        "role": component.role,
        "enabled": bool(component.enabled),
        "field": {
            "normal_coefficients": (
                component.multipole_field
                .normal_coefficients.tolist()
            ),
            "skew_coefficients": (
                component.multipole_field
                .skew_coefficients.tolist()
            ),
            "fringe_expansion_order": int(
                backend.fringe_expansion_order
            ),
        },
        "calibration": {
            "calibration_id": calibration.calibration_id,
            "reference_voltage_kv": float(
                calibration.reference_voltage_kv
            ),
            "reference_normal_coefficients": (
                calibration.reference_normal_coefficients.tolist()
            ),
            "reference_skew_coefficients": (
                calibration.reference_skew_coefficients.tolist()
            ),
            "normal_trim_coefficients": (
                calibration.normal_trim_coefficients.tolist()
            ),
            "skew_trim_coefficients": (
                calibration.skew_trim_coefficients.tolist()
            ),
        },
    }


def energy_filter_m12_from_dict(values, role, reference_voltage_kv):
    if not isinstance(values, dict):
        return _create_m12(role, reference_voltage_kv)
    expected_key = (
        ENERGY_FILTER_ENTRANCE_M12
        if role == "entrance"
        else ENERGY_FILTER_EXIT_M12
    )
    defaults = _create_m12(role, reference_voltage_kv)
    field_values = values.get("field", {})
    calibration_values = values.get("calibration", {})
    return EnergyFilterM12Component(
        name=str(values.get("name", defaults.name)),
        key=expected_key,
        field_backend=FiniteMultipoleField(
            MultipoleField(
                normal=field_values.get(
                    "normal_coefficients",
                    defaults.multipole_field.normal_coefficients,
                ),
                skew=field_values.get(
                    "skew_coefficients",
                    defaults.multipole_field.skew_coefficients,
                ),
            ),
            SoftEdgeEnvelope(
                length_m=defaults.field_backend.envelope.length_m,
                entrance_soft_edge_m=(
                    defaults.field_backend.envelope.entrance_soft_edge_m
                ),
                exit_soft_edge_m=(
                    defaults.field_backend.envelope.exit_soft_edge_m
                ),
            ),
            fringe_expansion_order=int(
                field_values.get(
                    "fringe_expansion_order",
                    defaults.field_backend.fringe_expansion_order,
                )
            ),
        ),
        frame=LocalCoordinateFrame(),
        bore_radius_m=float(defaults.bore_radius_m),
        outer_radius_m=float(defaults.outer_radius_m),
        pole_zero_angle_rad=float(defaults.pole_zero_angle_rad),
        enabled=bool(values.get("enabled", defaults.enabled)),
        role=role,
        calibration=M12VoltageCalibration(
            calibration_id=str(
                calibration_values.get(
                    "calibration_id",
                    f"{expected_key}_calibration",
                )
            ),
            reference_voltage_kv=float(
                calibration_values.get(
                    "reference_voltage_kv",
                    reference_voltage_kv,
                )
            ),
            reference_normal_coefficients=calibration_values.get(
                "reference_normal_coefficients",
                np.zeros(MultipoleField.ORDER_COUNT),
            ),
            reference_skew_coefficients=calibration_values.get(
                "reference_skew_coefficients",
                np.zeros(MultipoleField.ORDER_COUNT),
            ),
            normal_trim_coefficients=calibration_values.get(
                "normal_trim_coefficients",
                np.zeros(MultipoleField.ORDER_COUNT),
            ),
            skew_trim_coefficients=calibration_values.get(
                "skew_trim_coefficients",
                np.zeros(MultipoleField.ORDER_COUNT),
            ),
        ),
    )
