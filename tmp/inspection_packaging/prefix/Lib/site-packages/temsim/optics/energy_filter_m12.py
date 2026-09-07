"""Independent entrance and exit M12 units for the Energy Filter."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from temsim.component_keys import (
    ENERGY_FILTER_ENTRANCE_M12,
    ENERGY_FILTER_EXIT_M12,
    ENERGY_FILTER_MULTIPOLE_KEYS,
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
ILIAD_REFERENCE_RADIUS_M = 2.0e-3
# Solved reference fields at 300 kV.  Values are the field amplitude at
# ILIAD_REFERENCE_RADIUS_M, rather than raw SI coefficients, so the preset is
# easy to audit and remains numerically well-scaled.  The geometry is a
# documented reference calibration based on the public tapered-prism/ten-
# multipole topology; it is not claimed to reproduce proprietary factory
# excitation tables.
ILIAD_REFERENCE_QUADRUPOLE_FIELD_T = (
    -7.00701380e-3,
    -7.28743610e-4,
    4.36144547e-3,
    -4.67007982e-4,
    -1.22447848e-3,
    5.12576293e-7,
    6.78299991e-3,
    -6.89927e-3,
    5.49052e-3,
    -8.36259e-3,
)
ILIAD_REFERENCE_SEXTUPOLE_FIELD_T = (
    -4.38361777e-4,
    8.60942288e-4,
    2.83908043e-4,
    5.34073905e-5,
    -2.73658563e-4,
    6.60512471e-4,
    4.79346486e-4,
    0.0,
    0.0,
    0.0,
)


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
    """One independently powered 12-pole carrier in the Energy Filter.

    ``field_backend.length_m`` is the magnetic support length used by the
    solver. ``housing_length_m`` is a separate mechanical envelope used only
    by layout views and collision checks; it must never create extra field.
    """

    role: str = "entrance"
    housing_length_m: float = 22.0e-3
    calibration: M12VoltageCalibration = field(
        default_factory=lambda: M12VoltageCalibration(
            "unassigned_m12"
        )
    )

    def __post_init__(self):
        super().__post_init__()
        expected_by_role = {
            "entrance": ENERGY_FILTER_ENTRANCE_M12,
            "exit": ENERGY_FILTER_EXIT_M12,
            **{
                f"m{index:02d}": key
                for index, key in enumerate(
                    ENERGY_FILTER_MULTIPOLE_KEYS, start=1
                )
            },
        }
        expected_key = expected_by_role.get(self.role)
        if expected_key is None:
            raise ValueError(
                "M12 role must be entrance, exit, or m01 through m10."
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
        housing_length = float(self.housing_length_m)
        if not math.isfinite(housing_length):
            raise ValueError("Energy Filter M12 housing length must be finite.")
        if housing_length < self.length_m:
            raise ValueError(
                "Energy Filter M12 housing length cannot be shorter than "
                "its magnetic support length."
            )
        self.housing_length_m = housing_length

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


def _create_multipole(key, name, role, reference_voltage_kv):
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
        housing_length_m=22.0e-3,
        calibration=M12VoltageCalibration(
            calibration_id=f"{key}_calibration",
            reference_voltage_kv=reference_voltage,
        ),
    )


def _create_m12(role, reference_voltage_kv):
    if role == "entrance":
        key = ENERGY_FILTER_ENTRANCE_M12
        name = "Entrance M12"
    elif role == "exit":
        key = ENERGY_FILTER_EXIT_M12
        name = "Exit M12"
    else:
        raise ValueError("M12 role must be 'entrance' or 'exit'.")
    return _create_multipole(
        key, name, role, reference_voltage_kv
    )


def create_entrance_m12(
    reference_voltage_kv=DEFAULT_REFERENCE_VOLTAGE_KV,
):
    return _create_m12("entrance", reference_voltage_kv)


def create_exit_m12(
    reference_voltage_kv=DEFAULT_REFERENCE_VOLTAGE_KV,
):
    return _create_m12("exit", reference_voltage_kv)


def create_iliad_multipoles(
    reference_voltage_kv=DEFAULT_REFERENCE_VOLTAGE_KV,
):
    """Create ten independent carriers matching the public Iliad topology."""

    multipoles = [
        _create_multipole(
            key,
            f"Iliad Multipole {index:02d} (model index)",
            f"m{index:02d}",
            reference_voltage_kv,
        )
        for index, key in enumerate(ENERGY_FILTER_MULTIPOLE_KEYS, start=1)
    ]
    for element, quadrupole_t, sextupole_t in zip(
        multipoles,
        ILIAD_REFERENCE_QUADRUPOLE_FIELD_T,
        ILIAD_REFERENCE_SEXTUPOLE_FIELD_T,
    ):
        reference_normal = np.zeros(MultipoleField.ORDER_COUNT)
        reference_normal[1] = (
            quadrupole_t / ILIAD_REFERENCE_RADIUS_M
        )
        reference_normal[2] = (
            sextupole_t / ILIAD_REFERENCE_RADIUS_M**2
        )
        element.calibration.reference_normal_coefficients = (
            reference_normal
        )
        element.apply_voltage_match(reference_voltage_kv)
    return multipoles


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


def energy_filter_multipole_from_dict(
    values, index, reference_voltage_kv
):
    """Restore one of the ten Iliad carriers without legacy key migration."""

    index = int(index)
    if not 1 <= index <= len(ENERGY_FILTER_MULTIPOLE_KEYS):
        raise ValueError("Energy Filter multipole index must be 1 through 10.")
    key = ENERGY_FILTER_MULTIPOLE_KEYS[index - 1]
    role = f"m{index:02d}"
    defaults = _create_multipole(
        key,
        f"Iliad Multipole {index:02d} (model index)",
        role,
        reference_voltage_kv,
    )
    if not isinstance(values, dict):
        return defaults
    field_values = values.get("field", {})
    calibration_values = values.get("calibration", {})
    return EnergyFilterM12Component(
        name=str(values.get("name", defaults.name)),
        key=key,
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
            fringe_expansion_order=int(field_values.get(
                "fringe_expansion_order",
                defaults.field_backend.fringe_expansion_order,
            )),
        ),
        frame=LocalCoordinateFrame(),
        bore_radius_m=float(defaults.bore_radius_m),
        outer_radius_m=float(defaults.outer_radius_m),
        pole_zero_angle_rad=float(defaults.pole_zero_angle_rad),
        enabled=bool(values.get("enabled", defaults.enabled)),
        role=role,
        calibration=M12VoltageCalibration(
            calibration_id=str(calibration_values.get(
                "calibration_id", f"{key}_calibration"
            )),
            reference_voltage_kv=float(calibration_values.get(
                "reference_voltage_kv", reference_voltage_kv
            )),
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
