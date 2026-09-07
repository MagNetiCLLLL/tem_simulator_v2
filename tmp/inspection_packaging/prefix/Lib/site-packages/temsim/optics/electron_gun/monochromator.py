"""Finite crossed-field Wien monochromator owned by a cold FEG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Protocol, runtime_checkable

import numpy as np

from temsim import module_manifest
from temsim.component_keys import FEG_MONOCHROMATOR_WIEN
from temsim.optics.electron_gun.electrostatic import (
    _soft_window_with_derivatives,
)
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    SPEED_OF_LIGHT_M_PER_S,
)


def speed_from_kinetic_energy_ev(energy_ev):
    """Return the relativistic speed for a positive kinetic energy."""

    energy = max(0.0, float(energy_ev)) * ELEMENTARY_CHARGE_C
    rest = ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S**2
    gamma = 1.0 + energy / rest
    return SPEED_OF_LIGHT_M_PER_S * math.sqrt(
        max(0.0, 1.0 - 1.0 / (gamma * gamma))
    )


def _create_wien_element():
    module_path = "gun/FEG_Mono.toml"
    geometry = module_manifest.part_geometry(
        module_path, FEG_MONOCHROMATOR_WIEN
    )
    part = module_manifest.part_data(
        module_path, FEG_MONOCHROMATOR_WIEN
    )
    return FiniteWienElement(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        active_length_mm=float(part["active_length_mm"]),
    )


@runtime_checkable
class WienFieldProvider(Protocol):
    """Replaceable analytic/measured-map interface for one Wien element."""

    def field_at_global_positions_v_per_m(self, positions_m):
        ...

    def field_at_global_positions_t(self, positions_m):
        ...

    def potential_v_at_global_positions(self, positions_m):
        ...


@dataclass
class FiniteWienElement:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    active_length_mm: float
    name: str = "FEG Wien Monochromator"
    key: str = FEG_MONOCHROMATOR_WIEN
    soft_edge_mm: float = 2.0
    electric_field_v_per_m: float = field(
        default_factory=lambda: (
            speed_from_kinetic_energy_ev(4_000.0) * 30.0e-3
        )
    )
    magnetic_field_mt: float = 30.0
    electric_quadrupole_gradient_v_per_m2: float = 1.2e8
    field_center_offset_mm: float = 0.0
    enabled: bool = True
    colour: str = "#00897b"

    @property
    def label(self):
        return self.name

    @property
    def optical_reference_from_tip_mm(self):
        return (
            float(self.mechanical_center_from_tip_mm)
            + float(self.field_center_offset_mm)
        )

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * float(self.mechanical_clear_bore_diameter_mm)

    @property
    def kind(self):
        return "finite_crossed_field_wien"

    @property
    def shape_profile(self):
        return "wien_field_carrier"

    @property
    def field_support_mm(self):
        center = self.optical_reference_from_tip_mm
        half = 0.5 * float(self.active_length_mm)
        edge = float(self.soft_edge_mm)
        return center - half - edge, center + half + edge

    def validate(self):
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Wien housing length must be positive.")
        if self.active_length_mm <= 0.0 or self.soft_edge_mm <= 0.0:
            raise ValueError("Wien active length and soft edge must be positive.")
        if (
            self.active_length_mm + 2.0 * self.soft_edge_mm
            > self.mechanical_length_mm + 1.0e-12
        ):
            raise ValueError("Wien finite field must fit inside its housing.")
        if self.mechanical_clear_bore_diameter_mm <= 0.0:
            raise ValueError("Wien clear bore must be positive.")
        for value, label in (
            (self.electric_field_v_per_m, "electric field"),
            (self.magnetic_field_mt, "magnetic field"),
            (
                self.electric_quadrupole_gradient_v_per_m2,
                "electric quadrupole gradient",
            ),
            (self.field_center_offset_mm, "field-centre offset"),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"Wien {label} must be finite.")
        return self

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": (
                self.mechanical_center_from_tip_mm
            ),
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "active_length_mm": self.active_length_mm,
            "shape_profile": self.shape_profile,
        }


class AnalyticWienField:
    """Curl-free soft-edged electric field plus transverse magnetic field."""

    def __init__(self, element):
        self.element = element

    def _envelope(self, z_mm):
        component = self.element
        center = component.optical_reference_from_tip_mm
        half = 0.5 * component.active_length_mm
        return _soft_window_with_derivatives(
            z_mm,
            center - half,
            center + half,
            component.soft_edge_mm,
        )

    def potential_v_at_global_positions(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        if not self.element.enabled:
            return np.zeros(positions.shape[:-1], dtype=float)
        envelope = self._envelope(positions[..., 2] * 1000.0)[0]
        x = positions[..., 0]
        y = positions[..., 1]
        dipole = -float(self.element.electric_field_v_per_m) * x
        quadrupole = (
            -0.5
            * float(self.element.electric_quadrupole_gradient_v_per_m2)
            * (x * x - y * y)
        )
        return (dipole + quadrupole) * envelope

    def field_at_global_positions_v_per_m(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        result = np.zeros_like(positions)
        if not self.element.enabled:
            return result
        envelope, derivative_per_mm, _, _ = self._envelope(
            positions[..., 2] * 1000.0
        )
        strength = float(self.element.electric_field_v_per_m)
        gradient = float(
            self.element.electric_quadrupole_gradient_v_per_m2
        )
        x = positions[..., 0]
        y = positions[..., 1]
        result[..., 0] = (strength + gradient * x) * envelope
        result[..., 1] = -gradient * y * envelope
        result[..., 2] = (
            (
                strength * x
                + 0.5 * gradient * (x * x - y * y)
            )
            * derivative_per_mm
            * 1000.0
        )
        return result

    def field_at_global_positions_t(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        result = np.zeros_like(positions)
        if not self.element.enabled:
            return result
        envelope = self._envelope(positions[..., 2] * 1000.0)[0]
        result[..., 1] = (
            float(self.element.magnetic_field_mt) * 1.0e-3 * envelope
        )
        return result


@dataclass
class MonochromatorSlit:
    """Blade settings hosted by the physical C1 aperture mechanism."""

    name: str = "Monochromator Slit (C1 Aperture)"
    gap_um: float = 0.28
    maximum_gap_um: float = 1000.0
    centre_offset_um: float = 0.0
    inserted: bool = True
    colour: str = "#d81b60"

    def transmission_mask(self, x_m, y_m):
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        if not self.inserted:
            return np.ones_like(x, dtype=bool)
        half_gap_m = 0.5 * float(self.gap_um) * 1.0e-6
        centre_m = float(self.centre_offset_um) * 1.0e-6
        return np.abs(x - centre_m) <= half_gap_m

    def validate(self):
        if not 0.0 <= float(self.gap_um) <= float(self.maximum_gap_um):
            raise ValueError("Monochromator slit gap is outside its range.")
        if not math.isfinite(float(self.centre_offset_um)):
            raise ValueError("Monochromator slit centre must be finite.")
        return self


class CombinedElectricField:
    def __init__(self, base_field, wien_field):
        self.base_field = base_field
        self.wien_field = wien_field

    def field_at_global_positions_v_per_m(self, positions_m):
        return (
            self.base_field.field_at_global_positions_v_per_m(positions_m)
            + self.wien_field.field_at_global_positions_v_per_m(positions_m)
        )

    def potential_v_at_global_positions(self, positions_m):
        return (
            self.base_field.potential_v_at_global_positions(positions_m)
            + self.wien_field.potential_v_at_global_positions(positions_m)
        )


class CombinedMagneticField:
    def __init__(self, base_field, wien_field):
        self.base_field = base_field
        self.wien_field = wien_field

    def field_at_global_positions_t(self, positions_m):
        return (
            self.base_field.field_at_global_positions_t(positions_m)
            + self.wien_field.field_at_global_positions_t(positions_m)
        )


@dataclass
class WienMonochromatorAssembly:
    installed: bool = False
    installation_model_version: int = 3
    requested_pass_window_ev: float = 0.10
    matched_energy_ev: float = 4_000.0
    trace_step_mm: float = 0.05
    accelerator_restore_profile: dict | None = None
    wien: FiniteWienElement = field(default_factory=_create_wien_element)
    slit: MonochromatorSlit = field(default_factory=MonochromatorSlit)

    def __post_init__(self):
        self._field_provider_override = None

    @property
    def absent_bay_clearance_mm(self):
        lens = module_manifest.part_geometry(
            "gun/FEG.toml", "feg_electrostatic_lens"
        )
        accelerator = module_manifest.part_geometry(
            "gun/FEG.toml", "feg_accelerator"
        )
        return accelerator.start_z_mm - lens.end_z_mm

    @property
    def upstream_clearance_mm(self):
        lens = module_manifest.part_geometry(
            "gun/FEG_Mono.toml", "feg_electrostatic_lens"
        )
        wien = module_manifest.part_geometry(
            "gun/FEG_Mono.toml", FEG_MONOCHROMATOR_WIEN
        )
        return wien.start_z_mm - lens.end_z_mm

    @property
    def downstream_clearance_mm(self):
        wien = module_manifest.part_geometry(
            "gun/FEG_Mono.toml", FEG_MONOCHROMATOR_WIEN
        )
        accelerator = module_manifest.part_geometry(
            "gun/FEG_Mono.toml", "feg_accelerator"
        )
        return accelerator.start_z_mm - wien.end_z_mm

    @property
    def components(self):
        return (self.wien,)

    @property
    def field_provider(self):
        return (
            self._field_provider_override
            if self._field_provider_override is not None
            else AnalyticWienField(self.wien)
        )

    def set_field_provider(self, provider):
        if provider is not None and not isinstance(provider, WienFieldProvider):
            raise TypeError(
                "Replacement Wien field must implement electric, magnetic "
                "and potential evaluation."
            )
        self._field_provider_override = provider
        return self

    def match_to_energy(self, energy_ev):
        energy = max(0.0, float(energy_ev))
        magnetic_t = float(self.wien.magnetic_field_mt) * 1.0e-3
        self.wien.electric_field_v_per_m = (
            speed_from_kinetic_energy_ev(energy) * magnetic_t
        )
        self.matched_energy_ev = energy
        return self.wien.electric_field_v_per_m

    def validate(self):
        self.wien.validate()
        self.slit.validate()
        if int(self.installation_model_version) < 1:
            raise ValueError(
                "Monochromator installation model version must be positive."
            )
        for value, label in (
            (self.absent_bay_clearance_mm, "absent bay clearance"),
            (self.upstream_clearance_mm, "upstream clearance"),
            (self.downstream_clearance_mm, "downstream clearance"),
        ):
            if float(value) < 0.0:
                raise ValueError(
                    f"Monochromator {label} must not be negative."
                )
        if float(self.requested_pass_window_ev) <= 0.0:
            raise ValueError(
                "Requested monochromator pass window must be positive."
            )
        if float(self.matched_energy_ev) < 0.0:
            raise ValueError("Matched monochromator energy cannot be negative.")
        if float(self.trace_step_mm) <= 0.0:
            raise ValueError(
                "Monochromator tracing step must be positive."
            )
        return self

    def to_dict(self):
        return {
            "installed": bool(self.installed),
            "installation_model_version": int(
                self.installation_model_version
            ),
            "requested_pass_window_ev": float(
                self.requested_pass_window_ev
            ),
            "matched_energy_ev": float(self.matched_energy_ev),
            "trace_step_mm": float(self.trace_step_mm),
            "wien": {
                key: value
                for key, value in asdict(self.wien).items()
                if key not in {
                    "mechanical_center_from_tip_mm",
                    "mechanical_length_mm",
                    "mechanical_outer_diameter_mm",
                    "mechanical_clear_bore_diameter_mm",
                    "active_length_mm",
                }
            },
            "slit": asdict(self.slit),
        }


def monochromator_from_dict(data=None):
    if data is None:
        return WienMonochromatorAssembly().validate()
    values = dict(data)
    wien = _create_wien_element()
    for key, value in dict(values.get("wien", {})).items():
        if key in {
            "mechanical_center_from_tip_mm",
            "mechanical_length_mm",
            "mechanical_outer_diameter_mm",
            "mechanical_clear_bore_diameter_mm",
            "active_length_mm",
        }:
            continue
        if key in FiniteWienElement.__dataclass_fields__:
            setattr(wien, key, value)
    assembly = WienMonochromatorAssembly(
        installed=bool(values.get("installed", False)),
        installation_model_version=int(
            values.get("installation_model_version", 1)
        ),
        requested_pass_window_ev=float(
            values.get("requested_pass_window_ev", 0.10)
        ),
        matched_energy_ev=float(
            values.get("matched_energy_ev", 4_000.0)
        ),
        trace_step_mm=float(values.get("trace_step_mm", 0.05)),
        wien=wien,
        slit=MonochromatorSlit(
            **{
                key: value
                for key, value in dict(values.get("slit", {})).items()
                if key in MonochromatorSlit.__dataclass_fields__
            }
        ),
    )
    return assembly.validate()
