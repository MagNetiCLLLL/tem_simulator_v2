"""Finite axisymmetric electrostatic fields for a cold FEG."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from temsim.component_keys import (
    FEG_ACCELERATOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_EXTRACTOR,
)


def _smooth_step_with_derivatives(z_mm, start_mm, end_mm):
    """Quintic compact step and its first three derivatives per millimetre."""

    z = np.asarray(z_mm, dtype=float)
    length = max(float(end_mm) - float(start_mm), 1e-9)
    u = (z - float(start_mm)) / length
    inside = (u > 0.0) & (u < 1.0)
    t = np.clip(u, 0.0, 1.0)
    value = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    first = np.zeros_like(z)
    second = np.zeros_like(z)
    third = np.zeros_like(z)
    first[inside] = (
        30.0 * t[inside] ** 2
        - 60.0 * t[inside] ** 3
        + 30.0 * t[inside] ** 4
    ) / length
    second[inside] = (
        60.0 * t[inside]
        - 180.0 * t[inside] ** 2
        + 120.0 * t[inside] ** 3
    ) / length**2
    third[inside] = (
        60.0 - 360.0 * t[inside] + 360.0 * t[inside] ** 2
    ) / length**3
    return value, first, second, third


def _soft_window_with_derivatives(z_mm, start_mm, end_mm, edge_mm):
    edge = max(float(edge_mm), 1e-6)
    left = _smooth_step_with_derivatives(
        z_mm, float(start_mm) - edge, float(start_mm) + edge
    )
    right = _smooth_step_with_derivatives(
        z_mm, float(end_mm) - edge, float(end_mm) + edge
    )
    return tuple(a - b for a, b in zip(left, right))


@dataclass
class ExtractorElectrode:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    name: str = "Extractor"
    key: str = FEG_EXTRACTOR
    voltage_kv: float = 4.0
    transition_start_mm: float = 0.05
    transition_end_mm: float = 6.0
    field_center_offset_mm: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if name == "mechanical_center_from_tip_mm" and ready:
            delta = float(value) - float(self.mechanical_center_from_tip_mm)
            object.__setattr__(self, name, float(value))
            object.__setattr__(
                self, "transition_start_mm",
                float(self.transition_start_mm) + delta,
            )
            object.__setattr__(
                self, "transition_end_mm",
                float(self.transition_end_mm) + delta,
            )
            return
        object.__setattr__(self, name, value)

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_clear_bore_diameter_mm

    @property
    def optical_reference_from_tip_mm(self):
        return self.mechanical_center_from_tip_mm + self.field_center_offset_mm

    @property
    def kind(self):
        return "extractor_electrode"

    @property
    def shape_profile(self):
        return "electrode"

    def axial_potential_v_and_derivatives_per_mm(self, z_mm):
        values = _smooth_step_with_derivatives(
            z_mm,
            self.transition_start_mm + self.field_center_offset_mm,
            self.transition_end_mm + self.field_center_offset_mm,
        )
        return tuple(self.voltage_kv * 1000.0 * value for value in values)

    def validate(self):
        if not 0.0 <= self.voltage_kv <= 20.0:
            raise ValueError("Extractor voltage must lie between 0 and 20 kV.")
        if self.transition_end_mm <= self.transition_start_mm:
            raise ValueError("Extractor field transition must have finite length.")
        if self.mechanical_clear_bore_diameter_mm <= 0.0:
            raise ValueError("Extractor clear bore must be positive.")
        return self

    def draw_layout(self):
        return _draw_body(self, "electrode")


@dataclass
class ElectrostaticGunLens:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    name: str = "Electrostatic Gun Lens"
    key: str = FEG_ELECTROSTATIC_LENS
    voltage_kv: float = 1.2
    potential_scale: float = 4.22125
    soft_edge_mm: float = 1.5
    field_center_offset_mm: float = 0.0

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_clear_bore_diameter_mm

    @property
    def optical_reference_from_tip_mm(self):
        return self.mechanical_center_from_tip_mm + self.field_center_offset_mm

    @property
    def kind(self):
        return "electrostatic_lens"

    @property
    def shape_profile(self):
        return "electrostatic_electrode_stack"

    def axial_potential_v_and_derivatives_per_mm(self, z_mm):
        center = self.optical_reference_from_tip_mm
        half = 0.5 * self.mechanical_length_mm
        values = _soft_window_with_derivatives(
            z_mm, center - half, center + half, self.soft_edge_mm
        )
        amplitude = self.voltage_kv * 1000.0 * self.potential_scale
        return tuple(amplitude * value for value in values)

    def validate(self):
        if self.mechanical_length_mm <= 0.0 or self.soft_edge_mm <= 0.0:
            raise ValueError("Electrostatic gun lens lengths must be positive.")
        if self.mechanical_clear_bore_diameter_mm <= 0.0:
            raise ValueError("Electrostatic gun lens clear bore must be positive.")
        if self.potential_scale < 0.0:
            raise ValueError("Electrostatic gun lens scale must not be negative.")
        return self

    def draw_layout(self):
        return _draw_body(self, "electrostatic_electrode_stack")


@dataclass
class AcceleratorStage:
    center_from_tip_mm: float
    voltage_fraction: float
    soft_edge_mm: float = 3.0

    def validate(self):
        if self.center_from_tip_mm <= 0.0:
            raise ValueError("Accelerator stage must follow the emitter.")
        if not 0.0 < self.voltage_fraction <= 1.0:
            raise ValueError("Accelerator voltage fraction must lie in (0, 1].")
        if self.soft_edge_mm <= 0.0:
            raise ValueError("Accelerator soft edge must be positive.")
        return self


@dataclass
class AcceleratorColumn:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    stages: list[AcceleratorStage]
    name: str = "Accelerator Tube"
    key: str = FEG_ACCELERATOR
    high_tension_kv: float = 300.0
    field_center_offset_mm: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if name == "mechanical_center_from_tip_mm" and ready:
            delta = float(value) - float(self.mechanical_center_from_tip_mm)
            object.__setattr__(self, name, float(value))
            for stage in self.stages:
                stage.center_from_tip_mm += delta
            return
        object.__setattr__(self, name, value)

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_clear_bore_diameter_mm

    @property
    def optical_reference_from_tip_mm(self):
        return self.mechanical_center_from_tip_mm + self.field_center_offset_mm

    @property
    def kind(self):
        return "multistage_accelerator"

    @property
    def shape_profile(self):
        return "accelerator_stack"

    def normalized_potential_and_derivatives_per_mm(self, z_mm):
        result = [np.zeros_like(np.asarray(z_mm, dtype=float)) for _ in range(4)]
        previous_fraction = 0.0
        for stage in self.stages:
            increment = stage.voltage_fraction - previous_fraction
            values = _smooth_step_with_derivatives(
                z_mm,
                stage.center_from_tip_mm
                + self.field_center_offset_mm
                - stage.soft_edge_mm,
                stage.center_from_tip_mm
                + self.field_center_offset_mm
                + stage.soft_edge_mm,
            )
            for order in range(4):
                result[order] += increment * values[order]
            previous_fraction = stage.voltage_fraction
        return tuple(result)

    def validate(self):
        if not 30.0 <= self.high_tension_kv <= 300.0:
            raise ValueError(
                "Electron-gun high tension must lie between 30 and 300 kV."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Accelerator length must be positive.")
        if not self.stages:
            raise ValueError("Accelerator requires at least one electrode stage.")
        previous_z = -np.inf
        previous_fraction = 0.0
        for stage in self.stages:
            stage.validate()
            if stage.center_from_tip_mm <= previous_z:
                raise ValueError("Accelerator stages must be ordered by position.")
            if stage.voltage_fraction <= previous_fraction:
                raise ValueError("Accelerator voltage fractions must increase.")
            previous_z = stage.center_from_tip_mm
            previous_fraction = stage.voltage_fraction
        if not np.isclose(previous_fraction, 1.0):
            raise ValueError("Final accelerator stage must have voltage fraction 1.")
        return self

    def draw_layout(self):
        values = _draw_body(self, "accelerator_stack")
        values["stage_centers_from_tip_mm"] = [
            stage.center_from_tip_mm for stage in self.stages
        ]
        return values


class FegElectrostaticField:
    """Superposed axisymmetric potential expanded through radial order r²."""

    def __init__(self, emitter, extractor, electrostatic_lens, accelerator):
        self.emitter = emitter
        self.extractor = extractor
        self.electrostatic_lens = electrostatic_lens
        self.accelerator = accelerator

    def axial_potential_v_and_derivatives_per_mm(self, z_mm):
        extractor = self.extractor.axial_potential_v_and_derivatives_per_mm(
            z_mm
        )
        lens = self.electrostatic_lens.axial_potential_v_and_derivatives_per_mm(
            z_mm
        )
        ramp = self.accelerator.normalized_potential_and_derivatives_per_mm(
            z_mm
        )
        final_gain_v = (
            self.accelerator.high_tension_kv * 1000.0
            - self.emitter.emission_energy_ev
        )
        accelerator_gain_v = final_gain_v - self.extractor.voltage_kv * 1000.0
        return tuple(
            extractor[order] + lens[order] + accelerator_gain_v * ramp[order]
            for order in range(4)
        )

    def axial_potential_v(self, z_mm):
        return self.axial_potential_v_and_derivatives_per_mm(z_mm)[0]

    def potential_v_at_global_positions(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        z_mm = positions[..., 2] * 1000.0
        potential, _, d2_mm, _ = (
            self.axial_potential_v_and_derivatives_per_mm(z_mm)
        )
        d2_m = d2_mm * 1.0e6
        radius_squared = (
            positions[..., 0] ** 2 + positions[..., 1] ** 2
        )
        return potential - 0.25 * radius_squared * d2_m

    def field_at_global_positions_v_per_m(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        z_mm = positions[..., 2] * 1000.0
        _, d1_mm, d2_mm, d3_mm = (
            self.axial_potential_v_and_derivatives_per_mm(z_mm)
        )
        d1_m = d1_mm * 1.0e3
        d2_m = d2_mm * 1.0e6
        d3_m = d3_mm * 1.0e9
        x = positions[..., 0]
        y = positions[..., 1]
        radius_squared = x * x + y * y
        field = np.empty_like(positions)
        field[..., 0] = 0.5 * x * d2_m
        field[..., 1] = 0.5 * y * d2_m
        field[..., 2] = -d1_m + 0.25 * radius_squared * d3_m
        return field


def _draw_body(component, shape_profile):
    return {
        "key": component.key,
        "mechanical_center_from_tip_mm": (
            component.mechanical_center_from_tip_mm
        ),
        "mechanical_length_mm": component.mechanical_length_mm,
        "mechanical_outer_diameter_mm": (
            component.mechanical_outer_diameter_mm
        ),
        "mechanical_clear_bore_diameter_mm": (
            component.mechanical_clear_bore_diameter_mm
        ),
        "shape_profile": shape_profile,
    }
