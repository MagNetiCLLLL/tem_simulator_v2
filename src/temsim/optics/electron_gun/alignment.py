"""Finite magnetic alignment components inside a field-emission gun."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import FEG_DEFLECTOR, FEG_STIGMATOR
from temsim.optics.electron_gun.electrostatic import (
    _soft_window_with_derivatives,
)


@dataclass
class GunDeflector:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    upper_center_from_tip_mm: float
    lower_center_from_tip_mm: float
    coil_length_mm: float
    name: str = "Gun Deflector Pair"
    key: str = FEG_DEFLECTOR
    soft_edge_mm: float = 1.0
    upper_field_x_mt: float = 0.0
    upper_field_y_mt: float = 0.0
    lower_field_x_mt: float = 0.0
    lower_field_y_mt: float = 0.0
    enabled: bool = True
    colour: str = "#ab47bc"
    field_center_offset_mm: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if name == "mechanical_center_from_tip_mm" and ready:
            delta = float(value) - float(self.mechanical_center_from_tip_mm)
            object.__setattr__(self, name, float(value))
            object.__setattr__(
                self, "upper_center_from_tip_mm",
                float(self.upper_center_from_tip_mm) + delta,
            )
            object.__setattr__(
                self, "lower_center_from_tip_mm",
                float(self.lower_center_from_tip_mm) + delta,
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
        return "finite_paired_deflector"

    @property
    def shape_profile(self):
        return "paired_deflector_coils"

    def field_at_global_positions_t(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        result = np.zeros_like(positions)
        if not self.enabled:
            return result
        z_mm = positions[..., 2] * 1000.0
        half = 0.5 * self.coil_length_mm
        for center, field_x, field_y in (
            (
                self.upper_center_from_tip_mm,
                self.upper_field_x_mt,
                self.upper_field_y_mt,
            ),
            (
                self.lower_center_from_tip_mm,
                self.lower_field_x_mt,
                self.lower_field_y_mt,
            ),
        ):
            center += self.field_center_offset_mm
            envelope = _soft_window_with_derivatives(
                z_mm, center - half, center + half, self.soft_edge_mm
            )[0]
            result[..., 0] += float(field_x) * 1e-3 * envelope
            result[..., 1] += float(field_y) * 1e-3 * envelope
        return result

    def validate(self):
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Gun deflector body length must be positive.")
        if self.mechanical_outer_diameter_mm <= 0.0:
            raise ValueError("Gun deflector outer diameter must be positive.")
        if (
            self.mechanical_clear_bore_diameter_mm <= 0.0
            or self.mechanical_clear_bore_diameter_mm
            >= self.mechanical_outer_diameter_mm
        ):
            raise ValueError("Gun deflector bore must fit inside its body.")
        if self.coil_length_mm <= 0.0 or self.soft_edge_mm <= 0.0:
            raise ValueError("Gun deflector field lengths must be positive.")
        if self.upper_center_from_tip_mm >= self.lower_center_from_tip_mm:
            raise ValueError("Gun deflector coils must retain their order.")
        for value in (
            self.upper_field_x_mt,
            self.upper_field_y_mt,
            self.lower_field_x_mt,
            self.lower_field_y_mt,
            self.field_center_offset_mm,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("Gun deflector fields must be finite.")
        return self

    def draw_layout(self):
        return _draw_alignment(self)


@dataclass
class GunStigmator:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    effective_length_mm: float
    name: str = "Gun Stigmator"
    key: str = FEG_STIGMATOR
    soft_edge_mm: float = 1.0
    gradient_t_per_m: float = 0.0
    rotation_deg: float = 0.0
    enabled: bool = True
    colour: str = "#7b1fa2"
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
        return "finite_quadrupole_stigmator"

    @property
    def shape_profile(self):
        return "quadrupole_body"

    def field_at_global_positions_t(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        result = np.zeros_like(positions)
        if not self.enabled or self.gradient_t_per_m == 0.0:
            return result
        z_mm = positions[..., 2] * 1000.0
        center = self.optical_reference_from_tip_mm
        half = 0.5 * self.effective_length_mm
        envelope = _soft_window_with_derivatives(
            z_mm, center - half, center + half, self.soft_edge_mm
        )[0]
        angle = np.deg2rad(self.rotation_deg)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        x = positions[..., 0]
        y = positions[..., 1]
        u = cosine * x + sine * y
        v = -sine * x + cosine * y
        b_u = self.gradient_t_per_m * v * envelope
        b_v = self.gradient_t_per_m * u * envelope
        result[..., 0] = cosine * b_u - sine * b_v
        result[..., 1] = sine * b_u + cosine * b_v
        return result

    def validate(self):
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Gun stigmator body length must be positive.")
        if self.mechanical_outer_diameter_mm <= 0.0:
            raise ValueError("Gun stigmator outer diameter must be positive.")
        if (
            self.mechanical_clear_bore_diameter_mm <= 0.0
            or self.mechanical_clear_bore_diameter_mm
            >= self.mechanical_outer_diameter_mm
        ):
            raise ValueError("Gun stigmator bore must fit inside its body.")
        if self.effective_length_mm <= 0.0:
            raise ValueError("Gun stigmator field length must be positive.")
        if self.effective_length_mm > self.mechanical_length_mm:
            raise ValueError("Gun stigmator field must fit inside its body.")
        if self.soft_edge_mm <= 0.0:
            raise ValueError("Gun stigmator soft edge must be positive.")
        for value in (
            self.gradient_t_per_m,
            self.rotation_deg,
            self.field_center_offset_mm,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("Gun stigmator settings must be finite.")
        return self

    def draw_layout(self):
        return _draw_alignment(self)


class FegMagneticField:
    def __init__(self, deflector, stigmator):
        self.deflector = deflector
        self.stigmator = stigmator

    def field_at_global_positions_t(self, positions_m):
        return (
            self.deflector.field_at_global_positions_t(positions_m)
            + self.stigmator.field_at_global_positions_t(positions_m)
        )


def _draw_alignment(component):
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
        "shape_profile": component.shape_profile,
    }
