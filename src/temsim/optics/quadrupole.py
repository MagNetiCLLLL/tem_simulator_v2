"""Reusable distributed paraxial quadrupole-field component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim.optics.component_position import TipReferencedPosition

import numpy as np


@dataclass
class QuadrupoleComponent(TipReferencedPosition):
    name: str
    key: str
    z_mm: float
    strength_m2: float
    maximum_strength_m2: float
    effective_length_mm: float
    enabled: bool
    colour: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    corrector: str = "probe"

    EXPECTED_KEY: ClassVar[str | None] = None
    KIND: ClassVar[str] = "quadrupole"
    SHAPE_PROFILE: ClassVar[str] = "quadrupole_body"
    INTERACTION_KIND: ClassVar[str] = "distributed_quadrupole_field"

    @property
    def owner(self):
        return (
            "probe_corrector"
            if self.corrector == "probe"
            else self.corrector
        )

    @property
    def kind(self):
        return self.KIND

    @property
    def shape_profile(self):
        return self.SHAPE_PROFILE

    @property
    def interaction_kind(self):
        return self.INTERACTION_KIND

    @property
    def length_mm(self):
        return self.mechanical_length_mm

    @property
    def optical_active(self):
        return True

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def validate(self):
        if self.EXPECTED_KEY is not None and self.key != self.EXPECTED_KEY:
            raise ValueError(f"{self.name} key is not canonical.")
        if self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(
                f"{self.name} mechanical centre must follow the tip."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                f"{self.name} mechanical length must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(f"{self.name} bore must fit inside its body.")
        if not (
            0.0 < self.effective_length_mm <= self.mechanical_length_mm
        ):
            raise ValueError(
                f"{self.name} effective length must fit its body."
            )
        if self.maximum_strength_m2 <= 0.0:
            raise ValueError(
                f"{self.name} maximum strength must be positive."
            )
        if abs(self.strength_m2) > self.maximum_strength_m2:
            raise ValueError(
                f"{self.name} strength exceeds its configured limit."
            )
        return self

    def quadrupole_strength_m2(self, z_mm):
        """Return signed +x focusing strength; y receives the negative."""

        z = np.asarray(z_mm, dtype=float)
        if not self.enabled:
            return np.zeros_like(z)
        sigma_mm = max(self.effective_length_mm / 2.355, 1e-12)
        envelope = np.exp(
            -0.5 * ((z - self.z_mm) / sigma_mm) ** 2
        )
        return float(self.strength_m2) * envelope

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
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "effective_length_mm": self.effective_length_mm,
            "strength_m2": self.strength_m2,
            "enabled": self.enabled,
        }


def restore_quadrupole(component, values):
    values = dict(values)
    allowed = QuadrupoleComponent.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
        if attribute in allowed:
            object.__setattr__(component, attribute, value)
    if "optical_reference_from_tip_mm" not in values:
        object.__setattr__(
            component,
            "optical_reference_from_tip_mm",
            float(values.get("z_mm", component.z_mm)),
        )
    object.__setattr__(component, "_position_coupling_ready", True)
    return component.apply_optical_position().validate()
