"""Reusable distributed paraxial hexapole-field component."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np


@dataclass
class HexapoleComponent:
    """A signed, continuous hexapole with nonlinear transverse action.

    ``strength_m3`` is the on-axis envelope coefficient used by the paraxial
    equations.  The solver applies it to ``x**2 - y**2`` and ``2*x*y`` at
    every RK4 substep, so this component is deliberately not represented as a
    pair of linear quadrupole strengths.
    """

    name: str
    key: str
    z_mm: float
    strength_m3: float
    maximum_strength_m3: float
    effective_length_mm: float
    enabled: bool
    colour: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    corrector: str = "probe"
    orientation_rad: float = 0.0

    EXPECTED_KEY: ClassVar[str | None] = None
    KIND: ClassVar[str] = "hexapole"
    SHAPE_PROFILE: ClassVar[str] = "hexapole_body"
    INTERACTION_KIND: ClassVar[str] = "distributed_hexapole_field"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_from_tip_mm",
            "optical_reference_from_tip_mm",
        }:
            value = float(value)
        coupling_ready = self.__dict__.get(
            "_position_coupling_ready", False
        )
        if name == "mechanical_center_from_tip_mm" and coupling_ready:
            delta_mm = float(value) - float(
                self.mechanical_center_from_tip_mm
            )
            object.__setattr__(self, name, float(value))
            optical = float(self.optical_reference_from_tip_mm) + delta_mm
            object.__setattr__(
                self, "optical_reference_from_tip_mm", optical
            )
            object.__setattr__(self, "z_mm", optical)
            return
        if name == "optical_reference_from_tip_mm" and coupling_ready:
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "z_mm", float(value))
            return
        object.__setattr__(self, name, value)

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
        if self.maximum_strength_m3 <= 0.0:
            raise ValueError(
                f"{self.name} maximum strength must be positive."
            )
        if abs(self.strength_m3) > self.maximum_strength_m3:
            raise ValueError(
                f"{self.name} strength exceeds its configured limit."
            )
        if not math.isfinite(float(self.orientation_rad)):
            raise ValueError(f"{self.name} orientation must be finite.")
        return self

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_from_tip_mm)
        return self

    def hexapole_strength_m3(self, z_mm):
        """Return the normal Gaussian nonlinear-field coefficient.

        The compatibility scalar is the normal component.  The propagator uses
        :meth:`hexapole_strength_components_m3` so a Larmor-rotated second
        principal hexapole can also contribute its required skew component.
        """

        normal, _skew = self.hexapole_strength_components_m3(z_mm)
        return normal

    def hexapole_strength_components_m3(self, z_mm):
        """Return normal/skew coefficients of the oriented hexapole field."""

        z = np.asarray(z_mm, dtype=float)
        if not self.enabled:
            zeros = np.zeros_like(z)
            return zeros, zeros.copy()
        sigma_mm = max(self.effective_length_mm / 2.355, 1e-12)
        envelope = np.exp(
            -0.5 * ((z - self.z_mm) / sigma_mm) ** 2
        )
        amplitude = float(self.strength_m3) * envelope
        phase = 3.0 * float(self.orientation_rad)
        return amplitude * math.cos(phase), amplitude * math.sin(phase)

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
            "strength_m3": self.strength_m3,
            "orientation_rad": self.orientation_rad,
            "enabled": self.enabled,
        }


def restore_hexapole(component, values):
    values = dict(values)
    allowed = HexapoleComponent.__dataclass_fields__
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
