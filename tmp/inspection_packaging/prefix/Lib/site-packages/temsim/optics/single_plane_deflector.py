"""Reusable mechanics and thin-plane physics for one-plane deflectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SinglePlaneDeflectorComponent:
    name: str
    key: str
    z_mm: float
    kick_x_mrad: float
    kick_y_mrad: float
    effective_thickness_mm: float
    enabled: bool
    colour: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    maximum_kick_mrad: float
    corrector: str = "probe"

    EXPECTED_KEY: ClassVar[str | None] = None
    KIND: ClassVar[str] = "deflector"
    SHAPE_PROFILE: ClassVar[str] = "single_deflector_coil"
    INTERACTION_KIND: ClassVar[str] = "thin_transverse_kick"

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
    def kind(self):
        return self.KIND

    @property
    def owner(self):
        return (
            "probe_corrector"
            if self.corrector == "probe"
            else self.corrector
        )

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
        return bool(self.enabled)

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
            0.0
            < self.effective_thickness_mm
            <= self.mechanical_length_mm
        ):
            raise ValueError(
                f"{self.name} effective thickness must fit its body."
            )
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError(f"{self.name} maximum kick must be positive.")
        if max(abs(self.kick_x_mrad), abs(self.kick_y_mrad)) > (
            self.maximum_kick_mrad
        ):
            raise ValueError(
                f"{self.name} kick exceeds its configured limit."
            )
        return self

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_from_tip_mm)
        return self

    def kick_events(self):
        if not self.enabled:
            return ()
        return ((
            self.z_mm,
            self.kick_x_mrad * 1.0e-3,
            self.kick_y_mrad * 1.0e-3,
        ),)

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
            "effective_thickness_mm": self.effective_thickness_mm,
            "kick_x_mrad": self.kick_x_mrad,
            "kick_y_mrad": self.kick_y_mrad,
            "enabled": self.enabled,
        }


def restore_single_plane_deflector(component, values):
    values = dict(values)
    allowed = component.__dataclass_fields__
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
