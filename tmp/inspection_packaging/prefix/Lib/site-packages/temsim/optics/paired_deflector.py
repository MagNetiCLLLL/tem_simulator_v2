"""Reusable mechanics and thin-plane physics for double deflectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class PairedDeflectorComponent:
    """A physical two-coil assembly represented by two optical kick planes."""

    name: str
    key: str
    upper_z_mm: float
    lower_z_mm: float
    upper_x_mrad: float
    upper_y_mrad: float
    lower_x_mrad: float
    lower_y_mrad: float
    thickness_mm: float
    enabled: bool
    colour: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_upper_reference_from_tip_mm: float
    optical_lower_reference_from_tip_mm: float
    maximum_kick_mrad: float

    EXPECTED_KEY: ClassVar[str | None] = None
    OWNER: ClassVar[str] = "column"
    KIND: ClassVar[str] = "paired_deflector"
    SHAPE_PROFILE: ClassVar[str] = "paired_deflector_coils"
    INTERACTION_KIND: ClassVar[str] = "paired_transverse_kick"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        position_fields = {
            "upper_z_mm",
            "lower_z_mm",
            "mechanical_center_from_tip_mm",
            "optical_upper_reference_from_tip_mm",
            "optical_lower_reference_from_tip_mm",
        }
        if name in position_fields:
            value = float(value)
        coupling_ready = self.__dict__.get(
            "_position_coupling_ready", False
        )
        if name == "mechanical_center_from_tip_mm" and coupling_ready:
            delta_mm = float(value) - float(
                self.mechanical_center_from_tip_mm
            )
            object.__setattr__(self, name, float(value))
            upper = (
                float(self.optical_upper_reference_from_tip_mm)
                + delta_mm
            )
            lower = (
                float(self.optical_lower_reference_from_tip_mm)
                + delta_mm
            )
            object.__setattr__(
                self, "optical_upper_reference_from_tip_mm", upper
            )
            object.__setattr__(
                self, "optical_lower_reference_from_tip_mm", lower
            )
            object.__setattr__(self, "upper_z_mm", upper)
            object.__setattr__(self, "lower_z_mm", lower)
            return
        if (
            name == "optical_upper_reference_from_tip_mm"
            and coupling_ready
        ):
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "upper_z_mm", float(value))
            return
        if (
            name == "optical_lower_reference_from_tip_mm"
            and coupling_ready
        ):
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "lower_z_mm", float(value))
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return self.OWNER

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
    def optical_center_from_tip_mm(self):
        return 0.5 * (
            self.optical_upper_reference_from_tip_mm
            + self.optical_lower_reference_from_tip_mm
        )

    @optical_center_from_tip_mm.setter
    def optical_center_from_tip_mm(self, value):
        delta_mm = float(value) - self.optical_center_from_tip_mm
        self.optical_upper_reference_from_tip_mm += delta_mm
        self.optical_lower_reference_from_tip_mm += delta_mm

    @property
    def optical_plane_separation_mm(self):
        return abs(
            self.optical_lower_reference_from_tip_mm
            - self.optical_upper_reference_from_tip_mm
        )

    @optical_plane_separation_mm.setter
    def optical_plane_separation_mm(self, value):
        separation_mm = abs(float(value))
        center_mm = self.optical_center_from_tip_mm
        self.optical_upper_reference_from_tip_mm = (
            center_mm - separation_mm / 2.0
        )
        self.optical_lower_reference_from_tip_mm = (
            center_mm + separation_mm / 2.0
        )

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
        if self.mechanical_outer_diameter_mm <= 0.0:
            raise ValueError(
                f"{self.name} outer diameter must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(f"{self.name} bore must fit inside its body.")
        if self.optical_plane_separation_mm <= 0.0:
            raise ValueError(
                f"{self.name} optical planes must be separated."
            )
        if self.thickness_mm <= 0.0:
            raise ValueError(
                f"{self.name} effective thickness must be positive."
            )
        if self.thickness_mm > self.mechanical_length_mm:
            raise ValueError(
                f"{self.name} effective thickness exceeds its body."
            )
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError(
                f"{self.name} maximum kick must be positive."
            )
        for value in (
            self.upper_x_mrad,
            self.upper_y_mrad,
            self.lower_x_mrad,
            self.lower_y_mrad,
        ):
            if abs(float(value)) > self.maximum_kick_mrad:
                raise ValueError(
                    f"{self.name} kick exceeds its configured limit."
                )
        return self

    def apply_optical_positions(self):
        self.upper_z_mm = float(
            self.optical_upper_reference_from_tip_mm
        )
        self.lower_z_mm = float(
            self.optical_lower_reference_from_tip_mm
        )
        return self

    def kick_events(self):
        """Return the two thin-plane kicks consumed by the ray solver."""

        if not self.enabled:
            return ()
        return (
            (
                self.upper_z_mm,
                self.upper_x_mrad * 1.0e-3,
                self.upper_y_mrad * 1.0e-3,
            ),
            (
                self.lower_z_mm,
                self.lower_x_mrad * 1.0e-3,
                self.lower_y_mrad * 1.0e-3,
            ),
        )

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
            "coil_plane_separation_mm": (
                self.optical_plane_separation_mm
            ),
            "coil_thickness_mm": self.thickness_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "upper_plane_z_mm": self.upper_z_mm,
            "lower_plane_z_mm": self.lower_z_mm,
            "effective_coil_thickness_mm": self.thickness_mm,
            "upper_kick_mrad": (
                self.upper_x_mrad,
                self.upper_y_mrad,
            ),
            "lower_kick_mrad": (
                self.lower_x_mrad,
                self.lower_y_mrad,
            ),
            "enabled": self.enabled,
        }


def restore_paired_deflector(component, values):
    """Restore common dataclass fields without triggering delta coupling."""

    values = dict(values)
    allowed = component.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
        if attribute in allowed:
            object.__setattr__(component, attribute, value)
    if not {
        "optical_upper_reference_from_tip_mm",
        "optical_lower_reference_from_tip_mm",
    } <= values.keys():
        object.__setattr__(
            component,
            "optical_upper_reference_from_tip_mm",
            float(values.get("upper_z_mm", component.upper_z_mm)),
        )
        object.__setattr__(
            component,
            "optical_lower_reference_from_tip_mm",
            float(values.get("lower_z_mm", component.lower_z_mm)),
        )
    object.__setattr__(component, "_position_coupling_ready", True)
    component.apply_optical_positions()
    return component.validate()
