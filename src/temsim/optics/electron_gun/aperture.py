"""Continuous hard apertures physically owned by an electron gun."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim import module_manifest
from temsim.component_keys import C1_APERTURE, GUN_EXTRACTOR_APERTURE


@dataclass
class GunAperture:
    name: str
    key: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    radius_mm: float
    maximum_radius_mm: float
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    enabled: bool = True
    colour: str = "#8e24aa"
    field_center_offset_mm: float = 0.0

    @property
    def label(self):
        if getattr(self, "_slit_mode", False):
            return "Monochromator Slit (C1 Aperture)"
        return self.name

    @property
    def optical_reference_from_tip_mm(self):
        return (
            float(self.mechanical_center_from_tip_mm)
            + float(self.field_center_offset_mm)
        )

    @property
    def z_mm(self):
        return self.optical_reference_from_tip_mm

    @property
    def kind(self):
        if getattr(self, "_slit_mode", False):
            return "energy_selection_slit"
        return "continuous_aperture"

    @property
    def shape_profile(self):
        if getattr(self, "_slit_mode", False):
            return "two_blade_energy_slit"
        return "adjustable_circular_aperture"

    @property
    def interaction_kind(self):
        if getattr(self, "_slit_mode", False):
            return "hard_edge_two_blade_stop"
        return "hard_edge_circular_stop"

    @property
    def effective_aperture_radius_mm(self):
        if getattr(self, "_slit_mode", False):
            return 0.5 * self.mechanical_bore_diameter_mm
        return self.radius_mm

    @effective_aperture_radius_mm.setter
    def effective_aperture_radius_mm(self, value):
        self.radius_mm = float(value)

    @property
    def radius_um(self):
        return self.radius_mm * 1000.0

    @radius_um.setter
    def radius_um(self, value):
        self.radius_mm = float(value) / 1000.0

    @property
    def offset_x_um(self):
        return self.offset_x_mm * 1000.0

    @offset_x_um.setter
    def offset_x_um(self, value):
        self.offset_x_mm = float(value) / 1000.0

    @property
    def offset_y_um(self):
        return self.offset_y_mm * 1000.0

    @offset_y_um.setter
    def offset_y_um(self, value):
        self.offset_y_mm = float(value) / 1000.0

    def transmission_mask(self, x_mm, y_mm):
        slit = getattr(self, "_slit_profile", None)
        if getattr(self, "_slit_mode", False) and slit is not None:
            x = np.asarray(x_mm, dtype=float)
            y = np.asarray(y_mm, dtype=float)
            inside_bore = np.hypot(x, y) <= (
                0.5 * float(self.mechanical_bore_diameter_mm)
            )
            if not slit.inserted:
                return inside_bore
            half_gap_mm = 0.5 * float(slit.gap_um) * 1.0e-3
            centre_mm = float(slit.centre_offset_um) * 1.0e-3
            return inside_bore & (np.abs(x - centre_mm) <= half_gap_mm)
        if not self.enabled:
            return np.ones_like(np.asarray(x_mm), dtype=bool)
        return np.hypot(
            np.asarray(x_mm, dtype=float) - self.offset_x_mm,
            np.asarray(y_mm, dtype=float) - self.offset_y_mm,
        ) <= self.radius_mm

    def bind_slit_profile(self, slit_profile):
        """Bind the alternate two-blade setting of the C1 mechanism."""

        object.__setattr__(self, "_slit_profile", slit_profile)
        return self

    def select_slit_mode(self, enabled):
        if bool(enabled) and getattr(self, "_slit_profile", None) is None:
            raise ValueError("C1 slit mode requires a bound slit profile.")
        object.__setattr__(self, "_slit_mode", bool(enabled))
        return self

    def validate(self):
        if self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(f"{self.name} must follow the emitter tip.")
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(f"{self.name} length must be positive.")
        if not 0.0 <= self.radius_mm <= self.maximum_radius_mm:
            raise ValueError(f"{self.name} radius is outside its continuous range.")
        if 2.0 * self.maximum_radius_mm > self.mechanical_bore_diameter_mm:
            raise ValueError(f"{self.name} opening does not fit inside its bore.")
        if self.mechanical_bore_diameter_mm >= self.mechanical_outer_diameter_mm:
            raise ValueError(f"{self.name} bore must fit inside its body.")
        slit = getattr(self, "_slit_profile", None)
        if getattr(self, "_slit_mode", False) and slit is not None:
            slit.validate()
            occupied_half_width_mm = (
                abs(float(slit.centre_offset_um))
                + 0.5 * float(slit.gap_um)
            ) * 1.0e-3
            if occupied_half_width_mm > (
                0.5 * self.mechanical_bore_diameter_mm
            ):
                raise ValueError(
                    "Monochromator slit opening must fit in the C1 bore."
                )
        return self

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": self.mechanical_outer_diameter_mm,
            "mechanical_bore_diameter_mm": self.mechanical_bore_diameter_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "radius_mm": self.radius_mm,
            "offset_x_mm": self.offset_x_mm,
            "offset_y_mm": self.offset_y_mm,
            "enabled": self.enabled,
        }


def _create_feg_aperture(key, name, colour="#8e24aa"):
    part = module_manifest.part_data("gun/FEG.toml", key)
    geometry = module_manifest.part_geometry("gun/FEG.toml", key)
    return GunAperture(
        name=name,
        key=key,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_bore_diameter_mm=float(part["bore_diameter_mm"]),
        plate_thickness_mm=float(part["active_length_mm"]),
        radius_mm=2.0,
        maximum_radius_mm=3.0,
        colour=colour,
    )


def create_dpa_aperture():
    return _create_feg_aperture(
        GUN_EXTRACTOR_APERTURE, "Gun / DPA Aperture"
    )


def create_c1_aperture():
    return _create_feg_aperture(
        C1_APERTURE, "C1 Aperture", colour="#1e88e5"
    )
