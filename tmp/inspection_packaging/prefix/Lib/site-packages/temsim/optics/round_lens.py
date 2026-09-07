"""Reusable continuous axial-field round-lens component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim.optics.condenser_lens import AxialFieldTerm
from temsim.optics.lens_focal_length import focal_length_mm as _focal_length_mm
from temsim.optics.model import Gaussian, Lens


@dataclass
class RoundLensComponent:
    """One physical round lens shared by layout, solver, GUI and overlays."""

    name: str
    key: str
    z_mm: float
    b0_t: float
    a_mm: float
    percent: float
    max_percent: float
    colour: str
    gaussian: list[AxialFieldTerm]
    enabled: bool
    cs_mm: float | None
    cc_mm: float | None
    polarity: int
    normalise_profile_peak: bool
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_from_tip_mm: float
    corrector: str = "probe"

    EXPECTED_KEY: ClassVar[str | None] = None
    KIND: ClassVar[str] = "round_lens"
    SHAPE_PROFILE: ClassVar[str] = "magnetic_lens_yoke"
    INTERACTION_KIND: ClassVar[str] = "axial_magnetic_field"

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
        return self.bore_diameter_mm / 2.0

    def scale(self):
        return (
            self.b0_t * self.percent / 100.0
            if self.enabled else 0.0
        )

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
        if self.mechanical_outer_diameter_mm <= self.bore_diameter_mm:
            raise ValueError(
                f"{self.name} outer diameter must exceed its bore."
            )
        if self.bore_diameter_mm <= 0.0 or self.pole_gap_mm <= 0.0:
            raise ValueError(
                f"{self.name} bore and pole gap must be positive."
            )
        if self.pole_gap_mm > self.mechanical_length_mm:
            raise ValueError(
                f"{self.name} pole gap must fit its mechanical body."
            )
        if self.b0_t < 0.0 or self.a_mm <= 0.0:
            raise ValueError(
                f"{self.name} field and field width must be valid."
            )
        if not 0.0 <= self.percent <= self.max_percent:
            raise ValueError(
                f"{self.name} excitation exceeds its configured range."
            )
        if not self.gaussian or any(
            term.sigma <= 0.0 for term in self.gaussian
        ):
            raise ValueError(
                f"{self.name} requires valid Gaussian field terms."
            )
        return self

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_from_tip_mm)
        return self

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        field = np.zeros_like(z)
        if not self.enabled:
            return field
        for term in self.gaussian:
            sigma = max(abs(term.sigma * self.a_mm), 1e-12)
            centre = self.z_mm + term.offset * self.a_mm
            field += term.amplitude * np.exp(
                -0.5 * ((z - centre) / sigma) ** 2
            )
        return float(self.polarity) * self.scale() * field

    def focal_length_mm(self):
        return _focal_length_mm(self, 300.0)

    def field_support_mm(self, sigma_cutoff=7.0):
        reaches = [
            abs(term.offset * self.a_mm)
            + float(sigma_cutoff) * abs(term.sigma * self.a_mm)
            for term in self.gaussian
        ]
        half = max(reaches, default=0.0)
        return self.z_mm - half, self.z_mm + half

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
            "bore_diameter_mm": self.bore_diameter_mm,
            "pole_gap_mm": self.pole_gap_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        start, end = self.field_support_mm()
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_mm(),
            "enabled": self.enabled,
        }


def restore_round_lens(component, values):
    values = dict(values)
    allowed = component.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
        if attribute == "gaussian":
            value = [
                term
                if isinstance(term, AxialFieldTerm)
                else AxialFieldTerm(**term)
                for term in value
            ]
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


@dataclass(frozen=True)
class AnchoredRoundLensGeometry:
    """Resolved geometry for a lens fixed downstream of another lens."""

    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_z_mm: float


@dataclass
class AnchoredRoundLensComponent(Lens):
    """Round lens whose mechanical and optical positions follow an anchor."""

    anchor_key: str = ""
    mechanical_center_downstream_of_anchor_mm: float = 0.0
    optical_reference_downstream_of_anchor_mm: float = 0.0
    mechanical_length_mm: float = 1.0
    mechanical_outer_diameter_mm: float = 2.0
    mechanical_clear_bore_diameter_mm: float = 1.0
    pole_gap_mm: float = 1.0
    corrector: str = "projector"

    EXPECTED_KEY: ClassVar[str | None] = None
    EXPECTED_ANCHOR_KEY: ClassVar[str | None] = None
    KIND: ClassVar[str] = "round_lens"
    SHAPE_PROFILE: ClassVar[str] = "magnetic_lens_yoke"
    INTERACTION_KIND: ClassVar[str] = "axial_magnetic_field"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "mechanical_center_downstream_of_anchor_mm":
            value = float(value)
            delta_mm = value - float(getattr(self, name))
            object.__setattr__(self, name, value)
            object.__setattr__(
                self,
                "optical_reference_downstream_of_anchor_mm",
                float(self.optical_reference_downstream_of_anchor_mm)
                + delta_mm,
            )
            object.__setattr__(self, "z_mm", float(self.z_mm) + delta_mm)
            return
        if ready and name == "optical_reference_downstream_of_anchor_mm":
            value = float(value)
            delta_mm = value - float(getattr(self, name))
            object.__setattr__(self, name, value)
            object.__setattr__(self, "z_mm", float(self.z_mm) + delta_mm)
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return self.corrector

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
    def optical_active(self):
        return bool(self.enabled)

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def geometry_for(self, anchor_geometry):
        return AnchoredRoundLensGeometry(
            mechanical_center_below_sample_mm=(
                float(anchor_geometry.mechanical_center_below_sample_mm)
                + float(self.mechanical_center_downstream_of_anchor_mm)
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_z_mm=(
                float(anchor_geometry.optical_reference_z_mm)
                + float(self.mechanical_center_downstream_of_anchor_mm)
            ),
        )

    def resolve_against(self, anchor_geometry):
        object.__setattr__(
            self,
            "optical_reference_downstream_of_anchor_mm",
            float(self.mechanical_center_downstream_of_anchor_mm),
        )
        geometry = self.geometry_for(anchor_geometry)
        object.__setattr__(
            self, "z_mm", float(geometry.optical_reference_z_mm)
        )
        return geometry

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        field = np.zeros_like(z)
        if not self.enabled:
            return field
        for term in self.gaussian:
            sigma = max(abs(term.sigma * self.a_mm), 1e-12)
            centre = self.z_mm + term.offset * self.a_mm
            field += term.amplitude * np.exp(
                -0.5 * ((z - centre) / sigma) ** 2
            )
        if self.normalise_profile_peak:
            field /= max(float(np.max(np.abs(field))), 1e-15)
        return float(self.polarity) * self.scale() * field

    def focal_length_mm(self):
        return _focal_length_mm(self, 300.0)

    def field_support_mm(self, sigma_cutoff=7.0):
        reaches = [
            abs(term.offset * self.a_mm)
            + float(sigma_cutoff) * abs(term.sigma * self.a_mm)
            for term in self.gaussian
        ]
        half = max(reaches, default=0.0)
        return self.z_mm - half, self.z_mm + half

    def validate(self):
        if self.EXPECTED_KEY is not None and self.key != self.EXPECTED_KEY:
            raise ValueError(f"{self.name} key is not canonical.")
        if (
            self.EXPECTED_ANCHOR_KEY is not None
            and self.anchor_key != self.EXPECTED_ANCHOR_KEY
        ):
            raise ValueError(f"{self.name} anchor key is not canonical.")
        if self.mechanical_center_downstream_of_anchor_mm <= 0.0:
            raise ValueError(
                f"{self.name} must remain downstream of its anchor."
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
        if not 0.0 < self.pole_gap_mm <= self.mechanical_length_mm:
            raise ValueError(f"{self.name} pole gap must fit its body.")
        if self.b0_t < 0.0 or self.a_mm <= 0.0:
            raise ValueError(
                f"{self.name} field and field width must be valid."
            )
        if not 0.0 < self.max_percent <= 100.0:
            raise ValueError(
                f"{self.name} maximum excitation must lie in (0, 100]."
            )
        if not 0.0 <= self.percent <= self.max_percent:
            raise ValueError(
                f"{self.name} excitation exceeds its configured range."
            )
        if not self.gaussian or any(
            term.sigma <= 0.0 for term in self.gaussian
        ):
            raise ValueError(
                f"{self.name} requires valid Gaussian field terms."
            )
        return self

    def draw_layout(self, anchor_geometry):
        geometry = self.geometry_for(anchor_geometry)
        return {
            "key": self.key,
            "anchor_key": self.anchor_key,
            "mechanical_center_below_sample_mm": (
                geometry.mechanical_center_below_sample_mm
            ),
            "mechanical_center_downstream_of_anchor_mm": (
                self.mechanical_center_downstream_of_anchor_mm
            ),
            "mechanical_length_mm": geometry.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                geometry.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                geometry.mechanical_clear_bore_diameter_mm
            ),
            "pole_gap_mm": geometry.pole_gap_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        start, end = self.field_support_mm()
        return {
            "key": self.key,
            "anchor_key": self.anchor_key,
            "optical_reference_z_mm": self.z_mm,
            "optical_reference_downstream_of_anchor_mm": (
                self.optical_reference_downstream_of_anchor_mm
            ),
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_mm(),
            "enabled": self.enabled,
        }


def restore_anchored_round_lens(
    component,
    values,
    legacy_anchor_reference_z_mm=None,
):
    values = dict(values)
    allowed = component.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
        if attribute == "gaussian":
            value = [
                term if isinstance(term, Gaussian) else Gaussian(**term)
                for term in value
            ]
        if attribute in allowed:
            object.__setattr__(component, attribute, value)
    if "optical_reference_downstream_of_anchor_mm" not in values:
        anchor_reference = float(
            legacy_anchor_reference_z_mm
            if legacy_anchor_reference_z_mm is not None
            else component.z_mm
            - component.optical_reference_downstream_of_anchor_mm
        )
        object.__setattr__(
            component,
            "optical_reference_downstream_of_anchor_mm",
            float(values.get("z_mm", component.z_mm)) - anchor_reference,
        )
    object.__setattr__(component, "_position_coupling_ready", True)
    return component.validate()
