"""Independent topology-aware Diffraction Lens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    DIFFRACTION_LENS,
    SELECTED_AREA_APERTURE,
    canonical_lens_key,
)
from temsim.optics.lens_focal_length import (
    focal_length_mm as _focal_length_mm,
)
from temsim.optics.model import Gaussian, Lens
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm


STANDALONE_INSTALLATION = "standalone"
IMAGE_CORRECTED_INSTALLATION = "image_corrected"
_DEFAULT_MANIFEST_PART = module_manifest.part_data(
    "project_and_recording_system/EnergyFilter.toml",
    DIFFRACTION_LENS,
)


@dataclass(frozen=True)
class DiffractionLensGeometry:
    installation: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_z_mm: float
    owner: str = "projector"


@dataclass(frozen=True)
class DiffractionLensDefinition:
    key: str = DIFFRACTION_LENS
    label: str = "Diffraction Lens"
    anchor_key: str = SELECTED_AREA_APERTURE
    mechanical_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_LENS
    )
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_LENS
    )
    standalone_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    image_corrected_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    # Mechanical defaults come from the authoritative recording-system TOML.
    mechanical_length_mm: float = float(_DEFAULT_MANIFEST_PART["length_mm"])
    mechanical_outer_diameter_mm: float = float(
        _DEFAULT_MANIFEST_PART["mechanical_outer_diameter_mm"]
    )
    mechanical_clear_bore_diameter_mm: float = float(
        _DEFAULT_MANIFEST_PART["mechanical_clear_bore_diameter_mm"]
    )
    pole_gap_mm: float = float(_DEFAULT_MANIFEST_PART["pole_gap_mm"])
    standalone_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    image_corrected_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    b0_t: float = 0.75
    a_mm: float = 12.0
    percent: float = 26.495
    maximum_percent: float = 100.0
    colour: str = "#f57c00"
    owner: str = "projector"
    kind: str = "round_lens"
    shape_profile: str = "magnetic_lens_yoke"
    interaction_kind: str = "axial_magnetic_field"

    @property
    def name(self):
        return self.label

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def geometry_for(self, installation):
        if installation not in {
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        }:
            raise ValueError(
                f"Unsupported Diffraction Lens installation: {installation}"
            )
        return DiffractionLensGeometry(
            installation=installation,
            mechanical_center_below_sample_mm=getattr(
                self,
                f"{installation}_mechanical_center_below_sample_mm",
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_z_mm=getattr(
                self, f"{installation}_optical_reference_z_mm"
            ),
        )

    def create_component(self):
        return DiffractionLensComponent(
            name=self.label,
            key=self.key,
            z_mm=self.standalone_optical_reference_z_mm,
            b0_t=self.b0_t,
            a_mm=self.a_mm,
            percent=self.percent,
            max_percent=self.maximum_percent,
            colour=self.colour,
            gaussian=[
                Gaussian(0.09, -1.0, 0.90),
                Gaussian(0.82, 0.0, 0.55),
                Gaussian(0.09, 1.0, 0.90),
            ],
            enabled=True,
            cs_mm=None,
            cc_mm=None,
            polarity=int(_DEFAULT_MANIFEST_PART["field_polarity"]),
            normalise_profile_peak=False,
            anchor_key=self.anchor_key,
            mechanical_center_downstream_of_anchor_mm=(
                self.mechanical_center_downstream_of_anchor_mm
            ),
            optical_reference_downstream_of_anchor_mm=(
                self.optical_reference_downstream_of_anchor_mm
            ),
            standalone_mechanical_center_below_sample_mm=(
                self.standalone_mechanical_center_below_sample_mm
            ),
            image_corrected_mechanical_center_below_sample_mm=(
                self.image_corrected_mechanical_center_below_sample_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            standalone_optical_reference_z_mm=(
                self.standalone_optical_reference_z_mm
            ),
            image_corrected_optical_reference_z_mm=(
                self.image_corrected_optical_reference_z_mm
            ),
            active_installation=STANDALONE_INSTALLATION,
            corrector=self.owner,
        )


@dataclass
class DiffractionLensComponent(Lens):
    anchor_key: str = SELECTED_AREA_APERTURE
    mechanical_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_LENS
    )
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_LENS
    )
    standalone_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    image_corrected_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    mechanical_length_mm: float = float(_DEFAULT_MANIFEST_PART["length_mm"])
    mechanical_outer_diameter_mm: float = float(
        _DEFAULT_MANIFEST_PART["mechanical_outer_diameter_mm"]
    )
    mechanical_clear_bore_diameter_mm: float = float(
        _DEFAULT_MANIFEST_PART["mechanical_clear_bore_diameter_mm"]
    )
    pole_gap_mm: float = float(_DEFAULT_MANIFEST_PART["pole_gap_mm"])
    standalone_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    image_corrected_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_LENS)
    )
    active_installation: str = STANDALONE_INSTALLATION
    corrector: str = "projector"

    EXPECTED_KEY: ClassVar[str] = DIFFRACTION_LENS
    KIND: ClassVar[str] = "round_lens"
    SHAPE_PROFILE: ClassVar[str] = "magnetic_lens_yoke"
    INTERACTION_KIND: ClassVar[str] = "axial_magnetic_field"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        center_attributes = {
            "standalone_mechanical_center_below_sample_mm": (
                STANDALONE_INSTALLATION
            ),
            "image_corrected_mechanical_center_below_sample_mm": (
                IMAGE_CORRECTED_INSTALLATION
            ),
        }
        reference_attributes = {
            "standalone_optical_reference_z_mm": STANDALONE_INSTALLATION,
            "image_corrected_optical_reference_z_mm": (
                IMAGE_CORRECTED_INSTALLATION
            ),
        }
        if ready and name in center_attributes:
            value = float(value)
            delta_mm = value - float(getattr(self, name))
            installation = center_attributes[name]
            reference_attribute = (
                f"{installation}_optical_reference_z_mm"
            )
            reference = float(getattr(self, reference_attribute)) + delta_mm
            object.__setattr__(self, name, value)
            object.__setattr__(self, reference_attribute, reference)
            if self.active_installation == installation:
                object.__setattr__(self, "z_mm", reference)
            return
        if ready and name in reference_attributes:
            value = float(value)
            object.__setattr__(self, name, value)
            if self.active_installation == reference_attributes[name]:
                object.__setattr__(self, "z_mm", value)
            return
        if ready and name == "z_mm":
            value = float(value)
            object.__setattr__(self, name, value)
            object.__setattr__(
                self,
                f"{self.active_installation}_optical_reference_z_mm",
                value,
            )
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

    @property
    def mechanical_center_below_sample_mm(self):
        return getattr(
            self,
            f"{self.active_installation}"
            "_mechanical_center_below_sample_mm",
        )

    @mechanical_center_below_sample_mm.setter
    def mechanical_center_below_sample_mm(self, value):
        setattr(
            self,
            f"{self.active_installation}"
            "_mechanical_center_below_sample_mm",
            float(value),
        )

    @property
    def optical_reference_z_mm(self):
        return getattr(
            self,
            f"{self.active_installation}_optical_reference_z_mm",
        )

    @optical_reference_z_mm.setter
    def optical_reference_z_mm(self, value):
        setattr(
            self,
            f"{self.active_installation}_optical_reference_z_mm",
            float(value),
        )

    def geometry_for(self, installation):
        if installation not in {
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        }:
            raise ValueError(
                f"Unsupported Diffraction Lens installation: {installation}"
            )
        return DiffractionLensGeometry(
            installation=installation,
            mechanical_center_below_sample_mm=getattr(
                self,
                f"{installation}_mechanical_center_below_sample_mm",
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_z_mm=getattr(
                self, f"{installation}_optical_reference_z_mm"
            ),
        )

    def select_installation(self, installation):
        geometry = self.geometry_for(installation)
        object.__setattr__(self, "active_installation", installation)
        object.__setattr__(
            self, "z_mm", float(geometry.optical_reference_z_mm)
        )
        return self

    def resolve_against(self, selected_area_geometry):
        mechanical_center = (
            float(selected_area_geometry.mechanical_center_below_sample_mm)
            + float(self.mechanical_center_downstream_of_anchor_mm)
        )
        optical_reference = (
            float(selected_area_geometry.optical_reference_z_mm)
            + float(self.mechanical_center_downstream_of_anchor_mm)
        )
        object.__setattr__(
            self,
            "optical_reference_downstream_of_anchor_mm",
            float(self.mechanical_center_downstream_of_anchor_mm),
        )
        object.__setattr__(
            self,
            f"{self.active_installation}"
            "_mechanical_center_below_sample_mm",
            mechanical_center,
        )
        object.__setattr__(
            self,
            f"{self.active_installation}_optical_reference_z_mm",
            optical_reference,
        )
        object.__setattr__(self, "z_mm", optical_reference)
        return self.geometry_for(self.active_installation)

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
        if self.key != self.EXPECTED_KEY:
            raise ValueError("Diffraction Lens key is not canonical.")
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Diffraction Lens must follow the Selected Area Aperture."
            )
        for installation in (
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        ):
            if (
                self.geometry_for(
                    installation
                ).mechanical_center_below_sample_mm
                <= 0.0
            ):
                raise ValueError(
                    "Diffraction Lens must remain below the sample."
                )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Diffraction Lens mechanical length must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(
                "Diffraction Lens bore must fit inside its body."
            )
        if not 0.0 < self.pole_gap_mm <= self.mechanical_length_mm:
            raise ValueError(
                "Diffraction Lens pole gap must fit its body."
            )
        if self.b0_t < 0.0 or self.a_mm <= 0.0:
            raise ValueError(
                "Diffraction Lens field and field width must be valid."
            )
        if not 0.0 < self.max_percent <= 100.0:
            raise ValueError(
                "Diffraction Lens maximum excitation must lie in (0, 100]."
            )
        if not 0.0 <= self.percent <= self.max_percent:
            raise ValueError(
                "Diffraction Lens excitation exceeds its configured range."
            )
        if not self.gaussian or any(
            term.sigma <= 0.0 for term in self.gaussian
        ):
            raise ValueError(
                "Diffraction Lens requires valid Gaussian field terms."
            )
        return self

    def draw_layout(self):
        geometry = self.geometry_for(self.active_installation)
        return {
            "key": self.key,
            "installation": geometry.installation,
            "mechanical_center_below_sample_mm": (
                geometry.mechanical_center_below_sample_mm
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
            "optical_reference_z_mm": self.z_mm,
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_mm(),
            "enabled": self.enabled,
        }


DIFFRACTION_LENS_DEFINITION = DiffractionLensDefinition()


def create_diffraction_lens():
    return DIFFRACTION_LENS_DEFINITION.create_component().validate()


def diffraction_lens_from_dict(
    data,
    legacy_reference_z_mm=None,
    active_installation=None,
):
    values = dict(data)
    values["key"] = canonical_lens_key(values.get("key", ""))
    component = create_diffraction_lens()
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
    if "standalone_optical_reference_z_mm" not in values:
        standalone_reference = float(
            legacy_reference_z_mm
            if legacy_reference_z_mm is not None
            else values.get("z_mm", component.z_mm)
        )
        object.__setattr__(
            component,
            "standalone_optical_reference_z_mm",
            standalone_reference,
        )
        if "image_corrected_optical_reference_z_mm" not in values:
            reference_delta = (
                DIFFRACTION_LENS_DEFINITION
                .image_corrected_optical_reference_z_mm
                - DIFFRACTION_LENS_DEFINITION
                .standalone_optical_reference_z_mm
            )
            object.__setattr__(
                component,
                "image_corrected_optical_reference_z_mm",
                standalone_reference + reference_delta,
            )
    object.__setattr__(component, "_position_coupling_ready", True)
    component.key = DIFFRACTION_LENS
    component.name = DIFFRACTION_LENS_DEFINITION.label
    component.anchor_key = SELECTED_AREA_APERTURE
    component.mechanical_center_downstream_of_anchor_mm = (
        downstream_offset_mm(DIFFRACTION_LENS)
    )
    component.optical_reference_downstream_of_anchor_mm = (
        downstream_offset_mm(DIFFRACTION_LENS)
    )
    component.corrector = DIFFRACTION_LENS_DEFINITION.owner
    component.select_installation(
        active_installation
        or values.get("active_installation", STANDALONE_INSTALLATION)
    )
    return component.validate()
