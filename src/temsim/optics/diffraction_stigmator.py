"""Independent topology-aware Diffraction Stigmator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim.component_keys import (
    DIFFRACTION_STIGMATOR,
    SELECTED_AREA_APERTURE,
    canonical_stigmator_key,
)
from temsim.optics.model import Stigmator
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm


STANDALONE_INSTALLATION = "standalone"
IMAGE_CORRECTED_INSTALLATION = "image_corrected"


@dataclass(frozen=True)
class DiffractionStigmatorGeometry:
    installation: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_z_mm: float
    owner: str = "diffraction"


@dataclass(frozen=True)
class DiffractionStigmatorDefinition:
    key: str = DIFFRACTION_STIGMATOR
    label: str = "Diffraction Stigmator"
    anchor_key: str = SELECTED_AREA_APERTURE
    mechanical_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_STIGMATOR
    )
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_STIGMATOR
    )
    # Mechanical envelope dimensions remain provisional.
    standalone_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    image_corrected_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    mechanical_length_mm: float = 40.0
    mechanical_outer_diameter_mm: float = 120.0
    mechanical_clear_bore_diameter_mm: float = 20.0
    standalone_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    image_corrected_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    effective_length_mm: float = 8.0
    maximum_strength_m2: float = 300.0
    colour: str = "#fb8c00"
    owner: str = "diffraction"
    kind: str = "stigmator"
    shape_profile: str = "quadrupole_body"
    interaction_kind: str = "distributed_quadrupole_field"

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
                f"Unsupported Diffraction Stigmator installation: "
                f"{installation}"
            )
        return DiffractionStigmatorGeometry(
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
            optical_reference_z_mm=getattr(
                self, f"{installation}_optical_reference_z_mm"
            ),
        )

    def create_component(self):
        return DiffractionStigmatorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.standalone_optical_reference_z_mm,
            length_mm=self.effective_length_mm,
            max_strength_m2=self.maximum_strength_m2,
            strength_x_percent=0.0,
            strength_y_percent=0.0,
            enabled=True,
            colour=self.colour,
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
class DiffractionStigmatorComponent(Stigmator):
    anchor_key: str = SELECTED_AREA_APERTURE
    mechanical_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_STIGMATOR
    )
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        DIFFRACTION_STIGMATOR
    )
    standalone_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    image_corrected_mechanical_center_below_sample_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_mechanical_center_below_sample_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    mechanical_length_mm: float = 40.0
    mechanical_outer_diameter_mm: float = 120.0
    mechanical_clear_bore_diameter_mm: float = 20.0
    standalone_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    image_corrected_optical_reference_z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .image_corrected_optical_reference_z_mm
        + downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    active_installation: str = STANDALONE_INSTALLATION
    corrector: str = "diffraction"

    EXPECTED_KEY: ClassVar[str] = DIFFRACTION_STIGMATOR
    KIND: ClassVar[str] = "stigmator"
    SHAPE_PROFILE: ClassVar[str] = "quadrupole_body"
    INTERACTION_KIND: ClassVar[str] = "distributed_quadrupole_field"

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

    @property
    def strength_x_m2(self):
        return (
            float(self.max_strength_m2)
            * float(self.strength_x_percent)
            / 100.0
        )

    @property
    def strength_y_m2(self):
        return (
            float(self.max_strength_m2)
            * float(self.strength_y_percent)
            / 100.0
        )

    def geometry_for(self, installation):
        if installation not in {
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        }:
            raise ValueError(
                f"Unsupported Diffraction Stigmator installation: "
                f"{installation}"
            )
        return DiffractionStigmatorGeometry(
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

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError(
                "Diffraction Stigmator key is not canonical."
            )
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Diffraction Stigmator must follow the "
                "Selected Area Aperture."
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
                    "Diffraction Stigmator must remain below the sample."
                )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Diffraction Stigmator mechanical length must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(
                "Diffraction Stigmator bore must fit inside its body."
            )
        if not 0.0 < self.length_mm <= self.mechanical_length_mm:
            raise ValueError(
                "Diffraction Stigmator field length must fit its body."
            )
        if self.max_strength_m2 <= 0.0:
            raise ValueError(
                "Diffraction Stigmator maximum strength must be positive."
            )
        return self

    def quadrupole_strengths_m2(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        if not self.enabled:
            zero = np.zeros_like(z)
            return zero, zero
        sigma_mm = max(float(self.length_mm) / 2.355, 1e-12)
        envelope = np.exp(
            -0.5 * ((z - float(self.z_mm)) / sigma_mm) ** 2
        )
        signed_strength = 0.5 * (
            self.strength_x_m2 - self.strength_y_m2
        )
        x_strength = signed_strength * envelope
        return x_strength, -x_strength

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
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "effective_length_mm": self.length_mm,
            "strength_x_m2": self.strength_x_m2,
            "strength_y_m2": self.strength_y_m2,
            "enabled": self.enabled,
        }


DIFFRACTION_STIGMATOR_DEFINITION = DiffractionStigmatorDefinition()


def create_diffraction_stigmator():
    return DIFFRACTION_STIGMATOR_DEFINITION.create_component().validate()


def diffraction_stigmator_from_dict(
    data,
    legacy_reference_z_mm=None,
    active_installation=None,
):
    values = dict(data)
    values["key"] = canonical_stigmator_key(values.get("key", ""))
    component = create_diffraction_stigmator()
    allowed = component.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
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
                DIFFRACTION_STIGMATOR_DEFINITION
                .image_corrected_optical_reference_z_mm
                - DIFFRACTION_STIGMATOR_DEFINITION
                .standalone_optical_reference_z_mm
            )
            object.__setattr__(
                component,
                "image_corrected_optical_reference_z_mm",
                standalone_reference + reference_delta,
            )
    object.__setattr__(component, "_position_coupling_ready", True)
    component.key = DIFFRACTION_STIGMATOR
    component.name = "Diffraction Stigmator"
    component.anchor_key = SELECTED_AREA_APERTURE
    component.mechanical_center_downstream_of_anchor_mm = (
        downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    component.optical_reference_downstream_of_anchor_mm = (
        downstream_offset_mm(DIFFRACTION_STIGMATOR)
    )
    component.corrector = DIFFRACTION_STIGMATOR_DEFINITION.owner
    component.select_installation(
        active_installation
        or values.get("active_installation", STANDALONE_INSTALLATION)
    )
    return component.validate()
