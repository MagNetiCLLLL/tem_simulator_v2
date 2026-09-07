"""Owned post-objective Image/Diffraction double-deflector component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim import module_manifest
from temsim.component_keys import (
    IMAGE_DIFFRACTION_DEFLECTOR,
    canonical_deflector_key,
)

_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml", "exit"
)
_DEFAULT_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, IMAGE_DIFFRACTION_DEFLECTOR
)
_DEFAULT_SAMPLE = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, "sample"
)
_DEFAULT_INTERACTIONS = tuple(
    _DEFAULT_COLUMN_ORIGIN_Z_MM + float(value)
    for value in _DEFAULT_PART["interaction_centers_local_z_mm"]
)


@dataclass(frozen=True)
class ImageDiffractionDeflectorDefinition:
    key: str = IMAGE_DIFFRACTION_DEFLECTOR
    label: str = "Image / Diffraction Deflector Pair"
    mechanical_center_below_sample_mm: float = (
        float(_DEFAULT_PART["local_center_z_mm"])
        - float(_DEFAULT_SAMPLE["local_center_z_mm"])
    )
    mechanical_length_mm: float = float(_DEFAULT_PART["length_mm"])
    mechanical_outer_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_outer_diameter_mm"]
    )
    mechanical_clear_bore_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_clear_bore_diameter_mm"]
    )
    optical_upper_reference_z_mm: float = _DEFAULT_INTERACTIONS[0]
    optical_lower_reference_z_mm: float = _DEFAULT_INTERACTIONS[1]
    effective_coil_thickness_mm: float = float(
        _DEFAULT_PART["effective_thickness_mm"]
    )
    inter_coil_gap_mm: float = float(
        _DEFAULT_PART["mechanical_inter_coil_gap_mm"]
    )
    maximum_kick_mrad: float = 100.0
    colour: str = "#66bb6a"
    owner: str = "image"
    kind: str = "paired_deflector"
    shape_profile: str = "paired_deflector_coils"
    interaction_kind: str = "paired_transverse_kick"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return ImageDiffractionDeflectorComponent(
            name=self.label,
            key=self.key,
            upper_z_mm=self.optical_upper_reference_z_mm,
            lower_z_mm=self.optical_lower_reference_z_mm,
            upper_x_mrad=0.0,
            upper_y_mrad=0.0,
            lower_x_mrad=0.0,
            lower_y_mrad=0.0,
            thickness_mm=self.effective_coil_thickness_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_below_sample_mm=(
                self.mechanical_center_below_sample_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_upper_reference_z_mm=(
                self.optical_upper_reference_z_mm
            ),
            optical_lower_reference_z_mm=(
                self.optical_lower_reference_z_mm
            ),
            inter_coil_gap_mm=self.inter_coil_gap_mm,
            maximum_kick_mrad=self.maximum_kick_mrad,
        )


@dataclass
class ImageDiffractionDeflectorComponent:
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
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_upper_reference_z_mm: float
    optical_lower_reference_z_mm: float
    inter_coil_gap_mm: float
    maximum_kick_mrad: float

    EXPECTED_KEY: ClassVar[str] = IMAGE_DIFFRACTION_DEFLECTOR
    OWNER: ClassVar[str] = "image"
    KIND: ClassVar[str] = "paired_deflector"
    SHAPE_PROFILE: ClassVar[str] = "paired_deflector_coils"
    INTERACTION_KIND: ClassVar[str] = "paired_transverse_kick"

    def __post_init__(self):
        object.__setattr__(self, "_geometry_ready", False)
        self._sync_geometry()
        object.__setattr__(self, "_geometry_ready", True)
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        geometry_ready = self.__dict__.get("_geometry_ready", False)
        if geometry_ready and name in {
            "thickness_mm",
            "inter_coil_gap_mm",
        }:
            value = float(value)
            if value <= 0.0:
                raise ValueError(
                    "Image/Diffraction Deflector coil thickness and gap "
                    "must be positive."
                )
            object.__setattr__(self, name, value)
            self._sync_geometry()
            return
        if geometry_ready and name == "mechanical_length_mm":
            length_mm = float(value)
            gap_mm = length_mm - 2.0 * float(self.thickness_mm)
            if gap_mm <= 0.0:
                raise ValueError(
                    "Image/Diffraction Deflector mechanical length must "
                    "contain two coils and a positive gap."
                )
            object.__setattr__(self, "inter_coil_gap_mm", gap_mm)
            self._sync_geometry()
            return
        position_fields = {
            "upper_z_mm",
            "lower_z_mm",
            "mechanical_center_below_sample_mm",
            "optical_upper_reference_z_mm",
            "optical_lower_reference_z_mm",
        }
        if name in position_fields:
            value = float(value)
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "mechanical_center_below_sample_mm":
            delta_mm = value - float(
                self.mechanical_center_below_sample_mm
            )
            object.__setattr__(self, name, value)
            upper = float(self.optical_upper_reference_z_mm) + delta_mm
            lower = float(self.optical_lower_reference_z_mm) + delta_mm
            object.__setattr__(
                self, "optical_upper_reference_z_mm", upper
            )
            object.__setattr__(
                self, "optical_lower_reference_z_mm", lower
            )
            object.__setattr__(self, "upper_z_mm", upper)
            object.__setattr__(self, "lower_z_mm", lower)
            return
        if ready and name == "optical_upper_reference_z_mm":
            object.__setattr__(self, name, value)
            object.__setattr__(self, "upper_z_mm", value)
            return
        if ready and name == "optical_lower_reference_z_mm":
            object.__setattr__(self, name, value)
            object.__setattr__(self, "lower_z_mm", value)
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
    def optical_center_z_mm(self):
        return 0.5 * (
            self.optical_upper_reference_z_mm
            + self.optical_lower_reference_z_mm
        )

    @optical_center_z_mm.setter
    def optical_center_z_mm(self, value):
        delta_mm = float(value) - self.optical_center_z_mm
        self.optical_upper_reference_z_mm += delta_mm
        self.optical_lower_reference_z_mm += delta_mm

    @property
    def optical_plane_separation_mm(self):
        return abs(
            self.optical_lower_reference_z_mm
            - self.optical_upper_reference_z_mm
        )

    @optical_plane_separation_mm.setter
    def optical_plane_separation_mm(self, value):
        separation_mm = abs(float(value))
        gap_mm = separation_mm - float(self.thickness_mm)
        if gap_mm <= 0.0:
            raise ValueError(
                "Image/Diffraction Deflector plane separation must "
                "exceed the coil thickness."
            )
        self.inter_coil_gap_mm = gap_mm

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    @property
    def upper_surface_z_mm(self):
        return (
            self.optical_center_z_mm - self.mechanical_length_mm / 2.0
        )

    @property
    def lower_surface_z_mm(self):
        return (
            self.optical_center_z_mm + self.mechanical_length_mm / 2.0
        )

    def _sync_geometry(self):
        center_mm = 0.5 * (
            float(self.optical_upper_reference_z_mm)
            + float(self.optical_lower_reference_z_mm)
        )
        separation_mm = (
            float(self.thickness_mm) + float(self.inter_coil_gap_mm)
        )
        object.__setattr__(
            self,
            "mechanical_length_mm",
            2.0 * float(self.thickness_mm)
            + float(self.inter_coil_gap_mm),
        )
        object.__setattr__(
            self,
            "optical_upper_reference_z_mm",
            center_mm - separation_mm / 2.0,
        )
        object.__setattr__(
            self,
            "optical_lower_reference_z_mm",
            center_mm + separation_mm / 2.0,
        )
        object.__setattr__(
            self,
            "upper_z_mm",
            center_mm - separation_mm / 2.0,
        )
        object.__setattr__(
            self,
            "lower_z_mm",
            center_mm + separation_mm / 2.0,
        )

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError(
                "Image/Diffraction Deflector key is not canonical."
            )
        if self.mechanical_center_below_sample_mm <= 0.0:
            raise ValueError(
                "Image/Diffraction Deflector must remain below the sample."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Image/Diffraction Deflector body length must be positive."
            )
        expected_length_mm = (
            2.0 * self.thickness_mm + self.inter_coil_gap_mm
        )
        if abs(self.mechanical_length_mm - expected_length_mm) > 1.0e-9:
            raise ValueError(
                "Image/Diffraction Deflector length must equal "
                "coil + gap + coil."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(
                "Image/Diffraction Deflector bore must fit its body."
            )
        if self.optical_plane_separation_mm <= 0.0:
            raise ValueError(
                "Image/Diffraction Deflector planes must be separated."
            )
        expected_separation_mm = (
            self.thickness_mm + self.inter_coil_gap_mm
        )
        if (
            abs(
                self.optical_plane_separation_mm
                - expected_separation_mm
            )
            > 1.0e-9
        ):
            raise ValueError(
                "Image/Diffraction Deflector optical planes must use "
                "the same coil + gap geometry as its mechanical body."
            )
        if not 0.0 < self.thickness_mm <= self.mechanical_length_mm:
            raise ValueError(
                "Image/Diffraction Deflector coil thickness is invalid."
            )
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError(
                "Image/Diffraction Deflector maximum kick must be positive."
            )
        if max(
            abs(float(self.upper_x_mrad)),
            abs(float(self.upper_y_mrad)),
            abs(float(self.lower_x_mrad)),
            abs(float(self.lower_y_mrad)),
        ) > self.maximum_kick_mrad:
            raise ValueError(
                "Image/Diffraction Deflector kick exceeds its limit."
            )
        return self

    def apply_optical_positions(self):
        self.upper_z_mm = float(self.optical_upper_reference_z_mm)
        self.lower_z_mm = float(self.optical_lower_reference_z_mm)
        return self

    def kick_events(self):
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
            "mechanical_center_below_sample_mm": (
                self.mechanical_center_below_sample_mm
            ),
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "coil_plane_separation_mm": self.optical_plane_separation_mm,
            "coil_thickness_mm": self.thickness_mm,
            "inter_coil_gap_mm": self.inter_coil_gap_mm,
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


IMAGE_DIFFRACTION_DEFLECTOR_DEFINITION = (
    ImageDiffractionDeflectorDefinition()
)


def create_image_diffraction_deflector():
    return (
        IMAGE_DIFFRACTION_DEFLECTOR_DEFINITION
        .create_component()
        .validate()
    )


def image_diffraction_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_deflector_key(values.get("key", ""))
    component = create_image_diffraction_deflector()
    for attribute in (
        "upper_x_mrad",
        "upper_y_mrad",
        "lower_x_mrad",
        "lower_y_mrad",
        "enabled",
        "colour",
        "maximum_kick_mrad",
    ):
        if attribute in values:
            object.__setattr__(component, attribute, values[attribute])
    component.key = IMAGE_DIFFRACTION_DEFLECTOR
    component.name = "Image / Diffraction Deflector Pair"
    return component.apply_optical_positions().validate()
