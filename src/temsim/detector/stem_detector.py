"""Shared canonical BF, DF and HAADF recording-stop components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    BRIGHT_FIELD_DETECTOR,
    DARK_FIELD_DETECTOR,
    HAADF_DETECTOR,
    SELECTED_AREA_APERTURE,
    STEM_DETECTOR_KEYS,
    canonical_recording_plane_key,
)
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm
from temsim.optics.selected_area_downstream import downstream_optical_offset_mm
from temsim.detector.point_spread import validate_component_point_spread


_DEFAULT_RECORDING_MODULE = (
    "project_and_recording_system/NoEnergyFilter.toml"
)
_DEFAULT_MANIFEST_PARTS = {
    key: module_manifest.part_data(_DEFAULT_RECORDING_MODULE, key)
    for key in (
        HAADF_DETECTOR,
        DARK_FIELD_DETECTOR,
        BRIGHT_FIELD_DETECTOR,
    )
}


@dataclass(frozen=True)
class StemDetectorDefinition:
    key: str
    label: str
    optical_reference_downstream_of_anchor_mm: float
    layout_center_downstream_of_anchor_mm: float
    layout_length_mm: float
    geometry: str
    outer_width_mm: float
    inner_diameter_mm: float
    colour: str
    point_spread_model: str
    point_spread_sigma_x_mm: float
    point_spread_sigma_y_mm: float
    point_spread_rotation_deg: float
    point_spread_status: str
    point_spread_source: str
    anchor_key: str = SELECTED_AREA_APERTURE
    owner: str = "detector"
    kind: str = "detector"
    shape_profile: str = "detector_plane"
    interaction_kind: str = "recording_plane_stop"

    @property
    def name(self):
        return self.label

    def create_component(
        self,
        anchor_z_mm=(
            SELECTED_AREA_APERTURE_DEFINITION
            .standalone_optical_reference_z_mm
        ),
    ):
        return StemDetectorComponent(
            key=self.key,
            name=self.label,
            z_mm=(
                float(anchor_z_mm)
                + self.optical_reference_downstream_of_anchor_mm
            ),
            geometry=self.geometry,
            outer_width_mm=self.outer_width_mm,
            inner_diameter_mm=self.inner_diameter_mm,
            inserted=True,
            colour=self.colour,
            anchor_key=self.anchor_key,
            optical_reference_downstream_of_anchor_mm=(
                self.optical_reference_downstream_of_anchor_mm
            ),
            layout_center_downstream_of_anchor_mm=(
                self.layout_center_downstream_of_anchor_mm
            ),
            layout_length_mm=self.layout_length_mm,
            owner=self.owner,
            point_spread_model=self.point_spread_model,
            point_spread_sigma_x_mm=self.point_spread_sigma_x_mm,
            point_spread_sigma_y_mm=self.point_spread_sigma_y_mm,
            point_spread_rotation_deg=self.point_spread_rotation_deg,
            point_spread_status=self.point_spread_status,
            point_spread_source=self.point_spread_source,
        )


@dataclass
class StemDetectorComponent:
    key: str
    name: str
    z_mm: float
    geometry: str
    outer_width_mm: float
    inner_diameter_mm: float = 0.0
    inserted: bool = True
    colour: str = "#455a64"
    anchor_key: str = SELECTED_AREA_APERTURE
    optical_reference_downstream_of_anchor_mm: float = 0.0
    layout_center_downstream_of_anchor_mm: float = 0.0
    layout_length_mm: float = 1.0
    owner: str = "detector"
    point_spread_model: str = "gaussian"
    point_spread_sigma_x_mm: float = 0.1
    point_spread_sigma_y_mm: float = 0.1
    point_spread_rotation_deg: float = 0.0
    point_spread_status: str = "provisional_model_parameter"
    point_spread_source: str = (
        "adjustable_simulator_default_not_instrument_calibration"
    )

    NON_BLOCKING: ClassVar[bool] = False
    INTERACTION_KIND: ClassVar[str] = "recording_plane_stop"

    @property
    def outer_diameter_mm(self):
        return self.outer_width_mm

    @property
    def readout_enabled(self):
        return bool(self.inserted)

    @readout_enabled.setter
    def readout_enabled(self, value):
        self.inserted = bool(value)

    def validate(self):
        self.key = canonical_recording_plane_key(self.key)
        if self.key not in STEM_DETECTOR_KEYS:
            raise ValueError("Unknown STEM detector key.")
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "STEM detectors must follow the Selected Area Aperture."
            )
        geometry = str(self.geometry).lower()
        if geometry not in {"disk", "annulus"}:
            raise ValueError("STEM detector geometry must be disk or annulus.")
        self.geometry = geometry
        if self.outer_width_mm <= 0.0:
            raise ValueError("STEM detector outer diameter must be positive.")
        if self.inner_diameter_mm < 0.0:
            raise ValueError("STEM detector inner diameter cannot be negative.")
        if (
            geometry == "annulus"
            and self.inner_diameter_mm >= self.outer_width_mm
        ):
            raise ValueError(
                "STEM detector inner diameter must be smaller than its "
                "outer diameter."
            )
        if self.layout_length_mm <= 0.0:
            raise ValueError("STEM detector layout length must be positive.")
        validate_component_point_spread(self)
        return self

    def resolve_against(self, anchor_z_mm):
        self.z_mm = (
            float(anchor_z_mm)
            + float(self.optical_reference_downstream_of_anchor_mm)
        )
        return self

    def set_optical_reference_z_mm(self, anchor_z_mm, z_mm):
        offset_mm = (
            float(z_mm) - float(anchor_z_mm)
        )
        self.optical_reference_downstream_of_anchor_mm = offset_mm
        return self.resolve_against(anchor_z_mm)

    def hit_mask(self, x_mm, y_mm):
        x_mm = np.asarray(x_mm, dtype=float)
        y_mm = np.asarray(y_mm, dtype=float)
        radius = np.hypot(x_mm, y_mm)
        outer = float(self.outer_width_mm) / 2.0
        if self.geometry == "annulus":
            inner = float(self.inner_diameter_mm) / 2.0
            return (radius >= inner) & (radius <= outer)
        return radius <= outer


STEM_DETECTOR_DEFINITIONS = (
    StemDetectorDefinition(
        key=HAADF_DETECTOR,
        label=str(_DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["name"]),
        optical_reference_downstream_of_anchor_mm=downstream_optical_offset_mm(
            HAADF_DETECTOR
        ),
        layout_center_downstream_of_anchor_mm=downstream_offset_mm(
            HAADF_DETECTOR
        ),
        layout_length_mm=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["length_mm"]
        ),
        geometry="annulus",
        outer_width_mm=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["outer_width_mm"]
        ),
        inner_diameter_mm=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["inner_diameter_mm"]
        ),
        colour="#d81b60",
        point_spread_model=str(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_model"]
        ),
        point_spread_sigma_x_mm=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_sigma_x_mm"]
        ),
        point_spread_sigma_y_mm=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_sigma_y_mm"]
        ),
        point_spread_rotation_deg=float(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_rotation_deg"]
        ),
        point_spread_status=str(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_status"]
        ),
        point_spread_source=str(
            _DEFAULT_MANIFEST_PARTS[HAADF_DETECTOR]["point_spread_source"]
        ),
    ),
    StemDetectorDefinition(
        key=DARK_FIELD_DETECTOR,
        label=str(_DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["name"]),
        optical_reference_downstream_of_anchor_mm=downstream_optical_offset_mm(
            DARK_FIELD_DETECTOR
        ),
        layout_center_downstream_of_anchor_mm=downstream_offset_mm(
            DARK_FIELD_DETECTOR
        ),
        layout_length_mm=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["length_mm"]
        ),
        geometry="annulus",
        outer_width_mm=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["outer_width_mm"]
        ),
        inner_diameter_mm=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["inner_diameter_mm"]
        ),
        colour="#fb8c00",
        point_spread_model=str(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_model"]
        ),
        point_spread_sigma_x_mm=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_sigma_x_mm"]
        ),
        point_spread_sigma_y_mm=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_sigma_y_mm"]
        ),
        point_spread_rotation_deg=float(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_rotation_deg"]
        ),
        point_spread_status=str(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_status"]
        ),
        point_spread_source=str(
            _DEFAULT_MANIFEST_PARTS[DARK_FIELD_DETECTOR]["point_spread_source"]
        ),
    ),
    StemDetectorDefinition(
        key=BRIGHT_FIELD_DETECTOR,
        label=str(_DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["name"]),
        optical_reference_downstream_of_anchor_mm=downstream_optical_offset_mm(
            BRIGHT_FIELD_DETECTOR
        ),
        layout_center_downstream_of_anchor_mm=downstream_offset_mm(
            BRIGHT_FIELD_DETECTOR
        ),
        layout_length_mm=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["length_mm"]
        ),
        geometry="disk",
        outer_width_mm=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["outer_width_mm"]
        ),
        inner_diameter_mm=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["inner_diameter_mm"]
        ),
        colour="#1e88e5",
        point_spread_model=str(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_model"]
        ),
        point_spread_sigma_x_mm=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_sigma_x_mm"]
        ),
        point_spread_sigma_y_mm=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_sigma_y_mm"]
        ),
        point_spread_rotation_deg=float(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_rotation_deg"]
        ),
        point_spread_status=str(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_status"]
        ),
        point_spread_source=str(
            _DEFAULT_MANIFEST_PARTS[BRIGHT_FIELD_DETECTOR]["point_spread_source"]
        ),
    ),
)

STEM_DETECTOR_DEFINITION_BY_KEY = {
    definition.key: definition
    for definition in STEM_DETECTOR_DEFINITIONS
}


def create_stem_detectors(
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    return [
        definition.create_component(anchor_z_mm).validate()
        for definition in STEM_DETECTOR_DEFINITIONS
    ]


def stem_detector_from_dict(
    data,
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    values = dict(data)
    key = canonical_recording_plane_key(values.get("key", ""))
    definition = STEM_DETECTOR_DEFINITION_BY_KEY[key]
    component = definition.create_component(anchor_z_mm)
    known = component.__dataclass_fields__
    for field, value in values.items():
        if field in known and field in {
            "inserted",
            "colour",
        }:
            setattr(component, field, value)
    component.key = key
    component.name = definition.label
    legacy_anchor = values.get("anchor_key") != SELECTED_AREA_APERTURE
    component.anchor_key = SELECTED_AREA_APERTURE
    if legacy_anchor:
        component.optical_reference_downstream_of_anchor_mm = (
            downstream_optical_offset_mm(key)
        )
        component.layout_center_downstream_of_anchor_mm = downstream_offset_mm(
            key
        )
    return component.resolve_against(anchor_z_mm).validate()
