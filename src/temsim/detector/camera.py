"""Canonical square Camera detector and recording stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import CAMERA, SELECTED_AREA_APERTURE
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm


_DEFAULT_MANIFEST_PART = module_manifest.part_data(
    "project_and_recording_system/NoEnergyFilter.toml", CAMERA
)


@dataclass(frozen=True)
class CameraDetectorDefinition:
    key: str = CAMERA
    label: str = str(_DEFAULT_MANIFEST_PART["name"])
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        CAMERA
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        CAMERA
    )
    layout_length_mm: float = float(_DEFAULT_MANIFEST_PART["length_mm"])
    sensor_width_mm: float = float(
        _DEFAULT_MANIFEST_PART["outer_width_mm"]
    )
    pixels: int = 2048
    colour: str = "#5e35b1"
    anchor_key: str = SELECTED_AREA_APERTURE
    owner: str = "detector"
    kind: str = "detector"
    shape_profile: str = "detector_plane"
    external_envelope: str = "400-800 mm envelope"
    interaction_kind: str = "recording_plane_stop"
    detector_axis_rotation_deg: float = float(
        _DEFAULT_MANIFEST_PART["detector_axis_rotation_deg"]
    )
    detector_flip_x: bool = bool(
        _DEFAULT_MANIFEST_PART["detector_flip_x"]
    )
    detector_flip_y: bool = bool(
        _DEFAULT_MANIFEST_PART["detector_flip_y"]
    )
    detector_orientation_uncertainty_deg: float = float(
        _DEFAULT_MANIFEST_PART["detector_orientation_uncertainty_deg"]
    )
    detector_orientation_status: str = str(
        _DEFAULT_MANIFEST_PART["detector_orientation_status"]
    )
    detector_orientation_source: str = str(
        _DEFAULT_MANIFEST_PART["detector_orientation_source"]
    )

    @property
    def name(self):
        return self.label

    @property
    def active_width_mm(self):
        return self.sensor_width_mm

    def create_component(
        self,
        anchor_z_mm=(
            SELECTED_AREA_APERTURE_DEFINITION
            .standalone_optical_reference_z_mm
        ),
    ):
        return CameraDetectorComponent(
            key=self.key,
            name=self.label,
            z_mm=(
                float(anchor_z_mm)
                + self.optical_reference_downstream_of_anchor_mm
            ),
            geometry="square",
            outer_width_mm=self.sensor_width_mm,
            inner_diameter_mm=0.0,
            inserted=True,
            colour=self.colour,
            pixels=self.pixels,
            anchor_key=self.anchor_key,
            optical_reference_downstream_of_anchor_mm=(
                self.optical_reference_downstream_of_anchor_mm
            ),
            layout_center_downstream_of_anchor_mm=(
                self.layout_center_downstream_of_anchor_mm
            ),
            layout_length_mm=self.layout_length_mm,
            owner=self.owner,
            kind=self.kind,
            shape_profile=self.shape_profile,
            external_envelope=self.external_envelope,
            detector_axis_rotation_deg=self.detector_axis_rotation_deg,
            detector_flip_x=self.detector_flip_x,
            detector_flip_y=self.detector_flip_y,
            detector_orientation_uncertainty_deg=(
                self.detector_orientation_uncertainty_deg
            ),
            detector_orientation_status=self.detector_orientation_status,
            detector_orientation_source=self.detector_orientation_source,
        )


@dataclass
class CameraDetectorComponent:
    key: str = CAMERA
    name: str = str(_DEFAULT_MANIFEST_PART["name"])
    z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION.standalone_optical_reference_z_mm
        + downstream_offset_mm(CAMERA)
    )
    geometry: str = "square"
    outer_width_mm: float = float(
        _DEFAULT_MANIFEST_PART["outer_width_mm"]
    )
    inner_diameter_mm: float = float(
        _DEFAULT_MANIFEST_PART["inner_diameter_mm"]
    )
    inserted: bool = True
    colour: str = "#5e35b1"
    pixels: int = 2048
    anchor_key: str = SELECTED_AREA_APERTURE
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        CAMERA
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        CAMERA
    )
    layout_length_mm: float = float(_DEFAULT_MANIFEST_PART["length_mm"])
    owner: str = "detector"
    kind: str = "detector"
    shape_profile: str = "detector_plane"
    external_envelope: str = "400-800 mm envelope"
    detector_axis_rotation_deg: float = float(
        _DEFAULT_MANIFEST_PART["detector_axis_rotation_deg"]
    )
    detector_flip_x: bool = bool(
        _DEFAULT_MANIFEST_PART["detector_flip_x"]
    )
    detector_flip_y: bool = bool(
        _DEFAULT_MANIFEST_PART["detector_flip_y"]
    )
    detector_orientation_uncertainty_deg: float = float(
        _DEFAULT_MANIFEST_PART["detector_orientation_uncertainty_deg"]
    )
    detector_orientation_status: str = str(
        _DEFAULT_MANIFEST_PART["detector_orientation_status"]
    )
    detector_orientation_source: str = str(
        _DEFAULT_MANIFEST_PART["detector_orientation_source"]
    )

    NON_BLOCKING: ClassVar[bool] = False
    INTERACTION_KIND: ClassVar[str] = "recording_plane_stop"

    @property
    def width_mm(self):
        return self.outer_width_mm

    @width_mm.setter
    def width_mm(self, value):
        self.outer_width_mm = float(value)

    @property
    def active_width_mm(self):
        return self.outer_width_mm

    def validate(self):
        if self.key != CAMERA:
            raise ValueError("Camera has a non-canonical key.")
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Camera must follow the Selected Area Aperture."
            )
        if str(self.geometry).lower() != "square":
            raise ValueError("Camera sensor geometry must be square.")
        self.geometry = "square"
        if self.outer_width_mm <= 0.0:
            raise ValueError("Camera sensor width must be positive.")
        if self.inner_diameter_mm != 0.0:
            raise ValueError("Camera cannot have an inner diameter.")
        if int(self.pixels) < 1:
            raise ValueError("Camera pixel count must be positive.")
        self.pixels = int(self.pixels)
        if self.layout_length_mm <= 0.0:
            raise ValueError("Camera mechanical length must be positive.")
        if not np.isfinite(float(self.detector_axis_rotation_deg)):
            raise ValueError("Camera detector-axis rotation must be finite.")
        uncertainty = float(self.detector_orientation_uncertainty_deg)
        if not np.isfinite(uncertainty) or not 0.0 <= uncertainty <= 180.0:
            raise ValueError(
                "Camera orientation uncertainty must be between 0 and 180 deg."
            )
        if not str(self.detector_orientation_status).strip():
            raise ValueError("Camera orientation status must not be empty.")
        if not str(self.detector_orientation_source).strip():
            raise ValueError("Camera orientation source must not be empty.")
        return self

    def resolve_against(self, anchor_z_mm):
        self.optical_reference_downstream_of_anchor_mm = float(
            self.layout_center_downstream_of_anchor_mm
        )
        self.z_mm = (
            float(anchor_z_mm)
            + float(self.layout_center_downstream_of_anchor_mm)
        )
        return self

    def set_optical_reference_z_mm(self, anchor_z_mm, z_mm):
        offset_mm = (
            float(z_mm) - float(anchor_z_mm)
        )
        self.layout_center_downstream_of_anchor_mm = offset_mm
        self.optical_reference_downstream_of_anchor_mm = offset_mm
        return self.resolve_against(anchor_z_mm)

    def hit_mask(self, x_mm, y_mm):
        x_mm = np.asarray(x_mm, dtype=float)
        y_mm = np.asarray(y_mm, dtype=float)
        half_width = float(self.outer_width_mm) / 2.0
        return (
            (np.abs(x_mm) <= half_width)
            & (np.abs(y_mm) <= half_width)
        )


CAMERA_DETECTOR_DEFINITION = CameraDetectorDefinition()


def create_camera_detector(
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    return (
        CAMERA_DETECTOR_DEFINITION
        .create_component(anchor_z_mm)
        .validate()
    )


def camera_detector_from_dict(
    data,
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    values = dict(data)
    component = create_camera_detector(anchor_z_mm)
    known = component.__dataclass_fields__
    for field, value in values.items():
        if field in known and field in {
            "inserted",
            "pixels",
            "colour",
            "detector_axis_rotation_deg",
            "detector_flip_x",
            "detector_flip_y",
            "detector_orientation_uncertainty_deg",
            "detector_orientation_status",
            "detector_orientation_source",
        }:
            setattr(component, field, value)
    component.key = CAMERA
    component.name = CAMERA_DETECTOR_DEFINITION.label
    legacy_anchor = values.get("anchor_key") != SELECTED_AREA_APERTURE
    component.anchor_key = SELECTED_AREA_APERTURE
    if legacy_anchor:
        offset = downstream_offset_mm(CAMERA)
        component.optical_reference_downstream_of_anchor_mm = offset
        component.layout_center_downstream_of_anchor_mm = offset
    return component.resolve_against(anchor_z_mm).validate()
