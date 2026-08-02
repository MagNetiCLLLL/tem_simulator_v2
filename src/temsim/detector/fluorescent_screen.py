"""Canonical retractable fluorescent-screen recording stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim.component_keys import (
    FLUORESCENT_SCREEN,
    SELECTED_AREA_APERTURE,
)
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm


@dataclass(frozen=True)
class FluorescentScreenDefinition:
    key: str = FLUORESCENT_SCREEN
    label: str = "Fluorescent Screen"
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        FLUORESCENT_SCREEN
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        FLUORESCENT_SCREEN
    )
    layout_length_mm: float = 320.0
    active_diameter_mm: float = 80.0
    colour: str = "#43a047"
    anchor_key: str = SELECTED_AREA_APERTURE
    owner: str = "detector"
    kind: str = "detector"
    shape_profile: str = "detector_plane"
    external_envelope: str = "D400-600 mm"
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
        return FluorescentScreenComponent(
            key=self.key,
            name=self.label,
            z_mm=(
                float(anchor_z_mm)
                + self.optical_reference_downstream_of_anchor_mm
            ),
            geometry="disk",
            outer_width_mm=self.active_diameter_mm,
            inner_diameter_mm=0.0,
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
            kind=self.kind,
            shape_profile=self.shape_profile,
            external_envelope=self.external_envelope,
        )


@dataclass
class FluorescentScreenComponent:
    key: str = FLUORESCENT_SCREEN
    name: str = "Fluorescent Screen"
    z_mm: float = 1890.0
    geometry: str = "disk"
    outer_width_mm: float = 80.0
    inner_diameter_mm: float = 0.0
    inserted: bool = True
    colour: str = "#43a047"
    anchor_key: str = SELECTED_AREA_APERTURE
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        FLUORESCENT_SCREEN
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        FLUORESCENT_SCREEN
    )
    layout_length_mm: float = 320.0
    owner: str = "detector"
    kind: str = "detector"
    shape_profile: str = "detector_plane"
    external_envelope: str = "D400-600 mm"

    NON_BLOCKING: ClassVar[bool] = False
    INTERACTION_KIND: ClassVar[str] = "recording_plane_stop"

    @property
    def outer_diameter_mm(self):
        return self.outer_width_mm

    @property
    def active_diameter_mm(self):
        return self.outer_width_mm

    def validate(self):
        if self.key != FLUORESCENT_SCREEN:
            raise ValueError("Fluorescent Screen has a non-canonical key.")
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Fluorescent Screen must follow the Selected Area Aperture."
            )
        if str(self.geometry).lower() != "disk":
            raise ValueError("Fluorescent Screen geometry must be a disk.")
        self.geometry = "disk"
        if self.outer_width_mm <= 0.0:
            raise ValueError(
                "Fluorescent Screen active diameter must be positive."
            )
        if self.inner_diameter_mm != 0.0:
            raise ValueError(
                "Fluorescent Screen cannot have an inner diameter."
            )
        if self.layout_length_mm <= 0.0:
            raise ValueError(
                "Fluorescent Screen mechanical length must be positive."
            )
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
        radius = np.hypot(
            np.asarray(x_mm, dtype=float),
            np.asarray(y_mm, dtype=float),
        )
        return radius <= float(self.outer_width_mm) / 2.0


FLUORESCENT_SCREEN_DEFINITION = FluorescentScreenDefinition()


def create_fluorescent_screen(
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    return (
        FLUORESCENT_SCREEN_DEFINITION
        .create_component(anchor_z_mm)
        .validate()
    )


def fluorescent_screen_from_dict(
    data,
    anchor_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    values = dict(data)
    component = create_fluorescent_screen(anchor_z_mm)
    known = component.__dataclass_fields__
    for field, value in values.items():
        if field in known and field in {
            "inserted",
            "colour",
        }:
            setattr(component, field, value)
    component.key = FLUORESCENT_SCREEN
    component.name = FLUORESCENT_SCREEN_DEFINITION.label
    legacy_anchor = values.get("anchor_key") != SELECTED_AREA_APERTURE
    component.anchor_key = SELECTED_AREA_APERTURE
    if legacy_anchor:
        offset = downstream_offset_mm(FLUORESCENT_SCREEN)
        component.optical_reference_downstream_of_anchor_mm = offset
        component.layout_center_downstream_of_anchor_mm = offset
    return component.resolve_against(anchor_z_mm).validate()
