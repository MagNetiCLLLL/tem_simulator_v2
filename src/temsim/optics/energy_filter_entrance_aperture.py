"""Canonical continuously adjustable Energy Filter Entrance Aperture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.component_keys import (
    ENERGY_FILTER_ENTRANCE_APERTURE,
    SELECTED_AREA_APERTURE,
)
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm


@dataclass(frozen=True)
class EnergyFilterEntranceApertureDefinition:
    key: str = ENERGY_FILTER_ENTRANCE_APERTURE
    label: str = "Energy Filter Entrance Aperture"
    anchor_key: str = SELECTED_AREA_APERTURE
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        ENERGY_FILTER_ENTRANCE_APERTURE
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        ENERGY_FILTER_ENTRANCE_APERTURE
    )
    mechanical_length_mm: float = 4.0
    mechanical_outer_diameter_mm: float = 120.0
    mechanical_bore_diameter_mm: float = 12.5
    plate_thickness_mm: float = 1.0
    default_radius_mm: float = 2.5
    maximum_radius_mm: float = 6.25
    colour: str = "#6d4c41"
    owner: str = "energy_filter"
    kind: str = "continuous_aperture"
    shape_profile: str = "adjustable_circular_aperture"
    interaction_kind: str = "hard_edge_circular_stop"

    @property
    def name(self):
        return self.label

    @property
    def radius_mm(self):
        return self.default_radius_mm

    def create_component(
        self,
        selected_area_z_mm=(
            SELECTED_AREA_APERTURE_DEFINITION
            .standalone_optical_reference_z_mm
        ),
    ):
        return EnergyFilterEntranceApertureComponent(
            name=self.label,
            key=self.key,
            z_mm=(
                float(selected_area_z_mm)
                + self.optical_reference_downstream_of_anchor_mm
            ),
            radius_mm=self.default_radius_mm,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            enabled=True,
            installed=False,
            colour=self.colour,
            anchor_key=self.anchor_key,
            optical_reference_downstream_of_anchor_mm=(
                self.optical_reference_downstream_of_anchor_mm
            ),
            layout_center_downstream_of_anchor_mm=(
                self.layout_center_downstream_of_anchor_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_bore_diameter_mm=(
                self.mechanical_bore_diameter_mm
            ),
            plate_thickness_mm=self.plate_thickness_mm,
            maximum_radius_mm=self.maximum_radius_mm,
        )


@dataclass
class EnergyFilterEntranceApertureComponent:
    name: str = "Energy Filter Entrance Aperture"
    key: str = ENERGY_FILTER_ENTRANCE_APERTURE
    z_mm: float = (
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
        + downstream_offset_mm(ENERGY_FILTER_ENTRANCE_APERTURE)
    )
    radius_mm: float = 2.5
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    enabled: bool = True
    installed: bool = False
    colour: str = "#6d4c41"
    anchor_key: str = SELECTED_AREA_APERTURE
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        ENERGY_FILTER_ENTRANCE_APERTURE
    )
    layout_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        ENERGY_FILTER_ENTRANCE_APERTURE
    )
    mechanical_length_mm: float = 4.0
    mechanical_outer_diameter_mm: float = 120.0
    mechanical_bore_diameter_mm: float = 12.5
    plate_thickness_mm: float = 1.0
    maximum_radius_mm: float = 6.25

    @property
    def owner(self):
        return "energy_filter"

    @property
    def kind(self):
        return "continuous_aperture"

    @property
    def shape_profile(self):
        return "adjustable_circular_aperture"

    @property
    def interaction_kind(self):
        return "hard_edge_circular_stop"

    @property
    def effective_aperture_radius_mm(self):
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

    def validate(self):
        if self.key != ENERGY_FILTER_ENTRANCE_APERTURE:
            raise ValueError(
                "Energy Filter Entrance Aperture has a non-canonical key."
            )
        if self.anchor_key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Energy Filter Entrance Aperture must follow the "
                "Selected Area Aperture."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Energy Filter Entrance Aperture length must be positive."
            )
        if (
            self.mechanical_outer_diameter_mm
            <= self.mechanical_bore_diameter_mm
        ):
            raise ValueError(
                "Energy Filter Entrance Aperture outer diameter must "
                "exceed its bore."
            )
        if self.plate_thickness_mm <= 0.0:
            raise ValueError(
                "Energy Filter Entrance Aperture plate must have thickness."
            )
        if self.maximum_radius_mm <= 0.0:
            raise ValueError(
                "Energy Filter Entrance Aperture maximum radius must "
                "be positive."
            )
        if not 0.0 <= self.radius_mm <= self.maximum_radius_mm:
            raise ValueError(
                "Energy Filter Entrance Aperture radius must lie within "
                "its continuous range."
            )
        if 2.0 * self.maximum_radius_mm > self.mechanical_bore_diameter_mm:
            raise ValueError(
                "Energy Filter Entrance Aperture opening must fit "
                "inside its bore."
            )
        return self

    def resolve_against(self, selected_area_z_mm):
        self.optical_reference_downstream_of_anchor_mm = float(
            self.layout_center_downstream_of_anchor_mm
        )
        self.z_mm = (
            float(selected_area_z_mm)
            + float(self.layout_center_downstream_of_anchor_mm)
        )
        return self

    def set_optical_reference_z_mm(self, selected_area_z_mm, z_mm):
        offset_mm = (
            float(z_mm) - float(selected_area_z_mm)
        )
        self.layout_center_downstream_of_anchor_mm = offset_mm
        self.optical_reference_downstream_of_anchor_mm = offset_mm
        return self.resolve_against(selected_area_z_mm)

    def transmission_mask(self, x_mm, y_mm):
        radial_distance_mm = np.hypot(
            np.asarray(x_mm, dtype=float) - self.offset_x_mm,
            np.asarray(y_mm, dtype=float) - self.offset_y_mm,
        )
        return radial_distance_mm <= self.radius_mm


ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION = (
    EnergyFilterEntranceApertureDefinition()
)


def create_energy_filter_entrance_aperture(
    selected_area_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    return (
        ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION
        .create_component(selected_area_z_mm)
        .validate()
    )


def energy_filter_entrance_aperture_from_dict(
    data,
    selected_area_z_mm=(
        SELECTED_AREA_APERTURE_DEFINITION
        .standalone_optical_reference_z_mm
    ),
):
    values = dict(data)
    component = create_energy_filter_entrance_aperture(selected_area_z_mm)
    known = component.__dataclass_fields__
    for field, value in values.items():
        if field in known:
            setattr(component, field, value)
    component.key = ENERGY_FILTER_ENTRANCE_APERTURE
    component.name = ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION.label
    legacy_anchor = values.get("anchor_key") != SELECTED_AREA_APERTURE
    component.anchor_key = SELECTED_AREA_APERTURE
    if legacy_anchor:
        offset = downstream_offset_mm(
            ENERGY_FILTER_ENTRANCE_APERTURE
        )
        component.optical_reference_downstream_of_anchor_mm = offset
        component.layout_center_downstream_of_anchor_mm = offset
    if "optical_reference_downstream_of_anchor_mm" not in values:
        component.optical_reference_downstream_of_anchor_mm = (
            float(values.get("z_mm", component.z_mm))
            - float(selected_area_z_mm)
        )
    return component.resolve_against(selected_area_z_mm).validate()
