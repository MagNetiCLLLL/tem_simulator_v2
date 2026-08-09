"""Single-source condenser-aperture components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    canonical_aperture_key,
)


@dataclass(frozen=True)
class ContinuousApertureDefinition:
    key: str
    label: str
    center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    default_radius_mm: float
    maximum_radius_mm: float
    colour: str
    owner: str = "condenser_assembly"
    kind: str = "continuous_aperture"
    shape_profile: str = "adjustable_circular_aperture"
    interaction_kind: str = "hard_edge_circular_stop"

    @property
    def effective_aperture_radius_mm(self):
        return self.default_radius_mm

    @property
    def mechanical_center_from_tip_mm(self):
        """Compatibility view of the single canonical aperture coordinate."""

        return self.center_from_tip_mm

    @property
    def optical_reference_from_tip_mm(self):
        """Compatibility view of the single canonical aperture coordinate."""

        return self.center_from_tip_mm

    def create_component(self):
        component = ContinuousApertureComponent(
            name=self.label,
            key=self.key,
            z_mm=self.center_from_tip_mm,
            radius_mm=self.default_radius_mm,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            enabled=True,
            colour=self.colour,
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
        object.__setattr__(component, "_toml_geometry_owned", True)
        return component


@dataclass
class ContinuousApertureComponent:
    """One continuously adjustable circular aperture.

    ``radius_mm`` is a continuous floating-point control. There is deliberately
    no discrete aperture list or selected-index state.
    """

    name: str
    key: str
    z_mm: float
    radius_mm: float
    offset_x_mm: float
    offset_y_mm: float
    enabled: bool
    colour: str
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    maximum_radius_mm: float

    _TOML_GEOMETRY_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "z_mm",
        "mechanical_length_mm",
        "mechanical_outer_diameter_mm",
        "mechanical_bore_diameter_mm",
        "plate_thickness_mm",
        "maximum_radius_mm",
    })

    def __setattr__(self, name, value):
        if (
            name in self._TOML_GEOMETRY_FIELDS
            and getattr(self, "_toml_geometry_owned", False)
            and not getattr(self, "_applying_toml_geometry", False)
        ):
            raise AttributeError(
                f"{self.name} geometry is owned by the selected Column TOML"
            )
        object.__setattr__(self, name, value)

    def apply_manifest_geometry(self, **geometry):
        unknown = set(geometry) - self._TOML_GEOMETRY_FIELDS
        if unknown:
            raise ValueError(
                f"Unsupported {self.name} TOML geometry fields: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "_applying_toml_geometry", True)
        try:
            for name, value in geometry.items():
                object.__setattr__(self, name, float(value))
        finally:
            object.__setattr__(self, "_applying_toml_geometry", False)
        object.__setattr__(self, "_toml_geometry_owned", True)
        return self

    @property
    def mechanical_center_from_tip_mm(self):
        """Mechanical drawing coordinate; identical to the optical plane."""

        return self.z_mm

    @mechanical_center_from_tip_mm.setter
    def mechanical_center_from_tip_mm(self, value):
        self.z_mm = float(value)

    @property
    def optical_reference_from_tip_mm(self):
        """Ray-interaction coordinate; identical to the mechanical centre."""

        return self.z_mm

    @optical_reference_from_tip_mm.setter
    def optical_reference_from_tip_mm(self, value):
        self.z_mm = float(value)

    @property
    def owner(self):
        return "condenser_assembly"

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
        if self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(
                f"{self.name} mechanical centre must follow the tip."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                f"{self.name} mechanical length must be positive."
            )
        if (
            self.mechanical_outer_diameter_mm
            <= self.mechanical_bore_diameter_mm
        ):
            raise ValueError(
                f"{self.name} outer diameter must exceed its mechanical bore."
            )
        if self.plate_thickness_mm <= 0.0:
            raise ValueError(
                f"{self.name} plate thickness must be positive."
            )
        if self.maximum_radius_mm <= 0.0:
            raise ValueError(
                f"{self.name} maximum radius must be positive."
            )
        if not 0.0 <= self.radius_mm <= self.maximum_radius_mm:
            raise ValueError(
                f"{self.name} radius must lie within its continuous range."
            )
        if 2.0 * self.maximum_radius_mm > self.mechanical_bore_diameter_mm:
            raise ValueError(
                f"{self.name} continuous opening must fit inside its bore."
            )
        return self

    def apply_optical_position(self):
        return self

    def transmission_mask(self, x_mm, y_mm):
        """Return the hard-edge transmission mask at the optical plane."""

        radial_distance_mm = np.hypot(
            np.asarray(x_mm, dtype=float) - self.offset_x_mm,
            np.asarray(y_mm, dtype=float) - self.offset_y_mm,
        )
        return radial_distance_mm <= self.radius_mm

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_bore_diameter_mm": self.mechanical_bore_diameter_mm,
            "plate_thickness_mm": self.plate_thickness_mm,
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


_DEFAULT_COLUMN_MANIFEST = "column/C3_ProbeCorrector.toml"
_DEFAULT_GUN_MANIFEST = "gun/FEG.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    _DEFAULT_GUN_MANIFEST,
    "exit",
)


def _aperture_definition(key, label, colour):
    part = module_manifest.part_data(_DEFAULT_COLUMN_MANIFEST, key)
    return ContinuousApertureDefinition(
        key=key,
        label=label,
        center_from_tip_mm=(
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(part["optical_reference_local_z_mm"])
        ),
        mechanical_length_mm=float(part["length_mm"]),
        mechanical_outer_diameter_mm=float(
            part["mechanical_outer_diameter_mm"]
        ),
        mechanical_bore_diameter_mm=float(
            part["mechanical_bore_diameter_mm"]
        ),
        plate_thickness_mm=float(part["plate_thickness_mm"]),
        # In a non-monochromated column the C3 aperture is parked at its
        # approximately 2 mm open radius and does not limit illumination.
        # The C2 aperture remains the normal convergence-defining stop.
        default_radius_mm=(
            2.0 if key == CONDENSER_APERTURE_3 else 0.05
        ),
        maximum_radius_mm=float(part["maximum_radius_mm"]),
        colour=colour,
    )


CONDENSER_APERTURE_2_DEFINITION = _aperture_definition(
    CONDENSER_APERTURE_2,
    "C2 Aperture",
    "#00acc1",
)

CONDENSER_APERTURE_3_DEFINITION = _aperture_definition(
    CONDENSER_APERTURE_3,
    "C3 Aperture",
    "#00897b",
)

CONDENSER_APERTURE_DEFINITION_BY_KEY = {
    definition.key: definition
    for definition in (
        CONDENSER_APERTURE_2_DEFINITION,
        CONDENSER_APERTURE_3_DEFINITION,
    )
}


def create_condenser_aperture_2():
    return CONDENSER_APERTURE_2_DEFINITION.create_component()


def create_condenser_aperture_3():
    return CONDENSER_APERTURE_3_DEFINITION.create_component()


def condenser_aperture_from_dict(data):
    """Restore one condenser aperture into its owned component type."""

    values = dict(data)
    key = canonical_aperture_key(values["key"])
    component = CONDENSER_APERTURE_DEFINITION_BY_KEY[key].create_component()
    allowed = ContinuousApertureComponent.__dataclass_fields__
    for attribute, value in values.items():
        if (
            attribute in allowed
            and attribute
            not in ContinuousApertureComponent._TOML_GEOMETRY_FIELDS
        ):
            setattr(component, attribute, value)
    component.key = canonical_aperture_key(component.key)
    return component
