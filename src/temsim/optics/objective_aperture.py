"""Objective-aperture mechanics and its co-located hard-edge stop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim import module_manifest
from temsim.component_keys import OBJECTIVE_APERTURE

_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml",
    "exit",
)


@dataclass(frozen=True)
class ObjectiveApertureDefinition:
    key: str
    label: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    optical_plane_below_sample_mm: float
    default_radius_mm: float
    maximum_radius_mm: float
    colour: str
    owner: str
    kind: str
    shape_profile: str
    interaction_kind: str

    @property
    def name(self):
        return self.label

    @property
    def z_mm(self):
        return (
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(_DEFAULT_SAMPLE_PART["local_center_z_mm"])
            + self.optical_plane_below_sample_mm
        )

    @property
    def radius_mm(self):
        return self.default_radius_mm

    @property
    def effective_aperture_radius_mm(self):
        return self.default_radius_mm

    def create_component(self, sample_z_mm=None):
        if sample_z_mm is None:
            sample_z_mm = (
                _DEFAULT_COLUMN_ORIGIN_Z_MM
                + float(_DEFAULT_SAMPLE_PART["local_center_z_mm"])
            )
        return ObjectiveApertureComponent(
            name=self.label,
            key=self.key,
            z_mm=(
                float(sample_z_mm) + self.optical_plane_below_sample_mm
            ),
            radius_mm=self.default_radius_mm,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            enabled=False,
            colour=self.colour,
            mechanical_center_below_sample_mm=(
                self.mechanical_center_below_sample_mm
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
class ObjectiveApertureComponent:
    """Pole-gap cartridge whose plate centre is the interaction plane."""

    name: str
    key: str
    z_mm: float
    radius_mm: float
    offset_x_mm: float
    offset_y_mm: float
    enabled: bool
    colour: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    maximum_radius_mm: float

    @property
    def owner(self):
        return "objective"

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
        if self.key != OBJECTIVE_APERTURE:
            raise ValueError("Objective Aperture must use its canonical key.")
        if self.mechanical_center_below_sample_mm <= 0.0:
            raise ValueError(
                "Objective Aperture must remain below the specimen."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Objective Aperture mechanical length must be positive."
            )
        if (
            self.mechanical_outer_diameter_mm
            <= self.mechanical_bore_diameter_mm
        ):
            raise ValueError(
                "Objective Aperture body must exceed its mechanical bore."
            )
        if self.plate_thickness_mm <= 0.0:
            raise ValueError(
                "Objective Aperture plate thickness must be positive."
            )
        if self.maximum_radius_mm <= 0.0:
            raise ValueError(
                "Objective Aperture maximum radius must be positive."
            )
        if not 0.0 <= self.radius_mm <= self.maximum_radius_mm:
            raise ValueError(
                "Objective Aperture radius is outside its continuous range."
            )
        if 2.0 * self.maximum_radius_mm > self.mechanical_bore_diameter_mm:
            raise ValueError(
                "Objective Aperture opening must fit inside its bore."
            )
        return self

    def validate_co_located_with_mechanics(self, sample_z_mm):
        """Reject a virtual stop displaced from the physical aperture plate."""

        expected_z_mm = (
            float(sample_z_mm)
            + float(self.mechanical_center_below_sample_mm)
        )
        if not np.isclose(self.z_mm, expected_z_mm, atol=1.0e-9, rtol=0.0):
            raise ValueError(
                "Objective Aperture optical plane must equal its mechanical "
                "centre."
            )
        return self

    def validate_between_poles(self, objective_lens):
        """Validate the cartridge inside the pole-piece gap below sample."""

        lower_inner_face_mm = (
            -float(objective_lens.inner_face_gap_mm) / 2.0
            - float(objective_lens.sample_axial_offset_mm)
        )
        sample_plane_mm = -float(
            objective_lens.sample_axial_offset_mm
        )
        cartridge_center_mm = (
            -float(self.mechanical_center_below_sample_mm)
            - float(objective_lens.sample_axial_offset_mm)
        )
        cartridge_half_length_mm = self.mechanical_length_mm / 2.0
        if not (
            lower_inner_face_mm
            <= cartridge_center_mm - cartridge_half_length_mm
            and cartridge_center_mm + cartridge_half_length_mm
            <= sample_plane_mm
        ):
            raise ValueError(
                "Objective Aperture cartridge must remain below the "
                "sample and between the Objective pole pieces."
            )
        return self

    def validate_nested_in(self, objective_lens):
        """Backward-compatible alias for the pole-gap diagnostic."""

        return self.validate_between_poles(objective_lens)

    def transmission_mask(self, x_mm, y_mm):
        radial_distance_mm = np.hypot(
            np.asarray(x_mm, dtype=float) - self.offset_x_mm,
            np.asarray(y_mm, dtype=float) - self.offset_y_mm,
        )
        return radial_distance_mm <= self.radius_mm

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
            "mechanical_bore_diameter_mm": (
                self.mechanical_bore_diameter_mm
            ),
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


_DEFAULT_APERTURE_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH,
    OBJECTIVE_APERTURE,
)
_DEFAULT_SAMPLE_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH,
    "sample",
)
OBJECTIVE_APERTURE_DEFINITION = ObjectiveApertureDefinition(
    key=OBJECTIVE_APERTURE,
    label=str(_DEFAULT_APERTURE_PART["name"]),
    mechanical_center_below_sample_mm=(
        float(_DEFAULT_APERTURE_PART["local_center_z_mm"])
        - float(_DEFAULT_SAMPLE_PART["local_center_z_mm"])
    ),
    mechanical_length_mm=float(_DEFAULT_APERTURE_PART["length_mm"]),
    mechanical_outer_diameter_mm=float(
        _DEFAULT_APERTURE_PART["mechanical_outer_diameter_mm"]
    ),
    mechanical_bore_diameter_mm=float(
        _DEFAULT_APERTURE_PART["mechanical_bore_diameter_mm"]
    ),
    plate_thickness_mm=float(_DEFAULT_APERTURE_PART["plate_thickness_mm"]),
    optical_plane_below_sample_mm=(
        float(_DEFAULT_APERTURE_PART["optical_reference_local_z_mm"])
        - float(_DEFAULT_SAMPLE_PART["local_center_z_mm"])
    ),
    default_radius_mm=0.05,
    maximum_radius_mm=float(_DEFAULT_APERTURE_PART["maximum_radius_mm"]),
    colour="#e53935",
    owner="objective",
    kind="continuous_aperture",
    shape_profile="adjustable_circular_aperture",
    interaction_kind="hard_edge_circular_stop",
)


def create_objective_aperture(sample_z_mm=None):
    return OBJECTIVE_APERTURE_DEFINITION.create_component(
        sample_z_mm
    ).validate()


def objective_aperture_from_dict(
    data,
    sample_z_mm=None,
    legacy_objective_lens=None,
):
    """Restore schema-32 data or upgrade the earlier generic aperture row."""

    values = dict(data)
    component = create_objective_aperture(sample_z_mm)
    legacy_lens = dict(legacy_objective_lens or {})
    if "mechanical_center_below_sample_mm" not in values:
        values["mechanical_center_below_sample_mm"] = legacy_lens.get(
            "objective_aperture_depth_below_sample_mm",
            component.mechanical_center_below_sample_mm,
        )
    if "mechanical_length_mm" not in values:
        values["mechanical_length_mm"] = legacy_lens.get(
            "objective_aperture_mechanical_length_mm",
            component.mechanical_length_mm,
        )
    for attribute in (
        "radius_mm",
        "offset_x_mm",
        "offset_y_mm",
        "enabled",
        "colour",
    ):
        if attribute in values:
            setattr(component, attribute, values[attribute])
    component.key = OBJECTIVE_APERTURE
    component.name = "Objective Aperture"
    return component.validate()
