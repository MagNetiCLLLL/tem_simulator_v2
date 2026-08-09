"""Topology-aware Selected Area Aperture and its two installation stations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    IMAGE_DIFFRACTION_DEFLECTOR,
    IMAGE_CORRECTOR_SAD_PLANE,
    SELECTED_AREA_APERTURE,
)


STANDALONE_INSTALLATION = "standalone"
IMAGE_CORRECTED_INSTALLATION = "image_corrected"
DEFAULT_STANDALONE_GAP_AFTER_IMAGE_DEFLECTOR_MM = 5.0
# Retained as a serialization-migration alias for schema <= 59.
DEFAULT_STANDALONE_GAP_AFTER_DESCAN_MM = (
    DEFAULT_STANDALONE_GAP_AFTER_IMAGE_DEFLECTOR_MM
)

_DEFAULT_GUN_MODULE = "gun/FEG.toml"
_STANDALONE_COLUMN_MODULE = "column/C3_ProbeCorrector.toml"
_IMAGE_CORRECTED_COLUMN_MODULE = (
    "column/C3_ProbeCorrector_ImageCorrector.toml"
)
_DEFAULT_RECORDING_MODULE = (
    "project_and_recording_system/NoEnergyFilter.toml"
)
_SELECTED_AREA_PART = module_manifest.part_data(
    _DEFAULT_RECORDING_MODULE, SELECTED_AREA_APERTURE
)


def _module_span_mm(module_path):
    return (
        module_manifest.port_z_mm(module_path, "exit")
        - module_manifest.port_z_mm(module_path, "entrance")
    )


def _default_station(column_module):
    gun_span = _module_span_mm(_DEFAULT_GUN_MODULE)
    column_origin = (
        gun_span
        - module_manifest.port_z_mm(column_module, "entrance")
    )
    sample_part = module_manifest.part_data(column_module, "sample")
    sample_z_mm = (
        column_origin + float(sample_part["local_center_z_mm"])
    )
    recording_origin = (
        column_origin
        + module_manifest.port_z_mm(column_module, "exit")
        - module_manifest.port_z_mm(
            _DEFAULT_RECORDING_MODULE, "entrance"
        )
    )
    selected_area_center_z_mm = (
        recording_origin
        + float(_SELECTED_AREA_PART["local_center_z_mm"])
    )
    optical_reference_z_mm = (
        selected_area_center_z_mm
        + float(_SELECTED_AREA_PART["optical_reference_local_z_mm"])
        - float(_SELECTED_AREA_PART["local_center_z_mm"])
    )
    return (
        selected_area_center_z_mm - sample_z_mm,
        optical_reference_z_mm,
    )


(
    _STANDALONE_CENTER_BELOW_SAMPLE_MM,
    _STANDALONE_REFERENCE_Z_MM,
) = _default_station(_STANDALONE_COLUMN_MODULE)
(
    _IMAGE_CORRECTED_CENTER_BELOW_SAMPLE_MM,
    _IMAGE_CORRECTED_REFERENCE_Z_MM,
) = _default_station(_IMAGE_CORRECTED_COLUMN_MODULE)


@dataclass(frozen=True)
class SelectedAreaApertureGeometry:
    installation: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_bore_diameter_mm: float
    plate_thickness_mm: float
    optical_reference_z_mm: float
    reference_component_key: str
    owner: str = "image"


@dataclass(frozen=True)
class SelectedAreaApertureDefinition:
    key: str = SELECTED_AREA_APERTURE
    label: str = "Selected Area Aperture"
    default_radius_mm: float = 0.10
    maximum_radius_mm: float = float(
        _SELECTED_AREA_PART["maximum_radius_mm"]
    )
    colour: str = "#7cb342"
    standalone_mechanical_center_below_sample_mm: float = (
        _STANDALONE_CENTER_BELOW_SAMPLE_MM
    )
    standalone_mechanical_length_mm: float = float(
        _SELECTED_AREA_PART["length_mm"]
    )
    standalone_mechanical_outer_diameter_mm: float = float(
        _SELECTED_AREA_PART["mechanical_outer_diameter_mm"]
    )
    standalone_mechanical_bore_diameter_mm: float = float(
        _SELECTED_AREA_PART["mechanical_bore_diameter_mm"]
    )
    standalone_plate_thickness_mm: float = float(
        _SELECTED_AREA_PART["plate_thickness_mm"]
    )
    standalone_optical_reference_z_mm: float = (
        _STANDALONE_REFERENCE_Z_MM
    )
    image_corrected_mechanical_center_below_sample_mm: float = (
        _IMAGE_CORRECTED_CENTER_BELOW_SAMPLE_MM
    )
    image_corrected_mechanical_length_mm: float = float(
        _SELECTED_AREA_PART["length_mm"]
    )
    image_corrected_mechanical_outer_diameter_mm: float = float(
        _SELECTED_AREA_PART["mechanical_outer_diameter_mm"]
    )
    image_corrected_mechanical_bore_diameter_mm: float = float(
        _SELECTED_AREA_PART["mechanical_bore_diameter_mm"]
    )
    image_corrected_plate_thickness_mm: float = float(
        _SELECTED_AREA_PART["plate_thickness_mm"]
    )
    image_corrected_optical_reference_z_mm: float = (
        _IMAGE_CORRECTED_REFERENCE_Z_MM
    )
    kind: str = "continuous_aperture"
    shape_profile: str = "adjustable_circular_aperture"
    interaction_kind: str = "hard_edge_circular_stop"

    @property
    def name(self):
        return self.label

    @property
    def owner(self):
        return "image"

    @property
    def radius_mm(self):
        return self.default_radius_mm

    def geometry_for(self, installation):
        if installation not in {
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        }:
            raise ValueError(
                f"Unsupported Selected Area Aperture installation: "
                f"{installation}"
            )
        prefix = installation
        return SelectedAreaApertureGeometry(
            installation=installation,
            mechanical_center_below_sample_mm=getattr(
                self, f"{prefix}_mechanical_center_below_sample_mm"
            ),
            mechanical_length_mm=getattr(
                self, f"{prefix}_mechanical_length_mm"
            ),
            mechanical_outer_diameter_mm=getattr(
                self, f"{prefix}_mechanical_outer_diameter_mm"
            ),
            mechanical_bore_diameter_mm=getattr(
                self, f"{prefix}_mechanical_bore_diameter_mm"
            ),
            plate_thickness_mm=getattr(
                self, f"{prefix}_plate_thickness_mm"
            ),
            optical_reference_z_mm=getattr(
                self, f"{prefix}_optical_reference_z_mm"
            ),
            reference_component_key=(
                IMAGE_CORRECTOR_SAD_PLANE
                if installation == IMAGE_CORRECTED_INSTALLATION
                else IMAGE_DIFFRACTION_DEFLECTOR
            ),
        )

    def create_component(self):
        return SelectedAreaApertureComponent(
            name=self.label,
            key=self.key,
            z_mm=self.standalone_optical_reference_z_mm,
            radius_mm=self.default_radius_mm,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            enabled=False,
            colour=self.colour,
            maximum_radius_mm=self.maximum_radius_mm,
            standalone_mechanical_center_below_sample_mm=(
                self.standalone_mechanical_center_below_sample_mm
            ),
            standalone_mechanical_length_mm=(
                self.standalone_mechanical_length_mm
            ),
            standalone_mechanical_outer_diameter_mm=(
                self.standalone_mechanical_outer_diameter_mm
            ),
            standalone_mechanical_bore_diameter_mm=(
                self.standalone_mechanical_bore_diameter_mm
            ),
            standalone_plate_thickness_mm=(
                self.standalone_plate_thickness_mm
            ),
            standalone_optical_reference_z_mm=(
                self.standalone_optical_reference_z_mm
            ),
            image_corrected_mechanical_center_below_sample_mm=(
                self.image_corrected_mechanical_center_below_sample_mm
            ),
            image_corrected_mechanical_length_mm=(
                self.image_corrected_mechanical_length_mm
            ),
            image_corrected_mechanical_outer_diameter_mm=(
                self.image_corrected_mechanical_outer_diameter_mm
            ),
            image_corrected_mechanical_bore_diameter_mm=(
                self.image_corrected_mechanical_bore_diameter_mm
            ),
            image_corrected_plate_thickness_mm=(
                self.image_corrected_plate_thickness_mm
            ),
            image_corrected_optical_reference_z_mm=(
                self.image_corrected_optical_reference_z_mm
            ),
            active_installation=STANDALONE_INSTALLATION,
        )


@dataclass
class SelectedAreaApertureComponent:
    """One aperture retained across topology changes at two image stations."""

    name: str
    key: str
    z_mm: float
    radius_mm: float
    offset_x_mm: float
    offset_y_mm: float
    enabled: bool
    colour: str
    maximum_radius_mm: float
    standalone_mechanical_center_below_sample_mm: float
    standalone_mechanical_length_mm: float
    standalone_mechanical_outer_diameter_mm: float
    standalone_mechanical_bore_diameter_mm: float
    standalone_plate_thickness_mm: float
    standalone_optical_reference_z_mm: float
    image_corrected_mechanical_center_below_sample_mm: float
    image_corrected_mechanical_length_mm: float
    image_corrected_mechanical_outer_diameter_mm: float
    image_corrected_mechanical_bore_diameter_mm: float
    image_corrected_plate_thickness_mm: float
    image_corrected_optical_reference_z_mm: float
    active_installation: str = STANDALONE_INSTALLATION

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
            delta = value - float(getattr(self, name))
            installation = center_attributes[name]
            reference_attribute = (
                f"{installation}_optical_reference_z_mm"
            )
            reference = float(getattr(self, reference_attribute)) + delta
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
            reference_attr = self._active_attribute(
                "optical_reference_z_mm"
            )
            object.__setattr__(self, reference_attr, value)
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return "image"

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
    def conjugate_to(self):
        return "objective_image_plane"

    def _active_attribute(self, suffix):
        return f"{self.active_installation}_{suffix}"

    @property
    def mechanical_center_below_sample_mm(self):
        return getattr(
            self, self._active_attribute("mechanical_center_below_sample_mm")
        )

    @mechanical_center_below_sample_mm.setter
    def mechanical_center_below_sample_mm(self, value):
        attribute = self._active_attribute(
            "mechanical_center_below_sample_mm"
        )
        setattr(self, attribute, float(value))

    @property
    def mechanical_length_mm(self):
        return getattr(self, self._active_attribute("mechanical_length_mm"))

    @mechanical_length_mm.setter
    def mechanical_length_mm(self, value):
        setattr(
            self, self._active_attribute("mechanical_length_mm"), float(value)
        )

    @property
    def mechanical_outer_diameter_mm(self):
        return getattr(
            self,
            self._active_attribute("mechanical_outer_diameter_mm"),
        )

    @mechanical_outer_diameter_mm.setter
    def mechanical_outer_diameter_mm(self, value):
        setattr(
            self,
            self._active_attribute("mechanical_outer_diameter_mm"),
            float(value),
        )

    @property
    def mechanical_bore_diameter_mm(self):
        return getattr(
            self,
            self._active_attribute("mechanical_bore_diameter_mm"),
        )

    @mechanical_bore_diameter_mm.setter
    def mechanical_bore_diameter_mm(self, value):
        setattr(
            self,
            self._active_attribute("mechanical_bore_diameter_mm"),
            float(value),
        )

    @property
    def plate_thickness_mm(self):
        return getattr(self, self._active_attribute("plate_thickness_mm"))

    @plate_thickness_mm.setter
    def plate_thickness_mm(self, value):
        setattr(
            self, self._active_attribute("plate_thickness_mm"), float(value)
        )

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

    def geometry_for(self, installation):
        if installation not in {
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        }:
            raise ValueError(
                f"Unsupported Selected Area Aperture installation: "
                f"{installation}"
            )
        prefix = installation
        return SelectedAreaApertureGeometry(
            installation=installation,
            mechanical_center_below_sample_mm=getattr(
                self, f"{prefix}_mechanical_center_below_sample_mm"
            ),
            mechanical_length_mm=getattr(
                self, f"{prefix}_mechanical_length_mm"
            ),
            mechanical_outer_diameter_mm=getattr(
                self, f"{prefix}_mechanical_outer_diameter_mm"
            ),
            mechanical_bore_diameter_mm=getattr(
                self, f"{prefix}_mechanical_bore_diameter_mm"
            ),
            plate_thickness_mm=getattr(
                self, f"{prefix}_plate_thickness_mm"
            ),
            optical_reference_z_mm=getattr(
                self, f"{prefix}_optical_reference_z_mm"
            ),
            reference_component_key=(
                IMAGE_CORRECTOR_SAD_PLANE
                if installation == IMAGE_CORRECTED_INSTALLATION
                else IMAGE_DIFFRACTION_DEFLECTOR
            ),
        )

    def select_installation(self, installation):
        geometry = self.geometry_for(installation)
        object.__setattr__(self, "active_installation", installation)
        object.__setattr__(
            self, "z_mm", float(geometry.optical_reference_z_mm)
        )
        return self

    def validate(self):
        if self.key != SELECTED_AREA_APERTURE:
            raise ValueError(
                "Selected Area Aperture must use its canonical key."
            )
        if self.maximum_radius_mm <= 0.0:
            raise ValueError(
                "Selected Area Aperture maximum radius must be positive."
            )
        if not 0.0 <= self.radius_mm <= self.maximum_radius_mm:
            raise ValueError(
                "Selected Area Aperture radius is outside its range."
            )
        for installation in (
            STANDALONE_INSTALLATION,
            IMAGE_CORRECTED_INSTALLATION,
        ):
            geometry = self.geometry_for(installation)
            if geometry.mechanical_center_below_sample_mm <= 0.0:
                raise ValueError(
                    "Selected Area Aperture must remain below the specimen."
                )
            if geometry.mechanical_length_mm <= 0.0:
                raise ValueError(
                    "Selected Area Aperture body length must be positive."
                )
            if (
                geometry.mechanical_outer_diameter_mm
                <= geometry.mechanical_bore_diameter_mm
            ):
                raise ValueError(
                    "Selected Area Aperture body must exceed its bore."
                )
            if geometry.plate_thickness_mm <= 0.0:
                raise ValueError(
                    "Selected Area Aperture plate must have positive thickness."
                )
            if (
                2.0 * self.maximum_radius_mm
                > geometry.mechanical_bore_diameter_mm
            ):
                raise ValueError(
                    "Selected Area Aperture opening must fit inside its bore."
                )
        return self

    def transmission_mask(self, x_mm, y_mm):
        radial_distance_mm = np.hypot(
            np.asarray(x_mm, dtype=float) - self.offset_x_mm,
            np.asarray(y_mm, dtype=float) - self.offset_y_mm,
        )
        return radial_distance_mm <= self.radius_mm

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
            "mechanical_bore_diameter_mm": (
                geometry.mechanical_bore_diameter_mm
            ),
            "plate_thickness_mm": geometry.plate_thickness_mm,
            "reference_component_key": geometry.reference_component_key,
            "shape_profile": self.shape_profile,
        }


SELECTED_AREA_APERTURE_DEFINITION = SelectedAreaApertureDefinition()


def create_selected_area_aperture():
    return SELECTED_AREA_APERTURE_DEFINITION.create_component().validate()


def selected_area_aperture_from_dict(data):
    """Restore the canonical component or upgrade a generic aperture row."""

    values = dict(data)
    component = create_selected_area_aperture()
    legacy_z = values.get("z_mm")
    new_fields = SelectedAreaApertureComponent.__dataclass_fields__
    for attribute, value in values.items():
        if attribute in new_fields:
            object.__setattr__(component, attribute, value)
    component.key = SELECTED_AREA_APERTURE
    component.name = "Selected Area Aperture"
    if (
        "standalone_optical_reference_z_mm" not in values
        and legacy_z is not None
    ):
        object.__setattr__(
            component,
            "standalone_optical_reference_z_mm",
            float(legacy_z),
        )
    component.select_installation(
        values.get("active_installation", STANDALONE_INSTALLATION)
    )
    return component.validate()


def resolve_standalone_selected_area_aperture_anchor(state):
    """Anchor the standalone aperture after the Image/Diff deflector."""

    aperture = state.selected_area_aperture
    image_deflector = state.image_diffraction_deflector
    image_deflector_lower_surface_mm = (
        float(image_deflector.optical_center_z_mm)
        + float(image_deflector.mechanical_length_mm) / 2.0
    )
    state.standalone_selected_area_aperture_gap_after_descan_mm = (
        DEFAULT_STANDALONE_GAP_AFTER_IMAGE_DEFLECTOR_MM
    )
    center_mm = (
        image_deflector_lower_surface_mm
        + float(
            state.standalone_selected_area_aperture_gap_after_descan_mm
        )
        + float(aperture.standalone_mechanical_length_mm) / 2.0
    )
    aperture.standalone_mechanical_center_below_sample_mm = (
        center_mm - float(state.sample.z_mm)
    )
    aperture.standalone_optical_reference_z_mm = center_mm
    state._standalone_selected_area_aperture_resolved_positions_mm = {
        "image_deflector_lower_surface": (
            image_deflector_lower_surface_mm
        ),
        "selected_area_aperture": center_mm,
    }
    return center_mm
