"""Canonical mechanical stations downstream of the Selected Area Aperture.

Every value is a positive centre-to-centre distance in the beam direction
from the mechanical centre of the Selected Area Aperture.  The same offsets
also place the components on the ray-tracing axis, so no independent optical
position can drift away from its mechanical station.
"""

from temsim.component_keys import (
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    DARK_FIELD_DETECTOR,
    DIFFRACTION_LENS,
    DIFFRACTION_STIGMATOR,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    FLUORESCENT_SCREEN,
    HAADF_DETECTOR,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)


SELECTED_AREA_DOWNSTREAM_OFFSETS_MM = {
    DIFFRACTION_STIGMATOR: 15.0,
    DIFFRACTION_LENS: 72.5,
    INTERMEDIATE_LENS: 242.5,
    PROJECTOR_LENS_1: 422.5,
    PROJECTOR_LENS_2: 625.0,
    HAADF_DETECTOR: 770.0,
    FLUORESCENT_SCREEN: 890.0,
    DARK_FIELD_DETECTOR: 980.0,
    BRIGHT_FIELD_DETECTOR: 1050.0,
    CAMERA: 1140.0,
    ENERGY_FILTER_ENTRANCE_APERTURE: 1201.5,
}

SELECTED_AREA_DOWNSTREAM_KEYS = tuple(
    SELECTED_AREA_DOWNSTREAM_OFFSETS_MM
)


def downstream_offset_mm(component_key):
    """Return the canonical downstream centre offset for one component."""

    return float(SELECTED_AREA_DOWNSTREAM_OFFSETS_MM[component_key])


def downstream_mechanical_center_mm(selected_area_geometry, component_key):
    """Resolve a below-sample mechanical centre from the active SAA station."""

    return (
        float(selected_area_geometry.mechanical_center_below_sample_mm)
        + downstream_offset_mm(component_key)
    )


def downstream_optical_reference_mm(selected_area_geometry, component_key):
    """Resolve the ray-axis position from the same mechanical relationship."""

    return (
        float(selected_area_geometry.optical_reference_z_mm)
        + downstream_offset_mm(component_key)
    )
