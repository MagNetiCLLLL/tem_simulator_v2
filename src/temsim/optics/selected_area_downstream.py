"""TOML-backed bootstrap stations downstream of the Selected Area Aperture.

Every value is a positive centre-to-centre distance in the beam direction
from the mechanical centre of the Selected Area Aperture.  The same offsets
also place the components on the ray-tracing axis, so no independent optical
position can drift away from its mechanical station.
"""

from temsim import module_manifest
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


_DEFAULT_RECORDING_MODULE = (
    "project_and_recording_system/NoEnergyFilter.toml"
)
_ENERGY_FILTER_RECORDING_MODULE = (
    "project_and_recording_system/EnergyFilter.toml"
)


def _manifest_center(module_path, key):
    return float(
        module_manifest.part_data(module_path, key)["local_center_z_mm"]
    )


_DEFAULT_SELECTED_AREA_CENTER_MM = _manifest_center(
    _DEFAULT_RECORDING_MODULE, "selected_area_aperture"
)


def _manifest_downstream_offset(module_path, key):
    return (
        _manifest_center(module_path, key)
        - _DEFAULT_SELECTED_AREA_CENTER_MM
    )


# These values are bootstrap geometry for component construction. The selected
# ResolvedAssembly reapplies the corresponding active TOML part before use.
SELECTED_AREA_DOWNSTREAM_OFFSETS_MM = {
    key: _manifest_downstream_offset(_DEFAULT_RECORDING_MODULE, key)
    for key in (
        DIFFRACTION_STIGMATOR,
        DIFFRACTION_LENS,
        INTERMEDIATE_LENS,
        PROJECTOR_LENS_1,
        PROJECTOR_LENS_2,
        HAADF_DETECTOR,
        FLUORESCENT_SCREEN,
        DARK_FIELD_DETECTOR,
        BRIGHT_FIELD_DETECTOR,
        CAMERA,
    )
}
SELECTED_AREA_DOWNSTREAM_OFFSETS_MM[
    ENERGY_FILTER_ENTRANCE_APERTURE
] = _manifest_downstream_offset(
    _ENERGY_FILTER_RECORDING_MODULE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
)

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
