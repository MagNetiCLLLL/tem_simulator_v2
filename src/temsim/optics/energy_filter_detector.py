"""Iliad detector-end electrostatics and Zebra EELS detector model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    ENERGY_FILTER_BIAS_TUBE,
    ENERGY_FILTER_CAMERA_DEFLECTOR,
    ENERGY_FILTER_SHUTTER,
    ENERGY_FILTER_ZEBRA,
)


_ENERGY_FILTER_MODULE_PATH = (
    "project_and_recording_system/EnergyFilter.toml"
)


def _part(key):
    return module_manifest.part_data(_ENERGY_FILTER_MODULE_PATH, key)


_BIAS_PART = _part(ENERGY_FILTER_BIAS_TUBE)
_SHUTTER_PART = _part(ENERGY_FILTER_SHUTTER)
_CAMERA_DEFLECTOR_PART = _part(ENERGY_FILTER_CAMERA_DEFLECTOR)
_ZEBRA_PART = _part(ENERGY_FILTER_ZEBRA)

_TOML_OWNED_FIELDS = frozenset({
    "housing_length_mm",
    "clear_bore_diameter_mm",
    "mechanical_outer_diameter_mm",
    "maximum_abs_offset_ev",
    "offset_range_status",
    "electrode_length_mm",
    "electrode_gap_mm",
    "maximum_strip_count",
    "strip_count",
    "pixels_per_strip",
    "strip_height_pixels",
    "spectral_clear_height_mm",
    "alignment_pixels_x",
    "alignment_pixels_y",
    "pixel_size_um",
    "maximum_spectra_per_s",
    "provisional_strip_center_pitch_mm",
    "strip_center_pitch_status",
    "external_envelope_status",
})


@dataclass
class EnergyFilterBiasTube:
    name: str = "Iliad MultiEELS Bias Tube"
    key: str = ENERGY_FILTER_BIAS_TUBE
    enabled: bool = True
    offset_ev: float = 0.0
    maximum_abs_offset_ev: float = float(
        _BIAS_PART["maximum_abs_offset_ev"]
    )
    offset_range_status: str = str(_BIAS_PART["offset_range_status"])
    housing_length_mm: float = float(_BIAS_PART["housing_length_mm"])
    clear_bore_diameter_mm: float = float(
        _BIAS_PART["clear_bore_diameter_mm"]
    )
    mechanical_outer_diameter_mm: float = float(
        _BIAS_PART["mechanical_outer_diameter_mm"]
    )

    KIND = "electrostatic_bias_tube"
    INTERACTION_KIND = "kinetic_energy_offset"

    def validate(self):
        if self.key != ENERGY_FILTER_BIAS_TUBE:
            raise ValueError("Energy Filter bias tube key is not canonical.")
        if not all(math.isfinite(float(value)) for value in (
            self.offset_ev, self.maximum_abs_offset_ev
        )):
            raise ValueError("Bias-tube values must be finite.")
        if self.maximum_abs_offset_ev <= 0.0:
            raise ValueError("Bias-tube range must be positive.")
        if self.offset_range_status != (
            "provisional_simulator_limit_not_iliad_product_specification"
        ):
            raise ValueError("Bias-tube range must remain marked provisional.")
        if abs(self.offset_ev) > self.maximum_abs_offset_ev:
            raise ValueError("Bias-tube offset exceeds its calibrated range.")
        if min(
            self.housing_length_mm,
            self.clear_bore_diameter_mm,
            self.mechanical_outer_diameter_mm,
        ) <= 0.0:
            raise ValueError("Bias-tube mechanical dimensions must be positive.")
        return self


@dataclass
class EnergyFilterShutter:
    name: str = "Iliad Fast Electrostatic Shutter"
    key: str = ENERGY_FILTER_SHUTTER
    enabled: bool = True
    open: bool = True
    electrode_length_mm: float = float(_SHUTTER_PART["electrode_length_mm"])
    electrode_gap_mm: float = float(_SHUTTER_PART["electrode_gap_mm"])
    mechanical_outer_diameter_mm: float = float(
        _SHUTTER_PART["mechanical_outer_diameter_mm"]
    )

    KIND = "electrostatic_shutter"
    INTERACTION_KIND = "fast_beam_gate"

    def validate(self):
        if self.key != ENERGY_FILTER_SHUTTER:
            raise ValueError("Energy Filter shutter key is not canonical.")
        if min(
            self.electrode_length_mm,
            self.electrode_gap_mm,
            self.mechanical_outer_diameter_mm,
        ) <= 0.0:
            raise ValueError("Shutter mechanical dimensions must be positive.")
        return self


@dataclass
class EnergyFilterCameraDeflector:
    name: str = "Iliad Zebra Camera Deflector"
    key: str = ENERGY_FILTER_CAMERA_DEFLECTOR
    enabled: bool = True
    active_strip: int = 1
    maximum_strip_count: int = 5
    electrode_length_mm: float = float(
        _CAMERA_DEFLECTOR_PART["electrode_length_mm"]
    )
    electrode_gap_mm: float = float(
        _CAMERA_DEFLECTOR_PART["electrode_gap_mm"]
    )
    mechanical_outer_diameter_mm: float = float(
        _CAMERA_DEFLECTOR_PART["mechanical_outer_diameter_mm"]
    )

    KIND = "electrostatic_camera_deflector"
    INTERACTION_KIND = "detector_strip_selector"

    def validate(self):
        if self.key != ENERGY_FILTER_CAMERA_DEFLECTOR:
            raise ValueError(
                "Energy Filter camera-deflector key is not canonical."
            )
        if not 1 <= int(self.active_strip) <= int(self.maximum_strip_count):
            raise ValueError("Active Zebra strip is outside the detector.")
        if int(self.maximum_strip_count) != 5:
            raise ValueError("Iliad camera deflector requires five strips.")
        if min(
            self.electrode_length_mm,
            self.electrode_gap_mm,
            self.mechanical_outer_diameter_mm,
        ) <= 0.0:
            raise ValueError(
                "Camera-deflector mechanical dimensions must be positive."
            )
        return self


@dataclass
class ZebraEELSDetector:
    """Five one-dimensional spectrum strips plus one 2-D alignment area."""

    name: str = "Iliad Zebra Five-strip EELS Detector"
    key: str = ENERGY_FILTER_ZEBRA
    enabled: bool = True
    inserted: bool = True
    alignment_mode: bool = False
    strip_count: int = int(_ZEBRA_PART["strip_count"])
    pixels_per_strip: int = int(_ZEBRA_PART["pixels_per_strip"])
    strip_height_pixels: int = 1
    # Public active height of each independent 1-D sensor.  This must not be
    # replaced by the much taller 2-D alignment region.
    spectral_clear_height_mm: float = float(
        _ZEBRA_PART["strip_active_height_mm"]
    )
    alignment_pixels_x: int = int(
        _ZEBRA_PART["alignment_pixels_non_dispersive"]
    )
    alignment_pixels_y: int = int(
        _ZEBRA_PART["alignment_pixels_dispersive"]
    )
    pixel_size_um: float = float(_ZEBRA_PART["strip_pixel_pitch_um"])
    maximum_spectra_per_s: float = float(
        _ZEBRA_PART["maximum_spectra_per_s"]
    )
    provisional_strip_center_pitch_mm: float = float(
        _ZEBRA_PART["provisional_strip_center_pitch_mm"]
    )
    strip_center_pitch_status: str = str(
        _ZEBRA_PART["strip_center_pitch_status"]
    )
    external_envelope_status: str = str(
        _ZEBRA_PART["external_envelope_status"]
    )

    KIND = "zebra_eels_detector"
    INTERACTION_KIND = "terminal_spectrum_detector"

    def validate(self):
        if self.key != ENERGY_FILTER_ZEBRA:
            raise ValueError("Zebra detector key is not canonical.")
        integers = (
            self.strip_count,
            self.pixels_per_strip,
            self.strip_height_pixels,
            self.alignment_pixels_x,
            self.alignment_pixels_y,
        )
        if any(int(value) <= 0 for value in integers):
            raise ValueError("Zebra detector dimensions must be positive.")
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (
            self.pixel_size_um,
            self.maximum_spectra_per_s,
            self.spectral_clear_height_mm,
            self.provisional_strip_center_pitch_mm,
        )):
            raise ValueError("Zebra detector calibration must be positive.")
        if self.strip_count != 5 or self.pixels_per_strip != 2_048:
            raise ValueError("Zebra detector must contain five 2048-pixel strips.")
        if not math.isclose(
            float(self.spectral_clear_height_mm), 0.800, abs_tol=1.0e-12
        ):
            raise ValueError("Zebra strip active height must be 0.800 mm.")
        if self.provisional_strip_center_pitch_mm < self.spectral_clear_height_mm:
            raise ValueError("Zebra provisional strip pitch causes overlap.")
        if self.strip_center_pitch_status != "adjustable_unknown_not_public":
            raise ValueError("Zebra strip pitch must remain marked unknown.")
        return self

    @property
    def spectral_width_mm(self):
        return float(self.pixels_per_strip) * float(self.pixel_size_um) * 1.0e-3

    @property
    def alignment_height_mm(self):
        return float(self.alignment_pixels_x) * float(self.pixel_size_um) * 1.0e-3

    @property
    def spectral_height_mm(self):
        return float(self.spectral_clear_height_mm)

    @property
    def alignment_width_mm(self):
        return float(self.alignment_pixels_y) * float(self.pixel_size_um) * 1.0e-3

    def recording_mask(self, dispersive_m, non_dispersive_m):
        dispersive, non_dispersive = np.broadcast_arrays(
            np.asarray(dispersive_m, dtype=float),
            np.asarray(non_dispersive_m, dtype=float),
        )
        if not self.enabled or not self.inserted:
            return np.zeros(dispersive.shape, dtype=bool)
        half_dispersive = 0.5 * (
            self.alignment_width_mm
            if self.alignment_mode
            else self.spectral_width_mm
        ) * 1.0e-3
        half_non_dispersive = 0.5 * (
            self.alignment_height_mm
            if self.alignment_mode
            else self.spectral_height_mm
        ) * 1.0e-3
        return (
            (np.abs(dispersive) <= half_dispersive)
            & (np.abs(non_dispersive) <= half_non_dispersive)
        )


def serialise_detector_component(component):
    return {
        key: value for key, value in asdict(component).items()
        if key not in _TOML_OWNED_FIELDS
    }


def detector_component_from_dict(component_type, values):
    defaults = component_type()
    if not isinstance(values, dict):
        return defaults.validate()
    allowed = defaults.__dataclass_fields__.keys() - _TOML_OWNED_FIELDS
    component = component_type(**{
        key: value for key, value in values.items() if key in allowed
    })
    return component.validate()
