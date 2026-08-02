"""Independent post-sample Descan Deflector with raster compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim import module_manifest
from temsim.component_keys import (
    DESCAN_DEFLECTOR,
    canonical_corrector_element_key,
)

_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml", "exit"
)
_DEFAULT_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, DESCAN_DEFLECTOR
)
_DEFAULT_SAMPLE = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, "sample"
)
_DEFAULT_INTERACTIONS = tuple(
    float(value)
    for value in _DEFAULT_PART["interaction_centers_local_z_mm"]
)


@dataclass(frozen=True)
class DescanDeflectorDefinition:
    key: str = DESCAN_DEFLECTOR
    label: str = "Descan Deflector"
    mechanical_center_below_sample_mm: float = (
        float(_DEFAULT_PART["local_center_z_mm"])
        - float(_DEFAULT_SAMPLE["local_center_z_mm"])
    )
    mechanical_length_mm: float = float(_DEFAULT_PART["length_mm"])
    mechanical_coil_length_mm: float = float(
        _DEFAULT_PART["mechanical_coil_length_mm"]
    )
    mechanical_inter_coil_gap_mm: float = float(
        _DEFAULT_PART["mechanical_inter_coil_gap_mm"]
    )
    mechanical_outer_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_outer_diameter_mm"]
    )
    mechanical_clear_bore_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_clear_bore_diameter_mm"]
    )
    optical_reference_z_mm: float = (
        _DEFAULT_COLUMN_ORIGIN_Z_MM
        + float(_DEFAULT_PART["optical_reference_local_z_mm"])
    )
    optical_plane_separation_mm: float = (
        _DEFAULT_INTERACTIONS[1] - _DEFAULT_INTERACTIONS[0]
    )
    effective_thickness_mm: float = float(
        _DEFAULT_PART["effective_thickness_mm"]
    )
    maximum_kick_mrad: float = 100.0
    colour: str = "#00838f"
    owner: str = "shared_column"
    kind: str = "paired_deflector"
    shape_profile: str = "paired_deflector_coils"
    interaction_kind: str = "paired_transverse_kick"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return DescanDeflectorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_z_mm,
            kick_x_mrad=0.0,
            kick_y_mrad=0.0,
            effective_thickness_mm=self.effective_thickness_mm,
            mechanical_coil_length_mm=self.mechanical_coil_length_mm,
            mechanical_inter_coil_gap_mm=(
                self.mechanical_inter_coil_gap_mm
            ),
            optical_plane_separation_mm=(
                self.optical_plane_separation_mm
            ),
            enabled=True,
            colour=self.colour,
            mechanical_center_below_sample_mm=(
                self.mechanical_center_below_sample_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_z_mm=self.optical_reference_z_mm,
            maximum_kick_mrad=self.maximum_kick_mrad,
            corrector=self.owner,
            scan_enabled=False,
            scan_amplitude_x_mrad=-0.1,
            scan_amplitude_y_mrad=-0.1,
            scan_frame_period_s=1.0,
            scan_lines=32,
        )


@dataclass
class DescanDeflectorComponent:
    name: str
    key: str
    z_mm: float
    kick_x_mrad: float
    kick_y_mrad: float
    effective_thickness_mm: float
    mechanical_coil_length_mm: float
    mechanical_inter_coil_gap_mm: float
    optical_plane_separation_mm: float
    enabled: bool
    colour: str
    mechanical_center_below_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_z_mm: float
    maximum_kick_mrad: float
    corrector: str = "shared_column"
    scan_enabled: bool = False
    scan_amplitude_x_mrad: float = -0.1
    scan_amplitude_y_mrad: float = -0.1
    scan_frame_period_s: float = 1.0
    scan_lines: int = 32

    EXPECTED_KEY: ClassVar[str] = DESCAN_DEFLECTOR
    KIND: ClassVar[str] = "paired_deflector"
    SHAPE_PROFILE: ClassVar[str] = "paired_deflector_coils"
    INTERACTION_KIND: ClassVar[str] = "paired_transverse_kick"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_below_sample_mm",
            "optical_reference_z_mm",
        }:
            value = float(value)
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "mechanical_center_below_sample_mm":
            delta_mm = (
                value - float(self.mechanical_center_below_sample_mm)
            )
            object.__setattr__(self, name, value)
            optical = float(self.optical_reference_z_mm) + delta_mm
            object.__setattr__(self, "optical_reference_z_mm", optical)
            object.__setattr__(self, "z_mm", optical)
            return
        if ready and name == "optical_reference_z_mm":
            object.__setattr__(self, name, value)
            object.__setattr__(self, "z_mm", value)
            return
        object.__setattr__(self, name, value)

    @property
    def kind(self):
        return self.KIND

    @property
    def owner(self):
        return self.corrector

    @property
    def shape_profile(self):
        return self.SHAPE_PROFILE

    @property
    def interaction_kind(self):
        return self.INTERACTION_KIND

    @property
    def length_mm(self):
        return self.mechanical_length_mm

    @property
    def optical_active(self):
        return bool(self.enabled)

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError("Descan Deflector key is not canonical.")
        if self.mechanical_center_below_sample_mm <= 0.0:
            raise ValueError(
                "Descan Deflector must remain below the sample."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Descan Deflector mechanical length must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(
                "Descan Deflector bore must fit inside its body."
            )
        if not (
            0.0
            < self.effective_thickness_mm
            <= self.mechanical_coil_length_mm
        ):
            raise ValueError(
                "Descan effective thickness must fit one coil."
            )
        if (
            self.mechanical_coil_length_mm <= 0.0
            or self.mechanical_inter_coil_gap_mm < 0.0
            or self.optical_plane_separation_mm <= 0.0
        ):
            raise ValueError("Descan double-coil geometry must be positive.")
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError(
                "Descan maximum kick must be positive."
            )
        if max(
            abs(float(self.kick_x_mrad)),
            abs(float(self.kick_y_mrad)),
        ) > self.maximum_kick_mrad:
            raise ValueError(
                "Descan kick exceeds its configured limit."
            )
        if self.scan_frame_period_s <= 0.0:
            raise ValueError(
                "Descan scan frame period must be positive."
            )
        if int(self.scan_lines) != self.scan_lines or self.scan_lines < 2:
            raise ValueError(
                "Descan scan must contain at least two lines."
            )
        if max(
            abs(float(self.scan_amplitude_x_mrad)),
            abs(float(self.scan_amplitude_y_mrad)),
        ) > self.maximum_kick_mrad:
            raise ValueError(
                "Descan scan amplitude exceeds its configured limit."
            )
        return self

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_z_mm)
        return self

    def scan_factors(self, time_s):
        if not self.scan_enabled:
            return 0.0, 0.0
        frame_phase = (
            float(time_s) / float(self.scan_frame_period_s)
        ) % 1.0
        line_position = frame_phase * int(self.scan_lines)
        line_index = min(int(line_position), int(self.scan_lines) - 1)
        within_line = line_position - line_index
        x_factor = 2.0 * within_line - 1.0
        y_factor = (
            2.0 * line_index / (int(self.scan_lines) - 1) - 1.0
        )
        return x_factor, y_factor

    def scan_kick_mrad(self, time_s):
        x_factor, y_factor = self.scan_factors(time_s)
        return (
            self.scan_amplitude_x_mrad * x_factor,
            self.scan_amplitude_y_mrad * y_factor,
        )

    def dynamic_kick_mrad(self, time_s=0.0):
        return self.scan_kick_mrad(time_s)

    def instantaneous_kick_mrad(self, time_s=0.0):
        scan_x_mrad, scan_y_mrad = self.dynamic_kick_mrad(time_s)
        return (
            self.kick_x_mrad + scan_x_mrad,
            self.kick_y_mrad + scan_y_mrad,
        )

    def kick_events(self, time_s=0.0):
        if not self.enabled:
            return ()
        kick_x_mrad, kick_y_mrad = self.instantaneous_kick_mrad(
            time_s
        )
        half_separation_mm = self.optical_plane_separation_mm / 2.0
        half_x_rad = kick_x_mrad * 0.5e-3
        half_y_rad = kick_y_mrad * 0.5e-3
        return (
            (
                self.z_mm - half_separation_mm,
                half_x_rad,
                half_y_rad,
            ),
            (
                self.z_mm + half_separation_mm,
                half_x_rad,
                half_y_rad,
            ),
        )

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
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "mechanical_coil_length_mm": self.mechanical_coil_length_mm,
            "mechanical_inter_coil_gap_mm": (
                self.mechanical_inter_coil_gap_mm
            ),
            "optical_plane_separation_mm": (
                self.optical_plane_separation_mm
            ),
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "upper_plane_z_mm": (
                self.z_mm - self.optical_plane_separation_mm / 2.0
            ),
            "lower_plane_z_mm": (
                self.z_mm + self.optical_plane_separation_mm / 2.0
            ),
            "effective_thickness_mm": self.effective_thickness_mm,
            "kick_x_mrad": self.kick_x_mrad,
            "kick_y_mrad": self.kick_y_mrad,
            "enabled": self.enabled,
            "scan_enabled": self.scan_enabled,
            "scan_amplitude_x_mrad": self.scan_amplitude_x_mrad,
            "scan_amplitude_y_mrad": self.scan_amplitude_y_mrad,
            "scan_frame_period_s": self.scan_frame_period_s,
            "scan_lines": self.scan_lines,
        }


DESCAN_DEFLECTOR_DEFINITION = DescanDeflectorDefinition()


def create_descan_deflector():
    return DESCAN_DEFLECTOR_DEFINITION.create_component().validate()


def descan_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_descan_deflector()
    for attribute in (
        "kick_x_mrad",
        "kick_y_mrad",
        "enabled",
        "colour",
        "maximum_kick_mrad",
        "scan_enabled",
        "scan_amplitude_x_mrad",
        "scan_amplitude_y_mrad",
        "scan_frame_period_s",
        "scan_lines",
    ):
        if attribute in values:
            object.__setattr__(component, attribute, values[attribute])
    component.key = DESCAN_DEFLECTOR
    component.name = "Descan Deflector"
    component.corrector = DESCAN_DEFLECTOR_DEFINITION.owner
    return component.apply_optical_position().validate()
