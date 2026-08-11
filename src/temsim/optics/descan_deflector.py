"""Post-sample AC Descan Coil with image-plane raster compensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
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
    label: str = "AC Descan Coil"
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
    interaction_kind: str = "time_dependent_paired_transverse_kick"

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
            scan_pixels_x=32,
            scan_lines=32,
            scan_pixel_size_nm=1.0,
            upper_coil_gain=0.5,
            lower_coil_gain=0.5,
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
    scan_pixels_x: int = 32
    scan_lines: int = 32
    # This is the same specimen-raster pitch shown for the AC Scan Foils.
    # Descan uses the opposite calibrated AC command; it does not define a
    # second, independent specimen field of view.
    scan_pixel_size_nm: float = 1.0
    upper_coil_gain: float = 0.5
    lower_coil_gain: float = 0.5

    EXPECTED_KEY: ClassVar[str] = DESCAN_DEFLECTOR
    KIND: ClassVar[str] = "paired_deflector"
    SHAPE_PROFILE: ClassVar[str] = "paired_deflector_coils"
    INTERACTION_KIND: ClassVar[str] = (
        "time_dependent_paired_transverse_kick"
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_image_plane_lower_ratio_matrix",
            ((1.0, 0.0), (0.0, 1.0)),
        )
        object.__setattr__(self, "_image_plane_residual", 0.0)
        object.__setattr__(self, "_image_plane_target_z_mm", None)
        object.__setattr__(self, "_image_plane_target_key", "")
        object.__setattr__(self, "_image_plane_calibrated", False)
        object.__setattr__(
            self,
            "_scan_command_matrix_mrad",
            (
                (float(self.scan_amplitude_x_mrad), 0.0),
                (0.0, float(self.scan_amplitude_y_mrad)),
            ),
        )
        object.__setattr__(self, "_scan_scale_residual", 0.0)
        object.__setattr__(self, "_scan_scale_calibrated", False)
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_below_sample_mm",
            "optical_reference_z_mm",
        }:
            value = float(value)
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "upper_coil_gain":
            object.__setattr__(self, name, float(value))
            self._sync_image_plane_gain()
            return
        if ready and name == "lower_coil_gain":
            ratio = self._image_plane_lower_ratio_matrix
            representative = 0.5 * (
                float(ratio[0][0]) + float(ratio[1][1])
            )
            if abs(representative) <= 1.0e-12:
                raise ValueError(
                    "Descan lower foil gain is coupled to the upper foil gain."
                )
            object.__setattr__(
                self,
                "upper_coil_gain",
                float(value) / representative,
            )
            self._sync_image_plane_gain()
            return
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

    @property
    def upper_z_mm(self):
        return self.z_mm - self.optical_plane_separation_mm / 2.0

    @property
    def lower_z_mm(self):
        return self.z_mm + self.optical_plane_separation_mm / 2.0

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
        if not (
            isfinite(float(self.scan_pixel_size_nm))
            and 1.0e-3 <= float(self.scan_pixel_size_nm) <= 1.0e6
        ):
            raise ValueError(
                "Descan scan pixel size must be between 0.001 nm and 1 mm."
            )
        if int(self.scan_pixels_x) != self.scan_pixels_x or (
            self.scan_pixels_x < 2
        ):
            raise ValueError(
                "Descan scan must contain at least two X pixels."
            )
        if int(self.scan_lines) != self.scan_lines or self.scan_lines < 2:
            raise ValueError(
                "Descan scan must contain at least two lines."
            )
        coil_gains = (
            float(self.upper_coil_gain),
            float(self.lower_coil_gain),
        )
        if not all(isfinite(value) for value in coil_gains):
            raise ValueError("Descan upper/lower coil gains must be finite.")
        driven_values = (
            self.scan_amplitude_x_mrad,
            self.scan_amplitude_y_mrad,
            *(
                value
                for row in self.scan_command_matrix_mrad
                for value in row
            ),
        )
        if max(abs(float(value)) for value in driven_values) > (
            self.maximum_kick_mrad
        ):
            raise ValueError(
                "Descan scan amplitude exceeds its configured limit."
            )
        active_x_mrad = abs(float(self.kick_x_mrad))
        active_y_mrad = abs(float(self.kick_y_mrad))
        if self.scan_enabled:
            scan_matrix = self.scan_command_matrix_mrad
            active_x_mrad += sum(abs(float(value)) for value in scan_matrix[0])
            active_y_mrad += sum(abs(float(value)) for value in scan_matrix[1])
        maximum_foil_drives = tuple(
            abs(float(matrix[row][0])) * active_x_mrad
            + abs(float(matrix[row][1])) * active_y_mrad
            for matrix in self.coil_kick_matrices()
            for row in range(2)
        )
        if max(maximum_foil_drives, default=0.0) > float(
            self.maximum_kick_mrad
        ):
            raise ValueError(
                "Descan upper/lower coil drive exceeds its individual limit."
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
            2.0 * (line_index + 0.5) / int(self.scan_lines) - 1.0
        )
        return x_factor, y_factor

    def scan_kick_mrad(self, time_s):
        x_factor, y_factor = self.scan_factors(time_s)
        matrix = self.scan_command_matrix_mrad
        return (
            matrix[0][0] * x_factor + matrix[0][1] * y_factor,
            matrix[1][0] * x_factor + matrix[1][1] * y_factor,
        )

    def dynamic_kick_mrad(self, time_s=0.0):
        return self.scan_kick_mrad(time_s)

    def instantaneous_kick_mrad(self, time_s=0.0):
        scan_x_mrad, scan_y_mrad = self.dynamic_kick_mrad(time_s)
        return (
            self.kick_x_mrad + scan_x_mrad,
            self.kick_y_mrad + scan_y_mrad,
        )

    def _sync_image_plane_gain(self):
        ratio = self._image_plane_lower_ratio_matrix
        representative = 0.5 * (
            float(ratio[0][0]) + float(ratio[1][1])
        )
        object.__setattr__(
            self,
            "lower_coil_gain",
            float(self.upper_coil_gain) * representative,
        )

    def set_image_plane_coupling(
        self,
        lower_from_upper,
        residual=0.0,
        *,
        target_z_mm=None,
        target_key="",
    ):
        """Install the lower-foil map that matches AC at one image station."""

        rows = tuple(
            tuple(float(value) for value in row)
            for row in lower_from_upper
        )
        if len(rows) != 2 or any(len(row) != 2 for row in rows):
            raise ValueError("Descan image-plane coupling must be a 2x2 matrix.")
        if not all(isfinite(value) for row in rows for value in row):
            raise ValueError("Descan image-plane coupling must be finite.")
        residual = float(residual)
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("Descan image-plane residual must be finite.")
        if target_z_mm is not None and not isfinite(float(target_z_mm)):
            raise ValueError("Descan image-plane target Z must be finite.")
        object.__setattr__(self, "_image_plane_lower_ratio_matrix", rows)
        object.__setattr__(self, "_image_plane_residual", residual)
        object.__setattr__(
            self,
            "_image_plane_target_z_mm",
            None if target_z_mm is None else float(target_z_mm),
        )
        object.__setattr__(self, "_image_plane_target_key", str(target_key))
        object.__setattr__(self, "_image_plane_calibrated", True)
        self._sync_image_plane_gain()
        return self

    def set_scan_command_matrix_mrad(self, command_matrix, residual=0.0):
        """Install the command opposed to the calibrated AC raster."""

        rows = tuple(
            tuple(float(value) for value in row)
            for row in command_matrix
        )
        if len(rows) != 2 or any(len(row) != 2 for row in rows):
            raise ValueError("Descan scan command calibration must be 2x2.")
        if not all(isfinite(value) for row in rows for value in row):
            raise ValueError("Descan scan command calibration must be finite.")
        residual = float(residual)
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("Descan scan scale residual must be finite.")
        object.__setattr__(self, "_scan_command_matrix_mrad", rows)
        object.__setattr__(self, "_scan_scale_residual", residual)
        object.__setattr__(self, "_scan_scale_calibrated", True)
        return self

    @property
    def scan_command_matrix_mrad(self):
        return tuple(
            tuple(float(value) for value in row)
            for row in self._scan_command_matrix_mrad
        )

    @property
    def scan_scale_residual(self):
        return float(self._scan_scale_residual)

    @property
    def scan_field_of_view_x_nm(self):
        return float(self.scan_pixels_x) * float(self.scan_pixel_size_nm)

    @property
    def scan_field_of_view_y_nm(self):
        return float(self.scan_lines) * float(self.scan_pixel_size_nm)

    @property
    def image_plane_lower_ratio_matrix(self):
        return tuple(
            tuple(float(value) for value in row)
            for row in self._image_plane_lower_ratio_matrix
        )

    @property
    def image_plane_residual(self):
        return float(self._image_plane_residual)

    @property
    def image_plane_target_z_mm(self):
        value = self._image_plane_target_z_mm
        return None if value is None else float(value)

    @property
    def image_plane_target_key(self):
        return str(self._image_plane_target_key)

    def coil_kick_matrices(self):
        """Return upper/lower maps from one shared descan command."""

        gain = float(self.upper_coil_gain)
        ratio = self._image_plane_lower_ratio_matrix
        return (
            ((gain, 0.0), (0.0, gain)),
            tuple(
                tuple(gain * float(value) for value in row)
                for row in ratio
            ),
        )

    def coil_kicks_mrad(self, kick_x_mrad, kick_y_mrad):
        command_x = float(kick_x_mrad)
        command_y = float(kick_y_mrad)
        upper, lower = self.coil_kick_matrices()

        def apply(matrix):
            return (
                matrix[0][0] * command_x + matrix[0][1] * command_y,
                matrix[1][0] * command_x + matrix[1][1] * command_y,
            )

        return apply(upper), apply(lower)

    def kick_events(self, time_s=0.0):
        if not self.enabled:
            return ()
        kick_x_mrad, kick_y_mrad = self.instantaneous_kick_mrad(
            time_s
        )
        upper_kick, lower_kick = self.coil_kicks_mrad(
            kick_x_mrad,
            kick_y_mrad,
        )
        return (
            (
                self.upper_z_mm,
                upper_kick[0] * 1.0e-3,
                upper_kick[1] * 1.0e-3,
            ),
            (
                self.lower_z_mm,
                lower_kick[0] * 1.0e-3,
                lower_kick[1] * 1.0e-3,
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
            "scan_pixels_x": self.scan_pixels_x,
            "scan_lines": self.scan_lines,
            "scan_pixel_size_nm": self.scan_pixel_size_nm,
            "scan_field_of_view_x_nm": self.scan_field_of_view_x_nm,
            "scan_field_of_view_y_nm": self.scan_field_of_view_y_nm,
            "scan_command_matrix_mrad": self.scan_command_matrix_mrad,
            "scan_scale_residual": self.scan_scale_residual,
            "upper_coil_gain": self.upper_coil_gain,
            "lower_coil_gain": self.lower_coil_gain,
            "image_plane_lower_ratio_matrix": (
                self.image_plane_lower_ratio_matrix
            ),
            "image_plane_residual": self.image_plane_residual,
            "image_plane_target_z_mm": self.image_plane_target_z_mm,
            "image_plane_target_key": self.image_plane_target_key,
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
        "scan_pixels_x",
        "scan_lines",
        "scan_pixel_size_nm",
        "upper_coil_gain",
    ):
        if attribute in values:
            object.__setattr__(component, attribute, values[attribute])
    component.key = DESCAN_DEFLECTOR
    component.name = "AC Descan Coil"
    component.corrector = DESCAN_DEFLECTOR_DEFINITION.owner
    return component.apply_optical_position().validate()
