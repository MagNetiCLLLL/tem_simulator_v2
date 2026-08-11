"""Canonical paired AC Scan Coil and its downstream mechanical anchors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sin
from typing import ClassVar

from temsim import module_manifest
from temsim.component_keys import (
    AC_DEFLECTOR,
    canonical_corrector_element_key,
)


_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml", "exit"
)
_DEFAULT_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, AC_DEFLECTOR
)


@dataclass(frozen=True)
class AcDeflectorDefinition:
    key: str
    label: str
    center_from_source_mm: float
    mechanical_coil_length_mm: float
    mechanical_inter_coil_gap_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    effective_thickness_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "shared_column"
    kind: str = "paired_deflector"
    shape_profile: str = "paired_deflector_coils"
    interaction_kind: str = "time_dependent_paired_transverse_kick"

    @property
    def mechanical_center_from_tip_mm(self):
        return self.center_from_source_mm

    @property
    def mechanical_length_mm(self):
        return (
            2.0 * self.mechanical_coil_length_mm
            + self.mechanical_inter_coil_gap_mm
        )

    @property
    def optical_reference_from_tip_mm(self):
        return self.center_from_source_mm

    @property
    def upper_z_mm(self):
        return self.center_from_source_mm - 0.5 * (
            self.mechanical_coil_length_mm
            + self.mechanical_inter_coil_gap_mm
        )

    @property
    def lower_z_mm(self):
        return self.center_from_source_mm + 0.5 * (
            self.mechanical_coil_length_mm
            + self.mechanical_inter_coil_gap_mm
        )

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return AcDeflectorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.center_from_source_mm,
            kick_x_mrad=0.0,
            kick_y_mrad=0.0,
            effective_thickness_mm=self.effective_thickness_mm,
            mechanical_coil_length_mm=self.mechanical_coil_length_mm,
            mechanical_inter_coil_gap_mm=(
                self.mechanical_inter_coil_gap_mm
            ),
            enabled=True,
            colour=self.colour,
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            maximum_kick_mrad=self.maximum_kick_mrad,
            corrector=self.owner,
            wobble_enabled=True,
            wobble_amplitude_x_mrad=0.1,
            wobble_amplitude_y_mrad=0.0,
            wobble_period_s=1.0,
            wobble_phase_deg=0.0,
            scan_enabled=False,
            scan_amplitude_x_mrad=0.1,
            scan_amplitude_y_mrad=0.1,
            scan_frame_period_s=1.0,
            scan_pixels_x=32,
            scan_lines=32,
            upper_coil_gain=0.5,
            lower_coil_gain=-0.5,
            active_installation="probe",
        )


@dataclass
class AcDeflectorComponent:
    """Physical 15 mm + 10 mm + 15 mm AC double-deflector."""

    name: str
    key: str
    z_mm: float
    kick_x_mrad: float
    kick_y_mrad: float
    effective_thickness_mm: float
    mechanical_coil_length_mm: float
    mechanical_inter_coil_gap_mm: float
    enabled: bool
    colour: str
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    maximum_kick_mrad: float
    corrector: str = "shared_column"
    wobble_enabled: bool = True
    wobble_amplitude_x_mrad: float = 0.1
    wobble_amplitude_y_mrad: float = 0.0
    wobble_period_s: float = 1.0
    wobble_phase_deg: float = 0.0
    scan_enabled: bool = False
    scan_amplitude_x_mrad: float = 0.1
    scan_amplitude_y_mrad: float = 0.1
    scan_frame_period_s: float = 1.0
    scan_pixels_x: int = 32
    scan_lines: int = 32
    upper_coil_gain: float = 0.5
    lower_coil_gain: float = -0.5
    active_installation: str = "probe"

    EXPECTED_KEY: ClassVar[str] = AC_DEFLECTOR

    def __post_init__(self):
        object.__setattr__(
            self,
            "_pure_shift_lower_ratio_matrix",
            ((-1.0, 0.0), (0.0, -1.0)),
        )
        object.__setattr__(self, "_pure_shift_angular_residual", 0.0)
        object.__setattr__(self, "_pure_shift_calibrated", False)
        object.__setattr__(
            self,
            "lower_coil_gain",
            -float(self.upper_coil_gain),
        )
        object.__setattr__(self, "_geometry_ready", True)
        self._sync_mechanical_length()

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_geometry_ready", False)
        if ready and name == "upper_coil_gain":
            object.__setattr__(self, name, float(value))
            self._sync_pure_shift_gain()
            return
        if ready and name == "lower_coil_gain":
            ratio = self._pure_shift_lower_ratio_matrix
            representative = 0.5 * (
                float(ratio[0][0]) + float(ratio[1][1])
            )
            if abs(representative) <= 1.0e-12:
                raise ValueError(
                    "AC lower foil gain is coupled to the upper foil gain."
                )
            object.__setattr__(
                self,
                "upper_coil_gain",
                float(value) / representative,
            )
            self._sync_pure_shift_gain()
            return
        if ready and name in {
            "mechanical_coil_length_mm",
            "mechanical_inter_coil_gap_mm",
        }:
            value = float(value)
            if value <= 0.0:
                raise ValueError(
                    "AC Scan Coil coil length and gap must be positive."
                )
            object.__setattr__(self, name, value)
            self._sync_mechanical_length()
            return
        if ready and name == "mechanical_length_mm":
            length_mm = float(value)
            gap_mm = (
                length_mm - 2.0 * float(self.mechanical_coil_length_mm)
            )
            if gap_mm <= 0.0:
                raise ValueError(
                    "AC Scan Coil envelope must contain two coils and a gap."
                )
            object.__setattr__(
                self, "mechanical_inter_coil_gap_mm", gap_mm
            )
            object.__setattr__(self, name, length_mm)
            return
        object.__setattr__(self, name, value)

    def _sync_mechanical_length(self):
        object.__setattr__(
            self,
            "mechanical_length_mm",
            (
                2.0 * float(self.mechanical_coil_length_mm)
                + float(self.mechanical_inter_coil_gap_mm)
            ),
        )

    @property
    def owner(self):
        return self.corrector

    @property
    def kind(self):
        return "paired_deflector"

    @property
    def shape_profile(self):
        return "paired_deflector_coils"

    @property
    def interaction_kind(self):
        return "time_dependent_paired_transverse_kick"

    @property
    def optical_active(self):
        return bool(self.enabled)

    @property
    def mechanical_center_from_tip_mm(self):
        return self.z_mm

    @mechanical_center_from_tip_mm.setter
    def mechanical_center_from_tip_mm(self, value):
        self.z_mm = float(value)

    @property
    def optical_reference_from_tip_mm(self):
        return self.z_mm

    @optical_reference_from_tip_mm.setter
    def optical_reference_from_tip_mm(self, value):
        self.z_mm = float(value)

    @property
    def optical_plane_separation_mm(self):
        return (
            float(self.mechanical_coil_length_mm)
            + float(self.mechanical_inter_coil_gap_mm)
        )

    @property
    def upper_z_mm(self):
        return self.z_mm - self.optical_plane_separation_mm / 2.0

    @property
    def lower_z_mm(self):
        return self.z_mm + self.optical_plane_separation_mm / 2.0

    @property
    def upper_surface_z_mm(self):
        return self.z_mm - self.mechanical_length_mm / 2.0

    @property
    def lower_surface_z_mm(self):
        return self.z_mm + self.mechanical_length_mm / 2.0

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    @property
    def dc_offset_x_mrad(self):
        return self.kick_x_mrad

    @dc_offset_x_mrad.setter
    def dc_offset_x_mrad(self, value):
        self.kick_x_mrad = float(value)

    @property
    def dc_offset_y_mrad(self):
        return self.kick_y_mrad

    @dc_offset_y_mrad.setter
    def dc_offset_y_mrad(self, value):
        self.kick_y_mrad = float(value)

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError("AC Scan Coil key is not canonical.")
        if self.z_mm < 0.0:
            raise ValueError("AC Scan Coil must follow source z=0.")
        if (
            self.mechanical_coil_length_mm <= 0.0
            or self.mechanical_inter_coil_gap_mm <= 0.0
        ):
            raise ValueError("AC Scan Coil double-coil geometry is invalid.")
        expected_length_mm = (
            2.0 * self.mechanical_coil_length_mm
            + self.mechanical_inter_coil_gap_mm
        )
        if abs(self.mechanical_length_mm - expected_length_mm) > 1.0e-9:
            raise ValueError(
                "AC Scan Coil length must equal coil + gap + coil."
            )
        if not (
            0.0
            < self.effective_thickness_mm
            <= self.mechanical_coil_length_mm
        ):
            raise ValueError(
                "AC Scan Coil field thickness must fit one coil."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError("AC Scan Coil bore must fit inside its body.")
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError("AC Scan Coil kick limit must be positive.")
        if self.wobble_enabled and self.scan_enabled:
            raise ValueError(
                "AC wobble and raster scan are mutually exclusive."
            )
        if self.wobble_period_s <= 0.0:
            raise ValueError("AC wobble period must be positive.")
        if self.scan_frame_period_s <= 0.0:
            raise ValueError("AC scan frame period must be positive.")
        if int(self.scan_pixels_x) != self.scan_pixels_x or (
            self.scan_pixels_x < 2
        ):
            raise ValueError("AC scan must contain at least two X pixels.")
        if int(self.scan_lines) != self.scan_lines or self.scan_lines < 2:
            raise ValueError("AC scan must contain at least two lines.")
        coil_gains = (
            float(self.upper_coil_gain),
            float(self.lower_coil_gain),
        )
        if not all(isfinite(value) for value in coil_gains):
            raise ValueError("AC Scan Coil gains must be finite.")
        driven_values = (
            self.kick_x_mrad,
            self.kick_y_mrad,
            self.wobble_amplitude_x_mrad,
            self.wobble_amplitude_y_mrad,
            self.scan_amplitude_x_mrad,
            self.scan_amplitude_y_mrad,
        )
        if max(abs(float(value)) for value in driven_values) > (
            self.maximum_kick_mrad
        ):
            raise ValueError("AC Scan Coil drive exceeds its limit.")
        active_x_mrad = abs(float(self.kick_x_mrad))
        active_y_mrad = abs(float(self.kick_y_mrad))
        if self.scan_enabled:
            active_x_mrad += abs(float(self.scan_amplitude_x_mrad))
            active_y_mrad += abs(float(self.scan_amplitude_y_mrad))
        elif self.wobble_enabled:
            active_x_mrad += abs(float(self.wobble_amplitude_x_mrad))
            active_y_mrad += abs(float(self.wobble_amplitude_y_mrad))
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
                "AC upper/lower coil drive exceeds its individual limit."
            )
        if self.active_installation not in {"probe", "standalone"}:
            raise ValueError("AC Scan Coil installation is invalid.")
        return self

    def apply_optical_position(self):
        return self

    def apply_optical_positions(self):
        return self

    def wobble_factor(self, time_s):
        if not self.wobble_enabled or self.scan_enabled:
            return 0.0
        phase_rad = self.wobble_phase_deg * pi / 180.0
        return sin(
            2.0 * pi * float(time_s) / self.wobble_period_s + phase_rad
        )

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
        wobble_factor = self.wobble_factor(time_s)
        scan_x_mrad, scan_y_mrad = self.scan_kick_mrad(time_s)
        return (
            self.wobble_amplitude_x_mrad * wobble_factor + scan_x_mrad,
            self.wobble_amplitude_y_mrad * wobble_factor + scan_y_mrad,
        )

    def instantaneous_kick_mrad(self, time_s=0.0):
        dynamic_x_mrad, dynamic_y_mrad = self.dynamic_kick_mrad(time_s)
        return (
            self.kick_x_mrad + dynamic_x_mrad,
            self.kick_y_mrad + dynamic_y_mrad,
        )

    def kick_events(self, time_s=0.0):
        if not self.enabled:
            return ()
        kick_x_mrad, kick_y_mrad = self.instantaneous_kick_mrad(time_s)
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

    def _sync_pure_shift_gain(self):
        ratio = self._pure_shift_lower_ratio_matrix
        representative = 0.5 * (
            float(ratio[0][0]) + float(ratio[1][1])
        )
        object.__setattr__(
            self,
            "lower_coil_gain",
            float(self.upper_coil_gain) * representative,
        )

    def set_pure_shift_coupling(
        self,
        lower_from_upper,
        angular_residual=0.0,
    ):
        """Install the lower-foil map that cancels angle at the specimen."""

        rows = tuple(
            tuple(float(value) for value in row)
            for row in lower_from_upper
        )
        if len(rows) != 2 or any(len(row) != 2 for row in rows):
            raise ValueError("AC pure-shift coupling must be a 2x2 matrix.")
        if not all(isfinite(value) for row in rows for value in row):
            raise ValueError("AC pure-shift coupling must be finite.")
        residual = float(angular_residual)
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("AC pure-shift angular residual must be finite.")
        object.__setattr__(self, "_pure_shift_lower_ratio_matrix", rows)
        object.__setattr__(self, "_pure_shift_angular_residual", residual)
        object.__setattr__(self, "_pure_shift_calibrated", True)
        self._sync_pure_shift_gain()
        return self

    @property
    def pure_shift_lower_ratio_matrix(self):
        return tuple(
            tuple(float(value) for value in row)
            for row in self._pure_shift_lower_ratio_matrix
        )

    @property
    def pure_shift_angular_residual(self):
        return float(self._pure_shift_angular_residual)

    def coil_kick_matrices(self):
        """Return upper/lower maps from one shared command to both foils."""

        gain = float(self.upper_coil_gain)
        ratio = self._pure_shift_lower_ratio_matrix
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

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.z_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_coil_length_mm": self.mechanical_coil_length_mm,
            "mechanical_inter_coil_gap_mm": (
                self.mechanical_inter_coil_gap_mm
            ),
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "upper_plane_z_mm": self.upper_z_mm,
            "lower_plane_z_mm": self.lower_z_mm,
            "effective_coil_thickness_mm": self.effective_thickness_mm,
            "enabled": self.enabled,
            "wobble_enabled": self.wobble_enabled,
            "scan_enabled": self.scan_enabled,
            "scan_pixels_x": self.scan_pixels_x,
            "scan_lines": self.scan_lines,
            "upper_coil_gain": self.upper_coil_gain,
            "lower_coil_gain": self.lower_coil_gain,
            "pure_shift_lower_ratio_matrix": (
                self.pure_shift_lower_ratio_matrix
            ),
            "pure_shift_angular_residual": self.pure_shift_angular_residual,
        }


# The canonical Probe-Corrector package places the AC centre after
# DP12 → Condenser Stigmator → Beam Deflector with 5 mm clearances.
AC_DEFLECTOR_DEFINITION = AcDeflectorDefinition(
    key=AC_DEFLECTOR,
    label="AC Scan Coil",
    center_from_source_mm=(
        _DEFAULT_COLUMN_ORIGIN_Z_MM
        + float(_DEFAULT_PART["local_center_z_mm"])
    ),
    mechanical_coil_length_mm=float(
        _DEFAULT_PART["mechanical_coil_length_mm"]
    ),
    mechanical_inter_coil_gap_mm=float(
        _DEFAULT_PART["mechanical_inter_coil_gap_mm"]
    ),
    mechanical_outer_diameter_mm=float(
        _DEFAULT_PART["mechanical_outer_diameter_mm"]
    ),
    mechanical_clear_bore_diameter_mm=float(
        _DEFAULT_PART["mechanical_clear_bore_diameter_mm"]
    ),
    effective_thickness_mm=float(_DEFAULT_PART["effective_thickness_mm"]),
    maximum_kick_mrad=100.0,
    colour="#26c6da",
)


def resolve_ac_scan_coil_installation(state, *, probe_installed):
    """Resolve the AC coil as part of the upper-objective package."""

    from temsim.optics.upper_objective_package import (
        resolve_upper_objective_package,
    )

    resolve_upper_objective_package(
        state,
        probe_installed=probe_installed,
    )
    return state.ac_deflector.validate()


def resolve_ac_downstream_anchors(state, *, probe_installed):
    """Compatibility wrapper for the upper-objective package resolver.

    AC no longer owns the downstream chain.  The package is rooted at the
    condenser stigmator and contains Beam, AC, and Mini Condenser in that
    order inside the Upper Objective Lens body above the pole piece.
    """

    from temsim.optics.upper_objective_package import (
        resolve_upper_objective_package,
    )

    return resolve_upper_objective_package(
        state,
        probe_installed=probe_installed,
    )


def create_ac_deflector():
    return AC_DEFLECTOR_DEFINITION.create_component()


def ac_deflector_from_dict(data):
    """Restore a paired AC record or migrate the obsolete single plane."""

    values = dict(data)
    component = create_ac_deflector()
    for attribute in (
        "kick_x_mrad",
        "kick_y_mrad",
        "enabled",
        "colour",
        "maximum_kick_mrad",
        "wobble_enabled",
        "wobble_amplitude_x_mrad",
        "wobble_amplitude_y_mrad",
        "wobble_period_s",
        "wobble_phase_deg",
        "scan_enabled",
        "scan_amplitude_x_mrad",
        "scan_amplitude_y_mrad",
        "scan_frame_period_s",
        "scan_pixels_x",
        "scan_lines",
        "upper_coil_gain",
    ):
        if attribute in values:
            setattr(component, attribute, values[attribute])
    component.key = canonical_corrector_element_key(
        values.get("key", AC_DEFLECTOR)
    )
    component.key = AC_DEFLECTOR
    if component.wobble_enabled and component.scan_enabled:
        component.wobble_enabled = False
    component.corrector = AC_DEFLECTOR_DEFINITION.owner
    return component.validate()
