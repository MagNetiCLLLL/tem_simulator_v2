"""Modular probe-corrector assembly, built in physical column order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.mechanical_axis import resolve_mechanical_axis
from temsim.component_keys import (
    ADAPTER_LENS,
    BEAM_DEFLECTOR,
    CONDENSER_STIGMATOR,
    PROBE_DPH2_DEFLECTOR,
    PROBE_DP22_DEFLECTOR,
    PROBE_DP21_DEFLECTOR,
    PROBE_DP11_DEFLECTOR,
    PROBE_DP12_SCAN_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR,
    PROBE_HP2_HEXAPOLE,
    PROBE_HP1_HEXAPOLE,
    PROBE_HPOL_HEXAPOLE,
    PROBE_HPC_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE,
    PROBE_QPH1_QUADRUPOLE,
    PROBE_QPOL_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
    PROBE_CORRECTOR_KEYS,
    canonical_corrector_element_key,
    canonical_deflector_key,
    canonical_lens_key,
)
from temsim.optics.condenser_lens import AxialFieldTerm
from temsim.optics.lens_focal_length import focal_length_mm as _focal_length_mm
from temsim.optics.single_plane_deflector import (
    SinglePlaneDeflectorComponent,
    restore_single_plane_deflector,
)
from temsim.optics.quadrupole import (
    QuadrupoleComponent,
    restore_quadrupole,
)
from temsim.optics.hexapole import HexapoleComponent, restore_hexapole
from temsim.optics.round_lens import (
    RoundLensComponent,
    restore_round_lens,
)
from temsim.optics.paired_deflector import (
    PairedDeflectorComponent,
    restore_paired_deflector,
)


_PROBE_DEFAULT_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_PROBE_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml",
    "exit",
)
_PROBE_DEFAULT_PARTS = {
    key: module_manifest.part_data(_PROBE_DEFAULT_MODULE_PATH, key)
    for key in PROBE_CORRECTOR_KEYS
}

# First-order unit-magnification HP2 -> HP1 relay at 300 kV.  The downstream
# hexapole is rotated to cancel the pair's second-order threefold term in the
# Larmor frame, leaving the desired negative third-order spherical term.  The
# pair is calibrated at the exact specimen plane; the endpoint-safe RK4 grid
# must not reuse the former value fitted 0.03885 mm before that plane.
PROBE_MAIN_HEXAPOLE_STRENGTH_M3 = 5.08073490e5
PROBE_HP1_HEXAPOLE_STRENGTH_RATIO = 0.59513503
PROBE_HP1_HEXAPOLE_ORIENTATION_RAD = -0.04234211


def _probe_manifest_value(key, field):
    return float(_PROBE_DEFAULT_PARTS[key][field])


def _probe_manifest_field_polarity(key):
    return int(_PROBE_DEFAULT_PARTS[key]["field_polarity"])


def _probe_manifest_absolute(key, field):
    return (
        _PROBE_DEFAULT_COLUMN_ORIGIN_Z_MM
        + _probe_manifest_value(key, field)
    )


def _create_single_deflector(definition, component_type):
    return component_type(
        name=definition.label,
        key=definition.key,
        z_mm=definition.optical_reference_from_tip_mm,
        kick_x_mrad=0.0,
        kick_y_mrad=0.0,
        effective_thickness_mm=definition.effective_thickness_mm,
        enabled=True,
        colour=definition.colour,
        mechanical_center_from_tip_mm=(
            definition.mechanical_center_from_tip_mm
        ),
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=(
            definition.mechanical_outer_diameter_mm
        ),
        mechanical_clear_bore_diameter_mm=(
            definition.mechanical_clear_bore_diameter_mm
        ),
        optical_reference_from_tip_mm=(
            definition.optical_reference_from_tip_mm
        ),
        maximum_kick_mrad=definition.maximum_kick_mrad,
        corrector="probe",
    )


def _create_quadrupole(definition, component_type):
    return component_type(
        name=definition.label,
        key=definition.key,
        z_mm=definition.optical_reference_from_tip_mm,
        strength_m2=0.0,
        maximum_strength_m2=definition.maximum_strength_m2,
        effective_length_mm=definition.effective_length_mm,
        enabled=True,
        colour=definition.colour,
        mechanical_center_from_tip_mm=(
            definition.mechanical_center_from_tip_mm
        ),
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=(
            definition.mechanical_outer_diameter_mm
        ),
        mechanical_clear_bore_diameter_mm=(
            definition.mechanical_clear_bore_diameter_mm
        ),
        optical_reference_from_tip_mm=(
            definition.optical_reference_from_tip_mm
        ),
        corrector="probe",
    )


def _create_hexapole(definition, component_type):
    is_hp1 = definition.key == PROBE_HP1_HEXAPOLE
    return component_type(
        name=definition.label,
        key=definition.key,
        z_mm=definition.optical_reference_from_tip_mm,
        strength_m3=(
            PROBE_MAIN_HEXAPOLE_STRENGTH_M3
            * PROBE_HP1_HEXAPOLE_STRENGTH_RATIO
            if is_hp1 else 0.0
        ),
        orientation_rad=(
            PROBE_HP1_HEXAPOLE_ORIENTATION_RAD if is_hp1 else 0.0
        ),
        maximum_strength_m3=definition.maximum_strength_m3,
        effective_length_mm=definition.effective_length_mm,
        enabled=True,
        colour=definition.colour,
        mechanical_center_from_tip_mm=(
            definition.mechanical_center_from_tip_mm
        ),
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=(
            definition.mechanical_outer_diameter_mm
        ),
        mechanical_clear_bore_diameter_mm=(
            definition.mechanical_clear_bore_diameter_mm
        ),
        optical_reference_from_tip_mm=(
            definition.optical_reference_from_tip_mm
        ),
        corrector="probe",
    )


def _create_round_lens(definition, component_type):
    return component_type(
        name=definition.label,
        key=definition.key,
        z_mm=definition.optical_reference_from_tip_mm,
        b0_t=definition.maximum_peak_field_t,
        a_mm=definition.field_scale_half_width_mm,
        percent=definition.default_excitation_percent,
        max_percent=definition.maximum_excitation_percent,
        colour=definition.colour,
        gaussian=[
            AxialFieldTerm(0.09, -1.0, 0.90),
            AxialFieldTerm(0.82, 0.0, 0.55),
            AxialFieldTerm(0.09, 1.0, 0.90),
        ],
        enabled=True,
        cs_mm=None,
        cc_mm=None,
        polarity=_probe_manifest_field_polarity(definition.key),
        normalise_profile_peak=False,
        mechanical_center_from_tip_mm=(
            definition.mechanical_center_from_tip_mm
        ),
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=(
            definition.mechanical_outer_diameter_mm
        ),
        bore_diameter_mm=definition.bore_diameter_mm,
        pole_gap_mm=definition.pole_gap_mm,
        optical_reference_from_tip_mm=(
            definition.optical_reference_from_tip_mm
        ),
        corrector="probe",
    )


@dataclass(frozen=True)
class AdapterLensDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_from_tip_mm: float
    maximum_peak_field_t: float
    field_scale_half_width_mm: float
    default_excitation_percent: float
    maximum_excitation_percent: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "round_lens"
    shape_profile: str = "magnetic_lens_yoke"
    interaction_kind: str = "axial_magnetic_field"

    def create_component(self):
        gaussian = [
            AxialFieldTerm(0.09, -1.0, 0.90),
            AxialFieldTerm(0.82, 0.0, 0.55),
            AxialFieldTerm(0.09, 1.0, 0.90),
        ]
        return AdapterLensComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            b0_t=self.maximum_peak_field_t,
            a_mm=self.field_scale_half_width_mm,
            percent=self.default_excitation_percent,
            max_percent=self.maximum_excitation_percent,
            colour=self.colour,
            gaussian=gaussian,
            enabled=True,
            cs_mm=None,
            cc_mm=None,
            polarity=_probe_manifest_field_polarity(self.key),
            normalise_profile_peak=False,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            bore_diameter_mm=self.bore_diameter_mm,
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
        )


@dataclass
class AdapterLensComponent:
    """The single ADL record used by layout, magnetic solver and overlays."""

    name: str
    key: str
    z_mm: float
    b0_t: float
    a_mm: float
    percent: float
    max_percent: float
    colour: str
    gaussian: list[AxialFieldTerm]
    enabled: bool
    cs_mm: float | None
    cc_mm: float | None
    polarity: int
    normalise_profile_peak: bool
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_from_tip_mm: float

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_from_tip_mm",
            "optical_reference_from_tip_mm",
        }:
            value = float(value)
        coupling_ready = self.__dict__.get(
            "_position_coupling_ready", False
        )
        if name == "mechanical_center_from_tip_mm" and coupling_ready:
            delta_mm = float(value) - float(
                self.mechanical_center_from_tip_mm
            )
            object.__setattr__(self, name, float(value))
            optical = float(self.optical_reference_from_tip_mm) + delta_mm
            object.__setattr__(
                self, "optical_reference_from_tip_mm", optical
            )
            object.__setattr__(self, "z_mm", optical)
            return
        if name == "optical_reference_from_tip_mm" and coupling_ready:
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "z_mm", float(value))
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return "probe_corrector"

    @property
    def kind(self):
        return "round_lens"

    @property
    def shape_profile(self):
        return "magnetic_lens_yoke"

    @property
    def interaction_kind(self):
        return "axial_magnetic_field"

    @property
    def length_mm(self):
        """Compatibility name used by the ray-overlay renderer."""
        return self.mechanical_length_mm

    @property
    def optical_active(self):
        return True

    @property
    def effective_aperture_radius_mm(self):
        return self.bore_diameter_mm / 2.0

    def scale(self):
        return (
            self.b0_t * self.percent / 100.0
            if self.enabled else 0.0
        )

    def validate(self):
        if self.key != ADAPTER_LENS:
            raise ValueError("Adapter Lens key is not canonical.")
        if self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(
                "Adapter Lens mechanical centre must follow the tip."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Adapter Lens mechanical length must be positive."
            )
        if self.mechanical_outer_diameter_mm <= self.bore_diameter_mm:
            raise ValueError(
                "Adapter Lens outer diameter must exceed its bore."
            )
        if self.bore_diameter_mm <= 0.0 or self.pole_gap_mm <= 0.0:
            raise ValueError(
                "Adapter Lens bore and pole gap must be positive."
            )
        if self.b0_t < 0.0 or self.a_mm <= 0.0:
            raise ValueError(
                "Adapter Lens field and field width must be valid."
            )
        if not 0.0 <= self.percent <= self.max_percent:
            raise ValueError(
                "Adapter Lens excitation exceeds its configured range."
            )
        if not self.gaussian or any(
            term.sigma <= 0.0 for term in self.gaussian
        ):
            raise ValueError(
                "Adapter Lens requires valid Gaussian field terms."
            )
        return self

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_from_tip_mm)
        return self

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        field = np.zeros_like(z)
        if not self.enabled:
            return field
        for term in self.gaussian:
            sigma = max(abs(term.sigma * self.a_mm), 1e-12)
            centre = self.z_mm + term.offset * self.a_mm
            field += term.amplitude * np.exp(
                -0.5 * ((z - centre) / sigma) ** 2
            )
        return float(self.polarity) * self.scale() * field

    def focal_length_mm(self):
        return _focal_length_mm(self, 300.0)

    def field_support_mm(self, sigma_cutoff=7.0):
        reaches = [
            abs(term.offset * self.a_mm)
            + float(sigma_cutoff) * abs(term.sigma * self.a_mm)
            for term in self.gaussian
        ]
        half = max(reaches, default=0.0)
        return self.z_mm - half, self.z_mm + half

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": (
                self.mechanical_center_from_tip_mm
            ),
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "bore_diameter_mm": self.bore_diameter_mm,
            "pole_gap_mm": self.pole_gap_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        start, end = self.field_support_mm()
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_mm(),
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class Dph2DeflectorDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    effective_thickness_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "deflector"
    shape_profile: str = "single_deflector_coil"
    interaction_kind: str = "thin_transverse_kick"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return Dph2DeflectorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            kick_x_mrad=0.0,
            kick_y_mrad=0.0,
            effective_thickness_mm=self.effective_thickness_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            maximum_kick_mrad=self.maximum_kick_mrad,
            corrector="probe",
        )


@dataclass
class Dph2DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DPH2_DEFLECTOR


@dataclass(frozen=True)
class Dp22DeflectorDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    effective_thickness_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "deflector"
    shape_profile: str = "single_deflector_coil"
    interaction_kind: str = "thin_transverse_kick"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return Dp22DeflectorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            kick_x_mrad=0.0,
            kick_y_mrad=0.0,
            effective_thickness_mm=self.effective_thickness_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            maximum_kick_mrad=self.maximum_kick_mrad,
            corrector="probe",
        )


@dataclass
class Dp22DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DP22_DEFLECTOR


@dataclass(frozen=True)
class Dp21DeflectorDefinition(Dp22DeflectorDefinition):
    def create_component(self):
        return _create_single_deflector(self, Dp21DeflectorComponent)


@dataclass
class Dp21DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DP21_DEFLECTOR


@dataclass(frozen=True)
class Dph1DeflectorDefinition(Dp22DeflectorDefinition):
    def create_component(self):
        return _create_single_deflector(self, Dph1DeflectorComponent)


@dataclass
class Dph1DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DPH1_DEFLECTOR


@dataclass(frozen=True)
class Dp11DeflectorDefinition(Dp22DeflectorDefinition):
    def create_component(self):
        return _create_single_deflector(self, Dp11DeflectorComponent)


@dataclass
class Dp11DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DP11_DEFLECTOR


@dataclass(frozen=True)
class Qph2QuadrupoleDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    effective_length_mm: float
    maximum_strength_m2: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "quadrupole"
    shape_profile: str = "quadrupole_body"
    interaction_kind: str = "distributed_quadrupole_field"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return Qph2QuadrupoleComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            strength_m2=0.0,
            maximum_strength_m2=self.maximum_strength_m2,
            effective_length_mm=self.effective_length_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            corrector="probe",
        )


@dataclass
class Qph2QuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_QPH2_QUADRUPOLE


@dataclass(frozen=True)
class QpcQuadrupoleDefinition(Qph2QuadrupoleDefinition):
    def create_component(self):
        return _create_quadrupole(self, QpcQuadrupoleComponent)


@dataclass
class QpcQuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_QPC_QUADRUPOLE


@dataclass(frozen=True)
class Qph1QuadrupoleDefinition(Qph2QuadrupoleDefinition):
    def create_component(self):
        return _create_quadrupole(self, Qph1QuadrupoleComponent)


@dataclass
class Qph1QuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_QPH1_QUADRUPOLE


@dataclass(frozen=True)
class QpolQuadrupoleDefinition(Qph2QuadrupoleDefinition):
    def create_component(self):
        return _create_quadrupole(self, QpolQuadrupoleComponent)


@dataclass
class QpolQuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_QPOL_QUADRUPOLE


@dataclass(frozen=True)
class Hp2HexapoleDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_from_tip_mm: float
    effective_length_mm: float
    maximum_strength_m3: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "hexapole"
    shape_profile: str = "hexapole_body"
    interaction_kind: str = "distributed_hexapole_field"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return Hp2HexapoleComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            strength_m3=PROBE_MAIN_HEXAPOLE_STRENGTH_M3,
            orientation_rad=0.0,
            maximum_strength_m3=self.maximum_strength_m3,
            effective_length_mm=self.effective_length_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            corrector="probe",
        )


@dataclass
class Hp2HexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_HP2_HEXAPOLE


@dataclass(frozen=True)
class HpcHexapoleDefinition(Hp2HexapoleDefinition):
    def create_component(self):
        return HpcHexapoleComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            strength_m3=0.0,
            maximum_strength_m3=self.maximum_strength_m3,
            effective_length_mm=self.effective_length_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            corrector="probe",
        )


@dataclass
class HpcHexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_HPC_HEXAPOLE


@dataclass(frozen=True)
class Hp1HexapoleDefinition(Hp2HexapoleDefinition):
    def create_component(self):
        return _create_hexapole(self, Hp1HexapoleComponent)


@dataclass
class Hp1HexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_HP1_HEXAPOLE


@dataclass(frozen=True)
class HpolHexapoleDefinition(Hp2HexapoleDefinition):
    def create_component(self):
        return _create_hexapole(self, HpolHexapoleComponent)


@dataclass
class HpolHexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_HPOL_HEXAPOLE


@dataclass(frozen=True)
class Tl22LensDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_from_tip_mm: float
    maximum_peak_field_t: float
    field_scale_half_width_mm: float
    default_excitation_percent: float
    maximum_excitation_percent: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "round_lens"
    shape_profile: str = "magnetic_lens_yoke"
    interaction_kind: str = "axial_magnetic_field"

    @property
    def effective_aperture_radius_mm(self):
        return self.bore_diameter_mm / 2.0

    def create_component(self):
        return Tl22LensComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            b0_t=self.maximum_peak_field_t,
            a_mm=self.field_scale_half_width_mm,
            percent=self.default_excitation_percent,
            max_percent=self.maximum_excitation_percent,
            colour=self.colour,
            gaussian=[
                AxialFieldTerm(0.09, -1.0, 0.90),
                AxialFieldTerm(0.82, 0.0, 0.55),
                AxialFieldTerm(0.09, 1.0, 0.90),
            ],
            enabled=True,
            cs_mm=None,
            cc_mm=None,
            polarity=_probe_manifest_field_polarity(self.key),
            normalise_profile_peak=False,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            bore_diameter_mm=self.bore_diameter_mm,
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_from_tip_mm=(
                self.optical_reference_from_tip_mm
            ),
            corrector="probe",
        )


@dataclass
class Tl22LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_TL22_LENS


@dataclass(frozen=True)
class Tl21LensDefinition(Tl22LensDefinition):
    def create_component(self):
        return _create_round_lens(self, Tl21LensComponent)


@dataclass
class Tl21LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_TL21_LENS


@dataclass(frozen=True)
class Tl12LensDefinition(Tl22LensDefinition):
    def create_component(self):
        return _create_round_lens(self, Tl12LensComponent)


@dataclass
class Tl12LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_TL12_LENS


@dataclass(frozen=True)
class Dp12ScanDeflectorDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_upper_reference_from_tip_mm: float
    optical_lower_reference_from_tip_mm: float
    effective_thickness_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "probe_corrector"
    kind: str = "virtual_layout"
    shape_profile: str = "virtual_plane"
    interaction_kind: str = "none"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    @property
    def optical_reference_from_tip_mm(self):
        return 0.5 * (
            self.optical_upper_reference_from_tip_mm
            + self.optical_lower_reference_from_tip_mm
        )

    def create_component(self):
        return Dp12ScanDeflectorComponent(
            name=self.label,
            key=self.key,
            upper_z_mm=self.optical_upper_reference_from_tip_mm,
            lower_z_mm=self.optical_lower_reference_from_tip_mm,
            upper_x_mrad=0.0,
            upper_y_mrad=0.0,
            lower_x_mrad=0.0,
            lower_y_mrad=0.0,
            thickness_mm=self.effective_thickness_mm,
            enabled=True,
            colour=self.colour,
            mechanical_center_from_tip_mm=(
                self.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_upper_reference_from_tip_mm=(
                self.optical_upper_reference_from_tip_mm
            ),
            optical_lower_reference_from_tip_mm=(
                self.optical_lower_reference_from_tip_mm
            ),
            maximum_kick_mrad=self.maximum_kick_mrad,
        )


@dataclass
class Dp12ScanDeflectorComponent(PairedDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = PROBE_DP12_SCAN_DEFLECTOR
    OWNER: ClassVar[str] = "probe_corrector"
    KIND: ClassVar[str] = "virtual_layout"
    SHAPE_PROFILE: ClassVar[str] = "virtual_plane"
    INTERACTION_KIND: ClassVar[str] = "none"

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError("DP12 virtual layout key is not canonical.")
        if self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(
                "DP12 virtual layout centre must follow the source."
            )
        if self.mechanical_length_mm != 0.0:
            raise ValueError(
                "DP12 virtual layout must not consume mechanical length."
            )
        return self

    def kick_events(self):
        return ()

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "layout_only": True,
            "enabled": self.enabled,
        }


ADAPTER_LENS_DEFINITION = AdapterLensDefinition(
    key=ADAPTER_LENS,
    label="ADL (Adapter Lens)",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        ADAPTER_LENS, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(ADAPTER_LENS, "length_mm"),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        ADAPTER_LENS, "mechanical_outer_diameter_mm"
    ),
    bore_diameter_mm=_probe_manifest_value(ADAPTER_LENS, "bore_diameter_mm"),
    pole_gap_mm=_probe_manifest_value(ADAPTER_LENS, "pole_gap_mm"),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        ADAPTER_LENS, "optical_reference_local_z_mm"
    ),
    maximum_peak_field_t=0.12,
    field_scale_half_width_mm=7.0,
    default_excitation_percent=42.0,
    maximum_excitation_percent=100.0,
    colour="#6a1b9a",
)

DPH2_DEFLECTOR_DEFINITION = Dph2DeflectorDefinition(
    key=PROBE_DPH2_DEFLECTOR,
    label="DPH2 Deflector",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DPH2_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DPH2_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DPH2_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DPH2_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DPH2_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DPH2_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#8e24aa",
)

DP22_DEFLECTOR_DEFINITION = Dp22DeflectorDefinition(
    key=PROBE_DP22_DEFLECTOR,
    label="DP22 Deflector",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP22_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DP22_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DP22_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DP22_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP22_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DP22_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#8e24aa",
)

QPH2_QUADRUPOLE_DEFINITION = Qph2QuadrupoleDefinition(
    key=PROBE_QPH2_QUADRUPOLE,
    label="QPH2 Quadrupole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPH2_QUADRUPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_QPH2_QUADRUPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_QPH2_QUADRUPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_QPH2_QUADRUPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPH2_QUADRUPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_QPH2_QUADRUPOLE, "effective_length_mm"
    ),
    maximum_strength_m2=300.0,
    colour="#8e24aa",
)

HP2_HEXAPOLE_DEFINITION = Hp2HexapoleDefinition(
    key=PROBE_HP2_HEXAPOLE,
    label="HP2 Hexapole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_HP2_HEXAPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_HP2_HEXAPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_HP2_HEXAPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_HP2_HEXAPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_HP2_HEXAPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_HP2_HEXAPOLE, "effective_length_mm"
    ),
    maximum_strength_m3=1.0e9,
    colour="#7b1fa2",
)

HPC_HEXAPOLE_DEFINITION = HpcHexapoleDefinition(
    key=PROBE_HPC_HEXAPOLE,
    label="HPC Hexapole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_HPC_HEXAPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_HPC_HEXAPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_HPC_HEXAPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_HPC_HEXAPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_HPC_HEXAPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_HPC_HEXAPOLE, "effective_length_mm"
    ),
    maximum_strength_m3=1.0e9,
    colour="#7b1fa2",
)

QPC_QUADRUPOLE_DEFINITION = QpcQuadrupoleDefinition(
    key=PROBE_QPC_QUADRUPOLE,
    label="QPC Quadrupole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPC_QUADRUPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_QPC_QUADRUPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_QPC_QUADRUPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_QPC_QUADRUPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPC_QUADRUPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_QPC_QUADRUPOLE, "effective_length_mm"
    ),
    maximum_strength_m2=300.0,
    colour="#8e24aa",
)

DP21_DEFLECTOR_DEFINITION = Dp21DeflectorDefinition(
    key=PROBE_DP21_DEFLECTOR,
    label="DP21 Deflector",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP21_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DP21_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DP21_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DP21_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP21_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DP21_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#7b1fa2",
)

TL22_LENS_DEFINITION = Tl22LensDefinition(
    key=PROBE_TL22_LENS,
    label="TL22 Transfer Lens",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL22_LENS, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_TL22_LENS, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_TL22_LENS, "mechanical_outer_diameter_mm"
    ),
    bore_diameter_mm=_probe_manifest_value(
        PROBE_TL22_LENS, "bore_diameter_mm"
    ),
    pole_gap_mm=_probe_manifest_value(PROBE_TL22_LENS, "pole_gap_mm"),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL22_LENS, "optical_reference_local_z_mm"
    ),
    # Preserve the calibrated 0.31809425 T operating field at 60%.
    maximum_peak_field_t=0.5301570833333334,
    field_scale_half_width_mm=6.0,
    default_excitation_percent=60.0,
    maximum_excitation_percent=100.0,
    colour="#8e24aa",
)

TL21_LENS_DEFINITION = Tl21LensDefinition(
    key=PROBE_TL21_LENS,
    label="TL21 Transfer Lens",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL21_LENS, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_TL21_LENS, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_TL21_LENS, "mechanical_outer_diameter_mm"
    ),
    bore_diameter_mm=_probe_manifest_value(
        PROBE_TL21_LENS, "bore_diameter_mm"
    ),
    pole_gap_mm=_probe_manifest_value(PROBE_TL21_LENS, "pole_gap_mm"),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL21_LENS, "optical_reference_local_z_mm"
    ),
    # Preserve the calibrated 0.29864759 T operating field at 60%.
    maximum_peak_field_t=0.49774598333333335,
    field_scale_half_width_mm=6.0,
    default_excitation_percent=60.0,
    maximum_excitation_percent=100.0,
    colour="#c2185b",
)

DPH1_DEFLECTOR_DEFINITION = Dph1DeflectorDefinition(
    key=PROBE_DPH1_DEFLECTOR,
    label="DPH1 Deflector",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DPH1_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DPH1_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DPH1_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DPH1_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DPH1_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DPH1_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#7b1fa2",
)

QPH1_QUADRUPOLE_DEFINITION = Qph1QuadrupoleDefinition(
    key=PROBE_QPH1_QUADRUPOLE,
    label="QPH1 Quadrupole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPH1_QUADRUPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_QPH1_QUADRUPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_QPH1_QUADRUPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_QPH1_QUADRUPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPH1_QUADRUPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_QPH1_QUADRUPOLE, "effective_length_mm"
    ),
    maximum_strength_m2=300.0,
    colour="#8e24aa",
)

HP1_HEXAPOLE_DEFINITION = Hp1HexapoleDefinition(
    key=PROBE_HP1_HEXAPOLE,
    label="HP1 Hexapole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_HP1_HEXAPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_HP1_HEXAPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_HP1_HEXAPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_HP1_HEXAPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_HP1_HEXAPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_HP1_HEXAPOLE, "effective_length_mm"
    ),
    maximum_strength_m3=1.0e9,
    colour="#7b1fa2",
)

HPOL_HEXAPOLE_DEFINITION = HpolHexapoleDefinition(
    key=PROBE_HPOL_HEXAPOLE,
    label="HPol Hexapole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_HPOL_HEXAPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_HPOL_HEXAPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_HPOL_HEXAPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_HPOL_HEXAPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_HPOL_HEXAPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_HPOL_HEXAPOLE, "effective_length_mm"
    ),
    maximum_strength_m3=1.0e9,
    colour="#8e24aa",
)

QPOL_QUADRUPOLE_DEFINITION = QpolQuadrupoleDefinition(
    key=PROBE_QPOL_QUADRUPOLE,
    label="QPol Quadrupole",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPOL_QUADRUPOLE, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_QPOL_QUADRUPOLE, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_QPOL_QUADRUPOLE, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_QPOL_QUADRUPOLE, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_QPOL_QUADRUPOLE, "optical_reference_local_z_mm"
    ),
    effective_length_mm=_probe_manifest_value(
        PROBE_QPOL_QUADRUPOLE, "effective_length_mm"
    ),
    maximum_strength_m2=300.0,
    colour="#8e24aa",
)

DP11_DEFLECTOR_DEFINITION = Dp11DeflectorDefinition(
    key=PROBE_DP11_DEFLECTOR,
    label="DP11 Deflector",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP11_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DP11_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DP11_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DP11_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP11_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DP11_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#6a1b9a",
)

TL12_LENS_DEFINITION = Tl12LensDefinition(
    key=PROBE_TL12_LENS,
    label="TL12 Transfer Lens",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL12_LENS, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_TL12_LENS, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_TL12_LENS, "mechanical_outer_diameter_mm"
    ),
    bore_diameter_mm=_probe_manifest_value(
        PROBE_TL12_LENS, "bore_diameter_mm"
    ),
    pole_gap_mm=_probe_manifest_value(PROBE_TL12_LENS, "pole_gap_mm"),
    optical_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_TL12_LENS, "optical_reference_local_z_mm"
    ),
    # Preserve the calibrated 0.33 T operating field at 60%.
    maximum_peak_field_t=0.55,
    field_scale_half_width_mm=7.0,
    default_excitation_percent=60.0,
    maximum_excitation_percent=100.0,
    colour="#ad1457",
)

DP12_SCAN_DEFLECTOR_DEFINITION = Dp12ScanDeflectorDefinition(
    key=PROBE_DP12_SCAN_DEFLECTOR,
    label="DP12 (virtual)",
    mechanical_center_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP12_SCAN_DEFLECTOR, "local_center_z_mm"
    ),
    mechanical_length_mm=_probe_manifest_value(
        PROBE_DP12_SCAN_DEFLECTOR, "length_mm"
    ),
    mechanical_outer_diameter_mm=_probe_manifest_value(
        PROBE_DP12_SCAN_DEFLECTOR, "mechanical_outer_diameter_mm"
    ),
    mechanical_clear_bore_diameter_mm=_probe_manifest_value(
        PROBE_DP12_SCAN_DEFLECTOR, "mechanical_clear_bore_diameter_mm"
    ),
    optical_upper_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP12_SCAN_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    optical_lower_reference_from_tip_mm=_probe_manifest_absolute(
        PROBE_DP12_SCAN_DEFLECTOR, "optical_reference_local_z_mm"
    ),
    effective_thickness_mm=_probe_manifest_value(
        PROBE_DP12_SCAN_DEFLECTOR, "effective_thickness_mm"
    ),
    maximum_kick_mrad=100.0,
    colour="#00838f",
)


PROBE_CORRECTOR_DEFINITIONS_IN_MECHANICAL_ORDER = (
    ADAPTER_LENS_DEFINITION,
    DPH2_DEFLECTOR_DEFINITION,
    QPH2_QUADRUPOLE_DEFINITION,
    HP2_HEXAPOLE_DEFINITION,
    TL22_LENS_DEFINITION,
    DP22_DEFLECTOR_DEFINITION,
    HPC_HEXAPOLE_DEFINITION,
    QPC_QUADRUPOLE_DEFINITION,
    DP21_DEFLECTOR_DEFINITION,
    TL21_LENS_DEFINITION,
    DPH1_DEFLECTOR_DEFINITION,
    QPH1_QUADRUPOLE_DEFINITION,
    HP1_HEXAPOLE_DEFINITION,
    HPOL_HEXAPOLE_DEFINITION,
    QPOL_QUADRUPOLE_DEFINITION,
    DP11_DEFLECTOR_DEFINITION,
    TL12_LENS_DEFINITION,
    DP12_SCAN_DEFLECTOR_DEFINITION,
)

PROBE_CORRECTOR_DEFAULT_OFFSETS_FROM_ADAPTER_MM = {
    definition.key: (
        float(definition.mechanical_center_from_tip_mm)
        - float(ADAPTER_LENS_DEFINITION.mechanical_center_from_tip_mm)
    )
    for definition in PROBE_CORRECTOR_DEFINITIONS_IN_MECHANICAL_ORDER
}
def create_adapter_lens():
    return ADAPTER_LENS_DEFINITION.create_component()


def resolve_probe_corrector_entrance_mechanical_axis(
    beam_deflector,
    adapter_lens,
    dph2_deflector,
):
    """Resolve the Beam Shift/Tilt -> ADL -> DPH2 sequence."""

    if beam_deflector.key != BEAM_DEFLECTOR:
        raise ValueError(
            "Probe Corrector entrance requires Beam Shift/Tilt."
        )
    if adapter_lens.key != ADAPTER_LENS:
        raise ValueError("Probe Corrector entrance requires Adapter Lens.")
    if dph2_deflector.key != PROBE_DPH2_DEFLECTOR:
        raise ValueError("Probe Corrector entrance requires DPH2.")
    return resolve_mechanical_axis(
        (beam_deflector, adapter_lens, dph2_deflector),
        (
            BEAM_DEFLECTOR,
            ADAPTER_LENS,
            PROBE_DPH2_DEFLECTOR,
        ),
    )


def resolve_probe_corrector_upper_triplet_mechanical_axis(
    dph2_deflector,
    qph2_quadrupole,
    hp2_hexapole,
    tl22_lens,
    dp22_deflector,
):
    """Resolve QPH2, HP2 and TL22 with their two boundary neighbours."""

    components = (
        dph2_deflector,
        qph2_quadrupole,
        hp2_hexapole,
        tl22_lens,
        dp22_deflector,
    )
    expected = (
        PROBE_DPH2_DEFLECTOR,
        PROBE_QPH2_QUADRUPOLE,
        PROBE_HP2_HEXAPOLE,
        PROBE_TL22_LENS,
        PROBE_DP22_DEFLECTOR,
    )
    if tuple(component.key for component in components) != expected:
        raise ValueError(
            "Probe Corrector upper triplet requires canonical physical order."
        )
    return resolve_mechanical_axis(components, expected)


def resolve_probe_corrector_second_triplet_mechanical_axis(
    tl22_lens,
    dp22_deflector,
    hpc_hexapole,
    qpc_quadrupole,
    dp21_deflector,
):
    """Resolve DP22, HPC and QPC with their two boundary neighbours."""

    components = (
        tl22_lens,
        dp22_deflector,
        hpc_hexapole,
        qpc_quadrupole,
        dp21_deflector,
    )
    expected = (
        PROBE_TL22_LENS,
        PROBE_DP22_DEFLECTOR,
        PROBE_HPC_HEXAPOLE,
        PROBE_QPC_QUADRUPOLE,
        PROBE_DP21_DEFLECTOR,
    )
    if tuple(component.key for component in components) != expected:
        raise ValueError(
            "Probe Corrector second triplet requires canonical physical order."
        )
    return resolve_mechanical_axis(components, expected)


def resolve_probe_corrector_lower_transfer_mechanical_axis(
    qpc_quadrupole,
    dp21_deflector,
    tl21_lens,
    dph1_deflector,
    qph1_quadrupole,
):
    """Resolve DP21, TL21 and DPH1 with their two boundary neighbours."""

    components = (
        qpc_quadrupole,
        dp21_deflector,
        tl21_lens,
        dph1_deflector,
        qph1_quadrupole,
    )
    expected = (
        PROBE_QPC_QUADRUPOLE,
        PROBE_DP21_DEFLECTOR,
        PROBE_TL21_LENS,
        PROBE_DPH1_DEFLECTOR,
        PROBE_QPH1_QUADRUPOLE,
    )
    if tuple(component.key for component in components) != expected:
        raise ValueError(
            "Probe Corrector lower transfer requires canonical physical order."
        )
    return resolve_mechanical_axis(components, expected)


def resolve_probe_corrector_lower_triplet_mechanical_axis(
    dph1_deflector,
    qph1_quadrupole,
    hp1_hexapole,
    hpol_hexapole,
    qpol_quadrupole,
):
    """Resolve QPH1, HP1 and HPol with their two boundary neighbours."""

    components = (
        dph1_deflector,
        qph1_quadrupole,
        hp1_hexapole,
        hpol_hexapole,
        qpol_quadrupole,
    )
    expected = (
        PROBE_DPH1_DEFLECTOR,
        PROBE_QPH1_QUADRUPOLE,
        PROBE_HP1_HEXAPOLE,
        PROBE_HPOL_HEXAPOLE,
        PROBE_QPOL_QUADRUPOLE,
    )
    if tuple(component.key for component in components) != expected:
        raise ValueError(
            "Probe Corrector lower triplet requires canonical physical order."
        )
    return resolve_mechanical_axis(components, expected)


def resolve_probe_corrector_exit_mechanical_axis(
    hpol_hexapole,
    qpol_quadrupole,
    dp11_deflector,
    tl12_lens,
    dp12_scan_deflector,
):
    """Resolve QPol, DP11 and TL12 with their two boundary neighbours."""

    components = (
        hpol_hexapole,
        qpol_quadrupole,
        dp11_deflector,
        tl12_lens,
        dp12_scan_deflector,
    )
    expected_with_virtual = (
        PROBE_HPOL_HEXAPOLE,
        PROBE_QPOL_QUADRUPOLE,
        PROBE_DP11_DEFLECTOR,
        PROBE_TL12_LENS,
        PROBE_DP12_SCAN_DEFLECTOR,
    )
    if (
        tuple(component.key for component in components)
        != expected_with_virtual
    ):
        raise ValueError(
            "Probe Corrector exit requires canonical physical order."
        )
    physical_components = components[:-1]
    return resolve_mechanical_axis(
        physical_components,
        expected_with_virtual[:-1],
    )


@dataclass(frozen=True)
class _MechanicalAxisPart:
    key: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float


def resolve_probe_corrector_scan_deflector_mechanical_axis(
    tl12_lens,
    dp12_scan_deflector,
    condenser_stigmator,
):
    """Resolve the probe-corrector exit into the upper-pole package."""

    components = (
        tl12_lens,
        dp12_scan_deflector,
        _MechanicalAxisPart(
            key=condenser_stigmator.key,
            mechanical_center_from_tip_mm=float(
                condenser_stigmator.z_mm
            ),
            mechanical_length_mm=float(
                condenser_stigmator.mechanical_length_mm
            ),
        ),
    )
    expected = (
        PROBE_TL12_LENS,
        PROBE_DP12_SCAN_DEFLECTOR,
        CONDENSER_STIGMATOR,
    )
    if tuple(component.key for component in components) != expected:
        raise ValueError(
            "Probe Corrector scan deflectors require canonical physical order."
        )
    return resolve_mechanical_axis(components, expected)


def adapter_lens_from_dict(data):
    values = dict(data)
    values["key"] = canonical_lens_key(values.get("key", ""))
    component = create_adapter_lens()
    allowed = AdapterLensComponent.__dataclass_fields__
    object.__setattr__(component, "_position_coupling_ready", False)
    for attribute, value in values.items():
        if attribute == "gaussian":
            value = [
                term
                if isinstance(term, AxialFieldTerm)
                else AxialFieldTerm(**term)
                for term in value
            ]
        if attribute in allowed:
            object.__setattr__(component, attribute, value)
    if "optical_reference_from_tip_mm" not in values:
        object.__setattr__(
            component,
            "optical_reference_from_tip_mm",
            float(values.get("z_mm", component.z_mm)),
        )
    object.__setattr__(component, "key", ADAPTER_LENS)
    object.__setattr__(component, "_position_coupling_ready", True)
    return component.apply_optical_position().validate()


def create_dph2_deflector():
    return DPH2_DEFLECTOR_DEFINITION.create_component()


def dph2_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_dph2_deflector()
    restored = restore_single_plane_deflector(component, values)
    restored.key = PROBE_DPH2_DEFLECTOR
    return restored.validate()


def create_dp22_deflector():
    return DP22_DEFLECTOR_DEFINITION.create_component()


def dp22_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_dp22_deflector()
    restored = restore_single_plane_deflector(component, values)
    restored.key = PROBE_DP22_DEFLECTOR
    return restored.validate()


def create_qph2_quadrupole():
    return QPH2_QUADRUPOLE_DEFINITION.create_component()


def qph2_quadrupole_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_qph2_quadrupole()
    restored = restore_quadrupole(component, values)
    restored.key = PROBE_QPH2_QUADRUPOLE
    return restored.validate()


def create_hp2_hexapole():
    return HP2_HEXAPOLE_DEFINITION.create_component()


def hp2_hexapole_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_hp2_hexapole()
    restored = restore_hexapole(component, values)
    restored.key = PROBE_HP2_HEXAPOLE
    return restored.validate()


def create_hpc_hexapole():
    return HPC_HEXAPOLE_DEFINITION.create_component()


def hpc_hexapole_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_hpc_hexapole()
    restored = restore_hexapole(component, values)
    restored.key = PROBE_HPC_HEXAPOLE
    return restored.validate()


def _restore_single_deflector(factory, canonical_key, data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    restored = restore_single_plane_deflector(factory(), values)
    restored.key = canonical_key
    return restored.validate()


def _restore_quadrupole(factory, canonical_key, data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    restored = restore_quadrupole(factory(), values)
    restored.key = canonical_key
    return restored.validate()


def _restore_hexapole(factory, canonical_key, data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    restored = restore_hexapole(factory(), values)
    restored.key = canonical_key
    return restored.validate()


def _restore_probe_round_lens(factory, canonical_key, data):
    values = dict(data)
    values["key"] = canonical_lens_key(values.get("key", ""))
    restored = restore_round_lens(factory(), values)
    restored.key = canonical_key
    return restored.validate()


def create_qpc_quadrupole():
    return QPC_QUADRUPOLE_DEFINITION.create_component()


def qpc_quadrupole_from_dict(data):
    return _restore_quadrupole(
        create_qpc_quadrupole, PROBE_QPC_QUADRUPOLE, data
    )


def create_dp21_deflector():
    return DP21_DEFLECTOR_DEFINITION.create_component()


def dp21_deflector_from_dict(data):
    return _restore_single_deflector(
        create_dp21_deflector, PROBE_DP21_DEFLECTOR, data
    )


def create_tl21_lens():
    return TL21_LENS_DEFINITION.create_component()


def tl21_lens_from_dict(data):
    return _restore_probe_round_lens(
        create_tl21_lens, PROBE_TL21_LENS, data
    )


def create_dph1_deflector():
    return DPH1_DEFLECTOR_DEFINITION.create_component()


def dph1_deflector_from_dict(data):
    return _restore_single_deflector(
        create_dph1_deflector, PROBE_DPH1_DEFLECTOR, data
    )


def create_qph1_quadrupole():
    return QPH1_QUADRUPOLE_DEFINITION.create_component()


def qph1_quadrupole_from_dict(data):
    return _restore_quadrupole(
        create_qph1_quadrupole, PROBE_QPH1_QUADRUPOLE, data
    )


def create_hp1_hexapole():
    return HP1_HEXAPOLE_DEFINITION.create_component()


def hp1_hexapole_from_dict(data):
    return _restore_hexapole(
        create_hp1_hexapole, PROBE_HP1_HEXAPOLE, data
    )


def create_hpol_hexapole():
    return HPOL_HEXAPOLE_DEFINITION.create_component()


def hpol_hexapole_from_dict(data):
    return _restore_hexapole(
        create_hpol_hexapole, PROBE_HPOL_HEXAPOLE, data
    )


def create_qpol_quadrupole():
    return QPOL_QUADRUPOLE_DEFINITION.create_component()


def qpol_quadrupole_from_dict(data):
    return _restore_quadrupole(
        create_qpol_quadrupole, PROBE_QPOL_QUADRUPOLE, data
    )


def create_dp11_deflector():
    return DP11_DEFLECTOR_DEFINITION.create_component()


def dp11_deflector_from_dict(data):
    return _restore_single_deflector(
        create_dp11_deflector, PROBE_DP11_DEFLECTOR, data
    )


def create_tl12_lens():
    return TL12_LENS_DEFINITION.create_component()


def tl12_lens_from_dict(data):
    return _restore_probe_round_lens(
        create_tl12_lens, PROBE_TL12_LENS, data
    )


def create_dp12_scan_deflector():
    return DP12_SCAN_DEFLECTOR_DEFINITION.create_component()


def dp12_scan_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        canonical_deflector_key(values.get("key", ""))
    )
    component = create_dp12_scan_deflector()
    default_center_mm = float(
        component.optical_upper_reference_from_tip_mm
    )
    optical_center_mm = 0.5 * (
        float(values.get(
            "optical_upper_reference_from_tip_mm",
            values.get("upper_z_mm", default_center_mm),
        ))
        + float(values.get(
            "optical_lower_reference_from_tip_mm",
            values.get("lower_z_mm", default_center_mm),
        ))
    )
    values.update({
        "name": DP12_SCAN_DEFLECTOR_DEFINITION.label,
        "mechanical_length_mm": _probe_manifest_value(
            PROBE_DP12_SCAN_DEFLECTOR, "length_mm"
        ),
        "thickness_mm": _probe_manifest_value(
            PROBE_DP12_SCAN_DEFLECTOR, "effective_thickness_mm"
        ),
        "optical_upper_reference_from_tip_mm": optical_center_mm,
        "optical_lower_reference_from_tip_mm": optical_center_mm,
        "upper_z_mm": optical_center_mm,
        "lower_z_mm": optical_center_mm,
        "upper_x_mrad": 0.0,
        "upper_y_mrad": 0.0,
        "lower_x_mrad": 0.0,
        "lower_y_mrad": 0.0,
    })
    restored = restore_paired_deflector(component, values)
    restored.key = PROBE_DP12_SCAN_DEFLECTOR
    return restored.validate()


def create_tl22_lens():
    return TL22_LENS_DEFINITION.create_component()


def tl22_lens_from_dict(data):
    values = dict(data)
    values["key"] = canonical_lens_key(values.get("key", ""))
    component = create_tl22_lens()
    restored = restore_round_lens(component, values)
    restored.key = PROBE_TL22_LENS
    return restored.validate()


class ProbeCorrectorSystem:
    """State-backed entry point for the progressively modularised assembly."""

    def __init__(self, state):
        self.state = state

    @property
    def adapter_lens(self):
        return self.state.adapter_lens

    @property
    def dph2_deflector(self):
        return self.state.dph2_deflector

    @property
    def deflector_components(self):
        return (
            self.dph2_deflector,
            self.dp22_deflector,
            self.dp21_deflector,
            self.dph1_deflector,
            self.dp11_deflector,
        )

    @property
    def dp22_deflector(self):
        return self.state.dp22_deflector

    @property
    def dp21_deflector(self):
        return self.state.dp21_deflector

    @property
    def dph1_deflector(self):
        return self.state.dph1_deflector

    @property
    def dp11_deflector(self):
        return self.state.dp11_deflector

    @property
    def dp12_scan_deflector(self):
        return self.state.dp12_scan_deflector

    @property
    def paired_deflector_components(self):
        return ()

    @property
    def qph2_quadrupole(self):
        return self.state.qph2_quadrupole

    @property
    def quadrupole_components(self):
        return (
            self.qph2_quadrupole,
            self.qpc_quadrupole,
            self.qph1_quadrupole,
            self.qpol_quadrupole,
        )

    @property
    def qpc_quadrupole(self):
        return self.state.qpc_quadrupole

    @property
    def qph1_quadrupole(self):
        return self.state.qph1_quadrupole

    @property
    def qpol_quadrupole(self):
        return self.state.qpol_quadrupole

    @property
    def hp2_hexapole(self):
        return self.state.hp2_hexapole

    @property
    def hpc_hexapole(self):
        return self.state.hpc_hexapole

    @property
    def hp1_hexapole(self):
        return self.state.hp1_hexapole

    @property
    def hpol_hexapole(self):
        return self.state.hpol_hexapole

    @property
    def hexapole_components(self):
        return (
            self.hp2_hexapole,
            self.hpc_hexapole,
            self.hp1_hexapole,
            self.hpol_hexapole,
        )

    @property
    def tl22_lens(self):
        return self.state.tl22_lens

    @property
    def tl21_lens(self):
        return self.state.tl21_lens

    @property
    def tl12_lens(self):
        return self.state.tl12_lens

    @property
    def round_lens_components(self):
        return (
            self.adapter_lens,
            self.tl22_lens,
            self.tl21_lens,
            self.tl12_lens,
        )

    @property
    def components(self):
        return (
            self.adapter_lens,
            self.dph2_deflector,
            self.qph2_quadrupole,
            self.hp2_hexapole,
            self.tl22_lens,
            self.dp22_deflector,
            self.hpc_hexapole,
            self.qpc_quadrupole,
            self.dp21_deflector,
            self.tl21_lens,
            self.dph1_deflector,
            self.qph1_quadrupole,
            self.hp1_hexapole,
            self.hpol_hexapole,
            self.qpol_quadrupole,
            self.dp11_deflector,
            self.tl12_lens,
            self.dp12_scan_deflector,
        )

    def validate(self):
        for component in self.components:
            component.validate()
        return self

    def _sync_mechanical_anchors(self):
        from temsim.optics.beam_deflector import (
            resolve_beam_deflector_after_active_aperture,
        )

        resolve_beam_deflector_after_active_aperture(self.state)
        anchor_probe_corrector_to_beam_deflector(self.state)
        synchronise_probe_corrector_physical_axis(self.state)

    def resolve_complete_mechanical_axis(self):
        self._sync_mechanical_anchors()
        physical_components = tuple(
            component
            for component in self.components
            if component.key != PROBE_DP12_SCAN_DEFLECTOR
        )
        return resolve_mechanical_axis(
            physical_components,
            tuple(
                component.key for component in physical_components
            ),
        )

    def resolve_entrance_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_entrance_mechanical_axis(
            self.state.beam_deflector,
            self.adapter_lens,
            self.dph2_deflector,
        )

    def resolve_upper_triplet_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_upper_triplet_mechanical_axis(
            self.dph2_deflector,
            self.qph2_quadrupole,
            self.hp2_hexapole,
            self.tl22_lens,
            self.dp22_deflector,
        )

    def resolve_second_triplet_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_second_triplet_mechanical_axis(
            self.tl22_lens,
            self.dp22_deflector,
            self.hpc_hexapole,
            self.qpc_quadrupole,
            self.dp21_deflector,
        )

    def resolve_lower_transfer_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_lower_transfer_mechanical_axis(
            self.qpc_quadrupole,
            self.dp21_deflector,
            self.tl21_lens,
            self.dph1_deflector,
            self.qph1_quadrupole,
        )

    def resolve_lower_triplet_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_lower_triplet_mechanical_axis(
            self.dph1_deflector,
            self.qph1_quadrupole,
            self.hp1_hexapole,
            self.hpol_hexapole,
            self.qpol_quadrupole,
        )

    def resolve_exit_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_exit_mechanical_axis(
            self.hpol_hexapole,
            self.qpol_quadrupole,
            self.dp11_deflector,
            self.tl12_lens,
            self.dp12_scan_deflector,
        )

    def resolve_scan_deflector_mechanical_axis(self):
        self._sync_mechanical_anchors()
        return resolve_probe_corrector_scan_deflector_mechanical_axis(
            self.tl12_lens,
            self.dp12_scan_deflector,
            self.state.condenser_stigmator,
        )

    def apply_optical_positions(self):
        for component in self.components:
            if hasattr(component, "apply_optical_positions"):
                component.apply_optical_positions()
            else:
                component.apply_optical_position()
        return self
def anchor_probe_corrector_to_beam_deflector(state):
    """Compatibility hook; selected TOML owns every corrector anchor."""

    return state.probe_corrector_system.components


def synchronise_probe_corrector_physical_axis(state):
    """Compatibility hook; selected TOML owns optical reference positions."""

    return state.probe_corrector_system.components
