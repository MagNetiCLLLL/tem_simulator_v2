"""Coupled objective field with separate pole-piece mechanics and optical planes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import OBJECTIVE_LENS
from temsim.optics.condenser_lens import AxialFieldTerm
from temsim.optics.lens_focal_length import electron_momentum


ELECTRON_CHARGE_C = 1.602176634e-19

_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml",
    "exit",
)


def _profile_terms(part, prefix):
    amplitudes = tuple(part[f"{prefix}_field_profile_amplitudes"])
    offsets = tuple(part[f"{prefix}_field_profile_offsets"])
    sigmas = tuple(part[f"{prefix}_field_profile_sigmas"])
    if not (len(amplitudes) == len(offsets) == len(sigmas)):
        raise ValueError(f"Invalid Objective {prefix} field profile")
    return [
        AxialFieldTerm(float(amplitude), float(offset), float(sigma))
        for amplitude, offset, sigma in zip(amplitudes, offsets, sigmas)
    ]


@dataclass(frozen=True)
class ObjectiveLensDefinition:
    key: str
    label: str
    assembly_length_mm: float
    assembly_outer_diameter_mm: float
    pole_piece_center_separation_mm: float
    pole_piece_axial_length_mm: float
    upper_pole_piece_axial_length_mm: float
    pole_piece_outer_diameter_mm: float
    pole_piece_tip_diameter_mm: float
    upper_pole_piece_outer_diameter_mm: float
    upper_pole_piece_tip_diameter_mm: float
    pole_piece_bore_diameter_mm: float
    inner_face_gap_mm: float


@dataclass
class ObjectiveLensComponent:
    name: str
    key: str
    z_mm: float
    percent: float
    max_percent: float
    enabled: bool
    colour: str
    upper_b0_t: float
    lower_b0_t: float
    upper_a_mm: float
    lower_a_mm: float
    upper_gaussian: list[AxialFieldTerm]
    lower_gaussian: list[AxialFieldTerm]
    upper_field_center_z_mm: float
    lower_field_center_z_mm: float
    upper_field_center_above_sample_mm: float
    upper_pole_piece_center_z_mm: float
    lower_pole_piece_center_z_mm: float
    upper_objective_lens_center_z_mm: float
    lower_objective_lens_center_z_mm: float
    upper_objective_lens_axial_length_mm: float
    lower_objective_lens_axial_length_mm: float
    virtual_lens_reference_z_mm: float
    assembly_length_mm: float
    assembly_outer_diameter_mm: float
    pole_piece_center_separation_mm: float
    pole_piece_axial_length_mm: float
    upper_pole_piece_axial_length_mm: float
    pole_piece_outer_diameter_mm: float
    pole_piece_tip_diameter_mm: float
    upper_pole_piece_outer_diameter_mm: float
    upper_pole_piece_tip_diameter_mm: float
    pole_piece_bore_diameter_mm: float
    inner_face_gap_mm: float
    sample_axial_offset_mm: float
    virtual_lens_offset_below_lower_surface_mm: float
    nominal_voltage_kv: float
    nominal_focal_length_mm: float
    nominal_back_focal_plane_z_mm: float
    nominal_image_plane_z_mm: float
    cs_mm: float | None
    cc_mm: float | None
    polarity: int
    corrector: str

    EXPECTED_KEY: ClassVar[str] = OBJECTIVE_LENS
    KIND: ClassVar[str] = "round_lens"
    SHAPE_PROFILE: ClassVar[str] = "magnetic_lens_yoke"
    INTERACTION_KIND: ClassVar[str] = "coupled_distributed_axial_field"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "upper_field_center_z_mm",
            "lower_field_center_z_mm",
            "upper_pole_piece_center_z_mm",
            "lower_pole_piece_center_z_mm",
            "upper_objective_lens_center_z_mm",
            "lower_objective_lens_center_z_mm",
            "virtual_lens_reference_z_mm",
        }:
            value = float(value)
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return self.corrector

    @property
    def kind(self):
        return self.KIND

    @property
    def shape_profile(self):
        return self.SHAPE_PROFILE

    @property
    def interaction_kind(self):
        return self.INTERACTION_KIND

    @property
    def optical_active(self):
        return bool(self.enabled)

    @property
    def length_mm(self):
        return self.assembly_length_mm

    @property
    def b0_t(self):
        return self.upper_b0_t

    @b0_t.setter
    def b0_t(self, value):
        value = max(float(value), 0.0)
        old = max(float(self.upper_b0_t), 1e-15)
        ratio = value / old
        self.upper_b0_t = value
        self.lower_b0_t = float(self.lower_b0_t) * ratio

    @property
    def a_mm(self):
        return self.upper_a_mm

    @property
    def gaussian(self):
        return self.upper_gaussian

    def scale(self):
        return (
            self.upper_b0_t * self.percent / 100.0
            if self.enabled else 0.0
        )

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError("Objective Lens key is not canonical.")
        if self.assembly_length_mm <= 0.0:
            raise ValueError("Objective assembly length must be positive.")
        if self.assembly_outer_diameter_mm <= 0.0:
            raise ValueError("Objective assembly diameter must be positive.")
        if self.pole_piece_center_separation_mm <= 0.0:
            raise ValueError("Pole-piece centre separation must be positive.")
        if self.pole_piece_axial_length_mm <= 0.0:
            raise ValueError("Lower pole-piece axial length must be positive.")
        if self.upper_pole_piece_axial_length_mm <= 0.0:
            raise ValueError("Upper pole-piece axial length must be positive.")
        if min(
            self.upper_objective_lens_axial_length_mm,
            self.lower_objective_lens_axial_length_mm,
        ) <= 0.0:
            raise ValueError(
                "Upper and Lower Objective Lens lengths must be positive."
            )
        if not math.isclose(
            self.upper_objective_lens_axial_length_mm,
            self.lower_objective_lens_axial_length_mm,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Lower Objective Lens must match the Upper Objective Lens "
                "axial thickness."
            )
        if self.upper_field_center_above_sample_mm <= 0.0:
            raise ValueError(
                "Upper Objective field centre must remain above the sample."
            )
        expected_gap = (
            self.pole_piece_center_separation_mm
            - 0.5
            * (
                self.upper_pole_piece_axial_length_mm
                + self.pole_piece_axial_length_mm
            )
        )
        if not math.isclose(
            expected_gap, self.inner_face_gap_mm, abs_tol=1e-9
        ):
            raise ValueError(
                "Pole-piece centres and lengths must produce the configured "
                "flat-face gap."
            )
        if self.inner_face_gap_mm <= 0.0:
            raise ValueError("Objective pole-tip gap must be positive.")
        lower_diameters_valid = (
            0.0
            < self.pole_piece_bore_diameter_mm
            < self.pole_piece_tip_diameter_mm
            < self.pole_piece_outer_diameter_mm
        )
        upper_diameters_valid = (
            0.0
            < self.pole_piece_bore_diameter_mm
            < self.upper_pole_piece_tip_diameter_mm
            < self.upper_pole_piece_outer_diameter_mm
        )
        if not (lower_diameters_valid and upper_diameters_valid):
            raise ValueError(
                "Objective pole bore, flat tip and body diameters are invalid."
            )
        if not 0.0 < self.max_percent <= 100.0:
            raise ValueError(
                "Objective maximum excitation must lie in (0, 100]."
            )
        if not 0.0 <= self.percent <= self.max_percent:
            raise ValueError("Objective excitation exceeds its range.")
        if min(
            self.upper_b0_t,
            self.lower_b0_t,
            self.upper_a_mm,
            self.lower_a_mm,
        ) <= 0.0:
            raise ValueError("Objective field calibration must be positive.")
        for terms in (self.upper_gaussian, self.lower_gaussian):
            if not terms or any(term.sigma <= 0.0 for term in terms):
                raise ValueError("Objective field terms are invalid.")
        if self.nominal_voltage_kv <= 0.0:
            raise ValueError("Objective nominal voltage must be positive.")
        if self.nominal_focal_length_mm <= 0.0:
            raise ValueError("Objective nominal focal length must be positive.")
        if not (
            self.virtual_lens_reference_z_mm
            < self.nominal_back_focal_plane_z_mm
            < self.nominal_image_plane_z_mm
        ):
            raise ValueError(
                "Objective nominal reference planes must be ordered "
                "sample, BFP, image"
            )
        return self

    def sync_to_sample(self, sample):
        del sample
        # Mechanical and optical reference coordinates are manifest-owned.
        # This compatibility method deliberately performs no positioning.
        return self.validate()

    @staticmethod
    def _profile(z, center, half_width, terms):
        result = np.zeros_like(z, dtype=float)
        for term in terms:
            sigma = max(abs(term.sigma * half_width), 1e-12)
            term_center = center + term.offset * half_width
            result += term.amplitude * np.exp(
                -0.5 * ((z - term_center) / sigma) ** 2
            )
        return result

    def unit_excitation_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        upper = self.upper_b0_t * self._profile(
            z,
            self.upper_field_center_z_mm,
            self.upper_a_mm,
            self.upper_gaussian,
        )
        lower = self.lower_b0_t * self._profile(
            z,
            self.lower_field_center_z_mm,
            self.lower_a_mm,
            self.lower_gaussian,
        )
        return float(self.polarity) * (upper + lower)

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        if not self.enabled:
            return np.zeros_like(z)
        return (
            float(self.percent) / 100.0
        ) * self.unit_excitation_field_t(z)

    def field_support_mm(self, sigma_cutoff=7.0):
        upper_reach = max(
            abs(term.offset * self.upper_a_mm)
            + sigma_cutoff * abs(term.sigma * self.upper_a_mm)
            for term in self.upper_gaussian
        )
        lower_reach = max(
            abs(term.offset * self.lower_a_mm)
            + sigma_cutoff * abs(term.sigma * self.lower_a_mm)
            for term in self.lower_gaussian
        )
        return (
            self.upper_field_center_z_mm - upper_reach,
            self.lower_field_center_z_mm + lower_reach,
        )

    def unit_field_integral_t2_m(self, samples=8001):
        start, end = self.field_support_mm()
        z_mm = np.linspace(start, end, int(samples))
        field_t = self.unit_excitation_field_t(z_mm)
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is not None:
            return float(trapezoid(field_t * field_t, z_mm * 1e-3))
        return float(np.trapz(field_t * field_t, z_mm * 1e-3))

    def focal_length_for_voltage_mm(self, voltage_kv):
        if not self.enabled or self.percent == 0.0:
            return math.inf
        momentum = electron_momentum(voltage_kv)
        power = (
            ELECTRON_CHARGE_C / (2.0 * momentum)
        ) ** 2 * (self.percent / 100.0) ** 2 * (
            self.unit_field_integral_t2_m()
        )
        return math.inf if power <= 0.0 else 1000.0 / power

    def set_focal_length_for_voltage_mm(self, voltage_kv, target_focal_mm):
        target_m = float(target_focal_mm) * 1e-3
        if not math.isfinite(target_m) or target_m <= 0.0:
            raise ValueError("Focal length must be positive and finite.")
        momentum = electron_momentum(voltage_kv)
        coefficient = (
            ELECTRON_CHARGE_C / (2.0 * momentum)
        ) ** 2 * self.unit_field_integral_t2_m()
        required_percent = 100.0 * math.sqrt(
            1.0 / (target_m * coefficient)
        )
        if required_percent <= self.max_percent:
            self.percent = required_percent
            return "percentage"
        ratio = required_percent / max(abs(self.percent), 1e-12)
        self.b0_t = self.b0_t * ratio
        return "maximum field"

    def transfer_matrix(
        self,
        voltage_kv,
        start_z_mm,
        end_z_mm,
        step_mm=0.05,
    ):
        """Distributed-field paraxial matrix between two axial planes."""

        momentum = electron_momentum(voltage_kv)
        matrix = np.eye(2, dtype=float)
        z = float(start_z_mm)
        while z < end_z_mm - 1e-12:
            h_mm = min(float(step_mm), end_z_mm - z)
            midpoint = z + h_mm / 2.0
            field_t = float(self.magnetic_field_t(np.array([midpoint]))[0])
            k_m2 = (
                ELECTRON_CHARGE_C * field_t / (2.0 * momentum)
            ) ** 2
            h_m = h_mm * 1e-3
            if k_m2 > 1e-24:
                root_k = math.sqrt(k_m2)
                phase = root_k * h_m
                cosine = math.cos(phase)
                sine = math.sin(phase)
                step_matrix = np.array((
                    (cosine, sine / root_k),
                    (-root_k * sine, cosine),
                ))
            else:
                step_matrix = np.array(((1.0, h_m), (0.0, 1.0)))
            matrix = step_matrix @ matrix
            z += h_mm
        return matrix

    def _first_matrix_zero(
        self,
        voltage_kv,
        start_z_mm,
        end_z_mm,
        matrix_index,
        step_mm,
    ):
        matrix = np.eye(2, dtype=float)
        z_mm = float(start_z_mm)
        previous_value = float(matrix[matrix_index])
        previous_z_mm = z_mm
        left_start_plane = False
        while z_mm < end_z_mm - 1e-12:
            next_z_mm = min(z_mm + float(step_mm), end_z_mm)
            step_matrix = self.transfer_matrix(
                voltage_kv,
                z_mm,
                next_z_mm,
                step_mm=next_z_mm - z_mm,
            )
            matrix = step_matrix @ matrix
            z_mm = next_z_mm
            value = float(matrix[matrix_index])
            if not left_start_plane:
                left_start_plane = z_mm > start_z_mm + 1.0e-3
            elif previous_value * value <= 0.0:
                fraction = abs(previous_value) / max(
                    abs(previous_value) + abs(value),
                    1.0e-15,
                )
                return previous_z_mm + fraction * (
                    z_mm - previous_z_mm
                )
            previous_value = value
            previous_z_mm = z_mm
        return None

    def back_focal_plane_z_mm(self, voltage_kv, sample):
        """First post-specimen A=0 Fourier plane of the distributed field."""

        start_z_mm = (
            float(sample.z_mm) + float(sample.thickness_nm) * 0.5e-6
        )
        end_z_mm = float(sample.z_mm) + self.assembly_length_mm / 2.0
        return self._first_matrix_zero(
            voltage_kv,
            start_z_mm,
            end_z_mm,
            (0, 0),
            0.02,
        )

    def image_plane_z_mm(self, voltage_kv, sample, step_mm=0.1):
        """First post-specimen B=0 image plane of the distributed field."""

        start_z_mm = (
            float(sample.z_mm) + float(sample.thickness_nm) * 0.5e-6
        )
        end_z_mm = (
            float(sample.z_mm) + self.assembly_length_mm / 2.0
        )
        return self._first_matrix_zero(
            voltage_kv,
            start_z_mm,
            end_z_mm,
            (0, 1),
            step_mm,
        )

    def image_magnification(self, voltage_kv, sample, step_mm=0.1):
        image_z_mm = self.image_plane_z_mm(
            voltage_kv,
            sample,
            step_mm=step_mm,
        )
        if image_z_mm is None:
            return None
        start_z_mm = (
            float(sample.z_mm) + float(sample.thickness_nm) * 0.5e-6
        )
        return float(
            self.transfer_matrix(
                voltage_kv,
                start_z_mm,
                image_z_mm,
                step_mm=step_mm,
            )[0, 0]
        )

    def draw_layout(self):
        return {
            "key": self.key,
            "assembly_length_mm": self.assembly_length_mm,
            "assembly_outer_diameter_mm": (
                self.assembly_outer_diameter_mm
            ),
            "pole_piece_center_separation_mm": (
                self.pole_piece_center_separation_mm
            ),
            "pole_piece_axial_length_mm": (
                self.pole_piece_axial_length_mm
            ),
            "upper_pole_piece_axial_length_mm": (
                self.upper_pole_piece_axial_length_mm
            ),
            "upper_pole_piece_center_z_mm": (
                self.upper_pole_piece_center_z_mm
            ),
            "lower_pole_piece_center_z_mm": (
                self.lower_pole_piece_center_z_mm
            ),
            "upper_objective_lens_center_z_mm": (
                self.upper_objective_lens_center_z_mm
            ),
            "lower_objective_lens_center_z_mm": (
                self.lower_objective_lens_center_z_mm
            ),
            "upper_objective_lens_axial_length_mm": (
                self.upper_objective_lens_axial_length_mm
            ),
            "lower_objective_lens_axial_length_mm": (
                self.lower_objective_lens_axial_length_mm
            ),
            "upper_field_center_above_sample_mm": (
                self.upper_field_center_above_sample_mm
            ),
            "upper_pole_piece_outer_diameter_mm": (
                self.upper_pole_piece_outer_diameter_mm
            ),
            "upper_pole_piece_tip_diameter_mm": (
                self.upper_pole_piece_tip_diameter_mm
            ),
            "inner_face_gap_mm": self.inner_face_gap_mm,
            "pole_piece_profile": "flat_tip_cone",
        }

    def draw_ray_overlay(self, voltage_kv=300.0, sample=None):
        start, end = self.field_support_mm()
        result = {
            "key": self.key,
            "virtual_lens_z_mm": self.z_mm,
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_for_voltage_mm(
                voltage_kv
            ),
            "enabled": self.enabled,
        }
        if sample is not None:
            result["back_focal_plane_z_mm"] = (
                self.back_focal_plane_z_mm(voltage_kv, sample)
            )
            result["image_plane_z_mm"] = self.image_plane_z_mm(
                voltage_kv, sample
            )
        return result

@dataclass(frozen=True)
class _ReferenceSample:
    z_mm: float
    thickness_nm: float


def _component_from_manifest(
    module_path=_DEFAULT_OBJECTIVE_MODULE_PATH,
    *,
    root=None,
    column_origin_z_mm=_DEFAULT_COLUMN_ORIGIN_Z_MM,
):
    document = module_manifest.read_document(
        (module_manifest.MODULE_ROOT if root is None else root)
        / module_path
    )
    parts = {
        str(part["key"]): dict(part)
        for part in document["parts"]
    }
    part = parts[OBJECTIVE_LENS]
    sample = parts["sample"]
    upper_pole = parts["objective_upper_pole"]
    lower_pole = parts["objective_lower_pole"]
    origin = float(column_origin_z_mm)

    def absolute(value):
        return origin + float(value)

    upper_yoke_start = absolute(part["upper_yoke_start_local_z_mm"])
    upper_yoke_end = absolute(part["upper_yoke_end_local_z_mm"])
    lower_yoke_start = absolute(part["lower_yoke_start_local_z_mm"])
    lower_yoke_end = absolute(part["lower_yoke_end_local_z_mm"])
    sample_z_mm = absolute(sample["local_center_z_mm"])
    virtual_z_mm = absolute(part["virtual_reference_local_z_mm"])
    pole_gap_center_z_mm = 0.5 * (
        absolute(upper_pole["local_end_z_mm"])
        + absolute(lower_pole["local_start_z_mm"])
    )
    return ObjectiveLensComponent(
        name=str(part["name"]),
        key=OBJECTIVE_LENS,
        z_mm=virtual_z_mm,
        percent=float(part["nominal_excitation_percent"]),
        max_percent=float(part["maximum_excitation_percent"]),
        enabled=True,
        colour="#d32f2f",
        upper_b0_t=float(part["upper_peak_field_t"]),
        lower_b0_t=float(part["lower_peak_field_t"]),
        upper_a_mm=float(part["upper_field_half_width_mm"]),
        lower_a_mm=float(part["lower_field_half_width_mm"]),
        upper_gaussian=_profile_terms(part, "upper"),
        lower_gaussian=_profile_terms(part, "lower"),
        upper_field_center_z_mm=absolute(
            part["upper_field_reference_local_z_mm"]
        ),
        lower_field_center_z_mm=absolute(
            part["lower_field_reference_local_z_mm"]
        ),
        upper_field_center_above_sample_mm=(
            sample_z_mm
            - absolute(part["upper_field_reference_local_z_mm"])
        ),
        upper_pole_piece_center_z_mm=absolute(
            upper_pole["local_center_z_mm"]
        ),
        lower_pole_piece_center_z_mm=absolute(
            lower_pole["local_center_z_mm"]
        ),
        upper_objective_lens_center_z_mm=0.5 * (
            upper_yoke_start + upper_yoke_end
        ),
        lower_objective_lens_center_z_mm=0.5 * (
            lower_yoke_start + lower_yoke_end
        ),
        upper_objective_lens_axial_length_mm=(
            upper_yoke_end - upper_yoke_start
        ),
        lower_objective_lens_axial_length_mm=(
            lower_yoke_end - lower_yoke_start
        ),
        virtual_lens_reference_z_mm=virtual_z_mm,
        assembly_length_mm=float(part["length_mm"]),
        assembly_outer_diameter_mm=float(
            part["mechanical_outer_diameter_mm"]
        ),
        pole_piece_center_separation_mm=(
            float(lower_pole["local_center_z_mm"])
            - float(upper_pole["local_center_z_mm"])
        ),
        pole_piece_axial_length_mm=float(lower_pole["length_mm"]),
        upper_pole_piece_axial_length_mm=float(upper_pole["length_mm"]),
        pole_piece_outer_diameter_mm=float(
            lower_pole["mechanical_outer_diameter_mm"]
        ),
        pole_piece_tip_diameter_mm=float(
            lower_pole["mechanical_tip_diameter_mm"]
        ),
        upper_pole_piece_outer_diameter_mm=float(
            upper_pole["mechanical_outer_diameter_mm"]
        ),
        upper_pole_piece_tip_diameter_mm=float(
            upper_pole["mechanical_tip_diameter_mm"]
        ),
        pole_piece_bore_diameter_mm=float(
            lower_pole["mechanical_bore_diameter_mm"]
        ),
        inner_face_gap_mm=float(part["s_twin_pole_gap_mm"]),
        sample_axial_offset_mm=sample_z_mm - pole_gap_center_z_mm,
        virtual_lens_offset_below_lower_surface_mm=(
            virtual_z_mm - sample_z_mm
        ),
        nominal_voltage_kv=float(part["nominal_voltage_kv"]),
        nominal_focal_length_mm=float(part["nominal_focal_length_mm"]),
        nominal_back_focal_plane_z_mm=absolute(
            part["nominal_back_focal_plane_local_z_mm"]
        ),
        nominal_image_plane_z_mm=absolute(
            part["nominal_image_plane_local_z_mm"]
        ),
        cs_mm=float(part["spherical_aberration_mm"]),
        cc_mm=float(part["chromatic_aberration_mm"]),
        polarity=int(part["polarity"]),
        corrector="objective",
    ).validate()


_DEFAULT_OBJECTIVE_COMPONENT = _component_from_manifest()
OBJECTIVE_LENS_DEFINITION = ObjectiveLensDefinition(
    key=_DEFAULT_OBJECTIVE_COMPONENT.key,
    label=_DEFAULT_OBJECTIVE_COMPONENT.name,
    assembly_length_mm=_DEFAULT_OBJECTIVE_COMPONENT.assembly_length_mm,
    assembly_outer_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.assembly_outer_diameter_mm
    ),
    pole_piece_center_separation_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.pole_piece_center_separation_mm
    ),
    pole_piece_axial_length_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.pole_piece_axial_length_mm
    ),
    upper_pole_piece_axial_length_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.upper_pole_piece_axial_length_mm
    ),
    pole_piece_outer_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.pole_piece_outer_diameter_mm
    ),
    pole_piece_tip_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.pole_piece_tip_diameter_mm
    ),
    upper_pole_piece_outer_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.upper_pole_piece_outer_diameter_mm
    ),
    upper_pole_piece_tip_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.upper_pole_piece_tip_diameter_mm
    ),
    pole_piece_bore_diameter_mm=(
        _DEFAULT_OBJECTIVE_COMPONENT.pole_piece_bore_diameter_mm
    ),
    inner_face_gap_mm=_DEFAULT_OBJECTIVE_COMPONENT.inner_face_gap_mm,
)


def reference_objective_image_plane_z_mm(voltage_kv=300.0):
    """Return the default manifest's calculated Objective image plane."""

    component = _component_from_manifest()
    if math.isclose(
        float(voltage_kv),
        float(component.nominal_voltage_kv),
        abs_tol=1.0e-12,
    ):
        return float(component.nominal_image_plane_z_mm)
    sample_part = module_manifest.part_data(
        _DEFAULT_OBJECTIVE_MODULE_PATH,
        "sample",
    )
    sample = _ReferenceSample(
        _DEFAULT_COLUMN_ORIGIN_Z_MM
        + float(sample_part["local_center_z_mm"]),
        float(sample_part["length_mm"]) * 1.0e6,
    )
    return component.image_plane_z_mm(voltage_kv, sample)


def create_objective_lens(sample_z_mm=None, sample_thickness_nm=None):
    del sample_z_mm, sample_thickness_nm
    return _component_from_manifest()


def _field_terms(values, key, fallback):
    return [
        term
        if isinstance(term, AxialFieldTerm)
        else AxialFieldTerm(**term)
        for term in values.get(key, fallback)
    ]


def objective_lens_from_dict(data, sample_z_mm=None, sample_thickness_nm=None):
    del sample_z_mm, sample_thickness_nm
    values = dict(data)
    component = create_objective_lens()
    for attribute in (
        "percent", "enabled", "colour", "cs_mm", "cc_mm", "polarity"
    ):
        if attribute in values:
            object.__setattr__(component, attribute, values[attribute])
    object.__setattr__(component, "key", OBJECTIVE_LENS)
    object.__setattr__(component, "corrector", "objective")
    return component.validate()


def objective_lens_from_legacy_rows(
    upper,
    lower,
    sample_z_mm=None,
    sample_thickness_nm=None,
):
    component = create_objective_lens(sample_z_mm, sample_thickness_nm)
    upper = dict(upper or {})
    lower = dict(lower or {})
    if upper:
        component.percent = float(
            upper.get("percent", component.percent)
        )
        component.enabled = bool(
            upper.get("enabled", component.enabled)
        )
    if lower:
        component.enabled = component.enabled and bool(
            lower.get("enabled", True)
        )
    return component.validate()
