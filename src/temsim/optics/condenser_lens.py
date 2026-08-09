"""Single-source condenser-lens components and system.

Each condenser lens owns its mechanical geometry, optical reference plane,
default operating state, axial field model, validation, layout metadata and
ray-overlay data here.  The column factory only assembles the resulting lens
instances; renderers and solvers resolve the same state-backed components.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from temsim import module_manifest
from temsim.mechanical_axis import resolve_mechanical_axis
from temsim.component_keys import (
    C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    CONDENSER_DEFLECTOR,
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    CONDENSER_LENS_KEYS,
    THERMIONIC_C1_APERTURE,
)
from temsim.optics.lens_focal_length import focal_length_mm, unit_field_peak


@dataclass(frozen=True)
class GaussianTermDefinition:
    amplitude: float
    offset: float
    sigma: float


DEFAULT_GAUSSIAN_TERMS = (
    GaussianTermDefinition(0.09, -1.0, 0.90),
    GaussianTermDefinition(0.82, 0.0, 0.55),
    GaussianTermDefinition(0.09, 1.0, 0.90),
)


@dataclass
class AxialFieldTerm:
    amplitude: float
    offset: float
    sigma: float


@dataclass
class CondenserLensState:
    """Mutable operating state owned by a condenser component."""

    name: str
    key: str
    z_mm: float
    b0_t: float
    a_mm: float
    percent: float
    max_percent: float
    colour: str
    gaussian: list[AxialFieldTerm]
    enabled: bool = True
    cs_mm: float | None = None
    cc_mm: float | None = None
    polarity: int = 1
    normalise_profile_peak: bool = False

    def scale(self):
        return self.b0_t * self.percent / 100.0 if self.enabled else 0.0


@dataclass(frozen=True)
class CondenserLensDefinition:
    """Immutable defaults owned by one physical condenser-lens component."""

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
    gaussian_terms: tuple[GaussianTermDefinition, ...] = DEFAULT_GAUSSIAN_TERMS
    owner: str = "condenser_system"
    shape_profile: str = "magnetic_lens_yoke"
    field_kind: str = "axial_magnetic_field"
    effective_aperture_radius_mm: float | None = None
    enabled: bool = True
    spherical_aberration_mm: float | None = None
    chromatic_aberration_mm: float | None = None
    polarity: int = 1
    normalise_profile_peak: bool = False

    def create_lens(self) -> CondenserLensState:
        """Create the sole mutable runtime record for this component."""

        return CondenserLensState(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_from_tip_mm,
            b0_t=self.maximum_peak_field_t,
            a_mm=self.field_scale_half_width_mm,
            percent=self.default_excitation_percent,
            max_percent=self.maximum_excitation_percent,
            colour=self.colour,
            gaussian=[
                AxialFieldTerm(term.amplitude, term.offset, term.sigma)
                for term in self.gaussian_terms
            ],
            enabled=self.enabled,
            cs_mm=self.spherical_aberration_mm,
            cc_mm=self.chromatic_aberration_mm,
            polarity=self.polarity,
            normalise_profile_peak=self.normalise_profile_peak,
        )


_DEFAULT_COLUMN_MANIFEST = "column/C3_ProbeCorrector.toml"
_DEFAULT_GUN_MANIFEST = "gun/FEG.toml"
_C1_MANIFEST = module_manifest.part_data(
    _DEFAULT_COLUMN_MANIFEST,
    CONDENSER_LENS_1,
)
_C2_MANIFEST = module_manifest.part_data(
    _DEFAULT_COLUMN_MANIFEST,
    CONDENSER_LENS_2,
)
_C3_MANIFEST = module_manifest.part_data(
    _DEFAULT_COLUMN_MANIFEST,
    CONDENSER_LENS_3,
)
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    _DEFAULT_GUN_MANIFEST,
    "exit",
)


def _default_c1_absolute(local_key):
    return _DEFAULT_COLUMN_ORIGIN_Z_MM + float(_C1_MANIFEST[local_key])


def _default_column_absolute(part, local_key):
    return _DEFAULT_COLUMN_ORIGIN_Z_MM + float(part[local_key])


CONDENSER_LENS_1_DEFINITION = CondenserLensDefinition(
    key=CONDENSER_LENS_1,
    label=str(_C1_MANIFEST["name"]),
    mechanical_center_from_tip_mm=_default_c1_absolute(
        "local_center_z_mm"
    ),
    mechanical_length_mm=float(_C1_MANIFEST["length_mm"]),
    mechanical_outer_diameter_mm=float(
        _C1_MANIFEST["mechanical_outer_diameter_mm"]
    ),
    bore_diameter_mm=float(_C1_MANIFEST["bore_diameter_mm"]),
    pole_gap_mm=float(_C1_MANIFEST["pole_gap_mm"]),
    optical_reference_from_tip_mm=_default_c1_absolute(
        "optical_reference_local_z_mm"
    ),
    maximum_peak_field_t=0.28,
    field_scale_half_width_mm=10.0,
    default_excitation_percent=90.0,
    maximum_excitation_percent=100.0,
    colour="#1565c0",
    polarity=int(_C1_MANIFEST["field_polarity"]),
    effective_aperture_radius_mm=float(
        _C1_MANIFEST["effective_aperture_radius_mm"]
    ),
    normalise_profile_peak=True,
)

CONDENSER_LENS_2_DEFINITION = CondenserLensDefinition(
    key=CONDENSER_LENS_2,
    label=str(_C2_MANIFEST["name"]),
    mechanical_center_from_tip_mm=_default_column_absolute(
        _C2_MANIFEST, "local_center_z_mm"
    ),
    mechanical_length_mm=float(_C2_MANIFEST["length_mm"]),
    mechanical_outer_diameter_mm=float(
        _C2_MANIFEST["mechanical_outer_diameter_mm"]
    ),
    bore_diameter_mm=float(_C2_MANIFEST["bore_diameter_mm"]),
    pole_gap_mm=float(_C2_MANIFEST["pole_gap_mm"]),
    optical_reference_from_tip_mm=_default_column_absolute(
        _C2_MANIFEST, "optical_reference_local_z_mm"
    ),
    # The calibrated microprobe solution previously occupied 100%.  Rebase
    # the rating so the same 0.726 T field is produced at 70% with headroom.
    maximum_peak_field_t=1.0371428571428571,
    field_scale_half_width_mm=10.0,
    default_excitation_percent=35.0,
    maximum_excitation_percent=100.0,
    colour="#1976d2",
    polarity=int(_C2_MANIFEST["field_polarity"]),
    effective_aperture_radius_mm=(
        0.5 * float(_C2_MANIFEST["bore_diameter_mm"])
    ),
)

CONDENSER_LENS_3_DEFINITION = CondenserLensDefinition(
    key=CONDENSER_LENS_3,
    label=str(_C3_MANIFEST["name"]),
    mechanical_center_from_tip_mm=_default_column_absolute(
        _C3_MANIFEST, "local_center_z_mm"
    ),
    mechanical_length_mm=float(_C3_MANIFEST["length_mm"]),
    mechanical_outer_diameter_mm=float(
        _C3_MANIFEST["mechanical_outer_diameter_mm"]
    ),
    bore_diameter_mm=float(_C3_MANIFEST["bore_diameter_mm"]),
    pole_gap_mm=float(_C3_MANIFEST["pole_gap_mm"]),
    optical_reference_from_tip_mm=_default_column_absolute(
        _C3_MANIFEST, "optical_reference_local_z_mm"
    ),
    maximum_peak_field_t=0.38,
    field_scale_half_width_mm=9.0,
    default_excitation_percent=55.0,
    maximum_excitation_percent=100.0,
    colour="#0288d1",
    polarity=int(_C3_MANIFEST["field_polarity"]),
    effective_aperture_radius_mm=(
        0.5 * float(_C3_MANIFEST["bore_diameter_mm"])
    ),
)

CONDENSER_LENS_DEFINITIONS = (
    CONDENSER_LENS_1_DEFINITION,
    CONDENSER_LENS_2_DEFINITION,
    CONDENSER_LENS_3_DEFINITION,
)
CONDENSER_LENS_DEFINITION_BY_KEY = {
    definition.key: definition
    for definition in CONDENSER_LENS_DEFINITIONS
}


def create_condenser_lenses() -> list[CondenserLensState]:
    """Create one runtime lens record per condenser component."""

    return [
        definition.create_lens()
        for definition in CONDENSER_LENS_DEFINITIONS
    ]


@dataclass(frozen=True)
class _TranslatedMechanicalPart:
    key: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float


def _translated_mechanical_part(component, translation_mm=0.0):
    return _TranslatedMechanicalPart(
        str(component.key),
        (
            float(component.mechanical_center_from_tip_mm)
            + float(translation_mm)
        ),
        float(component.mechanical_length_mm),
    )


def resolve_condenser_entrance_mechanical_axis(
    gun_c1_aperture,
    condenser_components,
    c2_aperture,
    downstream_translation_mm=0.0,
):
    """Resolve the C1 mechanism through the C2-aperture mechanism."""

    condenser = {
        component.key: component for component in condenser_components
    }
    upstream_key = str(gun_c1_aperture.key)
    if upstream_key not in (C1_APERTURE, THERMIONIC_C1_APERTURE):
        raise ValueError(
            "The condenser entrance requires the active gun C1 mechanism."
        )
    if c2_aperture.key != CONDENSER_APERTURE_2:
        raise ValueError(
            "The condenser entrance requires the canonical C2 aperture."
        )
    missing = {
        CONDENSER_LENS_1,
        CONDENSER_LENS_2,
    } - condenser.keys()
    if missing:
        raise ValueError(
            "The condenser entrance is missing components: "
            + ", ".join(sorted(missing))
        )
    return resolve_mechanical_axis(
        (
            _translated_mechanical_part(gun_c1_aperture),
            _translated_mechanical_part(
                condenser[CONDENSER_LENS_1],
                downstream_translation_mm,
            ),
            _translated_mechanical_part(
                condenser[CONDENSER_LENS_2],
                downstream_translation_mm,
            ),
            _translated_mechanical_part(
                c2_aperture,
                downstream_translation_mm,
            ),
        ),
        (
            upstream_key,
            CONDENSER_LENS_1,
            CONDENSER_LENS_2,
            c2_aperture.key,
        ),
    )


C3_HARDWARE_OCCUPIED_LENGTH_MM = 230.0
C3_HARDWARE_DOWNSTREAM_CLEARANCE_MM = 170.0


def resolve_condenser_c3_hardware_mechanical_axis(
    c2_aperture,
    condenser_deflector,
    c3_lens,
    c3_aperture,
    *,
    installed,
    downstream_translation_mm=0.0,
):
    """Resolve the installed and absent C3-hardware profiles."""

    if c2_aperture.key != CONDENSER_APERTURE_2:
        raise ValueError("C3 hardware must follow the canonical C2 aperture.")
    if condenser_deflector.key != CONDENSER_DEFLECTOR:
        raise ValueError("C3 hardware requires the condenser deflector.")
    if c3_lens.key != CONDENSER_LENS_3:
        raise ValueError("C3 hardware requires the canonical C3 lens.")
    if c3_aperture.key != CONDENSER_APERTURE_3:
        raise ValueError("C3 hardware requires the canonical C3 aperture.")
    if installed:
        components = (
            _translated_mechanical_part(c2_aperture),
            _translated_mechanical_part(condenser_deflector),
            _translated_mechanical_part(c3_lens),
            _translated_mechanical_part(c3_aperture),
        )
        order = (
            CONDENSER_APERTURE_2,
            CONDENSER_DEFLECTOR,
            CONDENSER_LENS_3,
            CONDENSER_APERTURE_3,
        )
    else:
        components = (_translated_mechanical_part(c2_aperture),)
        order = (CONDENSER_APERTURE_2,)
    return resolve_mechanical_axis(components, order)


def condenser_lens_state_from_dict(data) -> CondenserLensState:
    """Restore one condenser record at the persistence boundary."""

    values = dict(data)
    definition = CONDENSER_LENS_DEFINITION_BY_KEY[values["key"]]
    defaults = definition.create_lens()
    restored = {
        key: getattr(defaults, key)
        for key in CondenserLensState.__dataclass_fields__
    }
    restored.update({
        key: value
        for key, value in values.items()
        if (
            key in CondenserLensState.__dataclass_fields__
            and key != "z_mm"
        )
    })
    restored["gaussian"] = [
        (
            term
            if isinstance(term, AxialFieldTerm)
            else AxialFieldTerm(**term)
        )
        for term in restored["gaussian"]
    ]
    return CondenserLensState(**restored)


class CondenserLensComponent:
    """One state-backed physical condenser lens."""

    _PLACEMENT_FIELDS = frozenset({
        "mechanical_center_from_tip_mm",
        "mechanical_length_mm",
        "mechanical_outer_diameter_mm",
        "bore_diameter_mm",
        "pole_gap_mm",
        "optical_reference_from_tip_mm",
        "effective_aperture_radius_mm",
    })

    def __init__(
        self,
        state,
        lens: CondenserLensState,
        definition: CondenserLensDefinition,
    ):
        self.state = state
        self._lens = lens
        self._base_definition = definition
        self._manifest_definition = definition

    @property
    def key(self):
        return self._base_definition.key

    @property
    def label(self):
        return self._base_definition.label

    @property
    def owner(self):
        return self._base_definition.owner

    @property
    def shape_profile(self):
        return self._base_definition.shape_profile

    @property
    def field_kind(self):
        return self._base_definition.field_kind

    @property
    def lens(self):
        return self._lens

    @property
    def definition(self):
        return self._manifest_definition

    def _set_placement(self, name, value):
        del name, value
        raise AttributeError(
            f"{self.label} geometry is owned by the selected Column TOML"
        )

    def apply_manifest_geometry(self, **geometry):
        unknown = set(geometry) - self._PLACEMENT_FIELDS
        if unknown:
            raise ValueError(
                f"Unsupported {self.label} TOML geometry fields: "
                + ", ".join(sorted(unknown))
            )
        self._manifest_definition = replace(
            self._base_definition,
            **{key: float(value) for key, value in geometry.items()},
        )
        self.apply_optical_position()
        return self

    @property
    def mechanical_center_from_tip_mm(self):
        return self.definition.mechanical_center_from_tip_mm

    @mechanical_center_from_tip_mm.setter
    def mechanical_center_from_tip_mm(self, value):
        previous_center = self.mechanical_center_from_tip_mm
        previous_optical = self.optical_reference_from_tip_mm
        delta_mm = float(value) - float(previous_center)
        self._set_placement("mechanical_center_from_tip_mm", value)
        self._set_placement(
            "optical_reference_from_tip_mm",
            float(previous_optical) + delta_mm,
        )
        self.apply_optical_position()

    @property
    def mechanical_length_mm(self):
        return self.definition.mechanical_length_mm

    @mechanical_length_mm.setter
    def mechanical_length_mm(self, value):
        self._set_placement("mechanical_length_mm", value)

    @property
    def mechanical_outer_diameter_mm(self):
        return self.definition.mechanical_outer_diameter_mm

    @mechanical_outer_diameter_mm.setter
    def mechanical_outer_diameter_mm(self, value):
        self._set_placement("mechanical_outer_diameter_mm", value)

    @property
    def bore_diameter_mm(self):
        return self.definition.bore_diameter_mm

    @bore_diameter_mm.setter
    def bore_diameter_mm(self, value):
        self._set_placement("bore_diameter_mm", value)

    @property
    def pole_gap_mm(self):
        return self.definition.pole_gap_mm

    @pole_gap_mm.setter
    def pole_gap_mm(self, value):
        self._set_placement("pole_gap_mm", value)

    @property
    def optical_reference_from_tip_mm(self):
        return self.definition.optical_reference_from_tip_mm

    @optical_reference_from_tip_mm.setter
    def optical_reference_from_tip_mm(self, value):
        self._set_placement("optical_reference_from_tip_mm", value)

    @property
    def effective_aperture_radius_mm(self):
        return self.definition.effective_aperture_radius_mm

    @effective_aperture_radius_mm.setter
    def effective_aperture_radius_mm(self, value):
        self._set_placement("effective_aperture_radius_mm", value)

    @property
    def parameters(self):
        lens = self.lens
        return {
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": self.mechanical_outer_diameter_mm,
            "bore_diameter_mm": self.bore_diameter_mm,
            "pole_gap_mm": self.pole_gap_mm,
            "optical_reference_from_tip_mm": self.optical_reference_from_tip_mm,
            "effective_aperture_radius_mm": self.effective_aperture_radius_mm,
            "maximum_peak_field_t": lens.b0_t,
            "excitation_percent": lens.percent,
            "maximum_excitation_percent": lens.max_percent,
            "field_scale_half_width_mm": lens.a_mm,
            "gaussian_terms": tuple(
                (term.amplitude, term.offset, term.sigma)
                for term in lens.gaussian
            ),
            "enabled": lens.enabled,
            "spherical_aberration_mm": lens.cs_mm,
            "chromatic_aberration_mm": lens.cc_mm,
            "polarity": lens.polarity,
            "normalise_profile_peak": lens.normalise_profile_peak,
        }

    def apply_optical_position(self):
        self.lens.z_mm = float(self.optical_reference_from_tip_mm)
        return self

    def validate(self):
        definition = self.definition
        lens = self.lens
        prefix = definition.label
        if definition.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(f"{prefix} mechanical centre must follow the tip.")
        if definition.mechanical_length_mm <= 0.0:
            raise ValueError(f"{prefix} mechanical length must be positive.")
        if (
            definition.mechanical_outer_diameter_mm
            <= definition.bore_diameter_mm
        ):
            raise ValueError(
                f"{prefix} outer diameter must exceed its bore diameter."
            )
        if definition.bore_diameter_mm <= 0.0 or definition.pole_gap_mm <= 0.0:
            raise ValueError(f"{prefix} bore and pole gap must be positive.")
        if (
            definition.effective_aperture_radius_mm is not None
            and definition.effective_aperture_radius_mm
            > definition.bore_diameter_mm / 2.0
        ):
            raise ValueError(
                f"{prefix} effective aperture must fit inside its bore."
            )
        if lens.b0_t < 0.0 or lens.a_mm <= 0.0:
            raise ValueError(
                f"{prefix} peak field must be non-negative and width positive."
            )
        if not 0.0 < lens.max_percent <= 100.0:
            raise ValueError(
                f"{prefix} maximum excitation must lie in (0, 100]."
            )
        if not 0.0 <= lens.percent <= lens.max_percent:
            raise ValueError(
                f"{prefix} excitation must lie within its configured range."
            )
        if not lens.gaussian:
            raise ValueError(f"{prefix} requires an axial magnetic-field profile.")
        if any(term.sigma <= 0.0 for term in lens.gaussian):
            raise ValueError(f"{prefix} Gaussian sigma values must be positive.")
        return self

    def magnetic_field_t(self, z_mm):
        """Return this component's signed on-axis Bz field."""

        lens = self.lens
        z = np.asarray(z_mm, dtype=float)
        field = np.zeros_like(z)
        if not lens.enabled:
            return field
        for term in lens.gaussian:
            sigma = max(abs(term.sigma * lens.a_mm), 1e-12)
            centre = lens.z_mm + term.offset * lens.a_mm
            field += term.amplitude * np.exp(
                -0.5 * ((z - centre) / sigma) ** 2
            )
        if lens.normalise_profile_peak:
            raw_peak = unit_field_peak(
                replace(lens, normalise_profile_peak=False)
            )
            field /= max(raw_peak, 1e-15)
        return float(lens.polarity) * lens.scale() * field

    def focal_length_mm(self):
        return focal_length_mm(self.lens, self.state.beam_voltage_kv)

    def field_support_mm(self, sigma_cutoff=7.0):
        lens = self.lens
        reaches = [
            abs(term.offset * lens.a_mm)
            + float(sigma_cutoff) * abs(term.sigma * lens.a_mm)
            for term in lens.gaussian
        ]
        half = max(reaches, default=0.0)
        return lens.z_mm - half, lens.z_mm + half

    def draw_layout(self):
        return self.definition

    def draw_ray_overlay(self):
        start, end = self.field_support_mm()
        return {
            "key": self.key,
            "optical_reference_z_mm": self.lens.z_mm,
            "field_support_start_z_mm": start,
            "field_support_end_z_mm": end,
            "focal_length_mm": self.focal_length_mm(),
        }


class CondenserSystem:
    """The sole runtime owner/view of the C1, C2 and C3 components."""

    def __init__(self, state):
        self.state = state
        lenses = {lens.key: lens for lens in state.lenses}
        missing = set(CONDENSER_LENS_KEYS) - lenses.keys()
        if missing:
            raise ValueError(
                "State is missing condenser lenses: "
                + ", ".join(sorted(missing))
            )
        self._components = {
            key: CondenserLensComponent(
                state,
                lenses[key],
                CONDENSER_LENS_DEFINITION_BY_KEY[key],
            )
            for key in CONDENSER_LENS_KEYS
        }

    def __iter__(self):
        return iter(self._components.values())

    def __getitem__(self, key):
        return self._components[key]

    @property
    def definitions(self):
        return tuple(component.definition for component in self)

    @property
    def condenser_lens_1(self):
        return self[CONDENSER_LENS_1]

    @property
    def condenser_lens_2(self):
        return self[CONDENSER_LENS_2]

    @property
    def condenser_lens_3(self):
        return self[CONDENSER_LENS_3]

    def validate(self):
        for component in self:
            component.validate()
        self.state.condenser_aperture_2.validate()
        self.state.condenser_deflector.validate()
        self.state.condenser_aperture_3.validate()
        return self

    def resolve_entrance_mechanical_axis(self):
        return resolve_condenser_entrance_mechanical_axis(
            self.state.electron_gun.c1_aperture,
            tuple(self),
            self.state.condenser_aperture_2,
        )

    def resolve_c3_hardware_mechanical_axis(self):
        installed = (
            getattr(
                self.state,
                "layout_c3_hardware",
                "three_condenser",
            )
            == "three_condenser"
        )
        return resolve_condenser_c3_hardware_mechanical_axis(
            self.state.condenser_aperture_2,
            self.state.condenser_deflector,
            self.condenser_lens_3,
            self.state.condenser_aperture_3,
            installed=installed,
        )

    def apply_optical_positions(self):
        for component in self:
            component.apply_optical_position()
        return self

    def apply_mode(self, mode):
        """Apply condenser excitation topology without duplicating components."""

        if mode not in ("three_lens", "two_lens_c3_off"):
            raise ValueError(f"Unknown column mode: {mode}")
        previous = getattr(self.state, "column_mode", "three_lens")
        if previous == mode:
            return self

        if mode == "two_lens_c3_off":
            self.state._three_lens_strengths = {
                component.key: (
                    component.lens.percent,
                    component.lens.enabled,
                )
                for component in self
            }
            c2 = self.condenser_lens_2.lens
            c3 = self.condenser_lens_3.lens
            c2_field = abs(float(c2.b0_t) * float(c2.percent) / 100.0)
            c3_field = abs(float(c3.b0_t) * float(c3.percent) / 100.0)
            compensated = (
                math.hypot(c2_field, c3_field)
                * 100.0
                / max(abs(float(c2.b0_t)), 1e-12)
            )
            if compensated > float(c2.max_percent):
                raise ValueError(
                    "C2 fixed maximum field cannot compensate the disabled C3"
                )
            c3.percent = 0.0
            c3.enabled = False
            c2.percent = compensated
            self.state.column_mode = mode
            return self

        saved = getattr(self.state, "_three_lens_strengths", {})
        for key, value in saved.items():
            if key in self._components:
                component = self[key]
                component.lens.percent, component.lens.enabled = value
        self.condenser_lens_3.lens.enabled = True
        self.state.column_mode = mode
        return self
