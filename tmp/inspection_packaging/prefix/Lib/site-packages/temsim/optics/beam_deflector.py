"""Beam Shift/Tilt Deflector with one canonical physical coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim import module_manifest
from temsim.component_keys import (
    BEAM_DEFLECTOR,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    canonical_deflector_key,
)
from temsim.optics.condenser_aperture import (
    CONDENSER_APERTURE_2_DEFINITION,
    CONDENSER_APERTURE_3_DEFINITION,
)


_DEFAULT_COLUMN_MODULE = "column/C3_ProbeCorrector.toml"
_DEFAULT_MANIFEST_PART = module_manifest.part_data(
    _DEFAULT_COLUMN_MODULE, BEAM_DEFLECTOR
)
_DEFAULT_C3_APERTURE_PART = module_manifest.part_data(
    _DEFAULT_COLUMN_MODULE, CONDENSER_APERTURE_3
)
_DEFAULT_COLUMN_ORIGIN_Z_MM = (
    module_manifest.port_z_mm("gun/FEG.toml", "exit")
    - module_manifest.port_z_mm(_DEFAULT_COLUMN_MODULE, "entrance")
)


@dataclass(frozen=True)
class BeamDeflectorDefinition:
    key: str
    label: str
    center_from_tip_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    effective_coil_thickness_mm: float
    inter_coil_gap_mm: float
    two_condenser_anchor_key: str
    three_condenser_anchor_key: str
    center_downstream_of_anchor_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "shared_column"
    kind: str = "paired_deflector"
    shape_profile: str = "paired_deflector_coils"
    interaction_kind: str = "paired_transverse_kick"

    @property
    def mechanical_center_from_tip_mm(self):
        return self.center_from_tip_mm

    @property
    def mechanical_length_mm(self):
        return (
            2.0 * self.effective_coil_thickness_mm
            + self.inter_coil_gap_mm
        )

    @property
    def mechanical_center_downstream_of_anchor_mm(self):
        return self.center_downstream_of_anchor_mm

    @property
    def optical_upper_reference_from_tip_mm(self):
        return self.center_from_tip_mm - 0.5 * (
            self.effective_coil_thickness_mm + self.inter_coil_gap_mm
        )

    @property
    def optical_lower_reference_from_tip_mm(self):
        return self.center_from_tip_mm + 0.5 * (
            self.effective_coil_thickness_mm + self.inter_coil_gap_mm
        )

    def create_component(self):
        return BeamDeflectorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.center_from_tip_mm,
            upper_x_mrad=0.0,
            upper_y_mrad=0.0,
            lower_x_mrad=0.0,
            lower_y_mrad=0.0,
            thickness_mm=self.effective_coil_thickness_mm,
            enabled=True,
            colour=self.colour,
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=self.mechanical_outer_diameter_mm,
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            maximum_kick_mrad=self.maximum_kick_mrad,
            inter_coil_gap_mm=self.inter_coil_gap_mm,
            anchor_key=self.three_condenser_anchor_key,
            mechanical_center_downstream_of_anchor_mm=(
                self.center_downstream_of_anchor_mm
            ),
        )


@dataclass
class BeamDeflectorComponent:
    """TOML-sized double deflector with one stored axial coordinate.

    ``z_mm`` is the sole stored position.  Mechanical drawing, ray interaction
    planes and numerical kick events are all derived from that coordinate.
    """

    name: str
    key: str
    z_mm: float
    upper_x_mrad: float
    upper_y_mrad: float
    lower_x_mrad: float
    lower_y_mrad: float
    thickness_mm: float
    enabled: bool
    colour: str
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    maximum_kick_mrad: float
    inter_coil_gap_mm: float = float(
        _DEFAULT_MANIFEST_PART["mechanical_inter_coil_gap_mm"]
    )
    anchor_key: str = CONDENSER_APERTURE_3
    mechanical_center_downstream_of_anchor_mm: float = (
        float(_DEFAULT_MANIFEST_PART["local_center_z_mm"])
        - float(_DEFAULT_C3_APERTURE_PART["local_center_z_mm"])
    )

    EXPECTED_KEY: ClassVar[str] = BEAM_DEFLECTOR
    OWNER: ClassVar[str] = "shared_column"

    def __post_init__(self):
        object.__setattr__(self, "_geometry_ready", False)
        object.__setattr__(self, "z_mm", float(self.z_mm))
        object.__setattr__(self, "thickness_mm", float(self.thickness_mm))
        object.__setattr__(
            self, "inter_coil_gap_mm", float(self.inter_coil_gap_mm)
        )
        self._sync_mechanical_length()
        object.__setattr__(self, "_geometry_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_geometry_ready", False)
        if ready and name in {"thickness_mm", "inter_coil_gap_mm"}:
            value = float(value)
            if value <= 0.0:
                raise ValueError(
                    f"{self.name}: coil thickness and gap must be positive."
                )
            object.__setattr__(self, name, value)
            self._sync_mechanical_length()
            return
        if ready and name == "mechanical_length_mm":
            length_mm = float(value)
            gap_mm = length_mm - 2.0 * float(self.thickness_mm)
            if gap_mm <= 0.0:
                raise ValueError(
                    f"{self.name}: mechanical length must contain two "
                    "coils and a positive gap."
                )
            object.__setattr__(self, "inter_coil_gap_mm", gap_mm)
            object.__setattr__(self, name, length_mm)
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return self.OWNER

    @property
    def kind(self):
        return "paired_deflector"

    @property
    def shape_profile(self):
        return "paired_deflector_coils"

    @property
    def interaction_kind(self):
        return "paired_transverse_kick"

    @property
    def mechanical_center_from_tip_mm(self):
        return self.z_mm

    @mechanical_center_from_tip_mm.setter
    def mechanical_center_from_tip_mm(self, value):
        value = float(value)
        delta_mm = value - float(self.z_mm)
        self.z_mm = value
        self.mechanical_center_downstream_of_anchor_mm += delta_mm

    @property
    def optical_center_from_tip_mm(self):
        return self.z_mm

    @optical_center_from_tip_mm.setter
    def optical_center_from_tip_mm(self, value):
        self.mechanical_center_from_tip_mm = value

    @property
    def optical_plane_separation_mm(self):
        return float(self.thickness_mm) + float(self.inter_coil_gap_mm)

    @optical_plane_separation_mm.setter
    def optical_plane_separation_mm(self, value):
        gap_mm = abs(float(value)) - float(self.thickness_mm)
        if gap_mm <= 0.0:
            raise ValueError(
                f"{self.name}: plane separation must exceed coil thickness."
            )
        self.inter_coil_gap_mm = gap_mm
        self._sync_mechanical_length()

    @property
    def optical_upper_reference_from_tip_mm(self):
        return self.upper_z_mm

    @property
    def optical_lower_reference_from_tip_mm(self):
        return self.lower_z_mm

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

    def _sync_mechanical_length(self):
        object.__setattr__(self, "mechanical_length_mm", (
            2.0 * float(self.thickness_mm)
            + float(self.inter_coil_gap_mm)
        ))

    def resolve_after_aperture(self, aperture):
        if aperture.key not in {
            CONDENSER_APERTURE_2,
            CONDENSER_APERTURE_3,
        }:
            raise ValueError(
                "Beam Shift/Tilt Deflector requires C2 or C3 Aperture."
            )
        self.anchor_key = aperture.key
        self.z_mm = (
            float(aperture.z_mm)
            + float(self.mechanical_center_downstream_of_anchor_mm)
        )
        return self

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError(f"{self.name} key is not canonical.")
        if self.z_mm < 0.0:
            raise ValueError(
                f"{self.name} mechanical centre must follow the source."
            )
        if self.thickness_mm <= 0.0 or self.inter_coil_gap_mm <= 0.0:
            raise ValueError(
                f"{self.name}: coil thickness and gap must be positive."
            )
        expected_length_mm = (
            2.0 * self.thickness_mm + self.inter_coil_gap_mm
        )
        if abs(self.mechanical_length_mm - expected_length_mm) > 1.0e-9:
            raise ValueError(
                "Beam Shift/Tilt Deflector length must equal "
                "coil + gap + coil."
            )
        if self.mechanical_outer_diameter_mm <= 0.0:
            raise ValueError(f"{self.name} outer diameter must be positive.")
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(f"{self.name} bore must fit inside its body.")
        if self.maximum_kick_mrad <= 0.0:
            raise ValueError(f"{self.name} kick limit must be positive.")
        for value in (
            self.upper_x_mrad,
            self.upper_y_mrad,
            self.lower_x_mrad,
            self.lower_y_mrad,
        ):
            if abs(float(value)) > self.maximum_kick_mrad:
                raise ValueError(
                    f"{self.name} kick exceeds its configured limit."
                )
        return self

    def apply_optical_positions(self):
        return self

    def kick_events(self):
        if not self.enabled:
            return ()
        return (
            (
                self.upper_z_mm,
                self.upper_x_mrad * 1.0e-3,
                self.upper_y_mrad * 1.0e-3,
            ),
            (
                self.lower_z_mm,
                self.lower_x_mrad * 1.0e-3,
                self.lower_y_mrad * 1.0e-3,
            ),
        )

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.z_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "coil_plane_separation_mm": self.optical_plane_separation_mm,
            "coil_thickness_mm": self.thickness_mm,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "upper_plane_z_mm": self.upper_z_mm,
            "lower_plane_z_mm": self.lower_z_mm,
            "effective_coil_thickness_mm": self.thickness_mm,
            "upper_kick_mrad": (
                self.upper_x_mrad,
                self.upper_y_mrad,
            ),
            "lower_kick_mrad": (
                self.lower_x_mrad,
                self.lower_y_mrad,
            ),
            "enabled": self.enabled,
        }


BEAM_DEFLECTOR_DEFINITION = BeamDeflectorDefinition(
    key=BEAM_DEFLECTOR,
    label=str(_DEFAULT_MANIFEST_PART["name"]),
    center_from_tip_mm=(
        _DEFAULT_COLUMN_ORIGIN_Z_MM
        + float(_DEFAULT_MANIFEST_PART["local_center_z_mm"])
    ),
    mechanical_outer_diameter_mm=float(
        _DEFAULT_MANIFEST_PART["mechanical_outer_diameter_mm"]
    ),
    mechanical_clear_bore_diameter_mm=float(
        _DEFAULT_MANIFEST_PART["mechanical_clear_bore_diameter_mm"]
    ),
    effective_coil_thickness_mm=float(
        _DEFAULT_MANIFEST_PART["effective_thickness_mm"]
    ),
    inter_coil_gap_mm=float(
        _DEFAULT_MANIFEST_PART["mechanical_inter_coil_gap_mm"]
    ),
    two_condenser_anchor_key=CONDENSER_APERTURE_2,
    three_condenser_anchor_key=CONDENSER_APERTURE_3,
    center_downstream_of_anchor_mm=(
        float(_DEFAULT_MANIFEST_PART["local_center_z_mm"])
        - float(_DEFAULT_C3_APERTURE_PART["local_center_z_mm"])
    ),
    maximum_kick_mrad=100.0,
    colour="#ef6c00",
)


def create_beam_deflector():
    return BEAM_DEFLECTOR_DEFINITION.create_component()


def resolve_beam_deflector_after_active_aperture(state):
    """Resolve the Beam Shift/Tilt root from the selected hardware aperture."""

    aperture = (
        state.condenser_aperture_3
        if getattr(state, "layout_c3_hardware", "three_condenser")
        == "three_condenser"
        else state.condenser_aperture_2
    )
    return state.beam_deflector.resolve_after_aperture(aperture)


def beam_deflector_from_dict(data):
    """Restore current records and migrate obsolete dual-coordinate records."""

    values = dict(data)
    values["key"] = canonical_deflector_key(values.get("key", ""))
    thickness_mm = float(values.get(
        "thickness_mm",
        BEAM_DEFLECTOR_DEFINITION.effective_coil_thickness_mm,
    ))
    mechanical_length_mm = float(values.get(
        "mechanical_length_mm",
        BEAM_DEFLECTOR_DEFINITION.mechanical_length_mm,
    ))
    gap_mm = float(values.get(
        "inter_coil_gap_mm",
        max(mechanical_length_mm - 2.0 * thickness_mm, 1.0e-9),
    ))
    anchor_key = values.get(
        "anchor_key",
        BEAM_DEFLECTOR_DEFINITION.three_condenser_anchor_key,
    )
    offset_mm = float(values.get(
        "mechanical_center_downstream_of_anchor_mm",
        BEAM_DEFLECTOR_DEFINITION.center_downstream_of_anchor_mm,
    ))
    anchor_definition = (
        CONDENSER_APERTURE_2_DEFINITION
        if anchor_key == CONDENSER_APERTURE_2
        else CONDENSER_APERTURE_3_DEFINITION
    )
    canonical_z_mm = float(values.get(
        "z_mm",
        anchor_definition.center_from_tip_mm + offset_mm,
    ))

    component = create_beam_deflector()
    allowed = BeamDeflectorComponent.__dataclass_fields__
    for attribute, value in values.items():
        if attribute in allowed and attribute not in {
            "z_mm",
            "mechanical_length_mm",
            "thickness_mm",
            "inter_coil_gap_mm",
        }:
            setattr(component, attribute, value)
    component.thickness_mm = thickness_mm
    component.inter_coil_gap_mm = gap_mm
    component._sync_mechanical_length()
    component.z_mm = canonical_z_mm
    component.anchor_key = anchor_key
    component.mechanical_center_downstream_of_anchor_mm = offset_mm
    component.key = BEAM_DEFLECTOR
    return component.validate()
