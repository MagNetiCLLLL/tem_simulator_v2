"""Canonical shared-column DC deflector component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim.component_keys import (
    DC_DEFLECTOR,
    canonical_corrector_element_key,
)
from temsim.optics.single_plane_deflector import (
    SinglePlaneDeflectorComponent,
    restore_single_plane_deflector,
)


@dataclass(frozen=True)
class DcDeflectorDefinition:
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
    owner: str = "shared_column"
    kind: str = "deflector"
    shape_profile: str = "single_deflector_coil"
    interaction_kind: str = "thin_transverse_kick"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return DcDeflectorComponent(
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
            corrector=self.owner,
        )


@dataclass
class DcDeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = DC_DEFLECTOR


DC_DEFLECTOR_DEFINITION = DcDeflectorDefinition(
    key=DC_DEFLECTOR,
    label="DC Deflector",
    mechanical_center_from_tip_mm=1411.0,
    mechanical_length_mm=8.0,
    mechanical_outer_diameter_mm=75.0,
    mechanical_clear_bore_diameter_mm=20.0,
    optical_reference_from_tip_mm=874.0,
    effective_thickness_mm=6.0,
    maximum_kick_mrad=100.0,
    colour="#00acc1",
)


def create_dc_deflector():
    return DC_DEFLECTOR_DEFINITION.create_component()


def dc_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_corrector_element_key(
        values.get("key", "")
    )
    component = create_dc_deflector()
    restored = restore_single_plane_deflector(component, values)
    restored.key = DC_DEFLECTOR
    restored.corrector = DC_DEFLECTOR_DEFINITION.owner
    return restored.validate()
