"""Single-source condenser double-deflector component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim.component_keys import (
    CONDENSER_DEFLECTOR,
    canonical_deflector_key,
)
from temsim.optics.paired_deflector import (
    PairedDeflectorComponent,
    restore_paired_deflector,
)


@dataclass(frozen=True)
class CondenserDeflectorDefinition:
    key: str
    label: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_upper_reference_from_tip_mm: float
    optical_lower_reference_from_tip_mm: float
    effective_coil_thickness_mm: float
    maximum_kick_mrad: float
    colour: str
    owner: str = "condenser_assembly"
    kind: str = "paired_deflector"
    shape_profile: str = "paired_deflector_coils"
    interaction_kind: str = "paired_transverse_kick"

    def create_component(self):
        return CondenserDeflectorComponent(
            name=self.label,
            key=self.key,
            upper_z_mm=self.optical_upper_reference_from_tip_mm,
            lower_z_mm=self.optical_lower_reference_from_tip_mm,
            upper_x_mrad=0.0,
            upper_y_mrad=0.0,
            lower_x_mrad=0.0,
            lower_y_mrad=0.0,
            thickness_mm=self.effective_coil_thickness_mm,
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
class CondenserDeflectorComponent(PairedDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = CONDENSER_DEFLECTOR
    OWNER: ClassVar[str] = "condenser_assembly"


CONDENSER_DEFLECTOR_DEFINITION = CondenserDeflectorDefinition(
    key=CONDENSER_DEFLECTOR,
    label="Condenser Deflector",
    mechanical_center_from_tip_mm=1034.0,
    mechanical_length_mm=40.0,
    mechanical_outer_diameter_mm=90.0,
    mechanical_clear_bore_diameter_mm=20.0,
    optical_upper_reference_from_tip_mm=610.0,
    optical_lower_reference_from_tip_mm=630.0,
    effective_coil_thickness_mm=10.0,
    maximum_kick_mrad=100.0,
    colour="#26a69a",
)


def create_condenser_deflector():
    return CONDENSER_DEFLECTOR_DEFINITION.create_component()


def condenser_deflector_from_dict(data):
    values = dict(data)
    values["key"] = canonical_deflector_key(values.get("key", ""))
    component = create_condenser_deflector()
    restored = restore_paired_deflector(component, values)
    restored.key = CONDENSER_DEFLECTOR
    return restored.validate()
