"""Projector Lens P2 anchored to the Diffraction Lens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temsim.component_keys import (
    PROJECTOR_LENS_2,
    SELECTED_AREA_APERTURE,
    canonical_lens_key,
)
from temsim.optics.selected_area_downstream import downstream_offset_mm
from temsim.optics.model import Gaussian
from temsim.optics.selected_area_aperture import (
    SELECTED_AREA_APERTURE_DEFINITION,
)
from temsim.optics.round_lens import (
    AnchoredRoundLensComponent,
    AnchoredRoundLensGeometry,
    restore_anchored_round_lens,
)


@dataclass(frozen=True)
class ProjectorLensP2Definition:
    key: str = PROJECTOR_LENS_2
    label: str = "Projector Lens P2"
    anchor_key: str = SELECTED_AREA_APERTURE
    mechanical_center_downstream_of_anchor_mm: float = downstream_offset_mm(
        PROJECTOR_LENS_2
    )
    optical_reference_downstream_of_anchor_mm: float = downstream_offset_mm(
        PROJECTOR_LENS_2
    )
    mechanical_length_mm: float = 185.0
    # The legacy drawing specifies only D260-340 mm.
    mechanical_outer_diameter_mm: float = 300.0
    mechanical_clear_bore_diameter_mm: float = 20.0
    pole_gap_mm: float = 20.0
    b0_t: float = 0.36
    a_mm: float = 20.0
    percent: float = 30.0
    maximum_percent: float = 100.0
    colour: str = "#5d4037"
    owner: str = "projector"
    kind: str = "round_lens"
    shape_profile: str = "magnetic_lens_yoke"
    interaction_kind: str = "axial_magnetic_field"

    @property
    def name(self):
        return self.label

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def geometry_for(self, anchor_geometry):
        return AnchoredRoundLensGeometry(
            mechanical_center_below_sample_mm=(
                float(anchor_geometry.mechanical_center_below_sample_mm)
                + self.mechanical_center_downstream_of_anchor_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            optical_reference_z_mm=(
                float(anchor_geometry.optical_reference_z_mm)
                + self.mechanical_center_downstream_of_anchor_mm
            ),
        )

    def create_component(self):
        return ProjectorLensP2Component(
            name=self.label,
            key=self.key,
            z_mm=(
                SELECTED_AREA_APERTURE_DEFINITION
                .standalone_optical_reference_z_mm
                + self.mechanical_center_downstream_of_anchor_mm
            ),
            b0_t=self.b0_t,
            a_mm=self.a_mm,
            percent=self.percent,
            max_percent=self.maximum_percent,
            colour=self.colour,
            gaussian=[
                Gaussian(0.09, -1.0, 0.90),
                Gaussian(0.82, 0.0, 0.55),
                Gaussian(0.09, 1.0, 0.90),
            ],
            enabled=True,
            cs_mm=None,
            cc_mm=None,
            polarity=1,
            normalise_profile_peak=False,
            anchor_key=self.anchor_key,
            mechanical_center_downstream_of_anchor_mm=(
                self.mechanical_center_downstream_of_anchor_mm
            ),
            optical_reference_downstream_of_anchor_mm=(
                self.optical_reference_downstream_of_anchor_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            pole_gap_mm=self.pole_gap_mm,
            corrector=self.owner,
        )


@dataclass
class ProjectorLensP2Component(AnchoredRoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = PROJECTOR_LENS_2
    EXPECTED_ANCHOR_KEY: ClassVar[str] = SELECTED_AREA_APERTURE


PROJECTOR_LENS_P2_DEFINITION = ProjectorLensP2Definition()


def create_projector_lens_p2():
    return PROJECTOR_LENS_P2_DEFINITION.create_component().validate()


def projector_lens_p2_from_dict(
    data,
    legacy_anchor_reference_z_mm=None,
    legacy_reference_z_mm=None,
):
    values = dict(data)
    values["key"] = canonical_lens_key(values.get("key", ""))
    if (
        "optical_reference_downstream_of_anchor_mm" not in values
        and legacy_reference_z_mm is not None
    ):
        values["z_mm"] = float(legacy_reference_z_mm)
    component = restore_anchored_round_lens(
        create_projector_lens_p2(),
        values,
        legacy_anchor_reference_z_mm=legacy_anchor_reference_z_mm,
    )
    component.key = PROJECTOR_LENS_2
    component.name = PROJECTOR_LENS_P2_DEFINITION.label
    component.anchor_key = SELECTED_AREA_APERTURE
    if values.get("anchor_key") != SELECTED_AREA_APERTURE:
        offset = downstream_offset_mm(PROJECTOR_LENS_2)
        component.mechanical_center_downstream_of_anchor_mm = offset
        component.optical_reference_downstream_of_anchor_mm = offset
    component.corrector = PROJECTOR_LENS_P2_DEFINITION.owner
    return component.validate()
