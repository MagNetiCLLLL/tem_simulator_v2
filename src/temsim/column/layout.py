"""Parameterized mechanical optical-column layouts.

Coordinates are mechanical drawing coordinates: the sample plane is ``Z = 0``
and the electron beam travels from positive Z toward negative Z.  These values
are provisional Titan G2-style mechanical defaults for drawing, collision
checking and topology work.  They are not electron-optical effective distances.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Tuple

from temsim.mechanical_axis import resolve_mechanical_axis
from temsim.component_names import (
    APERTURE_NAMES,
    DEFLECTOR_NAMES,
    LENS_NAMES,
    STIGMATOR_NAMES,
)
from temsim.optics.image_corrector import (
    IMAGE_CORRECTOR_COMPONENTS,
    REFERENCE_SAMPLE_EFFECTIVE_Z_MM,
    create_image_corrector_components,
)
from temsim.optics.electron_gun import FieldEmissionGun
from temsim.optics.condenser_lens import CONDENSER_LENS_DEFINITIONS
from temsim.optics.condenser_aperture import (
    CONDENSER_APERTURE_2_DEFINITION,
    CONDENSER_APERTURE_3_DEFINITION,
)
from temsim.optics.condenser_deflector import (
    CONDENSER_DEFLECTOR_DEFINITION,
)
from temsim.optics.beam_deflector import BEAM_DEFLECTOR_DEFINITION
from temsim.optics.ac_deflector import AC_DEFLECTOR_DEFINITION
from temsim.optics.mini_condenser import MINI_CONDENSER_DEFINITION
from temsim.optics.condenser_stigmator import (
    CONDENSER_STIGMATOR_DEFINITION,
)
from temsim.optics.diffraction_stigmator import (
    DIFFRACTION_STIGMATOR_DEFINITION,
    IMAGE_CORRECTED_INSTALLATION as DIFFRACTION_IMAGE_INSTALLATION,
    STANDALONE_INSTALLATION as DIFFRACTION_STANDALONE_INSTALLATION,
)
from temsim.optics.diffraction_lens import (
    DIFFRACTION_LENS_DEFINITION,
    IMAGE_CORRECTED_INSTALLATION as DIFFRACTION_LENS_IMAGE_INSTALLATION,
    STANDALONE_INSTALLATION as DIFFRACTION_LENS_STANDALONE_INSTALLATION,
)
from temsim.optics.intermediate_lens import (
    INTERMEDIATE_LENS_DEFINITION,
)
from temsim.optics.projector_lens_p1 import (
    PROJECTOR_LENS_P1_DEFINITION,
)
from temsim.optics.projector_lens_p2 import (
    PROJECTOR_LENS_P2_DEFINITION,
)
from temsim.detector.stem_detector import STEM_DETECTOR_DEFINITIONS
from temsim.detector.fluorescent_screen import (
    FLUORESCENT_SCREEN_DEFINITION,
)
from temsim.detector.camera import CAMERA_DETECTOR_DEFINITION
from temsim.optics.energy_filter_entrance_aperture import (
    ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION,
)
from temsim.optics.objective_stigmator import (
    OBJECTIVE_STIGMATOR_DEFINITION,
)
from temsim.optics.image_diffraction_deflector import (
    IMAGE_DIFFRACTION_DEFLECTOR_DEFINITION,
)
from temsim.optics.descan_deflector import DESCAN_DEFLECTOR_DEFINITION
from temsim.optics.objective_lens import OBJECTIVE_LENS_DEFINITION
from temsim.optics.objective_aperture import (
    OBJECTIVE_APERTURE_DEFINITION,
)
from temsim.optics.selected_area_aperture import (
    IMAGE_CORRECTED_INSTALLATION,
    SELECTED_AREA_APERTURE_DEFINITION,
    STANDALONE_INSTALLATION,
)
from temsim.optics.selected_area_downstream import (
    SELECTED_AREA_DOWNSTREAM_KEYS,
)
from temsim.mechanical_profiles import pole_piece_keys
from temsim.optics.probe_corrector import (
    ADAPTER_LENS_DEFINITION,
    DPH2_DEFLECTOR_DEFINITION,
    DP22_DEFLECTOR_DEFINITION,
    HP2_HEXAPOLE_DEFINITION,
    HPC_HEXAPOLE_DEFINITION,
    DP11_DEFLECTOR_DEFINITION,
    DP12_SCAN_DEFLECTOR_DEFINITION,
    DP21_DEFLECTOR_DEFINITION,
    DPH1_DEFLECTOR_DEFINITION,
    HP1_HEXAPOLE_DEFINITION,
    HPOL_HEXAPOLE_DEFINITION,
    QPH2_QUADRUPOLE_DEFINITION,
    QPC_QUADRUPOLE_DEFINITION,
    QPH1_QUADRUPOLE_DEFINITION,
    QPOL_QUADRUPOLE_DEFINITION,
    TL12_LENS_DEFINITION,
    TL21_LENS_DEFINITION,
    TL22_LENS_DEFINITION,
)
from temsim.component_keys import (
    AC_DEFLECTOR,
    ADAPTER_LENS,
    BEAM_DEFLECTOR,
    C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    CONDENSER_DEFLECTOR,
    CONDENSER_LENS_1,
    CONDENSER_LENS_1_LOWER_POLE,
    CONDENSER_LENS_2,
    CONDENSER_LENS_2_UPPER_POLE,
    CONDENSER_LENS_3,
    CONDENSER_LENS_3_LOWER_POLE,
    CONDENSER_LENS_3_UPPER_POLE,
    CONDENSER_STIGMATOR,
    FEG_ACCELERATOR,
    FEG_DEFLECTOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_EXTRACTOR,
    FEG_STIGMATOR,
    FEG_TIP,
    FEG_MONOCHROMATOR_WIEN,
    GUN_EXTRACTOR_APERTURE,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_C1_APERTURE,
    THERMIONIC_CATHODE,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_GUN_LENS,
    THERMIONIC_STIGMATOR,
    THERMIONIC_WEHNELT,
    MINI_CONDENSER,
    OBJECTIVE_LENS,
    OBJECTIVE_APERTURE,
    OBJECTIVE_STIGMATOR,
    IMAGE_DIFFRACTION_DEFLECTOR,
    IMAGE_CORRECTOR_ADAPTER_LENS,
    IMAGE_CORRECTOR_DP11_DEFLECTOR,
    IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR,
    IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_HP2_HEXAPOLE,
    IMAGE_CORRECTOR_HPOL_HEXAPOLE,
    IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_KEYS,
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    IMAGE_CORRECTOR_SAD_PLANE,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
    DESCAN_DEFLECTOR,
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    DARK_FIELD_DETECTOR,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    FLUORESCENT_SCREEN,
    DIFFRACTION_LENS,
    DIFFRACTION_LENS_LOWER_POLE,
    DIFFRACTION_LENS_UPPER_POLE,
    DIFFRACTION_STIGMATOR,
    INTERMEDIATE_LENS,
    INTERMEDIATE_LENS_LOWER_POLE,
    INTERMEDIATE_LENS_UPPER_POLE,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_1_LOWER_POLE,
    PROJECTOR_LENS_1_UPPER_POLE,
    PROJECTOR_LENS_2,
    PROJECTOR_LENS_2_LOWER_POLE,
    PROJECTOR_LENS_2_UPPER_POLE,
    HAADF_DETECTOR,
    PROBE_DPH2_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR,
    PROBE_DP11_DEFLECTOR,
    PROBE_DP12_SCAN_DEFLECTOR,
    PROBE_DP21_DEFLECTOR,
    PROBE_DP22_DEFLECTOR,
    PROBE_HP1_HEXAPOLE,
    PROBE_HPOL_HEXAPOLE,
    PROBE_HP2_HEXAPOLE,
    PROBE_HPC_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE,
    PROBE_QPH1_QUADRUPOLE,
    PROBE_QPOL_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
    SELECTED_AREA_APERTURE,
)


class CorrectorAssembly(str, Enum):
    NO_CORRECTOR = "no_corrector"
    PROBE_CORRECTOR = "probe_corrector"
    IMAGE_CORRECTOR = "image_corrector"
    DOUBLE_CORRECTOR = "double_corrector"


class C3Hardware(str, Enum):
    TWO_CONDENSER = "two_condenser"
    THREE_CONDENSER = "three_condenser"


class Branch(str, Enum):
    COMMON = "common"
    ILLUMINATION = "illumination"
    IMAGE = "image"
    DETECTION = "detection"
    ENERGY_FILTER = "energy_filter"


@dataclass(frozen=True)
class MechanicalEnvelope:
    """Provisional mechanical allocation, not a manufacturer dimension."""

    start_s_mm: float
    end_s_mm: float

    def __post_init__(self):
        if self.end_s_mm < self.start_s_mm:
            raise ValueError("A mechanical envelope end must not precede its start.")

    @property
    def center_s_mm(self) -> float:
        return (self.start_s_mm + self.end_s_mm) / 2.0


@dataclass(frozen=True)
class FieldSupport:
    """Reference/field range, deliberately independent of mechanical space."""

    start_s_mm: float
    end_s_mm: float


@dataclass(frozen=True)
class MechanicalShape:
    """Machine-readable axial shape owned by one column component."""

    axial_length_mm: float
    profile: str = "axial_envelope"
    external_envelope: str = ""
    outer_diameter_mm: Optional[float] = None
    active_diameter_mm: Optional[float] = None
    active_length_mm: Optional[float] = None


@dataclass(frozen=True)
class FieldModel:
    """Physical interaction provided by a component on the optical axis."""

    kind: str
    active: Optional[bool] = None


@dataclass(frozen=True)
class LayoutComponent:
    key: str
    name: str
    kind: str
    owner: str
    branch: Branch
    excitation_enabled: Optional[bool]
    mechanical: MechanicalEnvelope
    field_support: FieldSupport
    local_s_center_mm: float
    local_s_range_mm: Tuple[float, float]
    rendered_z_center_mm: float
    rendered_z_range_mm: Tuple[float, float]
    external_envelope: str = ""
    note: str = ""
    upstream_key: Optional[str] = None
    downstream_key: Optional[str] = None
    upstream_clearance_mm: Optional[float] = None
    downstream_clearance_mm: Optional[float] = None
    mechanical_shape: Optional[MechanicalShape] = None
    field_model: Optional[FieldModel] = None
    optical_reference_plane_z_mm: Optional[float] = None
    effective_aperture_radius_mm: Optional[float] = None
    optical_interaction_planes_z_mm: Tuple[float, ...] = ()
    nested_parent_key: Optional[str] = None
    mechanical_overlap_reason: str = ""

    @property
    def mechanical_center_z_mm(self):
        return self.rendered_z_center_mm


class LayoutResult(tuple):
    """Sample-centered mechanical layout components."""

    @property
    def source_to_sample_mm(self) -> float:
        components = {component.key: component for component in self}
        source_key = (
            FEG_TIP
            if FEG_TIP in components
            else THERMIONIC_CATHODE
        )
        return (
            components[source_key].local_s_center_mm
            - components["sample"].local_s_center_mm
        )


@dataclass(frozen=True)
class ObjectiveLayout:
    pole_piece_type: str = "s_twin"
    inner_face_gap_mm: float = 5.4
    sample_axial_offset_mm: float = 0.0
    specimen_thickness_mm: float = 0.0001

    def __post_init__(self):
        if not abs(self.inner_face_gap_mm - 5.4) <= 1e-9:
            raise ValueError(
                "Objective flat pole-tip separation must be 5.4 mm."
            )
        if abs(self.sample_axial_offset_mm) >= self.inner_face_gap_mm / 2.0:
            raise ValueError("Sample offset must remain between the objective inner faces.")
        if self.specimen_thickness_mm < 0.0:
            raise ValueError("Specimen thickness cannot be negative.")


@dataclass(frozen=True)
class LayoutConfiguration:
    """Selections that change physical topology, not optical excitations."""

    corrector: CorrectorAssembly = CorrectorAssembly.PROBE_CORRECTOR
    electron_gun_type: str = "cold_feg"
    c3_hardware: C3Hardware = C3Hardware.THREE_CONDENSER
    c3_excited: bool = True
    monochromator_installed: bool = False
    source_relative_column_offset_mm: float = 0.0
    sample_center_from_source_mm: float = 1149.0
    objective: ObjectiveLayout = field(default_factory=ObjectiveLayout)
    energy_filter_selected: bool = False
    gun_components: tuple = field(
        default_factory=lambda: FieldEmissionGun().components
    )
    condenser_components: tuple = CONDENSER_LENS_DEFINITIONS
    condenser_aperture_2_component: object = (
        CONDENSER_APERTURE_2_DEFINITION
    )
    condenser_aperture_3_component: object = (
        CONDENSER_APERTURE_3_DEFINITION
    )
    condenser_deflector_component: object = (
        CONDENSER_DEFLECTOR_DEFINITION
    )
    beam_deflector_component: object = BEAM_DEFLECTOR_DEFINITION
    ac_deflector_component: object = AC_DEFLECTOR_DEFINITION
    mini_condenser_component: object = MINI_CONDENSER_DEFINITION
    condenser_stigmator_component: object = (
        CONDENSER_STIGMATOR_DEFINITION
    )
    diffraction_stigmator_component: object = (
        DIFFRACTION_STIGMATOR_DEFINITION
    )
    diffraction_lens_component: object = DIFFRACTION_LENS_DEFINITION
    intermediate_lens_component: object = INTERMEDIATE_LENS_DEFINITION
    projector_lens_p1_component: object = PROJECTOR_LENS_P1_DEFINITION
    projector_lens_p2_component: object = PROJECTOR_LENS_P2_DEFINITION
    stem_detector_components: tuple = STEM_DETECTOR_DEFINITIONS
    fluorescent_screen_component: object = FLUORESCENT_SCREEN_DEFINITION
    camera_component: object = CAMERA_DETECTOR_DEFINITION
    energy_filter_entrance_aperture_component: object = (
        ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION
    )
    objective_lens_component: object = OBJECTIVE_LENS_DEFINITION
    objective_aperture_component: object = (
        OBJECTIVE_APERTURE_DEFINITION
    )
    selected_area_aperture_component: object = (
        SELECTED_AREA_APERTURE_DEFINITION
    )
    objective_stigmator_component: object = (
        OBJECTIVE_STIGMATOR_DEFINITION
    )
    image_diffraction_deflector_component: object = (
        IMAGE_DIFFRACTION_DEFLECTOR_DEFINITION
    )
    descan_deflector_component: object = DESCAN_DEFLECTOR_DEFINITION
    adapter_lens_component: object = ADAPTER_LENS_DEFINITION
    dph2_deflector_component: object = DPH2_DEFLECTOR_DEFINITION
    dp22_deflector_component: object = DP22_DEFLECTOR_DEFINITION
    qph2_quadrupole_component: object = QPH2_QUADRUPOLE_DEFINITION
    hp2_hexapole_component: object = HP2_HEXAPOLE_DEFINITION
    hpc_hexapole_component: object = HPC_HEXAPOLE_DEFINITION
    probe_corrector_tail_components: tuple = (
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
    tl22_lens_component: object = TL22_LENS_DEFINITION
    image_corrector_components: tuple = field(
        default_factory=create_image_corrector_components
    )
    resolved_assembly: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        gun_keys = tuple(
            component.key for component in self.gun_components
        )
        if not gun_keys:
            raise ValueError("Layout requires one complete electron gun.")
        if (
            self.electron_gun_type == "thermionic"
            and gun_keys[0] == FEG_TIP
        ):
            from temsim.optics.electron_gun import create_electron_gun

            object.__setattr__(
                self,
                "gun_components",
                create_electron_gun("thermionic").components,
            )
        if (
            self.electron_gun_type == "cold_feg"
            and self.monochromator_installed
            and FEG_MONOCHROMATOR_WIEN not in gun_keys
        ):
            from temsim.optics.electron_gun import create_electron_gun

            gun = create_electron_gun("cold_feg")
            gun.install_monochromator()
            object.__setattr__(self, "gun_components", gun.components)
        if (
            self.monochromator_installed
            and self.electron_gun_type != "cold_feg"
        ):
            raise ValueError(
                "A monochromator can only be installed with the FEG source."
            )
        condenser_keys = tuple(
            component.key for component in self.condenser_components
        )
        if condenser_keys != (
            CONDENSER_LENS_1,
            CONDENSER_LENS_2,
            CONDENSER_LENS_3,
        ):
            raise ValueError(
                "Condenser components must contain C1, C2 and C3 exactly once "
                "in optical order."
            )
        if (
            self.condenser_aperture_2_component.key
            != CONDENSER_APERTURE_2
        ):
            raise ValueError("The C2 aperture component has a non-canonical key.")
        if (
            self.condenser_aperture_3_component.key
            != CONDENSER_APERTURE_3
        ):
            raise ValueError("The C3 aperture component has a non-canonical key.")
        if self.condenser_deflector_component.key != CONDENSER_DEFLECTOR:
            raise ValueError(
                "The condenser deflector component has a non-canonical key."
            )
        if self.beam_deflector_component.key != BEAM_DEFLECTOR:
            raise ValueError(
                "The beam deflector component has a non-canonical key."
            )
        if self.ac_deflector_component.key != AC_DEFLECTOR:
            raise ValueError(
                "The AC deflector component has a non-canonical key."
            )
        if self.mini_condenser_component.key != MINI_CONDENSER:
            raise ValueError(
                "The Mini Condenser component has a non-canonical key."
            )
        if (
            self.condenser_stigmator_component.key
            != CONDENSER_STIGMATOR
        ):
            raise ValueError(
                "The Condenser Stigmator has a non-canonical key."
            )
        if (
            self.diffraction_stigmator_component.key
            != DIFFRACTION_STIGMATOR
        ):
            raise ValueError(
                "The Diffraction Stigmator has a non-canonical key."
            )
        if self.diffraction_lens_component.key != DIFFRACTION_LENS:
            raise ValueError(
                "The Diffraction Lens has a non-canonical key."
            )
        if self.intermediate_lens_component.key != INTERMEDIATE_LENS:
            raise ValueError(
                "The Intermediate Lens has a non-canonical key."
            )
        if self.projector_lens_p1_component.key != PROJECTOR_LENS_1:
            raise ValueError(
                "Projector Lens P1 has a non-canonical key."
            )
        if self.projector_lens_p2_component.key != PROJECTOR_LENS_2:
            raise ValueError(
                "Projector Lens P2 has a non-canonical key."
            )
        if tuple(
            detector.key for detector in self.stem_detector_components
        ) != (
            HAADF_DETECTOR,
            DARK_FIELD_DETECTOR,
            BRIGHT_FIELD_DETECTOR,
        ):
            raise ValueError(
                "STEM detectors must contain HAADF, DF and BF in "
                "canonical order."
            )
        if self.fluorescent_screen_component.key != FLUORESCENT_SCREEN:
            raise ValueError(
                "The Fluorescent Screen has a non-canonical key."
            )
        if self.camera_component.key != CAMERA:
            raise ValueError("The Camera has a non-canonical key.")
        if (
            self.energy_filter_entrance_aperture_component.key
            != ENERGY_FILTER_ENTRANCE_APERTURE
        ):
            raise ValueError(
                "The Energy Filter Entrance Aperture has a "
                "non-canonical key."
            )
        if self.objective_lens_component.key != OBJECTIVE_LENS:
            raise ValueError(
                "The Objective Lens assembly has a non-canonical key."
            )
        if self.objective_aperture_component.key != OBJECTIVE_APERTURE:
            raise ValueError(
                "The Objective Aperture has a non-canonical key."
            )
        if (
            self.selected_area_aperture_component.key
            != SELECTED_AREA_APERTURE
        ):
            raise ValueError(
                "The Selected Area Aperture has a non-canonical key."
            )
        if self.objective_stigmator_component.key != OBJECTIVE_STIGMATOR:
            raise ValueError(
                "The Objective Stigmator has an invalid key."
            )
        if (
            self.image_diffraction_deflector_component.key
            != IMAGE_DIFFRACTION_DEFLECTOR
        ):
            raise ValueError(
                "The Image/Diffraction Deflector has a non-canonical key."
            )
        if self.descan_deflector_component.key != DESCAN_DEFLECTOR:
            raise ValueError(
                "The Descan Deflector has a non-canonical key."
            )
        if self.adapter_lens_component.key != ADAPTER_LENS:
            raise ValueError("The Adapter Lens has a non-canonical key.")
        if self.dph2_deflector_component.key != PROBE_DPH2_DEFLECTOR:
            raise ValueError("The DPH2 deflector has a non-canonical key.")
        if self.dp22_deflector_component.key != PROBE_DP22_DEFLECTOR:
            raise ValueError("The DP22 deflector has a non-canonical key.")
        if self.qph2_quadrupole_component.key != PROBE_QPH2_QUADRUPOLE:
            raise ValueError("The QPH2 quadrupole has a non-canonical key.")
        if self.hp2_hexapole_component.key != PROBE_HP2_HEXAPOLE:
            raise ValueError("The HP2 hexapole has a non-canonical key.")
        if self.hpc_hexapole_component.key != PROBE_HPC_HEXAPOLE:
            raise ValueError("The HPC hexapole has a non-canonical key.")
        if tuple(
            component.key for component in self.image_corrector_components
        ) != IMAGE_CORRECTOR_KEYS:
            raise ValueError(
                "Image Corrector components must be complete and canonical."
            )
        tail_keys = tuple(
            component.key
            for component in self.probe_corrector_tail_components
        )
        if tail_keys != (
            PROBE_QPC_QUADRUPOLE,
            PROBE_DP21_DEFLECTOR,
            PROBE_TL21_LENS,
            PROBE_DPH1_DEFLECTOR,
            PROBE_QPH1_QUADRUPOLE,
            PROBE_HP1_HEXAPOLE,
            PROBE_HPOL_HEXAPOLE,
            PROBE_QPOL_QUADRUPOLE,
            PROBE_DP11_DEFLECTOR,
            PROBE_TL12_LENS,
            PROBE_DP12_SCAN_DEFLECTOR,
        ):
            raise ValueError(
                "Probe Corrector tail components must use canonical keys "
                "in physical order."
            )
        if self.tl22_lens_component.key != PROBE_TL22_LENS:
            raise ValueError("The TL22 lens has a non-canonical key.")
        if not self.electron_gun_type:
            raise ValueError("Electron-gun type must not be empty.")
        if self.c3_hardware is C3Hardware.TWO_CONDENSER and self.c3_excited:
            raise ValueError("C3 cannot be excited when the C3 hardware is absent.")
        if (
            self.c3_hardware is C3Hardware.TWO_CONDENSER
            and self.corrector is not CorrectorAssembly.NO_CORRECTOR
        ):
            raise ValueError(
                "Correctors require three-condenser hardware."
            )


@dataclass(frozen=True)
class _LayoutSpec:
    key: str
    name: str
    kind: str
    owner: str
    center_z_mm: float
    length_mm: float
    branch: Branch = Branch.COMMON
    external_envelope: str = ""
    installed_if: Tuple[str, ...] = ()
    excitation_enabled: Optional[bool] = None
    note: str = ""
    shape_profile: str = "axial_envelope"
    outer_diameter_mm: Optional[float] = None
    active_diameter_mm: Optional[float] = None
    active_length_mm: Optional[float] = None
    optical_reference_offset_mm: float = 0.0
    optical_reference_plane_mm: Optional[float] = None
    effective_aperture_radius_mm: Optional[float] = None
    optical_interaction_planes_mm: Tuple[float, ...] = ()
    nested_parent_key: Optional[str] = None
    mechanical_overlap_reason: str = ""


def _image_corrector_specs(
    components=None,
    sample_center_from_source_mm=REFERENCE_SAMPLE_EFFECTIVE_Z_MM,
):
    runtime_components = {
        component.key: component
        for component in (
            components
            if components is not None
            else create_image_corrector_components()
        )
    }
    specs = {}
    for definition in IMAGE_CORRECTOR_COMPONENTS:
        component = runtime_components[definition.key]
        mechanical_below_sample_mm = (
            float(component.mechanical_center_from_tip_mm)
            - float(sample_center_from_source_mm)
        )
        specs[definition.key] = _LayoutSpec(
            definition.key,
            component.name,
            component.kind,
            "image_corrector",
            -mechanical_below_sample_mm,
            component.mechanical_length_mm,
            Branch.IMAGE,
            installed_if=_IMAGE_MODES,
            excitation_enabled=component.enabled,
            shape_profile=component.shape_profile,
            outer_diameter_mm=(
                component.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=definition.active_diameter_mm,
            active_length_mm=definition.active_length_mm,
            effective_aperture_radius_mm=(
                component.effective_aperture_radius_mm
            ),
            optical_reference_plane_mm=float(component.z_mm),
            optical_interaction_planes_mm=(
                (float(component.z_mm),)
                if bool(getattr(component, "optical_active", True))
                else ()
            ),
            note=(
                "Conjugate to the Objective Lens image plane."
                if definition.conjugate_to == "objective_image_plane"
                else ""
            ),
            mechanical_overlap_reason=(
                "The zero-thickness Image Corrector SAD optical reference "
                "plane may coincide with the Selected Area Aperture "
                "mechanism."
                if definition.key == IMAGE_CORRECTOR_SAD_PLANE
                else ""
            ),
        )
    return specs


_PROBE_MODES = ("probe_corrector", "double_corrector")
_IMAGE_MODES = ("image_corrector", "double_corrector")

_UPPER_SEQUENCE = (
    FEG_TIP,
    FEG_EXTRACTOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_MONOCHROMATOR_WIEN,
    FEG_ACCELERATOR,
    GUN_EXTRACTOR_APERTURE,
    FEG_DEFLECTOR,
    FEG_STIGMATOR,
    C1_APERTURE,
    THERMIONIC_CATHODE,
    THERMIONIC_WEHNELT,
    THERMIONIC_GUN_LENS,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_STIGMATOR,
    THERMIONIC_C1_APERTURE,
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_APERTURE_2,
    CONDENSER_DEFLECTOR,
    CONDENSER_LENS_3,
    CONDENSER_APERTURE_3,
    BEAM_DEFLECTOR,
    ADAPTER_LENS,
    PROBE_DPH2_DEFLECTOR,
    PROBE_QPH2_QUADRUPOLE,
    PROBE_HP2_HEXAPOLE,
    PROBE_TL22_LENS,
    PROBE_DP22_DEFLECTOR,
    PROBE_HPC_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE,
    PROBE_DP21_DEFLECTOR,
    PROBE_TL21_LENS,
    PROBE_DPH1_DEFLECTOR,
    PROBE_QPH1_QUADRUPOLE,
    PROBE_HP1_HEXAPOLE,
    PROBE_HPOL_HEXAPOLE,
    PROBE_QPOL_QUADRUPOLE,
    PROBE_DP11_DEFLECTOR,
    PROBE_TL12_LENS,
    PROBE_DP12_SCAN_DEFLECTOR,
    CONDENSER_STIGMATOR,
    AC_DEFLECTOR,
    MINI_CONDENSER,
    "objective_upper_pole",
    "sample",
)

_LOWER_SEQUENCE = (
    "sample",
    OBJECTIVE_APERTURE,
    "objective_lower_pole",
    DESCAN_DEFLECTOR,
    "objective_stigmator",
    "image_diffraction_deflector",
    SELECTED_AREA_APERTURE,
    DIFFRACTION_STIGMATOR,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    HAADF_DETECTOR,
    FLUORESCENT_SCREEN,
    DARK_FIELD_DETECTOR,
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    "energy_filter",
)

_IMAGE_CORRECTOR_SEQUENCE = (
    "sample",
    OBJECTIVE_APERTURE,
    "objective_lower_pole",
    DESCAN_DEFLECTOR,
    "objective_stigmator",
    IMAGE_DIFFRACTION_DEFLECTOR,
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_HPOL_HEXAPOLE,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    IMAGE_CORRECTOR_DP11_DEFLECTOR,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_TL22_LENS,
    IMAGE_CORRECTOR_HP2_HEXAPOLE,
    IMAGE_CORRECTOR_ADAPTER_LENS,
    IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR,
    IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_SAD_PLANE,
    SELECTED_AREA_APERTURE,
    DIFFRACTION_STIGMATOR,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    HAADF_DETECTOR,
    FLUORESCENT_SCREEN,
    DARK_FIELD_DETECTOR,
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    "energy_filter",
)


def _mini_condenser_spec(configuration, component, gun_anchor_z_mm):
    installation = (
        "integrated"
        if configuration.corrector in (
            CorrectorAssembly.PROBE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        )
        else "standalone"
    )
    geometry = component.geometry_for(installation)
    return _LayoutSpec(
        component.key,
        component.name if hasattr(component, "name") else component.label,
        component.kind,
        geometry.owner,
        gun_anchor_z_mm - geometry.mechanical_center_from_tip_mm,
        geometry.mechanical_length_mm,
        Branch.ILLUMINATION,
        shape_profile=component.shape_profile,
        outer_diameter_mm=geometry.mechanical_outer_diameter_mm,
        active_diameter_mm=geometry.bore_diameter_mm,
        active_length_mm=geometry.pole_gap_mm,
        optical_reference_plane_mm=(
            geometry.optical_reference_from_tip_mm
        ),
        effective_aperture_radius_mm=geometry.bore_diameter_mm / 2.0,
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Mini Condenser Lens is embedded inside the Upper "
            "Objective Lens body above its pole piece."
        ),
    )


def _condenser_stigmator_spec(component):
    effective_length_mm = (
        component.length_mm
        if hasattr(component, "length_mm")
        else component.effective_length_mm
    )
    optical_reference_z_mm = (
        component.z_mm
        if hasattr(component, "z_mm")
        else component.optical_reference_z_mm
    )
    return _LayoutSpec(
        component.key,
        component.name if hasattr(component, "name") else component.label,
        component.kind,
        component.owner,
        component.mechanical_center_above_sample_mm,
        component.mechanical_length_mm,
        Branch.ILLUMINATION,
        shape_profile=component.shape_profile,
        outer_diameter_mm=component.mechanical_outer_diameter_mm,
        active_diameter_mm=(
            component.mechanical_clear_bore_diameter_mm
        ),
        active_length_mm=effective_length_mm,
        optical_reference_plane_mm=optical_reference_z_mm,
        effective_aperture_radius_mm=(
            component.effective_aperture_radius_mm
        ),
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Condenser Stigmator anchors the embedded Upper "
            "Objective Lens component package."
        ),
    )


def _objective_stigmator_spec(component):
    effective_length_mm = (
        component.length_mm
        if hasattr(component, "length_mm")
        else component.effective_length_mm
    )
    optical_reference_z_mm = (
        component.z_mm
        if hasattr(component, "z_mm")
        else component.optical_reference_z_mm
    )
    return _LayoutSpec(
        "objective_stigmator",
        component.name if hasattr(component, "name") else component.label,
        component.kind,
        component.owner,
        -component.mechanical_center_below_sample_mm,
        component.mechanical_length_mm,
        Branch.IMAGE,
        shape_profile=component.shape_profile,
        outer_diameter_mm=component.mechanical_outer_diameter_mm,
        active_diameter_mm=(
            component.mechanical_clear_bore_diameter_mm
        ),
        active_length_mm=effective_length_mm,
        optical_reference_plane_mm=optical_reference_z_mm,
        effective_aperture_radius_mm=(
            component.effective_aperture_radius_mm
        ),
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Objective Stigmator is installed inside the Lower "
            "Objective Lens assembly bore."
        ),
    )


def _image_diffraction_deflector_spec(component):
    upper_z_mm = (
        component.upper_z_mm
        if hasattr(component, "upper_z_mm")
        else component.optical_upper_reference_z_mm
    )
    lower_z_mm = (
        component.lower_z_mm
        if hasattr(component, "lower_z_mm")
        else component.optical_lower_reference_z_mm
    )
    thickness_mm = (
        component.thickness_mm
        if hasattr(component, "thickness_mm")
        else component.effective_coil_thickness_mm
    )
    return _LayoutSpec(
        IMAGE_DIFFRACTION_DEFLECTOR,
        component.name if hasattr(component, "name") else component.label,
        component.kind,
        component.owner,
        -component.mechanical_center_below_sample_mm,
        component.mechanical_length_mm,
        Branch.IMAGE,
        shape_profile=component.shape_profile,
        outer_diameter_mm=component.mechanical_outer_diameter_mm,
        active_diameter_mm=(
            component.mechanical_clear_bore_diameter_mm
        ),
        active_length_mm=thickness_mm,
        optical_reference_plane_mm=(
            0.5 * (upper_z_mm + lower_z_mm)
        ),
        optical_interaction_planes_mm=(upper_z_mm, lower_z_mm),
        effective_aperture_radius_mm=(
            component.effective_aperture_radius_mm
        ),
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Image / Diffraction Deflector is installed inside the "
            "Lower Objective Lens assembly bore."
        ),
    )


def _descan_deflector_spec(component):
    optical_z_mm = (
        component.z_mm
        if hasattr(component, "z_mm")
        else component.optical_reference_z_mm
    )
    thickness_mm = component.effective_thickness_mm
    half_separation_mm = (
        component.optical_plane_separation_mm / 2.0
    )
    return _LayoutSpec(
        DESCAN_DEFLECTOR,
        component.name if hasattr(component, "name") else component.label,
        component.kind,
        component.owner,
        -component.mechanical_center_below_sample_mm,
        component.mechanical_length_mm,
        Branch.IMAGE,
        shape_profile=component.shape_profile,
        outer_diameter_mm=component.mechanical_outer_diameter_mm,
        active_diameter_mm=component.mechanical_clear_bore_diameter_mm,
        active_length_mm=thickness_mm,
        optical_reference_plane_mm=optical_z_mm,
        optical_interaction_planes_mm=(
            optical_z_mm - half_separation_mm,
            optical_z_mm + half_separation_mm,
        ),
        effective_aperture_radius_mm=(
            component.effective_aperture_radius_mm
        ),
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The AC Descan coils are installed inside the Lower "
            "Objective Lens assembly bore."
        ),
    )


def _objective_lens_spec(component):
    label = (
        component.name if hasattr(component, "name") else component.label
    )
    length_mm = component.assembly_length_mm
    outer_diameter_mm = component.assembly_outer_diameter_mm
    virtual_z_mm = (
        component.z_mm
        if hasattr(component, "z_mm")
        else 1149.50005
    )
    back_focal_plane_z_mm = getattr(
        component, "_back_focal_plane_z_mm", None
    )
    return _LayoutSpec(
        OBJECTIVE_LENS,
        label,
        "round_lens",
        "objective",
        0.0,
        length_mm,
        Branch.COMMON,
        shape_profile="magnetic_lens_yoke",
        outer_diameter_mm=outer_diameter_mm,
        active_diameter_mm=component.pole_piece_bore_diameter_mm,
        active_length_mm=(
            component.upper_pole_piece_axial_length_mm
            + component.inner_face_gap_mm
            + component.pole_piece_axial_length_mm
        ),
        optical_reference_plane_mm=virtual_z_mm,
        optical_interaction_planes_mm=tuple(
            value
            for value in (virtual_z_mm, back_focal_plane_z_mm)
            if value is not None
        ),
        effective_aperture_radius_mm=(
            component.pole_piece_bore_diameter_mm / 2.0
        ),
    )


def _base_specs(configuration):
    gun = {component.key: component for component in configuration.gun_components}
    gun_anchor_z_mm = 1392.0

    def gun_spec(key):
        component = gun[key]
        nominal_center = (
            gun_anchor_z_mm - component.mechanical_center_from_tip_mm
        )
        nested_parent_key = None
        mechanical_overlap_reason = ""
        if key == GUN_EXTRACTOR_APERTURE:
            nested_parent_key = FEG_ACCELERATOR
            mechanical_overlap_reason = (
                "The DPA aperture is mounted inside the accelerator envelope."
            )
        elif key == THERMIONIC_ANODE_APERTURE:
            nested_parent_key = THERMIONIC_ACCELERATOR
            mechanical_overlap_reason = (
                "The anode aperture is mounted inside the accelerator envelope."
            )
        return _LayoutSpec(
            component.key,
            component.label,
            component.kind,
            configuration.electron_gun_type,
            nominal_center,
            component.mechanical_length_mm,
            external_envelope=(
                "D250-350 mm"
                if key in (FEG_ACCELERATOR, THERMIONIC_ACCELERATOR)
                else ""
            ),
            shape_profile=component.shape_profile,
            outer_diameter_mm=component.mechanical_outer_diameter_mm,
            optical_reference_offset_mm=(
                component.optical_reference_from_tip_mm - nominal_center
            ),
            optical_reference_plane_mm=(
                component.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                component.effective_aperture_radius_mm
            ),
            nested_parent_key=nested_parent_key,
            mechanical_overlap_reason=mechanical_overlap_reason,
        )

    condenser = {
        component.key: component
        for component in configuration.condenser_components
    }
    condenser_aperture_2 = configuration.condenser_aperture_2_component
    condenser_aperture_3 = configuration.condenser_aperture_3_component
    condenser_deflector = configuration.condenser_deflector_component
    beam_deflector = configuration.beam_deflector_component
    ac_deflector = configuration.ac_deflector_component
    mini_condenser = configuration.mini_condenser_component
    condenser_stigmator = (
        configuration.condenser_stigmator_component
    )
    selected_area_aperture = (
        configuration.selected_area_aperture_component
    )
    selected_area_installation = (
        IMAGE_CORRECTED_INSTALLATION
        if configuration.corrector in (
            CorrectorAssembly.IMAGE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        )
        else STANDALONE_INSTALLATION
    )
    selected_area_geometry = selected_area_aperture.geometry_for(
        selected_area_installation
    )
    diffraction_stigmator = (
        configuration.diffraction_stigmator_component
    )
    diffraction_stigmator_installation = (
        DIFFRACTION_IMAGE_INSTALLATION
        if configuration.corrector in (
            CorrectorAssembly.IMAGE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        )
        else DIFFRACTION_STANDALONE_INSTALLATION
    )
    diffraction_stigmator_geometry = diffraction_stigmator.geometry_for(
        diffraction_stigmator_installation
    )
    diffraction_lens = configuration.diffraction_lens_component
    diffraction_lens_installation = (
        DIFFRACTION_LENS_IMAGE_INSTALLATION
        if configuration.corrector in (
            CorrectorAssembly.IMAGE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        )
        else DIFFRACTION_LENS_STANDALONE_INSTALLATION
    )
    diffraction_lens_geometry = diffraction_lens.geometry_for(
        diffraction_lens_installation
    )
    intermediate_lens = configuration.intermediate_lens_component
    intermediate_lens_geometry = intermediate_lens.geometry_for(
        selected_area_geometry
    )
    projector_lens_p1 = configuration.projector_lens_p1_component
    projector_lens_p1_geometry = projector_lens_p1.geometry_for(
        selected_area_geometry
    )
    projector_lens_p2 = configuration.projector_lens_p2_component
    projector_lens_p2_geometry = projector_lens_p2.geometry_for(
        selected_area_geometry
    )
    stem_detectors = {
        detector.key: detector
        for detector in configuration.stem_detector_components
    }
    fluorescent_screen = configuration.fluorescent_screen_component
    camera = configuration.camera_component
    energy_filter_entrance_aperture = (
        configuration.energy_filter_entrance_aperture_component
    )

    def selected_area_downstream_center(offset_below_anchor_mm):
        return -(
            selected_area_geometry.mechanical_center_below_sample_mm
            + float(offset_below_anchor_mm)
        )
    objective_lens = configuration.objective_lens_component
    objective_aperture = configuration.objective_aperture_component
    objective_stigmator = configuration.objective_stigmator_component
    image_diffraction_deflector = (
        configuration.image_diffraction_deflector_component
    )
    descan_deflector = configuration.descan_deflector_component
    adapter_lens = configuration.adapter_lens_component
    dph2_deflector = configuration.dph2_deflector_component
    dp22_deflector = configuration.dp22_deflector_component
    qph2_quadrupole = configuration.qph2_quadrupole_component
    hp2_hexapole = configuration.hp2_hexapole_component
    hpc_hexapole = configuration.hpc_hexapole_component
    probe_corrector_tail = (
        configuration.probe_corrector_tail_components
    )
    tl22_lens = configuration.tl22_lens_component

    def condenser_spec(key):
        component = condenser[key]
        nominal_center = (
            gun_anchor_z_mm
            - component.mechanical_center_from_tip_mm
        )
        return _LayoutSpec(
            component.key,
            component.label,
            "magnetic_lens",
            component.owner,
            nominal_center,
            component.mechanical_length_mm,
            external_envelope=(
                f"D{component.mechanical_outer_diameter_mm:g} mm; "
                f"bore D{component.bore_diameter_mm:g} mm"
            ),
            shape_profile=component.shape_profile,
            outer_diameter_mm=component.mechanical_outer_diameter_mm,
            active_diameter_mm=component.bore_diameter_mm,
            active_length_mm=component.pole_gap_mm,
            optical_reference_plane_mm=(
                component.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                component.effective_aperture_radius_mm
            ),
            installed_if=(
                ("three_condenser",)
                if key == CONDENSER_LENS_3
                else ()
            ),
        )

    def condenser_pole_spec(lens_key, *, upper):
        component = condenser[lens_key]
        pole_length = 0.5 * max(
            component.mechanical_length_mm - component.pole_gap_mm, 0.001
        )
        key = f"{lens_key}_{'upper' if upper else 'lower'}_pole"
        return _LayoutSpec(
            key,
            f"{component.label} {'Upper' if upper else 'Lower'} Pole Piece",
            "pole_piece",
            component.owner,
            0.0,
            pole_length,
            installed_if=(
                ("three_condenser",)
                if lens_key == CONDENSER_LENS_3 else ()
            ),
            shape_profile="magnetic_pole_piece",
            outer_diameter_mm=0.67 * component.mechanical_outer_diameter_mm,
            active_diameter_mm=component.bore_diameter_mm,
            active_length_mm=max(2.0 * component.bore_diameter_mm, 1.0),
            nested_parent_key=lens_key,
            mechanical_overlap_reason=(
                f"The pole piece is part of the {component.label} assembly."
            ),
        )

    def probe_tail_spec(component):
        layout_only = (
            component.key == PROBE_DP12_SCAN_DEFLECTOR
        )
        optical_reference = getattr(
            component,
            "optical_reference_from_tip_mm",
            getattr(component, "optical_center_from_tip_mm", 0.0),
        )
        active_length = getattr(
            component,
            "effective_length_mm",
            getattr(
                component,
                "effective_thickness_mm",
                getattr(
                    component,
                    "pole_gap_mm",
                    getattr(component, "thickness_mm", 0.0),
                ),
            ),
        )
        active_diameter = getattr(
            component,
            "mechanical_clear_bore_diameter_mm",
            getattr(component, "bore_diameter_mm", 0.0),
        )
        interaction_planes = ()
        if (
            not layout_only
            and hasattr(component, "optical_upper_reference_from_tip_mm")
        ):
            interaction_planes = (
                component.optical_upper_reference_from_tip_mm,
                component.optical_lower_reference_from_tip_mm,
            )
        return _LayoutSpec(
            component.key,
            component.name
            if hasattr(component, "name")
            else component.label,
            component.kind,
            component.owner,
            gun_anchor_z_mm
            - component.mechanical_center_from_tip_mm,
            component.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=component.shape_profile,
            outer_diameter_mm=(
                component.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                0.0 if layout_only else active_diameter
            ),
            active_length_mm=0.0 if layout_only else active_length,
            optical_reference_plane_mm=optical_reference,
            optical_interaction_planes_mm=interaction_planes,
            effective_aperture_radius_mm=(
                None
                if layout_only
                else component.effective_aperture_radius_mm
            ),
            nested_parent_key=None,
            mechanical_overlap_reason="",
        )

    specs = {
        component.key: gun_spec(component.key)
        for component in configuration.gun_components
    }
    specs.update({
        CONDENSER_LENS_1: condenser_spec(CONDENSER_LENS_1),
        CONDENSER_LENS_1_LOWER_POLE: condenser_pole_spec(
            CONDENSER_LENS_1, upper=False
        ),
        CONDENSER_LENS_2: condenser_spec(CONDENSER_LENS_2),
        CONDENSER_LENS_2_UPPER_POLE: condenser_pole_spec(
            CONDENSER_LENS_2, upper=True
        ),
        CONDENSER_APERTURE_2: _LayoutSpec(
            CONDENSER_APERTURE_2,
            condenser_aperture_2.name
            if hasattr(condenser_aperture_2, "name")
            else condenser_aperture_2.label,
            "continuous_aperture",
            condenser_aperture_2.owner,
            (
                gun_anchor_z_mm
                - condenser_aperture_2.mechanical_center_from_tip_mm
            ),
            condenser_aperture_2.mechanical_length_mm,
            shape_profile=condenser_aperture_2.shape_profile,
            outer_diameter_mm=(
                condenser_aperture_2.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                2.0
                * condenser_aperture_2.effective_aperture_radius_mm
            ),
            active_length_mm=condenser_aperture_2.plate_thickness_mm,
            optical_reference_plane_mm=(
                condenser_aperture_2.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                condenser_aperture_2.effective_aperture_radius_mm
            ),
        ),
        CONDENSER_DEFLECTOR: _LayoutSpec(
            CONDENSER_DEFLECTOR,
            condenser_deflector.name
            if hasattr(condenser_deflector, "name")
            else condenser_deflector.label,
            condenser_deflector.kind,
            condenser_deflector.owner,
            (
                gun_anchor_z_mm
                - condenser_deflector.mechanical_center_from_tip_mm
            ),
            condenser_deflector.mechanical_length_mm,
            installed_if=("three_condenser",),
            shape_profile=condenser_deflector.shape_profile,
            outer_diameter_mm=(
                condenser_deflector.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                condenser_deflector.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=condenser_deflector.thickness_mm
            if hasattr(condenser_deflector, "thickness_mm")
            else condenser_deflector.effective_coil_thickness_mm,
            optical_reference_plane_mm=(
                condenser_deflector.optical_center_from_tip_mm
                if hasattr(
                    condenser_deflector,
                    "optical_center_from_tip_mm",
                )
                else 0.5 * (
                    condenser_deflector
                    .optical_upper_reference_from_tip_mm
                    + condenser_deflector
                    .optical_lower_reference_from_tip_mm
                )
            ),
            optical_interaction_planes_mm=(
                condenser_deflector
                .optical_upper_reference_from_tip_mm,
                condenser_deflector
                .optical_lower_reference_from_tip_mm,
            ),
        ),
        CONDENSER_LENS_3: condenser_spec(CONDENSER_LENS_3),
        CONDENSER_LENS_3_UPPER_POLE: condenser_pole_spec(
            CONDENSER_LENS_3, upper=True
        ),
        CONDENSER_LENS_3_LOWER_POLE: condenser_pole_spec(
            CONDENSER_LENS_3, upper=False
        ),
        CONDENSER_APERTURE_3: _LayoutSpec(
            CONDENSER_APERTURE_3,
            condenser_aperture_3.name
            if hasattr(condenser_aperture_3, "name")
            else condenser_aperture_3.label,
            "continuous_aperture",
            condenser_aperture_3.owner,
            (
                gun_anchor_z_mm
                - condenser_aperture_3.mechanical_center_from_tip_mm
            ),
            condenser_aperture_3.mechanical_length_mm,
            installed_if=("three_condenser",),
            shape_profile=condenser_aperture_3.shape_profile,
            outer_diameter_mm=(
                condenser_aperture_3.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                2.0
                * condenser_aperture_3.effective_aperture_radius_mm
            ),
            active_length_mm=condenser_aperture_3.plate_thickness_mm,
            optical_reference_plane_mm=(
                condenser_aperture_3.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                condenser_aperture_3.effective_aperture_radius_mm
            ),
        ),
        BEAM_DEFLECTOR: _LayoutSpec(
            BEAM_DEFLECTOR,
            beam_deflector.name
            if hasattr(beam_deflector, "name")
            else beam_deflector.label,
            beam_deflector.kind,
            beam_deflector.owner,
            (
                gun_anchor_z_mm
                - beam_deflector.mechanical_center_from_tip_mm
            ),
            beam_deflector.mechanical_length_mm,
            shape_profile=beam_deflector.shape_profile,
            outer_diameter_mm=(
                beam_deflector.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                beam_deflector.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=beam_deflector.thickness_mm
            if hasattr(beam_deflector, "thickness_mm")
            else beam_deflector.effective_coil_thickness_mm,
            optical_reference_plane_mm=(
                beam_deflector.optical_center_from_tip_mm
                if hasattr(beam_deflector, "optical_center_from_tip_mm")
                else 0.5 * (
                    beam_deflector.optical_upper_reference_from_tip_mm
                    + beam_deflector.optical_lower_reference_from_tip_mm
                )
            ),
            optical_interaction_planes_mm=(
                beam_deflector.optical_upper_reference_from_tip_mm,
                beam_deflector.optical_lower_reference_from_tip_mm,
            ),
        ),
        ADAPTER_LENS: _LayoutSpec(
            ADAPTER_LENS,
            adapter_lens.name
            if hasattr(adapter_lens, "name")
            else adapter_lens.label,
            adapter_lens.kind,
            adapter_lens.owner,
            gun_anchor_z_mm
            - adapter_lens.mechanical_center_from_tip_mm,
            adapter_lens.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=adapter_lens.shape_profile,
            outer_diameter_mm=adapter_lens.mechanical_outer_diameter_mm,
            active_diameter_mm=adapter_lens.bore_diameter_mm,
            active_length_mm=adapter_lens.pole_gap_mm,
            optical_reference_plane_mm=(
                adapter_lens.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                adapter_lens.effective_aperture_radius_mm
                if hasattr(adapter_lens, "effective_aperture_radius_mm")
                else adapter_lens.bore_diameter_mm / 2.0
            ),
        ),
        PROBE_DPH2_DEFLECTOR: _LayoutSpec(
            PROBE_DPH2_DEFLECTOR,
            dph2_deflector.name
            if hasattr(dph2_deflector, "name")
            else dph2_deflector.label,
            dph2_deflector.kind,
            dph2_deflector.owner,
            gun_anchor_z_mm
            - dph2_deflector.mechanical_center_from_tip_mm,
            dph2_deflector.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=dph2_deflector.shape_profile,
            outer_diameter_mm=(
                dph2_deflector.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                dph2_deflector.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=dph2_deflector.effective_thickness_mm,
            optical_reference_plane_mm=(
                dph2_deflector.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                dph2_deflector.effective_aperture_radius_mm
                if hasattr(dph2_deflector, "effective_aperture_radius_mm")
                else (
                    dph2_deflector.mechanical_clear_bore_diameter_mm
                    / 2.0
                )
            ),
        ),
        PROBE_QPH2_QUADRUPOLE: _LayoutSpec(
            PROBE_QPH2_QUADRUPOLE,
            qph2_quadrupole.name
            if hasattr(qph2_quadrupole, "name")
            else qph2_quadrupole.label,
            qph2_quadrupole.kind,
            qph2_quadrupole.owner,
            gun_anchor_z_mm
            - qph2_quadrupole.mechanical_center_from_tip_mm,
            qph2_quadrupole.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=qph2_quadrupole.shape_profile,
            outer_diameter_mm=(
                qph2_quadrupole.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                qph2_quadrupole.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=qph2_quadrupole.effective_length_mm,
            optical_reference_plane_mm=(
                qph2_quadrupole.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                qph2_quadrupole.effective_aperture_radius_mm
            ),
        ),
        PROBE_HP2_HEXAPOLE: _LayoutSpec(
            PROBE_HP2_HEXAPOLE,
            hp2_hexapole.name
            if hasattr(hp2_hexapole, "name")
            else hp2_hexapole.label,
            hp2_hexapole.kind,
            hp2_hexapole.owner,
            gun_anchor_z_mm
            - hp2_hexapole.mechanical_center_from_tip_mm,
            hp2_hexapole.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=hp2_hexapole.shape_profile,
            outer_diameter_mm=(
                hp2_hexapole.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                hp2_hexapole.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=hp2_hexapole.effective_length_mm,
            optical_reference_plane_mm=(
                hp2_hexapole.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                hp2_hexapole.effective_aperture_radius_mm
            ),
        ),
        PROBE_HPC_HEXAPOLE: _LayoutSpec(
            PROBE_HPC_HEXAPOLE,
            hpc_hexapole.name
            if hasattr(hpc_hexapole, "name")
            else hpc_hexapole.label,
            hpc_hexapole.kind,
            hpc_hexapole.owner,
            gun_anchor_z_mm
            - hpc_hexapole.mechanical_center_from_tip_mm,
            hpc_hexapole.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=hpc_hexapole.shape_profile,
            outer_diameter_mm=(
                hpc_hexapole.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                hpc_hexapole.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=hpc_hexapole.effective_length_mm,
            optical_reference_plane_mm=(
                hpc_hexapole.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                hpc_hexapole.effective_aperture_radius_mm
            ),
        ),
        **{
            component.key: probe_tail_spec(component)
            for component in probe_corrector_tail
        },
        PROBE_TL22_LENS: _LayoutSpec(
            PROBE_TL22_LENS,
            tl22_lens.name
            if hasattr(tl22_lens, "name")
            else tl22_lens.label,
            tl22_lens.kind,
            tl22_lens.owner,
            gun_anchor_z_mm
            - tl22_lens.mechanical_center_from_tip_mm,
            tl22_lens.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=tl22_lens.shape_profile,
            outer_diameter_mm=tl22_lens.mechanical_outer_diameter_mm,
            active_diameter_mm=tl22_lens.bore_diameter_mm,
            active_length_mm=tl22_lens.pole_gap_mm,
            optical_reference_plane_mm=(
                tl22_lens.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                tl22_lens.effective_aperture_radius_mm
            ),
        ),
        PROBE_DP22_DEFLECTOR: _LayoutSpec(
            PROBE_DP22_DEFLECTOR,
            dp22_deflector.name
            if hasattr(dp22_deflector, "name")
            else dp22_deflector.label,
            dp22_deflector.kind,
            dp22_deflector.owner,
            gun_anchor_z_mm
            - dp22_deflector.mechanical_center_from_tip_mm,
            dp22_deflector.mechanical_length_mm,
            Branch.ILLUMINATION,
            installed_if=_PROBE_MODES,
            shape_profile=dp22_deflector.shape_profile,
            outer_diameter_mm=(
                dp22_deflector.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                dp22_deflector.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=dp22_deflector.effective_thickness_mm,
            optical_reference_plane_mm=(
                dp22_deflector.optical_reference_from_tip_mm
            ),
            effective_aperture_radius_mm=(
                dp22_deflector.effective_aperture_radius_mm
            ),
        ),
        AC_DEFLECTOR: _LayoutSpec(
            AC_DEFLECTOR,
            ac_deflector.name
            if hasattr(ac_deflector, "name")
            else ac_deflector.label,
            ac_deflector.kind,
            ac_deflector.owner,
            gun_anchor_z_mm
            - ac_deflector.mechanical_center_from_tip_mm,
            ac_deflector.mechanical_length_mm,
            Branch.ILLUMINATION,
            shape_profile=ac_deflector.shape_profile,
            outer_diameter_mm=(
                ac_deflector.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                ac_deflector.mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=ac_deflector.effective_thickness_mm,
            optical_reference_plane_mm=(
                ac_deflector.optical_reference_from_tip_mm
            ),
            optical_interaction_planes_mm=(
                ac_deflector.upper_z_mm,
                ac_deflector.lower_z_mm,
            )
            if hasattr(ac_deflector, "upper_z_mm")
            else (),
            effective_aperture_radius_mm=(
                ac_deflector.effective_aperture_radius_mm
            ),
            nested_parent_key=OBJECTIVE_LENS,
            mechanical_overlap_reason=(
                "The AC Scan Coil is embedded inside the Upper "
                "Objective Lens component package."
            ),
        ),
        MINI_CONDENSER: _mini_condenser_spec(
            configuration, mini_condenser, gun_anchor_z_mm
        ),
        CONDENSER_STIGMATOR: _condenser_stigmator_spec(
            condenser_stigmator
        ),
        OBJECTIVE_LENS: _objective_lens_spec(objective_lens),
        "objective_upper_pole": _LayoutSpec(
            "objective_upper_pole", "Upper Objective", "magnetic_pole_piece",
            "objective", 10.0,
            objective_lens.upper_pole_piece_axial_length_mm,
            shape_profile="magnetic_pole_piece",
            outer_diameter_mm=(
                objective_lens.upper_pole_piece_outer_diameter_mm
            ),
            active_diameter_mm=(
                objective_lens.pole_piece_bore_diameter_mm
            ),
            active_length_mm=(
                objective_lens.upper_pole_piece_tip_diameter_mm
            ),
            nested_parent_key=OBJECTIVE_LENS,
            mechanical_overlap_reason=(
                "The Upper Objective pole piece is part of the "
                "Objective Lens assembly."
            ),
        ),
        "sample_stage": _LayoutSpec(
            "sample_stage", "Sample Stage / Goniometer", "stage", "objective",
            0.0, 260.0, external_envelope="650 x 500 mm transverse envelope",
            nested_parent_key=OBJECTIVE_LENS,
            mechanical_overlap_reason=(
                "The sample stage crosses the Objective Lens assembly "
                "through its central access plane."
            ),
        ),
        "sample": _LayoutSpec(
            "sample", "Sample", "sample", "objective", 0.0,
            configuration.objective.specimen_thickness_mm,
            shape_profile="specimen_slab",
            outer_diameter_mm=3.0,
            nested_parent_key="sample_stage",
            mechanical_overlap_reason=(
                "The specimen is mounted inside the sample stage."
            ),
        ),
        "objective_lower_pole": _LayoutSpec(
            "objective_lower_pole", "Lower Objective", "magnetic_pole_piece",
            "objective", -10.0,
            objective_lens.pole_piece_axial_length_mm,
            shape_profile="magnetic_pole_piece",
            outer_diameter_mm=(
                objective_lens.pole_piece_outer_diameter_mm
            ),
            active_diameter_mm=(
                objective_lens.pole_piece_bore_diameter_mm
            ),
            active_length_mm=(
                objective_lens.pole_piece_tip_diameter_mm
            ),
            nested_parent_key=OBJECTIVE_LENS,
            mechanical_overlap_reason=(
                "The Lower Objective pole piece is part of the "
                "Objective Lens assembly."
            ),
        ),
        OBJECTIVE_APERTURE: _LayoutSpec(
            OBJECTIVE_APERTURE,
            objective_aperture.name
            if hasattr(objective_aperture, "name")
            else objective_aperture.label,
            objective_aperture.kind,
            objective_aperture.owner,
            -objective_aperture.mechanical_center_below_sample_mm,
            objective_aperture.mechanical_length_mm,
            shape_profile=objective_aperture.shape_profile,
            outer_diameter_mm=(
                objective_aperture.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                2.0 * objective_aperture.radius_mm
            ),
            active_length_mm=(
                objective_aperture.plate_thickness_mm
            ),
            optical_reference_plane_mm=(
                objective_aperture.z_mm
            ),
            effective_aperture_radius_mm=(
                objective_aperture.radius_mm
            ),
        ),
        "objective_stigmator": _objective_stigmator_spec(
            objective_stigmator
        ),
        IMAGE_DIFFRACTION_DEFLECTOR: (
            _image_diffraction_deflector_spec(
                image_diffraction_deflector
            )
        ),
        DESCAN_DEFLECTOR: _descan_deflector_spec(descan_deflector),
        SELECTED_AREA_APERTURE: _LayoutSpec(
            SELECTED_AREA_APERTURE,
            selected_area_aperture.name
            if hasattr(selected_area_aperture, "name")
            else selected_area_aperture.label,
            selected_area_aperture.kind,
            selected_area_aperture.owner,
            -selected_area_geometry.mechanical_center_below_sample_mm,
            selected_area_geometry.mechanical_length_mm,
            shape_profile=selected_area_aperture.shape_profile,
            outer_diameter_mm=(
                selected_area_geometry.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                2.0 * selected_area_aperture.radius_mm
            ),
            active_length_mm=selected_area_geometry.plate_thickness_mm,
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
            ),
            effective_aperture_radius_mm=(
                selected_area_aperture.radius_mm
            ),
        ),
        DIFFRACTION_STIGMATOR: _LayoutSpec(
            DIFFRACTION_STIGMATOR,
            diffraction_stigmator.name,
            diffraction_stigmator.kind,
            diffraction_stigmator.owner,
            selected_area_downstream_center(
                diffraction_stigmator
                .mechanical_center_downstream_of_anchor_mm
            ),
            diffraction_stigmator_geometry.mechanical_length_mm,
            shape_profile=diffraction_stigmator.shape_profile,
            outer_diameter_mm=(
                diffraction_stigmator_geometry
                .mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                diffraction_stigmator_geometry
                .mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=(
                diffraction_stigmator.length_mm
                if hasattr(diffraction_stigmator, "length_mm")
                else diffraction_stigmator.effective_length_mm
            ),
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + diffraction_stigmator
                .mechanical_center_downstream_of_anchor_mm
            ),
            effective_aperture_radius_mm=(
                diffraction_stigmator.effective_aperture_radius_mm
            ),
        ),
        DIFFRACTION_LENS: _LayoutSpec(
            DIFFRACTION_LENS,
            diffraction_lens.name,
            "magnetic_lens",
            diffraction_lens.owner,
            selected_area_downstream_center(
                diffraction_lens
                .mechanical_center_downstream_of_anchor_mm
            ),
            diffraction_lens_geometry.mechanical_length_mm,
            external_envelope="D220-280 mm (provisional)",
            shape_profile=diffraction_lens.shape_profile,
            outer_diameter_mm=(
                diffraction_lens_geometry.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                diffraction_lens_geometry
                .mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=diffraction_lens_geometry.pole_gap_mm,
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + diffraction_lens
                .mechanical_center_downstream_of_anchor_mm
            ),
            effective_aperture_radius_mm=(
                diffraction_lens.effective_aperture_radius_mm
            ),
        ),
        INTERMEDIATE_LENS: _LayoutSpec(
            INTERMEDIATE_LENS,
            intermediate_lens.name,
            "magnetic_lens",
            intermediate_lens.owner,
            -intermediate_lens_geometry.mechanical_center_below_sample_mm,
            intermediate_lens_geometry.mechanical_length_mm,
            external_envelope="D240-300 mm (provisional)",
            shape_profile=intermediate_lens.shape_profile,
            outer_diameter_mm=(
                intermediate_lens_geometry.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                intermediate_lens_geometry
                .mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=intermediate_lens_geometry.pole_gap_mm,
            optical_reference_plane_mm=(
                intermediate_lens_geometry.optical_reference_z_mm
            ),
            effective_aperture_radius_mm=(
                intermediate_lens.effective_aperture_radius_mm
            ),
        ),
        PROJECTOR_LENS_1: _LayoutSpec(
            PROJECTOR_LENS_1,
            projector_lens_p1.name,
            "magnetic_lens",
            projector_lens_p1.owner,
            -projector_lens_p1_geometry.mechanical_center_below_sample_mm,
            projector_lens_p1_geometry.mechanical_length_mm,
            external_envelope="D250-320 mm (provisional)",
            shape_profile=projector_lens_p1.shape_profile,
            outer_diameter_mm=(
                projector_lens_p1_geometry.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                projector_lens_p1_geometry
                .mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=projector_lens_p1_geometry.pole_gap_mm,
            optical_reference_plane_mm=(
                projector_lens_p1_geometry.optical_reference_z_mm
            ),
            effective_aperture_radius_mm=(
                projector_lens_p1.effective_aperture_radius_mm
            ),
        ),
        PROJECTOR_LENS_2: _LayoutSpec(
            PROJECTOR_LENS_2,
            projector_lens_p2.name,
            "magnetic_lens",
            projector_lens_p2.owner,
            -projector_lens_p2_geometry.mechanical_center_below_sample_mm,
            projector_lens_p2_geometry.mechanical_length_mm,
            external_envelope="D260-340 mm (provisional)",
            shape_profile=projector_lens_p2.shape_profile,
            outer_diameter_mm=(
                projector_lens_p2_geometry.mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                projector_lens_p2_geometry
                .mechanical_clear_bore_diameter_mm
            ),
            active_length_mm=projector_lens_p2_geometry.pole_gap_mm,
            optical_reference_plane_mm=(
                projector_lens_p2_geometry.optical_reference_z_mm
            ),
            effective_aperture_radius_mm=(
                projector_lens_p2.effective_aperture_radius_mm
            ),
        ),
        HAADF_DETECTOR: _LayoutSpec(
            HAADF_DETECTOR,
            stem_detectors[HAADF_DETECTOR].name,
            "detector",
            stem_detectors[HAADF_DETECTOR].owner,
            selected_area_downstream_center(
                stem_detectors[HAADF_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
            stem_detectors[HAADF_DETECTOR].layout_length_mm,
            Branch.DETECTION,
            shape_profile="detector_plane",
            active_diameter_mm=(
                stem_detectors[HAADF_DETECTOR].outer_width_mm
            ),
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + stem_detectors[HAADF_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
        ),
        FLUORESCENT_SCREEN: _LayoutSpec(
            FLUORESCENT_SCREEN,
            fluorescent_screen.name,
            fluorescent_screen.kind,
            fluorescent_screen.owner,
            selected_area_downstream_center(
                fluorescent_screen
                .layout_center_downstream_of_anchor_mm
            ),
            fluorescent_screen.layout_length_mm,
            Branch.DETECTION,
            external_envelope=fluorescent_screen.external_envelope,
            shape_profile=fluorescent_screen.shape_profile,
            active_diameter_mm=fluorescent_screen.active_diameter_mm,
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + fluorescent_screen
                .layout_center_downstream_of_anchor_mm
            ),
        ),
        DARK_FIELD_DETECTOR: _LayoutSpec(
            DARK_FIELD_DETECTOR,
            stem_detectors[DARK_FIELD_DETECTOR].name,
            "detector",
            stem_detectors[DARK_FIELD_DETECTOR].owner,
            selected_area_downstream_center(
                stem_detectors[DARK_FIELD_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
            stem_detectors[DARK_FIELD_DETECTOR].layout_length_mm,
            Branch.DETECTION,
            shape_profile="detector_plane",
            active_diameter_mm=(
                stem_detectors[DARK_FIELD_DETECTOR].outer_width_mm
            ),
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + stem_detectors[DARK_FIELD_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
        ),
        BRIGHT_FIELD_DETECTOR: _LayoutSpec(
            BRIGHT_FIELD_DETECTOR,
            stem_detectors[BRIGHT_FIELD_DETECTOR].name,
            "detector",
            stem_detectors[BRIGHT_FIELD_DETECTOR].owner,
            selected_area_downstream_center(
                stem_detectors[BRIGHT_FIELD_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
            stem_detectors[BRIGHT_FIELD_DETECTOR].layout_length_mm,
            Branch.DETECTION,
            shape_profile="detector_plane",
            active_diameter_mm=(
                stem_detectors[BRIGHT_FIELD_DETECTOR].outer_width_mm
            ),
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + stem_detectors[BRIGHT_FIELD_DETECTOR]
                .layout_center_downstream_of_anchor_mm
            ),
        ),
        CAMERA: _LayoutSpec(
            CAMERA,
            camera.name,
            camera.kind,
            camera.owner,
            selected_area_downstream_center(
                camera.layout_center_downstream_of_anchor_mm
            ),
            camera.layout_length_mm,
            Branch.DETECTION,
            external_envelope=camera.external_envelope,
            shape_profile=camera.shape_profile,
            active_diameter_mm=camera.active_width_mm,
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + camera.layout_center_downstream_of_anchor_mm
            ),
        ),
        ENERGY_FILTER_ENTRANCE_APERTURE: _LayoutSpec(
            ENERGY_FILTER_ENTRANCE_APERTURE,
            energy_filter_entrance_aperture.name,
            energy_filter_entrance_aperture.kind,
            energy_filter_entrance_aperture.owner,
            selected_area_downstream_center(
                energy_filter_entrance_aperture
                .layout_center_downstream_of_anchor_mm
            ),
            energy_filter_entrance_aperture.mechanical_length_mm,
            Branch.ENERGY_FILTER,
            installed_if=("energy_filter",),
            shape_profile=(
                energy_filter_entrance_aperture.shape_profile
            ),
            outer_diameter_mm=(
                energy_filter_entrance_aperture
                .mechanical_outer_diameter_mm
            ),
            active_diameter_mm=(
                2.0 * energy_filter_entrance_aperture.radius_mm
            ),
            active_length_mm=(
                energy_filter_entrance_aperture.plate_thickness_mm
            ),
            optical_reference_plane_mm=(
                selected_area_geometry.optical_reference_z_mm
                + energy_filter_entrance_aperture
                .layout_center_downstream_of_anchor_mm
            ),
            effective_aperture_radius_mm=(
                energy_filter_entrance_aperture.radius_mm
            ),
        ),
        "energy_filter": _LayoutSpec(
            "energy_filter", "GIF / Camera Interface", "branch_interface",
            "energy_filter", selected_area_downstream_center(1140.0), 500.0,
            Branch.ENERGY_FILTER,
            external_envelope="400-800 mm envelope", installed_if=("energy_filter",),
        ),
    })
    specs.update(_image_corrector_specs(
        configuration.image_corrector_components,
        configuration.sample_center_from_source_mm,
    ))
    magnetic_pole_parents = {
        component.key: component
        for component in (
            diffraction_lens,
            intermediate_lens,
            projector_lens_p1,
            projector_lens_p2,
            mini_condenser,
            adapter_lens,
            tl22_lens,
            *probe_corrector_tail,
            *configuration.image_corrector_components,
        )
        if (
            component.key in specs
            and getattr(component, "interaction_kind", "")
            == "axial_magnetic_field"
        )
    }
    for lens_key, component in magnetic_pole_parents.items():
        parent = specs[lens_key]
        pole_gap = float(component.pole_gap_mm)
        if parent.active_length_mm is None:
            parent = replace(parent, active_length_mm=pole_gap)
            specs[lens_key] = parent
        pole_length = 0.5 * max(
            parent.length_mm - pole_gap,
            0.001,
        )
        offset = 0.5 * (pole_gap + pole_length)
        bore_diameter = getattr(component, "bore_diameter_mm", None)
        if bore_diameter is None:
            bore_diameter = component.mechanical_clear_bore_diameter_mm
        bore_diameter = float(bore_diameter)
        upper_pole_key, lower_pole_key = pole_piece_keys(lens_key)
        for pole_key, is_upper in (
            (upper_pole_key, True),
            (lower_pole_key, False),
        ):
            side = "Upper" if is_upper else "Lower"
            direction = 1.0 if is_upper else -1.0
            specs[pole_key] = _LayoutSpec(
                pole_key,
                f"{parent.name} {side} Pole Piece",
                "pole_piece",
                parent.owner,
                parent.center_z_mm + direction * offset,
                pole_length,
                parent.branch,
                installed_if=parent.installed_if,
                shape_profile="magnetic_pole_piece",
                outer_diameter_mm=(
                    0.67 * float(parent.outer_diameter_mm)
                ),
                active_diameter_mm=bore_diameter,
                active_length_mm=2.0 * bore_diameter,
                nested_parent_key=lens_key,
                mechanical_overlap_reason=(
                    f"The pole piece is part of the {parent.name} assembly."
                ),
            )
    return specs


def _condition_tokens(configuration: LayoutConfiguration):
    tokens = {configuration.corrector.value}
    tokens.add("three_condenser" if configuration.c3_hardware is C3Hardware.THREE_CONDENSER else "two_condenser")
    if configuration.corrector not in (CorrectorAssembly.PROBE_CORRECTOR, CorrectorAssembly.DOUBLE_CORRECTOR):
        tokens.add("no_probe_corrector")
    tokens.add(
        "monochromated"
        if (
            configuration.electron_gun_type == "cold_feg"
            and configuration.monochromator_installed
        )
        else "non_monochromated"
    )
    tokens.add("energy_filter" if configuration.energy_filter_selected else "no_energy_filter")
    return tokens


def _installed(spec: _LayoutSpec, tokens):
    return not spec.installed_if or any(token in tokens for token in spec.installed_if)


def _range(center, length):
    return (center - length / 2.0, center + length / 2.0)


def _field_kind(spec: _LayoutSpec):
    return {
        "source": "electron_source",
        "electrode": "electrostatic_field",
        "electrostatic_lens": "electrostatic_field",
        "magnetic_lens": "axial_magnetic_field",
        "round_lens": "axial_magnetic_field",
        "stigmator": "multipole_field",
        "quadrupole": "multipole_field",
        "hexapole": "multipole_field",
        "multipole_stack": "multipole_field",
        "deflector": "transverse_kick",
        "paired_deflector": "paired_transverse_kick",
        "aperture": "hard_aperture",
        "continuous_aperture": "hard_aperture",
        "aperture_plane": "hard_aperture",
        "sample": "specimen_scattering",
        "detector": "recording_plane",
        "branch_interface": "branch_interface",
    }.get(spec.kind, "passive")


def _nominal_gap(upstream: _LayoutSpec, downstream: _LayoutSpec):
    return (
        upstream.center_z_mm - upstream.length_mm / 2.0
        - (downstream.center_z_mm + downstream.length_mm / 2.0)
    )


def _pair_gaps(specs):
    gaps = {}
    for sequence in (
        _UPPER_SEQUENCE,
        _LOWER_SEQUENCE,
        _IMAGE_CORRECTOR_SEQUENCE,
    ):
        for a, b in zip(sequence, sequence[1:]):
            if a in specs and b in specs:
                gaps[(a, b)] = _nominal_gap(specs[a], specs[b])
    gaps.update({
        (C1_APERTURE, CONDENSER_LENS_1): 4.0,
        (FEG_ELECTROSTATIC_LENS, FEG_MONOCHROMATOR_WIEN): 4.0,
        # Layout allocation only: runtime coordinates retain the physical
        # 4 mm Wien-to-accelerator gap and the fixed C1 plane.
        (FEG_MONOCHROMATOR_WIEN, FEG_ACCELERATOR): 1.0,
        # DPA is embedded inside the accelerator body rather than consuming
        # an additional serial column length.
        (FEG_ACCELERATOR, GUN_EXTRACTOR_APERTURE): -271.0,
        (THERMIONIC_C1_APERTURE, CONDENSER_LENS_1): 4.0,
        (THERMIONIC_ACCELERATOR, THERMIONIC_DEFLECTOR): 5.0,
        (CONDENSER_APERTURE_2, CONDENSER_DEFLECTOR): 0.0,
        (CONDENSER_DEFLECTOR, CONDENSER_LENS_3): 0.0,
        (CONDENSER_LENS_3, CONDENSER_APERTURE_3): 0.0,
        (CONDENSER_APERTURE_3, BEAM_DEFLECTOR): 0.0,
        (CONDENSER_APERTURE_2, BEAM_DEFLECTOR): 0.0,
        (BEAM_DEFLECTOR, ADAPTER_LENS): 10.0,
        (BEAM_DEFLECTOR, CONDENSER_STIGMATOR): 5.0,
        (PROBE_DP12_SCAN_DEFLECTOR, CONDENSER_STIGMATOR): 5.0,
        (CONDENSER_STIGMATOR, AC_DEFLECTOR): 5.0,
        (AC_DEFLECTOR, MINI_CONDENSER): 5.0,
        (SELECTED_AREA_APERTURE, DIFFRACTION_STIGMATOR): 5.0,
        # The diffraction stigmator is nested at the entrance of the
        # diffraction-lens mechanical envelope.
        (DIFFRACTION_STIGMATOR, DIFFRACTION_LENS): -40.0,
        (SELECTED_AREA_APERTURE, DIFFRACTION_LENS): -72.5,
        (PROJECTOR_LENS_2, "energy_filter"): 40.0,
        (PROJECTOR_LENS_2, HAADF_DETECTOR): 2.5,
    })
    return gaps


def _gap_between(gaps, upstream_key, downstream_key):
    return gaps.get((upstream_key, downstream_key), 5.0)


def _place_axis(specs, configuration):
    tokens = _condition_tokens(configuration)
    upper = [key for key in _UPPER_SEQUENCE if key in specs and _installed(specs[key], tokens)]
    lower_sequence = (
        _IMAGE_CORRECTOR_SEQUENCE
        if configuration.corrector
        in (CorrectorAssembly.IMAGE_CORRECTOR, CorrectorAssembly.DOUBLE_CORRECTOR)
        else _LOWER_SEQUENCE
    )
    lower = [key for key in lower_sequence if key in specs and _installed(specs[key], tokens)]
    gaps = _pair_gaps(specs)
    centers = {"sample": 0.0}

    downstream_key = "sample"
    downstream = specs[downstream_key]
    for key in reversed([key for key in upper if key != "sample"]):
        spec = specs[key]
        gap = _gap_between(gaps, key, downstream_key)
        centers[key] = centers[downstream_key] + downstream.length_mm / 2.0 + gap + spec.length_mm / 2.0
        downstream_key = key
        downstream = spec

    upstream_key = "sample"
    upstream = specs[upstream_key]
    for key in [key for key in lower if key != "sample"]:
        spec = specs[key]
        gap = _gap_between(gaps, upstream_key, key)
        centers[key] = centers[upstream_key] - upstream.length_mm / 2.0 - gap - spec.length_mm / 2.0
        upstream_key = key
        upstream = spec

    centers[OBJECTIVE_LENS] = 0.0
    centers["sample_stage"] = 0.0
    # Composite modules own their internal mechanical coordinates. Optional
    # neighbours such as a monochromator must not compress those coordinates.
    source_key = configuration.gun_components[0].key
    tip_center = float(configuration.sample_center_from_source_mm)
    if tip_center is not None:
        for component in configuration.gun_components:
            if component.key in centers:
                centers[component.key] = (
                    tip_center - component.mechanical_center_from_tip_mm
                )
        for component in configuration.condenser_components:
            if component.key in centers:
                centers[component.key] = (
                    tip_center - component.mechanical_center_from_tip_mm
                )
        condenser_aperture_2 = (
            configuration.condenser_aperture_2_component
        )
        if condenser_aperture_2.key in centers:
            centers[condenser_aperture_2.key] = (
                tip_center
                - condenser_aperture_2.mechanical_center_from_tip_mm
            )
        condenser_aperture_3 = (
            configuration.condenser_aperture_3_component
        )
        if condenser_aperture_3.key in centers:
            centers[condenser_aperture_3.key] = (
                tip_center
                - condenser_aperture_3.mechanical_center_from_tip_mm
            )
        condenser_deflector = configuration.condenser_deflector_component
        if condenser_deflector.key in centers:
            centers[condenser_deflector.key] = (
                tip_center
                - condenser_deflector.mechanical_center_from_tip_mm
            )
        beam_deflector = configuration.beam_deflector_component
        if beam_deflector.key in centers:
            centers[beam_deflector.key] = (
                tip_center
                - beam_deflector.mechanical_center_from_tip_mm
            )
        ac_deflector = configuration.ac_deflector_component
        if ac_deflector.key in centers:
            centers[ac_deflector.key] = (
                tip_center
                - ac_deflector.mechanical_center_from_tip_mm
            )
        mini_condenser = configuration.mini_condenser_component
        mini_installation = (
            "integrated"
            if configuration.corrector in (
                CorrectorAssembly.PROBE_CORRECTOR,
                CorrectorAssembly.DOUBLE_CORRECTOR,
            )
            else "standalone"
        )
        mini_geometry = mini_condenser.geometry_for(mini_installation)
        if mini_condenser.key in centers:
            centers[mini_condenser.key] = (
                tip_center
                - mini_geometry.mechanical_center_from_tip_mm
            )
        adapter_lens = configuration.adapter_lens_component
        if adapter_lens.key in centers:
            centers[adapter_lens.key] = (
                tip_center - adapter_lens.mechanical_center_from_tip_mm
            )
        dph2_deflector = configuration.dph2_deflector_component
        if dph2_deflector.key in centers:
            centers[dph2_deflector.key] = (
                tip_center
                - dph2_deflector.mechanical_center_from_tip_mm
            )
        dp22_deflector = configuration.dp22_deflector_component
        if dp22_deflector.key in centers:
            centers[dp22_deflector.key] = (
                tip_center
                - dp22_deflector.mechanical_center_from_tip_mm
            )
        qph2_quadrupole = configuration.qph2_quadrupole_component
        if qph2_quadrupole.key in centers:
            centers[qph2_quadrupole.key] = (
                tip_center
                - qph2_quadrupole.mechanical_center_from_tip_mm
            )
        hp2_hexapole = configuration.hp2_hexapole_component
        if hp2_hexapole.key in centers:
            centers[hp2_hexapole.key] = (
                tip_center - hp2_hexapole.mechanical_center_from_tip_mm
            )
        hpc_hexapole = configuration.hpc_hexapole_component
        if hpc_hexapole.key in centers:
            centers[hpc_hexapole.key] = (
                tip_center - hpc_hexapole.mechanical_center_from_tip_mm
            )
        for component in configuration.probe_corrector_tail_components:
            if component.key in centers:
                centers[component.key] = (
                    tip_center
                    - component.mechanical_center_from_tip_mm
                )
        tl22_lens = configuration.tl22_lens_component
        if tl22_lens.key in centers:
            centers[tl22_lens.key] = (
                tip_center - tl22_lens.mechanical_center_from_tip_mm
            )
    condenser_poles = (
        (CONDENSER_LENS_1, CONDENSER_LENS_1_LOWER_POLE, False),
        (CONDENSER_LENS_2, CONDENSER_LENS_2_UPPER_POLE, True),
        (CONDENSER_LENS_3, CONDENSER_LENS_3_UPPER_POLE, True),
        (CONDENSER_LENS_3, CONDENSER_LENS_3_LOWER_POLE, False),
    )
    for lens_key, pole_key, is_upper_pole in condenser_poles:
        if lens_key not in centers:
            continue
        gap = float(specs[lens_key].active_length_mm)
        pole_length = float(specs[pole_key].length_mm)
        offset = 0.5 * (gap + pole_length)
        direction = 1.0 if is_upper_pole else -1.0
        centers[pole_key] = centers[lens_key] + offset * direction
    condenser_stigmator = configuration.condenser_stigmator_component
    if condenser_stigmator.key in centers:
        centers[condenser_stigmator.key] = (
            condenser_stigmator.mechanical_center_above_sample_mm
        )
    objective_lens = configuration.objective_lens_component
    objective_aperture = configuration.objective_aperture_component
    objective_stigmator = configuration.objective_stigmator_component
    image_diffraction_deflector = (
        configuration.image_diffraction_deflector_component
    )
    descan_deflector = configuration.descan_deflector_component
    selected_area_aperture = (
        configuration.selected_area_aperture_component
    )
    selected_area_installation = (
        IMAGE_CORRECTED_INSTALLATION
        if configuration.corrector in (
            CorrectorAssembly.IMAGE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        )
        else STANDALONE_INSTALLATION
    )
    selected_area_geometry = selected_area_aperture.geometry_for(
        selected_area_installation
    )
    centers["objective_upper_pole"] = (
        (
            tip_center
            - float(objective_lens.upper_pole_piece_center_z_mm)
            if hasattr(objective_lens, "upper_pole_piece_center_z_mm")
            else (
                objective_lens.inner_face_gap_mm / 2.0
                + objective_lens.upper_pole_piece_axial_length_mm / 2.0
                - configuration.objective.sample_axial_offset_mm
            )
        )
    )
    centers["objective_lower_pole"] = (
        (
            tip_center
            - float(objective_lens.lower_pole_piece_center_z_mm)
            if hasattr(objective_lens, "lower_pole_piece_center_z_mm")
            else (
                -objective_lens.pole_piece_center_separation_mm / 2.0
                - configuration.objective.sample_axial_offset_mm
            )
        )
    )
    centers[OBJECTIVE_APERTURE] = (
        -objective_aperture.mechanical_center_below_sample_mm
        - configuration.objective.sample_axial_offset_mm
    )
    centers["objective_stigmator"] = (
        -objective_stigmator.mechanical_center_below_sample_mm
    )
    if IMAGE_DIFFRACTION_DEFLECTOR in centers:
        centers[IMAGE_DIFFRACTION_DEFLECTOR] = (
            -image_diffraction_deflector
            .mechanical_center_below_sample_mm
        )
    if DESCAN_DEFLECTOR in centers:
        centers[DESCAN_DEFLECTOR] = (
            -descan_deflector.mechanical_center_below_sample_mm
        )
    if SELECTED_AREA_APERTURE in centers:
        centers[SELECTED_AREA_APERTURE] = (
            -selected_area_geometry.mechanical_center_below_sample_mm
        )
    downstream_components = {
        DIFFRACTION_STIGMATOR: (
            configuration.diffraction_stigmator_component
        ),
        DIFFRACTION_LENS: configuration.diffraction_lens_component,
        INTERMEDIATE_LENS: configuration.intermediate_lens_component,
        PROJECTOR_LENS_1: configuration.projector_lens_p1_component,
        PROJECTOR_LENS_2: configuration.projector_lens_p2_component,
        **{
            detector.key: detector
            for detector in configuration.stem_detector_components
        },
        FLUORESCENT_SCREEN: (
            configuration.fluorescent_screen_component
        ),
        CAMERA: configuration.camera_component,
        ENERGY_FILTER_ENTRANCE_APERTURE: (
            configuration.energy_filter_entrance_aperture_component
        ),
    }
    selected_area_center = centers.get(SELECTED_AREA_APERTURE)
    if selected_area_center is not None:
        for key in SELECTED_AREA_DOWNSTREAM_KEYS:
            component = downstream_components.get(key)
            if component is None or key not in centers:
                continue
            offset = (
                component.mechanical_center_downstream_of_anchor_mm
                if hasattr(
                    component,
                    "mechanical_center_downstream_of_anchor_mm",
                )
                else component.layout_center_downstream_of_anchor_mm
            )
            centers[key] = selected_area_center - float(offset)
    pole_keys_by_lens = {}
    for pole_key, pole_spec in specs.items():
        if (
            pole_spec.shape_profile == "magnetic_pole_piece"
            and pole_spec.nested_parent_key
        ):
            pole_keys_by_lens.setdefault(
                pole_spec.nested_parent_key, []
            ).append(pole_key)
    for lens_key, pole_keys in pole_keys_by_lens.items():
        if lens_key == OBJECTIVE_LENS or lens_key not in centers:
            continue
        gap = float(specs[lens_key].active_length_mm)
        for pole_key in pole_keys:
            pole_length = float(specs[pole_key].length_mm)
            offset = 0.5 * (gap + pole_length)
            direction = 1.0 if "_upper_pole" in pole_key else -1.0
            centers[pole_key] = centers[lens_key] + direction * offset
    expanded_upper = []
    for key in upper[:-1]:
        expanded_upper.append(key)
        expanded_upper.extend(pole_keys_by_lens.get(key, ()))
    expanded_lower = []
    for key in lower[1:]:
        expanded_lower.append(key)
        expanded_lower.extend(pole_keys_by_lens.get(key, ()))
    return centers, [
        *expanded_upper,
        OBJECTIVE_LENS,
        "sample_stage",
        "sample",
        *expanded_lower,
    ]


def _build_optics_layout_metadata(
    configuration: LayoutConfiguration = LayoutConfiguration(),
) -> LayoutResult:
    """Build a sample-centered mechanical layout for one physical column."""
    specs = _base_specs(configuration)
    objective = configuration.objective
    objective_lens = configuration.objective_lens_component
    upper_center = (
        (
            float(configuration.sample_center_from_source_mm)
            - float(objective_lens.upper_pole_piece_center_z_mm)
        )
        if hasattr(objective_lens, "upper_pole_piece_center_z_mm")
        else (
            float(objective_lens.inner_face_gap_mm) / 2.0
            + float(objective_lens.upper_pole_piece_axial_length_mm) / 2.0
            - objective.sample_axial_offset_mm
        )
    )
    lower_center = (
        (
            float(configuration.sample_center_from_source_mm)
            - float(objective_lens.lower_pole_piece_center_z_mm)
        )
        if hasattr(objective_lens, "lower_pole_piece_center_z_mm")
        else (
            -objective_lens.pole_piece_center_separation_mm / 2.0
            - objective.sample_axial_offset_mm
        )
    )
    specs["objective_upper_pole"] = _LayoutSpec(
        "objective_upper_pole", "Upper Objective", "magnetic_pole_piece",
        "objective", upper_center,
        objective_lens.upper_pole_piece_axial_length_mm,
        shape_profile="magnetic_pole_piece",
        outer_diameter_mm=(
            objective_lens.upper_pole_piece_outer_diameter_mm
        ),
        active_diameter_mm=objective_lens.pole_piece_bore_diameter_mm,
        active_length_mm=(
            objective_lens.upper_pole_piece_tip_diameter_mm
        ),
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Upper Objective pole piece is part of the "
            "Objective Lens assembly."
        ),
    )
    specs["sample"] = _LayoutSpec(
        "sample", "Sample", "sample", "objective", 0.0,
        objective.specimen_thickness_mm,
        shape_profile="specimen_slab",
        outer_diameter_mm=3.0,
        nested_parent_key="sample_stage",
        mechanical_overlap_reason=(
            "The specimen is mounted inside the sample stage."
        ),
    )
    specs["objective_lower_pole"] = _LayoutSpec(
        "objective_lower_pole", "Lower Objective", "magnetic_pole_piece",
        "objective", lower_center,
        objective_lens.pole_piece_axial_length_mm,
        shape_profile="magnetic_pole_piece",
        outer_diameter_mm=objective_lens.pole_piece_outer_diameter_mm,
        active_diameter_mm=objective_lens.pole_piece_bore_diameter_mm,
        active_length_mm=objective_lens.pole_piece_tip_diameter_mm,
        nested_parent_key=OBJECTIVE_LENS,
        mechanical_overlap_reason=(
            "The Lower Objective pole piece is part of the "
            "Objective Lens assembly."
        ),
    )

    centers, ordered_keys = _place_axis(specs, configuration)
    active_keys = [key for key in ordered_keys if key in centers]
    components = []
    seen = set()
    for key in active_keys:
        if key in seen or key not in centers:
            continue
        seen.add(key)
        spec = specs[key]
        length = max(float(spec.length_mm), 0.0)
        start, end = _range(centers[key], length)
        mechanical = MechanicalEnvelope(start, end)
        index = active_keys.index(key)
        upstream_key = active_keys[index - 1] if index > 0 else None
        downstream_key = active_keys[index + 1] if index + 1 < len(active_keys) else None
        upstream_clearance = None
        downstream_clearance = None
        if upstream_key is not None:
            upstream_spec = specs[upstream_key]
            upstream_start, _ = _range(
                centers[upstream_key], max(float(upstream_spec.length_mm), 0.0)
            )
            upstream_clearance = upstream_start - end
        if downstream_key is not None:
            downstream_spec = specs[downstream_key]
            _, downstream_end = _range(
                centers[downstream_key], max(float(downstream_spec.length_mm), 0.0)
            )
            downstream_clearance = start - downstream_end
        components.append(LayoutComponent(
            spec.key,
            spec.name,
            spec.kind,
            spec.owner,
            spec.branch,
            configuration.c3_excited
            if spec.key == CONDENSER_LENS_3 else spec.excitation_enabled,
            mechanical,
            FieldSupport(start, end),
            centers[key],
            (start, end),
            centers[key],
            (start, end),
            spec.external_envelope,
            spec.note,
            upstream_key,
            downstream_key,
            upstream_clearance,
            downstream_clearance,
            MechanicalShape(
                length,
                profile=spec.shape_profile,
                external_envelope=spec.external_envelope,
                outer_diameter_mm=spec.outer_diameter_mm,
                active_diameter_mm=spec.active_diameter_mm,
                active_length_mm=spec.active_length_mm,
            ),
            FieldModel(
                _field_kind(spec),
                configuration.c3_excited
                if spec.key == CONDENSER_LENS_3
                else spec.excitation_enabled,
            ),
            (
                float(spec.optical_reference_plane_mm)
                if spec.optical_reference_plane_mm is not None
                else centers[key] + float(spec.optical_reference_offset_mm)
            ),
            spec.effective_aperture_radius_mm,
            tuple(float(value) for value in spec.optical_interaction_planes_mm),
            spec.nested_parent_key,
            spec.mechanical_overlap_reason,
        ))
    return LayoutResult(components)


def build_optics_layout(
    configuration: LayoutConfiguration = LayoutConfiguration(),
    *,
    assembly=None,
) -> LayoutResult:
    from temsim.column.module_assembly import apply_module_assembly

    return apply_module_assembly(
        configuration,
        _build_optics_layout_metadata(configuration),
        assembly=assembly,
    )


@dataclass(frozen=True)
class _SourceReferencedAxisComponent:
    key: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float


def resolve_objective_entrance_mechanical_axis(configuration=None):
    """Diagnose the serial package embedded above the sample."""

    if configuration is None:
        configuration = LayoutConfiguration()
    layout = build_optics_layout(configuration)
    by_key = {component.key: component for component in layout}
    axis_order = (
        CONDENSER_STIGMATOR,
        AC_DEFLECTOR,
        MINI_CONDENSER,
        "objective_upper_pole",
        "sample",
    )
    components = tuple(
        _SourceReferencedAxisComponent(
            key,
            (
                layout.source_to_sample_mm
                - by_key[key].local_s_center_mm
            ),
            (
                by_key[key].mechanical.end_s_mm
                - by_key[key].mechanical.start_s_mm
            ),
        )
        for key in axis_order
    )
    return resolve_mechanical_axis(components, axis_order)


def resolve_objective_exit_mechanical_axis(configuration=None):
    """Diagnose the serial sample-to-Image/Deflector mechanical path."""

    if configuration is None:
        configuration = LayoutConfiguration()
    layout = build_optics_layout(configuration)
    by_key = {component.key: component for component in layout}
    component_keys = (
        "sample",
        OBJECTIVE_APERTURE,
        "objective_lower_pole",
        DESCAN_DEFLECTOR,
        "objective_stigmator",
        IMAGE_DIFFRACTION_DEFLECTOR,
    )
    axis_order = (
        "sample",
        OBJECTIVE_APERTURE,
        "objective_lower_pole",
        DESCAN_DEFLECTOR,
        "objective_stigmator",
        IMAGE_DIFFRACTION_DEFLECTOR,
    )
    components = tuple(
        _SourceReferencedAxisComponent(
            key,
            (
                layout.source_to_sample_mm
                - by_key[key].local_s_center_mm
            ),
            (
                by_key[key].mechanical.end_s_mm
                - by_key[key].mechanical.start_s_mm
            ),
        )
        for key in component_keys
    )
    return resolve_mechanical_axis(components, axis_order)


def resolve_post_objective_deflector_mechanical_axis(
    configuration=None,
):
    """Diagnose Descan, Objective Stigmator and Image/Deflector."""

    if configuration is None:
        configuration = LayoutConfiguration()
    layout = build_optics_layout(configuration)
    by_key = {component.key: component for component in layout}
    axis_order = (
        DESCAN_DEFLECTOR,
        "objective_stigmator",
        IMAGE_DIFFRACTION_DEFLECTOR,
    )
    components = tuple(
        _SourceReferencedAxisComponent(
            key,
            (
                layout.source_to_sample_mm
                - by_key[key].local_s_center_mm
            ),
            (
                by_key[key].mechanical.end_s_mm
                - by_key[key].mechanical.start_s_mm
            ),
        )
        for key in axis_order
    )
    return resolve_mechanical_axis(components, axis_order)


def resolve_installed_image_corrector_exit_mechanical_axis(
    configuration=None,
):
    """Diagnose the installed Image Corrector path through the SAA."""

    if configuration is None:
        configuration = LayoutConfiguration(
            corrector=CorrectorAssembly.IMAGE_CORRECTOR,
        )
    if configuration.corrector not in (
        CorrectorAssembly.IMAGE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    ):
        raise ValueError(
            "The Image Corrector exit mechanical axis requires an "
            "installed Image Corrector."
        )
    layout = build_optics_layout(configuration)
    by_key = {component.key: component for component in layout}
    axis_order = (
        IMAGE_DIFFRACTION_DEFLECTOR,
        IMAGE_CORRECTOR_OL_POST_LENS,
        IMAGE_CORRECTOR_HPOL_HEXAPOLE,
        IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
        IMAGE_CORRECTOR_DP11_DEFLECTOR,
        IMAGE_CORRECTOR_TL11_LENS,
        IMAGE_CORRECTOR_DP12_DEFLECTOR,
        IMAGE_CORRECTOR_HP1_HEXAPOLE,
        IMAGE_CORRECTOR_DP21_DEFLECTOR,
        IMAGE_CORRECTOR_TL21_LENS,
        IMAGE_CORRECTOR_DP22_DEFLECTOR,
        IMAGE_CORRECTOR_TL22_LENS,
        IMAGE_CORRECTOR_HP2_HEXAPOLE,
        IMAGE_CORRECTOR_ADAPTER_LENS,
        IMAGE_CORRECTOR_ISH_DEFLECTOR,
        IMAGE_CORRECTOR_DSH_DEFLECTOR,
        IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
        SELECTED_AREA_APERTURE,
    )
    components = tuple(
        _SourceReferencedAxisComponent(
            key,
            layout.source_to_sample_mm - by_key[key].local_s_center_mm,
            (
                by_key[key].mechanical.end_s_mm
                - by_key[key].mechanical.start_s_mm
            ),
        )
        for key in axis_order
    )
    return resolve_mechanical_axis(components, axis_order)


def resolve_standalone_selected_area_aperture_mechanical_axis(
    configuration=None,
):
    """Diagnose Image/Deflector-to-SAA without an Image Corrector."""

    if configuration is None:
        configuration = LayoutConfiguration(
            corrector=CorrectorAssembly.NO_CORRECTOR,
        )
    if configuration.corrector in (
        CorrectorAssembly.IMAGE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    ):
        raise ValueError(
            "The standalone Selected Area Aperture axis requires the "
            "Image Corrector to be absent."
        )
    layout = build_optics_layout(configuration)
    by_key = {component.key: component for component in layout}
    axis_order = (
        IMAGE_DIFFRACTION_DEFLECTOR,
        SELECTED_AREA_APERTURE,
    )
    components = tuple(
        _SourceReferencedAxisComponent(
            key,
            layout.source_to_sample_mm - by_key[key].local_s_center_mm,
            (
                by_key[key].mechanical.end_s_mm
                - by_key[key].mechanical.start_s_mm
            ),
        )
        for key in axis_order
    )
    return resolve_mechanical_axis(components, axis_order)
