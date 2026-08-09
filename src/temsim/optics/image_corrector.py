"""Modular CETCOR-style TEM image-corrector assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from temsim import module_manifest
from temsim.component_keys import (
    IMAGE_CORRECTOR_ADAPTER_LENS,
    IMAGE_CORRECTOR_DP11_DEFLECTOR,
    IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR,
    IMAGE_CORRECTOR_DPH2_DEFLECTOR,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR,
    IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_ELEMENT_KEYS,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_HP2_HEXAPOLE,
    IMAGE_CORRECTOR_HPOL_HEXAPOLE,
    IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_KEYS,
    IMAGE_CORRECTOR_LENS_KEYS,
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    IMAGE_CORRECTOR_SAD_PLANE,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL12_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
)
from temsim.optics.hexapole import HexapoleComponent, restore_hexapole
from temsim.optics.model import Gaussian
from temsim.optics.quadrupole import (
    QuadrupoleComponent,
    restore_quadrupole,
)
from temsim.optics.round_lens import RoundLensComponent, restore_round_lens
from temsim.optics.single_plane_deflector import (
    SinglePlaneDeflectorComponent,
    restore_single_plane_deflector,
)
from temsim.optics.objective_lens import (
    reference_objective_image_plane_z_mm,
)


REFERENCE_SAMPLE_EFFECTIVE_Z_MM = 935.0
IMAGE_CORRECTOR_RELAY_CALIBRATION_VOLTAGE_KV = 300.0
REFERENCE_OBJECTIVE_IMAGE_PLANE_Z_MM = (
    reference_objective_image_plane_z_mm(
        IMAGE_CORRECTOR_RELAY_CALIBRATION_VOLTAGE_KV
    )
)
REFERENCE_SOURCE_TO_SAMPLE_MECHANICAL_MM = 1705.0
_DEFAULT_IMAGE_CORRECTOR_MODULE_PATH = "column/C3_ImageCorrector.toml"
# Bootstrap source-z coordinate from the default FEG plus image-corrected
# column TOMLs. Keep the 1705 mm calibration span above only for legacy field
# profile scaling; it is not mechanical geometry.
DEFAULT_IMAGE_CORRECTOR_SAMPLE_CENTER_FROM_SOURCE_MM = (
    module_manifest.port_z_mm("gun/FEG.toml", "exit")
    - module_manifest.port_z_mm(
        _DEFAULT_IMAGE_CORRECTOR_MODULE_PATH, "entrance"
    )
    + float(module_manifest.part_data(
        _DEFAULT_IMAGE_CORRECTOR_MODULE_PATH, "sample"
    )["local_center_z_mm"])
)
REFERENCE_MECHANICAL_TO_EFFECTIVE_SCALE = (
    REFERENCE_SAMPLE_EFFECTIVE_Z_MM
    / REFERENCE_SOURCE_TO_SAMPLE_MECHANICAL_MM
)
IMAGE_CORRECTOR_RELAY_CALIBRATION_STEP_MM = 0.5
DEFAULT_IMAGE_CORRECTOR_UPSTREAM_GAP_MM = 5.0
DEFAULT_SELECTED_AREA_APERTURE_OFFSET_FROM_SAD_MM = 0.0
IMAGE_MAIN_HEXAPOLE_STRENGTH_M3 = 2.49050244e7
IMAGE_HP2_HEXAPOLE_STRENGTH_RATIO = 0.59946796
IMAGE_HP2_HEXAPOLE_ORIENTATION_RAD = -2.00126792


@dataclass(frozen=True)
class ImageCorrectorComponentDefinition:
    key: str
    label: str
    kind: str
    mechanical_center_from_specimen_mm: float
    mechanical_length_mm: float
    outer_diameter_mm: Optional[float]
    shape_profile: str
    optical_reference_from_specimen_mm: float
    effective_aperture_radius_mm: Optional[float] = None
    active_diameter_mm: Optional[float] = None
    active_length_mm: Optional[float] = None
    reference_peak_field_t: float = 0.0
    default_excitation_percent: float = 0.0
    maximum_strength_m2: float = 300.0
    maximum_strength_m3: float = 1.0e9
    default_strength_m3: float = 0.0
    orientation_rad: float = 0.0
    maximum_kick_mrad: float = 100.0
    conjugate_to: Optional[str] = None

    @property
    def sample_centered_mechanical_z_mm(self):
        return -float(self.mechanical_center_from_specimen_mm)

    @property
    def sample_centered_optical_reference_z_mm(self):
        return -float(self.optical_reference_from_specimen_mm)

    @property
    def reference_effective_z_mm(self):
        return (
            REFERENCE_OBJECTIVE_IMAGE_PLANE_Z_MM
            + float(self.optical_reference_from_specimen_mm)
            * REFERENCE_MECHANICAL_TO_EFFECTIVE_SCALE
        )

    @property
    def reference_effective_length_mm(self):
        return (
            float(self.mechanical_length_mm)
            * REFERENCE_MECHANICAL_TO_EFFECTIVE_SCALE
        )


_DEFAULT_IMAGE_CORRECTOR_MANIFEST = module_manifest.read_document(
    module_manifest.MODULE_ROOT
    / "column"
    / "C3_ImageCorrector.toml"
)
_DEFAULT_IMAGE_CORRECTOR_PARTS = {
    str(part["key"]): part
    for part in _DEFAULT_IMAGE_CORRECTOR_MANIFEST["parts"]
}
_DEFAULT_IMAGE_CORRECTOR_SAMPLE_Z_MM = float(
    _DEFAULT_IMAGE_CORRECTOR_PARTS["sample"]["local_center_z_mm"]
)


def _image_corrector_definition(
    key, label, kind, shape_profile, **calibration
):
    """Combine TOML-owned structure with code-owned field behaviour."""

    part = _DEFAULT_IMAGE_CORRECTOR_PARTS[key]
    bore_diameter_mm = part.get(
        "bore_diameter_mm",
        part.get("mechanical_clear_bore_diameter_mm"),
    )
    if kind != "reference_plane":
        missing = [
            field
            for field, value in (
                ("mechanical_outer_diameter_mm", part.get(
                    "mechanical_outer_diameter_mm"
                )),
                ("bore diameter", bore_diameter_mm),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"Image-corrector TOML part {key} is missing "
                + ", ".join(missing)
            )
    return ImageCorrectorComponentDefinition(
        key=key,
        label=label,
        kind=kind,
        mechanical_center_from_specimen_mm=(
            float(part["local_center_z_mm"])
            - _DEFAULT_IMAGE_CORRECTOR_SAMPLE_Z_MM
        ),
        mechanical_length_mm=float(part["length_mm"]),
        outer_diameter_mm=(
            float(part["mechanical_outer_diameter_mm"])
            if "mechanical_outer_diameter_mm" in part
            else None
        ),
        shape_profile=shape_profile,
        optical_reference_from_specimen_mm=(
            float(part["optical_reference_local_z_mm"])
            - _DEFAULT_IMAGE_CORRECTOR_SAMPLE_Z_MM
        ),
        effective_aperture_radius_mm=(
            0.5 * float(bore_diameter_mm)
            if bore_diameter_mm is not None
            else None
        ),
        active_diameter_mm=(
            float(part["active_diameter_mm"])
            if "active_diameter_mm" in part
            else None
        ),
        active_length_mm=(
            float(part.get(
                "effective_length_mm",
                part.get("effective_thickness_mm"),
            ))
            if (
                "effective_length_mm" in part
                or "effective_thickness_mm" in part
            )
            else None
        ),
        **calibration,
    )


IMAGE_CORRECTOR_COMPONENTS = (
    _image_corrector_definition(
        IMAGE_CORRECTOR_OL_POST_LENS,
        "OL post",
        "round_lens",
        "magnetic_lens_yoke",
        reference_peak_field_t=3.0364085833333334,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_HPOL_HEXAPOLE,
        "HPol", "hexapole", "hexapole_body",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
        "QPol", "quadrupole", "quadrupole_body",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DP11_DEFLECTOR,
        "DP11", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_TL11_LENS,
        "TL11",
        "round_lens",
        "magnetic_lens_yoke",
        reference_peak_field_t=0.5609032,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DP12_DEFLECTOR,
        "DP12", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_TL12_LENS,
        "TL12",
        "round_lens",
        "integrated_magnetic_lens_channel",
        reference_peak_field_t=0.3299537833333333,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DPH1_DEFLECTOR,
        "DPH1", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_HP1_HEXAPOLE,
        "HP1",
        "hexapole",
        "hexapole_body",
        default_strength_m3=IMAGE_MAIN_HEXAPOLE_STRENGTH_M3,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DP21_DEFLECTOR,
        "DP21", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_TL21_LENS,
        "TL21",
        "round_lens",
        "magnetic_lens_yoke",
        reference_peak_field_t=2.1599302,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DP22_DEFLECTOR,
        "DP22", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_TL22_LENS,
        "TL22",
        "round_lens",
        "magnetic_lens_yoke",
        reference_peak_field_t=2.1261313,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DPH2_DEFLECTOR,
        "DPH2", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_HP2_HEXAPOLE,
        "HP2",
        "hexapole",
        "hexapole_body",
        default_strength_m3=(
            IMAGE_MAIN_HEXAPOLE_STRENGTH_M3
            * IMAGE_HP2_HEXAPOLE_STRENGTH_RATIO
        ),
        orientation_rad=IMAGE_HP2_HEXAPOLE_ORIENTATION_RAD,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_ADAPTER_LENS,
        "ADL",
        "round_lens",
        "magnetic_lens_yoke",
        reference_peak_field_t=0.41919165,
        default_excitation_percent=60.0,
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_ISH_DEFLECTOR,
        "ISh", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DSH_DEFLECTOR,
        "DSh", "deflector", "single_deflector_coil",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
        "DStg", "quadrupole", "quadrupole_body",
    ),
    _image_corrector_definition(
        IMAGE_CORRECTOR_SAD_PLANE,
        "SAD plane",
        "reference_plane",
        "reference_plane",
        conjugate_to="objective_image_plane",
    ),
)

IMAGE_CORRECTOR_BY_KEY = {
    component.key: component for component in IMAGE_CORRECTOR_COMPONENTS
}
IMAGE_CORRECTOR_ROUND_LENSES = tuple(
    component
    for component in IMAGE_CORRECTOR_COMPONENTS
    if component.kind == "round_lens"
)


def default_image_corrector_offsets_from_ol_post_mm():
    """Return fixed internal centre offsets measured from OL Post."""

    ol_post_center_mm = IMAGE_CORRECTOR_BY_KEY[
        IMAGE_CORRECTOR_OL_POST_LENS
    ].mechanical_center_from_specimen_mm
    return {
        definition.key: (
            float(definition.mechanical_center_from_specimen_mm)
            - float(ol_post_center_mm)
        )
        for definition in IMAGE_CORRECTOR_COMPONENTS
        if definition.key != IMAGE_CORRECTOR_OL_POST_LENS
    }


class ImageCorrectorOlPostLensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_OL_POST_LENS


class ImageCorrectorTl11LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_TL11_LENS


class ImageCorrectorTl12LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_TL12_LENS


class ImageCorrectorTl21LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_TL21_LENS


class ImageCorrectorTl22LensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_TL22_LENS


class ImageCorrectorAdapterLensComponent(RoundLensComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_ADAPTER_LENS


class ImageCorrectorHpolHexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_HPOL_HEXAPOLE


class ImageCorrectorHp1HexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_HP1_HEXAPOLE


class ImageCorrectorHp2HexapoleComponent(HexapoleComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_HP2_HEXAPOLE


class ImageCorrectorQpolQuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_QPOL_QUADRUPOLE


class ImageCorrectorDstgQuadrupoleComponent(QuadrupoleComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DSTG_QUADRUPOLE


class ImageCorrectorDp11DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DP11_DEFLECTOR


class ImageCorrectorDp12DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DP12_DEFLECTOR


class ImageCorrectorDph1DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DPH1_DEFLECTOR


class ImageCorrectorDph2DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DPH2_DEFLECTOR


class ImageCorrectorDp21DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DP21_DEFLECTOR


class ImageCorrectorDp22DeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DP22_DEFLECTOR


class ImageCorrectorIshDeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_ISH_DEFLECTOR


class ImageCorrectorDshDeflectorComponent(SinglePlaneDeflectorComponent):
    EXPECTED_KEY: ClassVar[str] = IMAGE_CORRECTOR_DSH_DEFLECTOR


@dataclass
class ImageCorrectorSadPlaneComponent:
    name: str = "Image Corrector SAD plane"
    key: str = IMAGE_CORRECTOR_SAD_PLANE
    z_mm: float = (
        DEFAULT_IMAGE_CORRECTOR_SAMPLE_CENTER_FROM_SOURCE_MM
        + IMAGE_CORRECTOR_BY_KEY[
            IMAGE_CORRECTOR_SAD_PLANE
        ].mechanical_center_from_specimen_mm
    )
    mechanical_center_from_tip_mm: float = (
        DEFAULT_IMAGE_CORRECTOR_SAMPLE_CENTER_FROM_SOURCE_MM
        + IMAGE_CORRECTOR_BY_KEY[
            IMAGE_CORRECTOR_SAD_PLANE
        ].mechanical_center_from_specimen_mm
    )
    optical_reference_from_tip_mm: float = (
        DEFAULT_IMAGE_CORRECTOR_SAMPLE_CENTER_FROM_SOURCE_MM
        + IMAGE_CORRECTOR_BY_KEY[
            IMAGE_CORRECTOR_SAD_PLANE
        ].mechanical_center_from_specimen_mm
    )
    enabled: bool = True
    corrector: str = "image"
    colour: str = "#7cb342"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "mechanical_center_from_tip_mm":
            value = float(value)
            delta = value - float(self.mechanical_center_from_tip_mm)
            object.__setattr__(self, name, value)
            optical = float(self.optical_reference_from_tip_mm) + delta
            object.__setattr__(self, "optical_reference_from_tip_mm", optical)
            object.__setattr__(self, "z_mm", optical)
            return
        if ready and name == "optical_reference_from_tip_mm":
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "z_mm", float(value))
            return
        object.__setattr__(self, name, value)

    @property
    def label(self):
        return self.name

    @property
    def kind(self):
        return "reference_plane"

    @property
    def owner(self):
        return "image_corrector"

    @property
    def shape_profile(self):
        return "reference_plane"

    @property
    def interaction_kind(self):
        return "reference_plane"

    @property
    def mechanical_length_mm(self):
        return 0.0

    @property
    def length_mm(self):
        """Zero axial extent for generic overlay/component consumers."""

        return 0.0

    @property
    def mechanical_outer_diameter_mm(self):
        return None

    @property
    def effective_aperture_radius_mm(self):
        return None

    @property
    def optical_active(self):
        return False

    @property
    def note(self):
        return (
            "Conjugate to the Objective Lens image plane; this reference "
            "plane is not a second physical aperture."
        )

    def validate(self):
        if self.key != IMAGE_CORRECTOR_SAD_PLANE:
            raise ValueError("Image Corrector SAD plane key is not canonical.")
        return self

    def apply_optical_position(self):
        self.z_mm = self.optical_reference_from_tip_mm
        return self

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": (
                self.mechanical_center_from_tip_mm
            ),
            "mechanical_length_mm": 0.0,
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "enabled": self.enabled,
        }


_ROUND_LENS_TYPES = {
    IMAGE_CORRECTOR_OL_POST_LENS: ImageCorrectorOlPostLensComponent,
    IMAGE_CORRECTOR_TL11_LENS: ImageCorrectorTl11LensComponent,
    IMAGE_CORRECTOR_TL12_LENS: ImageCorrectorTl12LensComponent,
    IMAGE_CORRECTOR_TL21_LENS: ImageCorrectorTl21LensComponent,
    IMAGE_CORRECTOR_TL22_LENS: ImageCorrectorTl22LensComponent,
    IMAGE_CORRECTOR_ADAPTER_LENS: ImageCorrectorAdapterLensComponent,
}
_HEXAPOLE_TYPES = {
    IMAGE_CORRECTOR_HPOL_HEXAPOLE:
        ImageCorrectorHpolHexapoleComponent,
    IMAGE_CORRECTOR_HP1_HEXAPOLE:
        ImageCorrectorHp1HexapoleComponent,
    IMAGE_CORRECTOR_HP2_HEXAPOLE:
        ImageCorrectorHp2HexapoleComponent,
}
_QUADRUPOLE_TYPES = {
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE:
        ImageCorrectorQpolQuadrupoleComponent,
    IMAGE_CORRECTOR_DSTG_QUADRUPOLE:
        ImageCorrectorDstgQuadrupoleComponent,
}
_DEFLECTOR_TYPES = {
    IMAGE_CORRECTOR_DP11_DEFLECTOR:
        ImageCorrectorDp11DeflectorComponent,
    IMAGE_CORRECTOR_DP12_DEFLECTOR:
        ImageCorrectorDp12DeflectorComponent,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR:
        ImageCorrectorDph1DeflectorComponent,
    IMAGE_CORRECTOR_DPH2_DEFLECTOR:
        ImageCorrectorDph2DeflectorComponent,
    IMAGE_CORRECTOR_DP21_DEFLECTOR:
        ImageCorrectorDp21DeflectorComponent,
    IMAGE_CORRECTOR_DP22_DEFLECTOR:
        ImageCorrectorDp22DeflectorComponent,
    IMAGE_CORRECTOR_ISH_DEFLECTOR:
        ImageCorrectorIshDeflectorComponent,
    IMAGE_CORRECTOR_DSH_DEFLECTOR:
        ImageCorrectorDshDeflectorComponent,
}


def _mechanical_center_from_tip(definition):
    return (
        DEFAULT_IMAGE_CORRECTOR_SAMPLE_CENTER_FROM_SOURCE_MM
        + definition.mechanical_center_from_specimen_mm
    )


def _create_round_lens(key):
    definition = IMAGE_CORRECTOR_BY_KEY[key]
    component_type = _ROUND_LENS_TYPES[key]
    bore = 2.0 * float(definition.effective_aperture_radius_mm)
    mechanical_center_mm = _mechanical_center_from_tip(definition)
    return component_type(
        name=f"Image Corrector {definition.label}",
        key=key,
        z_mm=mechanical_center_mm,
        b0_t=definition.reference_peak_field_t,
        a_mm=max(definition.reference_effective_length_mm / 2.0, 0.5),
        percent=definition.default_excitation_percent,
        max_percent=100.0,
        colour="#3949ab",
        gaussian=[
            Gaussian(0.09, -1.0, 0.90),
            Gaussian(0.82, 0.0, 0.55),
            Gaussian(0.09, 1.0, 0.90),
        ],
        enabled=True,
        cs_mm=None,
        cc_mm=None,
        polarity=int(_DEFAULT_IMAGE_CORRECTOR_PARTS[key]["field_polarity"]),
        normalise_profile_peak=False,
        mechanical_center_from_tip_mm=mechanical_center_mm,
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=definition.outer_diameter_mm,
        bore_diameter_mm=bore,
        pole_gap_mm=float(
            _DEFAULT_IMAGE_CORRECTOR_PARTS[key]["pole_gap_mm"]
        ),
        optical_reference_from_tip_mm=mechanical_center_mm,
        corrector="image",
    ).validate()


def _create_hexapole(key):
    definition = IMAGE_CORRECTOR_BY_KEY[key]
    component_type = _HEXAPOLE_TYPES[key]
    mechanical_center_mm = _mechanical_center_from_tip(definition)
    return component_type(
        name=f"Image Corrector {definition.label} Hexapole",
        key=key,
        z_mm=mechanical_center_mm,
        strength_m3=definition.default_strength_m3,
        orientation_rad=definition.orientation_rad,
        maximum_strength_m3=definition.maximum_strength_m3,
        effective_length_mm=(
            definition.active_length_mm
            or definition.mechanical_length_mm
        ),
        enabled=True,
        colour="#8e24aa",
        mechanical_center_from_tip_mm=mechanical_center_mm,
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=definition.outer_diameter_mm,
        mechanical_clear_bore_diameter_mm=(
            2.0 * definition.effective_aperture_radius_mm
        ),
        optical_reference_from_tip_mm=mechanical_center_mm,
        corrector="image",
    ).validate()


def _create_quadrupole(key):
    definition = IMAGE_CORRECTOR_BY_KEY[key]
    component_type = _QUADRUPOLE_TYPES[key]
    mechanical_center_mm = _mechanical_center_from_tip(definition)
    return component_type(
        name=f"Image Corrector {definition.label} Quadrupole",
        key=key,
        z_mm=mechanical_center_mm,
        strength_m2=0.0,
        maximum_strength_m2=definition.maximum_strength_m2,
        effective_length_mm=(
            definition.active_length_mm
            or definition.mechanical_length_mm
        ),
        enabled=True,
        colour="#7b1fa2",
        mechanical_center_from_tip_mm=mechanical_center_mm,
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=definition.outer_diameter_mm,
        mechanical_clear_bore_diameter_mm=(
            2.0 * definition.effective_aperture_radius_mm
        ),
        optical_reference_from_tip_mm=mechanical_center_mm,
        corrector="image",
    ).validate()


def _create_deflector(key):
    definition = IMAGE_CORRECTOR_BY_KEY[key]
    component_type = _DEFLECTOR_TYPES[key]
    mechanical_center_mm = _mechanical_center_from_tip(definition)
    return component_type(
        name=f"Image Corrector {definition.label} Deflector",
        key=key,
        z_mm=mechanical_center_mm,
        kick_x_mrad=0.0,
        kick_y_mrad=0.0,
        effective_thickness_mm=(
            definition.active_length_mm
            or definition.mechanical_length_mm
        ),
        enabled=True,
        colour="#5c6bc0",
        mechanical_center_from_tip_mm=mechanical_center_mm,
        mechanical_length_mm=definition.mechanical_length_mm,
        mechanical_outer_diameter_mm=definition.outer_diameter_mm,
        mechanical_clear_bore_diameter_mm=(
            2.0 * definition.effective_aperture_radius_mm
        ),
        optical_reference_from_tip_mm=mechanical_center_mm,
        maximum_kick_mrad=definition.maximum_kick_mrad,
        corrector="image",
    ).validate()


def create_image_corrector_component(key):
    if key in _ROUND_LENS_TYPES:
        return _create_round_lens(key)
    if key in _HEXAPOLE_TYPES:
        return _create_hexapole(key)
    if key in _QUADRUPOLE_TYPES:
        return _create_quadrupole(key)
    if key in _DEFLECTOR_TYPES:
        return _create_deflector(key)
    if key == IMAGE_CORRECTOR_SAD_PLANE:
        return ImageCorrectorSadPlaneComponent().validate()
    raise KeyError(f"Unknown Image Corrector component: {key}")


def create_image_corrector_components():
    return tuple(
        create_image_corrector_component(key)
        for key in IMAGE_CORRECTOR_KEYS
    )


def create_image_corrector_lenses():
    return [
        create_image_corrector_component(key)
        for key in IMAGE_CORRECTOR_LENS_KEYS
    ]


def create_image_corrector_elements():
    return [
        create_image_corrector_component(key)
        for key in IMAGE_CORRECTOR_ELEMENT_KEYS
    ]


def is_image_corrector_component(component):
    """Return whether a runtime object has its canonical owned class."""

    key = str(getattr(component, "key", ""))
    expected_type = (
        _ROUND_LENS_TYPES.get(key)
        or _HEXAPOLE_TYPES.get(key)
        or _QUADRUPOLE_TYPES.get(key)
        or _DEFLECTOR_TYPES.get(key)
        or (
            ImageCorrectorSadPlaneComponent
            if key == IMAGE_CORRECTOR_SAD_PLANE
            else None
        )
    )
    return expected_type is not None and isinstance(
        component, expected_type
    )


def image_corrector_component_from_dict(data):
    values = dict(data)
    key = str(values.get("key", ""))
    component = create_image_corrector_component(key)
    if key in _ROUND_LENS_TYPES:
        return restore_round_lens(component, values)
    if key in _HEXAPOLE_TYPES:
        return restore_hexapole(component, values)
    if key in _QUADRUPOLE_TYPES:
        return restore_quadrupole(component, values)
    if key in _DEFLECTOR_TYPES:
        return restore_single_plane_deflector(component, values)
    if key == IMAGE_CORRECTOR_SAD_PLANE:
        object.__setattr__(component, "_position_coupling_ready", False)
        allowed = component.__dataclass_fields__
        for attribute, value in values.items():
            if attribute in allowed:
                object.__setattr__(component, attribute, value)
        object.__setattr__(component, "_position_coupling_ready", True)
        return component.apply_optical_position().validate()
    raise KeyError(f"Unknown Image Corrector component: {key}")


def create_image_corrector_ol_post_lens():
    return create_image_corrector_component(
        IMAGE_CORRECTOR_OL_POST_LENS
    )


def image_corrector_ol_post_lens_from_dict(data):
    values = dict(data)
    values["key"] = IMAGE_CORRECTOR_OL_POST_LENS
    return image_corrector_component_from_dict(values)


class ImageCorrectorSystem:
    """State-backed aggregate matching the ProbeCorrectorSystem pattern."""

    def __init__(self, state):
        self.state = state

    def component(self, key):
        if key in IMAGE_CORRECTOR_LENS_KEYS:
            return next(item for item in self.state.lenses if item.key == key)
        return next(
            item
            for item in self.state.corrector_elements
            if item.key == key
        )

    @property
    def ol_post_lens(self):
        return self.component(IMAGE_CORRECTOR_OL_POST_LENS)

    @property
    def hpol_hexapole(self):
        return self.component(IMAGE_CORRECTOR_HPOL_HEXAPOLE)

    @property
    def qpol_quadrupole(self):
        return self.component(IMAGE_CORRECTOR_QPOL_QUADRUPOLE)

    @property
    def dp11_deflector(self):
        return self.component(IMAGE_CORRECTOR_DP11_DEFLECTOR)

    @property
    def tl11_lens(self):
        return self.component(IMAGE_CORRECTOR_TL11_LENS)

    @property
    def dp12_deflector(self):
        return self.component(IMAGE_CORRECTOR_DP12_DEFLECTOR)

    @property
    def tl12_lens(self):
        return self.component(IMAGE_CORRECTOR_TL12_LENS)

    @property
    def dph1_deflector(self):
        return self.component(IMAGE_CORRECTOR_DPH1_DEFLECTOR)

    @property
    def hp1_hexapole(self):
        return self.component(IMAGE_CORRECTOR_HP1_HEXAPOLE)

    @property
    def dp21_deflector(self):
        return self.component(IMAGE_CORRECTOR_DP21_DEFLECTOR)

    @property
    def tl21_lens(self):
        return self.component(IMAGE_CORRECTOR_TL21_LENS)

    @property
    def dp22_deflector(self):
        return self.component(IMAGE_CORRECTOR_DP22_DEFLECTOR)

    @property
    def tl22_lens(self):
        return self.component(IMAGE_CORRECTOR_TL22_LENS)

    @property
    def dph2_deflector(self):
        return self.component(IMAGE_CORRECTOR_DPH2_DEFLECTOR)

    @property
    def hp2_hexapole(self):
        return self.component(IMAGE_CORRECTOR_HP2_HEXAPOLE)

    @property
    def adapter_lens(self):
        return self.component(IMAGE_CORRECTOR_ADAPTER_LENS)

    @property
    def ish_deflector(self):
        return self.component(IMAGE_CORRECTOR_ISH_DEFLECTOR)

    @property
    def dsh_deflector(self):
        return self.component(IMAGE_CORRECTOR_DSH_DEFLECTOR)

    @property
    def dstg_quadrupole(self):
        return self.component(IMAGE_CORRECTOR_DSTG_QUADRUPOLE)

    @property
    def components(self):
        return tuple(self.component(key) for key in IMAGE_CORRECTOR_KEYS)

    @property
    def round_lens_components(self):
        return tuple(
            self.component(key) for key in IMAGE_CORRECTOR_LENS_KEYS
        )

    @property
    def corrector_components(self):
        return tuple(
            self.component(key) for key in IMAGE_CORRECTOR_ELEMENT_KEYS
        )

    @property
    def deflector_components(self):
        return tuple(
            component
            for component in self.corrector_components
            if isinstance(component, SinglePlaneDeflectorComponent)
        )

    @property
    def quadrupole_components(self):
        return tuple(
            component
            for component in self.corrector_components
            if isinstance(component, QuadrupoleComponent)
        )

    @property
    def hexapole_components(self):
        return tuple(
            component
            for component in self.corrector_components
            if isinstance(component, HexapoleComponent)
        )

    @property
    def sad_plane(self):
        return self.component(IMAGE_CORRECTOR_SAD_PLANE)

    @property
    def relay_input_plane_z_mm(self):
        """Objective Lens image plane at the Image Corrector entrance."""

        return float(self.state.objective_image_plane_z_mm)

    @property
    def relay_output_plane_z_mm(self):
        return float(self.sad_plane.z_mm)

    def relay_transfer_matrix(self):
        """Full distributed-field transfer from specimen to SAD."""

        from temsim.physics.core import transfer

        return transfer(
            self.state,
            self.relay_input_plane_z_mm,
            self.relay_output_plane_z_mm,
        )

    def main_hexapole_transfer_matrix(self):
        """Internal relay between the HP1 and HP2 principal planes."""

        from temsim.physics.core import transfer

        return transfer(
            self.state,
            float(self.hp1_hexapole.z_mm),
            float(self.hp2_hexapole.z_mm),
        )

    def validate(self):
        if tuple(component.key for component in self.components) != (
            IMAGE_CORRECTOR_KEYS
        ):
            raise ValueError(
                "Image Corrector components are not in canonical order."
            )
        if len({id(component) for component in self.components}) != len(
            IMAGE_CORRECTOR_KEYS
        ):
            raise ValueError(
                "Image Corrector components must be independent instances."
            )
        for component in self.components:
            component.validate()
        return self

    def apply_optical_positions(self):
        for component in self.components:
            component.apply_optical_position()
        return self


def resolve_image_corrector_mechanical_anchors(state):
    """Resolve the installed Image Corrector as one OL-Post-anchored body.

    The OL Post upper surface follows the Image/Diff deflector lower surface.
    Every remaining corrector component keeps one editable centre offset from
    OL Post.  The image-corrected Selected Area Aperture is then anchored to
    the corrector SAD plane, which is itself part of that rigid internal chain.
    """

    system = state.image_corrector_system
    ol_post = system.ol_post_lens
    components = system.components
    offsets = state.image_corrector_component_offsets_from_ol_post_mm
    for key, value in default_image_corrector_offsets_from_ol_post_mm().items():
        offsets.setdefault(key, value)

    image_deflector = state.image_diffraction_deflector
    image_deflector_lower_surface_mm = (
        float(image_deflector.optical_center_z_mm)
        + float(image_deflector.mechanical_length_mm) / 2.0
    )
    state.image_corrector_upstream_gap_mm = (
        DEFAULT_IMAGE_CORRECTOR_UPSTREAM_GAP_MM
    )
    current_ol_upper_surface_mm = (
        float(ol_post.mechanical_center_from_tip_mm)
        - float(ol_post.mechanical_length_mm) / 2.0
    )
    previous = getattr(
        state, "_image_corrector_resolved_positions_mm", None
    )
    if isinstance(previous, dict):
        image_deflector_surface_shift_mm = (
            image_deflector_lower_surface_mm
            - float(
                previous.get(
                    "image_deflector_lower_surface",
                    image_deflector_lower_surface_mm,
                )
            )
        )
        ol_upper_shift_mm = (
            current_ol_upper_surface_mm
            - float(previous["ol_post_upper_surface"])
        )
        if abs(image_deflector_surface_shift_mm) <= 1.0e-12:
            if abs(ol_upper_shift_mm) <= 1.0e-12:
                for component in components:
                    if component.key == IMAGE_CORRECTOR_OL_POST_LENS:
                        continue
                    current_center_mm = float(
                        component.mechanical_center_from_tip_mm
                    )
                    manual_delta_mm = current_center_mm - float(
                        previous.get(component.key, current_center_mm)
                    )
                    if abs(manual_delta_mm) > 1.0e-12:
                        offsets[component.key] = (
                            float(offsets[component.key])
                            + manual_delta_mm
                        )

                selected_area_aperture = state.selected_area_aperture
                current_selected_area_center_mm = (
                    float(state.sample.z_mm)
                    + float(
                        selected_area_aperture
                        .image_corrected_mechanical_center_below_sample_mm
                    )
                )
                current_sad_center_mm = float(
                    system.sad_plane.mechanical_center_from_tip_mm
                )
                sad_shift_mm = current_sad_center_mm - float(
                    previous.get(
                        IMAGE_CORRECTOR_SAD_PLANE,
                        current_sad_center_mm,
                    )
                )
                selected_area_shift_mm = (
                    current_selected_area_center_mm
                    - float(
                        previous.get(
                            "selected_area_aperture",
                            current_selected_area_center_mm,
                        )
                    )
                )
                if (
                    abs(sad_shift_mm) <= 1.0e-12
                    and abs(selected_area_shift_mm) > 1.0e-12
                ):
                    state.selected_area_aperture_offset_from_sad_mm += (
                        selected_area_shift_mm
                    )

    ol_post_center_mm = (
        image_deflector_lower_surface_mm
        + float(state.image_corrector_upstream_gap_mm)
        + float(ol_post.mechanical_length_mm) / 2.0
    )
    ol_post.mechanical_center_from_tip_mm = ol_post_center_mm
    ol_post.optical_reference_from_tip_mm = ol_post_center_mm
    ol_post.apply_optical_position()

    resolved = {
        "image_deflector_lower_surface": (
            image_deflector_lower_surface_mm
        ),
        "ol_post_upper_surface": (
            ol_post_center_mm - float(ol_post.mechanical_length_mm) / 2.0
        ),
        IMAGE_CORRECTOR_OL_POST_LENS: ol_post_center_mm,
    }
    for component in components:
        if component.key == IMAGE_CORRECTOR_OL_POST_LENS:
            continue
        center_mm = ol_post_center_mm + float(offsets[component.key])
        component.mechanical_center_from_tip_mm = center_mm
        component.optical_reference_from_tip_mm = center_mm
        component.apply_optical_position()
        resolved[component.key] = center_mm

    sad_center_mm = float(
        system.sad_plane.mechanical_center_from_tip_mm
    )
    selected_area_center_mm = (
        sad_center_mm
        + float(state.selected_area_aperture_offset_from_sad_mm)
    )
    selected_area_aperture = state.selected_area_aperture
    selected_area_aperture.image_corrected_mechanical_center_below_sample_mm = (
        selected_area_center_mm - float(state.sample.z_mm)
    )
    selected_area_aperture.image_corrected_optical_reference_z_mm = (
        selected_area_center_mm
    )
    resolved["selected_area_aperture"] = selected_area_center_mm
    state._image_corrector_resolved_positions_mm = resolved
    return resolved
