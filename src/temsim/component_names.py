"""Canonical display names for physical column components."""

from temsim.optics.condenser_lens import CONDENSER_LENS_DEFINITION_BY_KEY
from temsim.optics.condenser_aperture import (
    CONDENSER_APERTURE_DEFINITION_BY_KEY,
)
from temsim.component_keys import (
    C1_APERTURE,
    CONDENSER_STIGMATOR,
    DIFFRACTION_LENS,
    DIFFRACTION_STIGMATOR,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    ENERGY_FILTER_TAPERED_PRISM,
    ENERGY_FILTER_SLIT,
    ENERGY_FILTER_MULTIPOLE_KEYS,
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
    ENERGY_FILTER_BIAS_TUBE,
    ENERGY_FILTER_SHUTTER,
    ENERGY_FILTER_CAMERA_DEFLECTOR,
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
    ENERGY_FILTER_ZEBRA,
    FEG_DEFLECTOR,
    FEG_STIGMATOR,
    GUN_EXTRACTOR_APERTURE,
    MINI_CONDENSER,
    IMAGE_DIFFRACTION_DEFLECTOR,
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL12_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
    IMAGE_CORRECTOR_ADAPTER_LENS,
    DESCAN_DEFLECTOR,
    DARK_FIELD_DETECTOR,
    FLUORESCENT_SCREEN,
    CAMERA,
    OBJECTIVE_LENS,
    OBJECTIVE_APERTURE,
    OBJECTIVE_STIGMATOR,
    PROBE_DP12_SCAN_DEFLECTOR,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
    SELECTED_AREA_APERTURE,
)

_CONDENSER_NAMES = {
    key: definition.label
    for key, definition in CONDENSER_LENS_DEFINITION_BY_KEY.items()
}

LENS_NAMES = {
    **_CONDENSER_NAMES,
    "adapter_lens": "ADL (Adapter Lens)",
    PROBE_TL22_LENS: "TL22 Transfer Lens",
    PROBE_TL21_LENS: "TL21 Transfer Lens",
    PROBE_TL12_LENS: "TL12 Transfer Lens",
    MINI_CONDENSER: "Mini Condenser Lens",
    OBJECTIVE_LENS: "Objective Lens Assembly",
    IMAGE_CORRECTOR_OL_POST_LENS: "Image Corrector OL Post Lens",
    IMAGE_CORRECTOR_TL11_LENS: "Image Corrector TL11",
    IMAGE_CORRECTOR_TL12_LENS: "Image Corrector TL12",
    IMAGE_CORRECTOR_TL21_LENS: "Image Corrector TL21",
    IMAGE_CORRECTOR_TL22_LENS: "Image Corrector TL22",
    IMAGE_CORRECTOR_ADAPTER_LENS: "Image Corrector ADL",
    DIFFRACTION_LENS: "Diffraction Lens",
    INTERMEDIATE_LENS: "Intermediate Lens",
    PROJECTOR_LENS_1: "Projector Lens P1",
    PROJECTOR_LENS_2: "Projector Lens P2",
}

LENS_SHORT_NAMES = {
    **_CONDENSER_NAMES,
    "adapter_lens": "ADL",
    PROBE_TL22_LENS: "TL22",
    PROBE_TL21_LENS: "TL21",
    PROBE_TL12_LENS: "TL12",
    MINI_CONDENSER: "Mini C",
    OBJECTIVE_LENS: "Obj",
    IMAGE_CORRECTOR_OL_POST_LENS: "IC OL post",
    IMAGE_CORRECTOR_TL11_LENS: "IC TL11",
    IMAGE_CORRECTOR_TL12_LENS: "IC TL12",
    IMAGE_CORRECTOR_TL21_LENS: "IC TL21",
    IMAGE_CORRECTOR_TL22_LENS: "IC TL22",
    IMAGE_CORRECTOR_ADAPTER_LENS: "IC ADL",
    DIFFRACTION_LENS: "Diff L",
    INTERMEDIATE_LENS: "Int L",
    PROJECTOR_LENS_1: "P1",
    PROJECTOR_LENS_2: "P2",
}

APERTURE_NAMES = {
    GUN_EXTRACTOR_APERTURE: "Gun Aperture",
    C1_APERTURE: "C1 Aperture",
    **{
        key: definition.label
        for key, definition in (
            CONDENSER_APERTURE_DEFINITION_BY_KEY.items()
        )
    },
    OBJECTIVE_APERTURE: "Objective Aperture",
    SELECTED_AREA_APERTURE: "Selected Area Aperture",
    ENERGY_FILTER_ENTRANCE_APERTURE: (
        "Iliad Spectrometer Entrance Aperture"
    ),
}

APERTURE_SHORT_NAMES = APERTURE_NAMES

ENERGY_FILTER_NAMES = {
    ENERGY_FILTER_ENTRANCE_APERTURE: (
        "Iliad Spectrometer Entrance Aperture"
    ),
    ENERGY_FILTER_TAPERED_PRISM: "Iliad Large Tapered Prism",
    **{
        key: f"Iliad Multipole {index:02d} (model index)"
        for index, key in enumerate(ENERGY_FILTER_MULTIPOLE_KEYS, start=1)
    },
    ENERGY_FILTER_SLIT: "XO / Optional EFTEM Energy Slit",
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE: (
        "Dynamic-focus Electrostatic Quadrupole"
    ),
    ENERGY_FILTER_BIAS_TUBE: "Iliad MultiEELS Bias Tube",
    ENERGY_FILTER_SHUTTER: "Iliad Fast Electrostatic Shutter",
    ENERGY_FILTER_CAMERA_DEFLECTOR: "Iliad Zebra Camera Deflector",
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE: "Optional EFTEM Output Plane",
    ENERGY_FILTER_ZEBRA: "Iliad Zebra EELS Detector",
}

STIGMATOR_NAMES = {
    FEG_STIGMATOR: "Gun Stigmator",
    CONDENSER_STIGMATOR: "Condenser Stigmator",
    OBJECTIVE_STIGMATOR: "Objective Stigmator",
    DIFFRACTION_STIGMATOR: "Diffraction Stigmator",
}

STIGMATOR_SHORT_NAMES = {
    FEG_STIGMATOR: "Gun Stig",
    CONDENSER_STIGMATOR: "Cond Stig",
    OBJECTIVE_STIGMATOR: "Obj Stig",
    DIFFRACTION_STIGMATOR: "Diff Stig",
}

DEFLECTOR_NAMES = {
    FEG_DEFLECTOR: "Gun Deflectors",
    "condenser_deflector": "Condenser Deflector",
    "beam_deflector": "BSh/BTlt Beam Shift/Tilt Deflector",
    "corr_pre_def": "Corrector DPH2/DP22 Deflectors",
    "corr_mid_def": "Corrector DP21/DPH1 Deflectors",
    "corr_post_def": "Corrector DP11/DP12 Deflectors",
    PROBE_DP12_SCAN_DEFLECTOR: "DP12 (virtual)",
    IMAGE_DIFFRACTION_DEFLECTOR: "Image/Diffraction Deflectors",
    DESCAN_DEFLECTOR: "Descan Deflector",
}

DEFLECTOR_SHORT_NAMES = {
    FEG_DEFLECTOR: "Gun Def",
    "condenser_deflector": "Cond Def",
    "beam_deflector": "BSh/BTlt",
    "corr_pre_def": "DPH2/DP22",
    "corr_mid_def": "DP21/DPH1",
    "corr_post_def": "DP11/DP12",
    PROBE_DP12_SCAN_DEFLECTOR: "DP12",
    IMAGE_DIFFRACTION_DEFLECTOR: "Img/Diff Def",
    DESCAN_DEFLECTOR: "Descan",
}

RECORDING_PLANE_NAMES = {
    "haadf": "HAADF Detector",
    FLUORESCENT_SCREEN: "Fluorescent Screen",
    DARK_FIELD_DETECTOR: "DF Detector",
    "bf": "BF Detector",
    CAMERA: "Camera",
}

RECORDING_PLANE_SHORT_NAMES = {
    "haadf": "HAADF",
    FLUORESCENT_SCREEN: "Flu Screen",
    DARK_FIELD_DETECTOR: "DF",
    "bf": "BF",
    CAMERA: "Camera",
}


def normalise_component_names(state):
    """Apply canonical display names to known keyed components."""
    for items, names in (
        (getattr(state, "lenses", []), LENS_NAMES),
        (getattr(state, "apertures", []), APERTURE_NAMES),
        (getattr(state, "stigmators", []), STIGMATOR_NAMES),
        (getattr(state, "deflectors", []), DEFLECTOR_NAMES),
        (getattr(state, "recording_planes", []), RECORDING_PLANE_NAMES),
    ):
        for item in items:
            if getattr(item, "key", None) in names:
                item.name = names[item.key]
    return state
