"""Canonical runtime component identifiers and load-boundary migrations."""

CONDENSER_LENS_1 = "condenser_lens_1"
CONDENSER_LENS_2 = "condenser_lens_2"
CONDENSER_LENS_3 = "condenser_lens_3"
CONDENSER_LENS_1_UPPER_POLE = "condenser_lens_1_upper_pole"
CONDENSER_LENS_1_LOWER_POLE = "condenser_lens_1_lower_pole"
CONDENSER_LENS_2_UPPER_POLE = "condenser_lens_2_upper_pole"
CONDENSER_LENS_2_LOWER_POLE = "condenser_lens_2_lower_pole"
CONDENSER_LENS_3_UPPER_POLE = "condenser_lens_3_upper_pole"
CONDENSER_LENS_3_LOWER_POLE = "condenser_lens_3_lower_pole"
CONDENSER_POLE_PIECE_KEYS = (
    CONDENSER_LENS_1_LOWER_POLE,
    CONDENSER_LENS_2_UPPER_POLE,
    CONDENSER_LENS_3_UPPER_POLE,
    CONDENSER_LENS_3_LOWER_POLE,
)
ADAPTER_LENS = "adapter_lens"
PROBE_DPH2_DEFLECTOR = "probe_dph2_deflector"
PROBE_QPH2_QUADRUPOLE = "probe_qph2_quadrupole"
PROBE_HP2_HEXAPOLE = "probe_hp2_hexapole"
PROBE_TL22_LENS = "probe_tl22_lens"
PROBE_DP22_DEFLECTOR = "probe_dp22_deflector"
PROBE_HPC_HEXAPOLE = "probe_hpc_hexapole"
PROBE_QPC_QUADRUPOLE = "probe_qpc_quadrupole"
PROBE_DP21_DEFLECTOR = "probe_dp21_deflector"
PROBE_TL21_LENS = "probe_tl21_lens"
PROBE_DPH1_DEFLECTOR = "probe_dph1_deflector"
PROBE_QPH1_QUADRUPOLE = "probe_qph1_quadrupole"
PROBE_HP1_HEXAPOLE = "probe_hp1_hexapole"
PROBE_HPOL_HEXAPOLE = "probe_hpol_hexapole"
PROBE_QPOL_QUADRUPOLE = "probe_qpol_quadrupole"
PROBE_DP11_DEFLECTOR = "probe_dp11_deflector"
PROBE_TL12_LENS = "probe_tl12_lens"
PROBE_DP12_SCAN_DEFLECTOR = "probe_dp12_scan_deflector"
PROBE_CORRECTOR_KEYS = (
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
)
IMAGE_CORRECTOR_OL_POST_LENS = "image_ol_post_lens"
IMAGE_CORRECTOR_HPOL_HEXAPOLE = "image_hpol_hexapole"
IMAGE_CORRECTOR_QPOL_QUADRUPOLE = "image_qpol_quadrupole"
IMAGE_CORRECTOR_DP11_DEFLECTOR = "image_dp11_deflector"
IMAGE_CORRECTOR_TL11_LENS = "image_tl11_lens"
IMAGE_CORRECTOR_DP12_DEFLECTOR = "image_dp12_deflector"
IMAGE_CORRECTOR_HP1_HEXAPOLE = "image_hp1_hexapole"
IMAGE_CORRECTOR_DP21_DEFLECTOR = "image_dp21_deflector"
IMAGE_CORRECTOR_TL21_LENS = "image_tl21_lens"
IMAGE_CORRECTOR_DP22_DEFLECTOR = "image_dp22_deflector"
IMAGE_CORRECTOR_TL22_LENS = "image_tl22_lens"
IMAGE_CORRECTOR_HP2_HEXAPOLE = "image_hp2_hexapole"
IMAGE_CORRECTOR_ADAPTER_LENS = "image_adapter_lens"
IMAGE_CORRECTOR_ISH_DEFLECTOR = "image_ish_deflector"
IMAGE_CORRECTOR_DSH_DEFLECTOR = "image_dsh_deflector"
IMAGE_CORRECTOR_DSTG_QUADRUPOLE = "image_dstg_quadrupole"
IMAGE_CORRECTOR_SAD_PLANE = "image_sad_plane"

IMAGE_CORRECTOR_LENS_KEYS = (
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
    IMAGE_CORRECTOR_ADAPTER_LENS,
)

IMAGE_CORRECTOR_ELEMENT_KEYS = (
    IMAGE_CORRECTOR_HPOL_HEXAPOLE,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    IMAGE_CORRECTOR_DP11_DEFLECTOR,
    IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_HP2_HEXAPOLE,
    IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR,
    IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_SAD_PLANE,
)

IMAGE_CORRECTOR_KEYS = (
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
)

CONDENSER_LENS_KEYS = (
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
)

CONDENSER_APERTURE_2 = "condenser_aperture_2"
FEG_TIP = "feg_tip"
FEG_EXTRACTOR = "feg_extractor"
FEG_ELECTROSTATIC_LENS = "feg_electrostatic_lens"
GUN_EXTRACTOR_APERTURE = "feg_dpa_aperture"
FEG_ACCELERATOR = "feg_accelerator"
FEG_DEFLECTOR = "feg_deflector"
FEG_STIGMATOR = "feg_stigmator"
C1_APERTURE = "feg_c1_aperture"
FEG_MONOCHROMATOR_WIEN = "feg_monochromator_wien"
# Legacy input identifier only. The FEI slit is now the alternate operating
# mode of the canonical C1 aperture mechanism and has no runtime/layout row.
FEG_MONOCHROMATOR_SLIT = "feg_monochromator_slit"
FEG_MONOCHROMATOR_COMPONENT_KEYS = (
    FEG_MONOCHROMATOR_WIEN,
)
THERMIONIC_CATHODE = "thermionic_cathode"
THERMIONIC_WEHNELT = "thermionic_wehnelt"
THERMIONIC_GUN_LENS = "thermionic_gun_lens"
THERMIONIC_ANODE_APERTURE = "thermionic_anode_aperture"
THERMIONIC_ACCELERATOR = "thermionic_accelerator"
THERMIONIC_DEFLECTOR = "thermionic_deflector"
THERMIONIC_STIGMATOR = "thermionic_stigmator"
THERMIONIC_C1_APERTURE = "thermionic_c1_aperture"
THERMIONIC_GUN_COMPONENT_KEYS = (
    THERMIONIC_CATHODE,
    THERMIONIC_WEHNELT,
    THERMIONIC_GUN_LENS,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_STIGMATOR,
    THERMIONIC_C1_APERTURE,
)
CONDENSER_APERTURE_3 = "condenser_aperture_3"
OBJECTIVE_APERTURE = "objective_aperture"
SELECTED_AREA_APERTURE = "selected_area_aperture"
ENERGY_FILTER_ENTRANCE_APERTURE = "energy_filter_entrance_aperture"
ENERGY_FILTER_ENTRANCE_M12 = "energy_filter_entrance_m12"
ENERGY_FILTER_EXIT_M12 = "energy_filter_exit_m12"
ENERGY_FILTER_SLIT = "energy_filter_slit"

CONDENSER_APERTURE_KEYS = (
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
)

CONDENSER_DEFLECTOR = "condenser_deflector"
BEAM_DEFLECTOR = "beam_deflector"
DC_DEFLECTOR = "dc_deflector"
AC_DEFLECTOR = "ac_deflector"
MINI_CONDENSER = "mini_condenser"
CONDENSER_STIGMATOR = "condenser_stigmator"
DIFFRACTION_STIGMATOR = "diffraction_stigmator"
DIFFRACTION_LENS = "diffraction_lens"
DIFFRACTION_LENS_UPPER_POLE = "diffraction_lens_upper_pole"
DIFFRACTION_LENS_LOWER_POLE = "diffraction_lens_lower_pole"
INTERMEDIATE_LENS = "intermediate_lens"
INTERMEDIATE_LENS_UPPER_POLE = "intermediate_lens_upper_pole"
INTERMEDIATE_LENS_LOWER_POLE = "intermediate_lens_lower_pole"
PROJECTOR_LENS_1 = "projector_lens_1"
PROJECTOR_LENS_1_UPPER_POLE = "projector_lens_1_upper_pole"
PROJECTOR_LENS_1_LOWER_POLE = "projector_lens_1_lower_pole"
PROJECTOR_LENS_2 = "projector_lens_2"
PROJECTOR_LENS_2_UPPER_POLE = "projector_lens_2_upper_pole"
PROJECTOR_LENS_2_LOWER_POLE = "projector_lens_2_lower_pole"
PROJECTOR_SYSTEM_POLE_PIECE_KEYS = (
    DIFFRACTION_LENS_UPPER_POLE,
    DIFFRACTION_LENS_LOWER_POLE,
    INTERMEDIATE_LENS_UPPER_POLE,
    INTERMEDIATE_LENS_LOWER_POLE,
    PROJECTOR_LENS_1_UPPER_POLE,
    PROJECTOR_LENS_1_LOWER_POLE,
    PROJECTOR_LENS_2_UPPER_POLE,
    PROJECTOR_LENS_2_LOWER_POLE,
)
HAADF_DETECTOR = "haadf"
DARK_FIELD_DETECTOR = "df"
BRIGHT_FIELD_DETECTOR = "bf"
FLUORESCENT_SCREEN = "flu_screen"
CAMERA = "camera"
EELS_PLANE = "eels_plane"
VIRTUAL_OBSERVATION_PLANE = "virtual_observation_plane"
STEM_DETECTOR_KEYS = (
    HAADF_DETECTOR,
    DARK_FIELD_DETECTOR,
    BRIGHT_FIELD_DETECTOR,
)
# Retain the established persisted identifier while the Objective Stigmator
# gains a canonical owned runtime component.
OBJECTIVE_STIGMATOR = "obj_stig"
IMAGE_DIFFRACTION_DEFLECTOR = "image_diffraction_deflector"
DESCAN_DEFLECTOR = "descan_deflector"
OBJECTIVE_LENS = "objective_lens"

APERTURE_KEYS = (
    GUN_EXTRACTOR_APERTURE,
    C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    OBJECTIVE_APERTURE,
    SELECTED_AREA_APERTURE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
)

_LEGACY_LENS_KEYS = {
    "c1": CONDENSER_LENS_1,
    "c2": CONDENSER_LENS_2,
    "c3": CONDENSER_LENS_3,
    "adl": ADAPTER_LENS,
    "tl22": PROBE_TL22_LENS,
    "tl21": PROBE_TL21_LENS,
    "tl1": PROBE_TL12_LENS,
    "tl12": PROBE_TL12_LENS,
    "minic": MINI_CONDENSER,
    "integrated_mini_condenser": MINI_CONDENSER,
    "standalone_mini_condenser": MINI_CONDENSER,
    "uobj": OBJECTIVE_LENS,
    "lobj": OBJECTIVE_LENS,
    "diff": DIFFRACTION_LENS,
    "il": INTERMEDIATE_LENS,
    "p1": PROJECTOR_LENS_1,
    "p2": PROJECTOR_LENS_2,
    "ic_ol_post": IMAGE_CORRECTOR_OL_POST_LENS,
    "ic_tl11": IMAGE_CORRECTOR_TL11_LENS,
    "ic_tl21": IMAGE_CORRECTOR_TL21_LENS,
    "ic_tl22": IMAGE_CORRECTOR_TL22_LENS,
    "ic_adl": IMAGE_CORRECTOR_ADAPTER_LENS,
}
def canonical_lens_key(key):
    """Translate legacy condenser keys only at persistence/input boundaries."""
    return _LEGACY_LENS_KEYS.get(str(key), str(key))


def canonical_aperture_key(key):
    """Return an aperture key without introducing runtime aliases."""
    return str(key)


_LEGACY_STIGMATOR_KEYS = {
    "cond_stig": CONDENSER_STIGMATOR,
    "obj_stig": OBJECTIVE_STIGMATOR,
    "diff_stig": DIFFRACTION_STIGMATOR,
}


def canonical_stigmator_key(key):
    """Translate historical stigmator keys at input boundaries."""
    return _LEGACY_STIGMATOR_KEYS.get(str(key), str(key))


_LEGACY_CORRECTOR_ELEMENT_KEYS = {
    "dph2": PROBE_DPH2_DEFLECTOR,
    "qph2": PROBE_QPH2_QUADRUPOLE,
    "hp2": PROBE_HP2_HEXAPOLE,
    "tl22": PROBE_TL22_LENS,
    "dp22": PROBE_DP22_DEFLECTOR,
    "hpc": PROBE_HPC_HEXAPOLE,
    "qpc": PROBE_QPC_QUADRUPOLE,
    "dp21": PROBE_DP21_DEFLECTOR,
    "tl21": PROBE_TL21_LENS,
    "dph1": PROBE_DPH1_DEFLECTOR,
    "qph1": PROBE_QPH1_QUADRUPOLE,
    "hp1": PROBE_HP1_HEXAPOLE,
    "hpol": PROBE_HPOL_HEXAPOLE,
    "qpol": PROBE_QPOL_QUADRUPOLE,
    "dp11": PROBE_DP11_DEFLECTOR,
    "tl12": PROBE_TL12_LENS,
    "dp12_virtual": PROBE_DP12_SCAN_DEFLECTOR,
    "dc": DC_DEFLECTOR,
    "ac": AC_DEFLECTOR,
    "descan": DESCAN_DEFLECTOR,
    "ic_dp12": IMAGE_CORRECTOR_DP12_DEFLECTOR,
    "ic_hp1": IMAGE_CORRECTOR_HP1_HEXAPOLE,
    "ic_dp21": IMAGE_CORRECTOR_DP21_DEFLECTOR,
    "ic_dp22": IMAGE_CORRECTOR_DP22_DEFLECTOR,
    "ic_hp2": IMAGE_CORRECTOR_HP2_HEXAPOLE,
    "ic_ish": IMAGE_CORRECTOR_ISH_DEFLECTOR,
    "ic_sad_plane": IMAGE_CORRECTOR_SAD_PLANE,
}


def canonical_corrector_element_key(key):
    return _LEGACY_CORRECTOR_ELEMENT_KEYS.get(str(key), str(key))


_LEGACY_DEFLECTOR_KEYS = {
    "cond_def": CONDENSER_DEFLECTOR,
    "beam_def": BEAM_DEFLECTOR,
    "post_scan_def": PROBE_DP12_SCAN_DEFLECTOR,
    "image_def": IMAGE_DIFFRACTION_DEFLECTOR,
}


def canonical_deflector_key(key):
    """Translate the old condenser-deflector key at input boundaries."""
    return _LEGACY_DEFLECTOR_KEYS.get(str(key), str(key))


def canonical_component_placement_key(key):
    return canonical_corrector_element_key(
        canonical_deflector_key(
            canonical_stigmator_key(
                canonical_aperture_key(canonical_lens_key(key))
            )
        )
    )


_LEGACY_RECORDING_PLANE_KEYS = {
    "df_s": DARK_FIELD_DETECTOR,
    "adf": DARK_FIELD_DETECTOR,
    "fluorescent_screen": FLUORESCENT_SCREEN,
    "ceta": CAMERA,
    "ccd_cmos": CAMERA,
}


def canonical_recording_plane_key(key):
    """Translate historical detector-plane keys at input boundaries."""
    return _LEGACY_RECORDING_PLANE_KEYS.get(str(key), str(key))


def canonical_binding_key(key):
    text = str(key)
    if text.startswith("recording:"):
        prefix, recording_key = text.split(":", 1)
        return f"{prefix}:{canonical_recording_plane_key(recording_key)}"
    if text == "corrector:tl22":
        return f"lens:{PROBE_TL22_LENS}"
    if text == "corrector:tl21":
        return f"lens:{PROBE_TL21_LENS}"
    if text == "corrector:tl12":
        return f"lens:{PROBE_TL12_LENS}"
    if text == "corrector:dp12_virtual":
        return f"deflector:{PROBE_DP12_SCAN_DEFLECTOR}"
    if text == "stigmator:hpol":
        return f"corrector:{PROBE_HPOL_HEXAPOLE}"
    if text.startswith("lens:"):
        prefix, lens_key = text.split(":", 1)
        return f"{prefix}:{canonical_lens_key(lens_key)}"
    if text.startswith("aperture:"):
        prefix, aperture_key = text.split(":", 1)
        return f"{prefix}:{canonical_aperture_key(aperture_key)}"
    if text.startswith("deflector:"):
        prefix, deflector_key = text.split(":", 1)
        return f"{prefix}:{canonical_deflector_key(deflector_key)}"
    if text.startswith("stigmator:"):
        prefix, stigmator_key = text.split(":", 1)
        return f"{prefix}:{canonical_stigmator_key(stigmator_key)}"
    if text.startswith("corrector:"):
        prefix, element_key = text.split(":", 1)
        return (
            f"{prefix}:{canonical_corrector_element_key(element_key)}"
        )
    return text
