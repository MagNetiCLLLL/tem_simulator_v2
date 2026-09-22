"""Current component identifiers and explicit input validation."""

NANOPULSER_DEFLECTOR = "nanopulser_deflector"
NANOPULSER_APERTURE = "nanopulser_aperture"

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
IMAGE_CORRECTOR_TL12_LENS = "image_tl12_lens"
IMAGE_CORRECTOR_DPH1_DEFLECTOR = "image_dph1_deflector"
IMAGE_CORRECTOR_HP1_HEXAPOLE = "image_hp1_hexapole"
IMAGE_CORRECTOR_DP21_DEFLECTOR = "image_dp21_deflector"
IMAGE_CORRECTOR_TL21_LENS = "image_tl21_lens"
IMAGE_CORRECTOR_DP22_DEFLECTOR = "image_dp22_deflector"
IMAGE_CORRECTOR_TL22_LENS = "image_tl22_lens"
IMAGE_CORRECTOR_DPH2_DEFLECTOR = "image_dph2_deflector"
IMAGE_CORRECTOR_HP2_HEXAPOLE = "image_hp2_hexapole"
IMAGE_CORRECTOR_ADAPTER_LENS = "image_adapter_lens"
IMAGE_CORRECTOR_ISH_DEFLECTOR = "image_ish_deflector"
IMAGE_CORRECTOR_DSH_DEFLECTOR = "image_dsh_deflector"
IMAGE_CORRECTOR_DSTG_QUADRUPOLE = "image_dstg_quadrupole"
IMAGE_CORRECTOR_SAD_PLANE = "image_sad_plane"

# Logical optical plane shared by the concentric STEM detector bank.  This is
# not an independently selectable detector and does not add a recording stop.
STEM_DIFFRACTION_REFERENCE_PLANE = "stem_diffraction_reference_plane"

IMAGE_CORRECTOR_LENS_KEYS = (
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL12_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
    IMAGE_CORRECTOR_ADAPTER_LENS,
)

IMAGE_CORRECTOR_ELEMENT_KEYS = (
    IMAGE_CORRECTOR_HPOL_HEXAPOLE,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    IMAGE_CORRECTOR_DP11_DEFLECTOR,
    IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_DPH2_DEFLECTOR,
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
    IMAGE_CORRECTOR_TL12_LENS,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR,
    IMAGE_CORRECTOR_HP1_HEXAPOLE,
    IMAGE_CORRECTOR_DP21_DEFLECTOR,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_TL22_LENS,
    IMAGE_CORRECTOR_DPH2_DEFLECTOR,
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
# The slit is the alternate optical mode of the canonical C1 aperture
# mechanism.  It also has a co-located mechanical-only manifest row so the
# distinct blade carrier is visible without introducing a second stop plane.
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
ENERGY_FILTER_TAPERED_PRISM = "energy_filter_tapered_prism"
ENERGY_FILTER_SLIT = "energy_filter_slit"
ENERGY_FILTER_MULTIPOLE_KEYS = tuple(
    f"energy_filter_multipole_{index:02d}" for index in range(1, 11)
)
ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE = (
    "energy_filter_dynamic_focus_electrostatic_quadrupole"
)
ENERGY_FILTER_BIAS_TUBE = "energy_filter_bias_tube"
ENERGY_FILTER_SHUTTER = "energy_filter_shutter"
ENERGY_FILTER_CAMERA_DEFLECTOR = "energy_filter_camera_deflector"
ENERGY_FILTER_EFTEM_OUTPUT_PLANE = "energy_filter_eftem_output_plane"
ENERGY_FILTER_ZEBRA = "energy_filter_zebra"
ENERGY_FILTER_INTERNAL_KEYS = (
    ENERGY_FILTER_ENTRANCE_APERTURE,
    ENERGY_FILTER_TAPERED_PRISM,
    *ENERGY_FILTER_MULTIPOLE_KEYS,
    ENERGY_FILTER_SLIT,
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
    ENERGY_FILTER_BIAS_TUBE,
    ENERGY_FILTER_SHUTTER,
    ENERGY_FILTER_CAMERA_DEFLECTOR,
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
    ENERGY_FILTER_ZEBRA,
)

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
EDS_DETECTOR_SYSTEM = "eds_detector_system"
POST_PROJECTOR_DETECTOR_CHAMBER = "post_projector_detector_chamber"
PROJECTION_CHAMBER_DPA_APERTURE = (
    "projection_chamber_dpa_aperture"
)
FLUORESCENT_SCREEN = "flu_screen"
CAMERA = "camera"
STEM_DETECTOR_KEYS = (
    HAADF_DETECTOR,
    DARK_FIELD_DETECTOR,
    BRIGHT_FIELD_DETECTOR,
)
# Retain the established persisted identifier while the Objective Stigmator
# gains a canonical owned runtime component.
OBJECTIVE_STIGMATOR = "objective_stigmator"
IMAGE_DIFFRACTION_DEFLECTOR = "image_diffraction_deflector"
DESCAN_DEFLECTOR = "descan_deflector"
OBJECTIVE_LENS = "objective_lens"

APERTURE_KEYS = (
    GUN_EXTRACTOR_APERTURE,
    THERMIONIC_ANODE_APERTURE,
    C1_APERTURE,
    THERMIONIC_C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    OBJECTIVE_APERTURE,
    SELECTED_AREA_APERTURE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    PROJECTION_CHAMBER_DPA_APERTURE,
    NANOPULSER_APERTURE,
)

# Hardware insertion is distinct from simulator geometry adjustability.
FIXED_APERTURE_KEYS = frozenset({
    GUN_EXTRACTOR_APERTURE,
    THERMIONIC_ANODE_APERTURE,
    PROJECTION_CHAMBER_DPA_APERTURE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    NANOPULSER_APERTURE,
})

# Retired identifiers are rejected, never translated into active components.
RETIRED_COMPONENT_KEYS = frozenset({
    "corr_pre_def", "corr_mid_def", "corr_post_def",
    "lorentz", "bsh_btlt", "tem_corrector",
    'ac',
    'adf',
    'adl',
    'beam_def',
    'c1',
    'c2',
    'c3',
    'ccd_cmos',
    'ceta',
    'cond_def',
    'cond_stig',
    'dc',
    'dc_deflector',
    'descan',
    'df_s',
    'diff',
    'diff_stig',
    'dp11',
    'dp12_virtual',
    'dp21',
    'dp22',
    'dph1',
    'dph2',
    'energy_filter_entrance_m12',
    'energy_filter_exit_m12',
    'fluorescent_screen',
    'hp1',
    'hp2',
    'hpc',
    'hpol',
    'ic_adl',
    'ic_dp12',
    'ic_dp21',
    'ic_dp22',
    'ic_dph1',
    'ic_dph2',
    'ic_dsh_dstg',
    'ic_hp1',
    'ic_hp2',
    'ic_hpol_qpol_dp11',
    'ic_ish',
    'ic_ol_post',
    'ic_sad_plane',
    'ic_tl11',
    'ic_tl12',
    'ic_tl21',
    'ic_tl22',
    'il',
    'image_def',
    'integrated_mini_condenser',
    'lobj',
    'minic',
    'obj_stig',
    'p1',
    'p2',
    'post_scan_def',
    'qpc',
    'qph1',
    'qph2',
    'qpol',
    'standalone_mini_condenser',
    'tl1',
    'tl12',
    'tl21',
    'tl22',
    'uobj',
})


def require_current_component_key(key, *, expected=None):
    """Validate an identifier without changing its value or physical owner."""
    if not isinstance(key, str) or not key or key.strip() != key:
        raise ValueError("A current component key must be a nonempty string without surrounding whitespace")
    if key in RETIRED_COMPONENT_KEYS:
        raise ValueError(f"Retired component key {key!r} is unsupported; use an explicit current component")
    if expected is not None and key != expected:
        raise ValueError(f"Expected current component key {expected!r}, found {key!r}")
    return key


def require_current_lens_key(key, *, expected=None):
    """Require a current lens identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_aperture_key(key, *, expected=None):
    """Require a current aperture identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_stigmator_key(key, *, expected=None):
    """Require a current stigmator identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_corrector_element_key(key, *, expected=None):
    """Require a current corrector element identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_deflector_key(key, *, expected=None):
    """Require a current deflector identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_component_placement_key(key, *, expected=None):
    """Require a current component placement identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_recording_plane_key(key, *, expected=None):
    """Require a current recording plane identifier; no alias conversion."""
    return require_current_component_key(key, expected=expected)


def require_current_binding_key(key):
    """Validate a typed layout binding while preserving its exact identity."""
    if not isinstance(key, str) or not key or key.strip() != key:
        raise ValueError("A current layout binding must be a nonempty string")
    if ":" in key:
        category, component = key.split(":", 1)
        if not category or not component:
            raise ValueError(f"Invalid layout binding {key!r}")
        require_current_component_key(component)
    else:
        require_current_component_key(key)
    return key
