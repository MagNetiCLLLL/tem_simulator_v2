from dataclasses import dataclass, asdict
from temsim.optics.image_corrector import (
    create_image_corrector_elements,
    image_corrector_component_from_dict,
    is_image_corrector_component,
)
from temsim.component_keys import (
    AC_DEFLECTOR,
    ADAPTER_LENS,
    DC_DEFLECTOR,
    DESCAN_DEFLECTOR,
    IMAGE_CORRECTOR_OL_POST_LENS,
    IMAGE_CORRECTOR_ELEMENT_KEYS,
    IMAGE_CORRECTOR_LENS_KEYS,
    PROBE_DP22_DEFLECTOR,
    PROBE_DP21_DEFLECTOR,
    PROBE_DP11_DEFLECTOR,
    PROBE_DPH2_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR,
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
)
from temsim.optics.ac_deflector import (
    AcDeflectorComponent,
    ac_deflector_from_dict,
    create_ac_deflector,
)
from temsim.optics.descan_deflector import (
    DescanDeflectorComponent,
    create_descan_deflector,
    descan_deflector_from_dict,
)
from temsim.optics.probe_corrector import (
    create_dph2_deflector,
    create_dp22_deflector,
    create_dp21_deflector,
    create_dp11_deflector,
    create_dph1_deflector,
    create_hp1_hexapole,
    create_hpol_hexapole,
    create_hp2_hexapole,
    create_hpc_hexapole,
    create_qpc_quadrupole,
    create_qph1_quadrupole,
    create_qpol_quadrupole,
    create_qph2_quadrupole,
    dph2_deflector_from_dict,
    qph2_quadrupole_from_dict,
)



@dataclass

class CorrectorElement:

    key: str

    name: str

    z_mm: float

    length_mm: float

    kind: str

    corrector: str

    colour: str

    enabled: bool = True

    optical_active: bool = False

    note: str = ""



def default_corrector_elements():

    # Normalised simulator coordinates. The sequence is grounded in the FEI

    # double-corrector column overview and the CEOS exported-element order; the

    # distances are editable simulator defaults, not manufacturer dimensions.

    image_elements = create_image_corrector_elements()

    return [

        create_dph2_deflector(),

        create_qph2_quadrupole(),

        create_hp2_hexapole(),

        create_dp22_deflector(),

        create_hpc_hexapole(),

        create_qpc_quadrupole(),

        create_dp21_deflector(),

        create_dph1_deflector(),

        create_qph1_quadrupole(),

        create_hp1_hexapole(),

        create_hpol_hexapole(),

        create_qpol_quadrupole(),

        create_dp11_deflector(),

        create_ac_deflector(),


        # FEI column order after objective post-field: objective stigmator,

        # image deflector, descan deflector, TEM corrector, diffraction section.

        create_descan_deflector(),

        *image_elements,

    ]



def ensure_corrector_structure(state):

    state.deflectors[:]=[q for q in state.deflectors if str(q.key) not in {"corr_pre_def","corr_mid_def","corr_post_def"}]

    normalised_elements = []
    for item in getattr(state, "corrector_elements", []):
        key = getattr(item, "key", None)
        if key == "dph2":
            item = dph2_deflector_from_dict(asdict(item))
        elif key == "qph2":
            item = qph2_quadrupole_from_dict(asdict(item))
        elif key in {"dc", DC_DEFLECTOR}:
            # DC deflector has been removed from the column. Silently discard
            # legacy saved rows instead of restoring the obsolete component.
            continue
        elif (
            key in {"ac", AC_DEFLECTOR}
            and not isinstance(item, AcDeflectorComponent)
        ):
            item = ac_deflector_from_dict(asdict(item))
        elif (
            key in {"descan", DESCAN_DEFLECTOR}
            and not isinstance(item, DescanDeflectorComponent)
        ):
            item = descan_deflector_from_dict(asdict(item))
        elif (
            key in IMAGE_CORRECTOR_ELEMENT_KEYS
            and not is_image_corrector_component(item)
        ):
            item = image_corrector_component_from_dict(asdict(item))
        elif key in {
            "ic_hpol_qpol_dp11",
            "ic_dsh_dstg",
            "ic_ol_post",
            "ic_tl11",
            "ic_tl21",
            "ic_tl22",
            "ic_adl",
        }:
            continue
        normalised_elements.append(item)
    unique_elements = []
    unique_modular_keys = {
        PROBE_DPH2_DEFLECTOR,
        PROBE_DP22_DEFLECTOR,
        PROBE_DP21_DEFLECTOR,
        PROBE_DP11_DEFLECTOR,
        PROBE_DPH1_DEFLECTOR,
        PROBE_HP1_HEXAPOLE,
        PROBE_HPOL_HEXAPOLE,
        PROBE_QPH2_QUADRUPOLE,
        PROBE_QPC_QUADRUPOLE,
        PROBE_QPH1_QUADRUPOLE,
        PROBE_QPOL_QUADRUPOLE,
        PROBE_HP2_HEXAPOLE,
        PROBE_HPC_HEXAPOLE,
        AC_DEFLECTOR,
        DESCAN_DEFLECTOR,
        *IMAGE_CORRECTOR_ELEMENT_KEYS,
    }
    seen_modular_keys = set()
    for item in normalised_elements:
        if item.key in unique_modular_keys:
            if item.key in seen_modular_keys:
                continue
            seen_modular_keys.add(item.key)
        unique_elements.append(item)
    state.corrector_elements = unique_elements

    defaults = default_corrector_elements()

    if not getattr(state,"corrector_elements",None):

        state.corrector_elements=defaults

    else:

        present = {item.key for item in state.corrector_elements}

        state.corrector_elements.extend(
            item for item in defaults if item.key not in present
        )

    state.corrector_elements[:]=[
        item for item in state.corrector_elements
        if item.key not in {
            "dc", DC_DEFLECTOR,
            "adl", ADAPTER_LENS,
            "tl22", PROBE_TL22_LENS,
            "tl21", PROBE_TL21_LENS,
            "tl12", PROBE_TL12_LENS,
            "dp12_virtual", IMAGE_CORRECTOR_OL_POST_LENS,
            "lorentz", "bsh_btlt", "tem_corrector",
            "ic_hpol_qpol_dp11", "ic_dsh_dstg",
            "ic_dp12", "ic_hp1", "ic_dp21", "ic_dp22",
            "ic_hp2", "ic_ish", "ic_sad_plane",
        }
    ]


    probe_on=bool(getattr(state,"probe_corrector_installed",True))

    image_on=bool(getattr(state,"image_corrector_installed",False))
    enabled_reference = getattr(state, "layout_reference_enabled", None)
    if not isinstance(enabled_reference, dict):
        enabled_reference = {}
        state.layout_reference_enabled = enabled_reference


    # Physical corrector elements follow their installation checkbox.

    for item in state.corrector_elements:

        if item.corrector=="probe":

            item.enabled=probe_on

        elif item.corrector=="image":
            binding = f"corrector:{item.key}"
            was_installed = bool(
                getattr(item, "_layout_installed", True)
            )
            if image_on:
                item.enabled = bool(enabled_reference.get(
                    binding,
                    getattr(
                        item,
                        "_layout_enabled_preference",
                        getattr(item, "enabled", True),
                    ),
                ))
                item._layout_enabled_preference = item.enabled
            else:
                if was_installed:
                    item._layout_enabled_preference = bool(
                        getattr(item, "enabled", True)
                    )
                elif binding in enabled_reference:
                    item._layout_enabled_preference = bool(
                        enabled_reference[binding]
                    )
                item.enabled = False
            item._layout_installed = image_on

        elif item.key == DESCAN_DEFLECTOR:

            item._layout_installed = True

        else:

            item.enabled=True


    # ADL/TL22/TL21/TL12 are the actual paraxial round lenses of the probe

    # corrector. They must leave the ray-transfer chain when the corrector is

    # uninstalled, not merely disappear from the overlay.

    for lens in state.lenses:

        if str(lens.key) in {
            ADAPTER_LENS, PROBE_TL22_LENS,
            PROBE_TL21_LENS, PROBE_TL12_LENS,
        }:

            lens.enabled=probe_on

        elif str(lens.key) in IMAGE_CORRECTOR_LENS_KEYS:
            binding = f"lens:{lens.key}"
            was_installed = bool(
                getattr(lens, "_layout_installed", True)
            )
            if image_on:
                lens.enabled = bool(enabled_reference.get(
                    binding,
                    getattr(
                        lens,
                        "_layout_enabled_preference",
                        getattr(lens, "enabled", True),
                    ),
                ))
                lens._layout_enabled_preference = lens.enabled
            else:
                if was_installed:
                    lens._layout_enabled_preference = bool(
                        getattr(lens, "enabled", True)
                    )
                elif binding in enabled_reference:
                    lens._layout_enabled_preference = bool(
                        enabled_reference[binding]
                    )
                lens.enabled = False
            lens._layout_installed = image_on


    # V6.0.17 kept three grouped compatibility deflectors. The explicit

    # corrector elements now own these planes, so the legacy groups must never

    # steer rays or create a second set of labels.

    for pair in state.deflectors:

        if str(pair.key) in {"corr_pre_def","corr_mid_def","corr_post_def"}:

            pair.enabled=False


    return state



def serialise_corrector_structure(state):

    ensure_corrector_structure(state)

    return [asdict(item) for item in state.corrector_elements]
