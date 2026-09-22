from dataclasses import dataclass, asdict
from temsim.optics.image_corrector import (
    create_image_corrector_elements,
)
from temsim.component_keys import (
    AC_DEFLECTOR,
    require_current_component_key,
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
    create_ac_deflector,
)
from temsim.optics.descan_deflector import (
    DescanDeflectorComponent,
    create_descan_deflector,
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

    for pair in state.deflectors:
        require_current_component_key(pair.key)
    defaults = default_corrector_elements()
    expected_types = {item.key: type(item) for item in defaults}
    elements = list(getattr(state, "corrector_elements", ()))
    seen = set()
    separately_owned = {ADAPTER_LENS, PROBE_TL22_LENS, PROBE_TL21_LENS,
                        PROBE_TL12_LENS, IMAGE_CORRECTOR_OL_POST_LENS,
                        "probe_dp12_scan_deflector"}
    for item in elements:
        key = require_current_component_key(item.key)
        if key in separately_owned:
            raise ValueError(f"{key}: this component belongs to the current lens or deflector collection")
        if key in seen:
            raise ValueError(f"Duplicate corrector component key {key!r}")
        seen.add(key)
        expected = expected_types.get(key)
        if expected is not None and not isinstance(item, expected):
            raise ValueError(f"{key}: current corrector component type {expected.__name__} is required")
    elements.extend(item for item in defaults if item.key not in seen)
    state.corrector_elements = elements

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


    return state



def serialise_corrector_structure(state):

    ensure_corrector_structure(state)

    return [item.to_dict() if isinstance(item, (AcDeflectorComponent, DescanDeflectorComponent))
            else asdict(item) for item in state.corrector_elements]
