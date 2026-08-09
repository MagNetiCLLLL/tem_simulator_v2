from temsim.optics.model import *

from temsim.optics.corrector_structure import ensure_corrector_structure

from temsim.optics.objective_station import normalise_objective_station

from temsim.detector.recording_system import ensure_recording_system

from temsim.component_names import normalise_component_names
from temsim.optics.image_corrector import (
    create_image_corrector_lenses,
)
from temsim.optics.condenser_lens import create_condenser_lenses
from temsim.optics.condenser_aperture import (
    create_condenser_aperture_2,
    create_condenser_aperture_3,
)
from temsim.optics.condenser_deflector import create_condenser_deflector
from temsim.optics.beam_deflector import create_beam_deflector
from temsim.optics.mini_condenser import create_mini_condenser
from temsim.optics.condenser_stigmator import (
    create_condenser_stigmator,
)
from temsim.optics.objective_stigmator import (
    create_objective_stigmator,
)
from temsim.optics.diffraction_stigmator import (
    create_diffraction_stigmator,
)
from temsim.optics.diffraction_lens import create_diffraction_lens
from temsim.optics.intermediate_lens import create_intermediate_lens
from temsim.optics.projector_lens_p1 import create_projector_lens_p1
from temsim.optics.projector_lens_p2 import create_projector_lens_p2
from temsim.optics.image_diffraction_deflector import (
    create_image_diffraction_deflector,
)
from temsim.optics.objective_lens import create_objective_lens
from temsim.optics.objective_aperture import create_objective_aperture
from temsim.optics.selected_area_aperture import (
    create_selected_area_aperture,
)
from temsim.optics.energy_filter_entrance_aperture import (
    create_energy_filter_entrance_aperture,
)
from temsim.optics.probe_corrector import (
    create_adapter_lens,
    create_dp12_scan_deflector,
    create_tl12_lens,
    create_tl21_lens,
    create_tl22_lens,
)
from temsim.component_keys import (
    OBJECTIVE_APERTURE,
    SELECTED_AREA_APERTURE,
)


def image_corrector_lenses():
    """Create the five independent Image Corrector round lenses."""
    return create_image_corrector_lenses()



def default_state():

    lenses = [

        *create_condenser_lenses(),

        create_adapter_lens(),

        create_tl22_lens(),

        create_tl21_lens(),

        create_tl12_lens(),

        create_mini_condenser(),

        create_objective_lens(),

        # CETCOR round lenses are present in state for field calibration but
        # disabled unless an image-corrector topology is selected.
        *image_corrector_lenses(),

        create_diffraction_lens(),

        create_intermediate_lens(),

        create_projector_lens_p1(),

        create_projector_lens_p2(),

    ]
    for lens in lenses:
        if lens.key.startswith("ic_"):
            lens.enabled = False


    # Condenser/filter apertures start inserted. The objective and selected-area
    # aperture cartridges are present mechanically but retracted by default.
    # Positions are normalized simulator

    # coordinates preserving the documented optical-plane order, not claimed

    # common-column coordinates, not claimed manufacturer dimensions.

    apertures = [

        create_condenser_aperture_2(),

        create_condenser_aperture_3(),

        create_objective_aperture(),

        create_selected_area_aperture(),

        create_energy_filter_entrance_aperture(),

    ]


    stigmators = [

        create_condenser_stigmator(),

        create_objective_stigmator(),

        create_diffraction_stigmator(),

    ]


    deflectors = [

        create_condenser_deflector(),

        create_beam_deflector(),

        create_dp12_scan_deflector(),

        # Titan alignment documentation places image coils below the objective lens.

        # In the normalized simulator they sit after the objective-aperture/BFP

        # station and before the selected-area/first-image-plane station.

        create_image_diffraction_deflector(),

    ]


    state = State(lenses, apertures, stigmators, deflectors)

    if hasattr(state, 'objective_coupled'):

        state.objective_coupled = True

    if hasattr(state, 'sync_objective'):

        state.sync_objective()


    state.monochromator_installed = False

    state.corrector_mode = "probe_corrector"

    state.energy_filter_mode = "energy_filter"

    state.column_mode = "three_lens"

    state.probe_corrector_installed=True

    state.image_corrector_installed=False

    state.energy_filter_installed=True

    state = normalise_component_names(
        ensure_recording_system(ensure_corrector_structure(state))
    )
    from temsim.optics.beam_deflector import (
        resolve_beam_deflector_after_active_aperture,
    )
    resolve_beam_deflector_after_active_aperture(state)
    from temsim.optics.probe_corrector import (
        anchor_probe_corrector_to_beam_deflector,
    )
    anchor_probe_corrector_to_beam_deflector(state)
    from temsim.optics.selected_area_aperture import (
        resolve_standalone_selected_area_aperture_anchor,
    )
    from temsim.column.state_layout import (
        resolve_selected_area_downstream_anchors,
    )
    resolve_standalone_selected_area_aperture_anchor(state)
    resolve_selected_area_downstream_anchors(
        state,
        image_corrected=False,
    )
    from temsim.column.state_layout import apply_physical_layout_to_state
    apply_physical_layout_to_state(
        state, preserve_operating_parameters=False
    )
    return state
