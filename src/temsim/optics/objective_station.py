from temsim.component_keys import (
    OBJECTIVE_APERTURE,
    OBJECTIVE_STIGMATOR,
    IMAGE_DIFFRACTION_DEFLECTOR,
    SELECTED_AREA_APERTURE,
)
from dataclasses import asdict

from temsim.optics.objective_aperture import (
    ObjectiveApertureComponent,
    objective_aperture_from_dict,
)
from temsim.optics.selected_area_aperture import (
    SelectedAreaApertureComponent,
    selected_area_aperture_from_dict,
)
from temsim.optics.objective_stigmator import (
    ObjectiveStigmatorComponent,
    objective_stigmator_from_dict,
)
from temsim.optics.image_diffraction_deflector import (
    ImageDiffractionDeflectorComponent,
    image_diffraction_deflector_from_dict,
)


def normalise_objective_station(state):

    """Apply the V6.0.16 normalised objective-station geometry.


    These are editable simulator coordinates, not claimed manufacturer

    mechanical dimensions.

    """

    stigs={item.key:item for item in state.stigmators}

    defs={item.key:item for item in state.deflectors}

    aps={item.key:item for item in state.apertures}


    if OBJECTIVE_STIGMATOR in stigs:
        stigmator = stigs[OBJECTIVE_STIGMATOR]
        if not isinstance(stigmator, ObjectiveStigmatorComponent):
            migrated = objective_stigmator_from_dict(asdict(stigmator))
            state.stigmators[
                state.stigmators.index(stigmator)
            ] = migrated
        else:
            stigmator.name = "Objective Stigmator"

    if IMAGE_DIFFRACTION_DEFLECTOR in defs:
        deflector = defs[IMAGE_DIFFRACTION_DEFLECTOR]
        if not isinstance(
            deflector, ImageDiffractionDeflectorComponent
        ):
            migrated = image_diffraction_deflector_from_dict(
                asdict(deflector)
            )
            state.deflectors[
                state.deflectors.index(deflector)
            ] = migrated
        else:
            deflector.name = "Image / Diffraction Deflector Pair"

    if OBJECTIVE_APERTURE in aps:
        aperture = aps[OBJECTIVE_APERTURE]
        if not isinstance(aperture, ObjectiveApertureComponent):
            migrated = objective_aperture_from_dict(
                asdict(aperture),
                state.sample.z_mm,
            )
            state.apertures[
                state.apertures.index(aperture)
            ] = migrated
        else:
            aperture.name = "Objective Aperture"

    if SELECTED_AREA_APERTURE in aps:
        aperture = aps[SELECTED_AREA_APERTURE]
        if not isinstance(aperture, SelectedAreaApertureComponent):
            migrated = selected_area_aperture_from_dict(asdict(aperture))
            state.apertures[
                state.apertures.index(aperture)
            ] = migrated
        else:
            aperture.name = "Selected Area Aperture"

    if hasattr(state, "objective_lens"):

        state.objective_lens.sync_to_sample(state.sample)

    # Normalisation is also an assembly boundary: selected TOML coordinates
    # replace any legacy or runtime mechanical positions.
    from temsim.column.state_layout import apply_physical_layout_to_state
    apply_physical_layout_to_state(state)

    return state
