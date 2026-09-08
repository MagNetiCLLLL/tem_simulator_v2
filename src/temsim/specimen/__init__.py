"""Real CIF specimen sources, shared geometry and numerical wave templates."""

from temsim.specimen.presets import (
    SpecimenColumn,
    SpecimenPreset,
    available_specimen_presets,
    default_specimen_preset_key,
    load_specimen_preset,
)
from temsim.specimen.source import (
    active_cif_path,
    active_specimen_source,
    selected_reference_preset_key,
    specimen_is_vacuum,
    specimen_structure_available,
    wave_template_preset_key,
)
from temsim.specimen.interaction_types import (
    ConservationCheck,
    IncidentElectronRay,
    IncidentRayBundle,
    InteractionEvent,
    InteractionProcess,
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
    SpecimenModelCoupling,
    SpecimenObservable,
)
from temsim.specimen.scene import SceneMaterialRegion, SpecimenScene

__all__ = [
    "SpecimenColumn",
    "SpecimenPreset",
    "available_specimen_presets",
    "default_specimen_preset_key",
    "load_specimen_preset",
    "active_cif_path",
    "active_specimen_source",
    "selected_reference_preset_key",
    "specimen_is_vacuum",
    "specimen_structure_available",
    "wave_template_preset_key",
    "ConservationCheck",
    "IncidentElectronRay",
    "IncidentRayBundle",
    "InteractionEvent",
    "InteractionProcess",
    "SpecimenInteractionRequest",
    "SpecimenInteractionResult",
    "SpecimenModelCoupling",
    "SpecimenObservable",
    "SceneMaterialRegion",
    "SpecimenScene",
]
