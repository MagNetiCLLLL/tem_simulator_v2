"""Transactional edits at the tip; no propagation or downstream source."""
from copy import copy


def candidate_tip_edit(gun, values):
    """Validate the entire gun before publishing a source/model change."""
    if gun.type_key != "cold_feg":
        raise ValueError("Tip emission editing requires a cold FEG")
    candidate = copy(gun)
    candidate.emitter = copy(gun.emitter)
    candidate.c1_aperture = copy(gun.c1_aperture)
    allowed = set(gun.emitter.__dataclass_fields__) | {"coherence", "surface_model", "curvature_nm_inv", "curvature_model"}
    if set(values) - allowed:
        raise ValueError("Only tip emission inputs may be edited here")
    for name, value in values.items():
        setattr(candidate.emitter, name, value)
    candidate.source_representation = "classical_particles"
    candidate._trace_cache = candidate._trace_cache_key = None
    candidate.validate()
    return candidate


def tip_model_label(gun):
    model = getattr(gun.emitter, "surface_model", None)
    if model is not None:
        return ("Curved tip · emitting apex cap "
                f"{model.emission.cap_half_angle_deg:g}° · {model.current_na:g} nA")
    k = getattr(gun.emitter, "curvature_nm_inv", 0.0)
    geometry = f"Curved tip · curvature {k:g} nm^-1 · R {1/k:g} nm" if k else "Flat tip · curvature 0"
    from temsim.optics.electron_gun.tip_curvature import ANGLE_ONLY_MODEL, LEGACY_MODEL
    scope = ("historical angle only; launch Z = 0" if gun.emitter.curvature_model == ANGLE_ONLY_MODEL
             else "historical sag + direction" if gun.emitter.curvature_model == LEGACY_MODEL
             else "centre Z = 0; edges bend upstream")
    return f"{geometry} · {scope} · analytic gun field · {gun.emitted_current_a*1e9:g} nA"
