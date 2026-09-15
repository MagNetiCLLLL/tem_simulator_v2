"""Detached, equal-projected-D95 classical tip comparisons.

Only tip geometry and its emitting-cap angle change. In particular this does
not narrow the local energy/angular law or create a downstream source. A flat
Gaussian and a uniform spherical cap match D95, not their entire distributions.
"""
from dataclasses import replace
import math


def flat_tip_d95_nm(emitter):
    """Exact radial quantile of the emitter's Gaussian truncated at 3 sigma."""
    if emitter.surface_model is not None or emitter.coherence is not None:
        raise ValueError("D95 reference requires classical flat tip emission")
    fwhm = float(emitter.virtual_source_fwhm_nm)
    if not math.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("Flat tip FWHM must be positive and finite")
    sigma = fwhm / 2.354820045  # Same conversion as ColdFieldEmitter.emit.
    return 2*sigma*math.sqrt(-2*math.log1p(-.95*(-math.expm1(-4.5))))


def matched_cap_angle_deg(diameter95_nm, radius_nm):
    """Invert the uniform-area cap's projected radial 95% quantile stably."""
    d, r = float(diameter95_nm), float(radius_nm)
    if not all(math.isfinite(v) and v > 0 for v in (d, r)) or d >= 2*r:
        raise ValueError("Positive finite D95 smaller than twice the radius required")
    q = (d/(2*r))**2
    h = q/(1+math.sqrt(1-q))/.95  # 1-cos(theta), without cancellation.
    if h >= 1:
        raise ValueError("D95 requires a cap beyond the forward hemisphere")
    return math.degrees(2*math.asin(math.sqrt(h/2)))


def matched_curved_candidate(state, radius_nm):
    """Change a detached physical part and runtime model together.

The original curved recipe's flux density (or current prescription) is kept;
the resulting total current is NOT silently matched to the flat source. The
default Flat selection and all old files remain untouched. Consumers must
compare current fractions as well as absolute currents.
"""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.column.module_assembly import _freeze
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.optics.electron_gun.tip_assembly import (
        model_from_part, derived_envelope, validate_tip_part,
    )
    if (state.electron_gun.type_key != "cold_feg"
            or state.electron_gun.source_representation != "classical_particles"):
        raise ValueError("Comparison requires classical FEG tip particles")
    d95 = flat_tip_d95_nm(state.electron_gun.emitter)
    candidate = capture_instrument_snapshot(state).restore()
    assembly = candidate._resolved_assembly
    part = assembly.part("feg_tip")
    data = dict(part.data)
    data.update(tip_radius_nm=float(radius_nm),
                emission_cap_half_angle_deg=matched_cap_angle_deg(d95, radius_nm))
    model = model_from_part(data)
    data.update(derived_envelope(model))
    validate_tip_part(data)
    origin = part.start_z_mm-float(part.data["local_start_z_mm"])
    new_part = replace(part, start_z_mm=origin+data["local_start_z_mm"],
        center_z_mm=origin+data["local_center_z_mm"],
        end_z_mm=origin+data["local_end_z_mm"], data=_freeze(data))
    modules = []
    for module in assembly.modules:
        if module.key == part.module_key:
            module = replace(module, parts=tuple(replace(p,
                start_z_mm=new_part.start_z_mm-origin,
                center_z_mm=new_part.center_z_mm-origin, end_z_mm=new_part.end_z_mm-origin,
                data=new_part.data) if p.key == part.key else p for p in module.parts))
        modules.append(module)
    assembly = replace(assembly, modules=tuple(modules),
        parts=tuple(new_part if p.key == part.key else p for p in assembly.parts))
    apply_physical_layout_to_state(candidate, assembly=assembly, preserve_operating_parameters=True)
    emitter = candidate.electron_gun.emitter
    emitter.curvature_nm_inv = 0.0  # explicit historical-model comparison
    emitter.surface_model = model
    emitter.coherence = None
    emitter.tip_radius_nm = model.geometry.apex_radius_nm
    emitter.tip_cone_half_angle_deg = model.geometry.cone_half_angle_deg
    candidate.electron_gun.validate()
    return candidate
