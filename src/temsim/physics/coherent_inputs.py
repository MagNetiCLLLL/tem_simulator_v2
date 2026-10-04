"""One authoritative physical tip for particle and wave representations.

Tip edits are validated transactionally before the caller publishes them to
the instrument. Wave preparation only captures that published source; it must
never install a different source in a private calculation copy. Source
preflight remains distinct from propagation and full-chain admission.
"""
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

from temsim.immutable_json import freeze_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.tip_coherence import (
    DRIVEN_GAUSSIAN_SCHELL, TIP_BOUNDARY_MODELS, TipCoherence, generate_tip_emission,
)


@dataclass(frozen=True)
class TipEmissionSettings:
    """A pending edit to Tip emission, never a separate calculation source.

    Geometry and all applied emission/phase parameters belong to the instrument
    Tip. ``enabled`` selects its declared coherence model, not a second Tip or
    an instruction to start wave propagation.
    """
    enabled: bool = False
    boundary_model: str | None = None
    incoherent_angle_rms_mrad: float | None = None
    surface_mean_energy_ev: float | None = None
    surface_energy_rms_ev: float | None = None
    surface_edge_phase_rad: float | None = None
    tip_fwhm_nm: float | None = None
    tip_mean_energy_ev: float | None = None
    tip_minimum_energy_ev: float | None = None
    tip_energy_spread_fwhm_ev: float | None = None
    tip_curvature_x_m1: float | None = None
    tip_curvature_xy_m1: float | None = None
    tip_curvature_y_m1: float | None = None
    tip_offset_x_nm: float | None = None
    tip_offset_y_nm: float | None = None
    tip_tilt_x_mrad: float | None = None
    tip_tilt_y_mrad: float | None = None

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("Tip coherence-model selection must be boolean")
        if self.boundary_model is not None and (
                not isinstance(self.boundary_model, str) or self.boundary_model not in TIP_BOUNDARY_MODELS):
            raise ValueError("Unknown Tip boundary model")
        for name in asdict(self):
            if name in ("enabled", "boundary_model"):
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Tip parameter {name} must be finite")
        if self.incoherent_angle_rms_mrad is not None and self.incoherent_angle_rms_mrad < 0:
            raise ValueError("Tip incoherent angular RMS must be non-negative")
        if self.surface_mean_energy_ev is not None and self.surface_mean_energy_ev <= 0:
            raise ValueError("Tip mean energy must be positive")
        if self.surface_energy_rms_ev is not None and self.surface_energy_rms_ev < 0:
            raise ValueError("Tip energy RMS must be non-negative")
        for name in ("tip_fwhm_nm", "tip_mean_energy_ev", "tip_minimum_energy_ev"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Tip parameter {name} must be positive")
        if self.tip_energy_spread_fwhm_ev is not None and self.tip_energy_spread_fwhm_ev < 0:
            raise ValueError("Tip energy FWHM must be non-negative")
        return self


def idealised_diffraction_tip_settings():
    """Historical idealised planar diffraction input record, not a default.

    These are prescribed physical-tip inputs, designed for the 1.5 nm specimen
    intensity FWHM with the example's fixed optics. They are not measured metal
    tip properties or an executed downstream beam. No runtime initialization
    uses this record. Applying it requires an explicit Tip parameter edit.
    """
    return TipEmissionSettings(
        tip_fwhm_nm=28390.10000542304,
        tip_mean_energy_ev=30.,
        tip_minimum_energy_ev=.01,
        tip_energy_spread_fwhm_ev=0.,
        tip_curvature_x_m1=624.3690658100246,
        tip_curvature_xy_m1=-9.348776164576383e-12,
        tip_curvature_y_m1=624.369065810008,
        tip_offset_x_nm=-.06696090871824144,
        tip_offset_y_nm=-.07266953387249751,
        tip_tilt_x_mrad=-4.18060377176702e-5,
        tip_tilt_y_mrad=-4.537013038384244e-5,
        incoherent_angle_rms_mrad=0.,
    )


_TIP_SCALARS = (
    ("tip_fwhm_nm", "virtual_source_fwhm_nm"),
    ("tip_mean_energy_ev", "emission_energy_ev"),
    ("tip_minimum_energy_ev", "minimum_kinetic_energy_ev"),
    ("tip_energy_spread_fwhm_ev", "energy_spread_fwhm_ev"),
)
_TIP_PHASE = (
    ("tip_curvature_x_m1", "curvature_x_m1"),
    ("tip_curvature_xy_m1", "curvature_xy_m1"),
    ("tip_curvature_y_m1", "curvature_y_m1"),
    ("tip_offset_x_nm", "offset_x_nm"), ("tip_offset_y_nm", "offset_y_nm"),
    ("tip_tilt_x_mrad", "tilt_x_mrad"), ("tip_tilt_y_mrad", "tilt_y_mrad"),
)


def _tip_emitter(state):
    gun = getattr(state, "electron_gun", None)
    emitter = getattr(gun, "emitter", None)
    if (getattr(gun, "type_key", "cold_feg") != "cold_feg"
            or not hasattr(emitter, "surface_model")
            or any(not hasattr(emitter, field) for _, field in _TIP_SCALARS)):
        raise ValueError("These Tip controls are unavailable for this gun family; "
                         "only a supported cold field-emission boundary can be used")
    return emitter


def source_settings_from_state(state):
    """Read the active physical tip without selecting a model or changing it.

    A classical surface has no inferred quantum reservoir. A flat classical
    source supplies its current scalar inputs and a neutral *unselected*
    phase draft; its old angular probabilities are not a coherence model.
    """
    emitter = _tip_emitter(state)
    surface = getattr(emitter, "surface_model", None)
    if surface is not None:
        phase = surface.coherence
        return TipEmissionSettings(enabled=phase is not None,
            surface_mean_energy_ev=surface.mean_energy_ev,
            surface_energy_rms_ev=surface.energy_sigma_ev,
            surface_edge_phase_rad=0. if phase is None else phase.edge_phase_rad)
    phase = getattr(emitter, "coherence", None)
    values = {control: getattr(emitter, field) for control, field in _TIP_SCALARS}
    # This is an unselected editor draft. Only explicit Apply can publish it;
    # loaded historical phase records retain their forward-Wigner semantics.
    draft = phase or TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    values.update({control: getattr(draft, field) for control, field in _TIP_PHASE})
    return TipEmissionSettings(enabled=phase is not None,
        boundary_model=draft.boundary_model,
        incoherent_angle_rms_mrad=draft.incoherent_angle_rms_mrad, **values)


def candidate_tip_emission(state, settings):
    """Return a validated gun candidate; publishing it is the caller's action.

    No source scalar, phase, physical component or numerical quadrature on the
    current instrument is mutated on success or failure. Disabled coherence
    is an explicit return to the classical law, with scalar edits retained.
    """
    settings.validate()
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
    require_physical_gun_source(state.electron_gun)
    emitter = _tip_emitter(state)
    surface = emitter.surface_model
    values = {}
    if surface is None:
        if any(value is not None for name, value in asdict(settings).items()
               if name.startswith("surface_")):
            raise ValueError("A surface reservoir cannot replace the selected Gaussian tip geometry")
        if settings.enabled and float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.:
            raise ValueError("The continuous curved tip needs a phase model defined on its surface; "
                             "a flat Gaussian wave cannot replace that geometry")
        previous = emitter.coherence
        if previous is not None and not isinstance(previous, TipCoherence):
            raise ValueError("Unsupported saved tip coherence model")
        for control, field in _TIP_SCALARS:
            value = getattr(settings, control)
            if value is not None:
                values[field] = float(value)
        phase = {field: float(getattr(settings, control)) for control, field in _TIP_PHASE
            if getattr(settings, control) is not None}
        if settings.incoherent_angle_rms_mrad is not None:
            phase["incoherent_angle_rms_mrad"] = float(settings.incoherent_angle_rms_mrad)
        if settings.boundary_model is not None:
            phase["boundary_model"] = settings.boundary_model
        values["coherence"] = (replace(previous or TipCoherence(), **phase).validate()
                               if settings.enabled else None)
    else:
        if settings.boundary_model is not None:
            raise ValueError("Gaussian boundary model cannot replace a physical surface reservoir")
        if any(value is not None for name, value in asdict(settings).items() if name.startswith("tip_")):
            raise ValueError("Gaussian tip edits cannot replace a physical surface reservoir")
        if settings.incoherent_angle_rms_mrad is not None:
            raise ValueError("Gaussian angular coherence cannot replace a physical surface reservoir")
        from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence
        coherence = surface.coherence
        if coherence is not None and not surface.shared_boundary:
            # Keep the historical boundary readable. An edit must explicitly
            # replace it in the physical source editor, not silently reinterpret
            # its independently prescribed spectrum as actual shared emission.
            if settings != source_settings_from_state(state):
                raise ValueError("Historical surface reservoir is read-only; explicitly replace "
                                 "it with a physical tip in the source editor")
            return candidate_tip_edit(state.electron_gun, {"surface_model": surface})
        mean = (surface.mean_energy_ev if settings.surface_mean_energy_ev is None
                else float(settings.surface_mean_energy_ev))
        rms = (surface.energy_sigma_ev if settings.surface_energy_rms_ev is None
               else float(settings.surface_energy_rms_ev))
        phase = (0. if settings.surface_edge_phase_rad is None
                 else float(settings.surface_edge_phase_rad))
        if settings.enabled:
            # This explicit Apply selects a physical boundary for BOTH methods.
            # Gamma/monoenergetic total energy is owned only by SurfaceEmission;
            # the coherent record contains phase, never another energy input.
            coherence = SharedSurfaceCoherence(edge_phase_rad=phase).validate()
            emission = replace(surface.emission, kinetic_mean_ev=mean, kinetic_sigma_ev=rms,
                energy_distribution="gamma" if rms > 0 else "monoenergetic",
                maximum_angle_deg=0., flux_profile="cosine_cap",
                spatial_sampling="uniform_area", spatial_stratum_allocation=(),
                angular_sampling="uniform_cdf", angular_stratum_allocation=())
        else:
            coherence = None
            emission = surface.emission
            if mean != surface.mean_energy_ev or rms != surface.energy_sigma_ev:
                emission = replace(emission, kinetic_mean_ev=mean, kinetic_sigma_ev=rms,
                    energy_distribution="gamma" if rms > 0 else "monoenergetic")
        values["surface_model"] = replace(surface, emission=emission, coherence=coherence).validate()
    candidate = candidate_tip_edit(state.electron_gun, values)
    if settings.enabled and surface is None:
        from temsim.optics.electron_gun.tip_source_domain import (
            validate_tip_boundary, validate_tip_geometric_domain, validate_tip_source_domain,
        )
        try:
            if candidate.emitter.coherence.boundary_model == DRIVEN_GAUSSIAN_SCHELL:
                validate_tip_boundary(candidate.emitter)
                validate_tip_geometric_domain(candidate.emitter)
            else:
                validate_tip_source_domain(candidate.emitter)
        except ValueError as error:
            raise ValueError("Tip unchanged: current source model cannot represent these "
                             f"inputs. {error}") from error
    return candidate


def prepare_coherent_state(state, settings=None):
    """Capture the exact published source; optional settings assert its identity.

    Display rounding is tolerated, but no rounded value is written back. An
    unpublished source draft must go through Apply tip parameters first.
    Existing surface wave records remain readable without claiming a particle
    representation for that boundary.
    """
    if settings is not None:
        settings.validate()
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    active = source_settings_from_state(state)
    if not active.enabled or (settings is not None and not settings.enabled):
        raise ValueError("Define the Tip emission and coherence model, then click Apply tip parameters before calculating a wave")
    emitter = state.electron_gun.emitter
    if emitter.surface_model is None and float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.:
        raise ValueError("The continuous curved tip needs a phase model defined on its surface; "
                         "a flat Gaussian wave cannot replace that geometry")
    emitter.validate()
    if settings is not None:
        for name, requested in asdict(settings).items():
            if name == "enabled" or requested is None:
                continue
            published = getattr(active, name)
            if name == "boundary_model":
                if requested != published:
                    raise ValueError("Apply tip parameters changes before calculating: boundary_model differs "
                                     "from the instrument's applied Tip parameters")
                continue
            # The source editor keeps at least 12 decimal places. This only
            # accepts sub-display-quantum differences; capture retains every
            # original bit and therefore the existing scientific cache identity.
            if published is None or not math.isclose(requested, published,
                    rel_tol=2e-14, abs_tol=5.1e-13):
                raise ValueError(f"Apply tip parameters changes before calculating: {name} differs "
                                 "from the instrument's applied Tip parameters")
    return capture_instrument_snapshot(state).restore()


def wave_input_summary(state, request):
    """Inspect tip inputs and mode cost without creating wave arrays/fields.

    SOURCE_READY means only that the declared source fits its development
    model. Installed-field, aperture, material, sampling and phase convergence
    must still be checked by the actual pipeline; they are never implied here.
    """
    request.validate()
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    emitter = _tip_emitter(state)
    surface = emitter.surface_model
    result = {"status": "SOURCE_NOT_CONFIGURED", "source_plane": "physical tip before extraction",
              "scope": "Source preflight only; not wave execution or TEM/STEM qualification",
              "mode_count": None, "minimum_initial_wave_bytes": None,
              "grid_pixels": request.source.grid_pixels,
              "energy_samples": request.source.energy_samples,
              "observation_z_mm": getattr(request, "observation_z_mm", None)}
    if surface is None:
        result.update(model="Gaussian-Schell tip boundary", inputs={name: getattr(emitter, name) for name in (
            "virtual_source_fwhm_nm", "emission_energy_ev", "minimum_kinetic_energy_ev",
            "energy_spread_fwhm_ev", "emission_current_na")},
            coherence=None if emitter.coherence is None else asdict(emitter.coherence))
        if emitter.coherence is None:
            result["reason"] = "No coherent tip mutual intensity is selected"
            return freeze_json(result)
        if float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.:
            result.update(status="SOURCE_UNSUPPORTED", reason="The continuous curved tip needs a coherent surface boundary; "
                          "flat Gaussian launch cannot replace its geometry")
            return freeze_json(result)
        try:
            emission = generate_tip_emission(state.electron_gun, request.source)
        except ValueError as error:
            result.update(status="SOURCE_UNSUPPORTED", reason=str(error))
            if hasattr(error, "report"):
                result["source_domain"] = error.report
            return freeze_json(result)
        count = emission.record["mode_count"]
        result.update(source_domain=emission.record["source_domain"],
                      omitted_source_probability=emission.record["omitted_probability"])
        if emitter.coherence.boundary_model == DRIVEN_GAUSSIAN_SCHELL:
            result.update(
                model="Driven Gaussian-Schell physical Tip boundary",
                boundary=emission.record["boundary"],
                particle_domain=emission.record["particle_domain"],
                particle_representation=emission.record["physical_source"]["particle_representation"],
                transport_requirement="Non-paraxial normal-Robin boundary solve through extraction; downstream propagation must retain the executed source state",
            )
    else:
        result.update(model=("Curved tip; two-way round-gun development"
                             if surface.shared_boundary else "Coherent cap reservoir; historical wave boundary"),
            particle_representation=("Geometric-optics normal rays of the same cap flux and energy law; "
                                     "wave diffraction and reflection are not classical ray effects"
                                     if surface.shared_boundary else
                                     "UNAVAILABLE: this boundary has no shared particle sampler"),
            inputs={"geometry": asdict(surface.geometry), "current_na": surface.current_na,
                    "emission": asdict(surface.emission),
                    "current_reference": "outward injected tip flux; not net escaping current"},
            coherence=None if surface.coherence is None else asdict(surface.coherence))
        if surface.coherence is None:
            result["reason"] = "Classical surface flux has no prescribed complex emission boundary"
            return freeze_json(result)
        try:
            surface.validate()
            energies, _ = surface.energy_quadrature(request.surface.energy_samples)
        except ValueError as error:
            result.update(status="SOURCE_UNSUPPORTED", reason=str(error))
            return freeze_json(result)
        count = len(energies)
        result["energy_samples"] = request.surface.energy_samples
        result["source_symmetry"] = "One axisymmetric spatial mode per energy"
    result.update(status="SOURCE_READY", mode_count=int(count),
        minimum_initial_wave_bytes=int(count)*request.source.grid_pixels**2*16,
        reason="Source inputs are explicit; propagation must establish field support and sampling separately")
    return freeze_json(result)
