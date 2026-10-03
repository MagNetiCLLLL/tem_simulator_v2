"""Explicit tip-only inputs for the separate coherent development workflow.

Preparing a wave request never changes the active particle instrument. A
captured instrument is restored first, then an explicitly selected quantum
boundary is attached to its tip. No exit/specimen amplitude is configurable.
The inexpensive source summary is not propagation or full-chain admission.
"""
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

from temsim.immutable_json import freeze_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.tip_coherence import TipCoherence, generate_tip_emission
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence


@dataclass(frozen=True)
class CoherentSourceSettings:
    enabled: bool = False
    incoherent_angle_rms_mrad: float = 0.
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
            raise ValueError("Coherent tip selection must be boolean")
        for name in asdict(self):
            if name == "enabled":
                continue
            value = getattr(self, name)
            if value is None and name != "incoherent_angle_rms_mrad":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Coherent tip {name} must be finite")
        if self.incoherent_angle_rms_mrad < 0:
            raise ValueError("Tip incoherent angular RMS must be non-negative")
        if self.surface_mean_energy_ev is not None and self.surface_mean_energy_ev <= 0:
            raise ValueError("Coherent surface mean energy must be positive")
        if self.surface_energy_rms_ev is not None and self.surface_energy_rms_ev < 0:
            raise ValueError("Coherent surface energy RMS must be non-negative")
        for name in ("tip_fwhm_nm", "tip_mean_energy_ev", "tip_minimum_energy_ev"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Coherent {name} must be positive")
        if self.tip_energy_spread_fwhm_ev is not None and self.tip_energy_spread_fwhm_ev < 0:
            raise ValueError("Coherent tip energy FWHM must be non-negative")
        return self


def default_gaussian_tip_settings():
    """User-selected defaults for the idealised planar diffraction example.

    These are prescribed physical-tip inputs, designed for the 1.5 nm specimen
    intensity FWHM with the example's fixed optics. They are not measured metal
    tip properties or an executed downstream beam. Selection stays explicit;
    saved coherent boundaries and classical source defaults are not replaced.
    """
    return CoherentSourceSettings(
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


def prepare_coherent_state(state, settings=CoherentSourceSettings()):
    """Return a detached captured instrument with an explicit tip boundary.

    Explicit optional edits apply only at the physical tip in this detached
    session. Current, geometry and all installed optics are preserved. Surface
    classical direction/energy probabilities do
    not specify a quantum reservoir; first use requires explicit mean and RMS.
    """
    settings.validate()
    if not settings.enabled:
        raise ValueError("Select a coherent tip boundary before calculating a wave")
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    working = capture_instrument_snapshot(state).restore()
    emitter = working.electron_gun.emitter
    surface = emitter.surface_model
    if surface is None:
        if float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.:
            raise ValueError("The continuous curved tip needs its own coherent surface boundary; "
                             "a flat Gaussian wave cannot replace that geometry")
        previous = emitter.coherence
        if previous is not None and not isinstance(previous, TipCoherence):
            raise ValueError("Unsupported saved tip coherence model")
        for control, field in (("tip_fwhm_nm", "virtual_source_fwhm_nm"),
                ("tip_mean_energy_ev", "emission_energy_ev"),
                ("tip_minimum_energy_ev", "minimum_kinetic_energy_ev"),
                ("tip_energy_spread_fwhm_ev", "energy_spread_fwhm_ev")):
            value = getattr(settings, control)
            if value is not None:
                setattr(emitter, field, float(value))
        emitter.validate()
        phase = {field: float(getattr(settings, control)) for control, field in (
            ("tip_curvature_x_m1", "curvature_x_m1"), ("tip_curvature_xy_m1", "curvature_xy_m1"),
            ("tip_curvature_y_m1", "curvature_y_m1"),
            ("tip_offset_x_nm", "offset_x_nm"), ("tip_offset_y_nm", "offset_y_nm"),
            ("tip_tilt_x_mrad", "tilt_x_mrad"), ("tip_tilt_y_mrad", "tilt_y_mrad"))
            if getattr(settings, control) is not None}
        emitter.coherence = replace(previous or TipCoherence(),
            incoherent_angle_rms_mrad=float(settings.incoherent_angle_rms_mrad), **phase).validate()
    else:
        if any(value is not None for name, value in asdict(settings).items() if name.startswith("tip_")):
            raise ValueError("Gaussian tip edits cannot replace a physical surface reservoir")
        coherence = surface.coherence
        updates = {name: float(value) for name, value in (
            ("mean_energy_ev", settings.surface_mean_energy_ev),
            ("energy_rms_ev", settings.surface_energy_rms_ev),
            ("edge_phase_rad", settings.surface_edge_phase_rad)) if value is not None}
        if coherence is None:
            if "mean_energy_ev" not in updates or "energy_rms_ev" not in updates:
                raise ValueError("A new coherent surface reservoir requires explicit tip mean energy and energy RMS; "
                                 "classical emission probabilities do not define its phase")
            coherence = SurfaceCoherence(**updates).validate()
        else:
            coherence = replace(coherence, **updates).validate()
        emitter.surface_model = replace(surface, coherence=coherence).validate()
    return working


def wave_input_summary(state, request):
    """Inspect tip inputs and mode cost without creating wave arrays/fields.

    SOURCE_READY means only that the declared source fits its development
    model. Installed-field, aperture, material, sampling and phase convergence
    must still be checked by the actual pipeline; they are never implied here.
    """
    request.validate()
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    emitter = state.electron_gun.emitter
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
    else:
        result.update(model="Coherent cap reservoir; two-way round-gun development",
            inputs={"geometry": asdict(surface.geometry), "current_na": surface.current_na},
            coherence=None if surface.coherence is None else asdict(surface.coherence))
        if surface.coherence is None:
            result["reason"] = "Classical surface flux has no prescribed complex emission boundary"
            return freeze_json(result)
        try:
            surface.validate()
            energies, _ = surface.coherence.energy_quadrature(request.surface.energy_samples)
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
