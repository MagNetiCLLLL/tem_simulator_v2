"""A positive mutual intensity specified ONLY at the physical FEG launch plane.

The circular Gaussian--Schell model is an explicit tip emission distribution.
Its angular input is the *incoherent* RMS spread; diffraction and the declared
wavefront curvature supply the remaining momentum variance. It does not infer
coherence from the old, independently truncated classical launch distribution.
Current, spatial FWHM and the Young/Boersch energy law belong to the emitter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np
from scipy.constants import c, e, h, m_e
from scipy.special import ndtri

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import WaveMode
from temsim.physics.wave_reference import AxialWaveReference

TIP_REFERENCE = "physical-feg-tip-emitted-electron"
PHYSICAL_TIP_SOURCE_SCHEMA = "physical-tip-source-description-v1"
FORWARD_GAUSSIAN_SCHELL = "forward_gaussian_schell"
DRIVEN_GAUSSIAN_SCHELL = "driven_gaussian_schell"
TIP_BOUNDARY_MODELS = (FORWARD_GAUSSIAN_SCHELL, DRIVEN_GAUSSIAN_SCHELL)


def wavelength_m(energy_ev):
    energy = np.asarray(energy_ev, dtype=float) * e
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Tip kinetic energies must be finite and positive")
    return h * c / np.sqrt(energy * (energy + 2 * m_e * c * c))


@dataclass(frozen=True)
class TipCoherence:
    """Additional physical tip inputs; None on the emitter retains its old law.

    The forward model selects its positive Wigner distribution for particle
    diagnostics. The driven model instead prescribes a complex injection
    boundary: particles are the geometric-optics limit of its phase gradient
    and incoherent tilts, not the quantum transverse-frequency distribution.
    The former classical angular RMS/cutoff and
    3-sigma position cutoff are retained for the legacy emission law; they are
    not apertures and do not truncate this different, explicitly selected law.
    Installed gun/column apertures always act downstream of emission.
    """
    incoherent_angle_rms_mrad: float = 0.0
    curvature_x_m1: float = 0.0
    curvature_xy_m1: float = 0.0
    curvature_y_m1: float = 0.0
    offset_x_nm: float = 0.0
    offset_y_nm: float = 0.0
    tilt_x_mrad: float = 0.0
    tilt_y_mrad: float = 0.0
    # Missing historical fields keep their original forward-Wigner semantics.
    # A driven boundary is a different, explicitly selected source model.
    boundary_model: str = FORWARD_GAUSSIAN_SCHELL

    def validate(self):
        if not isinstance(self.boundary_model, str) or self.boundary_model not in TIP_BOUNDARY_MODELS:
            raise ValueError("Unknown Tip boundary model")
        for name, value in asdict(self).items():
            if name == "boundary_model":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"Tip coherence {name} must be finite")
        if self.incoherent_angle_rms_mrad < 0:
            raise ValueError("Tip incoherent angular RMS must be non-negative")
        return self

    @property
    def curvature(self):
        return np.array(((self.curvature_x_m1, self.curvature_xy_m1),
                         (self.curvature_xy_m1, self.curvature_y_m1)), dtype=float)


@dataclass(frozen=True)
class TipWaveNumerics:
    """Sampling controls, never a user-defined downstream beam state."""
    grid_pixels: int = 128
    energy_samples: int = 9
    mode_tail_tolerance: float = 1e-6
    maximum_modes: int = 1024

    def validate(self):
        for name, minimum, maximum in (("grid_pixels", 32, 4096),
                                        ("energy_samples", 1, 4096),
                                        ("maximum_modes", 1, 1_000_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if not math.isfinite(self.mode_tail_tolerance) or not 0 < self.mode_tail_tolerance < 1:
            raise ValueError("Mode tail tolerance must lie strictly between zero and one")
        return self


def _require_emitter(emitter):
    if getattr(emitter, "surface_model", None) is not None:
        if emitter.surface_model.coherence is not None:
            raise ValueError("A coherent surface tip must use the joint two-way round-gun development solver; the Gaussian tip model cannot replace its complex boundary.")
        raise ValueError("The grounded surface model specifies classical outgoing flux, not coherent amplitudes. Surface-to-wave transport is not yet available; no legacy source will be substituted.")
    if getattr(emitter, "kind", None) != "cold_field_emitter":
        raise ValueError("Coherent emission is implemented only at a cold FEG tip")
    parameters = getattr(emitter, "coherence", None)
    if not isinstance(parameters, TipCoherence):
        raise ValueError("Configure the FEG tip coherence parameters explicitly before wave emission")
    emitter.validate()
    parameters.validate()
    if not math.isfinite(emitter.virtual_source_fwhm_nm) or emitter.virtual_source_fwhm_nm <= 0:
        raise ValueError("A coherent tip needs positive finite spatial FWHM")
    return parameters


def describe_physical_tip_source(emitter):
    """Describe the one published source using physical, canonical labels.

    ``virtual_source_fwhm_nm`` remains the single historical stored field. Its
    value is the intensity FWHM at the physical tip emission plane; it is not
    a back-projected virtual source or an independently placed exit beam.
    This record is input provenance, never an executed upstream checkpoint.
    Particle and wave algorithms may use it without copying or editing a
    second set of source parameters.
    """
    emitter.validate()
    surface = getattr(emitter, "surface_model", None)
    common = {
        "schema": PHYSICAL_TIP_SOURCE_SCHEMA,
        "reference_id": TIP_REFERENCE,
        "origin": "physical tip before extraction and acceleration",
        "geometry": {
            "tip_radius_nm": float(emitter.tip_radius_nm),
            "tip_cone_half_angle_deg": float(emitter.tip_cone_half_angle_deg),
            "curvature_nm_inv": float(emitter.curvature_nm_inv),
        },
        "current_na": float(emitter.emitted_current_a)*1e9,
        "scope": "Published source input description, not an executed beam or image qualification",
    }
    if surface is not None:
        common.update(
            model="physical-surface-emission",
            geometry=asdict(surface.geometry),
            surface=surface.to_dict(),
        )
        if surface.shared_boundary:
            common.update(
                current_definition="Incident injection at the physical tip; net escaping wave current is computed after reflection",
                particle_representation="Normal-launch geometric-optics comparison of the same injection boundary; not an exact quantum Wigner distribution",
                mean_kinetic_energy_ev=float(surface.mean_energy_ev),
                energy_rms_ev=float(surface.energy_sigma_ev),
                spatial_flux_profile=surface.emission.flux_profile,
            )
    else:
        phase = emitter.coherence
        common.update(
            model="gaussian-schell-tip-boundary" if phase is not None else "classical-truncated-gaussian-tip",
            emission_fwhm_nm=float(emitter.virtual_source_fwhm_nm),
            width_definition="Intensity FWHM at the physical tip plane",
            mean_kinetic_energy_ev=float(emitter.emission_energy_ev),
            minimum_kinetic_energy_ev=float(emitter.minimum_kinetic_energy_ev),
            energy_spread_fwhm_ev=float(emitter.energy_spread_fwhm_ev),
            energy_spread_definition="Existing positive energy law, RMS-equivalent FWHM",
            energy_law={
                "young_decay_width_ev": float(emitter.young_decay_width_ev),
                "boersch_sigma_ev": float(emitter.boersch_sigma_ev),
                "half_range_ev": float(emitter.energy_half_range_ev),
            },
            phase=None if phase is None else asdict(phase),
        )
        if phase is None:
            common["classical_direction_law"] = {
                "angular_rms_mrad": float(emitter.angular_rms_mrad),
                "angular_cutoff_mrad": float(emitter.angular_cutoff_mrad),
                "position_cutoff_sigma": 3.,
            }
        else:
            common["boundary_model"] = phase.boundary_model
            if phase.boundary_model == DRIVEN_GAUSSIAN_SCHELL:
                common.update(
                    current_definition="Incident injection at the physical tip; net escaping wave current is computed after reflection",
                    particle_representation="Geometric-optics phase-gradient rays of the same spatial injection and energy law; not an exact quantum Wigner distribution",
                    wave_representation="Driven complex Gaussian-Schell boundary with a normal Robin injection condition",
                )
            else:
                common["particle_representation"] = "Positive Gaussian-Schell Wigner distribution within its forward/paraxial source domain"
    return freeze_json(common)


def tip_energy_samples(emitter, count):
    """The same deterministic, positive Young/Boersch law as particle emission.

    Equal-weight sampling deliberately retains the existing energy model,
    including its RMS-equivalent FWHM and finite-sample moment matching.
    Energy-sample convergence must be checked separately from mode truncation.
    """
    from temsim.optics.electron_gun.emitter import _halton_dimensions
    from temsim.physics.chromatic import cold_feg_energy_offsets
    if count == 1 and emitter.energy_spread_fwhm_ev > 0:
        raise ValueError("One energy sample would discard the configured tip energy spread")
    quantiles = _halton_dimensions(count, (11, 13))
    offsets = cold_feg_energy_offsets(
        count, emitter.energy_spread_fwhm_ev, emitter.energy_half_range_ev,
        emitter.young_decay_width_ev, emitter.boersch_sigma_ev,
        quantiles=quantiles, mean_kinetic_energy_ev=emitter.emission_energy_ev,
        minimum_kinetic_energy_ev=emitter.minimum_kinetic_energy_ev)
    return float(emitter.emission_energy_ev) + offsets


def tip_covariance(emitter, energy_ev):
    """Analytic covariance in (x,y,px/p,py/p), including phase correlations."""
    p = _require_emitter(emitter)
    sigma = float(emitter.virtual_source_fwhm_nm) * 1e-9 / math.sqrt(8 * math.log(2))
    quantum_angle = float(wavelength_m(energy_ev)) / (4 * math.pi * sigma)
    angle2 = quantum_angle**2 + (p.incoherent_angle_rms_mrad * 1e-3)**2
    position = np.eye(2) * sigma**2
    cross = position @ p.curvature
    covariance = np.block([[position, cross], [cross.T, np.eye(2)*angle2 + p.curvature@cross]])
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Tip phase-space covariance is outside the finite numerical range")
    return covariance


def tip_particle_samples(emitter, count):
    """Sample the particle representation declared by the one Tip model.

    Driven boundaries use geometric rays of their phase and incoherent tilt
    distribution. Quantum diffraction is retained in the wave calculation;
    frequencies above the free-electron wave number never become real rays.
    """
    from temsim.optics.electron_gun.base import EmissionBundle
    from temsim.optics.electron_gun.emitter import _halton_dimensions
    p = _require_emitter(emitter)
    from temsim.optics.electron_gun.tip_source_domain import (
        validate_tip_geometric_domain, validate_tip_source_domain,
    )
    driven = p.boundary_model == DRIVEN_GAUSSIAN_SCHELL
    if driven:
        validate_tip_geometric_domain(emitter)
    else:
        validate_tip_source_domain(emitter)
    if isinstance(count, bool) or int(count) != count or count < 9:
        raise ValueError("Tip particle diagnostics require at least nine samples")
    n = int(count)
    normal = np.array([ndtri(u) for u in _halton_dimensions(n, (2, 3, 5, 7))])
    energies = tip_energy_samples(emitter, n)
    sigma = float(emitter.virtual_source_fwhm_nm) * 1e-9 / math.sqrt(8 * math.log(2))
    angle = (p.incoherent_angle_rms_mrad*1e-3 if driven else
             np.hypot(wavelength_m(energies)/(4*np.pi*sigma), p.incoherent_angle_rms_mrad*1e-3))
    xy = sigma * normal[:2]
    slope = p.curvature @ xy + angle * normal[2:]
    xy += np.array((p.offset_x_nm, p.offset_y_nm))[:, None] * 1e-9
    slope += np.array((p.tilt_x_mrad, p.tilt_y_mrad))[:, None] * 1e-3
    # These are paraxial canonical momenta. Converting them to ray directions
    # must retain px and py at fixed kinetic energy, rather than normalising
    # (px/p,py/p,1), which would silently narrow the distribution.
    transverse2 = np.sum(slope*slope, axis=0)
    if np.any(transverse2 >= 1):
        representation = "geometric" if driven else "Wigner"
        raise ValueError(f"Tip {representation} samples contain non-forward momenta; change the declared phase or angular distribution")
    tangent = slope / np.sqrt(1-transverse2)
    return EmissionBundle(xy[0], xy[1], tangent[0], tangent[1],
                          energies-emitter.emission_energy_ev,
                          np.full(n, 1/n), np.arange(n, dtype=np.int64))


def _mode_row(emitter, energy, numerics):
    p = emitter.coherence
    sigma = emitter.virtual_source_fwhm_nm*1e-9/math.sqrt(8*math.log(2))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Tip spatial width is outside the representable SI range")
    quantum = float(wavelength_m(energy))/(4*math.pi*sigma)
    occupation = math.hypot(1., p.incoherent_angle_rms_mrad*1e-3/quantum)
    if not math.isfinite(occupation):
        raise ValueError("Tip phase-space extent exceeds the coherent-mode budget")
    ratio = (occupation-1)/(occupation+1)
    if occupation == 1:
        count, tail = 1, 0.
    else:
        log_ratio = math.log1p(-2/(occupation+1)) if occupation >= 2 else math.log(ratio)
        log_root = math.log(numerics.mode_tail_tolerance)-math.log1p(math.sqrt(1-numerics.mode_tail_tolerance))
        required = log_root/log_ratio
        if not math.isfinite(required) or required > math.isqrt(numerics.maximum_modes):
            raise ValueError("Tip coherent-mode budget is insufficient for the requested tail tolerance")
        count = max(1, math.ceil(required))
        t = math.exp(count*log_ratio)
        tail = t*(2-t)
    return {"energy_ev": float(energy), "modes_per_axis": count,
            "omitted_probability": tail, "sigma_m": sigma,
            "ground_sigma_m": sigma/math.sqrt(occupation), "eigenvalue_ratio": ratio,
            "diffraction_rms_mrad": quantum*1e3,
            "uncorrelated_total_rms_mrad": quantum*occupation*1e3}


@dataclass(frozen=True)
class TipEmission:
    """Immutable input-bound tip modes; not an accelerated beam checkpoint."""
    parameters: object
    numerics: TipWaveNumerics
    record: object

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze_json(self.parameters))
        object.__setattr__(self, "record", freeze_json(self.record))

    @property
    def digest(self):
        return json_digest({"parameters": self.parameters, "numerics": asdict(self.numerics), "record": self.record})

    @property
    def reference_current_a(self):
        return self.parameters["emission_current_na"]*1e-9

    def modes(self):
        p = TipCoherence(**self.parameters["coherence"])
        origin = np.array((p.offset_x_nm, p.offset_y_nm))*1e-9
        tilt = np.array((p.tilt_x_mrad, p.tilt_y_mrad))*1e-3
        prior = 1/len(self.record["energy_modes"])
        for ei, row in enumerate(self.record["energy_modes"]):
            sigma, ground, count = row["sigma_m"], row["ground_sigma_m"], row["modes_per_axis"]
            half_extent = max(9*sigma, (math.sqrt(2*count)+8)*math.sqrt(2)*ground)
            dx = 2*half_extent/self.numerics.grid_pixels
            if dx > ground/3:
                raise ValueError("Tip eigenmodes are undersampled; increase grid_pixels")
            x = (np.arange(self.numerics.grid_pixels)-self.numerics.grid_pixels//2)*dx
            coordinate = x/(math.sqrt(2)*ground)
            functions = [np.exp(-x*x/(4*ground**2))/(2*np.pi*ground**2)**.25*math.sqrt(dx)]
            if count > 1:
                functions.append(math.sqrt(2)*coordinate*functions[0])
            for n in range(1, count-1):
                functions.append(math.sqrt(2/(n+1))*coordinate*functions[n]-math.sqrt(n/(n+1))*functions[n-1])
            ratio = row["eigenvalue_ratio"]
            for ny in range(count):
                for nx in range(count):
                    plane = PlaneWave(np.outer(functions[ny], functions[nx]), np.eye(2)*dx,
                                      origin, p.curvature, tilt)
                    yield WaveMode(plane, prior*(1-ratio)**2*ratio**(nx+ny), TIP_REFERENCE,
                                   f"{self.digest}:E{ei}:HG{nx},{ny}", row["energy_ev"]*1e-3,
                                   axial_reference=AxialWaveReference(0., 0.))


def _generate_tip_emission(gun, numerics, *, driven_boundary):
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(gun)
    emitter = gun.emitter
    _require_emitter(emitter)
    numerics.validate()
    from temsim.optics.electron_gun.tip_source_domain import (
        SOURCE_DOMAIN_SCHEMA, inspect_tip_source_domain,
        validate_tip_boundary, validate_tip_geometric_domain, validate_tip_source_domain,
    )
    boundary = validate_tip_boundary(emitter) if driven_boundary else None
    if driven_boundary:
        geometric_domain = validate_tip_geometric_domain(emitter)
        try:
            source_domain = inspect_tip_source_domain(emitter)
        except ValueError as error:
            # A legal positive energy distribution may approach zero kinetic
            # energy, leaving no finite worst-case paraxial bound. Preserve
            # that explicit limitation; the driven solver still has to admit
            # every actual positive energy component and its spatial grid.
            source_domain = {
                "schema": SOURCE_DOMAIN_SCHEMA,
                "status": "BOUND_UNAVAILABLE",
                "reason": str(error),
                "scope": "No ray or paraxial admission; the complex boundary remains separately defined",
            }
    else:
        source_domain = validate_tip_source_domain(emitter)
    energies = tip_energy_samples(emitter, numerics.energy_samples)
    rows = [_mode_row(emitter, energy, numerics) for energy in energies]
    count = sum(row["modes_per_axis"]**2 for row in rows)
    if count > numerics.maximum_modes:
        raise ValueError(f"Tip emission needs {count} modes, above maximum_modes={numerics.maximum_modes}")
    parameters = {name: getattr(emitter, name) for name in (
        "emission_current_na", "emission_energy_ev", "minimum_kinetic_energy_ev",
        "virtual_source_fwhm_nm", "energy_spread_fwhm_ev", "young_decay_width_ev",
        "boersch_sigma_ev", "energy_half_range_ev")}
    parameters["coherence"] = asdict(emitter.coherence)
    record = {
        "schema": "physical-tip-mutual-intensity-v2", "plane_z_mm": 0.,
        "source_domain": source_domain,
        "reference_id": TIP_REFERENCE, "model": "circular Gaussian-Schell at FEG tip",
        "energy_modes": rows, "mode_count": count,
        "omitted_probability": float(np.mean([row["omitted_probability"] for row in rows])),
        "energy_law": "Existing positive Young/Boersch quantile law; equal-weight energy samples",
        "energy_convergence": "NOT_VALIDATED", "emission_boundary": "After tunnelling, before extraction/acceleration",
        "physical_calibration": "USER_DEFINED_TIP_NOT_MEASURED"}
    if driven_boundary:
        record.update(
            schema="physical-tip-driven-gaussian-boundary-v1",
            boundary=boundary,
            physical_source=describe_physical_tip_source(emitter),
            representation="Driven complex boundary; non-propagating support is not a ray ensemble",
            injected_flux_status="REQUIRES_BOUNDARY_SOLVER",
            particle_domain=geometric_domain,
        )
    return TipEmission(parameters, numerics, record)


def generate_tip_emission(gun, numerics=TipWaveNumerics()):
    """Generate modes under the Tip's explicitly published boundary model."""
    if getattr(getattr(gun.emitter, "coherence", None), "boundary_model", None) == DRIVEN_GAUSSIAN_SCHELL:
        return generate_tip_boundary_emission(gun, numerics)
    return _generate_tip_emission(gun, numerics, driven_boundary=False)


def generate_tip_boundary_emission(gun, numerics=TipWaveNumerics()):
    """Build the same physical tip boundary for an explicit non-paraxial solve.

    No source width, energy, phase, current or mode shape is changed. The
    result is a prescribed complex boundary with L2-normalized mode fields.
    A two-way/near-field solver must supply the injection condition and flux
    normalization; these modes must not be consumed as paraxial rays. Calling
    this opt-in function does not change the Tip's published model. Production
    selection requires boundary_model='driven_gaussian_schell' on the Tip.
    """
    return _generate_tip_emission(gun, numerics, driven_boundary=True)
