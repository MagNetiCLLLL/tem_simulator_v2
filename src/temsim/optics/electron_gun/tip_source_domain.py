"""Physical Gaussian boundary definition and separate transport admission.

No truncation, renormalisation, change of energy or downstream source is used.
This is a model-domain budget, not a full tip-to-image accuracy certificate.
"""
import math

import numpy as np

from temsim.immutable_json import freeze_json


SOURCE_DOMAIN_SCHEMA = "gaussian-tip-forward-paraxial-v1"
TAIL_PROBABILITY_BUDGET = 1e-8
PARAXIAL_GENERATOR_RELATIVE_ERROR_BUDGET = .01
GAUSSIAN_BOUNDARY_SCHEMA = "gaussian-schell-tip-driven-boundary-v1"
GEOMETRIC_DOMAIN_SCHEMA = "gaussian-tip-geometric-forward-v1"


class TipSourceDomainError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = freeze_json(report)


def validate_tip_boundary(emitter):
    """Validate a tip-plane complex boundary without selecting a propagator.

    A Gaussian boundary field can have transverse spatial frequencies above
    the local free-electron wave number. Those are not classical ray momenta.
    A non-paraxial solver must retain them as near-field components and define
    the normal derivative and injected flux. This structural validation does
    not authorize a Wigner particle sampler or a paraxial propagation step.
    """
    from temsim.optics.electron_gun.tip_coherence import _require_emitter
    _require_emitter(emitter)
    if float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.:
        raise ValueError("The continuous curved tip needs its own complex surface boundary; "
                         "a planar Gaussian driven field cannot replace its geometry")
    return {
        "schema": GAUSSIAN_BOUNDARY_SCHEMA,
        "reference_plane": "physical FEG tip before extraction",
        "physical_boundary_status": "DEFINED",
        "emission_fwhm_nm": float(emitter.virtual_source_fwhm_nm),
        "normalization": "Boundary mode L2; physical injected flux must be established by the boundary solver",
        "injection_condition": "Normal Robin drive at the physical tip; reflection and escaping flux are solved, not prescribed downstream",
        "near_field_policy": "Retain non-propagating transverse support in the complex boundary; never turn it into rays",
        "phase_convention": "Curvature in inverse metres; historical tilt_mrad is transverse phase gradient divided by local wave number",
        "scope": "Tip boundary definition only; not transport, ray admission or image acceptance",
    }


def validate_tip_geometric_domain(emitter):
    """Admit the driven boundary's geometric-optics particle comparison.

    A ray's transverse momentum is the local phase gradient plus the declared
    incoherent tilt. The Fourier width of its amplitude is quantum diffraction,
    not an extra distribution of geometric rays. This bound contains curvature
    correlations but no quantum variance, and makes no paraxial-wave claim.
    Individual generated samples are still checked; none are clipped.
    """
    from temsim.optics.electron_gun.tip_coherence import _require_emitter
    p = _require_emitter(emitter)
    sigma = float(emitter.virtual_source_fwhm_nm)*1e-9/math.sqrt(8*math.log(2))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Tip geometric source requires positive finite spatial width")
    with np.errstate(over="ignore", invalid="ignore"):
        correlated = sigma*p.curvature
        covariance = correlated@correlated.T
        covariance += np.eye(2)*np.square(p.incoherent_angle_rms_mrad*1e-3)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Tip geometric source covariance exceeds the finite numerical range")
    std_bound = math.sqrt(max(float(np.linalg.eigvalsh(covariance)[-1]), 0.))
    centre = math.hypot(p.tilt_x_mrad, p.tilt_y_mrad)*1e-3
    radius = centre+std_bound*math.sqrt(-2*math.log(TAIL_PROBABILITY_BUDGET))
    separation = (1-centre)/std_bound if std_bound else math.inf
    forward_tail = 1. if centre >= 1 else (0. if separation > 40 else math.exp(-.5*separation**2))
    report = {
        "schema": GEOMETRIC_DOMAIN_SCHEMA,
        "reference_plane": "physical FEG tip before extraction",
        "representation": "Geometric-optics phase-gradient rays; not an exact quantum Wigner distribution",
        "mean_transverse_momentum_over_p": centre,
        "maximum_momentum_std_over_p": std_bound,
        "support_radius_over_p": radius,
        "tail_probability_budget": TAIL_PROBABILITY_BUDGET,
        "nonforward_probability_upper_bound": forward_tail,
        "tail_policy": "No clipping or renormalisation; every emitted ray is checked for forward support",
        "quantum_diffraction": "Retained by complex wave boundary, not added to geometric ray momenta",
        "scope": "Geometric particle admission only; no paraxial-wave or propagation qualification",
    }
    if not math.isfinite(radius) or radius >= 1:
        raise TipSourceDomainError(
            f"Tip geometric phase/angle distribution cannot establish forward support "
            f"at the declared tail budget (p_transverse/p bound={radius:.6g}); "
            "change the declared phase or incoherent angular spread. No momenta were clipped.", report)
    return report


def inspect_tip_source_domain(emitter):
    """Bound the full Wigner law, independent of ray/mode/energy sample count.

    For u~N(mu,S), ||u|| <= ||mu|| + sqrt(lambda_max(S))*||z||,
    z~N(0,I_2), and P(||z||>r)=exp(-r^2/2). This gives a conservative
    circular support bound including diffraction and curvature correlations.
    The lowest permitted kinetic energy bounds diffraction for every energy
    quadrature point. With no spread the actual monoenergetic value is used.

    For q=||p_transverse||/p<1, the relative error of q^2/2 against the exact
    transverse generator 1-sqrt(1-q^2) is (1-sqrt(1-q^2))/2. Require <=1%
    inside the 1-1e-8 probability region. The Gaussian tails are NOT removed;
    their probability allowance and lack of phase qualification are explicit.
    """
    from temsim.optics.electron_gun.tip_coherence import _require_emitter, wavelength_m
    _require_emitter(emitter)
    p = emitter.coherence
    sigma = float(emitter.virtual_source_fwhm_nm)*1e-9/math.sqrt(8*math.log(2))
    energy = float(emitter.minimum_kinetic_energy_ev if emitter.energy_spread_fwhm_ev > 0
                   else emitter.emission_energy_ev)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Tip source domain requires positive finite spatial width")
    quantum = float(wavelength_m(energy))/(4*math.pi*sigma)
    with np.errstate(over="ignore", invalid="ignore"):
        # Multiply in NumPy so extreme finite inputs produce a checked
        # non-finite covariance rather than an uncategorised OverflowError.
        correlated = sigma*p.curvature
        covariance = np.eye(2)*(np.square(quantum)+np.square(p.incoherent_angle_rms_mrad*1e-3))
        covariance += correlated@correlated.T
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Tip source domain covariance exceeds the finite numerical range")
    std_bound = math.sqrt(max(float(np.linalg.eigvalsh(covariance)[-1]), 0.))
    centre = math.hypot(p.tilt_x_mrad, p.tilt_y_mrad)*1e-3
    radius = centre+std_bound*math.sqrt(-2*math.log(TAIL_PROBABILITY_BUDGET))
    separation = (1-centre)/std_bound if std_bound else math.inf
    forward_tail = 1. if centre >= 1 else (0. if separation > 40 else math.exp(-.5*separation**2))
    error = radius**2/(2*(1+math.sqrt(1-radius**2))) if radius < 1 else None
    report = {"schema": SOURCE_DOMAIN_SCHEMA, "reference_plane": "physical FEG tip before extraction",
        "minimum_evaluated_energy_ev": energy, "mean_transverse_momentum_over_p": centre,
        "maximum_momentum_std_over_p": std_bound, "support_radius_over_p": radius,
        "tail_probability_budget": TAIL_PROBABILITY_BUDGET,
        "nonforward_probability_upper_bound": forward_tail,
        "paraxial_generator_relative_error_bound": error,
        "paraxial_generator_relative_error_budget": PARAXIAL_GENERATOR_RELATIVE_ERROR_BUDGET,
        "tail_policy": "No truncation or renormalisation; outside-region phase accuracy is not qualified",
        "scope": "Source-domain bound only; not propagation or image acceptance"}
    return report


def validate_tip_source_domain(emitter):
    """Require forward/paraxial admission for the existing Wigner law.

    This remains separate from a valid complex tip boundary. In particular,
    defining a non-paraxial boundary does not make q >= 1 a physical ray.
    """
    report = inspect_tip_source_domain(emitter)
    radius = report["support_radius_over_p"]
    error = report["paraxial_generator_relative_error_bound"]
    if radius >= 1:
        raise TipSourceDomainError(
            f"Tip source domain cannot establish forward support at the declared tail budget (p_transverse/p bound={radius:.6g}); "
            "a non-paraxial source model is required. Saved tip parameters were not changed.", report)
    if error > PARAXIAL_GENERATOR_RELATIVE_ERROR_BUDGET:
        raise TipSourceDomainError(
            f"Tip source exceeds the paraxial domain: generator error bound {error:.3%} > 1% "
            f"at tail probability {TAIL_PROBABILITY_BUDGET:g}. A wider-domain physical model is required; "
            "saved tip parameters were not changed.", report)
    return report


def validate_tip_particle_domain(emitter):
    """Validate the particle representation explicitly selected on the Tip."""
    from temsim.optics.electron_gun.tip_coherence import DRIVEN_GAUSSIAN_SCHELL, _require_emitter
    phase = _require_emitter(emitter)
    if phase.boundary_model == DRIVEN_GAUSSIAN_SCHELL:
        validate_tip_boundary(emitter)
        return validate_tip_geometric_domain(emitter)
    return validate_tip_source_domain(emitter)
