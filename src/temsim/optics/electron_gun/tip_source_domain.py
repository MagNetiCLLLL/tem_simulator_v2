"""Shared forward/paraxial admission for the declared Gaussian tip law.

No truncation, renormalisation, change of energy or downstream source is used.
This is a model-domain budget, not a full tip-to-image accuracy certificate.
"""
import math

import numpy as np

from temsim.immutable_json import freeze_json


SOURCE_DOMAIN_SCHEMA = "gaussian-tip-forward-paraxial-v1"
TAIL_PROBABILITY_BUDGET = 1e-8
PARAXIAL_GENERATOR_RELATIVE_ERROR_BUDGET = .01


class TipSourceDomainError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = freeze_json(report)


def validate_tip_source_domain(emitter):
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
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
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
