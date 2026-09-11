"""Historical Gaussian--Schell exit-source data and numerical representation.

Withdrawn: a user-defined exit ensemble bypasses the physical gun. The data
types remain readable for historical evidence. Public production and binding
entry points reject this model; it is not a source for new calculations.

Hermite--Gaussian eigenmodes and their geometric weights implement a positive
Gaussian density operator (Starikov & Wolf, JOSA 72, 923, 1982). Truncated weight
is reported, never renormalised into retained electrons. Ray samples represent
the SAME analytic Gaussian Wigner distribution, not its modal truncation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.special import ndtri

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import WaveMode

MODEL_ID = "gun-exit-gaussian-schell-v1"
REFERENCE_ID = "gun_exit_source_electron"


def _readonly(values):
    values = np.asarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class EffectiveGunSource:
    """Inputs refer to the installed gun EXIT, not its low-energy emitter.

    ``incoherent_angle_rms_mrad`` adds angular variance to the diffraction-
    limited wave. It is not the total angular width or a specimen aperture.
    Current must be explicitly supplied for this new equivalent model.
    """
    reference_current_a: float
    source_fwhm_nm: float = 5.0
    incoherent_angle_rms_mrad: float = 0.0
    energy_fwhm_ev: float = 0.3
    energy_nodes: int = 3
    dispersion_nm_per_ev: float = 0.0
    angular_dispersion_mrad_per_ev: float = 0.0
    mode_tail_tolerance: float = 1e-6
    maximum_modes: int = 4096
    grid_pixels: int = 128
    model_id: str = MODEL_ID
    bound_gun_digest: str = ""

    def __post_init__(self):
        positive = (self.source_fwhm_nm, self.mode_tail_tolerance)
        nonnegative = (self.reference_current_a, self.incoherent_angle_rms_mrad, self.energy_fwhm_ev)
        if (not all(math.isfinite(v) and v > 0 for v in positive)
                or not all(math.isfinite(v) and v >= 0 for v in nonnegative)
                or not all(math.isfinite(v) for v in (self.dispersion_nm_per_ev, self.angular_dispersion_mrad_per_ev))):
            raise ValueError("Effective gun source parameters must be finite and physically non-negative")
        if self.mode_tail_tolerance >= .01:
            raise ValueError("Effective-source omitted mode weight must be below 1 percent")
        for name, lower, upper in (("energy_nodes", 1, 31), ("maximum_modes", 1, 65536), ("grid_pixels", 32, 4096)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"Effective-source {name} must be an integer in [{lower}, {upper}]")
        if self.energy_fwhm_ev > 0 and self.energy_nodes < 3:
            raise ValueError("Nonzero gun energy spread needs at least three energy nodes")
        if self.model_id != MODEL_ID:
            raise ValueError("Unsupported effective gun source model version")


def gun_binding_digest(gun):
    """Full raw gun identity, not a to_dict subset or a caller origin label."""
    from copy import copy
    from temsim.instrument_snapshot import encode_instrument
    isolated = copy(gun)
    isolated.effective_source = None
    isolated.source_representation = "classical_particles"
    # Sampling changes do not redefine the calibrated effective source.
    isolated.emitter = copy(gun.emitter)
    isolated.emitter.ray_count = 0
    vars(isolated.emitter).pop("_tuning_boundary_probes", None)
    return json_digest(encode_instrument(isolated))


def bind_effective_source(gun, parameters):
    """Retired API: binding a label cannot replace upstream physics."""
    from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel, EXIT_SOURCE_REJECTION
    raise UnsupportedSourceModel(EXIT_SOURCE_REJECTION)


def validate_binding(gun, parameters):
    if not isinstance(parameters, EffectiveGunSource) or not parameters.bound_gun_digest:
        raise ValueError("The effective gun source needs an explicit gun calibration binding")
    if parameters.bound_gun_digest != gun_binding_digest(gun):
        raise ValueError("Effective gun calibration is stale: gun geometry or controls changed; no automatic rebind")


def wavelength_m(energy_ev):
    # Relativistic de Broglie wavelength using exact SI h, e, c.
    from scipy.constants import h, m_e, c, e
    kinetic = float(energy_ev) * e
    if not math.isfinite(kinetic) or kinetic <= 0:
        raise ValueError("Gun source energy must be positive")
    return h * c / math.sqrt(kinetic * (kinetic + 2*m_e*c*c))


def _mode_parameters(parameters, energy_ev):
    sigma = parameters.source_fwhm_nm * 1e-9 / math.sqrt(8*math.log(2))
    wavelength = wavelength_m(energy_ev)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Effective gun source size is outside the representable SI range")
    quantum_angle = wavelength / (4*math.pi*sigma)
    if not math.isfinite(quantum_angle) or quantum_angle <= 0:
        raise ValueError("Effective gun diffraction angle is outside the representable range")
    angle = math.hypot(quantum_angle, parameters.incoherent_angle_rms_mrad * 1e-3)
    if not all(math.isfinite(v*v) and v*v > 0 for v in (sigma, angle)):
        raise ValueError("Effective gun covariance is outside the representable SI range")
    occupation = angle / quantum_angle
    if not math.isfinite(occupation):
        raise ValueError("Effective gun mode count is above limit; source phase-space extent is too large")
    if occupation == 1:
        ratio, count, tail = 0., 1, 0.
    else:
        # Avoid cancellation at both ends: 1-sqrt(1-tol) may round to zero,
        # while (occupation-1)/(occupation+1) may round to one. Solve the
        # exact square 2-D tail criterion 2*q**n-q**(2*n) <= tolerance.
        ratio = (occupation-1) / (occupation+1)
        log_ratio = (math.log(ratio) if occupation < 2
                     else math.log1p(-2/(occupation+1)))
        tolerance = parameters.mode_tail_tolerance
        log_tail_root = math.log(tolerance) - math.log1p(math.sqrt(1-tolerance))
        required = log_tail_root / log_ratio
        # Refuse unbounded requests before converting a huge float to an
        # integer or allocating modes. The full multi-energy budget is also
        # checked below by generate_gun_emission.
        if not math.isfinite(required) or required > math.isqrt(parameters.maximum_modes):
            raise ValueError(f"Effective gun mode count is above limit {parameters.maximum_modes}; increase the budget explicitly")
        count = max(1, math.ceil(required))
        retained_axis_tail = math.exp(count*log_ratio)
        tail = retained_axis_tail*(2-retained_axis_tail)
    return sigma, angle, sigma/math.sqrt(occupation), ratio, count, tail


@dataclass(frozen=True)
class GunEmissionState:
    gun_model_digest: str
    source_parameters: EffectiveGunSource
    plane_z_mm: float
    reference_current_a: float
    energies_ev: np.ndarray
    energy_weights: np.ndarray
    covariance_by_energy: np.ndarray
    record: object

    def __post_init__(self):
        for name in ("energies_ev", "energy_weights", "covariance_by_energy"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "record", freeze_json(self.record))

    @property
    def digest(self):
        return json_digest({"parameters": asdict(self.source_parameters), "record": self.record,
                            "plane_z_mm": self.plane_z_mm, "gun_model_digest": self.gun_model_digest})

    def modes(self):
        """Stream immutable eigenmodes, keeping energy and source correlations."""
        p = self.source_parameters
        mean_energy = float(self.record["mean_energy_ev"])
        source_id = self.digest
        for ei, (energy, prior) in enumerate(zip(self.energies_ev, self.energy_weights)):
            sigma, _, ground_sigma, ratio, count, _ = _mode_parameters(p, energy)
            # All retained eigenfunctions fit on the same energy-local grid.
            half_extent = max(9*sigma, (math.sqrt(2*count)+8)*math.sqrt(2)*ground_sigma)
            dx = 2*half_extent / p.grid_pixels
            if dx > ground_sigma / 3:
                raise ValueError("Effective gun eigenmodes are undersampled; increase source grid pixels")
            x = (np.arange(p.grid_pixels) - p.grid_pixels//2)*dx
            hermite_coordinate = x / (math.sqrt(2)*ground_sigma)
            functions = [np.exp(-x*x/(4*ground_sigma**2)) / (2*math.pi*ground_sigma**2)**.25 * math.sqrt(dx)]
            if count > 1:
                functions.append(math.sqrt(2)*hermite_coordinate*functions[0])
            for n in range(1, count-1):
                functions.append(math.sqrt(2/(n+1))*hermite_coordinate*functions[n] - math.sqrt(n/(n+1))*functions[n-1])
            delta = float(energy-mean_energy)
            origin = np.array((p.dispersion_nm_per_ev*delta*1e-9, 0.))
            tilt = np.array((p.angular_dispersion_mrad_per_ev*delta*1e-3, 0.))
            for ny in range(count):
                for nx in range(count):
                    wave = PlaneWave(np.outer(functions[ny], functions[nx]), np.eye(2)*dx, origin, tilt_rad=tilt)
                    # WaveMode verifies the analytic normalisation. No numeric
                    # renormalisation may hide clipping or a coarse grid.
                    yield WaveMode(wave, float(prior)*(1-ratio)**2*ratio**(nx+ny),
                                   REFERENCE_ID, f"{source_id}:E{ei}:HG{nx},{ny}", float(energy)*1e-3)

    def particle_samples(self, count):
        """Positive Gaussian Wigner samples, carrying continuous energy labels."""
        from temsim.optics.electron_gun.base import EmissionBundle
        from temsim.optics.electron_gun.emitter import _halton_dimensions
        if isinstance(count, bool) or int(count) != count or count < 9:
            raise ValueError("Effective gun sampling requires at least nine particles")
        p = self.source_parameters
        normal = [ndtri(u) for u in _halton_dimensions(int(count), (2, 3, 5, 7, 11))]
        mean = float(self.record["mean_energy_ev"])
        delta = normal[4] * p.energy_fwhm_ev / math.sqrt(8*math.log(2))
        energies = mean + delta
        sigma = p.source_fwhm_nm*1e-9 / math.sqrt(8*math.log(2))
        quantum_angles = np.array([wavelength_m(e)/(4*math.pi*sigma) for e in energies])
        angles = np.hypot(quantum_angles, p.incoherent_angle_rms_mrad*1e-3)
        return EmissionBundle(*[_readonly(v) for v in (
            sigma*normal[0] + p.dispersion_nm_per_ev*delta*1e-9,
            sigma*normal[1], angles*normal[2] + p.angular_dispersion_mrad_per_ev*delta*1e-3,
            angles*normal[3], delta, np.full(int(count), 1/int(count)), np.arange(int(count)))])


def generate_gun_emission(gun, parameters=None):
    """Retired producer; never synthesize a new beam at the gun exit."""
    from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel, EXIT_SOURCE_REJECTION
    raise UnsupportedSourceModel(EXIT_SOURCE_REJECTION)


def _reconstruct_historical_emission(gun, parameters=None):
    """Decode old source metadata for read-only evidence and isolated math tests.

    This is not an admission path for transport, active profiles or cache reuse.
    """
    p = parameters if parameters is not None else getattr(gun, "effective_source", None)
    validate_binding(gun, p)
    mean_energy = float(gun.nominal_exit_energy_ev)
    if p.energy_fwhm_ev == 0:
        energies, weights = np.array([mean_energy]), np.array([1.])
    else:
        nodes, weights = np.polynomial.hermite.hermgauss(p.energy_nodes)
        energies = mean_energy + math.sqrt(2)*nodes*p.energy_fwhm_ev/math.sqrt(8*math.log(2))
        weights = weights / math.sqrt(math.pi)
    rows, covariances, count = [], [], 0
    for energy in energies:
        sigma, angle, ground_sigma, ratio, n, tail = _mode_parameters(p, energy)
        count += n*n
        rows.append({"energy_ev": float(energy), "modes_per_axis": n, "omitted_weight": tail,
                     "sigma_x_m": sigma, "sigma_angle_rad": angle, "ground_sigma_m": ground_sigma,
                     "eigenvalue_ratio": ratio})
        covariances.append(np.diag((sigma*sigma, sigma*sigma, angle*angle, angle*angle)))
    if count > p.maximum_modes:
        raise ValueError(f"Effective gun needs {count} modes for its declared tolerance, above limit {p.maximum_modes}; increase the budget explicitly")
    return GunEmissionState(p.bound_gun_digest, p, float(gun.exit_plane_z_mm), p.reference_current_a,
        energies, weights, np.asarray(covariances), {"model_id": MODEL_ID,
        "reference_plane": "Installed electron-gun exit port", "reference_id": REFERENCE_ID,
        "mean_energy_ev": mean_energy, "mode_count": count, "mode_rows": rows,
        "weighted_omitted_mode_probability": float(sum(w*r["omitted_weight"] for w, r in zip(weights, rows))),
        "energy_quadrature": "Gauss-Hermite for a Gaussian exit energy distribution",
        "energy_quadrature_convergence": "NOT_VALIDATED", "physical_calibration": "PHENOMENOLOGICAL_NOT_MEASURED",
        "particle_relation": "Full Gaussian Wigner distribution; modal truncation is separately recorded",
        "gun_internal_scope": "Bound effective exit model, not a resolved quantum gun calculation"})


def trace_effective_source(state, count=None):
    """Retired API: every particle must traverse the physical gun."""
    from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel, EXIT_SOURCE_REJECTION
    raise UnsupportedSourceModel(EXIT_SOURCE_REJECTION)
