"""Two-way, nonparaxial stationary propagation in a stratified scalar field.

This is a boundary-value operator, NOT an admitted gun or a downstream source.
For each transverse Fourier component, psi'' + k_z**2 psi = 0, with
k_z**2 = 2*m_e*(E_tip + phi(z)-phi_tip)*e/hbar**2 - |k_perp|**2.
The right boundary is outgoing (or decaying). The left boundary prescribes
the TOTAL complex field; its normal derivative is solved, not assumed ik psi.
Reflection and evanescent coupling are included. An evanescent amplitude is
not a lost electron, and |psi|**2 is not a forward-flux probability.

Only nonrelativistic, transversely uniform, real electrostatic layers are
represented. Radial focusing, magnetic fields, emitter geometry/tunnelling,
apertures and a flux-normalised tip boundary need their own joint operator.
The full FEG must never be replaced by this axial reduction.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import e, hbar, m_e

from temsim.physics.wave_execution import check_available_memory


@dataclass(frozen=True)
class BoundaryPropagation:
    """Unit left-field response; amplitudes and derivatives retain their gauge."""

    boundary_amplitude: np.ndarray
    boundary_derivative_per_m: np.ndarray
    log_transfer: np.ndarray
    current_per_boundary_density_m_per_s: np.ndarray
    evanescent_layers: np.ndarray


def propagate_low_energy_boundary(kinetic_energy_ev, transverse_wave_number_per_m,
        layer_widths_m, exit_energy_ev, *, maximum_working_bytes=512*1024**2,
        cancelled=lambda: False):
    """Solve uniform layers by stable right-to-left admittance recursion.

    kinetic_energy_ev has shape (layers,); positive values are the on-axis
    kinetic energies, INCLUDING executed electrostatic energy gain. The
    transverse wavenumbers (rad/m) can have any array shape. Layer edges are
    contiguous. The exit energy defines the semi-infinite outgoing medium.
    A 1 keV upper bound confines the nonrelativistic kernel's use; it does not
    change any caller's energy. Converge piecewise-constant field sampling.

    Scaled trigonometric functions avoid exp(+kappa*d) overflow. At k_z=0 the
    exact sin(k*d)/k -> d limit is used. No amplitude/flux renormalisation,
    spectral clipping, or paraxial square-root expansion is performed.
    """
    energy = np.asarray(kinetic_energy_ev, dtype=float)
    widths = np.asarray(layer_widths_m, dtype=float)
    transverse = np.asarray(transverse_wave_number_per_m, dtype=float)
    if energy.ndim != 1 or not energy.size or widths.shape != energy.shape:
        raise ValueError("Provide one energy and width per nonempty axial layer")
    if (not np.all(np.isfinite(energy)) or np.any(energy <= 0) or np.any(energy > 1000)
            or not math.isfinite(exit_energy_ev) or not 0 < exit_energy_ev <= 1000):
        raise ValueError("This nonrelativistic boundary kernel requires energies in (0, 1000] eV")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Layer widths must be finite and positive")
    if not transverse.size or not np.all(np.isfinite(transverse)) or np.any(transverse < 0):
        raise ValueError("Transverse wavenumbers must be finite and nonnegative")
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Boundary working memory budget must be a positive integer")
    count = transverse.size
    required = int((energy.size+1)*count*160+count*512)
    if required > maximum_working_bytes:
        raise MemoryError(f"Low-energy boundary calculation needs approximately {required} bytes")
    check_available_memory(required)
    if cancelled():
        raise InterruptedError("Low-energy boundary calculation cancelled")
    shape = (energy.size+1,)+transverse.shape
    admittance = np.empty(shape, dtype=complex)
    log_steps = np.empty((energy.size,)+transverse.shape, dtype=complex)
    evanescent = np.empty(log_steps.shape, dtype=bool)
    def longitudinal_squared(value):
        # Factoring the difference of squares avoids cancellation at grazing
        # incidence and preserves the exactly represented k_perp == k case.
        total = np.sqrt(2*m_e*value*e)/hbar
        return (total-transverse)*(total+transverse)
    with np.errstate(over="raise", invalid="raise"):
        admittance[-1] = 1j*np.sqrt(longitudinal_squared(exit_energy_ev).astype(complex))
        for i in range(energy.size-1, -1, -1):
            if cancelled():
                raise InterruptedError("Low-energy boundary calculation cancelled")
            k2 = longitudinal_squared(energy[i])
            kz = np.sqrt(k2.astype(complex))
            phase = kz*widths[i]
            attenuation = abs(phase.imag)
            positive = np.exp(1j*phase-attenuation)
            negative = np.exp(-1j*phase-attenuation)
            cosine = (positive+negative)/2
            sine_over_k = np.zeros_like(kz)
            small = abs(phase) < 1e-4
            np.divide(positive-negative, 2j*kz, out=sine_over_k, where=~small)
            # Polynomial of the entire function sinc, including the exact
            # grazing/turning case. The series error here is O(1e-24).
            sinc = 1-phase[small]**2/6+phase[small]**4/120
            sine_over_k[small] = widths[i]*np.exp(-attenuation[small])*sinc
            denominator = cosine-admittance[i+1]*sine_over_k
            if np.any(abs(denominator) < np.finfo(float).tiny):
                raise ValueError("Singular boundary admittance; a nodal boundary needs a two-component chart")
            admittance[i] = (admittance[i+1]*cosine+k2*sine_over_k)/denominator
            log_steps[i] = -attenuation-np.log(denominator)
            evanescent[i] = k2 < 0
    log_transfer = np.concatenate((np.zeros((1,)+transverse.shape, complex),
                                   np.cumsum(log_steps, axis=0)), axis=0)
    # Underflow represents an exponentially tiny amplitude, not renormalised
    # transmission. Keep its logarithm for reproducibility and diagnostics.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        amplitude = np.exp(log_transfer)
        derivative = amplitude*admittance
        current = hbar/m_e*np.imag(amplitude.conj()*derivative)
    if not all(np.all(np.isfinite(a)) for a in (amplitude, derivative, log_transfer, current)):
        raise ValueError("Non-finite low-energy boundary solution")
    frozen = tuple(np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for value in (amplitude, derivative, log_transfer, current, evanescent))
    return BoundaryPropagation(*frozen)
