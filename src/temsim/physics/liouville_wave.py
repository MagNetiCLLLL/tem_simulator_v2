"""Exact longitudinal-coordinate transformation, not a WKB truncation.

For D_z^2 f + (k(z)^2 I + H) f = 0, define ds/dz = alpha = k/k_ref
and f = alpha^(-1/2) u. Then G_s=G_z/alpha and
D_s^2 u + [k_ref^2 I + (H + a''/a I)/alpha^2] u = 0.
The scalar correction and derivative jumps at potential knots are RETAINED.
References: NIST DLMF 1.13(iv), changes of independent/dependent variables.
"""
from dataclasses import dataclass
import math
import numpy as np


def coordinate_terms(q, q_prime, q_second, reference_k):
    """q=k^2 and its physical-z derivatives; no fitted source quantities."""
    q, qp, qpp = np.broadcast_arrays(np.asarray(q, float), q_prime, q_second)
    if (np.any(q <= 0) or not np.all(np.isfinite((q, qp, qpp)))
            or not math.isfinite(reference_k) or reference_k <= 0):
        raise ValueError("Liouville coordinate needs a positive finite carrier and finite derivatives")
    alpha = np.sqrt(q)/reference_k
    log_derivative = -qp/(4*q)
    correction = 5*(qp/q)**2/16-qpp/(4*q)
    return alpha, log_derivative, correction


def derivative_jump(strength, reference_k, count):
    """Exact passive interface: u continuous, D_s u increases by strength*u."""
    if not np.isfinite(strength) or not np.isfinite(reference_k) or reference_k <= 0:
        raise ValueError("A derivative jump requires finite strength and positive chart wave number")
    transmission = 2j*reference_k/(2j*reference_k-strength)
    reflection = strength/(2j*reference_k-strength)
    eye = np.eye(count, dtype=complex)
    return reflection*eye, transmission*eye, transmission*eye, reflection*eye


@dataclass(frozen=True)
class PhysicalCoordinateLoad:
    """Expose the executed transformed load in the original physical variables."""
    transformed: object
    alpha: np.ndarray
    log_amplitude_derivative: np.ndarray

    @property
    def kappa(self):
        return self.transformed.kappa

    @property
    def input_admittance(self):
        admittance = self.transformed.input_admittance
        return self.alpha[0]*admittance+self.log_amplitude_derivative[0]*np.eye(len(admittance))

    @property
    def output_wave_number(self):
        if self.log_amplitude_derivative[-1] != 0:
            raise ValueError("A homogeneous exit load needs zero longitudinal amplitude derivative")
        return self.alpha[-1]*self.transformed.output_wave_number

    def propagate(self, total_left):
        amplitude = self.alpha**-.5
        values, derivatives = self.transformed.propagate(np.asarray(total_left)/amplitude[0])
        return (amplitude[:, None]*values,
                amplitude[:, None]*(self.alpha[:, None]*derivatives
                    + self.log_amplitude_derivative[:, None]*values))
