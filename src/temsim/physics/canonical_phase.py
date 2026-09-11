"""SI mechanical/canonical boundaries and sampled phase-carrier admission.

Canonical ordering is (x, y, p_x/p_ref, p_y/p_ref). Positions are metres;
momentum ratios and paraxial mechanical slopes are dimensionless. The gauge
is A=(-Bz*y/2, Bz*x/2, 0) at an axial, locally uniform boundary plane.
"""
from __future__ import annotations

import numpy as np

CANONICAL_BASIS = "laboratory-normalized-canonical"
SYMPLECTIC_TOLERANCE = 1e-7


def canonical_basis_matrix(*, charge_c: float, bz_t: float,
                           momentum_kg_m_s: float, reference_momentum_kg_m_s: float) -> np.ndarray:
    values = (charge_c, bz_t, momentum_kg_m_s, reference_momentum_kg_m_s)
    if not np.all(np.isfinite(values)) or min(values[2:]) <= 0:
        raise ValueError("Canonical conversion requires finite fields and positive physical momenta")
    matrix = np.eye(4)
    matrix[2:, 2:] *= momentum_kg_m_s / reference_momentum_kg_m_s
    g = charge_c * bz_t / (2 * reference_momentum_kg_m_s)
    matrix[2:, :2] = ((0., -g), (g, 0.))
    return matrix


def validate_canonical_map(matrix, *, basis=CANONICAL_BASIS,
                           tolerance=SYMPLECTIC_TOLERANCE) -> float:
    """Return dimensionless symplectic defect; never repair an invalid map.

    A common position/angle scaling balances B and C before checking M^T J M.
    It is a similarity in a single normalized basis, not a change of momentum
    reference between input and output. Mechanical maps must be converted first.
    """
    if basis != CANONICAL_BASIS:
        raise ValueError("Wave transport needs an explicitly normalized canonical basis")
    m = np.asarray(matrix, float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError("Canonical map must be a finite 4x4 matrix")
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-5:
        raise ValueError("Invalid symplectic admission tolerance")
    b, c = np.linalg.norm(m[:2, 2:], ord=2), np.linalg.norm(m[2:, :2], ord=2)
    length = np.sqrt(b / c) if b > 0 and c > 0 else (b if b > 0 else (1/c if c > 0 else 1.))
    scale = np.diag((1/length, 1/length, 1., 1.))
    scaled = scale @ m @ np.diag((length, length, 1., 1.))
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    defect = float(np.linalg.norm(scaled.T @ j @ scaled - j, ord=np.inf))
    if not np.isfinite(defect) or defect > tolerance:
        raise ValueError(f"Non-symplectic canonical map: residual {defect:.6g} exceeds {tolerance:.6g}")
    return defect


def mechanical_map_to_canonical(matrix, *, input_basis, output_basis):
    """Convert mechanical slopes at BOTH endpoints, then check admission."""
    result = np.asarray(output_basis) @ np.asarray(matrix) @ np.linalg.inv(input_basis)
    validate_canonical_map(result)
    return result


def expanded_phase_amplitude(wave, wavelength_m: float) -> np.ndarray:
    """Materialize the full complex CELL amplitude, including tilt/curvature.

    FFT sign is exp(-2 pi i f.x); a positive phase gradient is positive momentum.
    Occupied adjacent cells must sample the analytic carrier below Nyquist.
    No normalization, pupil replacement or resampling occurs here.
    """
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Phase expansion requires a finite positive wavelength")
    xy = wave.coordinates_m() - wave.origin_m[:, None, None]
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    t = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase = (2*np.pi/wavelength_m) * (
        .5*np.einsum("iyx,ij,jyx->yx", xy, q, xy) + np.einsum("i,iyx->yx", t, xy))
    occupied = np.abs(wave.amplitude) > np.max(np.abs(wave.amplitude))*1e-8
    for axis in (0, 1):
        low = np.take(occupied, np.arange(occupied.shape[axis]-1), axis=axis)
        high = np.take(occupied, np.arange(1, occupied.shape[axis]), axis=axis)
        a0 = np.take(wave.amplitude, np.arange(occupied.shape[axis]-1), axis=axis)
        a1 = np.take(wave.amplitude, np.arange(1, occupied.shape[axis]), axis=axis)
        # The envelope and analytic carrier share ONE Nyquist budget. The
        # envelope is assumed already sampled; adding a separately safe carrier
        # can still make the full physical field undersampled.
        total_difference = np.angle(a1*np.conj(a0)) + np.diff(phase, axis=axis)
        if np.any(np.abs(total_difference)[low & high] >= np.pi):
            raise ValueError("Full-wave phase carrier is undersampled; refine the physical grid")
    return wave.amplitude * np.exp(1j*phase)
