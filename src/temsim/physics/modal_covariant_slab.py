"""Complete mixed propagating/evanescent constant-slab scattering coordinates.

This optional kernel solves (d/dz+iG)^2 f + Q f = 0. Decaying solutions are
referenced at opposite faces, so a growing long transfer matrix is never
formed. All 2N solutions remain present. A rejected modal chart must fall
back to the full scattering-doubling method, never delete difficult modes.

The invariant-subspace and current checks are local numerical diagnostics,
not a source, axial/basis convergence certificate, or an image admission.
Scattering formulation reference: Ko and Inkson, PRB 38, 9945 (1988),
https://doi.org/10.1103/PhysRevB.38.9945 .
"""
import math

import numpy as np
from scipy.linalg import eig, svdvals

from temsim.physics.coupled_low_energy import _hermitian
from temsim.physics.covariant_boundary import _solve, _unitarity


def modal_covariant_slab(q, connection, width, kappa, cancelled=lambda: False):
    """Return (rL,tRL,tLR,rR), retaining absolute complex phase in every block.

    Q has units 1/length^2, G and kappa 1/length, width length. Signed width
    supports numerical Magnus stages; it does not move physical apertures.
    The common current coordinates are f=u+v, Df=i*kappa*(u-v).
    """
    q, connection = np.asarray(q, complex), np.asarray(connection, complex)
    if q.ndim != 2 or not len(q) or q.shape[0] != q.shape[1] or connection.shape != q.shape:
        raise ValueError("Modal slab requires matching nonempty square Q and G")
    _hermitian(q, "Modal slab Q")
    _hermitian(connection, "Modal slab G")
    if isinstance(width, bool) or not math.isfinite(width):
        raise ValueError("Modal slab width must be finite")
    if isinstance(kappa, bool) or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("Modal slab kappa must be positive and finite")
    if cancelled():
        raise InterruptedError("Modal covariant slab cancelled")
    n = len(q)
    eye = np.eye(n)
    if width == 0:
        return (eye*0, eye, eye, eye*0), {"modal_covariant_slab": True, "zero_width": True}
    h = (q-kappa*kappa*eye)/(2*kappa)
    generator = 1j*np.block([[kappa*eye+h-connection, h],
                             [-h, -kappa*eye-h-connection]])
    if not np.all(np.isfinite(generator)):
        raise ValueError("Modal generator exceeds finite numerical range")
    values, vectors = eig(generator, check_finite=False)
    singular = svdvals(vectors, check_finite=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] else math.inf
    if not math.isfinite(condition) or condition > 1e6:
        raise ValueError("Unresolved modal chart; retain all channels in scattering doubling")
    defect = float(np.linalg.norm(generator@vectors-vectors*values, ord=2)*abs(width))
    if not math.isfinite(defect) or defect > 1e-9:
        raise ValueError("Modal invariant phase defect exceeds 1e-9; use scattering doubling")
    # Values indistinguishable from the imaginary axis are classified by
    # physical current, not by the sign of a noisy real eigenvalue.
    spectral_noise = 128*np.finfo(float).eps*max(1., float(np.linalg.norm(generator, ord=2)))
    decaying = abs(values.real) > spectral_noise
    flux = np.sum(abs(vectors[:n])**2, axis=0)-np.sum(abs(vectors[n:])**2, axis=0)
    if np.any((~decaying) & (abs(flux) < 1e-10)):
        raise ValueError("Grazing modal current is unresolved; use scattering doubling")
    positive = np.where(decaying, values.real*width < 0, flux > 0)
    if np.count_nonzero(positive) != n:
        raise ValueError("Incomplete directional modal chart; use scattering doubling")
    vp, vm = vectors[:, positive], vectors[:, ~positive]
    # No eigenvalue or field is clipped or renormalised. Tiny real parts of
    # propagating eigenvalues stay in the exponential and the current guard.
    with np.errstate(over="raise", invalid="raise"):
        right = np.exp(values[positive]*width)
        left = np.exp(-values[~positive]*width)
    incoming = np.block([[vp[:n], vm[:n]*left], [vp[n:]*right, vm[n:]]])
    outgoing = np.block([[vp[n:], vm[n:]*left], [vp[:n]*right, vm[:n]]])
    if cancelled():
        raise InterruptedError("Modal covariant slab cancelled")
    full = _solve(incoming.T, outgoing.T).T
    blocks = full[:n, :n], full[:n, n:], full[n:, :n], full[n:, n:]
    current = _unitarity(blocks)
    if not np.any(blocks[1]) or not np.any(blocks[2]):
        raise ValueError("Modal transmission underflow requires log-scaled channel transport")
    return blocks, {"modal_covariant_slab": True,
        "evanescent_directional_solutions": int(np.count_nonzero(decaying)),
        "retained_directional_solutions": 2*n,
        "modal_chart_condition": condition, "modal_invariant_phase_defect": defect,
        "slab_unitarity_residual": current,
        "convergence": "REQUIRES_INDEPENDENT_AXIAL_BASIS_AND_DOMAIN_REFINEMENT"}
