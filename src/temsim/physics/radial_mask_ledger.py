"""Separate disk absorption from occupied-wave Galerkin projection loss.

For P=<Phi|disk|Phi>, <a|(I-P)|a> is the incident field outside the
opening; <a|(P-P^2)|a> is the transmitted field outside the retained basis.
Both incoming ports are retained. No field or transmission is rescaled.
"""
import numpy as np


def mask_loss_budget(projection, left_incoming, right_incoming, reference_k):
    p = np.asarray(projection)
    if not np.isfinite(reference_k) or reference_k <= 0:
        raise ValueError("Mask ledger needs a positive finite chart wave number")
    absorbed, unresolved, incoming = 0., 0., 0.
    for amplitude in (left_incoming, right_incoming):
        amplitude = np.asarray(amplitude)
        projected = p@amplitude
        norm = float(np.vdot(amplitude, amplitude).real)
        inside = float(np.vdot(amplitude, projected).real)
        retained = float(np.vdot(projected, projected).real)
        incoming += reference_k*norm
        absorbed += reference_k*(norm-inside)
        unresolved += reference_k*(inside-retained)
    # Small negative roundoff is reported, not hidden by clipping.
    return {"incoming_port_flux": incoming, "mask_absorbed": absorbed,
        "unresolved_transmitted": unresolved,
        "total_removed": absorbed+unresolved}


def physical_to_chart(field, derivative, alpha, log_derivative, reference_k):
    """Invert the exact Liouville field transform without new propagation."""
    value = np.sqrt(alpha)*np.asarray(field)
    covariant_derivative = (np.sqrt(alpha)*np.asarray(derivative)-log_derivative*value)/alpha
    return ((value+covariant_derivative/(1j*reference_k))/2,
            (value-covariant_derivative/(1j*reference_k))/2)
