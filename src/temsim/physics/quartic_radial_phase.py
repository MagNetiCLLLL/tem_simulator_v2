"""Exact finite-basis operators for a numerical quadratic + quartic phase.

Phi_n = exp(i*k*c*r^2/2 + i*g*r^4) L_n(r^2/b^2) exp(-r^2/2b^2)/(sqrt(pi)*b).
Lengths are nm. g is a NUMERICAL chart coefficient in nm^-4, not an added
physical aberration. Every phase term is retained in the operator and export.
"""
from functools import lru_cache
import math

import numpy as np


def radial_envelope(radius_nm, width_nm, coefficients):
    """Evaluate all complex coefficients in O(N*radii) without an N*radii array.

    The exponentially scaled Laguerre recurrence avoids 0*inf in empty tails.
    It does not truncate small coefficients or renormalise the result.
    """
    radius = np.asarray(radius_nm, float)
    values = np.asarray(coefficients, complex)
    if (not math.isfinite(width_nm) or width_nm <= 0 or not np.all(np.isfinite(radius))
            or np.any(radius < 0) or values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))):
        raise ValueError("Radial envelope needs finite radii, width and complex coefficients")
    x = (radius/width_nm)**2
    previous, current = np.zeros_like(x), np.exp(-x/2)
    result = np.zeros_like(x, dtype=complex)
    for n, value in enumerate(values):
        result += value*current
        previous, current = current, ((2*n+1-x)*current-n*previous)/(n+1)
    return result/(np.sqrt(np.pi)*width_nm)


@lru_cache(maxsize=16)
def _padded_operators(count):
    from temsim.physics.radial_gun_wave import laguerre_operators
    # X^3 needs three untruncated intermediate radial modes. G has bandwidth
    # two; its outside-basis Gram closure needs two extra modes as well.
    x, d, t = laguerre_operators(count+3)
    x2 = x@x
    result = (x, x2, x2@x, d, t, x@d+d@x)
    for value in result:
        value.setflags(write=False)
    return result


def quartic_operators(count, width, curvature, reference_k, width_log_rate=0.,
                      curvature_prime=0., quartic=0., quartic_prime=0.):
    """Return transverse kinetic K, Hermitian connection G and closure C.

    Let q=k*c, x=r^2/b^2 and D=1+r*d/dr. Then
    K=T/b^2+q^2*b^2*X+8*q*g*b^4*X^2+16*g^2*b^6*X^3
      -2*i*q*D-4*i*g*b^2*(XD+DX),
    G=k*c'*b^2*X/2+g'*b^4*X^2+i*(b'/b)*D.
    C=G_PQ G_QP retains the derivative outside the finite moving basis.
    Matrix products are formed BEFORE projection; squaring truncated X would
    omit real boundary terms even on the last retained diagonal.
    """
    if type(count) is not int or count < 1:
        raise ValueError("Quartic radial basis count must be a positive integer")
    values = (width, curvature, reference_k, width_log_rate, curvature_prime, quartic, quartic_prime)
    if not all(math.isfinite(v) for v in values) or min(width, reference_k) <= 0:
        raise ValueError("Quartic coordinates require finite values and positive width/wave number")
    x, x2, x3, d, t, xd = _padded_operators(count)
    q = reference_k*curvature
    kinetic = (t/width**2+(q*width)**2*x+8*q*quartic*width**4*x2
        +16*quartic**2*width**6*x3-2j*q*d-4j*quartic*width**2*xd)
    connection = (.5*reference_k*curvature_prime*width**2*x
        +quartic_prime*width**4*x2+1j*width_log_rate*d)
    outside = connection[count:, :count]
    return kinetic[:count, :count], connection[:count, :count], outside.conj().T@outside


def quartic_wave_moments(coefficients, width, curvature, reference_k, quartic=0.):
    """Choose a phase chart from the full wave's exact gradient moments.

    Minimise ||grad(envelope) - i*(d2*rho+d4*rho^3)*envelope||^2,
    rho=r/b. This least-squares coordinate choice never changes the wave.
    The phase coefficients add to the already executed analytic phase; do
    not refit/remove that phase in physical field comparisons.
    """
    a = np.asarray(coefficients, complex)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Quartic moments need a finite complex radial state")
    if not all(math.isfinite(v) for v in (width, curvature, reference_k, quartic)) or min(width, reference_k) <= 0:
        raise ValueError("Quartic moment coordinates must be finite and positive where required")
    scale = float(np.max(abs(a)))
    if scale == 0:
        raise ValueError("A zero wave cannot define a phase chart")
    a = np.r_[a/scale, np.zeros(3)]
    a /= np.linalg.norm(a)
    x, x2, x3, d, t, _ = _padded_operators(len(a)-3)
    moments = np.array([np.vdot(a, matrix@a).real for matrix in (x, x2, x3)])
    rhs = np.array([np.vdot(a, d@a).imag, np.vdot(x@a, d@a).imag])
    gram = np.array(((moments[0], moments[1]), (moments[1], moments[2])))
    if np.any(np.diag(gram) <= 0):
        raise ValueError("Radial gradient Gram matrix lost positive moments")
    # Scale the 2x2 system by its diagonal, not the physical source/current.
    diagonal = 1/np.sqrt(np.diag(gram))
    normal = diagonal[:, None]*gram*diagonal[None, :]
    if np.linalg.cond(normal) > 1e12:
        raise ValueError("Radial phase-coordinate fit is ill-conditioned")
    fit = diagonal*np.linalg.solve(normal, diagonal*rhs)
    intrinsic = float(np.vdot(a, t@a).real-rhs@fit)/width**2
    r2 = float(width**2*moments[0])
    if not math.isfinite(intrinsic) or intrinsic <= 0 or r2*intrinsic < 1-1e-9:
        raise ValueError("Quartic moments lost positive intrinsic momentum or uncertainty bound")
    return {"optimal_width_nm": (r2/intrinsic)**.25,
        "optimal_curvature_per_nm": curvature+float(fit[0])/(reference_k*width**2),
        "optimal_quartic_phase_per_nm4": quartic+float(fit[1])/(4*width**4),
        "radius_squared_nm2": r2, "intrinsic_momentum_squared_nm2": intrinsic,
        "minimum_mean_radial_order": (math.sqrt(r2*intrinsic)-1)/2}


def apply_quartic_on_grid(amplitude, radius_nm, step_nm, quartic):
    """Export the actual complex phase; reject an unresolved Cartesian grid.

    The 1e-12 tail is excluded only from the sampling diagnostic, never from
    transport or output. No norm/current repair is performed.
    """
    if not math.isfinite(quartic) or not math.isfinite(step_nm) or step_nm <= 0:
        raise ValueError("Quartic export needs finite phase and positive grid spacing")
    if quartic == 0:
        return amplitude
    radius = np.asarray(radius_nm, float)
    mass = abs(amplitude.ravel())**2
    if radius.shape != amplitude.shape or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(mass)):
        raise ValueError("Quartic export requires matching finite field and radius grids")
    order = np.argsort(radius.ravel())
    tail = np.cumsum(mass[order][::-1])[::-1]
    occupied = tail > mass.sum()*1e-12
    edge = float(radius.ravel()[order][occupied].max(initial=0.))
    delta = math.sqrt(2)*step_nm
    increment = abs(quartic)*delta*(4*edge**3+6*edge**2*delta+4*edge*delta**2+delta**3)
    if increment > np.pi/2:
        raise ValueError(f"Gun-exit quartic phase needs a finer Cartesian grid ({increment:.6g} rad/sample); increase grid_pixels")
    return amplitude*np.exp(1j*quartic*radius**4)
