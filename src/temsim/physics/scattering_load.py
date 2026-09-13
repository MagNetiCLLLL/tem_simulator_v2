"""Reusable two-way loads in a fixed current chart.

This internal linear algebra has no source admission. It composes exact
uniform Hermitian slabs, unitary connection rotations and passive absorbing
projections. A load is coupled back into a driven boundary solve, not used
to relabel an already calculated forward field.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh, cholesky, solve, solve_triangular

from temsim.physics.coupled_low_energy import _slab, _hermitian


def compose(left, right):
    """Redheffer product in physical left-to-right order: (rL,tRL,tLR,rR)."""
    a, b, c, d = left
    e, f, g, h = right
    den = np.eye(len(a))-d@e
    # Both complete right-hand sides use the SAME reflected interface.
    # Factor it once; no inverse, channel truncation or new approximation.
    response = solve(den, np.concatenate((c, d@f), axis=1), check_finite=False)
    fc, df = response[:, :len(a)], response[:, len(a):]
    return a+b@e@fc, b@(f+e@df), g@fc, h+g@df


def _shifted_slab(residual_q, carrier_k, width, kappa):
    """Exact slab without erasing a small transverse operator in k0**2 I.

    Its eigenvectors are those of the residual. Evaluate sqrt(k0**2+h)-k0
    as h/(sqrt(k0**2+h)+k0), keeping axial and residual phases separate.
    """
    if not math.isfinite(carrier_k) or carrier_k <= 0:
        raise ValueError("The scalar axial carrier must be positive and finite")
    values, vectors = eigh(residual_q, check_finite=False)
    k = np.sqrt((carrier_k**2+values).astype(complex))
    delta = values/(k+carrier_k)
    phase = k*width
    attenuation = abs(phase.imag)
    positive = np.exp(1j*carrier_k*width)*np.exp(1j*delta*width-attenuation)
    negative = np.exp(-1j*carrier_k*width)*np.exp(-1j*delta*width-attenuation)
    cosine = (positive+negative)/2
    sine_over_k = np.empty_like(k)
    small = abs(phase) < 1e-4
    sine_over_k[small] = width*np.exp(-attenuation[small])*(1-phase[small]**2/6+phase[small]**4/120)
    np.divide(positive-negative, 2j*k, out=sine_over_k, where=~small)
    total = carrier_k**2+values
    denominator = cosine-.5j*(kappa+total/kappa)*sine_over_k
    reflected = .5j*((carrier_k-kappa)*(carrier_k+kappa)+values)/kappa*sine_over_k/denominator
    log_t = -attenuation-np.log(denominator)
    if np.any(log_t.real < math.log(np.finfo(float).tiny)):
        raise ValueError("Slab transmission needs a log-scaled evanescent representation")
    transmitted = np.exp(log_t)
    error = max(float(abs(abs(reflected)**2+abs(transmitted)**2-1).max()),
                float(abs(2*np.real(reflected.conj()*transmitted)).max()))
    if not math.isfinite(error) or error > 1e-10:
        raise ValueError("Carrier-resolved slab did not conserve current")
    return ((vectors*reflected)@vectors.conj().T, (vectors*transmitted)@vectors.conj().T,
        {"evanescent_channels": int(np.count_nonzero(total < 0)),
         "grazing_channels": int(np.count_nonzero(total == 0)),
         "minimum_log_transmission_magnitude": float(log_t.real.min()),
         "slab_unitarity_residual": error, "separate_axial_carrier": True})


def covariant_carrier_slab(residual, connection, width, kappa, carrier, cancelled=lambda: False,
                           *, current_coordinates="spectral", mixed_coordinates="auto"):
    """Execute the simultaneous constant Q/G operator, not a split rotation.

    Solve the shifted forward/backward spectral pencils separately so that
    the small transverse phase is not added to a large carrier eigenvalue.
    The common current chart is restored before composing physical layers.
    Mixed propagating/evanescent channels first try a complete face-referenced
    modal chart. Unresolved/degenerate charts use the existing full two-way
    doubling solver. No channel is dropped to make a chart work. Explicit
    mixed_coordinates="doubling" retains the independent reference method.
    """
    n = len(residual)
    if current_coordinates not in ("spectral", "graph", "riccati"):
        raise ValueError("Unknown covariant carrier current coordinates")
    if mixed_coordinates not in ("auto", "doubling"):
        raise ValueError("Unknown mixed-channel slab coordinates")
    eye = np.eye(n)
    h = residual/(2*carrier)
    # J*H is positive for the propagating chart used here. Its Cholesky
    # similarity gives an ordinary Hermitian eigensystem and J-orthogonal
    # channels. A generic eigensolver loses that current orthogonality for
    # nearly degenerate high-energy modes.
    metric_h = np.block([[carrier*eye+h-connection, h], [h, carrier*eye+h+connection]])
    try:
        factor = cholesky(metric_h, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        from temsim.physics.covariant_boundary import _slab
        q = carrier**2*eye+residual
        fallback = {}
        if mixed_coordinates == "auto":
            from temsim.physics.modal_covariant_slab import modal_covariant_slab
            try:
                return modal_covariant_slab(q, connection, width, kappa, cancelled)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as cause:
                fallback = {"modal_fallback_reason": str(cause)}
        blocks, record = _slab(q, connection, width, kappa, cancelled)
        return blocks, {**record, **fallback}
    graph_record = {}
    graph_bases = None
    if current_coordinates == "riccati":
        from temsim.physics.carrier_subspaces import refine_current_graph
        graph_bases, graph_record = refine_current_graph(h, connection, carrier, width)
        eigen = np.r_[np.ones(n), -np.ones(n)]  # Signature labels, not physical eigenvalues.
    else:
        metric = np.diag(np.r_[np.ones(n), -np.ones(n)])
        eigen, rotation = eigh(factor.conj().T@metric@factor, check_finite=False)
        vectors = solve_triangular(factor.conj().T, rotation, lower=False,
                                   check_finite=False)*np.sqrt(abs(eigen))
        if current_coordinates == "graph":
            from temsim.physics.carrier_subspaces import graph_current_subspaces
            graph_bases, graph_record = graph_current_subspaces(vectors, eigen, h, connection, carrier, width)
        elif current_coordinates != "spectral":
            raise ValueError("Unknown covariant carrier current coordinates")
    groups = []
    for sign in (1, -1):
        selected = np.flatnonzero(sign*eigen > 0)
        if len(selected) != n:
            from temsim.physics.covariant_boundary import _slab
            return _slab(carrier**2*eye+residual, connection, width, kappa, cancelled)
        basis = vectors[:, selected] if graph_bases is None else graph_bases[0 if sign == 1 else 1]
        # Recover tiny residual eigenphases in each invariant sign subspace
        # without subtracting k0 from an already rounded large eigenvalue.
        shifted_metric = np.block([[(1-sign)*carrier*eye+h-connection, h],
                                    [h, (1+sign)*carrier*eye+h+connection]])
        reduced = sign*basis.conj().T@shifted_metric@basis
        values, mixing = eigh(reduced, check_finite=False)
        basis = basis@mixing
        # local chart f=u+v, Df=i*k0*(u-v) -> common kappa chart.
        ratio = carrier/kappa
        u = .5*((1+ratio)*basis[:n]+(1-ratio)*basis[n:])
        v = .5*((1-ratio)*basis[:n]+(1+ratio)*basis[n:])
        propagation = np.exp(1j*carrier*width)*np.exp(sign*1j*values*width)
        groups.append((u, v, propagation))
    up, vp, forward = groups[0]
    um, vm, backward = groups[1]
    incoming = np.block([[up, um*backward], [vp*forward, vm]])
    outgoing = np.block([[vp, vm*backward], [up*forward, um]])
    full = solve(incoming.T, outgoing.T, check_finite=False).T
    error = float(np.linalg.norm(full.conj().T@full-np.eye(2*n), ord=np.inf))
    if not math.isfinite(error) or error > 1e-9:
        if current_coordinates == "spectral":
            try:
                blocks, record = covariant_carrier_slab(residual, connection, width, kappa, carrier,
                    cancelled, current_coordinates="graph")
                return blocks, {**record, "original_spectral_current_residual": error}
            except (ValueError, np.linalg.LinAlgError) as cause:
                failure = ValueError(f"Covariant carrier slab current residual {error:.6g}; "
                                     f"current-graph coordinates also failed: {cause}")
        else:
            failure = ValueError(f"Covariant carrier slab current residual {error:.6g}; refine its representation")
        failure.slab_diagnostic = {"residual_real": residual.real.tolist(), "residual_imag": residual.imag.tolist(),
            "connection_real": connection.real.tolist(), "connection_imag": connection.imag.tolist(),
            "width": width, "kappa": kappa, "carrier": carrier, "current_residual": error,
            "current_coordinates": current_coordinates}
        raise failure
    return (full[:n, :n], full[:n, n:], full[n:, :n], full[n:, n:]), {
        "covariant_carrier_spectral_slab": True, "slab_unitarity_residual": error,
        "connection_split": False, "current_coordinates": current_coordinates, **graph_record}


def hermitian_slab(q, connection, width, kappa, *, carrier_k=None):
    """Current-conserving covariant slab in an explicitly selected chart.

    The exact Q slab retains backward/evanescent waves. The connection is a
    unitary change on BOTH directional components, not an extra physical
    lens. Carrier-resolved Q/G uses their simultaneous constant operator;
    the non-carrier path uses a second-order symmetric split. Variation of
    either physical operator requires independent axial refinement.
    """
    _hermitian(q, "Channel wave-number square")
    _hermitian(connection, "Channel connection")
    if carrier_k is not None and np.any(connection):
        return covariant_carrier_slab(q, connection, width, kappa, carrier_k)
    r, t, record = (_slab(q, width, kappa) if carrier_k is None
                    else _shifted_slab(q, carrier_k, width, kappa))
    if np.any(connection):
        values, vectors = eigh(connection, check_finite=False)
        u = (vectors*np.exp(-.5j*width*values))@vectors.conj().T
        return (u.conj().T@r@u, u.conj().T@t@u.conj().T,
                u@t@u, u@r@u.conj().T), record
    return (r, t, t, r), record


@dataclass(frozen=True)
class ScatteringLoad:
    kappa: float
    input_admittance: np.ndarray
    output_wave_number: np.ndarray
    reflections: tuple
    forwards: tuple
    rows: tuple

    def propagate(self, total_left):
        """Execute the retained complex response to the jointly solved trace."""
        f = np.asarray(total_left, complex)
        u = solve(np.eye(len(f))+self.reflections[0], f, check_finite=False)
        values, derivatives = [], []
        for index, reflection in enumerate(self.reflections):
            v = reflection@u
            values.append(u+v)
            derivatives.append(1j*self.kappa*(u-v))
            if index < len(self.forwards):
                u = self.forwards[index]@u
        return np.asarray(values), np.asarray(derivatives)


def outgoing_load(operators, exit_q, kappa, *, cancelled=lambda: False, progress_callback=None):
    """Embed an outgoing right half-space through every supplied operation."""
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("A positive numerical current-chart wave number is required")
    _hermitian(exit_q, "Exit wave-number square")
    n = len(exit_q)
    eigen, vectors = eigh(exit_q, check_finite=False)
    k = np.sqrt(eigen.astype(complex))
    out = (vectors*k)@vectors.conj().T
    reflection = (vectors*((kappa-k)/(kappa+k)))@vectors.conj().T
    reflections, forwards = [None]*(len(operators)+1), [None]*len(operators)
    reflections[-1] = reflection
    for i in range(len(operators)-1, -1, -1):
        if cancelled():
            raise InterruptedError("Two-way load cancelled")
        rl, tr, tl, rr = operators[i]
        forwards[i] = solve(np.eye(n)-rr@reflection, tl, check_finite=False)
        reflection = rl+tr@reflection@forwards[i]
        reflections[i] = reflection
        if progress_callback is not None and i % 64 == 0:
            progress_callback(len(operators)-i, len(operators), "Embedding downstream wave reflection")
    admittance = 1j*kappa*solve((np.eye(n)+reflection).T, (np.eye(n)-reflection).T,
                              check_finite=False).T
    return ScatteringLoad(kappa, admittance, out, tuple(reflections), tuple(forwards), ())
