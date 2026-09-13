"""Lossless two-way scalar boundary operator with a longitudinal connection.

Solve (d/dz + i G)^2 f + Q f = 0. Q (m^-2) and G (m^-1) are Hermitian
matrices in a fixed transverse orthonormal basis. For electron charge -e,
G = e A_z/hbar; Q must include the transverse minimal-coupling terms too.
This is not a gun/source model: fields, absorbing components and physical
emission boundary data are owned by the caller, never inferred here.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.constants import hbar, m_e
from scipy.linalg import eigh, expm, solve, LinAlgWarning
from scipy.special import logsumexp

from temsim.immutable_json import freeze_json
from temsim.physics.coupled_low_energy import _hermitian
from temsim.physics.wave_execution import check_available_memory


@dataclass(frozen=True)
class CovariantBoundaryResult:
    boundary_amplitude: np.ndarray
    boundary_covariant_derivative_per_m: np.ndarray
    current_per_boundary_density_m_per_s: np.ndarray
    record: object


def _solve(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            x = solve(a, b, check_finite=False)
        except (np.linalg.LinAlgError, LinAlgWarning) as error:
            raise ValueError("Unresolved covariant boundary solve; no channels were removed") from error
    scale = max(float(np.linalg.norm(a)*np.linalg.norm(x)+np.linalg.norm(b)), np.finfo(float).tiny)
    residual = float(np.linalg.norm(a@x-b)/scale)
    if not np.all(np.isfinite(x)) or not math.isfinite(residual) or residual > 1e-12:
        raise ValueError("Covariant boundary linear-solve residual exceeds 1e-12")
    return x


def _unitarity(blocks):
    rl, tr, tl, rr = blocks
    s = np.block([[rl, tr], [tl, rr]])
    residual = float(np.linalg.norm(s.conj().T@s-np.eye(len(s)), ord=np.inf))
    if not math.isfinite(residual) or residual > 1e-9:
        raise ValueError(f"Covariant slab lost current unitarity ({residual:.9g} > 1e-9); "
                         "refine the numerical representation")
    return residual


def _compose(first, second):
    """Redheffer composition in physical left-to-right order."""
    rl, tr, tl, rr = first
    bl, brt, blt, br = second
    eye = np.eye(len(rl))
    through = _solve(eye-rr@bl, tl)
    returning = _solve(eye-bl@rr, brt)
    return (rl+tr@bl@through, tr@returning,
            blt@through, br+blt@rr@returning)


def _slab(q, connection, width, kappa, cancelled, *, maximum_seed_log_growth=.5):
    """Short safe matrix exponential, then current-conserving doubling.

    A long transfer matrix containing exponentially growing evanescent waves
    is never constructed. Both propagation directions remain in the S matrix.
    Seed scaling bounds transient norm growth, not just rapid unitary phase.
    The optional smaller growth bound supports independent numerical checks;
    it does not change either the operator or the unitarity tolerance.
    """
    if (isinstance(maximum_seed_log_growth, bool)
            or not math.isfinite(maximum_seed_log_growth)
            or not 0 < maximum_seed_log_growth <= .5):
        raise ValueError("Seed log-growth bound must be finite and in (0, 0.5]")
    if cancelled():
        raise InterruptedError("Covariant boundary cancelled")
    n = len(q)
    eye = np.eye(n)
    # y=(f, Df/kappa), y = chart @ (u,v). G need not commute with Q.
    chart = np.block([[eye, eye], [1j*eye, -1j*eye]])
    inverse = .5*np.block([[eye, -1j*eye], [eye, 1j*eye]])
    generator = np.block([[-1j*connection, kappa*eye], [-q/kappa, -1j*connection]])
    generator = inverse@generator@chart
    # High-order exponential formulas may use signed numerical substeps.
    # Bound growth with |h|, while exp(A*h) retains its actual sign. Public
    # physical layer widths are still required to be positive below.
    size = float(np.linalg.norm(generator, ord=np.inf))*abs(width)
    if not math.isfinite(size):
        raise ValueError("Covariant slab is outside the finite numerical domain")
    # For Hermitian Q,G, H=(A+A*)/2 has eigenvalues +/- eigenvalues of
    # (kappa I-Q/kappa)/2. The -iG connection is skew-Hermitian and cannot
    # itself grow the norm. ||exp(+/- A h)||_2 <= exp(h ||H||_2), so a
    # log-growth bound of 0.5 keeps seed condition <= e, including transient
    # nonnormal growth. Using ||A||/0.25 instead needlessly doubles rapid
    # phase and amplifies roundoff. A separate ||A h||_inf <= 16 cap keeps
    # the exponential's phase argument moderate; no phase is removed.
    growth = .5*float(np.max(abs(eigh(kappa*eye-q/kappa,
                                     eigvals_only=True, check_finite=False))))
    scale = max(growth*abs(width)/maximum_seed_log_growth, size/16.)
    if not math.isfinite(scale):
        raise ValueError("Covariant slab is outside the finite numerical domain")
    doublings = max(0, math.ceil(math.log2(scale))) if scale else 0
    if doublings > 64:
        raise ValueError("Covariant slab exceeds the supported scattering-doubling range")
    transfer = expm(generator*(width/(2**doublings)))
    a, b = transfer[:n, :n], transfer[:n, n:]
    c, d = transfer[n:, :n], transfer[n:, n:]
    inv_d = _solve(d, eye)
    reflected = -_solve(d, c)
    blocks = (reflected, inv_d, a+b@reflected, b@inv_d)
    error = _unitarity(blocks)
    for _ in range(doublings):
        if cancelled():
            raise InterruptedError("Covariant boundary cancelled")
        blocks = _compose(blocks, blocks)
        error = max(error, _unitarity(blocks))
        if not np.any(blocks[1]) or not np.any(blocks[2]):
            raise ValueError("Coupled transmission underflow requires log-scaled channel transport")
    return blocks, {"scattering_doublings": doublings, "maximum_unitarity_residual": error,
        "seed_scaling": "hermitian-growth-and-phase-v1",
        "maximum_seed_log_growth": maximum_seed_log_growth,
        "seed_log_growth_bound": growth*abs(width)/(2**doublings),
        "seed_generator_norm": size/(2**doublings)}


def propagate_covariant_boundary(layer_q_m2, layer_connection_per_m, layer_widths_m,
        exit_q_m2, left_total_amplitude, *, reference_wave_number_per_m=None,
        maximum_channels=1024, maximum_working_bytes=72*1024**3,
        cancelled=lambda: False):
    """Execute the coupled magnetic/electric boundary kernel, not a full gun.

    The left total field is prescribed. The uniform right half-space has
    G=0 and outgoing/decaying derivative i sqrt(Q_exit). Nonzero exit G needs
    an explicitly solved boundary load, not an assumed scalar replacement.
    f and Df are continuous across layers, so ordinary derivatives can jump
    when G jumps. Return Df explicitly to avoid confusing it with f'.
    """
    q, g, widths, end, left = map(np.asarray, (layer_q_m2, layer_connection_per_m,
        layer_widths_m, exit_q_m2, left_total_amplitude))
    if q.ndim != 3 or min(q.shape) < 1 or q.shape[1] != q.shape[2]:
        raise ValueError("Provide square transverse matrices in nonempty axial layers")
    layers, n, _ = q.shape
    if g.shape != q.shape or end.shape != (n, n) or left.shape != (n,) or widths.shape != (layers,):
        raise ValueError("Covariant boundary matrix and field shapes do not match")
    if type(maximum_channels) is not int or not 1 <= n <= maximum_channels:
        raise ValueError("Covariant channel count exceeds its numerical budget")
    required = 16*((6*layers+120)*n*n+8*(layers+1)*n)+q.nbytes+g.nbytes+end.nbytes
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Covariant memory budget must be a positive integer")
    if required > maximum_working_bytes:
        raise MemoryError(f"Covariant boundary needs approximately {required} working bytes")
    check_available_memory(required)
    if cancelled():
        raise InterruptedError("Covariant boundary cancelled")
    if np.iscomplexobj(widths) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Covariant layer widths must be positive finite real metres")
    if not np.all(np.isfinite(left)):
        raise ValueError("Boundary amplitude must be finite")
    q, g, end, left = (np.array(v, dtype=complex, copy=True) for v in (q, g, end, left))
    widths = np.array(widths, dtype=float, copy=True)
    for i in range(layers):
        _hermitian(q[i], "Transverse kinetic operator")
        _hermitian(g[i], "Longitudinal connection")
    _hermitian(end, "Exit operator")
    magnitude = max(float(np.linalg.norm(end, ord=np.inf)),
                    max(float(np.linalg.norm(v, ord=np.inf)) for v in q))
    kappa = math.sqrt(magnitude) if reference_wave_number_per_m is None else reference_wave_number_per_m
    if isinstance(kappa, bool) or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("A positive finite chart wavenumber is required")
    values, vectors = eigh(end, check_finite=False)
    k = np.sqrt(values.astype(complex))
    reflection = (vectors*((kappa-k)/(kappa+k)))@vectors.conj().T
    reflections, forwards, rows = [None]*(layers+1), [None]*layers, [None]*layers
    reflections[-1] = reflection
    for i in range(layers-1, -1, -1):
        if cancelled():
            raise InterruptedError("Covariant boundary cancelled")
        (rl, tr, tl, rr), rows[i] = _slab(q[i], g[i], widths[i], kappa, cancelled)
        forwards[i] = _solve(np.eye(n)-rr@reflection, tl)
        reflection = rl+tr@reflection@forwards[i]
        reflections[i] = reflection
    u = _solve(np.eye(n)+reflections[0], left)
    scale = max(float(hbar/m_e*kappa*np.vdot(u, u).real), np.finfo(float).tiny)
    fields, covariant = [], []
    for i in range(layers+1):
        if cancelled():
            raise InterruptedError("Covariant boundary cancelled")
        v = reflections[i]@u
        fields.append(u+v)
        covariant.append(1j*kappa*(u-v))
        if i < layers:
            before = u
            u = forwards[i]@u
            if np.any(before) and not np.any(u):
                raise ValueError("Coupled field underflow requires log-scaled channel transport")
    fields, covariant = np.asarray(fields), np.asarray(covariant)
    current = hbar/m_e*np.imag(np.einsum("ij,ij->i", fields.conj(), covariant))
    # The outgoing load has a positive spectral flux form. Avoid subtracting
    # almost equal incident/reflected currents to infer tiny transmission.
    # Keep log flux if squaring a representable amplitude would underflow.
    exit_coefficients = vectors.conj().T@fields[-1]
    carrying = (k.real > 0) & (abs(exit_coefficients) > 0)
    exit_log_current = None
    if np.any(carrying):
        exit_log_current = float(math.log(hbar/m_e)+logsumexp(
            np.log(k.real[carrying])+2*np.log(abs(exit_coefficients[carrying]))))
        if not math.isfinite(exit_log_current) or exit_log_current > math.log(np.finfo(float).max):
            raise ValueError("Covariant outgoing current exceeds the finite numerical domain")
    current[-1] = 0. if exit_log_current is None else math.exp(exit_log_current)
    residual = float(abs(current-current[0]).max()/scale)
    if (not all(np.all(np.isfinite(v)) for v in (fields, covariant, current))
            or not math.isfinite(scale) or not math.isfinite(residual)
            or residual > 1e-9 or current.min() < -1e-9*scale):
        raise ValueError("Covariant boundary current balance failed")
    if cancelled():
        raise InterruptedError("Covariant boundary cancelled")
    frozen = [np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape) for v in (fields, covariant, current)]
    return CovariantBoundaryResult(*frozen, freeze_json({
        "schema": "covariant-two-way-boundary-v1", "channels": n, "layers": layers,
        "reference_wave_number_per_m": float(kappa), "estimated_working_bytes": required,
        "current_balance_residual": residual, "slabs": rows,
        "current_balance_scale": "incident numerical-chart current; not a relative bound on tiny transmitted current",
        "exit_log_current_per_boundary_density_m_per_s": exit_log_current,
        "exit_current_representation": ("zero" if exit_log_current is None else
            "log_below_normal_float_range" if exit_log_current < math.log(np.finfo(float).tiny) else "float_and_log"),
        "equation": "(d/dz + i G)^2 f + Q f = 0; G=e*A_z/hbar for charge -e",
        "boundary": "prescribed total left field; right G=0 outgoing/decaying half-space",
        "derivative": "covariant Df, not the ordinary derivative f'",
        "gun_source_admission": "NOT_A_GUN_CHECKPOINT",
        "convergence": "REQUIRES_INDEPENDENT_BASIS_DOMAIN_AND_AXIAL_REFINEMENT"}))
