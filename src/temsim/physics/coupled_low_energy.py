"""Two-way stationary scalar wave propagation with transverse mode coupling.

Solve f''(z) + Q(z) f(z) = 0 for piecewise-constant Hermitian Q (m^-2).
The transverse basis is fixed and orthonormal. The left TOTAL complex field
is prescribed; the right half-space is outgoing/decaying. Normal derivatives
and currents are solved, not inferred from |f|^2 or a forward-only launch.

This is an internal boundary operator, not a tip emission law or a gun cache.
Electrostatic callers must enforce the nonrelativistic model's energy domain.
Magnetic covariant derivatives, absorbing bores, curved emitter boundaries,
and matching to the full column are not implicitly supplied by this kernel.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.constants import hbar, m_e
from scipy.linalg import eigh, solve, LinAlgWarning

from temsim.immutable_json import freeze_json
from temsim.physics.wave_execution import check_available_memory


@dataclass(frozen=True)
class CoupledBoundaryResult:
    boundary_amplitude: np.ndarray        # (layer boundaries, transverse modes)
    boundary_derivative_per_m: np.ndarray
    current_per_boundary_density_m_per_s: np.ndarray
    record: object


def _hermitian(matrix, name):
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), np.finfo(float).tiny)
    if np.linalg.norm(matrix-matrix.conj().T, ord=np.inf) > 1e-12*scale:
        raise ValueError(f"{name} must be Hermitian; absorption needs a separate operator")


def _slab(q, width, kappa):
    """Exact uniform-slab scattering in a common positive reference chart.

    f = u+v, f' = i*kappa*(u-v). kappa is a numerical coordinate scale,
    NOT a physical injection energy or a replacement medium.
    Only exp(-Im(k)*d) is formed, never exp(+Im(k)*d).
    """
    values, vectors = eigh(q, check_finite=False)
    k = np.sqrt(values.astype(complex))
    phase = k*width
    attenuation = abs(phase.imag)
    positive, negative = np.exp(1j*phase-attenuation), np.exp(-1j*phase-attenuation)
    cosine = (positive+negative)/2
    s = np.empty_like(k)
    small = abs(phase) < 1e-4
    s[small] = width*np.exp(-attenuation[small])*(1-phase[small]**2/6+phase[small]**4/120)
    np.divide(positive-negative, 2j*k, out=s, where=~small)
    denominator = cosine-.5j*(kappa+values/kappa)*s
    reflected = .5j*(values/kappa-kappa)*s/denominator
    log_transmitted = -attenuation-np.log(denominator)
    if np.any(log_transmitted.real < math.log(np.finfo(float).tiny)):
        raise ValueError("Coupled-channel transmission needs a log-scaled representation; no evanescent channel was clipped")
    transmitted = np.exp(log_transmitted)
    residual = max(float(abs(abs(reflected)**2+abs(transmitted)**2-1).max()),
                   float(abs(2*np.real(reflected.conj()*transmitted)).max()))
    if not math.isfinite(residual) or residual > 1e-10:
        raise ValueError("Uniform coupled slab did not preserve current")
    r = (vectors*reflected)@vectors.conj().T
    t = (vectors*transmitted)@vectors.conj().T
    return r, t, {"evanescent_channels": int(np.count_nonzero(values < 0)),
        "grazing_channels": int(np.count_nonzero(values == 0)),
        "minimum_log_transmission_magnitude": float(log_transmitted.real.min()),
        "slab_unitarity_residual": residual}


def propagate_coupled_boundary(layer_q_m2, layer_widths_m, exit_q_m2,
        left_total_amplitude, *, reference_wave_number_per_m=None,
        maximum_channels=1024, maximum_working_bytes=72*1024**3,
        cancelled=lambda: False):
    """Reflection-embedding solution, retaining all declared transverse modes.

    Q has shape (layers, channels, channels), widths are metres. The exit Q
    defines a uniform semi-infinite load, not an arbitrary forward initial
    condition. Channel coefficients are complex cell/basis amplitudes. The
    result is NOT a flux-normalised BeamState and is never a gun-exit source.

    Converge the transverse basis, potential quadrature and axial subdivision
    independently. Cancellation/budgets fail before a result is published.
    Dense reference cost is O(layers*channels^3), not a production GPU claim.
    """
    q, widths, end, left = map(np.asarray, (layer_q_m2, layer_widths_m, exit_q_m2, left_total_amplitude))
    if q.ndim != 3 or q.shape[0] < 1 or q.shape[1] < 1 or q.shape[1] != q.shape[2]:
        raise ValueError("Provide nonempty square coupled-channel matrices per layer")
    layers, n, _ = q.shape
    if widths.shape != (layers,) or end.shape != (n, n) or left.shape != (n,):
        raise ValueError("Boundary field, exit operator and widths must match the layer channels")
    if type(maximum_channels) is not int or maximum_channels < 1 or n > maximum_channels:
        raise ValueError("Coupled-channel count exceeds its numerical budget")
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Coupled-channel memory budget must be a positive integer")
    required = 16*((5*layers+32)*n*n+8*(layers+1)*n)+q.nbytes+end.nbytes
    if required > maximum_working_bytes:
        raise MemoryError(f"Coupled boundary needs approximately {required} working bytes")
    check_available_memory(required)
    if cancelled():
        raise InterruptedError("Coupled boundary cancelled")
    if np.iscomplexobj(widths) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Layer widths must be positive finite real metres")
    if not np.all(np.isfinite(left)):
        raise ValueError("Total boundary field must be finite")
    # Fixed double-precision detached problem; cancellation callbacks and live
    # editors cannot alter a layer after it has been validated/factorised.
    q, end, left = (np.array(value, dtype=complex, copy=True) for value in (q, end, left))
    widths = np.array(widths, dtype=float, copy=True)
    for i in range(layers):
        _hermitian(q[i], f"Layer {i} operator")
    _hermitian(end, "Exit operator")
    magnitude = max(float(np.linalg.norm(end, ord=np.inf)),
                    max(float(np.linalg.norm(value, ord=np.inf)) for value in q))
    kappa = math.sqrt(magnitude) if reference_wave_number_per_m is None else reference_wave_number_per_m
    if isinstance(kappa, bool) or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("A positive finite reference wavenumber is required (explicitly for an all-grazing problem)")
    eye = np.eye(n, dtype=complex)
    residuals = []
    def checked_solve(a, b):
        if cancelled():
            raise InterruptedError("Coupled boundary cancelled")
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                x = solve(a, b, assume_a="gen", check_finite=False)
            except (np.linalg.LinAlgError, LinAlgWarning) as error:
                raise ValueError("Ill-conditioned coupled boundary; refine or change the numerical chart") from error
        scale = max(float(np.linalg.norm(a)*np.linalg.norm(x)+np.linalg.norm(b)), np.finfo(float).tiny)
        residual = float(np.linalg.norm(a@x-b)/scale)
        if not np.all(np.isfinite(x)) or residual > 1e-12:
            raise ValueError("Unresolved coupled-boundary linear solve")
        residuals.append(residual)
        return x
    values, vectors = eigh(end, check_finite=False)
    exit_k = np.sqrt(values.astype(complex))
    reflection = (vectors*((kappa-exit_k)/(kappa+exit_k)))@vectors.conj().T
    reflections, transfers, rows = [None]*(layers+1), [None]*layers, [None]*layers
    reflections[-1] = reflection
    for i in range(layers-1, -1, -1):
        if cancelled():
            raise InterruptedError("Coupled boundary cancelled")
        r, t, row = _slab(q[i], float(widths[i]), kappa)
        transfer = checked_solve(eye-r@reflection, t)
        reflection = r+t@reflection@transfer
        reflections[i], transfers[i], rows[i] = reflection, transfer, row
    u = checked_solve(eye+reflections[0], left.astype(complex))
    incident_chart_norm = float(np.vdot(u, u).real)
    fields, derivatives = [], []
    for i in range(layers+1):
        if cancelled():
            raise InterruptedError("Coupled boundary cancelled")
        v = reflections[i]@u
        fields.append(u+v)
        derivatives.append(1j*kappa*(u-v))
        if i < layers:
            previous = u
            u = transfers[i]@u
            if np.any(previous) and not np.any(u):
                raise ValueError("Coupled field underflow; a log-scaled propagation is required")
    fields, derivatives = np.asarray(fields), np.asarray(derivatives)
    if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(derivatives)):
        raise ValueError("Coupled boundary field exceeds the floating-point domain")
    current = hbar/m_e*np.imag(np.einsum("ij,ij->i", fields.conj(), derivatives))
    current_scale = max(hbar/m_e*kappa*incident_chart_norm, np.finfo(float).tiny)
    current_residual = float(abs(current-current[0]).max()/current_scale)
    if (not math.isfinite(current_scale) or not math.isfinite(current_residual)
            or not np.all(np.isfinite(current)) or current_residual > 1e-9
            or float(current.min()) < -1e-9*current_scale):
        raise ValueError("Coupled boundary does not conserve nonnegative outgoing current")
    if cancelled():
        raise InterruptedError("Coupled boundary cancelled")
    frozen = [np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape) for a in (fields, derivatives, current)]
    return CoupledBoundaryResult(*frozen, freeze_json({
        "schema": "coupled-scalar-boundary-v1", "channels": n, "layers": layers,
        "reference_wave_number_per_m": float(kappa), "estimated_working_bytes": required,
        "maximum_linear_solve_residual": max(residuals), "current_balance_residual": current_residual,
        "slabs": rows, "current_definition": "hbar/m_e Im(f* f'); not |f|^2",
        "boundary": "prescribed total left field; outgoing/decaying right half-space",
        "phase": "complex stationary field at every interface; no fabricated axial flight time",
        "convergence": "REQUIRES_TRANSVERSE_AND_AXIAL_REFINEMENT",
        "gun_source_admission": "NOT_A_GUN_CHECKPOINT"}))
