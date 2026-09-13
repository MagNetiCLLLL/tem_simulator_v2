"""Input-bound fixed FEM assembly for repeated physical tip/load solves.

Only volume/source/interface matrices are reused. Every downstream load is
still coupled and solved anew. This cache contains no independently chosen
boundary field, emitted wave, or output current.
"""
from hashlib import sha256
import math

import numpy as np
from scipy.sparse import coo_matrix


def assembly_identity(problem, energy, count, width, curvature, kappa, factor_phase):
    digest = sha256(repr((energy, count, width, curvature, kappa, factor_phase)).encode())
    def array(value):
        value = np.ascontiguousarray(value)
        digest.update(repr((value.shape, value.dtype.str)).encode())
        digest.update(value.view(np.uint8))
    for name in ("points", "triangles", "potential", "drive"):
        array(problem[name])
    for name in ("source", "side", "top"):
        array(problem["edges"][name])
    for matrix in problem["matrices"]:
        digest.update(repr((matrix.shape, matrix.format)).encode())
        for value in (matrix.data, matrix.indices, matrix.indptr):
            array(value)
    return digest.hexdigest()


def prepare_joint_assembly(problem, energy, count, width, curvature, kappa, factor_phase,
                           *, cache=None, maximum_working_bytes, cancelled):
    from temsim.physics.surface_wave import KINETIC_NM2_PER_EV, fem_volume, robin_port
    from temsim.physics.radial_gun_wave import REST_EV, basis_values, aperture_projection
    from temsim.physics.radial_phase_fem import radial_phase_terms, interface_gram
    from temsim.physics.wave_execution import check_available_memory
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Joint boundary assembly memory budget must be a positive integer")
    if cancelled():
        raise InterruptedError("Joint tip boundary assembly cancelled")
    identity = assembly_identity(problem, energy, count, width, curvature, kappa, factor_phase)
    if cache is not None and identity in cache:
        result = cache[identity]
        if result["retained_bytes"] > maximum_working_bytes:
            raise MemoryError("Cached joint boundary assembly exceeds its numerical memory budget")
        return result, True
    if cache is not None:
        cache.clear()  # Release the previous energy/chart, not an unbounded cache.
    p, t, edges = (problem[k] for k in ("points", "triangles", "edges"))
    # Conservative sparse-assembly allowance before allocating matrix products.
    required = 16*(128*len(p)+16*sum(matrix.nnz for matrix in problem["matrices"])
                   +8*len(np.unique(edges["top"]))*count)
    if required > maximum_working_bytes:
        raise MemoryError("Joint boundary assembly exceeds its numerical working-memory budget")
    check_available_memory(required)
    phi, drive = problem["potential"], problem["drive"]
    stiffness, mass, potential = problem["matrices"]
    _, _, potential_square = fem_volume(p, t, phi*phi)
    matrix = (stiffness-KINETIC_NM2_PER_EV*((energy+energy*energy/(2*REST_EV))*mass
        +(1+energy/REST_EV)*potential+potential_square/(2*REST_EV))).astype(complex)
    gamma = kappa*curvature if factor_phase else 0.
    phase = .5*gamma*p[:, 0]**2
    if gamma:
        linear, square = radial_phase_terms(p, t)
        matrix += 1j*gamma*linear+gamma*gamma*square
    effective_kinetic = (energy+phi)*(1+(energy+phi)/(2*REST_EV))
    source = robin_port(p, edges["source"], effective_kinetic)
    side = robin_port(p, edges["side"], effective_kinetic)
    matrix -= 1j*(source+side)
    incoming = np.asarray(drive).copy()*np.exp(-1j*phase)
    flux_in = float(np.vdot(incoming, source@incoming).real)
    if not math.isfinite(flux_in) or flux_in <= 0:
        raise ValueError("Physical tip reservoir has no incoming flux")
    incoming /= math.sqrt(flux_in)
    top = np.unique(edges["top"])
    interior = np.setdiff1d(np.arange(len(p)), top)
    trace = basis_values(p[top, 0], width, 0. if gamma else curvature, kappa, count)
    gram_eigen = np.linalg.eigvalsh(interface_gram(p, edges["top"], trace))
    domain_gram = aperture_projection(float(p[top, 0].max()), width, count, 0)
    row = np.r_[interior, np.repeat(top, count)]
    column = np.r_[np.arange(len(interior)), np.tile(len(interior)+np.arange(count), len(top))]
    data = np.r_[np.ones(len(interior)), trace.ravel()]
    transform = coo_matrix((data, (row, column)), shape=(len(p), len(interior)+count)).tocsc()
    reduced = (transform.conj().T@matrix@transform).tocsc()
    rhs = transform.conj().T@(-2j*(source@incoming))
    result = dict(reduced=reduced, rhs=rhs, transform=transform, source=source, side=side,
        incoming=incoming, phase=phase, gamma=gamma,
        domain_error=float(np.linalg.norm(np.eye(count)-domain_gram, 2)),
        gram_error=float(max(abs(gram_eigen-1))), matrix_inputs_digest=identity)
    retained = 0
    for value in result.values():
        if isinstance(value, np.ndarray):
            retained += value.nbytes
            value.setflags(write=False)
        elif hasattr(value, "indptr"):
            for buffer in (value.data, value.indices, value.indptr):
                retained += buffer.nbytes
                buffer.setflags(write=False)
    if retained > maximum_working_bytes:
        raise MemoryError("Joint boundary assembly exceeds its numerical working-memory budget")
    check_available_memory(retained)
    result["retained_bytes"] = retained
    if cancelled():
        raise InterruptedError("Joint tip boundary assembly cancelled before caching")
    if cache is not None:
        cache[identity] = result
    return result, False
