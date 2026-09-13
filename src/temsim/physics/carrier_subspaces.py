"""Current-orthogonal coordinates of a complete propagating two-way slab.

An invariant graph [I; X] has J-orthogonal complement [X*; I], J=diag(I,-I).
Cholesky factors implement these coordinate maps; no physical state, source
weight, S-matrix singular value or transmitted current is renormalised.
"""
import numpy as np
from scipy.linalg import cholesky, solve, solve_triangular, solve_sylvester


def graph_current_subspaces(vectors, eigenvalues, h, connection, carrier, width):
    n = len(h)
    positive = vectors[:, np.asarray(eigenvalues) > 0]
    if positive.shape != (2*n, n):
        raise ValueError("Current graph needs the complete positive invariant subspace")
    x = solve(positive[:n].T, positive[n:].T, check_finite=False).T
    return refine_current_graph(h, connection, carrier, width, initial=x)


def refine_current_graph(h, connection, carrier, width, *, initial=None):
    """Solve the complete invariant graph, without an initial 2N eigensystem.

    This is a Newton solve of the same full Riccati equation. The residual,
    phase budget, current Gram matrix and complete complementary subspace
    must pass exactly the same checks as the spectrally initialised graph.
    """
    n = len(h)
    x = -np.asarray(h, complex)/(2*carrier) if initial is None else np.array(initial, complex, copy=True)
    # Invariance defect evaluated with the carrier kept separate. A graph
    # correction may not replace an unresolved invariant subspace by merely
    # current-orthogonal channels. Include the physical step in the check.
    eye = np.eye(n)
    # Refine the complete Riccati equation, keeping the large +/- carrier
    # separate from the small transverse matrices. A raw eigensolver's tiny
    # subspace error can otherwise dominate a very long residual phase.
    for iteration in range(7):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = -h-(h+connection)@x-x@(h-connection)-x@h@x-2*carrier*x
        if not np.all(np.isfinite(residual)):
            raise ValueError("Carrier graph invariant-subspace solve diverged")
        defect = float(np.linalg.norm(residual, ord=2))
        if defect*abs(width) <= 1e-12 or iteration == 6:
            break
        left = -h-connection-x@h-carrier*eye
        right = -h+connection-h@x-carrier*eye
        x += solve_sylvester(left, right, -residual)
    if not np.isfinite(defect) or defect*abs(width) > 1e-9:
        raise ValueError(f"Carrier graph invariant-subspace phase defect {defect*abs(width):.6g} exceeds 1e-9")
    result = []
    for raw, gram in ((np.vstack((eye, x)), eye-x.conj().T@x),
                      (np.vstack((x.conj().T, eye)), eye-x@x.conj().T)):
        factor = cholesky(gram, lower=True, check_finite=False)
        result.append(solve_triangular(factor, raw.conj().T, lower=True, check_finite=False).conj().T)
    basis = np.concatenate(result, axis=1)
    metric = np.diag(np.r_[np.ones(n), -np.ones(n)])
    orthogonality = float(np.linalg.norm(basis.conj().T@metric@basis-metric, ord=np.inf))
    if not np.isfinite(orthogonality) or orthogonality > 1e-10:
        raise ValueError("Carrier graph current coordinates remain ill-conditioned")
    return tuple(result), {"current_graph_orthogonality": orthogonality,
                           "current_graph_invariant_phase_defect": defect*abs(width),
                           "current_graph_newton_iterations": iteration}
