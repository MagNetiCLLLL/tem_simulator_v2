"""Locally refined two-port propagation, with no fitted phase or flux repair.

Each estimate compares a CF4 interval with two independently sampled halves
in the same flux chart. The raw difference is used (no division by the
asymptotic factor 15). Accepted halves are composed, retaining reflection.
The accumulated indicator is not a rigorous global bound near resonances;
independent basis/coordinate and full-chain convergence remain necessary.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.magnus_scattering import GAUSS_FRACTIONS, cf4_slab
from temsim.physics.scattering_load import compose


@dataclass(frozen=True)
class AxialRefinement:
    tolerance: float = 1e-6
    maximum_evaluations: int = 4096
    maximum_depth: int = 20

    def validate(self):
        if isinstance(self.tolerance, bool) or not math.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("Axial refinement tolerance must lie in (0, 1)")
        if type(self.maximum_evaluations) is not int or self.maximum_evaluations < 3:
            raise ValueError("Axial refinement needs at least three step evaluations")
        if type(self.maximum_depth) is not int or not 1 <= self.maximum_depth <= 30:
            raise ValueError("Axial refinement maximum depth must lie in [1, 30]")
        return self


def scattering_distance(first, second):
    """Infinity norm of the full complex two-port difference, both directions."""
    a, b, c, d = (x-y for x, y in zip(first, second))
    return float(max(np.max(np.sum(abs(a), axis=1)+np.sum(abs(b), axis=1)),
                     np.max(np.sum(abs(c), axis=1)+np.sum(abs(d), axis=1))))


def adaptive_cf4(sample, width, reference_k, settings=AxialRefinement(), *,
                 cancelled=lambda: False, progress_callback=None):
    """sample(f) returns residual/connection at action fraction f in [0,1].

    Physical discontinuities MUST delimit calls, never lie inside a call.
    Numerical moving-coordinate derivatives must use one consistent guide
    throughout the call, independently of the adaptive subdivision.
    """
    settings.validate()
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Adaptive scattering needs a positive finite interval")
    evaluations, accepted, depth_used = 0, 0, 0
    total_indicator, largest_indicator = 0., 0.
    def evaluate(left, right):
        nonlocal evaluations
        if cancelled():
            raise InterruptedError("Adaptive coherent propagation cancelled")
        if evaluations >= settings.maximum_evaluations:
            raise ValueError(f"Axial refinement exhausted {evaluations} step evaluations; "
                "no unconverged interval was accepted")
        evaluations += 1
        if progress_callback is not None:
            progress_callback(evaluations, settings.maximum_evaluations, "Resolving coherent axial phase")
        samples = [sample(left+(right-left)*f) for f in GAUSS_FRACTIONS]
        result, _ = cf4_slab(*samples, (right-left)*width, reference_k, cancelled=cancelled)
        return result
    def refine(left, right, coarse, depth):
        nonlocal accepted, total_indicator, largest_indicator, depth_used
        middle = (left+right)/2
        first, second = evaluate(left, middle), evaluate(middle, right)
        fine = compose(first, second)
        indicator = scattering_distance(coarse, fine)
        if not math.isfinite(indicator):
            raise ValueError("Non-finite axial propagation error estimate")
        local_budget = settings.tolerance*(right-left)
        depth_used = max(depth_used, depth)
        if indicator <= local_budget:
            accepted += 2
            total_indicator += indicator
            largest_indicator = max(largest_indicator, indicator)
            return fine
        if depth >= settings.maximum_depth:
            raise ValueError(f"Axial propagation error {indicator:.6g} exceeds local budget "
                f"{local_budget:.6g} at depth {depth}; no unconverged interval was accepted")
        return compose(refine(left, middle, first, depth+1), refine(middle, right, second, depth+1))
    result = refine(0., 1., evaluate(0., 1.), 0)
    return result, {"axial_integrator": "adaptive_cf4", "step_evaluations": evaluations,
        "accepted_substeps": accepted, "maximum_depth_used": depth_used,
        "summed_local_indicator": total_indicator, "largest_local_indicator": largest_indicator,
        "requested_local_budget": settings.tolerance,
        "error_scope": "Raw two-port complex step-doubling indicator; not a full-chain accuracy certificate"}
