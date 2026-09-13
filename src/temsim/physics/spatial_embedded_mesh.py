"""Adaptive numerical leaves; immutable physical events are never subdivided.

Both halves are full two-way operators. The returned load preserves the
original physical boundary interface while retaining every numerical leaf.
No interpolation, extrapolation or source/phase renormalisation is used.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.occupied_axial_refinement import evaluate_interval, apply_two_port, internal_incoming


def size(operator):
    return sum(block.nbytes for block in operator)


class SpatialLeaf:
    expanded = False
    certificate = None
    divisions = 1  # Two new evaluations per leaf; scheduling only.

    def __init__(self, physical_index, sample, width, kappa, coarse, *,
                 start=0., end=1., depth=0, owns_coarse=False):
        self.physical_index, self.sample = physical_index, sample
        self.width, self.kappa, self.coarse = width, kappa, coarse
        self.start, self.end, self.depth = start, end, depth
        self.owns_coarse = owns_coarse
        self.operator, self.halves = None, None

    @property
    def can_recheck_without_propagation(self):
        """Current-port readout only; both complete executed halves are present."""
        return self.halves is not None

    def refine(self, left, right, unused_budget, work):
        work.check()
        if self.halves is None:
            if self.depth >= work.settings.maximum_depth:
                raise ValueError("Spatial embedded refinement reached its depth budget; no source was published")
            middle = (self.start+self.end)/2
            halves = tuple(evaluate_interval(self.sample, self.width, self.kappa, a, b, work)
                           for a, b in ((self.start, middle), (middle, self.end)))
            work.retain(sum(size(value) for value in halves))
            self.halves = halves
        # Both complete half-step matrices remain. For this particular error
        # readout, solve their reflected internal port for the CURRENT input
        # vectors, instead of retaining a third composed four-block matrix.
        # New BVP inputs are solved afresh; no old occupied vector is reused.
        first, second = self.halves
        back, forward = internal_incoming(first, second, left, right)
        fine = np.r_[first[0]@left+first[1]@back,
                     second[2]@forward+second[3]@right]
        error = float(np.linalg.norm(apply_two_port(self.coarse, left, right)-fine)*math.sqrt(self.kappa))
        if not math.isfinite(error):
            raise ValueError("Nonfinite spatial embedded indicator")
        return error, False

    def split(self, work):
        if self.halves is None:
            raise ValueError("Only executed halves may refine a numerical mesh")
        middle = (self.start+self.end)/2
        children = tuple(SpatialLeaf(self.physical_index, self.sample, self.width, self.kappa, op,
            start=a, end=b, depth=self.depth+1, owns_coarse=True) for op, (a, b) in
            zip(self.halves, ((self.start, middle), (middle, self.end))))
        work.release(size(self.coarse) if self.owns_coarse else 0)
        return children

    def adopt_worker_state(self, other):
        """Retain shared executed samplers, not one deserialised copy per leaf."""
        identity = ("physical_index", "width", "kappa", "start", "end", "depth", "owns_coarse")
        if not isinstance(other, SpatialLeaf) or any(getattr(self, key) != getattr(other, key) for key in identity):
            raise RuntimeError("Spatial worker returned a different numerical interval")
        if other.halves is None:
            raise RuntimeError("Spatial worker did not return complete two-way operators")
        self.halves = other.halves
        return self


@dataclass(frozen=True)
class PhysicalBoundarySubset:
    """Exact indexing of an executed load, including both faces of every stop."""
    load: object
    indices: tuple

    @property
    def kappa(self):
        return self.load.kappa

    @property
    def input_admittance(self):
        return self.load.input_admittance

    @property
    def output_wave_number(self):
        return self.load.output_wave_number

    def propagate(self, total_left):
        states = self.load.propagate(total_left)
        return tuple(value[np.asarray(self.indices)] for value in states)


def initial_mesh(plan):
    return [SpatialLeaf(index, *plan["samplers"][index], plan["kappa"], operator)
            if index in plan["samplers"] else (index, operator)
            for index, operator in enumerate(plan["operators"])]


def operators_and_boundaries(mesh, *, fine=False):
    operators, physical, coarse = [], [0], [0]
    for index, item in enumerate(mesh):
        if isinstance(item, SpatialLeaf):
            operators.extend(item.halves if fine else (item.coarse,))
            original = item.physical_index
            end_of_original = item.end == 1.
        else:
            original, operator = item
            operators.append(operator)
            end_of_original = True
        coarse.append(len(operators))
        if end_of_original:
            if original != len(physical)-1:
                raise RuntimeError("Adaptive wave mesh lost physical boundary order")
            physical.append(len(operators))
    return operators, tuple(physical), tuple(coarse)
