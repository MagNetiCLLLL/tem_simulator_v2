"""Deterministic subinterval parallelism for costly global mesh comparisons.

Every constant two-way substep is executed. Adjacent substeps are composed
into fixed-size chunks, then chunks are composed in their physical order.
Serial and parallel paths use the SAME grouping, never completion order.
No source, channel, boundary solve or acceptance criterion is changed.
"""
import math

import numpy as np

from temsim.physics.scattering_load import compose


CHUNK_STEPS = 128


def evaluate_chunk(sample, width, kappa, divisions, begin, end, work):
    from temsim.physics.occupied_axial_refinement import evaluate_interval
    result = None
    for j in range(begin, end):
        step = evaluate_interval(sample, width, kappa, j/divisions, (j+1)/divisions, work)
        result = step if result is None else compose(result, step)
    return result


def evaluate_embedded_interval(sample, width, kappa, divisions, work):
    result = None
    for begin in range(0, divisions, CHUNK_STEPS):
        chunk = evaluate_chunk(sample, width, kappa, divisions, begin, min(begin+CHUNK_STEPS, divisions), work)
        result = chunk if result is None else compose(result, chunk)
    return result


class _Chunk:
    expanded = False
    certificate = None

    def __init__(self, tree, begin, end):
        self.sample, self.width, self.kappa = tree.sample, tree.width, tree.kappa
        self.total_divisions = tree.divisions*2
        self.begin, self.end = begin, end
        self.divisions = end-begin  # Scheduling cost only.
        self.operator = None

    def refine(self, unused_left, unused_right, unused_budget, work):
        self.operator = evaluate_chunk(self.sample, self.width, self.kappa, self.total_divisions,
                                       self.begin, self.end, work)
        work.retain(sum(block.nbytes for block in self.operator))
        return 0., False


def refine_embedded_process_intervals(roots, incoming, work):
    """Distribute even one long interval; keep bounded, complete chunk results."""
    from temsim.physics.occupied_wave_processes import refine_process_intervals
    jobs, destinations = {}, {}
    for index, tree in roots.items():
        if tree.operator is not None:
            continue
        divisions = tree.divisions*2
        if divisions > 2**work.settings.maximum_depth:
            raise ValueError("Global embedded refinement reached its depth budget; no source was published")
        for begin in range(0, divisions, CHUNK_STEPS):
            job = len(jobs)
            jobs[job] = _Chunk(tree, begin, min(begin+CHUNK_STEPS, divisions))
            destinations[job] = index
    if jobs:
        zero = np.zeros(1, complex)  # Unused by a complete-operator chunk job.
        inputs = [(zero, zero)]*(len(jobs)+1)
        updates = refine_process_intervals(jobs, inputs, math.inf, work, maximum_group_size=1)
        del updates
        for job in range(len(jobs)):
            work.check()
            tree = roots[destinations[job]]
            chunk = jobs[job].operator
            if tree.operator is None:
                tree.operator = chunk
            else:
                # One stored aggregate replaces two retained operators.
                tree.operator = compose(tree.operator, chunk)
                work.release(sum(block.nbytes for block in chunk))
            jobs[job].operator = None
    result = []
    for index, tree in sorted(roots.items()):
        error, changed = tree.refine(incoming[index][0], incoming[index+1][1], math.inf, work)
        result.append((index, error, changed, tree.operator))
    return result
