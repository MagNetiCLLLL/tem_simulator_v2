"""Development global-mesh comparison for the complete reflected tip BVP.

Each candidate halves EVERY retained propagation substep and re-solves the
physical reservoir. All original complex fields AND normal derivatives are
compared, without phase fitting or Richardson extrapolation of a field.
Local full-port differences guide refinement only. Two successive uniform
mesh comparisons must pass; the L1 local-sum strategy remains independent.
Neither strategy certifies radial/domain/field or full-product convergence.
"""
import math

import numpy as np

from temsim.physics.occupied_axial_refinement import (
    _Work, _refine_intervals, evaluate_interval, apply_two_port)
from temsim.physics.scattering_load import compose
from temsim.physics.liouville_wave import PhysicalCoordinateLoad
from temsim.physics.radial_mask_ledger import physical_to_chart


class EmbeddedInterval:
    """Keep two complete root operators, not all their intermediate matrices."""
    expanded = False  # The local-sum tree compressor does not own this object.
    certificate = None

    def __init__(self, sample, width, kappa, coarse):
        self.sample, self.width, self.kappa = sample, width, kappa
        self.coarse, self.operator = coarse, None
        self.divisions, self.owns_coarse = 1, False

    def statistics(self):
        return {"nodes": self.divisions, "leaves": self.divisions,
                "maximum_depth": self.divisions.bit_length()-1, "compressed": True}

    def refine(self, left, right, unused_budget, work):
        work.check()
        if self.operator is None:
            divisions = self.divisions*2
            if divisions > 2**work.settings.maximum_depth:
                raise ValueError("Global embedded refinement reached its depth budget; no source was published")
            from temsim.physics.embedded_wave_chunks import evaluate_embedded_interval
            result = evaluate_embedded_interval(self.sample, self.width, self.kappa, divisions, work)
            work.retain(sum(block.nbytes for block in result))
            self.operator = result
        difference = tuple(a-b for a, b in zip(self.coarse, self.operator))
        indicator = float(np.linalg.norm(apply_two_port(difference, left, right))*math.sqrt(self.kappa))
        if not math.isfinite(indicator):
            raise ValueError("Nonfinite global-mesh local indicator")
        return indicator, False

    def promote(self, work):
        if self.operator is None:
            raise ValueError("Only an executed finer interval can become the next mesh")
        if self.owns_coarse:
            work.release(sum(block.nbytes for block in self.coarse))
        self.coarse, self.operator = self.operator, None
        self.divisions *= 2
        self.owns_coarse = True


def boundary_difference_details(first, second):
    rows = []
    for component, (a, b) in zip(("complex_field", "covariant_derivative"), zip(first, second)):
        scale = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
        scale = np.maximum(scale, np.finfo(float).tiny)
        absolute = np.linalg.norm(a-b, axis=1)
        relative = absolute/scale
        index = int(np.argmax(relative))
        # Diagnose the kind of discrepancy, never phase-fit the acceptance
        # field or replace the unfitted difference above. Unit-normalised
        # overlap is evaluated only for these read-only diagnostic scalars.
        norm_a, norm_b = float(np.linalg.norm(a[index])), float(np.linalg.norm(b[index]))
        overlap = None
        if norm_a and norm_b:
            overlap = complex(np.vdot(a[index]/norm_a, b[index]/norm_b))
        rows.append({"component": component, "boundary_index": index,
            "relative_difference": float(relative[index]), "absolute_difference": float(absolute[index]),
            "comparison_scale": float(scale[index]),
            "norm_relative_change_diagnostic": abs(norm_a-norm_b)/float(scale[index]),
            "overlap_phase_rad_diagnostic": None if overlap is None or overlap == 0 else float(np.angle(overlap)),
            "normalised_overlap_magnitude_diagnostic": None if overlap is None else float(abs(overlap)),
            "diagnostic_scope": "Overlap and norms describe the error only; acceptance uses the original unfitted complex difference"})
    if not all(math.isfinite(row["relative_difference"]) for row in rows):
        raise ValueError("Nonfinite full complex-boundary comparison")
    return {"maximum_relative_difference": max(row["relative_difference"] for row in rows),
            "worst_boundaries": rows}


def boundary_difference(first, second):
    return boundary_difference_details(first, second)["maximum_relative_difference"]


def refine_global_mode(plan, boundary_solver, settings, *, cancelled, progress_callback):
    kappa = plan["kappa"]
    roots = {i: EmbeddedInterval(sample, width, kappa, plan["operators"][i])
             for i, (sample, width) in plan["samplers"].items()}
    if not roots:
        raise ValueError("Global mesh comparison requires physical propagation intervals")
    work = _Work(settings, cancelled, progress_callback)
    from temsim.physics.incremental_scattering_load import IncrementalOutgoingLoad
    load_cache = IncrementalOutgoingLoad(settings.maximum_working_bytes)
    def solve(operators):
        work.check()
        transformed = load_cache.build(operators, plan["exit_q"], kappa, cancelled=cancelled,
                                       progress_callback=progress_callback)
        load = PhysicalCoordinateLoad(transformed, plan["alpha"], plan["log_derivative"])
        result = boundary_solver(load)
        states = load.propagate(result[1])
        return load, result, states
    coarse_operators = list(plan["operators"])
    coarse_load, coarse_result, coarse_states = solve(coarse_operators)
    del coarse_load, coarse_result
    history, successive = [], 0
    try:
        for iteration in range(settings.maximum_rounds):
            incoming = [physical_to_chart(f, d, a, l, kappa) for f, d, a, l in
                zip(*coarse_states, plan["alpha"], plan["log_derivative"])]
            if settings.executor == "process" and settings.workers > 1:
                from temsim.physics.embedded_wave_chunks import refine_embedded_process_intervals
                updates = refine_embedded_process_intervals(roots, incoming, work)
            else:
                updates = _refine_intervals(roots, incoming, math.inf, work)
            fine_operators = list(coarse_operators)
            indicators = {}
            for index, error, _, operator in updates:
                fine_operators[index] = operator
                indicators[index] = error
            fine_load, fine_result, fine_states = solve(fine_operators)
            comparison = boundary_difference_details(coarse_states, fine_states)
            change = comparison["maximum_relative_difference"]
            maximum_local = max(indicators.values())
            # No local difference is allowed to hide a large global defect by
            # cancellation. The summed indicator is recorded, NOT interpreted
            # as meeting the separate local_sum strategy's total L1 target.
            passed = change <= settings.tolerance and maximum_local <= settings.tolerance
            successive = successive+1 if passed else 0
            row = {"round": iteration, "step_evaluations": work.evaluations,
                "maximum_complex_boundary_change": change, "maximum_local_indicator": maximum_local,
                "complex_comparison": comparison,
                "summed_local_indicator_diagnostic_only": sum(indicators.values()),
                "coarse_substeps": sum(tree.divisions for tree in roots.values()),
                "fine_substeps": 2*sum(tree.divisions for tree in roots.values()),
                "successive_uniform_checks": successive,
                "exit_net_flux": float(np.vdot(fine_states[0][-1], fine_states[1][-1]).imag)}
            history.append(row)
            from temsim.physics.refinement_progress import emit_refinement_record
            emit_refinement_record(progress_callback, "global_embedded", row)
            if progress_callback:
                progress_callback(iteration+1, settings.maximum_rounds,
                    f"Global complex gun check {iteration+1}: change {change:.4g}; local {maximum_local:.4g}")
            if successive >= 2:
                return fine_load, fine_result, {"method": "global-embedded-two-port-v2", "rounds": history,
                    "composition_order": "128-substep chunks in physical order; identical serial/process grouping",
                    "tolerance": settings.tolerance, "evaluations": work.evaluations,
                    "workers": settings.workers, "executor": settings.executor, "integrator": settings.integrator,
                    "scope": "Two successive full-mesh complex comparisons; not an L1 local-sum or full-source certificate"}
            if passed:
                selected = set(roots)  # The next check uses a genuinely finer mesh EVERYWHERE.
                coarse_states = fine_states
            else:
                # Bulk marking is a cost heuristic, not an acceptance test.
                selected, cumulative = set(), 0.
                target = .5*sum(indicators.values())
                for index in sorted(indicators, key=lambda key: (-indicators[key], key)):
                    selected.add(index)
                    cumulative += indicators[index]
                    if cumulative >= target:
                        break
            for index in selected:
                roots[index].promote(work)
                coarse_operators[index] = roots[index].coarse
            del fine_load, fine_result, fine_operators, fine_states, updates
            if not passed:
                coarse_load, coarse_result, coarse_states = solve(coarse_operators)
                del coarse_load, coarse_result
        raise ValueError("Global embedded refinement exhausted its round budget; no source was published")
    except Exception as failure:
        failure.refinement_diagnostic = {"scope": "FAILED_GLOBAL_MESH_COMPARISON_NOT_SOURCE",
            "failed_constant_slab": getattr(failure, "slab_diagnostic", None),
            "completed_rounds": history, "evaluations": work.evaluations,
            "retained_operator_bytes": work.retained_bytes, "numerics": vars(settings),
            "coarse_mesh": [{"operator_index": i, "substeps": tree.divisions,
                              "z_interval_nm": plan.get("interval_z_nm", {}).get(i)}
                            for i, tree in roots.items()]}
        raise
