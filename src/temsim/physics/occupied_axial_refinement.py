"""Source-specific two-way axial refinement with repeated boundary solves.

Every channel and reflected port is retained. Only the error estimator is
weighted by the executed source. An operator refined this way is not certified
for another source. Its enclosing checkpoint must bind all upstream inputs.
Local indicators are not rigorous global bounds; full complex boundary-state
and independent numerical refinements remain required.
"""
from dataclasses import dataclass, replace
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np
from scipy.linalg import solve

from temsim.physics.magnus_scattering import cf4_slab, GAUSS_FRACTIONS
from temsim.physics.scattering_load import compose, outgoing_load
from temsim.physics.liouville_wave import PhysicalCoordinateLoad
from temsim.physics.radial_mask_ledger import physical_to_chart


@dataclass(frozen=True)
class OccupiedAxialRefinement:
    enabled: bool = False
    tolerance: float = 1e-3
    maximum_rounds: int = 12
    maximum_evaluations: int = 200_000
    maximum_depth: int = 18
    maximum_working_bytes: int = 16*1024**3
    workers: int = 1
    executor: str = "thread"
    refine_pilot_charts: bool = False
    integrator: str = "cf4"
    error_budget_allocation: str = "uniform"
    strategy: str = "local_sum"
    initial_mesh: tuple = ()
    initial_mesh_energy_ev: float | None = None

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("Occupied axial refinement must be explicitly enabled or disabled")
        if type(self.refine_pilot_charts) is not bool:
            raise ValueError("Pilot-chart refinement must be explicitly enabled or disabled")
        if self.executor not in ("thread", "process"):
            raise ValueError("Occupied axial executor must be thread or process")
        if self.integrator not in ("cf4", "cf6"):
            raise ValueError("Occupied axial integrator must be cf4 or cf6")
        if self.error_budget_allocation not in ("uniform", "initial_indicator"):
            raise ValueError("Unknown occupied axial error-budget allocation")
        if self.strategy not in ("local_sum", "global_embedded", "spatial_embedded"):
            raise ValueError("Unknown occupied axial refinement strategy")
        from temsim.physics.axial_mesh_seed import validate_mesh_seed
        validate_mesh_seed(self.initial_mesh, maximum_depth=self.maximum_depth)
        if self.initial_mesh and (not self.enabled or self.strategy != "spatial_embedded"):
            raise ValueError("Numerical mesh seeds require enabled spatial embedded refinement")
        if self.initial_mesh_energy_ev is not None:
            energy = self.initial_mesh_energy_ev
            if (not self.initial_mesh or isinstance(energy, bool) or not isinstance(energy, (int, float))
                    or not math.isfinite(energy) or energy <= 0):
                raise ValueError("A scoped mesh seed needs a finite positive emission energy")
        if (isinstance(self.tolerance, bool) or not math.isfinite(self.tolerance)
                or not 0 < self.tolerance < .1):
            raise ValueError("Occupied axial tolerance must be finite and in (0, 0.1)")
        for name, low, high in (("maximum_rounds", 2, 100), ("maximum_evaluations", 2, 10_000_000),
                                ("maximum_depth", 1, 30), ("workers", 1, 32)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid occupied axial {name}")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("Occupied axial memory budget must be a positive integer")
        return self

    def for_energy(self, energy_ev):
        """A mesh hint applies only to its specified energy; every mode executes."""
        if self.initial_mesh_energy_ev is None or energy_ev == self.initial_mesh_energy_ev:
            return self
        return replace(self, initial_mesh=(), initial_mesh_energy_ev=None)

    def applies_to_chart(self, index, last):
        if type(index) is not int or type(last) is not int or not 0 <= index <= last:
            raise ValueError("Wave chart iteration must belong to its requested sequence")
        # Pilots only select numerical coordinates. The final physical BVP
        # is always refined when enabled, including a zero-pilot calculation.
        return self.enabled and (index == last or self.refine_pilot_charts)


def apply_two_port(blocks, left, right):
    return np.r_[blocks[0]@left+blocks[1]@right, blocks[2]@left+blocks[3]@right]


def internal_incoming(first, second, left, right):
    """Solve the complete reflected internal interface, not forward-only input."""
    forward = solve(np.eye(len(left))-first[3]@second[0],
                    first[2]@left+first[3]@second[1]@right, check_finite=False)
    backward = second[0]@forward+second[1]@right
    return backward, forward


def evaluate_interval(sample, width, reference_k, start, end, work):
    """One budgeted complete covariant step; never changes the physical chart."""
    work.evaluate()
    if work.settings.integrator == "cf6":
        from temsim.physics.magnus_sixth import quadrature, cf6_slab
        values = [sample(start+(end-start)*fraction) for fraction in quadrature()[0]]
        return cf6_slab(values, width*(end-start), reference_k, cancelled=work.cancelled)[0]
    values = [sample(start+(end-start)*fraction) for fraction in GAUSS_FRACTIONS]
    return cf4_slab(*values, width*(end-start), reference_k, cancelled=work.cancelled)[0]


class RefinableSlab:
    """A bounded refinement tree for one continuous, already executed chart."""

    def __init__(self, sample, width, reference_k, operator, *, start=0., end=1., depth=0):
        self.sample, self.width, self.kappa = sample, width, reference_k
        self.start, self.end, self.depth = start, end, depth
        self.operator, self.children, self.expanded = operator, None, False
        self.error_operator = None
        self.certificate, self.partition = None, None

    def _indicator(self, difference, left, right):
        value = float(np.linalg.norm(apply_two_port(difference, left, right))*math.sqrt(self.kappa))
        if not math.isfinite(value):
            raise ValueError("Nonfinite occupied axial error indicator")
        return value

    def statistics(self):
        if self.partition is not None:
            return {"nodes": 2*len(self.partition)-1, "leaves": len(self.partition),
                    "maximum_depth": max(item[2] for item in self.partition), "compressed": True}
        nodes, leaves, depth = 1, int(self.children is None), self.depth
        for child in self.children or ():
            item = child.statistics()
            nodes += item["nodes"]
            leaves += item["leaves"]
            depth = max(depth, item["maximum_depth"])
        return {"nodes": nodes, "leaves": leaves, "maximum_depth": depth}

    def refine(self, left, right, tolerance, work):
        work.check()
        if self.certificate is not None:
            from temsim.physics.occupied_tree_certificate import certificate_indicator, restore_partition
            indicator = certificate_indicator(self, left, right)
            if indicator <= tolerance:
                return indicator, False
            restore_partition(self, work)
        if self.error_operator is not None:
            # Retain the full complex difference, not its action on an old
            # source. A newly reflected/occupied channel is checked afresh.
            indicator = self._indicator(self.error_operator, left, right)
            if indicator <= tolerance:
                return indicator, False
            work.release(sum(block.nbytes for block in self.error_operator))
            self.error_operator = None
        if self.children is None:
            midpoint = (self.start+self.end)/2
            children = []
            for a, b in ((self.start, midpoint), (midpoint, self.end)):
                operator = evaluate_interval(self.sample, self.width, self.kappa, a, b, work)
                work.retain(sum(block.nbytes for block in operator))
                children.append(RefinableSlab(self.sample, self.width, self.kappa, operator,
                                              start=a, end=b, depth=self.depth+1))
            self.children = tuple(children)
        first, second = self.children
        if self.expanded:
            back, forward = internal_incoming(first.operator, second.operator, left, right)
            a, changed_a = first.refine(left, back, tolerance/2, work)
            b, changed_b = second.refine(forward, right, tolerance/2, work)
            self.operator = compose(first.operator, second.operator)
            return a+b, changed_a or changed_b
        fine = compose(first.operator, second.operator)
        difference = tuple(coarse-refined for coarse, refined in zip(self.operator, fine))
        # The driven tip reservoir has unit incoming flux. sqrt(kappa) puts
        # both complex-port errors in that common source-current amplitude unit.
        indicator = self._indicator(difference, left, right)
        if indicator <= tolerance:
            # Two child scattering operators existed only for this comparison.
            # One full difference matrix gives exactly the same future test,
            # while the executed coarse operator and all its channels remain.
            work.retain(sum(block.nbytes for block in difference))
            self.error_operator = difference
            work.release(sum(block.nbytes for child in self.children for block in child.operator))
            self.children = None
            return indicator, False
        if self.depth >= work.settings.maximum_depth:
            raise ValueError(f"Occupied axial error {indicator:.6g} exceeds {tolerance:.6g} "
                             f"at depth {self.depth}; no unresolved source was published")
        back, forward = internal_incoming(first.operator, second.operator, left, right)
        error_first, _ = first.refine(left, back, tolerance/2, work)
        error_second, _ = second.refine(forward, right, tolerance/2, work)
        self.operator = compose(first.operator, second.operator)
        self.expanded = True
        return error_first+error_second, True


class _Work:
    def __init__(self, settings, cancelled, progress):
        self.settings, self.cancelled, self.progress = settings, cancelled, progress
        self.evaluations = 0
        self.retained_bytes = 0
        self.lock = Lock()
        self.aborted = Event()
        self.failure = None

    def check(self):
        if self.aborted.is_set() or self.cancelled():
            raise InterruptedError("Occupied axial refinement cancelled")

    def abort(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = error
            self.aborted.set()

    def evaluate(self):
        self.check()
        with self.lock:
            if self.evaluations >= self.settings.maximum_evaluations:
                raise ValueError("Occupied axial refinement exhausted its evaluation budget; no source was published")
            self.evaluations += 1
            evaluations = self.evaluations
        if self.progress is not None and evaluations % 64 == 0:
            self.progress(evaluations, self.settings.maximum_evaluations,
                          "Refining occupied complex gun wave")

    def retain(self, size):
        from temsim.physics.wave_execution import check_available_memory
        check_available_memory(4*size)
        with self.lock:
            self.retained_bytes += size
            if self.retained_bytes > self.settings.maximum_working_bytes:
                raise MemoryError("Occupied axial refinement exceeds its retained-operator memory budget")

    def release(self, size):
        with self.lock:
            if size < 0 or size > self.retained_bytes:
                raise RuntimeError("Invalid occupied-operator memory release")
            self.retained_bytes -= size


def _refine_intervals(roots, incoming, budget, work):
    """Only scheduling is parallel; reflected inputs belong to one solved BVP.

    A tree is owned by exactly one worker. All results are collected in physical
    order before rebuilding the global reflected load. Budgets are shared,
    not multiplied by the worker count. Any failed interval aborts the round.
    """
    def refine(item):
        index, tree = item
        try:
            local_budget = budget[index] if isinstance(budget, dict) else budget
            error, changed = tree.refine(incoming[index][0], incoming[index+1][1], local_budget, work)
            from temsim.physics.occupied_tree_certificate import compress_tree
            compressed = compress_tree(tree, incoming[index][0], incoming[index+1][1], work)
            if compressed is not None:
                error = compressed
            return index, error, changed, tree.operator
        except BaseException as failure:
            work.abort(failure)
            raise
    if work.settings.workers == 1:
        return [refine(item) for item in roots.items()]
    if work.settings.executor == "process":
        from temsim.physics.occupied_wave_processes import refine_process_intervals
        # An unchanged spatial leaf needs a NEW occupied-port error readout,
        # not another matrix propagation. Keep those complete matrices here
        # instead of copying them through worker files on every mesh round.
        local, pending = [], {}
        for index, tree in roots.items():
            if getattr(tree, "can_recheck_without_propagation", False):
                local.append(refine((index, tree)))
            else:
                pending[index] = tree
        if pending:
            local.extend(refine_process_intervals(pending, incoming, budget, work))
            roots.update(pending)
        return sorted(local, key=lambda row: row[0])
    try:
        with ThreadPoolExecutor(max_workers=work.settings.workers,
                                thread_name_prefix="occupied-wave") as executor:
            return list(executor.map(refine, roots.items()))
    except BaseException:
        if work.failure is not None:
            raise work.failure
        raise


def allocate_error_budget(indicators, tolerance, order):
    """Redistribute a FIXED total budget; never relax the summed error target.

    If local error e is order p and equal-size substeps cost n, minimizing
    sum_i (e_i/t_i)**(1/p) at fixed sum_i t_i gives t_i proportional to
    e_i**(1/(p+1)). This is only a scheduling heuristic. A 5% uniform reserve
    prevents a presently empty channel from receiving zero budget; every
    actual incoming port is still rechecked after each full boundary solve.
    """
    keys, values = list(indicators), np.array(list(indicators.values()), float)
    if (not len(values) or np.any(~np.isfinite(values)) or np.any(values < 0)
            or not math.isfinite(tolerance) or tolerance <= 0 or order not in (4, 6)):
        raise ValueError("Finite nonnegative local indicators and a positive total budget are required")
    maximum = values.max()
    weights = np.ones_like(values) if maximum == 0 else (values/maximum)**(1/(order+1))
    fractions = .05/len(values)+.95*weights/weights.sum()
    # Round downward to keep the budget sum below the same explicit target.
    budgets = fractions*(tolerance*(1-8*np.finfo(float).eps))
    return dict(zip(keys, map(float, budgets)))


def refine_joint_mode(plan, boundary_solver, settings, *, cancelled=lambda: False, progress_callback=None):
    """Recompute the full reflected load and tip BVP after every mesh update.

    plan contains executed operators, continuous numerical samplers and fixed
    Liouville boundary coordinates. boundary_solver(load) returns the genuine
    driven-tip solution tuple (near_field, total_left_trace, flux_record).
    No downstream source is configured, frozen or normalized here.
    """
    settings.validate()
    if not settings.enabled:
        raise ValueError("Occupied joint refinement must be explicitly requested")
    if settings.strategy == "spatial_embedded":
        from temsim.physics.spatial_embedded_refinement import refine_spatial_mode
        return refine_spatial_mode(plan, boundary_solver, settings,
            cancelled=cancelled, progress_callback=progress_callback)
    if settings.strategy == "global_embedded":
        from temsim.physics.global_embedded_refinement import refine_global_mode
        return refine_global_mode(plan, boundary_solver, settings,
            cancelled=cancelled, progress_callback=progress_callback)
    operators = list(plan["operators"])
    kappa, alpha, log_derivative = plan["kappa"], plan["alpha"], plan["log_derivative"]
    roots = {index: RefinableSlab(sample, width, kappa, operators[index])
             for index, (sample, width) in plan["samplers"].items()}
    if not roots:
        raise ValueError("The joint axial refinement plan has no physical propagation intervals")
    work = _Work(settings, cancelled, progress_callback)
    history, previous, stable = [], None, 0
    local_budget = settings.tolerance/len(roots)
    allocation_record = {"method": "uniform", "total_tolerance": settings.tolerance}
    for iteration in range(settings.maximum_rounds):
        work.check()
        transformed = outgoing_load(operators, plan["exit_q"], kappa,
                                    cancelled=cancelled, progress_callback=progress_callback)
        load = PhysicalCoordinateLoad(transformed, alpha, log_derivative)
        result = boundary_solver(load)
        fields, derivatives = load.propagate(result[1])
        incoming = [physical_to_chart(f, d, a, l, kappa)
                    for f, d, a, l in zip(fields, derivatives, alpha, log_derivative)]
        change = None
        if previous is not None:
            # Same physical planes and transverse charts, absolute complex phase.
            differences = []
            for current, old in zip((fields, derivatives), previous):
                denominator = np.maximum(np.linalg.norm(current, axis=1), np.linalg.norm(old, axis=1))
                denominator = np.maximum(denominator, np.finfo(float).tiny)
                differences.append(float(np.max(np.linalg.norm(current-old, axis=1)/denominator)))
            change = max(differences)
        indicator, changed = 0., 0
        try:
            if iteration == 0 and settings.error_budget_allocation == "initial_indicator":
                # Probe complete coarse/fine differences only. These infinite-
                # budget comparisons cannot return a source: the strict pass
                # below and all global boundary stability checks still follow.
                probes = _refine_intervals(roots, incoming, math.inf, work)
                local_budget = allocate_error_budget({index: error for index, error, _, _ in probes},
                    settings.tolerance, 4 if settings.integrator == "cf4" else 6)
                allocation_record = {"method": settings.error_budget_allocation,
                    "total_tolerance": settings.tolerance, "assigned_total": sum(local_budget.values()),
                    "minimum_interval_budget": min(local_budget.values()),
                    "maximum_interval_budget": max(local_budget.values()),
                    "initial_summed_indicator": sum(row[1] for row in probes),
                    "scope": "Cost allocation only; every strict local and global check remains required"}
            updates = _refine_intervals(roots, incoming, local_budget, work)
        except Exception as failure:
            ranked = []
            for index, tree in roots.items():
                ranked.append({"operator_index": index, **tree.statistics(),
                    "root_z_interval_nm": plan.get("interval_z_nm", {}).get(index)})
            ranked.sort(key=lambda row: (row["nodes"], row["maximum_depth"]), reverse=True)
            failure.refinement_diagnostic = {"scope": "FAILED_REFINEMENT_NOT_A_SOURCE_OR_CONVERGENCE_CERTIFICATE",
                "round": iteration, "completed_rounds": history, "evaluations": work.evaluations,
                "retained_operator_bytes": work.retained_bytes, "numerics": vars(settings),
                "error_budget_allocation": allocation_record,
                "largest_returned_trees": ranked[:32],
                "tree_scope": "Only coordinator-owned or completely returned trees; failed process-local trees may be absent"}
            raise
        for index, error, updated, operator in updates:
            indicator += error
            changed += int(updated)
            operators[index] = operator
        row = {"round": iteration, "step_evaluations": work.evaluations,
               "summed_occupied_indicator": indicator, "changed_intervals": changed,
               "maximum_complex_boundary_change": change,
               "exit_net_flux": float(np.vdot(fields[-1], derivatives[-1]).imag)}
        history.append(row)
        if progress_callback is not None:
            progress_callback(iteration+1, settings.maximum_rounds,
                f"Joint axial verification {iteration+1}: {changed} intervals updated; indicator {indicator:.4g}")
        # Two repeated solved-state checks prevent a one-pass pilot estimate
        # from admitting a state whose reflected load changed significantly.
        stable = stable+1 if changed == 0 and change is not None and change <= settings.tolerance else 0
        if stable >= 2 and indicator <= settings.tolerance:
            return load, result, {"method": f"occupied-two-port-{settings.integrator}-joint-v1", "rounds": history,
                "tolerance": settings.tolerance, "evaluations": work.evaluations, "workers": settings.workers,
                "executor": settings.executor,
                "integrator": settings.integrator,
                "error_budget_allocation": allocation_record,
                "evaluation_unit": "Complete slab; CF4 uses two exponentials, CF6 uses five",
                "scope": "Source-specific axial indicator and re-solved boundary stability; radial/full-chain convergence still required"}
        previous = (fields, derivatives)
        del load, transformed
    raise ValueError("Occupied axial refinement did not stabilize within its round budget; no source was published")
