"""Bounded process scheduling of independent, already-defined wave intervals.

Trusted internal trees travel through execution-owned temporary files. Large
matrices never travel through Windows named-pipe messages. No user-provided
pickle is read. Results retain complete complex states and physical order.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

from temsim.physics.wave_process_packets import write_packet, read_packet, read_worker_result


_SHARED = None
_BLAS_LIMIT = None


def _initialize(evaluations, retained, abort, guard, settings):
    global _SHARED, _BLAS_LIMIT
    from threadpoolctl import threadpool_limits
    _BLAS_LIMIT = threadpool_limits(1)
    _SHARED = evaluations, retained, abort, guard, settings


class _ProcessWork:
    def __init__(self):
        self.evaluations, self.retained, self.abort, self.guard, self.settings = _SHARED
        self.cancelled = self.abort.is_set

    def check(self):
        if self.cancelled():
            raise InterruptedError("Occupied process refinement cancelled")

    def evaluate(self):
        self.check()
        with self.guard:
            if self.evaluations.value >= self.settings.maximum_evaluations:
                raise ValueError("Occupied axial refinement exhausted its evaluation budget; no source was published")
            self.evaluations.value += 1

    def retain(self, size):
        from temsim.physics.wave_execution import check_available_memory
        check_available_memory(4*size)
        with self.guard:
            self.retained.value += size
            if self.retained.value > self.settings.maximum_working_bytes:
                raise MemoryError("Occupied axial refinement exceeds its retained-operator memory budget")

    def release(self, size):
        with self.guard:
            if size < 0 or size > self.retained.value:
                raise RuntimeError("Invalid occupied-operator memory release")
            self.retained.value -= size


def _execute(packet, output_path):
    work = _ProcessWork()
    try:
        entries, budget = read_packet(packet)
        result = []
        for index, tree, left, right in entries:
            local_budget = budget[index] if isinstance(budget, dict) else budget
            error, changed = tree.refine(left, right, local_budget, work)
            from temsim.physics.occupied_tree_certificate import compress_tree
            compressed = compress_tree(tree, left, right, work)
            if compressed is not None:
                error = compressed
            result.append((index, tree, error, changed))
        response = (True, result, None)
    except BaseException as failure:
        work.abort.set()
        response = (False, failure, traceback.format_exc())
    return write_packet(output_path, response)


def balanced_interval_groups(roots, maximum_group_size=16):
    """Dispatch costly intervals first; do not alter numerical decisions."""
    if type(maximum_group_size) is not int or maximum_group_size <= 0:
        raise ValueError("Refinement group size must be a positive integer")
    if not roots:
        return ()
    costs = {}
    for index, tree in roots.items():
        if hasattr(tree, "divisions"):
            cost = 2*tree.divisions if tree.operator is None else 1
        else:
            partition = getattr(tree, "partition", None)
            cost = len(partition) if partition is not None else tree.statistics()["leaves"]
        costs[index] = max(1, int(cost))
    target = math.ceil(sum(costs.values())/math.ceil(len(roots)/maximum_group_size))
    singleton_cost = max(2*min(costs.values()), target/2)
    groups, current, cost = [], [], 0
    for index in sorted(roots, key=lambda key: (-costs[key], key)):
        if costs[index] > singleton_cost:
            if current:
                groups.append(tuple(current))
                current, cost = [], 0
            groups.append((index,))
            continue
        if current and (len(current) == maximum_group_size or cost+costs[index] > target):
            groups.append(tuple(current))
            current, cost = [], 0
        current.append(index)
        cost += costs[index]
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def refine_process_intervals(roots, incoming, budget, work, *, maximum_group_size=16):
    context = mp.get_context("spawn")
    evaluations = context.Value("q", work.evaluations, lock=False)
    retained = context.Value("q", work.retained_bytes, lock=False)
    abort, guard = context.Event(), context.Lock()
    groups = balanced_interval_groups(roots, maximum_group_size)
    from temsim.physics.wave_execution import check_available_memory
    results, errors, last_progress = {}, [], work.evaluations
    next_group, in_flight_bytes, ready = 0, 0, None
    # Executor exits first; cleanup cannot delete an active worker's packet.
    # Only this newly created directory belongs to the invocation.
    with TemporaryDirectory(prefix="temsim-wave-workers-") as directory, ProcessPoolExecutor(
            max_workers=work.settings.workers, mp_context=context,
            initializer=_initialize, initargs=(evaluations, retained, abort, guard, work.settings)) as executor:
        pending = {}
        try:
            while pending or next_group < len(groups):
                if work.cancelled():
                    abort.set()
                while (not abort.is_set() and next_group < len(groups)
                       and len(pending) < work.settings.workers):
                    if ready is None:
                        entries = [(i, roots[i], incoming[i][0], incoming[i+1][1]) for i in groups[next_group]]
                        ready = write_packet(Path(directory)/f"input-{next_group}.bin", (entries, budget))
                        del entries
                    size = ready[1]
                    if retained.value+4*(in_flight_bytes+size) > work.settings.maximum_working_bytes:
                        if pending:
                            break
                        raise MemoryError("Occupied process transfer exceeds the shared working-memory budget")
                    check_available_memory(4*size)
                    output = Path(directory)/f"output-{next_group}.bin"
                    future = executor.submit(_execute, ready, str(output))
                    pending[future] = (ready, output)
                    in_flight_bytes += size
                    next_group += 1
                    ready = None
                if not pending:
                    break
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    packet, output = pending.pop(future)
                    in_flight_bytes -= packet[1]
                    try:
                        for index, tree, error, changed in read_worker_result(future.result()):
                            adopt = getattr(roots[index], "adopt_worker_state", None)
                            if adopt is not None:
                                tree = adopt(tree)
                            roots[index] = tree
                            results[index] = (index, error, changed, tree.operator)
                    except BaseException as error:
                        errors.append(error)
                        abort.set()
                    finally:
                        Path(packet[0]).unlink(missing_ok=True)
                        output.unlink(missing_ok=True)
                current = evaluations.value
                if work.progress is not None and current-last_progress >= 64:
                    work.progress(current, work.settings.maximum_evaluations, "Refining occupied complex gun wave (processes)")
                    last_progress = current
                if abort.is_set():
                    for future in pending:
                        future.cancel()
        finally:
            abort.set()
            work.evaluations, work.retained_bytes = evaluations.value, retained.value
    if errors:
        from concurrent.futures import CancelledError
        raise next((error for error in errors if not isinstance(error, (InterruptedError, CancelledError))), errors[0])
    if work.cancelled():
        raise InterruptedError("Occupied process refinement cancelled")
    if sorted(results) != sorted(roots):
        raise RuntimeError("Incomplete occupied process result; no wave was published")
    return [results[index] for index in sorted(results)]
