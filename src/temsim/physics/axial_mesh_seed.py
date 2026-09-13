"""Numerical subdivision hints, never cached fields or replacement sources.

Every hinted leaf is re-executed from the current physical sampler. The full
reflected boundary problem and two uniform complex comparisons still run.
Only dyadic action-coordinate intervals are accepted; no phase/current data
are read. The consumed tuple belongs in the numerical checkpoint identity.
"""
import json
import math
from pathlib import Path

import numpy as np


def validate_mesh_seed(seed, *, maximum_depth=30):
    if not isinstance(seed, tuple) or len(seed) > 1_000_000:
        raise ValueError("Axial mesh seed must be a bounded immutable tuple")
    groups = {}
    previous = None
    for row in seed:
        if not isinstance(row, tuple) or len(row) != 5:
            raise ValueError("Mesh seed rows are (physical Z start, Z end, action start, end, depth)")
        z0, z1, start, end, depth = row
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in row[:4]) or not 0 <= z0 < z1 or not 0 <= start < end <= 1
                or type(depth) is not int or not 0 <= depth < maximum_depth):
            raise ValueError("Mesh seed has invalid physical planes, fractions or refinement depth")
        scale = 2**depth
        if (end-start)*scale != 1 or start*scale != int(start*scale):
            raise ValueError("Mesh seed leaves must be exact dyadic intervals")
        order = z0, z1, start
        if previous is not None and order <= previous:
            raise ValueError("Mesh seed must be ordered without duplicated leaves")
        previous = order
        groups.setdefault((z0, z1), []).append((start, end, depth))
    for leaves in groups.values():
        if leaves[0][0] != 0 or leaves[-1][1] != 1 or any(a[1] != b[0] for a, b in zip(leaves, leaves[1:])):
            raise ValueError("Mesh seed cannot leave a gap or overlap in a physical interval")
    return groups


def read_mesh_seed(path):
    """Read bounded numeric-only hints or a failed spatial diagnostic report."""
    if path is None:
        return ()
    path = Path(path)
    if path.stat().st_size > 64*1024**2:
        raise ValueError("Mesh seed JSON exceeds the 64 MiB input budget")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") == "wave-axial-mesh-seed-v1":
        seed = tuple(tuple(row) for row in data["mesh_seed"])
    else:
        diagnostic = data.get("refinement_diagnostic", data)
        if diagnostic.get("scope") != "FAILED_SPATIAL_MESH_COMPARISON_NOT_SOURCE":
            raise ValueError("Expected numerical mesh hints or a failed spatial-mesh diagnostic")
        seed = tuple((*row["z_interval_nm"], row["fraction_start"], row["fraction_end"], row["depth"])
                     for row in diagnostic["coarse_mesh"])
    validate_mesh_seed(seed)
    return seed


def export_mesh_seed(mesh, plan, *, fine=False):
    from temsim.physics.spatial_embedded_mesh import SpatialLeaf
    if not plan.get("interval_z_nm"):
        return ()  # Nonphysical mathematical fixtures need not invent Z units.
    result = []
    for leaf in mesh:
        if not isinstance(leaf, SpatialLeaf):
            continue
        z = plan["interval_z_nm"][leaf.physical_index]
        middle = (leaf.start+leaf.end)/2
        parts = ((leaf.start, middle, leaf.depth+1), (middle, leaf.end, leaf.depth+1)) if fine else (
            (leaf.start, leaf.end, leaf.depth),)
        result.extend((*z, *part) for part in parts)
    return tuple(result)


class _SeedExecution:
    """Internal worker task; only the current sampler produces its matrix."""
    expanded = False
    certificate = None
    divisions = 1

    def __init__(self, leaf):
        self.leaf, self.operator = leaf, None

    def refine(self, left, right, unused_budget, work):
        from temsim.physics.occupied_axial_refinement import evaluate_interval
        from temsim.physics.spatial_embedded_mesh import size
        leaf = self.leaf
        self.operator = evaluate_interval(leaf.sample, leaf.width, leaf.kappa, leaf.start, leaf.end, work)
        work.retain(size(self.operator))
        return 0., False  # Scheduling result only; no error acceptance uses this.

    def adopt_worker_state(self, other):
        if not isinstance(other, _SeedExecution) or other.operator is None:
            raise RuntimeError("Mesh worker did not execute its complete operator")
        names = ("physical_index", "width", "kappa", "start", "end", "depth")
        if any(getattr(self.leaf, key) != getattr(other.leaf, key) for key in names):
            raise RuntimeError("Mesh worker changed its requested numerical interval")
        self.operator = other.operator
        return self


def execute_seed_mesh(plan, work):
    from temsim.physics.spatial_embedded_mesh import SpatialLeaf, initial_mesh
    from temsim.physics.occupied_axial_refinement import _refine_intervals
    seed = work.settings.initial_mesh
    if not seed:
        return initial_mesh(plan)
    groups = validate_mesh_seed(seed, maximum_depth=work.settings.maximum_depth)
    intervals = plan.get("interval_z_nm", {})
    available = {tuple(z) for index, z in intervals.items() if index in plan["samplers"]}
    if groups.keys()-available:
        raise ValueError("Mesh seed physical intervals do not match the current gun; no field was reused")
    mesh, tasks = [], {}
    for item in initial_mesh(plan):
        key = tuple(intervals[item.physical_index]) if isinstance(item, SpatialLeaf) else None
        parts = groups.get(key)
        if parts is None or parts == [(0., 1., 0)]:
            mesh.append(item)
            continue
        for start, end, depth in parts:
            leaf = SpatialLeaf(item.physical_index, item.sample, item.width, item.kappa, None,
                               start=start, end=end, depth=depth, owns_coarse=True)
            tasks[len(mesh)] = _SeedExecution(leaf)
            mesh.append(leaf)
    # Dummy incoming ports are never read by seed execution. Reuse the
    # existing bounded process scheduler without pretending to have a wave.
    zero = np.zeros(len(plan["exit_q"]), complex)
    _refine_intervals(tasks, [(zero, zero)]*(len(mesh)+1), math.inf, work)
    for index, task in tasks.items():
        mesh[index].coarse = task.operator
    return mesh
