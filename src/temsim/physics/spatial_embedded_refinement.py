"""Local mesh adaptation with unchanged full-boundary acceptance checks.

The bulk indicator selects numerical leaves, not original long intervals.
Every comparison halves ALL current leaves and re-solves the driven tip
against the full reflected load. Two consecutive uniform comparisons must
pass, including complex derivatives; no phase fitting or extrapolation.
"""
import math

import numpy as np

from temsim.physics.global_embedded_refinement import boundary_difference_details
from temsim.physics.incremental_scattering_load import IncrementalOutgoingLoad
from temsim.physics.liouville_wave import PhysicalCoordinateLoad
from temsim.physics.occupied_axial_refinement import _Work, _refine_intervals
from temsim.physics.spatial_embedded_mesh import (
    SpatialLeaf, PhysicalBoundarySubset, initial_mesh, operators_and_boundaries)


def refine_spatial_mode(plan, boundary_solver, settings, *, cancelled, progress_callback):
    from temsim.physics.axial_mesh_seed import execute_seed_mesh, export_mesh_seed
    work = _Work(settings, cancelled, progress_callback)
    mesh = execute_seed_mesh(plan, work)
    if not any(isinstance(item, SpatialLeaf) for item in mesh):
        raise ValueError("Spatial mesh comparison requires physical propagation intervals")
    cache = IncrementalOutgoingLoad(settings.maximum_working_bytes)
    alpha, rate, kappa = plan["alpha"], plan["log_derivative"], plan["kappa"]

    def solve(fine):
        operators, indices, matched = operators_and_boundaries(mesh, fine=fine)
        raw = cache.build(operators, plan["exit_q"], kappa, cancelled=cancelled,
                          progress_callback=progress_callback)
        load = PhysicalCoordinateLoad(PhysicalBoundarySubset(raw, indices), alpha, rate)
        result = boundary_solver(load)
        raw_states = raw.propagate(np.asarray(result[1])*math.sqrt(alpha[0]))
        field, derivative = (a[np.asarray(indices)] for a in raw_states)
        amplitude = alpha**-.5
        physical = (amplitude[:, None]*field,
                    amplitude[:, None]*(alpha[:, None]*derivative+rate[:, None]*field))
        return load, result, raw_states, physical, matched

    history, successive = [], 0
    try:
        coarse_load, coarse_result, coarse, physical, _ = solve(False)
        del coarse_load, coarse_result
        for iteration in range(settings.maximum_rounds):
            work.check()
            ports = list(zip((coarse[0]+coarse[1]/(1j*kappa))/2,
                             (coarse[0]-coarse[1]/(1j*kappa))/2))
            leaves = {i: leaf for i, leaf in enumerate(mesh) if isinstance(leaf, SpatialLeaf)}
            updates = _refine_intervals(leaves, ports, math.inf, work)
            for index, leaf in leaves.items():
                mesh[index] = leaf  # Process workers return complete private copies.
            fine_load, fine_result, fine, fine_physical, matched = solve(True)
            physical_comparison = boundary_difference_details(physical, fine_physical)
            chart_comparison = boundary_difference_details(coarse,
                tuple(a[np.asarray(matched)] for a in fine))
            change = max(physical_comparison["maximum_relative_difference"],
                         chart_comparison["maximum_relative_difference"])
            indicators = {index: error for index, error, _, _ in updates}
            maximum_local = max(indicators.values())
            passed = change <= settings.tolerance and maximum_local <= settings.tolerance
            successive = successive+1 if passed else 0
            row = {"round": iteration, "step_evaluations": work.evaluations,
                "maximum_complex_boundary_change": change,
                "maximum_local_indicator": maximum_local,
                "physical_complex_comparison": physical_comparison,
                "numerical_boundary_comparison": chart_comparison,
                "summed_local_indicator_diagnostic_only": sum(indicators.values()),
                "coarse_substeps": len(leaves), "fine_substeps": 2*len(leaves),
                "maximum_depth": max(leaf.depth for leaf in leaves.values())+1,
                "successive_uniform_checks": successive,
                "exit_net_flux": float(np.vdot(fine_physical[0][-1], fine_physical[1][-1]).imag)}
            history.append(row)
            from temsim.physics.refinement_progress import emit_refinement_record
            emit_refinement_record(progress_callback, "spatial_embedded", row)
            if progress_callback:
                progress_callback(iteration+1, settings.maximum_rounds,
                    f"Spatial complex gun check {iteration+1}: change {change:.4g}; local {maximum_local:.4g}; "
                    f"leaves {len(leaves)}; exit flux {row['exit_net_flux']:.9g}")
            if successive >= 2:
                return fine_load, fine_result, {"method": "spatial-embedded-two-port-v1", "rounds": history,
                    "tolerance": settings.tolerance, "evaluations": work.evaluations,
                    "workers": settings.workers, "executor": settings.executor,
                    "integrator": settings.integrator,
                    "mesh_seed": export_mesh_seed(mesh, plan, fine=True),
                    "initial_mesh_scope": "Numerical subdivisions only; all fields and boundary conditions re-executed",
                    "scope": "Two successive uniform complex checks on an adaptive mesh; not full-source certification"}
            if passed:
                selected = set(leaves)
            else:
                selected, cumulative = set(), 0.
                target = .5*sum(indicators.values())
                for index in sorted(indicators, key=lambda key: (-indicators[key], key)):
                    selected.add(index)
                    cumulative += indicators[index]
                    if cumulative >= target:
                        break
            mesh = [child for i, leaf in enumerate(mesh)
                    for child in (leaf.split(work) if i in selected else (leaf,))]
            del fine_load, fine_result, fine, fine_physical, leaves, updates, ports
            # Even after a successful comparison, rebuild the exact current
            # coarse mesh so the next uniform test cannot reuse a fitted state.
            coarse_load, coarse_result, coarse, physical, _ = solve(False)
            del coarse_load, coarse_result
        raise ValueError("Spatial embedded refinement exhausted its round budget; no source was published")
    except Exception as failure:
        failure.refinement_diagnostic = {"scope": "FAILED_SPATIAL_MESH_COMPARISON_NOT_SOURCE",
            "completed_rounds": history, "evaluations": work.evaluations,
            "failed_constant_slab": getattr(failure, "slab_diagnostic", None),
            "retained_operator_bytes": work.retained_bytes, "numerics": vars(settings),
            "coarse_mesh": [{"physical_operator_index": leaf.physical_index,
                "fraction_start": leaf.start, "fraction_end": leaf.end, "depth": leaf.depth,
                "z_interval_nm": plan.get("interval_z_nm", {}).get(leaf.physical_index)}
                for leaf in mesh if isinstance(leaf, SpatialLeaf)]}
        raise
