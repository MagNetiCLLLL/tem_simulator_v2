"""Deterministic cost dispatch only; no physics or result-order changes."""
from types import SimpleNamespace

import pytest

from temsim.physics.occupied_wave_processes import balanced_interval_groups


def roots(costs, cached=()):
    return {i: SimpleNamespace(divisions=value, operator=object() if i in cached else None)
            for i, value in enumerate(costs)}


def test_uniform_costs_keep_small_physical_order_groups():
    assert balanced_interval_groups(roots([1]*35)) == (tuple(range(12)), tuple(range(12, 24)), tuple(range(24, 35)))


def test_heavy_neighbours_are_dispatched_independently_first():
    values = [1]*32+[4096]*4+[1]*28
    groups = balanced_interval_groups(roots(values))
    assert groups[:4] == ((32,), (33,), (34,), (35,))
    assert all(len(group) <= 16 for group in groups)
    assert sorted(i for group in groups for i in group) == list(range(len(values)))
    assert groups == balanced_interval_groups(dict(reversed(list(roots(values).items()))))


def test_cached_fine_operator_is_not_scheduled_as_expensive_propagation():
    values = roots([4096, 128]+[1]*30, cached=(0,))
    assert balanced_interval_groups(values)[0] == (1,)


def test_no_intervals_and_invalid_group_limit():
    assert balanced_interval_groups({}) == ()
    with pytest.raises(ValueError): balanced_interval_groups(roots([1]), 0)


def test_round_budget_failure_preserves_completed_global_diagnostics():
    from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
    from test_occupied_axial_refinement import plan_and_boundary
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises(ValueError, match="round budget") as caught:
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
            strategy="global_embedded", maximum_rounds=2, tolerance=1e-8))
    report = caught.value.refinement_diagnostic
    assert len(report["completed_rounds"]) == 2
    assert report["coarse_mesh"][0]["substeps"] >= 2
    assert report["scope"] == "FAILED_GLOBAL_MESH_COMPARISON_NOT_SOURCE"


def test_private_packets_preserve_complex_arrays_without_large_pipe_messages(tmp_path):
    import pickle
    import numpy as np
    from temsim.physics.wave_process_packets import write_packet, read_packet
    data = np.arange(200_000, dtype=float)*(1+2j)
    descriptor = write_packet(tmp_path/"owned.bin", {"complex_state": data})
    assert len(pickle.dumps(descriptor)) < 1024
    assert descriptor[1] > data.nbytes
    np.testing.assert_array_equal(read_packet(descriptor)["complex_state"], data)
    with pytest.raises(FileExistsError): write_packet(tmp_path/"owned.bin", None)
    with pytest.raises(ValueError, match="Incomplete"):
        read_packet((descriptor[0], descriptor[1]-1))


def test_worker_failure_keeps_actual_slab_and_traceback_out_of_pipe(tmp_path):
    from temsim.physics.wave_process_packets import write_packet, read_worker_result
    failure = ValueError("unconverged slab")
    failure.slab_diagnostic = {"physical_step": .3, "q_real": [[1., 2.], [2., 3.]]}
    packet = write_packet(tmp_path/"failed.bin", (False, failure, "original worker traceback"))
    with pytest.raises(ValueError, match="unconverged") as caught:
        read_worker_result(packet)
    assert caught.value.slab_diagnostic == failure.slab_diagnostic
    assert caught.value.worker_traceback == "original worker traceback"
