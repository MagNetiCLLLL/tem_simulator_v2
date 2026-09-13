"""Cached complete half slabs stay local, but current inputs are rechecked."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, _Work, _refine_intervals
from temsim.physics.spatial_embedded_mesh import initial_mesh
from test_occupied_axial_refinement import plan_and_boundary


def test_cached_leaf_does_not_spawn_or_copy_matrices(monkeypatch):
    plan, _, _, _ = plan_and_boundary()
    leaf, = initial_mesh(plan)
    settings = OccupiedAxialRefinement(integrator="cf6", workers=2, executor="process")
    work = _Work(settings, lambda: False, None)
    leaf.refine(np.ones(2), np.zeros(2), np.inf, work)
    halves = leaf.halves
    def no_worker(*args, **kwargs):
        pytest.fail("Cached full matrices must not be copied to a process")
    monkeypatch.setattr("temsim.physics.occupied_wave_processes.refine_process_intervals", no_worker)
    rng = np.random.default_rng(614)
    measured = []
    for _ in range(4):
        incoming = rng.normal(size=(2, 2, 2))+1j*rng.normal(size=(2, 2, 2))
        expected = leaf.refine(incoming[0][0], incoming[1][1], np.inf, work)[0]
        rows = _refine_intervals({0: leaf}, incoming, np.inf, work)
        assert rows[0][1] == expected
        measured.append(expected)
    assert np.ptp(measured) > 1e-5
    assert work.evaluations == 2 and leaf.halves is halves


def test_only_missing_halves_go_to_worker_and_results_remain_in_physical_order(monkeypatch):
    plan, _, _, _ = plan_and_boundary()
    first, = initial_mesh(plan)
    second, = initial_mesh(plan)
    settings = OccupiedAxialRefinement(integrator="cf6", workers=2, executor="process")
    work = _Work(settings, lambda: False, None)
    first.refine(np.ones(2), np.zeros(2), np.inf, work)
    seen = []
    def execute_locally(roots, incoming, budget, parent):
        # Offline scheduling fixture; actual spawned workers are exercised by
        # the complete spatial/BVP process tests, not claimed by this stub.
        seen.extend(roots)
        local = _Work(replace(settings, workers=1), parent.cancelled, None)
        rows = _refine_intervals(roots, incoming, budget, local)
        parent.evaluations += local.evaluations
        parent.retain(local.retained_bytes)
        return rows
    monkeypatch.setattr("temsim.physics.occupied_wave_processes.refine_process_intervals", execute_locally)
    incoming = np.array([[[1., .2j], [.1, .3]], [[.2, 1.], [1j, .4]], [[.3, .1], [0., .6j]]])
    roots = {1: second, 0: first}
    rows = _refine_intervals(roots, incoming, np.inf, work)
    assert seen == [1] and [row[0] for row in rows] == [0, 1]
    assert work.evaluations == 4
    for index, error, _, _ in rows:
        assert error == roots[index].refine(incoming[index][0], incoming[index+1][1], np.inf, work)[0]


def test_cancelled_cached_recheck_is_not_reported_as_complete():
    plan, _, _, _ = plan_and_boundary()
    leaf, = initial_mesh(plan)
    settings = OccupiedAxialRefinement(workers=2, executor="process")
    work = _Work(settings, lambda: False, None)
    leaf.refine(np.ones(2), np.zeros(2), np.inf, work)
    work.cancelled = lambda: True
    with pytest.raises(InterruptedError):
        _refine_intervals({0: leaf}, np.ones((2, 2, 2)), np.inf, work)
