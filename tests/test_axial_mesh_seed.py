"""Safe reuse of numerical partitions; all physical complex waves re-execute."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.immutable_json import json_digest
from temsim.physics.axial_mesh_seed import read_mesh_seed, validate_mesh_seed
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
from test_occupied_axial_refinement import plan_and_boundary, varying
from test_spatial_embedded_refinement import independent_exit


def settings(**kwargs):
    return OccupiedAxialRefinement(enabled=True, strategy="spatial_embedded", integrator="cf6",
        tolerance=1e-6, maximum_rounds=32, **kwargs)


@pytest.mark.parametrize("executor,workers", (("thread", 1), ("thread", 2), ("process", 2)))
def test_seed_reexecutes_full_bvp_and_two_uniform_comparisons(executor, workers):
    plan, boundary, incoming, calls = plan_and_boundary()
    plan["interval_z_nm"] = {0: (0., 1.)}
    with threadpool_limits(1):
        original, result, record = refine_joint_mode(plan, boundary, settings())
        seed = record["mesh_seed"]
        before = len(calls)
        restarted, result, new = refine_joint_mode(plan, boundary, settings(
            executor=executor, workers=workers, initial_mesh=seed))
    values = restarted.propagate(result[1])[0][-1]
    np.testing.assert_allclose(values, independent_exit(varying, incoming), atol=1e-6, rtol=1e-6)
    assert len(calls) > before
    assert len(new["rounds"]) == 2
    assert new["rounds"][-1]["successive_uniform_checks"] == 2
    assert new["evaluations"] >= len(seed)*7  # Rebuilt seed, first halves and next halves.
    assert new["initial_mesh_scope"].startswith("Numerical subdivisions only")
    assert json_digest(asdict(settings(initial_mesh=seed))) != json_digest(asdict(settings()))


def test_failed_round_mesh_is_numeric_only_and_reusable(tmp_path):
    plan, boundary, incoming, _ = plan_and_boundary()
    plan["interval_z_nm"] = {0: (0., 1.)}
    with pytest.raises(ValueError, match="round budget") as failure:
        refine_joint_mode(plan, boundary, replace(settings(), maximum_rounds=2))
    path = tmp_path/"failed.json"
    path.write_text(json.dumps({"refinement_diagnostic": failure.value.refinement_diagnostic}))
    seed = read_mesh_seed(path)
    assert len(seed) > 1
    assert all(len(row) == 5 for row in seed)
    load, result, record = refine_joint_mode(plan, boundary, settings(initial_mesh=seed))
    np.testing.assert_allclose(load.propagate(result[1])[0][-1], independent_exit(varying, incoming),
                               atol=1e-6, rtol=1e-6)
    assert record["rounds"][-1]["successive_uniform_checks"] == 2


@pytest.mark.parametrize("seed", (
    ((0., 1., .5, 1., 1),),  # Gap.
    ((0., 1., 0., .5, 1), (0., 1., .25, .5, 2), (0., 1., .5, 1., 1)),
    ((0., 1., 0., 1., True),),
    ((0., 1., 0., 1., 1),),  # Depth and length disagree.
    ((0., float("nan"), 0., 1., 0),),
    ((1., 0., 0., 1., 0),),
    ((0., 1., 0., .3, 2), (0., 1., .3, 1., 1)),
))
def test_invalid_seed_rejected_before_any_physical_calculation(seed):
    with pytest.raises(ValueError):
        settings(initial_mesh=seed).validate()


def test_unmatched_physical_interval_rejected_not_interpolated():
    plan, boundary, _, calls = plan_and_boundary()
    plan["interval_z_nm"] = {0: (0., 1.)}
    with pytest.raises(ValueError, match="do not match"):
        refine_joint_mode(plan, boundary, settings(initial_mesh=((0., 2., 0., 1., 0),)))
    assert not calls


def test_seed_cannot_enable_or_skip_refinement():
    seed = ((0., 1., 0., 1., 0),)
    for changes in ({"enabled": False}, {"strategy": "local_sum"}):
        with pytest.raises(ValueError, match="require enabled spatial"):
            replace(settings(initial_mesh=seed), **changes).validate()


def test_explicit_seed_json_read_without_source_fields(tmp_path):
    path = tmp_path/"seed.json"
    path.write_text(json.dumps({"schema": "wave-axial-mesh-seed-v1", "mesh_seed": [[0., 1., 0., 1., 0]],
                               "amplitude": "ignored: not loaded as a physical field"}))
    assert read_mesh_seed(path) == ((0., 1., 0., 1., 0),)
    assert read_mesh_seed(None) == ()
    validate_mesh_seed(())


def test_cancelled_seed_does_not_return_a_source():
    plan, boundary, _, calls = plan_and_boundary()
    plan["interval_z_nm"] = {0: (0., 1.)}
    with pytest.raises(InterruptedError):
        refine_joint_mode(plan, boundary, settings(initial_mesh=((0., 1., 0., .5, 1), (0., 1., .5, 1., 1))),
                          cancelled=lambda: True)
    assert not calls
