"""Independent adaptive BVP checks, not physical-tip acceptance."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
from temsim.physics.scattering_load import hermitian_slab
from test_occupied_axial_refinement import plan_and_boundary, varying


def independent_exit(sample, incoming):
    def derivative(z, flat):
        q, g = sample(z)
        return (np.block([[-1j*g, np.eye(2)], [-4*np.eye(2)-q, -1j*g]])@flat.reshape(4, 4)).ravel()
    reference = solve_ivp(derivative, (0., 1.), np.eye(4, dtype=complex).ravel(),
                         method="DOP853", atol=2e-14, rtol=2e-13, max_step=.002)
    assert reference.success
    left = solve(reference.y[:, -1].reshape(4, 4), np.vstack((np.eye(2), 2j*np.eye(2))))
    return solve((left[:2]+left[2:]/2j)/2, incoming)


@pytest.mark.parametrize("integrator", ("cf4", "cf6"))
def test_adaptive_full_complex_solution_against_independent_ode(integrator):
    plan, boundary, incoming, calls = plan_and_boundary()
    exact = independent_exit(varying, incoming)
    errors = []
    for tolerance in (1e-3, 1e-6):
        with threadpool_limits(1):
            load, result, record = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(
                enabled=True, tolerance=tolerance, integrator=integrator, strategy="spatial_embedded",
                maximum_rounds=32))
        fields, derivatives = load.propagate(result[1])
        assert fields.shape == derivatives.shape == (2, 2)  # Original physical boundary contract.
        errors.append(np.linalg.norm(fields[-1]-exact))
        assert record["rounds"][-1]["successive_uniform_checks"] == 2
        assert record["rounds"][-1]["coarse_substeps"] == record["rounds"][-2]["fine_substeps"]
        current = np.imag(np.einsum("ij,ij->i", fields.conj(), derivatives))
        np.testing.assert_allclose(current, current[0], rtol=1e-10)
    assert errors[-1] < 1e-6 and errors[-1] < errors[0]/10
    assert np.linalg.norm(calls[0]-calls[-1]) > .01


@pytest.mark.parametrize("executor", ("thread", "process"))
def test_serial_and_parallel_adaptive_results_match_exactly(executor):
    plan, boundary, _, _ = plan_and_boundary()
    outputs = []
    settings = OccupiedAxialRefinement(enabled=True, strategy="spatial_embedded", integrator="cf6",
                                      tolerance=1e-5, maximum_rounds=32, executor=executor)
    with threadpool_limits(1):
        for workers in (1, 2):
            load, result, record = refine_joint_mode(plan, boundary, replace(settings, workers=workers))
            outputs.append((load.propagate(result[1]), record["rounds"]))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        np.testing.assert_array_equal(a, b)
    assert outputs[0][1] == outputs[1][1]


def test_localised_variation_uses_less_work_than_uniform_root_refinement():
    def local(z):
        peak = np.exp(-((z-.37)/.012)**2)
        return np.array([[2*peak, .4*peak], [.4*peak, -.3*peak]]), np.array([[.3, .1j], [-.1j, -.1]])*peak
    plan, boundary, incoming, _ = plan_and_boundary()
    # Deliberately one long root, but nonzero coarse/fine samples see this
    # feature. Additional physical reference refinement remains necessary.
    plan["samplers"] = {0: (local, 1.)}
    plan["operators"] = [hermitian_slab(*local(.5), 1., 2., carrier_k=2.)[0]]
    exact = independent_exit(local, incoming)
    outputs = []
    with threadpool_limits(1):
        for strategy in ("global_embedded", "spatial_embedded"):
            load, result, record = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(
                enabled=True, strategy=strategy, integrator="cf6", tolerance=1e-6, maximum_rounds=40))
            values = load.propagate(result[1])[0][-1]
            outputs.append((values, record["evaluations"]))
    np.testing.assert_allclose(outputs[0][0], exact, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(outputs[1][0], exact, atol=1e-6, rtol=1e-6)
    assert outputs[1][1] < outputs[0][1]/2


@pytest.mark.parametrize("options,message", [
    ({"maximum_evaluations": 2}, "evaluation budget"),
    ({"maximum_rounds": 2}, "round budget"),
    ({"maximum_depth": 1}, "depth budget"),
    ({"maximum_working_bytes": 1}, "memory budget"),
])
def test_unfinished_spatial_refinement_does_not_publish(options, message):
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises((ValueError, MemoryError), match=message) as failure:
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
            strategy="spatial_embedded", tolerance=1e-8, **options))
    assert failure.value.refinement_diagnostic["scope"] == "FAILED_SPATIAL_MESH_COMPARISON_NOT_SOURCE"


def test_both_faces_of_passive_physical_stop_and_backward_waves_are_preserved():
    plan, boundary, _, _ = plan_and_boundary()
    first, second = lambda f: varying(f/2), lambda f: varying(.5+f/2)
    zero, mask = np.zeros((2, 2), complex), np.diag([.8, .3]).astype(complex)
    plan.update(operators=[hermitian_slab(*first(.5), .5, 2., carrier_k=2.)[0],
        (zero, mask, mask, zero), hermitian_slab(*second(.5), .5, 2., carrier_k=2.)[0]],
        samplers={0: (first, .5), 2: (second, .5)}, alpha=np.ones(4), log_derivative=np.zeros(4))
    outputs = []
    with threadpool_limits(1):
        for strategy in ("global_embedded", "spatial_embedded"):
            load, result, _ = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
                strategy=strategy, integrator="cf6", tolerance=1e-7, maximum_rounds=40))
            outputs.append(load.propagate(result[1]))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7, rtol=1e-7)
    field, derivative = outputs[1]
    assert field.shape == derivative.shape == (4, 2)
    forward, backward = (field+derivative/(2j))/2, (field-derivative/(2j))/2
    np.testing.assert_allclose(forward[2], mask@forward[1], atol=1e-12)
    np.testing.assert_allclose(backward[1], mask@backward[2], atol=1e-12)
    assert np.linalg.norm(backward[2]) > .001
    currents = np.imag(np.einsum("ij,ij->i", field.conj(), derivative))
    assert currents[1] > currents[2]


def test_spatial_cancellation_and_fixed_plan_are_not_changed():
    plan, boundary, _, calls = plan_and_boundary()
    original = np.array(plan["operators"])
    with pytest.raises(InterruptedError):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
            strategy="spatial_embedded"), cancelled=lambda: True)
    assert calls == []
    np.testing.assert_array_equal(plan["operators"], original)


def test_worker_adoption_does_not_duplicate_the_shared_sampler_or_coarse_operator():
    import cloudpickle
    from temsim.physics.spatial_embedded_mesh import initial_mesh
    from temsim.physics.occupied_axial_refinement import _Work
    plan, _, _, _ = plan_and_boundary()
    leaf, = initial_mesh(plan)
    copied = cloudpickle.loads(cloudpickle.dumps(leaf))
    work = _Work(OccupiedAxialRefinement(), lambda: False, None)
    copied.refine(np.ones(2), np.zeros(2), np.inf, work)
    assert leaf.adopt_worker_state(copied) is leaf
    assert leaf.sample is plan["samplers"][0][0] and leaf.coarse is plan["operators"][0]
    assert leaf.halves is copied.halves
    copied.start = .25
    with pytest.raises(RuntimeError, match="different numerical interval"):
        leaf.adopt_worker_state(copied)
