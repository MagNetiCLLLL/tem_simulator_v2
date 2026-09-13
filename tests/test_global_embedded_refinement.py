"""Independent global boundary-mesh checks, not full source acceptance."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
from temsim.physics.global_embedded_refinement import boundary_difference
from temsim.physics.scattering_load import hermitian_slab
from test_occupied_axial_refinement import plan_and_boundary, varying


@pytest.mark.parametrize("integrator", ["cf4", "cf6"])
def test_global_mesh_comparison_reexecutes_boundary_and_matches_independent_ode(integrator):
    plan, boundary, incoming, calls = plan_and_boundary()
    def derivative(z, flat):
        q, g = varying(z)
        return (np.block([[-1j*g, np.eye(2)], [-4*np.eye(2)-q, -1j*g]])@flat.reshape(4, 4)).ravel()
    reference = solve_ivp(derivative, (0., 1.), np.eye(4, dtype=complex).ravel(),
                         method="DOP853", atol=2e-14, rtol=2e-13)
    assert reference.success
    left = solve(reference.y[:, -1].reshape(4, 4), np.vstack((np.eye(2), 2j*np.eye(2))))
    exact = solve((left[:2]+left[2:]/2j)/2, incoming)
    errors = []
    for tolerance in (1e-3, 1e-6):
        with threadpool_limits(1):
            load, result, record = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(
                enabled=True, tolerance=tolerance, integrator=integrator, strategy="global_embedded", maximum_rounds=24))
        fields, derivatives = load.propagate(result[1])
        errors.append(np.linalg.norm(fields[-1]-exact))
        assert record["rounds"][-1]["successive_uniform_checks"] == 2
        assert record["rounds"][-1]["maximum_complex_boundary_change"] <= tolerance
        assert record["rounds"][-1]["coarse_substeps"] == record["rounds"][-2]["fine_substeps"]
        currents = np.imag(np.einsum("ij,ij->i", fields.conj(), derivatives))
        np.testing.assert_allclose(currents, currents[0], rtol=2e-10)
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]/10
    assert np.linalg.norm(calls[0]-calls[-1]) > .01


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_global_mesh_serial_and_parallel_keep_identical_complex_states(executor):
    plan, boundary, _, _ = plan_and_boundary()
    ops, samplers = [], {}
    for index in range(20):
        sample = lambda f, index=index: varying((index+f)/20)
        ops.append(hermitian_slab(*sample(.5), 1/20, 2., carrier_k=2.)[0])
        samplers[index] = sample, 1/20
    plan.update(operators=ops, samplers=samplers, alpha=np.ones(21), log_derivative=np.zeros(21))
    outputs = []
    with threadpool_limits(1):
        for workers in (1, 4):
            load, result, report = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
                strategy="global_embedded", tolerance=1e-6, integrator="cf6", workers=workers,
                executor=executor, maximum_rounds=24))
            outputs.append((load.propagate(result[1]), report))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        np.testing.assert_array_equal(a, b)
    assert outputs[0][1]["rounds"] == outputs[1][1]["rounds"]


@pytest.mark.parametrize("options,message", [
    ({"maximum_evaluations": 2}, "evaluation budget"),
    ({"maximum_rounds": 2}, "round budget"),
    ({"maximum_depth": 1}, "depth budget"),
    ({"maximum_working_bytes": 1}, "memory budget"),
])
def test_global_budget_failures_do_not_publish_an_unresolved_wave(options, message):
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises((ValueError, MemoryError), match=message):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True,
            strategy="global_embedded", tolerance=1e-8, **options))


def test_global_comparison_neither_fits_phase_nor_ignores_intermediate_planes_or_derivatives():
    field = np.ones((3, 2), dtype=complex)
    derivative = 2j*field
    changed = field.copy(); changed[1] *= np.exp(.2j)
    assert boundary_difference((field, derivative), (changed, derivative)) > .19
    changed_derivative = derivative.copy(); changed_derivative[0] *= np.exp(.3j)
    assert boundary_difference((field, derivative), (field, changed_derivative)) > .29
    assert boundary_difference((field, derivative), (field, derivative)) == 0


def test_global_cancellation_and_unknown_strategy_remain_explicit():
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises(InterruptedError):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True, strategy="global_embedded"),
                          cancelled=lambda: True)
    with pytest.raises(ValueError, match="strategy"):
        OccupiedAxialRefinement(strategy="skip_checks").validate()


def test_global_diagnostic_locates_relative_error_without_hiding_small_field():
    from temsim.physics.global_embedded_refinement import boundary_difference_details
    a = np.array([[1., 0.], [1e-12, 0.]], complex)
    b = a.copy(); b[1, 1] = 1e-13j
    report = boundary_difference_details((a, 2j*a), (b, 2j*b))
    row = report["worst_boundaries"][0]
    assert row["component"] == "complex_field" and row["boundary_index"] == 1
    assert row["absolute_difference"] == pytest.approx(1e-13, rel=1e-15)
    assert row["comparison_scale"] == pytest.approx(np.sqrt(1.01)*1e-12, rel=1e-15)
    assert report["maximum_relative_difference"] == boundary_difference((a, 2j*a), (b, 2j*b))
    assert report["maximum_relative_difference"] > .09
