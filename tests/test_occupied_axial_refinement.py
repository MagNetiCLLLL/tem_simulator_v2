"""Independent two-way fixtures for source-specific axial error control."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from temsim.physics.occupied_axial_refinement import (
    OccupiedAxialRefinement, refine_joint_mode, internal_incoming, apply_two_port,
    RefinableSlab, _Work)
from temsim.physics.scattering_load import hermitian_slab, compose


def varying(f):
    return np.array([[.8*np.sin(19*f), .2*np.cos(7*f)], [.2*np.cos(7*f), .4*f]]), np.array(
        [[.7*np.cos(13*f), .15j*np.sin(11*f)], [-.15j*np.sin(11*f), -.2*f]])


def plan_and_boundary():
    kappa = 2.
    op, _ = hermitian_slab(*varying(.5), 1., kappa, carrier_k=kappa)
    plan = {"operators": [op], "samplers": {0: (varying, 1.)}, "exit_q": 4*np.eye(2),
            "kappa": kappa, "alpha": np.ones(2), "log_derivative": np.zeros(2)}
    incoming = np.array([1., .2j])/np.sqrt(2*1.04)
    calls = []
    def boundary(load):
        # Fixture: specified incoming reservoir at LEFT, with its total field
        # re-solved against every newly embedded downstream reflection.
        trace = solve(load.input_admittance+2j*np.eye(2), 4j*incoming)
        calls.append(trace.copy())
        return None, trace, {"fixture": True}
    return plan, boundary, incoming, calls


@pytest.mark.parametrize("integrator", ["cf4", "cf6"])
@pytest.mark.parametrize("allocation", ["uniform", "initial_indicator"])
def test_joint_refinement_preserves_reflection_and_matches_direct_ode(integrator, allocation):
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
        load, result, record = refine_joint_mode(plan, boundary,
            OccupiedAxialRefinement(enabled=True, tolerance=tolerance, integrator=integrator,
                                    error_budget_allocation=allocation))
        fields, derivatives = load.propagate(result[1])
        errors.append(np.linalg.norm(fields[-1]-exact))
        assert record["rounds"][-1]["summed_occupied_indicator"] <= tolerance
        assert record["rounds"][-1]["changed_intervals"] == 0
        currents = np.imag(np.einsum("ij,ij->i", fields.conj(), derivatives))
        np.testing.assert_allclose(currents, currents[0], rtol=1e-10)
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]/10
    assert len(calls) >= 6
    assert np.linalg.norm(calls[0]-calls[-1]) > .01


def test_internal_interface_includes_returning_wave():
    a, _ = hermitian_slab(np.diag([-.8, -.2]), np.zeros((2, 2)), .6, 2., carrier_k=2.)
    b, _ = hermitian_slab(np.diag([-.3, -.9]), np.zeros((2, 2)), .8, 2., carrier_k=2.)
    left, right = np.array([1., .2j]), np.array([.3j, -.2])
    back, forward = internal_incoming(a, b, left, right)
    expected = apply_two_port(compose(a, b), left, right)
    actual = np.r_[apply_two_port(a, left, back)[:2], apply_two_port(b, forward, right)[2:]]
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_expanded_tree_rechecks_leaves_for_changed_source():
    # Refining an initially occupied channel must not make the tree blind
    # when the re-solved load populates another, previously weak channel.
    sampler = lambda f: (np.diag([.8*np.sin(19*f), 2*np.cos(11*f)]), np.zeros((2, 2)))
    initial, _ = hermitian_slab(*sampler(.5), 1., 2., carrier_k=2.)
    tree = RefinableSlab(sampler, 1., 2., initial)
    work = _Work(OccupiedAxialRefinement(enabled=True, tolerance=1e-5), lambda: False, None)
    _, changed = tree.refine(np.array([1., 0.]), np.zeros(2), 1e-5, work)
    assert changed and tree.expanded
    earlier = work.evaluations
    error, changed = tree.refine(np.array([0., 10.]), np.zeros(2), 1e-7, work)
    assert changed
    assert work.evaluations > earlier
    assert error <= 1e-7


def test_compact_leaf_checks_new_channel_and_accounts_retained_matrices():
    sampler = lambda f: (np.diag([0., 2*np.cos(11*f)]), np.zeros((2, 2)))
    initial, _ = hermitian_slab(*sampler(.5), 1., 2., carrier_k=2.)
    tree = RefinableSlab(sampler, 1., 2., initial)
    work = _Work(OccupiedAxialRefinement(enabled=True), lambda: False, None)
    error, changed = tree.refine(np.array([1., 0.]), np.zeros(2), 1e-5, work)
    assert not changed and error < 1e-5
    assert tree.children is None and tree.error_operator is not None
    assert work.retained_bytes == sum(block.nbytes for block in tree.error_operator)
    first_count = work.evaluations
    tree.refine(np.array([.5j, 0.]), np.array([.2, 0.]), 1e-5, work)
    assert work.evaluations == first_count
    error, changed = tree.refine(np.array([0., 1.]), np.zeros(2), 1e-5, work)
    assert changed and tree.expanded and tree.error_operator is None
    assert work.evaluations > first_count and error <= 1e-5
    def retained(node, root=False):
        count = 0 if root else sum(block.nbytes for block in node.operator)
        count += sum(block.nbytes for block in (node.error_operator or ()))
        return count+sum(retained(child) for child in (node.children or ()))
    assert work.retained_bytes == retained(tree, root=True)
    with pytest.raises(RuntimeError, match="memory release"):
        work.release(work.retained_bytes+1)


def test_budget_cancellation_and_round_limit_do_not_publish_source():
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises(ValueError, match="evaluation budget") as failure:
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True, maximum_evaluations=2))
    evidence = failure.value.refinement_diagnostic
    assert evidence["scope"].startswith("FAILED_REFINEMENT")
    assert evidence["evaluations"] == 2
    assert evidence["largest_returned_trees"][0]["operator_index"] == 0
    with pytest.raises(ValueError, match="round budget"):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True, maximum_rounds=2))
    with pytest.raises(InterruptedError):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True), cancelled=lambda: True)
    with pytest.raises(MemoryError):
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True, maximum_working_bytes=1))


@pytest.mark.parametrize("executor", ["thread", "process"])
@pytest.mark.parametrize("numerics", [{}, {"error_budget_allocation": "initial_indicator"},
                                      {"integrator": "cf6", "error_budget_allocation": "initial_indicator"}])
def test_parallel_intervals_reproduce_serial_full_complex_solution(executor, numerics):
    plan, boundary, _, _ = plan_and_boundary()
    operators, samplers = [], {}
    for index in range(20):
        sample = lambda f, index=index: varying((index+f)/20)
        operator, _ = hermitian_slab(*sample(.5), 1/20, 2., carrier_k=2.)
        operators.append(operator)
        samplers[index] = (sample, 1/20)
    plan.update(operators=operators, samplers=samplers, alpha=np.ones(21), log_derivative=np.zeros(21))
    outputs = []
    with threadpool_limits(1):
        for workers in (1, 4):
            load, result, record = refine_joint_mode(plan, boundary,
                OccupiedAxialRefinement(enabled=True, tolerance=1e-6, workers=workers, executor=executor, **numerics))
            outputs.append((load.propagate(result[1]), record))
    for serial, parallel in zip(outputs[0][0], outputs[1][0]):
        np.testing.assert_array_equal(serial, parallel)
    assert outputs[0][1]["rounds"] == outputs[1][1]["rounds"]
    assert outputs[0][1]["evaluations"] == outputs[1][1]["evaluations"]


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_parallel_budget_failure_cannot_publish_a_partial_load(executor):
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises(ValueError, match="evaluation budget") as failure:
        refine_joint_mode(plan, boundary,
            OccupiedAxialRefinement(enabled=True, workers=4, maximum_evaluations=2, executor=executor))
    assert failure.value.refinement_diagnostic["evaluations"] == 2


def test_os_memory_pressure_aborts_before_new_retained_operator(monkeypatch):
    import temsim.physics.wave_execution as execution
    def no_space(size):
        raise MemoryError("test physical-memory reserve")
    monkeypatch.setattr(execution, "check_available_memory", no_space)
    work = _Work(OccupiedAxialRefinement(), lambda: False, None)
    with pytest.raises(MemoryError, match="physical-memory reserve"):
        work.retain(64)
    assert work.retained_bytes == 0


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"tolerance": float("nan")}, {"tolerance": 0.},
                                    {"maximum_rounds": 1}, {"maximum_working_bytes": 0},
                                    {"workers": 0}, {"workers": True}, {"workers": 33},
                                    {"integrator": "unsupported"}, {"error_budget_allocation": "unchecked"}])
def test_invalid_numerical_settings(kwargs):
    with pytest.raises(ValueError):
        OccupiedAxialRefinement(**kwargs).validate()


def test_final_physical_solve_cannot_be_skipped_by_pilot_scheduling():
    settings = OccupiedAxialRefinement(enabled=True)
    assert settings.applies_to_chart(0, 0)
    assert not settings.applies_to_chart(0, 1)
    assert settings.applies_to_chart(1, 1)
    assert OccupiedAxialRefinement(enabled=True, refine_pilot_charts=True).applies_to_chart(0, 1)
    assert not OccupiedAxialRefinement(enabled=False).applies_to_chart(0, 0)
    with pytest.raises(ValueError):
        settings.applies_to_chart(2, 1)


def test_adaptive_allocations_preserve_total_budget_and_keep_empty_channels_testable():
    from temsim.physics.occupied_axial_refinement import allocate_error_budget
    for order in (4, 6):
        budget = allocate_error_budget({0: 0., 2: 1e-16, 4: .1, 6: 1e100}, .001, order)
        assert set(budget) == {0, 2, 4, 6}
        assert all(v > 0 for v in budget.values())
        assert .000999999 < sum(budget.values()) <= .001
        assert budget[6] > budget[4] >= budget[0]
        assert max(allocate_error_budget({0: 0., 1: 0.}, .001, order).values()) <= .0005
