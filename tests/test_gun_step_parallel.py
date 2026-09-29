"""Parallel gun iterations must retain the serial global convergence decision."""
from threading import enumerate as threads
import numpy as np
import pytest

from temsim.cpu_resources import numerical_job
from temsim.physics import grounded_particle_step as steps
from temsim.physics.discrete_gradient import ITERATION_TOLERANCE, MAXIMUM_ITERATIONS
from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev
from test_closed_gun_execution import uniform_field


def inputs(count=8193, z=.04):
    x = np.zeros((count, 3))
    x[:, 0] = np.linspace(-1e-4, 1e-4, count)
    x[:, 2] = z
    p = momentum_from_kinetic_energy_ev(np.linspace(350., 500., count),
                                      np.tile([.01, .02, 1.], (count, 1)))
    # Actual local finite fields, with all particles inside their support.
    magnetic = np.array([40., 10., 1., .001, -.002, 40., 10., 1., 0., 0.,
                         40., 10., 1., .2, .3])
    return (x, p, 1e-12, ITERATION_TOLERANCE, MAXIMUM_ITERATIONS,
            steps._closed_field_data(uniform_field()), 2000., magnetic, True, True, None, None)


@pytest.mark.parametrize("budget", [1, 2, 4])
def test_parallel_step_is_bitwise_equal_with_global_iteration_barrier(budget):
    if steps.njit is None:
        pytest.skip("No Numba")
    args = inputs()
    expected = steps._step(*args)
    with numerical_job(budget), steps.gun_step_workers():
        pool = steps._STEP_POOL.get()
        actual = pool.step(*args)
        assert pool.workers <= budget
        if budget > 1:
            assert pool.executor is not None
        for got, wanted in zip(actual, expected):
            np.testing.assert_array_equal(got, wanted)
    assert steps._STEP_POOL.get() is None
    assert not any(thread.name.startswith("gun-step") for thread in threads())


def test_worker_failure_joins_children_before_releasing_numerical_scope():
    if steps.njit is None:
        pytest.skip("No Numba")
    args = inputs(z=.199999)
    with numerical_job(2), steps.gun_step_workers():
        pool = steps._STEP_POOL.get()
        with pytest.raises(ValueError, match="outside the solved gun field"):
            pool.step(*args)
    assert steps._STEP_POOL.get() is None
    assert not any(thread.name.startswith("gun-step") for thread in threads())


def test_nested_scope_reuses_workers_and_small_batches_stay_serial():
    if steps.njit is None:
        pytest.skip("No Numba")
    with numerical_job(2), steps.gun_step_workers():
        pool = steps._STEP_POOL.get()
        with steps.gun_step_workers():
            assert steps._STEP_POOL.get() is pool
            pool.step(*inputs(9))
        assert pool.executor is None
    assert steps._STEP_POOL.get() is None
