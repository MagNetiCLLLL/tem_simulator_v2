"""Exact complete fields versus independent uncached right-load rebuilding."""
import numpy as np
import pytest

from temsim.physics.scattering_load import outgoing_load, hermitian_slab
from temsim.physics.incremental_scattering_load import IncrementalOutgoingLoad


def operators(count=24):
    return [hermitian_slab(np.diag([4.+i/30, 8.-i/30]),
            np.array([[.01, .02j], [-.02j, -.01]]), .1, 2.)[0] for i in range(count)]


def test_only_dependent_prefix_rebuilt_with_identical_complete_complex_response():
    ops, exit_q = operators(), np.diag([3., 9.])
    cache = IncrementalOutgoingLoad(1<<20)
    previous = cache.build(ops, exit_q, 2.)
    initial_field = previous.propagate(np.array([1., .2j]))
    for index in (0, 7, 18, 3):
        ops[index] = hermitian_slab(np.array([[5., .1j], [-.1j, 6.]]), np.zeros((2, 2)), .23+index/100, 2.)[0]
        actual = cache.build(ops, exit_q, 2.)
        direct = outgoing_load(ops, exit_q, 2.)
        assert cache.last_recomputed_intervals == index+1
        assert actual.reflections[index+1] is previous.reflections[index+1]
        np.testing.assert_array_equal(actual.input_admittance, direct.input_admittance)
        np.testing.assert_array_equal(actual.propagate(np.array([1., .2j])), direct.propagate(np.array([1., .2j])))
        previous = actual
    repeated = cache.build(ops, exit_q, 2.)
    assert repeated is previous and cache.last_recomputed_intervals == 0
    assert not repeated.input_admittance.flags.writeable
    assert np.linalg.norm(repeated.propagate(np.array([1., .2j]))[0]-initial_field[0]) > .01


def test_operator_inplace_edits_and_changed_exit_or_chart_cannot_reuse_stale_response():
    ops, exit_q = operators(), np.diag([3., 9.])
    cache = IncrementalOutgoingLoad(1<<20)
    original = cache.build(ops, exit_q, 2.)
    original_fields = original.propagate(np.array([1., .2j]))
    ops[2][0][0, 0] += .001j
    edited = cache.build(ops, exit_q, 2.)
    assert cache.last_recomputed_intervals == 3
    np.testing.assert_array_equal(edited.input_admittance, outgoing_load(ops, exit_q, 2.).input_admittance)
    np.testing.assert_array_equal(original_fields, original.propagate(np.array([1., .2j])))
    for exit_changed, kappa in ((np.diag([4., 9.]), 2.), (exit_q, 1.8)):
        actual = cache.build(ops, exit_changed, kappa)
        assert cache.last_recomputed_intervals == len(ops)
        np.testing.assert_array_equal(actual.input_admittance, outgoing_load(ops, exit_changed, kappa).input_admittance)


def test_cancellation_preserves_prior_completed_response_and_memory_limit_is_enforced():
    ops, exit_q = operators(), np.diag([3., 9.])
    cache = IncrementalOutgoingLoad(1<<20)
    original = cache.build(ops, exit_q, 2.)
    with pytest.raises(InterruptedError):
        cache.build(ops, exit_q, 2., cancelled=lambda: True)
    assert cache.build(ops, exit_q, 2.) is original
    with pytest.raises(MemoryError):
        IncrementalOutgoingLoad(1).build(ops, exit_q, 2.)


@pytest.mark.parametrize("exit_q", [np.diag([0., 2.]), np.diag([-1., 2.])])
def test_grazing_and_evanescent_exit_channels_remain_present(exit_q):
    ops = operators(3)
    cached = IncrementalOutgoingLoad(1<<20).build(ops, exit_q, 2.)
    direct = outgoing_load(ops, exit_q, 2.)
    np.testing.assert_array_equal(cached.output_wave_number, direct.output_wave_number)
    np.testing.assert_array_equal(cached.propagate(np.array([1., .2j])), direct.propagate(np.array([1., .2j])))
