"""Full complex suffix reuse across adaptive mesh insertion and removal."""
import numpy as np
import pytest

from temsim.physics.incremental_scattering_load import IncrementalOutgoingLoad
from temsim.physics.scattering_load import hermitian_slab, outgoing_load


def slab(index, width=.1):
    q = np.array([[4.+index/40, .03j], [-.03j, 6.-index/50]])
    g = np.array([[.01, .02j], [-.02j, -.01]])
    return hermitian_slab(q, g, width, 2.)[0]


@pytest.mark.parametrize("index", [0, 3, 11])
@pytest.mark.parametrize("replacement_count", [0, 2, 5])
def test_changed_mesh_reuses_only_exact_suffix(index, replacement_count):
    ops = [slab(i) for i in range(12)]
    exit_q = np.diag([3., 9.])
    cache = IncrementalOutgoingLoad(1 << 20)
    original = cache.build(ops, exit_q, 2.)
    original_fields = original.propagate(np.array([1., .2j]))
    replacement = [slab(30+i, .04) for i in range(replacement_count)]
    changed = ops[:index]+replacement+ops[index+1:]
    actual = cache.build(changed, exit_q, 2.)
    direct = outgoing_load(changed, exit_q, 2.)
    new_start = index+replacement_count
    assert cache.last_recomputed_intervals == new_start
    for j in range(len(ops)-index):
        assert actual.reflections[new_start+j] is original.reflections[index+1+j]
    np.testing.assert_array_equal(actual.input_admittance, direct.input_admittance)
    np.testing.assert_array_equal(actual.propagate(np.array([1., .2j])), direct.propagate(np.array([1., .2j])))
    np.testing.assert_array_equal(original_fields, original.propagate(np.array([1., .2j])))
    assert cache.build(changed, exit_q, 2.) is actual


@pytest.mark.parametrize("exit_q,kappa", [(np.diag([4., 9.]), 2.), (np.diag([3., 9.]), 1.9)])
def test_changed_exit_or_chart_invalidates_even_matching_variable_suffix(exit_q, kappa):
    ops = [slab(i) for i in range(8)]
    cache = IncrementalOutgoingLoad(1 << 20)
    old = cache.build(ops, np.diag([3., 9.]), 2.)
    changed = ops[:2]+[slab(40), slab(41)]+ops[3:]
    actual = cache.build(changed, exit_q, kappa)
    assert cache.last_recomputed_intervals == len(changed)
    assert actual.reflections[-1] is not old.reflections[-1]
    np.testing.assert_array_equal(actual.input_admittance, outgoing_load(changed, exit_q, kappa).input_admittance)


def test_remove_entire_prefix_or_all_operators_preserves_complete_exit_response():
    ops, exit_q = [slab(i) for i in range(6)], np.diag([-1., 2.])
    cache = IncrementalOutgoingLoad(1 << 20)
    previous = cache.build(ops, exit_q, 2.)
    for changed in (ops[3:], []):
        actual = cache.build(changed, exit_q, 2.)
        assert cache.last_recomputed_intervals == 0
        assert actual.reflections[-1] is previous.reflections[-1]
        direct = outgoing_load(changed, exit_q, 2.)
        np.testing.assert_array_equal(actual.input_admittance, direct.input_admittance)
        np.testing.assert_array_equal(actual.propagate(np.array([1., .2j])), direct.propagate(np.array([1., .2j])))
        previous = actual


def test_cancellation_during_inserted_prefix_does_not_replace_completed_cache():
    ops, exit_q = [slab(i) for i in range(140)], np.diag([3., 9.])
    cache = IncrementalOutgoingLoad(1 << 20)
    previous = cache.build(ops, exit_q, 2.)
    changed = ops[:70]+[slab(40), slab(41)]+ops[71:]
    stopped = [False]
    def progress(*_):
        stopped[0] = True
    with pytest.raises(InterruptedError):
        cache.build(changed, exit_q, 2., cancelled=lambda: stopped[0], progress_callback=progress)
    assert cache.build(ops, exit_q, 2.) is previous
