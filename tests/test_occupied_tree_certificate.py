"""Conservative compression of local error indicators, not a physical source."""
import numpy as np
import pytest

from temsim.physics.occupied_axial_refinement import (
    RefinableSlab, OccupiedAxialRefinement, _Work, internal_incoming, apply_two_port)
from temsim.physics.occupied_tree_certificate import compress_tree, certificate_indicator
from temsim.physics.scattering_load import hermitian_slab


def fixture():
    sample = lambda f: (np.diag([.8*np.sin(19*f), 2*np.cos(11*f)]),
                         np.array([[0., .15j*np.sin(5*f)], [-.15j*np.sin(5*f), 0.]]))
    operator, _ = hermitian_slab(*sample(.5), 1., 2., carrier_k=2.)
    tree = RefinableSlab(sample, 1., 2., operator)
    work = _Work(OccupiedAxialRefinement(enabled=True), lambda: False, None)
    tree.refine(np.array([1., 0.]), np.array([0., .2j]), 1e-5, work)
    return tree, work


def leaf_indicator(tree, left, right):
    if tree.children is None:
        return np.sqrt(tree.kappa)*np.linalg.norm(apply_two_port(tree.error_operator, left, right))
    first, second = tree.children
    back, forward = internal_incoming(first.operator, second.operator, left, right)
    return leaf_indicator(first, left, back)+leaf_indicator(second, forward, right)


def test_certificate_bounds_every_tested_complex_input_without_changing_transport():
    tree, work = fixture()
    rng = np.random.default_rng(62)
    inputs = rng.normal(size=(50, 4))+1j*rng.normal(size=(50, 4))
    expected = [leaf_indicator(tree, row[:2], row[2:]) for row in inputs]
    transport = [block.copy() for block in tree.operator]
    previous_bytes = work.retained_bytes
    compress_tree(tree, inputs[0, :2], inputs[0, 2:], work)
    assert tree.children is None and tree.partition
    assert work.retained_bytes == tree.certificate.nbytes < previous_bytes/10
    for before, after in zip(transport, tree.operator):
        np.testing.assert_array_equal(before, after)
    for row, error in zip(inputs, expected):
        assert certificate_indicator(tree, row[:2], row[2:]) >= error*(1-1e-12)


def test_changed_source_that_fails_certificate_reexecutes_recorded_partition():
    tree, work = fixture()
    left, right = np.array([1., 0.]), np.array([0., .2j])
    compress_tree(tree, left, right, work)
    evaluations = work.evaluations
    bound = certificate_indicator(tree, left, right)
    error, changed = tree.refine(left, right, bound*2, work)
    assert not changed and work.evaluations == evaluations
    error, changed = tree.refine(np.array([0., 10.]), np.zeros(2), 1e-8, work)
    assert changed and work.evaluations > evaluations
    assert error <= 1e-8 and tree.certificate is None
    compress_tree(tree, np.array([0., 10.]), np.zeros(2), work)
    assert work.retained_bytes == tree.certificate.nbytes


def test_compression_rejects_an_unverified_leaf():
    tree, work = fixture()
    leaf = tree
    while leaf.children:
        leaf = leaf.children[0]
    leaf.error_operator = None
    with pytest.raises(ValueError, match="unverified"):
        compress_tree(tree, np.ones(2), np.zeros(2), work)
