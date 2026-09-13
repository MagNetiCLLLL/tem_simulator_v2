"""Analytic cell areas and equivalent CPU display backends, not optics."""
import numpy as np
import pytest

from temsim.physics.affine_cell_histogram import affine_cell_histogram


@pytest.mark.parametrize("basis", (np.eye(2), np.array([[1., .2], [0., 1.5]]),
                                  np.array([[.7, -.7], [.7, .7]]), np.array([[1., 0.], [0., -1.]])))
def test_uniform_affine_field_has_no_false_internal_stripes_and_conserves_current(basis):
    output, _ = affine_cell_histogram(np.ones((8, 8))/64, basis, np.zeros(2), ((-10., 10.), (-10., 10.)), 64, backend="python")
    assert output.sum() == pytest.approx(1., rel=1e-12)
    expected = (20/64)**2/(64*abs(np.linalg.det(basis)))
    np.testing.assert_allclose(output[29:34, 29:34], expected, atol=1e-14, rtol=1e-12)


def test_visible_half_cell_is_half_current_without_renormalisation():
    output, _ = affine_cell_histogram(np.array([[2.]]), np.eye(2), np.zeros(2), ((0., .5), (-.5, .5)), (3, 4), backend="python")
    np.testing.assert_allclose(output, 1/12, rtol=1e-13)
    assert output.sum() == pytest.approx(1.)


def test_outside_cell_and_empty_field_have_no_spurious_signal():
    for weight, origin in ((1., (100., 0.)), (0., (0., 0.))):
        output, _ = affine_cell_histogram(np.array([[weight]]), np.eye(2), origin, ((-1., 1.), (-1., 1.)), 8, backend="python")
        assert not np.any(output)


def test_python_and_numba_apply_the_same_conservative_polygons():
    pytest.importorskip("numba")
    weights = np.random.default_rng(17).random((31, 32))
    basis, origin = np.array([[.5, .3], [-.2, .6]]), np.array([.23, -.41])
    args = weights, basis, origin, ((-10., 10.), (-10., 10.)), (64, 63)
    cpu, _ = affine_cell_histogram(*args, backend="python")
    fast, record = affine_cell_histogram(*args, backend="numba")
    np.testing.assert_allclose(fast, cpu, rtol=1e-12, atol=1e-12)
    assert record["backend"] == "numba"


@pytest.mark.parametrize("change", [dict(basis=np.zeros((2, 2))), dict(bounds=((1., -1.), (-1., 1.))),
    dict(weights=[[np.nan]]), dict(bins=0), dict(backend="cuda")])
def test_invalid_display_inputs_are_explicit(change):
    args = dict(weights=np.ones((2, 2)), basis=np.eye(2), origin=(0., 0.), bounds=((-1., 1.), (-1., 1.)))
    args.update(change)
    with pytest.raises(ValueError):
        affine_cell_histogram(**args)
