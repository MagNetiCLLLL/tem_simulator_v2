"""Loaded, driven complex boundaries with upstream reflection feedback."""
import numpy as np
import pytest

from temsim.physics.scattering_load import hermitian_slab, outgoing_load
from temsim.physics.radial_gun_wave import laguerre_operators, aperture_projection


@pytest.mark.parametrize("count", [2, 16, 64])
def test_reflected_composition_factors_once_and_matches_separate_solve(count, monkeypatch):
    import temsim.physics.scattering_load as module
    from scipy.linalg import solve
    rng = np.random.default_rng(917)
    matrices = [(rng.normal(size=(count, count))+1j*rng.normal(size=(count, count)))*.02
                for _ in range(8)]
    a, b, c, d, e, f, g, h = matrices
    den = np.eye(count)-d@e
    fc, df = solve(den, c), solve(den, d@f)
    expected = a+b@e@fc, b@(f+e@df), g@fc, h+g@df
    calls = []
    def recorded(matrix, rhs, **kwargs):
        calls.append(rhs.shape)
        return solve(matrix, rhs, **kwargs)
    monkeypatch.setattr(module, "solve", recorded)
    actual = module.compose(matrices[:4], matrices[4:])
    assert calls == [(count, 2*count)]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)


def test_carrier_triangular_solve_matches_generic_factor_without_changed_phase(monkeypatch):
    import temsim.physics.scattering_load as module
    from scipy.linalg import solve
    rng = np.random.default_rng(153)
    count = 16
    raw = rng.normal(size=(2, count, count))+1j*rng.normal(size=(2, count, count))
    residual, connection = raw+raw.conj().transpose(0, 2, 1)
    connection *= .002
    actual, _ = module.covariant_carrier_slab(residual, connection, .17, 24., 24.)
    monkeypatch.setattr(module, "solve_triangular",
                        lambda a, b, **kwargs: solve(a, b, check_finite=False))
    expected, _ = module.covariant_carrier_slab(residual, connection, .17, 24., 24.)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)


def test_reflected_load_matches_scalar_step_and_preserves_total_trace():
    q = np.array([[4.]])
    op, _ = hermitian_slab(q, np.zeros((1, 1)), .37, 1.)
    loaded = outgoing_load([op], np.array([[9.]]), 1.)
    field, derivative = loaded.propagate(np.array([1.+.3j]))
    np.testing.assert_allclose(field[0], [1.+.3j], atol=1e-14)
    np.testing.assert_allclose(derivative[0], loaded.input_admittance@field[0], atol=1e-14)
    np.testing.assert_allclose(derivative[-1], 3j*field[-1], atol=1e-14)
    current = np.imag((field.conj()*derivative).sum(axis=1))
    assert current[0] == pytest.approx(current[-1], rel=1e-13)
    assert loaded.reflections[0][0, 0] != 0


def test_unitary_connection_changes_complex_field_and_preserves_flux():
    q = np.diag([4., 9.]); g = np.array([[0., .2], [.2, 0.]])
    op, _ = hermitian_slab(q, g, .1, 2.)
    loaded = outgoing_load([op], q, 2.)
    field, derivative = loaded.propagate(np.array([1.+0j, 0.]))
    current = np.imag((field.conj()*derivative).sum(axis=1))
    assert current[0] == pytest.approx(current[-1], rel=1e-12)
    assert abs(field[-1, 1]) > .01


def test_radial_kinetic_and_dilation_have_required_symmetries():
    x, d, k = laguerre_operators(10)
    np.testing.assert_allclose(x, x.T)
    np.testing.assert_allclose(d, -d.T)
    np.testing.assert_allclose(k, k.T)
    assert np.linalg.eigvalsh(k).min() > 0
    projection = aperture_projection(2., 1., 10, 80)
    eigen = np.linalg.eigvalsh(projection)
    assert eigen.min() > -1e-12 and eigen.max() < 1+1e-12
    np.testing.assert_allclose(aperture_projection(100., 1., 10, 80), np.eye(10))


def test_small_transverse_phase_survives_large_axial_carrier():
    # k0**2 +/- 1e-12 rounds to the same float, but the residual accumulates
    # a measurable differential phase over a long optical path.
    carrier, width = 3000., 6e14
    residual = np.diag([-1e-12, 1e-12])
    op, row = hermitian_slab(residual, np.zeros((2, 2)), width, carrier, carrier_k=carrier)
    phase_ratio = op[2][1, 1]/op[2][0, 0]
    assert phase_ratio == pytest.approx(np.exp(.2j), abs=1e-13)
    assert row["separate_axial_carrier"]


@pytest.mark.parametrize("residual", (-5., -4., -3., 2.))
def test_carrier_split_equals_unsplit_slab_in_resolved_domain(residual):
    original, _ = hermitian_slab(np.array([[4.+residual]]), np.zeros((1, 1)), .7, 1.5)
    split, _ = hermitian_slab(np.array([[residual]]), np.zeros((1, 1)), .7, 1.5, carrier_k=2.)
    np.testing.assert_allclose(split, original, rtol=1e-13, atol=1e-13)


def test_electrode_knot_quadrature_resolves_cusp_without_basis_fit():
    from temsim.physics.radial_gun_wave import resolved_radial_quadrature
    from scipy.integrate import quad
    r, weighted = resolved_radial_quadrature(1., 8, np.array([0., .713, 20.]), 32)
    actual = (weighted*abs(r-.713))@weighted.T
    exact = quad(lambda r: 2*r*np.exp(-r*r)*abs(r-.713), 0., .713, epsabs=1e-13)[0]
    exact += quad(lambda r: 2*r*np.exp(-r*r)*abs(r-.713), .713, 20., epsabs=1e-13)[0]
    assert actual[0, 0] == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("radius", (8., 10., 10.58, 12.))
def test_wide_bore_does_not_numerically_remove_transmitted_beam(radius):
    coarse = aperture_projection(radius, 1., 8, 32)
    fine = aperture_projection(radius, 1., 8, 96)
    np.testing.assert_allclose(coarse, fine, atol=2e-12, rtol=2e-12)
    eigen = np.linalg.eigvalsh(coarse)
    assert eigen.max() <= 1+2e-12
    assert eigen.min() >= 1-1e-8


def test_simultaneous_covariant_carrier_matches_independent_doubling():
    from temsim.physics.scattering_load import covariant_carrier_slab
    from temsim.physics.covariant_boundary import _slab
    residual = np.array([[1., .2], [.2, -1.]])
    connection = np.array([[.1, .4j], [-.4j, .3]])
    expected, _ = _slab(4*np.eye(2)+residual, connection, .7, 1.3, lambda: False)
    actual, record = covariant_carrier_slab(residual, connection, .7, 1.3, 2.)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    assert not record["connection_split"]


def test_covariant_small_phase_is_not_erased_at_accelerated_energy():
    from scipy.linalg import expm
    from temsim.physics.scattering_load import covariant_carrier_slab
    carrier, width = 3000., 6e14
    residual = np.array([[1., .2], [.2, -1.]])*1e-12
    connection = np.array([[.3, .15j], [-.15j, -.2]])*1e-16
    blocks, _ = covariant_carrier_slab(residual, connection, width, carrier, carrier)
    # Here the neglected next dispersion term is <1e-20 rad; the analytic
    # high-energy limit is a useful independent absolute complex check.
    expected = np.exp(1j*carrier*width)*expm(1j*width*(residual/(2*carrier)-connection))
    np.testing.assert_allclose(blocks[2], expected, atol=1e-11, rtol=1e-11)


def test_moving_basis_closure_keeps_derivative_outside_retained_modes():
    count, chirp, dilation_rate = 12, .21, -.34
    x, d, _ = laguerre_operators(count+1)
    full = chirp*x+1j*dilation_rate*d
    retained = full[:count, :count]
    direct_closure = (full@full)[:count, :count]-retained@retained
    exact = np.zeros((count, count)); exact[-1, -1] = count**2*(chirp**2+dilation_rate**2)
    np.testing.assert_allclose(direct_closure, exact, atol=1e-13)


def test_roundoff_equivalent_gun_planes_apply_absorption_once():
    from temsim.physics.radial_gun_wave import merge_gun_nodes
    plane = 18e6
    nodes = np.array([2., plane-1., plane, np.nextafter(plane, np.inf), plane+1., 450e6])
    merged, events = merge_gun_nodes(nodes, {18., 450.})
    assert len(merged) == 5
    assert list(events.values()) == [18e6, 450e6]
    assert np.all(np.diff(merged) > 0)
    # Distinct, resolvable planes remain distinct; this is not a physical
    # coarsening of aperture positions or a removed component.
    merged, events = merge_gun_nodes(nodes, {18., 18.+1e-9, 450.})
    assert len(events) == 3
    assert np.diff(list(events.values())).min() > 0
