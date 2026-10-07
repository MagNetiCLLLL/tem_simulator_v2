"""Native multipole wave jets, finite support, and full magnetic residual."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.constants import e
from scipy.spatial.transform import Rotation

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.posed_lens_wave import posed_lens_correction, vector_potential_jet
from temsim.physics.posed_multipole_wave import (
    multipole_potential_remainder_bound, multipole_vector_potential_remainders,
)
from temsim.physics.tip_gun_wave import _momentum_velocity


POLYNOMIALS = (((1, 0, -.02), (0, 1, .013)),
               ((2, 0, 70.), (0, 2, -70.), (1, 1, 30.)),
               ((3, 0, 1e6), (1, 2, -3e6), (2, 1, .8e6), (0, 3, -.8e6/3)))
J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])


def field(polynomial=1, kind="gaussian"):
    registration = CoordinateRegistration((2e-5, -3e-5, .5),
        tuple(map(tuple, Rotation.from_rotvec((.018, -.014, .07)).as_matrix())))
    return FrozenPosedMultipole("posed-test", "posed-test", registration,
        POLYNOMIALS[polynomial], kind, .004, .006, (-25., 25.), .003,
        float(_momentum_velocity(300000.)[0]))


def centre_global(provider, local_z=.005):
    return provider.registration.origin_array_m+provider.registration.rotation_array@np.array((1.2e-4, -7e-5, local_z))


def test_zero_pose_quadrupole_matches_native_scalar_hamiltonian_exactly():
    provider = replace(field(1, "uniform"), registration=CoordinateRegistration())
    p = float(_momentum_velocity(300000.)[0])
    correction = posed_lens_correction((provider,), .004, p, p)
    expected = np.zeros((4, 4))
    expected[2:, :2] = -e/p*np.array(((140., 30.), (30., -140.)))
    np.testing.assert_allclose(correction.generator, expected, rtol=2e-15, atol=0.)
    np.testing.assert_array_equal(correction.force, np.zeros(4))
    assert correction.constant_h == 0.
    assert correction.phase_error_bound((1e-4, 1e-4), (.001, .001), .001, 2e-12) == 0.


@pytest.mark.parametrize("polynomial", range(3))
@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_stable_native_residuals_match_exact_vector_potential_and_bound(polynomial, kind):
    provider = field(polynomial, kind)
    centre = centre_global(provider)
    xy_bound = np.array((1.1e-4, 9e-5))
    rng = np.random.default_rng(622)
    delta = rng.uniform(-1., 1., (2, 8, 9))*xy_bound[:, None, None]
    second, third = multipole_vector_potential_remainders(provider, centre, delta, np)
    points = np.moveaxis(np.concatenate((delta, np.zeros((1, 8, 9))), axis=0), 0, -1)+centre
    exact = np.moveaxis(provider.vector_potential_at_global_positions_t_m(points), -1, 0)
    a, d, dd = vector_potential_jet((provider,), centre)
    first = np.einsum("ij,jyx->iyx", d[:, :2], delta)
    quadratic = .5*np.einsum("ijk,jyx,kyx->iyx", dd[:, :2, :2], delta, delta)
    np.testing.assert_allclose(second, exact-a[:, None, None]-first, atol=3e-18, rtol=5e-8)
    np.testing.assert_allclose(third, exact-a[:, None, None]-first-quadratic, atol=3e-18, rtol=5e-8)
    bound, axial_bound = multipole_potential_remainder_bound(provider, centre, xy_bound)
    assert np.max(np.linalg.norm(third, axis=0)) <= bound+3e-18
    assert np.max(abs(third[2])) <= axial_bound+3e-18


@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_oblique_native_support_edge_has_finite_bound_and_exact_masked_residual(kind):
    provider = field(0, kind)
    centre = centre_global(provider, local_z=.025)
    xy_bound = np.array((1e-4, 1e-4))
    delta = np.array(tuple(product((-xy_bound[0], xy_bound[0]), (-xy_bound[1], xy_bound[1])))).T.reshape(2, 2, 2)
    second, third = multipole_vector_potential_remainders(provider, centre, delta, np)
    points = np.moveaxis(np.concatenate((delta, np.zeros((1, 2, 2))), axis=0), 0, -1)+centre
    exact = np.moveaxis(provider.vector_potential_at_global_positions_t_m(points), -1, 0)
    assert np.any(np.linalg.norm(exact, axis=0) == 0.)
    assert np.any(np.linalg.norm(exact, axis=0) > 0.)
    a, d, dd = provider.vector_potential_jet(centre)
    expected = exact-a[:, None, None]-np.einsum("ij,jyx->iyx", d[:, :2], delta)
    np.testing.assert_allclose(second, expected, atol=2e-18, rtol=1e-10)
    expected -= .5*np.einsum("ijk,jyx,kyx->iyx", dd[:, :2, :2], delta, delta)
    np.testing.assert_allclose(third, expected, atol=2e-18, rtol=1e-10)
    bound, axial = multipole_potential_remainder_bound(provider, centre, xy_bound)
    assert np.isfinite(bound) and bound > 0
    assert np.max(np.linalg.norm(third, axis=0)) <= bound
    assert np.max(abs(third[2])) <= axial


def test_mixed_native_multipoles_preserve_total_a_squared_cross_terms_and_residual():
    first = field(0)
    second = replace(field(2), lens_key="second", component_key="second",
        registration=CoordinateRegistration((-1e-5, 2e-5, .5),
            tuple(map(tuple, Rotation.from_rotvec((-.01, .02, -.03)).as_matrix()))))
    centre = centre_global(first)
    p = float(_momentum_velocity(300000.)[0])
    correction = posed_lens_correction((first, second), centre[2], p, p,
                                      center_phase_space=np.r_[centre[:2], [.002, -.001]])
    wave = PlaneWave(np.ones((8, 8), complex)/8, np.eye(2)*20e-6, centre[:2],
                     np.array(((2., .2), (.2, -.5))), np.array((.002, -.001)))
    velocity, scalar = correction.residual_coefficients(wave, 2e-12)
    xy = wave.coordinates_m()
    points = np.moveaxis(np.concatenate((xy, np.full((1, 8, 8), centre[2])), axis=0), 0, -1)
    a = sum(np.moveaxis(item.vector_potential_at_global_positions_t_m(points), -1, 0) for item in (first, second))
    dq = xy-wave.origin_m[:, None, None]
    carrier = wave.tilt_rad[:, None, None]+np.einsum("ij,jyx->iyx", wave.curvature_m1, dq)
    canonical = np.concatenate((xy, carrier))
    exact = e*np.sum(a[:2]*carrier, axis=0)/p+e*e*np.sum(a[:2]**2, axis=0)/(2*p*p)+e*a[2]/p
    quadratic = (correction.constant_h+np.einsum("i,iyx->yx", -J@correction.force, canonical)
        +.5*np.einsum("iyx,ij,jyx->yx", canonical, -J@correction.generator, canonical))
    np.testing.assert_allclose(scalar, exact-quadratic, atol=1e-15, rtol=1e-7)
    expected_velocity = e/p*(a[:2]-correction.vector_potential[:2, None, None]
        -np.einsum("ij,jyx->iyx", correction.potential_gradient[:2, :2], dq))
    np.testing.assert_allclose(velocity, expected_velocity, atol=1e-16, rtol=1e-8)
    combined_origin = posed_lens_correction((first, second), centre[2], p, p).constant_h
    independent_origins = sum(posed_lens_correction((item,), centre[2], p, p).constant_h for item in (first, second))
    a1, a2 = (item.vector_potential_at_global_positions_t_m((0., 0., centre[2]))[:2]
              for item in (first, second))
    expected_cross = e*e*(a1@a2)/(p*p)
    subtraction_roundoff = 64*np.finfo(float).eps*abs(combined_origin)
    assert abs(expected_cross) > 100*subtraction_roundoff
    assert combined_origin-independent_origins == pytest.approx(expected_cross, abs=subtraction_roundoff)


def test_real_gpu_native_multipole_residual_matches_cpu():
    from test_wave_device import _cuda
    from temsim.physics.wave_device import device_scope, to_host
    cp = _cuda()
    provider = field(2)
    centre = centre_global(provider)
    p = float(_momentum_velocity(300000.)[0])
    correction = posed_lens_correction((provider,), centre[2], p, p,
        center_phase_space=np.r_[centre[:2], [.002, -.001]])
    wave = PlaneWave(np.ones((8, 8), complex)/8, np.eye(2)*10e-6, centre[:2],
                     np.array(((2., .2), (.2, -.5))), np.array((.002, -.001)))
    expected = correction.residual_coefficients(wave, 2e-12)
    with device_scope("Require GPU") as device:
        assert device.backend == "cupy"
        actual = correction.residual_coefficients(replace(wave, amplitude=cp.asarray(wave.amplitude)), 2e-12)
        for host, gpu in zip(expected, actual):
            assert hasattr(gpu, "__cuda_array_interface__")
            np.testing.assert_allclose(to_host(gpu), host, atol=1e-17, rtol=2e-12)
