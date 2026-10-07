"""Independent field/canonical consistency tests for posed analytical lenses."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import e
from scipy.spatial.transform import Rotation

from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.posed_lens_wave import posed_lens_correction, vector_potential_jet
from temsim.physics.tip_gun_wave import _momentum_velocity


J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])


def lens(rot=(.003, -.002, .07), offset=(2e-5, -3e-5, .5)):
    registration = CoordinateRegistration(tuple(offset), tuple(map(tuple, Rotation.from_rotvec(rot).as_matrix())))
    return FrozenAnalyticField("test_lens", ((.3, 0., .006), (-.08, .008, .004)),
                               registration, (-25., 25.), .003)


def test_rotated_vector_potential_curl_is_same_field_as_particles():
    field = lens()
    for point in ((0., 0., .5), (2e-4, -1e-4, .509), (-4e-5, 1e-5, .48)):
        _, derivative, _ = vector_potential_jet((field,), point)
        curl = np.array((derivative[2, 1]-derivative[1, 2],
                         derivative[0, 2]-derivative[2, 0],
                         derivative[1, 0]-derivative[0, 1]))
        np.testing.assert_allclose(curl, field.field_at_global_positions_t(point), rtol=2e-14, atol=1e-17)


def test_vector_potential_derivatives_match_independent_finite_differences():
    field, point, step = lens(), np.array((8e-5, -2e-4, .507)), 1e-7
    _, derivative, second = vector_potential_jet((field,), point)
    for axis in range(3):
        delta = np.eye(3)[axis]*step
        plus, plus_d, _ = vector_potential_jet((field,), point+delta)
        minus, minus_d, _ = vector_potential_jet((field,), point-delta)
        np.testing.assert_allclose(derivative[:, axis], (plus-minus)/(2*step), rtol=2e-7, atol=1e-12)
        np.testing.assert_allclose(second[:, :, axis], (plus_d-minus_d)/(2*step), rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(second, second.swapaxes(1, 2), atol=1e-14)


def test_zero_pose_reproduces_existing_axial_canonical_generator():
    field = lens(rot=(0., 0., 0.), offset=(0., 0., 0.))
    p0 = float(_momentum_velocity(200000.)[0])
    p = float(_momentum_velocity(300000.)[0])
    z, aligned = .004, .08
    b_posed = float(field.field_at_global_positions_t((0., 0., z))[2])
    actual = posed_lens_correction((field,), z, p, p0, aligned_b_t=aligned)

    def magnetic_generator(bz):
        b = -e*bz/2
        rotation = np.array(((0., b/p), (-b/p, 0.)))
        return np.block([[rotation, np.zeros((2, 2))],
                         [-np.eye(2)*b*b/(p*p0), rotation]])

    np.testing.assert_allclose(actual.generator, magnetic_generator(aligned+b_posed)-magnetic_generator(aligned),
                               rtol=2e-15, atol=1e-12)
    assert actual.constant_h == 0.
    np.testing.assert_array_equal(actual.force, 0.)
    assert actual.phase_error_bound((.001, .001), (.05, .05), .1, 2e-12) == 0.


def _exact_magnetic_h(field, z, position, momentum, p, p0, aligned=0.):
    value = vector_potential_jet((field,), (*position, z))[0]
    value += np.array((-.5*aligned*position[1], .5*aligned*position[0], 0.))
    baseline = np.array((-.5*aligned*position[1], .5*aligned*position[0]))
    # Subtract the already executed aligned magnetic operator, not its drift.
    full = (2*p0*momentum@(e*value[:2])+e*e*(value[:2]@value[:2]))/(2*p*p0)+e*value[2]/p0
    return full-(2*p0*momentum@(e*baseline)+e*e*(baseline@baseline))/(2*p*p0)


def test_affine_and_scalar_terms_are_retained_in_fixed_global_gauge():
    field, z, aligned = lens(), .502, .12
    p = float(_momentum_velocity(300000.)[0])
    result = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned)
    h0 = _exact_magnetic_h(field, z, np.zeros(2), np.zeros(2), p, p, aligned)
    assert result.constant_h == pytest.approx(h0, rel=2e-14)
    assert abs(h0) > 0.
    expected_linear = -J@result.force
    assert np.linalg.norm(result.force[:2]) > 0.  # Kinetic/canonical offset.
    for axis, step in enumerate((1e-9, 1e-9, 1e-7, 1e-7)):
        delta = np.eye(4)[axis]*step
        plus = _exact_magnetic_h(field, z, delta[:2], delta[2:], p, p, aligned)
        minus = _exact_magnetic_h(field, z, -delta[:2], -delta[2:], p, p, aligned)
        assert expected_linear[axis] == pytest.approx((plus-minus)/(2*step), rel=3e-7, abs=1e-10)
    np.testing.assert_allclose(result.generator.T@J+J@result.generator, 0., atol=1e-12)


def test_higher_order_error_bound_covers_actual_hamiltonian_on_support():
    field = lens(rot=(.06, -.04, .07))
    z, aligned = .505, .15
    p = float(_momentum_velocity(300000.)[0])
    result = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned)
    curvature, linear = -J@result.generator, -J@result.force
    q_bound, u_bound = np.array((8e-5, 6e-5)), np.array((.005, .008))
    bound = result.hamiltonian_error_bound(q_bound, u_bound)
    assert bound > 0.
    rng = np.random.default_rng(89431)
    maximum_error = 0.
    for point in rng.uniform(-1., 1., (64, 4))*np.r_[q_bound, u_bound]:
        exact = _exact_magnetic_h(field, z, point[:2], point[2:], p, p, aligned)
        quadratic = result.constant_h+linear@point+.5*point@curvature@point
        maximum_error = max(maximum_error, abs(exact-quadratic))
    assert maximum_error > 0.
    assert maximum_error <= bound
    # Budget changes with physical represented support, not checkpoint count.
    small = result.hamiltonian_error_bound(q_bound*.5, u_bound*.5)
    assert small < bound/5
    assert result.phase_error_bound(q_bound, u_bound, .002, 2e-12) == pytest.approx(
        2*result.phase_error_bound(q_bound, u_bound, .001, 2e-12), rel=1e-14)


def test_translated_parallel_lens_has_exact_quadratic_magnetic_operator():
    field = lens(rot=(0., 0., .3))
    p = float(_momentum_velocity(300000.)[0])
    result = posed_lens_correction((field,), .505, p, p)
    assert result.hamiltonian_error_bound((.003, .003), (.1, .1)) == 0.


def test_superposed_posed_fields_preserve_cross_terms():
    first = lens()
    second = replace(first, lens_key="second", terms_t_m=((.12, .001, .004),))
    p = float(_momentum_velocity(300000.)[0])
    total = posed_lens_correction((first, second), .503, p, p)
    a = vector_potential_jet((first, second), (0., 0., .503))[0]
    assert total.constant_h == pytest.approx(e*e*(a[:2]@a[:2])/(2*p*p)+e*a[2]/p)
    independent = sum(posed_lens_correction((field,), .503, p, p).constant_h for field in (first, second))
    assert abs(total.constant_h-independent) > 1e-8


def test_small_tilt_canonical_centre_matches_native_particle_law():
    """Matching-order comparison, not a claim of exact full Lorentz transport."""
    from scipy.integrate import solve_ivp
    field = replace(lens(rot=(0., .001, 0.), offset=(0., 0., .5)),
                    terms_t_m=((.3, 0., .006),))
    p = float(_momentum_velocity(300000.)[0])
    start, end = .47, .53
    position, slope = np.array((1e-6, -2e-6)), np.array((.0001, -.0002))
    potential = vector_potential_jet((field,), (*position, start))[0][:2]
    initial = np.r_[position, slope-e*potential/p]

    def canonical(z, state):
        correction = posed_lens_correction((field,), z, p, p)
        return np.r_[state[2:], np.zeros(2)]+correction.generator@state+correction.force

    def particle(z, state):
        acceleration = field.slope_derivative(np.array(((*state[:2], z),)), state[None, 2:], -e/p)[0]
        return np.r_[state[2:], acceleration]

    wave_centre = solve_ivp(canonical, (start, end), initial, rtol=1e-10, atol=1e-13).y[:, -1]
    ray = solve_ivp(particle, (start, end), np.r_[position, slope], rtol=1e-10, atol=1e-13).y[:, -1]
    wave_centre[2:] += e*vector_potential_jet((field,), (*wave_centre[:2], end))[0][:2]/p
    np.testing.assert_allclose(wave_centre[:2], ray[:2], rtol=0., atol=3e-11)
    np.testing.assert_allclose(wave_centre[2:], ray[2:], rtol=0., atol=5e-10)


def test_zero_excitation_is_identity_magnetic_correction():
    field = replace(lens(), terms_t_m=((0., 0., .006),))
    result = posed_lens_correction((field,), .5, 1e-22, 1e-22, aligned_b_t=.2)
    np.testing.assert_array_equal(result.generator, 0.)
    np.testing.assert_array_equal(result.force, 0.)
    assert result.constant_h == 0.
    assert result.hamiltonian_error_bound((.1, .1), (.1, .1)) == 0.


def test_support_bounds_distinguish_global_and_lens_local_forward_domains():
    p = float(_momentum_velocity(300000.)[0])
    small = posed_lens_correction((lens(),), .502, p, p)
    bounds = small.support_bounds((1e-6, 1e-6), (.001, .001))
    assert bounds["global_transverse_over_longitudinal_bound"] < .1
    assert bounds["kinetic_relative_remainder_bound"] < .0025
    assert bounds["local"][0]["forward_direction_lower_bound"] > 0.
    assert bounds["local"][0]["transverse_over_longitudinal_bound"] < .1
    backwards = lens(rot=(0., np.pi, 0.))
    rejected = posed_lens_correction((backwards,), .5, p, p).support_bounds((1e-6, 1e-6), (.001, .001))
    assert rejected["local"][0]["forward_direction_lower_bound"] < 0.
    assert rejected["local"][0]["transverse_over_longitudinal_bound"] == np.inf


def test_scalar_potential_and_posed_magnetic_cross_terms_match_exact_hamiltonian():
    from itertools import product
    field, z, energy, aligned = lens(), .504, 300000., .12
    p = float(_momentum_velocity(energy)[0])
    gradient = np.array((2e10, -1e10))
    hessian = np.array(((2e15, 8e14), (8e14, -1e15)))
    correction = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned,
        potential_gradient_v_m=gradient, potential_hessian_v_m2=hessian, kinetic_energy_ev=energy)
    curvature, linear = -J@correction.generator, -J@correction.force

    def exact(v):
        q, u = v[:2], v[2:]
        a = vector_potential_jet((field,), (*q, z))[0]
        baseline_a = np.array((-.5*aligned*q[1], .5*aligned*q[0]))
        a[:2] += baseline_a
        local_p = float(_momentum_velocity(energy+gradient@q+.5*q@hessian@q)[0])
        full = np.sum((p*u+e*a[:2])**2)/(2*local_p*p)+e*a[2]/p
        baseline = np.sum((p*u+e*baseline_a)**2)/(2*p*p)
        return full-baseline

    steps = np.array((1e-9, 1e-9, 1e-6, 1e-6))
    origin = exact(np.zeros(4))
    for i in range(4):
        di = np.eye(4)[i]*steps[i]
        assert linear[i] == pytest.approx((exact(di)-exact(-di))/(2*steps[i]), rel=2e-7, abs=1e-10)
        for j in range(4):
            dj = np.eye(4)[j]*steps[j]
            finite_second = (exact(di+dj)-exact(di-dj)-exact(-di+dj)+exact(-di-dj))/(4*steps[i]*steps[j])
            assert curvature[i, j] == pytest.approx(finite_second, rel=8e-5, abs=2e-7)
    assert correction.constant_h == pytest.approx(origin)
    q_bound, u_bound = np.array((2e-7, 3e-7)), np.array((.001, .002))
    bound = correction.hamiltonian_error_bound(q_bound, u_bound)
    for signs in product((-1., 1.), repeat=4):
        point = np.r_[q_bound, u_bound]*signs
        polynomial = origin+linear@point+.5*point@curvature@point
        assert abs(exact(point)-polynomial) <= bound
    with pytest.raises(ValueError, match="zero kinetic energy"):
        correction.hamiltonian_error_bound((1e-4, 1e-4), u_bound)


def test_moving_centre_matches_exact_magnetic_jet_and_bounds_in_local_box():
    from itertools import product
    field, z, aligned = lens(rot=(.008, -.006, .07)), .505, .15
    p = float(_momentum_velocity(300000.)[0])
    centre = np.array((3e-5, -2e-5, .002, -.001))
    correction = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned).reexpanded(centre)
    np.testing.assert_array_equal(correction.center_phase_space, centre)
    curvature, linear = -J@correction.generator, -J@correction.force
    def exact(v):
        return _exact_magnetic_h(field, z, v[:2], v[2:], p, p, aligned)
    def polynomial(v):
        return correction.constant_h+linear@v+.5*v@curvature@v
    assert polynomial(centre) == pytest.approx(exact(centre), rel=1e-13)
    gradient_at_centre = linear+curvature@centre
    # q steps resolve the small mixed Hessian above floating-point cancellation.
    steps = np.array((1e-7, 1e-7, 1e-5, 1e-5))
    for i in range(4):
        di = np.eye(4)[i]*steps[i]
        assert gradient_at_centre[i] == pytest.approx((exact(centre+di)-exact(centre-di))/(2*steps[i]),
                                                      rel=3e-7, abs=2e-10)
        for j in range(4):
            dj = np.eye(4)[j]*steps[j]
            second = (exact(centre+di+dj)-exact(centre+di-dj)-exact(centre-di+dj)+exact(centre-di-dj))/(4*steps[i]*steps[j])
            assert curvature[i, j] == pytest.approx(second, rel=2e-4, abs=2e-6)
    q_bound, u_bound = np.array((3e-6, 2e-6)), np.array((.0001, .0002))
    bound = correction.hamiltonian_error_bound(q_bound, u_bound)
    for signs in product((-1., 1.), repeat=4):
        point = centre+np.r_[q_bound, u_bound]*signs
        assert abs(exact(point)-polynomial(point)) <= bound


def test_recentring_preserves_exact_parallel_operator_and_electric_fallback():
    p = float(_momentum_velocity(300000.)[0])
    centre = np.array((3e-5, -2e-5, .002, -.001))
    initial = posed_lens_correction((lens(rot=(0., 0., 0.)),), .505, p, p, aligned_b_t=.1)
    moved = initial.reexpanded(centre)
    np.testing.assert_allclose(moved.generator, initial.generator, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(moved.force, initial.force, rtol=1e-13, atol=1e-13)
    assert moved.constant_h == pytest.approx(initial.constant_h, rel=1e-12)
    electric = posed_lens_correction((lens(),), .505, p, p,
        potential_gradient_v_m=(1e6, 0.), kinetic_energy_ev=300000.)
    retained = electric.reexpanded(centre)
    np.testing.assert_array_equal(retained.center_phase_space, np.zeros(4))
    np.testing.assert_array_equal(retained.generator, electric.generator)
    np.testing.assert_array_equal(retained.force, electric.force)


def test_exact_magnetic_residual_matches_independent_hamiltonian_and_momentum_derivative():
    from temsim.physics.multiplane_wave import PlaneWave
    field, z, aligned = lens(rot=(.045, -.03, .07)), .505, .15
    p = float(_momentum_velocity(300000.)[0])
    centre = np.array((3e-5, -2e-5, .002, -.001))
    correction = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned,
                                      center_phase_space=centre)
    wave = PlaneWave(np.ones((4, 4), complex), np.array(((4e-5, 7e-6), (-3e-6, 3e-5))),
                     np.array((2e-4, -1e-4)), np.array(((12., 3.), (3., -7.))), np.array((.004, -.003)))
    velocity, scalar = correction.residual_coefficients(wave, 2e-12)
    curvature, linear = -J@correction.generator, -J@correction.force
    xy = wave.coordinates_m()
    carrier = wave.tilt_rad[:, None, None]+np.einsum(
        "ij,jyx->iyx", wave.curvature_m1, xy-wave.origin_m[:, None, None])

    def residual(q, u):
        v = np.r_[q, u]
        polynomial = correction.constant_h+linear@v+.5*v@curvature@v
        return _exact_magnetic_h(field, z, q, u, p, p, aligned)-polynomial

    for row, col in ((0, 0), (1, 3), (3, 1), (3, 3)):
        q, u = xy[:, row, col], carrier[:, row, col]
        assert scalar[row, col] == pytest.approx(residual(q, u), rel=4e-7, abs=2e-17)
        for axis in range(2):
            delta = np.eye(2)[axis]*.01
            derivative = (residual(q, u+delta)-residual(q, u-delta))/.02
            assert velocity[axis, row, col] == pytest.approx(derivative, rel=5e-7, abs=2e-15)
    # A tilted native field has a nonzero derivative residual: omitting this
    # term and applying only a scalar phase is not the same Hamiltonian.
    assert np.max(abs(velocity)) > 1e-8


def test_exact_magnetic_residual_zero_for_parallel_lens_and_rejects_varying_electric_field():
    from temsim.physics.multiplane_wave import PlaneWave
    p = float(_momentum_velocity(300000.)[0])
    wave = PlaneWave(np.ones((4, 4), complex), np.eye(2)*1e-6, np.array((1e-4, -2e-4)))
    correction = posed_lens_correction((lens(rot=(0., 0., .3)),), .5, p, p, aligned_b_t=.1)
    velocity, scalar = correction.residual_coefficients(wave, 2e-12)
    np.testing.assert_array_equal(velocity, 0.)
    np.testing.assert_array_equal(scalar, 0.)
    varying = posed_lens_correction((lens(),), .5, p, p, potential_gradient_v_m=(1., 0.))
    with pytest.raises(ValueError, match="varying electric-magnetic residuals"):
        varying.residual_coefficients(wave, 2e-12)


def test_imported_maps_and_invalid_support_are_rejected():
    with pytest.raises(ValueError, match="vector potential"):
        posed_lens_correction((object(),), .5, 1e-22, 1e-22)
    result = posed_lens_correction((lens(),), .5, 1e-22, 1e-22)
    with pytest.raises(ValueError, match="nonnegative"):
        result.hamiltonian_error_bound((np.inf, 1.), (.1, .1))
    with pytest.raises(ValueError, match="positive momenta"):
        posed_lens_correction((lens(),), .5, 0., 1e-22)


def test_axis_momentum_magnetic_tail_keeps_evidence_and_allows_same_residual():
    from temsim.physics.multiplane_wave import PlaneWave
    p = float(_momentum_velocity(300000.)[0])
    centre = np.array((30e-6, -20e-6, .002, -.001))
    hessian = np.diag((-1.0915, -1.0915))
    actual = posed_lens_correction((lens(),), .505, p, p, aligned_b_t=.1,
        potential_hessian_v_m2=hessian, axis_momentum_magnetic=True).reexpanded(centre)
    constant = posed_lens_correction((lens(),), .505, p, p, aligned_b_t=.1).reexpanded(centre)
    np.testing.assert_array_equal(actual.potential_hessian_v_m2, hessian)
    assert not actual.potential_hessian_v_m2.flags.writeable
    np.testing.assert_array_equal(actual.center_phase_space, centre)
    for name in ("generator", "force", "constant_h"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(constant, name))
    assert actual.supports_magnetic_residual
    wave = PlaneWave(np.ones((4, 4), complex), np.eye(2)*5e-9, centre[:2],
                     np.zeros((2, 2)), centre[2:])
    for a, b in zip(actual.residual_coefficients(wave, 2e-12), constant.residual_coefficients(wave, 2e-12)):
        np.testing.assert_array_equal(a, b)
    q_bound, u_bound = np.full(2, 2e-7), np.full(2, 1e-4)
    bound = actual.unresolved_electric_phase_error_bound(q_bound, u_bound, .001, 2e-12)
    assert 0 < bound < .01*.001
    assert actual.phase_error_bound(q_bound, u_bound, .001, 2e-12) >= bound


def test_axis_momentum_magnetic_omission_bounds_exact_cross_term_at_global_positions():
    from itertools import product
    p = float(_momentum_velocity(300000.)[0])
    centre = np.array((30e-6, -20e-6, .002, -.001))
    gradient, hessian = np.array((1e8, -2e8)), np.diag((1e10, -2e10))
    field, aligned, z = lens(), .1, .505
    actual = posed_lens_correction((field,), z, p, p, aligned_b_t=aligned,
        potential_gradient_v_m=gradient, potential_hessian_v_m2=hessian,
        axis_momentum_magnetic=True).reexpanded(centre)
    q_bound, u_bound = np.full(2, 1e-6), np.full(2, 1e-4)
    bound = actual.unresolved_electric_hamiltonian_error_bound(q_bound, u_bound)
    assert bound > 0
    for signs in product((-1., 1.), repeat=4):
        point = centre+np.r_[q_bound, u_bound]*signs
        q, u = point[:2], point[2:]
        potential = gradient@q+.5*q@hessian@q
        local_p = float(_momentum_velocity(300000.+potential)[0])
        a = vector_potential_jet((field,), (*q, z))[0][:2]
        a += .5*aligned*np.array((-q[1], q[0]))
        exact_difference = (1/local_p-1/p)*(e*a@u+e*e*(a@a)/(2*p))
        assert abs(exact_difference) <= bound
    # This term is not corrected by magnetic spatial refinement. The caller
    # must retain it in the gate and final record even after a residual step.
    phase_bound = actual.unresolved_electric_phase_error_bound(q_bound, u_bound, .001, 2e-12)
    assert phase_bound > .01*.001
    assert actual.supports_magnetic_residual  # Does not imply budget admission.
    invalid = posed_lens_correction((field,), z, p, p,
        potential_gradient_v_m=(2e10, 0.), axis_momentum_magnetic=True).reexpanded(centre)
    with pytest.raises(ValueError, match="zero kinetic energy"):
        invalid.unresolved_electric_phase_error_bound(q_bound, u_bound, .001, 2e-12)
    with pytest.raises(ValueError, match="zero kinetic energy"):
        invalid.support_bounds(q_bound, u_bound)
