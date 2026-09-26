"""Analytical magnetic-only fixtures; these do not qualify the TEM chain."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import MagneticSceneField, _Source
from temsim.magnetic_test_particle import (
    TestElectronSettings, electron_momentum_and_speed, trace_test_electron,
)
from temsim.physics.relativistic_lorentz import ELEMENTARY_CHARGE_C


def _settings(**kwargs):
    kwargs.setdefault("kinetic_energy_ev", 300_000.)
    return TestElectronSettings(**kwargs)


class UniformScene:
    def __init__(self, field=(0., 0., 0.), bounds=((-1., -1., -1.), (1., 1., 1.))):
        self.field = np.array(field)
        self.bounds_m = np.array(bounds)
        self.queries = []

    def contains(self, points):
        return ((points >= self.bounds_m[0]) & (points <= self.bounds_m[1])).all(axis=1)

    def field_at_global_positions_t(self, points):
        assert self.contains(points).all(), "No extrapolation is allowed"
        self.queries.append(points.copy())
        return np.broadcast_to(self.field, points.shape).copy()


def exact_uniform_z(settings, field_z, lengths):
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    # Negative charge: n initially +X turns toward +Y for B along +Z.
    omega = ELEMENTARY_CHARGE_C*field_z/p
    theta, phi = np.deg2rad((settings.polar_angle_deg, settings.azimuth_angle_deg))
    phase = phi+omega*lengths
    xy = np.sin(theta)/omega*np.column_stack((np.sin(phase)-np.sin(phi), np.cos(phi)-np.cos(phase)))
    return np.column_stack((xy, np.cos(theta)*lengths))+settings.position_m


@pytest.mark.parametrize("polar, azimuth", [(0., 0.), (90., 0.), (90., 90.), (180., 0.), (130., -37.)])
def test_zero_field_straight_line_with_backward_and_transverse_emission(polar, azimuth):
    settings = _settings(position_m=(.01, .02, .03), polar_angle_deg=polar,
                                    azimuth_angle_deg=azimuth, max_path_length_m=.005, step_m=.0003)
    result = trace_test_electron(UniformScene(), settings)
    assert result.reason == "path_limit" and result.completed
    expected = np.asarray(settings.position_m)+result.path_length_m[:, None]*result.directions[0]
    np.testing.assert_allclose(result.positions_m, expected, atol=3e-16)
    assert result.path_length_m[-1] == pytest.approx(.005)
    np.testing.assert_allclose(result.time_s, result.path_length_m/result.speed_m_per_s, rtol=1e-15)
    assert result.energy_invariant_relative_error < 1e-14


@pytest.mark.parametrize("field_z", [.12, -.12])
def test_uniform_field_helix_radius_charge_sign_pitch_and_energy(field_z):
    settings = _settings(polar_angle_deg=63., azimuth_angle_deg=14., max_path_length_m=.04, step_m=2e-5)
    result = trace_test_electron(UniformScene((0., 0., field_z)), settings)
    expected = exact_uniform_z(settings, field_z, result.path_length_m)
    np.testing.assert_allclose(result.positions_m, expected, atol=1e-8, rtol=0.)
    assert result.energy_invariant_relative_error < 3e-14
    assert result.reason == "path_limit"
    # p_perp / |q B| is the independent analytical cyclotron radius.
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    radius = p*np.sin(np.deg2rad(settings.polar_angle_deg))/(ELEMENTARY_CHARGE_C*abs(field_z))
    signed_radius = np.copysign(radius, field_z)
    phi = np.deg2rad(settings.azimuth_angle_deg)
    center = np.array((-signed_radius*np.sin(phi), signed_radius*np.cos(phi)))
    np.testing.assert_allclose(np.linalg.norm(result.positions_m[:, :2]-center, axis=1), radius, atol=1e-13)


def test_higher_energy_curvature_decreases_without_modifying_fixed_field():
    scene = UniformScene((0., .05, 0.))
    settings = _settings(max_path_length_m=.001, step_m=1e-5)
    low = trace_test_electron(scene, replace(settings, kinetic_energy_ev=20_000.))
    high = trace_test_electron(scene, replace(settings, kinetic_energy_ev=300_000.))
    assert low.positions_m[-1, 0] > high.positions_m[-1, 0] > 0.
    np.testing.assert_array_equal(scene.field, (0., .05, 0.))


def test_parallel_field_is_straight_at_any_magnetic_strength():
    result = trace_test_electron(UniformScene((0., 0., 10.)), _settings(max_path_length_m=.001))
    np.testing.assert_array_equal(result.positions_m[:, :2], 0.)
    np.testing.assert_allclose(result.positions_m[:, 2], result.path_length_m)


def test_step_halving_gives_second_order_global_error():
    settings = _settings(polar_angle_deg=64., max_path_length_m=.015, step_m=4e-4)
    errors = []
    for divisor in (1., 2., 4.):
        result = trace_test_electron(UniformScene((0., 0., .03)), replace(settings, step_m=settings.step_m/divisor))
        exact = exact_uniform_z(settings, .03, np.array((settings.max_path_length_m,)))[0]
        errors.append(np.linalg.norm(result.positions_m[-1]-exact))
    assert 3.7 < errors[0]/errors[1] < 4.3
    assert 3.7 < errors[1]/errors[2] < 4.3


def test_spatially_varying_field_converges_to_independent_dop853_solution():
    from scipy.integrate import solve_ivp

    class VaryingField(UniformScene):
        def field_at_global_positions_t(self, points):
            # div B = 0: By varies with Z, not with Y.
            values = np.zeros_like(points)
            values[:, 1] = .01*(1.+20.*points[:, 2])
            return values

    settings = _settings(polar_angle_deg=27., azimuth_angle_deg=19.,
                                    max_path_length_m=.02, step_m=.001)
    scene = VaryingField()
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    theta, phi = np.deg2rad((settings.polar_angle_deg, settings.azimuth_angle_deg))
    initial = (*settings.position_m, np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))
    def derivative(_s, values):
        field = scene.field_at_global_positions_t(values[None, :3])[0]
        return np.concatenate((values[3:], -ELEMENTARY_CHARGE_C/p*np.cross(values[3:], field)))
    reference = solve_ivp(derivative, (0., settings.max_path_length_m), initial, method="DOP853",
                          rtol=2e-13, atol=1e-15)
    assert reference.success
    errors = []
    for divisor in (1., 2., 4.):
        result = trace_test_electron(scene, replace(settings, step_m=settings.step_m/divisor))
        errors.append(np.linalg.norm(result.positions_m[-1]-reference.y[:3, -1]))
        assert result.energy_invariant_relative_error < 1e-14
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1
    assert errors[-1] < 1e-8


def test_returning_vz_is_allowed_and_not_used_as_independent_coordinate():
    scene = UniformScene((0., .1, 0.))
    settings = _settings(max_path_length_m=.15, step_m=1e-4)
    result = trace_test_electron(scene, settings)
    assert result.reason == "path_limit"
    assert result.directions[:, 2].min() < -.99
    assert np.any(np.diff(result.positions_m[:, 2]) < 0.)
    assert np.all(np.diff(result.path_length_m) > 0.)


def test_domain_exit_never_extrapolates_and_result_arrays_are_immutable():
    scene = UniformScene(bounds=((-1., -1., 0.), (1., 1., .001)))
    result = trace_test_electron(scene, _settings(max_path_length_m=.1, step_m=.0003))
    assert result.reason == "domain_exit" and result.completed
    assert result.positions_m[-1, 2] == pytest.approx(.001, abs=2e-13)
    for values in (result.positions_m, result.directions, result.time_s, result.path_length_m):
        assert not values.flags.writeable


def test_initial_invalid_state_and_cancellation_are_explicit():
    scene = UniformScene()
    result = trace_test_electron(scene, _settings(position_m=(2., 0., 0.)))
    assert result.reason == "initial_outside_domain" and result.steps == 0 and not result.completed
    result = trace_test_electron(scene, _settings(), cancelled=lambda: True)
    assert result.reason == "cancelled" and result.steps == 0 and not result.completed
    calls = [0]
    def cancelled():
        calls[0] += 1
        return calls[0] > 9
    result = trace_test_electron(scene, _settings(), cancelled=cancelled)
    assert result.reason == "cancelled" and 0 < result.steps < 10 and not result.completed


def test_step_budget_is_not_reported_as_completion():
    result = trace_test_electron(UniformScene(), _settings(max_steps=2))
    assert result.steps == 2 and result.reason == "step_limit" and not result.completed


@pytest.mark.parametrize("kwargs", [{"kinetic_energy_ev": 0.}, {"kinetic_energy_ev": np.nan},
    {"position_m": (0., 0.)}, {"position_m": (0., np.inf, 0.)}, {"polar_angle_deg": -1.},
    {"polar_angle_deg": 181.}, {"azimuth_angle_deg": np.nan}, {"max_path_length_m": 0.},
    {"step_m": -1.}, {"max_steps": True}, {"max_steps": 1.5}, {"max_steps": 0}])
def test_invalid_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _settings(**kwargs)


def captured_scene(*sources, bounds=((-1., -1., 0.), (1., 1., .01))):
    return MagneticSceneField(np.array(bounds), 1., tuple(s.key for s in sources), (),
                             tuple(s.bounds_m for s in sources), tuple(sources), np.array(bounds))


def test_known_axial_gaps_are_zero_but_active_missing_radial_field_is_unknown():
    provider = UniformScene((0., 0., .01))
    bounds = np.array(((-.001, -.001, .003), (.001, .001, .004)))
    source = _Source("coil", provider, bounds, .001)
    scene = captured_scene(source)
    assert scene.diagnostic_field_at_global_position_t((.002, 0., .002)) is not None
    assert scene.diagnostic_field_at_global_position_t((.002, 0., .0035)) is None
    result = trace_test_electron(scene, _settings(position_m=(.002, 0., 0.), step_m=.005))
    assert result.reason == "domain_exit"
    assert result.positions_m[-1, 2] <= .003


def test_missing_native_map_support_is_unknown_and_known_zero_sources_do_not_restrict():
    provider = UniformScene((0., 0., .01))
    bounds = np.array(((-.001, -.001, .003), (.001, .001, .004)))
    field_map = SimpleNamespace(map_type="axisymmetric_rz",
        registration=SimpleNamespace(positions_global_to_local_m=lambda points: points),
        axes_m=(np.array((0., .0002)), np.array((.003, .004))))
    scene = captured_scene(_Source("mapped", provider, bounds, .001, field_map=field_map))
    assert scene.diagnostic_position_is_valid((.0001, 0., .0035))
    assert not scene.diagnostic_position_is_valid((.0003, 0., .0035))
    assert scene.diagnostic_field_at_global_position_t((.0003, 0., .0035)) is None
    zero = captured_scene(_Source("zero", provider, bounds, .001, known_zero=True))
    assert zero.diagnostic_position_is_valid((.002, 0., .0035))
    np.testing.assert_array_equal(zero.diagnostic_field_at_global_position_t((.002, 0., .0035)), 0.)


def test_step_cannot_jump_over_a_narrow_nonzero_field_region():
    provider = UniformScene((0., .1, 0.))
    bounds = np.array(((-.01, -.01, .003), (.01, .01, .00301)))
    scene = captured_scene(_Source("coil", provider, bounds, .01))
    settings = _settings(max_path_length_m=.007, step_m=.005)
    result = trace_test_electron(scene, settings)
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    expected = ELEMENTARY_CHARGE_C*.1*.00001/p
    assert result.directions[-1, 0] == pytest.approx(expected, rel=1e-5)


def test_transverse_launch_cannot_gyrate_across_a_thin_coil_in_one_step():
    provider = UniformScene((0., .1, 0.))
    bounds = np.array(((-.01, -.01, 0.), (.01, .01, 1e-7)))
    scene = captured_scene(_Source("coil", provider, bounds, .01),
                           bounds=((- .1, -.1, -.01), (.1, .1, .01)))
    settings = _settings(position_m=(0., 0., 5e-8), polar_angle_deg=90.,
                                    max_path_length_m=.002, step_m=.001, max_steps=200_000)
    result = trace_test_electron(scene, settings)
    fine = trace_test_electron(scene, replace(settings, step_m=1e-7))
    assert result.reason == fine.reason == "path_limit"
    assert result.directions[-1, 2] == pytest.approx(fine.directions[-1, 2], rel=2e-5, abs=1e-8)
    # Uniform-B circle reaches Z=0 after a drop of 50 nm; this fixes its
    # exit direction independently of either numerical step selection.
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    radius = p/(ELEMENTARY_CHARGE_C*.1)
    expected = -np.sqrt(1.-(1.-5e-8/radius)**2)
    assert result.directions[-1, 2] == pytest.approx(expected, rel=2e-5)


def test_native_map_transverse_grid_feature_is_not_skipped_by_large_requested_step():
    from temsim.physics.lens_field_provider import (
        CoordinateRegistration, MagneticFieldMap, FieldMapProvenance,
    )
    axes = (np.array((-.01, 0., .0002, .00025, .0003, .002, .01)),
            np.array((-.01, .01)), np.array((-.01, .01)))
    zero = np.zeros((7, 2, 2))
    by = zero.copy()
    by[3, :, :] = .1
    field_map = MagneticFieldMap("cartesian_xyz", axes, (zero, by, zero),
        CoordinateRegistration(), "synthetic", 100., 1,
        FieldMapProvenance("fem", "synthetic", "0"*64, "Numerical review fixture"))
    bounds = np.array(((-.01, -.01, -.01), (.01, .01, .01)))
    scene = captured_scene(_Source("map", field_map, bounds, .01, field_map=field_map), bounds=bounds)
    settings = _settings(polar_angle_deg=90., max_path_length_m=.001, step_m=.001)
    result = trace_test_electron(scene, settings)
    fine = trace_test_electron(scene, replace(settings, step_m=1e-6))
    assert result.reason == fine.reason == "path_limit"
    assert result.directions[-1, 2] == pytest.approx(-.00238021, rel=2e-4)
    assert result.directions[-1, 2] == pytest.approx(fine.directions[-1, 2], rel=2e-4)
    assert result.steps >= 40


def test_invalid_provider_output_is_not_rendered_as_a_zero_field():
    scene = UniformScene((0., np.nan, 0.))
    with pytest.raises(ValueError, match="invalid XYZ"):
        trace_test_electron(scene, _settings())


class UniformElectromagneticScene(UniformScene):
    has_electric_field = True

    def __init__(self, electric=(0., 0., 0.), magnetic=(0., 0., 0.), **kwargs):
        super().__init__(magnetic, **kwargs)
        self.electric = np.asarray(electric, float)

    def diagnostic_fields_at_global_position(self, point):
        if not self.contains(point[None, :])[0]:
            return None
        return self.field, self.electric, -float(self.electric@point)


def test_uniform_electric_acceleration_from_sub_ev_emission_obeys_work_and_flight_time():
    scene = UniformElectromagneticScene(electric=(0., 0., -1e6))
    settings = _settings(kinetic_energy_ev=.3, max_path_length_m=.003, step_m=.001)
    result = trace_test_electron(scene, settings)
    assert result.reason == "path_limit"
    assert result.kinetic_energy_ev[-1] == pytest.approx(3000.3, abs=1e-5)
    assert result.energy_invariant_error_ev < 1e-7
    p0, _ = electron_momentum_and_speed(.3)
    p1, _ = electron_momentum_and_speed(3000.3)
    expected_time = (p1-p0)/(ELEMENTARY_CHARGE_C*1e6)
    assert result.time_s[-1] == pytest.approx(expected_time, rel=3e-5)
    assert result.speed_m_per_s[-1] > 50.*result.speed_m_per_s[0]
    assert result.positions_m[-1, 2] == pytest.approx(.003, abs=1e-11)
    assert np.all(np.diff(result.kinetic_energy_ev) >= 0.)


def test_electric_retardation_turns_and_returns_without_negative_kinetic_energy():
    scene = UniformElectromagneticScene(electric=(0., 0., 1e5))
    settings = _settings(kinetic_energy_ev=10., max_path_length_m=.0003, step_m=.0001)
    result = trace_test_electron(scene, settings)
    assert result.reason == "path_limit"
    assert np.any(result.directions[:, 2] < 0.)
    assert result.positions_m[:, 2].max() == pytest.approx(.0001, abs=2e-8)
    assert result.positions_m[-1, 2] == pytest.approx(-.0001, abs=2e-8)
    assert result.kinetic_energy_ev[-1] == pytest.approx(20., abs=.002)
    assert result.kinetic_energy_ev.min() >= 0.
    assert np.all(np.diff(result.time_s) > 0.)
    assert np.all(np.diff(result.path_length_m) > 0.)
    assert result.energy_invariant_error_ev < 1e-7


def test_crossed_fields_balance_at_the_selected_relativistic_velocity():
    energy = 100_000.
    _, speed = electron_momentum_and_speed(energy)
    # v along +Z and B along +Y give v cross B along -X; E cancels it.
    scene = UniformElectromagneticScene(electric=(speed*.02, 0., 0.), magnetic=(0., .02, 0.))
    settings = _settings(kinetic_energy_ev=energy, max_path_length_m=.01, step_m=.0002)
    result = trace_test_electron(scene, settings)
    assert result.reason == "path_limit"
    np.testing.assert_allclose(result.positions_m[:, :2], 0., atol=1e-12)
    np.testing.assert_allclose(result.kinetic_energy_ev, energy, atol=1e-6)


def test_discrete_gradient_matches_existing_reference_step():
    from temsim.magnetic_test_particle import _Sampler, _discrete_gradient_step
    from temsim.physics.relativistic_lorentz import (
        RelativisticPhaseSpace, ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
    )
    from temsim.physics.static_energy_lorentz import static_energy_step
    scene = UniformElectromagneticScene(electric=(1e4, -2e4, -5e5), magnetic=(.001, .002, .005))
    class Electric:
        def field_at_global_positions_v_per_m(self, points):
            return np.broadcast_to(scene.electric, points.shape)
        def potential_v_at_global_positions(self, points):
            return -points@scene.electric
    p0, _ = electron_momentum_and_speed(500.)
    phase = RelativisticPhaseSpace(np.zeros((1, 3)), np.array(((0., 0., p0),)))
    scale = ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S
    dt = 2e-13
    result = _discrete_gradient_step(_Sampler(scene), phase.position_m[0], phase.momentum_kg_m_per_s[0]/scale,
                                    scene.diagnostic_fields_at_global_position(phase.position_m[0]), dt, lambda: False)
    reference = static_energy_step(phase, dt, scene, Electric(), tolerance=1e-12)
    np.testing.assert_allclose(result[0], reference.position_m[0], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(result[1]*scale, reference.momentum_kg_m_per_s[0], rtol=1e-9, atol=1e-31)


def test_electric_domain_stop_cancellation_and_budget_remain_explicit():
    scene = UniformElectromagneticScene(electric=(0., 0., -1e5),
        bounds=((-1., -1., 0.), (1., 1., .0001)))
    result = trace_test_electron(scene, _settings(kinetic_energy_ev=.3, max_path_length_m=.001))
    assert result.reason == "domain_exit" and result.completed
    assert result.positions_m[-1, 2] == pytest.approx(.0001, abs=1e-11)
    result = trace_test_electron(scene, _settings(kinetic_energy_ev=.3), cancelled=lambda: True)
    assert result.reason == "cancelled" and not result.completed
    result = trace_test_electron(scene, _settings(kinetic_energy_ev=.3, max_steps=2))
    assert result.reason == "step_limit" and result.steps == 2 and not result.completed


def test_electric_hardware_intercept_stops_without_specimen_or_detector_signals():
    class ApertureScene(UniformElectromagneticScene):
        def diagnostic_segment_stop(self, start, end):
            if start[2] <= .0001 <= end[2] and end[2] > start[2]:
                return (.0001-start[2])/(end[2]-start[2]), "aperture_stop"
            return None
    scene = ApertureScene(electric=(0., 0., -1e5))
    result = trace_test_electron(scene, _settings(kinetic_energy_ev=.3, max_path_length_m=.001))
    assert result.reason == "aperture_stop" and result.completed
    assert result.positions_m[-1, 2] == pytest.approx(.0001, abs=1e-11)
    assert result.kinetic_energy_ev[-1] == pytest.approx(10.3, abs=1e-6)


def test_nonuniform_electric_and_magnetic_trajectory_agrees_with_independent_reference():
    from scipy.integrate import solve_ivp
    from temsim.physics.relativistic_lorentz import ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S

    class HarmonicPotential(UniformElectromagneticScene):
        def diagnostic_fields_at_global_position(self, point):
            # Laplace phi=Az+G/2*(z²-(x²+y²)/2); no fictitious vacuum charge.
            x, y, z = point
            a, g = 1e5, 2e7
            potential = a*z+.5*g*(z*z-.5*(x*x+y*y))
            return np.array((0., .001, 0.)), np.array((.5*g*x, .5*g*y, -a-g*z)), potential
    scene = HarmonicPotential()
    settings = _settings(kinetic_energy_ev=10_000., polar_angle_deg=5., max_path_length_m=.01,
                         step_m=.00005, relative_tolerance=1e-5)
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    theta = np.deg2rad(settings.polar_angle_deg)
    scale = ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S
    initial = np.array((0., 0., 0., p/scale*np.sin(theta), 0., p/scale*np.cos(theta), 0.))
    def derivative(_time, values):
        u = values[3:6]
        velocity = SPEED_OF_LIGHT_M_PER_S*u/np.sqrt(1.+u@u)
        b, e, _phi = scene.diagnostic_fields_at_global_position(values[:3])
        du = -ELEMENTARY_CHARGE_C/scale*(e+np.cross(velocity, b))
        return np.r_[velocity, du, np.linalg.norm(velocity)]
    def arrival(_time, values):
        return values[6]-settings.max_path_length_m
    arrival.terminal = True
    reference = solve_ivp(derivative, (0., 1e-8), initial, method="DOP853", events=arrival,
                          rtol=2e-12, atol=1e-14)
    assert reference.success and len(reference.t_events[0]) == 1
    result = trace_test_electron(scene, settings)
    assert result.reason == "path_limit"
    np.testing.assert_allclose(result.positions_m[-1], reference.y[:3, -1], atol=2e-10, rtol=0.)
    assert result.time_s[-1] == pytest.approx(reference.t[-1], rel=1e-7)
    assert result.energy_invariant_error_ev < 1e-6
    coarse = trace_test_electron(scene, replace(settings, step_m=settings.step_m*2.))
    fine_error = np.linalg.norm(result.positions_m[-1]-reference.y[:3, -1])
    coarse_error = np.linalg.norm(coarse.positions_m[-1]-reference.y[:3, -1])
    assert 3.7 < coarse_error/fine_error < 4.3


def test_complete_electromagnetic_solver_also_resolves_transverse_thin_coil():
    provider = UniformScene((0., .1, 0.))
    bounds = np.array(((-.01, -.01, 0.), (.01, .01, 1e-7)))
    magnetic = captured_scene(_Source("coil", provider, bounds, .01),
                              bounds=((- .1, -.1, -.01), (.1, .1, .01)))
    class Scene:
        has_electric_field = True
        def __getattr__(self, name):
            return getattr(magnetic, name)
        def diagnostic_fields_at_global_position(self, point):
            b = magnetic.diagnostic_field_at_global_position_t(point)
            return None if b is None else (b, np.zeros(3), 0.)
    settings = _settings(position_m=(0., 0., 5e-8), polar_angle_deg=90.,
                          max_path_length_m=.002, step_m=.001)
    result = trace_test_electron(Scene(), settings)
    p, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    radius = p/(ELEMENTARY_CHARGE_C*.1)
    expected = -np.sqrt(1.-(1.-5e-8/radius)**2)
    assert result.reason == "path_limit"
    assert result.directions[-1, 2] == pytest.approx(expected, rel=2e-5)
    assert result.energy_invariant_error_ev < 1e-7


def test_field_reuse_is_exact_bounded_and_owned_by_one_trace():
    from temsim.magnetic_test_particle import _Sampler
    class Counted(UniformElectromagneticScene):
        calls = 0
        def diagnostic_fields_at_global_position(self, point):
            self.calls += 1
            return super().diagnostic_fields_at_global_position(point)
    scene = Counted(electric=(0., 0., -1e5))
    sampler = _Sampler(scene)
    point = np.array((0., 0., .0001))
    first = sampler.fields(point)
    assert sampler.fields(point.copy()) is first
    assert scene.calls == 1
    assert not first[0].flags.writeable and not first[1].flags.writeable
    # Distinct neighbouring floats are not conflated into a spatial cache bin.
    point[2] = np.nextafter(point[2], np.inf)
    sampler.fields(point)
    assert scene.calls == 2
    for z in np.linspace(0., .01, 100):
        sampler.fields(np.array((0., 0., z)))
    assert len(sampler._field_cache) == 64
    _Sampler(scene).fields(point)
    assert scene.calls == 103


def test_unsupported_field_stop_is_not_a_completed_hardware_collision():
    class Unsupported(UniformElectromagneticScene):
        def diagnostic_segment_stop(self, start, end):
            if start[2] <= .0001 <= end[2] and end[2] > start[2]:
                return (.0001-start[2])/(end[2]-start[2]), "unsupported_field:electrostatic_blanker"
            return None
    result = trace_test_electron(Unsupported(electric=(0., 0., -1e5)),
                                 _settings(kinetic_energy_ev=.3, max_path_length_m=.001))
    assert result.reason == "unsupported_field:electrostatic_blanker" and not result.completed
    assert result.positions_m[-1, 2] == pytest.approx(.0001, abs=1e-11)
    # A scene also clips its field domain at the unavailable device entrance.
    # An out-of-domain trial must preserve that explicit stop reason.
    clipped = Unsupported(electric=(0., 0., -1e5), bounds=((-1., -1., 0.), (1., 1., .0001)))
    result = trace_test_electron(clipped, _settings(kinetic_energy_ev=.3, max_path_length_m=.001))
    assert result.reason == "unsupported_field:electrostatic_blanker" and not result.completed
    assert result.positions_m[-1, 2] == pytest.approx(.0001, abs=1e-11)
