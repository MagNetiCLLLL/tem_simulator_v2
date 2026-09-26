"""Analytical transport checks independent of the gun geometry field solver."""
import numpy as np
import pytest

from scripts.diagnose_planar_gun import (
    CHARGE, LIGHT, MASS, kinetic_energy, trace,
)


class UniformField:
    def __init__(self, ez):
        self.ez = float(ez)

    def field_at_global_positions_v_per_m(self, positions):
        result = np.zeros_like(positions)
        result[:, 2] = self.ez
        return result


def emitted_state(energy_ev, slopes=((0., 0.),), time_s=0.):
    # Independent algebra for mechanical momentum at prescribed kinetic energy.
    kinetic_over_rest = np.asarray(energy_ev, float)*CHARGE/(MASS*LIGHT**2)
    magnitude = np.sqrt(kinetic_over_rest*(kinetic_over_rest+2))
    direction = np.column_stack((np.asarray(slopes), np.ones(len(slopes))))
    momentum = direction/np.linalg.norm(direction, axis=1)[:, None]*magnitude[..., None]
    return np.column_stack((np.zeros((len(slopes), 2)), momentum,
                            np.full(len(slopes), time_s)))


def exact_uniform(initial, distance, ez):
    """Constant dp_z/dt solution, with transverse mechanical momentum fixed."""
    result = initial.copy()
    u = initial[:, 2:5]
    gamma0 = np.sqrt(1+np.sum(u*u, axis=1))
    transverse_rest = np.sqrt(1+np.sum(u[:, :2]**2, axis=1))
    gain_per_m = -CHARGE*ez/(MASS*LIGHT**2)
    gain = gain_per_m*distance
    uz1 = np.sqrt(u[:, 2]**2+gain*(2*gamma0+gain))
    dt = (uz1-u[:, 2])/(gain_per_m*LIGHT)
    transverse_integral = (
        np.arcsinh(uz1/transverse_rest)-np.arcsinh(u[:, 2]/transverse_rest)
    )/gain_per_m
    result[:, :2] += u[:, :2]*transverse_integral[:, None]
    result[:, 4] = uz1
    result[:, 5] += dt
    return result


def test_uniform_acceleration_preserves_energy_direction_and_clock():
    initial = emitted_state(.3, ((0., 0.), (.003, -.002), (-.003, .002)), 2e-9)
    field = UniformField(-5.5e5)
    distance = .006
    planes = np.r_[np.geomspace(1e-9, 1e-5, 15), .003]
    z, states, _ = trace(field, initial, 0., distance, (.001, .002), sample_planes=planes)
    expected = exact_uniform(initial, distance, field.ez)
    np.testing.assert_array_equal(states[0], initial)
    np.testing.assert_allclose(kinetic_energy(states[-1]), .3-field.ez*distance,
                               rtol=2e-9, atol=2e-7)
    np.testing.assert_allclose(states[-1, :,:2], expected[:, :2], rtol=5e-8, atol=2e-14)
    np.testing.assert_allclose(states[-1, :,2:4], initial[:, 2:4], rtol=0., atol=1e-16)
    np.testing.assert_allclose(states[-1, :,5], expected[:, 5], rtol=2e-9, atol=2e-17)
    assert np.all(np.diff(z) > 0)
    for plane in planes:
        index = np.argmin(abs(z-plane))
        assert abs(z[index]-plane) < 1e-16
        np.testing.assert_allclose(kinetic_energy(states[index]), .3-field.ez*plane,
                                   rtol=2e-9, atol=2e-7)
        np.testing.assert_allclose(states[index, :, 5],
                                   exact_uniform(initial, plane, field.ez)[:, 5],
                                   rtol=2e-9, atol=2e-17)


def test_zero_field_drift_has_exact_initial_slope_and_speed():
    initial = emitted_state(300000., ((.03, -.04), (-.03, .04)), 3e-9)
    start, end = .025, .037
    _, states, _ = trace(UniformField(0), initial, start, end)
    expected = initial.copy()
    expected[:, :2] += initial[:, 2:4]/initial[:, 4, None]*(end-start)
    gamma = np.sqrt(1+np.sum(initial[:, 2:5]**2, axis=1))
    expected[:, 5] += gamma/(LIGHT*initial[:, 4])*(end-start)
    np.testing.assert_allclose(states[-1], expected, rtol=2e-13, atol=1e-18)


@pytest.mark.parametrize("pz", [0., -0.001])
def test_nonforward_launch_rejected(pz):
    initial = emitted_state(.3)
    initial[:, 4] = pz
    with pytest.raises(ValueError, match="Invalid forward"):
        trace(UniformField(-1e6), initial, 0., .001)


def test_electron_turning_is_not_accepted_as_forward_continuation():
    with pytest.raises(ValueError, match="Backstreaming"):
        trace(UniformField(1e6), emitted_state(.3), 0., .001)


def test_spatial_step_refinement_agrees_with_analytic_uniform_solution():
    initial = emitted_state(20000., ((.08, -.04),))
    field = UniformField(-2e6)
    distance = .03
    expected = exact_uniform(initial, distance, field.ez)
    errors = []
    for step in (.03, .00375, .0009375):
        _, state, _ = trace(field, initial, 0., distance, max_step_m=step, rtol=.02)
        # Normalize position and clock with physical interval scales.
        errors.append(max(np.max(abs(state[-1, :, :2]-expected[:, :2]))/distance,
                          np.max(abs(state[-1, :, 5]-expected[:, 5]))/(distance/LIGHT)))
    assert errors[-1] < 1e-10
    assert errors[-1] < errors[0]/100


def test_executed_midpoint_continuation_keeps_full_momentum_and_elapsed_time():
    initial = emitted_state(.3, ((.003, -.002),), 1e-9)
    field = UniformField(-5.5e5)
    _, first, _ = trace(field, initial, 0., .003)
    _, resumed, _ = trace(field, first[-1].copy(), .003, .006)
    expected = exact_uniform(initial, .006, field.ez)
    np.testing.assert_allclose(resumed[-1, :, :2], expected[:, :2], rtol=5e-8, atol=2e-14)
    np.testing.assert_allclose(kinetic_energy(resumed[-1]), 3300.3, rtol=2e-9, atol=2e-7)
    np.testing.assert_allclose(resumed[-1, :, 5], expected[:, 5], rtol=2e-9, atol=2e-17)


def test_sub_ev_energy_spread_with_tight_tolerance_remains_forward():
    # A tight generic initial-step estimate can overshoot the very short launch
    # scale and create negative *trial* momentum in an entirely accelerating
    # field. No physical trajectory in this analytical fixture can turn back.
    count = 193
    energies = np.linspace(.01, .6, count)
    angle = np.linspace(0., 2*np.pi, count, endpoint=False)
    slopes = np.column_stack((.003*np.cos(angle), .003*np.sin(angle)))
    initial = emitted_state(energies, slopes)
    field = UniformField(-5e5)
    distance = .001
    planes = np.geomspace(1e-10, 1e-5, 20)
    z, states, _ = trace(field, initial, 0., distance, rtol=2e-11,
                         sample_planes=planes)
    assert np.all(states[..., 4] > 0)
    np.testing.assert_array_equal(states[0], initial)
    expected = exact_uniform(initial, distance, field.ez)
    np.testing.assert_allclose(kinetic_energy(states[-1]), energies-field.ez*distance,
                               rtol=2e-10, atol=2e-8)
    np.testing.assert_allclose(states[-1, :, :2], expected[:, :2], rtol=2e-9, atol=2e-14)
    np.testing.assert_allclose(states[-1, :, 5], expected[:, 5], rtol=2e-9, atol=2e-18)
    for plane in planes:
        index = np.argmin(abs(z-plane))
        assert abs(z[index]-plane) < 1e-16
        np.testing.assert_allclose(kinetic_energy(states[index]), energies-field.ez*plane,
                                   rtol=2e-10, atol=2e-8)
        np.testing.assert_allclose(states[index, :, 5],
                                   exact_uniform(initial, plane, field.ez)[:, 5],
                                   rtol=2e-9, atol=2e-18)
