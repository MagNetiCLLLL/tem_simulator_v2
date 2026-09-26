"""Physical event definitions must not confuse force, slope and position."""
import numpy as np
import pytest

from scripts.analyze_accelerator_turns import (
    CHARGE, MASS, LIGHT, curvature_components, force_metrics, sign_changes,
    trajectory_metrics,
)


def saved_fixture(z, x, slope):
    state = np.zeros((len(z), 1, 6))
    state[:, 0, 0] = x
    state[:, 0, 2] = slope
    state[:, 0, 4] = 1.
    return state


def test_resolved_sign_events_ignore_plateau_and_bounded_small_oscillations():
    z = np.arange(7.)
    values = [2., 0., 1e-12, -1e-12, 1e-12, 0., -2.]
    assert len(sign_changes(z, values)) == 3
    events = sign_changes(z, values, 1e-10)
    assert len(events) == 1
    assert events[0] == {"z_m": 3., "bracket_m": [0., 6.], "before": 1, "after": -1}


def test_axis_crossing_with_constant_slope_is_not_transverse_turn():
    z = np.linspace(0., 2., 11)
    state = saved_fixture(z, z-1., np.ones(len(z)))
    result = trajectory_metrics(z, state, [1.], interval_m=(0., 2.))
    assert result["x_axis_crossings"]["resolved"]["per_particle"] == [1]
    assert result["x_slope_turns"]["resolved"]["per_particle"] == [0]
    assert result["radial_extrema"]["resolved"]["per_particle"] == [1]
    assert result["longitudinal"]["all_sampled_forward"]


def test_transverse_turn_away_from_axis_is_not_a_crossover():
    z = np.linspace(0., 2., 11)
    state = saved_fixture(z, 1.+(z-1.)**2, 2*(z-1.))
    result = trajectory_metrics(z, state, [1.], interval_m=(0., 2.))
    assert result["x_axis_crossings"]["resolved"]["per_particle"] == [0]
    assert result["x_slope_turns"]["resolved"]["per_particle"] == [1]
    assert result["radial_extrema"]["resolved"]["per_particle"] == [1]


def test_backward_longitudinal_samples_are_reported_separately():
    z = np.linspace(0., 2., 11)
    state = saved_fixture(z, np.ones(len(z)), np.zeros(len(z)))
    state[5:, 0, 4] = -1.
    result = trajectory_metrics(z, state, [1.], interval_m=(0., 2.))
    assert not result["longitudinal"]["all_sampled_forward"]
    assert result["longitudinal"]["nonforward_particle_count"] == 1
    assert result["x_slope_turns"]["resolved"]["max"] == 0


def test_axial_acceleration_bends_x_of_z_without_a_transverse_force():
    state = np.array([[2e-6, 0., .01, 0., 1., 0.]])
    field = np.array([[0., 0., -1e6]])
    transverse, acceleration = curvature_components(state, field)
    np.testing.assert_array_equal(transverse, [[0., 0.]])
    assert acceleration[0, 0] < 0
    # Exact derivative of px/pz under qEz with x-component momentum fixed.
    gamma = np.sqrt(2.0001)
    expected = CHARGE*gamma/(MASS*LIGHT**2)*.01*(-1e6)
    np.testing.assert_allclose(acceleration[0, 0], expected, rtol=1e-14)


def test_radial_force_projection_is_invariant_under_transverse_rotation():
    z = np.linspace(0., 2., 11)
    state = saved_fixture(z, np.full(len(z), 1e-6), np.ones(len(z))*.001)
    electric = np.zeros((len(z), 1, 3))
    electric[:, 0, 0] = z-1.
    first = force_metrics(z, state, electric, [1.], interval_m=(0., 2.))
    rotated = state.copy()
    rotated[..., 0], rotated[..., 1] = -state[..., 1], state[..., 0]
    rotated[..., 2], rotated[..., 3] = -state[..., 3], state[..., 2]
    rotated_electric = electric.copy()
    rotated_electric[..., 0], rotated_electric[..., 1] = -electric[..., 1], electric[..., 0]
    second = force_metrics(z, rotated, rotated_electric, [1.], interval_m=(0., 2.))
    assert first["radial_force_direction_reversals"] == second["radial_force_direction_reversals"]
    assert first["radial_force_direction_reversals"]["resolved"]["max"] == 1
    assert first["peak_transverse_force_n"] == second["peak_transverse_force_n"]


def test_analysis_rejects_nonmonotonic_saved_planes_and_negative_weights():
    state = saved_fixture(np.arange(3.), np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="increasing z"):
        trajectory_metrics([0., 2., 1.], state, [1.], interval_m=(0., 1.))
    with pytest.raises(ValueError, match="nonnegative"):
        trajectory_metrics([0., 1., 2.], state, [-1.], interval_m=(0., 1.))
