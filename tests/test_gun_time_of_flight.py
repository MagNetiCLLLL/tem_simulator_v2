"""Gun clocks are read from executed transport, never from exit velocity."""

from types import SimpleNamespace

import numpy as np

from temsim.optics.electron_gun.base import GunEqualTimeHistory, GunPlaneArrival
from temsim.optics.electron_gun.tracing import (
    _resample_gun_flight_times,
    _resample_gun_paths,
    trace_feg_to_exit,
)


def history(z_mm, time_ns, *, alive=None, completed=None, x_um=None):
    z = np.asarray(z_mm, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    zeros = np.zeros_like(z)
    x = zeros if x_um is None else np.asarray(x_um, dtype=float).reshape(z.shape) * 1e-6
    return GunEqualTimeHistory(
        time_s=np.asarray(time_ns, dtype=float) * 1e-9,
        z_mm=z, x_m=x, y_m=-x, tx_rad=x * 1e4, ty_rad=-x * 1e4,
        alive=np.ones_like(z, dtype=bool) if alive is None else np.asarray(alive, bool),
        completed=(np.zeros_like(z, dtype=bool) if completed is None
                   else np.asarray(completed, bool)),
    )


def arrival(z_mm, time_ns, *, transmitted=None, x_um=0.):
    times = np.atleast_1d(time_ns).astype(float) * 1e-9
    return GunPlaneArrival(
        key="test_plane", name="Test plane", z_mm=z_mm, time_s=times,
        x_m=np.full_like(times, x_um * 1e-6), y_m=np.full_like(times, -x_um * 1e-6),
        reached=np.isfinite(times),
        transmitted=(np.isfinite(times) if transmitted is None
                     else np.atleast_1d(transmitted).astype(bool)),
    )


def resample_display(h, *, exit_z_mm=4., history_step_mm=.5, **timing_options):
    """Apply the production display adapter without running a field solver."""
    gun = SimpleNamespace(
        exit_plane_z_mm=exit_z_mm, history_step_mm=history_step_mm,
        dpa_aperture=SimpleNamespace(z_mm=1.), components=(),
    )
    position = np.stack((h.x_m, h.y_m, h.z_mm * 1e-3), axis=-1)
    count = h.z_mm.shape[1]
    common_z, *paths = _resample_gun_paths(
        gun, position, h.tx_rad, h.ty_rad, position[-1], h.tx_rad[-1], h.ty_rad[-1],
        np.zeros(count, dtype=bool), np.full(count, np.nan), None,
    )
    legacy_paths = tuple(values.copy() for values in paths)
    times = _resample_gun_flight_times(common_z, h, path_outputs=paths, **timing_options)
    return common_z, paths, times, legacy_paths


def test_time_resampling_uses_executed_accelerating_history_and_exact_events():
    h = history([0., 1., 4., 9.], [0., 1., 2., 3.])
    z = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.5, 9., 10.])
    exact = arrival(2., 1.25)
    actual = _resample_gun_flight_times(z, h, plane_arrivals=[exact])
    np.testing.assert_allclose(actual[:, 0] * 1e9,
        [0., .5, 1., 1.125, 1.25, 1.625, 2., 2.5, 3., np.nan],
        rtol=2e-15, atol=1e-15)
    assert actual.dtype == np.float64
    assert actual[z == 2., 0][0] == exact.time_s[0]


def test_terminal_events_replace_overshoot_and_frozen_rows_without_tail_times():
    h = history([[0., 0.], [1., 1.], [2., 1.3], [2., 1.3]], [0., 1., 2., 3.],
        alive=[[True, True], [True, True], [True, False], [True, False]],
        completed=[[False, False], [False, False], [True, False], [True, False]])
    z = np.array([0., 1., 1.1, 1.2, 1.5, 2., 2.1])
    exact = [arrival(1.2, [np.nan, 1.2], transmitted=[False, False]),
             arrival(2., [1.75, np.nan])]
    actual = _resample_gun_flight_times(z, h, plane_arrivals=exact,
        terminal_z_mm=[2., 1.2], terminal_time_s=np.array([1.75, 1.2]) * 1e-9)
    np.testing.assert_allclose(actual[:, 0] * 1e9,
        [0., 1., 1.075, 1.15, 1.375, 1.75, np.nan])
    np.testing.assert_allclose(actual[:, 1] * 1e9,
        [0., 1., 1.1, 1.2, np.nan, np.nan, np.nan])


def test_distinct_tip_surface_launch_positions_have_no_pre_emission_time():
    h = history([[-.1, .2], [.4, .7], [.9, 1.2]], [0., 1., 2.])
    z = np.array([-.2, -.1, 0., .1, .2, .3])
    actual = _resample_gun_flight_times(z, h)
    np.testing.assert_allclose(actual[:, 0] * 1e9,
        [np.nan, 0., .2, .4, .6, .8], atol=1e-15)
    np.testing.assert_allclose(actual[:, 1] * 1e9,
        [np.nan, np.nan, np.nan, np.nan, 0., .2], atol=1e-15)


def test_turning_path_uses_first_crossing_and_keeps_pre_stop_excursion():
    h = history([0., 2., 1., 3., 2.5, 2.5], [0., 1., 2., 3., 4., 5.],
                alive=[[True], [True], [True], [True], [False], [False]])
    z = np.array([0., 1., 2., 2.5, 3., 3.5])
    actual = _resample_gun_flight_times(z, h,
        terminal_z_mm=[2.5], terminal_time_s=[4e-9])
    np.testing.assert_allclose(actual[:, 0] * 1e9,
        [0., .5, 1., 2.75, 3., np.nan])


def test_backwards_initial_segment_and_later_forward_crossing_are_chronological():
    h = history([1., -1., 0., 2.], [0., 1., 2., 3.])
    z = np.array([-2., -1., 0., 1., 1.5, 2.])
    actual = _resample_gun_flight_times(z, h)
    np.testing.assert_allclose(actual[:, 0] * 1e9,
        [np.nan, 1., .5, 0., 2.75, 3.])


def test_unknown_stop_time_does_not_treat_inactive_samples_as_arrivals():
    h = history([0., 1., 1.3, 1.3], [0., 1., 2., 3.],
                alive=[[True], [True], [False], [False]])
    actual = _resample_gun_flight_times([0., 1., 1.2, 1.3], h)
    np.testing.assert_allclose(actual[:, 0] * 1e9, [0., 1., np.nan, np.nan])


def test_reaching_an_absorbing_plane_has_a_time_without_transmission():
    h = history([0., 1., 2.2, 2.2], [0., 1., 2., 3.],
                alive=[[True], [True], [False], [False]])
    absorbing_plane = arrival(2., 1.75, transmitted=False)
    exit_not_reached = arrival(2., np.nan, transmitted=False)
    actual = _resample_gun_flight_times([0., 1., 2., 2.2], h,
        plane_arrivals=[absorbing_plane, exit_not_reached],
        terminal_z_mm=[2.], terminal_time_s=absorbing_plane.time_s)
    np.testing.assert_allclose(actual[:, 0] * 1e9, [0., 1., 1.75, np.nan])
    assert actual[2, 0] == absorbing_plane.time_s[0]


def test_terminal_clock_can_follow_the_last_retained_snapshot():
    h = history([0., 1.], [0., 1.])
    actual = _resample_gun_flight_times([0., 1., 1.5, 2., 3.], h,
        terminal_z_mm=[2.], terminal_time_s=[2e-9])
    np.testing.assert_allclose(actual[:, 0] * 1e9, [0., 1., 1.5, 2., np.nan])


def test_turning_display_coordinates_and_clock_use_the_same_first_crossing_segment():
    h = history([0., 3., 1., 4.], [0., 1., 2., 3.], x_um=[0., 10., 20., 30.])
    z, paths, times, _ = resample_display(h)
    row = np.flatnonzero(z == 3.5)[0]
    fraction = (3.5 - 1.) / (4. - 1.)
    np.testing.assert_allclose(times[row, 0], (2. + fraction) * 1e-9)
    for path, values in zip(paths, (h.x_m, h.y_m, h.tx_rad, h.ty_rad)):
        expected = values[2, 0] + fraction * (values[3, 0] - values[2, 0])
        np.testing.assert_allclose(path[row, 0], expected)
    earlier = np.flatnonzero(z == 1.5)[0]
    np.testing.assert_allclose(paths[0][earlier, 0], 5e-6)
    np.testing.assert_allclose(times[earlier, 0], .5e-9)


def test_plane_event_coordinates_and_clock_refine_the_same_display_segments():
    h = history([0., 3., 1., 4.], [0., 1., 2., 3.], x_um=[0., 10., 20., 30.])
    event = arrival(3.5, 2.9, x_um=31.)
    z, paths, times, _ = resample_display(h, history_step_mm=.25, plane_arrivals=[event])
    row = np.flatnonzero(z == 3.5)[0]
    assert times[row, 0] == event.time_s[0]
    assert paths[0][row, 0] == event.x_m[0]
    assert paths[1][row, 0] == event.y_m[0]
    # Both adjacent rows use the recorded event in the same piecewise path as
    # their clocks, rather than merely correcting the coordinate at the plane.
    np.testing.assert_allclose(paths[0][z == 3.25, 0], [29.9e-6])
    np.testing.assert_allclose(times[z == 3.25, 0], [2.81e-9])
    np.testing.assert_allclose(paths[0][z == 3.75, 0], [30.5e-6])
    np.testing.assert_allclose(times[z == 3.75, 0], [2.95e-9])
    np.testing.assert_allclose(paths[0][z == 4., 0], [30e-6])
    np.testing.assert_allclose(times[z == 4., 0], [3e-9])


def test_monotone_display_interpolation_is_unchanged_without_resolved_events():
    h = history([0., 1., 4.], [0., 1., 3.], x_um=[0., 10., 18.])
    z, paths, times, legacy = resample_display(h, exit_z_mm=5.)
    for path, original in zip(paths, legacy):
        np.testing.assert_allclose(path, original, rtol=2e-15, atol=1e-20)
    assert np.all(np.isnan(times[z > 4.]))
    np.testing.assert_array_equal(paths[0][z > 4.], legacy[0][z > 4.])


def test_absorbing_plane_uses_recorded_xy_without_timing_the_readability_tail():
    h = history([0., 1., 2.2, 2.2], [0., 1., 2., 3.],
                x_um=[0., 10., 22., 22.],
                alive=[[True], [True], [False], [False]])
    event = arrival(2., 1.75, transmitted=False, x_um=9.)
    z, paths, times, legacy = resample_display(
        h, plane_arrivals=[event], terminal_z_mm=[2.], terminal_time_s=event.time_s,
    )
    np.testing.assert_array_equal(paths[0][z == 2., 0], event.x_m)
    np.testing.assert_array_equal(paths[1][z == 2., 0], event.y_m)
    np.testing.assert_array_equal(times[z == 2., 0], event.time_s)
    assert np.all(np.isnan(times[z > 2.]))
    for path, original in zip(paths, legacy):
        np.testing.assert_array_equal(path[z > 2.], original[z > 2.])


def test_real_tip_transport_publishes_original_exit_and_plane_clocks():
    from temsim.optics.column import default_state
    gun = default_state().electron_gun
    assert gun.emitter.surface_model is None
    result = trace_feg_to_exit(gun, 9)
    times = result.flight_time_s
    assert times.dtype == np.float64
    assert times.shape == result.x_m.shape
    np.testing.assert_array_equal(times[0], np.zeros(9))
    assert result.exit_bundle.flight_time_s.dtype == np.float64
    exit_plane = result.plane_arrivals[-1]
    np.testing.assert_array_equal(result.exit_bundle.flight_time_s, exit_plane.time_s)
    assert np.any(result.exit_bundle.alive)
    assert np.all(np.isfinite(result.exit_bundle.flight_time_s[result.exit_bundle.alive]))
    assert np.all(result.exit_bundle.flight_time_s[result.exit_bundle.alive] > 0.)
    assert np.all(np.isnan(result.exit_bundle.flight_time_s[~exit_plane.reached]))
    for displayed, exact in ((result.tx_rad, result.exit_bundle.tx_rad),
                             (result.ty_rad, result.exit_bundle.ty_rad)):
        np.testing.assert_array_equal(displayed[-1, exit_plane.reached], exact[exit_plane.reached])
    for plane in result.plane_arrivals:
        row = np.flatnonzero(result.z_mm == plane.z_mm)
        if row.size:
            np.testing.assert_array_equal(times[row[0], plane.reached], plane.time_s[plane.reached])
            np.testing.assert_array_equal(result.x_m[row[0], plane.reached], plane.x_m[plane.reached])
            np.testing.assert_array_equal(result.y_m[row[0], plane.reached], plane.y_m[plane.reached])
    for ray in range(9):
        finite = times[:, ray][np.isfinite(times[:, ray])]
        assert np.all(np.diff(finite) >= 0.)
        if np.isfinite(result.blocked_z_mm[ray]):
            assert np.all(np.isnan(times[result.z_mm > result.blocked_z_mm[ray], ray]))
