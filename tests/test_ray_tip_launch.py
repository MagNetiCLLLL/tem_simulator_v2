"""Ray Diagram launch-prefix regressions; classical particle traces only."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.electron_gun import FieldEmissionGun


def test_launch_prefix_uses_each_actual_start_and_preserves_negative_stop():
    z = np.array([0.0, 1.0, 2.0])
    values = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]) * 1e-6
    launch_z = np.array([[-.2, -.1, -.3], [-.05, .1, -.25], [.1, .2, -.25]])
    launch_values = np.array([[.5, 1.5, 2.5], [.8, 2.1, 2.8], [1.1, 2.2, 2.8]]) * 1e-6
    actual_z, actual_mm = VisualizationWorkspace._bundle_lines(
        z, values, 3, np.array([np.nan, 1.5, -.25]),
        launch_z=launch_z, launch_values=launch_values)
    separators = np.flatnonzero(np.isnan(actual_z))
    starts = np.r_[0, separators[:-1] + 1]
    for ray, (start, stop) in enumerate(zip(starts, separators)):
        assert actual_z[start] == launch_z[0, ray]
        assert actual_mm[start] == pytest.approx(launch_values[0, ray] * 1000)
        assert np.min(actual_z[start:stop]) == launch_z[0, ray]
    assert actual_z[separators[1] - 1] == 1.5
    assert actual_z[separators[2] - 1] == -.25
    assert np.all(actual_z[starts[2]:separators[2]] < 0)


@pytest.mark.parametrize("curvature", [0., 1e-8, .01])
def test_actual_tip_launch_display_preserves_transport_and_rotation(qtbot, curvature):
    gun = FieldEmissionGun()
    gun.emitter.curvature_nm_inv = curvature
    trace = gun.trace_to_exit(49)
    branch = SimpleNamespace(name="incident", z=trace.z_mm, x=trace.x_m, y=trace.y_m,
                             blocked_z=trace.blocked_z_mm)
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    view._last_result = SimpleNamespace(simulation=SimpleNamespace(incident=branch, gun_trace=trace))
    history = trace.equal_time_history
    before_z = trace.z_mm.copy()
    before_x = trace.x_m.copy()
    for angle in (0., 90.):
        view._projection_angle_deg = angle
        z, projected = view._display_bundle_lines(branch, np.arange(49))
        stops = np.flatnonzero(np.isnan(z))
        starts = np.r_[0, stops[:-1] + 1]
        assert len(starts) == 49
        np.testing.assert_array_equal(z[starts], history.z_mm[0])
        expected_start = view._project_transverse(history.x_m[0], history.y_m[0]) * 1000
        np.testing.assert_allclose(projected[starts], expected_start, atol=1e-15)
        # The launch prefix cannot change or extrapolate the already clipped
        # shared-plane traces, including the final exit/stop point.
        base_z, base_values = view._bundle_lines(branch.z,
            view._project_transverse(branch.x, branch.y), 49, branch.blocked_z)
        keep = (z >= 0) | np.isnan(z)
        np.testing.assert_array_equal(z[keep], base_z)
        np.testing.assert_allclose(projected[keep], base_values, equal_nan=True)
        if curvature == 0:
            np.testing.assert_array_equal(z, base_z)
        else:
            assert np.min(z[np.isfinite(z)]) < 0
    assert view.ray_display_cache_info()["hits"] == 1
    np.testing.assert_array_equal(trace.z_mm, before_z)
    np.testing.assert_array_equal(trace.x_m, before_x)
    assert gun.trace_to_exit(49) is trace
    # A downstream branch with the same ray count must not inherit the prefix.
    outgoing = SimpleNamespace(**vars(branch))
    outgoing.z = trace.z_mm + 100
    downstream_z, _ = view._display_bundle_lines(outgoing)
    assert np.nanmin(downstream_z) >= 100
