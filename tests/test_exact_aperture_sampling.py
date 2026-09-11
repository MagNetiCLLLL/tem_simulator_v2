"""Operator-level drift/clipping tests, not a complete gun-wave validation."""
from types import SimpleNamespace

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.core import build_propagation_plan, propagate


def test_sparse_plot_history_retains_the_physical_aperture_plane():
    state = default_state()
    state.step_mm = .8
    state.history_step_mm = 10.
    state.acceleration_enabled = False
    state.apertures = [SimpleNamespace(key="test-stop", z_mm=.37,
        enabled=True, installed=True, radius_mm=.037, offset_x_mm=0., offset_y_mm=0.)]
    for lens in state.lenses:
        lens.enabled = False
    plan = build_propagation_plan(state, 0., 1.)
    saved = plan.z_mm[plan.save_index]
    assert np.count_nonzero(saved == .37) == 1
    initial = (np.array([.036e-3]), np.array([.004]), np.zeros(1), np.zeros(1))
    z, x, tx, y, ty = propagate(state, 0., 1., *initial)
    index, = np.flatnonzero(z == .37)
    expected = .036e-3+.004*.37e-3
    # The retained drawing history is float32 even on the float64 CPU solver.
    # Allow two storage ULPs, not a relaxation of the aperture-location check.
    np.testing.assert_allclose(x[index], expected, rtol=0,
        atol=2*abs(np.spacing(np.float32(expected))))
    alive, blocked, keys = clip_segment(state, z, x, y)
    assert not alive[0]
    assert blocked[0] == .37
    assert keys == ["test-stop"]
    # The old sparse history selected z=0, which incorrectly transmitted it.
    assert clip_segment(state, z[[0, -1]], x[[0, -1]], y[[0, -1]])[0][0]


def test_inactive_aperture_does_not_add_a_physical_action_plane():
    state = default_state()
    state.step_mm = .8
    state.apertures = [SimpleNamespace(key="test-stop", z_mm=.37,
        enabled=False, installed=True)]
    plan = build_propagation_plan(state, 0., 1.)
    assert not np.any(plan.z_mm == .37)
