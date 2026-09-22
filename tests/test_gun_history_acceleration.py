"""Lossless history conversion and exact rejected-trial reuse."""
import numpy as np
import pytest

from test_analytic_particle_step import gun, _phase
from temsim.optics.electron_gun import tracing
from temsim.physics import analytic_particle_step as compiled


def test_history_finalization_keeps_every_value_and_releases_consumed_rows():
    rng = np.random.default_rng(724)
    positions = [rng.normal(size=(7, 3)) for _ in range(11)]
    momenta = [rng.normal(size=(7, 3)) for _ in range(11)]
    momenta[3][2, 2] = 0.
    original_x, original_p = np.asarray(positions), np.asarray(momenta)
    times = np.arange(11, dtype=float) * 1e-13
    alive = [rng.random(7) > .2 for _ in times]
    completed = [rng.random(7) > .7 for _ in times]
    history = tracing._finalize_gun_history(positions, momenta, times, alive, completed)
    for values, expected in ((history.x_m, original_x[:, :, 0]),
                             (history.y_m, original_x[:, :, 1]),
                             (history.z_mm, original_x[:, :, 2] * 1000.),
                             (history.time_s, times), (history.alive, alive),
                             (history.completed, completed)):
        np.testing.assert_array_equal(values, expected)
    for column, values in ((0, history.tx_rad), (1, history.ty_rad)):
        expected = np.divide(original_p[:, :, column], original_p[:, :, 2],
                             out=np.zeros(original_p.shape[:2]), where=np.abs(original_p[:, :, 2]) > 0.)
        np.testing.assert_array_equal(values, expected)
    assert all(value is None for value in positions + momenta)
    assert all(value.flags.owndata for value in (history.x_m, history.y_m, history.z_mm))


@pytest.mark.parametrize("values", [
    [0., 1., 1., .5, 2., np.nan, np.inf, -np.inf, 3.],
    [0., .5e-12, 1e-12, 1.5e-12, 2e-12, 3e-12],
    [-2., -1., 0., -3., 2.], [np.nan, np.inf],
])
def test_compiled_history_records_match_chronological_threshold(values):
    expected, previous = [], -np.inf
    for index, value in enumerate(values):
        if np.isfinite(value) and value > previous + 1e-12:
            expected.append(index)
            previous = value
    array = np.asarray(values, dtype=float)
    np.testing.assert_array_equal(tracing._history_indices(array), expected)
    np.testing.assert_array_equal(tracing._retained_axial_indices(array), expected)


def test_history_processing_observes_cancellation():
    with pytest.raises(RuntimeError, match="Superseded"):
        tracing._finalize_gun_history([np.zeros((1,3))], [np.ones((1,3))],
            [0.], [np.ones(1, bool)], [np.zeros(1, bool)], cancelled=lambda: True)


def test_duplicate_exact_events_keep_original_stable_last_event_ownership():
    from temsim.optics.electron_gun.base import GunEqualTimeHistory, GunPlaneArrival
    z = np.array([[0.], [1.], [2.], [3.]])
    times = np.arange(4, dtype=float) * 1e-9
    h = GunEqualTimeHistory(times, z, z * 1e-6, z * 2e-6,
        z * 1e-4, z * 2e-4, np.ones_like(z, bool), np.zeros_like(z, bool))
    def event(x):
        return GunPlaneArrival("event", "event", 2., np.array([2e-9]),
            np.array([x]), np.array([0.]), np.array([True]), np.array([True]))
    outputs = tuple(np.zeros((4, 1)) for _ in range(4))
    actual = tracing._resample_gun_flight_times([0., 1., 2., 3.], h,
        plane_arrivals=(event(4e-6), event(8e-6)), path_outputs=outputs,
        terminal_z_mm=[3.], terminal_time_s=[times[-1]])
    np.testing.assert_array_equal(actual[:, 0], times)
    assert outputs[0][2, 0] == 8e-6


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
def test_each_rejected_trial_reuses_exactly_one_already_executed_half_step(gun, monkeypatch):
    phase, invariant = _phase(gun, [1., 5., 13., 14., 22., 41., 77., 402., 418., 433., 550.])
    active = np.ones(len(invariant), dtype=bool)
    active[[1, 4]] = False
    fields = compiled._field_parameters(gun.electric_field, gun.magnetic_field)
    calls = []
    original = compiled._advance
    def counted(*args):
        calls.append(args[6])
        return original(*args)
    monkeypatch.setattr(compiled, "_advance", counted)
    x, p, dt = compiled._step(phase.position_m, phase.momentum_kg_m_per_s,
                              1e-12, active, invariant[active], *fields)
    rejections = int(round(np.log2(1e-12 / dt)))
    assert rejections > 0
    assert len(calls) == np.count_nonzero(active) * (3 + 2 * rejections)
    monkeypatch.setattr(compiled, "_advance", original)
    actual = compiled.try_analytic_step(gun, phase, 1e-12, active,
        gun.magnetic_field, gun.electric_field, invariant[active])
    assert actual[1] == dt
    np.testing.assert_array_equal(actual[0].position_m, x)
    np.testing.assert_array_equal(actual[0].momentum_kg_m_per_s, p)
