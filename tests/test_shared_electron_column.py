"""Virtual/production column parity; small prescribed-field fixtures only."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.magnetic_field_scene import prepare_magnetic_scene
from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
from temsim.optics.column import default_state
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.core import propagate
from temsim.physics.instrument_electric import InstrumentElectricField
from temsim.physics.instrument_magnetic import active_column_events
from temsim.test_electron_scene import prepare_test_electron_scene


@pytest.fixture
def column(monkeypatch):
    state = default_state()
    state.simulation_mode = "custom"
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = state.history_step_mm = .25
    state.equivalent_image_lenses_enabled = False
    for lens in state.lenses:
        lens.enabled = False
    for item in state.stigmators:
        item.enabled = False
    for item in state.corrector_elements:
        item.enabled = False
    r, z = np.array([0., .01]), np.array([0., .4, 1., 4.])
    base = ClosedGunField({}, r, z, np.broadcast_to(1e5*z, (2, 4)))
    field = InstrumentElectricField(base, base, state.electron_gun,
        np.array(((-.01, -.01, 0.), (.01, .01, 4.))), "prescribed", "prescribed", "prescribed", ())
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", lambda state: field)
    monkeypatch.setattr("temsim.test_electron_scene._prepare_electric_provider", lambda state, stop: (state.electron_gun, base, ()))
    scene = prepare_test_electron_scene(state, prepare_magnetic_scene(state), z_limits_mm=(0., 503.))
    # The prescribed potential has no physical solver request identity, so
    # production correctly refuses automatic column replay. Select the adapter
    # explicitly for this isolated mathematical fixture; do not invent a
    # verified physical identity or relax the production admission rule.
    from temsim.instrument_snapshot import encode_instrument
    scene = replace(scene, _column_input_graph=encode_instrument(state),
                    _column_handoff_z_m=state.electron_gun.exit_plane_z_mm*1e-3)
    return state, scene


@pytest.mark.parametrize("compiled", [False, True])
def test_virtual_uses_production_column_trajectory_energy_and_clock(column, compiled):
    state, scene = column
    settings = TestElectronSettings(position_m=(1e-6, -2e-6, .5), kinetic_energy_ev=200000.,
        polar_angle_deg=.03, azimuth_angle_deg=23., max_path_length_m=.01,
        step_m=.00025, max_steps=10000)
    actual = trace_test_electron(scene, settings, use_compiled=compiled)
    theta, phi = np.radians((settings.polar_angle_deg, settings.azimuth_angle_deg))
    tx, ty = np.tan(theta)*np.cos(phi), np.tan(theta)*np.sin(phi)
    expected = propagate(state, 500., 503., np.array([1e-6]), np.array([tx]),
        np.array([-2e-6]), np.array([ty]), events=active_column_events(state),
        initial_kinetic_energy_ev=np.array([200000.]), initial_time_s=np.zeros(1),
        return_flight_times=True, return_checkpoints=True,
        checkpoint_z_mm=(501., 502., 503.))[-1]
    assert actual.completed and actual.reason == "domain_exit"
    for i, z in enumerate(expected.z_mm*1e-3):
        row = np.argmin(abs(actual.positions_m[:, 2]-z))
        assert abs(actual.positions_m[row, 2]-z) < 1e-14
        np.testing.assert_allclose(actual.positions_m[row, :2],
            [expected.x_m[i, 0], expected.y_m[i, 0]], rtol=2e-11, atol=1e-16)
        assert actual.kinetic_energy_ev[row] == pytest.approx(expected.kinetic_energy_ev[i, 0], abs=1e-7)
        assert actual.time_s[row] == pytest.approx(expected.flight_time_s[i, 0], rel=1e-11)
    assert actual.kinetic_energy_ev[-1]-actual.kinetic_energy_ev[0] == pytest.approx(300., abs=1e-7)
    assert actual.energy_invariant_error_ev < 1e-7


def test_captured_column_inputs_do_not_follow_later_live_edits(column):
    state, scene = column
    settings = TestElectronSettings(position_m=(1e-6, 0., .5), kinetic_energy_ev=200000.,
        max_path_length_m=.01, step_m=.00025)
    before = trace_test_electron(scene, settings, use_compiled=False)
    state.electron_gun.accelerator.high_tension_kv = 70.
    for lens in state.lenses:
        lens.enabled = True
    after = trace_test_electron(scene, settings, use_compiled=False)
    np.testing.assert_array_equal(after.positions_m, before.positions_m)
    np.testing.assert_array_equal(after.kinetic_energy_ev, before.kinetic_energy_ev)


def test_column_progress_prefix_remains_an_executed_state(column):
    _, scene = column
    settings = TestElectronSettings(position_m=(0., 0., .5), kinetic_energy_ev=200000.,
        max_path_length_m=.01, step_m=.00025)
    prefixes = []
    result = trace_test_electron(scene, settings, use_compiled=False,
        progress=prefixes.append, progress_interval_s=0.)
    assert prefixes
    for prefix in prefixes:
        assert prefix.reason == "in_progress" and not prefix.completed
        np.testing.assert_array_equal(prefix.positions_m, result.positions_m[:len(prefix.positions_m)])
        assert not prefix.positions_m.flags.writeable


def test_column_budget_returns_only_executed_states(column):
    _, scene = column
    settings = TestElectronSettings(position_m=(0., 0., .5), kinetic_energy_ev=200000.,
        max_path_length_m=.01, step_m=.00025, max_steps=3)
    result = trace_test_electron(scene, settings, use_compiled=False)
    assert result.reason == "step_limit" and not result.completed
    assert result.steps == 3 and len(result.positions_m) == 4
    assert np.isfinite(result.positions_m).all()
    assert result.energy_invariant_error_ev < 1e-7


def test_axial_column_path_agrees_with_physical_displacement(column):
    _, scene = column
    settings = TestElectronSettings(position_m=(0., 0., .5), kinetic_energy_ev=200000.,
        max_path_length_m=.002, step_m=.00025)
    result = trace_test_electron(scene, settings, use_compiled=False)
    assert result.reason == "path_limit" and result.completed
    assert result.path_length_m[-1] == pytest.approx(.002, abs=1e-12)
    # Independent geometric check of the documented second-order v dt
    # reconstruction in a purely axial, accelerating prescribed field.
    displacement = result.positions_m[-1, 2]-result.positions_m[0, 2]
    assert displacement == pytest.approx(result.path_length_m[-1], abs=1e-10)
