"""Execution-local gun scratch data retain the existing step and live fields."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tracing import _enforce_static_field_energy
from temsim.physics import analytic_particle_step as stepping
from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev


pytestmark = pytest.mark.skipif(stepping._compiled_step is None, reason="Numba optional")


def setup(count=11):
    gun = FieldEmissionGun()
    gun.emitter.surface_model = None
    gun.deflector.upper_field_x_mt = .7
    gun.deflector.lower_field_y_mt = -.3
    gun.stigmator.gradient_t_per_m = 19.
    gun.stigmator.rotation_deg = 37.
    z = np.resize([.01, 1., 5., 13., 14., 41., 77., 402., 418., 433., 550.], count)
    i = np.arange(count)
    x = np.column_stack((1e-5*np.cos(i), 2e-5*np.sin(i), z*1e-3))
    invariant = np.full(count, gun.emitter.emission_energy_ev)
    electric, magnetic = gun.electric_field, gun.magnetic_field
    p = momentum_from_kinetic_energy_ev(invariant+electric.potential_v_at_global_positions(x),
        np.column_stack((.004*np.cos(i), .003*np.sin(i), np.ones(count))))
    phase = RelativisticPhaseSpace(x, p)
    active = np.ones(count, dtype=bool)
    active[::11] = False
    return gun, electric, magnetic, phase, invariant, active


@pytest.mark.parametrize("count,threads", [(11, 1), (1150, 1), (1150, 4)])
def test_prepared_step_and_time_bound_exactly_match_compatibility(count, threads):
    from numba import get_num_threads, set_num_threads
    gun, electric, magnetic, phase, invariant, active = setup(count)
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, count)
    previous = get_num_threads()
    try:
        set_num_threads(min(threads, previous))
        for proposed_dt in (1e-12, 4e-10, 1e-14):
            assert execution.begin_step(phase, active)
            bounded = execution.time_step(proposed_dt, .025)
            assert bounded == stepping.try_analytic_time_step(phase, proposed_dt,
                active, magnetic, electric, .025)
            expected, expected_dt = stepping.try_analytic_step(gun, phase, bounded,
                active, magnetic, electric, invariant[active])
            actual, actual_dt = execution.step(bounded, invariant[active])
            assert actual_dt == expected_dt
            assert actual.time_s == expected.time_s
            np.testing.assert_array_equal(actual.position_m, expected.position_m)
            np.testing.assert_array_equal(actual.momentum_kg_m_per_s, expected.momentum_kg_m_per_s)
            np.testing.assert_array_equal(actual.position_m[~active], phase.position_m[~active])
            assert not np.shares_memory(actual.position_m, phase.position_m)
            assert not np.shares_memory(actual.momentum_kg_m_per_s, phase.momentum_kg_m_per_s)
            phase = actual
    finally:
        set_num_threads(previous)


def test_fields_repack_only_on_change_and_workspace_tracks_changed_masks(monkeypatch):
    gun, electric, magnetic, phase, invariant, active = setup()
    original = stepping._field_parameters
    packed = []
    def packing(*args):
        packed.append(1)
        return original(*args)
    monkeypatch.setattr(stepping, "_field_parameters", packing)
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    workspace = tuple(id(getattr(execution, name)) for name in
        ("_indices", "_status", "_half", "_limits", "_invalid", "_out_x", "_out_p"))
    for offset in (0., 0., .3, .3, -.2):
        gun.deflector.upper_field_x_mt = .7+offset
        assert execution.begin_step(phase, active)
        actual, dt = execution.step(1e-14, invariant[active])
        count = len(packed)
        expected, expected_dt = stepping.try_analytic_step(gun, phase, 1e-14,
            active, magnetic, electric, invariant[active])
        packed.pop()  # Compatibility deliberately repacks on every call.
        assert len(packed) == count
        assert dt == expected_dt
        np.testing.assert_array_equal(actual.position_m, expected.position_m)
        np.testing.assert_array_equal(actual.momentum_kg_m_per_s, expected.momentum_kg_m_per_s)
        active[:] = ~active
    assert len(packed) == 3
    assert workspace == tuple(id(getattr(execution, name)) for name in
        ("_indices", "_status", "_half", "_limits", "_invalid", "_out_x", "_out_p"))


@pytest.mark.parametrize("component,attribute,value", [
    ("emitter", "emission_energy_ev", .61),
    ("extractor", "voltage_kv", 6.),
    ("extractor", "transition_end_mm", 6.2),
    ("extractor", "field_center_offset_mm", .1),
    ("electrostatic_lens", "potential_scale", 4.3),
    ("electrostatic_lens", "mechanical_center_from_tip_mm", 15.),
    ("electrostatic_lens", "soft_edge_mm", 1.7),
    ("accelerator", "high_tension_kv", 250.),
    ("accelerator", "field_center_offset_mm", .1),
    ("deflector", "beam_blanked", True),
    ("deflector", "enabled", False),
    ("deflector", "coil_length_mm", 7.),
    ("stigmator", "rotation_deg", 53.),
    ("stigmator", "enabled", False),
    ("stigmator", "field_center_offset_mm", .13),
])
def test_current_consumed_field_inputs_are_not_frozen(component, attribute, value):
    gun, electric, magnetic, phase, invariant, active = setup()
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    before = execution._parameters
    setattr(getattr(gun, component), attribute, value)
    assert execution.begin_step(phase, active)
    assert execution._parameters is not before
    current = stepping._field_parameters(electric, magnetic)
    for actual, expected in zip(execution._parameters, current):
        np.testing.assert_array_equal(actual, expected)


def test_stage_replacement_and_nonfinite_fields_invalidate_prepared_parameters():
    gun, electric, magnetic, phase, invariant, active = setup()
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    stages = list(gun.accelerator.stages)
    stages[0] = replace(stages[0], voltage_fraction=stages[0].voltage_fraction*.7)
    gun.accelerator.stages = tuple(stages)
    assert execution.begin_step(phase, active)
    for a, b in zip(execution._parameters, stepping._field_parameters(electric, magnetic)):
        np.testing.assert_array_equal(a, b)
    gun.deflector.upper_field_x_mt = float("nan")
    assert not execution.begin_step(phase, active)
    assert execution.time_step(1e-14, .025) is None
    assert execution.step(1e-14, invariant[active]) is None


def test_custom_or_disabled_providers_and_unavailable_numba_fall_back(monkeypatch):
    gun, electric, magnetic, phase, invariant, active = setup()
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    magnetic.compiled_particle_steps = False
    assert not execution.begin_step(phase, active)
    assert stepping.prepare_analytic_execution(gun, magnetic, electric, len(active)) is None
    magnetic.compiled_particle_steps = True
    assert execution.begin_step(phase, active)
    magnetic.field_at_global_positions_t = lambda points: np.zeros_like(points)
    assert not execution.begin_step(phase, active)
    assert execution.project_momentum(phase.position_m, phase.momentum_kg_m_per_s, invariant) is None
    del magnetic.field_at_global_positions_t
    monkeypatch.setattr(stepping, "_compiled_prepared_step", None)
    assert stepping.prepare_analytic_execution(gun, magnetic, electric, len(active)) is None
    assert not execution.begin_step(phase, active)


def test_class_overrides_gun_disable_and_noncanonical_phase_defer(monkeypatch):
    gun, electric, magnetic, phase, invariant, active = setup()
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    gun.compiled_particle_steps = False
    assert not execution.begin_step(phase, active)
    gun.compiled_particle_steps = True
    assert execution.begin_step(phase, active)
    # Normal PhaseSpace construction owns float64 arrays. A caller that later
    # replaces one with a lower-precision array must retain compatibility-path
    # rounding, rather than being silently promoted by a float64 workspace.
    original_position = phase.position_m
    phase.position_m = original_position.astype(np.float32)
    assert not execution.begin_step(phase, active)
    phase.position_m = original_position
    monkeypatch.setattr(type(gun.stigmator), "field_at_global_positions_t",
                        lambda self, points: np.zeros_like(points))
    assert not execution.begin_step(phase, active)
    assert stepping.prepare_analytic_execution(gun, magnetic, electric, len(active)) is None


@pytest.mark.parametrize("count", [11, 1150])
def test_projection_retains_numpy_operation_order_and_current_parameters(count):
    gun, electric, magnetic, phase, invariant, active = setup(count)
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, count)
    for change in (0., .25, -.1):
        gun.extractor.voltage_kv = 4.+change
        actual = execution.project_momentum(phase.position_m, phase.momentum_kg_m_per_s, invariant)
        expected = _enforce_static_field_energy(gun, phase.position_m, phase.momentum_kg_m_per_s, invariant)
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, phase.momentum_kg_m_per_s)


def test_empty_active_mask_keeps_state_and_new_step_is_required():
    gun, electric, magnetic, phase, invariant, active = setup()
    active[:] = False
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    assert execution.begin_step(phase, active)
    assert execution.time_step(1e-14, .025) == 1e-14
    advanced, dt = execution.step(1e-14, invariant[active])
    np.testing.assert_array_equal(advanced.position_m, phase.position_m)
    np.testing.assert_array_equal(advanced.momentum_kg_m_per_s, phase.momentum_kg_m_per_s)
    assert advanced.time_s == dt == 1e-14
    assert execution.step(1e-14, invariant[active]) is None


@pytest.mark.parametrize("count,failure", [(11, "zero"), (1150, "zero"), (1150, "field")])
def test_prepared_path_retains_all_particle_errors(count, failure):
    gun, electric, magnetic, phase, invariant, active = setup(count)
    active[-1] = True
    if failure == "zero":
        phase.position_m[-1, 2] = 1.
        phase.momentum_kg_m_per_s[-1] = 0.
    else:
        phase.position_m[-1, 0] = 1e300
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, count)
    assert execution.begin_step(phase, active)
    with pytest.raises(ValueError):
        execution.step(1e-14, invariant[active])
