"""Shared captured-field and section-boundary contracts, not microscope validation.

The default instrument check uses its actual component graph. Short bare-state
fixtures isolate finite-coil support, specimen coordinates and cache identity;
they do not replace the tip-origin source or qualify the full transport chain.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import prepare_magnetic_scene
from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
from temsim.optics.column import default_state
from temsim.optics.model import DeflectorPair, Stigmator
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.instrument_magnetic import (
    capture_instrument_magnetic_field, column_dipole_fields,
    events_overlapping_interval,
)
from temsim.specimen.vector_field_transport import SpecimenFieldTransport


def bare_state(**kwargs):
    values = dict(lenses=(), stigmators=(), corrector_elements=(), deflectors=(),
                  apertures=(), beam_voltage_kv=300., simulation_mode="analytical",
                  simulation_time_s=.237, sample=SimpleNamespace(z_mm=20.),
                  step_mm=.2, history_step_mm=.2, acceleration_enabled=False,
                  acceleration_backend="CPU", equivalent_image_lenses_enabled=False)
    values.update(kwargs)
    return SimpleNamespace(**values)


def coil_events(state):
    return tuple((coil.event_z_mm, coil.event_dx_rad, coil.event_dy_rad)
                 for coil in column_dipole_fields(state))


@pytest.fixture
def active_default():
    state = default_state()
    state.stigmators[0].strength_x_percent = 17.
    state.stigmators[0].strength_y_percent = -11.
    state.deflectors[0].upper_x_mrad = .8
    state.deflectors[0].upper_y_mrad = -.3
    state.deflectors[0].lower_x_mrad = -.2
    return state


def test_default_captured_graph_equals_diagnostic_sum_and_isolated_from_live_edits(active_default):
    state = active_default
    captured = capture_instrument_magnetic_field(state)
    scene = prepare_magnetic_scene(state)
    lens, stigmator, deflector = state.lenses[0], state.stigmators[0], state.deflectors[0]
    z = (lens.z_mm, stigmator.z_mm, deflector.upper_z_mm, deflector.lower_z_mm)
    points = np.array([(2e-8, -3e-8, plane * 1e-3) for plane in z])
    assert all(scene.diagnostic_position_is_valid(point) for point in points)
    expected = np.stack([scene.diagnostic_field_at_global_position_t(point) for point in points])
    before = captured.field_at_global_positions_t(points)
    np.testing.assert_allclose(before, expected, rtol=2e-15, atol=1e-18)
    np.testing.assert_array_equal(captured.field_at_global_positions_t(points.reshape(2, 2, 3)),
                                  before.reshape(2, 2, 3))
    assert before[0, 2] != 0. and np.linalg.norm(before[1:, :2]) > 0.
    assert {"lens", "stigmator", "deflector"} <= set(scene.source_categories)
    assert captured.physical_identity == scene.physical_identity
    # The production sum and bounded diagnostic sampler have different query
    # contracts. They share captured physics, not a false numerical identity.
    assert captured.numerical_identity and scene.numerical_identity
    assert captured.numerical_identity != scene.numerical_identity
    assert capture_instrument_magnetic_field(state).numerical_identity == captured.numerical_identity

    lens.percent *= .7
    stigmator.strength_x_percent *= -1.
    deflector.upper_x_mrad += .5
    state.electron_gun.high_tension_kv = 200.
    state.simulation_time_s = .125
    after = capture_instrument_magnetic_field(state)
    np.testing.assert_array_equal(captured.field_at_global_positions_t(points), before)
    np.testing.assert_array_equal(np.stack([scene.diagnostic_field_at_global_position_t(p) for p in points]), before)
    assert after.physical_identity != captured.physical_identity
    assert not np.array_equal(after.field_at_global_positions_t(points), before)


def test_virtual_energy_changes_response_without_rebuilding_captured_b(active_default, monkeypatch):
    state = active_default
    scene = prepare_magnetic_scene(state)
    captured = capture_instrument_magnetic_field(state)
    point = (0., 0., state.deflectors[0].upper_z_mm * 1e-3)
    before = captured.field_at_global_positions_t(point)
    identity = (scene.physical_identity, scene.numerical_identity)
    settings = TestElectronSettings(kinetic_energy_ev=300_000., position_m=point,
        max_path_length_m=4e-5, step_m=4e-6, max_steps=500,
        relative_tolerance=1e-6, position_tolerance_m=1e-13)
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene",
                        lambda *_a, **_k: pytest.fail("Changing virtual energy must not recapture B"))
    high = trace_test_electron(scene, settings, use_compiled=False)
    low = trace_test_electron(scene, replace(settings, kinetic_energy_ev=80_000.), use_compiled=False)
    assert high.completed and low.completed
    assert np.linalg.norm(low.directions[-1, :2]) > np.linalg.norm(high.directions[-1, :2])
    np.testing.assert_array_equal(captured.field_at_global_positions_t(point), before)
    assert (scene.physical_identity, scene.numerical_identity) == identity
    assert (high.physical_field_identity, high.numerical_field_identity) == identity
    assert (low.physical_field_identity, low.numerical_field_identity) == identity


def test_display_window_cannot_clip_the_production_capture(active_default):
    state = active_default
    first = capture_instrument_magnetic_field(state)
    point = (0., 0., state.lenses[0].z_mm * 1e-3)
    value = first.field_at_global_positions_t(point)
    assert value[2] != 0.
    clipped = prepare_magnetic_scene(state, z_limits_mm=(1800., 1900.))
    assert clipped.diagnostic_field_at_global_position_t(point) is None
    later = capture_instrument_magnetic_field(state)
    np.testing.assert_array_equal(first.field_at_global_positions_t(point), value)
    np.testing.assert_array_equal(later.field_at_global_positions_t(point), value)
    assert later.numerical_identity == first.numerical_identity


def test_specimen_local_nanometres_query_same_crossing_coil_and_frozen_sum():
    coil = DeflectorPair("Deflector", "d", 19., 29., 1.2, -.8, 0., 0., thickness_mm=4.)
    stigmator = Stigmator("Stigmator", "s", 20., strength_x_percent=15.)
    state = bare_state(deflectors=(coil,), stigmators=(stigmator,))
    captured = capture_instrument_magnetic_field(state)
    transport = SpecimenFieldTransport(state)
    local_nm = np.array(((10., -20., -50.), (-30., 10., 0.), (0., 0., 50.)))
    local_m = local_nm * 1e-9
    global_m = local_m + (0., 0., state.sample.z_mm * 1e-3)
    expected = captured.field_at_global_positions_t(global_m)
    assert np.all(np.linalg.norm(expected[:, :2], axis=1) > 0.)
    np.testing.assert_array_equal(transport.field_at_global_positions_t(local_m), expected)
    np.testing.assert_array_equal(transport.field_at_global_positions_t(local_m[1]), expected[1])
    assert transport.magnetic_field.numerical_identity == captured.numerical_identity
    # Also evaluate a point beyond the downstream coil end in local nm.
    outside_nm = np.array((10., 20., 1.1e6))
    outside_m = outside_nm * 1e-9
    np.testing.assert_array_equal(transport.field_at_global_positions_t(outside_m),
        captured.field_at_global_positions_t(outside_m + transport.origin_m))
    coil.upper_x_mrad *= 2.
    stigmator.strength_x_percent *= 2.
    np.testing.assert_array_equal(transport.field_at_global_positions_t(local_m), expected)
    assert not np.array_equal(SpecimenFieldTransport(state).field_at_global_positions_t(local_m), expected)


def test_axial_identity_tracks_only_intersecting_support_and_zero_to_active_sources():
    near = DeflectorPair("Near", "near", 25., 35., thickness_mm=4.)
    distant = DeflectorPair("Far", "far", 100., 110., 1., 0., 0., 0., thickness_mm=4.)
    state = bare_state(deflectors=(near, distant))
    first = capture_instrument_magnetic_field(state)
    extent = (.024, .026)
    baseline = first.identity_for_axial_range(*extent)
    assert baseline is not None
    assert next(source for source in first._sources if source.key == "near:0").known_zero
    distant.upper_x_mrad = 2.
    distant_changed = capture_instrument_magnetic_field(state)
    assert distant_changed.numerical_identity != first.numerical_identity
    assert distant_changed.identity_for_axial_range(*extent) == baseline
    near.lower_x_mrad = .7  # Same component, but this coil is outside this axial interval.
    other_coil_changed = capture_instrument_magnetic_field(state)
    assert other_coil_changed.identity_for_axial_range(*extent) == baseline
    near.upper_x_mrad = .5
    local_changed = capture_instrument_magnetic_field(state)
    assert local_changed.identity_for_axial_range(*extent) != baseline
    assert first.identity_for_axial_range(*extent) == baseline
    np.testing.assert_array_equal(first.field_at_global_positions_t((0., 0., .025)), 0.)
    assert local_changed.field_at_global_positions_t((0., 0., .025))[1] != 0.
    # A fringe segment is a dependency even though its centre is outside it.
    assert local_changed.identity_for_axial_range(.0225, .0235) != first.identity_for_axial_range(.0225, .0235)


@pytest.mark.parametrize("limits", [(0., 0.), (.03, .02), (float("nan"), .02), (0., float("inf"))])
def test_axial_identity_rejects_invalid_range(limits):
    captured = capture_instrument_magnetic_field(bare_state())
    with pytest.raises(ValueError, match="finite and increasing"):
        captured.identity_for_axial_range(*limits)


def _execute(state, start, stop, events, initial=None, *, initial_kicks=True):
    selected = events_overlapping_interval(state, events, start, stop)
    plan = build_propagation_plan(state, start, stop, selected,
        include_spherical_aberration=False, include_hexapole=False,
        save_z_mm=(start, stop), checkpoint_z_mm=(stop,))
    if initial is None:
        initial = tuple(np.zeros(1) for _ in range(4))
    result = execute_propagation_plan(state, plan, *initial,
                                     include_initial_plane_kicks=initial_kicks)
    checkpoints = result[-1]
    endpoint = tuple(getattr(checkpoints, name)[-1] for name in
                     ("x_m", "tx_rad", "y_m", "ty_rad"))
    assert all(array.dtype == np.float64 for array in endpoint)
    return plan, endpoint


def test_crossing_coils_remain_in_both_sections_when_centres_are_outside_them():
    coil = DeflectorPair("Deflector", "d", 19., 21., 1., -.5, -.4, .7, thickness_mm=4.)
    state = bare_state(deflectors=(coil,))
    events = coil_events(state)
    assert events_overlapping_interval(state, events, 15., 20.) == events
    assert events_overlapping_interval(state, events, 20., 25.) == events
    assert events_overlapping_interval(state, events, 13., 17.) == ()
    assert events_overlapping_interval(state, events, 23., 27.) == ()
    full_plan, full = _execute(state, 15., 25., events)
    first_plan, first = _execute(state, 15., 20., events)
    second_plan, resumed = _execute(state, 20., 25., events, first, initial_kicks=False)
    for plan in (full_plan, first_plan, second_plan):
        assert np.count_nonzero(plan.dipole_by_t) > 0
        np.testing.assert_array_equal(plan.kick_x_rad, 0.)
        np.testing.assert_array_equal(plan.kick_y_rad, 0.)
    np.testing.assert_allclose(resumed, full, rtol=2e-12, atol=1e-15)


def test_thin_event_on_cutoff_is_applied_once_by_after_action_continuation():
    state = bare_state()
    events = ((20., .003, -.002),)
    # Closed input intervals are intentional: the execution checkpoint owns
    # whether the starting-plane action has already been performed.
    assert events_overlapping_interval(state, events, 15., 20.) == events
    assert events_overlapping_interval(state, events, 20., 25.) == events
    _, full = _execute(state, 15., 25., events)
    _, first = _execute(state, 15., 20., events)
    _, resumed = _execute(state, 20., 25., events, first, initial_kicks=False)
    np.testing.assert_allclose(resumed, full, rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose((resumed[1][0], resumed[3][0]), (.003, -.002), rtol=2e-15)
    np.testing.assert_allclose((resumed[0][0], resumed[2][0]), (15e-6, -10e-6), rtol=2e-14)


def test_crossing_coil_and_independent_thin_event_do_not_replace_each_other():
    coil = DeflectorPair("Deflector", "d", 19., 21., .4, -.2, -.1, .3, thickness_mm=4.)
    state = bare_state(deflectors=(coil,))
    thin = (20., .001, -.002)
    events = (*coil_events(state), thin)
    full_plan, full = _execute(state, 15., 25., events)
    _, first = _execute(state, 15., 20., events)
    _, resumed = _execute(state, 20., 25., events, first, initial_kicks=False)
    np.testing.assert_allclose(resumed, full, rtol=2e-12, atol=1e-15)
    assert np.sum(full_plan.kick_x_rad) == thin[1]
    assert np.sum(full_plan.kick_y_rad) == thin[2]
    assert np.count_nonzero(full_plan.dipole_by_t) > 0
