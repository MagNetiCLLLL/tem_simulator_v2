"""Bounded classical trends; executed gun and installed first-lens segment.

No coherent source, downstream replacement source, measured calibration or
whole-column/image qualification. Lens-only limits are labelled separately.
Numerical comparisons hold emitted particles and observation planes fixed.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.core import propagate, fields, electron


@pytest.fixture(scope="module")
def gun_state():
    state = default_state()
    assert state.electron_gun.emitter.surface_model is None
    assert state.electron_gun.emitter.coherence is None
    assert not state.vacuum_map.enabled
    return state


@pytest.mark.parametrize("count", [49, 193])
def test_prescribed_current_scales_executed_gun_flux_without_changing_paths(gun_state, count, record_property):
    gun = deepcopy(gun_state.electron_gun)
    before = gun.trace_to_exit(count)
    gun.emitter.emission_current_na *= 2
    after = gun.trace_to_exit(count)
    assert after is not before
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "alive", "weight"):
        np.testing.assert_array_equal(getattr(after.exit_bundle, name), getattr(before.exit_bundle, name))
    assert before.exit_bundle.alive.any()
    for name in ("emitted_current_a", "dpa_transmitted_current_a", "c1_transmitted_current_a"):
        assert getattr(after, name) == pytest.approx(2 * getattr(before, name), rel=1e-13)
    assert before.exit_bundle.weight.sum() == pytest.approx(1., abs=1e-14)
    record_property("exit_current_ratio", after.c1_transmitted_current_a / before.c1_transmitted_current_a)


@pytest.mark.parametrize("count", [49, 193])
def test_nested_gun_apertures_have_nonincreasing_losses(gun_state, count, record_property):
    gun = deepcopy(gun_state.electron_gun)
    radii = [0., .0005, .001, .002, .004, .01]  # mm, same physical C1 plane
    previous = np.zeros(count, dtype=bool)
    currents = []
    emitted = gun.emit(count)
    for radius in radii:
        gun.c1_aperture.radius_mm = radius
        trace = gun.trace_to_exit(count)
        transmitted = trace.exit_bundle.alive
        assert np.all(~previous | transmitted)
        assert np.array_equal(trace.exit_bundle.ray_id, emitted.ray_id)
        assert np.array_equal(trace.exit_bundle.weight, emitted.weight)
        expected = gun.emitted_current_a * emitted.weight[transmitted].sum()
        assert trace.c1_transmitted_current_a == pytest.approx(expected, abs=1e-20)
        assert 0 <= expected <= trace.dpa_transmitted_current_a + 1e-20
        currents.append(expected)
        previous = transmitted
    assert currents[0] == 0 and currents[-1] > 0
    assert any(0 < current < currents[-1] for current in currents)
    record_property("radii_mm", json.dumps(radii))
    record_property("transmitted_current_a", json.dumps(currents))


@pytest.mark.parametrize("transition_start_mm", [.05, -1.])
def test_gun_handoff_energy_keeps_actual_launch_potential(gun_state, transition_start_mm, record_property):
    gun = deepcopy(gun_state.electron_gun)
    gun.extractor.transition_start_mm = transition_start_mm
    emitted = gun.emit(49)
    trace = gun.trace_to_exit(49)
    live = trace.exit_bundle.alive
    assert live.any()
    launch = trace.emission_reference["position_m"]
    exit_positions = np.column_stack((trace.exit_bundle.x_m, trace.exit_bundle.y_m,
                                     np.full(49, gun.exit_plane_z_mm * 1e-3)))
    potential = gun.electric_field.potential_v_at_global_positions
    # Independent energy bookkeeping at actual physical endpoints, in eV.
    expected = (gun.emitter.emission_energy_ev + emitted.energy_offset_ev
                + potential(exit_positions) - potential(launch))
    actual = gun.nominal_exit_energy_ev + trace.exit_bundle.energy_offset_ev
    error = float(np.max(np.abs(actual[live] - expected[live])))
    record_property("maximum_energy_error_ev", error)
    np.testing.assert_allclose(actual[live], expected[live], rtol=0, atol=1e-8)


def test_installed_lens_field_polarity_and_strength_have_bounded_analytic_limits(gun_state, record_property):
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = state.active_backend = "CPU"
    lens = state.lenses[0]
    # Field-only limiting case: actual installed geometry, ideal provider,
    # fixed energy. This is not a replacement source in the application.
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "ideal")
    z = np.linspace(450., 550., 4001)
    integrals = []
    for strength in (1., 2., 4.):
        lens.percent = strength
        lens.polarity = 1
        positive = fields(z, state)[0]
        lens.polarity = -1
        negative = fields(z, state)[0]
        np.testing.assert_allclose(negative, -positive, rtol=1e-13, atol=1e-15)
        integrals.append(float(np.trapezoid(positive**2, z*1e-3)))
    np.testing.assert_allclose(np.array(integrals)/integrals[0], [1, 4, 16], rtol=1e-12)
    record_property("relative_field_squared_integrals", json.dumps((np.array(integrals)/integrals[0]).tolist()))


def test_tip_origin_first_lens_polarity_and_step_refinement(gun_state, record_property):
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = state.active_backend = "CPU"
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "analytical")
    # Trace the actual upstream gun. The completed output is consumed directly;
    # extraction, acceleration and gun stops are never replaced by a fixture.
    gun = state.electron_gun
    coarse = gun.trace_to_exit(49)
    gun.trace_step_mm /= 2
    gun.drift_step_mm /= 2
    refined = gun.trace_to_exit(49)
    live = coarse.exit_bundle.alive & refined.exit_bundle.alive
    assert live.any()
    np.testing.assert_array_equal(coarse.exit_bundle.alive, refined.exit_bundle.alive)
    left = np.stack([getattr(coarse.exit_bundle, n)[live] for n in ("x_m", "y_m")])
    right = np.stack([getattr(refined.exit_bundle, n)[live] for n in ("x_m", "y_m")])
    # Position and angular budgets are independent; these are local error
    # budgets (10 nm and 1 microradian), not a full-source convergence claim.
    assert float(np.max(np.abs(right-left))) < 1e-8
    for name in ("tx_rad", "ty_rad"):
        assert np.max(np.abs(getattr(coarse.exit_bundle, name)[live]
                             - getattr(refined.exit_bundle, name)[live])) < 1e-6
    bundle = refined.exit_bundle
    lens = state.lenses[0]
    lens.percent = 2.  # bounded weak-field diagnostic, never a production edit
    results = []
    node_counts = []
    for step in (.4, .2, .1):
        state.step_mm = state.history_step_mm = step
        signed = []
        for polarity in (1, -1):
            lens.polarity = polarity
            z, _x, _tx, _y, _ty, checkpoints = propagate(state, gun.exit_plane_z_mm, 550.,
                bundle.x_m[live], bundle.tx_rad[live], bundle.y_m[live], bundle.ty_rad[live],
                energy_offset_ev=bundle.energy_offset_ev[live],
                checkpoint_z_mm=(550.,), return_checkpoints=True)
            # Display histories are float32. Numerical evidence must use the
            # full-precision executed checkpoint, not a quantized plot.
            assert checkpoints.x_m.dtype == np.float64
            signed.append(checkpoints.x_m[-1] + 1j*checkpoints.y_m[-1])
        node_counts.append(len(z))
        np.testing.assert_allclose(np.abs(signed[0]), np.abs(signed[1]), rtol=2e-6, atol=1e-12)
        lens.polarity = 1
        # Energy spread remains per particle, including its rotation response.
        q, _, _ = electron(state)
        from temsim.optics.lens_focal_length import electron_momentum
        momentum = np.array([electron_momentum((state.beam_voltage_kv*1000+e)/1000)
                             for e in bundle.energy_offset_ev[live]])
        theta = -q*np.trapezoid(fields(z, state)[0], z*1e-3)/(2*momentum)
        np.testing.assert_allclose(signed[1], signed[0]*np.exp(-2j*theta), rtol=2e-5, atol=1e-12)
        results.append(signed[0])
    assert node_counts[0] < node_counts[1] < node_counts[2]
    differences = [float(np.max(np.abs(b-a))) for a, b in zip(results, results[1:])]
    for coarse_result, fine_result in zip(results, results[1:]):
        np.testing.assert_allclose(fine_result, coarse_result, rtol=2e-5, atol=1e-12)
    assert differences[1] <= differences[0] + 1e-15  # allow a floating-point floor
    record_property("gun_maximum_position_step_difference_m", float(np.max(np.abs(right-left))))
    record_property("column_integration_nodes", json.dumps(node_counts))
    record_property("column_position_step_differences_m", json.dumps(differences))


def test_handoff_energy_change_invalidates_gun_and_downstream_cache(gun_state, monkeypatch):
    from temsim.optics.electron_gun import tracing
    from temsim.calculation_cache import calculation_signatures, matching_products
    state = default_state()
    with monkeypatch.context() as patch:
        patch.setattr(tracing, "ANALYTIC_ENERGY_SCHEMA", "launch-potential-reference-v1")
        old_gun = state.electron_gun._cache_key(49)
        old_stages = calculation_signatures(state)
    assert state.electron_gun._cache_key(49) != old_gun
    assert not {"incident", "column", "elastic", "eds", "stem", "tem"} & matching_products(
        old_stages, calculation_signatures(state))
