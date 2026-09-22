"""Bounded classical response tests; elementary limits are not full-chain acceptance."""
import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import beam_observation as observation
from temsim.physics.first_order import trace_transverse_transfer
from temsim.simulation_modes import switch_mode
from temsim.physics.core import propagate


def endpoint(state, x, tx, y, ty):
    return propagate(state, 500., 600., np.atleast_1d(x), np.atleast_1d(tx),
        np.atleast_1d(y), np.atleast_1d(ty), checkpoint_z_mm=(600.,), return_checkpoints=True)[-1]


def field_free_fixture():
    """Local mathematical limit only, never an application source/preset."""
    state = default_state()
    switch_mode(state, "analytical")
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = .25
    state.history_step_mm = 1.
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        component.enabled = False
    return state


def test_first_order_map_uses_full_precision_not_drawing_history():
    state = field_free_fixture()
    length_m = (600.1234 - 500.) * 1e-3
    transfer = trace_transverse_transfer(state, 500., 600.1234)
    np.testing.assert_allclose(transfer.j_diff_m_per_rad, np.eye(2)*length_m, rtol=0, atol=1e-13)


def test_scan_response_is_the_on_axis_derivative_not_a_nonlinear_test_ray():
    state = field_free_fixture()
    hp = next(c for c in state.corrector_elements if c.kind == "hexapole")
    hp.optical_reference_from_tip_mm = 550.
    hp.strength_m3 = min(1e6, hp.maximum_strength_m3)
    hp.enabled = True
    # A centred hexapole has zero derivative at the axis. Its nonlinear
    # forces still act on finite rays in the actual particle propagator.
    position, angle = observation.transverse_kick_phase_space_response(state, 500., 600.)
    np.testing.assert_allclose(position, .1*np.eye(2), rtol=0, atol=1e-13)
    np.testing.assert_allclose(angle, np.eye(2), rtol=0, atol=1e-12)
    z, path = observation.transverse_kick_response_path(state, 500., 600., save_z_mm=(573.1234,))
    np.testing.assert_allclose(path[-1], position, rtol=0, atol=1e-13)
    index = np.flatnonzero(z == 573.1234).item()
    np.testing.assert_allclose(path[index], .0731234*np.eye(2), rtol=0, atol=1e-13)


def test_stigmator_sign_exchanges_axes_and_skew_rotates_the_response():
    state = field_free_fixture()
    stig = state.stigmators[0]
    stig.z_mm = 550.
    stig.enabled = True
    stig.field_model = "normal_skew"
    stig.strength_x_percent = 10.
    stig.strength_y_percent = 0.
    positive = endpoint(state, 1e-6, 0., 1e-6, 0.)
    stig.strength_x_percent = -10.
    negative = endpoint(state, 1e-6, 0., 1e-6, 0.)
    assert positive.x_m.item() < 1e-6 < positive.y_m.item()
    np.testing.assert_allclose(positive.x_m, negative.y_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(positive.y_m, negative.x_m, rtol=0, atol=1e-15)
    # Pure Y is the same quadrupole physically rotated by 45 degrees.
    stig.strength_x_percent = 0.
    stig.strength_y_percent = 10.
    skew = endpoint(state, 0., 0., np.sqrt(2)*1e-6, 0.)
    np.testing.assert_allclose(skew.x_m, (positive.x_m-positive.y_m)/np.sqrt(2), rtol=0, atol=1e-15)
    np.testing.assert_allclose(skew.y_m, (positive.x_m+positive.y_m)/np.sqrt(2), rtol=0, atol=1e-15)


def test_hexapole_rotation_quadratic_limit_and_step_refinement(record_property):
    import json
    from scipy.integrate import solve_ivp
    state = field_free_fixture()
    hp = next(c for c in state.corrector_elements if c.kind == "hexapole")
    hp.optical_reference_from_tip_mm = 550.
    hp.effective_length_mm = 6.
    hp.strength_m3 = 1e6
    hp.enabled = True
    radius = np.array([1e-6, 2e-6])
    zero = np.zeros(2)
    results = []
    for step in (1., .5, .25):
        state.step_mm = step
        hp.orientation_rad = 0.
        points = endpoint(state, radius, zero, zero, zero)
        results.append(np.r_[points.x_m[-1], points.tx_rad[-1]])
    # Independent adaptive integration of the scalar on-axis-plane equation
    # x'' = -H(z) x², using metre coordinates, not the production RK4 routine.
    sigma_m = hp.effective_length_mm/2.355 * 1e-3
    def equation(z, values):
        x, angle = values[:2], values[2:]
        h = hp.strength_m3*np.exp(-.5*((z-.55)/sigma_m)**2)
        return np.r_[angle, -h*x*x]
    reference = solve_ivp(equation, (.5, .6), np.r_[radius, zero], method="DOP853",
                         max_step=1e-4, rtol=1e-11, atol=1e-17).y[:, -1]
    position_errors = [float(np.max(np.abs(v[:2]-reference[:2]))) for v in results]
    angle_errors = [float(np.max(np.abs(v[2:]-reference[2:]))) for v in results]
    for errors, budget in ((position_errors, 1e-14), (angle_errors, 1e-13)):
        assert errors[-1] < budget and errors[1] < errors[0] and errors[2] < errors[1]
    # Weak-deflection, fixed geometry limit; finite nonlinear paths need not
    # scale exactly with initial radius outside this declared small range.
    np.testing.assert_allclose(points.tx_rad[0, 1]/points.tx_rad[0, 0], 4., rtol=1e-3)
    base = points.x_m + 1j*points.y_m
    hp.orientation_rad = 2*np.pi/3
    repeated = endpoint(state, radius, zero, zero, zero)
    np.testing.assert_allclose(repeated.x_m + 1j*repeated.y_m, base, rtol=0, atol=1e-18)
    hp.orientation_rad = np.pi/3
    rotated = endpoint(state, radius, zero, zero, zero)
    hp.orientation_rad = 0.
    hp.strength_m3 *= -1
    reversed_field = endpoint(state, radius, zero, zero, zero)
    np.testing.assert_allclose(rotated.x_m + 1j*rotated.y_m,
                              reversed_field.x_m + 1j*reversed_field.y_m, rtol=0, atol=1e-18)
    assert points.tx_rad[0, 0] < 0 < reversed_field.tx_rad[0, 0]
    record_property("hexapole_step_mm", json.dumps([1., .5, .25]))
    record_property("hexapole_max_position_error_m", json.dumps(position_errors))
    record_property("hexapole_max_angle_error_rad", json.dumps(angle_errors))


def test_scan_frame_period_changes_timing_not_phase_path_and_gain_reverses_drive():
    state = default_state()
    ac = state.ac_deflector
    ac.wobble_enabled = False
    ac.scan_enabled = True
    ac.set_scan_command_matrix_mrad([[.1, .02], [-.03, .12]])
    phases = np.array([.013, .29, .673, .99])
    before = np.array([ac.scan_kick_mrad(t*ac.scan_frame_period_s) for t in phases])
    ac.scan_frame_period_s *= 3
    after = np.array([ac.scan_kick_mrad(t*ac.scan_frame_period_s) for t in phases])
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-14)
    old_kicks = np.array(ac.coil_kicks_mrad(*before[1]))
    ac.upper_coil_gain *= -2
    np.testing.assert_allclose(ac.coil_kicks_mrad(*before[1]), -2*old_kicks, rtol=0, atol=1e-15)


def test_response_model_change_invalidates_dependent_cached_products(monkeypatch):
    from temsim.physics import first_order
    from temsim.calculation_cache import calculation_signatures, matching_products
    state = default_state()
    current = calculation_signatures(state)
    monkeypatch.setattr(first_order, "FIRST_ORDER_RESPONSE_SCHEMA", "historical-one-sided-plot-precision")
    previous = calculation_signatures(state)
    assert not {"incident", "column", "scan_geometry", "scan_ray_paths", "stem"} & matching_products(previous, current)


def test_tip_origin_stigmator_response_is_locally_signed_and_survives_column_refinement(record_property):
    import json
    from temsim.alignment_constraints import evaluate_incident
    state = default_state()
    state.electron_gun.emitter.ray_count = 49
    # 0.5 -> 0.25 mm gave a 24.5 nm displacement difference in the full
    # installed column; 0.25 -> 0.125 mm still exceeded the 1 urad angular
    # budget. Refine the integration instead of relaxing either error budget.
    state.step_mm = .125
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    switch_mode(state, "analytical")
    assert state.electron_gun.emitter.coherence is None
    assert not state.vacuum_map.enabled
    stig = state.stigmators[0]
    stig.field_model = "normal_skew"
    def measured(x, y):
        stig.strength_x_percent, stig.strength_y_percent = x, y
        return evaluate_incident(state)[1]
    baseline = measured(0., 0.)
    live = baseline["alive"]
    assert np.count_nonzero(live) >= 3
    def difference(values, reference):
        np.testing.assert_array_equal(values["alive"], baseline["alive"])
        for name in ("weight", "energy_offset_ev", "gun_ray_id"):
            np.testing.assert_array_equal(values[name], baseline[name])
        # Dimensionless vector: fixed diagnostic scales, not fitted weights.
        return np.concatenate([(values[name]-reference[name])[live]/scale for name, scale in
            (("x_m", 1e-6), ("y_m", 1e-6), ("tx_rad", 1e-3), ("ty_rad", 1e-3))])
    derivatives, errors = [], []
    positive_runs = []
    for dx, dy in ((.1, 0.), (0., .1)):
        plus, minus = measured(dx, dy), measured(-dx, -dy)
        forward, reverse = difference(plus, baseline), difference(minus, baseline)
        assert np.linalg.norm(forward) > 1e-12
        error = np.linalg.norm(forward+reverse)/np.linalg.norm(forward-reverse)
        assert error < .01  # bounded +/-0.1 percent, same incident ensemble
        derivatives.append((forward-reverse)/.2)
        positive_runs.append((dx, dy, plus))
        errors.append(float(error))
    assert np.linalg.matrix_rank(np.column_stack(derivatives)) == 2
    state.step_mm /= 2
    differences = []
    for dx, dy, coarse in positive_runs:
        refined = measured(dx, dy)
        difference(refined, coarse)  # also checks survivor and source identity
        for name, budget in (("x_m", 1e-8), ("y_m", 1e-8), ("tx_rad", 1e-6), ("ty_rad", 1e-6)):
            error = float(np.max(np.abs(refined[name][live]-coarse[name][live])))
            differences.append(dict(channel="X" if dx else "Y", observable=name, error=error, budget=budget))
            assert error < budget
    record_property("tip_origin_local_odd_response_relative_errors", json.dumps(errors))
    record_property("tip_origin_surviving_rays", int(np.count_nonzero(live)))
    record_property("tip_origin_column_step_differences", json.dumps(differences))
