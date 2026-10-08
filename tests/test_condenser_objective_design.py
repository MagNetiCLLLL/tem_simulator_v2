"""Physical invariants and guarded acceptance for the read-only CM study."""
import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.condenser_objective_design import (
    CondenserObjectiveTargets,
    condenser_objective_position_bounds,
    evaluate_condenser_objective_design,
    solve_condenser_objective_design,
)


def test_cm_bounds_follow_real_ac_body_and_upper_pole_faces():
    state = default_state()
    bounds = condenser_objective_position_bounds(state, clearance_mm=0.1)
    half = state.mini_condenser.length_mm / 2
    assert bounds.minimum_center_z_mm - half == pytest.approx(
        state.ac_deflector.lower_surface_z_mm + 0.1)
    assert bounds.maximum_center_z_mm + half == pytest.approx(
        state.objective_lens.upper_pole_piece_center_z_mm
        - state.objective_lens.upper_pole_piece_axial_length_mm / 2 - 0.1)
    with pytest.raises(ValueError, match="mechanical envelope"):
        evaluate_condenser_objective_design(state, center_z_mm=bounds.maximum_center_z_mm + 1)
    state.mini_condenser.enabled = False
    with pytest.raises(ValueError, match="installed, enabled"):
        evaluate_condenser_objective_design(state)


def test_isolated_round_lens_reversal_preserves_circular_beam_size():
    state = default_state()
    for lens in state.lenses:
        lens.enabled = lens is state.mini_condenser
    covariance = np.diag((1e-12, 1e-12, 1e-6, 1e-6))
    result = evaluate_condenser_objective_design(
        state, excitation_magnitude_percent=90,
        input_covariance=covariance,
    )
    # Changing the sign of an isolated axisymmetric field changes rotation,
    # not radial lens power. It cannot alone implement two probe modes.
    assert result.micro.rms_radius_m == pytest.approx(result.nano.rms_radius_m, rel=1e-8)
    assert result.micro.rms_mechanical_angle_rad == pytest.approx(
        result.nano.rms_mechanical_angle_rad, rel=1e-8)
    assert result.polarity_focusing_integral_difference_t2_mm == pytest.approx(0, abs=1e-12)
    assert result.micro.mechanical_transfer[0, 1] == pytest.approx(
        -result.nano.mechanical_transfer[0, 1], abs=1e-9)


def test_same_magnitude_combines_with_objective_without_changing_live_inputs():
    state = default_state()
    before_cm = vars(state.mini_condenser).copy()
    before_objective = vars(state.objective_lens).copy()
    before_tip = state.electron_gun.to_dict()
    result = evaluate_condenser_objective_design(state, excitation_magnitude_percent=90)
    assert result.micro.polarity == 1 and result.nano.polarity == -1
    assert result.micro.mechanical_transfer.shape == (4, 4)
    assert result.micro.canonical_output_transfer.shape == (4, 4)
    assert 0 < result.cm_objective_normalized_overlap < 0.02
    assert result.polarity_focusing_integral_difference_t2_mm > 0
    assert not result.targets_met and result.unmet
    assert not result.micro.mechanical_transfer.flags.writeable
    assert vars(state.mini_condenser) == before_cm
    assert vars(state.objective_lens) == before_objective
    assert state.electron_gun.to_dict() == before_tip


def test_unattainable_targets_return_best_candidate_with_unmet_and_cancel():
    state = default_state()
    result = solve_condenser_objective_design(
        state, candidate_count=2, excitation_magnitude_percent=90,
        input_covariance=np.diag((1e-12, 1e-12, 1e-6, 1e-6)),
        targets=CondenserObjectiveTargets(1e-15, 1e-15),
    )
    assert len(result.candidates) == 2
    assert result.best.objective_value > 1
    assert not result.best.targets_met
    assert len(result.best.unmet) == 2
    with pytest.raises(InterruptedError):
        solve_condenser_objective_design(state, candidate_count=2, cancel_check=lambda: True)


def test_geometry_bound_field_models_are_not_translated_as_analytic_profiles():
    state = default_state()
    state.simulation_mode = "linear_geometry"
    with pytest.raises(ValueError, match="geometry-matched"):
        evaluate_condenser_objective_design(state)
