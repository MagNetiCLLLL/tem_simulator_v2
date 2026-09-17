"""Independent deterministic quadrature preserves the physical tip law."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.emitter import EmissionQuadrature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.calculation_cache import calculation_signatures


def emitter_with_plan(spatial=9, directions=9, energies=9):
    state = default_state()
    emitter = state.electron_gun.emitter
    emitter.quadrature = EmissionQuadrature(spatial, directions, energies)
    emitter.ray_count = emitter.quadrature.total
    return state, emitter


@pytest.mark.parametrize("axis", ["spatial", "directions", "energies"])
def test_single_factor_preserves_other_marginals_and_total_weight(axis):
    state, emitter = emitter_with_plan()
    old = emitter.emit()
    original = emitter.quadrature
    emitter.quadrature = replace(original, **{axis: getattr(original, axis)*2})
    emitter.ray_count = emitter.quadrature.total
    new = emitter.emit()
    assert new.weight.sum() == pytest.approx(1.)
    assert emitter.emitted_current_a == state.electron_gun.emitter.emitted_current_a
    if axis != "spatial":
        assert np.array_equal(np.unique(new.x_m), np.unique(old.x_m))
    if axis != "directions":
        assert np.array_equal(np.unique(new.tx_rad), np.unique(old.tx_rad))
    if axis != "energies":
        assert np.array_equal(np.unique(new.energy_offset_ev), np.unique(old.energy_offset_ev))
    assert np.all(np.abs(new.energy_offset_ev) <= emitter.energy_half_range_ev)


def test_sampling_identity_survives_snapshot_and_profile_and_invalidates_consumers():
    state, emitter = emitter_with_plan()
    old = calculation_signatures(state)
    emitter.quadrature = replace(emitter.quadrature, directions=18)
    emitter.ray_count = emitter.quadrature.total
    new = calculation_signatures(state)
    assert old["incident"] != new["incident"]
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun.emitter.quadrature == emitter.quadrature
    from temsim.optics.electron_gun.field_emission import _component_payload, _restore_component_settings
    record = _component_payload(emitter)
    other = default_state().electron_gun.emitter
    _restore_component_settings(other, record)
    assert other.quadrature == emitter.quadrature
    _restore_component_settings(other, _component_payload(default_state().electron_gun.emitter))
    assert other.quadrature is None


def test_no_silent_partial_tensor_when_requested_budget_does_not_match():
    _, emitter = emitter_with_plan()
    with pytest.raises(ValueError, match="quadrature"):
        emitter.emit(100)


def test_independent_comparison_executes_actual_tip_chain_and_records_only_selected_factor():
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.sampling_convergence import ConvergenceRequest, run_convergence
    state = default_state()
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 1.
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    report = run_convergence(ConvergenceRequest(point, "directions", 54, 3, 3, 3))
    before, after = report["runs"]
    assert before["planes"][0]["emitted_samples"] == 27
    assert after["planes"][0]["emitted_samples"] == 54
    assert before["sampling_rule"]["independent_product"]["directions"] == 3
    assert after["sampling_rule"]["independent_product"]["directions"] == 6
    assert len(after["input_changes"]) == 2  # Direction factor and its derived total count.
    assert before["planes"][0]["source_current_a"] == after["planes"][0]["source_current_a"]
    assert report["physical_validation"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("maximum_angle", [0., 30., 90.])
def test_surface_conditional_energy_law_and_full_cap_weight(maximum_angle):
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    state, emitter = emitter_with_plan(9, 128, 128)
    model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    # Preserve the exact exponential N/T law conditioned on the local cone.
    model = replace(model, emission=replace(model.emission,
        maximum_angle_deg=maximum_angle, angular_sampling="uniform_cdf",
        spatial_sampling="apex_stratified_v1", directions_per_position=1,
        angular_stratum_allocation=(), spatial_stratum_allocation=()))
    emitter.surface_model = model
    bundle = emitter.emit()
    assert bundle.weight.sum() == pytest.approx(1.)
    assert np.all(bundle.surface_energy_ev > 0)
    measured = np.sum(bundle.surface_energy_ev*bundle.weight)
    assert measured == pytest.approx(model.emission.mean_energy_ev, rel=.04)
    assert np.unique(bundle.surface_position_m[:, 0]).size == 9
    assert np.allclose(np.linalg.norm(bundle.surface_direction, axis=1), 1.)
    assert emitter.emitted_current_a == model.current_na*1e-9
