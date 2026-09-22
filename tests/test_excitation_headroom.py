import pytest

from temsim.optics.column import default_state
from temsim.optics.excitation_policy import rebase_peak_field
from temsim.optics.lens_focal_length import focal_length_mm, set_focal_length


def test_rebase_peak_field_preserves_the_physical_operating_field():
    maximum_t, percent = rebase_peak_field(0.726, 100.0, 70.0)

    assert maximum_t == pytest.approx(1.0371428571428571)
    assert percent == pytest.approx(70.0)
    assert maximum_t * percent / 100.0 == pytest.approx(0.726)


def test_focal_length_adjustment_rebases_a_saturated_round_lens_to_sixty_percent():
    state = default_state()
    lens = state.condenser_lens_1.lens
    lens.percent = 100.0
    target_focal_mm = focal_length_mm(lens, state.beam_voltage_kv)
    previous_maximum_t = lens.b0_t
    lens.percent = 50.0

    changed = set_focal_length(
        lens, state.beam_voltage_kv, target_focal_mm
    )

    assert changed == "maximum field"
    assert lens.percent == pytest.approx(60.0)
    assert lens.b0_t == pytest.approx(previous_maximum_t / 0.6)
    assert focal_length_mm(
        lens, state.beam_voltage_kv
    ) == pytest.approx(target_focal_mm)


def test_focal_length_adjustment_rebases_a_saturated_objective_to_sixty_percent():
    state = default_state()
    objective = state.objective_lens
    objective.percent = 100.0
    target_focal_mm = objective.focal_length_for_voltage_mm(
        state.beam_voltage_kv
    )
    previous_maximum_t = objective.b0_t
    objective.percent = 50.0

    changed = objective.set_focal_length_for_voltage_mm(
        state.beam_voltage_kv, target_focal_mm
    )

    assert changed == "maximum field"
    assert objective.percent == pytest.approx(60.0)
    assert objective.b0_t == pytest.approx(previous_maximum_t / 0.6)
    assert objective.focal_length_for_voltage_mm(
        state.beam_voltage_kv
    ) == pytest.approx(target_focal_mm)


def test_old_excitation_schema_is_rejected_without_rescaling_fields():
    from copy import deepcopy
    payload = default_state().to_dict()
    payload["schema_version"] = 62
    before = deepcopy(payload)
    with pytest.raises(ValueError, match="schema"):
        type(default_state()).from_dict(payload)
    assert payload == before
