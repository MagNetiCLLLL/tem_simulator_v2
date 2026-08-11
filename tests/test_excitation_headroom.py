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


def test_v62_state_migration_preserves_fields_while_adding_headroom():
    state = default_state()
    payload = state.to_dict()
    payload["schema_version"] = 62
    factors = {
        "condenser_lens_2": 0.7,
        "probe_tl22_lens": 0.6,
        "probe_tl21_lens": 0.6,
        "probe_tl12_lens": 0.6,
        "image_ol_post_lens": 0.6,
        "image_tl11_lens": 0.6,
        "image_tl12_lens": 0.6,
        "image_tl21_lens": 0.6,
        "image_tl22_lens": 0.6,
        "image_adapter_lens": 0.6,
    }
    current = {lens.key: lens for lens in state.lenses}
    expected_fields = {
        key: current[key].b0_t * current[key].percent / 100.0
        for key in factors
    }
    expected_fields["objective_lens"] = (
        current["objective_lens"].b0_t
        * current["objective_lens"].percent
        / 100.0
    )

    for row in payload["lenses"]:
        key = row["key"]
        if key in factors:
            factor = factors[key]
            row["b0_t"] *= factor
            row["percent"] /= factor
        elif key == "objective_lens":
            row["percent"] /= 0.7

    restored = type(state).from_dict(payload)
    restored_by_key = {lens.key: lens for lens in restored.lenses}

    assert restored.schema_version == default_state().schema_version
    for key, expected_field_t in expected_fields.items():
        lens = restored_by_key[key]
        assert lens.b0_t * lens.percent / 100.0 == pytest.approx(
            expected_field_t
        )
