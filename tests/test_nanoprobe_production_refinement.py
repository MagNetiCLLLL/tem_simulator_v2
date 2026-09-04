from types import SimpleNamespace

import numpy as np
import pytest

from temsim.operating_modes import direct_alignment_by_key
import temsim.optics.direct_alignment as alignment


def _state():
    return SimpleNamespace(lenses=[
        SimpleNamespace(key=key, max_percent=100.0)
        for key in alignment.CONDENSER_KEYS
    ])


def _measurement(value, waist_mm):
    return alignment.DirectAlignmentMeasurement(
        key="nanoprobe_convergence", value=value, unit="mrad",
        constraint_value=waist_mm, constraint_unit="mm",
    )


def test_acceptable_angular_quantile_keeps_c2_while_focusing_c3(monkeypatch):
    # At the 240 um pupil a 55.5866 mrad production quantile satisfies the
    # existing 60 mrad request tolerance. A jump between clipped rays makes
    # a C2 derivative unsuitable even though the C3 focus root is smooth.
    initial = np.asarray((30.0, 20.97119414232055))
    focus_c3 = 20.97145310038172
    calls = []

    def validate(_state, _definition, vector, _step):
        calls.append(np.asarray(vector).copy())
        assert vector[0] == initial[0], "An acceptable pupil must not receive a C2 Newton step"
        return _measurement(55.5866143809, (vector[1] - focus_c3) * 0.1)

    monkeypatch.setattr(alignment, "_validate_condenser_production", validate)
    refined, evaluations = alignment._refine_nanoprobe_production_focus(
        _state(), direct_alignment_by_key("nanoprobe_convergence"),
        60.0, initial, 0.1,
    )
    assert refined[0] == initial[0]
    assert refined[1] == pytest.approx(focus_c3, abs=1.0e-7)
    assert evaluations == len(calls)
    np.testing.assert_array_equal(initial, (30.0, 20.97119414232055))


def test_production_valid_seed_is_not_perturbed_toward_an_unreachable_target(monkeypatch):
    initial = np.asarray((30.0, 20.97145310038172))

    def validate(_state, _definition, vector, _step):
        np.testing.assert_array_equal(vector, initial)
        return _measurement(55.5866143809, 0.0)

    monkeypatch.setattr(alignment, "_validate_condenser_production", validate)
    refined, evaluations = alignment._refine_nanoprobe_production_focus(
        _state(), direct_alignment_by_key("nanoprobe_convergence"),
        60.0, initial, 0.1,
    )
    np.testing.assert_array_equal(refined, initial)
    assert evaluations == 1


def test_out_of_tolerance_angle_still_uses_coupled_production_correction(monkeypatch):
    initial = np.asarray((22.0, 21.0))
    focus_c3 = 21.003

    def validate(_state, _definition, vector, _step):
        return _measurement(
            30.0 * (1.0 + 0.1 * (vector[0] - 20.0)),
            (vector[1] - focus_c3) * 0.1,
        )

    monkeypatch.setattr(alignment, "_validate_condenser_production", validate)
    definition = direct_alignment_by_key("nanoprobe_convergence")
    refined, _evaluations = alignment._refine_nanoprobe_production_focus(
        _state(), definition, 30.0, initial, 0.1,
    )
    final = validate(None, None, refined, None)
    assert refined[0] != initial[0]
    assert abs(np.log(final.value / 30.0)) <= definition.targets["maximum_relative_error"]
    assert abs(final.constraint_value) <= definition.targets["maximum_waist_offset_mm"]
