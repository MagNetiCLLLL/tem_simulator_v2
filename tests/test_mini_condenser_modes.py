"""CM mode control preserves finite excitation, including persisted settings."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import (
    _apply_values, apply_operating_mode_pair,
    load_mini_condenser_default_excitation_percent, mode_by_key,
)
from temsim.optics.column import default_state
from temsim.optics.mini_condenser import create_mini_condenser
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def test_startup_magnitude_has_one_authoritative_configuration():
    configured = load_mini_condenser_default_excitation_percent()
    assert create_mini_condenser().percent == configured > 0.0
    for key in ("micro_probe", "nano_probe"):
        mode = mode_by_key(key)
        assert "percent" not in mode.devices["mini_condenser"]
        assert mode.calibration_status == "retained_not_recomputed_after_signed_cm_mode_control"


@pytest.mark.parametrize("magnitude", [10.0, 37.25, 90.0])
def test_mode_round_trip_reverses_finite_field_without_changing_magnitude(magnitude):
    mini = create_mini_condenser()
    mini.signed_excitation_percent = -magnitude
    field_planes = mini.z_mm + np.array([-8.0, 0.0, 8.0])
    original_field = mini.magnetic_field_t(field_planes)
    original_b0 = mini.b0_t
    assert np.all(np.abs(original_field) > 0.0)
    mini.set_probe_mode("micro_probe")
    assert mini.signed_excitation_percent == magnitude
    assert mini.enabled
    np.testing.assert_allclose(mini.magnetic_field_t(field_planes), -original_field,
                               rtol=0.0, atol=0.0)
    mini.set_probe_mode("nano_probe")
    assert mini.signed_excitation_percent == -magnitude
    assert mini.percent == magnitude
    assert mini.b0_t == original_b0
    np.testing.assert_array_equal(mini.magnetic_field_t(field_planes), original_field)


def test_public_mode_application_preserves_live_cm_magnitude_and_other_presets():
    state = default_state()
    state.mini_condenser.signed_excitation_percent = -53.125
    for key, sign in (("micro_probe", 1), ("nano_probe", -1), ("micro_probe", 1)):
        applied = apply_operating_mode_pair(state, key, "diffraction")
        assert state.mini_condenser.signed_excitation_percent == sign * 53.125
        assert state.mini_condenser.enabled
        assert "mini_condenser" in applied.changed_devices
        assert state.objective_lens.percent == mode_by_key(key).devices["objective_lens"]["percent"]
        c2 = next(lens for lens in state.lenses if lens.key == "condenser_lens_2")
        assert c2.percent == mode_by_key(key).devices["condenser_lens_2"]["percent"]


@pytest.mark.parametrize("key,historical_magnitude,sign", [
    ("micro_probe", 35.0, 1), ("nano_probe", 41.5570797718, -1),
])
def test_historical_mode_definition_cannot_restore_old_cm_drive(key, historical_magnitude, sign):
    state = default_state()
    state.mini_condenser.signed_excitation_percent = -62.75
    mode = mode_by_key(key)
    mode = replace(mode, devices={**mode.devices, "mini_condenser": {
        "percent": historical_magnitude, "field_polarity": -sign, "enabled": False,
    }})
    _apply_values(state, mode)
    assert state.mini_condenser.signed_excitation_percent == sign * 62.75
    assert state.mini_condenser.enabled


@pytest.mark.parametrize("column", ["C2", "C3", "C3 + Probe Corrector",
    "C3 + Image Corrector", "C3 + Probe Corrector + Image Corrector"])
def test_assembly_seed_uses_same_signed_cm_control(column):
    from temsim.optics.assembly_illumination import seed_illumination
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "No Energy Filter"))
    state.mini_condenser.signed_excitation_percent = -48.125
    for mode, sign in (("micro_probe", 1), ("nano_probe", -1)):
        seed_illumination(state, mode)
        assert state.mini_condenser.signed_excitation_percent == sign * 48.125


@pytest.mark.parametrize("signed", [-67.125, 67.125])
def test_signed_cm_survives_profile_and_snapshot_round_trip(tmp_path, signed):
    state = default_state()
    state.mini_condenser.signed_excitation_percent = signed
    path = tmp_path / "cm-profile.toml"
    selection = AssemblyCatalog().default_selection()
    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    restored = default_state()
    apply_profile_values(restored, values)
    snapshot = capture_instrument_snapshot(state).restore()
    assert loaded_selection == selection
    for candidate in (restored, snapshot):
        mini = candidate.mini_condenser
        assert mini.signed_excitation_percent == signed
        assert mini.enabled
        assert abs(mini.magnetic_field_t([mini.z_mm])[0]) > 0.0
        mini.set_probe_mode("nano_probe" if signed > 0.0 else "micro_probe")
        assert mini.signed_excitation_percent == -signed


@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf"), -101.0, True])
def test_invalid_signed_drive_is_rejected_without_partial_mutation(invalid):
    mini = create_mini_condenser()
    before = (mini.percent, mini.polarity, mini.enabled)
    with pytest.raises(ValueError, match="Mini Condenser"):
        mini.signed_excitation_percent = invalid
    assert (mini.percent, mini.polarity, mini.enabled) == before


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_or_disabled_cm_is_not_silently_treated_as_nanoprobe(disabled):
    state = default_state()
    if disabled:
        state.mini_condenser.enabled = False
    else:
        state.mini_condenser.percent = 0.0
    before = state.to_dict()
    with pytest.raises(ValueError, match="nonzero"):
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert state.to_dict() == before
