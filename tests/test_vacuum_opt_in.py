"""Vacuum transport is opt-in; saved choices and broad invalidation survive."""
from copy import deepcopy

import pytest

from temsim.optics.column import default_state
from temsim.vacuum import VacuumMap, bind_gun_environment, resolve_regions


@pytest.fixture
def state():
    return default_state()


def test_new_models_and_missing_switch_default_to_disabled():
    assert not VacuumMap().enabled
    assert not VacuumMap.from_dict({}).enabled
    assert not VacuumMap.load().enabled


def test_new_instrument_keeps_editable_pressures_without_transport(state):
    assert not state.vacuum_map.enabled
    assert not resolve_regions(state)
    assert len(resolve_regions(state, include_disabled=True)) == 6
    assert state.vacuum_map.regions[0].medium.pressure_mbar == 3e-11
    assert bind_gun_environment(state) == ()


def test_standalone_gun_does_not_silently_enable_vacuum():
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.vacuum import ensure_standalone_gun_environment

    gun = FieldEmissionGun()
    ensure_standalone_gun_environment(gun)
    assert gun._vacuum_regions == ()


@pytest.mark.parametrize("enabled", [False, True])
def test_saved_map_profile_and_snapshot_preserve_explicit_choice(state, tmp_path, enabled):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.profile_io import save_profile, read_profile, apply_profile_values

    state.vacuum_map.enabled = enabled
    state.vacuum_map.regions[-1].medium.pressure_mbar = 2e-7
    expected = state.vacuum_map.to_dict()
    path = tmp_path / "vacuum.toml"
    state.vacuum_map.save(path)
    assert VacuumMap.load(path).to_dict() == expected
    assert capture_instrument_snapshot(state).restore().vacuum_map.to_dict() == expected
    profile = tmp_path / "operating.toml"
    save_profile(profile, state, AssemblyCatalog().default_selection())
    _, values = read_profile(profile)
    restored = default_state()
    restored.vacuum_map.enabled = not enabled
    apply_profile_values(restored, values)
    assert restored.vacuum_map.to_dict() == expected


def test_setup_choice_before_preview_is_shared_by_all_qualities(state, qtbot):
    from temsim.gui.calculate_setup import CalculateSetupDialog
    from temsim.gui.calculation_controller import CalculationController
    from temsim.gui.vacuum_map_page import VacuumMapPage

    dialog = CalculateSetupDialog(state)
    page = VacuumMapPage()
    qtbot.addWidget(dialog)
    qtbot.addWidget(page)
    page.set_state(state)
    assert not dialog.controls["vacuum"][0].isChecked()
    assert not page.enabled.isChecked()
    assert len(page.diagram.rows) == 6
    original = deepcopy(state.to_dict())
    for enabled in (True, False):
        dialog.controls["vacuum"][0].setChecked(enabled)
        assert dialog.apply()
        page.set_state(state)
        assert page.enabled.isChecked() is enabled
        for quality in ("Preview", "Medium", "High accuracy"):
            snapshot = CalculationController._calculation_snapshot(state, quality, 49, .1)
            assert snapshot.vacuum_map.enabled is enabled
    assert state.to_dict() == original


def test_vacuum_toggle_and_active_map_edit_invalidate_every_stage(state):
    from temsim.calculation_cache import calculation_signatures

    off = calculation_signatures(state)
    state.vacuum_map.enabled = True
    on = calculation_signatures(state)
    assert off.keys() == on.keys()
    assert all(off[key] != on[key] for key in off)
    # Deliberately broad: changing only the downstream chamber may invalidate
    # upstream products too. The user explicitly permits this cache policy.
    state.vacuum_map.regions[-1].medium.pressure_mbar *= 2
    edited = calculation_signatures(state)
    assert all(on[key] != edited[key] for key in on)
