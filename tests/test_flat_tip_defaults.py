"""Flat-tip defaults, explicit alternatives and saved-state ownership."""
from copy import deepcopy
from dataclasses import replace

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_assembly import apply_tip_part, model_from_part
from temsim.optics.electron_gun.tip_edit import tip_model_label


def tip_part():
    return deepcopy(module_manifest.part_data("gun/FEG.toml", "feg_tip"))


def planar_emitter():
    return ColdFieldEmitter(0., 1., 1.)


def assert_flat(gun):
    assert gun.emitter.surface_model is None
    assert gun.emitter.coherence is None
    assert gun.source_representation == "classical_particles"
    assert tip_model_label(gun).startswith("Flat tip")


def test_emitter_gun_and_headless_state_defaults_are_flat():
    assert planar_emitter().surface_model is None
    gun = FieldEmissionGun()
    assert_flat(gun)
    assert_flat(default_state().electron_gun)
    # Default selection changes neither the scalar emission law nor the gun.
    assert gun.emitter.emission_current_na == planar_emitter().emission_current_na
    assert {"feg_tip", "feg_extractor", "feg_electrostatic_lens",
            "feg_accelerator", "feg_dpa_aperture", "feg_c1_aperture"} <= {
                item.key for item in gun.components}


@pytest.mark.parametrize("path", ["gun/FEG.toml", "gun/FEG_Mono.toml",
                                 "../sources/FEG_tip.toml"])
def test_every_shipped_tip_declares_flat_default_and_keeps_curved_recipe(path):
    part = module_manifest.part_data(path, "feg_tip")
    assert part["default_tip_emission"] == "flat_tip"
    assert model_from_part(part).geometry.apex_radius_nm == part["tip_radius_nm"]


@pytest.mark.parametrize("gun_name", ["FEG", "FEG + Mono"])
def test_install_resets_to_flat_but_geometry_refresh_preserves_user_choice(gun_name):
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), gun=gun_name)
    assembly = catalog.apply(state, selection)
    gun = state.electron_gun
    assert_flat(gun)
    planar_key = gun._cache_key(49)
    curved = model_from_part(assembly.part("feg_tip").data)
    gun.emitter.surface_model = curved
    assert gun._cache_key(49) != planar_key
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert gun.emitter.surface_model == curved
    catalog.apply(state, selection, preserve_operating_parameters=False)
    assert_flat(gun)


@pytest.mark.parametrize("curved", [False, True])
def test_saved_source_choice_is_not_replaced_by_new_defaults(curved):
    gun = FieldEmissionGun()
    if curved:
        gun.emitter.surface_model = model_from_part(tip_part())
    payload = gun.to_dict()
    restored = field_emission_gun_from_dict(payload)
    assert restored.emitter.surface_model == gun.emitter.surface_model
    assert restored.to_dict() == payload


def test_thermionic_installation_keeps_its_source_family():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun="Thermionic"))
    assert state.electron_gun.type_key == "thermionic"


def test_legacy_tip_definition_without_default_field_starts_flat():
    part = tip_part()
    part.pop("default_tip_emission")
    emitter = planar_emitter()
    apply_tip_part(emitter, part)
    assert emitter.surface_model is None
    # An explicit catalog default is still possible; it is not a new source.
    part["default_tip_emission"] = "curved_surface_particles"
    apply_tip_part(emitter, part, reset_source=True)
    assert emitter.surface_model == model_from_part(part)


@pytest.mark.parametrize("value", ["unknown", True, None, []])
def test_invalid_default_is_rejected(value):
    part = tip_part()
    part["default_tip_emission"] = value
    with pytest.raises(ValueError, match="Default tip emission"):
        model_from_part(part)


def test_source_dialog_exposes_flat_default_without_enabling_curved_or_waves(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    assert not dialog.surface_enabled.isChecked()
    assert "Historical" in dialog.surface_enabled.text()
    assert "Flat tip" in dialog.model_change_summary.text()
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    assert dialog.value()["surface_model"] is None
    assert dialog.value()["coherence"] is None
    assert_flat(gun)


def test_main_window_uses_same_default_and_does_not_reset_it_on_reload(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(
        str(tmp_path / "window.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    window = main_window.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    assert_flat(window.state.electron_gun)
    curved = model_from_part(window.state._resolved_assembly.part("feg_tip").data)
    window.state.electron_gun.emitter.surface_model = curved
    window.reload_toml_catalog()
    assert window.state.electron_gun.emitter.surface_model == curved
