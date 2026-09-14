"""Shared physical tip edits reach every linked assembly without new sources."""
from dataclasses import replace
from pathlib import Path

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.manifest_editor import ManifestEditor
from temsim.optics.column import default_state
from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.shared_tip import LINK, copy_catalog_tree

TIP = ("parts", "feg_tip")


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return root


def shared_path(root):
    return root.parent / "sources/FEG_tip.toml"


def change_radius(root, radius):
    draft = PartModelDocument(shared_path(root))
    draft.set_dimension(TIP + ("tip_radius_nm",), radius)
    draft.save()


@pytest.mark.parametrize("filename", ["FEG.toml", "FEG_Mono.toml", "../../sources/FEG_tip.toml"])
def test_save_any_linked_tip_updates_both_runtime_variants(root, filename):
    catalog = AssemblyCatalog(root)
    states = [default_state(), default_state()]
    selections = [replace(catalog.default_selection(), gun=gun) for gun in ("FEG", "FEG + Mono")]
    for state, selection in zip(states, selections):
        catalog.apply(state, selection, preserve_operating_parameters=True)
    previous = [state.electron_gun._cache_key(9) for state in states]
    draft = PartModelDocument(root / "gun" / filename)
    draft.set_dimension(TIP + ("tip_radius_nm",), 125.)
    draft.set_dimension(TIP + ("emission_maximum_angle_deg",), 35.)
    draft.save()
    assert not draft.dirty
    for state, selection, key in zip(states, selections, previous):
        assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
        gun = state.electron_gun
        assert assembly.part("feg_tip").data["tip_radius_nm"] == 125.
        assert gun.emitter.surface_model.geometry.apex_radius_nm == 125.
        assert gun.emitter.surface_model.emission.maximum_angle_deg == 35.
        assert gun.emitter.surface_model.coherence is None
        assert gun._cache_key(9) != key
        assert gun.extractor.voltage_kv == 4.
        assert gun.accelerator.high_tension_kv == 300.


def test_embedded_snapshot_is_not_authority_and_unrelated_save_preserves_shared_edit(root):
    path = root / "gun/FEG.toml"
    original = path.read_bytes()
    change_radius(root, 145.)
    assert path.read_bytes() == original
    draft = PartModelDocument(path)
    assert draft.part("feg_tip")["tip_radius_nm"] == 145.
    draft.part("feg_extractor")["default_voltage_kv"] = 5.
    draft.save()
    assert module_manifest.part_data(shared_path(root), "feg_tip")["tip_radius_nm"] == 145.
    assert module_manifest.part_data(root / "gun/FEG_Mono.toml", "feg_extractor")["default_voltage_kv"] == 4.
    assert module_manifest.part_data(path, "feg_extractor")["default_voltage_kv"] == 5.


def test_stale_linked_draft_cannot_overwrite_new_shared_definition(root):
    draft = PartModelDocument(root / "gun/FEG_Mono.toml")
    draft.set_dimension(TIP + ("tip_radius_nm",), 120.)
    change_radius(root, 140.)
    original = draft.path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        draft.save()
    assert draft.path.read_bytes() == original
    assert module_manifest.part_data(shared_path(root), "feg_tip")["tip_radius_nm"] == 140.


def test_validation_failure_changes_neither_module_nor_shared_file(root, monkeypatch):
    draft = PartModelDocument(root / "gun/FEG.toml")
    draft.set_dimension(TIP + ("tip_radius_nm",), 150.)
    originals = {path: path.read_bytes() for path in (draft.path, shared_path(root))}
    def fail(_self):
        raise ValueError("Rejected by another compatible assembly")
    monkeypatch.setattr(ManifestEditor, "validate_catalog", fail)
    with pytest.raises(ValueError, match="compatible assembly"):
        draft.save()
    assert all(path.read_bytes() == value for path, value in originals.items())


def test_failed_second_file_write_rolls_back_first(root, monkeypatch):
    draft = PartModelDocument(root / "gun/FEG.toml")
    draft.set_dimension(TIP + ("tip_radius_nm",), 150.)
    originals = {path: path.read_bytes() for path in (draft.path, shared_path(root))}
    atomic_write = module_manifest._atomic_write_text
    def fail(path, text):
        if Path(path).resolve() == shared_path(root):
            raise OSError("Shared file is locked")
        return atomic_write(path, text)
    monkeypatch.setattr(module_manifest, "_atomic_write_text", fail)
    with pytest.raises(OSError, match="locked"):
        draft.save()
    assert all(path.read_bytes() == value for path, value in originals.items())


def test_missing_definition_does_not_fall_back_to_embedded_tip(root):
    shared_path(root).unlink()
    with pytest.raises(ValueError, match="unavailable"):
        module_manifest.read_document(root / "gun/FEG.toml")


def test_absolute_link_cannot_escape_isolated_catalog_validation(root):
    from temsim.shared_tip import definition_path
    with pytest.raises(ValueError, match="relative to the assembly TOML"):
        definition_path(root / "gun/FEG.toml", {"key": "feg_tip", LINK: str(shared_path(root))})


def test_copy_and_save_copy_detach_from_shared_tip(root, tmp_path):
    change_radius(root, 135.)
    source = PartModelDocument(root / "gun/FEG.toml")
    target = PartModelDocument(root / "gun/FEG_Mono.toml")
    target.copy_component_from(source.document, "feg_tip", "spare_tip", 400.)
    spare = target.part("spare_tip")
    assert LINK not in spare and spare["mechanical_only"]
    assert spare["tip_radius_nm"] == 135.
    copy = tmp_path / "export.toml"
    source.save_copy(copy)
    assert LINK not in PartModelDocument(copy).part("feg_tip")
    change_radius(root, 160.)
    assert PartModelDocument(copy).part("feg_tip")["tip_radius_nm"] == 135.
    assert spare["tip_radius_nm"] == 135.


def test_save_copy_cannot_replace_shared_authority(root):
    draft = PartModelDocument(root / "gun/FEG.toml")
    with pytest.raises(ValueError, match="shared tip definition"):
        draft.save_copy(shared_path(root))


def test_archived_inputs_include_the_consumed_tip_and_have_their_own_copy(root, tmp_path):
    from temsim.shared_tip import copy_catalog_inputs
    from temsim.calculation_manifest import _external_inputs
    change_radius(root, 135.)
    catalog = AssemblyCatalog(root)
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    tip_identity = next(row for row in _external_inputs(state) if row.role == "assembly:tip_definition")
    assert Path(tip_identity.path) == shared_path(root) and tip_identity.available
    archive = tmp_path / "archive/instrument_inputs"
    copy_catalog_inputs(root, archive, ("catalog.toml", *catalog.selected_paths(selection).values()))
    copy_catalog_tree(archive, tmp_path / "relocated/instrument_inputs")
    change_radius(root, 165.)
    assert module_manifest.part_data(archive / "gun/FEG.toml", "feg_tip")["tip_radius_nm"] == 135.
    assert module_manifest.part_data(tmp_path / "relocated/instrument_inputs/gun/FEG.toml", "feg_tip")["tip_radius_nm"] == 135.
    changed_identity = next(row for row in _external_inputs(state) if row.role == "assembly:tip_definition")
    assert changed_identity.sha256 != tip_identity.sha256


@pytest.mark.parametrize("historical_planar", [False, True])
def test_saved_source_is_not_silently_converted_when_shared_defaults_change(root, monkeypatch, historical_planar):
    monkeypatch.setattr(module_manifest, "MODULE_ROOT", root)
    gun = FieldEmissionGun()
    if historical_planar:
        gun.emitter.surface_model = None
    saved = gun.to_dict()
    change_radius(root, 170.)
    restored = field_emission_gun_from_dict(saved)
    if historical_planar:
        assert restored.emitter.surface_model is None
    else:
        assert restored.emitter.surface_model.geometry.apex_radius_nm == 100.
    assert "tip_assembly_defaults" not in saved["components"]["feg_tip"]


@pytest.fixture
def window(qtbot, root, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window
    monkeypatch.setattr(module_manifest, "MODULE_ROOT", root)
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(
        str(tmp_path / "window.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    return widget


@pytest.mark.parametrize("relative", ["gun/FEG.toml", "gun/FEG_Mono.toml", "../sources/FEG_tip.toml"])
def test_editor_save_refreshes_active_gun_even_from_inactive_variant(window, root, relative):
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(root / relative, selected_key="feg_tip")
    assert "shared" in page.source_label.text()
    page.session.set_dimension(TIP + ("tip_radius_nm",), 175.)
    assert page.save()
    assert not page.session.dirty
    assert window.state.electron_gun.emitter.surface_model.geometry.apex_radius_nm == 175.
    assert module_manifest.part_data(root / "gun/FEG_Mono.toml", "feg_tip")["tip_radius_nm"] == 175.


def test_external_shared_edit_and_reload_refreshes_clean_model_and_source_dialog(window, root, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(root / "gun/FEG.toml", selected_key="feg_tip")
    change_radius(root, 180.)
    window.reload_toml_catalog()
    assert page.session.part("feg_tip")["tip_radius_nm"] == 180.
    assert window.state.electron_gun.emitter.surface_model.geometry.apex_radius_nm == 180.
    dialog = GunSourceDialog(window.state.electron_gun)
    qtbot.addWidget(dialog)
    assert dialog._assembly_tip_path == shared_path(root)
    assert dialog._surface_draft.geometry.apex_radius_nm == 180.


def test_legacy_runtime_choice_survives_explicit_shared_reload(window, root):
    window.state.electron_gun.emitter.surface_model = None
    change_radius(root, 190.)
    window.reload_toml_catalog()
    assert window.state.electron_gun.emitter.surface_model is None


def test_tip_reload_applies_only_changed_electrode_defaults(window, root):
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(root / "gun/FEG.toml", selected_key="feg_tip")
    page.session.set_dimension(TIP + ("tip_radius_nm",), 125.)
    page.session.part("feg_extractor")["default_voltage_kv"] = 5.
    assert page.save()
    assert window.state.electron_gun.extractor.voltage_kv == 5.
    assert module_manifest.part_data(root / "gun/FEG_Mono.toml", "feg_extractor")["default_voltage_kv"] == 4.
    window.state.electron_gun.extractor.voltage_kv = 6.
    page.session.set_dimension(TIP + ("tip_radius_nm",), 130.)
    assert page.save()
    assert window.state.electron_gun.extractor.voltage_kv == 6.
