"""Editable installed tip and classical launch contracts; no wave calculations."""
from copy import deepcopy
from dataclasses import replace
import shutil

import numpy as np
import pytest

from temsim import module_manifest
from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_assembly import model_from_part, validate_tip_part
from temsim.optics.electron_gun.tip_surface import emit_surface, TipSurfaceModel
from temsim.part_model_document import PartModelDocument
from temsim.paths import CONFIG_ROOT, INSTRUMENT_CONFIG_ROOT


def test_shared_definition_and_installed_tips_resolve_the_same_model():
    template = PartModelDocument(CONFIG_ROOT / "sources/FEG_tip.toml")
    template.validate()
    expected = model_from_part(template.part("feg_tip"))
    for name in ("FEG.toml", "FEG_Mono.toml"):
        part = module_manifest.part_data("gun/" + name, "feg_tip")
        validate_tip_part(part)
        assert model_from_part(part) == expected
    gun = FieldEmissionGun()
    assert gun.emitter.surface_model == expected
    assert expected.coherence is None
    assert gun.source_representation == "classical_particles"
    assert gun.emitted_current_a == pytest.approx(100e-9)
    assert gun.extractor.voltage_kv == 4
    assert gun.accelerator.high_tension_kv == 300


def test_tip_dimensions_save_reload_and_surface_selection(tmp_path):
    from temsim.part_model_3d import part_model_from_document
    path = tmp_path / "tip.toml"
    shutil.copyfile(CONFIG_ROOT / "sources/FEG_tip.toml", path)
    draft = PartModelDocument(path)
    draft.set_dimension(("parts", "feg_tip", "tip_radius_nm"), 150)
    draft.set_dimension(("parts", "feg_tip", "length_mm"), .8)
    tip = draft.part("feg_tip")
    assert (tip["local_start_z_mm"], tip["local_center_z_mm"], tip["local_end_z_mm"]) == (-.8, -.4, 0)
    mesh = part_model_from_document(draft.document, "feg_tip").meshes[0]
    assert mesh.vertices[:, 2].min() == pytest.approx(-.8)
    assert mesh.vertices[:, 2].max() == 0
    assert 2 * mesh.vertices[:, 0].max() == pytest.approx(tip["outer_diameter_mm"])
    paths = {path for metadata in mesh.surfaces.values() for path in metadata["parameter_paths"]}
    assert ("parts", "feg_tip", "tip_radius_nm") in paths
    assert ("parts", "feg_tip", "tip_cone_half_angle_deg") in paths
    draft.save()
    loaded = PartModelDocument(path)
    assert model_from_part(loaded.part("feg_tip")).geometry.apex_radius_nm == 150
    before = deepcopy(loaded.document)
    with pytest.raises(ValueError, match="must follow"):
        loaded.set_dimension(("parts", "feg_tip", "outer_diameter_mm"), 10)
    assert loaded.document == before


def test_density_scales_with_real_curved_surface_area():
    model = FieldEmissionGun().emitter.surface_model
    twice_radius = replace(model, geometry=replace(model.geometry, apex_radius_nm=200)).validate()
    assert twice_radius.current_na == pytest.approx(4 * model.current_na)
    assert TipSurfaceModel.from_dict(twice_radius.to_dict()) == twice_radius
    assert "current_na" not in model.to_dict()["emission"]
    with pytest.raises(ValueError, match="not both"):
        replace(model, emission=replace(model.emission, current_na=100)).validate()


def test_tip_copy_is_independent_positioned_solid_without_second_emitter():
    from temsim.part_model_3d import part_model_from_document
    source = PartModelDocument(CONFIG_ROOT / "sources/FEG_tip.toml")
    target = PartModelDocument(INSTRUMENT_CONFIG_ROOT / "gun/FEG.toml")
    target.copy_component_from(source.document, "feg_tip", "spare_tip", 50.)
    part = target.part("spare_tip")
    assert part["mechanical_only"] and part["axial_vacuum_context_only"]
    assert part["local_center_z_mm"] == 50
    target.set_dimension(("parts", "spare_tip", "tip_radius_nm"), 200)
    assert source.part("feg_tip")["tip_radius_nm"] == 100
    mesh = part_model_from_document(target.document, "spare_tip").meshes[0]
    assert mesh.vertices[:, 2].max() == pytest.approx(50.5)
    assert {component.key for component in FieldEmissionGun().components}.isdisjoint({"spare_tip"})


@pytest.mark.parametrize("key,field,value", [
    ("feg_extractor", "default_voltage_kv", -1.),
    ("feg_accelerator", "default_high_tension_kv", 400.),
    ("feg_electrostatic_lens", "default_voltage_kv", float("nan")),
])
def test_invalid_voltage_defaults_rejected_before_saving(key, field, value):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / "gun/FEG.toml")
    draft.part(key)[field] = value
    with pytest.raises(ValueError):
        draft.validate()


@pytest.mark.parametrize("field,value", [("length_mm", True), ("length_mm", "1"),
    ("tip_cone_half_angle_deg", True), ("emission_cap_half_angle_deg", True),
    ("emission_flux_electrons_per_nm2_s", float("nan"))])
def test_tip_toml_does_not_coerce_invalid_input_types(field, value):
    tip = module_manifest.part_data("gun/FEG.toml", "feg_tip")
    tip[field] = value
    with pytest.raises(ValueError, match="finite number"):
        model_from_part(tip)


@pytest.mark.parametrize("law", ["normal_tangential_exponential", "gamma", "monoenergetic"])
@pytest.mark.parametrize("angle", [0., 20., 90.])
def test_emission_energy_and_local_angular_limits(law, angle):
    model = FieldEmissionGun().emitter.surface_model
    model = replace(model, emission=replace(model.emission, energy_distribution=law,
        maximum_angle_deg=angle, kinetic_mean_ev=1.0, kinetic_sigma_ev=.2))
    position, direction, energy, weight = emit_surface(model, 32768)
    radius = model.geometry.apex_radius_nm * 1e-9
    normal = (position + np.array([0, 0, radius])) / radius
    cosine = np.sum(normal * direction, axis=1)
    assert np.all(cosine >= np.cos(np.deg2rad(angle)) - 1e-14)
    assert np.isfinite(direction).all() and np.isfinite(energy).all()
    assert np.all(energy > 0)
    assert weight.sum() == pytest.approx(1)
    assert np.mean(energy) == pytest.approx(model.emission.mean_energy_ev, rel=.003)
    assert np.std(energy) == pytest.approx(model.emission.energy_sigma_ev, rel=.008, abs=1e-14)


@pytest.mark.parametrize("gun_name,module", [("FEG", "FEG.toml"), ("FEG + Mono", "FEG_Mono.toml")])
def test_installed_toml_save_reaches_runtime_and_invalidates_consumed_inputs(tmp_path, gun_name, module):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    root = tmp_path / "instruments"
    from temsim.shared_tip import copy_catalog_tree
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    state = default_state()
    catalog = AssemblyCatalog(root)
    selection = replace(catalog.default_selection(), gun=gun_name)
    catalog.apply(state, selection, preserve_operating_parameters=True)
    gun = state.electron_gun
    previous = gun._cache_key(9)
    field_before = field_request(gun)
    editor = ManifestEditor(root)
    editor.save(ManifestTarget("gun/" + module, "feg_tip"), {
        ("parts", "feg_tip", "tip_radius_nm"): 125.,
        ("parts", "feg_tip", "emission_flux_electrons_per_nm2_s"): 1e8,
        ("parts", "feg_extractor", "default_voltage_kv"): 5.,
        ("parts", "feg_accelerator", "default_high_tension_kv"): 200.,
        ("parts", "feg_accelerator", "electrode_thickness_mm"): 1.5,
    }, layout_configuration_from_state(state))
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert gun.emitter.surface_model.geometry.apex_radius_nm == 125
    assert gun.emitter.surface_model.emission.flux_electrons_per_nm2_s == 1e8
    assert gun.extractor.voltage_kv == 5
    assert gun.accelerator.high_tension_kv == 200
    assert gun._cache_key(9) != previous
    assert field_request(gun) != field_before
    ring = next(row for row in field_request(gun)["rings"] if row[0] == "accelerator:0")
    assert ring[2] - ring[1] == pytest.approx(.0015)
    gun.extractor.voltage_kv = 6
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert gun.extractor.voltage_kv == 6  # unchanged default preserves operating edit


def test_historical_source_without_surface_field_is_not_converted():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = None
    payload = gun.to_dict()
    payload["components"]["feg_tip"].pop("surface_model", None)
    restored = field_emission_gun_from_dict(payload)
    assert restored.emitter.surface_model is None
    assert restored.to_dict()["integrator"]["method"] == "boris"


def test_editor_density_and_energy_law_apply_only_valid_draft(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    original = gun.emitter.surface_model
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    assert dialog.flux_density_enabled.isChecked()
    assert "total current 100 nA" in dialog.surface_derived.text()
    dialog.energy_law.setCurrentIndex(dialog.energy_law.findData("monoenergetic"))
    dialog.surface_inputs["kinetic_mean_ev"].setText("0.5")
    dialog.surface_inputs["maximum_angle_deg"].setText("25")
    dialog.geometry_inputs["apex_radius_nm"].setText("200")
    dialog.accept()
    value = dialog.value()["surface_model"]
    assert value.current_na == pytest.approx(400)
    assert value.emission.maximum_angle_deg == 25
    assert value.emission.mean_energy_ev == .5
    assert gun.emitter.surface_model is original
    assert not dialog.near_field_button.isEnabled()
