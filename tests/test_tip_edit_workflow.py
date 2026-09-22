"""Tip geometry, cache identity and the single physical editing workflow."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
from temsim import module_manifest
from temsim.optics.electron_gun.tip_assembly import model_from_part


def curved_model():
    return model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))


@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("component,attribute", [
    ("extractor", "mechanical_clear_bore_diameter_mm"),
    ("dpa_aperture", "mechanical_center_from_tip_mm"),
    ("deflector", "upper_center_from_tip_mm"),
    ("stigmator", "effective_length_mm"),
])
def test_consumed_geometry_invalidates_each_tip_model(curved, component, attribute):
    gun = FieldEmissionGun()
    if curved:
        gun.emitter.surface_model = curved_model()
    before = gun._cache_key(49)
    target = getattr(gun, component)
    old = getattr(target, attribute)
    setattr(target, attribute, old + .1)
    assert gun._cache_key(49) != before
    setattr(target, attribute, old)
    assert gun._cache_key(49) == before


def test_changed_bore_reexecutes_actual_aperture_loss():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = None
    first = gun.trace_to_exit(49)
    assert first.exit_bundle.alive.all()
    gun.extractor.mechanical_clear_bore_diameter_mm = 1e-6
    after = gun.trace_to_exit(49)
    assert after is not first
    assert not after.exit_bundle.alive.any()
    assert set(after.blocked_key) == {gun.extractor.key}
    assert gun.trace_to_exit(49) is after


def test_invalid_model_switch_is_atomic_and_does_not_reinterpret_voltage(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    gun.emitter.surface_model = curved_model()
    gun.electrostatic_lens.voltage_reference = "tip"
    before = gun.to_dict()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(False)
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert "voltage reference" in dialog.error.text()
    assert gun.to_dict() == before
    assert dialog._value is None


def test_valid_switch_changes_model_identity_but_keeps_optics():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = curved_model()
    before = gun.to_dict()
    off = candidate_tip_edit(gun, {"surface_model": None, "coherence": None})
    assert off.emitter.surface_model is None
    assert gun.emitter.surface_model is not None
    assert off._cache_key(49) != gun._cache_key(49)
    for key, settings in before["components"].items():
        if key != "feg_tip":
            assert off.to_dict()["components"][key] == settings
    on = candidate_tip_edit(off, {"surface_model": gun.emitter.surface_model})
    assert on._cache_key(49) == gun._cache_key(49)


def test_emitting_mesh_boundary_matches_the_actual_particle_cap():
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument
    from temsim.paths import CONFIG_ROOT
    from temsim.optics.electron_gun.tip_surface import emit_surface
    document = PartModelDocument(CONFIG_ROOT / "sources/FEG_tip.toml")
    g = FieldEmissionGun()
    g.emitter.surface_model = curved_model()
    for degrees in (2., 10., 25.):
        model = replace(g.emitter.surface_model, emission=replace(g.emitter.surface_model.emission, cap_half_angle_deg=degrees))
        mesh = part_model_from_document(document.document, "feg_tip", runtime_values={
            "feg_tip": {"tip_surface_model": model.to_dict()}})
        patch = next(m for m in mesh.meshes if "emitting_cap" in m.surfaces)
        cap_vertices = patch.vertices[np.unique(patch.faces[patch.face_groups == "emitting_cap"])]
        positions, _, _, weights = emit_surface(model, 4096)
        radius = model.geometry.apex_radius_nm*1e-6
        depth = 2*radius*np.sin(np.radians(degrees)/2)**2
        assert cap_vertices[:, 2].min() == pytest.approx(-depth)
        assert cap_vertices[:, 2].max() == 0
        assert np.min(positions[:, 2]*1000) >= -depth
        assert np.all(positions[:, 2] <= 0)
        assert weights.sum() == pytest.approx(1)
        assert ("parts", "feg_tip", "emission_cap_half_angle_deg") in patch.surfaces["emitting_cap"]["parameter_paths"]
    off = part_model_from_document(document.document, "feg_tip", runtime_values={"feg_tip": {"tip_surface_model": None}})
    assert all("emitting_cap" not in m.surfaces for m in off.meshes)


def test_input_design_is_reusable_but_computed_history_remains_read_only():
    from temsim.optics.column import default_state
    from temsim.optics.gun_matching import input_working_point
    point = input_working_point(default_state(), label="Input design")
    changed = replace(point, snapshot=replace(point.snapshot, implementation="older-solver"))
    restored = changed.compatible_state()
    assert restored.electron_gun.emitter.surface_model == point.compatible_state().electron_gun.emitter.surface_model
    assert restored.electron_gun._trace_cache is None
    assert changed.observables.get("alpha95").status == "NOT_COMPUTED"
    assert changed.snapshot.implementation == "older-solver"
    computed = replace(changed, metadata={"quality": "High accuracy"})
    with pytest.raises(ValueError, match="historical viewing"):
        computed.compatible_state()
    # Merely tagging retained results as an input design cannot bypass the gate.
    mislabeled = replace(changed, arrays={"some_result": np.ones(2)})
    with pytest.raises(ValueError, match="historical viewing"):
        mislabeled.compatible_state()


def test_installed_curved_design_remains_readable_without_overriding_new_defaults():
    from pathlib import Path
    from temsim.working_point import WorkingPointCheckpoint
    path = Path(__file__).parents[1]/"profiles/particle_tip_30mrad_20260915.temwp"
    before = path.read_bytes()
    point = WorkingPointCheckpoint.read_package(path)
    assert point.is_input_design and not point.arrays
    nodes = point.snapshot.graph["nodes"]
    tip = next(node for node in nodes if node.get("type", "").endswith(":ColdFieldEmitter"))
    surface = nodes[tip["attributes"]["_surface_model"]["ref"]]
    geometry = nodes[surface["attributes"]["geometry"]["ref"]]
    assert geometry["attributes"]["apex_radius_nm"] == 100.
    # Historical input designs require matching dependencies. Neither their
    # archived model choice nor this guard may be rewritten for new defaults.
    with pytest.raises(ValueError, match="Changed assembly:"):
        point.compatible_state()
    assert path.read_bytes() == before
    assert FieldEmissionGun().emitter.surface_model is None


def test_input_design_rejects_changed_dependency(tmp_path):
    from hashlib import sha256
    from temsim.optics.column import default_state
    from temsim.optics.gun_matching import input_working_point
    point = input_working_point(default_state(), label="Input design")
    path = tmp_path / "extra.toml"
    content = b"radius_nm = 100\n"
    path.write_bytes(content)
    row = {"role": "geometry", "path": str(path), "sha256": sha256(content).hexdigest(),
           "content_hex": content.hex()}
    snapshot = replace(point.snapshot, external_inputs=(*point.snapshot.external_inputs, row))
    design = replace(point, snapshot=snapshot)
    path.write_bytes(content.replace(b"\n", b"\r\n"))
    assert design.compatible_state().electron_gun._trace_cache is None
    path.write_bytes(b"radius_nm = 120\r\n")
    with pytest.raises(ValueError, match="Changed geometry"):
        design.compatible_state()


def test_emitting_cap_can_be_fitted_picked_and_located_in_parameters(qtbot):
    from PySide6.QtCore import QPointF, Qt
    from temsim.gui.part_model_editor import PartModelEditorPage
    from temsim.paths import CONFIG_ROOT
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.resize(1500, 900)
    assert page.open_path(CONFIG_ROOT / "sources/FEG_tip.toml", selected_key="feg_tip")
    page.show()
    qtbot.wait(50)
    assert page._mesh_records, page.status.text()
    assert page.region.findData("emitting_cap") == -1  # not an independent material
    page._fit_tip_emission()
    assert page.view._radius < .0001
    # Pick a triangle interior using the same camera/depth buffer as users.
    patch = next(mesh for mesh in page._mesh_records if "emitting_cap" in mesh["surfaces"])
    picked = None
    for face in patch["faces"][patch["face_groups"] == "emitting_cap"][::17]:
        point = page.view.project_points([patch["vertices"][face].mean(axis=0)])[0]
        picked = page.view.pick_topology_at(QPointF(*point[:2]), mode="face")
        if picked is not None and picked["id"] == "emitting_cap":
            break
    assert picked is not None and picked["id"] == "emitting_cap"
    page.view.set_topology_selection([picked], emit=True)
    highlighted = {tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole))
                   for row in range(page.dimensions.rowCount())
                   if page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole + 1)}
    assert ("parts", "feg_tip", "emission_cap_half_angle_deg") in highlighted
    assert ("parts", "feg_tip", "tip_radius_nm") in highlighted
    assert not page.session.dirty


def test_active_patch_override_is_distinguished_from_saved_dimensions():
    from temsim.part_model_3d import part_dimension_specs
    from temsim.part_model_document import PartModelDocument
    from temsim.paths import CONFIG_ROOT
    document = PartModelDocument(CONFIG_ROOT / "sources/FEG_tip.toml")
    model = curved_model()
    model = replace(model, emission=replace(model.emission, cap_half_angle_deg=5.))
    specs = part_dimension_specs(document.document, "feg_tip", runtime_values={
        "feg_tip": {"tip_surface_model": model.to_dict()}})
    saved = next(s for s in specs if s.path == ("parts", "feg_tip", "emission_cap_half_angle_deg"))
    active = next(s for s in specs if s.path == ("runtime", "feg_tip", "tip_surface_model", "emission", "cap_half_angle_deg"))
    assert saved.editable and saved.value == 10. and saved.label.startswith("Saved default")
    assert not active.editable and active.value == 5. and active.label.startswith("Active")
