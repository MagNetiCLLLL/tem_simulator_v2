"""Small renderer-neutral assembly checks; no GUI or optical solve."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim.assembly_model_3d import assembly_model_fingerprint, assembly_model_from_assembly
from temsim.column.module_assembly import AssemblyPart, ModulePart, VacuumLinerSegment
from temsim.part_model_3d import part_model_from_document


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _part(key="coil", **changes):
    row = dict(key=key, name=key, branch="illumination", parent_key=None,
               local_start_z_mm=10.0, local_center_z_mm=12.0, local_end_z_mm=14.0,
               length_mm=4.0, mechanical_profile="magnetic_excitation_coil",
               mechanical_inner_diameter_mm=4.0, mechanical_outer_diameter_mm=8.0,
               material_class="insulated_copper_winding")
    row.update(changes)
    return row


def _assembly(rows, *, shift=100.0, liners=(), module_key="module", active_keys=None):
    module_parts, active = [], []
    for row in rows:
        data = _freeze(deepcopy(row))
        values = (row["key"], row["name"], row["branch"], row["local_start_z_mm"],
                  row["local_center_z_mm"], row["local_end_z_mm"], row["length_mm"],
                  row.get("parent_key"), data)
        module_parts.append(ModulePart(*values))
        if active_keys is None or row["key"] in active_keys:
            active.append(AssemblyPart(module_key, "captured.toml", *values[:3],
                                      values[3] + shift, values[4] + shift, values[5] + shift,
                                      *values[6:]))
    return SimpleNamespace(modules=(SimpleNamespace(key=module_key, parts=tuple(module_parts)),),
                           parts=tuple(active), vacuum_liner_segments=tuple(liners))


def test_world_translation_matches_source_and_moves_semantic_edges_without_mutation():
    row = _part()
    assembly = _assembly([row], shift=987.0)
    source = part_model_from_document({"parts": [row]}, "coil", angular_segments=8).meshes[0]
    output = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not output.errors and not output.omitted_keys
    mesh, = output.meshes
    np.testing.assert_array_equal(mesh.faces, source.faces)
    np.testing.assert_allclose(mesh.vertices, source.vertices + (0, 0, 987), rtol=0, atol=0)
    assert mesh.vertices[:, 2].min() == 997.0
    assert mesh.vertices[:, 2].max() == 1001.0
    assert len(mesh.edges) == len(source.edges) > 0
    for edge, original in zip(mesh.edges, source.edges, strict=True):
        np.testing.assert_array_equal(edge["vertices"], original["vertices"] + (0, 0, 987))
        assert edge["id"] == original["id"]
        assert edge["parameter_paths"] == original["parameter_paths"]
        with pytest.raises(TypeError):
            edge["label"] = "changed"
    for array in (mesh.vertices, mesh.faces, mesh.face_groups, mesh.edges[0]["vertices"]):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        mesh.surfaces["outer"] = {}
    assert assembly.parts[0].data["local_start_z_mm"] == 10.0
    assert source.vertices[:, 2].min() == 10.0


def test_distinct_modules_use_their_own_resolved_origins():
    first = _assembly([_part("first")], shift=100, module_key="one")
    second = _assembly([_part("second")], shift=900, module_key="two")
    assembly = SimpleNamespace(modules=first.modules + second.modules,
                               parts=first.parts + second.parts, vacuum_liner_segments=())
    model = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not model.errors
    assert {mesh.key: mesh.vertices[:, 2].min() for mesh in model.meshes} == {"first": 110, "second": 910}


def test_source_parent_context_preserves_actual_split_objective_intervals():
    root = Path(__file__).resolve().parents[1]
    document = tomllib.loads((root / "configs/instruments/column/C3_ProbeCorrector.toml").read_text())
    keys = {"objective_lens", "objective_lens_excitation_coil"}
    rows = [row for row in document["parts"] if row["key"] in keys]
    # Source-only parent remains available for interval resolution, not drawn.
    assembly = _assembly(rows, shift=350, active_keys={"objective_lens_excitation_coil"})
    expected = part_model_from_document(document, "objective_lens_excitation_coil", include_children=False,
                                        angular_segments=8).meshes
    model = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not model.errors
    assert [mesh.region for mesh in model.meshes] == ["upper", "lower"]
    for actual, local in zip(model.meshes, expected, strict=True):
        np.testing.assert_array_equal(actual.vertices, local.vertices + (0, 0, 350))


def test_shared_child_is_rendered_once_and_optical_parent_envelopes_are_omitted():
    rows = [_part("c1", mechanical_profile="magnetic_lens_assembly"),
            _part("c2", mechanical_profile="magnetic_lens_assembly"),
            _part("shared", parent_key="c1", magnetic_lens_keys=["c1", "c2"])]
    model = assembly_model_from_assembly(_assembly(rows), angular_segments=8)
    assert [mesh.key for mesh in model.meshes] == ["shared"]
    assert set(model.omitted_keys) == {"c1", "c2"}
    assert not model.errors


@pytest.mark.parametrize("changes", [
    dict(mechanical_profile="branch_reference_plane"),
    dict(branch_path_only=True, housing_length_mm=5.0),
    dict(local_start_z_mm=12., local_end_z_mm=12., length_mm=0.),
    dict(virtual=True),
])
def test_reference_or_branch_duplicates_are_not_presented_as_axial_material(changes):
    model = assembly_model_from_assembly(_assembly([_part(**changes)]), angular_segments=8)
    assert not model.meshes and model.omitted_keys == ("coil",)
    assert not model.errors


def test_custom_solid_on_optical_parent_is_preserved():
    pytest.importorskip("manifold3d")
    model_3d = dict(schema_version=1, base=dict(kind="box", width_mm=6., height_mm=4., length_mm=2.),
                    features=[], transform=dict(offset_mm=[1., 2., 3.], rotation_deg=[0., 0., 30.]))
    rows = [_part("lens", mechanical_profile="magnetic_lens_assembly", model_3d=model_3d),
            _part("child", parent_key="lens")]
    model = assembly_model_from_assembly(_assembly(rows), angular_segments=8)
    assert not model.errors
    assert {mesh.key for mesh in model.meshes} == {"lens", "child"}
    parent = next(mesh for mesh in model.meshes if mesh.key == "lens")
    assert parent.vertices[:, 2].min() == 114.0
    assert parent.vertices[:, 2].max() == 116.0


def test_invalid_part_does_not_erase_other_parts_and_unknown_shape_is_honest():
    rows = [_part("good"), _part("bad", mechanical_inner_diameter_mm=20),
            _part("approx", mechanical_profile="unknown_mechanism")]
    model = assembly_model_from_assembly(_assembly(rows), angular_segments=8)
    assert {mesh.key for mesh in model.meshes} == {"good", "approx"}
    assert model.omitted_keys == ("bad",)
    assert len(model.errors) == 1 and model.errors[0].startswith("bad:")
    assert not next(mesh for mesh in model.meshes if mesh.key == "approx").is_exact
    assert any("Envelope only" in note for note in model.notes)


def test_unrepresented_axial_resizing_is_reported_not_silently_mistranslated():
    assembly = _assembly([_part()])
    assembly.parts = (replace(assembly.parts[0], end_z_mm=130.0),)
    model = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not model.meshes
    assert "translation alone is insufficient" in model.errors[0]


def test_resolved_vacuum_liner_is_material_annulus_and_bore_is_not_a_solid():
    liner = VacuumLinerSegment("@liner", "Existing tube", 700., 720., 4., 6., 1.)
    assembly = _assembly([], liners=(liner,))
    assembly.vacuum_bore_segments = (SimpleNamespace(key="vacuum", inner_diameter_mm=4.0),)
    model = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not model.errors
    mesh, = model.meshes
    assert mesh.key == "@liner" and mesh.is_exact
    assert set(mesh.vertices[:, 2]) == {700., 720.}
    np.testing.assert_allclose(np.unique(np.round(np.linalg.norm(mesh.vertices[:, :2], axis=1), 8)), [2., 3.])


def test_fingerprint_tracks_all_geometry_but_not_unrelated_operating_values():
    assembly = _assembly([_part()])
    before = assembly_model_fingerprint(assembly)
    assert assembly_model_fingerprint(_assembly([_part()])) == before
    assert assembly_model_fingerprint(assembly, {"coil": {"excitation": 42, "current_a": 5}}) == before
    assert assembly_model_fingerprint(_assembly([_part()], shift=101)) != before
    assert assembly_model_fingerprint(_assembly([_part(mechanical_outer_diameter_mm=10)])) != before
    liner = VacuumLinerSegment("@liner", "tube", 700., 720., 4., 6., 1.)
    assert assembly_model_fingerprint(_assembly([_part()], liners=(liner,))) != before


def _strip():
    return _part("aperture", mechanical_profile="circular_aperture", aperture_plate_form="perforated_strip",
                  plate_thickness_mm=.1, mechanical_bore_diameter_mm=4.)


def test_aperture_runtime_opening_and_offsets_match_renderer_and_fingerprint():
    pytest.importorskip("manifold3d")
    assembly = _assembly([_strip()])
    values = {"aperture": {"radius_mm": .2, "offset_x_mm": .1, "offset_y_mm": .3}}
    before = deepcopy(values)
    model = assembly_model_from_assembly(assembly, runtime_values=values, angular_segments=8)
    assert not model.errors
    mesh, = model.meshes
    assert "0.4 mm (runtime)" in mesh.description
    digest = assembly_model_fingerprint(assembly, values)
    assert assembly_model_fingerprint(assembly, {"aperture": {**values["aperture"], "excitation": 99}}) == digest
    for field in ("radius_mm", "offset_x_mm", "offset_y_mm"):
        changed = deepcopy(values)
        changed["aperture"][field] += .1
        assert assembly_model_fingerprint(assembly, changed) != digest
    assert values == before


def test_explicit_custom_aperture_base_ignores_unused_runtime_opening():
    row = _strip()
    row["model_3d"] = dict(schema_version=1, base=dict(kind="box", width_mm=6., height_mm=4., length_mm=2.), features=[])
    assembly = _assembly([row])
    assert assembly_model_fingerprint(assembly, {"aperture": {"radius_mm": 1.}}) == assembly_model_fingerprint(
        assembly, {"aperture": {"radius_mm": 5.}})


@pytest.mark.parametrize("count", [True, 2, 2.5, 4097])
def test_tessellation_is_validated_even_for_an_empty_assembly(count):
    with pytest.raises(ValueError, match="angular_segments"):
        assembly_model_from_assembly(_assembly([]), angular_segments=count)


def test_build_and_fingerprint_do_not_read_source_files(monkeypatch):
    assembly = _assembly([_part()])
    def no_io(*args, **kwargs):
        pytest.fail("Captured assembly rendering must not open a source file")
    monkeypatch.setattr(Path, "open", no_io)
    assert len(assembly_model_from_assembly(assembly, angular_segments=8).meshes) == 1
    assert len(assembly_model_fingerprint(assembly)) == 64


def test_saved_copper_assignment_matches_parts_editor_and_captured_model_is_immutable():
    from temsim.gui.part_model_editor import PartModelEditorPage
    from temsim.part_materials import configured_region_colour, material_catalog

    row = _part(material_class="soft_magnetic")
    row["material_regions"] = {
        "body": next(item for item in material_catalog() if item["material_key"] == "copper"),
    }
    source_before = deepcopy(row)
    assembly = _assembly([row])
    whole = assembly_model_from_assembly(assembly, angular_segments=8)
    mesh, = whole.meshes
    expected = (200 / 255, 135 / 255, 78 / 255, 1.)
    assert mesh.color == expected
    assert configured_region_colour(row) == expected

    # Exercise the actual editor publication method without rendering Qt/GPU
    # surfaces or replacing any real editor session / user source file.
    editor = SimpleNamespace(
        session=SimpleNamespace(document={"parts": [row]}, part=lambda key: row),
        _selected_key="coil", _aperture_index=0, _topology_selection=(),
        _model_runtime_values=lambda: {}, scope=SimpleNamespace(currentData=lambda: "part"),
        view=SimpleNamespace(set_meshes=lambda *args, **kwargs: None,
                             set_topology_selection=lambda value: None, topology_selection=()),
        _sync_view_selection=lambda: None, _topology_changed=lambda *args, **kwargs: None,
        _message=lambda message, **kwargs: pytest.fail(message),
    )
    assert PartModelEditorPage._render(editor)
    assert editor._mesh_records[0]["color"] == mesh.color
    assert row == source_before
    row["material_regions"]["body"]["material_key"] = "aluminum"
    assert assembly.parts[0].data["material_regions"]["body"]["material_key"] == "copper"
    assert mesh.color == expected


def test_region_palette_fallback_and_explicit_region_assignment_preserve_geometry():
    from temsim.part_materials import configured_region_colour, material_catalog

    row = _part(mechanical_profile="magnetic_lens_yoke")
    fallback = (.12, .34, .56, .28)
    assert configured_region_colour(row, "lower", fallback) == fallback
    materials = {item["material_key"]: item for item in material_catalog()}
    row["material_regions"] = {"body": materials["copper"], "upper": materials["aluminum"]}
    before = deepcopy(row)
    assert configured_region_colour(row, "upper", fallback) == (183 / 255, 197 / 255, 211 / 255, 1.)
    assert configured_region_colour(row, "lower", fallback) == (200 / 255, 135 / 255, 78 / 255, 1.)
    assert row == before
    assembly = _assembly([row])
    captured = assembly_model_from_assembly(assembly, angular_segments=8)
    assert not captured.errors
    bare = deepcopy(row)
    bare.pop("material_regions")
    original = assembly_model_from_assembly(_assembly([bare]), angular_segments=8)
    np.testing.assert_array_equal(captured.meshes[0].vertices, original.meshes[0].vertices)
    np.testing.assert_array_equal(captured.meshes[0].faces, original.meshes[0].faces)
    assert assembly_model_fingerprint(assembly) != assembly_model_fingerprint(_assembly([bare]))
