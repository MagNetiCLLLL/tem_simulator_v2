"""Small geometric checks; no field integration, coherent solve or GPU."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim.assembly_model_3d import assembly_model_fingerprint, assembly_model_from_assembly
from temsim.energy_filter_model_3d import (
    FILTER_TO_COLUMN_ROTATION, energy_filter_component_pose, energy_filter_frame_mm, energy_filter_render_values,
    mount_energy_filter_mesh, rigid_mesh, supports_energy_filter_part,
)
from temsim.part_model_3d import module_model_from_document, part_dimension_specs, part_model_from_document


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def document():
    return tomllib.loads((ROOT / "configs/subassemblies/energy_filter.toml").read_text(encoding="utf-8"))


def _parts(document):
    return {row["key"]: row for row in document["parts"]}


def _assembly(document, *, shift=2000.):
    rows = deepcopy(document["parts"])
    module = SimpleNamespace(key="filter", parts=tuple(SimpleNamespace(data=row) for row in rows))
    parts = tuple(SimpleNamespace(
        key=row["key"], module_key="filter", data=row, parent_key=row.get("parent_key"),
        start_z_mm=row["local_start_z_mm"] + shift,
        center_z_mm=row["local_center_z_mm"] + shift,
        end_z_mm=row["local_end_z_mm"] + shift, length_mm=row["length_mm"],
    ) for row in rows)
    return SimpleNamespace(parts=parts, modules=(module,), vacuum_liner_segments=())


def _model(document, key, **kwargs):
    return part_model_from_document(document, key, angular_segments=16, include_children=False, **kwargs)


def test_mount_is_right_handed_and_90_degree_exit_is_horizontal(document):
    rotation = np.array(FILTER_TO_COLUMN_ROTATION)
    np.testing.assert_array_equal(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == 1.
    np.testing.assert_array_equal(rotation @ (1., 0., 0.), (0., 0., 1.))
    np.testing.assert_array_equal(rotation @ (0., 0., -1.), (1., 0., 0.))
    parts = _parts(document)
    camera_origin, camera_frame = energy_filter_frame_mm(parts["energy_filter_zebra"], parts)
    np.testing.assert_allclose(camera_origin, (240., 0., -610.), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rotation @ camera_origin, (610., 0., 240.), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rotation @ camera_frame[:, 2], (1., 0., 0.), rtol=0, atol=1e-12)


def test_all_supported_parts_share_builder_and_mounted_snapshot_preserves_selection(document):
    before = deepcopy(document)
    assembly = _assembly(document)
    runtime = {"energy_filter_slit": {"gap_m": 36e-6, "centre_m": 5e-6}}
    result = assembly_model_from_assembly(assembly, runtime_values=runtime, angular_segments=16)
    assert not result.errors
    assert set(result.omitted_keys) == {"energy_filter", "energy_filter_eftem_output_plane"}
    supported = {row["key"] for row in document["parts"] if supports_energy_filter_part(row)}
    assert len(supported) == 18
    assert {mesh.key for mesh in result.meshes} == supported
    assert len({(mesh.key, mesh.region) for mesh in result.meshes}) == len(result.meshes)
    for row in document["parts"]:
        if row["key"] not in supported:
            continue
        local = _model(document, row["key"], runtime_values=runtime)
        output = [mesh for mesh in result.meshes if mesh.key == row["key"]]
        for mesh, original in zip(output, local.meshes, strict=True):
            expected = mount_energy_filter_mesh(original, 2000.)
            np.testing.assert_allclose(mesh.vertices, expected.vertices, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(mesh.faces, original.faces)
            np.testing.assert_array_equal(mesh.face_groups, original.face_groups)
            assert mesh.surfaces.keys() == original.surfaces.keys()
            for edge, expected_edge in zip(mesh.edges, expected.edges, strict=True):
                assert edge["id"] == expected_edge["id"]
                assert edge["parameter_paths"] == expected_edge["parameter_paths"]
                np.testing.assert_allclose(edge["vertices"], expected_edge["vertices"], rtol=0, atol=1e-12)
            with pytest.raises(ValueError):
                mesh.vertices.setflags(write=True)
    assert document == before


def test_expanded_module_datum_does_not_translate_branch_twice(document):
    original = _assembly(document, shift=2000.)
    for row in document["parts"]:
        for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
            if field in row:
                row[field] += 1234.
    expanded = _assembly(document, shift=766.)
    before = assembly_model_from_assembly(original, angular_segments=8)
    after = assembly_model_from_assembly(expanded, angular_segments=8)
    assert not before.errors and not after.errors
    for a, b in zip(before.meshes, after.meshes, strict=True):
        assert (a.key, a.region) == (b.key, b.region)
        np.testing.assert_allclose(a.vertices, b.vertices, rtol=0, atol=1e-10)


def test_prism_has_only_gap_boundaries_and_radial_channel_no_yoke(document):
    model = _model(document, "energy_filter_tapered_prism")
    assert {mesh.region for mesh in model.meshes} == {
        "lower_pole_boundary", "upper_pole_boundary", "inner_radial_channel", "outer_radial_channel"}
    for mesh in model.meshes:
        assert all(item["open_surface"] for item in mesh.surfaces.values())
        if "pole_boundary" in mesh.region:
            expected_y = -15. if mesh.region.startswith("lower") else 15.
            np.testing.assert_array_equal(mesh.vertices[:, 1], expected_y)
            assert not mesh.wireframe
            normal = mesh.surfaces[mesh.region]["normal"]
            np.testing.assert_allclose(normal, (0., -np.sign(expected_y), 0.))
        else:
            assert mesh.wireframe
            radial = np.linalg.norm(mesh.vertices[:, (0, 2)] - (105., -135.), axis=1)
            np.testing.assert_allclose(radial, 105. if mesh.region.startswith("inner") else 165.)
    path = next(edge["vertices"] for mesh in model.meshes for edge in mesh.edges
                if edge["id"] == "reference_centerline")
    np.testing.assert_allclose(path[0], (0., 0., 0.))
    np.testing.assert_allclose(path[-1], (240., 0., -610.))


def test_multipole_uses_housing_length_and_keeps_bore_open(document):
    parts = _parts(document)
    for index in range(1, 11):
        key = f"energy_filter_multipole_{index:02d}"
        mesh, = _model(document, key).meshes
        origin, rotation = energy_filter_frame_mm(parts[key], parts)
        local = (mesh.vertices - origin) @ rotation
        assert np.ptp(local[:, 2]) == pytest.approx(22. if index <= 3 else 28.)
        np.testing.assert_allclose(np.unique(np.round(np.linalg.norm(local[:, :2], axis=1), 9)), (7.5, 35.))
        counts = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                         for a, b in zip(face, np.roll(face, -1)))
        assert set(counts.values()) == {2}
        assert parts[key]["magnetic_support_length_mm"] == 20.
        axial = [surface for surface in mesh.surfaces.values() if surface["kind"] == "cap"]
        assert all(("parts", key, "housing_length_mm") in surface["parameter_paths"] for surface in axial)
        assert all(("parts", key, "length_mm") not in surface["parameter_paths"] for surface in axial)


def test_runtime_slit_gap_and_center_move_two_facing_blade_edges_and_cache(document):
    key = "energy_filter_slit"
    runtime = {key: {"gap_m": 36e-6, "centre_m": 20e-6}}
    model = _model(document, key, runtime_values=runtime)
    parts = _parts(document)
    origin, rotation = energy_filter_frame_mm(parts[key], parts)
    for mesh, expected_x, direction in zip(model.meshes, (.002, .038), (1., -1.), strict=True):
        local = (mesh.vertices - origin) @ rotation
        np.testing.assert_allclose(local[:, 0], expected_x, rtol=0, atol=1e-12)
        assert np.ptp(local[:, 1]) == pytest.approx(12.5)
        assert np.ptp(local[:, 2]) == pytest.approx(1.)
        surface = mesh.surfaces[mesh.region]
        np.testing.assert_allclose(rotation.T @ surface["normal"], (direction, 0., 0.), atol=1e-12)
        assert ("runtime", key, "gap_m") in surface["parameter_paths"]
    assembly = _assembly(document)
    initial = assembly_model_fingerprint(assembly, runtime)
    for field in ("gap_m", "centre_m"):
        changed = deepcopy(runtime)
        changed[key][field] += 1e-6
        assert assembly_model_fingerprint(assembly, changed) != initial
    assert assembly_model_fingerprint(assembly, {key: {**runtime[key], "requested_width_ev": 99}}) == initial
    assert any("maximum travel" in note for note in _model(document, key).notes)


def test_zebra_active_regions_have_no_package_or_invented_alignment_offset(document):
    key = "energy_filter_zebra"
    parts = _parts(document)
    origin, rotation = energy_filter_frame_mm(parts[key], parts)
    meshes = _model(document, key).meshes
    assert len(meshes) == 6
    strips, alignment = meshes[:5], meshes[5]
    for index, mesh in enumerate(strips):
        local = (mesh.vertices - origin) @ rotation
        np.testing.assert_allclose(np.ptp(local, axis=0), (28.672, .8, 0.), atol=1e-12)
        assert local[:, 1].mean() == pytest.approx(index - 2.)
        assert not mesh.wireframe
    assert alignment.wireframe and alignment.region == "alignment_active_area"
    local = (alignment.vertices - origin) @ rotation
    np.testing.assert_allclose(local.mean(axis=0), 0., atol=1e-12)
    np.testing.assert_allclose(np.ptp(local, axis=0), (28.672, 3.584, 0.), atol=1e-12)
    assert "relative sensor layout unknown" in alignment.description


@pytest.mark.parametrize("key", ["energy_filter_shutter", "energy_filter_camera_deflector"])
def test_unknown_deflection_electrodes_are_guides_not_vacuum_solids(document, key):
    meshes = _model(document, key).meshes
    assert len(meshes) == 3 and all(mesh.wireframe for mesh in meshes)
    assert all(not mesh.is_exact for mesh in meshes)


def test_geometry_capture_and_mesh_builder_do_not_ensure_state_or_read_files(document, monkeypatch):
    state = SimpleNamespace(energy_filter=SimpleNamespace(energy_slit=SimpleNamespace(gap_m=36e-6, centre_m=2e-6)))
    before = deepcopy(vars(state.energy_filter.energy_slit))
    assert energy_filter_render_values(state) == {"energy_filter_slit": before}
    assert vars(state.energy_filter.energy_slit) == before
    assert energy_filter_render_values(SimpleNamespace()) == {}
    assert energy_filter_render_values(SimpleNamespace(energy_filter=SimpleNamespace(energy_slit=None))) == {}
    # Imports are complete; captured geometry must not reopen TOML or solve a field.
    module_model_from_document(document, angular_segments=8)
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("Captured geometry reopened a file"))
    result = assembly_model_from_assembly(_assembly(document), angular_segments=8)
    assert not result.errors


def test_branch_position_dimensions_are_available_and_zero_packing_length_readonly(document):
    fields = {spec.path[-1]: spec for spec in part_dimension_specs(document, "energy_filter_multipole_04")}
    assert fields["path_center_mm"].editable
    assert fields["housing_length_mm"].editable
    assert not fields["length_mm"].editable


@pytest.mark.parametrize("changes, message", [
    ({"prism_radius_mm": -1.}, "positive"),
    ({"radial_clear_half_width_mm": 140.}, "curvature centre"),
    ({"bend_angle_deg": 180.}, "180"),
    ({"local_center_z_mm": 10.}, "entrance datum"),
])
def test_invalid_branch_dimensions_fail_without_erasing_other_geometry(document, changes, message):
    _parts(document)["energy_filter_tapered_prism"].update(changes)
    with pytest.raises(ValueError, match=message):
        _model(document, "energy_filter_tapered_prism")
    result = assembly_model_from_assembly(_assembly(document), angular_segments=8)
    assert result.errors and "energy_filter_tapered_prism" in result.omitted_keys
    assert any(mesh.key == "energy_filter_multipole_01" for mesh in result.meshes)


def test_reflections_are_rejected_and_normals_rotate_with_surfaces(document):
    mesh = _model(document, "energy_filter_slit").meshes[0]
    with pytest.raises(ValueError, match="right-handed"):
        rigid_mesh(mesh, np.diag((1., 1., -1.)), (0., 0., 0.))
    mounted = mount_energy_filter_mesh(mesh, 2000.)
    for region, surface in mesh.surfaces.items():
        np.testing.assert_allclose(mounted.surfaces[region]["normal"],
                                   np.array(FILTER_TO_COLUMN_ROTATION) @ surface["normal"])


def test_part_editor_refresh_dependency_follows_actual_slit_position(document):
    from temsim.gui.part_model_editor import PartModelEditorPage
    runtime = {"energy_filter_slit": {"gap_m": 36e-6, "centre_m": 0.}}
    editor = SimpleNamespace(session=SimpleNamespace(document=document), _selected_key="energy_filter_slit",
                             scope=SimpleNamespace(currentData=lambda: "part"), _model_runtime_values=lambda: runtime)
    before = PartModelEditorPage._runtime_mesh_dependencies(editor)
    runtime["energy_filter_slit"]["gap_m"] = 50e-6
    assert PartModelEditorPage._runtime_mesh_dependencies(editor) != before
    runtime["energy_filter_slit"]["gap_m"] = 36e-6
    runtime["energy_filter_slit"]["requested_width_ev"] = 12.
    assert PartModelEditorPage._runtime_mesh_dependencies(editor) == before


def test_filter_display_regions_share_validated_body_palette_without_new_material_domains(document):
    from temsim.part_materials import configured_region_colour, material_catalog, validated_material_regions
    row = _parts(document)["energy_filter_zebra"]
    row["material_regions"] = {"body": next(item for item in material_catalog() if item["material_key"] == "copper")}
    result = assembly_model_from_assembly(_assembly(document), angular_segments=8)
    assert not result.errors
    for mesh in result.meshes:
        if mesh.key == row["key"]:
            assert mesh.color == (200/255, 135/255, 78/255, 1.)
            assert configured_region_colour(row, mesh.region) == mesh.color
    assert set(validated_material_regions(row)) == {"body"}
    with pytest.raises(ValueError, match="Unsupported material region"):
        configured_region_colour(row, "active_strip_999")
    with pytest.raises(ValueError, match="Unsupported material region"):
        configured_region_colour({"key": "other"}, "active_strip_1")
    row["material_regions"]["body"]["relative_permeability"] = -1.
    with pytest.raises(ValueError, match="permeability"):
        configured_region_colour(row, "active_strip_1")


def _custom_model():
    return dict(schema_version=1, base=dict(kind="box", width_mm=6., height_mm=4., length_mm=8.), features=[],
                transform=dict(offset_mm=[1., 2., 3.], rotation_deg=[0., 0., 30.]))


def test_explicit_entrance_cad_preserves_existing_column_axes_and_resolved_datum(document):
    key = "energy_filter_entrance_aperture"
    row = _parts(document)[key]
    row["model_3d"] = _custom_model()
    plain = {**deepcopy(row), "key": "ordinary_axial_part", "branch": "column"}
    expected, = part_model_from_document({"parts": [plain]}, plain["key"], angular_segments=8).meshes
    result = assembly_model_from_assembly(_assembly(document), angular_segments=8)
    assert not result.errors
    actual, = [mesh for mesh in result.meshes if mesh.key == key]
    np.testing.assert_allclose(actual.vertices, expected.vertices + (0., 0., 2000.), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.faces, expected.faces)


def test_explicit_downstream_cad_uses_local_beam_axes_before_branch_mount(document):
    key = "energy_filter_multipole_04"
    row = _parts(document)[key]
    row["model_3d"] = _custom_model()
    plain = {**deepcopy(row), "key": "ordinary_axial_part", "branch": "column", "branch_path_only": False}
    expected, = part_model_from_document({"parts": [plain]}, plain["key"], angular_segments=8).meshes
    actual, = _model(document, key).meshes
    origin, rotation = energy_filter_frame_mm(row, _parts(document))
    np.testing.assert_allclose((actual.vertices - origin) @ rotation, expected.vertices, rtol=0, atol=1e-12)


def test_existing_prism_cad_transform_keeps_open_edges_and_reference_path(document):
    key = "energy_filter_tapered_prism"
    before = _model(document, key)
    _parts(document)[key]["model_3d"] = dict(schema_version=1, base=dict(kind="existing"), features=[],
                                            transform=dict(offset_mm=[0., 2., 0.]))
    after = _model(document, key)
    for old, new in zip(before.meshes, after.meshes, strict=True):
        np.testing.assert_allclose(new.vertices, old.vertices + (0., 2., 0.))
        assert {edge["id"] for edge in old.edges} <= {edge["id"] for edge in new.edges}
    path = next(edge for mesh in after.meshes for edge in mesh.edges if edge["id"] == "reference_centerline")
    np.testing.assert_allclose(path["vertices"][0], (0., 2., 0.))


def test_actual_energy_filter_assembly_has_no_errors_or_missing_branch_material():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    state = default_state()
    assembly = AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3", "Energy Filter"))
    model = assembly_model_from_assembly(assembly, runtime_values=energy_filter_render_values(state), angular_segments=8)
    assert not model.errors
    expected = {part.key for part in assembly.parts if supports_energy_filter_part(part.data)}
    assert len(expected) == 18
    assert expected <= {mesh.key for mesh in model.meshes}
    inlet = assembly.part("energy_filter_entrance_aperture").center_z_mm
    camera = np.vstack([mesh.vertices for mesh in model.meshes if mesh.key == "energy_filter_zebra"])
    assert camera[:, 0].mean() == pytest.approx(610.)
    assert camera[:, 2].mean() == pytest.approx(inlet + 240.)


@pytest.mark.parametrize("key", ["energy_filter_entrance_aperture", "energy_filter_tapered_prism", "energy_filter_multipole_04"])
@pytest.mark.parametrize("kind", ["hole", "slot"])
def test_picked_branch_feature_returns_to_component_cad_axes(document, key, kind):
    from temsim.gui.part_model_editor import PartModelEditorPage
    from temsim.part_model_features import feature_placement, _transform
    parts = _parts(document)
    # Resolved module-local datums can be far from zero, while branch points
    # are measured from the spectrometer entrance. Exercise both transforms.
    for row in parts.values():
        for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
            if field in row:
                row[field] += 1234.
    part = parts[key]
    part["model_3d"] = _custom_model()
    matrix, offset = _transform(part)
    point = matrix @ np.array((2., 1., 4.)) + offset + (0., 0., 1234.)
    normal = matrix @ np.array((0., 0., 1.))
    expected_center, expected_axis, expected_depth = feature_placement(part, point, normal)
    origin, rotation = energy_filter_component_pose(part, parts)
    hit = dict(key=key, point=origin + rotation @ (point - (0., 0., 1234.)), normal=rotation @ normal)
    editor = SimpleNamespace(session=SimpleNamespace(document=document, part=lambda chosen: parts[chosen]),
                             _selected_key=key, _topology_selection=(hit,))
    feature = PartModelEditorPage._feature_defaults(editor, kind)
    np.testing.assert_allclose(feature["center_mm"], expected_center, rtol=0, atol=1e-12)
    assert feature["axis"] == expected_axis
    assert feature["depth_mm"] == pytest.approx(expected_depth)
