"""The 3-D view uses declared material, holes and ownership, without Qt."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.part_model_3d import (
    module_model_from_document, part_dimension_specs, part_model_from_document, revolve_section,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "instruments"


def _document(path):
    return tomllib.loads((CONFIGS / path).read_text(encoding="utf-8-sig"))


def _part(document, key):
    return next(part for part in document["parts"] if part["key"] == key)


def _assert_mesh(mesh, *, closed=True):
    assert mesh.vertices.ndim == mesh.faces.ndim == 2
    assert mesh.vertices.shape[1] == mesh.faces.shape[1] == 3
    assert np.isfinite(mesh.vertices).all()
    assert np.issubdtype(mesh.faces.dtype, np.integer)
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < len(mesh.vertices)
    triangles = mesh.vertices[mesh.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    assert (areas > 0).all()
    assert np.isfinite(areas).all()
    if closed:
        counts = Counter(tuple(sorted(edge)) for a, b, c in mesh.faces
                         for edge in ((a, b), (b, c), (c, a)))
        assert set(counts.values()) == {2}
        # Positive signed volume verifies winding, including mirrored poles.
        shifted = triangles - mesh.vertices.mean(axis=0)
        volume = np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6
        assert volume > 0


@pytest.fixture
def document():
    return {"parts": [{
        "key": "test_coil", "name": "Test winding", "parent_key": "lens",
        "mechanical_profile": "magnetic_excitation_coil",
        "local_start_z_mm": 10.0, "local_center_z_mm": 13.0,
        "local_end_z_mm": 20.0, "length_mm": 10.0,
        "mechanical_inner_diameter_mm": 8.0, "mechanical_outer_diameter_mm": 12.0,
        "vacuum_inner_diameter_mm": 2.0, "material_class": "insulated_copper_winding",
    }]}


@pytest.mark.parametrize("section,axis_vertices", [
    ([(0, 0), (0, 3), (5, 3), (5, 0)], 2),  # Solid cylinder.
    ([(0, 0), (0, 3), (5, 0)], 2),  # A cone's tip is one vertex.
    ([(0, 1), (0, 3), (5, 3), (5, 1)], 0),  # Real through bore.
    ([(0, 2), (0, 3), (5, 3), (5, .5)], 0),  # Conical bore.
])
@pytest.mark.parametrize("reverse", [False, True])
def test_revolved_sections_are_closed_outward_and_never_duplicate_axis(section, axis_vertices, reverse):
    mesh = revolve_section(section[::-1] if reverse else section, key="primitive", angular_segments=16)
    _assert_mesh(mesh)
    assert (np.linalg.norm(mesh.vertices[:, :2], axis=1) == 0).sum() == axis_vertices
    assert len(np.unique(mesh.vertices, axis=0)) == len(mesh.vertices)
    assert not mesh.vertices.flags.writeable
    assert not mesh.faces.flags.writeable


def test_annulus_preserves_local_origin_real_material_bore_and_metadata(document):
    before = deepcopy(document)
    model = part_model_from_document(document, "test_coil", angular_segments=16)
    mesh, = model.meshes
    _assert_mesh(mesh)
    assert mesh.key == mesh.part_key == "test_coil"
    assert mesh.region == mesh.region_key == "body"
    assert mesh.material_class == "insulated_copper_winding"
    assert mesh.is_exact
    assert set(np.round(np.linalg.norm(mesh.vertices[:, :2], axis=1), 6)) == {4.0, 6.0}
    assert set(mesh.vertices[:, 2]) == {10.0, 20.0}
    assert len(mesh.color) == 4 and all(0 <= value <= 1 for value in mesh.color)
    assert document == before


def test_radial_profile_uses_both_piecewise_radii_at_each_knot(document):
    part = document["parts"][0]
    part.update(mechanical_profile="magnetic_lens_yoke",
                magnetic_radial_profile_mm=[[0, 1, 6], [3, 2, 4], [10, 1.5, 5]])
    mesh, = part_model_from_document(document, part["key"]).meshes
    _assert_mesh(mesh)
    for offset, inner, outer in part["magnetic_radial_profile_mm"]:
        ring = mesh.vertices[mesh.vertices[:, 2] == 10 + offset]
        assert set(np.round(np.linalg.norm(ring[:, :2], axis=1), 7)) == {inner, outer}


@pytest.mark.parametrize("kind", ["excitation_coil", "yoke"])
def test_actual_objective_layers_use_two_physical_intervals_and_source_dimensions(kind):
    document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    part = _part(document, f"objective_lens_{kind}")
    parent = _part(document, "objective_lens")
    before = deepcopy(document)
    model = part_model_from_document(document, part["key"], include_children=False)
    intervals = objective_layer_intervals_mm(parent, parent["local_start_z_mm"], part["mechanical_profile"])
    assert [mesh.region for mesh in model.meshes] == ["upper", "lower"]
    for mesh, (start, end) in zip(model.meshes, intervals):
        _assert_mesh(mesh)
        assert mesh.vertices[:, 2].min() == pytest.approx(start)
        assert mesh.vertices[:, 2].max() == pytest.approx(end)
    length = next(field for field in model.fields if field.path == ("parts", part["key"], "length_mm"))
    assert not length.editable and "physical sections" in length.reason
    names = {field.path for field in model.fields if field.editable}
    for side in ("upper", "lower"):
        for edge in ("start", "end"):
            assert ("parts", parent["key"], f"{side}_yoke_{edge}_local_z_mm") in names
    if kind == "excitation_coil":
        assert ("parts", parent["key"], "mechanical_coil_axial_inset_mm") in names
    assert document == before


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_actual_objective_pole_has_connector_shank_shoulder_nose_and_open_bore(side):
    document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    part = _part(document, f"objective_{side}_pole")
    mesh, = part_model_from_document(document, part["key"]).meshes
    _assert_mesh(mesh)
    assert mesh.region == "body" and mesh.is_exact
    radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    assert radius.min() == pytest.approx(part["mechanical_bore_diameter_mm"] / 2)
    assert radius.max() == pytest.approx(part["mechanical_outer_diameter_mm"] / 2)
    face = part["local_end_z_mm"] if side == "upper" else part["local_start_z_mm"]
    assert radius[mesh.vertices[:, 2] == face].max() == pytest.approx(part["mechanical_tip_diameter_mm"] / 2)
    outside = part["local_start_z_mm"] if side == "upper" else part["local_end_z_mm"]
    direction = 1 if side == "upper" else -1
    expected_z = {outside, face,
                  outside + direction * part["pole_vacuum_connector_axial_length_mm"],
                  outside + direction * part["pole_mounting_shank_axial_length_mm"],
                  face - direction * part["pole_nose_axial_length_mm"]}
    assert set(mesh.vertices[:, 2]) == expected_z


@pytest.mark.parametrize("key,face_at_end", [
    ("condenser_lens_1_lower_pole", True), ("condenser_lens_2_upper_pole", False),
])
def test_actual_shared_cartridge_pole_faces_follow_physical_direction(key, face_at_end):
    document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    part = _part(document, key)
    mesh, = part_model_from_document(document, key).meshes
    face = part["local_end_z_mm" if face_at_end else "local_start_z_mm"]
    radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    assert radius[mesh.vertices[:, 2] == face].max() == pytest.approx(part["mechanical_tip_diameter_mm"] / 2)


def test_shared_bodies_are_not_duplicated_or_claimed_as_solid_homogeneous_material():
    document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    for key in ("condenser_lens_1", "condenser_lens_2"):
        model = part_model_from_document(document, key)
        shared = [mesh for mesh in model.meshes if mesh.key == "c1_c2_pole_piece_cartridge"]
        assert len(shared) == 1
        assert not shared[0].is_exact
        assert "Envelope" in shared[0].description
        assert len({(mesh.key, mesh.region) for mesh in model.meshes}) == len(model.meshes)
    module = module_model_from_document(document, angular_segments=8)
    assert sum(mesh.key == "c1_c2_pole_piece_cartridge" for mesh in module.meshes) == 1


def test_multi_gap_shared_profile_is_shown_as_one_shaped_body_for_each_channel(document):
    body = document["parts"][0]
    body.update(parent_key="channel_a", mechanical_profile="magnetic_lens_yoke",
                magnetic_lens_keys=["channel_a", "channel_b"],
                magnetic_radial_profile_mm=[[0, 1, 6], [4, 2, 4], [10, 1, 6]])
    for key in ("channel_a", "channel_b"):
        document["parts"].append({**body, "key": key, "parent_key": "", "magnetic_lens_keys": [],
                                  "mechanical_profile": "magnetic_lens_assembly",
                                  "magnetic_circuit_topology": "shared_pole_multi_gap"})
        document["parts"][-1].pop("magnetic_radial_profile_mm")
    for key in ("channel_a", "channel_b"):
        model = part_model_from_document(document, key)
        shared = [mesh for mesh in model.meshes if mesh.key == body["key"]]
        assert len(shared) == 1 and shared[0].is_exact


def test_explicit_disjoint_intervals_keep_gaps_in_one_body_region(document):
    document["parts"][0]["material_intervals_mm"] = [[10, 12], [18, 20]]
    mesh, = part_model_from_document(document, "test_coil").meshes
    _assert_mesh(mesh)
    assert mesh.region == "body"
    assert set(mesh.vertices[:, 2]) == {10, 12, 18, 20}
    triangles = mesh.vertices[mesh.faces]
    assert not ((triangles[:, :, 2].min(axis=1) < 15) & (triangles[:, :, 2].max(axis=1) > 15)).any()


def test_dimensions_include_only_existing_scalars_and_flatten_existing_arrays(document):
    part = document["parts"][0]
    part.update(aperture_hole_diameters_mm=[.1, .2],
                pole_root_fillet_radius_range_mm=[2, 4],
                magnetic_radial_profile_mm=[[0, 1, 6], [10, 1, 6]],
                installed=True, note="unchanged")
    model = part_model_from_document(document, part["key"])
    fields = {field.path: field for field in model.fields}
    for index, value in enumerate((.1, .2)):
        field = fields[("parts", part["key"], "aperture_hole_diameters_mm", index)]
        assert field.value == value and field.unit == "mm" and field.editable
    assert fields[("parts", part["key"], "magnetic_radial_profile_mm", 1, 2)].value == 6
    assert not any("installed" in path or "note" in path or "thickness_mm" in path for path in fields)


def test_actual_aperture_uses_declared_hole_not_larger_vacuum_constraint():
    document = _document("beam_blanker/NanoPulser.toml")
    mesh, = part_model_from_document(document, "nanopulser_aperture").meshes
    _assert_mesh(mesh)
    assert mesh.is_exact
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(.1)


@pytest.mark.parametrize("unit,diameters", [("mm", [.1, .2]), ("um", [100, 200])])
def test_existing_aperture_array_index_changes_the_actual_preview_hole(unit, diameters):
    document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    part = _part(document, "condenser_aperture_2")
    field = f"aperture_hole_diameters_{unit}"
    part[field] = diameters
    before = deepcopy(document)
    first = part_model_from_document(document, part["key"], aperture_index=0)
    second = part_model_from_document(document, part["key"], aperture_index=1)
    for model, radius, index in ((first, .05, 0), (second, .1, 1)):
        mesh, = model.meshes
        _assert_mesh(mesh)
        assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(radius)
        assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(part["plate_thickness_mm"])
        assert f"index={index}" in mesh.description
        assert not mesh.is_exact and "positions are not defined" in mesh.description
    assert document == before
    part[field][1] *= 1.5
    updated = part_model_from_document(document, part["key"], aperture_index=1)
    assert np.linalg.norm(updated.meshes[0].vertices[:, :2], axis=1).min() == pytest.approx(.15)
    with pytest.raises(ValueError, match="existing aperture"):
        part_model_from_document(document, part["key"], aperture_index=2)


def test_dimension_introspection_remains_available_when_draft_geometry_is_invalid(document):
    document["parts"][0]["mechanical_inner_diameter_mm"] = 20
    with pytest.raises(ValueError):
        part_model_from_document(document, "test_coil")
    specs = part_dimension_specs(document, "test_coil")
    assert next(field.value for field in specs if field.name == "mechanical_inner_diameter_mm") == 20
    split_document = _document("column/C3_ProbeCorrector_ImageCorrector.toml")
    parent = _part(split_document, "objective_lens")
    parent["upper_yoke_end_local_z_mm"] = parent["upper_yoke_start_local_z_mm"] - 1
    specs = part_dimension_specs(split_document, "objective_lens_excitation_coil")
    assert any(field.path == ("parts", "objective_lens", "upper_yoke_end_local_z_mm") and field.editable for field in specs)


def test_unsupported_mechanism_is_explicit_envelope_but_dimensions_remain_available():
    document = _document("beam_blanker/NanoPulser.toml")
    model = part_model_from_document(document, "nanopulser_deflector")
    assert not model.meshes[0].is_exact
    assert "Envelope" in model.meshes[0].description
    assert model.notes
    assert "plate_gap_mm" in {field.path[-1] for field in model.fields}


def test_zero_reference_plane_does_not_gain_synthetic_thickness(document):
    part = document["parts"][0]
    part.update(mechanical_profile="unknown_plane", length_mm=0, local_end_z_mm=10)
    mesh, = part_model_from_document(document, part["key"]).meshes
    _assert_mesh(mesh, closed=False)
    assert set(mesh.vertices[:, 2]) == {10}
    assert not mesh.is_exact


@pytest.mark.parametrize("value", [True, "12", float("nan"), float("inf"), -1, 8])
def test_invalid_material_diameter_is_rejected(document, value):
    document["parts"][0]["mechanical_outer_diameter_mm"] = value
    with pytest.raises(ValueError):
        part_model_from_document(document, "test_coil")


@pytest.mark.parametrize("count", [True, 2, 4.5, "16", 4097])
def test_invalid_angular_resolution_is_rejected(document, count):
    with pytest.raises(ValueError, match="angular_segments"):
        part_model_from_document(document, "test_coil", angular_segments=count)


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.toml")), ids=lambda path: str(path.relative_to(CONFIGS)))
def test_all_shipped_modules_have_finite_preview_without_changing_source(path):
    original = path.read_bytes()
    document = tomllib.loads(original.decode("utf-8-sig"))
    before = deepcopy(document)
    model = module_model_from_document(document, angular_segments=8)
    for mesh in model.meshes:
        _assert_mesh(mesh, closed="plane" not in mesh.description)
    assert len({(mesh.key, mesh.region) for mesh in model.meshes}) == len(model.meshes)
    assert document == before
    assert path.read_bytes() == original


def test_module_import_has_no_qt_or_gui_dependency():
    result = subprocess.run([sys.executable, "-c", "import sys; import temsim.part_model_3d; "
                             "assert not any(k.startswith(('PySide6', 'PyQt', 'temsim.gui')) for k in sys.modules)"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
