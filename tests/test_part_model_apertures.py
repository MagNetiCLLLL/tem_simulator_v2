"""Aperture plate thickness and working openings are not carrier envelopes."""

from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.part_model_3d import (
    module_model_from_document, part_dimension_specs, part_model_from_document,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "instruments"
COLUMN = "column/C3_ProbeCorrector_ImageCorrector.toml"
C2 = "condenser_aperture_2"


def _read(path):
    return tomllib.loads((CONFIGS / path).read_text(encoding="utf-8-sig"))


def _part(document, key):
    return next(part for part in document["parts"] if part["key"] == key)


def _strip_cases():
    return [(path.relative_to(CONFIGS).as_posix(), part["key"])
            for path in sorted(CONFIGS.rglob("*.toml"))
            for part in tomllib.loads(path.read_text(encoding="utf-8-sig")).get("parts", ())
            if part.get("aperture_plate_form") == "perforated_strip"]


STRIPS = _strip_cases()


def _meshes(model, key):
    return tuple(mesh for mesh in model.meshes if mesh.key == key)


def _vertices(model, key):
    meshes = _meshes(model, key)
    assert meshes, f"No aperture plate was generated for {key}"
    return np.concatenate([mesh.vertices for mesh in meshes])


def _sorted_vertices(vertices):
    return vertices[np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))]


def _cap_surfaces(model, key):
    found = []
    for mesh in _meshes(model, key):
        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        assert np.isfinite(lengths).all() and (lengths > 0).all()
        axial = np.abs(normals[:, 2]) / lengths > 1 - 1e-8
        for group in set(mesh.face_groups[axial]):
            found.append(mesh.surfaces[group])
    assert found, "The plate must expose its real upstream/downstream faces"
    return found


def _working_wall_radii(model, key):
    runtime_path = ("runtime", key, "radius_mm")
    radii = []
    for mesh in _meshes(model, key):
        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        for index, group in enumerate(mesh.face_groups):
            if runtime_path in mesh.surfaces[group]["parameter_paths"] and abs(normals[index, 2]) / lengths[index] < 1e-8:
                radii.extend(np.linalg.norm(triangles[index, :, :2], axis=1))
    assert radii, "The working aperture wall must carry its runtime radius provenance"
    return np.asarray(radii)


def _assert_schematic(model, key):
    assert all(not mesh.is_exact for mesh in _meshes(model, key))
    text = " ".join(model.notes).lower()
    assert "schematic" in text
    assert "transverse" in text
    assert any(word in text for word in ("unspecified", "undefined", "not defined", "not specified"))


@pytest.mark.parametrize("path,key", STRIPS, ids=[f"{path}:{key}" for path, key in STRIPS])
def test_every_shipped_perforated_strip_uses_real_plate_thickness_and_optical_plane(path, key):
    source = (CONFIGS / path).read_bytes()
    document = _read(path)
    before = deepcopy(document)
    part = _part(document, key)
    runtime = {key: {"radius_mm": .05}}
    model = part_model_from_document(document, key, include_children=False, runtime_values=runtime)
    vertices = _vertices(model, key)
    thickness = part.get("plate_thickness_mm", part.get("active_length_mm"))
    center = part.get("optical_reference_local_z_mm", part["local_center_z_mm"])
    assert np.ptp(vertices[:, 2]) == pytest.approx(thickness)
    assert (vertices[:, 2].min() + vertices[:, 2].max()) / 2 == pytest.approx(center)
    assert np.ptp(vertices[:, 2]) != pytest.approx(part["length_mm"])
    _assert_schematic(model, key)
    assert document == before
    assert runtime == {key: {"radius_mm": .05}}
    assert (CONFIGS / path).read_bytes() == source


def test_c2_plate_thickness_edit_changes_solid_without_repurposing_length_or_outer_diameter():
    document = _read(COLUMN)
    part = _part(document, C2)
    assert part["plate_thickness_mm"] == .2
    assert part["length_mm"] == 20 and part["mechanical_outer_diameter_mm"] == 80
    runtime = {C2: {"radius_mm": .05}}
    original = part_model_from_document(document, C2, runtime_values=runtime)
    part["plate_thickness_mm"] = .35
    changed = part_model_from_document(document, C2, runtime_values=runtime)
    assert np.ptp(_vertices(original, C2)[:, 2]) == pytest.approx(.2)
    assert np.ptp(_vertices(changed, C2)[:, 2]) == pytest.approx(.35)
    assert part["length_mm"] == 20 and part["mechanical_outer_diameter_mm"] == 80
    vertices = _vertices(changed, C2)
    # The 80-mm carrier OD must not become a full plate's transverse size.
    assert np.linalg.norm(vertices[:, :2], axis=1).max() < 40
    for surface in _cap_surfaces(changed, C2):
        paths = surface["parameter_paths"]
        assert ("parts", C2, "plate_thickness_mm") in paths
        assert ("parts", C2, "length_mm") not in paths
        assert ("parts", C2, "mechanical_outer_diameter_mm") not in paths


def test_changing_c2_assembly_envelope_length_does_not_change_plate_thickness_or_mesh():
    document = _read(COLUMN)
    part = _part(document, C2)
    runtime = {C2: {"radius_mm": .05}}
    first = part_model_from_document(document, C2, runtime_values=runtime)
    center = part["local_center_z_mm"]
    part.update(length_mm=25, local_start_z_mm=center - 12.5, local_end_z_mm=center + 12.5)
    second = part_model_from_document(document, C2, runtime_values=runtime)
    assert np.ptp(_vertices(second, C2)[:, 2]) == pytest.approx(.2)
    assert part["plate_thickness_mm"] == .2 and part["mechanical_outer_diameter_mm"] == 80
    assert _sorted_vertices(_vertices(first, C2)) == pytest.approx(_sorted_vertices(_vertices(second, C2)))


def test_runtime_opening_is_real_hole_and_readonly_diameter_with_runtime_provenance():
    document = _read(COLUMN)
    part = _part(document, C2)
    before = deepcopy(part)
    runtime = {C2: {"radius_mm": .05}}
    model = part_model_from_document(document, C2, runtime_values=runtime)
    assert _working_wall_radii(model, C2) == pytest.approx(.05)
    assert part["mechanical_bore_diameter_mm"] == 4
    runtime_path = ("runtime", C2, "radius_mm")
    fields = part_dimension_specs(document, C2, runtime_values=runtime)
    spec, = [field for field in fields if field.path == runtime_path]
    assert not spec.editable and spec.value == pytest.approx(.1) and spec.unit == "mm"
    assert "working opening diameter" in spec.label.lower()
    assert any(field.path == runtime_path and field.value == pytest.approx(.1) for field in model.fields)
    for mesh in _meshes(model, C2):
        for surface in mesh.surfaces.values():
            if runtime_path in surface["parameter_paths"]:
                assert ("parts", C2, "mechanical_bore_diameter_mm") not in surface["parameter_paths"]
    assert part == before


def test_runtime_radius_change_only_changes_working_opening_not_authoritative_plate_dimensions():
    document = _read(COLUMN)
    before = deepcopy(document)
    first = part_model_from_document(document, C2, runtime_values={C2: {"radius_mm": .05}})
    second = part_model_from_document(document, C2, runtime_values={C2: {"radius_mm": .1}})
    assert _working_wall_radii(first, C2) == pytest.approx(.05)
    assert _working_wall_radii(second, C2) == pytest.approx(.1)
    assert np.ptp(_vertices(first, C2)[:, 2]) == pytest.approx(np.ptp(_vertices(second, C2)[:, 2]))
    assert document == before


def test_missing_runtime_opening_is_not_filled_in_from_carrier_bore():
    document = _read(COLUMN)
    model = part_model_from_document(document, C2)
    notes = " ".join(model.notes).lower()
    assert "working" in notes
    assert any(word in notes for word in ("undefined", "unspecified", "not defined", "not specified"))
    assert not any(field.path == ("runtime", C2, "radius_mm") for field in model.fields)
    for mesh in _meshes(model, C2):
        for surface in mesh.surfaces.values():
            assert ("parts", C2, "mechanical_bore_diameter_mm") not in surface["parameter_paths"]
    _assert_schematic(model, C2)


def test_plate_center_uses_optical_reference_then_local_center_fallback():
    document = _read(COLUMN)
    part = _part(document, C2)
    part["optical_reference_local_z_mm"] = part["local_center_z_mm"] + 1.25
    referenced = _vertices(part_model_from_document(document, C2), C2)
    assert (referenced[:, 2].min() + referenced[:, 2].max()) / 2 == pytest.approx(part["optical_reference_local_z_mm"])
    del part["optical_reference_local_z_mm"]
    fallback = _vertices(part_model_from_document(document, C2), C2)
    assert (fallback[:, 2].min() + fallback[:, 2].max()) / 2 == pytest.approx(part["local_center_z_mm"])


@pytest.mark.parametrize("path,key", [(path, key) for path, key in STRIPS if path.startswith("gun/")])
def test_gun_strip_active_length_fallback_has_correct_surface_parameter_source(path, key):
    document = _read(path)
    part = _part(document, key)
    assert "plate_thickness_mm" not in part
    part["active_length_mm"] = .3
    model = part_model_from_document(document, key, runtime_values={key: {"radius_mm": .05}})
    assert np.ptp(_vertices(model, key)[:, 2]) == pytest.approx(.3)
    for surface in _cap_surfaces(model, key):
        assert ("parts", key, "active_length_mm") in surface["parameter_paths"]
        assert ("parts", key, "length_mm") not in surface["parameter_paths"]


def test_module_preview_uses_same_runtime_hole_and_readonly_field_as_selected_part():
    document = _read(COLUMN)
    runtime = {C2: {"radius_mm": .05}, "objective_aperture": {"radius_mm": .125}}
    module = module_model_from_document(document, angular_segments=8, runtime_values=runtime)
    for key, radius in ((C2, .05), ("objective_aperture", .125)):
        selected = part_model_from_document(document, key, angular_segments=8, runtime_values=runtime)
        assert _working_wall_radii(module, key) == pytest.approx(radius)
        assert _sorted_vertices(_vertices(module, key)) == pytest.approx(_sorted_vertices(_vertices(selected, key)))
        specs = [field for field in module.fields if field.path == ("runtime", key, "radius_mm")]
        assert len(specs) == 1 and not specs[0].editable and specs[0].value == pytest.approx(2 * radius)


def test_nanopulser_circular_aperture_keeps_its_declared_thickness_and_fixed_bore():
    document = _read("beam_blanker/NanoPulser.toml")
    key = "nanopulser_aperture"
    part = _part(document, key)
    model = part_model_from_document(document, key)
    vertices = _vertices(model, key)
    assert np.ptp(vertices[:, 2]) == pytest.approx(part["length_mm"])
    assert np.linalg.norm(vertices[:, :2], axis=1).min() == pytest.approx(part["bore_diameter_mm"] / 2)
    assert all(mesh.is_exact for mesh in _meshes(model, key))


@pytest.mark.parametrize("path", ["project_and_recording_system/EnergyFilter.toml",
                                  "project_and_recording_system/NoEnergyFilter.toml"])
def test_projection_dpa_zero_thickness_reference_plane_is_preserved(path):
    document = _read(path)
    key = "projection_chamber_dpa_aperture"
    part = _part(document, key)
    assert part["length_mm"] == 0 and part["mechanical_profile"] == "fixed_differential_pumping_aperture"
    model = part_model_from_document(document, key)
    vertices = _vertices(model, key)
    assert np.ptp(vertices[:, 2]) == 0
    assert vertices[:, 2] == pytest.approx(part["local_center_z_mm"])
    assert np.linalg.norm(vertices[:, :2], axis=1).min() == pytest.approx(part["mechanical_bore_diameter_mm"] / 2)
