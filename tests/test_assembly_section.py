"""Small CPU-only geometric sections with independently known boundaries."""

from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.assembly_model_3d import AssemblyModel3D
from temsim.assembly_section import assembly_section_from_model, mesh_section
from temsim.part_model_3d import TriangleMesh, revolve_section


def _box(half=(2., 1., 3.), *, shift=(0., 0., 0.), key="box"):
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float) * half + shift
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=int)
    return TriangleMesh(vertices, faces, key)


def _bounds(section):
    points = section.segments_mm.reshape(-1, 2)
    return points.min(axis=0), points.max(axis=0)


def _length(section):
    return np.linalg.norm(section.segments_mm[:, 1] - section.segments_mm[:, 0], axis=1).sum()


@pytest.mark.parametrize("angle, radial_extent", [(0., 2.), (90., 1.), (45., math.sqrt(2.))])
def test_rotating_plane_cuts_asymmetric_body_instead_of_projecting_it(angle, radial_extent):
    result = mesh_section(_box(), angle)
    np.testing.assert_allclose(_bounds(result), [[-3., -radial_extent], [3., radial_extent]], atol=1e-12)
    assert _length(result) == pytest.approx(12. + 4 * radial_extent)
    # Every segment belongs to the boundary of the actual section rectangle.
    midpoint = result.segments_mm.mean(axis=1)
    assert np.all(np.isclose(np.abs(midpoint[:, 0]), 3.)
                  | np.isclose(np.abs(midpoint[:, 1]), radial_extent))


def test_body_outside_plane_is_invisible_until_cutting_plane_reaches_it():
    mesh = _box(half=(1., 1., 2.), shift=(0., 3., 20.))
    assert mesh_section(mesh, 0.).segments_mm.shape == (0, 2, 2)
    np.testing.assert_allclose(_bounds(mesh_section(mesh, 90.)), [[18., 2.], [22., 4.]])


@pytest.mark.parametrize("angle", [0., 45., 90., 180., 270.])
def test_axial_bore_remains_open_with_two_material_section_boundaries(angle):
    mesh = revolve_section([(10., 2.), (14., 2.), (14., 4.), (10., 4.)],
                           key="tube", angular_segments=8)
    result = mesh_section(mesh, angle)
    # Two 4 x 2 material rectangles, with no edge bridging the central bore.
    assert _length(result) == pytest.approx(24.)
    endpoints = result.segments_mm[..., 1]
    assert np.all(np.abs(endpoints) >= 2. - 1e-12)
    assert np.all(np.sign(endpoints[:, 0]) == np.sign(endpoints[:, 1]))
    np.testing.assert_allclose(_bounds(result), [[10., -4.], [14., 4.]], atol=1e-12)


def test_tilted_decentered_solid_uses_placed_vertices():
    theta = .2
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    mesh = _box()
    shifted = replace(mesh, vertices=mesh.vertices @ rotation.T + (7., 0., 100.))
    result = mesh_section(shifted, 0.)
    points = result.segments_mm.reshape(-1, 2)
    world = np.column_stack((points[:, 1], np.zeros(len(points)), points[:, 0]))
    local = (world - (7., 0., 100.)) @ rotation
    assert np.allclose(local[:, 1], 0.)
    assert np.all(np.abs(local[:, 0]) <= 2. + 1e-12)
    assert np.all(np.abs(local[:, 2]) <= 3. + 1e-12)
    assert np.all(np.isclose(np.abs(local[:, 0]), 2.) | np.isclose(np.abs(local[:, 2]), 3.))
    assert _length(result) == pytest.approx(20.)


def test_out_of_plane_tilt_changes_the_section_instead_of_projecting_the_local_xz_face():
    cosine = sine = math.sqrt(.5)
    rotation = np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    mesh = _box(half=(2., 1., 4.))
    mesh = replace(mesh, vertices=mesh.vertices @ rotation.T)
    result = mesh_section(mesh, 0.)
    # Cutting Y = 0 meets the side walls at Z = +/-sqrt(2); the projected
    # silhouette would instead extend to +/-5/sqrt(2).
    np.testing.assert_allclose(_bounds(result), [[-math.sqrt(2.), -2.], [math.sqrt(2.), 2.]], atol=1e-12)
    assert _length(result) == pytest.approx(8. + 4 * math.sqrt(2.))


@pytest.mark.parametrize("duplicated_vertices", [False, True])
def test_coplanar_face_keeps_perimeter_without_triangulation_diagonal(duplicated_vertices):
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [2., 0., 4.], [0., 0., 4.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if duplicated_vertices:
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(6).reshape(-1, 3)
    result = mesh_section(TriangleMesh(vertices, faces, "sheet"), 0.)
    assert len(result.segments_mm) == 4
    assert _length(result) == pytest.approx(12.)
    assert np.all(np.any(np.isclose(result.segments_mm[:, 0], result.segments_mm[:, 1]), axis=1))


def test_plane_on_box_face_has_no_duplicate_edges_or_diagonal():
    result = mesh_section(_box(shift=(0., 1., 0.)), 0.)
    assert len(result.segments_mm) == 4
    assert _length(result) == pytest.approx(20.)


@pytest.mark.parametrize("vertices, expected", [
    ([[0., 0., 0.], [2., 0., 4.], [2., 1., 0.]], [[[0., 0.], [4., 2.]]]),
    ([[0., 0., 0.], [2., 1., 2.], [2., -1., 4.]], [[[0., 0.], [3., 2.]]]),
    ([[0., 0., 0.], [2., 1., 2.], [2., 1., 4.]], []),
])
def test_on_plane_edge_vertex_and_tangent_point(vertices, expected):
    result = mesh_section(TriangleMesh(np.asarray(vertices), np.array([[0, 1, 2]]), "triangle"), 0.)
    np.testing.assert_allclose(result.segments_mm, np.asarray(expected).reshape(-1, 2, 2))


def test_coplanar_diagonal_is_removed_across_chunk_boundaries(monkeypatch):
    monkeypatch.setattr("temsim.assembly_section._FACE_CHUNK_SIZE", 1)
    result = mesh_section(_box(shift=(0., 1., 0.)), 0.)
    assert len(result.segments_mm) == 4
    assert _length(result) == pytest.approx(20.)


def test_capture_metadata_is_preserved_and_cached_segments_cannot_be_mutated():
    mesh = replace(_box(), region="coil", color=(.8, .4, .1, .7),
                   material_class="copper", is_exact=False, description="Approximate winding envelope")
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    result = mesh_section(mesh, 20.)
    assert (result.key, result.region, result.color, result.material_class,
            result.is_exact, result.description) == (
        mesh.key, mesh.region, mesh.color, mesh.material_class, mesh.is_exact, mesh.description)
    with pytest.raises(ValueError):
        result.segments_mm.setflags(write=True)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_partial_assembly_preserves_errors_skips_guides_and_does_not_call_a_miss_an_omission():
    good = _box(key="good")
    invalid = replace(_box(key="bad"), faces=np.array([[0, 1, 99]]))
    guide = replace(_box(key="channel"), wireframe=True)
    outside = _box(key="outside", shift=(0., 10., 0.))
    source = AssemblyModel3D((good, invalid, guide, outside), ("Original approximation",),
                             ("unavailable",), ("Prior geometry error",))
    result = assembly_section_from_model(source, 360.)
    assert result.angle_deg == 0.
    assert [mesh.key for mesh in result.meshes] == ["good"]
    assert result.omitted_keys == ("unavailable", "bad", "channel")
    assert result.errors[0] == "Prior geometry error"
    assert "bad: Mesh face index" in result.errors[1]
    assert "Original approximation" in result.notes
    assert not len(mesh_section(guide, 0.).segments_mm)


@pytest.mark.parametrize("angle,tolerance", [
    (float("nan"), 1e-8), (float("inf"), 1e-8), (True, 1e-8),
    (0., 0.), (0., -1.), (0., float("nan")), (0., True),
])
def test_invalid_plane_parameters_are_rejected_even_for_empty_assembly(angle, tolerance):
    with pytest.raises(ValueError):
        assembly_section_from_model(AssemblyModel3D(()), angle, tolerance_mm=tolerance)


def test_angles_are_periodic_and_negative_angles_are_supported():
    mesh = _box()
    np.testing.assert_allclose(mesh_section(mesh, -45.).segments_mm,
                               mesh_section(mesh, 315.).segments_mm, atol=1e-12)
