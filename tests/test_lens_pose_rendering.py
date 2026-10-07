"""Lens rigid placement must agree across editor, assembly and section views."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.part_model_3d import (
    module_model_from_document, part_dimension_specs, part_model_from_document,
)


def _row(key, profile, *, parent=None, centre=12., **extra):
    return dict(key=key, name=key, parent_key=parent, mechanical_profile=profile,
                local_start_z_mm=centre-2, local_center_z_mm=centre,
                local_end_z_mm=centre+2, length_mm=4.,
                mechanical_inner_diameter_mm=4., mechanical_outer_diameter_mm=8.,
                **extra)


def _assembly(rows, shift):
    parts = tuple(SimpleNamespace(key=row["key"], parent_key=row.get("parent_key"),
        module_key="module", data=row, start_z_mm=row["local_start_z_mm"]+shift,
        center_z_mm=row["local_center_z_mm"]+shift,
        end_z_mm=row["local_end_z_mm"]+shift, length_mm=row["length_mm"])
        for row in rows)
    return SimpleNamespace(parts=parts,
        modules=(SimpleNamespace(key="module", parts=parts),), vacuum_liner_segments=())


def _y_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0., s), (0., 1., 0.), (-s, 0., c)))


def test_lens_pose_moves_parent_children_and_edges_about_parent_centre_once():
    rows = [_row("lens", "magnetic_lens_assembly"),
            _row("coil", "magnetic_excitation_coil", parent="lens", centre=15.)]
    document = dict(parts=rows)
    baseline = part_model_from_document(document, "lens", angular_segments=8)
    rows[0].update(offset_x_mm=.7, offset_y_mm=-.4, rotation_y_mrad=30.)
    captured = deepcopy(document)
    posed = part_model_from_document(document, "lens", angular_segments=8)
    whole = module_model_from_document(document, angular_segments=8)
    rotation = _y_rotation(.030)
    centre = np.array((0., 0., 12.))
    transform = lambda values: (values-centre) @ rotation.T + centre + (.7, -.4, 0.)
    for original, actual, module_mesh in zip(baseline.meshes, posed.meshes, whole.meshes, strict=True):
        np.testing.assert_allclose(actual.vertices, transform(original.vertices), atol=1e-13)
        np.testing.assert_array_equal(actual.vertices, module_mesh.vertices)
        np.testing.assert_array_equal(actual.faces, original.faces)
        assert not actual.vertices.flags.writeable
        for edge, expected in zip(actual.edges, original.edges, strict=True):
            np.testing.assert_allclose(edge["vertices"], transform(expected["vertices"]), atol=1e-13)
    assert document == captured
    assembly = assembly_model_from_assembly(_assembly(rows, 987.), angular_segments=8)
    assert not assembly.errors
    # The optical-parent envelope is replaced by its coil in the whole column.
    coil, = assembly.meshes
    np.testing.assert_allclose(coil.vertices, posed.meshes[1].vertices + (0., 0., 987.))


def test_legacy_rigid_lens_pose_is_not_applied_twice_and_canonical_values_override():
    row = _row("lens", "magnetic_lens_assembly")
    before = part_model_from_document(dict(parts=[row]), "lens", angular_segments=8).meshes[0]
    row["model_3d"] = dict(schema_version=1, base=dict(kind="existing"), features=[],
        transform=dict(offset_mm=[1., 2., 3.], rotation_deg=[0., 90., 0.], scale_xy=[1., 1.]))
    row.update(offset_x_mm=4., rotation_y_mrad=0.)
    actual = part_model_from_document(dict(parts=[row]), "lens", angular_segments=8).meshes[0]
    np.testing.assert_allclose(actual.vertices, before.vertices + (4., 2., 3.), atol=1e-13)


def test_shared_cartridge_uses_declared_parent_once():
    rows = [_row("c1", "magnetic_lens_assembly", offset_x_mm=1.),
            _row("c2", "magnetic_lens_assembly", offset_x_mm=8.),
            _row("shared", "magnetic_excitation_coil", parent="c1", magnetic_lens_keys=["c1", "c2"])]
    model = assembly_model_from_assembly(_assembly(rows, 100.), angular_segments=8)
    assert not model.errors
    mesh, = model.meshes
    assert mesh.key == "shared"
    assert np.mean(mesh.vertices[:, 0]) == pytest.approx(1.)
    assert np.mean(mesh.vertices[:, 2]) == pytest.approx(112.)


def test_pose_dimension_defaults_use_mrad_and_hide_old_rigid_cad_controls():
    row = _row("lens", "magnetic_lens_assembly")
    row["model_3d"] = dict(schema_version=1, base=dict(kind="existing"), features=[],
        transform=dict(offset_mm=[1., 2., 3.], rotation_deg=[0., 1., 0.], scale_xy=[1., 1.]))
    fields = part_dimension_specs(dict(parts=[row]), "lens")
    by_name = {item.path[2]: item for item in fields if len(item.path) == 3}
    assert by_name["offset_x_mm"].value == 1.
    assert by_name["rotation_x_mrad"].value == 0.
    assert by_name["rotation_y_mrad"].value == pytest.approx(np.pi / 180 * 1000)
    assert by_name["rotation_y_mrad"].unit == "mrad"
    assert not any(item.path[2:5] in (("model_3d", "transform", "offset_mm"),
        ("model_3d", "transform", "rotation_deg")) for item in fields)


def test_section_projection_uses_same_global_pose_and_leaves_y_out_of_plane():
    from temsim.gui.diagnostic_tabs import PhysicalLayoutView
    row = _row("lens", "magnetic_lens_assembly", offset_x_mm=.4, offset_y_mm=6., rotation_y_mrad=20.)
    part = _assembly([row], 800.).parts[0]
    view = SimpleNamespace(_part_by_key={"lens": part})
    view._lens_pose_global_transform = lambda key: PhysicalLayoutView._lens_pose_global_transform(view, key)
    z, x = PhysicalLayoutView._posed_section_point(view, "lens", 814., 1.)
    expected = _y_rotation(.020) @ np.array((1., 0., 2.)) + (.4, 6., 812.)
    assert z == pytest.approx(expected[2])
    assert x == pytest.approx(expected[0])


def test_real_objective_moves_its_magnetic_hardware_not_independent_devices():
    from temsim.lens_pose import effective_part_transform_mm
    root = Path(__file__).resolve().parents[1]
    document = tomllib.loads((root / "configs/instruments/column/C3_ProbeCorrector.toml").read_text())
    rows = {row["key"]: row for row in document["parts"]}
    rows["objective_lens"].update(offset_x_mm=.2, rotation_y_mrad=10.)
    for key in ("sample", "sample_stage", "condenser_stigmator", "ac_deflector",
                "descan_deflector", "objective_stigmator", "image_diffraction_deflector"):
        rotation, offset = effective_part_transform_mm(rows[key], rows)
        np.testing.assert_array_equal(rotation, np.eye(3))
        np.testing.assert_array_equal(offset, np.zeros(3))
        fields = part_dimension_specs(document, key)
        assert not any(item.path[1] == "objective_lens" and item.path[2] == "rotation_y_mrad" for item in fields)
    for key in ("objective_lens_excitation_coil", "objective_upper_pole", "mini_condenser"):
        rotation, _ = effective_part_transform_mm(rows[key], rows)
        np.testing.assert_allclose(rotation, _y_rotation(.01))
    fields = part_dimension_specs(document, "objective_lens_excitation_coil")
    field = next(item for item in fields if item.path == ("parts", "objective_lens", "rotation_y_mrad"))
    assert field.label.startswith("Parent lens:") and field.value == 10.
