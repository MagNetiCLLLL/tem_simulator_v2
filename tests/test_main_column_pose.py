"""Main-column pose ownership is shared, without invented mechanical hosts."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim.lens_pose import (
    PHYSICAL_POSE_FIELDS, column_pose_kind, effective_part_transform_mm,
    lens_pose_registration, physical_pose_values, supports_physical_column_pose,
    validate_physical_lens_pose,
)


def _document():
    path = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector.toml"
    return tomllib.loads(path.read_text())


@pytest.mark.parametrize("key", ["condenser_stigmator", "objective_stigmator", "ac_deflector",
    "condenser_deflector", "image_diffraction_deflector", "condenser_aperture_2",
    "objective_aperture", "probe_hp1_hexapole", "probe_qph1_quadrupole"])
def test_main_column_devices_have_independent_pose_despite_packing_parent(key):
    rows = {row["key"]: row for row in _document()["parts"]}
    rows["objective_lens"].update(offset_x_mm=3., rotation_y_mrad=80.)
    row = rows[key]
    assert supports_physical_column_pose(row)
    row.update(offset_x_mm=.25, rotation_x_mrad=2.)
    rotation, shift = effective_part_transform_mm(row, rows)
    centre = np.array((0., 0., row["local_center_z_mm"]))
    np.testing.assert_allclose(rotation @ centre + shift, centre + [.25, 0., 0.], atol=1e-12)
    assert rotation[2, 1] == pytest.approx(np.sin(.002))
    assert rotation[0, 2] == 0.


@pytest.mark.parametrize("key", ["feg_tip", "feg_extractor", "feg_accelerator", "feg_c1_aperture",
    "thermionic_gun_lens", "energy_filter_multipole_01", "haadf_detector", "camera",
    "probe_dp12_scan_deflector", "sample"])
def test_pose_scope_does_not_expand_to_gun_receivers_filter_or_virtual_planes(key):
    row = dict(key=key, rotation_y_mrad=1.)
    assert not supports_physical_column_pose(row)
    with pytest.raises(ValueError, match="post-gun"):
        validate_physical_lens_pose(row)


def test_shared_descan_has_exactly_one_host_pose_and_no_editable_copy():
    from temsim.component_representation import shared_deflector_field_owner
    rows = {row["key"]: row for row in _document()["parts"]}
    host, channel = rows["image_diffraction_deflector"], rows["descan_deflector"]
    host.update(offset_y_mm=.1, rotation_y_mrad=3.)
    assert not supports_physical_column_pose(channel)
    for name in PHYSICAL_POSE_FIELDS:
        assert shared_deflector_field_owner(channel["key"], name) == host["key"]
    for actual, expected in zip(effective_part_transform_mm(channel, rows),
                                effective_part_transform_mm(host, rows)):
        np.testing.assert_allclose(actual, expected)
    parts = tuple(SimpleNamespace(key=key, data=row, module_key="column",
                  center_z_mm=row["local_center_z_mm"]+450.) for key, row in rows.items())
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=parts))
    assert lens_pose_registration(state, channel["key"]) == lens_pose_registration(state, host["key"])


def test_channel_pose_ui_is_field_coordinates_without_new_material_claim():
    from temsim.part_model_3d import part_dimension_specs
    from temsim.parameter_semantics import describe_parameter
    document = _document()
    row = next(row for row in document["parts"] if row["key"] == "probe_qph1_quadrupole")
    assert column_pose_kind(row) == "field"
    path = ("parts", row["key"], "rotation_y_mrad")
    field = next(item for item in part_dimension_specs(document, row["key"]) if item.path == path)
    assert field.editable and field.label.startswith("Field ")
    assert "without creating a material body" in describe_parameter(row, path).description
    assert physical_pose_values(row)["rotation_y_mrad"] == 0.


def test_aperture_impact_does_not_claim_a_magnetic_field():
    from temsim.parameter_impact import describe_parameter_impact
    row = next(row for row in _document()["parts"] if row["key"] == "objective_aperture")
    impact = describe_parameter_impact(row, ("parts", row["key"], "rotation_y_mrad"))
    assert "beam_clearance" in impact.effects
    assert "magnetic_field" not in impact.effects


def test_existing_component_editor_stages_new_poses_and_preserves_undo():
    from temsim.part_model_document import PartModelDocument
    from temsim import module_manifest
    path = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector.toml"
    draft = PartModelDocument(path)
    for key in ("objective_aperture", "ac_deflector", "probe_qph1_quadrupole"):
        draft.set_dimension(("parts", key, "rotation_y_mrad"), 1.25)
        assert draft.part(key)["rotation_y_mrad"] == 1.25
    module_manifest.validate_document(draft.document)
    draft.undo()
    assert physical_pose_values(draft.part("probe_qph1_quadrupole"))["rotation_y_mrad"] == 0.
    assert draft.part("ac_deflector")["rotation_y_mrad"] == 1.25
    with pytest.raises(ValueError, match="physical host"):
        draft.set_physical_pose("descan_deflector", {"rotation_y_mrad": 2.})


def test_aperture_material_mesh_uses_the_same_rigid_transform():
    from temsim.part_model_3d import part_model_from_document
    document = _document()
    rows = {row["key"]: row for row in document["parts"]}
    key = "objective_aperture"
    before = part_model_from_document(document, key, include_children=False)
    rows[key].update(rotation_y_mrad=3., offset_x_mm=.012)
    after = part_model_from_document(document, key, include_children=False)
    rotation, translation = effective_part_transform_mm(rows[key], rows)
    assert before.meshes and len(before.meshes) == len(after.meshes)
    for original, placed in zip(before.meshes, after.meshes, strict=True):
        np.testing.assert_allclose(placed.vertices, original.vertices @ rotation.T + translation, atol=1e-12)
