"""Control channels keep their metadata, without becoming mechanical solids."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.component_representation import IMAGE_CONTROL_CHANNEL_KEYS
from temsim.part_model_3d import module_model_from_document, part_model_from_document


CHANNEL_KEYS = (
    "probe_dph2_deflector", "probe_qph2_quadrupole", "probe_dp22_deflector",
    "probe_hpc_hexapole", "probe_qpc_quadrupole", "probe_dp21_deflector",
    "probe_dph1_deflector", "probe_qph1_quadrupole", "probe_hpol_hexapole",
    "probe_qpol_quadrupole", "probe_dp11_deflector", "probe_dp12_scan_deflector",
)


def _row(key, **changes):
    row = dict(key=key, name=key, local_start_z_mm=10., local_center_z_mm=12.,
               local_end_z_mm=14., length_mm=4., mechanical_profile="unknown_multipole",
               mechanical_inner_diameter_mm=20., mechanical_outer_diameter_mm=75.,
               effective_length_mm=2.)
    row.update(changes)
    return row


def _assembly(rows):
    parts = tuple(SimpleNamespace(key=row["key"], module_key="corrector", parent_key=row.get("parent_key"),
                                 start_z_mm=row["local_start_z_mm"] + 100.,
                                 center_z_mm=row["local_center_z_mm"] + 100.,
                                 end_z_mm=row["local_end_z_mm"] + 100., data=row)
                  for row in rows)
    return SimpleNamespace(parts=parts, modules=(), vacuum_liner_segments=())


@pytest.mark.parametrize("key", (*CHANNEL_KEYS, *sorted(IMAGE_CONTROL_CHANNEL_KEYS)))
def test_legacy_nonzero_channel_envelopes_remain_inspectable_without_solids(key):
    document = {"parts": [_row(key)]}
    before = deepcopy(document)
    part = part_model_from_document(document, key, angular_segments=8)
    whole = assembly_model_from_assembly(_assembly(document["parts"]), angular_segments=8)
    assert part.part_key == key and part.name == key
    assert not part.meshes and not whole.meshes and not whole.errors
    assert whole.omitted_keys == (key,)
    assert part.notes and any(key in note for note in whole.notes)
    values = {field.name: field.value for field in part.fields}
    assert values["mechanical_outer_diameter_mm"] == 75.
    assert values["effective_length_mm"] == 2.
    assert all("not a dimension" in field.reason for field in part.fields)
    assert all(field.meaning.category_label == "Channel / reference metadata" for field in part.fields)
    assert document == before


@pytest.mark.parametrize("changes", [
    {"layout_role": "control_channel"},
    {"layout_role": "virtual_reference"},
    {"mechanical_profile": "virtual_plane"},
    {"mechanical_profile": "virtual_layout"},
])
def test_non_material_classification_precedes_explicit_solid_generation(changes):
    row = _row("imported_channel", **changes)
    row["model_3d"] = dict(schema_version=1,
                           base=dict(kind="box", width_mm=6., height_mm=4., length_mm=2.), features=[])
    document = {"parts": [row]}
    assert not part_model_from_document(document, row["key"], angular_segments=8).meshes
    assert not module_model_from_document(document, angular_segments=8).meshes
    whole = assembly_model_from_assembly(_assembly([row]), angular_segments=8)
    assert not whole.meshes and not whole.errors and whole.omitted_keys == (row["key"],)


@pytest.mark.parametrize("prefix,child_key", (
    ("probe", "probe_qph1_quadrupole"), ("image", "image_dph1_deflector"),
))
def test_virtual_children_do_not_hide_their_material_parent(prefix, child_key):
    parent = _row(f"{prefix}_hp1_hexapole", mechanical_profile="magnetic_lens_assembly")
    child = _row(child_key, parent_key=parent["key"])
    rows = [parent, child, _row(f"{prefix}_hp2_hexapole")]
    whole = assembly_model_from_assembly(_assembly(rows), angular_segments=8)
    module = module_model_from_document({"parts": rows}, angular_segments=8)
    assert not whole.errors
    for model in (whole, module):
        assert {mesh.key for mesh in model.meshes} == {f"{prefix}_hp1_hexapole", f"{prefix}_hp2_hexapole"}


def test_real_zero_thickness_detector_surface_is_still_available_in_part_preview():
    row = _row("detector", mechanical_profile="detector_surface", length_mm=0., local_end_z_mm=10.)
    mesh, = part_model_from_document({"parts": [row]}, row["key"], angular_segments=8).meshes
    assert set(mesh.vertices[:, 2]) == {10.}
    assert len(mesh.faces) > 0
