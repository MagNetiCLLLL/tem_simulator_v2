"""Control-plane metadata cannot create material, collisions or beam stops."""
from types import SimpleNamespace

import pytest

from temsim.column.module_assembly import _module_vacuum_segments
from temsim.module_manifest import _validate_column_mechanical_overlaps
from temsim.part_geometry import geometry_from_part
from temsim.part_materials import (
    is_magnetostatic_body,
    material_application_scope,
    material_catalog,
    part_material_updates,
    validated_material_regions,
)


@pytest.mark.parametrize("channel", (
    {"key": "probe_qph1_quadrupole"},  # Legacy saved rows have no role.
    {"key": "probe_dp12_scan_deflector"},
    {"key": "image_ish_deflector"},
    {"key": "image_dsh_deflector"},
    {"key": "image_hpol_hexapole"},
    {"key": "test_control", "layout_role": "control_channel"},
    {"key": "test_reference", "layout_role": "virtual_reference"},
))
def test_channel_bore_does_not_replace_real_vacuum_wall(channel):
    module = SimpleNamespace(
        key="column", entrance_z_mm=0.0, exit_z_mm=20.0,
        geometry={
            "vacuum_drift_inner_diameter_mm": 12.0,
            "c1_c2_objective_vacuum_tube_start_z_mm": 5.0,
            "c1_c2_objective_vacuum_tube_end_z_mm": 15.0,
            "c1_c2_objective_vacuum_tube_inner_diameter_mm": 5.76,
            "c1_c2_objective_vacuum_tube_outer_diameter_mm": 19.2,
        },
    )
    wall = SimpleNamespace(
        module_key="column", key="real_wall", name="Real wall",
        start_z_mm=0.0, end_z_mm=20.0, length_mm=20.0,
        data={"key": "real_wall", "vacuum_inner_diameter_mm": 8.0},
    )
    control = SimpleNamespace(
        module_key="column", key=channel["key"], name="Channel",
        start_z_mm=6.0, end_z_mm=14.0, length_mm=8.0,
        data={**channel, "vacuum_inner_diameter_mm": 0.5},
    )
    expected = _module_vacuum_segments(module, 0.0, (wall,))
    actual = _module_vacuum_segments(module, 0.0, (wall, control))
    assert actual == expected
    assert [(s.start_z_mm, s.end_z_mm, s.inner_diameter_mm) for s in actual] == [
        (0.0, 5.0, 8.0), (5.0, 15.0, 5.76), (15.0, 20.0, 8.0),
    ]


def test_channel_envelope_does_not_collide_but_physical_envelopes_still_do():
    body = dict(key="body", length_mm=10.0, local_start_z_mm=0.0,
                local_center_z_mm=5.0, local_end_z_mm=10.0)
    channel = {**body, "key": "probe_hpc_hexapole"}
    _validate_column_mechanical_overlaps((body, channel))
    with pytest.raises(ValueError, match="Undeclared mechanical overlap"):
        _validate_column_mechanical_overlaps((body, {**channel, "key": "second_body"}))


@pytest.mark.parametrize("key", ("probe_qpc_quadrupole", "image_hpol_hexapole", "image_dph1_deflector"))
def test_saved_material_metadata_cannot_turn_channel_into_magnetic_body(key):
    iron = next(row for row in material_catalog() if row["material_key"] == "femm_pure_iron")
    channel = {"key": key, "mechanical_profile": "magnetic_lens_yoke",
               "material_regions": {"body": iron}}
    assert validated_material_regions(channel)["body"] == iron
    assert not is_magnetostatic_body(channel)
    assert "no independent material body" in material_application_scope(channel)
    with pytest.raises(ValueError, match="no material body"):
        part_material_updates(channel, "copper")
    with pytest.raises(ValueError, match="no material body"):
        geometry_from_part(channel)
