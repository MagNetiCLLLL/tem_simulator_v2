from pathlib import Path

import pytest

from temsim.optics.column import default_state
from temsim.specimen.scene import SpecimenScene


def test_scene_uses_sample_centre_as_the_shared_local_z_origin():
    state = default_state()
    state.sample.thickness_nm = 20.0
    state.sample.eds_support_material_key = "copper"

    scene = SpecimenScene.from_state(state, include_eds_materials=True)

    assert scene.sample_top_nm == pytest.approx(-10.0)
    assert scene.sample_bottom_nm == pytest.approx(10.0)
    assert scene.support_top_nm == pytest.approx(10.0)
    assert scene.sample_top_z_mm == pytest.approx(state.sample.z_mm - 10.0e-6)
    assert scene.sample_bottom_z_mm == pytest.approx(
        state.sample.z_mm + 10.0e-6
    )


def test_scene_keeps_real_and_virtual_structure_sources_mutually_exclusive():
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "dormant.cif"

    virtual = SpecimenScene.from_state(state)

    assert virtual.source_kind == "preset"
    assert virtual.source_key == "preset:si_110"
    assert virtual.preset_key == "si_110"
    assert virtual.cif_path == ""

    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(Path("user-sample.cif"))
    real = SpecimenScene.from_state(state)

    assert real.source_kind == "cif"
    assert real.source_key == "cif:user-sample.cif"
    assert real.preset_key == ""
    assert real.cif_path.endswith("user-sample.cif")


def test_scene_recognises_virtual_vacuum_as_a_non_interacting_reference():
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 25.0

    scene = SpecimenScene.from_state(state)

    assert scene.is_vacuum
    assert scene.interacting_thickness_nm == 25.0
    assert scene.matter_thickness_nm == 0.0
    assert scene.source_key == "preset:vacuum"


def test_scene_axial_material_query_matches_sample_and_support_geometry():
    state = default_state()
    state.sample.thickness_nm = 12.0
    state.sample.eds_support_material_key = "copper"
    state.sample.eds_support_mesh_key = "square_200"

    scene = SpecimenScene.from_state(state, include_eds_materials=True)
    opening = scene.axial_material_regions(0.0, 0.0)
    bar = scene.axial_material_regions(60_000.0, 0.0)

    assert [region.source_key for region in opening] == ["sample"]
    assert opening[0].z_start_nm == pytest.approx(-6.0)
    assert opening[0].z_end_nm == pytest.approx(6.0)
    assert [region.source_key for region in bar] == [
        "sample",
        "support:bar",
    ]
    assert bar[1].z_start_nm == pytest.approx(scene.support_top_nm)
    assert bar[1].path_length_nm == pytest.approx(25_000.0)

    state.sample.inserted = False
    retracted = SpecimenScene.from_state(state, include_eds_materials=True)
    assert retracted.reference_z_mm == pytest.approx(state.sample.z_mm)
    assert retracted.axial_material_regions(60_000.0, 0.0) == ()
