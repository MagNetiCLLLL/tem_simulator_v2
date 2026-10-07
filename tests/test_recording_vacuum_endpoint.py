"""Recording mechanics, observation and electrostatics share one axial end.

These checks resolve input geometry only: no field solve or ray tracing is
needed, including for the shortest column and optional upstream hardware.
"""
from dataclasses import replace
from itertools import product

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE
from temsim.optics.column import default_state
from temsim.physics.closed_gun_field import closed_field_request
from temsim.physics.continuous_gun_field import continuous_field_request
from temsim.physics.instrument_electric import (
    configure_instrument_electric_domain,
    instrument_electric_end_mm,
)
from temsim.physics.particle_sections import section_limits
from temsim.physics.recording_stop import determine_tem_stop_z


COLUMNS = (
    "C2",
    "C3",
    "C3 + Probe Corrector",
    "C3 + Image Corrector",
    "C3 + Probe Corrector + Image Corrector",
)
RECORDINGS = ("No Energy Filter", "Energy Filter")
BLANKERS = ("None", "Electrostatic beam blanker")


@pytest.fixture(scope="module")
def catalog():
    return AssemblyCatalog()


@pytest.mark.parametrize("column,blanker,recording", tuple(product(COLUMNS, BLANKERS, RECORDINGS)))
def test_recording_endpoint_follows_installed_hardware(catalog, column, blanker, recording):
    state = default_state()
    assembly = catalog.apply(state, AssemblySelection("FEG", column, recording, blanker))
    camera = assembly.part("camera")
    module = next(item for item in assembly.modules if item.key == camera.module_key)
    # The camera's physical plane must retain its original module-local value;
    # upstream optional modules change the global coordinate, not this spacing.
    origin = sum(item.length_mm for item in assembly.modules if item is not module)
    assert camera.data["optical_reference_local_z_mm"] == pytest.approx(1172.25)
    assert state.camera.z_mm == pytest.approx(origin + 1172.25)
    assert camera.end_z_mm == pytest.approx(origin + 1172.75)
    assert bool(state.nanopulser.installed) == (blanker != "None")

    if recording == "No Energy Filter":
        endpoint = camera.end_z_mm
        assert module.exit_z_mm == pytest.approx(camera.data["local_end_z_mm"])
        assert assembly.exit_z_mm == pytest.approx(endpoint)
        housing = assembly.part("camera_recording_housing")
        chamber = assembly.part("post_projector_detector_chamber")
        assert housing.start_z_mm == pytest.approx(chamber.end_z_mm)
        assert housing.end_z_mm == pytest.approx(camera.end_z_mm)
        assert housing.data["mechanical_only"] is True
        assert housing.data["axial_vacuum_context_only"] is True
    else:
        inlet = assembly.part(ENERGY_FILTER_ENTRANCE_APERTURE)
        endpoint = inlet.center_z_mm
        assert endpoint == pytest.approx(assembly.part("energy_filter").center_z_mm)
        assert assembly.exit_z_mm > inlet.end_z_mm

    assert max(row.end_z_mm for row in assembly.vacuum_bore_segments) == pytest.approx(endpoint)
    assert max(row.end_z_mm for row in assembly.vacuum_liner_segments) == pytest.approx(endpoint)
    assert instrument_electric_end_mm(state) == pytest.approx(endpoint)
    lower, observation_end = section_limits(state)
    assert lower < state.camera.z_mm <= observation_end <= endpoint + 1e-9
    if recording == "Energy Filter":
        assert observation_end == pytest.approx(endpoint)
    else:
        assert state.camera.z_mm <= determine_tem_stop_z(state) <= endpoint + 1e-9


REPRESENTATIVE_SELECTIONS = (
    AssemblySelection("FEG", "C2", "No Energy Filter"),
    AssemblySelection(
        "FEG", "C3 + Probe Corrector + Image Corrector", "Energy Filter",
        "Electrostatic beam blanker",
    ),
)


@pytest.mark.parametrize("selection", REPRESENTATIVE_SELECTIONS, ids=("short-no-filter", "long-filter-blanker"))
def test_grounded_field_requests_cover_the_shared_endpoint_without_solving(catalog, selection):
    state = default_state()
    catalog.apply(state, selection)
    endpoint = instrument_electric_end_mm(state)
    gun = state.electron_gun
    configure_instrument_electric_domain(gun, endpoint)
    gun.emitter.curvature_nm_inv = 0.0
    planar = closed_field_request(gun, cells_per_bore=4)
    gun.emitter.curvature_nm_inv = 0.01
    curved = continuous_field_request(gun, cells_per_bore=4)

    for request in (planar, curved):
        assert request["domain"]["exit_m"] == pytest.approx(endpoint * 1e-3)
        rows = request["grounded_liner"]
        assert rows[-1]["stop_m"] == pytest.approx(endpoint * 1e-3)
        assert rows[0]["start_m"] < request["domain"]["gun_exit_m"]
        for left, right in zip(rows, rows[1:]):
            assert left["stop_m"] == pytest.approx(right["start_m"], abs=1e-12)
    assert planar["grounded_liner"] == curved["grounded_liner"]


@pytest.mark.parametrize("selection", REPRESENTATIVE_SELECTIONS, ids=("short-no-filter", "long-filter-blanker"))
def test_rendered_liner_uses_the_same_endpoint(catalog, selection):
    state = default_state()
    assembly = catalog.apply(state, selection)
    # Only synthesize the resolved liner, avoiding a full instrument mesh.
    model = assembly_model_from_assembly(replace(assembly, modules=(), parts=()), angular_segments=8)
    assert not model.errors
    assert model.meshes
    assert max(float(mesh.vertices[:, 2].max()) for mesh in model.meshes) == pytest.approx(
        instrument_electric_end_mm(state)
    )
    if selection.recording == "No Energy Filter":
        housing = assembly.part("camera_recording_housing")
        housing_model = assembly_model_from_assembly(
            replace(assembly, parts=(housing,), vacuum_liner_segments=()), angular_segments=8,
        )
        assert not housing_model.errors
        assert housing_model.meshes
        assert min(float(mesh.vertices[:, 2].min()) for mesh in housing_model.meshes) == pytest.approx(
            assembly.part("post_projector_detector_chamber").end_z_mm
        )
        assert max(float(mesh.vertices[:, 2].max()) for mesh in housing_model.meshes) == pytest.approx(
            assembly.part("camera").end_z_mm
        )
