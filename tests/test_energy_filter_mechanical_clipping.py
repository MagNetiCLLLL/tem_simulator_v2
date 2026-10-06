"""Mechanical branch boundaries, without field propagation or a GUI."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.column.module_assembly import _module_vacuum_segments
from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE
from temsim.optics.column import default_state
from temsim.optics.energy_filter_raytrace import _m12_bore_blocked
from temsim.physics.column_wall import COLUMN_WALL_KEY, clip_column_wall


@pytest.fixture(scope="module")
def installed_filter():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state, AssemblySelection("FEG", "C3", "Energy Filter")
    )
    return state, assembly


def test_main_vacuum_and_liner_reach_filter_optical_handoff(installed_filter):
    _state, assembly = installed_filter
    inlet = assembly.part(ENERGY_FILTER_ENTRANCE_APERTURE)
    assert max(segment.end_z_mm for segment in assembly.vacuum_bore_segments) == pytest.approx(inlet.center_z_mm)
    assert max(segment.end_z_mm for segment in assembly.vacuum_liner_segments) == pytest.approx(inlet.center_z_mm)
    # The module's declared envelope still describes the installed assembly;
    # shortening the pipe must not move its ports or optical components.
    assert assembly.exit_z_mm > inlet.end_z_mm

    # Exercise the renderer-neutral 3D liner builder with the actual resolved
    # liner geometry. Other instrument parts are irrelevant to this boundary.
    model = assembly_model_from_assembly(
        replace(assembly, modules=(), parts=()), angular_segments=8
    )
    assert not model.errors
    assert max(float(mesh.vertices[:, 2].max()) for mesh in model.meshes) == pytest.approx(inlet.center_z_mm)


def test_filter_carrier_envelope_change_does_not_move_axial_handoff(installed_filter):
    _state, assembly = installed_filter
    inlet = assembly.part(ENERGY_FILTER_ENTRANCE_APERTURE)
    module = next(item for item in assembly.modules if item.key == inlet.module_key)
    local_inlet = next(item for item in module.parts if item.key == inlet.key)
    origin = inlet.center_z_mm - local_inlet.center_z_mm
    # Increasing the mechanism's upstream envelope must not open a gap before
    # the unchanged optical handoff or change the fixed electric domain.
    moved_inlet = replace(inlet, start_z_mm=inlet.start_z_mm - 3.0, length_mm=inlet.length_mm + 3.0)
    moved_parts = tuple(moved_inlet if item.key == inlet.key else item for item in assembly.parts)
    segments = _module_vacuum_segments(module, origin, moved_parts)
    assert segments[-1].end_z_mm == pytest.approx(inlet.center_z_mm)


def test_filter_optical_handoff_adjustment_updates_shared_vacuum_endpoint(installed_filter):
    _state, assembly = installed_filter
    inlet = assembly.part(ENERGY_FILTER_ENTRANCE_APERTURE)
    module = next(item for item in assembly.modules if item.key == inlet.module_key)
    local_inlet = next(item for item in module.parts if item.key == inlet.key)
    origin = inlet.center_z_mm - local_inlet.center_z_mm
    moved_inlet = replace(inlet, start_z_mm=inlet.start_z_mm + 3.0,
                          center_z_mm=inlet.center_z_mm + 3.0, end_z_mm=inlet.end_z_mm + 3.0)
    moved_parts = tuple(moved_inlet if item.key == inlet.key else item for item in assembly.parts)
    segments = _module_vacuum_segments(module, origin, moved_parts)
    assert segments[-1].end_z_mm == pytest.approx(moved_inlet.center_z_mm)


def test_no_filter_retains_full_declared_main_vacuum_span():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state, AssemblySelection("FEG", "C3", "No Energy Filter")
    )
    assert max(segment.end_z_mm for segment in assembly.vacuum_bore_segments) == pytest.approx(assembly.exit_z_mm)
    assert max(segment.end_z_mm for segment in assembly.vacuum_liner_segments) == pytest.approx(assembly.exit_z_mm)


@pytest.mark.parametrize("downstream", [False, True])
def test_runtime_column_wall_ends_at_filter_inlet(installed_filter, downstream):
    _state, assembly = installed_filter
    last = assembly.vacuum_bore_segments[-1]
    inlet = assembly.part(ENERGY_FILTER_ENTRANCE_APERTURE)
    z = np.array([inlet.center_z_mm - 1.0, inlet.center_z_mm - 0.5])
    if downstream:
        z += 1.5
    x = np.full((2, 1), (0.5 * last.inner_diameter_mm + 1.0) * 1.0e-3)
    alive, blocked_z, blocked_key = clip_column_wall(
        SimpleNamespace(_resolved_assembly=assembly), z, x, np.zeros_like(x)
    )
    assert alive.tolist() == [downstream]
    if downstream:
        assert np.isnan(blocked_z[0])
        assert blocked_key == [""]
    else:
        assert blocked_z[0] == pytest.approx(z[0])
        assert blocked_key == [COLUMN_WALL_KEY]


@pytest.mark.parametrize("index", [0, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_m12_bore_uses_housing_ends_in_each_oriented_bank(installed_filter, index, enabled):
    state, _assembly = installed_filter
    element = replace(state.energy_filter.multipoles[index], enabled=enabled)
    half_housing = 0.5 * element.housing_length_m
    between_field_and_housing = 0.25 * (element.length_m + element.housing_length_m)
    assert 0.5 * element.length_m < between_field_and_housing < half_housing
    outside_bore = element.bore_radius_m + 1.0e-3
    local = np.array([
        [outside_bore, 0.0, -between_field_and_housing],
        [0.0, outside_bore, between_field_and_housing],
        [0.0, 0.0, between_field_and_housing],
        [outside_bore, 0.0, -half_housing - 1.0e-6],
        [outside_bore, 0.0, half_housing + 1.0e-6],
    ])
    positions = local @ element.frame.rotation_local_to_global.T + element.frame.origin_m
    np.testing.assert_array_equal(
        _m12_bore_blocked(element, positions), [True, True, False, False, False]
    )
