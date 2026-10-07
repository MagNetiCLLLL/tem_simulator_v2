"""Finite column endpoints include all actual interaction stations."""
from types import SimpleNamespace

import pytest

from temsim.physics.recording_stop import (
    MINIMUM_TEM_STOP_Z_MM, RECORDING_STOP_MARGIN_MM, determine_tem_stop_z,
)


def test_empty_collections_retain_explicit_minimum_endpoint():
    state = SimpleNamespace(recording_planes=(), apertures=(), lenses=(),
        deflectors=(), stigmators=(), corrector_elements=())
    assert determine_tem_stop_z(state) == MINIMUM_TEM_STOP_Z_MM + RECORDING_STOP_MARGIN_MM


@pytest.mark.parametrize("collection", [
    "recording_planes", "apertures", "lenses", "deflectors", "stigmators", "corrector_elements",
])
def test_downstream_component_and_driven_interaction_are_not_skipped(collection):
    events = []
    def kicks(time_s):
        events.append(time_s)
        return ((4100. + time_s, 0., 0.),)
    component = SimpleNamespace(z_mm=3500., kick_events=kicks)
    state = SimpleNamespace(simulation_time_s=2., **{collection: (component,)})
    assert determine_tem_stop_z(state) == 4102. + RECORDING_STOP_MARGIN_MM
    assert events == [2.]


@pytest.mark.parametrize("end_z_mm", [2550.0, 3025.0, 3400.0])
@pytest.mark.parametrize("inserted, step_mm", [(False, 0.1), (True, 5.0)])
def test_assembled_camera_endpoint_follows_module_placement(end_z_mm, inserted, step_mm):
    camera = SimpleNamespace(z_mm=end_z_mm - 0.5, inserted=inserted)
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=end_z_mm, parts=()),
        recording_planes=(camera,),
        step_mm=step_mm,
    )

    assert determine_tem_stop_z(state) == end_z_mm


def test_observation_margin_is_limited_by_the_physical_endpoint():
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2600.0, parts=()),
        recording_planes=(SimpleNamespace(z_mm=2599.9),),
    )

    assert determine_tem_stop_z(state) == 2600.0


def test_assembled_observation_plane_remains_after_last_interaction():
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2700.0, parts=()),
        recording_planes=(SimpleNamespace(z_mm=2600.0),),
    )

    assert determine_tem_stop_z(state) == 2600.0 + RECORDING_STOP_MARGIN_MM


def test_empty_assembled_column_uses_its_physical_endpoint():
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2600.0, parts=()),
    )

    assert determine_tem_stop_z(state) == 2600.0


@pytest.mark.parametrize("driven_event", [False, True])
def test_actual_station_outside_assembled_column_is_not_silently_truncated(driven_event):
    component = SimpleNamespace(z_mm=2600.1, inserted=False)
    if driven_event:
        component.z_mm = 2500.0
        component.kick_events = lambda time_s: ((2600.1, 0.0, 0.0),)
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2600.0, parts=()),
        recording_planes=(component,),
    )

    with pytest.raises(ValueError, match="beyond the assembled column endpoint"):
        determine_tem_stop_z(state)


@pytest.mark.parametrize("interface_key", ["energy_filter", "energy_filter_entrance_aperture"])
def test_installed_filter_retains_its_separate_downstream_recording_coordinates(interface_key):
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(
            exit_z_mm=2600.0,
            parts=(SimpleNamespace(key=interface_key),),
        ),
        recording_planes=(SimpleNamespace(z_mm=3100.0),),
    )

    assert determine_tem_stop_z(state) == 3100.0 + RECORDING_STOP_MARGIN_MM


def test_uninstalled_filter_entrance_placeholder_does_not_extend_camera_column():
    # NoEnergyFilter retains an editable filter entrance object at its default
    # coordinate, although the installed column now ends at the camera shell.
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2413.15, parts=()),
        recording_planes=(SimpleNamespace(z_mm=2412.65, inserted=False),),
        apertures=(SimpleNamespace(
            key="energy_filter_entrance_aperture", z_mm=2451.9,
            installed=False, enabled=True,
        ),),
    )

    assert determine_tem_stop_z(state) == pytest.approx(2413.15)


@pytest.mark.parametrize("installation_flag", ["installed", "_layout_installed"])
def test_uninstalled_optional_component_does_not_contribute_kick_events(installation_flag):
    def absent_kicks(time_s):
        pytest.fail("Uninstalled hardware must not be queried for physical interactions")

    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2600.0, parts=()),
        recording_planes=(SimpleNamespace(z_mm=2599.5),),
        corrector_elements=(SimpleNamespace(
            z_mm=2800.0, kick_events=absent_kicks, **{installation_flag: False},
        ),),
    )

    assert determine_tem_stop_z(state) == 2600.0


def test_disabled_installed_hardware_outside_column_is_still_rejected():
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(exit_z_mm=2600.0, parts=()),
        apertures=(SimpleNamespace(
            z_mm=2600.1, installed=True, _layout_installed=True, enabled=False,
        ),),
    )

    with pytest.raises(ValueError, match="beyond the assembled column endpoint"):
        determine_tem_stop_z(state)
