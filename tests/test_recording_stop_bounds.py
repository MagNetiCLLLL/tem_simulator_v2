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
