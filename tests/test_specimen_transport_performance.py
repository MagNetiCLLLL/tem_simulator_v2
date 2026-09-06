"""Boundary-work regressions and analytic fixtures, not OEM validation."""

import math

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.relativistic_lorentz import ELECTRON, momentum_from_kinetic_energy_ev
from temsim.simulation_modes import switch_mode
from temsim.specimen.elastic_transport import (
    ElasticTransportGeometry, IncidentElectronRay, simulate_elastic_point_transport,
)
from temsim.specimen.vector_field_transport import SpecimenFieldTransport


def _state():
    state = default_state()
    switch_mode(state, "ideal")
    state.sample.thickness_nm = 10.0
    state.sample.eds_support_material_key = "vacuum"
    return state


def _uniform_transport(field):
    transport = object.__new__(SpecimenFieldTransport)
    transport.spatial_step_nm = 20_000.0
    transport.field_at_global_positions_t = lambda p: np.broadcast_to(field, np.asarray(p).shape)
    return transport


def _reference_boundary(self, geometry, position, direction, region, *, maximum_distance_nm, energy_ev):
    """Pre-optimisation magnetic chord search retained only as a test oracle."""
    point, unit = np.asarray(position), np.asarray(direction)
    momentum = float(np.linalg.norm(momentum_from_kinetic_energy_ev(energy_ev, unit)))
    travelled = 0.0
    for _ in range(100_000):
        if region is None and (
            (point[2] > geometry.support_bottom_nm and unit[2] > 0)
            or (point[2] < geometry.sample_top_nm and unit[2] < 0)
            or (geometry.sample_material is None and geometry.support_material is None)
        ):
            return None
        remaining = maximum_distance_nm - travelled
        if remaining <= 1e-12:
            return None
        field = np.linalg.norm(np.cross(unit, self.field_at_global_positions_t(point * 1e-9)))
        curvature = abs(ELECTRON.charge_c) * field / momentum * 1e-9
        step = min(remaining, self.spatial_step_nm,
                   math.sqrt(2 * geometry.epsilon_nm / max(curvature, 1e-30)))
        endpoint, final = self.advance(point, unit, step, energy_ev=energy_ev)
        chord = endpoint - point
        norm = np.linalg.norm(chord)
        hit = geometry.next_region_boundary_distance_nm(
            point, chord / norm, region, maximum_distance_nm=norm,
        )
        if hit is not None:
            return travelled + step * hit / norm
        travelled += step
        point, unit = endpoint, final
    raise AssertionError("Reference boundary search exceeded its test budget")


def test_material_bounds_ignore_vacuum_support_but_keep_real_foil():
    state = _state()
    geometry = ElasticTransportGeometry.from_state(state)
    assert geometry.support_bottom_nm == 25_005.0
    assert geometry.material_z_bounds_nm == (-5.0, 5.0)
    state.sample.eds_support_material_key = "copper"
    assert ElasticTransportGeometry.from_state(state).material_z_bounds_nm == (-5.0, 25_005.0)
    state.sample.specimen_preset_key = "vacuum"
    assert ElasticTransportGeometry.from_state(state).material_z_bounds_nm == (5.0, 25_005.0)
    state.sample.inserted = False
    assert ElasticTransportGeometry.from_state(state).material_z_bounds_nm is None


@pytest.mark.parametrize("vacuum", [False, True])
def test_no_magnetic_search_after_leaving_actual_matter(vacuum):
    state = _state()
    if vacuum:
        state.sample.specimen_preset_key = "vacuum"
    geometry = ElasticTransportGeometry.from_state(state)
    transport = _uniform_transport((0.0, 0.0, 1.0))
    def forbidden(*args, **kwargs):
        raise AssertionError("There is no downstream material to search")
    transport.advance = forbidden
    assert transport.boundary_distance(
        geometry, (0.0, 0.0, 5.01), (0.0, 0.0, 1.0), None,
        maximum_distance_nm=1e6, energy_ev=300_000.0,
    ) is None


def test_near_entrance_does_not_integrate_a_long_trial_flight():
    geometry = ElasticTransportGeometry.from_state(_state())
    transport = _uniform_transport((0.0, 0.0, 1.0))
    calls = []
    advance = transport.advance
    def measured(position, direction, distance, **kwargs):
        calls.append(distance)
        return advance(position, direction, distance, **kwargs)
    transport.advance = measured
    distance = transport.boundary_distance(
        geometry, (0.0, 0.0, -5.0 - 8 * geometry.epsilon_nm), (0.0, 0.0, 1.0), None,
        maximum_distance_nm=1e6, energy_ev=300_000.0,
    )
    assert distance == pytest.approx(8 * geometry.epsilon_nm, abs=1e-12)
    assert len(calls) == 1
    assert max(calls) <= 32 * geometry.epsilon_nm


def test_no_tangent_hit_still_finds_a_magnetically_curved_sidewall_crossing():
    state = _state()
    state.sample.envelope_shape = "rectangle"
    state.sample.size_x_nm = state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 1000.0
    geometry = ElasticTransportGeometry.from_state(state)
    position = np.array((-7.0, 0.0, -510.0))
    direction = np.array((0.0, 0.0, 1.0))
    assert geometry.next_region_boundary_distance_nm(
        position, direction, None, maximum_distance_nm=1000.0,
    ) is None
    transport = _uniform_transport((0.0, 100.0, 0.0))
    distance = transport.boundary_distance(
        geometry, position, direction, None,
        maximum_distance_nm=1000.0, energy_ev=300_000.0,
    )
    radius_nm = np.linalg.norm(momentum_from_kinetic_energy_ev(300_000.0, direction)) / (
        abs(ELECTRON.charge_c) * 100.0
    ) * 1e9
    expected = 2 * radius_nm * np.arcsin(np.sqrt(2.0 / (2 * radius_nm)))
    assert distance == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("tilt_mrad", [0.0, 20.0])
def test_seeded_material_paths_match_preoptimisation_reference(monkeypatch, tilt_mrad):
    state = _state()
    direction = np.array((tilt_mrad * 1e-3, 0.0, 1.0))
    direction /= np.linalg.norm(direction)
    rays = tuple(IncidentElectronRay(i, (0.0, 0.0), tuple(direction), 300_000.0, 1/8) for i in range(8))
    actual = simulate_elastic_point_transport(state, incident_rays=rays, seed=73)
    monkeypatch.setattr(SpecimenFieldTransport, "boundary_distance", _reference_boundary)
    expected = simulate_elastic_point_transport(state, incident_rays=rays, seed=73)
    for name in ("source_ray_index", "kinetic_energy_ev", "weight", "outcome", "event_count", "has_scattered"):
        np.testing.assert_array_equal(getattr(actual.terminal_electrons, name), getattr(expected.terminal_electrons, name))
    np.testing.assert_allclose(actual.terminal_electrons.position_nm, expected.terminal_electrons.position_nm, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual.terminal_electrons.direction, expected.terminal_electrons.direction, rtol=0, atol=1e-12)
    np.testing.assert_allclose([t.path_length_nm for t in actual.eds_tracks], [t.path_length_nm for t in expected.eds_tracks], rtol=0, atol=1e-10)
    assert [t.source_key for t in actual.eds_tracks] == [t.source_key for t in expected.eds_tracks]


def test_real_support_is_still_intercepted_after_the_sample():
    state = _state()
    state.sample.eds_support_material_key = "copper"
    geometry = ElasticTransportGeometry.from_state(state)
    # Travel sideways across the central mesh opening while within its foil.
    position = np.array((0.0, 0.0, geometry.support_top_nm + 100.0))
    direction = np.array((1.0, 0.0, 0.0))
    transport = _uniform_transport((0.0, 0.0, 0.0))
    distance = transport.boundary_distance(
        geometry, position, direction, None,
        maximum_distance_nm=1e6, energy_ev=300_000.0,
    )
    expected = 0.5 * geometry.support_grid.mesh.hole_width_um * 1000.0
    assert distance == pytest.approx(expected, abs=geometry.epsilon_nm)
