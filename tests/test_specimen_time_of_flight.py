"""Executed specimen clocks; aggregate inelastic event times remain unknown."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S,
)
from temsim.specimen import elastic_transport as elastic
from temsim.specimen import downstream_transport as downstream
from temsim.specimen.vector_field_transport import SpecimenFieldTransport


def _speed(energy_ev):
    gamma = 1. + energy_ev * ELEMENTARY_CHARGE_C / (
        ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S**2)
    return SPEED_OF_LIGHT_M_PER_S * np.sqrt(1. - gamma**-2)


def _field_free_state():
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements):
        for component in collection:
            component.enabled = False
    state.sample.eds_support_material_key = "vacuum"
    return state


def test_specimen_signed_plane_matching_time_uses_actual_tilted_flight():
    transport = SpecimenFieldTransport(_field_free_state())
    direction = np.array((.3, -.2, 1.))
    direction /= np.linalg.norm(direction)
    start = np.array((11., -12., 0.))
    energy = 200_000.
    end, final, elapsed = transport.to_plane(
        start, direction, -50., energy_ev=energy, return_elapsed_time=True)
    assert elapsed == pytest.approx(-50e-9 / direction[2] / _speed(energy), rel=2e-14)
    recovered, _, forward = transport.to_plane(
        end, final, 0., energy_ev=energy, return_elapsed_time=True)
    np.testing.assert_allclose(recovered, start, atol=1e-12)
    assert forward == pytest.approx(-elapsed, rel=2e-14)
    plain = transport.to_plane(start, direction, -50., energy_ev=energy)
    np.testing.assert_array_equal(plain[0], end)
    np.testing.assert_array_equal(plain[1], final)


def test_magnetic_specimen_clock_counts_curved_path_and_signed_inverse():
    # Isolated uniform magnetic field: energy and v_z remain constant, while
    # the transverse direction rotates. It is not a full-column fixture.
    transport = SpecimenFieldTransport(_field_free_state())
    transport.field_at_global_positions_t = lambda _p: np.array((0., 0., 10.))
    transport.spatial_step_nm = 10.
    direction = np.array((.3, -.2, 1.))
    direction /= np.linalg.norm(direction)
    endpoint, final, elapsed = transport.advance(
        np.zeros(3), direction, 1000., energy_ev=200_000., return_elapsed_time=True)
    assert elapsed == pytest.approx(1000e-9 / _speed(200_000.), rel=2e-14)
    assert elapsed > endpoint[2] * 1e-9 / _speed(200_000.)
    assert not np.allclose(final[:2], direction[:2], rtol=1e-4, atol=0.)
    recovered, _, reverse = transport.to_plane(
        endpoint, final, 0., energy_ev=200_000., return_elapsed_time=True)
    np.testing.assert_allclose(recovered, np.zeros(3), atol=1e-9)
    assert reverse == pytest.approx(-elapsed, rel=2e-14)


def test_elastic_offsets_follow_each_full_track_including_boundary_flights(monkeypatch):
    state = _field_free_state()
    state.sample.thickness_nm = 100.
    state.sample.size_x_nm = state.sample.size_y_nm = 1000.
    monkeypatch.setattr(elastic, "elastic_scattering_rates_nm_inverse", lambda *_: ((14, .05),))
    monkeypatch.setattr(elastic, "sample_screened_rutherford_angle", lambda *_: (.04, .3))
    direction = np.array((.02, -.01, 1.))
    direction /= np.linalg.norm(direction)
    # Two descendants may share a parent index, but their sampled trajectories
    # and therefore travel delays must remain independent rows.
    rays = tuple(elastic.IncidentElectronRay(0, (0., 0.), tuple(direction), 200_000., .5)
                 for _ in range(2))
    result = elastic.simulate_elastic_point_transport(
        state, incident_rays=rays, seed=12, stored_trajectory_count=2)
    terminal = result.terminal_electrons
    assert terminal.reference_time_offset_s.dtype == np.float64
    assert not terminal.reference_time_offset_s.flags.writeable
    assert not np.isclose(*terminal.reference_time_offset_s, rtol=1e-8, atol=0.)
    for index, track in enumerate(result.trajectories):
        points = track.points_nm
        initial_signed_length = points[0, 2] / direction[2]
        length = (initial_signed_length + np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                  + np.linalg.norm(terminal.position_nm[index] - points[-1]))
        assert terminal.reference_time_offset_s[index] == pytest.approx(
            length * 1e-9 / _speed(200_000.), rel=3e-12, abs=1e-27)
    np.testing.assert_array_equal(terminal.kinetic_energy_ev, np.full(2, 200_000.))
    assert terminal.weight.sum() == pytest.approx(1.)


def _downstream_fixture(monkeypatch, *, offsets=True, exact_plane=True):
    state = SimpleNamespace(
        beam_voltage_kv=200., chromatic_aberration_enabled=False,
        objective_lens=SimpleNamespace(cc_mm=2.), sample=SimpleNamespace(z_mm=1.),
        lenses=(), stigmators=(), condenser_system={}, deflectors=(), corrector_elements=(),
        vacuum_map=SimpleNamespace(enabled=False))
    incident = SimpleNamespace(
        alive=np.array((True, True)), ray_weight=np.array((.5, .5)),
        z=np.array((.9, 1. if exact_plane else 1.1)),
        flight_time_s=np.array(((9e-10, 1.9e-9), (1e-9, 2e-9))),
        source_ray_id=np.array((100, 200)), source_azimuth_rad=np.array((0., 1.)))
    terminal = elastic.ElasticTerminalBundle(
        source_ray_index=np.array((1, 0, 1)),
        position_nm=np.array(((0., 0., 5.), (0., 0., 5.), (0., 0., 5.))),
        direction=np.tile((0., 0., 1.), (3, 1)), kinetic_energy_ev=np.full(3, 200_000.),
        weight=np.full(3, 1/3), outcome=("transmitted",)*3,
        event_count=np.zeros(3, dtype=int), has_scattered=np.zeros(3, dtype=bool),
        material_path_nm=np.full(3, 5.),
        reference_time_offset_s=np.array((1e-16, 2e-16, 3e-16)) if offsets else None)
    transport = elastic.ElasticTransportResult((), (), {}, terminal_electrons=terminal)
    def drift(_s, start, stop, x, tx, y, ty, _events, energy, **kw):
        z = np.array((start, stop))
        time = np.asarray(kw["initial_time_s"])
        elapsed = (z-start)[:, None] * 1e-3 * np.sqrt(1+tx*tx+ty*ty)[None, :] / _speed(200_000.+energy)[None, :]
        return z, np.tile(x, (2, 1)), np.tile(tx, (2, 1)), np.tile(y, (2, 1)), np.tile(ty, (2, 1)), time+elapsed
    monkeypatch.setattr(downstream, "propagate", drift)
    monkeypatch.setattr(downstream, "determine_tem_stop_z", lambda _: 2.)
    for name in ("clip_recording_planes", "clip_column_wall"):
        monkeypatch.setattr(downstream, name, lambda _s, _z, _x, _y, alive, blocked, keys: (alive, blocked, keys))
    return state, SimpleNamespace(incident=incident), transport


def test_downstream_keeps_distinct_descendant_delays_and_parent_column_mapping(monkeypatch):
    state, simulation, transport = _downstream_fixture(monkeypatch)
    result = downstream.build_geometric_specimen_exit(state, simulation, transport)
    branch = result.branches[0]
    assert np.all(np.isnan(branch.flight_time_s[0]))  # Virtual pre-terminal reference row.
    expected = np.array((2e-9, 1e-9, 2e-9)) + np.array((1e-16, 2e-16, 3e-16))
    expected += (1e-3-5e-9)/_speed(200_000.)
    np.testing.assert_allclose(branch.flight_time_s[-1], expected, rtol=1e-15, atol=0.)
    np.testing.assert_array_equal(branch.source_ray_id, (200, 100, 200))
    assert branch.flight_time_s[-1, 2] > branch.flight_time_s[-1, 0]
    assert branch.weight == pytest.approx(1.)


@pytest.mark.parametrize("missing", ["terminal", "reference"])
def test_legacy_or_sparse_incident_timing_stays_unknown(monkeypatch, missing):
    state, simulation, transport = _downstream_fixture(
        monkeypatch, offsets=missing != "terminal", exact_plane=missing != "reference")
    result = downstream.build_geometric_specimen_exit(state, simulation, transport)
    assert np.all(np.isnan(result.branches[0].flight_time_s))
    assert result.branches[0].weight == pytest.approx(1.)


def test_aggregate_inelastic_loss_has_no_invented_event_time(monkeypatch):
    state, simulation, transport = _downstream_fixture(monkeypatch)
    distribution = SimpleNamespace(channels=(
        SimpleNamespace(key="real_zero_loss", probability=.7, characteristic_angle_mrad=0., energy_loss_ev=0.),
        SimpleNamespace(key="real_plasmon", probability=.2, characteristic_angle_mrad=2., energy_loss_ev=20.)),
        tracked_probability=.9, absorbed_probability=.1)
    result = downstream.build_geometric_specimen_exit(state, simulation, transport, distribution)
    zero, loss = result.branches
    assert np.all(np.isfinite(zero.flight_time_s[-1]))
    assert np.all(np.isnan(loss.flight_time_s))
    assert result.metrics["flight_time_missing_inelastic_event_depth"]
    assert sum(b.weight for b in result.branches) == pytest.approx(.9)


def test_specimen_exit_integrates_clock_through_real_column_propagator(monkeypatch):
    # Small field-free column segment, using the production RK clock path.
    # Detailed specimen parent matching remains the real downstream adapter.
    _, simulation, transport = _downstream_fixture(monkeypatch)
    state = _field_free_state()
    state.sample.z_mm = 1.
    state.electron_gun.accelerator.high_tension_kv = 200.
    state.step_mm = state.history_step_mm = .1
    state.acceleration_enabled = False
    state.chromatic_aberration_enabled = False
    state.deflectors = []
    from temsim.physics.core import propagate
    monkeypatch.setattr(downstream, "propagate", propagate)
    result = downstream.build_geometric_specimen_exit(state, simulation, transport)
    expected = np.array((2e-9, 1e-9, 2e-9)) + np.array((1e-16, 2e-16, 3e-16))
    expected += (1e-3 - 5e-9) / _speed(200_000.)
    np.testing.assert_allclose(result.branches[0].flight_time_s[-1], expected, rtol=3e-15, atol=0.)
