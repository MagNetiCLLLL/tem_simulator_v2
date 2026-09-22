"""Display lookup uses recorded launch data without replaying source physics."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.emission_source_data import EmissionSourceData


def make_simulation():
    # Deliberately unsorted sparse IDs: downstream rows need not follow launch.
    reference = {
        "ray_id": np.array([42, 7, 20, 3]),
        "position_m": np.array([[2e-9, 0., -1e-10], [0., 3e-9, 0.],
                                [-4e-9, 0., 0.], [0., 0., 0.]]),
        "direction": np.array([[1., 0., 1.], [0., -1., 2.],
                               [-1., 0., -1.], [0., 0., 1.]]),
        "normal": np.array([[0., 0., 1.], [0., -1., 0.],
                            [0., 0., 1.], [0., 0., 1.]]),
    }
    reference["direction"] /= np.linalg.norm(reference["direction"], axis=1)[:, None]
    return SimpleNamespace(gun_trace=SimpleNamespace(emission_reference=reference))


def test_original_positions_and_angles_are_independent_and_follow_reordered_descendants():
    simulation = make_simulation()
    data = EmissionSourceData.from_simulation(simulation)
    query = np.array([[20, 7, 20], [42, 3, 999]])
    np.testing.assert_array_equal(data.indices_for(query), [[2, 1, 2], [0, 3, -1]])
    np.testing.assert_allclose(data.values(query, "source"),
                               [[np.pi, np.pi/2, np.pi], [0., np.nan, np.nan]], equal_nan=True)
    np.testing.assert_allclose(data.values(query, "emission_direction"),
                               [[np.pi, 3*np.pi/2, np.pi], [0., np.nan, np.nan]], equal_nan=True)
    np.testing.assert_allclose(data.values(query, "emission_angle"),
                               [[3*np.pi/4, np.arctan(2.), 3*np.pi/4],
                                [np.pi/4, 0., np.nan]], equal_nan=True)
    assert data.angle_to_normal_rad[2] > np.pi/2  # A backward launch is retained.
    assert data.status == "Ready"


def test_arrays_are_independent_and_irreversibly_read_only():
    simulation = make_simulation()
    data = EmissionSourceData.from_simulation(simulation)
    positions = data.position_m.copy()
    simulation.gun_trace.emission_reference["position_m"][:] = 10.
    simulation.gun_trace.emission_reference["direction"][:] = 0.
    np.testing.assert_array_equal(data.position_m, positions)
    for array in (data.source_ids, data.position_m, data.source_azimuth_rad,
                  data.direction_azimuth_rad, data.angle_to_normal_rad,
                  data.indices_for([42]), data.values([42], "source"), data.display_indices(2)):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        data.status = "changed"


def test_partial_record_keeps_directions_when_launch_positions_are_missing():
    simulation = make_simulation()
    del simulation.gun_trace.emission_reference["position_m"]
    data = EmissionSourceData.from_simulation(simulation)
    assert data.position_m.shape == (4, 3)
    assert np.all(np.isnan(data.position_m))
    assert np.all(np.isnan(data.source_azimuth_rad))
    np.testing.assert_allclose(data.values(data.source_ids, "emission_direction"),
                               [0., 3*np.pi/2, np.pi, np.nan], equal_nan=True)
    np.testing.assert_allclose(data.values(data.source_ids, "emission_angle"),
                               [np.pi/4, np.arctan(2.), 3*np.pi/4, 0.])
    assert "Partial" in data.status and "position_m" in data.status


def test_missing_normal_preserves_direction_azimuth_but_not_angle_to_normal():
    simulation = make_simulation()
    del simulation.gun_trace.emission_reference["normal"]
    data = EmissionSourceData.from_simulation(simulation)
    assert data.values([7], "emission_direction")[0] == pytest.approx(3*np.pi/2)
    assert np.all(np.isnan(data.angle_to_normal_rad))
    assert "normal" in data.status


def test_bad_rows_are_unavailable_without_destroying_other_rows_or_quantities():
    simulation = make_simulation()
    record = simulation.gun_trace.emission_reference
    record["position_m"][0] = np.nan
    record["direction"][1] = 0.
    record["normal"][2] = 0.
    data = EmissionSourceData.from_simulation(simulation)
    assert np.isnan(data.source_azimuth_rad[0])
    assert data.direction_azimuth_rad[0] == 0.
    assert np.isnan(data.direction_azimuth_rad[1]) and np.isnan(data.angle_to_normal_rad[1])
    assert np.isnan(data.angle_to_normal_rad[2])
    assert data.source_azimuth_rad[2] == np.pi
    assert "zero vectors" in data.status and "non-finite" in data.status


@pytest.mark.parametrize("ids", [[1, 1], [-1, 2], [1., 2.], [True, False],
                                  [[1, 2]], np.array([2**63], dtype=np.uint64)])
def test_invalid_source_identity_is_rejected_without_guessing(ids):
    simulation = make_simulation()
    simulation.gun_trace.emission_reference["ray_id"] = ids
    data = EmissionSourceData.from_simulation(simulation)
    assert data.source_ids.size == 0
    assert "Invalid source IDs" in data.status
    assert np.isnan(data.values([1], "source")[0])


@pytest.mark.parametrize("record", [None, "legacy", {}, {"ray_id": np.array([], dtype=int)}])
def test_missing_historical_launch_never_falls_back_to_incident_coordinates(record):
    simulation = SimpleNamespace(gun_trace=SimpleNamespace(emission_reference=record),
                                 incident=SimpleNamespace(x=np.ones((2, 10)), y=np.ones((2, 10))))
    data = EmissionSourceData.from_simulation(simulation)
    assert data.source_ids.size == 0 and data.position_m.shape == (0, 3)
    assert data.display_indices(100).size == 0
    assert np.all(data.indices_for([0, 1]) == -1)
    assert data.status != "Ready"


@pytest.mark.parametrize("query", [np.array(7), np.array([], dtype=int),
                                    np.array([[42, 7], [-1, 8]]),
                                    np.array([2**64-1], dtype=np.uint64),
                                    np.array([True]), np.array([7.]), np.array(["7"])])
def test_lookup_preserves_shape_and_never_coerces_invalid_ids(query):
    data = EmissionSourceData.from_simulation(make_simulation())
    indices = data.indices_for(query)
    values = data.values(query, "source")
    assert indices.shape == values.shape == query.shape
    assert indices.dtype == np.int64
    if query.dtype.kind not in "iu":
        assert np.all(indices == -1) and np.all(np.isnan(values))


def test_display_sample_uses_original_rows_and_lookup_never_resorts(monkeypatch):
    simulation = make_simulation()
    data = EmissionSourceData.from_simulation(simulation)
    def unexpected(*args, **kwargs):
        pytest.fail("Display access must not sort or resample physical emission")
    monkeypatch.setattr(np, "argsort", unexpected)
    monkeypatch.setattr(np, "unique", unexpected)
    np.testing.assert_array_equal(data.display_indices(3), [0, 1, 3])
    np.testing.assert_array_equal(data.display_indices(100), np.arange(4))
    assert data.display_indices(0).size == 0
    for ids in ([7], [20, 20, 42], [3, 999]):
        data.values(ids, "source")
        data.values(ids, "emission_direction")
    np.testing.assert_array_equal(data.display_indices(3), [0, 1, 3])


@pytest.mark.parametrize("limit", [-1, 2.5, True, "4"])
def test_invalid_display_budget_is_explicit(limit):
    data = EmissionSourceData.from_simulation(make_simulation())
    with pytest.raises(ValueError, match="non-negative integer"):
        data.display_indices(limit)


def test_unknown_colour_quantity_is_explicit():
    data = EmissionSourceData.from_simulation(make_simulation())
    with pytest.raises(ValueError, match="Unknown emission colour"):
        data.values([42], "current_direction")
