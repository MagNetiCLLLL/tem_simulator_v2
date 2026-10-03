"""Interception projection reads retained histories without new transport."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.plane_cutoff_events import (
    PlaneCutoffEventsCache, sample_plane_cutoff_events,
)
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def branch(count=3, *, z=(0., 10.), name="incident", **changes):
    z = np.asarray(z, float)
    # Independent linear trajectories, x=(2*Z+column) um and y=(-Z+3*column) um.
    columns = np.arange(count)
    values = dict(name=name, z=z, x=(2. * z[:, None] + columns) * 1.e-6,
                  y=(-z[:, None] + 3. * columns) * 1.e-6,
                  blocked_z=np.array([2., 4., 8.]) if count == 3 else np.linspace(z[0], z[-1], count),
                  blocked_key=["a"] * count, source_ray_id=10 + columns * 7,
                  ray_weight=np.full(count, 1. / max(count, 1)), weight=1.)
    values.update(changes)
    return SimpleNamespace(**values)


def result(incident=None, downstream=()):
    return SimpleNamespace(
        simulation=SimpleNamespace(incident=branch() if incident is None else incident,
                                   branches={str(i): row for i, row in enumerate(downstream)}),
        signatures={"sample_downstream": "current"},
        aperture_stops=[{"key": "a", "name": "Aperture A"}],
        state_snapshot=SimpleNamespace(recording_planes=[SimpleNamespace(key="haadf")]),
    )


def test_interception_positions_use_each_ray_stop_not_selected_plane():
    data = sample_plane_cutoff_events(result(), 9.)
    group, = data.groups
    np.testing.assert_array_equal(group.stop_z_mm, [2., 4., 8.])
    np.testing.assert_allclose(group.x_m, np.array([4., 9., 18.]) * 1.e-6)
    np.testing.assert_allclose(group.y_m, np.array([-2., -1., -2.]) * 1.e-6)
    assert group.stop_kind == "hardware" and group.blocked_key == "a"
    assert group.provenance == "Incident" and group.observed_count == 3
    assert data.total_recorded_count == data.displayed_count == 3
    np.testing.assert_array_equal(group.source_ray_id, [10, 17, 24])


@pytest.mark.parametrize("selected,expected", [(1., []), (2., [2.]), (4., [2., 4.]), (7., [2., 4.])])
def test_selected_plane_includes_only_already_encountered_stops(selected, expected):
    data = sample_plane_cutoff_events(result(), selected)
    stops = np.concatenate([group.stop_z_mm for group in data.groups]) if data.groups else []
    np.testing.assert_array_equal(stops, expected)


def test_exact_last_history_row_and_after_history_observation_are_valid():
    incident = branch(count=2, blocked_z=np.array([0., 10.]))
    data = sample_plane_cutoff_events(result(incident), 30.)
    np.testing.assert_allclose(data.groups[0].x_m, [0., 21.e-6])
    np.testing.assert_allclose(data.groups[0].y_m, [0., -7.e-6])


def test_numerical_and_nonhardware_stops_do_not_claim_component_absorption():
    incident = branch(blocked_key=["projected_field_domain", "medium_removal:gas", "unknown_device"])
    data = sample_plane_cutoff_events(result(incident), 10.)
    assert [group.stop_kind for group in data.groups] == ["numerical", "non_hardware", "recorded_stop"]
    assert [group.blocked_key for group in data.groups] == incident.blocked_key


def test_captured_detector_and_column_stops_are_hardware():
    incident = branch(blocked_key=["haadf", "column_wall", "feg_tip_reabsorbed"])
    data = sample_plane_cutoff_events(result(incident), 10.)
    assert all(group.stop_kind == "hardware" for group in data.groups)


def test_detailed_exit_excludes_optical_reference_and_empty_exit_does_not_fallback():
    incident = branch(blocked_z=np.full(3, np.nan))
    reference = branch(z=(10., 20.), name="reference", blocked_z=np.full(3, 13.))
    detailed = branch(z=(10., 20.), name="detailed", blocked_z=np.full(3, 14.),
                      blocked_key=["haadf"] * 3)
    value = result(incident, [reference])
    value.specimen_exit = GeometricSpecimenExit(
        (detailed,), {"tracked_downstream_source_probability": 1.,
                      "inelastic_absorbed_source_probability": 0.}, "current")
    data = sample_plane_cutoff_events(value, 20.)
    assert len(data.groups) == 1 and data.groups[0].provenance == "Specimen exit"
    np.testing.assert_array_equal(data.groups[0].stop_z_mm, [14.] * 3)
    value.specimen_exit = GeometricSpecimenExit(
        (), {"tracked_downstream_source_probability": 0.,
             "inelastic_absorbed_source_probability": 1.}, "current")
    assert sample_plane_cutoff_events(value, 20.).groups == ()


def test_incident_boundary_stop_not_duplicated_in_reference():
    incident = branch(count=1, blocked_z=np.array([10.]))
    reference = branch(count=1, z=(10., 20.), name="reference", blocked_z=np.array([10.]))
    data = sample_plane_cutoff_events(result(incident, [reference]), 20.)
    assert data.total_recorded_count == 1
    assert data.groups[0].provenance == "Incident"


def test_downstream_siblings_sharing_source_ids_remain_distinct_path_representatives():
    incident = branch(count=1, blocked_z=np.array([np.nan]))
    first = branch(count=1, z=(10., 20.), name="first", blocked_z=np.array([15.]))
    second = branch(count=1, z=(10., 20.), name="second", blocked_z=np.array([15.]))
    data = sample_plane_cutoff_events(result(incident, [first, second]), 20.)
    assert data.total_recorded_count == 2
    assert len(set(data.groups[0].event_id)) == 2
    np.testing.assert_array_equal(data.groups[0].source_ray_id, [10, 10])


@pytest.mark.parametrize("changes,reason", [
    ({"blocked_key": []}, "invalid or unavailable"),
    ({"blocked_z": np.array([2., 4.])}, "invalid or unavailable"),
    ({"z": np.array([10., 0.])}, "invalid or unavailable"),
    ({"x": np.zeros((2, 2))}, "invalid or unavailable"),
    ({"blocked_key": ["", None, "a"]}, "missing stop keys"),
    ({"blocked_z": np.array([-1., 4., 11.])}, "outside retained history"),
    ({"blocked_z": np.array([np.inf, np.nan, 8.])}, "non-finite interception Z"),
    ({"x": np.full((2, 3), np.nan)}, "non-finite position"),
])
def test_invalid_metadata_never_extrapolates_or_fabricates_stops(changes, reason):
    data = sample_plane_cutoff_events(result(branch(**changes)), 20.)
    assert any(reason in item for item in data.diagnostics)
    assert all(np.all(np.isfinite(group.x_m)) and np.all(np.isfinite(group.y_m)) for group in data.groups)


def test_missing_lineage_remains_unknown_instead_of_manufacturing_source_ids():
    data = sample_plane_cutoff_events(result(branch(source_ray_id=None)), 10.)
    np.testing.assert_array_equal(data.groups[0].source_ray_id, [-1, -1, -1])
    assert any("source lineage unavailable" in item for item in data.diagnostics)


def test_sampling_has_global_budget_and_stable_ids_when_z_moves():
    value = result(branch(count=100))
    early = sample_plane_cutoff_events(value, 6., max_events=7)
    late = sample_plane_cutoff_events(value, 20., max_events=7)
    assert late.displayed_count == 7 and late.total_recorded_count == 100
    assert late.groups[0].observed_count == 100
    assert set(early.groups[0].event_id).issubset(set(late.groups[0].event_id))
    np.testing.assert_array_equal(late.groups[0].column_index, np.linspace(0, 99, 7, dtype=int))
    again = sample_plane_cutoff_events(value, 20., max_events=7)
    np.testing.assert_array_equal(again.groups[0].x_m, late.groups[0].x_m)


def test_zero_weight_support_probes_are_omitted():
    data = sample_plane_cutoff_events(result(branch(ray_weight=np.array([1., 0., 0.]))), 10.)
    assert data.total_recorded_count == 1
    np.testing.assert_array_equal(data.groups[0].source_ray_id, [10])


def test_arrays_are_immutable_and_inputs_are_never_modified():
    incident = branch()
    original = {key: getattr(incident, key).copy() for key in ("x", "y", "blocked_z", "source_ray_id")}
    data = sample_plane_cutoff_events(result(incident), 10.)
    for key, values in original.items():
        np.testing.assert_array_equal(getattr(incident, key), values)
    for key in ("x_m", "y_m", "stop_z_mm", "source_ray_id", "column_index", "event_id"):
        values = getattr(data.groups[0], key)
        with pytest.raises(ValueError):
            values.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        data.groups[0].observed_count = 5


def test_cache_has_bounded_entries_and_explicit_invalidation():
    cache = PlaneCutoffEventsCache(max_events=3)
    value = result()
    first = cache.sample(value, 10.)
    assert cache.sample(value, 10.) is first
    for z in range(11, 20):
        cache.sample(value, z)
    assert len(cache._entries) == 4
    different = result(branch(blocked_key=["haadf"] * 3))
    changed = cache.sample(different, 10.)
    assert len(cache._entries) == 1 and changed is not first
    cache.clear()
    assert not cache._entries and cache._result is None


def test_new_observation_z_never_reextracts_or_interpolates_recorded_stops(monkeypatch):
    import temsim.gui.plane_cutoff_events as module
    incident = branch(count=100)
    value = result(incident)
    cache = PlaneCutoffEventsCache(max_events=7)
    reference = {z: sample_plane_cutoff_events(value, z, max_events=7)
                 for z in (2., 6., 11.)}
    cache.sample(value, 1.)

    def forbidden(*args, **kwargs):
        raise AssertionError("Moving observation Z must not re-read/interpolate ray histories")

    monkeypatch.setattr(module, "_branch_events", forbidden)
    for z, expected in reference.items():
        actual = cache.sample(value, z)
        assert actual.total_recorded_count == expected.total_recorded_count
        assert actual.displayed_count == expected.displayed_count
        assert actual.diagnostics == expected.diagnostics
        assert len(actual.groups) == len(expected.groups)
        for new, old in zip(actual.groups, expected.groups):
            assert (new.blocked_key, new.provenance, new.stop_kind, new.observed_count) == (
                old.blocked_key, old.provenance, old.stop_kind, old.observed_count)
            for key in ("x_m", "y_m", "stop_z_mm", "source_ray_id", "column_index", "event_id"):
                np.testing.assert_array_equal(getattr(new, key), getattr(old, key))


def test_cache_clear_reextracts_same_published_result_object(monkeypatch):
    import temsim.gui.plane_cutoff_events as module
    value = result()
    cache = PlaneCutoffEventsCache()
    extract = module._branch_events
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return extract(*args, **kwargs)

    monkeypatch.setattr(module, "_branch_events", counted)
    cache.sample(value, 3.)
    cache.sample(value, 7.)
    assert len(calls) == 1
    value.simulation.incident.x += 3.e-6
    cache.clear()
    data = cache.sample(value, 7.)
    assert len(calls) == 2
    np.testing.assert_allclose(data.groups[0].x_m, [7.e-6, 12.e-6])


@pytest.mark.parametrize("budget", [0, -1, 16385, True, 1.5])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError, match="display budget"):
        sample_plane_cutoff_events(result(), 10., max_events=budget)


def test_no_result_or_nonfinite_plane_reports_unavailable():
    assert sample_plane_cutoff_events(None, 10.).groups == ()
    assert sample_plane_cutoff_events(result(), np.nan).groups == ()


def test_module_does_not_import_execution_or_calculation_entry_points():
    import temsim.gui.plane_cutoff_events as module
    import ast
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("simulation_pipeline" in name or "calculation" in name or "tracing" in name
                   for name in imported)
