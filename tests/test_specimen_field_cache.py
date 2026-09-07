"""Exact scalar field reuse; seeded transport fixtures are not OEM validation."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.simulation_modes import switch_mode
import temsim.specimen.elastic_transport as elastic_module
import temsim.specimen.vector_field_transport as field_module
from temsim.specimen.interaction_types import IncidentElectronRay
from temsim.specimen.vector_field_transport import SpecimenFieldTransport


def _context(monkeypatch, *, size=8, state=None):
    if state is None:
        state = SimpleNamespace(
            beam_voltage_kv=300.0, sample=SimpleNamespace(z_mm=1.0),
            lenses=[SimpleNamespace(key="synthetic", enabled=True)],
            stigmators=[], corrector_elements=[], strength=0.5,
        )

    def provider(_state, _key, _native):
        strength = state.strength
        return SimpleNamespace(
            field_support_mm=lambda: (-10.0, 10.0),
            field_at_global_positions_t=lambda p: np.asarray(p) + strength,
        )

    monkeypatch.setattr(field_module, "resolve_runtime_lens_field_provider", provider)
    transport = SpecimenFieldTransport(state, field_cache_size=size)
    uncached = Mock(wraps=transport._field_at_global_positions_t_uncached)
    monkeypatch.setattr(transport, "_field_at_global_positions_t_uncached", uncached)
    return transport, uncached, state


def test_scalar_hit_returns_owned_copies_and_does_not_retain_input(monkeypatch):
    transport, uncached, _state = _context(monkeypatch)
    point = np.array((1e-9, 2e-9, 3e-9))
    original_point = point.copy()
    first = transport.field_at_global_positions_t(point)
    expected = first.copy()
    first[:] = -100.0
    point[:] = 1.0
    second = transport.field_at_global_positions_t(original_point)
    np.testing.assert_array_equal(second, expected)
    second[:] = 100.0
    third = transport.field_at_global_positions_t(original_point)
    np.testing.assert_array_equal(third, expected)
    assert not np.shares_memory(second, third)
    assert uncached.call_count == 1
    assert transport.field_cache_info() == {
        "capacity": 8, "entries": 1, "hits": 2, "misses": 1,
    }


def test_keys_keep_nextafter_and_signed_zero_distinct(monkeypatch):
    transport, uncached, _state = _context(monkeypatch)
    for x in (0.0, -0.0, np.nextafter(0.0, 1.0)):
        transport.field_at_global_positions_t((x, 0.0, 0.0))
    assert uncached.call_count == 3
    assert transport.field_cache_info()["entries"] == 3


def test_lru_retains_at_most_eight_scalar_entries(monkeypatch):
    transport, uncached, _state = _context(monkeypatch)
    for i in range(8):
        transport.field_at_global_positions_t((float(i), 0.0, 0.0))
    transport.field_at_global_positions_t((0.0, 0.0, 0.0))
    transport.field_at_global_positions_t((8.0, 0.0, 0.0))
    transport.field_at_global_positions_t((0.0, 0.0, 0.0))
    assert uncached.call_count == 9
    transport.field_at_global_positions_t((1.0, 0.0, 0.0))
    assert uncached.call_count == 10
    assert transport.field_cache_info()["entries"] == 8
    assert all(array.shape == (3,) for array in transport._field_cache.values())
    assert all(not array.flags.writeable for array in transport._field_cache.values())


def test_batches_bypass_scalar_cache_without_retaining_large_arrays(monkeypatch):
    transport, uncached, _state = _context(monkeypatch)
    batch = np.zeros((1024, 3))
    first = transport.field_at_global_positions_t(batch)
    second = transport.field_at_global_positions_t(batch)
    np.testing.assert_array_equal(first, second)
    assert first.shape == batch.shape
    assert uncached.call_count == 2
    assert transport.field_cache_info()["entries"] == 0


def test_zero_capacity_preserves_uncached_reference(monkeypatch):
    transport, uncached, _state = _context(monkeypatch, size=0)
    first = transport.field_at_global_positions_t((0.0, 0.0, 0.0))
    second = transport.field_at_global_positions_t((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(first, second)
    assert uncached.call_count == 2
    assert transport.field_cache_info() == {
        "capacity": 0, "entries": 0, "hits": 0, "misses": 0,
    }


def test_new_calculation_context_does_not_reuse_previous_state_fields(monkeypatch):
    previous, _uncached, state = _context(monkeypatch)
    point = (0.0, 0.0, 0.0)
    before = previous.field_at_global_positions_t(point)
    state.strength = 0.75
    current, uncached, _state = _context(monkeypatch, state=state)
    after = current.field_at_global_positions_t(point)
    assert not np.array_equal(before, after)
    assert uncached.call_count == 1
    assert current._field_cache is not previous._field_cache
    assert current.field_cache_info()["hits"] == 0


def test_failed_evaluation_is_not_cached(monkeypatch):
    transport, uncached, _state = _context(monkeypatch)
    uncached.side_effect = ValueError("Invalid field fixture")
    for _ in range(2):
        with pytest.raises(ValueError, match="Invalid field fixture"):
            transport.field_at_global_positions_t((0.0, 0.0, 0.0))
    assert uncached.call_count == 2
    assert transport.field_cache_info()["entries"] == 0


@pytest.mark.parametrize("size", [-1, 9, 1000, 1.5, True])
def test_invalid_cache_capacity_is_rejected_before_resolving_fields(size):
    with pytest.raises(ValueError, match="integer from 0 to 8"):
        SpecimenFieldTransport(None, field_cache_size=size)


def _assert_exact_transport(actual, expected):
    assert actual.metrics == expected.metrics
    assert actual.eds_tracks == expected.eds_tracks
    assert actual.material_flights == expected.material_flights
    for field in fields(actual.terminal_electrons):
        left = getattr(actual.terminal_electrons, field.name)
        right = getattr(expected.terminal_electrons, field.name)
        np.testing.assert_array_equal(left, right)
    for left, right in zip(actual.trajectories, expected.trajectories, strict=True):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
def test_seeded_elastic_histories_match_uncached_fields_exactly(monkeypatch, mode):
    state = default_state()
    switch_mode(state, mode)
    state.sample.thickness_nm = 100.0
    state.sample.eds_support_material_key = "vacuum"
    unit = np.array((0.015, -0.01, 1.0))
    unit /= np.linalg.norm(unit)
    rays = tuple(
        IncidentElectronRay(
            i, (i * 0.02, i * -0.03), tuple(unit), 300_000.0 + i * 0.001, 1 / 16,
        )
        for i in range(16)
    )
    contexts = []

    def cached_context(snapshot):
        context = SpecimenFieldTransport(snapshot, field_cache_size=8)
        contexts.append(context)
        return context

    monkeypatch.setattr(elastic_module, "SpecimenFieldTransport", cached_context)
    actual = elastic_module.simulate_elastic_point_transport(
        state, incident_rays=rays, seed=101, stored_trajectory_count=16,
    )
    assert actual.metrics["total_elastic_events"] > 0
    assert contexts[0].field_cache_info()["hits"] > 0
    monkeypatch.setattr(
        elastic_module, "SpecimenFieldTransport",
        lambda snapshot: SpecimenFieldTransport(snapshot, field_cache_size=0),
    )
    expected = elastic_module.simulate_elastic_point_transport(
        state, incident_rays=rays, seed=101, stored_trajectory_count=16,
    )
    _assert_exact_transport(actual, expected)
