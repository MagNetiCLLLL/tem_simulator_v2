"""Executed diagnostic reference reuse never replaces physical beam transport."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics import energy_filter_raytrace as tracing


@pytest.fixture
def filter_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    # Small actual curved-filter fixture; integration refinement is not the
    # purpose of these cache tests and the production default is unchanged.
    state.energy_filter.ray_step_mm = 2.0
    return state


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    from temsim import calculation_manifest
    # Parallel workspace development must not spuriously invalidate a test
    # halfway through. A separate test checks this implementation dependency.
    monkeypatch.setattr(calculation_manifest, "solver_source_identity", lambda: "test-solver")
    with tracing._REFERENCE_CACHE_LOCK:
        tracing._REFERENCE_CACHE.clear()
    yield
    with tracing._REFERENCE_CACHE_LOCK:
        tracing._REFERENCE_CACHE.clear()


def assert_same_batch(actual, expected):
    for field in fields(tracing.EnergyFilterTraceBatch):
        first, second = getattr(actual, field.name), getattr(expected, field.name)
        if first is None or second is None:
            assert first is second
        else:
            assert first.dtype == second.dtype
            np.testing.assert_array_equal(first, second, err_msg=field.name)


def test_actual_reference_cache_matches_uncached_positions_interceptions_and_clocks(filter_state, monkeypatch):
    state = filter_state
    original = tracing.trace_energy_filter_batch
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(tracing, "trace_energy_filter_batch", counted)
    cold = tracing._trace_energy_filter_reference(state)
    warm = tracing._trace_energy_filter_reference(state)
    assert calls == [1]
    fresh = tracing._trace_energy_filter_reference(state, use_cache=False)
    assert calls == [1, 1]
    assert_same_batch(warm, fresh)
    assert_same_batch(cold, fresh)
    assert warm.time_s.dtype == np.float64
    assert warm.time_s[-1] > 0
    for field in fields(tracing.EnergyFilterTraceBatch):
        a, b = getattr(cold, field.name), getattr(warm, field.name)
        if a is not None:
            assert not np.shares_memory(a, b)
    cold.positions_m[:] = 1.
    warm.time_s[:] = 10.
    warm.stop_key[:] = "mutated consumer"
    assert_same_batch(tracing._trace_energy_filter_reference(state), fresh)
    assert calls == [1, 1]


def _small_batch():
    """Counting-only fixture; the actual numerical parity test is above."""
    arrays = {}
    for field in fields(tracing.EnergyFilterTraceBatch):
        arrays[field.name] = (np.zeros((2, 1, 3)) if field.name == "positions_m"
                             else np.array([""]) if field.name == "stop_key"
                             else np.zeros(1))
    return tracing.EnergyFilterTraceBatch(**arrays)


@pytest.mark.parametrize("change", [
    lambda s: setattr(s, "_propagation_energy_kev", 290.),
    lambda s: setattr(s.energy_filter, "ray_step_mm", 1.5),
    lambda s: setattr(s.energy_filter, "prism_radius_mm", s.energy_filter.prism_radius_mm + .1),
    lambda s: setattr(s.energy_filter, "alignment_x_mrad", .1),
    lambda s: s.energy_filter.multipoles[0].multipole_field.set_component(2, normal=.25),
    lambda s: setattr(s.energy_filter.multipoles[0].field_backend, "fringe_expansion_order", 1),
    lambda s: setattr(s.energy_filter.multipoles[0], "frame", replace(
        s.energy_filter.multipoles[0].frame,
        origin_m=s.energy_filter.multipoles[0].frame.origin_m + [0., .001, 0.])),
    lambda s: setattr(s.energy_filter.multipoles[0].field_backend, "envelope", replace(
        s.energy_filter.multipoles[0].field_backend.envelope, length_m=.025)),
    lambda s: setattr(s.energy_filter.energy_slit, "gap_m", .0001),
    lambda s: setattr(s.energy_filter.fast_shutter, "open", False),
    lambda s: setattr(s.energy_filter, "operating_mode", "eftem"),
    lambda s: setattr(s.energy_filter.bias_tube, "offset_ev", 10.),
], ids=["voltage", "step", "sector-geometry", "alignment", "actual-field", "fringe-order",
        "actual-frame", "field-support", "slit", "shutter", "detector", "bias"])
def test_consumed_input_changes_require_a_new_executed_reference(filter_state, monkeypatch, change):
    state = filter_state
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return _small_batch()
    monkeypatch.setattr(tracing, "trace_energy_filter_batch", counted)
    tracing._trace_energy_filter_reference(state)
    tracing._trace_energy_filter_reference(state)
    assert len(calls) == 1
    change(state)
    tracing._trace_energy_filter_reference(state)
    assert len(calls) == 2


def test_model_change_invalidates_and_custom_field_is_never_cached(filter_state, monkeypatch):
    from temsim import calculation_manifest
    state = filter_state
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return _small_batch()
    monkeypatch.setattr(tracing, "trace_energy_filter_batch", counted)
    tracing._trace_energy_filter_reference(state)
    monkeypatch.setattr(calculation_manifest, "solver_source_identity", lambda: "new-solver")
    tracing._trace_energy_filter_reference(state)
    assert len(calls) == 2
    state.energy_filter.multipoles[0].field_backend.field_at_local_positions_t = lambda p: p
    assert tracing._reference_cache_key(state) is None
    tracing._trace_energy_filter_reference(state)
    tracing._trace_energy_filter_reference(state)
    assert len(calls) == 4


def test_cache_is_bounded_and_failed_execution_is_not_retained(filter_state, monkeypatch):
    state = filter_state
    monkeypatch.setattr(tracing, "_REFERENCE_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(tracing, "trace_energy_filter_batch", lambda *a, **k: _small_batch())
    for angle in (0., .1, .2):
        state.energy_filter.alignment_x_mrad = angle
        tracing._trace_energy_filter_reference(state)
    assert len(tracing._REFERENCE_CACHE) == 2
    monkeypatch.setattr(tracing, "_REFERENCE_CACHE_MAX_BYTES", 1)
    with tracing._REFERENCE_CACHE_LOCK:
        tracing._REFERENCE_CACHE.clear()
    tracing._trace_energy_filter_reference(state)
    assert not tracing._REFERENCE_CACHE
    def failed(*args, **kwargs):
        raise RuntimeError("interrupted reference")
    monkeypatch.setattr(tracing, "trace_energy_filter_batch", failed)
    with pytest.raises(RuntimeError, match="interrupted reference"):
        tracing._trace_energy_filter_reference(state)
    assert not tracing._REFERENCE_CACHE


def test_simulation_reuses_only_reference_and_retraces_physical_population(filter_state, monkeypatch):
    state = filter_state
    z = state.energy_filter.entrance_z_mm
    branch = SimpleNamespace(name="incident fixture", colour="#fff", z=np.array([z-1., z+1.]),
        x=np.zeros((2, 2)), y=np.zeros((2, 2)), tx=np.zeros((2, 2)), ty=np.zeros((2, 2)),
        blocked_z=np.full(2, np.nan), alive=np.ones(2, bool), weight=1.,
        ray_weight=np.array([.5, .5]), energy_offset_ev=np.zeros(2),
        source_ray_id=np.arange(2), source_azimuth_rad=np.full(2, np.nan),
        flight_time_s=np.array([[1., 2.], [3., 4.]])*1e-9)
    simulation = SimpleNamespace(incident=None, branches={"fixture": branch},
        metrics={"branch_weights_are_absolute": True})
    calls = []
    original = tracing.trace_energy_filter_batch
    def counted(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)
    monkeypatch.setattr(tracing, "trace_energy_filter_batch", counted)
    first = tracing.simulate_energy_filter(state, simulation)
    second = tracing.simulate_energy_filter(state, simulation)
    assert calls == [1, 2, 2]
    for a, b in zip(first.paths_u_mm, second.paths_u_mm):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first.timed_planes, second.timed_planes):
        for name in ("time_s", "x_m", "y_m", "tx_rad", "ty_rad", "reached"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
