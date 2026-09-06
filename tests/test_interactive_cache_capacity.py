"""Cache ownership and exact live-tuning reuse; no specimen solve required."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.artifact_store import ArtifactStore
from temsim.cache_memory import RetainedMemoryLedger, estimate_result_cache_bytes
from temsim.gui.calculation_controller import CalculationController
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult


def result(key, values=None):
    return CalculationResult(
        simulation=SimpleNamespace(payload=np.zeros(64) if values is None else values),
        energy_filter=None,
        signatures={"request": key},
    )


def complete_worker(controller, worker):
    completed = result(worker.request_signatures["request"])
    completed.model_signature = worker.model_signature
    completed.state_snapshot = worker.state
    completed.signatures = worker.request_signatures
    controller._accept_result(worker.generation, worker.quality, completed, .1)
    controller._accept_finished(worker.generation, worker.quality)
    return completed


def test_revisiting_tuning_settings_uses_exact_history_without_worker(qtbot):
    controller = CalculationController(persistent_cache_enabled=False)
    workers = []
    controller.pool.start = workers.append
    state = default_state()
    original_percent = state.objective_lens.percent
    controller.submit(state, "Preview", 49, 1.)
    original = complete_worker(controller, workers[-1])
    state.objective_lens.percent += .1
    controller.submit(state, "Preview", 49, 1.)
    complete_worker(controller, workers[-1])
    state.objective_lens.percent = original_percent
    with qtbot.waitSignal(controller.result_ready, timeout=1_000) as received:
        controller.submit(state, "Preview", 49, 1.)
    quality, cached, duration = received.args
    assert len(workers) == 2
    assert quality == "Preview" and duration == 0
    assert cached.cache_hit and cached.simulation is original.simulation
    assert not original.cache_hit
    assert controller.completed_high_accuracy_results() == ()
    assert controller.cache_statistics()["tuning_hits"] == 1
    assert controller.cache_statistics()["tuning_misses"] == 2


@pytest.mark.parametrize("change", ("quality", "step", "rays", "aperture", "geometry"))
def test_tuning_cache_never_reuses_different_request(change):
    controller = CalculationController(persistent_cache_enabled=False)
    workers = []
    controller.pool.start = workers.append
    state = default_state()
    controller.submit(state, "Preview", 49, 1.)
    complete_worker(controller, workers[-1])
    quality, rays, step = "Preview", 49, 1.
    if change == "quality":
        quality = "Medium"
    elif change == "rays":
        rays = 53
    elif change == "step":
        step = .75
    elif change == "aperture":
        state.condenser_aperture_2.radius_mm *= .75
    else:
        state.objective_lens.cs_mm += .1
    controller.submit(state, quality, rays, step)
    assert len(workers) == 2


def test_stale_queued_tuning_hit_is_suppressed(qtbot):
    controller = CalculationController(persistent_cache_enabled=False)
    workers, delivered = [], []
    controller.pool.start = workers.append
    state = default_state()
    controller.submit(state, "Preview", 49, 1.)
    complete_worker(controller, workers[-1])
    controller.result_ready.connect(lambda *args: delivered.append(args))
    controller.submit(state, "Preview", 49, 1.)
    controller.invalidate_pending()
    qtbot.wait(20)
    assert delivered == []


def test_configure_evicts_lru_without_changing_display_or_job():
    controller = CalculationController(persistent_cache_enabled=False)
    first, second = result("first"), result("second")
    controller._cache_result(first)
    controller._cache_result(second)
    controller._cache_tuning_result("Preview", first)
    controller._cache_tuning_result("Preview", second)
    generation, cancel_event = controller.generation, controller._cancel_event
    controller.configure_cache(high_cache_limit=1, tuning_cache_limit=1)
    assert controller.completed_high_accuracy_results() == (second,)
    assert tuple(controller._tuning_cache.values()) == (second,)
    np.testing.assert_array_equal(first.simulation.payload, np.zeros(64))
    assert controller.generation == generation
    assert controller._cancel_event is cancel_event and not cancel_event.is_set()
    controller.configure_cache(high_cache_budget_bytes=0, tuning_cache_budget_bytes=0)
    assert controller.completed_high_accuracy_results() == ()
    assert controller._tuning_seeds == {}
    assert controller.cache_statistics()["high_used_bytes"] == 0
    assert controller.cache_statistics()["tuning_used_bytes"] == 0


def test_tuning_lru_counts_shared_storage_and_releases_seed_references():
    owner = np.zeros(100_000)
    first, second = result("first", owner), result("second", owner[::2])
    budget = estimate_result_cache_bytes(first, second)
    controller = CalculationController(persistent_cache_enabled=False, tuning_cache_budget_bytes=budget)
    controller._cache_tuning_result("Preview", first)
    controller._cache_tuning_result("Preview", second)
    assert controller.cache_statistics()["tuning_entries"] == 2
    assert controller.cache_statistics()["tuning_used_bytes"] == budget
    controller.configure_cache(tuning_cache_budget_bytes=estimate_result_cache_bytes(second))
    assert tuple(controller._tuning_cache.values()) == (second,)
    assert controller._tuning_seeds == {"Preview": second}


def test_oversized_tuning_result_keeps_existing_history():
    controller = CalculationController(persistent_cache_enabled=False, tuning_cache_budget_bytes=16_384)
    first = result("first")
    controller._cache_tuning_result("Preview", first)
    controller._cache_tuning_result("Preview", result("too-big", np.zeros(16_384)))
    assert tuple(controller._tuning_cache.values()) == (first,)


def test_invalid_configuration_is_atomic():
    controller = CalculationController(persistent_cache_enabled=False)
    before = controller.cache_statistics()
    for values in ({"high_cache_budget_bytes": -1}, {"tuning_cache_limit": 0},
                   {"high_cache_limit": True}, {"tuning_cache_budget_bytes": float("nan")},
                   {"high_cache_limit": 1, "disk_cache_budget_bytes": 0}):
        with pytest.raises(ValueError):
            controller.configure_cache(**values)
        assert controller.cache_statistics() == before


def test_disk_quota_change_does_not_scan_or_prune(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts", quota_bytes=1024**2)
    controller = CalculationController(artifact_store=store)
    monkeypatch.setattr(store, "_store_size", lambda: pytest.fail("Synchronous disk scan"))
    monkeypatch.setattr(store, "_prune_to_quota", lambda **_kw: pytest.fail("Synchronous deletion"))
    controller.configure_cache(disk_cache_budget_bytes=16 * 1024**3)
    assert controller.cache_statistics()["disk_budget_bytes"] == 16 * 1024**3


def test_buffer_views_and_object_arrays_are_counted_once():
    buffer = bytearray(80_000)
    first = np.frombuffer(buffer, dtype=np.float64)
    second = np.frombuffer(buffer, dtype=np.uint8)
    separate = estimate_result_cache_bytes(first) + estimate_result_cache_bytes(second)
    together = estimate_result_cache_bytes(first, second)
    assert separate - together >= len(buffer)
    assert together >= len(buffer)
    payload = bytearray(40_000)
    objects = np.empty(2, dtype=object)
    objects[:] = [payload, payload]
    assert estimate_result_cache_bytes(objects) >= len(payload) + objects.nbytes


def test_incremental_ledger_matches_full_estimate_for_shared_results():
    owner = np.zeros(100_000)
    first, second = result("first", owner), result("second", owner[1:])
    ledger = RetainedMemoryLedger()
    ledger.replace("first", first)
    ledger.replace("second", second)
    assert ledger.total_bytes == estimate_result_cache_bytes(first, second)
    ledger.remove("first")
    assert ledger.total_bytes == estimate_result_cache_bytes(second)
    ledger.clear()
    assert ledger.total_bytes == 0


def test_statistics_do_not_rescan_result_graph(monkeypatch):
    controller = CalculationController(persistent_cache_enabled=False)
    controller._cache_result(result("high"))
    controller._cache_tuning_result("Preview", result("preview"))
    monkeypatch.setattr("temsim.cache_memory.retained_memory_inventory",
                        lambda *_values: pytest.fail("Cache inventory rescanned"))
    assert controller.cache_statistics()["high_used_bytes"] > 0
    assert controller.cache_statistics()["tuning_used_bytes"] > 0
    controller.configure_cache(high_cache_limit=1, tuning_cache_limit=1)
