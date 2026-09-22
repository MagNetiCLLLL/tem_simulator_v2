"""Cancellation must reach the physical gun without publishing partial work."""

from collections import OrderedDict

from temsim.gui.calculation_controller import CalculationWorker
from temsim.optics.column import default_state
from temsim.optics.electron_gun import field_emission, tracing


def test_high_accuracy_cancel_during_gun_transport_discards_partial_result(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.stem_wave_enabled = False
    state.ac_deflector.scan_enabled = False
    state.electron_gun.emitter.ray_count = 9
    cache = OrderedDict()
    monkeypatch.setattr(field_emission, "_SHARED_TRACE_CACHE", cache)

    worker = CalculationWorker(1, "High accuracy", state)
    results, errors, finished = [], [], []
    worker.signals.result.connect(lambda *args: results.append(args))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.signals.finished.connect(lambda *args: finished.append(args))

    integrated_steps = []
    original_analytic_step = tracing._analytic_step

    def cancel_after_integration(*args, **kwargs):
        integrated_steps.append(1)
        if len(integrated_steps) > 1:
            raise AssertionError("Gun integration ignored worker cancellation")
        result = original_analytic_step(*args, **kwargs)
        worker.cancel_event.set()
        return result

    monkeypatch.setattr(tracing, "_analytic_step", cancel_after_integration)
    monkeypatch.setattr(
        worker, "_persist_incident_seed",
        lambda *_args: results.append("unexpected partial persistence"),
    )
    worker.run()

    assert errors == []
    assert integrated_steps == [1], "Gun transport must stop after its cancelled step"
    assert not cache
    assert state.electron_gun._trace_cache is None
    assert state.electron_gun._trace_cache_key is None
    assert results == []
    assert finished == [(1, "High accuracy")]
    assert not hasattr(state, "_tuning_cancelled")
