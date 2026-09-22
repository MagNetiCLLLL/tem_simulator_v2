"""Offline lifecycle checks: section requests must not alias full previews."""
from types import SimpleNamespace

import pytest


def test_section_target_and_components_isolate_exact_preview_cache(qtbot, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim import simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    state = default_state()
    calls = []

    def trace(snapshot, **kwargs):
        calls.append(kwargs)
        metrics = {"tuning_quality": "Preview"}
        if "observation_stop_z_mm" in kwargs:
            metrics.update(section_target_z_mm=kwargs["observation_stop_z_mm"],
                           section_component_keys=kwargs["tuning_component_keys"])
        return SimpleNamespace(incident=object(), branches={}, metrics=metrics)

    monkeypatch.setattr(module, "run_ray_simulation", trace)
    def physical(snapshot, **kwargs):
        simulation = trace(snapshot, observation_stop_z_mm=kwargs["target_z_mm"],
                           tuning_component_keys=kwargs["component_keys"])
        return pipeline.CalculationResult(simulation=simulation, state_snapshot=snapshot,
                                          energy_filter=None, signatures={})
    monkeypatch.setattr(pipeline, "calculate_particle_section", physical)
    monkeypatch.setattr("temsim.detector.particle_readout.measure_particle_detectors", lambda result: ())
    monkeypatch.setattr(module, "detect_all_lens_crossovers", lambda *a: ())
    controller = module.CalculationController(persistent_cache_enabled=False)
    workers, results = [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.result_ready.connect(lambda _q, result, _time: results.append(result))

    def submit(section=None):
        previous_workers = len(workers)
        before = len(results)
        controller.submit(state, "Preview", 49, 1., section_request=section)
        if len(workers) > previous_workers:
            workers[-1].run()
        qtbot.waitUntil(lambda: len(results) > before)

    submit()
    first = {"target_z_mm": state.sample.z_mm, "component_keys": ("objective_lens",)}
    submit(first)
    assert len(calls) == 2
    assert "observation_stop_z_mm" not in calls[0]
    assert calls[1]["observation_stop_z_mm"] == state.sample.z_mm
    submit(first)
    assert len(calls) == 2 and results[-1].cache_hit
    submit(dict(first, target_z_mm=state.sample.z_mm - 5.))
    submit(dict(first, component_keys=("condenser_lens_2",)))
    assert len(calls) == 4
    assert results[-1].simulation.metrics["section_component_keys"] == ("condenser_lens_2",)
    submit()
    assert len(calls) == 4
    assert "section_target_z_mm" not in results[-1].simulation.metrics


def test_background_capture_detaches_section_options(qtbot, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim.optics.column import default_state
    state = default_state()
    controller = module.CalculationController(persistent_cache_enabled=False)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    request = {"target_z_mm": state.sample.z_mm, "component_keys": ["objective_lens"]}
    controller.submit_background(state, "Preview", 49, 1., section_request=request)
    captured = controller._requests[controller.generation]["section_request"]
    request["target_z_mm"] += 100.
    request["component_keys"].append("condenser_lens_2")
    assert captured == {"target_z_mm": state.sample.z_mm, "component_keys": ("objective_lens",)}
    controller.invalidate_pending()
    for worker in workers:
        worker.release_inputs()


def test_high_accuracy_section_keeps_requested_budget_and_needs_no_tuning_ranges(qtbot, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim import simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    state = default_state()
    owner = module.CalculationController(persistent_cache_enabled=False)
    workers, results, observed = [], [], []
    monkeypatch.setattr(owner.pool, "start", workers.append)
    monkeypatch.setattr(module, "prepare_tuning_snapshot", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("High accuracy must keep the requested budget")))
    def section(snapshot, **kwargs):
        observed.append((snapshot.electron_gun.ray_count, snapshot.step_mm, kwargs))
        return pipeline.CalculationResult(simulation=SimpleNamespace(metrics={"tuning_quality":"High accuracy"}),
                                          state_snapshot=snapshot, energy_filter=None, signatures={})
    monkeypatch.setattr(pipeline, "calculate_particle_section", section)
    monkeypatch.setattr("temsim.detector.particle_readout.measure_particle_detectors", lambda result: ())
    owner.result_ready.connect(lambda _q, result, _t: results.append(result))
    owner.submit(state, "High accuracy", 5000, .2, section_request={
        "target_z_mm": state.sample.z_mm, "component_keys": ()})
    assert len(workers) == 1
    workers[0].run()
    qtbot.waitUntil(lambda: len(results) == 1)
    assert observed[0][:2] == (5000, .2)
    assert observed[0][2]["component_keys"] == ()
    assert results[0].simulation.metrics["tuning_quality"] == "High accuracy"


def test_physical_preview_preserves_requested_scan_and_cannot_hit_optical_cache(qtbot, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim import simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    state = default_state()
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    calls = []
    def physical(snapshot, **kwargs):
        assert snapshot._particle_tuning and not snapshot._optical_tuning
        assert snapshot.ac_deflector.scan_enabled
        assert not snapshot.sample.wave_enabled and not snapshot.sample.stem_wave_enabled
        calls.append("physical")
        return pipeline.CalculationResult(
            simulation=SimpleNamespace(incident=object(), branches={},
                metrics={"particle_tuning":True,"tuning_quality":"Preview"}),
            state_snapshot=snapshot, energy_filter=None,
            signatures={"request":"inner scoped physical request", "sample_downstream":"bounded material"})
    def optical(snapshot, **kwargs):
        assert not snapshot.ac_deflector.scan_enabled
        calls.append("optical")
        return SimpleNamespace(incident=object(), branches={}, metrics={"tuning_quality":"Preview"})
    monkeypatch.setattr(pipeline, "calculate_particle_section", physical)
    monkeypatch.setattr(module, "run_ray_simulation", optical)
    monkeypatch.setattr(module, "detect_all_lens_crossovers", lambda *a: ())
    monkeypatch.setattr("temsim.detector.particle_readout.measure_particle_detectors", lambda result: ("measured",))
    controller = module.CalculationController(persistent_cache_enabled=False)
    workers, results = [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.result_ready.connect(lambda q, result, t: results.append(result))
    for particle in (False, True, True):
        count = len(workers)
        delivered = len(results)
        controller.submit(state, "Preview", 49, 1., particle_tuning=particle)
        if len(workers) > count:
            workers[-1].run()
        qtbot.waitUntil(lambda: len(results)>delivered)
    assert calls == ["optical", "physical"]
    assert results[-1].cache_hit
    assert results[-1].particle_signals == ("measured",)
    assert results[-1].signatures["sample_downstream"] == "bounded material"
    assert results[0].signatures["request"] != results[-1].signatures["request"]
    assert state.ac_deflector.scan_enabled
