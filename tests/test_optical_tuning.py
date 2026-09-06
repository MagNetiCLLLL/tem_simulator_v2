"""Ray-only tuning boundaries, numerical equivalence and UI cache isolation."""
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.gui.calculation_controller import CalculationController, CalculationWorker
from temsim.physics.optical_tuning import (
    TUNING_PROFILES, prepare_tuning_snapshot, projected_support,
)


def tuning_result(quality="Medium", state=None):
    source = state or default_state()
    p = TUNING_PROFILES[quality]
    snapshot = CalculationController._calculation_snapshot(source, quality, p.rays, p.step_mm)
    worker = CalculationWorker(1, quality, snapshot)
    results, errors = [], []
    worker.signals.result.connect(lambda _g, _q, r, _t: results.append(r))
    worker.signals.error.connect(lambda *a: errors.append(a))
    worker.run()
    assert not errors
    assert len(results) == 1
    return results[0]


@pytest.fixture(scope="module")
def medium_result():
    return tuning_result()


def test_medium_support_probes_preserve_weighted_source_distribution():
    source = default_state()
    p = TUNING_PROFILES["Medium"]
    s = CalculationController._calculation_snapshot(source, "Medium", p.rays, p.step_mm)
    original = s.electron_gun.emitter.emit()
    prepare_tuning_snapshot(s, "Medium")
    sampled = s.electron_gun.emitter.emit()
    assert sampled.weight.sum() == pytest.approx(1.)
    assert np.count_nonzero(sampled.weight == 0) == 33
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev"):
        np.testing.assert_array_equal(getattr(sampled, name)[:160], getattr(original, name)[:160])
    radii = np.hypot(sampled.tx_rad[160:-1], sampled.ty_rad[160:-1])
    np.testing.assert_allclose(radii, s.electron_gun.emitter.angular_cutoff_mrad*1e-3)
    assert not hasattr(source.electron_gun.emitter, "_tuning_boundary_probes")
    assert sampled.x_m[-1] == sampled.y_m[-1] == 0.


def test_tuning_does_not_call_specimen_or_scan_solvers(monkeypatch):
    import temsim.specimen.inelastic as inelastic
    import temsim.gui.calculation_controller as module
    import temsim.physics.scan_geometry as scan
    import temsim.simulation_pipeline as pipeline
    def forbidden(*a, **k):
        pytest.fail("Expensive signal calculation in optical tuning")
    monkeypatch.setattr(inelastic, "real_inelastic_distribution", forbidden)
    monkeypatch.setattr(module, "calculate", forbidden)
    monkeypatch.setattr(scan, "calculate_scan_geometry", forbidden)
    monkeypatch.setattr(scan, "calculate_scan_ray_paths", forbidden)
    monkeypatch.setattr(pipeline, "calculate_stem_scan_frame", forbidden)
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "must-not-load-during-tuning.cif"
    state.sample.wave_enabled = state.sample.stem_wave_enabled = True
    state.ac_deflector.scan_enabled = True
    result = tuning_result("Preview", state)
    assert result.wave_imaging is result.stem_scan is result.sample_region is None
    assert result.simulation.real_interactions is None
    assert result.simulation.metrics["sample_scattering_applied"] is False
    assert state.ac_deflector.scan_enabled and state.sample.wave_enabled


def test_medium_is_not_cached_as_a_high_accuracy_result(medium_result):
    controller = CalculationController(persistent_cache_enabled=False)
    controller._generation = 7
    controller._accept_result(7, "Medium", medium_result, .1)
    assert controller.completed_high_accuracy_results() == ()
    assert controller._tuning_seeds["Medium"] is medium_result
    controller._accept_result(6, "Medium", object(), .1)
    assert controller._tuning_seeds["Medium"] is medium_result


def test_live_objective_edit_updates_rays_and_reuses_only_valid_prefix():
    controller = CalculationController(persistent_cache_enabled=False)
    workers, results = [], []
    controller.pool.start = workers.append
    controller.result_ready.connect(lambda _q, r, _t: results.append(r))
    state = default_state()
    controller.submit(state, "Preview", 49, 1.)
    workers[-1].run()
    state.objective_lens.percent += .1
    controller.submit(state, "Preview", 49, 1.)
    workers[-1].run()
    assert len(results) == 2
    assert results[1].state_snapshot.objective_lens.percent == state.objective_lens.percent
    assert results[1].simulation.metrics["column_segment_cache"]["mode"] == "checkpoint"
    assert not np.allclose(results[0].simulation.incident.x[-1],
                           results[1].simulation.incident.x[-1], rtol=1e-8, atol=1e-15)
    assert controller.completed_high_accuracy_results() == ()


def test_superseded_tuning_worker_does_no_work(monkeypatch):
    import temsim.gui.calculation_controller as module
    monkeypatch.setattr(module, "run_ray_simulation", lambda *a, **k: pytest.fail("Cancelled work ran"))
    worker = CalculationWorker(1, "Preview", default_state())
    worker.cancel_event = Event()
    worker.cancel_event.set()
    results = []
    worker.signals.result.connect(lambda *a: results.append(a))
    worker.run()
    assert results == []


def test_support_guide_respects_rotation_and_physical_stops():
    b = SimpleNamespace(z=np.array([0., 1., 2.]),
        x=np.array([[0., .002], [0., .004], [0., .006]]),
        y=np.array([[.003, 0.], [.003, 0.], [.003, 0.]]), blocked_z=np.array([np.nan, 1.]))
    low, high = projected_support(b)
    np.testing.assert_allclose(low, 0)
    np.testing.assert_allclose(high, [2, 4, 0])
    low, high = projected_support(b, 90)
    np.testing.assert_allclose(high, 3)
    np.testing.assert_allclose(low, [0, 0, 3], atol=1e-12)


def test_serial_tuning_kernel_matches_numpy_and_falls_back(monkeypatch):
    from temsim.physics import core
    if not core.NUMBA_AVAILABLE:
        pytest.skip("Numba unavailable")
    state = default_state()
    for part in (*state.lenses, *state.stigmators, *state.corrector_elements):
        part.enabled = False
    state.step_mm = .1
    state.history_step_mm = .2
    monkeypatch.setattr(core, "fields", lambda z, _s: (np.full_like(z, .1), np.zeros_like(z), np.zeros_like(z)))
    plan = core.build_propagation_plan(state, 1500., 1503., checkpoint_z_mm=(1503.,))
    source = tuple(np.array(v) for v in ([2e-6, -1e-6], [.001, -.002], [-1e-6, 3e-6], [.002, .001]))
    state.acceleration_enabled = False
    reference = core.execute_propagation_plan(state, plan, *source)
    state.acceleration_enabled = True
    state.acceleration_backend = "Auto"
    state._optical_tuning = True
    accelerated = core.execute_propagation_plan(state, plan, *source)
    assert state._tuning_kernel == "serial_numba"
    for name in ("x_m", "y_m", "tx_rad", "ty_rad"):
        np.testing.assert_allclose(getattr(accelerated[5], name), getattr(reference[5], name), rtol=5e-12, atol=1e-17)
    def unavailable(*a):
        raise RuntimeError("offline JIT failure")
    monkeypatch.setattr(core, "_serial_rk4", unavailable)
    fallback = core.execute_propagation_plan(state, plan, *source)
    np.testing.assert_array_equal(fallback[1], reference[1])


def test_medium_view_preserves_high_products_and_user_plot_range(qtbot, medium_result):
    from temsim.gui.visualization import VisualizationWorkspace
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    widget.display_result(medium_result, "Medium")
    old = SimpleNamespace(model_signature="older high result")
    widget._high_accuracy_result = old
    widget._high_accuracy_current = False
    widget.plot.setRange(xRange=(1450, 1800), yRange=(-.01, .01), padding=0)
    expected = np.asarray(widget.plot.getViewBox().viewRange())
    widget.display_result(medium_result, "Medium")
    assert widget._high_accuracy_result is old
    np.testing.assert_allclose(widget.plot.getViewBox().viewRange(), expected)
    assert widget._tuning_envelopes
    assert "optical tuning only" in widget.heading.text()
    assert widget._last_quality == "Medium"
    widget.interactive_calculation.shutdown()


def test_live_range_uses_continuous_values_without_building_bank(qtbot, monkeypatch):
    from temsim.gui.interactive_calculation import InteractiveCalculationPage
    from temsim.component_keys import OBJECTIVE_LENS
    page = InteractiveCalculationPage()
    qtbot.addWidget(page)
    page.set_source(default_state())
    index = next(i for i in range(page.choice.count()) if page.choice.itemData(i).key == OBJECTIVE_LENS)
    page.choice.setCurrentIndex(index)
    page._add_range()
    page.ranges.cellWidget(0, 1).setText("67")
    page.ranges.cellWidget(0, 2).setText("70")
    monkeypatch.setattr(page.controller, "build", lambda *a: pytest.fail("Unexpected high bank"))
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode
    assert page.controller.bank is None
    slider_value = next(iter(page.live_widgets.values()))
    slider_value.setValue(68.123456)
    page.timer.stop()
    with qtbot.waitSignal(page.tuning_changed) as output:
        page._read()
    assert output.args[0][0][1] == pytest.approx(68.123456)
    assert not page.live_plan.precompute
    page.shutdown()


def test_main_live_tuning_applies_final_value_then_requests_high_accuracy_once(qtbot, monkeypatch):
    from temsim.gui.main_window import MainWindow
    from temsim.component_keys import OBJECTIVE_LENS
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    page = window.workspace.interactive_calculation
    calls = []
    monkeypatch.setattr(window.calculations, "submit", lambda *args: calls.append(args))
    window._capture_interactive_settings()
    index = next(i for i in range(page.choice.count()) if page.choice.itemData(i).key == OBJECTIVE_LENS)
    page.choice.setCurrentIndex(index)
    page._add_range()
    page.ranges.cellWidget(0, 1).setText("67")
    page.ranges.cellWidget(0, 2).setText("70")
    page.start_live_tuning()
    page.timer.stop()
    value = next(iter(page.live_widgets.values()))
    # Mimic an in-flight low-cost request: new tuning must supersede it,
    # rather than reject every edit while a preview is running.
    window._progress_owners.add("calculation")
    value.setValue(68.123456)
    page._request_high_accuracy()
    assert window.state.objective_lens.percent == pytest.approx(68.123456)
    assert len(calls) == 1 and calls[0][1] == "High accuracy"
    assert not window.preview_timer.isActive()
    page.shutdown()


def test_many_live_controls_do_not_create_a_cartesian_high_accuracy_grid():
    from temsim.interactive_calculation import InteractivePlan, CalculationRange, available_controls
    controls = [c for c in available_controls(default_state()) if c.group == "lens"][:5]
    axes = tuple(CalculationRange(c, 10, 11, 5) for c in controls)
    assert InteractivePlan(axes, 1024, precompute=False).point_count == 1
    with pytest.raises(ValueError, match="combinations"):
        InteractivePlan(axes, 1024)
