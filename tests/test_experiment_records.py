from dataclasses import replace
from types import SimpleNamespace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures
from temsim.design_explorer import capture_design_snapshot, HighAccuracyRequest
from temsim.design_experiments import recipe_from_snapshot, plan_parameter_sweep, SweepAxis, ToleranceRule
from temsim.design_sweep_execution import execute_parameter_sweep
from temsim.experiment_records import (save_experiment, load_experiment, export_experiment_table,
    plan_robustness, Perturbation, pareto_points)
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult


def source():
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    state.extra_captured_input = {'disabled_component_parameter': 17.125}
    snapshot = capture_design_snapshot(state, selection, slot='A', request=HighAccuracyRequest(9, 2.5))
    return state, recipe_from_snapshot(snapshot, name='saved experiment')


def calculator(calls, *, fail_at=None):
    def calculate(state, **kwargs):
        value = state.objective_lens.percent
        calls.append(value)
        assert state.extra_captured_input == {'disabled_component_parameter': 17.125}
        if value == fail_at:
            raise RuntimeError('declared fixture failure')
        return CalculationResult(simulation=SimpleNamespace(metrics={
            'sample_illumination_diameter_95_um': value,
            'sample_surviving_current_pa': value}), energy_filter=None, state_snapshot=state,
            signatures=calculation_signatures(state))
    return calculate


def test_cancel_save_resume_preserves_complete_inputs_failures_and_live_state(tmp_path):
    live, recipe = source()
    before = live.to_dict()
    sweep = plan_parameter_sweep(recipe, (SweepAxis('lenses[objective_lens].percent', (20., 40., 60.)),))
    calls = []
    first = execute_parameter_sweep(recipe, sweep, calculator=calculator(calls), cancel_requested=lambda: len(calls) == 1)
    assert first.cancelled and first.completed_points == 1
    path = tmp_path / 'captured.temexp'
    save_experiment(path, recipe, sweep, first)
    saved_recipe, saved_sweep, saved_result, rules = load_experiment(path)
    interrupted = execute_parameter_sweep(saved_recipe, saved_sweep, resume=saved_result,
        cancel_requested=lambda: True, calculator=calculator(calls))
    assert interrupted.cancelled and interrupted.point_results == saved_result.point_results
    resumed = execute_parameter_sweep(saved_recipe, saved_sweep, resume=saved_result,
        calculator=calculator(calls, fail_at=40.))
    assert calls == [20., 40., 60.]
    assert [row.status for row in resumed.point_results] == ['COMPLETE', 'FAILED', 'COMPLETE']
    assert 'fixture failure' in resumed.point_results[1].failure_reason
    assert resumed.point_results[1].metrics == {}
    assert all(row.numerical_status == 'NOT_RUN' for row in resumed.point_results)
    assert live.to_dict() == before
    with pytest.raises(ValueError, match='Resume requires identical'):
        execute_parameter_sweep(saved_recipe, saved_sweep, resume=saved_result,
            tolerance_rules=(ToleranceRule('sample_surviving_current_pa', minimum=1.),))
    assert pareto_points(resumed, {'sample_illumination_diameter_95_um': 'min',
        'sample_surviving_current_pa': 'max'}) == (0, 2)
    export_experiment_table(tmp_path / 'observations.csv', saved_recipe, saved_sweep, resumed)
    assert 'FAILED' in (tmp_path / 'observations.csv').read_text(encoding='utf-8-sig')
    assert (tmp_path / 'observations.csv.manifest.json').exists()
    path.write_text(path.read_text().replace('disabled_component_parameter', 'tampered_input'))
    with pytest.raises(ValueError, match='checksum'):
        load_experiment(path)


def test_explicit_normal_perturbations_repeat_and_retain_invalid_samples():
    _, recipe = source()
    options = (Perturbation('lenses[objective_lens].percent', 200.),)
    plan = plan_robustness(recipe, options, samples=8, seed=12)
    duplicate = plan_robustness(recipe, options, samples=8, seed=12)
    assert [p.state_digest for p in plan.points] == [p.state_digest for p in duplicate.points]
    calls = []
    result = execute_parameter_sweep(recipe, plan, calculator=calculator(calls))
    assert len(result.point_results) == 8
    assert any(row.status == 'FAILED' for row in result.point_results)
    assert 'not an OEM' in plan.assumptions['scope']


def test_experiment_plot_and_load_are_read_only(qtbot, tmp_path):
    from temsim.gui.design_explorer import DesignExplorerPage
    _, recipe = source()
    sweep = plan_parameter_sweep(recipe, (SweepAxis('lenses[objective_lens].percent', (20., 40.)),))
    calls = []
    result = execute_parameter_sweep(recipe, sweep, calculator=calculator(calls))
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    page.experiment_tools.bind_pending(recipe, sweep, ())
    page.set_sweep_result(result, 1.)
    assert page.experiment_tools.record[2] is result
    page.experiment_tools.metric.setCurrentIndex(1)
    page.experiment_tools._compare()
    assert calls == [20., 40.]
    assert page.sweep_result_table.item(0, 0).text().endswith('COMPLETE')
    assert page.experiment_tools.shutdown()


def test_full_recipe_rebuild_does_not_migrate_source_vacuum_or_sample_orientation():
    from temsim.design_sweep_execution import rebuild_recipe_state
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    state.electron_gun.emitter.surface_model = model_from_part(state._resolved_assembly.part('feg_tip').data)
    state.simulation_mode = 'analytical'
    state.vacuum_map.enabled = True
    state.sample.optional_record = {'keep': 1.23456789012345}
    source_before = state.electron_gun.emitter.surface_model.to_dict()
    orientation = state.sample.specimen_orientation_quaternion_wxyz
    snapshot = capture_design_snapshot(state, selection, slot='A', request=HighAccuracyRequest(9, 2.5))
    recipe = recipe_from_snapshot(snapshot, name='exact model preservation')
    restored, _ = rebuild_recipe_state(recipe)
    assert restored.electron_gun.emitter.surface_model.to_dict() == source_before
    assert restored.simulation_mode == 'analytical'
    assert restored.vacuum_map.enabled is True
    assert restored.sample.optional_record == {'keep': 1.23456789012345}
    assert restored.sample.specimen_orientation_quaternion_wxyz == orientation


def test_two_dimensional_points_background_record_io_and_plot_export(qtbot, monkeypatch, tmp_path):
    from temsim.design_sweep_execution import SweepExecutionResult, SweepPointExecution, SWEEP_METRICS
    from temsim.gui.experiment_tools import ExperimentTools, QFileDialog
    from temsim.immutable_json import json_digest
    _, recipe = source()
    sweep = plan_parameter_sweep(recipe, (SweepAxis('lenses[objective_lens].percent', (20., 40.)),
        SweepAxis('lenses[projector_lens_2].percent', (10., 30.))))
    rows = tuple(SweepPointExecution(point.index, point.coordinates, point.state_digest, '', '',
        {'sample_surviving_current_pa': float(point.index)} if point.index != 2 else {}, (), (), 0.,
        status='FAILED' if point.index == 2 else 'COMPLETE', failure_reason='Unavailable fixture' if point.index == 2 else '') for point in sweep.points)
    result = SweepExecutionResult(recipe.digest, rows, (), (), SWEEP_METRICS, False, sweep.axes,
        execution_inputs={'scope': 'plot fixture, no numerical qualification'},
        execution_identity=json_digest({'scope': 'plot fixture, no numerical qualification'}))
    tools = ExperimentTools()
    qtbot.addWidget(tools)
    tools.set_record((recipe, sweep, result, ()))
    tools.metric.setCurrentIndex(tools.metric.findData('sample_surviving_current_pa'))
    assert 'crosses' in tools.plot.plotItem.titleLabel.text
    record = tmp_path / 'saved.temexp'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *_: (str(record), ''))
    tools._file('save')
    qtbot.waitUntil(lambda: tools.worker is None, timeout=10000)
    assert record.exists()
    loaded = []
    tools.record_loaded.connect(loaded.append)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *_: (str(record), ''))
    tools._file('load')
    qtbot.waitUntil(lambda: tools.worker is None, timeout=10000)
    assert loaded[0][2].point_results[2].status == 'FAILED'
    png = tmp_path / 'plot.png'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *_: (str(png), ''))
    tools._export_plot()
    from hashlib import sha256
    import json
    plot_manifest = png.with_suffix('.png.manifest.json')
    assert png.exists() and plot_manifest.exists()
    original_plot_manifest = plot_manifest.read_bytes()
    assert json.loads(original_plot_manifest)['output_sha256'] == sha256(png.read_bytes()).hexdigest()
    table = png.with_suffix('.csv')
    table_manifest = export_experiment_table(table, recipe, sweep, result)
    assert table_manifest != plot_manifest
    assert json.loads(table_manifest.read_bytes())['output_sha256'] == sha256(table.read_bytes()).hexdigest()
    assert plot_manifest.read_bytes() == original_plot_manifest
    assert tools.shutdown()


def test_background_design_capture_preserves_at_click_inputs_and_releases_assets(qtbot, monkeypatch):
    from temsim.gui.design_explorer import DesignExplorerPage
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.design_sweep_execution import rebuild_recipe_state
    from threading import get_ident
    import temsim.design_explorer as model
    live, _ = source()
    original = live.objective_lens.percent
    calls = []
    capture = model.capture_design_snapshot
    def observed(*args, **kwargs):
        calls.append(get_ident())
        return capture(*args, **kwargs)
    monkeypatch.setattr(model, 'capture_design_snapshot', observed)
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    request = CapturedCalculationRequest.capture(live, 'High accuracy', 9, 2.5)
    page.capture_in_background(request, AssemblyCatalog().default_selection(), 'A')
    live.objective_lens.percent = original + 1.
    qtbot.waitUntil(lambda: page._capture_worker is None, timeout=20000)
    assert calls and calls == [calls[0]] and calls[0] != get_ident()
    assert request._input_assets._closed
    state, _ = rebuild_recipe_state(page.recipe_for_slot('A'))
    assert state.objective_lens.percent == original
    assert state.extra_captured_input == live.extra_captured_input
    assert page.shutdown()


def test_local_sensitivity_objective_uses_neighbors_and_does_not_bridge_failed_points():
    from temsim.design_experiments import ParameterSweep, SweepPoint
    from temsim.design_sweep_execution import SweepPointExecution, SweepMetricDefinition
    from temsim.experiment_records import with_local_sensitivities
    path = 'lenses[objective_lens].percent'
    points = tuple(SweepPoint(i, {path:x}, {}, 'fixture') for i,x in enumerate((1.,2.,4.)))
    sweep = ParameterSweep('fixture', (SweepAxis(path, (1.,2.,4.), '%'),), points)
    rows = tuple(SweepPointExecution(p.index,p.coordinates,'fixture','','',{'spot':2*p.coordinates[path]**2},(),(),0.) for p in points)
    definition = SweepMetricDefinition('spot','Spot','um','fixture')
    measured, definitions = with_local_sensitivities(sweep, rows, (definition,))
    key = definitions[-1].key
    assert measured[1].metrics[key] == pytest.approx(8.)
    assert measured[1].evidence['local_sensitivities'][key]['point_indices'] == (0,1,2)
    failed = replace(rows[1], metrics={}, status='FAILED', failure_reason='Unavailable')
    measured, definitions = with_local_sensitivities(sweep, (rows[0], failed, rows[2]), (definition,))
    assert definitions == (definition,)
    assert all(key not in row.metrics for row in measured)
