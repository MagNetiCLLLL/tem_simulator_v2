from pathlib import Path
from types import SimpleNamespace

import pytest

from test_experiment_records import source
from temsim import input_io
from temsim.calculation_cache import calculation_signatures
from temsim.design_sweep_execution import execute_parameter_sweep
from temsim.experiment_records import save_experiment, load_experiment
from temsim.geometry_experiments import plan_geometry_sweep, candidate_state, validate_active_fields
from temsim.simulation_pipeline import CalculationResult


def test_geometry_candidates_validate_retain_failures_and_never_write_live_toml(tmp_path):
    live, recipe = source()
    paths = [Path(row.path) for row in live._resolved_assembly.external_inputs] if hasattr(live._resolved_assembly, 'external_inputs') else []
    paths += list(Path(live._resolved_assembly.root).rglob('*.toml'))
    before = {path: path.read_bytes() for path in paths}
    controls = {lens.key: lens.percent for lens in live.lenses}
    sweep = plan_geometry_sweep(recipe, 'column', 'condenser_aperture_2', 'vacuum_inner_diameter_mm', (4., 5., -1.))
    calls = []

    def calculate(state, **kwargs):
        assert {lens.key: lens.percent for lens in state.lenses} == controls
        part = next(part for part in state._resolved_assembly.parts if part.key == 'condenser_aperture_2')
        calls.append(part.data['vacuum_inner_diameter_mm'])
        assert input_io.archive_payload(state) is not None
        return CalculationResult(SimpleNamespace(metrics={'sample_surviving_current_pa': 1.}),
            None, state_snapshot=state, signatures=calculation_signatures(state))

    result = execute_parameter_sweep(recipe, sweep, calculator=calculate)
    assert [row.status for row in result.point_results] == ['COMPLETE', 'COMPLETE', 'FAILED'], [row.failure_reason for row in result.point_results]
    assert calls == [4., 5.]
    assert result.point_results[0].geometry_fingerprint != result.point_results[1].geometry_fingerprint
    assert result.point_results[0].evidence['candidate_toml_sha256'] != result.point_results[1].evidence['candidate_toml_sha256']
    assert result.point_results[0].evidence['controls'] == 'Fixed-control comparison'
    assert all(path.read_bytes() == content for path, content in before.items())
    destination = tmp_path / 'geometry.temexp'
    save_experiment(destination, recipe, sweep, result)
    r, s, saved, rules = load_experiment(destination)
    resumed = execute_parameter_sweep(r, s, resume=saved, calculator=lambda *_args, **_kw: pytest.fail('retained points recalculated'))
    assert len(resumed.point_results) == 3


def test_geometry_candidate_executes_actual_tip_chain_and_rejects_missing_active_map():
    from temsim.alignment_constraints import evaluate_incident
    live, recipe = source()
    archive = input_io.capture_input_archive(live)
    sweep = plan_geometry_sweep(recipe, 'column', 'condenser_aperture_2', 'vacuum_inner_diameter_mm', (4.,))
    state, selection, evidence = candidate_state(recipe, sweep.points[0], sweep, archive)
    metrics, arrays, summary, topology = evaluate_incident(state)
    assert arrays['x_m'].shape[0] == 9
    state.simulation_mode = 'custom'
    state.lens_field_map_descriptors = {'objective_lens': {'source_path': 'absent-map.npz', 'geometry_fingerprint': 'invalid'}}
    with input_io.input_scope(state), pytest.raises(ValueError, match='requested field map'):
        validate_active_fields(state)


def test_geometry_editor_captures_explicit_controls_without_mutation(qtbot):
    from temsim.gui.geometry_experiments import GeometryExperimentEditor
    live, recipe = source()
    before = live.to_dict()
    editor = GeometryExperimentEditor(lambda slot: recipe)
    qtbot.addWidget(editor)
    requests = []
    editor.requested.connect(lambda *args: requests.append(args))
    editor.request()
    assert requests[0][1].assumptions['optimization'] is None
    editor.mode.setCurrentIndex(1)
    editor.constraints.controls.item(0, 0).setCheckState(__import__('PySide6.QtCore', fromlist=['Qt']).Qt.CheckState.Unchecked)
    editor.request()
    assert set(requests[1][1].assumptions['optimization']['options']['bounds']) == {'condenser_lens_3'}
    assert live.to_dict() == before
