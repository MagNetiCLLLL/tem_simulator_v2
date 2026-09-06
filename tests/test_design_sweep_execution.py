from dataclasses import replace
import shutil
from types import SimpleNamespace

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.calculation_cache import calculation_signatures
from temsim.design_experiments import (
    ToleranceResult,
    ToleranceRule,
    SweepAxis,
    plan_parameter_sweep,
    recipe_from_snapshot,
)
from temsim.design_explorer import (
    HighAccuracyRequest,
    capture_design_snapshot,
)
from temsim.design_sweep_execution import (
    PointToleranceAssessment,
    SWEEP_METRICS,
    SweepExecutionResult,
    SweepPointExecution,
    execute_parameter_sweep,
    rebuild_recipe_state,
)
from temsim.gui.design_explorer import DesignExplorerPage
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.simulation_pipeline import CalculationResult


def _recipe(*, objective_percent=73.25, selection=None):
    catalog = AssemblyCatalog()
    selection = selection or catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    state.objective_lens.percent = objective_percent
    snapshot = capture_design_snapshot(
        state,
        selection,
        slot="A",
        request=HighAccuracyRequest(49, 2.5),
        captured_at_utc="2026-09-05T12:00:00+00:00",
    )
    return (
        catalog,
        state,
        recipe_from_snapshot(snapshot, name="sweep source"),
    )


def test_recipe_rebuild_resolves_selection_and_keeps_lens_strengths():
    selected = AssemblySelection(
        gun="FEG + Mono",
        column="C3 + Probe Corrector",
        recording="Energy Filter",
        beam_blanker="NanoPulser",
    )
    catalog, _live, recipe = _recipe(
        objective_percent=81.5,
        selection=selected,
    )

    rebuilt, selection = rebuild_recipe_state(
        recipe,
        catalog=catalog,
        validate_recipe_geometry=True,
    )

    assert selection == selected
    assert rebuilt.objective_lens.percent == pytest.approx(81.5)
    assert dict(rebuilt._resolved_assembly.selected_module_paths) == (
        catalog.selected_paths(selected)
    )
    assert rebuilt.nanopulser.installed
    assert rebuilt.monochromator_installed


def test_executable_sweep_is_detached_reuses_pipeline_and_analyses_results():
    catalog, live, recipe = _recipe(objective_percent=77.0)
    original_payload = live.to_dict()
    sweep = plan_parameter_sweep(
        recipe,
        (SweepAxis(
            "lenses[projector_lens_2].percent",
            (20.0, 40.0, 60.0),
            "%",
        ),),
    )
    calculated_states = []
    existing_inputs = []
    progress = []

    def fake_calculate(state, *, progress_callback, existing_result=None):
        calculated_states.append(state)
        existing_inputs.append(existing_result)
        progress_callback(1, 2, "Tracing detached point")
        p2 = next(
            lens for lens in state.lenses if lens.key == "projector_lens_2"
        )
        signatures = calculation_signatures(state)
        return CalculationResult(
            simulation=SimpleNamespace(metrics={
                "sample_illumination_diameter_95_um": 2.0 * p2.percent,
                "sample_surviving_current_pa": 10.0,
            }),
            energy_filter=None,
            state_snapshot=state,
            signatures=signatures,
            calculated_products=(
                frozenset({"incident", "column"})
                if existing_result is None
                else frozenset({"column"})
            ),
            reused_products=(
                frozenset()
                if existing_result is None
                else frozenset({"incident"})
            ),
        )

    result = execute_parameter_sweep(
        recipe,
        sweep,
        catalog=catalog,
        calculator=fake_calculate,
        tolerance_rules=(ToleranceRule(
            "sample_surviving_current_pa", minimum=5.0
        ),),
        progress_callback=progress.append,
    )

    assert not result.cancelled
    assert result.completed_points == 3
    assert live.to_dict() == original_payload
    assert len({id(state) for state in calculated_states}) == 3
    assert all(state is not live for state in calculated_states)
    assert [
        next(
            lens for lens in state.lenses
            if lens.key == "projector_lens_2"
        ).percent
        for state in calculated_states
    ] == pytest.approx([20.0, 40.0, 60.0])
    assert all(
        state.electron_gun.ray_count == recipe.request.ray_count
        for state in calculated_states
    )
    assert existing_inputs[0] is None
    assert existing_inputs[1] is not None
    assert "incident" in result.point_results[1].reused_products
    assert result.sensitivities[0].metric == (
        "sample_illumination_diameter_95_um"
    )
    assert result.sensitivities[0].derivative == pytest.approx(2.0)
    assert all(
        assessment.results[0].passed
        for assessment in result.tolerances
    )
    assert progress[-1].completed_points == 3
    assert progress[-1].stage == "Point complete"


def test_sweep_cancellation_discards_inflight_point():
    catalog, _live, recipe = _recipe()
    sweep = plan_parameter_sweep(
        recipe,
        (SweepAxis(
            "lenses[projector_lens_2].percent", (20.0, 40.0)
        ),),
    )
    cancel = {"requested": False}

    def fake_calculate(state, *, progress_callback, existing_result=None):
        cancel["requested"] = True
        progress_callback(1, 2, "Halfway")
        raise AssertionError("Cancellation must interrupt the calculator")

    result = execute_parameter_sweep(
        recipe,
        sweep,
        catalog=catalog,
        calculator=fake_calculate,
        cancel_requested=lambda: cancel["requested"],
    )

    assert result.cancelled
    assert result.completed_points == 0
    assert result.sensitivities == ()
    assert "cancelled" in result.analysis_notes[0].lower()


def test_sweep_rejects_stale_geometry_before_calculating():
    catalog, _live, recipe = _recipe()
    stale_recipe = replace(recipe, geometry_fingerprint="0" * 64)
    sweep = plan_parameter_sweep(
        stale_recipe,
        (SweepAxis(
            "lenses[projector_lens_2].percent", (20.0, 40.0)
        ),),
    )

    with pytest.raises(ValueError, match="geometry has changed"):
        execute_parameter_sweep(
            stale_recipe,
            sweep,
            catalog=catalog,
            calculator=lambda *_args, **_kwargs: None,
        )


def test_sweep_rejects_cif_content_drift_before_calculating(tmp_path):
    cif_path = tmp_path / "specimen.cif"
    cif_path.write_text("data_original\n", encoding="utf-8")
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(cif_path)
    snapshot = capture_design_snapshot(
        state,
        selection,
        slot="A",
        request=HighAccuracyRequest(49, 2.5),
    )
    recipe = recipe_from_snapshot(snapshot, name="CIF guard")
    sweep = plan_parameter_sweep(
        recipe,
        (SweepAxis("lenses[objective_lens].percent", (40.0, 60.0)),),
    )
    cif_path.write_text("data_changed\n", encoding="utf-8")
    called = []

    with pytest.raises(ValueError, match="external model differs"):
        execute_parameter_sweep(
            recipe,
            sweep,
            catalog=catalog,
            calculator=lambda *_args, **_kwargs: called.append(True),
        )

    assert called == []


def test_sweep_rechecks_field_map_bytes_before_each_point(tmp_path):
    field_path = tmp_path / "objective-field.npz"
    field_path.write_bytes(b"field-map-v1")
    catalog, state, _unused = _recipe()
    state.lens_field_map_descriptors = {
        "objective_lens": {"source_path": str(field_path)}
    }
    snapshot = capture_design_snapshot(
        state,
        catalog.default_selection(),
        slot="A",
        request=HighAccuracyRequest(49, 2.5),
    )
    recipe = recipe_from_snapshot(snapshot, name="field-map guard")
    sweep = plan_parameter_sweep(
        recipe,
        (SweepAxis("lenses[objective_lens].percent", (40.0, 60.0)),),
    )
    calls = []

    def fake_calculate(point_state, *, progress_callback, existing_result=None):
        calls.append(point_state)
        signatures = calculation_signatures(point_state)
        result = CalculationResult(
            simulation=SimpleNamespace(metrics={}),
            energy_filter=None,
            state_snapshot=point_state,
            signatures=signatures,
        )
        field_path.write_bytes(b"field-map-v2")
        return result

    with pytest.raises(RuntimeError, match="inputs changed"):
        execute_parameter_sweep(
            recipe,
            sweep,
            catalog=catalog,
            calculator=fake_calculate,
        )

    assert len(calls) == 1


def test_sweep_rejects_assembly_toml_content_drift(tmp_path):
    root = tmp_path / "instrument"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    snapshot = capture_design_snapshot(
        state,
        selection,
        slot="A",
        request=HighAccuracyRequest(49, 2.5),
    )
    recipe = recipe_from_snapshot(snapshot, name="TOML guard")
    sweep = plan_parameter_sweep(
        recipe,
        (SweepAxis("lenses[objective_lens].percent", (40.0, 60.0)),),
    )
    selected = dict(state._resolved_assembly.selected_module_paths)
    module_path = root / selected["column"]
    module_path.write_text(
        module_path.read_text(encoding="utf-8") + "\n# content drift\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="external inputs changed"):
        execute_parameter_sweep(
            recipe,
            sweep,
            catalog=catalog,
            calculator=lambda *_args, **_kwargs: None,
        )


@pytest.mark.parametrize(
    "path",
    (
        "lenses[objective_lens].z_mm",
        "lenses[objective_lens].mechanical_profile",
        "sample.cif_path",
        "component_placements.objective_lens.z_mm",
    ),
)
def test_sweep_rejects_non_runtime_or_external_parameter_paths(path):
    _catalog, _state, recipe = _recipe()

    with pytest.raises(ValueError, match="not an allowed runtime sweep"):
        plan_parameter_sweep(
            recipe,
            (SweepAxis(path, (1.0, 2.0)),),
        )


@pytest.mark.parametrize(
    ("path", "values"),
    (
        ("column_current_limit_percent", (70.0, 90.0)),
        (
            "electron_gun.components.feg_tip.emission_current_na",
            (8_000.0, 10_000.0),
        ),
        ("sample.thickness_nm", (10.0, 20.0)),
    ),
)
def test_runtime_sweep_allowlist_supports_current_beam_and_sample_controls(
    path, values
):
    _catalog, _state, recipe = _recipe()

    sweep = plan_parameter_sweep(recipe, (SweepAxis(path, values),))

    assert [point.coordinates[path] for point in sweep.points] == list(values)


def test_runtime_beam_current_and_sample_coordinates_survive_rebuild():
    catalog, live, recipe = _recipe()
    original_payload = live.to_dict()
    source_path = "electron_gun.components.feg_tip.emission_current_na"
    sweep = plan_parameter_sweep(
        recipe,
        (
            SweepAxis("column_current_limit_percent", (82.0,)),
            SweepAxis(source_path, (8_500.0,)),
            SweepAxis("sample.thickness_nm", (24.0,)),
        ),
    )
    rebuilt = []

    def fake_calculate(state, *, progress_callback, existing_result=None):
        rebuilt.append(state)
        return CalculationResult(
            simulation=SimpleNamespace(metrics={}),
            energy_filter=None,
            state_snapshot=state,
            signatures=calculation_signatures(state),
        )

    result = execute_parameter_sweep(
        recipe,
        sweep,
        catalog=catalog,
        calculator=fake_calculate,
    )

    assert result.completed_points == 1
    assert rebuilt[0].column_current_limit_percent == pytest.approx(82.0)
    assert rebuilt[0].electron_gun.emitter.emission_current_na == pytest.approx(
        8_500.0
    )
    assert rebuilt[0].sample.thickness_nm == pytest.approx(24.0)
    assert live.to_dict() == original_payload


def test_design_explorer_gui_plans_a_real_bounded_sweep(qtbot):
    catalog, live, _recipe_source = _recipe()
    snapshot = capture_design_snapshot(
        live,
        catalog.default_selection(),
        slot="A",
        request=HighAccuracyRequest(49, 2.5),
    )
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    page.set_capture(snapshot)
    page.sweep_parameter.setEditText(
        "lenses[projector_lens_2].percent"
    )
    page.sweep_values.setText("20, 40, 60")

    with qtbot.waitSignal(page.sweep_requested) as signal:
        page.run_sweep_button.click()

    recipe, sweep, tolerance_rules = signal.args
    assert recipe.digest == page.recipe_for_slot("A").digest
    assert [
        point.coordinates["lenses[projector_lens_2].percent"]
        for point in sweep.points
    ] == [20.0, 40.0, 60.0]
    assert tolerance_rules == ()
    assert sweep.axes[0].unit == "%"


def test_design_explorer_gui_displays_sweep_provenance(qtbot):
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    point = SweepPointExecution(
        point_index=0,
        coordinates={"lenses[projector_lens_2].percent": 40.0},
        state_digest="state",
        geometry_fingerprint="geometry",
        request_signature="request",
        metrics={"sample_surviving_current_pa": 12.5},
        calculated_products=("column",),
        reused_products=("incident",),
        duration_s=0.25,
    )
    result = SweepExecutionResult(
        recipe_digest="recipe",
        point_results=(point,),
        sensitivities=(),
        tolerances=(PointToleranceAssessment(
            0,
            (ToleranceResult(
                "sample_surviving_current_pa",
                12.5,
                True,
                "Within tolerance",
            ),),
        ),),
        metric_definitions=SWEEP_METRICS,
        cancelled=False,
    )

    page.set_sweep_running(1)
    page.set_sweep_result(result, 0.25)
    page.set_sweep_finished()

    assert page.sweep_result_table.rowCount() == 1
    assert "12.5 pA" in page.sweep_result_table.item(0, 2).text()
    assert page.sweep_result_table.item(0, 3).text() == "1 products"
    assert page.sweep_result_table.item(0, 4).text() == "Pass"
    assert page.run_sweep_button.isEnabled()
    assert not page.cancel_sweep_button.isEnabled()


def test_design_explorer_gui_plans_multiple_axes_without_mutating_live_state(qtbot):
    from PySide6.QtWidgets import QTableWidgetItem
    catalog, live, _ = _recipe()
    before = live.to_dict()
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    page.set_capture(capture_design_snapshot(live, catalog.default_selection(), slot="A", request=HighAccuracyRequest(49, 2.5)))
    page.sweep_parameter.setEditText("lenses[projector_lens_2].percent")
    page.sweep_values.setText("20, 40")
    page.additional_axes.setRowCount(1)
    page.additional_axes.setItem(0, 0, QTableWidgetItem("sample.thickness_nm"))
    page.additional_axes.setItem(0, 1, QTableWidgetItem("10, 20, 30"))
    with qtbot.waitSignal(page.sweep_requested) as signal:
        page.run_sweep_button.click()
    sweep = signal.args[1]
    assert len(sweep.axes) == 2
    assert len(sweep.points) == 6
    assert live.to_dict() == before
    page.additional_axes.item(0, 0).setText("lenses[projector_lens_2].percent")
    with qtbot.waitSignal(page.sweep_error):
        page.run_sweep_button.click()
