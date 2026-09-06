"""Synthetic field studies and offline Qt workflow checks; not OEM validation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
from threading import Event

import numpy as np
import pytest

from temsim.magnetic_validation import (
    ValidationOptions, ValidationCache, ValidationCancelled, input_signature,
    prepare_problems, run_validation, paraxial_metrics, compare_cases,
    extend_problem_grid, report_payload, _solve, snapshot_state,
)


def state_fixture():
    helpers = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    state = helpers["_state"](helpers["_shared_rows"]())
    state.beam_voltage_kv = 300.
    return state


@pytest.fixture(scope="module")
def linear_report():
    state = state_fixture()
    options = ValidationOptions(mesh_factor=1.2, boundary_factor=1.2)
    original = deepcopy(state.lens_field_map_descriptors)
    progress = []
    report = run_validation(state, "a", options, progress=lambda *a: progress.append(a))
    assert progress[0][:2] == (0, 5) and progress[-1][:2] == (5, 5)
    assert [p[0] for p in progress] == sorted(p[0] for p in progress)
    assert state.lens_field_map_descriptors == original
    assert not hasattr(state, "_runtime_lens_field_provider_cache")
    assert not hasattr(state, "_field_provider_diagnostics")
    return report


def test_synthetic_study_has_fixed_planes_and_separate_convergence(linear_report):
    report = linear_report
    assert len(report.cases) == 5 and len(report.comparisons) == 16
    assert all(c.axis_field_t.shape == report.z_m.shape for c in report.cases)
    assert all(not c.axis_field_t.flags.writeable for c in report.cases)
    assert not report.passed  # Coarse synthetic iron grid is not a 1% reference.
    assert any(c.passed is False for c in report.comparisons if c.stage.startswith("Mesh"))
    assert all(np.isfinite(r) and r < 1e-7 for c in report.cases for r in c.residuals)
    assert not report.scene.regions.flags.writeable
    assert set(np.unique(report.scene.regions)) == {0, 1, 2}
    assert np.max(np.abs(report.scene.flux_per_radian_wb)) > 0


def test_boundary_extension_preserves_every_interior_node(linear_report):
    for previous, expanded in zip(linear_report.cases[2:4], linear_report.cases[3:5]):
        for a, b in zip(previous.problems, expanded.problems):
            old, new = _solve(a)[0].axes_m, _solve(b)[0].axes_m
            for old_axis, new_axis in zip(old, new):
                assert all(np.any(new_axis == node) for node in old_axis)
                assert new_axis.size > old_axis.size


def test_linear_problems_preserve_channel_ownership_and_require_no_solve(monkeypatch):
    from temsim.physics import axisymmetric_magnetostatics as fem
    monkeypatch.setattr(fem, "_solve_bound_geometry", lambda *a: pytest.fail("Preparing must not solve"))
    state = state_fixture()
    state.lenses[0].percent = 37
    state.lenses[1].polarity = -1
    prepared = prepare_problems(state, "a")
    assert [p.scale for p in prepared] == [.37, -1.]
    assert [json.loads(p.geometry_json)["lens_key"] for p in prepared] == ["a", "b"]
    assert state.lens_field_map_descriptors["a"]["radial_nodes"] == 22


def test_nonlinear_prepare_is_joint_and_does_not_solve(monkeypatch):
    from temsim.physics import nonlinear_circuits
    from temsim.magnetic_materials import reference_materials
    monkeypatch.setattr(nonlinear_circuits, "_joint_map", lambda *a: pytest.fail("Preparing must not solve"))
    state = state_fixture()
    for row in state.lens_field_map_descriptors.values():
        row.update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0])
    state.lenses[1].percent = 20
    problems = prepare_problems(state, "a")
    assert len(problems) == 1 and problems[0].scale == 1
    assert json.loads(problems[0].settings_json)["channel_ampere_turns"] == {"a": 100., "b": 20.}
    assert not hasattr(state, "_runtime_nonlinear_provider_cache")


def test_nonlinear_study_runs_through_the_production_fem():
    from temsim.magnetic_materials import reference_materials
    state = state_fixture()
    for row in state.lens_field_map_descriptors.values():
        row.update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0],
                   ampere_turns=1000, radial_nodes=12, axial_nodes=20, padding_factor=2)
    report = run_validation(state, "a", ValidationOptions(mesh_factor=1.1, boundary_factor=1.1))
    assert len(report.cases) == 5
    assert all(len(c.problems) == 1 for c in report.cases)
    assert all(c.problems[0].scale == 1 for c in report.cases)
    assert report.scene.magnitude_t.max() > 0
    assert state.lens_field_map_descriptors["a"]["radial_nodes"] == 12


def test_paraxial_uniform_field_matches_exact_transfer_and_signed_rotation():
    from temsim.physics.core import electron
    from types import SimpleNamespace
    z = np.linspace(0, .03, 250)
    b = np.full(z.size, .01)
    q, p, _ = electron(SimpleNamespace(beam_voltage_kv=300))
    g = -q*.01/(2*p)
    result = paraxial_metrics(z, b, 300, 1)
    assert result["Effective focal length"] == pytest.approx(1000/(g*np.sin(g*.03)), rel=1e-7)
    assert result["Larmor rotation"] == pytest.approx(np.degrees(g*.03), rel=1e-12)
    assert result["Exit test-beam radius"] == pytest.approx(abs(np.cos(g*.03)), rel=1e-8)
    reversed_field = paraxial_metrics(z, -b, 300, 1)
    assert reversed_field["Larmor rotation"] == -result["Larmor rotation"]
    assert reversed_field["Effective focal length"] == result["Effective focal length"]
    vacuum = paraxial_metrics(z, b*0, 300, 1)
    assert vacuum["Effective focal length"] is None
    assert vacuum["Exit test-beam radius"] == 1


def test_near_zero_metrics_do_not_get_fake_relative_errors(linear_report):
    case = replace(linear_report.cases[0], axis_field_t=np.zeros_like(linear_report.z_m),
                   metrics=(("Effective focal length", None), ("Larmor rotation", 0), ("Exit test-beam radius", 1)))
    comparison = compare_cases(case, case, linear_report.options, "zero")
    assert comparison[0].relative_change is None and comparison[0].passed
    assert comparison[1].passed is None
    assert comparison[2].relative_change is None and comparison[2].passed


@pytest.mark.parametrize("change", [{"mesh_factor": 1}, {"field_absolute_t": 0}, {"relative_tolerance": float("nan")}])
def test_invalid_targets_rejected(change):
    with pytest.raises(ValueError):
        ValidationOptions(**change).validate()


def test_resource_and_mode_guards_are_explicit():
    state = state_fixture()
    with pytest.raises(ValueError, match="limit"):
        prepare_problems(state, "a", mesh_multiplier=100)
    with pytest.raises(ValueError, match="padding"):
        prepare_problems(state, "a", padding_multiplier=10)
    state.simulation_mode = "ideal"
    with pytest.raises(ValueError, match="inactive"):
        prepare_problems(state, "a")


def test_cancellation_does_not_publish_or_mutate_existing_results(linear_report):
    cache = ValidationCache()
    cache.put(linear_report)
    with pytest.raises(ValidationCancelled):
        run_validation(state_fixture(), "a", cancelled=lambda: True)
    assert cache.get(linear_report.signature) is linear_report


def test_geometry_current_material_voltage_and_options_invalidate_but_sample_does_not():
    state, options = state_fixture(), ValidationOptions()
    original = input_signature(state, "a", options)
    state.sample = object()
    assert input_signature(state, "a", options) == original
    state.lenses[1].percent -= 1
    assert input_signature(state, "a", options) != original
    state.lenses[1].percent += 1
    assert input_signature(state, "a", options) == original
    state.lens_field_map_descriptors["a"]["relative_permeability"] += 1
    assert input_signature(state, "a", options) != original
    state = state_fixture(); state.beam_voltage_kv = 200
    assert input_signature(state, "a", options) != original
    state = state_fixture(); state.lenses[0].z_mm += 1
    assert input_signature(state, "a", options) != original
    state = state_fixture()
    state._resolved_assembly.parts = tuple(replace(p, data={**p.data, "mechanical_outer_diameter_mm": 32})
                                         if p.key == "yoke" else p for p in state._resolved_assembly.parts)
    assert input_signature(state, "a", options) != original
    assert input_signature(state_fixture(), "a", replace(options, relative_tolerance=.02)) != original


def test_export_contains_reproducible_inputs_without_claiming_external_validation(linear_report):
    payload = report_payload(linear_report)
    assert json.loads(json.dumps(payload, allow_nan=False))
    assert payload["beam_voltage_kv"] == 300
    assert payload["external_validation"] == "Not checked"
    assert payload["cs_cc_validation"] == "Not checked"
    assert payload["cases"][-1]["problems"][0]["geometry"]["lens_assembly"]["parts"]
    assert payload["cases"][-1]["problems"][0]["settings"]["validation_grid_axes_m"]


def test_production_snapshot_preserves_resolved_geometry_and_detaches_controls():
    from temsim.optics.column import default_state
    state = default_state()
    copied = snapshot_state(state)
    assert copied._resolved_assembly is state._resolved_assembly  # Immutable assembly snapshot.
    copied.lenses[0].percent += 1
    assert copied.lenses[0].percent != state.lenses[0].percent
    copied.lens_field_map_descriptors["new"] = {}
    assert "new" not in state.lens_field_map_descriptors


def test_unrelated_distant_lens_changes_preserve_a_linear_circuit_study():
    helpers = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    rows = helpers["_shared_rows"]()+[
        helpers["_part"]("far", "magnetic_lens_assembly", 1000, 1010, 2, 30),
        helpers["_part"]("far_coil", "magnetic_excitation_coil", 1001, 1009, 16, 20, "far"),
    ]
    state = helpers["_state"](rows); state.beam_voltage_kv = 300
    before = input_signature(state, "a", ValidationOptions())
    state.lenses[-1].percent = 20
    state.lens_field_map_descriptors["far"]["relative_permeability"] = 80
    assert input_signature(state, "a", ValidationOptions()) == before


def test_translating_physical_circuit_translates_its_field_without_retuning():
    helpers = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    rows = helpers["_shared_rows"]()
    original = helpers["_state"](deepcopy(rows)); original.beam_voltage_kv = 300
    for row in rows:
        for name in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm"):
            row[name] += 5
    shifted = helpers["_state"](rows); shifted.beam_voltage_kv = 300
    z = np.linspace(-.03, .03, 101)
    first, second = [], []
    for problem in prepare_problems(original, "a"):
        first.append(_solve(problem)[0].field_at_global_positions_t(np.column_stack((z*0, z*0, z)))[:, 2])
    for problem in prepare_problems(shifted, "a"):
        second.append(_solve(problem)[0].field_at_global_positions_t(np.column_stack((z*0, z*0, z+.005)))[:, 2])
    np.testing.assert_allclose(np.sum(first, axis=0), np.sum(second, axis=0), rtol=1e-8, atol=1e-10)
    assert [l.percent for l in original.lenses] == [l.percent for l in shifted.lenses]


def test_material_change_recomputes_field_and_returning_reuses_it():
    state = state_fixture()
    problem = prepare_problems(state, "a")[0]
    before = _solve(problem)[0]
    for recipe in state.lens_field_map_descriptors.values():
        recipe["relative_permeability"] = 80
    after = _solve(prepare_problems(state, "a")[0])[0]
    assert after.content_fingerprint != before.content_fingerprint
    assert not np.allclose(after.field_at_global_positions_t([0, 0, -.005]), before.field_at_global_positions_t([0, 0, -.005]), rtol=1e-5, atol=1e-10)
    assert _solve(problem)[0] is before


def test_new_active_coil_in_expanded_nonlinear_domain_fails_before_any_solve(monkeypatch):
    import temsim.magnetic_validation as validation
    from temsim.magnetic_materials import reference_materials
    helpers = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    rows = helpers["_shared_rows"]()+[
        helpers["_part"]("c", "magnetic_lens_assembly", 70, 80, 2, 30),
        helpers["_part"]("coil_c", "magnetic_excitation_coil", 72, 78, 16, 20, "c", field_source_key="c"),
    ]
    state = helpers["_state"](rows); state.beam_voltage_kv = 300
    state.lens_field_map_descriptors.pop("c")
    for row in state.lens_field_map_descriptors.values():
        row.update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0])
    monkeypatch.setattr(validation, "_solve", lambda *a, **k: pytest.fail("Preflight must reject before solving"))
    with pytest.raises(ValueError, match="Active coil c overlaps"):
        run_validation(state, "a")


def test_worker_failure_preserves_the_displayed_report(qtbot, monkeypatch, linear_report):
    import temsim.gui.magnetic_validation as gui
    def fail(*args, **kwargs):
        raise ValueError("Synthetic validation failure")
    monkeypatch.setattr(gui, "run_validation", fail)
    page = gui.MagneticValidationPage(); qtbot.addWidget(page)
    page.refinement.setValue(1.2)
    state = state_fixture(); page.set_context(state, "a")
    page.display_report(linear_report)
    page._cache.put(linear_report)
    state.lenses[0].percent = 12
    page.start()
    qtbot.waitUntil(lambda: not page.running, timeout=5000)
    assert page._report is linear_report
    assert "Synthetic validation failure" in page.status.text()
    assert page._cache.get(linear_report.signature) is linear_report


def test_validation_gui_rendering_cache_and_no_rerun_on_tab_or_layer_change(qtbot, monkeypatch, linear_report):
    from temsim.gui.magnetic_validation import MagneticValidationPage
    import temsim.gui.magnetic_validation as gui
    monkeypatch.setattr(gui, "run_validation", lambda *a, **k: pytest.fail("Display must not calculate"))
    page = MagneticValidationPage(); qtbot.addWidget(page)
    page.refinement.setValue(1.2)
    page.set_context(state_fixture(), "a")
    page._cache.put(linear_report)
    page.start()
    assert page._report is linear_report and not page.running
    assert page.table.rowCount() == 16
    page.resize(1100, 700); page.show(); qtbot.wait(10)
    page.tabs.setCurrentIndex(1)
    page.layer.setCurrentIndex(1)
    np.testing.assert_array_equal(page.image.image, linear_report.scene.regions)
    assert np.shares_memory(page.image.image, linear_report.scene.regions)
    page.flux.setChecked(False); page.flux.setChecked(True)
    page.layer.setCurrentIndex(0)
    assert len(page.image.childItems()) == len(page._contours)
    page.scene_plot.setXRange(-5, 5, padding=0)
    bounds = page.scene_plot.viewRange()[0]
    page.refresh_identity()
    assert page.scene_plot.viewRange()[0] == bounds
    assert "not checked" in page.status.text().lower()
    page._state.lenses[0].percent -= 1
    page.refresh_identity()
    assert "Inputs changed" in page.status.text()
    page._state.lenses[0].percent += 1
    page.refresh_identity()
    assert "Inputs changed" not in page.status.text()


def test_worker_result_for_old_inputs_is_cached_without_replacing_the_current_view(qtbot, monkeypatch, linear_report):
    import temsim.gui.magnetic_validation as gui
    ready, release = Event(), Event()
    def run(state, key, options, **kwargs):
        ready.set()
        if not release.wait(5):
            raise RuntimeError("Test worker not released")
        return linear_report
    monkeypatch.setattr(gui, "run_validation", run)
    page = gui.MagneticValidationPage(); qtbot.addWidget(page)
    page.refinement.setValue(1.2)
    state = state_fixture(); page.set_context(state, "a")
    try:
        page.start()
        qtbot.waitUntil(ready.is_set, timeout=3000)
        state.lenses[0].percent = 12
    finally:
        release.set()
    qtbot.waitUntil(lambda: not page.running, timeout=5000)
    assert page._report is None and "earlier inputs" in page.status.text()
    assert page._cache.get(linear_report.signature) is linear_report
    state.lenses[0].percent = 100
    page.refresh_identity()
    assert page._report is linear_report
