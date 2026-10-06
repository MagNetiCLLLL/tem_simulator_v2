"""Scope receipts are software evidence, never whole-instrument qualification."""
from copy import deepcopy
from pathlib import Path

import pytest

from temsim.acceptance import (
    ACCEPTANCE_SCOPES, CLASSICAL_CRITERIA, CLASSICAL_TESTS,
    software_report, scope_test_files,
)


ROOT = Path(__file__).resolve().parents[1]


def receipt_for(scope):
    nodes = [path + "::synthetic_receipt_only" for path in scope_test_files(scope)]
    return {"collected": nodes, "complete": True, "cases": {
        node: {"setup": "passed", "call": "passed", "teardown": "passed"}
        for node in nodes}}


def evaluate(scope, receipt=None, **kwargs):
    return software_report(receipt if receipt is not None else receipt_for(scope), scope=scope,
        pytest_exit_code=kwargs.pop("pytest_exit_code", 0),
        source_unchanged=kwargs.pop("source_unchanged", True), **kwargs)


def test_classical_contract_and_report_namespace_remain_unchanged():
    assert ACCEPTANCE_SCOPES["classical"]["criteria"] is CLASSICAL_CRITERIA
    assert scope_test_files("classical") == CLASSICAL_TESTS
    report = software_report(receipt_for("classical"), pytest_exit_code=0, source_unchanged=True)
    assert report["schema"] == "classical-software-acceptance-v1"
    assert report["scope"] == "classical-particle-software"
    assert report["software_scope_status"] == "PASS"


@pytest.mark.parametrize("scope", tuple(ACCEPTANCE_SCOPES))
def test_declared_scopes_use_existing_unique_whole_files_and_namespaced_criteria(scope):
    definition = ACCEPTANCE_SCOPES[scope]
    paths = scope_test_files(scope)
    assert definition["description"] and definition["evidence_kind"]
    assert paths and len(paths) == len(set(paths))
    assert all("::" not in path and (ROOT / path).is_file() for path in paths)
    if scope != "classical":
        assert all(key.startswith(scope + "/") for key in definition["criteria"])
    report = evaluate(scope)
    assert report["scope_key"] == scope
    assert report["software_scope_status"] == "PASS" and report["exit_code"] == 0
    assert report["evidence_kind"] == definition["evidence_kind"]
    assert report["selected_tests"] == list(paths)
    assert report["full_simulator_qualification"] == "UNQUALIFIED"
    assert {"coherent-tip-to-image", "native-desktop", "actual-gpu-scientific-parity",
            "experimental-calibration"} <= report["exclusions"].keys()
    if scope != "classical":
        assert report["scope"] == scope
        assert report["schema"] == "scoped-software-acceptance-v1"


def test_new_lanes_explicitly_cover_reviewed_feature_and_runtime_boundaries():
    required = {
        "gun-fields": {"test_closed_gun_field.py", "test_continuous_tip_curvature.py",
                       "test_tip_curvature_comparison.py", "test_axisymmetric_cut_field.py",
                       "test_diagnostic_field_identity.py", "test_diagnostic_gun_domains.py",
                       "test_grounded_field_identity.py", "test_instrument_electric.py"},
        "electron-execution": {"test_magnetic_test_particle.py", "test_test_electron_scene.py",
            "test_test_electron_execution.py", "test_test_electron_compiled.py",
            "test_test_electron_intercepts.py", "test_closed_gun_execution.py",
            "test_test_electron_performance.py", "test_numba_cache.py",
            "test_diagnostic_scene_identity.py", "test_diagnostic_execution_identity.py",
            "test_electron_execution_faults.py", "test_electron_execution_diagnostics.py",
            "test_electron_execution_protocol.py", "test_electron_resource_cleanup.py", "test_cpu_resources.py",
            "test_electron_diagnostic_session.py", "test_shared_electron_column.py"},
        "field-ui": {"test_magnetic_test_electron_gui.py", "test_continuous_electron_gui.py",
            "test_virtual_electron_dock.py", "test_magnetic_field_3d.py", "test_magnetic_field_canvas.py",
            "test_magnetic_field_lines.py", "test_magnetic_field_scene.py", "test_incremental_magnetic_scene.py",
            "test_magnetic_navigation_link.py", "test_hardware_tuning_gui.py", "test_hardware_tuning_bindings.py",
            "test_magnetic_field_identity.py", "test_electron_failure_gui.py",
            "test_hardware_tuning_feedback.py", "test_electron_session_gui.py", "test_diagnostic_electron_record.py",
            "test_assembly_selection_state.py", "test_assembly_navigation.py", "test_instrument_configuration.py",
            "test_working_point_restore_gui.py", "test_selected_plane.py", "test_selected_plane_gui.py",
            "test_conjugate_planes.py", "test_conjugate_plane_gui.py", "test_conjugate_plane_workspace.py",
            "test_coherent_beam_gui.py", "test_shared_tip_workflow.py", "test_tip_source_gui.py",
            "test_electron_beam_observation.py", "test_wave_beam_analysis.py",
            "test_coherent_state_controller.py", "test_coherent_state_list.py", "test_coherent_state_set.py",
            "test_beam_comparison.py", "test_particle_energy_handoff.py",
            "test_plane_hardware_geometry.py", "test_plane_hardware_overlay.py",
            "test_plane_cutoff_events.py", "test_lazy_ray_panels.py", "test_ray_extent_workspace.py",
            "test_incremental_ray_scene.py",
            "test_beam_analysis_modes.py", "test_transverse_source_tracking.py", "test_filter_plane_analysis.py",
            "test_ray_flight_time_colours.py", "test_transverse_plot_sizes.py", "test_transverse_plot_size_persistence.py",
            "test_energy_filter_mechanical_clipping.py", "test_energy_filter_model_3d.py",
            "test_energy_filter_workspace_views.py", "test_live_beam_refresh.py"},
        "particle-continuation": {"test_particle_sections.py", "test_particle_section_io.py",
            "test_completed_particle_sections.py", "test_material_particle_sections.py",
            "test_material_section_resume.py", "test_particle_section_eds_archive.py",
            "test_particle_section_eds_reuse.py", "test_particle_archive_compression.py",
            "test_section_archive_identity.py", "test_result_file_request_routing.py", "test_result_files_gui.py",
            "test_downstream_transport.py", "test_specimen_time_of_flight.py"},
        "acceptance-policy": {"test_acceptance_scopes.py", "test_acceptance_runner.py", "test_validation_process.py"},
        "performance-observation": {"test_electron_execution_performance.py", "test_electron_response_benchmark.py",
            "test_continuous_electron_response_benchmark.py", "test_particle_benchmark.py"},
        "coherent-development": {"test_coherent_inputs.py", "test_tip_coherent_emission.py",
            "test_electron_detection.py", "test_electron_beam_observation.py",
            "test_wave_beam_analysis.py", "test_wave_plane_observables.py",
            "test_shared_tip_workflow.py", "test_beam_comparison.py",
            "test_coherent_state_controller.py", "test_coherent_state_list.py", "test_coherent_state_set.py",
            "test_radial_phase_fem.py", "test_shared_surface_source.py", "test_surface_wave_integration.py",
            "test_inelastic_wave.py", "test_galerkin_potential.py", "test_galerkin_specimen.py", "test_wave_device.py",
            "test_multislice.py", "test_wave_fft.py", "test_tem_flux_contract.py",
            "test_stem_cuda_pipeline.py", "test_gpu_capture_contract.py", "test_execution_migration_contract.py",
            "test_wave_checkpoint_publication.py",
            "test_tip_wave_pipeline.py", "test_column_wave_transport.py", "test_coherent_beam_gui.py",
            "test_column_wave_electric.py", "test_tip_gun_wave.py", "test_tip_gun_shared_fields.py",
            "test_electrostatic_column_transport.py", "test_wave_grid.py",
            "test_planar_gun_field.py", "test_canonical_action.py", "test_planar_tip_boundary.py", "test_driven_tip_gun.py"},
    }
    for scope, expected in required.items():
        assert {Path(path).name for path in scope_test_files(scope)} == expected


@pytest.mark.parametrize("path", ("tests/test_selected_plane.py", "tests/test_selected_plane_gui.py",
                                "tests/test_conjugate_planes.py", "tests/test_conjugate_plane_gui.py",
                                "tests/test_conjugate_plane_workspace.py"))
def test_selected_plane_requires_both_numerical_and_gui_evidence(path):
    receipt = receipt_for("field-ui")
    missing = next(node for node in receipt["collected"] if node.startswith(path + "::"))
    receipt["collected"].remove(missing)
    del receipt["cases"][missing]
    report = evaluate("field-ui", receipt)
    assert report["software_scope_status"] == "INCOMPLETE"
    assert report["criteria"]["field-ui/FU-09"]["status"] != "PASS"


def test_shared_component_regressions_are_mandatory_classical_evidence():
    required = {"tests/test_component_position_contract.py", "tests/test_component_persistence.py"}
    assert required <= set(scope_test_files("classical"))
    for path in required:
        receipt = receipt_for("classical")
        missing = next(node for node in receipt["collected"] if node.startswith(path + "::"))
        receipt["collected"].remove(missing)
        del receipt["cases"][missing]
        assert evaluate("classical", receipt)["software_scope_status"] == "INCOMPLETE"


@pytest.mark.parametrize("scope", tuple(ACCEPTANCE_SCOPES))
@pytest.mark.parametrize("fault", ["missing-file", "empty", "skipped", "crash", "interrupted"])
def test_mandatory_scope_cannot_pass_without_its_complete_executed_evidence(scope, fault):
    receipt = receipt_for(scope)
    code = 0
    if fault == "missing-file":
        receipt["cases"].pop(receipt["collected"].pop())
    elif fault == "empty":
        receipt["collected"], receipt["cases"] = [], {}
    elif fault == "skipped":
        receipt["cases"][receipt["collected"][0]]["call"] = "skipped"
    elif fault == "crash":
        code = -1
    else:
        receipt["complete"] = False
    assert evaluate(scope, receipt, pytest_exit_code=code)["software_scope_status"] == "INCOMPLETE"


def test_receipt_from_another_scope_cannot_be_relabelled():
    receipt = receipt_for("electron-execution")
    before = deepcopy(receipt)
    assert evaluate("field-ui", receipt)["exit_code"] == 1
    assert evaluate("electron-execution", receipt, selected=scope_test_files("field-ui"))["exit_code"] == 1
    assert receipt == before


def test_unknown_scope_is_rejected_in_selection_and_reporting():
    with pytest.raises(ValueError, match="Unknown acceptance scope"):
        scope_test_files("unlisted")
    with pytest.raises(ValueError, match="Unknown acceptance scope"):
        evaluate("unlisted", receipt_for("classical"))


@pytest.mark.parametrize("payload", [None, [], "not an object", 1])
def test_malformed_receipt_object_is_reported_without_aborting(payload):
    report = software_report(payload, scope="acceptance-policy", pytest_exit_code=0,
                             source_unchanged=True)
    assert report["exit_code"] == 1
    assert "Invalid pytest receipt" in report["errors"][0]


@pytest.mark.parametrize("field,value", [
    ("collected", None), ("collected", {}), ("collected", "one-test"),
    ("collected", [[]]), ("collected", [None]),
    ("cases", None), ("cases", []), ("cases", "passed"),
    ("deselected", {}), ("deselected", ""), ("deselected", [None]),
    ("complete", "true"), ("complete", 1),
])
def test_malformed_receipt_fields_cannot_pass_or_break_reporting(field, value):
    receipt = receipt_for("acceptance-policy")
    receipt[field] = value
    report = evaluate("acceptance-policy", receipt)
    assert report["exit_code"] == 1 and report["errors"]


@pytest.mark.parametrize("outcomes", [None, [], "passed", {"call": []}, {1: "passed"}])
def test_malformed_case_entry_is_nonpassing_evidence(outcomes):
    receipt = receipt_for("acceptance-policy")
    receipt["cases"][receipt["collected"][0]] = outcomes
    report = evaluate("acceptance-policy", receipt)
    assert report["exit_code"] == 1
    assert any("Invalid test outcome" in error for error in report["errors"])


def _gpu_skip_receipt():
    receipt = receipt_for("classical")
    node = "tests/test_compute_backend.py::test_auto_cuda_ray_trace_matches_cpu_with_energy_spread"
    receipt["collected"].append(node)
    receipt["cases"][node] = {"setup": "passed", "call": "skipped", "teardown": "passed"}
    receipt["skip_reasons"] = {node: "Skipped: CUDA device unavailable"}
    return receipt, node


def test_cpu_lane_may_report_reviewed_gpu_absence_without_claiming_gpu_success():
    receipt, node = _gpu_skip_receipt()
    assert evaluate("classical", receipt)["exit_code"] == 1
    report = evaluate("classical", receipt, allow_gpu_skips=True)
    assert report["exit_code"] == 0
    assert report["gpu_hardware"]["status"] == "NOT_RUN"
    assert report["gpu_hardware"]["not_run"] == [node]
    assert report["cases"][node]["call"] == "skipped"
    assert report["criteria"]["round2/R2-AT-10"]["gpu_not_run"] == [node]
    assert report["full_simulator_qualification"] == "UNQUALIFIED"


@pytest.mark.parametrize("outcomes", [
    {"setup": "passed", "call": "failed", "teardown": "passed"},
    {"setup": "passed", "call": "skipped", "teardown": "failed"},
    {"setup": "passed", "call": "skipped"},
    {"setup": "passed", "call": "passed", "teardown": "skipped"},
    {},
])
def test_cpu_lane_never_ignores_gpu_failures_or_incomplete_execution(outcomes):
    receipt, node = _gpu_skip_receipt()
    receipt["cases"][node] = outcomes
    assert evaluate("classical", receipt, allow_gpu_skips=True)["exit_code"] == 1


def test_cpu_lane_still_rejects_unreviewed_skips_and_records_gpu_execution():
    receipt, node = _gpu_skip_receipt()
    receipt["cases"][node]["call"] = "passed"
    report = evaluate("classical", receipt, allow_gpu_skips=True)
    assert report["gpu_hardware"]["status"] == "EXECUTED"
    assert not report["gpu_hardware"]["not_run"]
    receipt["cases"][receipt["collected"][0]]["call"] = "skipped"
    assert evaluate("classical", receipt, allow_gpu_skips=True)["exit_code"] == 1


@pytest.mark.parametrize("reason", ["", "Atomistic backend unavailable", "Known failure", "Skipped: memory mismatch"])
def test_cpu_lane_rejects_non_device_skip_reasons_even_for_reviewed_gpu_tests(reason):
    receipt, node = _gpu_skip_receipt()
    receipt["skip_reasons"][node] = reason
    assert evaluate("classical", receipt, allow_gpu_skips=True)["exit_code"] == 1


def test_gpu_setup_skip_still_requires_completed_teardown_and_device_reason():
    receipt, node = _gpu_skip_receipt()
    receipt["cases"][node] = {"setup": "skipped", "teardown": "passed"}
    receipt["skip_reasons"][node] = "Skipped: could not import 'cupy': No module named 'cupy'"
    assert evaluate("classical", receipt, allow_gpu_skips=True)["exit_code"] == 0
    receipt["cases"][node].pop("teardown")
    assert evaluate("classical", receipt, allow_gpu_skips=True)["exit_code"] == 1


def test_gpu_registry_keeps_cpu_parameters_mandatory_and_matches_existing_tests():
    import ast
    from temsim.acceptance_gpu import GPU_TEST_IDS, GPU_PARAMETER_IDS, is_gpu_hardware_test

    for node in GPU_TEST_IDS | GPU_PARAMETER_IDS:
        filename, name = node.split("::", 1)
        definitions = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
        assert name.split("[", 1)[0] in {item.name for item in definitions.body if isinstance(item, ast.FunctionDef)}
    prefix = ("tests/test_electrostatic_column_transport.py::"
              "test_closed_electric_and_magnetic_actions_match_reference_backend")
    assert is_gpu_hardware_test(prefix + "[CUDA GPU]")
    assert not is_gpu_hardware_test(prefix + "[Numba CPU]")
    assert not is_gpu_hardware_test(prefix + "[CPU]")
    assert not is_gpu_hardware_test("tests/test_stem_cuda_pipeline.py::test_toolbar_require_gpu_does_not_retry_stem_on_cpu")
