"""Fail-closed, namespaced software evidence; never physical qualification."""
from collections import Counter


CLASSICAL_CRITERIA = {
    "product-usability/AT-01": ("Physical source admission", ("tests/test_source_admission.py",)),
    "product-usability/AT-02": ("Complete captured instrument", ("tests/test_working_point_contract.py", "tests/test_input_assets.py",
        "tests/test_component_position_contract.py", "tests/test_component_persistence.py", "tests/test_shared_tip.py")),
    "product-usability/AT-06": ("Transactional alignment", ("tests/test_alignment_transactions.py",)),
    "product-usability/AT-12": ("Unknown-input and opt-in vacuum invalidation", ("tests/test_parameter_registry.py", "tests/test_vacuum_opt_in.py")),
    "round2/R2-AT-03": ("Explicit classical acceptance", ("tests/test_classical_acceptance.py",)),
    "round2/R2-AT-04": ("Legacy blockers preserved", ("tests/test_development_acceptance.py",)),
    "round2/R2-AT-05": ("Namespaced, fail-closed receipts", ("tests/test_classical_acceptance.py",)),
    "round2/R2-AT-06": ("Worker entry and transition evidence", ("tests/test_job_lifecycle.py",)),
    "round2/R2-AT-07": ("Captured inputs and latest live work", ("tests/test_background_calculation_requests.py", "tests/test_background_preview_gui.py", "tests/test_calculation_controller.py")),
    "round2/R2-AT-08": ("Exactly-once terminal ownership", ("tests/test_job_lifecycle.py", "tests/test_job_coordination.py")),
    "round2/R2-AT-09": ("Independent High and experiment ownership", ("tests/test_job_coordination_gui.py", "tests/test_result_readout.py")),
    "round2/R2-AT-10": ("Backend absence/resource policy", ("tests/test_ray_gpu_policy.py", "tests/test_backend_failure_semantics.py")),
    "round2/R2-AT-11": ("Original non-retryable failures", ("tests/test_backend_failure_semantics.py",)),
    "round2/R2-AT-12": ("Stage reporting and device ownership fixtures", ("tests/test_backend_failure_semantics.py", "tests/test_ray_device_residency.py", "tests/test_calculation_performance.py")),
}
CLASSICAL_TESTS = tuple(sorted({path for _, paths in CLASSICAL_CRITERIA.values() for path in paths}))


ACCEPTANCE_SCOPES = {
    "classical": {
        "description": "Existing classical particle software contracts",
        "evidence_kind": "software-contracts-with-offscreen-ui",
        "criteria": CLASSICAL_CRITERIA,
    },
    "gun-fields": {
        "description": "Classical tip geometry and bounded electrostatic field checks",
        "evidence_kind": "bounded-numerical-and-software-checks",
        "criteria": {
            "gun-fields/GF-01": ("Conductor geometry, field boundaries and exact field cache", (
                "tests/test_closed_gun_field.py", "tests/test_axisymmetric_cut_field.py")),
            "gun-fields/GF-02": ("Continuous tip emission, transport and saved inputs", (
                "tests/test_continuous_tip_curvature.py", "tests/test_tip_curvature_comparison.py")),
            "gun-fields/GF-03": ("Exact field identity, conservative reuse and domain comparison metrics", (
                "tests/test_diagnostic_field_identity.py", "tests/test_diagnostic_gun_domains.py")),
        },
    },
    "electron-execution": {
        "description": "Virtual electron electromagnetic transport and isolated execution",
        "evidence_kind": "bounded-numerical-and-real-process-checks",
        "criteria": {
            "electron-execution/EE-01": ("Independent electromagnetic references and supported scene", (
                "tests/test_magnetic_test_particle.py", "tests/test_test_electron_scene.py")),
            "electron-execution/EE-02": ("Compiled field and trajectory parity with complete hardware stops", (
                "tests/test_closed_gun_execution.py", "tests/test_test_electron_compiled.py",
                "tests/test_test_electron_intercepts.py", "tests/test_test_electron_performance.py")),
            "electron-execution/EE-08": ("Shared production column and virtual-electron transport", (
                "tests/test_shared_electron_column.py",)),
            "electron-execution/EE-03": ("Owned real process, cancellation and accepted progress", (
                "tests/test_test_electron_execution.py",)),
            "electron-execution/EE-04": ("Code-bound native caches without deleting historical evidence", (
                "tests/test_numba_cache.py",)),
            "electron-execution/EE-05": ("Captured field and execution identity through real metadata boundaries", (
                "tests/test_diagnostic_scene_identity.py", "tests/test_diagnostic_execution_identity.py")),
            "electron-execution/EE-06": ("Bounded failure recovery and retained worker diagnostics", (
                "tests/test_electron_execution_faults.py", "tests/test_electron_execution_diagnostics.py",
                "tests/test_electron_execution_protocol.py", "tests/test_electron_resource_cleanup.py",
                "tests/test_cpu_resources.py")),
            "electron-execution/EE-07": ("Safe independent diagnostic history archives and source rejection", (
                "tests/test_electron_diagnostic_session.py",)),
        },
    },
    "field-ui": {
        "description": "Combined magnetic fields, virtual electron interaction and hardware editing",
        "evidence_kind": "offscreen-ui-and-bounded-field-checks",
        "criteria": {
            "field-ui/FU-01": ("Combined field support and direction-aware field-line geometry", (
                "tests/test_magnetic_field_scene.py", "tests/test_magnetic_field_lines.py")),
            "field-ui/FU-02": ("Cached field rendering and physical display-range linkage", (
                "tests/test_magnetic_field_3d.py", "tests/test_magnetic_field_canvas.py",
                "tests/test_incremental_magnetic_scene.py", "tests/test_magnetic_navigation_link.py")),
            "field-ui/FU-03": ("Latest electron edits, continuous progress and dock lifetime", (
                "tests/test_magnetic_test_electron_gui.py", "tests/test_continuous_electron_gui.py",
                "tests/test_virtual_electron_dock.py")),
            "field-ui/FU-04": ("In-place actual hardware bindings and transactional edits", (
                "tests/test_hardware_tuning_bindings.py", "tests/test_hardware_tuning_gui.py",
                "tests/test_hardware_tuning_feedback.py")),
            "field-ui/FU-05": ("Captured magnetic identities and equivalent deflector reference response", (
                "tests/test_magnetic_field_identity.py",)),
            "field-ui/FU-06": ("Explicit diagnostic retry and rejection of late failure signals", (
                "tests/test_electron_failure_gui.py",)),
            "field-ui/FU-07": ("Historical sessions and centralized diagnostic record ownership", (
                "tests/test_electron_session_gui.py", "tests/test_diagnostic_electron_record.py")),
            "field-ui/FU-08": ("Explicit assembly selection, configuration and working-point restoration", (
                "tests/test_assembly_selection_state.py", "tests/test_assembly_navigation.py",
                "tests/test_instrument_configuration.py", "tests/test_working_point_restore_gui.py")),
            "field-ui/FU-09": ("Captured-optics selected-Z conjugacy and latest cached plane readout", (
                "tests/test_selected_plane.py", "tests/test_selected_plane_gui.py")),
            "field-ui/FU-11": ("Selected-plane upstream hardware projections, recorded interceptions and beam views", (
                "tests/test_plane_hardware_geometry.py", "tests/test_plane_hardware_overlay.py",
                "tests/test_plane_cutoff_events.py", "tests/test_lazy_ray_panels.py",
                "tests/test_ray_extent_workspace.py", "tests/test_incremental_ray_scene.py",
                "tests/test_beam_analysis_modes.py", "tests/test_transverse_source_tracking.py",
                "tests/test_filter_plane_analysis.py", "tests/test_ray_flight_time_colours.py",
                "tests/test_transverse_plot_sizes.py", "tests/test_transverse_plot_size_persistence.py")),
            "field-ui/FU-10": ("Separate coherent sessions and continuous latest-Z interaction", (
                "tests/test_coherent_beam_gui.py",)),
        },
    },
    "particle-continuation": {
        "description": "Executed classical checkpoints, material reuse and exact result archives",
        "evidence_kind": "software-and-bounded-particle-persistence-checks",
        "criteria": {
            "particle-continuation/PC-01": ("Executed optical checkpoints and full-calculation continuation", (
                "tests/test_particle_sections.py", "tests/test_completed_particle_sections.py")),
            "particle-continuation/PC-02": ("Material continuation with incident-state dependency checks", (
                "tests/test_material_particle_sections.py", "tests/test_material_section_resume.py",
                "tests/test_particle_section_eds_reuse.py")),
            "particle-continuation/PC-03": ("Exact saved records and lossless compact continuation", (
                "tests/test_particle_section_io.py", "tests/test_particle_archive_compression.py",
                "tests/test_particle_section_eds_archive.py")),
            "particle-continuation/PC-04": ("Archive identity and latest file-request ownership", (
                "tests/test_section_archive_identity.py", "tests/test_result_file_request_routing.py",
                "tests/test_result_files_gui.py")),
        },
    },
    "performance-observation": {
        "description": "Opt-in performance receipts and bounded benchmark admissions",
        "evidence_kind": "software-measurement-contract-checks-not-performance-qualification",
        "criteria": {
            "performance-observation/PO-01": ("Actual child execution observations preserve unprofiled results", (
                "tests/test_electron_execution_performance.py",)),
            "performance-observation/PO-02": ("Declared diagnostic cases, reference checks and executed-result reuse", (
                "tests/test_electron_response_benchmark.py",)),
            "performance-observation/PO-03": ("Input-to-painted-result latency and exact GUI cache accounting", (
                "tests/test_continuous_electron_response_benchmark.py",)),
            "performance-observation/PO-04": ("Particle identity, executed prefixes and bounded benchmark ownership", (
                "tests/test_particle_benchmark.py",)),
        },
    },
    "acceptance-policy": {
        "description": "Declared acceptance scopes and bounded fail-closed evidence collection",
        "evidence_kind": "synthetic-and-real-child-software-policy-checks",
        "criteria": {
            "acceptance-policy/AP-01": ("Explicit scope selection and complete mandatory receipts", (
                "tests/test_acceptance_scopes.py",)),
            "acceptance-policy/AP-02": ("Runner records failures, missing tests, empty collection and timeout", (
                "tests/test_acceptance_runner.py",)),
            "acceptance-policy/AP-03": ("Bounded owned process cleanup and explicit cleanup failure", (
                "tests/test_validation_process.py",)),
        },
    },
    "coherent-development": {
        "description": "Bounded tip-only coherent inputs and exact post-gun observation workflow",
        "evidence_kind": "development-operators-and-offscreen-software-checks-not-full-chain-qualification",
        "criteria": {
            "coherent-development/CW-01": ("Explicit tip boundary, domain preflight and preserved source inputs", (
                "tests/test_coherent_inputs.py", "tests/test_tip_coherent_emission.py")),
            "coherent-development/CW-02": ("Executed upstream waves, exact plane routing and physical absorption", (
                "tests/test_tip_wave_pipeline.py", "tests/test_column_wave_transport.py",
                "tests/test_wave_checkpoint_publication.py",
                "tests/test_column_wave_electric.py", "tests/test_tip_gun_wave.py",
                "tests/test_electrostatic_column_transport.py", "tests/test_wave_grid.py",
                "tests/test_tip_gun_shared_fields.py", "tests/test_canonical_action.py")),
            "coherent-development/CW-03": ("Captured wave sessions, latest Z, resource ownership and bounded display", (
                "tests/test_coherent_beam_gui.py",)),
            "coherent-development/CW-04": ("Vacuum electrostatic boundaries do not depend on prescribed source phase", (
                "tests/test_planar_gun_field.py",)),
            "coherent-development/CW-05": ("Conditional material waves, exact deterministic reuse and slice continuation", (
                "tests/test_inelastic_wave.py",)),
        },
    },
}


def _scope_definition(scope: str) -> dict:
    try:
        return ACCEPTANCE_SCOPES[scope]
    except KeyError:
        raise ValueError(f"Unknown acceptance scope: {scope}") from None


def scope_test_files(scope: str) -> tuple[str, ...]:
    """Return complete test files for one declared scope; never infer a lane."""
    criteria = _scope_definition(scope)["criteria"]
    return tuple(sorted({path for _, paths in criteria.values() for path in paths}))


def merge_criteria(*groups):
    result = {}
    for group in groups:
        for key, value in group.items():
            if "/" not in key or key in result:
                raise ValueError(f"Missing namespace or duplicate criterion: {key}")
            result[key] = value
    return result


def software_report(receipt, *, pytest_exit_code, source_unchanged,
                    scope="classical", selected=None, criteria=None):
    """Require every collected item, including setup/teardown, without skips.

    The allowlist is a scope contract, not inferred from whichever tests happened
    to run. Unknown/omitted files, deselection, duplicates and empty collection
    fail closed. Synthetic receipts exercise this policy, not any physics.
    """
    definition = _scope_definition(scope)
    if criteria is None:
        criteria = definition["criteria"]
    if selected is None:
        selected = scope_test_files(scope)
    expected_files = {path for _, paths in criteria.values() for path in paths}
    selected = tuple(selected)
    errors = []
    if not isinstance(receipt, dict):
        errors.append("Invalid pytest receipt: expected an object")
        receipt = {}
    collected = receipt.get("collected", [])
    if not isinstance(collected, list) or not all(isinstance(node, str) for node in collected):
        errors.append("Invalid collected test IDs: expected a list of strings")
        collected = []
    cases = receipt.get("cases", {})
    if not isinstance(cases, dict):
        errors.append("Invalid test outcomes: expected an object")
        cases = {}
    else:
        valid_cases = {}
        for node, outcomes in cases.items():
            if not isinstance(node, str) or not isinstance(outcomes, dict) or not all(
                    isinstance(phase, str) and isinstance(outcome, str)
                    for phase, outcome in outcomes.items()):
                errors.append("Invalid test outcome entry: expected named phase outcomes")
                continue
            valid_cases[node] = outcomes
        cases = valid_cases
    deselected = receipt.get("deselected", [])
    if not isinstance(deselected, list) or not all(isinstance(node, str) for node in deselected):
        errors.append("Invalid deselected test IDs: expected a list of strings")
    if set(selected) != expected_files or len(selected) != len(set(selected)):
        errors.append("Selected tests differ from the declared scope")
    if receipt.get("complete") is not True:
        errors.append("Missing or interrupted pytest receipt")
    if deselected:
        errors.append("Tests were deselected")
    if any(n != 1 for n in Counter(collected).values()):
        errors.append("Duplicate collected test IDs")
    if any(node.split("::")[0] not in expected_files for node in collected):
        errors.append("Unknown test outside the declared scope")
    if set(cases) - set(collected):
        errors.append("Uncollected test outcomes")
    if not source_unchanged:
        errors.append("Source or input definitions changed during execution")
    if pytest_exit_code != 0:
        errors.append(f"pytest exited {pytest_exit_code}")
    rows = {}
    for key, (title, files) in criteria.items():
        nodes = [node for node in collected if node.split("::")[0] in files]
        missing = [path for path in files if not any(node.startswith(path + "::") for node in nodes)]
        status = "PASS"
        for node in nodes:
            outcomes = cases.get(node, {})
            if "failed" in outcomes.values():
                status = "FAIL"
                break
            if outcomes != {"setup": "passed", "call": "passed", "teardown": "passed"}:
                status = "NOT_RUN"
        if missing or not nodes:
            status = "NOT_RUN" if status != "FAIL" else status
        rows[key] = {"description": title, "status": status, "tests": nodes, "missing_files": missing}
    passed = not errors and bool(rows) and all(row["status"] == "PASS" for row in rows.values())
    return {
        "schema": ("classical-software-acceptance-v1" if scope == "classical"
                   else "scoped-software-acceptance-v1"),
        "scope": "classical-particle-software" if scope == "classical" else scope,
        "scope_key": scope, "description": definition["description"],
        "evidence_kind": definition["evidence_kind"],
        "software_scope_status": "PASS" if passed else "INCOMPLETE",
        "full_simulator_qualification": "UNQUALIFIED",
        "exit_code": 0 if passed else 1, "criteria": merge_criteria(rows), "errors": errors,
        "selected_tests": list(selected), "collected_count": len(set(collected)),
        "cases": cases, "source_unchanged_during_tests": source_unchanged,
        "pytest_exit_code": pytest_exit_code,
        "exclusions": {
            "coherent-tip-to-image": "Development resumed; bounded operator and source-admission tests are not full image qualification",
            "actual-gpu-scientific-parity": "NOT_RUN in this software lane; emulated policy tests are not hardware evidence",
            "native-desktop": "Offscreen tests only",
            "experimental-calibration": "NOT_RUN; no OEM or experimental qualification",
            **({"round2/R2-AT-13..40": "Later packages are not selected by the R2-00..03 software lane"}
               if scope == "classical" else {}),
        },
    }
