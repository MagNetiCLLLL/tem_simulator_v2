"""Fail-closed, namespaced software evidence; never physical qualification."""
from collections import Counter


CLASSICAL_CRITERIA = {
    "product-usability/AT-01": ("Physical source admission", ("tests/test_source_admission.py",)),
    "product-usability/AT-02": ("Complete captured instrument", ("tests/test_working_point_contract.py", "tests/test_input_assets.py")),
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


def merge_criteria(*groups):
    result = {}
    for group in groups:
        for key, value in group.items():
            if "/" not in key or key in result:
                raise ValueError(f"Missing namespace or duplicate criterion: {key}")
            result[key] = value
    return result


def classical_report(receipt, *, pytest_exit_code, source_unchanged,
                     selected=CLASSICAL_TESTS, criteria=CLASSICAL_CRITERIA):
    """Require every collected item, including setup/teardown, without skips.

    The allowlist is a scope contract, not inferred from whichever tests happened
    to run. Unknown/omitted files, deselection, duplicates and empty collection
    fail closed. Synthetic receipts exercise this policy, not any physics.
    """
    expected_files = {path for _, paths in criteria.values() for path in paths}
    selected = tuple(selected)
    collected = receipt.get("collected", [])
    cases = receipt.get("cases", {})
    errors = []
    if set(selected) != expected_files or len(selected) != len(set(selected)):
        errors.append("Selected tests differ from the declared scope")
    if not receipt.get("complete"):
        errors.append("Missing or interrupted pytest receipt")
    if receipt.get("deselected"):
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
        "schema": "classical-software-acceptance-v1", "scope": "classical-particle-software",
        "software_scope_status": "PASS" if passed else "INCOMPLETE",
        "full_simulator_qualification": "UNQUALIFIED",
        "exit_code": 0 if passed else 1, "criteria": merge_criteria(rows), "errors": errors,
        "selected_tests": list(selected), "collected_count": len(set(collected)),
        "cases": cases, "source_unchanged_during_tests": source_unchanged,
        "pytest_exit_code": pytest_exit_code,
        "exclusions": {
            "coherent-tip-to-image": "Paused by user; source-admission tests are not image qualification",
            "actual-gpu-scientific-parity": "NOT_RUN in this software lane; emulated policy tests are not hardware evidence",
            "native-desktop": "Offscreen tests only",
            "experimental-calibration": "NOT_RUN; no OEM or experimental qualification",
            "round2/R2-AT-13..40": "Later packages are not selected by the R2-00..03 software lane",
        },
    }
