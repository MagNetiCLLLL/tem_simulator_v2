"""Acceptance evidence must not turn retired or partial image tests green."""
import ast
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / "scripts/validate_development_spec.py"))
CRITERIA = VALIDATOR["CRITERIA"]
evaluate = VALIDATOR["evaluate_criteria"]
exit_code = VALIDATOR["acceptance_exit_code"]


def passing_cases():
    return [{"test": prefix, "status": "PASS"}
            for prefix in sorted({prefix for _, prefixes in CRITERIA.values() for prefix in prefixes})]


def test_every_acceptance_prefix_has_an_existing_test():
    missing = []
    for number, (_, prefixes) in CRITERIA.items():
        for prefix in prefixes:
            module, name = prefix.split(".", 1)
            tree = ast.parse((ROOT / "tests" / f"{module}.py").read_text(encoding="utf-8"))
            if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name.startswith(name) for node in tree.body):
                missing.append((number, prefix))
    assert not missing


def test_partial_numerical_passes_do_not_qualify_production_images():
    rows = evaluate(passing_cases(), {"status": "PASS", "wave_images": {"status": "UNAVAILABLE"}})
    assert {key for key, row in rows.items() if row["status"] == "BLOCKED"} == {
        "AT-12", "AT-13", "AT-14", "AT-31"}
    assert all(not row["missing_test_prefixes"] for row in rows.values())
    assert exit_code(rows, source_unchanged=True, pytest_exit_code=0, allow_no_gpu=True) == 1


@pytest.mark.parametrize("status", ["FAIL", "NOT_RUN"])
def test_migration_blocker_preserves_failed_or_skipped_evidence(status):
    cases = passing_cases()
    prefix = CRITERIA[12][1][0]
    next(row for row in cases if row["test"] == prefix)["status"] = status
    row = evaluate(cases, {"status": "NOT_RUN"})["AT-12"]
    assert row["status"] == status
    assert row["blocked_reason"]


def test_missing_required_prefix_is_reported_even_with_other_passing_tests():
    prefix = CRITERIA[12][1][0]
    cases = [row for row in passing_cases() if row["test"] != prefix]
    row = evaluate(cases, {"status": "NOT_RUN"})["AT-12"]
    assert row["status"] == "NOT_RUN"
    assert row["tests"]
    assert row["missing_test_prefixes"] == [prefix]


@pytest.mark.parametrize("receipt, expected", [
    ({"status": "PASS"}, "BLOCKED"),
    ({"status": "PASS", "wave_images": {"status": "NOT_RUN"}}, "BLOCKED"),
    ({"status": "PASS", "wave_images": {"status": "PASS"}}, "PASS"),
    ({"status": "FAIL"}, "FAIL"),
    ({"status": "INCONCLUSIVE"}, "INCONCLUSIVE"),
    ({"status": "NOT_RUN"}, "NOT_RUN"),
])
def test_installation_scope_does_not_silently_drop_the_wave_case(receipt, expected):
    assert evaluate(passing_cases(), receipt)["AT-31"]["status"] == expected


def test_only_actual_gpu_not_run_can_be_waived():
    rows = {f"AT-{number:02}": {"status": "PASS"} for number in CRITERIA}
    rows["AT-25"]["status"] = rows["AT-26"]["status"] = "NOT_RUN"
    assert exit_code(rows, source_unchanged=True, pytest_exit_code=0) == 1
    assert exit_code(rows, source_unchanged=True, pytest_exit_code=0, allow_no_gpu=True) == 0
    rows["AT-12"]["status"] = "NOT_RUN"
    assert exit_code(rows, source_unchanged=True, pytest_exit_code=0, allow_no_gpu=True) == 1
    rows["AT-12"]["status"] = "PASS"
    rows["AT-25"]["status"] = "FAIL"
    assert exit_code(rows, source_unchanged=True, pytest_exit_code=0, allow_no_gpu=True) == 1


@pytest.mark.parametrize("unchanged, pytest_status", [(False, 0), (True, 1), (True, 2), (True, 5)])
def test_changed_source_or_unsuccessful_test_run_cannot_pass(unchanged, pytest_status):
    rows = {f"AT-{number:02}": {"status": "PASS"} for number in CRITERIA}
    assert exit_code(rows, source_unchanged=unchanged, pytest_exit_code=pytest_status) == 1
