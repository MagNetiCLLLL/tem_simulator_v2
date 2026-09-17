"""Receipt policy fixtures, not image/particle/GPU scientific qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from temsim.acceptance import CLASSICAL_CRITERIA, CLASSICAL_TESTS, classical_report, merge_criteria

ROOT = Path(__file__).resolve().parents[1]


def passing_receipt():
    nodes = [path + "::synthetic_receipt_only" for path in CLASSICAL_TESTS]
    return {"collected": nodes, "complete": True, "cases": {
        node: {"setup": "passed", "call": "passed", "teardown": "passed"} for node in nodes}}


def evaluate(receipt, **kwargs):
    return classical_report(receipt, pytest_exit_code=kwargs.pop("pytest_exit_code", 0),
                            source_unchanged=kwargs.pop("source_unchanged", True), **kwargs)


def test_synthetic_pass_qualifies_only_declared_software_scope():
    report = evaluate(passing_receipt())
    assert report["exit_code"] == 0 and report["software_scope_status"] == "PASS"
    assert report["full_simulator_qualification"] == "UNQUALIFIED"
    assert "actual-gpu-scientific-parity" in report["exclusions"]
    script = runpy.run_path(str(ROOT / "scripts/validate_classical_scope.py"))
    legacy = script["legacy_exclusions"](ROOT)
    combined = merge_criteria(report["criteria"], legacy)
    assert combined["product-usability/AT-12"]["status"] == "PASS"
    assert combined["legacy-development/AT-12"]["status"] == "BLOCKED"
    assert combined["round2/R2-AT-12"]["status"] == "PASS"
    assert {key for key, row in legacy.items() if row["status"] == "BLOCKED"} == {
        "legacy-development/AT-12", "legacy-development/AT-13", "legacy-development/AT-14"}


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
@pytest.mark.parametrize("outcome", ["failed", "skipped", "missing"])
def test_each_test_phase_is_required(phase, outcome):
    receipt = passing_receipt()
    case = next(iter(receipt["cases"].values()))
    if outcome == "missing":
        case.pop(phase)
    else:
        case[phase] = outcome
    assert evaluate(receipt)["exit_code"] == 1


@pytest.mark.parametrize("problem", ["missing_file", "missing_case", "unknown", "deselected", "duplicate", "incomplete"])
def test_missing_unknown_or_incomplete_evidence_never_passes(problem):
    receipt = passing_receipt()
    node = receipt["collected"][0]
    if problem == "missing_file":
        receipt["collected"].remove(node)
        receipt["cases"].pop(node)
    elif problem == "missing_case":
        receipt["cases"].pop(node)
    elif problem == "unknown":
        receipt["collected"].append("tests/not_selected.py::test_unknown")
    elif problem == "deselected":
        receipt["deselected"] = [node]
    elif problem == "duplicate":
        receipt["collected"].append(node)
    else:
        receipt["complete"] = False
    assert evaluate(receipt)["exit_code"] == 1


def test_scope_omission_and_source_change_fail_closed():
    assert evaluate(passing_receipt(), selected=CLASSICAL_TESTS[:-1])["exit_code"] == 1
    assert evaluate(passing_receipt(), source_unchanged=False)["exit_code"] == 1
    for code in (1, 2, 3, 4, 5, 124, None):
        assert evaluate(passing_receipt(), pytest_exit_code=code)["exit_code"] == 1
    with pytest.raises(ValueError):
        merge_criteria({"AT-12": {}})
    with pytest.raises(ValueError):
        merge_criteria(CLASSICAL_CRITERIA, CLASSICAL_CRITERIA)


def test_allowlist_files_exist_and_ci_uses_explicit_classical_scope():
    assert all((ROOT / path).is_file() for path in CLASSICAL_TESTS)
    workflow = (ROOT / ".github/workflows/tem-p0.yml").read_text()
    assert "validate_classical_scope.py --scope classical" in workflow
    assert "validate_development_spec.py --allow-no-gpu" not in workflow


def test_real_failing_pytest_child_produces_nonzero_scope_exit(tmp_path):
    # Deliberately failing isolated software test verifies the same plugin and
    # aggregator used by CI; the main suite expects that child's failure.
    target = tmp_path / "test_intentional.py"
    target.write_text("def test_receipt_failure():\n    assert False, 'intentional acceptance failure'\n")
    output = tmp_path / "receipt.json"
    run = subprocess.run([sys.executable, "-m", "pytest", "test_intentional.py", "-p", "temsim.acceptance_pytest"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=45,
                         env={**os.environ, "TEMSIM_ACCEPTANCE_RECEIPT": str(output),
                              "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""})
    assert run.returncode == 1, run.stdout + run.stderr
    receipt = json.loads(output.read_text())
    report = classical_report(receipt, pytest_exit_code=run.returncode, source_unchanged=True,
        selected=("test_intentional.py",), criteria={"round2/R2-AT-05": ("Failing fixture", ("test_intentional.py",))})
    assert report["exit_code"] == 1
    assert report["criteria"]["round2/R2-AT-05"]["status"] == "FAIL"


def test_source_inventory_detects_code_and_configuration_changes(tmp_path):
    script = runpy.run_path(str(ROOT / "scripts/validate_classical_scope.py"))
    for name in ("pyproject.toml", "AGENTS.md", "PROJECT_FUNCTION_SPEC.md"):
        (tmp_path / name).write_text("fixture")
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "fixture.toml"
    path.write_text("x = 1")
    before = script["source_hashes"](tmp_path)
    path.write_text("x = 2")
    assert before != script["source_hashes"](tmp_path)


def test_validation_timeout_terminates_owned_interpreter(tmp_path):
    from temsim.validation_process import run_bounded
    from time import monotonic
    log = tmp_path / "timeout.log"
    started = monotonic()
    with log.open("w") as stream:
        result = run_bounded([sys.executable, "-c", "import time; time.sleep(60)"],
                             stdout=stream, stderr=stream, timeout=1)
    assert result.returncode == 124
    assert monotonic() - started < 20
