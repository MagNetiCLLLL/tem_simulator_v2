"""Real child-process checks for acceptance evidence, not microscope physics."""
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    return runpy.run_path(str(ROOT / "scripts/validate_classical_scope.py"))


def test_real_child_gpu_skip_is_recorded_separately_in_explicit_cpu_lane(runner, tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    target = tests / "test_compute_backend.py"
    target.write_text(
        "import pytest\n"
        "def test_cpu_contract():\n    assert True\n"
        "def test_auto_cuda_ray_trace_matches_cpu_with_energy_spread():\n"
        "    pytest.skip('CUDA device unavailable')\n", encoding="utf-8",
    )
    relative = "tests/test_compute_backend.py"
    report = runner["run_scope"](tmp_path, tmp_path / "evidence", "acceptance-policy", 45,
        criteria={"policy/cpu-fixture": ("CPU contract and real-device dependency", (relative,))},
        selected=(relative,), allow_gpu_skips=True)
    assert report["exit_code"] == 0
    assert report["collected_count"] == 2
    assert report["gpu_hardware"]["status"] == "NOT_RUN"
    assert len(report["gpu_hardware"]["not_run"]) == 1
    receipt = json.loads(Path(report["artifacts"]["receipt"]["path"]).read_text())
    assert receipt["cases"][relative + "::test_auto_cuda_ray_trace_matches_cpu_with_energy_spread"]["call"] == "skipped"


@pytest.mark.parametrize("case, expected_code", [
    ("pass", 0), ("missing", 4), ("empty", 5), ("fail", 1),
    ("timeout", 124), ("source-change", 0),
])
def test_real_child_evidence_and_fail_closed_report(runner, tmp_path, case, expected_code):
    target = tmp_path / "test_fixture.py"
    definitions = {
        "pass": "def test_ok():\n    assert 1 + 1 == 2\n",
        "empty": "# deliberately no tests\n",
        "fail": "def test_failure():\n    assert False, 'intentional failure'\n",
        "timeout": "import time\ndef test_timeout():\n    time.sleep(60)\n",
        "source-change": "from pathlib import Path\ndef test_change():\n    Path('configs/input.toml').write_text('x=2')\n",
    }
    if case != "missing":
        target.write_text(definitions[case], encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/input.toml").write_text("x=1")
    out = tmp_path / "evidence"
    report = runner["run_scope"](tmp_path, out, "acceptance-policy", 1 if case == "timeout" else 45,
        criteria={"policy/fixture": ("Isolated software fixture", (target.name,))},
        selected=(target.name,))
    assert report["pytest_exit_code"] == expected_code, report
    assert (report["software_scope_status"] == "PASS") == (case == "pass")
    assert report["exit_code"] == (0 if case == "pass" else 1)
    assert report["full_simulator_qualification"] == "UNQUALIFIED"
    assert report["artifacts"]["log"]["exists"]
    if case != "timeout":
        assert report["artifacts"]["junit"]["exists"]
        assert report["artifacts"]["receipt"]["exists"]
    if case == "source-change":
        assert not report["source_unchanged_during_tests"]
    assert json.loads((out / "report.json").read_text()) == report
    selection = json.loads((out / f"selection-{report['run_id']}.json").read_text())
    assert selection["selected_tests"] == [target.name]
    assert selection["dependency_versions"]["pytest"]
    assert selection["validation_environment"]["TEMSIM_CPU_THREADS"] == "1"
    assert selection["validation_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert selection["git"]["sha"] is None  # fixture is explicitly not a Git checkout
    assert (out / f"report-{report['run_id']}.json").is_file()


def test_launch_failure_replaces_previous_success_and_retains_diagnostics(runner, tmp_path, monkeypatch):
    execute = runner["run_scope"]
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "report.json").write_text('{"software_scope_status":"PASS","run_id":"old"}')

    def failed_start(*args, **kwargs):
        # An incomplete new attempt must invalidate the old top-level PASS
        # before attempting to launch the child.
        pending = json.loads((out / "report.json").read_text())
        assert pending["software_scope_status"] == "NOT_RUN"
        raise OSError("intentional process launch failure")

    monkeypatch.setitem(execute.__globals__, "run_bounded", failed_start)
    report = execute(tmp_path, out, "acceptance-policy", 1,
                     criteria={"policy/launch": ("Launch failure", ("test_missing.py",))},
                     selected=("test_missing.py",))
    assert report["exit_code"] == 1
    assert report["pytest_exit_code"] is None
    assert "intentional process launch failure" in " ".join(report["errors"])
    assert report["run_id"] != "old"
    assert not report["artifacts"]["junit"]["exists"]


def test_lockfile_changes_are_bound_to_validation(runner, tmp_path):
    path = tmp_path / "requirements/validation-cpu-lock.txt"
    path.parent.mkdir()
    path.write_text("numpy==1.0\n")
    before = runner["source_hashes"](tmp_path)
    assert "requirements/validation-cpu-lock.txt" in before
    path.write_text("numpy==2.0\n")
    assert runner["source_hashes"](tmp_path) != before
    path.unlink()
    assert runner["source_hashes"](tmp_path) != before


def test_preparation_failure_cannot_leave_previous_success(runner, tmp_path, monkeypatch):
    execute = runner["run_scope"]
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "report.json").write_text('{"software_scope_status":"PASS","run_id":"old"}')

    def unreadable_source(root):
        raise OSError("intentional source inventory failure")

    monkeypatch.setitem(execute.__globals__, "source_hashes", unreadable_source)
    with pytest.raises(OSError, match="source inventory"):
        execute(tmp_path, out, "acceptance-policy", 1)
    pending = json.loads((out / "report.json").read_text())
    assert pending["software_scope_status"] == "NOT_RUN"
    assert pending["exit_code"] == 1 and pending["run_id"] != "old"


def test_report_only_does_not_run_paused_calculations(runner, tmp_path, monkeypatch):
    execute = runner["run_scope"]
    monkeypatch.setitem(execute.__globals__, "run_bounded",
                        lambda *a, **k: pytest.fail("Report-only must not launch pytest"))
    report = execute(ROOT, tmp_path, "full-report", 1)
    assert report["software_scope_status"] == "NOT_RUN"
    assert report["exit_code"] == 1
    assert report["command"] == []
    assert report["git"]["sha"]
    assert report["git"]["dirty"] in (True, False)


def test_ci_runs_every_declared_scope_and_retains_failure_evidence():
    import yaml
    from temsim.acceptance import ACCEPTANCE_SCOPES

    jobs = yaml.safe_load((ROOT / ".github/workflows/tem-p0.yml").read_text())["jobs"]
    feature = jobs["feature-acceptance"]
    assert set(feature["strategy"]["matrix"]["scope"]) == set(ACCEPTANCE_SCOPES) - {"classical"}
    assert feature["strategy"]["fail-fast"] is False
    steps = feature["steps"]
    assert any("--scope ${{ matrix.scope }}" in step.get("run", "") for step in steps)
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"
    assert "${{ matrix.scope }}" in upload["with"]["name"]
    cpu_steps = jobs["cpu-acceptance"]["steps"]
    commands = "\n".join(step.get("run", "") for step in cpu_steps)
    assert "pip wheel" in commands and "acceptance-install" in commands
    wheel = next(step for step in cpu_steps if step.get("id") == "wheel")
    assert "$env:RUNNER_TEMP" in wheel["run"] and "[guid]::NewGuid()" in wheel["run"]
    assert '$Wheels.Count -ne 1' in wheel["run"]
    assert 'tem_simulator_v2-*.whl' in wheel["run"]
    assert 'pip check' in wheel["run"]
    smoke = next(step for step in cpu_steps if ' -I ' in step.get("run", ""))
    assert 'Push-Location $env:INSTALL_ROOT' in smoke["run"]
    assert '$env:GITHUB_WORKSPACE "scripts/installation_diagnostic_smoke.py"' in smoke["run"]
    assert 'exit $SmokeCode' in smoke["run"]
    classical = next(step for step in cpu_steps if '--scope classical' in step.get("run", ""))
    assert classical["if"] == "${{ !cancelled() && steps.environment.outcome == 'success' }}"
    assert not any(step.get("continue-on-error") for step in cpu_steps)
    report_commands = "\n".join(step.get("run", "") for step in jobs["full-scope-status"]["steps"])
    assert "pip install -c requirements/validation-cpu-lock.txt threadpoolctl" in report_commands
