"""Bounded, explicit acceptance scopes; never full-instrument qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from uuid import uuid4

from temsim.acceptance import ACCEPTANCE_SCOPES, software_report, scope_test_files, merge_criteria
from temsim.validation_process import run_bounded


def source_hashes(root):
    paths = set()
    for directory in ("src/temsim", "tests", "scripts", "configs", ".github/workflows", "requirements"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix.lower() in
                     {".py", ".toml", ".json", ".cif", ".mcif", ".yml", ".yaml", ".txt"})
    paths.update(root / name for name in ("pyproject.toml", "AGENTS.md", "PROJECT_FUNCTION_SPEC.md")
                 if (root / name).is_file())
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def legacy_exclusions(root):
    legacy = runpy.run_path(str(root / "scripts/validate_development_spec.py"))
    blockers = legacy["SOURCE_MIGRATION_BLOCKERS"]
    return {f"legacy-development/AT-{number:02}": {
        "description": title, "status": "BLOCKED" if number in blockers else "NOT_RUN",
        "reason": blockers.get(number, "Not executed in the bounded classical lane"),
    } for number, (title, _) in legacy["CRITERIA"].items()}


def git_metadata(root):
    """Record local provenance without fetching or modifying the checkout."""
    def query(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    try:
        sha = query("rev-parse", "HEAD")
        status = query("status", "--porcelain=v1", "--untracked-files=normal")
        return {"sha": sha, "dirty": bool(status), "status": status, "error": None}
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return {"sha": None, "dirty": None, "status": None, "error": str(exc)}


def validation_environment(receipt_path):
    # Every validation lane is serial, including imported BLAS and child jobs.
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTEST_ADDOPTS": "",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "TEMSIM_ACCEPTANCE_RECEIPT": str(receipt_path),
           "OMP_MAX_ACTIVE_LEVELS": "1", "OMP_NESTED": "FALSE"}
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run_scope(root, out, scope, timeout, *, criteria=None, selected=None, allow_gpu_skips=False):
    """Execute one declared scope and retain evidence even on child failure.

    Explicit criteria/selection are used by isolated command-level policy tests;
    the CLI only admits registered scopes and their complete file allowlists.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_only = scope == "full-report"
    executed_scope = "classical" if report_only else scope
    declared = ACCEPTANCE_SCOPES[executed_scope]["criteria"] if criteria is None else criteria
    selected = scope_test_files(executed_scope) if selected is None else tuple(selected)
    run_id = uuid4().hex
    evidence_path = out / f"pytest-{run_id}.json"
    log_path = out / f"pytest-{run_id}.log"
    junit_path = out / f"pytest-{run_id}.xml"
    # Invalidate the previous summary before even reading source/metadata:
    # preparation failure or interruption is not evidence for an old PASS.
    pending = {"software_scope_status": "NOT_RUN", "full_simulator_qualification": "UNQUALIFIED",
               "exit_code": 1, "run_id": run_id, "requested_scope": scope,
               "errors": ["Execution has not produced a completed acceptance report"]}
    (out / "report.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")
    before = source_hashes(root)
    env = validation_environment(evidence_path)
    command = [] if report_only else [
        sys.executable, "-m", "pytest", *selected, "-p", "temsim.acceptance_pytest",
        "-p", "pytestqt.plugin", "-o", "addopts=", "--tb=short", f"--junitxml={junit_path}"]
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_scope": scope, "run_id": run_id, "command": command,
        "timeout_seconds": timeout, "selected_tests": list(selected),
        "source_sha256": before, "git": git_metadata(root),
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependency_versions": dict(sorted((d.metadata["Name"], d.version)
                                            for d in metadata.distributions() if d.metadata["Name"])),
        "validation_environment": {k: v for k, v in env.items() if k in {
            "QT_QPA_PLATFORM", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS",
            "TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR", "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS", "OMP_NESTED"}},
    }
    # Write selection/provenance before launching. Abrupt runner death cannot
    # turn a previous successful report into evidence for this new attempt.
    (out / f"selection-{run_id}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    receipt, code, execution_errors = {}, None, []
    if not report_only:
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = run_bounded(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=timeout, env=env)
                code = completed.returncode
            except (OSError, subprocess.SubprocessError) as exc:
                execution_errors.append(f"Validation process failed: {type(exc).__name__}: {exc}")
                log.write(execution_errors[-1] + "\n")
        if evidence_path.exists():
            try:
                loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("Receipt must be a JSON object")
                receipt = loaded
            except (ValueError, OSError) as exc:
                execution_errors.append(f"Invalid pytest receipt: {exc}")
    report = software_report(receipt, scope=executed_scope, pytest_exit_code=code,
                             source_unchanged=before == source_hashes(root),
                             selected=selected, criteria=declared,
                             allow_gpu_skips=allow_gpu_skips)
    if executed_scope == "classical":
        report["criteria"] = merge_criteria(report["criteria"], legacy_exclusions(root))
    report.update(provenance, completed_utc=datetime.now(timezone.utc).isoformat(),
                  git_after=git_metadata(root),
                  artifacts={name: {"path": str(path), "exists": path.is_file()} for name, path in
                             (("receipt", evidence_path), ("junit", junit_path), ("log", log_path))})
    if execution_errors:
        report["errors"].extend(execution_errors)
        report["software_scope_status"], report["exit_code"] = "INCOMPLETE", 1
    if report_only:
        report["software_scope_status"] = "NOT_RUN"
        report["errors"] = ["Report only: paused full-image calculations were not launched"]
    for path in (out / f"report-{run_id}.json", out / "report.json"):
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=(*ACCEPTANCE_SCOPES, "full-report"), default="classical")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-gpu-skips", action="store_true",
                        help="CPU CI only: report reviewed real-GPU skips as NOT_RUN; other skips and failures still fail")
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "outputs" / "agent-validation" / args.scope
    report = run_scope(root, output, args.scope, args.timeout_seconds,
                       allow_gpu_skips=args.allow_gpu_skips)
    print(json.dumps({key: report[key] for key in (
        "software_scope_status", "full_simulator_qualification", "exit_code", "run_id")}), flush=True)
    if report["gpu_hardware"]["not_run"]:
        print(f"GPU hardware: NOT_RUN ({len(report['gpu_hardware']['not_run'])} skipped cases)", flush=True)
    if report["exit_code"]:
        for node, phases in report["cases"].items():
            if "failed" in phases.values() or ("skipped" in phases.values()
                    and node not in report["allowed_gpu_skips"]):
                print(f"Nonpassing test: {node} | {phases}", flush=True)
        for error in report["errors"]:
            print(f"Acceptance error: {error}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
