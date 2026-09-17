"""Bounded classical software acceptance, separate from legacy full-image gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from uuid import uuid4

from temsim.acceptance import CLASSICAL_TESTS, classical_report, merge_criteria
from temsim.validation_process import run_bounded


def source_hashes(root):
    paths = set()
    for directory in ("src/temsim", "tests", "scripts", "configs", ".github/workflows"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.suffix.lower() in {".py", ".toml", ".json", ".cif", ".mcif", ".yml", ".yaml"})
    paths.update(root / name for name in ("pyproject.toml", "AGENTS.md", "PROJECT_FUNCTION_SPEC.md"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def legacy_exclusions(root):
    legacy = runpy.run_path(str(root / "scripts/validate_development_spec.py"))
    blockers = legacy["SOURCE_MIGRATION_BLOCKERS"]
    return {f"legacy-development/AT-{number:02}": {
        "description": title, "status": "BLOCKED" if number in blockers else "NOT_RUN",
        "reason": blockers.get(number, "Not executed in the bounded classical lane"),
    } for number, (title, _) in legacy["CRITERIA"].items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("classical", "full-report"), default="classical")
    parser.add_argument("--output", type=Path, default=Path("outputs/classical-acceptance"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Unique per-run evidence prevents a stale receipt from an earlier process
    # being admitted after timeout/crash. Complete logs stay in this directory.
    run_id = uuid4().hex
    evidence_path = out / f"pytest-{run_id}.json"
    before = source_hashes(root)
    started = datetime.now(timezone.utc).isoformat()
    command, receipt, code = [], {}, None
    if args.scope == "classical":
        command = [sys.executable, "-m", "pytest", *CLASSICAL_TESTS, "-p", "temsim.acceptance_pytest",
                   "-o", "addopts=", "--tb=short", f"--junitxml={out / ('pytest-' + run_id + '.xml')}"]
        with (out / f"pytest-{run_id}.log").open("w", encoding="utf-8") as log:
            completed = run_bounded(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds, env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
                    "PYTEST_ADDOPTS": "", "TEMSIM_ACCEPTANCE_RECEIPT": str(evidence_path)})
            code = completed.returncode
        if evidence_path.exists():
            try:
                receipt = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                receipt = {}  # Corrupt evidence is missing evidence, never PASS.
    report = classical_report(receipt, pytest_exit_code=code, source_unchanged=before == source_hashes(root))
    report["criteria"] = merge_criteria(report["criteria"], legacy_exclusions(root))
    report.update(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  command=command, requested_scope=args.scope, run_id=run_id,
                  source_sha256=before, python=sys.version, platform=platform.platform())
    if args.scope == "full-report":
        report["software_scope_status"] = "NOT_RUN"
        report["errors"] = ["Report only: paused full-image calculations were not launched"]
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("software_scope_status", "full_simulator_qualification", "exit_code", "run_id")}), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
