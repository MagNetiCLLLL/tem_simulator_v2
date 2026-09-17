"""Bounded cold-process repeats plus warm/rapid-edit Qt ownership scenarios.

This is a software stress receipt, not a numerical or native-GUI benchmark.
Worker-entry assertions remain five seconds; process timeouts are only safety
limits for the whole test group. No source calculations are enabled here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from uuid import uuid4
from temsim.validation_process import run_bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--process-timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("outputs/job-lifecycle-stress"))
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10 or not 10 <= args.process_timeout <= 600:
        parser.error("Use 1..10 repetitions and a 10..600 second process budget")
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve() / uuid4().hex
    out.mkdir(parents=True)
    rows = []
    for index in range(args.repetitions):
        command = [sys.executable, "-m", "pytest", "tests/test_job_lifecycle.py",
            "tests/test_background_preview_gui.py::test_live_edits_during_preparation_keep_one_frame_and_latest_pending",
            "-o", "addopts=", "--tb=short", f"--junitxml={out / f'run-{index}.xml'}"]
        started = monotonic()
        with (out / f"run-{index}.log").open("w", encoding="utf-8") as log:
            child = run_bounded(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTEST_ADDOPTS": ""},
                timeout=args.process_timeout)
            code = child.returncode
        rows.append(dict(repetition=index, exit_code=code, seconds=monotonic()-started, command=command))
        print(json.dumps(rows[-1]), flush=True)
        if code != 0:
            break
    passed = len(rows) == args.repetitions and all(row["exit_code"] == 0 for row in rows)
    report = dict(scope="offscreen software stress; no physical qualification", passed=passed,
                  worker_entry_timeout_seconds=5, intended_repetitions=args.repetitions, runs=rows,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Receipt: {out / 'report.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
