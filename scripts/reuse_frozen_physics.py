"""Execute another numerical comparison against exactly one frozen solver.

This is a development runner, not a source-cache importer. Reusing the same
verified directory also preserves absolute configuration identities between
independent numerical runs. Every run has a separate, non-overwriting receipt.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.freeze_physics_run import inventory
from scripts.physics_driver_log import run_logged_driver


def verified_inputs(snapshot):
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot/"execution.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "frozen-physics-execution-v1"
            or Path(manifest["snapshot_root"]).resolve() != snapshot):
        raise ValueError("A matching frozen physics execution record is required")
    expected = manifest["input_sha256"]
    if "src/temsim/calculation_manifest.py" not in expected or inventory(snapshot) != expected:
        raise ValueError("Frozen solver or physical inputs changed; comparison was not started")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--driver", choices=("inspect_surface_column", "trace_tip_wave", "audit_current_gun_integrators"), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    snapshot, receipt = args.snapshot.resolve(), args.receipt.resolve()
    manifest = verified_inputs(snapshot)
    # A driver output or its receipt must never mutate frozen dependencies.
    if receipt == snapshot or snapshot in receipt.parents:
        parser.error("The new execution receipt must be outside the frozen directory")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if "--output" not in arguments:
        parser.error("An explicit, separate driver output directory is required")
    output = Path(arguments[arguments.index("--output")+1]).resolve()
    if output.exists() or output == snapshot or snapshot in output.parents:
        parser.error("Use a new driver output directory outside the frozen solver")
    command = [sys.executable, "-m", "scripts."+args.driver, *arguments]
    environment = dict(os.environ, PYTHONPATH=str(snapshot/"src"), TEMSIM_PROJECT_ROOT=str(snapshot),
                       PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    report = {"schema": "reused-frozen-physics-execution-v1", "snapshot_root": str(snapshot),
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "input_sha256": manifest, "status": "RUNNING"}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    try:
        log = receipt.with_suffix(".driver.log")
        report["driver_log"] = str(log)
        code = run_logged_driver(command, cwd=snapshot, env=environment, log_path=log)
        report.update(exit_code=code,
                      status="DRIVER_COMPLETED" if code == 0 else "DRIVER_FAILED")
    except BaseException as failure:
        report.update(status="INTERRUPTED_OR_ERROR", error=f"{type(failure).__name__}: {failure}")
        raise
    finally:
        report["inputs_unchanged"] = inventory(snapshot) == manifest
        if not report["inputs_unchanged"]:
            report["status"] = "INPUTS_CHANGED_UNQUALIFIED"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["output_sha256"] = {p.name: sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file()} if output.is_dir() else {}
        receipt.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return int(report["status"] != "DRIVER_COMPLETED")


if __name__ == "__main__":
    raise SystemExit(main())
