"""Run a development physics driver against a verified copy of its inputs.

Long reference calculations must not consume source edits made during the run.
The copied code/configuration is retained with a manifest; it is not a new
electron source, an admission certificate, or a replacement for external-input
checks performed by the driver. No existing outputs are overwritten.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from scripts.physics_driver_log import run_logged_driver


def inventory(root):
    files = []
    for name in ("src", "scripts", "configs", "data"):
        directory = root/name
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts
                         and p.suffix not in {".pyc", ".pyo"})
    files.extend(p for p in (root/"pyproject.toml", root/"AGENTS.md") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def freeze(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents and any(
            (root/name) == destination or (root/name) in destination.parents
            for name in ("src", "scripts", "configs", "data")):
        raise ValueError("The frozen destination must be outside source/input trees")
    files = inventory(root)
    if not files or "src/temsim/calculation_manifest.py" not in files:
        raise ValueError("A complete TEM simulator checkout is required")
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    # Detect a concurrent edit during copying instead of publishing a hybrid.
    if inventory(root) != files or inventory(destination) != files:
        raise RuntimeError("Source/input files changed while taking the frozen copy")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--driver", choices=("inspect_surface_column", "trace_tip_wave", "audit_current_gun_integrators"), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    snapshot = args.snapshot.resolve()
    manifest = freeze(root, snapshot)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    environment = dict(os.environ, PYTHONPATH=str(snapshot/"src"), TEMSIM_PROJECT_ROOT=str(snapshot),
                       PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "scripts."+args.driver, *arguments]
    report = {"schema": "frozen-physics-execution-v1", "id": uuid.uuid4().hex,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "original_root": str(root), "snapshot_root": str(snapshot),
              "input_sha256": manifest, "command": command, "status": "RUNNING"}
    record = snapshot/"execution.json"
    record.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    try:
        log = snapshot/"driver.log"
        report["driver_log"] = str(log)
        code = run_logged_driver(command, cwd=snapshot, env=environment, log_path=log)
        report.update(exit_code=code, status="DRIVER_COMPLETED" if code == 0 else "DRIVER_FAILED")
    except BaseException as error:
        report.update(status="INTERRUPTED_OR_ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["inputs_unchanged"] = inventory(snapshot) == manifest
        if not report["inputs_unchanged"]:
            report["status"] = "INPUTS_CHANGED_UNQUALIFIED"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return 0 if report["status"] == "DRIVER_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
