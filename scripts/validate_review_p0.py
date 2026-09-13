"""Run scoped review regressions and preserve raw, input-bound evidence.

This does not certify full-chain physics, GPU performance or microscope access.
Every run has its own directory; failures are retained, never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", default=["tests/test_review_p0_regressions.py"])
    parser.add_argument("--output", type=Path, default=Path("docs/development/evidence"))
    parser.add_argument("--label", default="review-p0", help="Short name for this scoped evidence run")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.label):
        parser.error("--label must contain only lowercase letters, digits and hyphens")
    root = Path(__file__).resolve().parents[1]
    started = datetime.now(timezone.utc)
    output = (root/args.output/(started.strftime("%Y%m%dT%H%M%SZ")+"-"+args.label+"-"+uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True)
        return result.stdout
    inventory = sorted({*root.glob("src/**/*.py"), *root.glob("tests/**/*.py"),
                        *root.glob("scripts/**/*.py"), *root.glob("configs/**/*.toml"),
                        *root.glob("configs/**/*.json"),
                        *root.glob("data/**/*.toml"), *root.glob("data/**/*.json"),
                        root/"pyproject.toml", Path(__file__).resolve()})
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
              for path in inventory if path.is_file()}
    packages = {}
    for name in ("numpy", "scipy", "numba", "cupy-cuda12x", "abtem", "xraylib", "PySide6", "pytest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", "-o", "junit_family=xunit1", "-p", "pytestqt.plugin",
               *args.tests, "-q", "--tb=short", f"--junitxml={output/'pytest.xml'}"]
    report = {"schema": "review-validation-evidence-v1", "started_utc": started.isoformat(),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "working_tree_status_before": git("status", "--short").decode(),
        "working_tree_diff_sha256": sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "input_sha256": hashes, "command": command, "python": sys.version,
        "platform": platform.platform(), "packages": packages, "status": "RUNNING",
        "scope": "Offline engineering and isolated numerical/identity regressions only",
        "full_chain_scientific_acceptance": "NOT_RUN", "gpu_validation": "NOT_RUN",
        "hardware_acquisition": "NOT_RUN"}
    def save():
        temporary = output/"report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(output/"report.json")
    save()
    print(f"Evidence directory: {output}", flush=True)
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    try:
        with (output/"pytest.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report["exit_code"] = result.returncode
        report["status"] = "PASSED" if result.returncode == 0 else "FAILED"
        xml = output/"pytest.xml"
        if xml.exists():
            cases = list(ET.parse(xml).getroot().iter("testcase"))
            report["counts"] = {"cases": len(cases), "failed": sum(c.find("failure") is not None for c in cases),
                "errors": sum(c.find("error") is not None for c in cases),
                "skipped": sum(c.find("skipped") is not None for c in cases)}
            report["counts"]["passed"] = len(cases)-sum(report["counts"][key] for key in ("failed", "errors", "skipped"))
        return_code = result.returncode
    except BaseException as error:
        report.update(status="INTERRUPTED_OR_ERROR", error=f"{type(error).__name__}: {error}")
        return_code = 1
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    changed_inputs = [name for name, digest in hashes.items()
                      if not (root/name).is_file() or sha256((root/name).read_bytes()).hexdigest() != digest]
    report["changed_inputs_during_run"] = changed_inputs
    if changed_inputs:
        report["status"] = "INPUTS_CHANGED_UNQUALIFIED"
        return_code = 1
    report["artifacts"] = {path.name: sha256(path.read_bytes()).hexdigest()
                           for path in (output/"pytest.log", output/"pytest.xml") if path.exists()}
    save()
    print(json.dumps({key: report.get(key) for key in ("status", "counts", "exit_code")}), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
