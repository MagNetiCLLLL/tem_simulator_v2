"""Run the first development-spec batch and save an auditable CPU report.

Usage: python scripts/validate_tem_p0.py --extended --output outputs/tem-p0
No instrument settings, CIFs or application profiles are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


CORE_TESTS = [
    "tests/test_tem_flux_contract.py", "tests/test_record_plane.py",
    "tests/test_record_plane_detector_masks.py", "tests/test_tem_result_sources.py",
]
EXTENDED_TESTS = [
    "tests/test_wave_imaging.py", "tests/test_six_stage_physics.py",
    "tests/test_interactive_calculation.py", "tests/test_calculation_cache_reuse.py",
    "tests/test_calculation_manifest_artifacts.py", "tests/test_wave_specimen_cache_version.py",
    "tests/test_stem_dose_cache.py", "tests/test_eels_forward.py", "tests/test_wave_fft.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/tem-p0"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    xml_path = output / "pytest.xml"
    command = [sys.executable, "-m", "pytest", *CORE_TESTS,
               *(EXTENDED_TESTS if args.extended else []), "--tb=short", f"--junitxml={xml_path}"]
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    started = datetime.now(timezone.utc).isoformat()
    print("Running TEM P0 acceptance and related regressions; output:", output, flush=True)
    with (output / "pytest.log").open("w", encoding="utf-8") as log:
        run = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    cases = []
    if xml_path.exists():
        for case in ET.parse(xml_path).iter("testcase"):
            status = ("FAIL" if case.find("failure") is not None or case.find("error") is not None else
                      "NOT_RUN" if case.find("skipped") is not None else "PASS")
            cases.append({"test": case.attrib.get("classname", "") + "." + case.attrib["name"], "status": status})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    versions = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")}
    report = {
        "schema": "tem-p0-validation-v1", "started_utc": started,
        "git_base": git_head, "working_tree_modified": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "command": command, "return_code": run.returncode,
        "status": "PASS" if run.returncode == 0 and cases else "FAIL",
        "counts": {s: sum(c["status"] == s for c in cases) for s in ("PASS", "FAIL", "NOT_RUN")},
        "acceptance": {"fixture": "manufactured two-order and phase-grating production TEM",
                       "backend": "NumPy CPU / complex128", "absolute_probability_tolerance": 1e-10},
        "norm_guard_tolerances": {"complex128_rtol": 1e-10, "complex64_rtol": 2e-5, "atol": 1e-14},
        "independent_material_validation": "NOT_RUN", "cuda_physical_parity": "NOT_RUN",
        "arbitrary_wavefront_and_A5": "NOT_RUN (later work packages)",
        "python": sys.version, "platform": platform.platform(), "installed_versions": versions,
        "cases": cases,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}), flush=True)
    if run.returncode:
        print((output / "pytest.log").read_text(encoding="utf-8")[-16000:])
    return run.returncode or (0 if cases else 1)


if __name__ == "__main__":
    raise SystemExit(main())
