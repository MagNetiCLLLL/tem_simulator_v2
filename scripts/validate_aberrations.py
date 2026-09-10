"""Run WP-04 phase, field-fit and numerical-study acceptance with provenance."""

from datetime import datetime, timezone
from importlib import metadata
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

CORE = ["tests/test_aberration_wp04.py", "tests/test_aberration_model.py",
        "tests/test_six_stage_physics.py", "tests/test_model_inspector.py",
        "tests/test_simulation_modes.py", "tests/test_stem_wave_control_ownership.py"]
EXTENDED = ["tests/test_illumination_modes.py", "tests/test_tem_flux_contract.py",
            "tests/test_wave_imaging.py", "tests/test_interactive_calculation.py",
            "tests/test_calculation_cache_reuse.py", "tests/test_profile_optional_values.py",
            "tests/test_vector_field_transport.py", "tests/test_lens_field_provider.py",
            "tests/test_fourdstem_cache_products.py", "tests/test_tem_result_sources.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/aberrations"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    files = sorted((root / "src/temsim").rglob("*.py")) + [root / file for file in CORE + EXTENDED] + [Path(__file__).resolve()]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    command = [sys.executable, "-m", "pytest", *CORE, *(EXTENDED if args.extended else []),
               "--tb=short", "-o", "junit_family=legacy", "--junitxml=" + str(out / "pytest.xml")]
    print("Running WP-04 validation:", out, flush=True)
    with (out / "pytest.log").open("w", encoding="utf-8") as log:
        run = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
    cases, evidence = [], {}
    if (out / "pytest.xml").exists():
        for case in ET.parse(out / "pytest.xml").iter("testcase"):
            status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "NOT_RUN" if case.find("skipped") is not None else "PASS"
            cases.append({"test": case.attrib.get("classname", "") + "." + case.attrib["name"], "status": status})
            for prop in case.findall("properties/property"):
                if prop.attrib["name"].startswith("wp04_"):
                    evidence[prop.attrib["name"]] = json.loads(prop.attrib["value"])
    unchanged = all(hashlib.sha256((root / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    passed = run.returncode == 0 and len(evidence) == 6 and unchanged
    report = {"schema": "wp04-aberration-validation-v1", "started_utc": started,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "command": command, "exit_code": run.returncode, "status": "PASS" if passed else "FAIL",
              "counts": {s: sum(c["status"] == s for c in cases) for s in ("PASS", "FAIL", "NOT_RUN")},
              "evidence": evidence, "cases": cases, "python": sys.version, "platform": platform.platform(),
              "versions": {n: metadata.version(n) for n in ("numpy", "scipy", "PySide6", "abtem", "pytest")},
              "source_sha256_at_start": hashes, "source_unchanged_during_tests": unchanged,
              "external_field_or_material_validation": "NOT_RUN", "GPU_phase_parity": "NOT_RUN",
              "limitations": ["A5/C56 is not a complete fourth/fifth-order basis",
                              "Analytic map and paraxial hexapole fixtures do not calibrate a physical corrector",
                              "Convergence bounds are local and include explicit per-coefficient absolute floors",
                              "Local response reports sensitivity and joint residuals; no automatic tuning or current calibration"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}), flush=True)
    if not passed:
        print((out / "pytest.log").read_text(encoding="utf-8")[-10000:])
    return run.returncode or (0 if passed else 1)


if __name__ == "__main__":
    raise SystemExit(main())
