"""Run WP-03 acceptance and report source/energy quadrature convergence."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib import metadata

CORE = ["tests/test_illumination_modes.py", "tests/test_tem_flux_contract.py", "tests/test_stem_wave_control_ownership.py"]
EXTENDED = ["tests/test_wave_imaging.py", "tests/test_interactive_calculation.py",
    "tests/test_stem_finite_absorption.py", "tests/test_stem_recording_deflection.py",
    "tests/test_aberration_model.py", "tests/test_fourdstem.py", "tests/test_fourdstem_cache_products.py",
    "tests/test_tem_result_sources.py", "tests/test_calculation_cache_reuse.py",
    "tests/test_sample_profile_v2.py", "tests/test_reference_sample_profile.py", "tests/test_profile_optional_values.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/illumination"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    source_hashes = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root / "src/temsim").rglob("*.py"))}
    command = [sys.executable, "-m", "pytest", *CORE, *(EXTENDED if args.extended else []),
               "--tb=short", "--junitxml=" + str(out / "pytest.xml")]
    print("Running WP-03 validation:", out, flush=True)
    with (out / "pytest.log").open("w", encoding="utf-8") as log:
        run = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
    cases, convergence = [], {}
    if (out / "pytest.xml").exists():
        for case in ET.parse(out / "pytest.xml").iter("testcase"):
            status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "NOT_RUN" if case.find("skipped") is not None else "PASS"
            cases.append({"test": case.attrib.get("classname", "") + "." + case.attrib["name"], "status": status})
            for prop in case.findall("properties/property"):
                if prop.attrib["name"].startswith("wp03_convergence_"):
                    convergence[prop.attrib["name"]] = json.loads(prop.attrib["value"])
    report = {"schema": "wp03-illumination-validation-v1", "started_utc": started,
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "command": command, "exit_code": run.returncode,
              "status": "PASS" if run.returncode == 0 and len(convergence) == 2 else "FAIL",
              "counts": {s: sum(c["status"] == s for c in cases) for s in ("PASS", "FAIL", "NOT_RUN")},
              "source_energy_convergence": convergence, "cases": cases,
              "python": sys.version, "platform": platform.platform(),
              "versions": {n: metadata.version(n) for n in ("numpy", "scipy", "PySide6", "abtem", "pytest")},
              "source_sha256_at_start": source_hashes,
              "source_unchanged_during_tests": all(hashlib.sha256((root / p).read_bytes()).hexdigest() == sha for p, sha in source_hashes.items()),
              "external_material_validation": "NOT_RUN", "GPU_mode_parity": "NOT_RUN",
              "limitations": ["Declared specimen-entrance modes, not full gun coherence or upstream chromatic source transport",
                              "Multi-mode 4D-STEM, ray-based Rutherford tail coupling and bulk absorption mixtures unsupported",
                              "Fixture-local quadrature convergence does not certify wave-grid or frozen-phonon convergence"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}), flush=True)
    if run.returncode:
        print((out / "pytest.log").read_text(encoding="utf-8")[-10000:])
    return run.returncode or (0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
