"""Trace all AT-01..33 to executed acceptance tests and measured evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

# One test can cover several criteria; a file/prefix selector includes all of
# its parameterizations. Missing, skipped or failed evidence is never PASS.
CRITERIA={
1:("Physical aperture identity executes once",["test_tem_flux_contract.test_at01_at03", "test_tem_flux_contract.test_duplicate"]),
2:("Offset objective aperture",["test_tem_flux_contract.test_at02"]),
3:("Transfer-dependent aperture acceptance",["test_tem_flux_contract.test_at01_at03"]),
4:("Closed, removed, disabled, uninstalled",["test_tem_flux_contract.test_at04"]),
5:("Exclusive physical/equivalent strategies",["test_tem_flux_contract.test_at05"]),
6:("Known absolute transmitted probability",["test_tem_flux_contract.test_at06", "test_gpu_capture_contract.test_stem_known_bandwidth"]),
7:("Weight/current/dose scaling",["test_tem_flux_contract.test_at07"]),
8:("Incoherent configuration intensity sum",["test_tem_flux_contract.test_at08"]),
9:("Lossless norm drift guard",["test_tem_flux_contract.test_at09"]),
10:("Exclusive physical sinks; overlapping virtual observers",["test_record_plane.test_sequential", "test_record_plane.test_overlapping_virtual"]),
11:("Distinct angular current quantiles",["test_illumination_modes.test_at11"]),
12:("Noncircular offset complex pupil",["test_illumination_modes.test_sampled_intensity", "test_source_admission.test_t201_api_refuses_legacy_pupil", "test_tip_source_contract.test_even_a_matching_exit_binding"]),
13:("Independent source mode sum",["test_illumination_modes.test_at13", "test_effective_gun_source.test_correlated_energy_samples"]),
14:("Source/energy quadrature refinement",["test_illumination_modes.test_source_priors_and_independent_dimensions", "test_effective_gun_source.test_tiny_mode_tail_tolerances"]),
15:("Gaussian width, centre and curvature under LCT",["test_illumination_modes.test_at15"]),
16:("A5 angular rotation and scale",["test_aberration_wp04.test_a5_rotation", "test_aberration_wp04.test_phase_matches_abtem"]),
17:("Mixed coefficients and independent holdout",["test_aberration_wp04.test_mixed_coefficients", "test_aberration_wp04.test_missing_term"]),
18:("Pupil, step and field-grid convergence with observables",["test_aberration_wp04.test_aperture_step", "test_aberration_wp04.test_local_response_reaches"]),
19:("Cc fixed-hardware energy study",["test_aberration_wp04.test_field_fit_reports", "test_illumination_modes.test_energy_override"]),
20:("Known N/I/NI conversion; unknown turns unavailable",["test_control_geometry_contract.test_known_turns", "test_control_geometry_contract.test_unknown_turns", "test_simulation_modes.test_mode_switch_preserves"]),
21:("Joint saturation current vector, polarity and cache",["test_nonlinear_magnetostatics.test_joint_field", "test_nonlinear_magnetostatics.test_reference_saturation", "test_nonlinear_magnetostatics.test_shared_recipe"]),
22:("Queryable geometry effects and unsupported CAD admission",["test_control_geometry_contract.test_geometry_", "test_execution_migration_contract.test_cad_display"]),
23:("Finite solenoid and simplified pole references",["test_scientific_wp06.test_finite_solenoid", "test_scientific_wp06.test_simplified_pole"]),
24:("Multislice grid/FOV/slices/bandwidth/phonon convergence",["test_scientific_wp06.test_si_multislice"]),
25:("Actual CPU/GPU and independent propagation parity",["test_gpu_capture_contract.test_actual_gpu", "test_scientific_wp06.test_external_abtem"]),
26:("Resident GPU full diffraction capture and reintegration",["test_gpu_capture_contract.test_actual_gpu"]),
27:("Bounded host output, integrity, cancellation, resume",["test_gpu_capture_contract.test_capture_host", "test_gpu_capture_contract.test_failed_checkpoint", "test_gpu_capture_contract.test_resume_detects", "test_gpu_capture_contract.test_cancel_resume"]),
28:("Explicit backend policy failures",["test_gpu_capture_contract.test_policy"]),
29:("One actual execution description and separate verification",["test_execution_migration_contract.test_execution_export", "test_tem_flux_contract.test_export"]),
30:("Scoped identities and display-only reuse",["test_execution_migration_contract.test_cad_display", "test_calculation_manifest_artifacts.test_geometry_fingerprint", "test_fourdstem.test_resume_rejects", "test_fourdstem.test_raw_cube_can"]),
31:("Locked isolated CPU install/import/GUI/small case",[]),
32:("Explicit migration without automatic hardware retuning",["test_execution_migration_contract.test_old_profile", "test_simulation_modes.test_mode_switch_preserves", "test_profile_optional_values.test_legacy_profile"]),
33:("Fixed control to field to crystal to aperture to camera",["test_scientific_wp06.test_at33"]),
}


# Current tests provide partial mathematics/admission evidence only. The
# withdrawn specimen-pupil producer cannot qualify a gun-to-image chain.
SOURCE_MIGRATION_BLOCKERS = {
    12: "Independent exit/specimen sources are prohibited. Noncircular illumination requires coherent transport from the tip through the entire gun and image chain.",
    13: "Mode addition and historical exit-source mathematics are covered separately; the required physical tip-origin multi-mode image chain is not implemented.",
    14: "Tip-source modes and a development quadratic accelerating-gun operator have scoped tests; energy/source refinement through the complete production image chain is not implemented.",
}


def evaluate_criteria(cases, installation):
    """Retain failures and missing evidence; partial tests never close a blocker."""
    criteria = {}
    for number, (title, prefixes) in CRITERIA.items():
        matches = [row for row in cases if any(row["test"].startswith(prefix) for prefix in prefixes)]
        missing = [prefix for prefix in prefixes if not any(row["test"].startswith(prefix) for row in cases)]
        if number == 31:
            status = installation["status"]
        elif any(row["status"] == "FAIL" for row in matches):
            status = "FAIL"
        elif missing or not matches or any(row["status"] != "PASS" for row in matches):
            status = "NOT_RUN"
        elif number in SOURCE_MIGRATION_BLOCKERS:
            status = "BLOCKED"
        else:
            status = "PASS"
        row = {"description": title, "status": status, "tests": [item["test"] for item in matches],
               "missing_test_prefixes": missing}
        if number in SOURCE_MIGRATION_BLOCKERS:
            row["blocked_reason"] = SOURCE_MIGRATION_BLOCKERS[number]
            row["evidence_scope"] = "Partial numerical/admission tests; production image acceptance remains open"
        if number == 31 and status == "PASS" and installation.get("wave_images", {}).get("status") != "PASS":
            row["status"] = "BLOCKED"
            row["blocked_reason"] = "Package checks pass, but the installed production wave-image case has not been validated"
            row["evidence_scope"] = installation.get("scope", "Installation receipt lacks explicit wave-image evidence")
        criteria[f"AT-{number:02}"] = row
    return criteria


def acceptance_exit_code(criteria, *, source_unchanged, pytest_exit_code, allow_no_gpu=False):
    acceptable = all(row["status"] == "PASS" or (
        allow_no_gpu and key in {"AT-25", "AT-26"} and row["status"] == "NOT_RUN"
    ) for key, row in criteria.items())
    return 0 if source_unchanged and pytest_exit_code == 0 and acceptable else 1


def source_hashes(root):
    paths=sorted((root/"src/temsim").rglob("*.py"))+sorted((root/"configs").rglob("*.toml"))+sorted((root/"configs/reference_samples").glob("*.cif"))
    paths+=sorted((root/"tests").glob("test_*.py"))+[Path(__file__).resolve(), root/"scripts/installation_smoke.py"]
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/spec-acceptance"))
    parser.add_argument("--installed-python",type=Path)
    parser.add_argument("--allow-no-gpu",action="store_true",help="CPU CI may leave actual GPU criteria NOT_RUN, never PASS")
    parser.add_argument("--full-suite",action="store_true")
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]; out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    before=source_hashes(root); started=datetime.now(timezone.utc).isoformat()
    files=sorted({"tests/"+prefix.split(".")[0]+".py" for _,prefixes in CRITERIA.values() for prefix in prefixes})
    files+= ["tests/test_development_acceptance.py","tests/test_model_inspector.py","tests/test_parameter_impact.py","tests/test_fourdstem_user_wiring.py","tests/test_fourdstem_cache_products.py","tests/test_stem_cuda_pipeline.py","tests/test_wave_fft.py","tests/test_multislice.py"]
    command=[sys.executable,"-m","pytest",*( ["tests"] if args.full_suite else files),"--tb=short","-o","junit_family=legacy","--junitxml="+str(out/"pytest.xml")]
    print("Running development acceptance:",out,flush=True)
    with (out/"pytest.log").open("w",encoding="utf-8") as log:
        run=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"QT_QPA_PLATFORM":"offscreen"})
    cases=[]; evidence={}
    if (out/"pytest.xml").exists():
        for row in ET.parse(out/"pytest.xml").iter("testcase"):
            name=row.attrib.get("classname","").removeprefix("tests.")+"."+row.attrib["name"]
            status="FAIL" if row.find("failure") is not None or row.find("error") is not None else "NOT_RUN" if row.find("skipped") is not None else "PASS"
            cases.append({"test":name,"status":status,"seconds":float(row.attrib.get("time",0))})
            for prop in row.findall("properties/property"):
                if prop.attrib["name"].startswith(("wp04_","wp06_","wp07_")):
                    evidence[prop.attrib["name"]]=json.loads(prop.attrib["value"])
    install={"status":"NOT_RUN","reason":"isolated installed interpreter was not supplied"}
    if args.installed_python:
        install_command=[str(args.installed_python.resolve()),"-I",str(root/"scripts/installation_smoke.py"),"--output",str(out/"installation.json")]
        with (out/"installation.log").open("w",encoding="utf-8") as log:
            completed=subprocess.run(install_command,cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"QT_QPA_PLATFORM":"offscreen"})
        install=json.loads((out/"installation.json").read_text()) if completed.returncode==0 else {"status":"FAIL","exit_code":completed.returncode}
        from temsim.calculation_manifest import solver_source_identity
        if install.get("status")=="PASS" and install["solver_source_sha256"]!=solver_source_identity():
            install["status"]="INCONCLUSIVE"; install["reason"]="installed wheel differs from current source"
    criteria=evaluate_criteria(cases, install)
    unchanged=before==source_hashes(root)
    all_pass=all(row["status"]=="PASS" for row in criteria.values())
    report={"schema":"development-spec-acceptance-v2","started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS" if all_pass and unchanged and run.returncode==0 else "INCOMPLETE", "criteria":criteria,
        "counts":{s:sum(row["status"]==s for row in cases) for s in ("PASS","FAIL","NOT_RUN")},"cases":cases,"measurements":evidence,
        "installation":install,"command":command,"pytest_exit_code":run.returncode,"source_unchanged_during_tests":unchanged,
        "source_sha256":before,"python":sys.version,"platform":platform.platform(),
        "versions":{n:metadata.version(n) for n in ("numpy","scipy","abtem","PySide6","pytest")},
        "scope":"Partial numerical and engineering evidence during the gun-source migration. BLOCKED criteria are not release acceptance even when all their listed tests pass. Installation smoke covers particles and a separate specimen kernel, not production TEM/STEM images. External experimental calibration remains NOT_RUN. A5 is a partial higher-order basis; 3D CAD requires a supported field approximation.",
        "external_experimental_calibration":"NOT_RUN"}
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"counts":report["counts"],"criteria":{k:v["status"] for k,v in criteria.items()}},ensure_ascii=False),flush=True)
    return acceptance_exit_code(criteria, source_unchanged=unchanged,
        pytest_exit_code=run.returncode, allow_no_gpu=args.allow_no_gpu)


if __name__=="__main__": raise SystemExit(main())
