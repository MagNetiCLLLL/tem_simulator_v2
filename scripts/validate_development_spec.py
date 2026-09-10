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
12:("Noncircular offset complex pupil",["test_illumination_modes.test_at12", "test_illumination_modes.test_sampled_intensity"]),
13:("Independent source mode sum",["test_illumination_modes.test_at13", "test_illumination_modes.test_stem_source_energy"]),
14:("Source/energy quadrature refinement",["test_illumination_modes.test_at14"]),
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


def source_hashes(root):
    paths=sorted((root/"src/temsim").rglob("*.py"))+sorted((root/"configs").rglob("*.toml"))+sorted((root/"configs/reference_samples").glob("*.cif"))
    paths+=sorted((root/"tests").glob("test_*.py"))+[Path(__file__).resolve()]
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
    files+= ["tests/test_model_inspector.py","tests/test_parameter_impact.py","tests/test_fourdstem_user_wiring.py","tests/test_fourdstem_cache_products.py","tests/test_stem_cuda_pipeline.py","tests/test_wave_fft.py","tests/test_multislice.py"]
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
    criteria={}
    for number,(title,prefixes) in CRITERIA.items():
        matches=[row for row in cases if any(row["test"].startswith(prefix) for prefix in prefixes)]
        all_found=all(any(row["test"].startswith(prefix) for row in cases) for prefix in prefixes)
        status=("FAIL" if any(row["status"]=="FAIL" for row in matches) else "PASS" if matches and all_found and all(row["status"]=="PASS" for row in matches) else "NOT_RUN")
        criteria[f"AT-{number:02}"]={"description":title,"status":install["status"] if number==31 else status,"tests":[row["test"] for row in matches]}
    unchanged=before==source_hashes(root)
    all_pass=all(row["status"]=="PASS" for row in criteria.values())
    report={"schema":"development-spec-acceptance-v1","started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS" if all_pass and unchanged and run.returncode==0 else "INCOMPLETE", "criteria":criteria,
        "counts":{s:sum(row["status"]==s for row in cases) for s in ("PASS","FAIL","NOT_RUN")},"cases":cases,"measurements":evidence,
        "installation":install,"command":command,"pytest_exit_code":run.returncode,"source_unchanged_during_tests":unchanged,
        "source_sha256":before,"python":sys.version,"platform":platform.platform(),
        "versions":{n:metadata.version(n) for n in ("numpy","scipy","abtem","PySide6","pytest")},
        "scope":"Acceptance of declared computational models and fixtures; external experimental calibration remains NOT_RUN. A5 is a partial higher-order basis. 3D CAD requires explicit supported field approximation. Multi-energy/multisource 4D capture remains UNSUPPORTED.",
        "external_experimental_calibration":"NOT_RUN"}
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"counts":report["counts"],"criteria":{k:v["status"] for k,v in criteria.items()}},ensure_ascii=False),flush=True)
    cpu_allowed=args.allow_no_gpu and all(row["status"]=="PASS" or (key in {"AT-25","AT-26"} and row["status"]=="NOT_RUN") for key,row in criteria.items())
    return 0 if unchanged and run.returncode==0 and (all_pass or cpu_allowed) else 1


if __name__=="__main__": raise SystemExit(main())
