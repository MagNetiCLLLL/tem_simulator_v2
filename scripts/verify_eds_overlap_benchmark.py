"""Verify saved spectra and export equivalent explicit diagnostic profiles.

No particle or spectrum calculation is performed. The settings are reconstructed
from the benchmark's recorded baseline and explicit overrides.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import diagnose_eds_cached_incident as d
from temsim.profile_io import save_profile
from temsim.specimen.source import active_cif_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=d.ROOT / "outputs/eds_overlap_sampling/broad_cached_raw_comparison")
    output = parser.parse_args().output.resolve()
    report = json.loads((output / "diagnostic.json").read_text())
    state = d.default_state()
    selection, values = d.read_profile(output / "baseline_profile.toml")
    d.AssemblyCatalog().apply(state, selection)
    assert d.apply_profile_values(state, values) == []
    state.sample.size_x_nm = state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 5.0
    state.sample.centre_x_nm = state.sample.centre_y_nm = 0.0
    state.sample.envelope_shape = "disk"
    state.sample.inserted = True
    state.sample.eds_support_material_key = "vacuum"
    state.sample.eds_poisson_enabled = False
    state.objective_lens.percent = 68.0
    state.ac_deflector.scan_enabled = False
    assert float(state.sample.z_mm) == report["baseline_sample_z_mm"]
    cif = Path(active_cif_path(state.sample))
    shutil.copyfile(cif, output / "input_Si.cif")
    checks = dict(
        scope="Read-only numerical verification plus equivalent profile exports from the benchmark's recorded baseline and overrides; no numerical rerun.",
        original_terminal_identical=len({row["original_terminal_sha256"] for row in report["cases"].values()} | {report["original_terminal_sha256"]}) == 1,
        original_zero_hits=report["original_elastic_metrics"]["sample_hit_trajectory_count"] == 0,
        source_hash_matches_current={key:hashlib.sha256((d.ROOT/key).read_bytes()).hexdigest()==value for key,value in report["source_sha256"].items()},
        source_hash_note="Recorded acquisition hashes are preserved. A false comparison marks code changed after the recorded run; the later eds_signal quadrature-ID provenance addition does not change the numerical estimator.",
        cif_original=str(cif), cif_sha256=hashlib.sha256(cif.read_bytes()).hexdigest(),
        cif_copy_identical=cif.read_bytes()==(output/"input_Si.cif").read_bytes(),
        spectra={}, profiles={})
    for label, row in report["cases"].items():
        with d.np.load(output / (label+"_spectrum.npz")) as arrays:
            expected = arrays["expected_counts"]
            checks["spectra"][label] = dict(
                finite_nonnegative=bool(d.np.isfinite(expected).all() and (expected>=0).all()),
                sum_matches_metrics=bool(d.np.isclose(expected.sum(), row["total_expected_counts"], rtol=1e-12, atol=0)))
        state.sample.eds_overlap_sampling_enabled = row["enabled"]
        state.sample.eds_overlap_sampling_points = row["requested_points"]
        profile = output / (label+"_equivalent_profile.toml")
        save_profile(profile, state, selection)
        (output / (label+"_equivalent_state.json")).write_text(json.dumps(state.to_dict(),indent=2),encoding="utf-8")
        restored = d.default_state()
        restored_selection, restored_values = d.read_profile(profile)
        d.AssemblyCatalog().apply(restored, restored_selection)
        skipped = d.apply_profile_values(restored, restored_values)
        assert restored.sample.eds_overlap_sampling_enabled == row["enabled"]
        assert restored.sample.eds_overlap_sampling_points == row["requested_points"]
        assert restored.objective_lens.percent == 68.0
        assert restored.sample.z_mm == state.sample.z_mm
        assert (restored.sample.size_x_nm,restored.sample.size_y_nm,restored.sample.thickness_nm)==(10.,10.,5.)
        checks["profiles"][label] = dict(sampling_settings_reload_match=True,objective_percent=restored.objective_lens.percent,
            sample_z_mm=restored.sample.z_mm,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),skipped=skipped,
            limitation="Loading the profile computes new optical rays; this diagnostic used a separately stored real cached bundle with unknown original column state.")
    assert checks["original_terminal_identical"] and checks["original_zero_hits"] and checks["cif_copy_identical"]
    assert all(all(row.values()) for row in checks["spectra"].values())
    (output/"artifact_verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
