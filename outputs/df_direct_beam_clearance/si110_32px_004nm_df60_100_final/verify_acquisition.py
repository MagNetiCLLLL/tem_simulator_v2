"""Read-only numerical checks; writes only acquisition_verification.json here."""
from pathlib import Path
import hashlib
import json
import sys
import tomllib
import zipfile
import numpy as np
from PIL import Image
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import read_profile, apply_profile_values
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.detector.stem_signal import physical_angular_detectors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = json.loads((HERE / "parameters.json").read_text())
    m = json.loads((HERE / "metrics.json").read_text())
    source = Path(p["input_provenance"]["source_archive"])
    with np.load(HERE / "raw_scan.npz", allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    x, y = raw["scan_x_nm"], raw["scan_y_nm"]
    assert x.shape == y.shape == (32, 32)
    assert np.allclose(np.diff(x, axis=1), .04, atol=1e-10)
    assert np.allclose(np.diff(y, axis=0), .04, atol=1e-10)
    assert np.isclose(x.max()-x.min()+.04, 1.28)
    assert np.isclose(y.max()-y.min()+.04, 1.28)
    rng = np.random.default_rng(42)
    images = {}
    for key in ("haadf", "df", "bf"):
        values = raw[key + "_fraction"]
        assert values.shape == (32,32) and np.isfinite(values).all() and values.min() >= 0
        assert np.ptp(values) > 0
        assert np.array_equal(tifffile.imread(HERE / f"{key}.tiff"), values.astype(np.float32))
        display = np.rint((values-values.min()) / (values.max()-values.min()) * 65535).astype(np.uint16)
        assert np.array_equal(np.asarray(Image.open(HERE / f"{key}.png")), np.flipud(display))
        tail = raw[key + "_tail_fraction"]
        assert np.all(tail >= 0) and np.all(values-tail >= -1e-15)
        assert np.allclose(raw[key + "_coherent_scaled_fraction"]+tail, values, rtol=0, atol=1e-18)
        expected = values * (100e-12 * 10e-6 / 1.602176634e-19)
        assert np.array_equal(expected, raw[key + "_expected_electrons"])
        counts = rng.poisson(expected)
        assert np.array_equal(counts, raw[key + "_poisson_counts"])
        assert np.array_equal(counts, tifffile.imread(HERE / f"{key}_poisson_counts.tiff"))
        count_display = np.rint(counts/max(int(counts.max()),1)*65535).astype(np.uint16)
        assert np.array_equal(np.asarray(Image.open(HERE / f"{key}_poisson_counts.png")), np.flipud(count_display))
        images[key] = dict(mean_fraction=float(values.mean()), range=[float(values.min()),float(values.max())],
            tail_share=float(tail.sum()/values.sum()), expected_electrons_mean=float(expected.mean()),
            poisson_counts_mean=float(counts.mean()), tif_png_npz_verified=True)
    budget = sum(raw[k + "_fraction"] for k in ("haadf","df","bf")) + raw["uncollected_fraction"] + raw["absorbed_fraction"]
    budget_error = float(np.max(np.abs(budget-1)))
    assert budget_error < 1e-12
    prod = m["production_metrics"]
    assert prod["cuda_resident_pipeline"] and prod["wave_compute_backend"] == "CuPy CUDA"
    assert prod["specimen_configuration_count"] == prod["cuda_configuration_count"] == 4
    assert prod["grid_pixels_x"] == prod["grid_pixels_y"] == 1024 and prod["field_of_view_angstrom"] == 40
    assert p["acquisition_frame_period_s"] == 1.0
    assert m["readout"]["dwell_time_s"] == 10e-6
    assert sha(HERE / "input_Si.cif") == sha(source / "input_Si.cif") == p["input_provenance"]["source_cif_sha256"]
    with zipfile.ZipFile(HERE / "implementation_snapshot.zip") as archive:
        for name, expected in p["implementation"]["sha256_by_file"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        assert hashlib.sha256(archive.read("scripts/run_df_clearance_scan.py")).hexdigest() == p["script_sha256"]
    actual_root = HERE / "instrument_inputs"
    for relative in ("gun/FEG.toml", "column/C3_ProbeCorrector.toml"):
        assert sha(actual_root / relative) == sha(source / "instrument_inputs" / relative)
    old = tomllib.loads((source / "instrument_inputs/project_and_recording_system/EnergyFilter.toml").read_text())
    new = tomllib.loads((actual_root / "project_and_recording_system/EnergyFilter.toml").read_text())
    new_df = next(part for part in new["parts"] if part["key"] == "df")
    new_df["inner_diameter_mm"] = 2.
    new_df["outer_width_mm"] = 14.
    assert new == old, "Unexpected mechanical changes beyond the DF ID/OD"
    catalog = AssemblyCatalog(root=actual_root)
    selection, values = read_profile(HERE / "operating_profile.toml")
    loaded = default_state()
    catalog.apply(loaded, selection)
    skipped = apply_profile_values(loaded, values)
    assert skipped == [], skipped
    apply_physical_layout_to_state(loaded, assembly_root=actual_root, preserve_operating_parameters=True)
    assert loaded.ac_deflector.scan_frame_period_s == p["acquisition_frame_period_s"]
    assert loaded.ac_deflector.scan_pixels_x == loaded.ac_deflector.scan_lines == 32
    assert loaded.ac_deflector.scan_pixel_size_nm == .04
    df = next(d for d in loaded.stem_detectors if d.key == "df")
    assert df.inner_diameter_mm == 60 and df.outer_diameter_mm == 100
    lens_rows = {row["key"]:row for row in p["state"]["lenses"]}
    for lens in loaded.lenses:
        if lens.key in lens_rows:
            for field in ("percent","b0_t","z_mm"):
                if field in lens_rows[lens.key] and hasattr(lens,field):
                    assert np.isclose(getattr(lens,field),lens_rows[lens.key][field],rtol=1e-11,atol=1e-10), (lens.key,field)
    for field in ("size_x_nm","size_y_nm","thickness_nm","wave_grid_pixels","wave_field_of_view_angstrom",
                  "wave_defocus_nm","wave_frozen_phonon_configurations","wave_frozen_phonon_seed"):
        assert getattr(loaded.sample,field) == p["state"]["sample"][field], field
    _, angles = physical_angular_detectors(loaded, loaded.stem_detectors)
    for key, angle in angles.items():
        assert np.isclose(angle.inner_mrad,p["physical_detector_reference_angles"][key]["inner_mrad"],rtol=1e-10)
        assert np.isclose(angle.outer_mrad,p["physical_detector_reference_angles"][key]["outer_mrad"],rtol=1e-10)
    result = dict(passed=True, images=images, probability_error=budget_error,
        array_count=len(raw), grid=[1024,1024], scan=[32,32], pitch_nm=.04, pixel_fov_nm=1.28,
        cuda_resident=True, frozen_phonons=4, seed=707, input_cif_sha256=sha(HERE / "input_Si.cif"),
        archive_hashes_verified=True, mechanical_changes_only_df_2_to_60_14_to_100=True,
        profile_with_independent_catalog_import=dict(skipped=skipped, df_id_mm=60, df_od_mm=100,
            lenses_wave_specimen_and_reference_angles_match=True),
        scope="No new ray/wave simulation; array, export, catalog and profile-import checks only.")
    (HERE / "acquisition_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
