# Si [110] acquisition index

**Superseding tail correction:** A later diagnostic identified scan-ROI-dependent
Gaussian weighting of the Rutherford tail. Corrected near-focus probability
recompositions, with the saved coherent multislice arrays reused, are in
`../si110_underfocus_100nm/near_focus_tail_corrected`. The original arrays below
remain archived. The earlier 512/1024 HAADF mean difference of 52.13% becomes
21.81% after the correction; neither comparison establishes convergence.

The requested acquisition is in `acquisition_gpu_1024_final/`. A completed run
contains `raw_scan.npz`, `metrics.json`, three float32 TIFFs, three 16-bit PNGs,
and `haadf_df_bf_comparison.png`. This run completed successfully in 1603.59 s
of acquisition time (26 min 44 s; 1620.51 s including setup and exports).
`output_verification.json` records the completed data/export checks.

Requested inputs: the user's unmodified `Si.cif` (a = 5.44370237 Å, eight atoms
after symmetry expansion), [110] beam direction, 10 nm disk diameter, 5 nm
thickness, 300 kV, 64 × 64 pixels, and 0.02 nm pitch. The wave calculation uses
1024 × 1024 pixels over 40 Å, 25 slices, four frozen phonon configurations, and
seed 707. Si displacement RMS = 0.085 Å is an explicit model assumption, not a
value measured in the CIF. The actual column probe is propagated at 0.05 mm
ray-integration steps; incident coordinates and slopes are not overridden.

The real post-sample D/I/P1/P2 lenses are set by
`recording_calibration/recording_calibration.json`. All physical aperture stops
remain active. Reference collection angles are HAADF 60–330 mrad,
DF 1.001–7.008 mrad, and BF 0–0.568 mrad. DF is a low-angle annulus inside the
approximately 24.8 mrad illumination disk; it does not exclude that entire
direct-beam disk.

`sampling_comparison.json` compares identical 4 × 4 scans at 512 and 1024 wave
pixels. The 512 HAADF mean is 52.13% lower, so it fails the agreed 5% reduction
criterion. This comparison supports retaining 1024; it does not establish that
1024 is fully converged. The 1024 coherent wave coverage ends at 168.80 mrad.
The separately saved high-angle contribution uses the structure-derived,
Moliere-screened Rutherford approximation through the same sequential stops;
it is neither a Mott calculation nor recovered high-angle multislice
interference. There is no shot noise.

Reproduce from the project root using the archived CIF and calibration, with
the recorded source and instrument versions:

```powershell
.venv\Scripts\python.exe scripts/generate_si110_cif_stem_scan.py --mode run --grid 1024 --phonons 4 --seed 707 --backend 'CUDA GPU' --cif outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final/input_Si.cif --recording-calibration outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final/recording_calibration.json --output outputs/si110_cif_5nm_64px_002nm/reproduction_run
```

The script refuses to overwrite an existing raw acquisition. The final folder
archives the exact generation and calibration scripts under `reproduction/`,
the selected instrument TOMLs under `instrument_inputs/`, full inputs and
source hashes in `parameters.json`, and an App-loadable format-5
`operating_profile.toml`. The profile restores the inputs; rendering the saved
images does not require rerunning the scan. A changed source/configuration or
GPU runtime can change numerical results; the recorded versions identify the
conditions used here.

Final full-raster results, expressed as fractions of emitted electrons:

| Detector | Mean signal | Share from the Rutherford extension |
| --- | ---: | ---: |
| HAADF | 0.0007951001 | 8.5886% |
| DF | 0.04376677 | 0% |
| BF | 0.0003552799 | 0% |

All arrays are finite and nonnegative; the actual pixel centres span -0.63 to
+0.63 nm on both axes with 0.02 nm spacing. TIFFs exactly match the raw arrays
after float32 conversion; PNGs match their recorded 16-bit mappings. The
coherent and tail arrays exactly reconstruct each detector signal. Maximum
per-pixel probability-conservation error is 2.22e-16. The production backend
was CuPy CUDA with its resident pipeline, 1024 × 1024 wave pixels, 25 slices
and four phonon configurations.

The delivered `operating_profile.toml` loads without skipped-field warnings.
`operating_profile_raw.toml` preserves the original export; only the two
unsupported export-only fields named in the reproduction README were removed.
After normal scan recalibration, the physical inputs match the acquisition
state. The full-grid tail share above is the final result; the 4 × 4 benchmark
statistics describe their smaller timing rasters only.

Included diagnostic folders are not substitutes for the final acquisition:

- `benchmark_1024_gpu_calibrated/` and `benchmark_512_gpu_calibrated/` are valid
  4 × 4 sampling/timing tests, not the requested 64 × 64 image.
- `prepare_step005/` verifies the production incident beam.

The following failed or superseded attempts remain local only, are excluded by
`.gitignore`, and are not included in the repository:

- `prepare/`, `benchmark_1024/`, and `benchmark_512_single/` predate the final
  recording calibration and common-stop tail fix. Their image signals must not
  be used as the final physical result.
- `acquisition_gpu_1024/` is the interrupted initial launch. The complete run
  uses the included `acquisition_gpu_1024_final/` folder.
