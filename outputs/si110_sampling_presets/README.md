# Si [110] scan sampling presets

`si110_preview_32px_004nm.toml` is a faster preview configuration based on the
calibrated, 12 mm-DPA profile. It changes only the synchronized AC/descan raster:

| Setting | Existing full scan | Quick preview |
| --- | ---: | ---: |
| Scan dimensions | 64 × 64 | 32 × 32 |
| Probe positions | 4096 | 1024 |
| Pixel spacing | 0.02 nm | 0.04 nm |
| Pixel-footprint field of view | 1.28 × 1.28 nm | 1.28 × 1.28 nm |
| Internal wave grid | 1024 × 1024 | 1024 × 1024 |
| Wave window | 4 × 4 nm | 4 × 4 nm |
| Frozen phonons | 4 | 4 |
| Si specimen | 10 nm diameter, 5 nm thick | 10 nm diameter, 5 nm thick |

The smaller raster reduces the number of propagated probe positions by 75%.
It samples the image more coarsely; it does not downsample an existing image.
Setup time and other fixed costs remain, so a fourfold total speedup is not
claimed. No new full image was acquired while preparing this profile.

The original 64 × 64 configuration is
`../haadf_dpa_clearance/si110_haadf_dpa_12mm.toml`. Load either file through the
App's operating-profile open action. This adds an explicit preview choice;
it does not silently replace the application's global numerical defaults.

The internal wave grid stays at 1024. After correcting the scan-ROI-dependent
Rutherford overlap, the matched 4 × 4 probe comparison at 4 nm FOV still gives
a 21.81% HAADF mean difference when reducing it to 512. The earlier 52.13%
figure included that overlap bug and is superseded. Reducing the grid cuts
the coherent isotropic angular coverage from 168.80 to 84.10 mrad; the
approximate Rutherford extension does not restore the lost multislice signal.
The corrected comparison is in
`../si110_underfocus_100nm/corrected_grid_comparison.json`. It reuses the saved
coherent multislice arrays with the corrected probability weights; it is a
limited grid-sensitivity check, not evidence of 1024-grid convergence.

`profile_verification.json` confirms that only raster fields changed, the
profile loads without skipped fields, and production scan calibration yields
32 × 32 positions spaced by 0.04 nm. Pixel centres span -0.62 to +0.62 nm,
while the pixel footprints still span -0.64 to +0.64 nm.
