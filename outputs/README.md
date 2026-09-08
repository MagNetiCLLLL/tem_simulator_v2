# Saved calculations and validation evidence

This directory contains selected, reproducible result archives and diagnostics.
It is not a live application cache. Incomplete preparations, invalid exploratory
outputs, packaging artifacts and transient test logs are kept local and excluded
by the repository's `.gitignore`; the relevant experiment README records why.
The existing calibration and older image directories remain historical records.

## Completed specimen calculations

| Archive | What it contains | Interpretation |
| --- | --- | --- |
| [DF direct-beam clearance](df_direct_beam_clearance/README.md), [final acquisition](df_direct_beam_clearance/si110_32px_004nm_df60_100_final/README.md) | New simultaneous Si [110] HAADF/DF/BF scan, 32 × 32 pixels at 0.04 nm, 1024² wave grid, four frozen-phonon configurations; independent DF 60/100 mm geometry | DF excludes the tested illumination disk for this archived optical mapping. Dimensions and angles are not universal defaults. Upstream interception remains physical. |
| [Original Si [110] acquisition](si110_cif_5nm_64px_002nm/README.md), [final archive](si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final/README.md) | User CIF, 10 nm specimen diameter, 5 nm thickness; 64 × 64 pixels at 0.02 nm, 1024² wave grid, four frozen-phonon configurations | Historical result with the original tail-overlap weighting. Its low-angle DF lies inside the illumination disk. Use the correction below for corrected near-focus probabilities. |
| [Near-focus correction](si110_underfocus_100nm/README.md), [corrected data](si110_underfocus_100nm/near_focus_tail_corrected/) | Corrected Rutherford/coherent probability budgets using the saved original coherent arrays | A probability recomposition, not a new multislice acquisition. The original archive is preserved. |
| [100 nm underfocus](si110_underfocus_100nm/README.md), [final ensemble](si110_underfocus_100nm/acquisition_16px_4seeds/README.md) | Four independent 16 × 16 scans at 0.08 nm, seeds 707–710, averaged in intensity; 2048² wave grid over 8 nm | Independent seeds are specified by the ensemble recipe. The equivalent profile alone does not reproduce the exact random sequence. BF is not quantitatively converged. |

These finite-grid, finite-thermal-ensemble calculations do not establish complete
numerical or experimental convergence. The high-angle tail uses the documented
screened-Rutherford approximation. The calibrated 512/1024 benchmarks, corrected
grid comparison, 8/10 nm static-window benchmarks and resident-pipeline test are
included beside their associated results as supporting evidence, not full scans.

## Diagnostics and display checks

- [EDS objective-defocus diagnosis](eds_defocus_diagnostic/README.md) distinguishes
  reaching the sample plane, crossing material and collecting X-rays. Included
  cached-ray hybrid cases declare their assumed material, dose and assembly;
  they do not reconstruct the complete unknown live application state.
- [Weighted beam/sample overlap](eds_overlap_sampling/README.md) preserves the
  actual incident phase-space inputs, bounded integration comparison and spectra.
  Its two scan-preview screenshots are synthetic reproductions of reported raw
  values for UI testing, not newly calculated Si images.
- [STEM geometry-preview diagnosis](stem_geometry_diagnostic/README.md) explains
  discrete ray fractions and display contrast; its counting fixture is labelled.
- [Poisson display comparison](stem_poisson_display/README.md) applies recorded
  current, dwell and seed to saved corrected Si fractions. It is count readout
  postprocessing, not a new wave calculation or the user's current exposure.
- [DF clearance UI fixtures](df_direct_beam_clearance/README.md) explicitly label
  `ui_*_layout_fixture.png` as synthetic layout checks. Only the final acquisition
  subdirectory contains the new physical scan.
- [HAADF/DPA geometry](haadf_dpa_clearance/README.md) and
  [sampling presets](si110_sampling_presets/README.md) retain the profiles and
  assumptions used by the corresponding calculations.

## Reproduction and file meaning

Read each archive's README and verification records before using its arrays.
Raw NPZ and TIFF preserve numeric signals; PNG contrast mappings and vertical
orientation are documented per acquisition. Black in an independently stretched
fraction image need not mean zero. Expected-electron and Poisson-count arrays
have separate units and exposure assumptions from source-normalized fractions.

Final archives preserve input CIFs, operating profiles, instrument TOMLs, source
hashes and implementation snapshots where listed. Their zip files are historical
source snapshots, not installed dependencies or build caches. Restore an archive
in an isolated project copy: current code or global instrument geometry may have
changed. In particular, the DF profile must be paired with its independent
`instrument_inputs` catalog. A profile import alone does not install that catalog.
The README commands target new output directories and do not require excluded
preparation or failed-result directories. The cached-ray EDS diagnostic commands
still read original machine-local cache manifests: their required sample-plane
NPZ inputs are preserved, but those scripts do not yet accept the archived NPZ
as an alternative input. Their README explicitly states this fresh-checkout
limitation; the saved evidence remains inspectable without that local cache.
Numerical runtimes can affect results;
exact agreement across different GPU/software versions is not claimed.

Git attributes disable text conversion under `outputs/` and for reference CIFs
so new archives retain their recorded bytes despite local line-ending settings.
No historical tracked files are renormalized as part of this archival policy.
