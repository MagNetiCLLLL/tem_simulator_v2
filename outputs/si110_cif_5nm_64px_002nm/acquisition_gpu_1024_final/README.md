# Si [110] STEM acquisition

**2026-09-08 correction:** These original acquisition files are retained as
an immutable numerical record. A later diagnostic found that the Rutherford
Gaussian overlap used the scan rectangle as a material boundary, reducing
the tail near raster edges. Corrected probability recompositions are in
`../../si110_underfocus_100nm/near_focus_tail_corrected`; they reuse the
saved coherent multislice arrays without a new wave acquisition. The full
HAADF mean becomes 0.0008019003 (formerly 0.0007951001), with 9.365% tail
contribution (formerly 8.589%). See that folder's `correction.json` and the
reproduction script `scripts/correct_archived_stem_tail_overlap.py`.

64×64 pixels at 0.02 nm; 300 kV; 10 nm disk diameter and 5 nm thickness.

The specimen is the archived user CIF, oriented [110] along the beam and [1 -1 0] along X. The actual production nanoprobe/diffraction preset is used with a 0.05 mm ray-integration step. No incident coordinates or slopes were edited.

Open `operating_profile.toml` in the App to restore inputs. `raw_scan.npz` retains unnormalised source-probability fractions, coherent and tail contributions, and actual raster coordinates. TIFFs contain float32 fractions. The 16-bit PNGs map each channel's min/max to 0/65535; exact mapping is in metrics.json. The comparison figure uses independently labelled probability scales.

Finite-grid elastic frozen-phonon multislice and the structure-derived screened Rutherford high-angle extension are separate models. The extension is not a Mott calculation or recovered high-angle multislice interference. The displayed cutoff and detector tail fractions disclose that distinction. Four configurations are a finite ensemble, not an established convergence study. The explicit Si RMS assumption is 0.085 Å and is not a thermal parameter measured from this CIF. No shot noise is added.

Parameters, script/source hashes, physical detector angles and hardware are in parameters.json; production diagnostics and image statistics are in metrics.json. The archived instrument_inputs preserve the selected input TOMLs.

Acquisition completed in 1603.59 s (26 min 44 s); total setup, acquisition and
export time was 1620.51 s. The actual CuPy CUDA pipeline stayed resident at
1024 × 1024 wave pixels, with 25 slices and four phonon configurations.
`output_verification.json` records complete 64 × 64 finite/nonnegative arrays,
0.02 nm X/Y pitch, exact float32 TIFF and 16-bit PNG export checks, exact
coherent-plus-tail reconstruction, and maximum probability-budget error
2.22e-16. Full-raster HAADF tail share is 8.5886%; DF/BF tail shares are zero.

`operating_profile.toml` is the equivalent App export with zero skipped-field
warnings. `operating_profile_raw.toml` preserves the original bytes. The two
removed export-only fields and the successful state/recalibration comparison
are recorded under `reproduction/`. That folder also contains the actual
source ZIP, verified startup hashes, exact scripts, pip freeze and GPU runtime
installation details. These exports do not modify the acquisition inputs or
the archived source snapshot.

Array row zero is minimum sample Y. The comparison figure displays Y upwards;
standalone PNG/TIFF files retain the original array row order. Their pixel
values are not spatially resampled.
