# Si [110]: DF diagnosis and 100 nm underfocus

This folder separates a diagnosis of the previous near-focus image, a correction
of its Rutherford probability weights, and a new underfocus acquisition.

**The new acquisition is complete.** All four 16 × 16 rasters ran on resident
CuPy CUDA without CPU fallback. Acquisition time summed to 1214.44 s; total
ensemble wall time was 1310.33 s (21 min 50 s). The final mean and raw data are
in `acquisition_16px_4seeds`, including:

- `haadf_df_bf_absolute_scale.png`: each labelled scale starts at zero.
- `haadf_df_bf_comparison.png`: independent automatic contrast per channel.
- `raw_scan.npz` and channel TIFFs: unnormalised emitted-electron fractions.
- `ensemble_sem.npz`: variation across the four thermal configurations.
- `ensemble_equivalent_profile.toml`: loadable physical settings; see the
  independent-seed reproduction qualification below.
- `independent_output_review.json`: independent checks of exact arithmetic
  averages, SEM, PNG/TIFF encoding, scan pitch, C1, CUDA and original CIF identity.

| Channel | Mean fraction of emitted electrons | Spatial relative standard deviation |
| --- | ---: | ---: |
| HAADF | 0.0008113768 | 9.11% |
| DF | 0.0438218535 | 2.20% |
| BF | 0.0003649501 | 21.37% |

The maximum final probability-budget error is 2.22e-16. The approximate tail
contributes 9.26% of HAADF and zero to these DF/BF channels. Spatial image
variation in this table is not a numerical uncertainty estimate. BF's sampling
limitation remains material; see the final section before using absolute values.

## Why the previous DF looked like disks separated by black lines

The original 64 × 64 DF array contains no zero pixels: its minimum and maximum
are 0.038165636956 and 0.052082687409 per emitted electron. Independent automatic
contrast maps that positive minimum to black. See `df_display_diagnostic.png`
for the same array with both stretched and zero-based scales and a line profile.

The actual recording geometry sends approximately 1.001–7.008 mrad to DF, inside
the approximately 24.805 mrad illumination disk. This channel contains coherent
low-angle contrast and is not a pure annular signal outside the direct disk.
The masks do not move discontinuously across this raster. See
`physics_diagnostic.json` for raw-data, detector-mask and normalisation checks.
These checks explain the display and rule out several implementation causes;
they do not establish full experimental accuracy or numerical convergence.

The DF hardware has a 2 mm inner and 14 mm outer diameter. Its current
angle-to-position scale is approximately 0.999 m/rad. This geometry itself
sets the low-angle interval; the DPA is not clipping an intended high-angle
ring into a low-angle one. At this scale, placing the inner edge outside the
illumination disk would require an inner diameter above 49.55 mm, exceeding
the existing outer diameter. A future outside-disk DF configuration therefore
requires a different recording map or hardware geometry, with the upstream
HAADF interception of 60–330 mrad checked again. See
`df_geometry_interpretation.json` for checks against the acquired profiles.

The app now shows the automatic intensity limits and detector angular interval,
including a warning when the interval overlaps the illumination disk.

## A separate Rutherford error was corrected

The old tail overlap calculation implicitly treated material beyond the scan
raster as vacuum. It now integrates a Gaussian footprint over the physical
sample envelope at each probe coordinate. Changing scan size no longer changes
the inferred material under the same probe. For these central rasters within a
10 nm disk, the calculated overlap is 1 throughout.

`near_focus_tail_corrected` contains the corrected previous 64 × 64 scan. This
is a probability recomposition using the original coherent multislice results,
not a newly acquired wave calculation. It rescales the coherent contribution
from `(1 - P * old_overlap)` to `(1 - P * physical_overlap)` and adjusts the
stored tail and uncollected budgets consistently. The original acquisitions
are untouched. `correction.json` and `recomposition_verification.json` record
the independent algebra, input hashes, probability and image-encoding checks.

The corrected near-focus HAADF mean is 0.0008019003; its approximate high-angle
tail contributes 9.365% of HAADF. The similarly corrected 512/1024 wave-grid
benchmark difference is 21.81% in HAADF mean (`corrected_grid_comparison.json`),
superseding the old 52.13% figure that also included the raster-overlap error.

## New underfocus settings

The target is a focus **100 nm downstream of the specimen reference plane**.
The production propagation convention uses effective C1 = −100 nm. A synthetic
vacuum propagation of this probe by +100 nm reproduces the zero-C1 focus with
relative complex-wave error 1.35e-14. This agrees with the C10/defocus sign
relationship in [abTEM's conventions](https://abtem.github.io/doc/user_guide/appendix/conventions.html).
The saved additional C1 compensates the small traced-ray waist offset. Lens
strengths remain those of the calibrated recording profile; this calculation
sets coherent-probe aberration rather than retuning physical lens currents.

- Source: the user's unmodified `Si.cif`; beam [110], X [1 −1 0].
- Specimen: 10 nm diameter disk, 5 nm thickness; 300 kV.
- Raster: 16 × 16, 0.08 nm step; 1.28 nm pixel-footprint field of view. Reducing
  the raster was authorised by the user; this is not the earlier 64 × 64 scan.
- Wave: 2048 × 2048 over 8 nm; 0.00390625 nm spatial step; 25 slices of 0.2 nm.
- Thermal model: four independent complete single-configuration rasters with
  seeds 707–710, averaged in intensity; Si RMS displacement 0.0085 nm.
- Recording: calibrated first-order optics, 12 mm DPA opening; HAADF 60–330,
  DF approximately 1.001–7.008 and BF 0–0.568 mrad. No Poisson counting noise.
- Coherent isotropic angular cutoff: 168.8004 mrad. HAADF above the coherent
  domain uses the approximate Rutherford extension, not converged full-angle
  multislice/Mott scattering.

The final acquisition directory is `acquisition_16px_4seeds`. Its constituent
folders contain exact profiles, input CIF copies, raw arrays, progress and
metrics. The top-level ensemble images and data are generated only after all
four rasters succeed. The equivalent app profile shares physical settings and
configuration count, but exact reproduction requires the four independent
seeds in `ensemble_recipe.json` and `scripts/run_stem_profile_ensemble.py`.

## Window and sampling limits

A 100 nm defocus broadens the geometrical probe to roughly 5 nm diameter, larger
than this scan field. Atomic-column resolution should not be expected from this
defocused acquisition. An 8 nm window was chosen to retain the coherent angular
support and contain the broader wave; the old 4 nm window would be too small.

Before the full acquisition, nine static-lattice points were compared using
8 nm/2048 and 10 nm/2560 windows at the same spatial step. Mean HAADF differed
by 0.0344% and DF by 0.4507%. The 8 nm probe's worst outer 0.5 nm boundary strip
contained 0.0935% of incident intensity. These limited checks support using the
8 nm window for this preview; they do not constitute full convergence.

**BF is not quantitatively converged.** Its narrow disk contains only 21
reciprocal-grid samples at 8 nm and 25 at 10 nm. Their represented disk areas
differ by 31.25%, explaining most of the 30.70% BF mean difference. BF values
should be treated as qualitative until fractional-pixel detector integration
or finer angular sampling is validated. Details are in
`static_window_comparison.json` and the two static benchmark directories.
The included directories are `benchmark_static_8nm_v2/` and
`benchmark_static_10nm/`; the incomplete first `benchmark_static_8nm/`
preparation remains local only and is excluded from the repository.

Four thermal configurations are a finite ensemble. Their standard error only
measures this thermal sampling, not wave-grid, recording-optics or physical-model
uncertainty. The real tail still uses a ray-RMS Gaussian envelope approximation;
it does not integrate the full defocused coherent probe footprint.
