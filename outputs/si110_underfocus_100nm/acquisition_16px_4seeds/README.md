# STEM underfocus ensemble

16 × 16 pixels, 0.08 nm step; effective C1=-100 nm.

Arithmetic mean of four independent single-phonon complete rasters, seeds [707, 708, 709, 710]. Each child directory contains the exact operating profile, input CIF, raw arrays and metrics. The ensemble-equivalent App profile reproduces the physical settings and configuration count, but not this independent-seed random-realisation sequence. Rerun this script for the exact recipe.

raw_scan.npz contains the mean of every total/coherent/tail/uncollected/absorbed/truncated array. ensemble_sem.npz contains thermal-configuration standard errors, not convergence or physical-model error bars. PNGs are independently min/max scaled and flipped for laboratory +Y-up viewing; TIFF/NPZ retain minimum-Y-first rows and unnormalised emitted-electron fractions. haadf_df_bf_absolute_scale.png starts each labelled scale at zero; no smoothing or sharpening is applied.

## Physical settings and interpretation

The input is the user's Si CIF, originally `C:/Users/Royal_Gray/Downloads/Si.cif`,
with lattice constant 5.44370237 Angstrom, oriented Si[110]. Its archived copy,
the original download, and `configs/reference_samples/Si.cif` share SHA-256
`944c5c81df5d813d051f96102eb9be37fac4b7647c2a9d1492d6333122be42c7`.
The finite specimen is a 10 nm diameter, 5 nm thick disk at 300 kV. The
per-element thermal RMS displacement of 0.085 Angstrom is an explicit model
assumption; it is not a thermal displacement measured from this CIF.

Underfocus 100 nm means focus 100 nm downstream of the specimen. The traced
ray contribution is +0.0836592007 nm; configured coherent C1 is
-100.0836592007 nm, giving effective C1 = -100.0000000000 nm. The existing
physical lens settings and incident ray bundle are retained. The traced
95-percent semiangle is 24.80467236 mrad, with 56.1% of emitted ray weight
surviving to the sample. The selected instrument retains its real sequential
stops and 12 mm DPA-to-HAADF spacing.

The coherent wave uses 2048 x 2048 samples over 8 nm, with 0.0390625 Angstrom
sampling, 25 slices, and a 168.8004 mrad isotropic bandwidth. The separate
Rutherford tail covers the high-angle contribution beyond coherent support;
the saved arrays retain its contribution separately. The corrected tail model
integrates the real specimen envelope independently of the scan ROI. Its
Gaussian-overlap width comes from the geometric ray RMS, not an exact integral
of the coherent defocused probe intensity.

| Channel | Actual collection band (mrad) | Mean emitted-electron fraction | Spatial relative standard deviation | Thermal relative SEM |
| --- | ---: | ---: | ---: | ---: |
| HAADF | 60.000-330.000 | 0.000811376831 | 9.1068% | 1.4088% |
| DF | 1.001151-7.008055 | 0.043821853549 | 2.2034% | 0.3366% |
| BF | 0-0.567885 | 0.000364950107 | 21.3650% | 3.0110% |

The DF annulus lies inside the illumination disk. It is a low-angle detector
signal and should not be described as an annulus excluding the direct beam.
Its minimum is 0.04200662, rather than zero: the auto-contrast image maps that
nonzero minimum to black. Its Michelson modulation is 5.3247%. HAADF tail share
is 9.2556%; the DF and BF tail contributions are zero. Thermal relative SEM is
the L2 norm of the per-pixel SEM divided by the L2 norm of the mean image.

## Sampling limits

This is a 16 x 16 preview with 0.08 nm pitch and 1.28 nm pixel-edge field of
view, not an interpolated 64 x 64 image. A matched static centre/corner/edge
check compared 8 nm and 10 nm wave windows at identical real-space sampling.
The largest fraction in the outer 0.5 nm strip of the 8 nm incident probe
window was 0.09349%. This edge-strip fraction is not a measurement of total
probability outside the window. The small static check does not establish
full spatial, angular or thermal convergence.

BF is particularly sensitive to the discrete angular mask: the 8 nm window
has 21 BF pixels, whereas the 10 nm window has 25 smaller pixels. Their summed
angular areas differ by 31.25%; this explains most of the approximately 30.7%
BF difference in the static window check. Absolute BF should therefore be
treated as limited by angular sampling. That diagnostic preceded the final
scan-ROI-independent tail correction; the correction does not change the BF
or DF coherent wave masks.

## Verification and reproduction

All four constituent scans used resident CuPy CUDA. Total runner wall time was
1310.326 s (21 min 50 s), including setup/export; summed acquisition time was
1214.441 s and reported GPU-pipeline time was 1112.660 s. Maximum per-wave
relative intensity drift across all four configurations was 0.0001254503.
Mean signals plus uncollected and absorbed fractions conserve emitted-electron
probability to 2.22e-16. Truncation is reported separately and is not added
again to this probability budget.

`output_verification.json` records successful exact arithmetic-mean and SEM
checks, float32 TIFF equality, the 16-bit PNG linear mapping and Y flip,
finite/nonnegative signals, original CIF equality, all 83 startup source
hashes, and archived acquisition/ensemble script hashes. Every constituent
profile and the sampling-equivalent App profile loads with zero skipped
fields. Sample, lenses, apertures, deflectors, recording planes and aberration
settings match the saved acquisition state. Runtime backend state and derived
corrector calibration are recalculated by the App.

`metrics.json` aggregates maxima across all constituent runs and explicitly
labels remaining inherited diagnostics as first-constituent scope. Complete
per-configuration metrics remain in each child directory. `reproduction/`
contains the source archive, package versions, verification script and exact
runner instructions. The App four-configuration profile is only physically
and statistically equivalent; exact independent-seed reproduction uses the
four child profiles or the archived ensemble runner.
