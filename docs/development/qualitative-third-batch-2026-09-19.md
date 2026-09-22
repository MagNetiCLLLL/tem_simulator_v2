# Third qualitative-science batch: detector response and current accounting

## Scope

This batch extends the existing classical detector/current diagnostics and shared
parameter descriptions. It uses the simulator's geometry and existing detector
response models. No commercial calibration, sensor technology, physical default,
independent downstream source, new emission law or acquisition is introduced.
Coherent tip-to-column development remains paused; the energy filter remains a
later integration step. The work is local and uncommitted.

## Reproduced defects and repairs

1. The virtual ray detector image replaced an all-zero weight population with
   equal positive weights. Two zero-weight particles therefore reported an
   accepted weight of 2 instead of 0. `detector/plane_image.py` now preserves zero
   weights and validates finite, non-negative weights before binning. Negative
   weights are rejected instead of being clipped and replaced. NaN/Inf were
   already rejected later by the intensity stage; they now fail at the weight
   boundary with the appropriate explanation.
2. The same diagnostic cropped images around the column origin even for a moved
   detector. Translating both a 2 mm disk and its unit-weight incident population
   from (0, 0) to (3, -4) mm changed the recorded weight from 1 to 0. Histogram
   bounds and mask coordinates now follow the physical detector centre; both
   cases retain weight 1. This changes diagnostic sampling, not electron paths.
3. `physics/interaction_budget.py` trusted downstream stop metadata without also
   applying the authoritative incident survival mask. A reference branch could
   reintroduce a particle already stopped before the specimen. The isolated
   reproduction had 0.25 upstream loss yet reported 1.0 downstream population.
   The budget now reports 0.75 downstream, 0.25 upstream loss and zero residual.

The PSF overlay in the existing transverse view is explicitly labelled a virtual
optical-reference response. It ignores recording stops for diagnostic inspection,
including retracted detectors; it is not acquired counts or collected current.
Only its display array is peak-normalized. Weighted input/response arrays retain
their original scale. The physical stop and integrated-current routes remain
separate and unchanged; disabling readout still leaves detector interception.

These diagnostic quantities are calculated from retained results when requested;
no new transported-state cache or source identity is introduced by this batch.

## Parameter meanings and units

`parameter_registry.py` now explains detector insertion, electronic readout,
active width/opening, centre offsets, native camera pixel count, PSF widths and
orientation, and provenance. Existing runtime and structural editors reuse the
descriptions. The structural editor retains its existing category vocabulary:
active widths are physical dimensions, PSF values are response/operating values,
and provenance does not acquire a material interpretation. No sweep permissions
or live physical settings changed.

Existing STEM and pixelated-response controls now distinguish:

- Fixed incident dose, collection fraction, current and electrons per scan pixel.
- Independent Poisson sampling, its seed and the deterministic expectation.
- Mean detection efficiency versus DQE and physical absorption. Missing detected
  signal does not become a transmitted electron population.
- PSF width in physical mm versus pixel-response width in output pixels.
- Dark electrons **per pixel per exposure**, not a current automatically scaled
  with dwell time; electron-equivalent read noise and saturation; counts/electron
  gain and additive counts offset.

The existing adjustable pixel response applies independent Poisson sampling
after blurring its expected signal. It is a simplified mean-response/noise model,
not an event-by-event charge-sharing covariance model. Its Gaussian read noise is
clipped at zero before upper saturation/gain, biasing low-signal means upward.
These limitations are now stated in the controls; this batch does not silently
replace the stored model or resume wave acquisition. No detector binning control
or CCD-specific model is invented.

## Scientific evidence

| Check | Declared conditions and result |
| --- | --- |
| Zero/invalid weights | Zero stays exactly zero; negative/non-finite weights fail explicitly |
| Translation symmetry | Same disk and incident population translated together; unit accepted/response weight preserved |
| PSF covariance | Fixed 0.005 mm grid; sigma 0.04/0.02 mm, doubled widths, and 45-degree rotation. Total weight 7 retained to 2e-13; covariance agrees with rotated analytic Gaussian within 0.2%; wider PSF lowers raw peak without creating weight |
| Linear response and saturation | Uniform 40-electron input, efficiency 0.25, gain 2, offset 3 gives 23 counts; double input doubles offset-subtracted signal. A 12-electron saturation gives 27 counts instead of linear growth |
| Shot noise | 320 x 320 independent samples at mean 10; fixed seed; mean/variance each satisfy predeclared six-standard-error bounds and repeat exactly |
| Dwell and raster | Fixed 0.12 s frame and collection fraction, 3 x 4 versus 6 x 8 scan grids: same total expected electrons, smaller dose per pixel. Doubling frame time doubles expected electrons but leaves current fixed |
| Upstream loss | Incomplete downstream reference metadata cannot restore a particle lost before the specimen |
| Actual tip-origin chain | 49 emitted particles, analytic classical transport, installed extraction/acceleration/apertures/column, retracted specimen, default-off vacuum; physical stop fractions and current budget checked at each detector |

The actual-chain test executes column steps 0.125 and 0.0625 mm. Both produce:

| Physical interception | Fraction of emitted weight |
| --- | --- |
| Annular high-angle detector | 23/49 = 0.46938775510204095 |
| Fluorescent screen | 23/49 = 0.46938775510204095 |
| Projection-chamber differential-pumping aperture | 2/49 = 0.04081632653061224 |
| Column wall | 1/49 = 0.02040816326530612 |

All 49 reach the specimen entrance; the listed downstream losses sum to one.
Turning off the annular detector's electronic readout preserves stop identities,
weights and trajectories in the existing physical calculation/reuse path.
Current-budget residuals are below 2e-14 (dimensionless). These fractions describe
this finite-particle diagnostic, not measured detector efficiencies. Stable stop
classification across two column steps is not independent source-sampling,
gun-step, boundary-location or image-resolution convergence.

The PSF covariance test uses well-resolved, finite-support kernels. Extremely
broad/undersampled PSFs and their existing support cap remain outside this receipt;
qualify support truncation and pixel sampling separately before using such cases
quantitatively. Pixel-response tests are isolated readout tests, not evidence of
an admitted coherent source or complete camera acquisition.

## Validation receipts

Generated receipts remain local under `tmp/qualitative-detectors-20260919/`.
All test counts overlap; do not sum them as unique cases.

- Initial five-file baseline: **107 passed**, `baseline.xml`.
- New defect reproductions before fixes: **6 failed**, `reproduced.xml`; afterward
  **6 passed**, `fixed-reproducers.xml`. Two of these check earlier validation
  of already-rejected NaN/Inf values, not newly admitted physics failures.
- Initial new trend/meaning tests: **29 passed**, `trends-first.xml`.
- Actual tip-chain test: **1 passed**, `actual-chain.xml`, including recorded
  interception fractions and both column steps.
- Expanded dependency run: **261 passed, 2 failed**, `final-regression.xml`.
  Both failures are historical coherent-image admission tests, listed below.
- Editor integration was refined after two focused failures: the structural
  editor must preserve its category enum and recognize canonical detector keys.
  The complete affected editor/semantics checks and selected PSF view test then
  passed **77 cases**, `editor-corrected.xml`.
- Final complete classical detector selection: **262 passed, 2 deselected**,
  `final-classical-detectors.xml`. The two exclusions are named below. There
  were no skips or failures in the selected set; this is not a full-repository pass.
- Final declared classical software lane: **259 passed, exit 0**,
  `classical-baseline/report.json`, run `8414a2ebbfed48e4935688b10b317609`.
  Software scope is PASS; full simulator remains UNQUALIFIED. All **960** recorded
  source/configuration/input hashes were unchanged during execution and matched
  the final working tree afterward.
- Serial `compileall -q src/temsim tests scripts` completed with exit 0 after
  tests. Final whitespace and local documentation-link checks passed. Existing
  Pydantic deprecation warnings remain. No full-repository, native interactive
  desktop, microscope hardware or long coherent-wave run was performed.

### Preserved coherent-image blockers

The following unchanged tests in `tests/test_fourdstem.py` fail with
`UnsupportedWaveSource` at source admission, before a long coherent calculation:

- `test_stem_wave_solver_streams_configuration_averaged_diffraction_cube`
- `test_high_accuracy_adapter_separates_cube_and_downstream_plan_signatures`

Both failures reproduce in the unchanged archived HEAD `7c52fa0` implementation
(`historical-wave-gates.xml`). Source-admission and wave-entry files were compared
with committed HEAD. Neither test, gate nor source model was changed to force a
pass. The final classical detector selection explicitly deselects these two
cases; their failed evidence remains visible. They do not qualify as passing or
newly fixed cases. The four earlier absolute alignment targets also remain open
and were not rerun in this detector batch.

### Reproduction

Use the project `.venv`, `QT_QPA_PLATFORM=offscreen`, and a fresh output directory.
The focused test entry is `tests/test_qualitative_detector_trends.py`; affected
integration files include detector PSF, record-plane masks/routing, beam-plane
budgets, detector controls, dose caching, parameter semantics and registry.
The final detector receipt records its complete test selection and deselections.
Run the separate declared classical lane through
`scripts/validate_classical_scope.py --scope classical`.

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$detectorChecks = @(
    'tests/test_detector_point_spread.py', 'tests/test_record_plane_detector_masks.py',
    'tests/test_record_plane.py', 'tests/test_illumination_current.py',
    'tests/test_beam_plane_data.py', 'tests/test_stem_detector_control.py',
    'tests/test_stem_dose_cache.py', 'tests/test_fourdstem.py',
    'tests/test_parameter_semantics.py', 'tests/test_parameter_semantics_ui.py',
    'tests/test_parameter_registry.py', 'tests/test_real_inelastic.py',
    'tests/test_reference_physics_routing.py', 'tests/test_detailed_ray_presentation_guards.py',
    'tests/test_qualitative_detector_trends.py', 'tests/test_qualitative_parameters.py'
)
$classicalSelection = 'not test_stem_wave_solver_streams_configuration_averaged_diffraction_cube and not test_high_accuracy_adapter_separates_cube_and_downstream_plan_signatures'
.venv/Scripts/python.exe -m pytest @detectorChecks -k $classicalSelection -q -o addopts= -o junit_family=xunit1 --tb=short --junitxml=tmp/detector-next/regression.xml
.venv/Scripts/python.exe scripts/validate_classical_scope.py --scope classical --output tmp/detector-next/classical --timeout-seconds 900
```

## References and remaining scope

The dose conversion uses the exact elementary charge
([NIST](https://physics.nist.gov/cuu/Constants/Value/e.html)). The finite image
convolution uses zero padding, which permits response to leave the output field
([SciPy fftconvolve documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html)).
These references support the limiting checks, not simulator geometry or measured
calibration. Signs, collection planes and actual weights come from the executed
local implementation.

Next work should qualify detector sampling/PSF support across bounded parameter
ranges and present physical loss versus response loss consistently, then continue
remaining multipole meanings and bounded sweep/convergence presentation. Preserve
the current filter boundary and integrate its physical response last. Native
interactive GUI operation, full TEM/STEM images, real hardware and full-chain GPU
performance are not qualified here.
