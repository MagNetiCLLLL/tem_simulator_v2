# Bounded coherent Si diffraction evidence

## 4 October 2026 — actual GPU execution and bounded comparison

The complete public `TipWaveObservationSession` executed from the same applied
Tip through the gun, column and imported Si specimen to
**Z = 1605.1934995243355 mm** on an **NVIDIA RTX 5090, 32 GB**. The resulting
image resolves the direct beam and eight surrounding diffraction peaks. Column
and material operators used CuPy **complex128 / float64**, without CPU fallback;
near-tip and gun propagation remained on CPU. Extraction, acceleration,
focusing, multipoles, apertures, walls and physical detector interception were
retained. No source width, energy, optical strength or numerical resolution was
changed to obtain the speed improvement.

The physical input snapshot is identical to the small-source CPU example below
(SHA256 `a891504c434e03f06aa5df31bf647837845270f9af3ffa0650dc5c766ce3f8d3`).
The execution request selects **Require GPU**, a **24 GiB device working
limit**, and a fresh checkpoint directory. It retains three energy modes,
reference material grid 96, 40 Å field, 10 Å target slices, 0.2 mm column step,
and the final **three 6720 × 6720 complex modes**. This is the user's monatomic
Fm-3m FCC structure, not a substituted diamond-Si structure.

Baseline: dirty `master`, HEAD
`c4bbd11f87610e3df06fc35c9b61e7b1c8cf5990`; existing edits and deletions were
preserved. Executed solver source identity:
`7a97ef1d2372d1c7117cbb75b677f8320bf243235859f2a2612cfdc53d3560e4`.
Environment: Windows, Python 3.12.3, NumPy 2.4.6, SciPy 1.18.0, CuPy 14.1.1,
PySide6 6.8.3, NVIDIA driver 610.62. Numerical jobs were serialized with a
maximum 16-worker CPU budget on the 32-logical-CPU host; regression tests used
one numerical thread. Device modes share one bounded allocator and execute
sequentially.

| Executed stage | GPU-enabled run | Historical CPU run | GPU retained probability per tip electron |
| --- | ---: | ---: | ---: |
| Tip through specimen entrance, including readout | 213.16 s | 568.16 s | 0.9981835308440273 |
| Specimen and column to diffraction Z, including readout | 118.97 s | 903.66 s | 0.989817072141876 |
| Complete run, including output handling | **343.24 s** | **1473.33 s** | Same final field |

The approximately 4.29-fold end-to-end reduction compares the earlier CPU
source revision with this GPU revision, using identical physical inputs and
discretisation. It is historical whole-chain evidence, not a same-revision
controlled speed ratio. Both executions used fresh propagation-cache locations;
OS file caches and library initialization were not independently controlled.

A separate **same-revision** comparison loaded the completed diffraction
checkpoint and propagated its three full 6720² modes through one actual column
step, from 1605.1934995243355 to 1605.1944995243355 mm. CPU took **49.4056 s**;
Require GPU took **11.2720 s**, a **4.383-fold** local reduction including device
transfers and setup. CPU ran first; neither path received explicit FFT warmup.
The maximum relative complex-field L2 difference was **9.283 × 10⁻¹⁶**, without
fitted phase or renormalisation. Coordinate basis, origin, curvature, tilt and
final Z matched exactly; mode-weight differences were at most 5.56 × 10⁻¹⁷.
This checks the actual column operator, not all possible material or hardware
settings, and is not a whole-application speed guarantee.

On identical 800 × 800 physical readout bins over ±100 µm, the completed GPU
intensity differs from the historical CPU result by **1.534 × 10⁻⁹ relative
L2**. Full per-mode complex comparisons, with no fitted phase, report a worst
nodal residual of **7.708 × 10⁻⁶** and corresponding coordinate differences
below 9.52 × 10⁻¹⁵ m. These different-revision residuals are reported evidence,
not an invented acceptance tolerance or physical convergence certification.

The initial GPU attempt stopped on the third energy mode because private
allocator split blocks survived scope cleanup. Its failure receipt was retained.
The fix keeps allocator ownership until live arrays are released and then drains
unused blocks; it does not relax memory admission or lower precision. After the
fix, 24 repeated real material operations held device use constant at
1,712,914,432 bytes. The successful full run and controlled comparison finished
with no retained private pool.

Focused validation on the final source:

- **115 passed, zero failed/skipped, exit 0**, 41.426 s: complete coherent-page
  test file, device test file and the backward-plane continuation regression.
  This covers offscreen Qt dispatch, backend capture, errors and actual CUDA
  device cases. Receipt: `outputs/gpu-coherent-20261004/final-gui-device-regression.xml`.
- **11 CUDA material regression cases passed, exit 0**, including repeated
  allocator cleanup and preservation of live device results. This is a separate
  selected suite; the counts are not a claim of a complete registered scope.
- An earlier continuation run correctly rejected a solver identity changed
  during execution. After freezing the source, that specific case passed in the
  115-case run; the original failed receipt remains available.

Generated evidence remains ignored under
`outputs/gpu-coherent-20261004/si-public-final-gpu-v2/`: `receipt.json`,
`diffraction-roi.png`, `readout-comparison.json`,
`historical-complex-comparison.json`, `same-revision-column-benchmark.json`,
the input snapshot, full execution records and complex checkpoints. No generated
arrays or calculation caches are intended for Git. Changes are local and
uncommitted. Native-desktop interaction, a fresh isolated-wheel install and full
scope/convergence validation are **NOT_RUN** for this acceleration change.
The physical-model limitations of the small-source CPU example remain in force.

## 4 October 2026 — small-source CPU result

The public tip-to-observation pipeline completed with **5 nm planar emission
FWHM, 0.3 eV mean kinetic energy and 0.3 eV RMS-equivalent FWHM**, using the
explicit driven Gaussian-Schell boundary. The source was neither widened nor
raised to the older demonstration's 30 eV. The user's Fm-3m FCC Si (a =
3.82166108 Å, [001]/[100], 10 nm disk, 5 nm thickness) produced a direct beam and
**eight distinct surrounding 200/220 diffraction peaks** at
**Z = 1605.1934995243355 mm**, the Ray Diagram objective diffraction reference.
The independent fine-field reference is 1605.1934964 mm; the production 0.2 mm
quadratic discretisation predicts 1605.1935413 mm. These are numerical
comparisons within this column model, not instrument calibration.

Execution used Windows CPU, Python **3.12.3**, PySide6 **6.8.3**, with bounded
internal parallelism. Repository baseline was `master` at
`c4bbd11f87610e3df06fc35c9b61e7b1c8cf5990`, with pre-existing uncommitted work
preserved. Executed solver source identity:
`9cd4bdf0a3620eb6fe005c90d64e34c28dda59e973f4b05fd25d8daaec85cc08`.

| Executed stage | Time | Retained probability per tip electron |
| --- | ---: | ---: |
| Tip through specimen entrance | 568.16 s | 0.9981835308440414 |
| Specimen and remaining column to reference Z | 903.66 s | 0.989817072208214 |
| Complete local run including readout | 1473.33 s | Same final field |

Three energy modes retain the physical nonzero energy spread. Material inputs
were reference grid 96 over an explicit 40 Å field, 10 Å target slices and a
24 GiB working budget. The entire incident field was retained, with upward
FFT rounding and carrier-covariant Galerkin propagation; final complex modes
were 6720 × 6720. These qualitative settings do not establish grid,
slice-thickness, energy-quadrature or diffraction-intensity convergence.
The exact optical changes and editable GUI controls are in
[COHERENT_BEAM.md](COHERENT_BEAM.md).

Independent reciprocal-lattice predictions use the imported cell and current
optics, without fitting the image. On a common ±100 µm ROI with 0.25 µm bins,
fixed-window maxima are **0.497–0.526 µm** from the eight predictions. These
measure maxima, not fitted centroids. Probability outside that ROI is
**3.5826 × 10⁻⁸**. A vacuum control propagated the same executed incident field
through the same downstream optics and lacks the distinct surrounding Si
spots. Neither readout was renormalised to conceal numerical bandwidth loss.

The real `CoherentBeamPage` accepted the published final checkpoint and
displayed all eight spots in its normal logarithmic **128 × 128-bin** readout.
Offscreen Qt screenshots have readable verified fonts; no synthetic intensity,
markers or extra propagation were injected. Native-desktop calculation,
real GPU validation and a new installed-wheel check are **NOT_RUN** for this
update. Earlier installation or scope results below are historical evidence,
not a claim that they were rerun on this source.

Targeted regressions passed with one numerical thread: **53 cases in 12.42 s**
across wave-domain planning, material-grid refinement and Galerkin potential
(budget boundaries/cache isolation, real small Si complex-state equality,
upward-only grids, FFT phase/norm and CPU caps); then **28 cases in 17.41 s**
across Galerkin specimen and material refinement (independent dense
carrier-conjugated operators/projectors, sampled-path rejection, real Si,
cache/cancellation). These selections overlap and are not complete registered
validation scopes. The three actual incident modes also passed bounded
carrier-preserving zero-potential admission, with relative complex errors
below 5.52 × 10⁻¹⁶; that check alone is not specimen validation.

Local evidence is under ignored
`outputs/realistic-tip-optics-20261004/si-public-final/`: input/request records,
published checkpoint identities, ROI/vacuum comparisons and offscreen readout
receipts. Generated arrays and images are not part of this lightweight record.
Changes remain local and uncommitted. Planar driven emission is not metal
tunnelling; approximate near-tip transverse fields, paraxial continuation after
the executed prefix, current/brightness calibration and complete microscope
convergence remain unqualified. The bounded near-tip refinement comparison
does not establish convergence of the small reflected fraction.

The remaining sections preserve the separate **3 October 2026** large-source
demonstration and its historical checks.

## Result and scope

A completed development calculation from the physical tip boundary produced
periodic Si diffraction peaks at **Z = 1605.1939574922505 mm**. A vacuum control
with the same source, hardware and upstream numerical inputs produced only the
direct beam in its retained computational domain. This is evidence for the
declared coherent development example, not full TEM/STEM qualification.

The source was a single spatial and energy mode at the tip. Extraction,
acceleration, installed lenses, stigmators, deflectors, multipole corrections,
apertures and column walls remained in the executed path. No specimen-plane or
gun-exit source was supplied. Recording detectors were explicitly retracted to
observe a nonabsorbing virtual screen; this does not disable absorption when a
physical detector is inserted. The energy filter was not installed.

The starting repository HEAD was `14640259ea44dd8ac3b9bde675685591e5b9aa96`
on `master`, with pre-existing uncommitted development work. Input snapshots and
executed checkpoint identities, rather than that HEAD alone, identify these
calculations. Generated arrays, caches, screenshots and full receipts remain
local under ignored output or temporary directories.

## Fixed inputs and numerical comparison

The imported structure is the user's **Fm-3m monatomic FCC Si**, cell parameter
3.82166108 Å, oriented [001] along Z and [100] along X. It was not replaced by
diamond Fd-3m Si. The finite specimen is a 10 nm diameter disk, 5 nm thick, with
static atoms and explicitly zero inelastic material rates in this example.
One material trajectory is exact for those zero rates; it is not a trajectory
convergence demonstration for nonzero inelastic scattering.

The source and request are described in [COHERENT_BEAM.md](COHERENT_BEAM.md).
The Gaussian intensity FWHM is 28390.10000542304 nm, initial kinetic energy is
30 eV, and the two diagonal wavefront curvatures are approximately
624.3690658100246 m⁻¹. These are idealised planar-cathode emission boundary
parameters, not a prediction of a nanometre metal FEG tip. The source energy is
accelerated by the captured electrodes to approximately 300.030 keV.

| Run | Requested sample grid | Slice thickness | Actual potential grid | Final complex grid | Wall time, one CPU worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Si coarse | 192 | 2 Å | 1268 × 1268 | 1268 × 1268 | 301.324 s |
| Si finer | 256 | 1 Å | 1688 × 1688 | 3376 × 3376 | 660.212 s |
| Vacuum | 256, unused material request | 1 Å, unused | Not applied | 1024 × 1024 | 164.536 s |

The three runs used the same source grid 128, one energy sample, maximum gun
step 0.2 mm, fractional energy-change limit 0.00025, column step 1 mm, maximum
grid 4096 and working-memory limit 8 GiB. Automatic grid changes were recorded,
not hidden reductions of enabled physics. These times include actual cold
execution and checkpoint handling in the local environment; they are not a
portable performance guarantee.

Coarse and finer Si calculations have **exactly equal specimen-entrance complex
states**: amplitude and coordinate/carrier arrays, weights, energy, flight time
and action phase reference match. Their native complex-state difference is
zero. The comparison therefore isolates the joint material grid/slice change;
it does not establish source, gun, column, mode or full-grid convergence.

## Independent peak check and probability ledger

The comparison rebinds neither sources nor checkpoints. Each completed result,
snapshot and checkpoint was checked against its recorded identity. Predictions
use the actual CIF cell and orientation, accelerated energy, and the current
post-specimen quadratic optics matrix. The direct-beam origin comes from the
independent vacuum run. No Si peaks were used to fit a scale, rotation, cell,
source or lens strength.

All eight common 200/220 peaks are two-dimensional local maxima. On the shared
1024 × 1024 physical readout over ±110 µm, their maximum position discrepancy
from that independent prediction is **0.07150 µm**. Coarse and finer peak bins
are equal; this is limited by the readout bin width, not proof of zero continuous
position error.

Fixed 5 and 7 µm integration windows yield maximum coarse/finer intensity
differences of **4.522% for 200** and **10.504% for 220**, relative to the finer
result. Peak identity and location are established for this example, while
quantitative intensities remain incompletely converged. Vacuum has no positive
local maximum in these windows in its retained finite domain; an infinite
physical contrast ratio is not inferred.

| Run | Full retained probability per tip-reference electron | Material numerical band loss | Probability outside the comparison ROI |
| --- | ---: | ---: | ---: |
| Si coarse | 0.9961501061973809 | 0.0038498938031946 | about 1.45 × 10⁻¹¹ |
| Si finer | 0.9969287413813909 | 0.0030712586173358 | about 0.00242 |
| Vacuum | 1.0000000000005593 | 0 | about 2.4 × 10⁻¹⁴ |

The common figure uses the same absolute intensity-density logarithmic colour
scale for all panels. Conservative physical-area display binning preserves the
retained complex waves. Neither specimen images nor vacuum are normalised to
their own maxima. Numerical band loss is distinct from physical absorption and
is not repaired by renormalisation. The unrepresented continuum outside each
finite wave domain is not qualified by the ROI ledger.

## Reproduction and boundaries

Local deliverables are in `outputs/coherent-si-diffraction-20261003/`: actual
figures, original input packages and reports, exact numerical requests, and an
explicit input-only reproduction entry point. Input packages are designs for
new execution; they are not completed downstream beam states. The package's
wave request is not automatically applied by the GUI. Manual control entry and
GUI decimal precision support qualitative reproduction, while the API driver
retains exact serialized input values. Original dependency digests and request
provenance remain available when preparing a portable input design.

This example does not establish metal tunnelling, space/image charge, source
brightness/current calibration, frozen-phonon or nonzero-inelastic ensemble
convergence, all arbitrary Z positions, specimen-interior observation, or energy
filter transport. A source-256 / column-step-0.5 mm trial was explicitly cancelled
because of its cost and is not accepted convergence evidence. CPU and offscreen
Qt checks do not establish real GPU consistency or native desktop experience.

Software regression and isolated-install receipts are recorded separately from
the physical result. Their PASS status must not be promoted to full-instrument
qualification.

## Actual software and installation checks

The local checks used Windows, Python 3.12.3, PySide6 6.8.3 and the existing
locked CPU dependency environment. The registered runner was invoked as
`python scripts/validate_classical_scope.py --scope NAME --output FRESH_DIRECTORY
--timeout-seconds 1800`, with one numerical worker, one BLAS/Numba/OpenMP thread
and offscreen Qt. Complete selections were executed without filters.

| Complete scope | Passed cases | Failures / errors / skipped | Test time |
| --- | ---: | --- | ---: |
| classical | 295 | 0 / 0 / 0 | 211.23 s |
| gun-fields | 122 | 0 / 0 / 0 | 64.54 s |
| electron-execution | 488 | 0 / 0 / 0 | 86.24 s |
| field-ui | 560 | 0 / 0 / 0 | 166.43 s |
| particle-continuation | 126 | 0 / 0 / 0 | 217.65 s |
| performance-observation | 88 | 0 / 0 / 0 | 13.38 s |
| acceptance-policy | 107 | 0 / 0 / 0 | 6.23 s |
| coherent-development, full rerun | 312 | 0 / 0 / 0 | 256.21 s |

These total 2098 test executions, including cases selected by more than one
scope; they are not 2098 unique tests. Every scope returned exit code 0 and
`PASS`; each retained its `UNQUALIFIED` full-simulator status. Simulated GPU
policy fixtures are distinct from real GPU scientific validation.

The initial coherent scope recorded 309 passes and two failures. Its continuation
fixture used a 64-cell, 1 µm-step Gaussian of intensity sigma 6 µm; the nonzero
periodic boundary produced about 4.63 × 10⁻¹⁰ probability above the 0.8π guard,
exceeding its unchanged 10⁻¹² check tail. The corresponding continuous Gaussian
and its 0.2 mm drift at 300 keV have negligible bandwidth at that boundary.
Changing **only the test window to 128 cells**, with spacing, sigma, weight,
phase, origin and all original continuation assertions held fixed, puts that
artificial high-frequency probability below 5 × 10⁻²⁸. A new regression verifies
that the old subdomain is rejected and the expanded fixture is admitted; no
runtime guard or comparison tolerance was relaxed. The initial failed scope and
the corrected complete rerun remain separate records.

The seven other scopes' runtime/configuration/acceptance sources and selected
test files were checked byte-for-byte against their original receipts. Their
only changed globally fingerprinted file was this **unselected** coherent test
fixture. The full coherent selection was rerun on the final fixture. The solver
used by the actual Si/vacuum calculations was unchanged by this test correction.

A fresh wheel was built in an empty output directory, installed non-editably
in a new virtual environment using the locked dependencies, and passed `pip
check`. The existing installation smoke ran from outside the source checkout
with the installed interpreter's `-I` flag and passed: actual tip-origin classical
column, independent diagnostic fixture, persistence and offscreen panels.
A supplementary installed Coherent beam page import/open/close check passed
without starting wave propagation or changing the physical state. It is passive
installation evidence, not an installed-wheel coherent numerical run.

All three new portable input designs passed read/restore/request/source preflight
without propagation. They retain the original input/result provenance and exact
wave requests; original completed caches were not relabelled or supplied as new
sources. The separate reproduction driver's propagation command was not run
again after preparing those portable inputs. Actual numerical evidence remains
the three completed calculations above.

The local `validation-summary.json`, XML receipts, original failure, input
preflight receipts and installation reports accompany the ignored deliverables.
No generated arrays or caches have been staged or committed. Changes remain
uncommitted; no GitHub run or native-desktop/GPU/experimental acceptance is
claimed for this local result.
