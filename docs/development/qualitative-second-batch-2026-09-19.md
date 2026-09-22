# Second qualitative-science batch

## Scope

Stigmators, quadrupole/hexapole correctors and scan/descan controls now have
specific units and causal explanations in the existing parameter editor. The
scan panel uses the same registry. Descriptions state coordinates, held conditions,
compatibility-only scalars and model limits; they do not import commercial drive
tables or guarantee that increasing a setting improves an image. Existing
operating defaults, physical source/column structure and sweep permissions remain.

This is classical qualitative validation. No coherent tip calculation was resumed,
no independently configurable downstream source was added, and vacuum stays
opt-in. The energy filter has not been newly integrated or qualified.

## Reproduced defects and changes

1. `strength_m2` and `strength_m3` were displayed without their inverse-length
   units, although the corresponding maximum fields already had units. Both
   actual controls now show m^-2 and m^-3. Hexapole controls explain triple-angle
   orientation and inactivity in ideal mode. Stigmator explanations distinguish
   a pure-channel sign reversal from reversing one component of a mixed tensor.
2. First-order matrices used float32 drawing histories. An independent field-free
   drift from 500 to 600.1234 mm had a matrix error of 1.99403757e-9 m/rad.
   Requested planes now use executed float64 checkpoints.
3. Scan observers previously divided one finite positive test-ray displacement
   by a 1 microradian kick while leaving nonlinear fields on. A centred hexapole
   (1e6 m^-3, 6 mm Gaussian FWHM) produced a spurious first-order cross response
   of 7.96216554e-7 m/rad, where its on-axis derivative is zero. Scan, pivot,
   descan and plane classification now share the same linear observer. The
   actual finite-particle propagator continues to execute nonlinear fields.
   Analytic multipoles use the column-axis linearisation; mapped fields retain
   their central-difference local affine response. This does not qualify a large
   off-axis nonlinear scan or calibrate corrector feed-down around such an orbit.
4. A filter boundary guarded explicit descan targets but not arbitrary scan
   response requests or secondary preview planes. These requests now obey the
   same boundary. Pre-filter previews remain available; unsupported planes are
   reported, and downstream playback response is unavailable (NaN), not a zero
   or unfiltered continuation. This is a boundary correction, not filter physics.
5. `FIRST_ORDER_RESPONSE_SCHEMA = axis-linear-float64-checkpoints-v2` enters
   existing calculation identities, including dependent physical scan results.
   Historical results and held calibration records remain readable. Held ratios
   are preserved until explicit recalibration; an old result is not silently
   admitted under the new response model.

## Independent evidence and limits

- A field-free drift and centred hexapole's zero on-axis first derivative are
  analytical reference cases. These diagnostic bases are not active sources.
- A local isolated stigmator exchanges focusing/defocusing axes under a pure
  sign reversal; the second basis rotates its response by 45 degrees.
- A finite hexapole trajectory is compared with independent adaptive DOP853
  integration of `x'' = -H(z) x²`, with metre coordinates. Separate position and
  angular errors are checked at 1, 0.5 and 0.25 mm production steps. Tests also
  check the weak-displacement quadratic limit, 120-degree periodicity and the
  equivalence of 60-degree rotation to signed-strength reversal.
- A full executed classical tip → extraction → acceleration → apertures →
  installed column → specimen-entrance chain uses 49 emitted particles and
  independent stigmator channels at +/-0.1 percent. Energy offsets, ray IDs,
  weights and survivor masks are preserved across the comparisons. It tests
  local signed response and rank two, not global spot-size monotonicity.
- The initial 0.5 → 0.25 mm column refinement gave a 24.467 nm position
  difference, failing the declared 10 nm budget. The calculation was refined
  to 0.25 → 0.125 mm, where the angular difference was still 5.586 microradians
  against a 1 microradian budget. At 0.125 → 0.0625 mm the largest position and
  angular differences were 0.134 nm and 0.400 microradians. Neither budget was
  relaxed. Both local signed-response residuals were approximately 2.615e-6
  (dimensionless) and all 49 rays survived. Gun-step and source-sampling
  convergence are independent and are not established by this column test.
- Timing is checked at equal raster phase. A signed pair-gain change is checked
  with held command matrices. Existing regression verifies held lens/pivot
  errors, shared clock, FOV rescaling, physical targets and calibration rollback.
- Existing relay/corrector and CPU/Numba/CUDA tensor checks remain element or
  segment evidence; they do not certify a complete microscope or image.

## Execution receipt

Generated detailed receipts remain local under `tmp/qualitative-20260919`.

| Run | Result | Scope |
| --- | --- | --- |
| Classical software baseline | 259 passed, exit 0, source/input hashes unchanged during execution | Complete declared classical software lane; full simulator remains UNQUALIFIED |
| Initial response regression | 52 passed | First-order transfer, mapped fields, scan/held calibration and initial analytical reproducers |
| Control/tensor/corrector regression | 64 passed, 1 failed | The failure was the deliberately enforced column-step budget; refined in the final run, not relaxed |
| Final controls regression | 60 passed, no skips | Complete final qualitative-control, parameter/UI, held-scan, scan-system and ray-presentation files |
| Expanded dependency regression | Initially 88 passed, 6 failed, 1 skipped | Direct alignment, precalibration, cache reuse, detector orientation and corrector envelopes |
| Corrected cache file | All 36 cases passed | Entire cache-reuse file rerun after the independent-job fixture correction |
| Corrected alignment guard | 1 passed | Both wrong-mode and out-of-range requests checked after message-contract correction |

The baseline receipt is `classical-baseline/report.json` (run
`757d32b2cd13479a9c629a91dbfdafa0`). It passed before the final test-only refinement
from a 0.25 mm starting step to 0.125 mm. Source/input hash comparison afterward
confirmed that **no production source or configuration changed**. Later changes
also correct the two stale tests described below. The final 60-case run uses
the refined physical test. These counts overlap; they are not a summed
count of unique cases or one claim of a clean full-repository suite.

Two of the six expanded-regression failures were stale test contracts, also
reproduced on the unchanged archive: the mode/range validation messages had
moved to the shared transaction guard, and Preview no longer cancels an
independent High-accuracy job. The tests now check the current guard and preserve
the stronger safety assertions: rejected requests leave lens settings unchanged,
Preview completion cannot clear High's ownership, identical High requests
deduplicate, and High completion clears only its own token. No production
controller or alignment code was changed. The skipped test already declares
that its stored aperture-scaling metrics predate the aperture relocation.

Across the expanded run and the explicit corrections, 90 of its 95 cases have
passing evidence, four absolute-target cases remain failing, and one remains
explicitly skipped. This is not a single clean 95-case rerun. Final `compileall`
over source/tests/scripts and `git diff --check` passed. Existing Pydantic
deprecation and small CUDA-grid occupancy warnings do not certify or disqualify
the full physical model. Native interactive GUI operation, full-image science,
GPU performance and the complete filter-reaching chain remain unqualified.

### Existing absolute-target failures

The expanded direct-alignment regression exposed four historical absolute-target
assertions. All four were independently reproduced in the archived, unchanged
`7c52fa0593dfe73b36bcb12c042f55ba09f3334d` implementation. The archived transfer,
observer and gun-transport files were checked against HEAD. This reproduction
did not use the new observer or gun-energy fix from either qualitative batch.

| Historical request | Archived achieved value | Result |
| --- | --- | --- |
| 60 mrad, 240 micrometre condenser aperture | 51.22657 mrad | Solver returned unsuccessful |
| 1.5 micrometre illumination diameter | 1.104680 micrometres | Solver returned unsuccessful |
| 2.0 micrometre illumination diameter | 1.104691 micrometres | Solver returned unsuccessful |
| 2.2 micrometre illumination diameter | 1.104694 micrometres | Solver returned unsuccessful |

These are unresolved historical target/preset qualifications, not passing evidence
and not proof of global mathematical unreachability. No target, source,
mechanical geometry, current table or assertion was altered to force success.
The next alignment-focused work should qualify feasible local operating ranges
and explicit rejected targets for the current source, rather than promise all
historical absolute values. The current user scope requires qualitative trends,
not matching those numerical presets.
