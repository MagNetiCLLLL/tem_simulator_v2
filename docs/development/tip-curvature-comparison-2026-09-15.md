# Matched-D95 tip curvature comparison

## Outcome

The reduced-curvature input design is available, but **no replacement optical
preset is qualified or installed**. The existing Flat default and all published
profiles remain unchanged by this comparison. Coherent propagation is still
paused. Every calculation here used classical emission, extraction, the gun
lens, acceleration and physical apertures/walls.

The emitting-region projected D95 is **10.05689406165185 nm** in both models.
This is the 95%-current radial diameter of the Flat 5 nm FWHM Gaussian truncated
at 3 sigma, not its FWHM or full edge diameter.

| Curvature radius | Emitting-cap half-angle | Projected D95 |
| --- | --- | --- |
| 100 nm | 2.9571973691 degrees | 10.0568940617 nm |
| 200 nm | 1.4781247257 degrees | 10.0568940617 nm |
| 500 nm | 0.5911968652 degrees | 10.0568940617 nm |

R=500 nm has one fifth of the curvature of R=100 nm. The cone half-angle remains
5 degrees. The comparison changes the actual tip part and runtime geometry
together, not just a drawing or source label. It leaves local emission energy,
angular law, electrode positions and voltages unchanged.

## Accelerator comparison

All cases start from the existing Flat Nanoprobe + Diffraction settings:
4.0 kV extraction, gun lens 1.2 kV **relative to the extractor**, 300 kV final
acceleration. The accelerator spans Z=30 to 370 mm. Ten electrode-stage planes
are recorded in each JSON report, together with RMS radius, D95, maximum radius
from the axis, local bore, current fraction and absolute current.

At 193 particles, gun trace step 0.2 mm, history step 0.2 mm:

| Model | Entrance D95 | Exit D95 | Gun transmitted current fraction |
| --- | --- | --- | --- |
| Flat | 0.941 micrometres | 4.974 micrometres | 100% |
| Curved R100 | 1.650 mm | 1.789 mm | 100% |
| Curved R200 | 1.172 mm | 1.283 mm | 100% |
| Curved R500 | 0.860 mm | 0.944 mm | 100% |

R500 at 1,000 particles gives 0.847 mm entrance and 0.930 mm exit D95. Its exit
D95 changes by about 1.5% from 193 particles. Refining electrode cells/bore from
8 to 16 and trace/history steps to 0.1 mm at 193 particles gives 0.945 mm exit
D95 (about 0.15% change). This is a limited refinement check, not full field or
statistical certification.

**D95 matching does not make the two sources physically identical.** Flat uses
a truncated Gaussian spatial law, its prescribed narrow angular law and the
legacy analytic gun field. Curved uses uniform emitting surface area, the
normal/tangential energy law and the grounded-electrode field. Curved-to-curved
comparisons keep those laws fixed; Flat-to-curved differences cannot all be
attributed to curvature.

The curved recipe retains its prescribed surface flux density. Its smaller cap
therefore emits about **8.760 nA**, rather than the original wide cap's 100 nA.
Flat retains 10,000 nA. These are independent existing prescriptions, not an
equal-current comparison; no normalization was used to hide the difference.
Space charge remains neglected by the existing grounded-field model.

## Preset exploration: not accepted

Directly using Flat lens values with the matched curved sources produced no
sample survivors in the 193-particle diagnostic. Some already-lost trajectories
overflowed during downstream integration; those runs are not focus acceptance.

Bounded first-order proposals were followed by actual tip-origin transport.
The best continued R500 trial used these excitation percentages:

| Lens | Flat starting value | Exploratory value |
| --- | --- | --- |
| C1 | 51.6666666667 | 49.3534546415 |
| C2 | 25.2577271725 | 11.8614833948 |
| C3 | 21.2736229083 | 34.9192996232 |
| Objective | 68.9801000000 | 68.4198579965 |

At 1,000 particles it delivered 5% of source current to the sample upper surface
(Z=1599.1999975 mm). RMS radius was approximately **184 nm**, D95 **714 nm** and
alpha95 **162.5 mrad**, not 30 mrad. Estimated local waist offset was -1.40 nm at
0.05 mm column step and -1.43 nm at 0.025 mm. Solver termination did not pass the
physical acceptance gate.

The 193-particle topology audit also rejected the trial: there are still five
incident crossover candidates, but one moved from downstream of TL12 to the
DP11--TL12 interval. This violates the required component ordering. The gun and
post-specimen crossover chains and downstream diffraction were not qualified.

## Additional blocker: Flat gun step dependence

The Flat gun calculation has a significant numerical-step dependence. At fixed
193 particles, accelerator exit D95 was:

| Gun step | Exit D95 |
| --- | --- |
| 0.2 mm | 4.974 micrometres |
| 0.1 mm | 3.277 micrometres |
| 0.05 mm | 2.971 micrometres |
| 0.025 mm | 3.046 micrometres |
| 0.0125 mm | 3.247 micrometres |

The smallest steps are not yet converged. Do not use the Flat numbers as a
certified calibration reference, or present the ratio between Flat and curved
sizes as a precise physical measurement. The order-of-magnitude difference
remains, but the Flat integrator requires investigation before final matching.
No production integration defaults were changed in this work.

## Loadable input designs

In **Working Points**, choose **Import...**, then explicitly **Restore working
point**. Restoration loads a complete instrument setup; save any current user
settings first. These packages contain only `manifest.json`, not rays, images,
wave arrays or calculation caches. All results need fresh execution.

- `configs/design_candidates/matched_tip_d95_20260915_flat.temwp`: unchanged
  Flat preset reference inputs.
- `configs/design_candidates/matched_tip_d95_20260915_r500.temwp`: R500 matched
  D95 source with the original Flat lens values, deliberately not qualified.
- `docs/development/tip_curvature_candidate_audit_20260915.temwp`: rejected
  exploratory lens trial, retained for reproducibility, not routine use.

## Reproduce and continue

Use the project virtual environment. No script installs operating settings.

```powershell
.venv\Scripts\python.exe scripts/compare_tip_curvature.py --rays 193 --column --output tmp/curvature-new.json
.venv\Scripts\python.exe scripts/compare_tip_curvature.py --rays 1000 --radii-nm 500 --output tmp/curvature-1000-new.json
.venv\Scripts\python.exe scripts/fit_matched_tip_preset.py --reference docs/development/tip_curvature_193_20260915.json --output tmp/fit-new.json
```

Outputs must be new files. Scalar evidence is in `tip_curvature_*.json` and
`tip_flat_step_*_20260915.json` beside this report. Geometry matching, detached
state preservation, input-package restoration and cache exclusion have focused
regression coverage. These software tests are not physical focus qualification.

Validation: 69 focused tests passed across curvature matching, Flat defaults,
gun design input packages, gun matching and surface-focus gates. The four new
Python files compiled successfully. Unrelated suites and TEM/STEM wave imaging
were not run.

Next: resolve Flat gun step convergence, then fit the condenser/corrector/objective
chain with explicit crossover-interval constraints, independently checking
particle sampling, field resolution, the upper-surface spot and alpha95. Only
then requalify downstream diffraction and consider publishing a new preset.
