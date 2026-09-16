# Assembly illumination calibration (in progress)

## Requested defaults

- Nanoprobe: current-weighted alpha95 = 30 mrad at the specimen **upper surface**.
- With a probe corrector: geometric D95 < 1 nm. Without one: nanometre-scale;
  a provisional 10 nm screening ceiling is used, not an atomic-resolution claim.
- Microprobe: current-weighted D95 = 2 um, alpha95 <= 0.3 mrad, with small
  radial position-angle covariance (quasi-parallel illumination).
- Preserve intermediate crossover count, optical-component intervals and order
  for the established FEG + C3 + Probe Corrector / no-blanker reference. The user
  authorised separate baselines for other assemblies on 2026-09-17.
- Specimen current: 2-200 pA. Physical C2 aperture diameter: 20-250 um.
- The user authorised the retained total-flux control in addition to physical
  aperture losses. Set it once on the fitted candidate and hold it fixed for
  independent validation. Do not renormalise the transmitted ray population.
- Execute classical tip emission, extraction, acceleration, every installed lens
  and every physical aperture. Coherent gun-wave development remains paused.

## Publication status

**No new preset has been installed.** Optimizer success is not qualification.
The historical operating-mode strengths remain unchanged. Their catalog status
and UI label now explicitly say stored reference / recalibration pending. The
old calibration metrics are not validation for the revised gun integrator.

The assembly catalog currently contains 18 pre-specimen combinations (three
gun variants, three condenser/corrector variants and two blanker choices),
represented in 60 complete assemblies. Image-corrector/recording variants must
not be aliased without establishing upstream equivalence for the active model.

## Implemented numerical work

1. Gun steps now inspect all active equal-time electrons, not only the leader.
   A trailing electron may still be inside an extraction or acceleration field.
2. Analytic particle transport bounds the local Lorentz impulse and compares
   one full Boris step with two half steps. Static-energy projection alone
   cannot establish transverse-trajectory accuracy.
3. The adaptive iteration allowance no longer assumes every step advances the
   full requested spatial increment. This also applies to thermionic and
   monochromated classical guns.
4. A compiled evaluator reproduces the existing compact polynomial potential
   and its derivatives. Custom electrode providers use the original fallback.
5. Detached calibration reuses an executed, private upstream checkpoint when
   only downstream lens excitations vary. Full-path fine-step validation is
   still required. This checkpoint is not a configurable downstream source.
6. Calibration budgets count actual physical evaluations, including numerical
   Jacobian probes. Failed searches preserve candidate diagnostics and never
   overwrite the user's operating state.

## Evidence so far

The FEG + C3 + Probe Corrector candidate obtained with 193 rays and a 120 um
C2 aperture reached alpha95 = 30 mrad and geometric D95 approximately 0.46 nm.
Its intermediate crossovers remain in the original five component intervals:

1. C1 to C2.
2. C2 to C2 aperture.
3. Probe DP22 to HPC.
4. Probe TL12 to condenser stigmator.
5. Mini condenser to objective/sample.

Actual root refinement at 2048 rays confirmed these intervals and a terminal
waist within 1 nm of the upper surface. However, unchanged candidate settings
gave alpha95 = 30.524 mrad at 1024 rays and 29.531 mrad at 2048 rays. This fails
the 1% angular criterion and sampling-convergence gate. It is **not** a default.

The initial C2/C3 searches also failed at least one illumination criterion.
A failed bounded search does not prove physical impossibility. More complete
branch exploration and statistical convergence are still needed.

## Reproducible tools

`scripts/calibrate_assembly_illumination.py` generates scalar candidate reports
from the current assembled physical model. Use `--help` for explicit assembly,
mode, ray budget, lens controls and evaluation budget. `--upstream-only` explores
the 18 upstream selections; `--all` enumerates all 60 complete assemblies.

`scripts/check_illumination_candidate.py` re-executes a candidate at independently
specified ray counts and two column steps. `--topology` also refines crossover
roots with real trajectory checkpoints.

Reports explicitly say `NOT_QUALIFIED` / `NOT_INSTALLED`. Neither script enables
coherent waves, calculates specimen images, edits assembly geometry, installs
presets, writes array caches, changes the running GUI, or uploads anything.

`assembly-illumination-evidence-20260916.json` retains selected scalar results
and candidate strengths. It contains no ray arrays or calculation caches.
Some early reports predate the final solver-source identity field; they remain
historical evidence, not an installable bank. The tools now record the current
implementation and reject source edits during execution.

The later 4096-ray corrected FEG fit reaches 30.000 mrad, geometric D95
0.427 nm and surface-waist offset -0.032 nm (0.025 mm column step). Independent
2048-ray re-execution gives 30.268 mrad, D95 0.452 nm and offset +0.105 nm.
Its five intermediate crossover intervals still match the reference.

The paired 4096-ray microprobe fit reaches 2.000 um and 0.2213 mrad. At 2048
rays, D95 is 1.9565 um: the 1% diameter/sampling gate is **not passed**, even
though angular spread and radial curvature satisfy their limits. These are
reported candidates, not an installed or image-validated default pair.

At 512 emitted FEG rays, bounded branch exploration also found fixed-budget
microprobe candidates for bare C2 (C1/objective controls) and C3 columns, and a
bare-C3 nanoprobe candidate (C3/objective controls). These have not passed the
independent sampling/topology gates. The C2 nanoprobe searches, including a
170 um physical pupil and C1/C2/objective trials, did not reach 30 mrad in the
searched branches. This is not a proof of infeasibility.

For the thermionic + probe-corrector assembly, the inherited FEG condenser
seed gives no surviving nanoprobe rays at the surface at 512 emitted rays.
The microprobe trial has fewer than 16 transmitted rays. Those searches are
under-resolved, not accepted presets and not evidence that a physical thermal
gun cannot be focused. They need a distinct upstream condenser branch and
adequate transmitted-current sampling.

Failed trajectory executions now consume the same evaluation budget as
successful ones. Repeating an already failed vector does not run it again.
Nanoprobe branch searches solve physical objective focus before testing the
angular target, and an oversized corrected spot cannot terminate the search.

## Validation in this round

- Final focused run: 35 passed (`test_assembly_illumination`,
  `test_analytic_gun_field`, `test_gun_adaptive_step`,
  `test_illumination_checkpoint`, and the operating-mode storage test).
  GUI checks use Qt offscreen, not a visible desktop acceptance test.
- Existing `test_operating_modes` particle-path checks for both probe modes:
  2 passed. These retain their original broad historical assertions; they do
  not certify the new 30 mrad / upper-surface / sub-nanometre targets.
- Earlier curvature/path/checkpoint regression group: 53 passed. Actual
  monochromated and thermionic aperture-contract tests: 2 passed.
- The full suite, specimen TEM/STEM images and hardware were not tested.

No calibration job is left running. No new default strengths, generated array
caches, Git publication or shutdown are part of this handoff.

## Remaining acceptance work

- The corrector-dependent diameter gate is now enforced in the candidate report.
- Refine and validate source sampling, gun step and column step independently.
- Complete both illumination modes for each supported assembly or state why a
  combination remains unqualified; do not silently loosen limits.
- Bind qualified values to source/model/geometry inputs and integrate separate
  upstream preset selection without overwriting downstream projector settings.
- Validate TEM/STEM image production separately: a geometric probe diameter is
  not a wave-probe FWHM or proof of image resolution.

The immediate next numerical task is a source-specific upstream condenser
search for C2, monochromated FEG and thermionic assemblies, with adequate
transmitted-ray support. Repeating the same FEG mini/objective seed at a low
particle budget is not a useful continuation. The corrected FEG microprobe
also needs an independent larger-sample check before its diameter is certified.

## 2026-09-17 continuation

The default assembled state uses **Custom / Per-lens Models**, not Ideal Optics.
An earlier verbal description and one interrupted report label said Ideal;
that label has been corrected without changing the measured numerical results.
New reports record the actual simulation-mode key explicitly.

The current and C2-pupil limits are now authoritative TOML settings. The
calibration selects a 100 pA interior setpoint when physically available, or
the available current if between 2 and 100 pA. It never amplifies above 100%
source flux. This uses the existing `column_current_limit_percent` field, not
a new downstream source or a change to tip size, angles or energy. The retired
Direct Alignment spot-number inverse control is not re-enabled. This scalar
flux ceiling is not a model of C1-dependent source brightness or space charge.

Each report retains physical source-to-specimen transmission, explicit source
ceiling, effective source current and specimen current separately. Lens values
and the flux setting are frozen across sampling, column-step and gun-step
checks. The snapshot digest includes the flux setting. Current/readout changes
continue to retain geometry caches while invalidating count products.

Expanded branch searches found a 512-ray FEG + C2 nanoprobe candidate with a
250 um pupil: alpha95 30.258 mrad, geometric D95 0.263 nm and surface-waist offset
-0.036 nm. This is a fixed-budget candidate, not a published default. Earlier
failure at smaller pupils was not a proof that this assembly was impossible.

At 1024 emitted rays, source-specific searches found monochromated-FEG
candidates for C2 microprobe (2.000 um / 0.200 mrad) and C3 nanoprobe
(30.000 mrad / 1.642 nm). Thermionic searches remain unqualified: a 30 mrad
candidate still has micrometre-scale D95. Source dimensions and distributions
have not been reduced to force an apparent success.

`validate_assembly_illumination.py` now checks pupil/current gates and explicit
relative changes in angle, diameter and transmission, rather than accepting
two independent target checks alone. No result is published just because an
optimizer converged. `check_illumination_equivalence.py` compares the executed
gun inputs, complete analytic column plans, walls and apertures at both column
steps. Equality permits reuse of an already executed upstream result only;
it does not certify downstream images or bypass any active component.
