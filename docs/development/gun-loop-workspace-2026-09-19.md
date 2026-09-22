# Classical gun loop workspace optimization — 2026-09-19

This round reduces repeated preparation and array copying in the complete
classical analytic gun calculation. It follows the earlier
[retry/history optimization](gun-performance-2026-09-19.md). The CPU limit
remains half of the logical CPUs available to the process, with serial BLAS.

## Changes and physical scope

The gun tracer now owns a workspace for one executed tip-to-exit calculation.
It reuses active-index, error-status, trial-half-step, impulse-limit and output
buffers. Current field-provider identity and every consumed field parameter
are checked at each outer step. Changed parameters rebuild their packed
representation; custom, overridden or unsupported providers use the existing
general path. The workspace is neither a persistent beam state nor a source.

Returned particle states still own independent float64 position and momentum
arrays. This allows the tracer to retain the previous state's arrays for
crossing interpolation and use the advanced state directly, avoiding an extra
full copy and reconstruction. History snapshots, vacuum segments and mutable
survival masks keep the independent copies that their consumers need.

The outer energy projection uses the prepared field representation when
supported. Its axial evaluator, radial potential correction, momentum
conversion and NumPy operation order are unchanged. Both the integrator's
energy projection and this existing outer projection still execute.

No extraction, acceleration, focusing, alignment, aperture or physical stop is
removed. Spatial and relative-impulse caps, adaptive rejection, the full-step
versus two-half-step error budget, maximum attempts, float64 state and history
sampling remain unchanged. The curved-tip/general-provider paths retain their
existing algorithms. Coherent-wave development remains paused.

The persisted restart identity continues to include production solver source.
An archive from an older source version remains readable, but changed solver
identity prevents silently treating it as a current executed prefix. New
compatible section archives retain the existing upstream-reuse workflow.

## Controlled complete-gun measurements

The unchanged 5,000-particle flat-tip input from the previous gun benchmark
was executed without an upstream-result cache. Both runs used 16 numerical
threads on this 32-logical-CPU host, serial BLAS, the same Python environment,
and already warmed compiled kernels. No other task-owned numerical job ran
during these two measurements. They ran serially in separate processes.

| Implementation | Complete gun wall time | Kernel warmup, separately | Sampled peak process RSS |
| --- | ---: | ---: | ---: |
| Before this round | 95.836 s | 0.212 s | 1.060 GB |
| After this round | 79.792 s | 0.252 s | 1.084 GB |

The paired wall-time reduction is **16.74%**, or **1.20 times throughput**.
This is one controlled pair, not a statistical scaling study or a cold-JIT
measurement. The earlier report's 85.75 s sixteen-thread result was a separate
run and is not the baseline used for this round's percentage. RSS includes
Python/JIT overhead; this round does not demonstrate a memory reduction.

Both completed runs delivered 5,000 surviving rays with a 3,536 by 5,000
equal-time history. Their complete-output digest agrees with each other and
the earlier optimization's baseline:

`a3b1841c60f657aba4b0daf57e57912e4dbffedcd44388650972440026b0c24c`

The digest covers the benchmark's trajectory and clock arrays, equal-time
history, exit phase space, identities, weights, survival and stop labels.
Separate instrumented runs accepted the same **35,336** outer steps.
This is a complete gun measurement; it is not the time for a complete
microscope, specimen interaction, STEM frame or GUI request.

Nested diagnostic timings attribute the largest preparation improvement to
the step bound: its exclusive time fell from 16.69 s to 4.25 s. The prepared
compiled bound now also reduces and validates its reused buffers internally.
Adaptive integration remains the main cost (40.76 s in the new diagnostic),
followed by the outer loop (14.87 s) and the retained energy projection
(12.50 s). The field-parameter pack was built once rather than 70,672 times;
current field checks still run on each step. These diagnostic totals contain
instrumentation overhead and the new profile's early phase may overlap a
short test run, so they are not used for the speed percentage above.

The profiling harness initially added the nested prepared-step and outer-step
counters, producing an erroneous 70,672-step mismatch. The receipt preserves
that initial status and its correction to 35,336; the output digest already
matched throughout. No production solver change was needed for this harness
metadata correction.

## Validation and reproducibility

The new workspace's 28 dedicated tests cover serial/parallel exact parity,
successive steps and changing active masks, field-value and provider changes,
repacking, invalid particles, empty activity, unsupported/disabled providers,
noncanonical dtypes, optional acceleration and exact energy projection.
Five ownership cases additionally check independent returned states, readonly
inputs, the general surface step and bounded cancellation with retained
history. The vacuum ownership cases explicitly inject deterministic stops;
they are not statistical validation of vacuum scattering.

The final integrated run completed **223 passed, zero failed/error/skipped**
in **379.272 s** (`final-regression.xml`), with 16 existing Pydantic
deprecation warnings. It includes the gun reference/compiled comparison,
adaptive stepping, physical apertures, curved-tip and monochromator
compatibility, voltage/current/TOF, history, cancellation, numerical cache
identity, CPU budgeting, section/controller/archive workflows, material
checkpoint continuation and persistent clock validation.

Real small-source fixtures verify nine-particle complete-pipeline archive
round trips and 49-particle executed-section save/load/extension. The tests
make a repeated gun execution fail explicitly when a matching saved prefix
should be reused. Changed lens settings, malformed clocks and reduced-precision
checkpoints exercise invalidation. Bounded artificial short-column/material
fixtures also participate; passing this selection is not full-instrument or
experimental qualification, and the entire repository suite was not rerun.

A separate process blocked Numba imports and successfully executed five
general-path steps from a real nine-particle tip launch before requested
cancellation (`no-numba-smoke.json`). This establishes import/fallback operation
for that bounded case, not a complete uncompiled gun performance result.
`compileall` and the tracked diff whitespace check passed after the tests.
The measured production source hashes were verified again unchanged.

Measured production SHA-256:

- `analytic_particle_step.py`: `0e0ba516243cd0e045744212d8fc0309fbea896866b19a0af386a238331a07c7`
- `tracing.py`: `bf46762fedd1e9e972db2a8a3024ed0268a5ea9e0ceba6cc51b6f008aa4b38ec`

Before/after receipts bind their executed source snapshots and verify that
the final production sources and input settings did not change during the
measurements. The frozen before-files reproduce the prior implementation
without changing the live checkout.

Local evidence: `tmp/gun-loop-20260919/comparison-5000-16.json`, the four
referenced profile/plain receipts, frozen before-files, `profile_worker.py`,
`prepared-final-tests.xml`, and the final regression receipt. Temporary
scripts, numerical arrays and receipts remain local; no generated calculation
data are committed. No application restart, Git commit or push was performed.
