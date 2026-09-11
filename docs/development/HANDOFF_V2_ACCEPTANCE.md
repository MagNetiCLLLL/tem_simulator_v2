# HANDOFF v2 implementation and acceptance ledger

Authoritative request: user-supplied `HANDOFF.md`, design dated 2026-09-10,
baseline `5eac9855ff8eefa2a3d68ddba3029f7c23e03b0c`.
This ledger supplements, and does not replace, the original design.

## Current user constraint: physical tip origin (2026-09-11)

The user withdrew permission for an independent effective exit source. This
requirement supersedes the effective-source authorization described in the
historical implementation notes below. Electrons must originate at the FEG
tip and undergo the existing extraction, acceleration, focusing, deflection
and aperture operations. Changes to numerical methods may not remove existing
capabilities. Equivalent states are only cached upstream computation results.

The exit-source UI/production paths are withdrawn. The source editor now edits
the existing tip emitter. Historical exit-source profiles cannot be activated,
and labels or matching bindings do not qualify them for calculation or reuse.
Historical Gaussian/column tests remain explicitly isolated numerical evidence.
They do not validate coherent propagation through the physical gun.

The tip mutual-intensity model and a development variable-energy quadratic
gun phase solver are now implemented. The development API also
connects axial column-field tails inside the gun, distributed multipole phase,
finite-specimen propagation and selectable detector complex/phase readout.
Physical AC/Descan coils now use per-energy axial arrival times; conditional
material-model inelastic histories propagate at their changed energy. Column
segments and specimen trajectory slices stream through executed disk caches.
General-angle/imported-field physics, longitudinal packets, atomic transition
potentials and the installed post-column energy filter remain unfinished.
See [the current parameter and implementation inventory](TIP_TO_IMAGE_WAVE.md).
The full scientific release remains incomplete. Module tests and these
development connections do not close its release gates; the actual 128-grid
tip profile previously stopped at a column Cs sampling budget before the specimen.
The 96 GB host now has larger bounded work/cache settings, but this alone is not
evidence of successful full-grid or full-microscope convergence.

Working-point Restore/Continue/Undo now synchronizes assembly and backend
selectors with the captured state, including rollback on failure. Obsolete DA
error/validation-step expectations were corrected without changing tolerances.
The development acceptance report identifies missing prefixes and explicit
source-migration blockers. Installation smoke covers physical particles, a
separate Si multislice kernel and the GUI; it does not qualify the installed
production image case (AT-31 remains blocked).

## Required delivery order

1. Complete snapshot, immutable numeric products and observable definitions.
2. Explicit full-phase conversion and normalized canonical-map admission.
3. Gun-owned source ensemble and actual component propagation; shared TEM/STEM entrance.
4. Content-pinned working-point packages, conservative dependency-aware reuse.
5. Registered Direct Alignment candidates, forward validation, atomic apply/undo.
6. Read-only working-point browser, explicit restore/fork and unified entry gates.
7. Actual operating-point, specimen, CPU/GPU and regression evidence.

## Historical implementation before the tip-origin constraint

The user authorized a **new versioned effective gun source**, keeping the legacy
model, its parameters and historical results unchanged. The cold-FEG exit
Gaussian-Schell model now owns both positive-Wigner particle samples and weighted
coherent modes. Its reference plane is fixed by the installed gun exit. The
required exit current is explicit; no legacy current or low-energy launch width
is silently reinterpreted as a quantum source calibration.

Implemented additions include the source-model dialog and profile shelf,
quadratic shared-field propagation, full phase-bearing persistent/portable
checkpoints, separate canonical-wave diagnostics, exact incident-operator cache
reuse, complete-snapshot browser, and registered DA transactions with
cancel/stale/idempotence/undo protection. These are **partial implementations**,
not completion of NWP00–09.

The production TEM/STEM adapters are still missing. New images therefore remain
gated rather than receiving a reconstructed legacy specimen pupil. Affine
scalar action and continuous metaplectic lift now have bounded complex-field
tests; complete mixed-conjugacy/domain validation, portable external-loader remapping, actual
32/40 mrad operating-point studies, independent specimen reference comparisons,
GPU streaming/resume and complete-workflow qualification remain open.

`GUN_SOURCE_V1.md` describes use and limitations. `ACCEPTANCE_MATRIX.json` records
each T2 requirement, actual test IDs where available, run evidence and remaining
work. Tests of synthetic transactions or isolated waves are explicitly not
hardware/source-to-image validation. The broad regression currently has failures
in its original report and is not an all-pass result. Its stopped cases now
have scoped follow-up evidence; a complete-suite rerun remains outstanding.

## Latest defect audit

The latest scoped regression passed **141 tests in 188.48 s**. It covers
complete affine phase, continuous lift, incident-operator cache reuse,
same-path CIF replacement, immutable mode containers, source/package/CLI
contracts, read-only diagnostics, transactions and isolated camera/flux checks.
The source code was held unchanged throughout this run; no production inputs or tolerances
were adjusted to pass it. The report is retained under
`evidence/handoff-canonical-final-regression.xml`.

The old equivalent-projector sweep now explicitly selects its physical model
before alignment. Its **six original targets passed** (78.93 s, 38 deselected),
with unchanged tolerances and constraints. Three historical A5/frozen-phonon
algorithm tests also passed (13.42 s): they first verify production rejects
their synthetic source, then isolate the old numerical algorithm using a
test-local admission override. Neither group qualifies the new source-to-image
chain. The migration reasons and original baseline references are preserved in
`tests/historical/PROJECTOR_MODEL_SELECTION.md` and `WAVE_ENTRY_MIGRATION.md`.
Both run reports are retained under `evidence/`.

An older corruption-injection test initially returned the kernel's obsolete
single-value format after that internal kernel began returning wave plus chart
phase. Only its test injection was adapted; the physical lossless-norm guard
and tolerance were retained, and the 141-test run includes that guard. A mode
container ownership defect was also fixed: a frozen `BeamState` now owns a
tuple rather than retaining a mutable caller-supplied list.

Serial `compileall` over `src`, `scripts` and `tests` passed after these changes.
Test wall times above are not isolated cold/warm or hardware benchmarks.

The earlier scoped regression passed **189 tests
in 196.11 s** with no failures or skips. The report is
`.pytest_cache/handoff-current-regression.xml`; it includes the six existing
nanoprobe target cases, exact-aperture sampling, source admission, detached
controller/alignment state, upstream coherent transport, persistent/portable
checkpoints, phase contracts and read-only observables. This is a scoped
regression, not the complete historical suite or source-to-image acceptance.
Pyqtgraph emitted a teardown signal-disconnection warning after successful
completion. Serial `compileall` over `src`, `scripts` and `tests` also passed.

The original broad audit stopped at its declared 15-failure limit after 720 passing
tests (414.23 s). It did not run the entire suite. The six controller failures
were reconciled with the detached-graph and isolated progress/memory-test
contracts; the controller/model subset subsequently passed 62 tests. The three
legacy wave-entry expectations and projector auto-model-selection assumptions
were subsequently migrated as described above. The original report, including
all failures, is retained in `evidence/handoff-regression-audit.xml`. Scoped
follow-up passes are not a replacement for a fresh complete regression run.

The small-aperture nanoprobe failures exposed a real propagation inconsistency:
full-column clipping could select the nearest sparse drawing-history row,
whereas Direct Alignment retained the exact aperture plane. One reproduced
3 mrad request differed by 8.4 percent at final validation and correctly left
the complete starting snapshot unchanged. The shared propagation planner now
retains every active aperture's exact plane. Six existing nanoprobe target
cases then passed without changing their target or tolerance. A separate drift
test checks clipping independently; its coordinate comparison allows two
float32 storage ULPs because drawing histories use float32.

Projector search now follows the captured distributed/equivalent field-model
choice, not merely whether an equivalent model is geometrically supported.
Search-basis construction owns a scratch snapshot and no longer invalidates
the caller's objective reference planes. Forward-particle and observable
validation backends are reported separately. Process-local objective-cache
signatures and memoized equivalent calibrations are not physical model inputs.

Effective-source mode-count arithmetic handles small tails without subtractive
cancellation and refuses unrepresentable covariance or excessive mode budgets
before array allocation. The upstream coherent checkpoint now stops at the
finite specimen's upper face, not its centre; this is still upstream transport,
not a completed finite-specimen or detector adapter.

The incident cache now compares exact consumed operator inputs independently
of complete snapshot ownership. Replacing Si with Au at the same CIF path can
reuse the unaffected upper-face modes, while the new work point records the
new content and parent/executed snapshot links. Changing the face, thickness,
objective field, aperture or numerical step invalidates that operator. A
separate full cold execution matches the warm result's arrays and weights
exactly. Full portable restoration with missing model files is still not
implemented; no current file is silently substituted for archived content.

## Acceptance inventory

All statuses below require new evidence. Historical passes are not imported.

| Tests | Requirement | Status |
| --- | --- | --- |
| T2-01–03 | Production-source gates, historical migration, verified provenance | IN_PROGRESS |
| T2-04–08 | Complete immutable content-pinned snapshots and exact restoration | IN_PROGRESS |
| T2-09–12 | Registered read-only observations and lazy diagnostics | IN_PROGRESS |
| T2-13–17 | Phase carriers, canonical maps, composition, inverse and flux | PARTIAL; scoped numerical tests |
| T2-18–21 | Gun/energy/aperture/scan component chain | PARTIAL; upstream only |
| T2-22–26 | Stage reuse, shared fields, display/readout, new planes, integrity | PARTIAL |
| T2-27–32 | Allowed inverse targets, forward validation and transactions | PARTIAL; physical target acceptance open |
| T2-33–34 | Unified UI/CLI and read-only incompatible history | PARTIAL |
| T2-35–38 | Actual 32/40 mrad, specimen, GPU and model-domain admission | NOT_RUN |
| T2-39–40 | Non-conflicting regression and complete workflow | Regression in progress; final workflow incomplete |

G1/G2/G3 remain open until associated evidence is recorded. An unsupported
physical model, unreachable target or unavailable accelerator must retain its
actual status; none may be relabeled as a successful scientific validation.

## Physical references checked for this implementation

- Collins (1970), ray-matrix diffraction integral:
  https://opg.optica.org/josa/abstract.cfm?uri=josa-60-9-1168
- abTEM, incoherent ensemble treatment (not a production-source interface):
  https://abtem.readthedocs.io/en/latest/user_guide/tutorials/partial_coherence.html
- abTEM, phase/aberration conventions:
  https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html

Consulted 2026-09-10. Additional primary phase references checked on 2026-09-11
are listed in `CANONICAL_PATH_PHASE.md`. Physical controls and calibration
values are not changed to make an acceptance test pass.

The user subsequently requested publication of the current in-progress state
on 2026-09-11, after being informed that the complete HANDOFF was not finished.
This supersedes the design's original local-only boundary for this checkpoint;
it does not close G1/G2/G3 or qualify the missing source-to-image adapters.
The earlier shutdown instruction remains conditional on completing all work.
No shutdown accompanies this in-progress publication.
