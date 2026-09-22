# Classical gun batches, archive identities and visible timings

Recorded 2026-09-19. This continuation preserves the classical tip-origin
source and all modelled upstream operations. Coherent development remains
paused. It builds on [the prepared gun workspace](gun-loop-workspace-2026-09-19.md)
and the existing section/archive/continuation workflow.

## Executed gun optimization

The ordinary analytic field-emission gun now advances consecutive accepted
steps inside compiled execution. Each step still uses the original spatial
and relative-impulse limits, full-step versus two-half-step acceptance,
adaptive rejection and separate outer static-energy projection. The compiled
outer projection preserves the existing NumPy operation order; no relaxed
floating-point option or larger integration step is introduced.

A batch contains at most 16 accepted steps and ends at the next existing
history-save step. A possible aperture, mechanical bore, exit-plane or
backstream event returns its final segment to the original event handlers.
Those handlers still resolve transmission, stops, identities and crossing
times. Intermediate accepted steps are not independently configurable source
states. Returned arrays remain independently owned, and inactive particles
retain their state.

The fast path requires the supported original gun, fields and execution/event
methods, an ordinary flat emission patch, and either no cancellation callback
or a genuine `threading.Event.is_set` callback. GUI cancellation already uses
that callback. Unknown callbacks and custom hooks retain their per-step
semantics. Curved-tip, monochromator, vacuum and unsupported provider paths
continue through their existing algorithms. Cancellation is observed between
bounded batches. Current field/settings eligibility is rechecked per batch.

## Controlled 5,000-particle measurement

Both timings use the same physical inputs, 16 numerical threads on the
32-logical-CPU host, serial BLAS and warmed compiled kernels. The before
implementation was frozen before editing and loaded into a separate process;
the live checkout was not rolled back. Task-owned numerical jobs ran serially.

| Complete gun execution | Wall time | Separate kernel warmup |
| --- | ---: | ---: |
| Before this round | 83.996 s | 9.324 s |
| After this round | 78.763 s | 0.304 s |

This pair shows **6.23% less wall time**, or 1.066 times throughput. It is one
controlled pair, not a statistical scaling study or a full-microscope timing.
Kernel loading/compilation above is outside both transport timings.
The previous round's 79.792 s result is not this round's baseline. The gain
is modest: the accepted physical integration work remains necessary.

The measured new execution accepted 35,336 steps in 7,980 batches, with a
maximum batch length of 10 because the unchanged history cadence is tighter
than the hard batch cap. Both results have 5,000 survivors and an equal-time
history of 3,536 by 5,000. Their complete benchmark output digest is identical:

`a3b1841c60f657aba4b0daf57e57912e4dbffedcd44388650972440026b0c24c`

The digest includes trajectories, flight times, full exit state, identities,
weights, survival, history and stop labels. Final measured solver-file hashes
were unchanged during execution. The initial after-run also matched this
digest but its diagnostic recorder read a nonexistent instance schema field
after finishing transport. That receipt retains the metadata error. The
recorder was corrected to read the module's execution schema, and the complete
after-run was repeated; only the successful repeated timing is used above.

## Archive work

`WorkingPointCheckpoint` calculates each immutable numerical array's content
identity once per checkpoint object. Payload identity and package manifests
reuse that verified table. On load, identities are calculated from the actual
decoded arrays and checked against every manifest entry and the complete
checkpoint digest. An archive cannot supply a trusted precomputed identity
table. Byte-backed immutable storage prevents later changes invalidating the
memoized identity; compatibility identities and archive schemas are unchanged.

A separate synthetic storage benchmark used four deterministic arrays totaling
128 MiB, stored ZIP members and three trials per implementation. Median save
time fell from 0.412 to 0.358 s (13.23%); load fell from 0.542 to 0.481 s (11.31%).
Content hash calls fell from eight to four for both save and load. Archive size
and checkpoint identity were unchanged. This is a storage microbenchmark,
not a measured full-microscope speed improvement.

Older archives remain readable. As before, production source is included in
restart compatibility, so older solver versions cannot silently be reused as
current upstream execution. No generated arrays or archives are committed.

## Visible calculation details

The Ray Diagram's **Cached signals** page now contains a collapsed
**Calculation details** panel. It displays available current worker/pipeline
and stage timings, executed upstream reuse and the actual completed section.
Worker time includes internal cache IO but excludes queue wait, GUI drawing
and the separate section-archive worker. Gun substeps are explicitly included
in transport and must not be added again to the total. Full-result cache hits
suppress inherited old stage times. Changed inputs label retained details as
the previous result; absent legacy metadata is unavailable, not zero.

Section status displays measured save, load or existing-archive verification
time only after successful IO. Reusing an in-memory archive record labels its
retained measurement as previous. Failed/pending work never claims a completed
save. The total pipeline timer includes final checkpoint capture and input
validation. Timing text is selectable and copying it does not execute physics.

## Actual material-hit continuation

A local acceptance input uses 5,000 physical tip-origin particles, the normal
assembled column at 300 kV with a 0.5 mm column step, a 5 nm Si reference
specimen with a finite 100 micrometre disk, and EDS enabled. Scan, vacuum,
energy filter and wave calculation are off. The larger finite specimen is
only a benchmark input; user profiles and mechanical settings are unchanged.

The executed initial cutoff is 1589.2 mm, 10 mm before the specimen. It took
59.995 s, saved in 1.861 s and loaded in 1.953 s. Its gun and column checkpoint
arrays compared exactly after restoration. Subsequent ordinary full
continuation forbids any gun retrace. All 5,000 histories hit material, with
166 elastic events and a mean material path of approximately 5 nm.

The successful material phase used twelve numerical threads while the related
regression used four, keeping their combined numerical budget at sixteen.
These are workflow measurements with concurrent test activity, not an
additional controlled speed comparison.

| Full continuation stage | Wall time |
| --- | ---: |
| Column transport from saved state | 6.165 s |
| Material transport | 31.650 s |
| EDS photon transport/readout | 176.176 s |
| Downstream specimen-exit propagation | 13.744 s |
| Optical diagnostics | 1.382 s |
| Complete continuation call, including setup/capture | 230.640 s |

The full result saved in 6.316 s and loaded in 7.209 s. Gun state, axial
checkpoints, material terminal state and downstream checkpoint arrays all
compared exactly, including float64 clocks and unavailable values. The source
probability ledger error was approximately -1.11e-16. Zero-loss paths retain
executed clocks; positive-loss aggregate channels have no invented event-depth
time. Simulated EDS response was finite and nonzero; it is not measured data.
Peak process RSS during the full-state save/load/comparison reached 8.29 GB,
including simultaneously retained original and restored arrays.

The initial acceptance script incorrectly classified any negative relative
energy offset as an inelastic loss. A legitimate zero-loss particle can have
a negative offset due to source energy spread. That first receipt preserves
the failed post-calculation assertion. Classification now uses actual channel
energy loss; the repeated phase passed all checks and saves its complete state
before checking. No production flight-time model was changed to satisfy it.

EDS dominates this input: 82,592 photon-emission entries are transported.
The current particle restart package retains material state but does not
retain a complete optional EDS readout. A future EDS reuse change should bind
executed photon/source/material and detector data to their full dependency
identities. Disabling an enabled readout or omitting its interactions is not
an acceptable shortcut.

The second independent phase loaded that full archive in 6.788 s, increased
the second projector lens by 0.1 percentage point and continued in 192.535 s.
Repeated gun execution and repeated elastic material transport were forbidden
by explicit hooks; neither hook was called. The restored elastic object was
reused, all 2,310 incident integration nodes were retained, and all eight
material branches resumed at Z = 2299.2 mm and reached the axial endpoint
3026.4 mm. The same probability/clock checks passed. The changed result saved
successfully in 5.660 s. EDS alone took 172.024 s, confirming that avoiding
gun/material repetition does not yet eliminate repeated optional EDS work
after archive restoration. Peak phase-two RSS was 7.31 GB.

## Validation

The dedicated batch/compatibility run completed 74 tests with no failures or
skips. It includes exact serial/parallel consecutive-step comparisons and
complete nine-particle traces for ordinary exit, aperture/bore blocking,
nonzero alignment settings and a refined step. Event candidates, ownership,
custom hooks, cancellation, vacuum and disabled acceleration are covered.

The archive-focused run passed 10 cases, including corrupt array contents,
forged manifests, immutable ownership, legacy digest compatibility and atomic
failed writes. A separate UI fixture visually checks collapsed/expanded and
previous-result labels; its displayed numbers are illustrative and are not
simulation evidence.

The main related regression completed **316 passed, zero failures/errors/
skips**, in 498.694 s. It covers gun integration, optional acceleration,
physical events, TOF, ownership, CPU limits, archive corruption/compatibility,
actual small-source continuation, material reuse and GUI/controller workflows.
A separate process with Numba unavailable passed a real nine-particle launch
and five accepted general-path steps before requested cancellation. This is
a bounded fallback check, not an uncompiled complete-gun performance claim.

The actual 5,000-particle material phases both completed successfully with
their own bounded processes. Final timing-label edits additionally suppress
old gun/prefix metadata when the complete column is reused and show ordinary
pipeline checkpoint reuse. These edits do not alter the physical solver or
archive implementation measured above. The final UI/controller selection
passed **49 tests, zero failures/errors/skips**, in 38.752 s
(`final-timing-ui.xml`); this overlaps the main regression and must not be
added to it as an independent case count. Final syntax and diff-whitespace
checks passed, and the three measured gun-solver source hashes still match.
The whole repository suite, native desktop workflow,
coherent physics and experimental instrument performance were not qualified
by this focused run.

Local evidence lives under `tmp/gun-batch-20260919`: frozen before-files,
`comparison-5000-16.json`, successful `after-confirmed-5000-16.json`,
`archive-identity-benchmark.json`, dedicated XML receipts and UI fixture PNGs.
The material folder retains the original failed harness receipt, separate
successful `acceptance-phase1.json` and `acceptance-phase2.json`, and local
restart archives. Solver source remains part of compatibility; any older
source-version archive stays readable but must pass current reuse checks.
No application restart, Git commit or push was performed.
