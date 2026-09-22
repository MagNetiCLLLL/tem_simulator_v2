# Particle speed, cutoff planes and automatic restart archives

Current scope is classical particles. Coherent development stays paused. The
physical tip, extraction, acceleration, focusing, apertures, specimen interactions
and detector absorption are retained. No independent downstream source is added.

## Delivered workflow

1. Open Live tuning and capture the current settings. Enable **Calculation
   section (optional)** and choose a named plane or **Custom Z**. Component
   selection is optional for a one-shot calculation. When tuning, choose the
   components whose controls will participate.
2. Preview, Medium and **Run high-accuracy once at current settings** can stop
   at that plane. High accuracy retains the requested particle count and step;
   it is not silently reduced to a preview budget.
3. Successful current particle calculations, including ordinary complete High
   accuracy runs, automatically enqueue a background `.temsection` archive.
   **Saved particle state** displays its actual endpoint, resumable endpoint,
   quality, emitted population, timestamp, full path and reuse explanation.
   "Saving", "Saved" and a failed write are distinct states. Calculation
   results remain available if a write fails.
4. Change the cutoff and continue. Compatible gun, incident-column and material
   checkpoints are reused. **Load saved section...** makes a saved execution
   available in a later session. Loading keeps the current instrument settings;
   reuse requires matching upstream inputs. Changed source, actual fields,
   specimen, model implementation or numerical settings invalidate the affected
   portion. This is also shown in the status text.

The currently supported custom coordinate is the straight-column Z interval
from the gun exit to the last axial recording section, or to the entrance of
an enabled energy filter. This is not an internal-gun time-domain checkpoint or
an arbitrary point along the filter's bent path. Full requests execute the
assembled filter after the upstream path. Both full and explicitly bounded
particle paths now stop axial integration at its entrance; later kicks cannot
be folded onto that endpoint.

The automatic directory is `particle_sections` beneath the application's
artifact-store root. In the default desktop installation this is under local
application data, `TEM Simulator v2/high_accuracy_artifacts/particle_sections`.
The actual full filename is displayed after a successful write. Manually saved
files can use another directory. No user archive is automatically deleted.
Generated archives and arrays remain local and are excluded from Git.

## Numerical and persistence details

- CPU numerical jobs use at most half of the logical CPUs available to the
  process: 16 on the measured 32-CPU host. Affinity and explicitly lower limits
  are respected. The budget is applied inside workers; BLAS is serial and
  numerical jobs share admission, preventing nested full-size pools. This is a
  thread budget, not a promise of a constant OS utilisation percentage.
- The gun integrator reuses an exactly identical half-step after an adaptive
  rejection. It preserves the same error thresholds and accepted steps.
  History finalization releases consumed rows; compiled chronological indexing
  and small-event merging remove repeated Python work and sorting.
- Full and bounded column paths use compatible exact checkpoint planes.
  Restarts compare the actually consumed field nodes, interval midpoints,
  impulses and mapped fields. The initial checkpoint is retained as well as
  the terminal checkpoint. Full-precision values, not plotting precision,
  cross segment and specimen boundaries.
- Full High accuracy retains its existing diagnostics and outputs. Its executed
  transport can additionally seed a section. Validated archived material state
  is accepted by the ordinary full pipeline, avoiding a second material solve;
  outgoing material segments can resume from their own checkpoints.
- Packages contain checked JSON and numeric arrays with a fixed data-class
  allowlist, no pickle or file-directed imports. Atomic replacement prevents
  failed writes from replacing an earlier valid file. Array writes and reads
  are streamed, and contiguous buffers are hashed without whole-array byte
  copies. Particle archives use uncompressed ZIP entries for speed, so they
  trade more disk space for lower CPU cost while retaining every stored value.
- Repeated identical completed requests do not rewrite an existing archive.
  A loaded archive is a restart seed, not a complete EDS/STEM/filter product
  cache hit. Seeds and completed results share their configured byte and entry
  budgets. Cancelled, obsolete and mismatched results are not automatically
  published as current saves.
- High-accuracy section seeds are ranked by matching dependencies before
  comparing saved depth. An older loaded section cannot unconditionally displace
  a more useful completed section from the current session. Load admission
  failures, including a cache budget that is too small, appear as failures in
  the same status area rather than remaining labelled as loading.
- Vacuum remains opt-in. With vacuum participation, the existing conservative
  collision-state policy can require a fresh execution; no missing random or
  collision state is invented. Aggregate finite-loss channels continue to carry
  unknown flight times when their interaction depth is undefined.
- Disabled energy filters neither request an entrance state nor add a filter
  tracing stage. The actual 5,000-particle acceptance also exposed a conditional
  weight sum of `1.0000000000000002`. Stable summation, local normalisation of
  already admitted conditional weights and directly summing the nontransmitted
  population now retain bounded, conserved probabilities. Original terminal
  arrays remain unchanged. Negative probabilities and excessive discrepancies
  still fail, and the strict checkpoint validator has not been weakened.

## Measured gun improvement

The same 5,000-particle flat-tip gun input was executed completely in each case:

| Implementation | Threads | Gun wall time | Sampled peak process RSS |
| --- | ---: | ---: | ---: |
| Before this optimization | 4 | 161.048 s | 2.455 GB |
| After, same thread budget | 4 | 130.137 s | 1.084 GB |
| After, half-host thread budget | 16 | 85.750 s | 1.096 GB |

The fixed-thread reduction is 19.19%; the combined algorithm/thread reduction
is 46.75%. All three accepted 35,336 steps and retained identical trajectory,
flight-time, equal-time history, identity and exit-state digests. These are
single completed standalone gun timings with warm compiled kernels, not full
microscope or clean-JIT timings. See the detailed
[gun report](gun-performance-2026-09-19.md) and
[CPU budget](cpu-budget-2026-09-19.md).

## Actual 5,000-particle archive and continuation

The final-source initial run completed the production section and full
continuation. Its harness then made an invalid Python object-identity assertion
about the reused gun: the production pipeline may clone the object. That failed
receipt is retained. A separate recovery run loaded the same saved section,
verified all 49 gun fields exactly (including array shape, dtype, source IDs and
unknown-clock NaNs), completed the full pipeline and wrote its final archive.
Neither continuation called the gun tracer. No production source changed
between these two runs.

| Executed operation | Wall time |
| --- | ---: |
| Initial section, including tip-to-gun-exit calculation | 86.908 s |
| Save section | 2.109 s |
| Load section in initial run | 2.015 s |
| First full continuation from section | 23.275 s |
| Recovery load of the same section | 2.122 s |
| Recovery full continuation with exact-state checks | 24.151 s |
| Save completed full state | 3.007 s |

These are single-run measurements, not a distribution or a whole-instrument
speedup ratio. The section file is 1,171,123,051 bytes; the full-state file is
1,761,801,495 bytes. Both preserve 5,000 emitted particles. The continuation
starts at the exact saved Z = 1589.2 mm and finishes at Z = 3026.4 mm. Its
incident-column cache reused 948 history rows / 2,289 solver nodes, adding only
8 history rows / 21 nodes before the specimen reference. The final strict
specimen-exit validator passes.

Within the recovery continuation, the reported stages took 0.415 s for layout,
5.342 s for column work, 8.615 s for finite-specimen transport/routing, 2.511 s
for EDS, 5.754 s for downstream specimen-exit transport and 1.024 s for optical
diagnostics. Remaining call overhead is outside those stage timers. Thus the
gun is no longer paid again, while the newly requested physical segment and
outputs still execute. These material timings refer to the finite-specimen
miss case described below and must not be extrapolated to a thick illuminated
specimen or a raster scan.

Current-pixel detector outputs were available, including approximately 2,362
weighted simulated electrons at the high-angle annular detector and 2,282 at
the fluorescent screen. These are simulation population weights. No physical
exposure duration was supplied, so exposure-integrated electron counts were
left unavailable; source-current-derived rates are separate outputs.

Local ignored evidence is under `tmp/particle-acceptance-20260919/`:
`section-roundtrip-5000-final.json` retains the completed production timings and
the harness assertion failure; `section-roundtrip-5000-recovery.json` records
successful exact-state validation and final persistence. Worker sources and
archives remain in the same local directory. The earlier
`continuation-failure-diagnostic.json` retains the actual pre-fix probability
overflow. Final implementation identity:
`e3121502634baeb7df463481746a698b7f8652921d90a45b5b38ef54d49f1da1`.

## Validation status

The new complete-calculation/section bridge suite passed 23 tests, including
actual small-bundle gun transport, full-to-section, section-to-full, archive
round trips, full-precision checkpoint admission and the filter cutoff. The
separate gun suite passed 76 tests.

The broad integration run collected 403 tests: 401 passed and two existing
progress expectations still counted a disabled energy filter as a calculation
stage. These were test-fixture errors against the intended new behavior. The
fixtures now explicitly disable the filter, forbid invoking it and expect
three/six stages respectively. Both affected test files then passed all 36
tests. No production source changed for those corrections. The original
403-test receipt is retained at `tmp/particle-autosave-integration-20260919.xml`;
the final focused repeat of its two corrected failures is recorded separately
at `tmp/particle-progress-final-20260919.xml` (2 passed, 11.551 s). Compilation
of `src`, `tests` and `scripts`, and the repository's normal whitespace check
also pass. The live solver identity still matches the final 5,000-particle run.
This is targeted integration
coverage and correction verification, not a claim that the entire repository
test suite ran green in one invocation.

The final 5,000-particle scenario uses classical flat-tip emission, 300 kV,
0.5 mm column steps, 2 mm recorded history spacing, and a 0.2 mm gun trace
step. The section ends at Z = 1589.2 mm, 10 mm before the specimen reference;
the full axial destination is Z = 3026.4 mm. EDS is requested with 256 overlap
samples, the finite reference specimen is Si [110], 5 nm thick, and vacuum,
raster scanning, wave propagation and the energy filter are disabled.

In this particular broad-beam scenario the 5,000 terminal histories miss the
finite specimen. It verifies actual large-bundle transport, specimen/EDS
request routing, detector bookkeeping and archive continuation; it does not
qualify 5,000 material-hit scattering histories. Actual material-hit restart
and no-repeated-collision behavior are covered separately by the small physical
material bridge tests. The 5,000-weight ledger tests also cover mixed material,
missed, backscattered and inelastic populations using explicit transport stubs.

The 1420 by 900 offscreen view in `tmp/section-archives-final.png` was inspected.
Its metadata are a labelled render fixture, not an acquired or timed 5,000-ray
result. No native user window was restarted or manipulated. This delivery does
not claim full-instrument qualification, coherent acceptance, or a guaranteed
interactive frame rate.
