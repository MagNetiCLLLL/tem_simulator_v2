# TEM Simulator v2 — Project Handoff

Last updated: **2026-09-25**. Current checkpoint:
**classical particle transport and qualitative scientific trends**.

Latest continuation: [Condenser adjustment and accelerator annotations](docs/development/condenser-adjustment-display-2026-09-25.md).
The Ray Diagram action is now Auto-adjust condensers, with captured C1/C2/C3
before/after values and explicit optical-validation scope. A persisted
Acceleration gaps toggle draws snapshot-based stage annotations without changing
trajectories. Completed optical validation supplies its actual straight-column
extent; it does not acquire a resumable particle checkpoint. All 116 distinct
targeted cases passed. A native 193-source-sample candidate passed application,
undo and GUI publication using an isolated compilation cache; both screenshots
were inspected. The note records the earlier default-cache native crash and
verification limits. No solver smoothing, coherent development, commit/push or
user-application restart was performed. Restart to load the changes.

Previous repair: [Match transport result signatures](docs/development/match-transport-signatures-2026-09-25.md).
The transport result producer now copies immutable manifest signatures into the
current result dictionary contract. Two new tests reproduced the user's exact
error before the fix; all 33 targeted checks pass afterward. No physical solver
or compatibility policy changed. Restart the application to load the repair.

Previous continuation: [Shared beam plot sizes and ray colours](docs/development/unified-beam-colours-2026-09-25.md).
The two transverse plots share fixed dimensions and axis gutters. Plot choices
now synchronize Ray Diagram colours in both directions. Source/angle colours
stay fixed per emitted identity; TOF uses a saved cumulative-time gradient with
one common result-wide palette, while relative arrival delay remains in hover.
The distance-versus-time clarification was left open; cumulative time was the
explicit implementation assumption. Geometric path length was not added.
Gradient rendering retains stop clipping, missing-clock grey paths, rotation
and the existing screen-detail caches. Numerical transport was not changed.
Focused validation covers 252 distinct passing cases; see the linked note for
scope and the existing old-archive source-identity rejection. No commit/push
or user-application restart was performed. Coherent work remains paused.

Previous continuation: [Working points inside Live tuning](docs/development/live-tuning-working-points-2026-09-22.md).
The independent Working Points workspace tab has been removed. Live tuning now
contains Calculation and Working points subpages; infrequent record tools,
captured details and convergence checks are in a default-collapsed Advanced area.
Calculation has one Open/Export pair and scrollable cutoff/range controls. The
same records, signals, restoration and continuation paths remain connected.
New/reset Results layouts open the nested Working points page, while existing
custom layouts are preserved. All 190 distinct related tests passed. Source and
isolated installed GUI checks each passed 16/16 with calculations forbidden;
all four tested page states fit 900 × 700. The final wheel matches all 449 current
Python modules. Compilation and whitespace checks passed. This is targeted
verification; the prior complete-suite live-slider timing issue below remains
unresolved. Restart to use the changed interface. No commit or push was made.

Previous continuation: [Compact result files and startup results](docs/development/result-files-and-startup-2026-09-22.md).
The toolbar and File menu now provide Open result / Export result (Ctrl+O / Ctrl+S),
named saved results and an optional startup result. Files preserve calculated
inputs, full-precision executed state and dependency-checked continuation;
opening restores editable settings and saved views without transport or an
automatic Preview. Exports use lossless compression and exact array deduplication;
a real 49-particle fixture shrank 39.53%. Internal automatic archives retain
their low-overhead storage policy. Generated result packages remain excluded
from Git, and the named library is separate from disposable calculation caches.

Result-file, archive and library checks pass, including actual Preview/High
accuracy file loading with physical calculations forbidden. Full regression
executed 6,000 cases: 5,998 passed, one historical skip, one live-slider timing
failure. Following the final menu optimization, all 31 result/archive GUI cases
(including two new menu checks) passed as part of a 38-case run: 37 passed,
the same slider final-frame timeout failed. Final collection has 6,002 cases, all covered by
the full run or affected-module follow-up. The unchanged slider single case
then passed on both HEAD and current code with matched one-thread budgets, but
the intermittent module timing failure remains unresolved: do not claim a
fully green regression or proven new transport regression. Compilation,
dependency checks, final wheel byte comparison (449 Python modules), and
isolated installed GUI startup checks passed. Restart the application to use the new controls;
no GUI restart, commit or push was performed for this feature.
Coherent development remains paused.

Previous continuation: [Project audit repairs](docs/development/project-audit-fixes-2026-09-21.md).
The five independently reproduced defects are fixed: all filter-plane plots use
executed physical crossings, archive reuse checks verified file identity,
ordinary sample/EDS/STEM cancellation reaches existing safe boundaries, camera
pixels must be positive, and Experiments layout restoration refreshes its status.
Wheel builds use fresh private staging to exclude deleted source modules.
Fresh full regression: **5,926 passed, one historical-reference skip, zero
failures or unexecuted cases**. Production inputs remained fixed throughout;
two test-fixture corrections were followed by restarting their entire test group.
Final 1,056-input identity verification, compilation, dependency and whitespace
checks passed. The directly built wheel matches all 447 current Python modules,
and isolated installed GUI startup passed. Restart the application to load the
changes; no GUI restart, Git commit or push was performed. Coherent development
remains paused. This result supersedes the earlier test counts below.

Previous continuation: [Unified transverse tracking plots](docs/development/transverse-unified-plots-2026-09-20.md).
The normal Plot selector now fixes both coordinates and colour: source position
maps to plane position, emission direction maps to plane angles, and TOF maps
to plane position with per-path delay colours. The second coordinate selector
and its callback were deleted. Independent combinations are explicitly Advanced
diagnostics. Both plots retain shared identity/path colours; mode changes reuse
cached transport. All 179 related tests passed; compilation and whitespace checks
passed. An offscreen synthetic display fixture was visually checked. Restart the
GUI to use these changes. Coherent development remains paused.

Previous continuation: [Current interface and persistence audit](docs/development/current-contract-audit-2026-09-20.md).
Current accepted coverage is **5,858 passed and one skipped**, from a complete
5,859-case regression followed by full affected-module retests (25/94/98 cases).
There are no unexecuted current cases or remaining failures. The 581 production
files remained unchanged during this final verification; the single skip uses
obsolete C2 aperture-position numerical reference data. `compileall`, static
explicit-call checks and `git diff --check` passed.

Current State/profile/particle-archive schemas are **78 / 12 / 2**. Removed
fields, former component aliases and old schemas are rejected, not automatically
migrated. Quaternion sample orientation, held scan calibration, independent
stigmator controls, physical slit state, ten multipole field/calibration records,
tip quadrature and accelerator stage controls preserve their current ownership
through saves and loads. General and assembly-triggered condenser calibration
use detached state; only accepted controls reach the live instrument.

This round also fixed in-flight cache clearing, stale GUI callbacks, filter
control edits overwriting mode state, near-pole Euler conversion and a small
complex64 FFT norm drift while retaining physical absorption/band losses.
Coherent development remains paused. Restart the running GUI to use this code;
no application restart, Git commit or push was performed. This latest audit
supersedes earlier compatibility and unresolved-test summaries for active code;
those older entries remain a historical record.

Previous continuation: [Ray Diagram navigation](docs/development/ray-navigation-performance-2026-09-20.md).
Screen-resolution rendering retains original paths, stops, colours, full-data
Fit/export and physical results. A single-thread compiled simplifier with a
NumPy fallback and bounded scale caches removes repeated dense-line painting.
Actual saved-trajectory offscreen redraws fell from 420 to 9 ms for pan and
378 to 12 ms for zoom; first new zoom levels were at most 63 ms in this fixture.
These are isolated plot timings, not a measured full desktop frame rate.
Final targeted tests passed 58/58; the broader selection had 188 passes and
11 existing GUI-shell failures described in the report, so it is not all-green.
Restart is needed to use the new UI code. Coherent development remains paused.

Previous continuation: [EDS dose and spectral readout cache](docs/development/eds-readout-cache-2026-09-19.md).
Executed per-electron expectations now support safe dose, spectral range/bin,
FWHM and Poisson readout updates in full and section continuations. Static
frame-period/dwell changes reuse transport only when both scans are off;
material, incidence and detector geometry remain dependencies. Complete
records and coefficients survive archives, including JSON mapping reordering.
Duplicate event-ledger work is removed, and startup native compilation caches
are isolated by complete source version. Coherent development remains paused.
All **434 related regression cases passed**; the report records separate final
progress-label checks and the measured/final implementation identities.
In one actual 5,000-particle loaded-archive dose-halving comparison, the whole
call fell from **110.294 s to 11.777 s (89.3%)**; its EDS stage fell from
**99.627 s to 1.086 s**. Physical records and ledgers agreed within the declared
tolerance, with zero repeated gun/material/EDS physics on replay. Noise-only
and bin/FWHM updates took **1.428 / 1.580 s** in memory. Full archive load/save
still cost about **45 / 39 s**; this does not accelerate the first EDS execution.
Restart the application after updating code; historical results remain readable
but old solver identities are not rebound to make their calculation caches fit.

Previous continuation: [Persistent exact EDS results](docs/development/eds-persistent-cache-2026-09-19.md).
Complete EDS products now survive ordinary/full and cutoff-plane particle
archives. Both continuation paths reuse them only after actual incident,
material, model, acquisition-call and independent EDS dependency checks.
Historical missing EDS stays readable without claiming a completed spectrum.
All **257 related regression cases passed**, including a real nine-particle
tip-origin archive/continuation test. The actual 5,000-material-hit archive
round-trip preserved the complete EDS records exactly; a loaded downstream
lens edit took **13.339 s** with zero repeated gun/material/EDS executions,
against **118.209 s** when the same saved EDS was deliberately recomputed.
Complete EDS records matched exactly. Including larger-archive IO and checks,
the controlled pair took **201.757 / 93.831 s** (recompute / reuse).
See the report for settings, separate IO costs and the single-pair scope.

Previous continuation: [Classical gun batches, archive identities and timings](docs/development/gun-batches-and-timing-2026-09-19.md).
Bounded compiled batches retain the existing accepted steps, physical event
handlers and history cadence. The current controlled 5,000-particle pair at
sixteen threads took 84.00 s before and 78.76 s after (6.23% reduction), with
identical full-output digests. Immutable array identities are shared between
archive validation and manifests. Cached signals now shows current timings
and reuse details; archive status distinguishes save/load/verification and
historical measurements. See the report for final validation and the separate
actual material-hit continuation evidence. Coherent work remains paused.
All 316 main related regression cases and the final 49-case UI/controller
selection passed (overlapping selections, not a summed total). The actual
5,000-particle material/loaded-lens continuation passed with zero repeated gun
or elastic-material executions; eight branches reused their saved prefixes.

Previous continuation: [Fixed Transverse beam picture sizes](docs/development/transverse-plot-sizes-2026-09-19.md).
The Plot sizes dialog sets source, selected-plane and colour-legend dimensions
independently. Pictures remain fixed inside a local scroll area and follow
named workspace layout autosave/restore. All 101 related cases passed in two
runs (86 + 15), plus syntax and visual checks. No transport calculation changed.

Previous continuation: [User-defined cutoff and Ray Diagram extent](docs/development/ray-cutoff-display-2026-09-19.md).
The existing section/save/load/continuation workflow is confirmed. A narrow
spatial strip below the ray plot now distinguishes completed coverage, the
requested cutoff and the available restart plane. It follows the physical Z
axis while panning/zooming and marks earlier results after input changes.
It does not report time percentages or turn a requested plane into completion.
The report records the supported straight-column range and final UI checks:
139 related cases passed in two runs (133 + 6), plus syntax and visual checks.
The extended GUI test cleanup now drains delayed Qt widget deletion within
each test, avoiding accumulation at a later hidden-panel transition.

Previous continuation: [Classical gun loop workspace optimization](docs/development/gun-loop-workspace-2026-09-19.md).
At the same sixteen-thread budget, a new controlled pair of complete
5,000-particle gun runs took 95.84 s before and 79.79 s after this round
(16.74% less time). Trajectory, TOF, identities, weights and history digests
remain identical. Execution-local buffers and field preparation are reused;
the original physical operations and numerical error budgets remain in place.
This is a warmed gun-only measurement, not a full-microscope timing. The
latest report records final regression and archive-continuation verification.
The final related regression passed all 223 cases, including real small-source
archive/continuation and clock checks. A separate Numba-unavailable process
passed a bounded general-path smoke test; syntax and diff checks passed.

Previous continuation: [Particle speed, cutoff planes and automatic restart archives](docs/development/particle-speed-and-autosave-2026-09-19.md).
Numerical CPU work now uses at most 50% of available logical CPUs. The complete
5,000-particle gun benchmark improved from 161.05 s at four threads to 130.14 s
at the same budget and 85.75 s at sixteen threads, retaining identical outputs.
High accuracy now accepts cutoff planes and retains full-precision restart
state. Completed current particle calculations are automatically archived in
the background with explicit endpoint, quality, population, timestamp and path.
An actual 5,000-particle section took 86.91 s, saved/loaded in about 2.1/2.0 s,
and continued to the full axial destination in 23–24 s without any gun retrace.
Exact saved-gun state equality and final full-state persistence were verified;
the report records the separate storage recovery and finite-specimen-miss scope.
The 403-test integration run had 401 passes and two outdated disabled-filter
progress fixtures; after correcting only those fixtures, both affected test
files passed all 36 tests. Separate gun coverage passed 76 tests.

Previous continuation: [Particle section tuning and live detector signals](docs/development/particle-section-live-tuning-2026-09-19.md), with
[measured performance and next gun optimizations](docs/development/particle-performance-2026-09-19.md).
Main-window Preview/Medium now use the classical material/detector path. Users
can select a section and participating components, reuse executed upstream
states, save/load them, and extend material branches from their checkpoints.
Current-pixel counts and classical 2D scan products update with accepted results.
The default new assembly omits the energy filter; explicit saved choices remain.

Previous continuation: [Classical time-of-flight tracking](docs/development/transverse-time-of-flight-2026-09-19.md).
All three source/plane presets are available. TOF uses executed clocks; missing
history and aggregate finite-loss paths remain explicitly unknown. The [High accuracy Stage 2 performance and cancellation repair](docs/development/high-accuracy-performance-2026-09-19.md)
is retained.
The preceding [detector response and current accounting batch](docs/development/qualitative-third-batch-2026-09-19.md) remains in place.

Read [AGENTS.md](AGENTS.md), then the
[detailed current handoff](docs/development/HANDOFF_2026-09-19.md).
Coherent tip-to-column development is **paused**. The previous root document is
preserved in [the historical archive](HANDOFF_HISTORY_2026-09-13.md); its old
continuation and shutdown requests are not active instructions.

## Current objective and boundaries

The simulator must respond correctly to parameter changes within declared
physical regimes, using its own mechanical structure. Matching a real
microscope's numerical settings or absolute performance is not required.
Use scientific/functional equipment names in the UI; preserve original records,
reference URLs and compatibility identifiers as evidence.

Keep the physical tip → extraction → acceleration → focusing/apertures → column
chain. Do not add an independently configurable downstream source, renormalize
away current losses or remove interactions when a readout is disabled. Vacuum
remains opt-in/default-off. Integrate the energy filter last; pre-filter paths
remain upstream and unsupported downstream paths must not silently bypass it.

## Local repository state

- Workspace: `F:\tem_simulator_v2`; branch: `master`.
- HEAD: `7c52fa0593dfe73b36bcb12c042f55ba09f3334d`.
- Local `origin/master` points to the same commit; the live remote was not checked
  for this handoff.
- Scientific naming, all three qualitative batches and the performance repair are **uncommitted**. Preserve
  unrelated changes, especially `tests/test_stem_cuda_pipeline.py` and the
  untracked user brief `CODEX_OPTIMIZATION_ROUND2.md`.
- The latest detector batch includes code changes and bounded classical checks.
  No commit, push, application restart or shutdown was performed.

## Completed work

| Area | Delivered behavior | Detailed evidence |
| --- | --- | --- |
| Scientific names | Functional component/camera labels; original record IDs and numerical settings preserved | [Scope and naming](docs/development/qualitative-scientific-scope-2026-09-18.md) |
| Tip, gun, lenses, apertures | Parameter meanings/units, bounded current/aperture/lens checks; corrected nonzero-launch-potential energy handoff and cache identity | [First batch](docs/development/qualitative-first-batch-2026-09-18.md) |
| Stigmators, correctors, scan | Shared explanations; float64 first-order observations; correct on-axis linearization; filter-boundary guards; actual tip-origin local stigmator checks | [Second batch](docs/development/qualitative-second-batch-2026-09-19.md) |
| Active/inactive controls | Source, physical, numerical and historical metadata separated in the existing registry | [Parameter inventory](docs/development/qualitative-parameter-inventory-2026-09-18.md) |
| Detector response/current | Preserve zero weights and moved-detector collection in diagnostics; retain upstream losses in current budgets; shared response/readout meanings and bounded dose/PSF checks | [Third batch](docs/development/qualitative-third-batch-2026-09-19.md) |
| High accuracy gun performance | Same-error-budget compiled CPU/parallel gun update and impulse bound; High cancellation reaches physical gun steps; cache version includes numerical implementation | [Performance repair](docs/development/high-accuracy-performance-2026-09-19.md) |
| Source/plane tracking | Real source scatter; position/launch-angle presets; fixed ancestry colours and display cache | [Source tracking](docs/development/transverse-source-tracking-2026-09-19.md) |
| Classical TOF | Executed per-path clocks through gun, column, zero-loss specimen paths and physical filter planes; checkpoint/cache persistence; incomplete times grey | [Time-of-flight tracking](docs/development/transverse-time-of-flight-2026-09-19.md) |
| Particle sections and live signals | Executed gun/column/material prefix reuse; exact selected stop; checked section persistence; current-pixel detector counts and bounded classical STEM frames | [Section workflow and scope](docs/development/particle-section-live-tuning-2026-09-19.md) |

The model identifiers are `launch-potential-handoff-v2` for analytic gun
energy, `all-active-field-step-doubled-boris-compiled-v2` for analytic stepping,
and `axis-linear-float64-checkpoints-v2` for first-order response.
Changed calculations invalidate dependent caches; historical files and held scan
calibrations remain readable. No physical defaults were retuned to pass tests.

## Verification and open failures

- Latest TOF regression: **152 display tests**, **103 physical-transport tests**
  and **54 cache/restart tests** passed in separate completed runs. The final
  filter/source-routing/progress group also passed **54 tests** (receipt:
  `tmp/filter-tof-pipeline-20260919.xml`). Receipts
  are in `tmp/tof-20260919/` and the [TOF report](docs/development/transverse-time-of-flight-2026-09-19.md).
  This includes actual small CUDA execution, real tip-origin cache restarts,
  unknown loss clocks and malformed-clock rejection. An actual 49-particle
  tip-to-specimen run supplied the offscreen three-panel preview; its coarse
  column step does not establish whole-instrument convergence. Missing
  event-resolved finite-loss timing remains explicit.
  After correcting gun first-crossing geometry/time alignment, a final **39-test
  gun/cache/restart/reference group passed** (`tmp/tof-20260919/final-gun-cache.xml`).
- Latest source-view regression: **155 passed**, exit 0. Actual default-emission
  samples were rendered offscreen; no gun/column run was performed for the image.
  Full GUI exploration also found nine failures: eight reproduced before this
  feature, while one timeout passed separately on both versions and remains
  unexplained in the combined run. See the source-tracking report; full GUI
  acceptance is not claimed.
- Latest declared classical software lane: **259 passed**, exit 0; source and
  configuration hashes unchanged during execution and all 965 final inventory
  hashes match. Performance-repair run: `c5a31d64a673452988bacf33f6e7b82a`.
  Full simulator: **UNQUALIFIED**.
- Final performance/physical-source regression: **58 passed**. Complete 49-ray
  gun reference/compiled comparisons retain particle identity, energy, current
  and stops. A 15,000-ray local update-plus-impulse benchmark measured about
  **5.35x** acceleration with four CPU threads; this is not a whole-High timing.
  A final 15,000-ray default-gun diagnostic reached 33,335 steps in a 360-second
  budget and was deliberately cancelled before completion. Large-bundle total
  runtime and history/progress overhead remain open; earlier whole-High speed
  has not been demonstrated.
  The original running application was not restarted and retains its old code.
- Second-batch final control regression: **60 passed**, no skips. The actual-chain column
  refinement met unchanged 10 nm / 1 microradian budgets, with measured
  differences 0.134 nm / 0.400 microradians. This is a bounded local check.
- Expanded dependencies: initially 88 passed / 6 failed / 1 skipped. After
  scoped fixture corrections, **90 cases have passing evidence, four remain
  failing and one is skipped**. This is not a single clean 95-case rerun.
- The four failures reproduce on unchanged HEAD: a requested 60 mrad setting
  reached about 51.22657 mrad; requested 1.5, 2.0 and 2.2 micrometre illumination
  diameters reached about 1.10469 micrometres. Retain them as unresolved historical
  target qualifications; do not force source/geometry/presets to fit them.
- Third-batch detector regression: **262 passed, 2 explicitly deselected**; editor
  integration: **77 passed**. The two deselected coherent-image tests also fail
  on unchanged HEAD at source admission; neither the tests nor gate was bypassed.
  Actual 49-particle tip-to-detector runs retain current balance and the same
  stop fractions at 0.125/0.0625 mm column steps, including with readout disabled.
- Counts overlap; do not sum them or claim the full repository passed.
- Final serial Python compilation, whitespace and documentation-link checks passed.

Receipts remain locally under `tmp/qualitative-20260918/` and
`tmp/qualitative-20260919/`, plus `tmp/qualitative-detectors-20260919/`.
See the batch reports for exact runs, limitations,
unchanged-HEAD reproductions and commands. Generated arrays/caches stay out of Git.
Native interactive GUI, full-image physics, full-chain GPU performance and
filter-reaching transport have not been qualified by these batches.

## Next step

The [EDS cache plan](docs/development/eds-cache-plan-2026-09-19.md) now has its
exact-result persistence and per-electron response phases implemented. Dose,
spectral binning/broadening and Poisson sampling can use the executed response
when their physical dependencies match.
Incident direction remains a physical dependency through material and photon
paths; changing it cannot be reduced to changing electron counts.

Profile the first EDS calculation's substages before changing its algorithm.
Also reduce archive JSON/record reconstruction overhead without dropping
executed records or weakening provenance. The response cache does not speed up
first acquisition; repeated dose/readout edits now avoid EDS physics. Gun history,
prepared workspaces and bounded same-step batches are already implemented;
grouped adaptive stepping still needs separate numerical validation.

Earlier particle-section validation: **265 passed** in 187.11 seconds, receipt
`tmp/particle-sections-delivery-20260919.xml`. The group includes short physical
transport fixtures, real small tip-origin persistence, material probability/TOF,
actual 2x2 classical detector arrays, default no-filter startup, and GUI/controller
lifecycle tests. Expected existing Pydantic deprecation and pyqtgraph teardown
warnings were emitted. This is a focused suite, not the entire repository suite.
Do not add overlapping earlier agent totals. A stronger persistence-only rerun
passed **11 tests** in 16.23 seconds and checks restored material-branch
checkpoints (`tmp/particle-section-persistence-delivery-20260919.xml`). Compilation
passed. A fresh isolated 49-particle no-filter classical pipeline completed in
6.625 s after these checks; see the performance report for conditions and scope.
Coherent calculations remain disabled in particle tuning. Current scan material
contrast uses one reference exit distribution; per-pixel material Monte Carlo
and coherent atomic contrast remain outside this change.

The third Transverse Beam preset now reads executed per-path time. Remaining
timing work is **event-resolved finite-loss transport** and explicitly defined
arbitrary filter sections; see the [TOF report](docs/development/transverse-time-of-flight-2026-09-19.md).
Do not invent missing event depths, infer clocks from drawing paths or equate
TOF with wave phase. Coherent development remains paused.

Apply the performance repair by saving the current desktop configuration/results
and restarting the application; this also loads the new source views. Recheck the same High request; live inspection
showed the old job eventually left the gun for EDS. See the performance report
for bounded-run limitations and remaining history/progress-label overhead.

Continue **detector sampling/PSF support and loss presentation** across bounded
parameter ranges. The third-batch mean-response checks do not qualify extreme
PSF widths, arbitrary pixel sampling or charge-sharing noise covariance. Keep
physical interception, electronic response loss and virtual diagnostics distinct.
Use actual tip-origin transport for integration evidence and separately label
isolated checks. Keep coherent work paused and integrate the filter last.

Then continue remaining multipole semantics and bounded sweep/convergence
presentation. Existing working-point and numerical-qualification workflows should
be extended, not rebuilt. Historical alignment targets and earlier Round 2 gaps
remain separate open items in the
[detailed resumption record](docs/development/HANDOFF_2026-09-19.md) and
[Round 2 history](docs/development/ROUND2_OPTIMIZATION_PROGRESS.md).
