# Product usability progress

## Session: 2026-09-17

- Actual start HEAD: `9ea3a7cc3940d81a956770e6a3785f3dc9994363`, branch `master`.
  The working tree was clean. This is the exact baseline reviewed by the guide;
  no checkout/reset or history rewrite was performed.
- The user requested implementation of the supplied development guide. Its
  exact original was copied from the Desktop to `CODEX_DEVELOPMENT_GUIDE.md`.
  Attached historical publication/shutdown instructions are not current actions.
- Root `AGENTS.md` and `PROJECT_FUNCTION_SPEC.md` apply. The 2026-09-13 classical
  particle scope and coherent-development pause supersede older wave plans;
  the 2026-09-14 opt-in vacuum policy remains intact. No source/optical default,
  preset, physical emission law or hardware control was changed. Existing cache
  budgets remain unchanged; the continuation adds a separately configurable
  256 MiB input-asset retention budget. No commit, push, hardware access or
  application restart was requested.
- Requirement: UR-029. Work continued through every package with an available
  prerequisite. The table below supersedes the initial slice status. Software
  integration, numerical convergence, physical qualification and hardware
  acceptance remain separate; the guide is not declared universally complete.

## Package status

The user's continuation requested all ready work and a report of tasks that
cannot finish or need a decision. Dated intermediate failures below are retained
as diagnostic history; later passing receipts supersede them only in their
tested scope. Counts from overlapping runs must not be added.

| Package | Status | Delivered / remaining acceptance |
| --- | --- | --- |
| WP-00 | CPU baseline and bounded measurements recorded; hardware scope open | Original 32-test baseline, CPU timing harness, small/large input assets and cross-package workflow. CUDA transfers/isolated device kernels cannot be measured here. Final measurement receipts follow below. |
| WP-01 | Implemented; bounded UI/service acceptance passed | Complete records, transactional illumination Apply/undo, lazy indexing, migration, frozen A/B and scoped topology references. Includes real tip-chain relocated restore; this does not qualify a production optical working point. |
| WP-02 | Implemented; numerical results remain unresolved | Independent source factors, three grounded-gun mesh axes, executed common-population crossovers and prerequisite-gated joint checks. Under-resolved actual cases remain explicitly unresolved as the guide permits. |
| WP-03 | Implemented; final conservative-input regression recorded below | Registry-backed pilot tooltips/sweeps and reuse explanations; unclassified public extension inputs now enter actual stage identities and survive request preparation. Existing invalidation is never narrowed. |
| WP-04 | Implemented; ownership and integration verified | Pinned immutable asset graphs, configurable retention, complete background High/A/B preparation, cancellation cleanup and shared-buffer accounting. Measured memory refers to traced allocations, not whole-system RSS. |
| WP-05 | Implemented; moved-input and corruption acceptance passed | Read-only portable resolver, full structural/external input archive, exact NPZ/CSV restoration, runtime migration, temporary-original relocation and actual nine-ray rerun. Original live files remain untouched. |
| WP-06 | Implemented; actual CUDA acceptance blocked | CPU policy/ownership/context-reset fixtures pass. No compatible CUDA device is available for numerical equivalence, real VRAM or transfer measurements. |
| WP-07 | Implemented; CPU scheduling/ownership acceptance passed | Shared FIFO admission, explicit High ownership, latest live requests, independent experiments, shared retention, cancellation/failure release and bounded library threads. VRAM reservation is not driver-total accounting. |
| WP-08 | Implemented; offscreen acceptance passed | Four task layouts, compact result readout, frozen A/B, scrolling and keyboard/wheel tests. 100%, 125%, 150% offscreen layouts inspected; native desktop interaction is not claimed. |
| WP-09 | Supported tasks implemented; additional physical contracts unresolved | Joint condenser constraints and four-control beam centre/direction alignment use actual forward checks. Arbitrary two-axis stigmation is rank-deficient in the existing field model; dynamic pivot/scan-descan targets need definition. The optional observer does not yet support active vacuum. |
| WP-10 | Implemented; bounded experiment acceptance passed | Complete detached runtime/geometry studies, declared perturbations, local sensitivities, objective trade-offs, retained failures, strict resume, independent candidate TOMLs, explicit per-candidate optimization and provenance exports. |
| WP-11 | Bounded integration verified; exceptions retained | Combined GUI/actual-chain/archive regression, requirement ledger and build/isolated-install receipts. Full hardware/physical qualification and paused waves are excluded explicitly. |

## User-visible behavior and implementation

Open **Working Points**. The index supports source/assembly/mode/status/date
filtering and column sorting. Selecting a row displays captured parameters and
weighted readouts without restoring or evaluating the instrument. **Pin A**,
**Pin B** and **Compare A / B** preserve the actual records and show exact input
differences with comparison-context warnings. **Save input candidate** stores
the full snapshot using the existing input-only `.temwp` contract.

**Sampling and Convergence** offers one selected axis and a per-run ray limit.
The assistant starts from the selected snapshot, runs the existing physical
gun and incident-column audit twice, and leaves live state/results untouched.
It records exact component planes through specimen entrance, which may differ
from the selected result's sample-centre plane. Full path arrays are released
between runs; exported evidence contains scalars and input differences only.
Each run records the emitter type, emission budget, deterministic sampling rule,
surface quadrature settings when present, and absence of random replication.
Cancellation is checked at existing solver boundaries and before publication;
previous complete evidence is retained on cancellation or failure. Application
closure requests cancellation and defers closure if the worker is still active.

Sample count and N_eff are screening diagnostics, not convergence proof.
The minimum of the existing Microprobe/Nanoprobe relative tolerances is captured
before execution. Additional absolute comparison floors are declared in the
report before evaluation. They do not replace illumination acceptance targets.
`STABLE_FOR_CHECKED_AXIS` is only a two-setting numerical comparison;
whole numerical and physical qualification remain `NOT_ESTABLISHED`.
Nine-ray execution is deliberately insufficient under the retained 16-sample
screen; no new qualified default is installed.

Files:

- `src/temsim/working_point.py`: exact plane publication, captured current/date,
  immutable identity memoization and reuse of derived ray records per reader.
- `src/temsim/physics/simulation.py`: reserve the exact sample endpoint within
  the existing checkpoint-memory budget. This does not add a field operator,
  alter an integration grid or increase the 512 MiB cache budget.
- `src/temsim/sampling_diagnostics.py`: pure weighted scalar observations and
  raw snapshot index labels, without restoration or physics calls.
- `src/temsim/sampling_convergence.py`: bounded captured-input comparisons,
  predeclared gates, cancellation, per-plane differences and atomic JSON export.
- `src/temsim/gui/working_point_panel.py`, `gui/sampling_panel.py`,
  `gui/main_window.py`: existing manager integration and safe worker shutdown.
- `scripts/benchmark_product_usability.py`: reproducible small CPU benchmark.
- `tests/test_product_usability.py`: weighted/precision/evidence/UI contracts and
  an actual small classical tip-origin refinement.
- `tests/test_segmented_column_cache.py`: endpoint-budget coverage and correction
  of a stale source-color expectation to the existing real-emission coordinates.

## Validation receipts

Original baseline, before any source edits:

```text
.venv\Scripts\python.exe -m pytest tests/test_working_point_contract.py tests/test_working_point_packages.py tests/test_working_point_restore_gui.py tests/test_source_admission.py --disable-warnings --junitxml=tmp\product_usability_baseline_20260917.xml
exit 0: 32 passed in 153.72 s
log: tmp/product_usability_baseline_20260917.log
```

Post-change checks (overlapping scopes; counts must not be added):

```text
.venv\Scripts\python.exe -m pytest tests/test_product_usability.py tests/test_working_point_packages.py tests/test_working_point_restore_gui.py --disable-warnings --junitxml=tmp\product_usability_first.xml
exit 1: 22 passed, 1 failed in 141.44 s
log: tmp/product_usability_first.log

.venv\Scripts\python.exe -m pytest tests/test_product_usability.py tests/test_working_point_packages.py tests/test_working_point_restore_gui.py tests/test_working_point_contract.py tests/test_segmented_column_cache.py tests/test_source_admission.py tests/test_vacuum_opt_in.py tests/test_alignment_transactions.py --disable-warnings --junitxml=tmp\product_usability_regression.xml
exit 1: 75 passed, 1 failed in 293.21 s
log: tmp/product_usability_regression.log

.venv\Scripts\python.exe -m pytest tests/test_product_usability.py tests/test_working_point_packages.py tests/test_working_point_restore_gui.py tests/test_segmented_column_cache.py tests/test_background_preview_gui.py tests/test_ray_source_colour_mode.py --disable-warnings --junitxml=tmp\product_usability_final.xml
exit 0: 56 passed in 256.65 s
log: tmp/product_usability_final.log

.venv\Scripts\python.exe -m pytest tests/test_product_usability.py --disable-warnings --junitxml=tmp\product_usability_polish.xml
exit 0: 14 passed in 71.94 s
log: tmp/product_usability_polish.log
```

Taking the latest outcome per test ID across the regression, final and polish
JUnit files yields **97 distinct tests, 97 passed**. This is a deduplicated
receipt across focused runs, not a claim that the entire repository suite ran.

The first failure exposed missing terminal full-precision checkpoints in the
existing periodic cache. Reserving the exact endpoint fixed actual gun-result
publication, verified by the subsequent suites. A later failure was a preexisting
source-color assertion: it recentered a finite display population, while the
current implementation uses actual emission positions relative to the tip apex.
Running that unchanged assertion with HEAD's checkpoint allocator reproduced
the same failure (`tmp/product_usability_colour_baseline.log`, exit 1). The test
now verifies the existing launch-coordinate contract; color physics and drawing
code were not changed. The corrected test passes in the 56-test run.

Checks cover weighted versus count fractions, zero/empty/nonfinite populations,
under-resolved rejection, stale evidence, immutable readouts, full-identity
filtering, sorting/pinning without restoration or gun/column execution, exact
restore and rollback, cache-prefix equivalence, preserved source admission and
vacuum choices, asynchronous result routing, cancellation and Preview isolation.
The real 9-to-18-ray refinement executes the physical gun and incident path and
stays unresolved. GUI layout fixtures and transaction mocks certify software
behavior only. Existing pyqtgraph teardown warnings were also present in the
baseline; dependencies were not changed to suppress them.

Compilation of changed/new Python files and `git diff --check` pass. The compile
scope is the seven modified/new source modules, the benchmark script and the
two modified/new test modules (using `.venv\Scripts\python.exe -m py_compile`). Isolated
offscreen renders of both working-point tabs use the Windows Segoe UI font;
the initial missing-font render was replaced and inspected again. Full-table
columns remain horizontally scrollable, with complete values in tooltips.
No user's application was restarted. A full suite, long coherent calculations,
hardware, GPU benchmarks and numerical/physical qualification across assemblies
were not run; those exceed the current slice and paused-wave boundaries.

## Benchmark scope

Original receipt: `product-usability-baseline-20260917.json`.
Command: `.venv\Scripts\python.exe scripts\benchmark_product_usability.py --output docs\development\product-usability-baseline-20260917.json`.
Exit 0. Log: `tmp/product_usability_benchmark_before.log`.

- Windows 11 build 26100, Python 3.12.10; Intel Core Ultra 5 225U, 12 cores /
  14 logical CPUs; OS reports 102,471,254,016 physical memory bytes. Display
  adapter inventory reports Intel Graphics, driver 32.0.101.6452.
- CPU, default resolved assembly, 9 emitted rays, 1 mm column step, 5 mm display
  history step, 9 survivors at specimen entrance. Vacuum stays at its disabled
  default. No specimen interactions, detector or wave acceptance is inferred.
- One cold and five warm runs; 20 isolated offscreen detail refreshes for p95.
  Full input and implementation identities and individual samples are in JSON.
- Cold measured wall: 25.646 s. Warm median measured wall: 2.416 s; capture
  0.308 s, restore 0.440 s, gun including fields 0.0185 s, column 0.326 s,
  checkpoint hash 0.0926 s, archive write 0.407 s, archive read 0.289 s.
  Timers are nested; these values must not be added together.
- Original isolated detail refresh median: 1.519 s; p95: 1.883 s (20 samples).
- Field construction is included in gun time. GPU transfer/allocation, peak
  memory, large-asset capture and main-window event latency are not measured.
  No overall speed-up claim is made from this small case.

Post-change receipt: `product-usability-after-20260917.json`, produced by the
same command with that output path; exit 0, log
`tmp/product_usability_benchmark_after.log`. The case again retains 9 survivors.
Cold measured wall is 10.249 s; five-run warm median wall is 1.329 s. Warm
medians: capture 0.252 s, restore 0.320 s, gun including fields 0.0199 s,
column 0.162 s, checkpoint hash 0.0473 s, archive write 0.160 s and archive
read 0.196 s. Twenty isolated offscreen detail refreshes give a median of
0.0257 s and p95 of 0.0278 s, with the richer new index/readouts present.

The exact current input digest is
`90ca1ee288e472b208b802d0cf5c921336711229c1c0cad49bfe42921f13b73f`;
the implementation digest is
`1b84d96055dabc5d64f88e3bbb8ff5b31cb26eef818b6c3a54bfaf00cf6a75c2`.
Recapturing the benchmark inputs matches the after digest; substituting only
the original implementation identity reproduces the before input digest
(`tmp/product_usability_input_comparison.log`: both checks True, exit 0).
Thus the captured geometry, source, numerical settings and external bytes are
identical apart from implementation identity. CPU load/frequency and existing
disk/JIT caches are not controlled, so differences in unchanged solver stages
are not attributed to this patch. The detail-refresh result applies only to
this small local benchmark, not all working-point sizes or full GUI latency.

## First-slice limits (historical; continuation below supersedes delivered items)

The manager still imports complete archives eagerly and does not yet offer an
illumination-only Apply transaction or migrated historical identities. It does
not load historical calibration reports as current qualifications. The distinct
approved Microprobe six-crossover and historical Nanoprobe five-crossover
baselines remain unchanged in existing configuration; this slice does not
qualify topology. No result/default is silently promoted.

The assistant rejects active vacuum or coherent sources explicitly, preserving
their input choices. This is an audit capability boundary, not removal of the
main simulator's existing vacuum or historical wave code. Independent spatial,
direction and energy quadrature controls require further source-sampler work;
total-count refinement is labeled joint. Its worker follows existing Qt worker
patterns but is not yet part of the future shared resource coordinator.

Next ready step: finish WP-01 illumination-only Apply. Add a detached explicit
patch over existing runtime target setters, preview its exact allowed-control
diff, reject source/assembly/schema mismatch, and integrate the existing
main-window transactional installation/stale-generation gate. Start tests with
non-target and downstream equality plus injected-failure rollback. Then add
lazy archive indexing and assembly/mode-bound report evidence before marking
WP-01 complete. Continue WP-02 independent numerical axes without changing
source support or current, followed by the WP-03 dependency registry.

Read this receipt and `CODEX_DEVELOPMENT_GUIDE.md`, reconcile actual HEAD and
local changes, reuse verified work, and update actual receipts for the next
bounded slice. Do not reset user changes, repeat completed tests without a
reason, restart paused wave work or infer publication authorization.

## First-slice local artifacts and process accounting

Test logs/JUnit XML and offscreen previews stay under ignored `tmp/`. Benchmark
receipts contain scalars only. Benchmark `.temwp` packages are created under a
task-owned temporary directory and removed by its context manager. No generated
ray/wave arrays are proposed for version control. All task-owned test, benchmark,
input-comparison and offscreen-render processes completed. The final layout
fixture is labeled as such and did not launch any physics. No helper,
calculation or background follow-up is left running.

## Continuation implementation and verification

The user explicitly asked why work had stopped. Stopping at the first bounded
slice was an execution mistake; the complete development-guide request remains
active. No new source, preset, hardware, publication or wave-work authorization
is inferred. No subagents or autonomous follow-up tasks were created.

- `illumination_apply.py` previews the exact allowed existing runtime controls,
  checks source/assembly/mode/calibration/insertion compatibility and verifies
  the complete non-target diff. The existing installation transaction rolls back
  state/selectors/checkpoint on failure; the existing undo history also owns this
  input-only apply. Revision checks reject stale and A-to-B-to-A requests.
  Objective and downstream controls are outside this declared patch.
- `.temwp` manifest indexing now defers NPY loading. Explicit **Load retained
  data** verifies every array off-thread; browsing/sorting/pinning remains pure.
  Array headers, sizes, duplicate/undeclared entries and checksums are checked
  before allocation. Indexed scalar readouts are explicitly unverified.
  Embedded convergence receipts are associated with their actual identities.
- **Migrate inputs** creates a new input-only identity with parent and migration
  diff, retains the original result, requires recalculation and rejects custom
  exit sources. It does not update live controls.
- New optional tip product quadrature preserves old absent-parameter behavior.
  Spatial, direction and conditional-energy factors are independent numerical
  choices. For surface exponential emission, the angular marginal and energy
  conditional on direction preserve the existing joint N/T energy law and its
  local angular conditioning. Full-cap weights and current are retained. The
  assistant records the selected-to-baseline method change explicitly.
- The parameter registry reuses existing setters, sweep validators, field-support
  inventory and stage signatures. It adds explanations without narrowing an
  existing invalidation decision. Unmapped changes yield conservative plans.
- Bulk input arrays can be captured as versioned content references with pinned
  immutable bytes, while the original inline graph reader remains supported.
  Magnetic field ingestion now owns bytes independently of mutable input arrays.
  High-accuracy capture no longer decodes the complete graph on the GUI thread;
  preparation, memory estimation and manifest construction run in its worker.
  The synchronous calculation API remains available. Input-asset retention is
  separately configurable and included in the managed RAM budget.

Actual continuation receipts (logs and JUnit under ignored `tmp/`):

- `illumination_apply`: 25 passed, one test fixture used obsolete `_ray_result`;
  corrected to the workspace's actual `_last_result` and rechecked below.
- `working_point_continuation`: 33 passed, three failures identified use of a
  removed private NumPy header helper. Replaced with bounded public NPY v1/v2
  readers; the following archive checks passed.
- `sampling_independent`: 35 passed, including old archive compatibility,
  deferred reading, malformed archive rejection, independent quadrature and
  weighted source-law moment checks. These do not qualify microscope physics.
- `registry_sampling_apply`: 31 passed, including an actual 27-to-54-ray
  independent direction comparison through the physical gun/incident column,
  conservative dependency planning, illumination rollback and existing restore.
- `input_assets`: 69 passed, three expectation failures (whole-graph size
  estimate, old foreground High routing, and RAM total missing the new input
  cache). Expectations now test the intended new behaviors; expanded rerun is
  in progress. This is not yet a passing rerun receipt.

Continue with measured large-asset capture and remaining crossover/portable
archive integration, then the remaining execution, workspace and experiment
packages. Do not stop merely because another bounded slice has passed.

### Continued acceptance and measurements

- `topology_sampling.xml`: 55 passed in 91.161 s. The new bracketing-axis test
  executes the actual existing tip/gun/incident-column chain at 40/20 mm bracket
  spacing, retains identical transport inputs, and correctly remains unresolved
  with nine emitted particles. Existing clipping/common-population tests pass.
  Scoped references include source-file hashes and remain targets, not current
  validation. Every raw executed root and separately classified terminal root is
  retained in the scalar receipt.
- `sampling-field-grid-20260917.json`: the real grounded surface model ran with
  nine rays, radial mesh 32 then 64, axial mesh 64 and apex resolution 4. Execution
  took 15.143 s; comparison is **UNRESOLVED**, first screened unresolved plane
  450 mm. Coarse transport produced overflow warnings, retained in the local log.
  This is execution coverage, not field/source convergence. No field or source
  default was changed. Axial/apex meshes and joint-check workflow remain open.
- `input_assets_verified.log` reached 72/80 tests and then stopped making
  progress. The task-owned process was terminated; that run is not a pass.
  The isolated `background_preview_assets.xml` then passed all eight tests in
  59.29 s. The subsequent expanded run below completed without that stall; its
  original cause was not established and is not claimed fixed.
- `assets_workspaces.xml`: 92 passed, one failed, 172.166 s. The failure assumed
  the initial component tab was index zero. HEAD already selects Assembly (index
  three); the test now records and restores the actual initial selection.
  This completed run covers input capture/ownership, all existing snapshot
  contracts, cache settings and nine background toolbar/preparation checks.
- `workspaces_fields_verified.xml`: 18 passed, one layout-proportion failure.
  Adding the compact readout reduced available height enough for splitter minimum
  sizes to constrain the old 2200x1100 fixture. The proportional-layout fixture
  now reserves 100 additional pixels; tolerance is unchanged. Small-window
  usability remains a separate open gate, not hidden by that fixture change.
- `workspaces_fields_final.xml`: all 19 passed, 59.241 s, including named-layout
  persistence/reset, retained-result identity/staleness, exact weighted readouts
  and magnetic-field-map behavior. Existing pyqtgraph disconnect warnings remain.
- Offscreen English/Segoe UI previews of both Working Points tabs and full Results
  and Alignment arrangements were inspected. No physics ran for these labelled
  layout fixtures. At requested 1500x920 the main window clamps to 1500x1031;
  controls remain readable at that actual size. Smaller-window and scaled-display
  acceptance is still open. Images/scripts remain under ignored `tmp/`.
- `ray_gpu_policy_verified.xml`: 21 passed, one actual-CUDA test skipped, 9.756 s.
  Ray `Prefer GPU` and `Require GPU` now distinguish explained fallback from strict
  admission. Invalid input/physics/cancellation errors are not CPU retry reasons.
  CPU-only mapped/medium stages reject strict GPU execution rather than bypassing
  physical effects. This is not a GPU residency or hardware performance receipt.
  The earlier broader `ray_gpu_policy.xml` also exposed a historical wave-budget
  test that expects MemoryError before the current closed production source gate.
  That existing test still fails with UnsupportedWaveSource; no gate was weakened
  and no wave-development task was resumed to make it pass.

Large-asset measurement: `input-asset-capture-20260917.json` records five repeats
per operation, exact input round-trip checks, implementation identity, cold/warm
scope and tracemalloc peak allocations. Both fixtures use the supported
MagneticFieldMap representation, attached as explicit serialization inputs rather
than treated as calibrated active optical fields.

| 48 MiB component fixture | Median time | Peak incremental traced allocation |
| --- | ---: | ---: |
| Inline graph encoding | 302.91 ms | 114.03 MiB |
| Warm asset-reference encoding | 244.06 ms | 3.73 MiB |
| Inline graph hash | 546.64 ms | See scalar receipt |
| Asset-reference graph hash | 271.79 ms | See scalar receipt |
| Complete High request capture | 259.92 ms | See scalar receipt |

The 0.75 MiB fixture's asset encoding was slower than inline encoding (370.34 vs
299.76 ms); this is not a universal latency improvement. Timings include tracing
overhead, exclude physics and do not describe full-process RSS or VRAM. Later
source edits make the recorded implementation identity historical; the receipt
is retained as measured rather than relabelled as a newer implementation.

### Exact continuation instruction

Continue in the current dirty checkout; do not reset, publish, enable coherent
source admission, change optical presets or restart the user's application.
First review the files and receipts above, including the subsequent joint-check
and small-window verification below, then implement WP-05 using an explicit,
read-only, content-verified dependency resolver that covers structural and
specimen/field inputs consistently. Original-file checks still govern current
exact Restore; embedded bytes alone are not a portable resolver. Preserve old
records and reject missing/incompatible content atomically. Continue the ready
WP-06/07/09 work and then WP-10/11; do not treat missing CUDA hardware as a blocker
for CPU/UI or archive work. GPU residency/Auto timing decisions and cross-worker
memory reservations have not been implemented by the boundary-policy slice.

All task-owned test, measurement and offscreen rendering processes described
above have exited. One stalled task-owned test process was explicitly stopped;
no unrelated application was terminated. No background automation, helper,
subagent, commit, push or application restart was created. Generated arrays and
temporary packages remain outside versioned changes; tracked additions proposed
here are source, settings, documentation and lightweight scalar receipts only.

### Joint checks and small-window follow-through

The continuation did not stop at the earlier passing boundary. Joint gun/column
step refinement now requires independently completed, identity-verified gun-step
and column-step reports for the same input record. The joint receipt links those
reports, changes both steps explicitly and preserves every unresolved verdict.
The browser retains/exports all matching axis receipts. New sampling controls
and the result selector use the existing wheel-safe input policy.

The existing tallest pages, rather than any particular new input, determine the
workspace minimum height. A scrollable page region now retains their original
controls while keeping the result readout fixed above it. Fresh offscreen
Results/Alignment fixtures now attain the requested **1500x920**, and the small-
window test checks access through the scroll region without starting physics.
The earlier 1500x1031 limitation above records the pre-fix measurement.

Grounded-field follow-up receipts (`sampling-field-axial-20260917.json` and
`sampling-field-apex-20260917.json`) both executed the existing tip-origin path.
They independently doubled axial nodes and apex cells respectively from the
declared coarse baseline. Both remain **UNRESOLVED** at the first 450 mm screen,
taking 10.481 s and 9.453 s. They ran alongside UI regression, so their durations
are execution accounting, not isolated performance measurements.

The first combined joint/UI run was deliberately stopped after a separate render
exposed a local import shadowing the existing QScrollArea import. The import was
removed before rerunning; that incomplete run is not counted as passing. This
was the second task-owned stopped test process, separate from the earlier stall.
The replacement run, `joint_small_workspace_verified.xml`, passed 38 tests with
one failure and one associated Qt callback error in 161.14 s. The asynchronous
worker fixture omitted the axis field now required for retaining all axis
receipts; production reports already provide it. The fixture was corrected and
a multi-axis package/import history test was added. `joint_evidence_final.xml`
then passed all 23 working-point/index tests in 49.872 s. The replacement run's
actual joint-chain and small-window checks passed. No physics or source gate was
changed to resolve the UI bug.

### Export scope follow-through

The export dialog now distinguishes full captured inputs plus verified retained
results, full captured inputs requiring recalculation, and read-only metadata.
Input-only derivatives preserve the exact input graph and external input bytes,
link their original parent, and do not inherit executed-result qualification.
Metadata derivatives omit graph-array bytes, external content and all retained
numeric products; scalar parameter trees and historical scalar readouts remain
readable. Their distinct graph schema and UI guards prevent Restore, Fork,
illumination Apply, migration and sampling execution. Input/metadata export from
an indexed record does not load NPY products. Exports reuse the existing atomic
package writer and bounded, checksum-checked reader.

This is **not** portable execution. Inspection found implicit specimen preset,
reference catalog, support and EDS coefficient loaders, shared structural
definitions, mapped-field loaders and cached parsed values that must all consume
the same resolver. Simply removing original-path checks would mix archived and
live inputs. No such bypass was introduced. Complete input export describes the
captured payload; its original-dependency restoration requirement is visible in
the dialog and user guide. WP-05 remains in progress.

Validation receipts (commands use the existing `.venv/Scripts/python.exe -m
pytest`, with `-q --disable-warnings` and the named `--junitxml=tmp/...`):

- `export_scope.xml`: 47 collected, 46 passed, one failed, zero errors, exit 1,
  88.095 s; scope: `test_working_point_export.py`, `test_working_point_index.py`,
  `test_product_usability.py`, `test_working_point_contract.py`. The new size test
  incorrectly assumed a 50% reduction despite retaining the full scalar graph.
  It now checks actual bulk-content omission and size reduction without that
  unsupported ratio. No application code was changed to satisfy the ratio.
- `export_scope_verified.xml`: 24 passed, zero failures/errors, exit 0,
  64.792 s; scope: export, illumination apply and archive-index tests. Coverage
  includes exact input/result values, no result load for input/metadata export,
  missing input rejection, metadata execution guards and atomic write failure.
- `export_dialog_final.xml`: seven export tests passed, zero failures/errors,
  exit 0, 17.381 s. A further UI fix captures the originally selected record before
  opening either export dialog; a new result arriving during the dialog cannot
  silently change which record is exported. Its regression simulates that event.
- All 51 changed Python files passed syntax parsing; `git diff --check` passed.
  These checks do not constitute a full-suite or release acceptance run.

Final process accounting for this continuation: all task-owned tests, physics
measurements and offscreen renderers have exited. Two explicitly recorded test
processes were stopped earlier (one unexplained stall, one import-shadowing UI
failure); neither was a passing run. No user application or unrelated process
was terminated. No automation, subagent, commit, push or application restart was
created. Numeric test fixtures and images remain in ignored temporary locations.

Latest exact continuation: retain this dirty checkout and reuse the passing
receipts. WP-05's next implementation is a state/request-owned read-only archive
resolver with content-isolated parsed caches, consumed consistently by assembly,
shared tip/subassembly, specimen/reference/support, EDS and mapped-field readers.
Test it using copied temporary dependencies, move only those fixtures aside,
and reproduce the same bounded tip-origin calculation before enabling portable
Restore. Preserve original paths as provenance and the live TOML write boundary.
Then extend the existing controllers with WP-07 job-specific ownership and shared
reservations, including an explicit High job surviving later Preview edits. Do
not create a second State/controller, bypass absent dependencies, enable coherent
admission or treat a passing export-scope test as portable-execution acceptance.
WP-06 residency/measured Auto, WP-09 constraints, WP-10 experiments and final
WP-11 integration remain authorized and unfinished as listed in the table.

## Continued instruction: finish the report (2026-09-17)

The user now explicitly requests continued implementation through all report
content, with genuinely blocked tasks or necessary questions collected for the
next instruction. The existing dirty changes above are retained. No publication,
hardware operation, application restart or coherent-wave resumption is implied.

WP-05 is being integrated using `input_io.py`: a content-verified, read-only,
request-local resolver and separately keyed parsed-input caches. Explicit
portable input copies include the configuration tree and declared external
inputs, preserve the original record, and require recalculation. Profile/full-
graph request copies retain archive ownership. Captured Python/scientific-library
versions prevent continuation across runtime drift; explicit input migration
creates a new identity. The bounded embedded file budget is 96 MiB / 10000 files;
the existing package reader's overall/manifest bounds still apply. This input
budget is separate from the user's 96 GB host and calculation/cache allowances.

Actual early checks:

- `archive_capture_smoke.log`: 40 configuration files captured; exact snapshot
  digest survived restore and profile payload retained archive association.
- `portable_inputs_initial.xml`: 37 tests, 36 passed, one fixture filename
  failure, zero errors, exit 1, 85.217 s. The moved-directory test ran the actual
  nine-ray physical gun/incident column before and after moving only temporary
  copied inputs aside. Float64 checkpoints, masks, weights, energies and parsed
  CIF atom/cell arrays matched exactly. Preview and High request preparation
  retained archive association. The cache-isolation fixture named si_110.toml
  instead of the actual 10_si_110.toml; its name has been corrected.
- `portable_inputs_verified.xml`: 43 tests, 15 passed, 20 failures and eight
  setup errors, exit 1, 56.501 s. A test-only configuration redirection leaked
  into modules imported after the patch was established. The fixture now
  restores every temporary alias before yielding, so subsequent work uses the
  ordinary live roots and must resolve the archive explicitly. This failed
  run is not acceptance evidence. The replacement run is being recorded next.

Portable UI integration, map-loader coverage and the remaining report packages
are still in progress; no whole-package completion is claimed at this point.

### Portable inputs and worker lifecycle receipts

- `portable_inputs_isolated.xml`: all 43 passed, 111.660 s. The corrected fixture
  restored temporary import aliases before use. This includes the actual
  nine-ray moved-input reproduction, runtime migration, complete snapshots,
  illumination transactions and field-provider regressions.
- `portable_integrated.xml`: 53 tests, 50 passed, one failure and two setup
  errors, 109.81 s. Both NPZ and CSV original-file removal tests passed with
  identical field values, geometry binding and source SHA. All six malformed
  inner-NPZ tests passed before NumPy allocation. GUI failures exposed surplus
  Qt signal arguments through decorated callbacks; those signatures were fixed.
- `portable_gui_gc.xml`: both restored-window/return-to-live and background
  portable-copy UI tests passed, 17.79 s. Worker lifetime/queued GUI receivers
  are explicit. Repeated stalled mixed-GUI runs led to moving automatic Python
  cyclic collection to an application-owned GUI timer; ordinary reference
  counting still frees array buffers immediately. The focused worker test
  verifies finalizers run only on the GUI thread while a worker is active.
- `coordinator_initial.xml`: all 74 passed, 101.23 s: shared FIFO admission,
  UI event responsiveness, failure/cancellation release, alias-aware retained
  bytes, High-versus-live ownership, GUI garbage collection, portable inputs,
  cache settings, background preparation and illumination apply/undo.

Shared job admission is configurable in **Performance and cache**, separately
from retention caps. The initial shared reservation cap is 40 GiB, bounded to
three quarters of detected physical RAM by saved preferences; the existing
24 GiB individual High-calculation estimate remains. These are conservative
ownership/admission estimates, not an RSS cap or measured GPU allocation. The
coordinator currently runs one numerical worker with at most four BLAS/Numba
threads. FIFO order prevents later previews overtaking already queued explicit
work. A completed High result for older inputs is retained under its captured
working point without replacing the live display. No optical defaults changed.

During diagnosis, only task-owned test processes were stopped: portable GUI
initial (18956/27060), signal-signature diagnostic (12036/8904), and mixed GUI
cycle diagnostic (29644/22916). They produced no successful acceptance receipt.
No user application or calculation was stopped. Subsequent completed test runs
exit normally. Wider adapter tests are running next; their outcome must be
recorded before package completion.

### Portable round trip, task adapters and paused baseline

- `coordinator_adapters.xml`: 66 passed, five failed, 170.34 s. Two interface
  fixtures were updated to capture the newly required real input state and
  optional High parent identity. Three unchanged production-wave tests failed
  at the existing coherent tip-source admission boundary.
- `baseline_wave_gate_verified.xml`: the same three failures were reproduced
  in an isolated source/tests/configs export of original HEAD, using explicitly
  `pythonpath=src`, 16.99 s. The first baseline attempt imported the editable
  current package and is not used as baseline evidence. No original files or
  history were replaced. Production wave admission remains closed.
- `coordinator_portable_verified.xml`: 48 passed, three specifically named
  baseline wave cases deselected, 133.64 s. Full portable inputs/results package
  round trip retained exact arrays, and actual incident transport was rerun
  after only copied temporary originals were moved aside. High completion for
  older inputs preserves live display and publishes the captured parent.
- `ray_residency_coordination.xml`: 21 passed, one CUDA-hardware skip. Ownership
  fixtures demonstrate unchanged-plan reuse without plan upload/allocation,
  capacity reuse, fresh independent host results, context/reset invalidation,
  bounded admission and failure cleanup. These are CPU emulation fixtures,
  not a passed hardware numerical or speed comparison.
- `ray_device_weak_ownership.xml`: eight passed. Ownership follows the installed
  Numba allocation registry through weak references, including proxy owners;
  a context reset cannot keep a stale allocation alive through this cache.
- `joint_alignment_constraints.xml`: 24 passed, two failed, 87.70 s. One test
  incorrectly called an emitter serialization method that does not exist;
  the other exposed operational readback mutation when the new observer was
  called directly. The observer now owns a detached graph even for direct
  callers. A corrected focused receipt follows when verified.

CUDA residency retains at most 512 MiB in one process-local plan/buffer entry.
Requested checkpoints remain float64 and host results own fresh storage. The
shared worker device scope checks required ray allocations against its VRAM
budget; this does not claim accounting for driver overhead or unrelated device
users. No device handles enter archives. Timings distinguish plan/particle
upload, allocation, launch, synchronization and download; host synchronization
includes outstanding device work, so these are not isolated hardware kernel
measurements. Auto learns only from actually requested compatible executions,
requires at least two observations per compared backend and a 10% median
advantage, and retains the deterministic fallback without sufficient evidence.
Explicit CPU remains CPU. Hardware CUDA acceptance remains pending.

Joint condenser requests are opt-in, do not change production presets and keep
all non-authorized inputs fixed. The current optional entrance-plane observer
rejects active vacuum transport explicitly; it does not disable scattering.
The independent gun/column comparisons and effective-sample screen do not
qualify source sampling, field meshes, wave imaging or the entire instrument.


### Constrained alignment, complete experiments and integration continuation

- `joint_alignment_verified.xml`: all six corrected joint-constraint tests passed,
  54.379 s. Target-versus-constraint failures, independent numerical checks,
  guarded apply/undo and direct observer ownership were exercised.
- `beam_joint_gui.xml`: 18 passed; one background dispatch fixture timed out,
  114.553 s. Its isolated rerun passed in 13.653 s (`beam_preview_dispatch.xml`).
  The later 66-case integration run also passed that test. An isolated rerun
  alone was not used to claim the intermittent failure resolved.
- `design_full_snapshot.xml`: 42 passed and three failed, 61.025 s. Two fixtures
  expected a later error message; full snapshot restore now detects changed
  external bytes earlier. The third revealed that allowed specimen thickness
  edits must compare the exact external inventory rather than a geometry-bearing
  legacy external signature. Corrected focused checks: three passed, 15.700 s.
- `experiment_records_initial.xml`: 27 passed, 92.480 s. Save/load, cancelled-prefix
  resumption, changed-tolerance rejection, explicit seeded perturbations, retained
  failed points, scalar export and read-only plot behavior were exercised.
- `geometry_experiments_initial.xml`: 60 passed and two failed, 117.182 s.
  A diagnostic confirmed legacy profile application changed specimen Euler
  readbacks. It also exposed a broader risk: that migration path could reset
  source, model and vacuum settings. Complete experiment reconstruction now
  restores validated captured scalar controls directly, without historical
  profile migration. The full model graph still owns non-scalar inputs.
- `geometry_experiments_verified.xml`: 24 passed, 99.971 s, including actual
  nine-ray tip/gun/column execution for a detached aperture geometry, map-mismatch
  rejection, unchanged live TOMLs, fixed lens controls and saved geometry resume.
  This is not numerical convergence or a field-model calibration claim.
- `report_integration.xml`: 65 passed and one test-fixture failure, 228.283 s.
  The new integration fixture used a nonexistent wrapper `.percent` property;
  it now edits the actual registered lens. Background dispatch, joint/beam
  transactions, keyboard/wheel behavior, layouts and existing UI regressions
  passed. The original offscreen screenshot was clamped by Qt's virtual monitor;
  subsequent QA explicitly sets the requested logical viewport after layout load.
- `report_workflow_verified.xml`: 36 passed and two failed, 87.416 s. The new
  full workflow found a real archive issue: `input_scope` tried to attach a
  resolver to a frozen AlignmentRequest. Immutable request objects now use the
  scoped resolver without mutation. The other test needed to await the new
  background A/B capture before reading the slot. These failures require the
  subsequent combined receipt; they are not counted as acceptance passes.

The added static beam task uses the existing upper/lower deflector kicks and
weighted position/slope measurements at the entrance plane, with measured local
rank/conditioning and three independent forward runs. Its actual nine-ray test
shows a real response to a physical kick. The condenser stigmator's implemented
normal quadrupole depends only on half the X-minus-Y strengths; the two nominal
controls have rank one. Arbitrary two-axis stigmation is therefore unavailable.
Dynamic pivot/scan-descan targets and observation planes are not silently invented.

Experiments retain exact input graphs, including attributes outside State.to_dict.
A/B capture now pins the existing immutable input assets on the GUI thread and
performs file checks, full serialization and signatures in the shared worker.
Clearing captures cancels pending publication. The complete recipe/history is
included in shared retention accounting. Independent jobs no longer require the
current live microscope to have the same inputs as their captured recipe.

Geometry candidates stage independent materialized TOML content through
PartModelDocument, deriving dimensions and validating existing graph/clearance
rules before rebuilding the captured assembly. Active field-map mismatches are
failed points, not silent analytic substitutions. Configured analytical modes are
explicitly retained as analytical. Optional per-candidate optimization uses the
same constrained AlignmentRequest/CommitGate and retains its solved controls and
forward evidence. Invalid geometry and failed optimization remain in the record.
Failed points are not silently retried on resume. Completed calculation arrays are not
added to the experiment history; scalar CSV/PNG exports include provenance.

Initial packaging check: `pip wheel --no-deps --no-build-isolation` succeeded
using the installed validation environment. This is build evidence only; final
wheel inspection/import and final-source construction are recorded separately.

### Combined acceptance and final corrections

- `report_archive_workflow.xml`: 57 passed, 168.200 s. The full isolated GUI path
  uses actual nine-ray tip/gun/incident transport, compatible restore, changed
  physical controls, independent A/B, a cancelled and a failing alignment,
  complete export, moved temporary originals and an array-exact rerun. The frozen
  request resolver defect from the previous run is covered and fixed.
- `report_core_regression.xml`: 177 passed, 205.898 s. Covers source admission,
  complete working points, arrays, weighted diagnostics, independent sampling,
  topology, cache semantics, CPU scheduling/device-policy fixtures, detector
  interception despite disabled readout, and preserved vacuum choices.
- `report_experiment_final.xml`: 43 passed, 166.027 s. Covers local sensitivity
  objectives on nonuniform grids, no derivatives across failed neighbors,
  repeated cancellation retaining a resumed prefix, detached geometry inputs,
  saved records and background Preview routing.
- `report_unknown_input_guard.xml`: 71 passed, four failures, 200.774 s. The
  new conservative extension guard initially treated already-supported dynamic
  simulation time as unknown and unnecessarily detached immutable assembly
  definitions. Time now retains its existing explicit cache/preparation owner.
- `report_final_inputs_exports.xml`: 84 passed, one intermittent dispatch
  failure, 238.079 s. Unknown nested extensions invalidate actual product keys
  and survive Preview/High capture; export collision and checksum checks pass.
  The background preparation startup failure is investigated separately; this
  mixed receipt alone does not close that issue.
- `report_ui_125.xml`: one passed, 20.079 s. `report_ui_150.xml`: one passed,
  19.537 s. Together with the 100% run, each exercises all four task layouts,
  scrollable geometry optimization and keyboard focus at 1500 x 920 logical
  pixels. Representative scaled screenshots were inspected. These are offscreen
  Qt tests with isolated settings, not native-desktop manual acceptance.
- `report_capture_dispatch_verified.xml`: 58 passed, 121.728 s. The final
  extension guard preserves admitted large arrays without inline copies;
  conservative signatures, Preview/High capture and shared job ownership pass.
- `report_dispatch_stress.xml`: five separate parameterized live-edit scenarios
  passed, 33.054 s. The earlier startup timeout was not reproduced and its cause
  is not established; it remains a known intermittent issue, not a claimed fix.
  Failed harness experiment `report_dispatch_five.xml` duplicated pytest Item
  objects and raised four fixture KeyErrors; it is excluded from product evidence.
  The replacement stress run creates proper independent parametrized fixtures.

The final registry guard hashes unknown public extension values into every
affected product key rather than merely displaying a warning. Its known-label
exceptions are restricted to the owning component types. Large admitted arrays
contribute immutable content references without a new full-size inline copy.
CSV/PNG provenance filenames now include the output extension and its SHA-256;
exporting a matching stem cannot replace the other format's manifest. CSV writes
are atomic, and cancellation also clears the global running-status message.

### Acceptance identifier mapping

These mappings identify executed coverage, not blanket certification. Fixtures
and actual transport are deliberately distinguished above. File names are under
`tests/`; receipts are local under `tmp/`.

| Guide ID | Coverage / actual boundary |
| --- | --- |
| AT-01 | `test_source_admission.py`, `test_working_point_index.py`: prohibited exit-source activation rejected; historical records remain readable. |
| AT-02 | `test_working_point_contract.py`, `test_input_assets.py`, `test_experiment_records.py`: full graph, disabled inputs, aliases and exact arrays. |
| AT-03 | `test_background_calculation_requests.py`, `test_job_coordination_gui.py`: immutable click-time input and stale High ownership. |
| AT-04 | `test_product_usability.py`, `test_working_point_index.py`, `test_report_workflow.py`: view/sort/pin/layout without physical edits or solver dispatch. |
| AT-05 | `test_illumination_apply.py`: exact declared illumination patch, non-target equality, undo and invalidation. |
| AT-06 | `test_illumination_apply.py`, `test_alignment_transactions.py`, `test_report_workflow.py`: incompatible/stale/failed transactions, cancelled alignment and prior-result preservation. |
| AT-07 | `test_product_usability.py`, `test_result_readout.py`: weighted currents and containment at exact checkpoints. |
| AT-08 | `test_product_usability.py`: unavailable/under-resolved population does not become zero-size focus or validation. |
| AT-09 | `test_independent_emission_sampling.py`, `test_topology_evidence.py`: independent dimensions and identity-bound joint prerequisites; actual bounded runs remain unresolved. |
| AT-10 | `test_topology_evidence.py`: ordered intervals, multiplicity and separate Microprobe/Nanoprobe references. |
| AT-11 | `test_parameter_registry.py`, `test_segmented_column_cache.py`: overlapping field-support and exact stage dependencies. |
| AT-12 | `test_parameter_registry.py`, `test_vacuum_opt_in.py`: actual unknown-input signatures and conservative vacuum invalidation. |
| AT-13 | `test_input_assets.py`: immutable admission, retained pins, original-array mutation and no repeated large-payload copy. |
| AT-14 | `test_portable_inputs.py`, `test_portable_field_maps.py`, `test_report_workflow.py`: moved fixture originals, structural inputs, exact field bytes and actual transport. |
| AT-15 | `test_working_point_index.py`, `test_portable_inputs.py`: content/schema/path/size validation before use. |
| AT-16 | `test_working_point_index.py`, `test_working_point_export.py`: historical views, exact-restore gate, explicit new migration identity. |
| AT-17 | `test_ray_device_residency.py`: CPU ownership/emulated device fixtures only; real CUDA execution remains unverified. |
| AT-18 | `test_ray_gpu_policy.py`: eligible accelerator failure retry, strict GPU errors and no input/model-error masking. |
| AT-19 | BLOCKED: compatible CUDA hardware absent; no CPU/GPU scientific equivalence claim. |
| AT-20 | `test_job_coordination.py`, `test_job_coordination_gui.py`, `test_background_preview_gui.py`: FIFO, scoped cancellation, retained High and latest pending live input; intermittent startup diagnostic recorded above. |
| AT-21 | `test_product_usability.py`, `test_design_explorer.py`, `test_result_readout.py`: result/plane-specific summaries and read-only comparisons. |
| AT-22 | `test_product_usability.py`, `test_segmented_column_cache.py`: exact float64 checkpoint population distinct from reduced display history. |
| AT-23 | `test_record_plane_detector_masks.py`, `test_stem_detector_control.py`: unread intercepting detectors still physically block electrons. |
| AT-24 | `test_geometry_experiments.py`, `test_experiment_records.py`: detached validated TOMLs, failed candidates, exact fixed controls and unchanged live definitions. |
| AT-25 | `test_source_admission.py`, scoped alignment GUI tests: explicit paused/unsupported gates; rejection is not completion of missing physics. |
| AT-26 | Build/isolated installed-package receipt below; task layout tests use separate settings and cache directories. |

### Remaining scope requiring hardware, physical definitions or qualification

1. **CUDA hardware:** complete AT-17/AT-19 on a compatible device, including
   accepted coordinate/metric/weight/stop comparisons, context reset, transfers,
   isolated kernels and observed VRAM. CPU emulation cannot certify these.
2. **Additional alignment physics:** the current condenser stigmator has one
   independent normal-quadrupole response. Arbitrary two-axis stigmation needs a
   specified additional physical response/calibration. Dynamic pivot and
   scan/descan matching need time-dependent targets and observation-plane
   definitions. None were invented. The optional joint incident observer also
   lacks supported active-vacuum statistical validation; it rejects this request
   without changing the saved vacuum choice.
3. **Numerical and physical qualification:** actual nine-ray/source-refinement
   and bounded grounded-field studies are unresolved. No production defaults
   were replaced and no assembly-wide accepted working point was fabricated.
   Higher-budget qualification must retain the configured targets and evidence
   scope. Software acceptance of unresolved reporting does not resolve physics.
4. **Native UI and measurement scope:** the tests cover offscreen layouts at
   three scale factors. Native graphics-driver behavior and populated large-mesh
   interactive timing remain unmeasured; no user's running application was
   restarted. The RAM admission budget is an estimate, not an OS RSS limit.
5. **Paused coherent work:** per root `AGENTS.md` dated 2026-09-13, coherent
   tip-to-column development is still paused. Three preexisting production-wave
   test failures were reproduced at original HEAD and remain outside this
   classical scope. Neither source rejection nor classical particle agreement
   qualifies TEM/STEM phase imaging.
6. **Intermittent startup:** one mixed regression still observed a preparation
   worker failing to enter within the test's five-second window. A subsequent
   58-test group and five independent repeated live-edit scenarios passed.
   Diagnostic stack capture now accompanies a recurrence. Without a reproduced
   cause this issue is left open rather than weakening the timeout or claiming
   that rerunning fixed it. Existing pyqtgraph signal-disconnect teardown warnings
   also remain nonfatal; the report does not claim warning-free execution.

Exact resumption instruction: read this current package table and final receipts,
then root `AGENTS.md`. Continue only the selected remaining scope above. Preserve
the uncommitted implementation and all existing physical source/default/vacuum
contracts. Do not rerun paused wave work, recreate completed features, or claim
hardware/physical acceptance from CPU/offscreen fixtures. For new stigmation or
dynamic alignment, obtain the missing physical target/model definition first.

### Final CPU workflow measurements

Measurements below ran sequentially, without this task's concurrent tests or
benchmarks. All are local measurements on the detected Windows/Python host.
They are not throughput promises for high-accuracy imaging or other assemblies.

- `product-usability-final-20260917.json`: one first call and five warm calls,
  nine physical-tip rays, 1 mm incident column step. Warm median total including
  capture/restore, transport, checkpoint and archive I/O: **1.189 s**. First call:
  **10.215 s**. Warm capture 0.151 s, restore 0.283 s, gun/cache 0.011 s, column
  0.163 s, archive write/read 0.122/0.237 s. First electric-field provider
  construction/cache lookup was 0.066 s nested in the 8.959 s gun stage;
  trajectory field evaluation and JIT are not separately timed. Twenty
  working-point-detail refreshes: median 71.8 ms, p95 86.5 ms. Earlier baseline
  timings remain available, but no isolated causal speed-up is inferred from
  changing code, import/JIT state and workstation load.
- `workflow-performance-20260917.json`: actual default classical production
  pipeline at nine rays, 2.5 mm steps, CPU and four library threads. Initial run
  took **22.228 s**, including the existing energy-filter stage (**11.555 s**).
  Five exact product reuses had median **0.704 s**, no recalculated products,
  and reused column, diagnostics, elastic, EDS and energy-filter products.
  Five explicit P2 control edits had median **12.682 s**: incident, elastic and
  EDS products reused; column, diagnostics and energy-filter products recalculated.
  This records the actual dependency decisions, without bypassing the filter.
- The same workflow receipt executes a three-point production sweep, cancels
  after one completed point, saves/loads the 2,246,869-byte scalar/input record,
  and resumes exactly the remaining two points. First segment took 15.831 s;
  resume took 30.004 s. Prefix and execution identity were unchanged, and all
  three observations stayed `NOT_RUN` for numerical qualification. Sampled peak
  process RSS during resume was 243,499,008 bytes; 20 ms sampling may miss brief
  peaks and is not a general memory bound. Only one cancellation/resume example
  was measured, so no percentile is reported for this expensive scenario.
- Isolated MainWindow construction in the workflow receipt took 3.341 s.
  Twenty offscreen task-layout changes at 1500 x 920 logical pixels had median
  **55.8 ms**, p95 **71.4 ms**, zero submitted physics calls and unchanged full
  instrument identity. The scene had no completed large mesh; populated-mesh
  rebuilding and native graphics-driver timings are not inferred.

The physical RAM inventory is 102,471,254,016 bytes. This host provides no usable
CUDA device; GPU/kernel/VRAM measurements remain absent, not zero-cost successes.

`input-asset-final-20260917.json` repeats the 0.75 MiB and 48 MiB serialization
fixtures five times per operation against the final source. With allocation
tracing enabled, the large already-admitted input's High request capture median
is **290.0 ms**, with **3.995 MiB** peak incremental traced allocation. Reference
inline encoding needs **114.035 MiB**, while asset encoding needs **3.728 MiB**;
their median times are 276.2 and 277.5 ms respectively. There is no final measured
encoding latency win to claim. Asset decoding takes 984.9 ms / 7.711 MiB versus
inline decoding's 1565.3 ms / 342.034 MiB. Full background High preparation still
takes 7875.6 ms with tracing and peaks at **440.242 MiB** for this 48 MiB fixture;
moving it off the GUI and pinning inputs does not eliminate complete validation
and manifest serialization costs. Small-fixture High capture is 257.8 ms /
3.992 MiB. These are traced operation allocations, not total process RSS or a
guarantee for arbitrarily large models. Both exact array round-trips passed.

The final measurements share implementation identity
`a626f87fde3c1c165816052176f01a6e134ec1515ee88d0a95cd273061a7c842`.

### Final packaging, workspace and process receipt

The final source was built with the existing project interpreter and installed
with `--no-deps` into `tmp/report-installed-final`. No environment or dependency
upgrade was performed. `installation-20260917.json` records the verified target:

- Wheel: `tem_simulator_v2-0.1.0-py3-none-any.whl`, **1,913,357 bytes**.
- SHA-256: `bdf91e2bf959bb37dce62e7eadb07cec365b239dc0df32ed6c72da01639ff548`.
- Every installed `temsim/*.py` recursively matches the current source byte for
  byte; the installed configuration root and new topology reference are present.
- MainWindow was constructed and closed from the installed package with separate
  settings/cache paths. Geometry and saved-experiment pages were present. The
  delayed initial Preview was stopped; no coherent computation was started.
- Logs: `tmp/report_final_build.log`, `tmp/report_final_install.log`,
  `tmp/report_final_install_check.log`. All three operations exited 0.
- Compilation of source and the task's benchmark/validation scripts passed.
  Default-repository `git diff --check` passed. The separate diagnostic using
  `core.autocrlf=false` misinterpreted existing Windows line endings and is not
  used as a whitespace result; no repository line-ending setting was changed.
- No generated `.npy`, `.npz`, `.temwp`, `.temexp`, image, wheel or bytecode file
  is among the unignored new files. Temporary arrays, screenshots, test reports
  and installation targets remain locally ignored. Lightweight scalar receipts
  and input settings remain eligible for review.

All task-started tests, benchmarks, build/install checks and isolated UI checks
have exited. The earlier explicitly identified diagnostic test processes were
the only processes stopped by the task. No user application, acquisition,
hardware state or unrelated process was restarted/stopped. At the implementation
handoff, changes remained in the working tree without a commit, push, pull request
or published-history rewrite.

The guide is **not declared completely accepted**: remaining hardware, model,
measurement and intermittent-startup boundaries are enumerated above. The user
requested that unresolved items be summarized for their next instruction; this
receipt leaves those decisions and the exact resumption scope explicit.

### Publication follow-up

The user subsequently authorized committing and uploading the completed current
step to Git without further pre-upload checks. The recorded test receipts remain
the validation evidence; no new test/build/review pass is claimed for publication.
The unresolved acceptance items above remain open after publication.
