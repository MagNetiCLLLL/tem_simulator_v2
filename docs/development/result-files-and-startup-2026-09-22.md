# Compact result files and startup results — 2026-09-22

Status: implemented; result-file checks pass. The broader regression retains
intermittent live-slider timing failures, described below.

The user requested direct result export/import, smaller files, and reusable
named settings/results that can be opened at startup without repeating the
calculation. The chosen export contract retains exact segment continuation.
Coherent development remains paused.

## Workflow

- Export the displayed completed classical result together with the inputs that
  produced it. New edits that have not been calculated are not mixed into it.
- Open a result by restoring an independent editable state and the recorded
  observables. Opening does not execute the gun, column, specimen or detectors.
- Save a completed result under a user-defined name and optionally select it
  as the startup result. There are no invented precomputed demonstrations.
- Parameter-only operating profiles remain separate from computed results.
- A failed open keeps previous state/results and does not silently calculate a
  replacement. Out-of-order background load completions cannot replace a newer
  request. Actual resume admission still checks physical/numerical dependencies.
- While loading, independent Live tuning and open editing dialogs are also
  disabled. A successful load restores the actual calculated and resumable
  endpoints; live tuning remains paused until explicitly requested.

## Storage contract

The `.temresult` extension uses the current checked classical particle package;
no schema migration or pickle loader is introduced. Manual exports and named
results use DEFLATE level 1 and share only byte-identical arrays with identical
dtype and shape. Hash matches are verified by bounded byte comparisons,
including signed zero, NaN payloads and non-contiguous inputs. No precision
reduction, particle resampling or trajectory thinning is used.

The existing complete histories are consumed by plots, backtracking and
dependency-checked continuation. They are not discarded as merely visual data.
Input-derived layout and assembly remain reconstructed from the saved input
snapshot, independently of the result arrays. Automatic internal archives keep
their uncompressed low-overhead policy; compact export is a deliberate file
operation, rather than extra compression on every completed calculation.

The named-result index contains only scalar metadata and a startup selection.
It is atomically replaced under a bounded cross-process transaction lock,
validates its schema and identifiers, and never reads large arrays to populate a
menu. Updating a name preserves its identifier and
startup selection; previous files are not deleted. The library lives beside,
not inside, the disposable artifact cache. Generated `.temresult` files are
excluded from Git.

Live result/progress events only update action availability. Reading the named
library and rebuilding its menus happens when the File menu opens or a library
setting changes. A synthetic 1,000-entry UI measurement reduced the availability
refresh median from 47.24 ms to 0.003 ms, with zero library reads on that path.
This is a menu-only measurement, not a transport performance result.

File requests bind the initial package digest as well as their path and operation
token. A queued file replacement is rejected before loading numeric arrays.
Named and startup entries also bind the package digest recorded when saved.
Export captures mutable result records, metadata and input controls on the
owning thread before queueing. Later sample/EDS enrichment cannot change that
request or leave the export button waiting for a different result identity.
Completed numerical buffers follow the existing read-only publication contract;
the background worker freezes and checks their bytes rather than duplicating
whole histories on the GUI thread.

## Size and scientific preservation evidence

For a newly executed 49-particle Preview section, the original save strategy
produced **7,157,409 bytes**. Compact export produced **4,328,261 bytes**, a
**39.53% reduction**. Array entries fell from **126 to 85**. Every retained field
and numeric byte was restored exactly, and a real continuation reused the
executed gun and upstream column. This is a bounded fixture, not a guaranteed
size ratio for every 5,000-particle calculation.

A separate ZIP-level measurement of existing current-format archives found
39.3% and 66.2% reductions with DEFLATE level 1. Those historical files were
inspected and recompressed without changing any entry bytes; incompatible
solver identities were not bypassed or activated. Higher compression levels
saved little extra in those samples and took longer.

A separate, genuinely executed High accuracy fixture used 49 particles, a
2 mm column integration step and a cutoff at Z = 1629.2 mm (30 mm after the
specimen reference). It used a flat tip with specimen insertion, EDS, scan,
energy filtering and waves disabled. Calculation took 11.739 s, compact save
0.722 s and complete GUI load/display 5.232 s (file IO: 1.228 s). The file was
4,328,049 bytes. The loaded High accuracy slot was current, writable controls
were independent of the retained snapshot, and forbidden transport entry
points were never called during loading. These are bounded workflow timings
collected during concurrent regression, not a 5,000-particle performance claim.
The toolbar and File menu were visually checked in offscreen captures.

## Verification

Focused evidence includes 69 codec-related tests, 46 result-library tests and
four request-ordering tests.
The six compact-codec tests were rerun to record size properties; that rerun is
not counted as six additional distinct cases.

Independent review first reproduced lost updates between separate library
instances/processes, a queued file replacement accepted as a direct Open, and
result enrichment changing a queued save's identity. Regression tests cover
those failure paths. Library tests also cover timeout, atomic write failure,
and automatic lock release when a process exits without running cleanup.
GUI coverage includes actual compressed-file background loading with gun and
column entry points forbidden, startup dispatch without preset calculation,
independent editable inputs, superseded requests and late display rollback.

A fresh complete regression executed all 6,000 collected cases with inputs held
fixed: **5,998 passed, one historical-reference skip and one failure**. The failure
is `test_real_ray_frames_arrive_during_sustained_slider_motion`: the requested
final frame did not arrive within the original 15-second limit. No cases were
left unexecuted. The skip concerns stored metrics predating the C2 aperture move.

After removing repeated menu IO/rebuilding, all 12 result-file GUI tests and
19 section-archive GUI tests passed. The seven-test live-slider module still
had its same final-frame timeout: **37 passed / one failed** across the three
modules, 167.555 s. Only the result menu module and its two new tests changed
after the complete regression. The source files and original 30-/15-second
limits of the slider test were not modified to obtain a pass.

Standalone attempts show variable first-frame/final-frame timeouts. Observation
confirmed zero export-capture calls and no automatic-preview hold during that
test. A committed HEAD copy passed once, but an initial resource comparison was
not matched: the broad runner used two numerical threads, whereas the baseline
and menu follow-up used one because OpenMP supplied the lower limit. Separate
LOCALAPPDATA folders also imply fresh revision-specific native-code caches.
An additional current-version single-case run with the same actual one-thread
budget, serial BLAS, fresh caches and unchanged time limits passed: 51.590 s
test body / 56.302 s total, versus HEAD's 48.618 / 54.457 s. The menu follow-up
module run also used one thread, so thread count alone does not explain its
failure. These results do not establish a new transport regression or a reliably
passing latency contract; this limitation must not be described as an all-green
suite.

Final collection contains 6,002 cases, including the two added menu checks;
every case was executed in the full run or affected-module follow-up. Compilation
of source/tests/scripts and dependency checks passed. The frozen final input
set contains 1,063 files; its SHA-256 is
`d439ee50e2ed42e3de904ae316093b6e6f102f1a443b688e280d9a001fca42d5`.

The final wheel contains all 449 current Python modules byte-for-byte, 36
configuration files and nine SVG assets. A fresh isolated installation opened
the GUI with all 286 loaded project modules coming from that installation;
the result actions, shortcuts, empty-result availability and File menu passed,
with no submitted calculation or GUI error. Startup optical calibration was
stubbed and physical calculation entry points were forbidden, so this checks
packaging/UI integration only. The separate real-calculation fixtures above
cover result restoration. Final wheel SHA-256:
`bf144f1735cc3bdc797b239e9e79b0ce7129a3867476c9cc4003ae3fb60e6e2d`.
Screenshots were visually inspected. The lightweight machine-readable record is
[result-files-verification-2026-09-22.json](result-files-verification-2026-09-22.json).

## Local evidence

- `tmp/result-files-20260922/full-run-before-menu-optimization.json` and
  `full-run-ledger.json`: complete regression with the original failure retained.
- `tmp/result-files-20260922/verification-summary.json`: final collection,
  affected-file comparison and follow-up coverage.
- `tmp/result-actions-hotpath-20260922.xml`: all three affected GUI modules.
- `tmp/live-slider-budget1-current-20260922-a/`: matched-budget single-case
  environment, pass and timings; no timing limit changes.
- `tmp/high-result-file-20260922/receipt.json`: actual High accuracy save/load
  workflow, forbidden transport checks and offscreen screenshots.
- `tmp/archive-size-audit-20260922/`: exact restoration, continuation and size
  measurements; generated packages remain local and excluded from Git.
- `tmp/result-files-20260922/packaging-final/`: final wheel/source comparison,
  isolated installation and installed GUI smoke receipts and screenshots.
