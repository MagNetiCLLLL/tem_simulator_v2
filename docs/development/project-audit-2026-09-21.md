# Project audit - 2026-09-21

Follow-up: all five findings and the packaging risk were fixed and verified in
[the repair report](project-audit-fixes-2026-09-21.md). The observations below
describe the original read-only audit, before those fixes.

Status: completed. Fresh full regression: **5,868 passed, one skipped, zero
failures and zero unexecuted cases**. Independent review confirmed five
functional defects outside the existing test coverage and one local packaging risk. Production code and existing tests are
unchanged by the audit. Existing uncommitted work is preserved.

## Confirmed findings

### P1 - Filter output plots can display the unfiltered population

Only the TOF route selects executed filter-plane data in
[`beam_analysis.py`](../../src/temsim/gui/beam_analysis.py#L491).
The ordinary angular sampler and the separate position path in
[`diagnostic_tabs.py`](../../src/temsim/gui/diagnostic_tabs.py#L4697)
instead read straight-column histories.

At the same named filter output, a deterministic saved-data fixture displays two
actual filter arrivals at X = [1, 2] micrometres and 30% of source in TOF mode.
Switching to source position displays 16 incident rays; angular mode displays
16 rays and reports 100% of source. No calculation or selected component changes.
This is a display/data-routing defect, not evidence that the underlying filter
transport was recalculated incorrectly.

All observables should share physical-plane selection and the filter's local
coordinate frame. Missing physical crossings must remain unavailable rather
than borrowing straight-column rays. Reproduction uses saved-data fixtures,
not a newly executed physical filter calculation.

### P2 - Overwritten archives can be reported as another saved result

[`calculation_controller.py`](../../src/temsim/gui/calculation_controller.py#L774)
accepts a cached identity-to-path record whenever the file still exists.
Successful saves at line 822 do not invalidate older identities sharing that
path.

A real 49-particle Preview section A stopped at 1629.2 mm, then continued as B to
1630.2 mm. A and B were saved successively to the same file. Reading that file
confirmed B, but requesting automatic archival of A reported A as saved/reused
at 1629.2 mm and queued zero file jobs. The displayed archive endpoint can
therefore disagree with its actual contents.

Invalidate records on same-path replacement and verify file identity before
reporting reuse. These are real executed particle sections and actual archive
save/load operations; the controller's completion dispatch was driven directly
to keep the reproduction bounded.

### P2 - Ordinary calculation cancellation is not consumed at sample progress boundaries

[`CalculationWorker._report_progress`](../../src/temsim/gui/calculation_controller.py#L601)
only emits progress. The ordinary pipeline's sample callback at
[`simulation_pipeline.py`](../../src/temsim/simulation_pipeline.py#L742)
does not check cancellation either. EDS and geometric STEM use the same
non-checking progress route. The section pipeline at line 1163 does check it.

In a real 32-particle elastic material calculation, cancellation was set at the
first 0/32 callback. All 32 particles still completed, with 33 callbacks. An
explicit check of the same state's token immediately raised the expected
cancellation exception. The dynamic reproduction covers elastic transport;
EDS/STEM share the problematic callback wiring but were not separately timed.

This delays cancellation and holds the shared numerical-job slot. No evidence
was found that a cancelled result is published as current. Check cancellation
at safe progress boundaries rather than forcibly interrupting an active kernel.

### P2 - Camera pixel count accepts zero before later state validation fails

[`validate_runtime_assignment`](../../src/temsim/runtime_parameters.py#L255)
does not enforce a positive camera pixel count. The operating-parameter editor
and profile application both use this validator.

The actual offscreen Camera operating table accepted 2048 -> 0, emitted
`runtime_changed` and no error, and left the live camera at zero pixels. The
next `State.from_dict(state.to_dict())` failed with
`Camera pixel count must be positive.` Applying the same value through a
profile reproduced the inconsistency. Reject non-positive counts at the shared
assignment boundary, before committing them to live state.

### P2 - Experiments workspace selection omits a read-only status refresh

[`WorkspaceLayouts._apply`](../../src/temsim/gui/workspace_layouts.py#L226)
blocks tab signals while restoring a workspace. This suppresses the Design
Explorer entry callback in
[`main_window.py`](../../src/temsim/gui/main_window.py#L768), without replacing
its read-only refresh.

In an actual offscreen MainWindow with dirty explorer status, selecting
Experiments displayed the page but scheduled zero refreshes. Leaving and
re-entering through the tab scheduled one. Explicitly schedule the status refresh
after layout restoration while continuing to suppress calculation requests.

## Packaging risk

Building directly from this checkout's existing build directory succeeded but
included ten Python modules already removed from current source, including
`effective_axis.py` and `presets.py`. A separate clean build contained exactly
the current 446 Python modules, with no missing or extra modules. Both included
36 packaged configuration files and nine SVG assets.

This is stale local build-output contamination, not a claim that those modules
are still active in the checkout. Release builds need a clean staging directory
and a source-versus-wheel inventory check. The clean wheel was installed only
in an ignored temporary target. Its main window opened with 13 pages and the
installed configuration root; automatic transport was disabled for this smoke.

## Verification and scope

All 458 current test files were executed in four non-overlapping groups. A
separate fresh full collection reconciles every one of the 5,869 current test
IDs with the execution ledger. All four groups exited successfully; no retest
replacement or test expectation edits were needed in this audit.

| Group | Cases | Passed | Skipped | Exit | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,742 | 1,742 | 0 | 0 | 1021.771 |
| 1 | 540 | 539 | 1 | 0 | 1010.155 |
| 2 | 1,794 | 1,794 | 0 | 0 | 1224.004 |
| 3 | 1,793 | 1,793 | 0 | 0 | 1281.300 |

The single skip is
`test_direct_alignment.py::test_fixed_lens_aperture_scaling_matches_recalculated_toml_metrics`:
its absolute reference values predate the C2 aperture relocation. That reference
is not considered validated. Existing tests pass while the five independent
reproductions above expose missing regression coverage.

The 1,051 source/configuration/test/build-input files hashed at audit start
were byte-for-byte unchanged after execution. Their inventory digest is
`d42b6b70f4176577a465db7f70ccf2e84b6eb7a97219efa3f00a02795bb49aed`.
Serial compilation of `src`, `scripts`, `tests` and `main.py` passed after all
tests exited. `git diff --check` also passed. The
[machine-readable report](project-audit-2026-09-21.json) records counts, source
identity, packaging inventories and finding locations. Per-test receipts remain
in `tmp/project-audit-20260921/shard-{0,1,2,3}/`; the reconciled ledger is
`tmp/project-audit-20260921/final-ledger.json`.


Environment checks found no broken installed requirements. Importing `main`
created no QApplication. Static inspection parsed 1,011 Python files and
resolved 18,786 explicit calls, with no syntax or definite argument-binding
errors. Dynamic dispatch and state validity still need runtime checks, as the
findings demonstrate.

The complete suite ran in four isolated settings/cache workspaces, two
numerical threads each and one thread per nested numerical library. Independent
reproductions used one numerical thread each, remaining below the host's
16-thread ceiling. No running user GUI was restarted or terminated. No coherent
production calculation or new 5,000-particle whole-instrument benchmark was run.

Detailed receipts are local under `tmp/project-audit-20260921/`. The archive
overwrite evidence and executed file are in `tmp/audit-persistence-20260921/`.
Generated caches and numerical outputs remain excluded from Git. This audit
does not certify the complete physical microscope model or commercial-equipment
performance. No repair, Git commit or push is included.
