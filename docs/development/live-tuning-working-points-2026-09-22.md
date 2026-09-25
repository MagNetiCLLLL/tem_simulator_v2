# Working points inside Live tuning — 2026-09-22

The independent Working Points workspace tab has been removed. Its existing
browser and operations now live in **Ray Diagram → Live tuning → Working points**,
beside the **Calculation** subpage. Both subpages use the same dock, which can
still be floated, closed and reopened without replacing captured records,
editable controls or completed results.

The Calculation subpage scrolls its range table and matching live controls as a
single unit. An expanded cutoff section therefore stays usable in a 900 × 700
floating dock instead of forcing the window taller than the available space.

Calculation retains the cutoff plane, participating control ranges, single-run
calculation and compatible segment continuation. Its **Saved result** area now
contains one Open result / Export result pair; the duplicate open button was
removed. Opening is available before capturing a new set of settings, and
bank work disables the result actions while busy.

Working points normally shows the selected-record status, filter, record table,
Load retained data and Restore working point. **Advanced** starts collapsed and
contains the record import/export, portable-input copy, input candidate,
comparisons, illumination apply, fork, undo and A/B tools. Captured parameter
details and Sampling and Convergence are in that same advanced area. Scrollable
content keeps those tools available in a dock-sized window. Expanding the tools
does not run transport or numerical validation.

Working-point `.temwp` actions retain their explicit record/input semantics.
Complete classical results and exact segment continuation still use the
unified `.temresult` Open/Export workflow. No new archive conversion, source
representation, physical approximation or numerical method was introduced.

The named-layout system saves the nested tab selection by its current widget
name. New/reset Results layouts select Ray Diagram plus Live tuning / Working
points; Alignment selects Calculation. Existing saved user layouts are preserved;
Reset adopts a built-in layout's new defaults. The old main-tab title has no
compatibility adapter.

## Validation

All **190 distinct related tests passed**: 148 GUI/section/result-file integration
cases, 12 workspace-layout cases, six new advanced-area cases and 24 existing
record-browser/archive cases. The final 148-case run took 483.57 s and kept all
449 source modules and ten selected test files unchanged. This is focused
integration coverage; the entire repository suite was not rerun, and the prior
full-suite live-slider timing limitation is not claimed resolved by this UI work.

Both source and fresh installed GUI smoke checks passed **16/16**, using an
explicit input-only display fixture with startup optical fitting stubbed and
all numerical dispatch forbidden. Calculation, collapsed/expanded Working
points and Sampling all fit the requested 900 × 700 floating window; their eight
screenshots were inspected. The fresh wheel and isolated installation match
all 449 current Python modules byte-for-byte, with 36 configuration files and
nine SVG assets. Compilation and whitespace checks passed.

The first visual pass exposed a 768 px minimum-height problem; the Calculation
scroll container fixed it and all affected integration/layout tests were rerun.
An early interrupted test run and the first target-install harness failure
(missing target root) remain in the local evidence and are not counted as passes.

Generated receipts and screenshots are in `tmp/live-working-points-20260922/`;
no generated calculation data is part of this change. The lightweight summary is
[live-tuning-working-points-verification-2026-09-22.json](live-tuning-working-points-verification-2026-09-22.json).
