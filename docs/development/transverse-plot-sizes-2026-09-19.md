# Fixed Transverse beam picture sizes — 2026-09-19

The Beam analysis / Transverse beam panel now has a **Plot sizes…** button.
Users can set the source picture, selected-plane picture and optional colour
legend widths and heights independently. **Apply** updates the pictures;
**OK** applies and closes; **Cancel** discards unapplied edits. Restore Defaults
fills the default values and waits for Apply or OK.

Sizes include axes and use Qt logical display pixels, following monitor display
scaling. Defaults are 400 × 196 for the source plot, 400 × 360 for the plane
plot and 184 × 184 for the legend. Plot minima are 240 × 160 and legend minima
120 × 120; the maximum is 2400 per dimension. Changing the legend canvas keeps
the angular colour wheel circular and centred.

Pictures retain their sizes when the window changes, the selected plane or
tracking preset changes, new cached results arrive, or the panel is hidden and
shown. A local horizontal/vertical scroll area makes large pictures reachable
without forcing the main window to grow or shrinking the pictures. The size
button stays outside the scrolling content. The existing physical axis units,
aspect constraints, source identities and interaction data remain unchanged.

Sizes are presentation settings saved automatically by the existing named
workspace-layout manager. Each layout can have its own sizes; reopening the
application restores the active layout. Missing legacy settings or malformed
entries use defaults. Restoring a layout does not write during restoration or
schedule a calculation. This does not change source, optics or cache identities.

## Verification

- 86 widget, source/plane mode, TOF display, emission display, hidden-panel and
  embedding tests passed in 27.773 s. Receipt:
  `tmp/transverse-plot-sizes/final-widget-regression.xml`.
- 15 named-layout and persistence tests passed in 99.716 s, including autosave,
  layout selection/reset, hidden-panel restart, legacy restoration and guards
  against changing physical inputs or launching computation. Receipt:
  `tmp/transverse-layout-20260919/persistence.xml`.
- Syntax checks passed for `src` and `tests` after the final test runs.
- Offscreen screenshots were inspected for default/custom sizes, the settings
  dialog, a 350 × 500 window and a resized colour legend. Actual widget sizes
  matched the requested values; the narrow window retained its dimensions and
  exposed both scrollbars. Images and dimension records are under
  `tmp/transverse-plot-sizes/`. These are artificial cached UI fixtures, not
  evidence of numerical propagation or instrument performance.

All 101 final related tests passed, with no failures, errors or skips. Existing
Pydantic deprecation and pyqtgraph shutdown-disconnect warnings remain. No
physical solver work, coherent development, user-application restart, commit or
push was performed.
