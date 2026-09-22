# Ray Diagram navigation performance — 2026-09-20

Dragging and zooming a completed High accuracy result spent most of their
time rasterising the full stored polylines. Ordinary viewport navigation did
not rerun gun/column/specimen/EDS calculations or rebuild Beam Analysis.
Source-position colour grouped 432 selected rays into 134 graphics items,
but those items still carried 1,171,088 path points. A representative baseline
profile spent about 90% of its two-paint frame inside QPainter.drawLines.

## Implemented

- `gui/ray_curve_item.py` retains the full original display-source coordinates
  and draws a screen-resolution subset in its child Qt curve. Each finite ray
  is separate; launch/stop endpoints and missing-data gaps are preserved.
  Viewport clipping retains crossing-edge neighbours. Turning paths are never
  sorted or treated as monotonic Z data.
- The point-to-segment simplification bound is 0.35 **Qt logical pixels**, using
  conservative dyadic X/Y scales. Zooming and resizing restore required detail.
  This is a display tolerance, not a changed physical integration tolerance.
  Original `getData`, Fit bounds, export coordinates, colours, selected rays,
  particle weights, angular statistics and saved calculation states are unchanged.
- `gui/ray_curve_simplification.py` compiles the RDP traversal as one CPU thread,
  with no parallel pool or fast-math. An independent NumPy implementation remains
  the fallback when compilation is unavailable. Native and NumPy subsets agree
  in the tested smooth, narrow-tail, reversing and noisy fixtures.
- Each colour item retains at most eight scales and 4 MiB of simplified arrays;
  new coordinates invalidate them. These are internal static limits, not a live
  user cache setting. Oversized entries still render without cache retention.
- `visualization.py` uses the new item for both colour modes, caches navigation
  bounds and the most recent visible-slope window in its existing bounded display
  cache, and skips unchanged scale-notice text. Result publication, changed array
  identities, projection and detailed specimen-exit selection invalidate reuse.

No physical solver, cross section, source, detector interaction, calculation
cutoff, continuation dependency, or stored particle state was changed. The
previously identified large-angle/paraxial propagation limitation is unchanged.

## Real saved-trajectory replay

The 15,000-particle completed archive with identity
`a8283b7991c02d10c2d7ac4e142c33fd89029cf9a6df5705c4db861a5f1bb929`
supplied nine branches and the existing deterministic 48-ray subset per branch.
The input subset digest matched before/after. Only selected stored numeric
arrays were read; this was not a full archive checksum validation or physics
rerun. Generated arrays and screenshots remain under ignored
`tmp/ray-navigation-20260920/`.

The same 1200×650 offscreen PlotWidget, projection 298.3°, pens and paths were
used for both runs. Each scenario has twelve measured frames. The primary
number below is the median forced QImage raster redraw, including actual Qt
painting rather than merely queuing a range change.

| View movement | Original redraw | Optimised redraw |
| --- | ---: | ---: |
| Horizontal pan | 420.26 ms | 8.82 ms |
| Zoom | 377.84 ms | 11.54 ms |
| Vertical pan | 438.18 ms | 11.11 ms |
| Full-column pan | 507.63 ms | 5.50 ms |

The harness also processes the range event and forces a second image paint;
its combined medians became 28.22 / 46.67 / 36.17 / 24.83 ms respectively.
Those combined timings contain two paint deliveries and are not application
frame rates. First visits to new zoom levels took at most 63.19 ms in this run.
The first native compilation plus item creation cost 969.06 ms once, against
64.95 ms for original item creation; initial paint then fell 754.78→57.54 ms
in the two-paint measurement. The cold compiler cache was isolated explicitly.

An intermediate NumPy-only version improved repainting but left cold zoom
near one second; it was replaced by the compiled traversal before delivery.
The final compiled and NumPy-rendered PNGs were byte-identical at the recorded
view. Both were visually checked against the original, including the prominent
yellow, purple and blue scattering tails. This is evidence for this view, not
a universal image-equality claim.

The benchmark isolates ray rendering: it excludes apparatus labels, optional
panels, desktop compositing and monitor presentation. Complete visible Windows
interaction performance has not been measured. Dragging the cyan selected-Z
cursor is a separate path that refreshes transverse analysis, outside this
ordinary pan/zoom change.

Measured code hashes and timing detail are retained in
[the compact report](ray-navigation-performance-2026-09-20.json). Reproduction
script, profiles and complete receipts are local in the ignored directory.

## Validation

- Final compiled implementation: **58/58 passed**, no skips/errors, 69.220 s,
  `tmp/ray-navigation-final-kernel.xml`. Covers simplification error, narrow
  excursions, gaps, reversals, stops, Fit, resize, export, native/fallback parity,
  disabled cache retention, metadata invalidation, colour modes and incremental
  scene publication.
- Earlier expanded selection: **188 passed / 11 failed**, 199 total, 367.283 s,
  `tmp/ray-navigation-regression.xml`. This run preceded the final compiled
  traversal. All failures are in existing `test_gui_shell.py` cases: filter
  installation assumptions/navigation, alignment worker/status expectations,
  component selection, and a raw result with `signatures=None` failing before
  ray drawing in `ResultReadout.publish`. These must not be described as a
  fully passing application regression.
- Before-optimisation control: restored the old PlotDataItem and original
  navigation callbacks in a test-only process, with temporary settings and a
  guard against physical calculation. All ten non-solver GUI failures recurred
  at the same assertions. The remaining 25-ray case was deliberately skipped;
  a pure-widget fixture reproduced its unchanged `signatures=None` readout
  exception before drawing. Receipt: `tmp/ray-navigation-before-control.xml`,
  10 expected failures / 1 reproduction passed / 1 skipped, 57.988 s. This
  establishes that these failures are not introduced by the navigation change;
  their underlying test/compatibility issues remain unresolved.
- Changed Python files compile; `git diff --check` passes (existing Windows
  line-ending notices remain). No commit/push or application restart performed.
