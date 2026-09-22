# Unified transverse tracking plots - 2026-09-20

The normal tracking view now has one Plot selector. The separate selected-plane
position/angular selector and its callback were removed. A single preset table
defines both the lower coordinates and colour quantity; switching a preset
updates both before drawing once.

| Plot | Upper coordinates | Lower coordinates | Shared colour |
| --- | --- | --- | --- |
| Source position | Actual tip X/Y, nm | Selected-plane position, micrometres | Original source-position azimuth |
| Emission direction | Actual tip X/Y, nm | Selected-plane angles, mrad | Original launch-direction azimuth |
| Time of flight | Arriving paths at their tip X/Y, nm | Selected-plane position, micrometres | Per-path arrival delay at this plane |

The selected-plane coordinate readout is text, not another selector. Independent
analysis controls remain explicitly under Advanced diagnostics for intensity,
histograms, interactions and phase-space inspection. A custom coordinate/colour
combination cannot be labelled as a normal tracking preset. Selecting a normal
preset always leaves diagnostics and restores its complete definition.

Launch colours follow recorded source identity through focusing, clipping and
scattered descendants. The upper angle-coloured plot retains actual positions;
it does not manufacture an angular distribution or force uniform mixing.
Angles below are current projected propagation angles, not original launch
angles. Projection rotates coordinates without redefining launch hues.

TOF retains each arriving path separately, including different times from one
source ancestor. Both plots use the same path colours and the time scale uses
all timed arrivals before display subsampling. Changing the plane can change
arrival delay; this is classical time, not coherent phase.

## Validation

- 179 related cases passed across nine modules, with no failures or skips:
  `tmp/transverse-unified-20260920/core.xml` (151, 14.492 s) and
  `tmp/transverse-unified-20260920/tracking.xml` (28, 6.411 s).
- Coverage includes atomic preset changes, coordinates and units, shared colours,
  explicit diagnostic combinations, missing launch/timing data, descendants,
  cache reuse without result mutation, fixed plot sizes and restoration after
  reading an already executed wave checkpoint.
- Offscreen visual inspection of the three normal modes used a synthetic
  512-particle display fixture; no physical gun or column solve
  was run. Preview: `tmp/transverse-unified-20260920/three-tracking-views.png`.
- Changed Python files compile and `git diff --check` passes. Numerical test
  jobs used two threads each and serial nested libraries, below the CPU budget.

This supersedes the independent lower-coordinate control documented in
`transverse-source-tracking-2026-09-19.md`. Existing generated results were not
modified. Coherent development remains paused. Restart the running application
to load these display changes; no Git commit or push was performed.
