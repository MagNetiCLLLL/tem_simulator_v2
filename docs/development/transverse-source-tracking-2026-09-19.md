# Actual emission-source tracking views — 2026-09-19

This records the first two presets. The subsequent [classical TOF implementation](transverse-time-of-flight-2026-09-19.md)
supersedes the unavailable-TOF status and next-step section below.

## Delivered behavior

The Transverse Beam panel now draws recorded tip emission positions above the
selected-plane view. The old static colour wheel remains available as an
explicit, collapsed colour legend. It is no longer presented as the source.

- **Source position tracking:** actual emission X/Y above, selected-plane
  position below, with the same original position-azimuth colours.
- **Emission angle tracking:** actual emission X/Y above, selected-plane
  angular X/Y below, coloured by the original launch-direction azimuth.
- The lower coordinates can independently switch between position and angle.
  The preset controls the tracked origin quantity, so an optical crossover,
  aperture or specimen interaction does not reset a particle's colour.
- **Time of flight** is visibly unavailable. Full path timing is not yet
  transported through every stage; no arrival time or wave phase is invented.
- Existing intensity, angle-distribution, interaction and phase-space views
  remain under **Advanced diagnostics**. The reader for already executed
  historical wave checkpoints is preserved; no coherent calculation is enabled.

Hue encodes azimuth only, not angular magnitude, radius, current or probability.
The launch angle to the local surface normal is available on hover and as an
existing advanced colour quantity. Angular coordinates below refer to the
selected plane, not necessarily the launch direction. Source coordinates use nm;
lower position uses µm and angular coordinates use mrad. Both follow the shared
U/V projection while physical hues remain fixed.

For a flat emitter with independent position and angle distributions, launch
direction colours naturally mix across the source. Curved-emitter correlations
are retained. Each numerical sample carries one actual direction; no extra
directions or position jitter are manufactured. Exact coincident source X/Y
positions are counted explicitly, with up to eight actual launch records in
hover text and a remaining-record count. Overlaid opaque colours are not claimed
to display a weighted angular probability distribution.

## Data, performance and compatibility

`EmissionSourceData` copies and freezes the original `gun_trace.emission_reference`
once per result publication. It validates source IDs, computes launch quantities
and sorts an ancestry lookup once. Source brushes are reused across redraws.
The upper panel is not redrawn for a Z-only change. Changing the published
result invalidates the lookup and colour cache, even if the wrapper is reused.

Each panel draws at most 2,000 samples. The upper sample comes from the complete
emission record and remains stable after downstream clipping. Matching source
IDs share colours, including repeated descendants. Large downstream descendant
populations can use a different bounded subset; a missing visible upper marker
does not imply missing ancestry or a missing original particle.

Missing original launch positions are explicit. The upper panel never uses a
gun-exit or incident-column position as a substitute tip. Historical lower-plane
position colours remain readable. Direction azimuth can remain available when
an old record lacks normals; the corresponding angle to the normal is unavailable.
Unknown direction values remain grey. Position-only or malformed historical
records cannot define a new active source.

All changes are in presentation and display-data lookup. Emission, extraction,
acceleration, focusing, apertures, sample interactions, detectors, vacuum and
energy-filter transport were not changed. Switching views and fitting either
plot do not call a solver. No claim is made about whole-High execution speed.

## Validation

- Final focused GUI/data regression: **155 passed**, exit 0, in 8.74 seconds.
  Receipt: `tmp/transverse-tracking-20260919/final-ui.xml`.
  This covers source provenance, independent launch quantities, missing/invalid
  records, reordered descendants, clipping, fixed source sampling, publication
  invalidation, brush reuse, projection/coordinate switching, fit, exact overlap
  explanation, historical wave-reader restoration and existing plane observables.
- These are display and cached-particle fixtures, not physical column acceptance.
  The existing wave-reader tests only inspect small preconstructed fixtures.
- Offscreen visual inspection used 3,072 actual default flat-emitter samples,
  with 2,000 displayed. The lower panel fixture repeats the launch values at Z=0;
  no gun or column transport was run. Both source maps were visually inspected,
  including labels, units, mixed angle colours and independent lower coordinates.
  Preview: `tmp/transverse-tracking-20260919/two-source-tracking-views.png`.
- An exploratory full `test_gui_shell.py` run had **63 passed, 9 failed**;
  an additional nine new source-tracking tests passed in that same invocation.
  The nine failures concern main-window historical calibration text, alignment
  workers/progress, lens/aperture navigation, a source-button label and a missing
  result-signature dictionary. Do not describe this as a clean full GUI suite.
- Isolated comparison copied the current source tree, keeping earlier scientific
  changes, but restored the three modified integration modules to their pre-change
  versions (plus the pre-existing PSF wording). Imported module paths and hashes
  were checked. **Eight of those nine failures reproduced before this feature.**
  The worker-error timeout passed both in the baseline subset and on a separate
  current-tree rerun; its failure in the combined run remains unexplained and
  potentially order/state dependent. It is not proven to be a feature regression.
  Receipts: `baseline-gui-shell-nine.xml` (8 failed / 1 passed),
  `current-worker-error.xml` (1 passed), `baseline-provenance.json` and
  `baseline-attribution.json` in the same ignored report directory.
- Serial compilation and whitespace checks passed. The final render was also
  checked at a compact 400 × 900 panel size. No native desktop interaction or
  full-repository pass is claimed.

The running user application was not restarted. Existing uncommitted changes
were preserved. No commit or push was performed; generated previews and numerical
receipts remain in ignored `tmp/`.

## Next: per-path time of flight

Continue from [the timing audit](transverse-beam-assessment-2026-09-19.md).
Accumulate elapsed time inside the existing physical transport, then preserve it
through gun exits, branches, sample descendants, checkpoints and persistent
caches. Do not add a second source or retrace solely to draw a view. Include the
filter only for paths physically reaching it, as the final transport integration.

Use a declared emission event and an explicit selected-plane reference for any
delay. Each descendant may have a different arrival time; ancestor ID alone is
not a time key. Non-arrivals have no arrival time, not zero delay. Validate drift
L/v, oblique path length, relativistic energy-dependent speed, acceleration,
apertures, branch timing, cache round trips and time-step refinement. TOF is not
quantum phase. Coherent tip-to-column development remains paused.
