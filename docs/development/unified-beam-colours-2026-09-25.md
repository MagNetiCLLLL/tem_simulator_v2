# Shared beam plot sizes and ray colours — 2026-09-25

The two transverse beam images now use one fixed width/height setting. Both
outer canvases and internal plotting rectangles match. The optional colour
wheel keeps its own size. The old independent source/plane size fields are
replaced by `beam`; layouts without that setting use the shared default.

The Transverse Beam **Plot** choice and Ray Diagram **Colour by** are connected
in both directions. Source position and emission-direction azimuth use the
same recorded tip-emission quantity and colour mapping in all three views.
Advanced emission-angle-to-normal colouring also shares this mapping. Colour
does not change when a path bends, focuses, scatters or rotates in the display.
Missing launch data remain grey; no downstream direction is substituted.

## Time-of-flight meaning

The user's wording referred both to TOF and to distance-dependent colouring.
A clarification was requested; implementation proceeded with the stated
assumption of **cumulative classical flight time**, retaining the TOF name.
Geometric path length was not added. Coherent development remains paused.

Ray Diagram now uses the recorded clock at successive trajectory vertices.
The source and selected-plane scatter plots use the same palette evaluated at
each arriving path's time at the selected plane. The source scatter therefore
shows arrival times mapped back to emission positions, rather than clocks at
the instant of emission. Both plot headings and tooltips identify this.

Colour normalization is 0 to the largest known physically covered time in the
captured result, including named physical filter crossings. It is independent
of the display subset, selected plane and view rotation. The original relative
arrival delay remains available in hover text, but is no longer the colour
quantity. Small arrival differences can occupy the same colour bin on this
global scale. Missing clocks are grey and non-arriving particles are omitted
from selected-plane views.

The palette has 128 bins. Segment endpoints are interpolated at colour-bin
boundaries before the existing screen-space ray simplification, so a straight
line can retain a time gradient. Disconnected paths remain disconnected;
aperture and other stops use the same clipped geometry. Equal-time gun launch
prefixes retain temporal order through repeated/turning Z positions. Inactive
or frozen gun rows do not establish an arrival clock and remain untimed.

Projection changes rebuild the grouped geometry. Ordinary pan/zoom reuses the
groups and the existing bounded ray-rendering caches. TOF shows captured
trajectories rather than adding schematic scan-playback offsets to a clock
that was calculated for another path. Numerical solvers, physics fields,
checkpoint schema, continuation rules and archive precision are unchanged.

## Validation and limits

The focused regression covers 252 distinct cases, with no failures or skips:
shared source/angle colours, missing records, descendant identity, preset
switching, source/plane geometry, size persistence, clipped time interpolation,
turning gun prefixes, physical stop limits, detailed-exit publication into the
same result object, hidden panels, projection, display-cache reuse, existing
wave readers and result-file UI restore/rollback.

Most cases use explicitly synthetic cached trajectories. The result-file GUI
suite also executes its existing small 49-particle fixture, then forbids fresh
transport during opening/restoration. This is focused UI and data-handling
validation, not full-instrument numerical qualification or a full test-suite
rerun. The earlier intermittent live-slider timing issue remains outside this
change's verification scope.

A previous real 49-particle `.temresult` was tried for visual inspection but
rejected by the existing source-identity check: `Solver implementation changed;
historical viewing only`. `solver_source_identity()` currently hashes every
Python module, including GUI modules. That policy was not bypassed or changed.
The offscreen visual check instead uses clearly labelled synthetic cached data.
The failed real-archive attempt remains in the local verification record.

Individual XML reports and screenshots are under `tmp/`; a lightweight receipt
is kept beside this note. No running user application was restarted. No commit
or push was requested for this change.
