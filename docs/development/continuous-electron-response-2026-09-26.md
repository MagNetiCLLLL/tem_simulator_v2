# Continuous virtual-electron response — 2026-09-26

## User-visible behaviour

Virtual-electron sliders now update while held. Their previous trajectory stays
visible as a dashed comparison; actual integrated prefixes and completed sampled
settings replace it progressively. No interpolation invents intermediate physical
paths. An earlier sample is explicitly labelled and cannot displace an exact
current cached result. On release, the latest settings receive a final calculation.
Numeric typing commits on Enter/focus loss as before.

The polar-angle editor uses **mrad**, with the equivalent degrees in the same row.
It spans 0 to pi × 1000 mrad, measured from +Z. Internal diagnostic settings retain
their established degree convention; cache identity is based on executed settings.
Unedited coordinates and energies are not rounded through the editor.

Changing captured fields or removing an electron discards its previous display
and invalidates its work. Keeping an old same-field curve while editing is only
a display aid, never a new electron source or continuation checkpoint.

## Execution and drawing

- One captured E+B scene and persistent hidden compute process are reused.
  The process-wide numerical admission and existing half-logical-CPU ceiling
  remain in force. Single-electron integration remains sequential and uses one
  numerical worker; compilation removes Python overhead instead of multiplying
  redundant integrations.
- Input starts after a 60 ms coalescing interval. While a slider is held, a sampled
  request gets up to 350 ms before newer settings cancel it. Release immediately
  requests the latest value. The window includes admission, IPC and display
  latency; a 120 ms window was shown to starve real updates despite fast numerics.
- Progress contains immutable, chronologically accepted states, normally emitted
  after about 80 ms of integration. The compiled loop returns to Python every
  64 accepted steps for cancellation and publication. Prefixes never enter the
  exact-result cache. Cancelled IPC streams are drained before the next request.
- The dense, unchanged magnetic background is rasterized once per physical camera
  and layout state. New electron frames reuse that display layer. Geometry,
  projection, pan/zoom, device pixel ratio, plot edges, resize and presentation
  mode invalidate the layer; hardware fields and physical coordinates are unchanged.

## Numerical scope

The accelerated path packs the existing closed axisymmetric electric grid,
supported analytic magnetic lenses, quadrupoles, hexapoles, deflectors and gun
coils. It preserves extraction, electrostatic focusing, acceleration, apertures,
electrode/bore contacts and explicit unsupported-field boundaries. The original
field interpolation, relativistic discrete-gradient update, step doubling,
spatial limits and tolerances remain in use. Fastmath and parallel arithmetic
are disabled.

Admission requires supported concrete classes and their original field-method
identities. Curved/custom/mapped/Wien providers or unrepresented geometry keep
the complete reference calculation. An unavailable compiler also keeps that
calculation; compilation errors log an explicit fallback. This is an alternative
numerical implementation, not retained legacy source settings or omitted physics.

Reference comparisons include full-axis transport, off-axis aperture stops,
oblique propagation and return towards the cathode. Nonzero stigmator,
quadrupole, rotated hexapole, deflector and gun-coil fields are checked individually.
For the measured default full path and 2.61-degree case, step counts and physical
termination reasons match the reference. The largest reported endpoint difference
was about 2.3e-13 m, with arrival-time difference about 9.7e-22 s. Energy-invariant,
full saved-array agreement, hardware-contact and streamed-prefix checks are
covered by the targeted tests. This does not qualify an entire microscope or
resume paused coherent-wave work.

## Measurements and reproduction

Measured warm single-trajectory integration, with the same numerical settings:

| Case | Reference | Compiled | Accepted steps |
| --- | ---: | ---: | ---: |
| Default on-axis, 3.0264 m | 7.52 s | 0.202 s | 8,799 |
| 2.61-degree off-axis, ending at aperture | 4.69 s | 0.096 s | 3,862 |

These integration timings exclude process startup, field preparation, painting
and first compilation. First compilation took approximately 9 s in this run;
later calls reuse compiled kernels. A process restarted with a populated disk
cache took about 3 s for preparation plus its initial path. Complex unsupported
providers retain their original runtime rather than being approximated for speed.

The actual Qt dock/canvas and process benchmark uses the default captured fields,
0.3 eV emission, X near -2.65 nm and polar angle 45 mrad. It sends 80 distinct held
slider values, verifies new integrated geometry before release, checks that the
plot never becomes empty, and waits for the final exact latest-settings result.
It also checks process/scene reuse and process exit on closing. See the final
local JSON for end-to-end timings; these include IPC, painting and input coalescing
and should not be confused with pure integration timings above.

Final measured dock/canvas run: 80 distinct slider values over **1.574 s**,
**6 complete path updates and 7 partial updates**, zero empty display states,
and **0.200 s** from release to the final exact result. Warm parameter updates
including input coalescing and display took 0.415 s for the full path and
0.271 s for the off-axis case. The 10 ms Qt timer had 23.1 ms p95 and 39.6 ms
maximum gaps. Initial process/field setup plus its first path took 3.768 s with
the compiled cache available. Screenshots were checked with a loaded Windows
font; these measurements use offscreen native Qt drawing.

Final combined regression: **447 passed**, zero failures/errors/skips,
**199.494 s**. Compilation and whitespace checks also passed. Coverage includes
delayed publication under sustained input, an earlier live completion arriving
after exact cache restoration, degree/mrad boundaries, raster cache lifecycle,
nonzero field contributions, reference trajectories, hardware contacts, IPC
cancellation and numerical CPU limits.

Run with the project interpreter, `PYTHONPATH=src;.` and
`QT_QPA_PLATFORM=offscreen`:

```powershell
.venv\Scripts\python.exe scripts/validate_continuous_electron_response.py --output tmp/continuous-electron-live-20260926
```

Local evidence (generated files are excluded from Git):

- `tmp/continuous-electron-final.xml`: combined affected-feature regression.
- `tmp/continuous-electron-live-20260926/report.json`: actual Qt/process measurements.
- `tmp/continuous-electron-live-20260926/completed.png` and `during-drag.png`.
- `tmp/test-continuous-electron-old-window-negative.xml`: intentional negative
  control; both delayed-publication cases fail under the former 120 ms window.
- `tmp/continuous-electron-mrad-dock.json` and 360/420/480 px screenshots:
  native angle-row layout verification, without a numerical field run.

The narrow-dock checks found no horizontal overflow or angle-label overlap.
Tests distinguish synthetic GUI scheduling fixtures from real executed fields.
Offscreen measurements are not a monitor refresh-rate guarantee. Restart the
application to load the revised controls, renderer and numerical backend.
