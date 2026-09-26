# Virtual electron navigation and response — 2026-09-26

## Interaction

The spatial magnetic-field and electron-trajectory views now have physical
Z and projected U axes. U = X cos(theta) + Y sin(theta), using the existing
Ray Diagram projection angle. Both navigation interfaces use millimetres
internally and SI-prefixed metre labels. Stored physical XYZ remains metres;
the display enlargement does not enter the axis values.

- Wheel inside the plot scales both axes about the pointer.
- Wheel over the bottom or left axis scales only Z or U. Axis dragging pans
  only its direction. The wheel factor and anchor match pyqtgraph's Ray Diagram.
- **Link Ray Diagram** shares both physical ranges in either direction.
  Disable it to keep an independent view; re-enable it to adopt the current
  Ray Diagram range. Workspace layouts retain the choice.
- Fit establishes an explicit viewport. New paths, clearing a pending path,
  background changes and resizing preserve that viewport until another view
  action. Source range limits are respected when linked.
- The 2D combined-field profile retains its tesla axis; fitting it does not
  alter the Ray Diagram's transverse range.
- **Start at view centre** uses the actual current Z interval, including in
  independent mode. Navigation itself never changes the electron's initial Z.

Camera operations use existing physical paths and do not call a field provider
or integrator. The software QPainter renderer is retained. Tick spacing reuses
pyqtgraph's axis helpers; screen transforms and physical geometry stay separate.

## Parameter editing

A number typed over several keystrokes commits on Enter or focus loss. A held
slider commits on release. Each intermediate change still invalidates the old
path and cancels obsolete work, but does not launch a new integration. The
existing 150 ms per-record coalescing remains. Other requested overlay records
can continue. Switching records, hiding the dialog and hiding the view release
edit transactions, and stale results cannot be published.

A separate dialog bug was reproduced: Enter could activate the default Reset
to tip action and overwrite a newly entered number. Action buttons no longer
take Enter away from the numerical editor.

An offline Qt fixture with 220 ms between keystrokes launched six calculations
for the prefixes of `250000` before this change. It now launches only the final
submitted value. This fixture tests dispatch with synthetic trajectory results;
it does not measure physical integration accuracy or speed.

## Equivalent numerical optimizations

The integrator, step budgets, tolerances, physical fields and stopping rules are
unchanged. Three measured hot paths were optimized:

1. Single-point calls use the same bilinear potential and exact-gradient
   interpolation without setting up general array broadcasts. Arithmetic order
   is preserved, including squared-radius multiplication. Boundary checks and
   the enclosing closed-field conductor logic remain active.
2. Hardware bores are first filtered by overlap with the chronological segment's
   axial interval. All surviving candidates use the original exact intersection
   code; radial, tapered, reverse and transverse contact behavior is retained.
3. Gaussian normalization peaks use a bounded 128-entry cache. Its key includes
   width, every Gaussian amplitude/sigma/offset, normalization flag, sample count
   and the scalar numeric types. Changing consumed field inputs invalidates
   reuse. The sampled normalization formula is unchanged.

## Actual configured-field timing and equivalence

Serial measurements used one numerical CPU worker, one BLAS thread and one
Numba thread on a host with 32 available logical CPUs. Both traces started at
the configured flat tip at 0.3 eV. Maximum spatial step was 1 mm, the budget was
20,000 steps, relative tolerance 1e-4 and position tolerance 1e-12 m.

| Path | Before | After | Accepted steps |
| --- | ---: | ---: | ---: |
| Forward default, 3.0264 m | 13.2185 s | 7.1321 s | 8,799 |
| Polar 5°, azimuth 45°, 550 mm | 7.9190 s | 3.7939 s | 3,566 |

These are individual trajectory timings in an already prepared E+B scene,
excluding application startup, initial field preparation and GUI painting.
They are bounded examples, not a universal response-time guarantee. Source or
field changes can still require preparation; repeated electron parameter edits
reuse the scene.

Both paths reached their requested length. All stored positions, momenta,
times, accumulated path lengths, kinetic energies and electrostatic potentials
were **bitwise identical** before and after. Termination, step counts, flight
times and maximum static-energy invariant errors were unchanged. Those errors
were about 1.96e-8 eV (full path) and 1.11e-8 eV (off-axis path). The separately
profiled off-axis run also matched all six arrays.

Local evidence, including inputs, full arrays and profiles, is retained under:

- `tmp/test-electron-performance-baseline-20260926/`
- `tmp/test-electron-performance-optimized-20260926/`
- `tmp/test-electron-performance-optimized-20260926/full_array_comparison.json`

The numerical arrays and caches remain excluded from Git.

## GUI evidence and test scope

The final combined affected-feature regression passed **309 tests**, with
**one explicitly deselected existing failure**, in 97.15 s. Receipt:
`tmp/test-electron-navigation-performance-final.xml`. It covers the particle
solver, field scene, planar interpolation, condenser normalization, performance
equivalence, controller editing, canvas/page navigation, linked views and
workspace persistence. Changed Python files compiled and the fatal-error lint
selection passed; existing whole-file style findings were not reformatted.

`scripts/validate_test_electron_navigation.py` replays the stored 3,567-point
off-axis result through real Qt wheel events. It checks plot/bottom/left wheel
behavior, linked and independent ranges, re-linking, resize and unchanged
physical-array digests while forbidding a field solve or trajectory calculation.
This is a replay check, not another physical calculation. Captures and receipt:
`tmp/test-electron-navigation-20260926/`. Additional 900×500 and 420×300 canvas
captures are in `environment/magnetic_navigation_checks/`.

One existing test, `test_default_electrodes_use_actual_voltages_and_keep_analytic_default`,
expects the former analytic default provider. The current configured flat tip
uses `ClosedGunField`; that test fails equally with the old interpolation,
uncached normalization and unfiltered bore loop restored in a separate control
process. It is excluded from this change's combined regression, not counted as
passed. Its default-model expectation was not changed here.

The work does not qualify an entire microscope or add specimen, detector,
electron-electron or coherent-wave physics to the virtual-electron diagnostic.
