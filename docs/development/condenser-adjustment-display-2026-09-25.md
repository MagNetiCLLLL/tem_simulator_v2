# Condenser adjustment and accelerator annotations — 2026-09-25

## User-visible behavior

- Ray Diagram's **Match transport** is now **Auto-adjust condensers**. Its
  tooltip explains that it changes C1/C2/C3 and automatically applies a validated
  candidate. This is transmission adjustment, not probe/image-focus calibration.
- A selectable, wrapping readout stores and shows C1/C2/C3 before/after values,
  the actual source population and the measured transmission plane. It belongs
  to the displayed calculation, so later live edits do not rewrite its history.
  Other results hide it; malformed metadata receives an unavailable message.
- **Acceleration gaps** draws amber stage markers from the displayed result's
  captured gun geometry. For the analytic field, each band is the field centre
  plus/minus its soft edge, including the configured field-centre offset. For
  the solved curved-tip field only electrode centres are marked; analytic
  transition widths are not used to describe that solved field. Tooltips give
  coordinates. The toggle is retained with the workspace layout.
- Completed optical validation now supplies the range bar with the successful
  straight-column solver extent. It remains an optical reference without
  specimen interactions, not an executed resumable particle section. A filter
  entrance endpoint is not reported as execution through the filter.

The fine bends in the gun were already present in numerical trajectories and
coincide with modelled acceleration transitions. An earlier bounded step-halving
check found a maximum X difference of about 0.955 nm over 50–400 mm for that
default configuration. This is limited numerical evidence, not a general gun
or complete-instrument qualification. The very large transverse display scale
can make these small bends prominent. This change annotates those transitions;
it does not smooth rays or change the physical force/integration model.

## Implementation

- `transport_matching.py` captures before/after controls only on a candidate
  that passes the existing refined validation. The existing detached candidate,
  commit gate and undo flow remain in use.
- `transport_adjustment_readout.py` renders that captured metadata with explicit
  optical-only scope. The message also states that specimen signals are deferred
  and probe/image focus is not calibrated.
- `accelerator_gap_overlay.py` reads the captured snapshot, reuses graphics,
  ignores their bounds for auto-range and accepts no mouse buttons. Toggle and
  result publication do not construct field providers or request transport.
  The resolved snapshot already contains assembly translation; it is not added
  a second time.
- `simulation.py` records `optical_execution_extent` after successful downstream
  propagation with verified endpoints. `ray_extent_data.py` requires that
  explicit receipt and its coordinate/physics scope; it does not infer coverage
  from requested planes or mutable displayed ray tails. Resumable Z stays absent.

## Verification

Environment: project Python environment, offscreen Qt, one or two numerical
workers, single-thread BLAS/OpenMP. The user's running application was not
modified or restarted.

- First targeted run: **94 passed**, no failures/errors/skips, 137.985 s.
  Local receipt: `tmp/condenser-display-20260925.xml`.
- After final readout styling and toggle-persistence assertions: **29 passed**,
  no failures/errors/skips, 23.842 s. This includes the 22 optical-extent cases.
  Local receipt: `tmp/condenser-display-final-20260925.xml`.
- These receipts contain **116 distinct passing cases**. They cover captured
  control values, GUI publication/clearing, stage geometry/graphics caching,
  layout persistence, range scope, malformed receipts, a real nine-ray CPU
  column run and rejection of an incorrect propagation endpoint. The filter
  entrance test is a bounded drift fixture using the actual propagation kernel;
  it is not an energy-filter optics validation.
- The production matching path ran a native search and **193-source-sample**
  validation/refinement, followed by actual candidate application and undo.
  Default-case values were C1 90 → 88.8596205%, C2 35 → 13.9136293% and
  C3 55 → 40.7845901%. All 193 samples crossed the target entrance at
  Z 2586.9 mm; the executed column extent ended at Z 3026.4 mm with no resumable
  section. These are test-case values, not the user's current live settings.
- Native matching completed in 29.699 s on the initial isolated-cache run and
  11.954 s on the subsequent run for final styled screenshots. This is a smoke
  check, not a controlled speed benchmark. GUI checks confirmed gap toggles
  preserve graphics identity, view ranges and ray arrays, and result clearing
  removes the readout/annotations. Full-column and gun-region screenshots were
  visually inspected with the production application stylesheet.
- Seven changed production modules compiled; whitespace checks passed.

Local, ignored native artifacts are under `tmp/condenser-display-20260925/`:
`native-validation.json`, `full-column.png`, `gun-gaps.png`. The harness is
`tmp/check_condenser_display_20260925.py`. Generated artifacts were not added to
version control.

An initial agent-side physical test crashed in the existing native Numba call
at `analytic_particle_batch.py:232`, before the new extent receipt was produced.
Using a separate `NUMBA_CACHE_DIR` completed the affected tests and both native
matching checks. The cause of that default-cache crash was not established or
repaired; successful clean-cache runs do not establish that the old cache is
healthy. The full project suite was not rerun for this bounded display change;
the earlier intermittent live-slider timing failure remains documented in
HANDOFF. Existing Pydantic deprecation warnings remain.

No compatibility fallback, physical smoothing, new downstream source or coherent
mode was introduced. No commit or push was performed. Restart the application
to use the changed interface.
