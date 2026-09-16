# Cache input isolation and tip launch display

## Cause and scope

The Advanced bank readout failed while capturing a detached input state:
`Unregistered working-point model: temsim.optics.energy_filter_raytrace:EnergyFilterResult`.
The pipeline attached this completed output to `State`; the input snapshot
encoder then tried to serialize it as an instrument component.

This predates the component split. `instrument_snapshot.py`,
`interactive_calculation.py`, and `simulation_pipeline.py` have no changes
between pre-split revision `e41d102` and revision `be1c4c0`. The former already
contains the result alias and the snapshot path that consumes it.

## Changes

- Keep energy-filter results and crossover diagnostics in `CalculationResult`.
  Do not add their redundant aliases to new pipeline states.
- Exclude only the two known legacy output aliases on the root instrument
  State from new input snapshots. Unknown model inputs still fail closed.
  Old readable snapshot graphs are verified without rewriting their contents;
  old in-memory completed results remain available to their original readers.
- Draw the curved-tip launch prefix from the executed equal-time gun history.
  Each electron starts at its own negative Z, with the tip centre still at zero.
  No ray is extrapolated upstream of its emission point. Flat-tip drawing and
  the existing shared-plane transport arrays are unchanged.
- Cache the short launch display prefix with the existing bounded display
  cache. Rotation reuses the X/Y bases; it does not retrace the gun. No new
  large common-Z-by-ray matrix or persistent calculation cache is created.
- Update old Direct Alignment GUI fixtures to the immutable candidate API.
  Check exact coupled controls, stale/cancelled rejection, and preservation
  of either captured image-lens model. Synthetic candidates validate the GUI
  transaction only, not optical alignment accuracy.

Extraction, acceleration, apertures, lens controls, energy-filter transport,
and numerical source parameters are unchanged. Coherent source development
remains paused. Verification is targeted; no full TEM/STEM high-accuracy
acceptance or real-desktop interaction is claimed.

## Verification

The final combined regression run passed **146 tests**, with **3 wave-related
cases deselected**, in 299.81 s (exit 0). It covered input snapshots, production
particle-bank reuse and readout, curved-tip drawing, bounded display caching,
source colours, gun timing, continuous curvature, alignment transactions,
calculation orchestration, filter geometry, and the updated GUI-shell cases.
The production bank test forbids calling propagation again during readout and
checks that each original energy-filter result remains available.

Checks used the project virtual environment and Qt offscreen rendering.
Existing Pydantic `json_encoders` deprecation warnings and a pyqtgraph teardown
disconnect warning remain; neither caused a test failure. No calculation
arrays were added to version control, and no commit or push was performed.
