# Independent Hardware tuning tab — 2026-09-25

## Scope and workflow

The supplied control-mapping document informed the functional grouping. The
user requested a separate manual editor, not implementation of the document's
entire calibration/control architecture. No commercial labels or raw interface
names were copied into the new page. Existing Direct Alignment remains intact.

**Hardware tuning** is a main-workspace tab immediately after Ray Diagram:

1. Choose an alignment in the searchable left column.
2. Edit its hardware drives in the right column; each control shows its unit.
3. Press Enter or leave a numeric field. Checkboxes apply immediately.
4. Existing preview/calculation settings handle accepted physical edits. Merely
   selecting an alignment is read-only and stays on the same page.

Both columns remain visible in compact windows. Hardware cards retain readable
labels and editor heights; long lists scroll. The existing workspace layout
mechanism saves the named horizontal splitter and selected workspace tab.

## Bindings and scientific meaning

There are **27 task entries**, including two explicitly unsupported standalone
three-fold tasks. They cover:

| Task group | Actual hardware fields |
|---|---|
| Gun shift/tilt | Upper/lower magnetic-field X/Y channels, in mT |
| Gun stigmation | Magnetic gradient in T/m and orientation in degrees |
| Extraction/gun focusing | Existing electrode voltages in kV; reference shown |
| Current/spot size and convergence | C1/C2/C3 excitation percentages |
| Condenser/beam/image shift and tilt | Shared upper/lower X/Y deflector kicks in mrad |
| Image/probe focus | Objective and, for probe focus, mini-condenser excitations |
| Projection/diffraction tasks | Diffraction, intermediate and two projector lens excitations |
| Two-fold stigmation | Separate condenser/objective/diffraction normal/skew channels |
| Probe corrector | The 17 declared multipole, relay-lens and steering components |
| Scan/descan static controls | Static kicks, upper gain and supported pivot offsets |
| Beam wobble | Existing sinusoidal amplitudes, period and temporal drive phase |

The registry references current `RuntimeTarget` objects and intersects fields
with the existing editable-parameter policy. There is no second copy of the
actuator state. Gun selectors resolve the actual installed source components,
including thermionic configurations. Geometry, source generation, transmission,
focus results and angular statistics are not overwritten by this page.

Shift/tilt tasks deliberately share a physical pair; manual drive changes do
not guarantee a pure shift or pure tilt at any observation plane. Focus and
projection tasks expose coupled hardware without claiming a calibrated defocus,
magnification or camera-length command. Effective multipole coefficients retain
their declared units and are not renamed as measured currents or wave aberrations.
Temporal wobble phase is the existing drive phase; coherent work stays paused.

Unavailable components have explicit notices. Standalone three-fold stigmators
do not borrow two-fold or corrector channels. Ideal Optics locks the excluded
hexapole field terms. Installed disabled components remain adjustable, except
C3 and mini-condenser activation, which is owned by the column layout and is
shown read-only. Derived lower-coil gains, raster clock and calibration records
are not exposed as independent writable state.

## Edit lifecycle

`hardware_tuning.py` contains the immutable task/binding registry and read-only
resolution. `gui/hardware_tuning_panel.py` builds the controls, reads current
values and validates edits. Before mutating a live component, an edit passes
the ordinary scalar validator and the existing component validator on a detached
copy. This checks combined drives, including derived lower-coil gains, and the
scan/wobble exclusion. Surface-source electrode validation retains its grounded
field convention. It performs no field solve.

Invalid edits restore the current displayed value and emit no physical-change
signal. Replaced forms reject delayed signals, and a draft based on an externally
changed value is rejected for review. MainWindow routes accepted edits through
its existing revision, stale-result, candidate-invalidation and preview path.
The original component editor and scan controls refresh from the same state.
State replacement, mode changes and ordinary parameter changes refresh this
page through the existing context refresh. No navigation or automatic alignment
is initiated by task selection or editing.

## Verification

Project Python environment; offscreen Qt; one numerical worker per test process;
single-thread BLAS/OpenMP. The user's application/settings were not modified.

- Registry: **17 passed**, 8.38 s. Checks include shared object identities,
  units, installation/model gating, layout-owned activation, source selectors,
  unsupported three-fold tasks and functional names.
- GUI: **25 passed**, 20.557 s, zero failures/errors/skips. Receipt:
  `tmp/hardware-tuning-gui-tests.xml`. Tests cover valid/invalid edits, both
  directions of parameter synchronization, actual main-window invalidation and
  preview scheduling, independent Direct Alignment, stale drafts, state swaps,
  compact scrolling, splitter persistence and text/control geometry under the
  production stylesheet. These tests forbid calculation dispatch.
- Five domain cases were first reproduced as failures: zero/negative wobble
  period, wobble enabled during raster scanning, excessive static kick and
  excessive upper gain changing the derived lower drive. After repair all five
  preserve the complete component dictionary on rejection. Valid scan-gain and
  wobble edits are separately exercised.
- Existing affected regression: **55 passed**, 104.779 s, zero failures/errors/
  skips. Receipt: `tmp/hardware-tuning-20260925/existing-regression.xml`. Includes
  the existing ray/geometry test with actual particle transport, workspace
  layouts, condenser matching UI, incremental ray scenes and Working points.
  This group used an isolated `NUMBA_CACHE_DIR`.
- Total: **97 distinct passing cases**. Production modules compile and
  whitespace checks pass. This is targeted integration validation, not a full
  project suite or new physical-control calibration.

The first attempt at the existing ray/geometry test crashed inside the old
native `analytic_particle_batch.py:232` path using the default compilation cache.
This recurs from the preceding task; the cause has not been established or fixed.
The entire selected group completed with an isolated cache. No existing test
was weakened or bypassed, and clean-cache success is not a claim that the old
cache is healthy. Existing Pydantic deprecation warnings remain.

Visual QA iterated all 27 task selections without edits/calculations, rendered
the standalone page at 900 × 700 and the actual main window at 1500 × 920, and
inspected gun/beam, condenser, corrector and unavailable-task states. The first
screenshots exposed compressed form labels/cards; minimum layout constraints
and geometry regressions now prevent that collapse. Startup preset fitting was
suppressed only in the visual/test harness. Artifacts and the harness remain
under ignored `tmp/`; no generated images or calculations were added to Git.

No commit, push or user-application restart was performed. Restart to use the
new tab.
