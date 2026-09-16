# Unit-based instrument configuration

## Operator workflow

Open **Instrument configuration → Configure instrument…**. The separate window
lists units in source-to-detector order:

1. Electron source: Cold FEG or Thermionic.
2. Monochromator.
3. Electrostatic beam blanker.
4. C3 lens.
5. Probe corrector.
6. Image corrector.
7. Energy filter.

Optional units use checkboxes. Monochromator requires Cold FEG; both correctors
require C3. Removing a prerequisite clears and disables its dependent options.
C1/C2, the objective, sample stage and recording hardware remain in the base
assembly. This window does not edit individual component definitions.

**Check** resolves an independent assembly draft and updates its linked 2D/3D
physical review. Selecting a unit focuses its hardware; selecting hardware
identifies its unit. The preview explicitly distinguishes the installed
assembly, a checked draft and a previous view after further edits.

**Assemble** becomes available only after Check. It applies the checked state
through the existing working-point transaction, updates the main physical
layout and schedules one Preview. Closing or checking a draft alone does not
change the instrument. Changing live settings or consumed definitions after
Check invalidates the draft rather than applying outdated inputs.

The window size and splitter are saved separately for each workspace layout.

## Compatibility and calculation boundaries

- Existing TOML assembly modules, interface connections and position anchors
  remain authoritative. Composite template names are internal compatibility
  identifiers for saved profiles, not visible operator choices.
- Existing defaults retain Cold FEG, C3, probe correction and energy filtering.
  Explicitly choosing no energy filter is now respected; the previous implicit
  replacement with an installed filter has been removed.
- Shared operating parameters are retained where supported. Checking an
  unchanged configuration preserves its exact inputs, including tip geometry.
- No lens-preset optimization is performed while checking or assembling. Use
  Direct Alignment or Live tuning after completing the hardware selection.
- Check validates assembly consistency, not optical performance. It does not
  calculate rays, coherent waves, specimen signals or calibrated lens values.
- Real assembly changes invalidate previous results through the existing
  working-point installation path; previous results are not current evidence
  for newly installed hardware.

## Regression coverage

The unit-selection tests enumerate all 60 currently supported combinations
(three source/monochromator choices, five column topologies, two blanker choices
and two filter choices). Tests cover dependency rejection, exact unchanged
drafts, shared operating values, stale checks, physical-review linkage, window
layout persistence and the main-window apply transaction without automatic
preset calculation. Related catalog, navigation, structure and working-point
restore tests cover compatibility. This is offline software validation, not a
new high-accuracy optics or microscope-hardware qualification.

The related 123-test regression passed. Its Thermionic/C2 propagation smoke
test emitted overflow/invalid-value warnings; passing the assembly tests does
not qualify that optical operating point. Deprecation and offscreen teardown
warnings were also reported. No high-accuracy calculation was run for this UI
change.

After adding persistence when the main window closes first, the focused
configuration, profile and working-point restore rerun passed all 41 tests.
