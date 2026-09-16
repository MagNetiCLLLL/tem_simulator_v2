# Active tip emission in Physical Layout / 3D Parts

The analytic tip's operating curvature previously changed launch positions and
directions while 3D Parts continued to show only the TOML metal tip. The saved
100 nm radius was also labelled Active without regard to the selected source.

The editor now receives a read-only description of the current analytic
emission, including its geometry version, curvature, source FWHM and angular
support. Changing these inputs refreshes the visible launch surface. Hidden
pages refresh when shown. Camera rotation, zoom and centre survive operating
edits; Fit emitting surface explicitly resets the framing.

## Display and parameters

- **Active emitting surface** shows a gold launch surface and green local
  emission axes. These axes are not propagated rays. XYZ use equal scales.
- **Reference body** shows the saved conducting tip solid.
- **Surface + reference outline** locates the launch surface relative to the
  reference body, without letting an opaque reference hide it.
- The dimension table lists active curvature, derived radius, projected FWHM,
  support diameter and edge depth. Flat emission has infinite radius. Selecting
  the surface or its boundary highlights these related entries. Use the existing
  operating controls or Tip model / emission to edit the source.
- Saved geometry and alternate field-model emission settings are marked
  **Reference** while analytic emission is active; derived quantities are marked
  **Derived**, not editable independent source parameters.

For the existing 5 nm projected FWHM, the support diameter is approximately
12.739827 nm. At 0.01 nm⁻¹, radius is 100 nm and edge depth is 0.203085 nm;
at 0.02 nm⁻¹, radius is 50 nm and depth is approximately 0.407418 nm. The small
depth is deliberately not exaggerated. The saved reference radius remains
100 nm in either case unless separately edited and saved.

## Scope and invariants

The display calls the same emission geometry operator used by the particle
source. Historical angle-only inputs still display a flat launch plane with
curved emission axes; they are labelled explicitly and are not converted.
Zero spatial support does not invent a finite emitting surface. Mechanical
copies and unconnected files do not acquire active emission overlays.

Open launch surfaces and direction guides are inserted only into the 3D Parts
view. The CAD model, material regions, Boolean operations, copied solids and
saved TOML are unchanged. The assembly overview continues to show reference
hardware. No new source is defined after extraction or acceleration.

The field solver, voltages, emission distribution and particle propagation are
unchanged. No coherent-wave work is enabled by this display update. This does
not certify a self-consistent electrode field for the analytic operating surface.

Focused tests cover source/mesh geometry agreement, flat and historical modes,
reference preservation, live control wiring, read-only parameter selection,
camera preservation, dirty drafts and hidden-page refresh. Local test records
and screenshots are kept under `outputs/setup-validation/`; only lightweight
verification receipts are eligible for version control.

## Verification receipt

Base revision: `e41d102`. Relevant tests: 193 unique cases, all passing on their
latest run. This combines the focused emission/edit/preview suite (54 cases)
and CAD, parameter semantics, source GUI and assembly rendering regressions
(139 cases); reruns replace earlier outcomes rather than adding to the count.
Two pre-existing source-label assertions and an editor rendering test double
were updated to match the current controls. One new GUI test was corrected to
reacquire the operating widget after navigating to Physical Layout.

Offscreen screenshots were inspected for flat, 100 nm and 50 nm emitting
surfaces. The curves use the same XYZ scale, and the table separates active
emission geometry from the saved 100 nm reference. This is display and
integration verification, not a new field calibration or full-chain image
acceptance. No long wave calculation was run.
