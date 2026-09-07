# Physical Layout views

Physical Layout contains three presentation modes:

| Mode | Purpose | Source |
| --- | --- | --- |
| 2D | Whole-column mechanical section | Existing resolved layout display |
| 3D Parts | Inspect and edit a part, its neighbours or its module | Existing file-backed part editor; edits remain a draft until Save |
| 3D | Rotate, inspect and section the assembled column | Current saved `ResolvedAssembly`, with supported live aperture openings |

No high-accuracy calculation is required for 3D. A view change does not run
presets, propagate electrons, invalidate physics caches or replace images.
The assembly page does not display unsaved 3D Parts drafts; Save them first.

## Using the assembly view

- Drag to rotate, right-drag to pan, and use the wheel to zoom.
- **Side view** puts the source above the detectors: +X right, +Z downstream
  and downward. **Isometric** gives an oblique orthographic view.
- **Section** removes the positive-Y half to expose the interior from those
  default viewpoints. It clips existing surfaces; it does not create new
  material at the cut.
- Check or uncheck a component to show or hide it. Vacuum liners have their
  own read-only rows. **Schematic envelopes** filters approximate surfaces.
- Selecting a component highlights it without changing the camera.
  **Fit selected** deliberately centres and zooms to its visible geometry.
  **Open in 3D Parts** opens the existing editor, including its draft guard.
- **Fit assembly** restores an overview. The long, slender column is shown
  with equal physical scale on all three axes, not stretched to fill the view.

## Coordinate and model limits

Meshes reuse the configured part models. Their module-local Z positions are
translated to the resolved global column positions; semantic edges receive the
same translation. Camera rotation never reflects vertices or changes optics.
Lengths are in millimetres throughout the view.

Configured geometry does not imply verified OEM CAD accuracy. Parts without
defined solid construction retain explicit schematic-envelope labels. Virtual
planes, zero-thickness markers and duplicate optical-parent envelopes are not
invented as solids. Curvilinear, branch-only Energy Filter internals remain in
the Energy Filter view instead of becoming false axial hardware.

The existing renderer supports working strip-aperture opening and XY offset
changes. It does not infer insertion mechanisms, detector motion, or a magnetic
field from a displayed solid. The status tooltip describes omissions and any
unavailable part geometry. Saved regional materials use the same colours in
3D Parts and 3D.

## Retention and layouts

Meshes are built only when the assembly page is visible. Hidden updates retain
the latest assembly, not a queue of builds. Exact source geometry, resolved
positions, liner dimensions and supported aperture changes govern reuse;
unrelated excitation edits do not rebuild meshes. Rotating, fitting, sectioning
or filtering only redraws retained geometry. Returning to the page preserves
the camera and hidden components.

The new splitter and view tabs participate in existing named layout storage.
Old `2D section` and `3D model editor` selections migrate to `2D` and `3D Parts`.
The renderer is software-based and works without an OpenGL context.

## Validation

The focused geometry, material, viewport, part-editor and layout suite passed
242 tests on 2026-09-08. A native hidden-window render of the saved default
assembly produced 181 meshes / 23,124 triangles without geometry errors.
The full-column and Objective-section views were inspected, and the mesh-build
counter remained at one through camera, section and selection changes. This
checks configured geometry and presentation, not OEM dimensional accuracy or
electron-optical performance. No high-accuracy propagation was run.
