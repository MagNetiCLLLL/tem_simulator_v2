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

## Components and assembly files

The 3D Parts editor separates a component's shape and ownership from its storage
file. FEG, C3 Probe Corrector and Energy Filter are existing assembly presets and
coordinate sections, not restrictions on the kinds of mechanical parts they can
contain. Their interface order remains compatible with existing instruments.

- **New component** creates a tube (including its bore), box or elliptic cylinder.
  Choose its name, unique key, dimensions, parent and centre. Materials and further
  holes/slots can then be edited in the existing parameter tabs.
- **Place** moves the selected component and, by default, its children. **Local Z**
  is measured in the destination file. **Global Z** is available only for files
  in the currently resolved instrument, using that file's actual origin.
  Moving a component does not automatically extend module ports or move neighbours.
- **Copy to assembly** makes an independent copy in another existing catalog or
  external module TOML, or in the same file. The source remains unchanged. Choose
  a new root key, target parent and centre. Child keys and internal references are
  remapped; unresolved shared dependencies must be resolved before copying.

Each operation updates a previewable draft and is one Undo/Redo step. **Save**
persists the destination. Cross-file copying requires resolving any existing
source draft first. Inactive catalog files can be saved without installing their
optical preset or replacing current simulation results. Catalog saves validate
all compatible assemblies in a temporary copy before replacing the destination;
changed source files and duplicate keys are rejected.

New components and copies are mechanical CAD definitions. Copying a lens or coil
does not create a new optical control, current source, vacuum stop or FEM region.
Existing dimensions remain editable; copied pole orientation and split winding
material intervals are retained. Arbitrary CAD shapes remain excluded from the
current axisymmetric field solver. Position previews describe axial envelopes,
not a complete solid-interference or mechanical-fit analysis.

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
