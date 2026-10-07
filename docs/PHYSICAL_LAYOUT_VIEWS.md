# Physical Layout views

Physical Layout contains the following presentation modes:

| Mode | Purpose | Source |
| --- | --- | --- |
| 2D | Existing fixed X-Z mechanical projection | Existing resolved layout display; unchanged for comparison |
| Rotating section | A longitudinal cut through the column axis, with X-Z, Y-Z and arbitrary azimuth | Current saved assembly meshes, including physical placement and supported live aperture openings |
| 3D Parts | Inspect and edit a part, its neighbours or its module | Existing file-backed part editor; edits remain a draft until Save |
| 3D | Rotate, inspect and section the assembled column | Current saved `ResolvedAssembly`, with supported live aperture openings |
| Energy Filter | Inspect the curved branch without compressing it onto the main column Z axis | The same installed filter dimensions used by its ray view |

No high-accuracy calculation is required for 3D or Rotating section. A view change does not run
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

## Physical magnetic-lens placement

Round magnetic-lens assemblies have physical X/Y/Z offsets in mm and X/Y/Z
rotation angles in mrad. Rotations are right-handed, applied X then Y then Z
about the unposed mechanical centre. The same rigid transform places the lens
magnetic field, its owned poles/coils/housing, and its owned bore or aperture.
The 3D Parts view previews a draft; Save and apply it before recalculating rays.
Selecting a physical child exposes its **Parent lens** placement controls.
The original 2D view projects the posed local X-Z section into global X-Z.
Use Rotating section to compare cuts in different directions, or 3D to
inspect the complete posed solid.

Ownership follows the lens's mechanical hierarchy. A shared C1/C2 cartridge
appears once and follows its declared parent, not both optical channels.
Nested round lenses inherit the containing lens pose. Separately mounted
sample/stage, steering and stigmator devices, control channels and virtual
references do not inherit it: their `parent_key` describes packing or navigation,
not a rigid mount. Stationary column vacuum tubes remain stationary.
For posed lenses, clipping retains each declared overlapping part bore and the
fixed continuous vacuum tube; a wider tube cannot disappear merely because the
centred layout previously retained only its narrower overlapping lens bore.

Legacy lens CAD offsets and degree-valued rigid rotations are read as physical
placement and exposed by the editor in mm/mrad, without applying them twice.
Changing a winding thickness, material shape, CAD scale or Boolean cut does not
quantitatively change the prescribed magnetic field strength. Individual child
CAD edits remain shape edits; use the parent lens placement controls to move
its magnetic axis and complete magnetic assembly together.

The existing thin equivalent spherical-aberration kick is attached to that
same lens frame. It remains an empirical coefficient model, not an aberration
fit inferred from the solid. Posed field transport runs on the CPU and does not
require CUDA; the unchanged centred case retains its existing fast path.
The analytical path retains its local paraxial law under the rigid transform,
so an infinitesimal placement does not introduce full-Lorentz higher-order
terms. Imported spatial field maps retain their existing vector Lorentz law.
The coherent column-wave solver does not yet support these placed spatial
fields; use Run high-accuracy once for the particle/ray result.

The TOML entries are `offset_x_mm`, `offset_y_mm`, `offset_z_mm`, and
`rotation_x_mrad`, `rotation_y_mrad`, `rotation_z_mrad`. Zero preserves the
nominal placement. For example, `rotation_y_mrad = 1.0` tilts the lens by
0.001 rad about Y. Field-map placement composes with the map's own registration;
it does not require solving or reimporting the field shape. An inseparable
joint nonlinear field can move only as one common rigid frame.

## Comparing the projection and rotating section

The original **2D** page remains unchanged. **Rotating section** intersects the
same saved, globally placed geometry used by **3D** with a plane containing
the nominal column Z axis. At viewing azimuth phi, the displayed transverse
coordinate is `U = X cos(phi) + Y sin(phi)` and the cut plane is
`V = -X sin(phi) + Y cos(phi) = 0`. **X-Z** selects 0 degrees and **Y-Z** 90
degrees; the slider and numeric field select intermediate angles. This angle
is in degrees and changes only the view, independently of lens installation
rotations in mrad.

**Follow Ray Diagram** shares the viewing azimuth in both directions. Uncheck
it for independent comparison. Ray Diagram projects all retained rays; the
section shows only where material intersects the chosen plane. A displaced
body outside that plane can disappear, and a hole appears only when the plane
actually cuts it. A rotation does not retrace particles, recalculate fields,
change optics or replace retained scientific results.

Material-coloured contours preserve bores and CAD openings. These are cuts of
the configured tessellated solids, not verified manufacturer CAD. Schematic
envelopes remain marked as approximate and can be hidden. Virtual channels
and non-material guides are excluded. The view uses the saved assembly and
current supported geometry inputs; unsaved 3D Parts drafts are excluded, and
retained ray results can belong to earlier settings.

Only a visible page builds geometry. The rotating-section and 3D pages share
one retained column mesh model, with 32 angular segments for revolved parts;
cuts between mesh vertices therefore have a small tessellation error. Angle
changes reuse those meshes, are throttled during dragging, and keep a bounded
cache of recent cuts. No CUDA or OpenGL context is required. Changing angle
preserves zoom; **Fit assembly** and **Fit selected** explicitly recenter it.

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
invented as solids. Curvilinear Energy Filter parts use their branch frames,
then one rigid placement at the main column entrance; they are never packed
onto their placeholder axial Z coordinates.

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

Instrument configuration uses one **Assemble** action: choose installed units,
then apply them. Geometry and interface validation runs within that action;
failure leaves the current instrument unchanged and permits another attempt.
The review shows the installed assembly until application, and closing the
dialog does not apply a draft. Its read-only view does not construct a hidden
part editor. Filter structure is drawn only when requested; full-column label
placement tests screen rectangles before moving graphics items and runs once
after the final fit range, avoiding repeated layout during opening.

## Validation

On 2026-10-07, an offscreen CPU check of **FEG + C3 + Probe Corrector + No
Energy Filter** rendered 165 meshes / 45,320 triangles without geometry errors.
Whole-column and Objective X-Z/Y-Z screenshots were inspected. Initial mesh
construction and display took 2.18 s; four angle changes took 173-213 ms each
including the page refresh, with the geometry intersection alone taking 48-53
ms. The mesh build count remained one. These are local measurements of this
configuration, not a frame-rate guarantee; no particle or field solve ran.

The focused geometry, material, viewport, part-editor and layout suite passed
242 tests on 2026-09-08. A native hidden-window render of the saved default
assembly produced 181 meshes / 23,124 triangles without geometry errors.
The full-column and Objective-section views were inspected, and the mesh-build
counter remained at one through camera, section and selection changes. This
checks configured geometry and presentation, not OEM dimensional accuracy or
electron-optical performance. No high-accuracy propagation was run.

## Energy Filter construction and evidence

The filter is an installed branch of the recording system. Its incoming axis
continues the main column; the configured 90-degree prism makes the outgoing
axis horizontal. This installation does not introduce another optical bend.
**Physical Layout > Energy Filter** shows the mechanical structure and dimensions.
**Ray Diagram > Energy Filter rays** shows calculated internal trajectories with
reference axes, component centres and optical planes; it does not repeat the
mechanical housings. The standalone **Energy Filter** page contains parameter
editing, **Calculate energy filter**, **EELS spectrum** and **EFTEM image**, with
no duplicate structure or ray plot. The ray and output pages use the same accepted
calculation result and retain previous results with a warning after input changes.
Switching pages does not calculate. Saved layouts that selected the removed
**Physical + rays** output tab now select **EELS spectrum**.

With Energy Filter installed, the main-column bore and its material liner end
at the declared optical entrance plane, inside the entrance-aperture carrier.
This fixed plane also ends the main-column electric domain and hands particle
transport to the branch. The same resolved segments feed the 2D view, 3D view,
grounded electric boundary and column-wall interception. The recording-module
envelope and all lens/sample positions remain unchanged. Without Energy Filter,
the recording-module exit and straight pipe end at the Camera downstream
surface (local Z=1172.75 mm), removing the old empty tail to 1350 mm. A separate
**Camera / Lower Recording Enclosure (schematic)** spans the original viewing
chamber end to this terminal surface. It adds no field source or clipping
boundary; its 180/200 mm radial envelope is provisional, and camera packaging,
end caps and pumping ports are not inferred. The original viewing/STEM chamber
and all active detector coordinates are unchanged.

Both physical variants obtain their global endpoints from the resolved
assembly. Adding a NanoPulser, probe corrector or image corrector changes module
origins, and the liner, 2D/3D/rotating-section geometry and fixed electric domain
follow automatically. No absolute global camera coordinate is hard-coded.
For an unfiltered assembly, the particle observation margin is bounded by the
same mechanical end, without the legacy 2800 mm minimum. Explicitly uninstalled
runtime placeholders do not extend the column; a real installed interaction
beyond its assembled boundary remains an error. Existing retained calculations
keep their captured input geometry until recalculated.

The endpoint regression matrix covers C2, C3, probe-corrected, image-corrected
and double-corrected columns, each with and without the electrostatic NanoPulser
and Energy Filter (20 assemblies). The associated boundary, grounded-field
request, recording-stop and chamber checks passed 85 tests on 2026-10-07.
A software-rendered 3D check of FEG + double correctors + NanoPulser without
Energy Filter placed the camera back surface, enclosure, liner and electric
domain at the same Z=3493.15 mm, with no mesh errors or protruding pipe tail.
A nine-particle CPU-only C2 / No Energy Filter calculation also returned its
completed particle result in 46.9 s at a 1 mm integration step, without a
domain or liner-coverage error. This is an execution smoke check, not a
high-accuracy convergence qualification.

Multipole interception uses mechanical
housing length, independently of its shorter magnetic-field support.

The local filter frame enters along +X and bends toward -Z. Installation uses
the right-handed transform `(X,Y,Z) -> (-Z,Y,X) + (0,0,entrance_z)` in mm.
This is a placement of the existing optics, not a second bend or a change to
the tracer's established transverse-sign convention. Single-part and whole
assembly views share the same geometry adapter. The current slit opening and
centre are read from the existing operating state; no duplicate control is
added. They participate in mesh reuse, so opening a tab or rotating a view
does not request another calculation.

Public references reviewed for the reconstruction:

| Source | What it supports | What it does not establish |
| --- | --- | --- |
| [Iliad features](https://www.thermofisher.com/de/de/home/electron-microscopy/products/transmission-electron-microscopes/iliad/features.html) | Tapered prism and ten multipoles | Individual carrier positions, radii or OEM pole assignment |
| [Iliad Ultra DS0510](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf), page 2 | Bias/shutter/deflector functions; five 2048-pixel strips; 14 um pixels; 256x2048 alignment region | Detector package, strip centre spacing or strip height |
| [US9966219B2](https://patents.google.com/patent/US9966219B2/en), Figs. 1-4 | Post-column bend, downstream optics and detector; opposed energy-selecting blades | This instrument's dimensions or the Iliad production arrangement |
| [US10431420B2](https://patents.google.com/patent/US10431420B2/en) | A post-column prism and post-slit multipole layout | Mechanical dimensions of this model |
| [JP2011103273A](https://patents.google.com/patent/JP2011103273A/en), Figs. 3/7 | Opposed pole faces, beam gap and electrostatic plates in section | Its different in-column architecture is not adopted |

Patent drawings establish construction principles, not a dimensional ruler.
The existing 135 mm prism radius, 30 mm pole gap, carrier bores and spacings
remain adjustable non-OEM model dimensions. For the default 90-degree branch,
the reference orbit is 792.0575 mm from entrance to Zebra. Its detector centre
is 610 mm horizontally and 240 mm downstream from the entrance after mounting.
These distances exclude unknown magnetic yokes and camera packaging.

Zebra's 28.672 mm spectral width and 28.672 x 3.584 mm alignment area follow
from the published pixel geometry. The configured 0.8 mm strip height and
1 mm centre spacing are engineering assumptions. Unknown housings are not
invented. The optional EFTEM output remains a reference plane, not a solid
detector body. Displayed electrostatic envelopes do not imply that a finite
electrode field or dynamic-focus solver has been implemented.

Integration checks on 2026-10-06 passed: 163 part/assembly/material/editor
checks, 29 mechanical-clipping and compatibility checks, and 10 targeted
workspace-view checks. The obsolete branch-exclusion expectations were replaced
with supported-boundary and explicit-omission checks. An isolated Qt offscreen
render of FEG + C3/Probe Corrector + Energy Filter produced 196 meshes, including
all 18 drawable filter part keys, with no geometry errors. Structure, full
assembly and mount detail images were inspected; the retained mesh build count
was one. That geometry/UI check did not execute optical propagation and missed
an electric-domain mismatch: the liner stopped at the carrier's upstream face
while the electric domain still used the recording-module envelope endpoint.
The corrected joint reaches the optical entrance at 3025.9 mm for this assembly,
2 mm inside the carrier. Grounded-liner coverage and connectivity checks remain
active; changing an observation plane does not change this physical boundary.

The repaired boundary was exercised through a real offscreen Qt MainWindow:
select Energy Filter, click Assemble, and let the scheduled Preview finish.
The 49-ray result completed in 22.815 s and was published to Ray Diagram;
the column used CUDA GPU, while the existing gun and filter stages used CPU.
No cached final result or mocked physics was used. The default inserted-hardware
state passed no rays through the filter entrance, so this verifies restoration
of the automatic preview, not an EELS spectrum or filter optical calibration.
The 32 focused electric-domain/mechanical regressions and 14 downstream-stop
regressions passed. The latter also cover the specimen transport fallback:
without an explicit stop it uses the same straight-column handoff, rather than
continuing 0.5 mm past the installed entrance.
