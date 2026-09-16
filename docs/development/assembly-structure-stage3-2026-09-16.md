# Assembly navigation and vacuum references

The default component navigation is now **Assembly**, organised by physical
function. Gun/column/NanoPulser selectors remain under **Instrument
configuration → Change instrument templates…** for loading compatible variants.
They describe configuration templates, not three physical vacuum compartments.
Optical, Mechanical and Direct Alignment views remain available.

The initial selection is an assembly heading. Opening the instrument does not
move beam analysis to the tip body's centre upstream of its emitting surface.
Selecting a component synchronises the optional optical/mechanical trees when
they contain that component. Right-click an Assembly component to open its
**3D Parts**, **Ray Diagram** or **Vacuum map** location. These actions use the
same selection and navigation path as the Physical Layout context menu.

## Physical Layout

**3D Parts** groups the composed recording template into its three actual
subassemblies: imaging/projector stack, projection/detector chamber and Iliad
filter. Existing physical parent/child relationships remain intact. Group
headings do not become components or edit dimensions. The selected component's
definition file remains visible; **Subassemblies / placement…** edits axial
placement, while existing dimension/material controls edit the part.

The full Assembly tree groups by function across storage files. Its tooltips
also identify the actual subassembly file and stable component ID. A stored
template may reference several files; it is no longer described as one shared
dimension file for every component.

## Vacuum map

The upper band shows actual subassemblies and functional sections for the
remaining gun/column components. Bounds come from installed physical geometry.
Overlapping extents occupy separate rows; they are not shortened to create
artificial partitions. Filter-internal path-only parts retain their curvilinear
coordinate and are not offered as new straight-column vacuum boundaries.

The horizontal range and millimetres per screen pixel still follow Physical
Layout. Selection, pressure edits and changing tabs do not fit the Z axis.
Pressure controls appear when a vacuum region is selected. The selected region
lists intersecting physical sections with their actual storage references.

New boundary choices use readable component names with stable instance IDs,
or named section/subassembly start, end and origin references. Numeric Z can
use global coordinates, a named section origin, or a legacy template origin.
Moving an installed component updates its referenced boundary; moving a saved
subassembly updates its local frame after assembly reload. A section's physical
extent is an annotation, **not a separate pressure compartment**. Existing
pressure regions can cross it.

Existing map references are retained verbatim, including references that are
currently missing. A pressure-only edit cannot silently bind a missing boundary
to a different component. Historical component keys and module anchors remain
readable. Gaps still receive linear pressure transitions, overlapping pressure
regions are rejected, and the projection region remains tied to the DPA.
Default pressure values, region ranges and vacuum participation (off) are
unchanged. Explicit on/off choices survive saving and restoring maps.

## Captured topology and compatibility

When loading a composed module, the same read that supplies its physical parts
captures subassembly membership, local origins and relative definition paths.
The immutable metadata is stored as `navigation_subassemblies` in the captured
module geometry mapping. It adds no physical parts or independently configurable
electron state. Normal editable TOML materialisation does not contain this
generated metadata.

Working-point snapshots retain that mapping without changing their dataclass
schema. Navigation and vacuum anchor resolution use only the captured assembly,
never current definition files. Old snapshots without the metadata display
functional sections; missing subassembly references are not guessed. Identity
seeds and all 141 default component IDs remain unchanged. Renaming an instance
key or moving it to a different module authority still requires an explicit
identity migration; changing a filename alone does not.

## Verification

The receipt in `evidence/assembly-structure-stage3-20260916.json` records the
latest checks and final archive identity. Related tests cover all 30 supported
assembly combinations, snapshot restoration without file access, component and
subassembly anchors, pressure editing and map persistence, default-off vacuum,
shared Z viewports, parent selection, context menus, placement transactions and
working-point packages. Offscreen screenshots were inspected at 1800 × 1000.
The final focused acceptance set contains 197 distinct passing cases. Across
all checks attempted in this stage, the latest outcomes are 235 passing and
the seven unrelated historical test failures described below.

The default flat-tip calculation was executed afresh through extraction,
acceleration, column apertures and the installed downstream filter. All 81
numerical arrays match the original stage-1 baseline exactly; 25 of 49 preview
rays reach the sample at Z 1599.2 mm. No archived rays are injected.

Snapshot hashes intentionally change because captured navigation metadata was
added. The comparison explicitly verifies this new metadata against its
definition files and compares every remaining input graph value exactly.
Stage-2 storage repacking is separately verified. A test confirms that this
allowance rejects unverified metadata and does not hide a physical input edit.

A broader GUI-shell attempt also found seven pre-existing Direct Alignment
test cases using the superseded `apply_direct_alignment` monkeypatch hook or
passing a bare result to the current candidate-based commit API. Those test
bodies and alignment implementation are unchanged by this stage. They are
recorded separately, not counted as passing or treated as whole-suite success.
The navigation-related failures found during development were corrected and
rechecked. Coherent-wave development remains paused.
