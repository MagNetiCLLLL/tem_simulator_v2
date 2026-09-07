# Parameter definitions, evidence and calculation use

Physical Layout, the 3D dimension inspector and the TOML parameter panel share
one parameter description. Select a dimension to see its meaning, declared
source and use by the active simulation mode. These are independent properties:
a physical dimension can have an unspecified source, and a displayed dimension
can be unused by the current field model.

## Reading dimensions

| Meaning | Interpretation |
| --- | --- |
| Physical dimension | Defines a supported material body, bore or material interval. |
| Envelope / allocation | Locates or bounds a mechanism without establishing its internal construction. |
| Vacuum passage | Defines a beam passage or clearance independently of material thickness. |
| Operating value / limit | Current optical setting or a permitted operating range. |
| Position / reference | Placement or reference coordinate. |
| Material definition | Material identity or constitutive response. |
| User CAD geometry | Explicit `model_3d` base, transform or Boolean feature. |
| Meaning not classified | The supported geometry rules do not establish the field's meaning. |

For the current C2 strip aperture, the 80 mm outer diameter and 20 mm length
describe the mechanism envelope. The plate thickness is 0.2 mm, the carrier
bore is 4 mm, and the vacuum passage is 5 mm. Its working opening comes from
the active operating radius: a 0.05 mm radius gives a 0.1 mm diameter. Neither
the carrier bore nor the maximum permitted radius substitutes for that opening.
The strip's unspecified transverse outline remains schematic.

Coil radial thickness is `(mechanical_outer_diameter_mm -
mechanical_inner_diameter_mm) / 2`. Changing it is different from changing the
vacuum passage. Parent-owned split coils and yokes expose their actual material
intervals as well as their allocation envelopes. An unused fillet-radius range
does not imply that the displayed solid has fillets.

## Evidence and catalog audit

The source labels are **Measured (declared)**, **Documented (declared)**,
**Estimated / reconstructed**, **Source unspecified**, **Current operating
value** and **User-defined**. A stored declaration is reported without
independent certification. Uncalibrated photographs or a statement about
mechanism topology do not establish a measured dimension.

Use **File > Dimension definitions and evidence audit…**, or **Dimension audit…**
inside the 3D editor. Search by component, raw TOML field or source file, and
filter by meaning, missing evidence or review status. Double-click a record to
open its source and locate the parameter. Existing unsaved-file guards apply.
**Export report…** writes Markdown or JSON; the JSON retains every numeric
value and its exact field/array path.

The main-window audit reads the saved catalog, including alternative modules;
runtime openings and unsaved drafts are excluded. A review item is not
necessarily a validation error: zero-thickness reference planes, incomplete
mechanism geometry and runtime-defined openings can all be intentional.
**Needs review** additionally includes unclassified meanings and unspecified
evidence, so its row count can exceed the structural review-item count.

The initial [catalog report](reports/dimension_audit.md) covers 11 module files,
482 component definitions and 4,480 numeric dimensional values. It identifies
280 structural review items, 182 values with estimation evidence and 4,298
values without established field-specific evidence. No values are promoted to
measured/documented, and the audit does not change source files. Repeated parts
in alternative modules are counted separately.

Generate a fresh report from the project environment:

```powershell
.venv/Scripts/python.exe -m temsim.dimension_audit --root configs/instruments --markdown docs/reports/dimension_audit.md --json docs/reports/dimension_audit.json
```

Existing field-specific `*_status` and `*_source` declarations are read
conservatively. An optional per-part `parameter_metadata` table can give an
exact field or dotted array path a source declaration. For example, add this
inside the relevant part declaration, before beginning another TOML table:

```toml
parameter_metadata = { plate_thickness_mm = { source_kind = "estimated", source_note = "Design target for the prototype; no measurement available." } }
```

Use a specific drawing/measurement reference for a documented/measured
declaration. This metadata describes evidence only; it does not override
geometry, validation or solver routing. The initial audit adds no declarations
to the instrument catalog.

## Calculation use

The 3D **Use** column has a concise status. Select the row or hover it to read
which results the field can affect and why.

| Status | Interpretation |
| --- | --- |
| Active | The current calculation has a supported route for this value. |
| Inactive | This value is not consumed by the current optical/field mode; its mechanical preview can still change. |
| Setup | A geometry field route needs explicit valid solver configuration or a matching custom map. |
| CAD only | This CAD parameter changes the preview but is excluded from beam-clearance and field calculations. |
| Check | Context or parameter support is not established. |
| Unsupported | The current calculation does not implement the requested parameter route. |

For coil ID/OD and magnetic material, Ideal Optics and Analytical Field retain
their existing field definitions. Linear Geometry Field and static Nonlinear
Material Field consume supported original geometry through configured
axisymmetric recipes. A custom imported map does not regenerate itself when
a coil is resized. Missing excitation, material or solver setup is explained
instead of being treated as a successful geometry solve.

One parameter can have several routes. A coil length can define both its
magnetic source extent and the axial vacuum interval. Ideal mode can therefore
use the vacuum interval while leaving magnetic geometry inactive. **Active**
means a route exists, not that every edit must change a numerical result:
overlapping narrower passages, operating settings and other constraints can
mask its effect. These descriptions do not replace full assembly validation or
solver diagnostics.

`model_3d` bases, X/Y transforms, holes and slots remain excluded from physics.
The component status retains **CAD changes excluded from physics** after
saving or calculating, including CAD modifications in visible child/shared
bodies. The tooltip lists the ignored parameters. This work does not add a
three-dimensional magnetic solver or infer collision geometry from CAD meshes.

## Drafts and results

The status above the 3D parameter tabs displays the active simulation mode and
the calculation lifecycle: **Not calculated**, **Results out of date**,
**Calculating**, **Result current for active model**, or **Calculation failed**.
It follows the existing calculation controller. A completed intermediate result
does not mark newer pending input as current.

**Unsaved draft · simulation still uses saved dimensions** takes precedence
when the file has changes or invalid input. Its tooltip lists affected results
to review after Save. Editing a draft does not apply it to the running
instrument. Saving an active module reloads the assembly and uses the existing
calculation scheduling/invalidation path. A current result is for the active
model and saved inputs; the CAD exclusion and field-setup notices remain
visible. External or inactive module files show **Not linked to active
simulation** and cannot claim to have been calculated with the active assembly.

When another editor saves a source file, an assembly refresh reloads a clean
3D document while retaining its selected component and camera view. Valid and
invalid drafts are preserved. A changed or unavailable source is explicitly
flagged and cannot display **Result current** until the source is resolved.
Saving an external copy or changing the active module selection also refreshes
the displayed working openings, so an inactive file cannot retain a borrowed
runtime aperture.

Mode/context updates refresh annotations without replacing typed drafts,
resetting the selected component or rebuilding its geometry. TOML tables keep
their raw field names and existing editable value column.
