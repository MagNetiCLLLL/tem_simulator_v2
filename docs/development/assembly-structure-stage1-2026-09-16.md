# Assembly structure: first compatibility stage

The new **Assembly** tab in the left instrument panel presents the whole
instrument by function, with every existing mechanical parent preserved.
For example, the intermediate/projector lenses are in Imaging and projection,
while HAADF/DF/BF are in Detection and recording. Only the actual filter
components appear in Energy filter. EDS, the sample stage and nested scan coils
remain under the objective assembly according to their existing parent links.

The existing gun/column selectors, TOML files, module ports, assembly resolver,
collision checks, vacuum map and particle calculation remain authoritative.
Groups in this tree are navigation folders, not new physical modules or vacuum
boundaries. Selecting a component uses the same parameter and Physical Layout
editor as before. Double-clicking reveals the physical component. Reloading the
assembly refreshes the selected parameter context; selection survives a storage
file rename when the module/component keys are unchanged.

## Identity contract

`assembly_structure.py` creates an immutable view of a captured assembly,
without opening current source files or adding fields to the instrument state.
Existing working-point snapshots can therefore be decoded without schema
migration. Their historical path-scoped definition authorities remain intact.

An installed component receives a UUID in the fixed v1 namespace, seeded by its
existing module key and component key. The UUID does not depend on its path,
label, dimensions, position or list order. Module and navigation-group IDs have
separate seeds. Distinct instances receive distinct IDs, including independent
copies. Explicit shared-tip definition references remain shared references;
they do not collapse the two installed instances into one object.

**Export assembly map...** writes the ID-to-component mapping, legacy storage
authority, parent IDs, resolved positions and filter-path references as JSON.
`identity_map_from_document()` reads the bindings without applying any geometry.
Duplicate IDs, duplicate locators, absent components, parent cycles and missing
parents are reported instead of silently dropping components.

The future module split or component-key rename must explicitly carry these
IDs into `build_assembly_structure(..., identity_map=...)`. This first stage
does not automatically rename keys, split files, import geometry from the map
or turn the navigation groups into editable placement constraints. A storage
rename alone already preserves identity. The captured default mapping is in
`evidence/default-assembly-identity-map-v1.json`.

## Frozen flat-tip comparison

The user selected the current project default configuration as the reference:

- FEG / C3 + Probe Corrector / installed Energy Filter.
- Nanoprobe / Diffraction, analytic classical flat tip, curvature zero.
- 49 Preview rays; 1 mm column propagation and history sampling.
- Vacuum transport disabled, preserving the project default.
- Preview optical reference: no specimen-image, EDS or coherent-wave execution.
- A separate filter calculation consumes the same executed upstream particles.

The baseline was executed **before** adding the new structure layer. It is
stored locally in:

`outputs/assembly-baselines/flat-tip-before-20260916-01/`

The archive contains the exact complete working point, source/configuration
ZIP, per-ray gun/column/filter arrays, array metadata, a physically scaled XZ/YZ
trajectory plot, and checksum/report files. The 81 arrays retain positions,
directions, weights, lineage, energy information and physical stops supplied by
the current result models. Units follow their model fields: column/gun Z in mm,
X/Y in m and transverse slopes in radians. Filter-local path coordinates retain
their separate references; the axial plot is the normal column Preview view.

Reproduction/comparison, using a **new, nonexistent** output directory:

```powershell
.venv/Scripts/python.exe scripts/archive_flat_tip_baseline.py `
  --output outputs/assembly-baselines/flat-tip-next-comparison `
  --compare outputs/assembly-baselines/flat-tip-before-20260916-01
```

This executes tip emission, extraction, acceleration, column optics, apertures,
detector interception and the installed filter again. Archived rays are not
reused as an injected downstream source. Changed working parameters or unequal
result arrays fail the comparison. A code hash change remains explicit and is
not treated as permission to restore old snapshots as active calculations.

The final comparison is in
`outputs/assembly-baselines/flat-tip-after-20260916-final/`.
The exact input graph, external inputs, result metadata and all 81 result arrays
match the baseline. This establishes behavior preservation for this classical
reference, not physical calibration or full TEM/STEM acceptance.

Numerical arrays, screenshots and source bundles stay local and are excluded
from Git. The lightweight verification receipt and input identity map can be
versioned. They do not replace the local numerical baseline.
