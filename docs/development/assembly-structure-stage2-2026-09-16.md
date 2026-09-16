# Assembly structure: reusable subassemblies and persistent axial placement

The oversized recording-system file is now a compatibility entry referencing
three independent physical part files. It retains its module key, interfaces,
overall envelope and calibrated geometry settings. Existing selections and
operating presets still load it. No component key, instance UUID, optical
operation or default physical position was changed by this migration.

| Definition file under `configs/subassemblies/` | Contents | Parts |
| --- | --- | ---: |
| `projector_stack.toml` | Selected-area aperture and D / I / P1 / P2 lenses, poles, coils and housings | 26 |
| `detector_chamber.toml` | DPA, chamber, HAADF, DF, BF, screen and camera | 7 |
| `iliad_filter.toml` | Actual filter entrance and internal prism, multipoles, slit and readout path | 20 |

`configs/instruments/project_and_recording_system/EnergyFilter.toml` contains
the references and placements. Definitions are outside `configs/instruments`
so the catalog's module-selection inventory remains unambiguous. Each file
declares `module.type = "subassembly"` and local axial coordinates. The loader
combines parts in their original `order` before ordinary module, collision and
whole-instrument validation.

## Editing in Physical Layout

Open a recording component in **Physical Layout → 3D Parts**. The file label
shows both the installed module and the component's actual definition file.
Existing dimension, material, feature, new-component and independent-copy
controls continue to work. Saving a coil dimension writes its subassembly;
the wrapper is not rewritten unless its own content changes.

**Subassemblies / placement…** opens an axial placement editor. Choose a
subassembly, then either a fixed module-local origin or a persistent reference
to another component's start, center or end. The table previews affected
components' previous and new center positions. **Apply to draft** changes the
3D draft, supports Undo/Redo, and does not save files. **Save** checks all
compatible instruments before writing. Edits to a referenced component also
move explicitly dependent subassemblies in the draft.

The anchor equation, in millimetres, is:

`subassembly origin = reference component Z + offset − local datum`

Default placements preserve the existing layout exactly:

- Projector stack: fixed origin at local Z 0.
- Detector chamber: origin at the end of the P2 housing (local Z 772.5).
- Iliad filter: origin 38.75 mm after the camera's end (local Z 1211.5).

The default offsets are compatibility placements, not new OEM measurements.
Individual length edits retain their existing center-preserving behavior.
Only explicit constraints move dependent groups; unrelated groups are not
automatically packed together. Existing envelope/clearance limits remain
enforced. A placement outside the module's configured envelope requires an
explicit compatible envelope change, rather than silently enlarging it.

Filter-internal `path_*` coordinates remain curvilinear distances referenced
to the filter entrance/prism exit. Moving the filter's Z origin does not
translate its internal path s. All existing upstream detector absorption and
the separately executed energy-filter path remain in the particle model.

## Storage, compatibility and invalidation

The stage-1 instance map is unchanged. Runtime component authority remains the
legacy module key and part key; the new physical file locations are storage
dependencies, not newly created optical instances. Historical snapshot
dataclasses were not extended or silently converted.

All referenced files enter calculation input identities, frozen snapshots,
catalog copies and wheel packaging. An edited file invalidates its previous
identity. No archived trajectories are admitted as a downstream source.
The dimension audit expands the composition and still covers 11 selectable or
historical modules and 482 part definitions.

Saving validates a temporary complete catalog, checks captured revisions,
then replaces the affected files. External revisions prevent overwriting a
stale draft. File replacement failure or runtime assembly reload failure
restores the transaction's previous files. Dimension/material edits preserve
part-file comments; changing placement tables may reformat the wrapper TOML.
New components are independent root additions. Cross-file component copies
retain the existing independent mechanical-copy behavior. **Save copy** of
a composed module materializes an independent flat file and detaches links.

This stage supports one level of subassemblies and acyclic axial constraints.
Missing files, duplicate instances/keys, nested composition and reference
cycles fail explicitly. It does not introduce free 3D assembly joints or
automatically reclassify the vacuum regions. Gun/column selection wrappers and
the stage-1 functional tree remain available for the subsequent UI migration.
Coherent-wave development remains paused; vacuum transport stays opt-in.

## Verification

- 734 distinct related regression cases passed after adapting old standalone
  TOML fixtures to materialize the new composition. Coverage includes physical
  dimensions, 3D selection, materials, copies, transactional saves, stale-file
  detection, rollback, detector edits, identity maps and working-point packages.
- All 30 supported assembly combinations validate, including from the three
  subassembly files extracted from the built wheel.
- The default 141-part identity/position map is exactly equal to stage 1.
- The final default flat-tip calculation was executed afresh from the tip,
  through extraction, acceleration and the column, then through the filter.
  All 81 numerical arrays equal the original frozen baseline, with maximum
  absolute difference 0. Sample Z remains 1599.2 mm and 25 of 49 preview rays
  reach the sample plane.

The comparison explicitly permits only the named EnergyFilter storage repack.
It verifies equality of the entire resolved physical document and state graph,
checks all new dependency bytes, and rejects unrelated input changes. Different
code and storage hashes are recorded, not treated as reusable old computation.

Original local baseline: `outputs/assembly-baselines/flat-tip-before-20260916-01/`.
Final comparison: `outputs/assembly-baselines/flat-tip-stage2-20260916-final/`.
Lightweight receipt: `evidence/assembly-structure-stage2-20260916.json`.
Generated arrays, archives, test XML and wheel outputs remain local and ignored.
