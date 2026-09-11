# Working-point implementation — HANDOFF v2

## Baseline and scope

The supplied 2026-09-10 handoff supersedes the independent specimen-entrance
illumination design. Development starts at commit
`5eac9855ff8eefa2a3d68ddba3029f7c23e03b0c`. Historical validation reports are
not evidence for this revision. This first slice was originally local-only;
the later publication instruction and current scope are recorded in
`HANDOFF_V2_ACCEPTANCE.md`.

This document records the earlier first data/control slice. The newer
`GUN_SOURCE_V1.md` and `HANDOFF_V2_ACCEPTANCE.md` supersede its implementation
status: source modes, quadratic upstream transport, working-point browser and
Direct Alignment transactions now have partial implementations and new scoped
tests. The full source-to-image chain and all release gates remain incomplete.

## Implemented in this slice

- `InstrumentSnapshot` captures the effective model graph, shared component
  ownership, disabled hardware, exact parameter values, assembly information,
  field-map arrays, and content-pinned external inputs. It does not use the
  intentionally lossy/migrating operating-profile serializer for restoration.
- Frozen JSON and byte-backed arrays prevent caller edits from modifying a
  saved working point or numeric propagation checkpoint. Export returns a copy.
- Restoration returns a **new State**. It does not replace the GUI's current
  state, run constructors/presets, resolve geometry, or refocus. Magnetic-map
  interpolation accelerators alone are rebuilt from captured field values.
- Model types are allow-listed. Changed dataclass field schemas, unknown model
  objects, bad hashes, missing files, changed file contents, and changed solver
  implementation reject continuation. Their serialized records remain readable.
- External bytes are retained for evidence, but this revision does not rewrite
  changed live files or automatically remap loaders to archived model files.
- High-accuracy request capture preserves effective lens/geometry values and
  changes only requested ray sampling and integration/history steps. Preview
  capture is unchanged to avoid adding full-snapshot work to slider updates.
- Calculation manifests contain the complete snapshot. Incident checkpoint
  and restart-seed artifacts persist it with the numeric product. Completed
  GUI calculation results retain their immutable manifest.
- Persistent numeric reuse includes the installed source-code digest, even
  for uncommitted changes with an unchanged package version. Historical
  manifests without the new fields remain readable, not upgraded implicitly.
- Snapshot validation precedes cancellation, cache eviction and worker
  dispatch. A rejected snapshot leaves the previous request and cache intact.
- Seed diagnostics record current-weighted alpha95/alpha99, finite sampled
  maximum, radius95, beam centre, and source fraction. The physical pupil edge
  remains unavailable unless independently known. Diagnostics use the retained
  **sample entrance** plane, not the last upstream restart plane. Coordinate
  storage precision and the estimator definition are explicit.
- Independent `specimen_entrance_pupil_modes` inputs are rejected in new
  calculations, profile application, working-point restoration and TEM
  reprojection. The former source editor is read-only and its production
  button is hidden. No GUI or private source-node flag bypass is provided.
- The existing `ray_conditioned_reduced_order` approximation remains explicitly
  labeled as such during transition. It is **not** a validated gun-derived
  coherent mode, nor proof of the handoff's final unique wave-source chain.

## Developer use

```python
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot

saved = capture_instrument_snapshot(state)
document = saved.to_dict()          # Detached JSON; viewing does not apply it.
loaded = InstrumentSnapshot.from_dict(document)
candidate = loaded.restore()       # Validates dependencies; no GUI mutation.
```

Incident artifacts written by `ArtifactStore.put_incident_simulation_seed`
contain `metadata["working_point"]`, `metadata["full_snapshot_id"]`, and
`metadata["beam_observables"]`. Their dependency signature still identifies
the local product; the full snapshot identifies the original working point.
Reading a compatible upstream artifact does not claim that its original full
snapshot is the current full instrument. The artifact content digest is a third,
separate identity.

The explicit **Working Points** restore/fork/import/export UI is now implemented.
Normal operating-profile loading retains its existing migration semantics and
must not be described as exact working-point restoration.

## Roadmap / remaining acceptance

| Package | Current status | Remaining work |
| --- | --- | --- |
| NWP00 | Partial | Finish source-entry inventory and isolate retired numerical helpers as test fixtures |
| NWP01 | First slice | Broader assembly/model coverage, archived dependency remapping, payload deduplication |
| NWP02 | First slice | Shared GUI/DA observable registry and lazy derived products |
| NWP03 | Not implemented here | Phase-carrier and canonical-coordinate interfaces and checks |
| NWP04 | Not implemented here | Gun-generated coherent/mixed modes through validated component operators |
| NWP05 | Not implemented here | Verified field/shared-circuit dependency graph and complete restore/reuse workflow |
| NWP06 | Not implemented here | Registered DA target transaction, final validation, stale/cancel/undo protection |
| NWP07 | Not implemented here | Explicit read-only working-point browser and atomic restore/apply UI |
| NWP08 | Not run | Actual 32/40 mrad domain and independent physical reference validation |
| NWP09 | Not run | CPU/GPU production-scale performance and release qualification |

G1, G2 and G3 are **not claimed complete**. A cache hit does not certify physics.
No TEM/STEM source-coherence benchmark or production high-accuracy image run
is claimed by the first-slice software tests.

## Validation

Targeted commands (run with the project virtual environment):

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p pytestqt.plugin tests/test_working_point_contract.py tests/test_calculation_cache_reuse.py tests/test_illumination_modes.py tests/test_tem_flux_contract.py tests/test_execution_migration_contract.py tests/test_profile_optional_values.py
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p pytestqt.plugin tests/test_calculation_manifest_artifacts.py tests/test_high_accuracy_cache_cli.py tests/test_segmented_column_cache.py tests/test_background_calculation_requests.py tests/test_artifact_quota.py
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p pytestqt.plugin tests/test_cache_settings_integration.py tests/test_background_preview_gui.py -k 'not live_edits'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p pytestqt.plugin tests/test_background_preview_gui.py::test_live_edits_during_preparation_keep_one_frame_and_latest_pending
```

The old production-pupil acceptance source remains in
`tests/historical/illumination_wp03_contract.py` for traceability and is not
collected as current acceptance. Pure mathematical checks remain active in
`tests/test_illumination_modes.py`; production-entry rejection tests replace
the withdrawn production-source expectations. Physical flux/aperture tests
were not relaxed.

Historical first-slice observations on 2026-09-10 (not rerun evidence for the
current source-model revision):

| Targeted group | Result |
| --- | --- |
| Working-point contract, reuse, source gate, TEM flux, execution migration, profiles | 112 passed (108.94 s) |
| Artifact manifests, CLI readback, segmented column, background preparation, quotas | 81 passed (145.49 s) |
| Cache-settings window and Preview/Medium/error routing | 4 passed (17.84 s) |
| Live edits during background preparation, isolated | 1 passed (12.85 s) |

Total: **198 distinct targeted tests passed**. All four final commands exited
successfully. Reports are local `.pytest_cache/working-point-contract-final.xml`,
`working-point-storage-final.xml`, `gui-final.xml`, and `gui-isolated.xml`.

An earlier combined GUI invocation did not finish and its test processes were
stopped. The final GUI runs above passed in separate processes; the combined-run
stall was not independently diagnosed and is not claimed fixed.

The working-point regression includes an actual small nine-ray/default-column
CPU propagation, disk save/readback, and restored-state propagation. It compares
sample-entrance alpha95 to retained coordinates, and verifies raw lens values.
Other tests use controlled software fixtures where appropriate. This is not a
production-size high-accuracy image, a GPU benchmark, or a full-suite/physical
accuracy qualification. No preset values were recalculated for delivery.

After tests, `python -m compileall -q src scripts tests` and `git diff --check`
both passed. At that first-slice checkpoint, changes were local and uncommitted
and no push had been performed. See the current acceptance ledger for the
subsequent in-progress publication authorization.
