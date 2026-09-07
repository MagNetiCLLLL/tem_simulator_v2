# Maintenance and cache verification — 2026-09-07

Baseline: `12ff7e3` (`add 3d model editor`). No instrument TOML dimensions,
lens excitations, specimen defaults, or calculated preset values were changed.

## Repairs

- Preserve each parameter panel's independent TOML draft across geometry saves,
  including uncommitted cell-editor text and freshly resolved runtime objects.
- Rebuild 3D aperture meshes only when their displayed runtime dependencies
  change. Hidden pages defer work; module views include all displayed openings.
- Replace complete material-assignment tables regardless of TOML inline,
  dotted-key, or expanded-table formatting, retaining comments and other fields.
- Validate overlap against actual material intervals, with parent-defined
  objective sections taking precedence over child intervals.
- Preserve original LF, CRLF, and mixed line endings in transaction rollback.
- Export calculation manifests as detached JSON-compatible data, without
  changing scientific identities or cache signatures.
- Install the declared `manifold3d` dependency in the project environment.

## Scratch cleanup

Removed 28 generated directories beneath `tmp/`: 8,180 tracked files,
253,945,554 bytes (242.18 MiB). These were isolated pytest/preview fixture
copies, packaging copies of installed dependencies, and a multiprocessing
marker. No production code or formal test depends on those directories.

Removed five superseded one-off scripts, keeping their historical logs:

| Retired script under `tmp/` | Maintained regression coverage |
| --- | --- |
| `project_inspection_repro_cache.py` | `tests/test_artifact_quota.py` |
| `project_inspection_repro_descan.py` | `tests/test_stem_recording_deflection.py` |
| `project_inspection_repro_profile.py` | `tests/test_profile_optional_values.py` |
| `project_inspection_repro_stem.py` | `tests/test_stem_recording_deflection.py`, `tests/test_wave_imaging.py` |
| `project_fix_wave_baseline.py` | `test_defocused_image_wave_uses_symplectic_specimen_canonical_transfer` in `tests/test_wave_imaging.py` |

The retired scripts either assert former defects or repeat a formal test.
No formal test was removed simply because it failed or took time to run.
The formal test sources had no byte-identical duplicate files or overwritten
duplicate test definitions in the same scope during the cleanup audit.

Scientific baseline/convergence/calibration directories, original PDFs, NPZ
before/after data, and historical reports and screenshots remain intact.
Previously tracked retained files stay tracked; new scratch files and
`outputs/high_accuracy/` are ignored. Deleted tracked artifacts can be
recovered from commit `12ff7e3` if needed.

## Reproducible cache population

`scripts/cache_high_accuracy.py` runs one explicit request using the same
snapshot, calculation pipeline and persistent incident-seed codec as the GUI.
Choose either `--profile PATH` or `--startup-defaults`; `--describe` only
inspects the request. The default route loads stored startup operating values
without a new lens-preset solve. It never changes an already open GUI.

```powershell
.\.venv\Scripts\python.exe scripts/cache_high_accuracy.py --startup-defaults --describe
.\.venv\Scripts\python.exe scripts/cache_high_accuracy.py --startup-defaults --output outputs/high_accuracy/NEW-RUN
```

An output directory must be new. It retains the request profile, manifest,
and a verification report written only after a successful calculation and
cache readback. Persistence uses the application's real per-user cache and
configured disk quota. Readback validates checksums and compares phase space,
current weights, checkpoint arrays and the propagation plan.

Current persistence scope is **incident propagation only**. Complete TEM,
STEM, and EDS results are not restored automatically across application
restarts by this codec. Loading the saved request profile and requesting the
same calculation lets the GUI reuse the incident seed; downstream products
may still need calculation. This is not a claim of full-result persistence.

## One completed physical calculation

The authorized run used desktop startup defaults: FEG, C3 + Probe Corrector,
Energy Filter, Ideal Optics, 300 kV, virtual Si[110], 15,000 rays, and a 0.1 mm
integration step. Default-disabled TEM/STEM wave calculations stayed disabled.
The pipeline reached `Complete`, including the requested specimen/EDS stages.

Windows package redirection sent the newly written cache files into Codex's
private LocalCache while the nominal cache root resolved outside it. The
store correctly rejected that inconsistent path during readback. The complete
incident object and reference were recovered into
`outputs/high_accuracy_artifacts/`; no path-safety check was relaxed and the
physical calculation was **not repeated**.

Verified retained data: 49 files, 1,434,564,101 bytes before the cache lock;
incident array shape 2,526 axial records by 15,000 rays; 7,891 surviving sample
rays. Array checksums, metadata, logical content digest, external inputs and
request identity were validated by the application's normal incident-seed
reader. The pre-recovery process did not retain its elapsed-time report, so
no timing or speedup claim is made.

The request profile, full manifest and successful readback record are local:
`outputs/high_accuracy/2026-09-07-default-verified/`. Recheck existing data
without running physics:

```powershell
.\.venv\Scripts\python.exe scripts/cache_high_accuracy.py --verify-existing --output outputs/high_accuracy/2026-09-07-default-verified --cache-root outputs/high_accuracy_artifacts
```

The recovered cache is separate from the normal per-user store. It is an
optional fallback for this source checkout, consulted only after the normal
store misses or fails, with the same scientific identities and checksums.
Disabling persistence or explicitly supplying a test/custom cache isolates
the caller from this fallback. Existing per-user cache contents are retained.

## Verification

- A combined 26-file affected regression run passed 722 tests in 153.44 s.
  Coverage includes geometry transactions, 3D runtime refresh, aperture and
  CSG models, parameter semantics, profiles, cache quota and recording paths.
- The cache utility's nine offline tests passed in 2.24 s. They check retained
  arrays and failure reporting; they do not run physical calculations.
- Recovery routing and existing cache/controller checks passed 36 tests in
  15.40 s, including 19 new mocked or temporary-directory routing tests.
- A separate real-data check captured the CURRENT default GUI request again,
  without calculating it. All product signatures and solver identity matched
  the saved manifest. `CalculationWorker._load_persistent_incident_seed`
  loaded the recovered (2,526, 15,000) data with 7,891 surviving sample rays.
  The calculation entry point was replaced with a fail-if-called guard during
  this check, confirming that recovery did not repeat propagation.
- Serial compilation of `src`, `tests`, `scripts` and `main.py` passed.
  These are targeted regressions, not a claim that the entire suite or native
  OpenGL interaction was validated. An existing pyqtgraph shutdown-disconnect
  warning did not fail the combined run.

Restart the application to load the updated code. With the same default
request, its next High accuracy request can reuse this incident seed. The
request still calculates any required downstream products that are not cached.
The local cache and run outputs are not intended for Git upload.
