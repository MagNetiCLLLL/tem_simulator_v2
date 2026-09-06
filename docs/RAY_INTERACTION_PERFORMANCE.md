# Ray interaction and cache settings

Open **Simulation > Performance and cache...** to adjust retained data. Apply
does not launch a calculation, change the instrument, or reset the current plots.
Settings persist across restarts and are independent of named workspace layouts.

| Retained data | Default upper limit | Purpose |
| --- | --- | --- |
| High-accuracy results | 8 GiB / 32 entries | Reuse complete results and dependency-compatible stages |
| Live-tuning results | 1 GiB / 128 entries | Revisit exact Preview or Medium settings without retracing |
| Ray display | 512 MiB | Reuse clipped line coordinates and projected-angle statistics |
| Disk checkpoints | 16 GiB | Restore supported incident checkpoints between sessions |

RAM defaults scale down on smaller computers. The settings dialog limits the
combined managed RAM budgets to half of detected physical RAM, leaving room for
workers, displayed results, other caches and the operating system. These are
retention limits, not preallocations or a whole-process memory guarantee. The
existing per-calculation memory guard remains separate. Advanced-bank pinned
memory retains its separate explicit limit.

Oldest unused history entries are evicted when a retention limit is reached.
Shared NumPy storage is counted once within each calculation cache. Eviction
releases history ownership; it does not erase an already displayed result or
invalidate a worker's seed. Statistics use retained metadata, not repeated array
or disk scans. Disk quota changes are enforced on the next checkpoint write.
Only supported checkpoints persist on disk, not all TEM/STEM/EDS images.

## What became faster

- Main-window Preview/Medium requests capture independent editable values on
  the GUI thread. Configuration loading, assembly normalization, reference-plane
  lookup and full model/request signatures are prepared in the background.
  Cache lookup and publication remain on the GUI thread under the same request
  generation. The original live-geometry model identity and the authoritative
  normalized calculation identity remain separate and unchanged.
- Hidden Physical Layout, Magnetic Field and Transverse beam panels keep only
  their newest pending result. They update when visible, retaining requested
  plane, projection, selection and user plot ranges. Hidden Magnetic Field no
  longer performs its field/rotation diagnostic traces. Until updated, dependent
  text explicitly says its diagnostics are pending rather than showing old
  numbers as current. EDS, TEM/STEM and shared result publication is not deferred.
- A live scalar edit validates the component, bounds and complete edit batch
  without reconstructing the instrument. Existing parameter editors update in
  place. The actual worker still receives an independent resolved snapshot.
- Repeated analytical Objective reference-plane solves reuse exact scalar
  results. Field parameters, Gaussian terms, voltage, integration boundaries and
  resolution participate in the key. Custom per-instance field implementations
  and subclasses bypass this analytical cache. The cache holds at most 1024
  scalar entries, not instrument objects.
- Preview and Medium retain separate, bounded exact-result histories. Reuse
  still requires the existing complete request signatures. New settings use the
  existing dependency-checked segmented propagation; there is no interpolation
  between lens strengths. Superseded callbacks remain generation-guarded.
- Rotating Ray Diagram updates existing graphics, including aperture openings
  and detector offsets. Display-only coordinates and slope summaries are reused;
  published replacement results invalidate those derived caches. Pan, zoom and
  plane selection keep the user's plot ranges. Explicit Fit remains available.

Preview/Medium still defer specimen scattering, multislice, spectra and scanning
images. High-accuracy products remain retained and are marked stale after relevant
physical edits; low-count tuning does not replace them. Cache reuse does not
increase the underlying numerical accuracy or guarantee a fixed frame rate.

The background path currently applies to Preview/Medium. Explicit High accuracy
keeps its existing synchronous request validation/deduplication and numerical
worker. Native field/propagation kernels are not force-killed: obsolete work is
discarded at generation-checked phase boundaries. Live dragging still completes
one frame while keeping only the latest pending setting. Closing the window
invalidates pending preparation before waiting for workers.

## Reproducible checks

`scripts/benchmark_live_tuning_edits.py` compares the old per-edit snapshot
validation path with scalar validation. It excludes ray tracing and painting.
The old path now benefits from the shared Objective root cache too.

`scripts/benchmark_ray_display_cache.py` measures warmed display-data preparation
for a deterministic synthetic bundle: 2048 axial planes, 512 rays, 48 displayed
rays and 256 slope samples. It excludes Qt painting, GPU work and physical
propagation. Neither script calculates lens presets or full high-accuracy images.

`scripts/benchmark_request_preparation.py` separates the previous foreground
model/snapshot/signature preparation from new GUI capture and background
preparation. It checks exact snapshot and signature equality. This measures
work moved off the GUI thread, not work eliminated from the physical solver.

Local Windows measurements on 2026-09-06 (warmed CPU data preparation):

| Operation | Before / reference | After |
| --- | --- | --- |
| Live-edit validation, 12-repeat median | 168.34 ms with per-edit snapshot and new root memoization | 0.093 ms |
| Repeated full snapshot construction, 5 warm repeats | 695.84 ms without root memoization | 154.05 ms |
| Synthetic ray rotation data | 1.600 ms | 0.068 ms |
| Synthetic pan slope statistics | 6.006 ms | 0.012 ms |
| Request preparation on GUI thread, 8-repeat median | 530.39 ms | 5.57 ms capture |

The synthetic display benchmark retained 10.1 MiB. Measurements describe these
operations only; they are not end-to-end frame rates or full solver speedups.
The request benchmark ran separately after tests: a further 545.13 ms of
preparation moved to the background. It did not disappear. All eight requests
matched the previous snapshot and both exact signatures; neither path traced
rays or painted widgets in this benchmark.

Regression coverage checks exact projection equivalence, clipping endpoints,
scan offsets, bounded memory, cache invalidation, stale callback rejection,
retained high-accuracy products and persistent settings without new solves.

The initial cache/display optimization check passed 126 tests in 149.10 s,
including real small-bundle
frames during a held slider drag. Qt tests used the offscreen platform and only
the required pytest-qt plugin; unrelated Napari auto-discovery was disabled
because its cache initialization attempted an unavailable external directory.
This is not a full desktop/GPU rendering or high-accuracy image benchmark.

The background-request/lazy-panel follow-up passed 182 focused tests in three
separate selections:

- 97 request/controller/cache-capacity/cache-reuse tests (93.91 s).
- 33 lazy-panel/ray-display tests and 12 selected GUI-shell regressions.
- 40 main-window preparation, slider, live-dock, cache-dialog and workspace-layout
  tests (101.01 s), including a real small-bundle CPU solver during held dragging.

These cover detached request ownership, edits while preparing, superseded work,
preparation errors, reentrant cache delivery, newest-only hidden panels, current
Ray/Magnetic X bounds, hidden Z/projection changes, retained high-accuracy data
and viewport-dependent component-label placement. No preset recalculation or
full high-accuracy image run was performed. Python compilation and whitespace
checks passed. An existing non-fatal pyqtgraph destroyed-signal disconnect
warning remains at Qt teardown.

## Stage 1: incremental result presentation

The enhancement roadmap's first stage now keeps value-keyed component, vacuum
wall, specimen and crossover layers in the Ray Diagram. A new result updates
existing ray and intercept groups with their new coordinates. Only changed or
removed drawing layers are replaced; completed-result publication no longer
clears the whole scene after the initial waiting notice.

Drawing signatures include mechanical positions, aperture openings and offsets,
detector geometry and insertion/readout state, specimen state and crossover
coordinates. They do not substitute a geometry-only key for a physical result
identity. Published arrays still invalidate the display-data cache, including
explicit republication of the same mutable result object.

Selected-Z cursor identity and user X/Y ranges survive new results, including
geometry changes. The selected Z is clamped only when outside the new simulated
extent; a subsequently selected detector remains the transverse-view focus.
Publication also resolves a pending projection angle for apparatus graphics, so
rays, aperture gaps and detector ranges cannot display different angles.

Magnetic Field retains total/per-lens curves, specimen and reference-plane
markers, legend samples and selected support graphics. New snapshots still
evaluate the authoritative field and rotation providers: this is graphics reuse,
not an additional field-value cache. User Y ranges and the linked Ray Diagram X
range remain unchanged after the first field presentation.

`ray_scene_info()` separates synchronous ray-scene update time from solver time,
optional-panel preparation and deferred painting. The new reproducible benchmark
is:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_incremental_ray_scene.py --repeats 10 --warmups 2
```

It uses three prescribed synthetic bundles (2048 planes and 512 rays each,
48 displayed rays per bundle), 80 synthetic components and a 1600 x 900 offscreen
Qt workspace with Transverse X-Y visible. No field, specimen, high-accuracy image
or preset solve is performed. Workspace creation and fixture generation are
excluded. Synchronous update, bounded Qt event dispatch and a separate forced
raster capture are reported independently. A forced capture is not added to the
event-inclusive time as though it were an extra display stage.

The 50 ms display-response target is evaluated separately from solver speed.
Passing synchronous-update timing alone does not establish the complete target
or an actual desktop/GPU frame rate.

### Measured result and validation (2026-09-07)

The final benchmark ran separately after the test processes stopped, with ten
measured samples and two warmups per operation. The
[complete benchmark output](benchmarks/RAY_SCENE_2026-09-07.json) records the
environment, fixture and counters.

| Operation | Synchronous update median / p95 | Update plus event dispatch median / p95 |
| --- | --- | --- |
| First publication in a fresh scene | 263.38 / 386.95 ms | 788.13 / 908.03 ms |
| New result with unchanged geometry | 94.87 / 98.56 ms | 270.60 / 277.70 ms |
| Projection rotation | 40.50 / 46.25 ms | 210.26 / 217.37 ms |
| Selected-Z movement | 26.45 / 29.39 ms | 77.89 / 126.22 ms |

Ray-scene work alone decreased from a 228.62 ms median for fresh construction
to 70.57 ms for same-geometry publication. These are two paths in the current
implementation, not a controlled old-version/new-version speedup. Across twelve
publications, 81 unchanged static layers were reused each time; only the changing
synthetic crossover layer was replaced. Rotating and moving Z caused no further
scene-build counters to increase. Display-data cache use was about 31.34 MiB.

The synchronous rotation/Z targets passed, but their event-inclusive p95 values
exceeded 50 ms. The full display-response target remains open. Forced software
raster capture alone had medians of approximately 155-159 ms for these reused
scenes. Further painting optimization requires separate work; it must not be
reported as a solved performance target or achieved by silently changing the
underlying ray data. Native desktop/GPU rendering remains unbenchmarked.

Final combined regression selection: **110 passed in 288.84 s**. This includes
24 new incremental-ray cases, seven magnetic-scene cases, existing display-cache
and lazy-panel checks, calculation-cache retention, shared image sources and
held-slider refresh. Another 16 selected existing GUI-shell tests passed in a
separate run. The new scene tests use prescribed synthetic data and verify
equivalence to fresh drawing, including deferred-rotation publication, detector
focus retention, in-flight cursor dragging and failed replacement cleanup.
Existing small-bundle integration tests are not full high-accuracy calculations.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -p pytestqt.plugin tests/test_incremental_ray_scene.py tests/test_incremental_magnetic_scene.py tests/test_ray_display_cache.py tests/test_lazy_ray_panels.py tests/test_calculation_cache_reuse.py tests/test_shared_image_sources.py tests/test_live_slider_refresh.py --override-ini=addopts= -q --tb=short
```

The existing non-fatal pyqtgraph destroyed-signal warning remains at teardown.
No lens preset recalculation, full high-accuracy image run, numerical-model change
or implementation of roadmap stages 2-6 was performed.
