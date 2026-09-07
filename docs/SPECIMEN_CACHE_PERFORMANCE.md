# Specimen preparation and display reuse

This optimisation changes retention and repeated work, not specimen sampling,
electron counts, slice thickness, random seeds or detector acceptance.

## Independent caches

| Cache | Default cap | Reused data |
| --- | --- | --- |
| Specimen potentials | 256 MiB RAM, 8 entries | Potential configurations, mean projection, axes and slice thicknesses |
| Sample atom display | 128 MiB RAM, 128 entries | CIF display atoms, element numbers, bonds and display geometry |
| CUDA transmission | Smaller of 512 MiB and one eighth of free VRAM | Slice transmission functions within one resident STEM calculation |

RAM defaults scale down on smaller machines. The first two limits are available
under **Simulation > Performance and cache...** and persist independently of
physical settings and workspace layouts. The managed RAM safety allowance
includes these new limits. They are retention caps, not preallocated memory or
a bound on the whole application's memory. Active results, graphics, workers,
CuPy allocation pools and temporary copies use additional space. Cache admission
may create an immutable copy of a potential that fits its configured budget;
oversize products bypass retention without that extra full copy.

These caches are process-local. Existing on-disk incident checkpoints and
complete-result reuse are unchanged. No saved user result is deleted.

### Specimen potentials

Exact keys include the parsed specimen preset, current CIF bytes, specimen
state/shape/centre/extent, crystal orientation, laboratory ROI, requested grid
and FOV, slice thickness, frozen-phonon parameters and potential backend identity.
File content is checked even when the path, size and modification time match.
Different physical inputs never reuse a merely nearby cached configuration.

Incident waves, lens strengths, electron energy and detector response are not
potential data. Changing them can reuse a compatible potential, but still
requires the appropriate new wave propagation or detector calculation. If a
focus change enlarges the resolved ROI/grid, a new potential is necessary.
TEM and STEM share a product only when their actual preparation inputs match.

Admitted arrays have immutable backing storage; each caller receives separate
metadata. Concurrent identical requests share one preparation. Failed builds
and temporary atomistic-to-qualitative fallbacks are not retained. A file change
during preparation prevents admission under the old content key.

### CUDA transmission functions

Each retained frozen-phonon configuration captures the same `exp(i sigma V)`
expression used by the reference GPU path. It is reused across probe batches,
not interpolated between potentials or electron energies. Handles belong to
one plan, including its energy-dependent interaction constant and sampling.
Raw mutable potential arguments continue to be evaluated from their current
contents. Capacity exhaustion or optional cache allocation failure uses the
existing on-the-fly GPU calculation. Whole-frame CPU retry remains available
for a failed CUDA pipeline. This does not add GPU support for raw 4D-STEM sinks.

### Sample display

The display cache is separate from physics and retains bounded equilibrium
atom previews only. Reusing it does not change the calculation ROI, displayed
full-specimen outline or structural-preview provenance. Repeated geometry uses
existing graphics; view changes do not rebuild a specimen potential. A hidden
Sample page retains its newest refresh request and prepares its scene when
shown. The rendering atom limit remains a display limit, not a physical limit.

## Reading performance information

- The calculation log lists disjoint stage wall times for the current pipeline
  call. These exclude GUI drawing and worker setup; their sum need not equal
  the toolbar's total elapsed time.
- TEM/STEM lines report the actual wave backend and potential cache hit/build.
  Potential preparation and GPU diagnostics are nested within their wave
  stage, not extra intervals to add to the total.
- A complete result-cache hit does not repeat the old acquisition's timings.
  Recollection from a retained diffraction cube reports readout time, not old
  potential preparation or GPU propagation counts.
- **Refresh statistics** in the cache dialog shows retained bytes, entry counts,
  hits and hit rate. No new physical calculation or periodic disk scan is run.

## Small preparation benchmark

A real CPU Si [110] potential-only fixture used 1 nm thickness, requested
64-pixel/16-angstrom sampling, 2-angstrom slices, no frozen phonons and seed 100.
The generated potential contained 144 atoms with shape `(6, 65, 61)`.
The backend was abTEM 1.0.10 with ASE 3.29.0.

After backend warmup, a cold cache build took **0.038696 s**. Three hits took
**0.001388 / 0.001173 / 0.001393 s** (median 1.388 ms). Potential configurations,
mean potential, axes and slice thicknesses were exactly equal to the uncached
reference. One entry retained 135,637 bytes in that run. The earlier first call
took 3.161808 s including startup and is not a fair warmed speedup baseline.

Reproduce the bounded fixture with:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_prepared_specimen_cache.py
```

This does not trace a column, calculate an image or run a full 100 x 100 scan.
These timings are local stage measurements, not promised acquisition speedups.

## Small display-data benchmark

An independent synthetic CIF fixture used ASE diamond Si with a 5.43-angstrom
cubic cell, zone [110], in-plane [1, -1, 0], a 10 nm diameter/thickness disk,
an XY request of [-5, +5] nm and a 2,500-atom display limit. The capped display
window was approximately 3.303 nm per axis, with 1,815 atoms and 3,264 bonds.
One repeat measured **39.37 ms uncached**, **33.49 ms cold cache**, and
**0.857 ms per warm hit** (mean of five). Positions, element numbers, bonds and
cell data matched exactly. This measures structural display-data preparation,
not OpenGL painting, GUI FPS or physical specimen propagation.

## Rendering verification

A separate hidden native Qt window on the local RTX 5090/OpenGL 4.6 renderer
used 36 prescribed Si display sites (not a specimen calculation). A draft
rotation and subsequent redraw retained all seven graphics items and the atom
mesh: one model build, two reuses. Coordinate checks and changed framebuffer
pixels verified that draft and camera rotations still worked without resetting
the camera on redraw. The styled cache dialog fit all six budget controls.
Its displayed cache statistics were synthetic UI fixtures. No active user
window, physical setting or saved result was changed.

## Validation scope

The final focused integration selection passed **343 tests in 227.19 s**.
It included real CUDA comparisons, exact cache-on/off wave results, immutable
ownership and concurrent preparation, file-content and geometry invalidation,
optional-cache memory failure, retained-result/controller checks, hidden-page
coalescing and settings/layout regressions. The tiny CUDA fixture emitted an
under-utilisation warning; the existing pyqtgraph teardown-disconnect warning
also remains. This was not the full repository suite or a production scan.
