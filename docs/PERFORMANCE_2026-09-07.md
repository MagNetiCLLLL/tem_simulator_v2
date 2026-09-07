# STEM wave and specimen/EDS throughput

## Specimen/EDS follow-up

The screenshot's `Elastic specimen history 7891/7891` reports completion of
electron histories, not completion of EDS or the STEM frame. EDS still generates
vacancies/lines, traces photons through the specimen/support/pole geometry, bins
the spectrum, and records its event ledger. The former progress calculation
mixed electron counts with probe-batch counts, assigning almost all bar space
to EDS and showing approximately 99% too early.

This follow-up preserves the physics, floating-point accumulation order, event
identities, random seeds, trajectory count and discretisation:

- The event ledger indexes material flights once by ray/source/material/history.
  The original deterministic position sampler receives the matching flights in
  their original order, including duplicates. A work-count test reduced scanned
  flight records from 172,800 to 1,440 plus 360 one-time indexing entries. These
  are operation counts, not a whole-application speedup measurement.
- Photon transport retains up to 1,024 exact origin/direction intersections per
  acquisition. Different transitions from one vacancy reuse geometry, while
  attenuation, detector weight, energy-dependent blocking and identity remain
  photon-specific. New acquisitions cannot inherit stale geometry.
- Atomic attenuation (4,096 entries) and relaxation yields (1,024 entries) have
  bounded exact-key caches. Gaussian detector-response weights are reused at
  identical line energies within one spectrum. Counts are still accumulated in
  the same order; noise is sampled only after the final expected spectrum.
- Each specimen-field context retains at most eight exact scalar positions.
  Boundary searches and subsequent flights can reuse repeated field queries.
  No interpolation or position rounding is introduced; returned arrays cannot
  mutate retained data. Batch arrays bypass the cache. Capacity zero retains
  the uncached reference route.
- Progress now shows named, equally allocated pipeline stages and local work:
  elastic histories, ionisation tracks, photon emissions and spectrum lines.
  Event-ledger finalisation has its own label. Stage fractions are bookkeeping,
  not estimates of remaining time. Main and bank displays use the same scheme;
  cached stages are omitted and nested resets cannot move the outer bar back.

The field microbenchmark used 32 prescribed rays with distinct positions and
energies, 100 nm thickness, seed 101 and 22 elastic events. Three paired warm CPU
runs took 0.3497/0.3531/0.3532 s uncached and 0.2618/0.2692/0.2884 s cached.
Medians were **0.3531 s versus 0.2692 s** (23.8% less time for this fixture).
An independent cold measurement was 0.4234 s. Each cached run reused 219 of
712 field requests; terminal arrays, trajectories/events, EDS material tracks,
flights and metrics were exactly equal. This is not a prediction for 7,891
histories or for a complete acquisition.

A separate photon-only benchmark used 100 prescribed emission origins, four
Si K transitions and six quadrature directions: 2,400 photons and 600 distinct
paths through the installed two-pole-piece/sample/support geometry. Warm CPU
timings were 0.24206/0.26017/0.24414 s without the geometry cache and
0.07885/0.10699/0.07836 s with it. Medians were **0.24414 versus 0.07885 s**,
about 3.10x for photon transport only. Complete result objects, stored paths,
order, counts and metrics matched exactly in every pair. Reproduce using
`.\.venv\Scripts\python.exe scripts/benchmark_eds_photon_cache.py`; no electron
propagation or new specimen simulation runs in this benchmark.

The outer elastic-history loop still uses one shared seeded random stream.
Naively splitting histories across threads would change its variable random
draw order, so this follow-up does not do that. Higher utilisation is not itself
a correctness or speed metric. The earlier GPU STEM optimisation below is
separate from these CPU-side specimen/EDS stages.

Follow-up validation: **122 focused tests passed in 44.63 s**, covering EDS
spectra/Poisson equality, exact caches, photon shadows, field transport, ledger
events, point progress, cached-stage reuse and GUI/controller integration. No
full 100 x 100 acquisition or new production high-accuracy run was performed;
the user's running application and saved caches were not interrupted or cleared.

## Changes

- Physical recording planes and per-pixel detector offsets no longer force all
  STEM wave propagation back to NumPy. Multislice, FFT, phonon averaging and
  detector reductions stay on CUDA; exact physical acceptance masks come from CPU.
- The mask-only router caches the angular projection once per plane. It applies
  the signed position map before broadcasting and omits unused diagnostic arrays.
  The full diagnostic router remains the independent reference.
- GPU batches adapt to available VRAM and estimated host/device scratch space,
  up to 128 probes. CPU reference batches remain at 8. No scan positions, slices,
  wave-grid samples or phonon configurations are removed or interpolated.
- Bandwidth truncation is now returned by CUDA, in the same final bulk transfer.
  GPU failure discards partial work and restarts the complete frame on CPU.

Existing result/checkpoint caches are not cleared. Restart the application to
load the changes; select **Auto (GPU / CPU)** or **CUDA GPU** for new calculations.
The progress label reports `STEM GPU probes` when the resident path is active.

## Measured small-stage benchmark

RTX 5090, CuPy 14.1.1; prescribed 30 mrad beam, Si [110], 10 nm thickness,
2 angstrom target slices, 8 x 8 scan positions at 0.02 nm pitch, requested
256 grid (resolved 243 x 246), one configuration, synthetic physical stops.
Times include specimen preparation and the wave/detector stage, but not column
ray tracing or application startup. The second run is reported as warm.

| Path | Warm time for 64 positions | Positions/s |
| --- | ---: | ---: |
| Original CPU / full diagnostic router | 15.122 s | 4.23 |
| CUDA / full diagnostic router, batch 32 | 4.560 s | 14.04 |
| CPU / prepared mask router | 11.368 s | 5.63 |
| CUDA / prepared mask router, batch 64 | 0.833 s | 76.85 |

Final CUDA first-run time was 1.749 s. Warm speedup over the original CPU path
was about 18.2x. Largest detector-fraction absolute differences versus the CPU
reference were DF 9.01e-7 and BF 6.00e-7. HAADF was zero at this benchmark's
angular sampling cutoff; this is not a quantitative HAADF reference image.
CUDA retains its existing complex64/float32 precision; CPU is the complex128
reference. Physical geometry and discretisation are unchanged.

These are small benchmark timings, not a promised 100 x 100 acquisition time.
CIF size, thickness, grid, phonon count, detector routing and cache warmth all
affect throughput. No complete 100 x 100 calculation was run in this change.

Reproduce in PowerShell using the project environment:

```powershell
$env:CUPY_CACHE_DIR = 'F:\tem_simulator_v2\tmp\stem_cuda_kernel_cache'
.\.venv\Scripts\python.exe scripts/benchmark_stem_wave.py --repeat 2
```

The explicit compiler-cache path is needed in the restricted test environment:
initial tests stalled while CuPy attempted to create cache files in its default
user directory. A writable project cache resolved this without a driver,
dependency or production application cache change. Profiling also verified that
the first GPU version spent most of its time in broadcast CPU recording maps.

## Validation

85 focused tests passed in 38.07 s on this machine, including actual CUDA/CPU
signal comparisons, exact masks against the diagnostic router, static/raster
deflection, nonuniform slices, frozen-phonon uncertainty, bandwidth truncation,
partial GPU failure with complete CPU retry, sampling/contrast and 4D capture
regressions. The tiny GPU fixture emitted an underutilisation warning. This was
not a full project-suite run.

## Remaining limit

Complete 4D-STEM capture (including an interactive bank retaining raw diffraction
frames) still uses the CPU route. File-sink resume semantics skip completed frames;
streaming partial GPU data then retrying on CPU would mix outputs. This patch does
not bypass that safeguard. Ordinary BF/DF/HAADF image generation without a raw
diffraction sink uses the accelerated route.
