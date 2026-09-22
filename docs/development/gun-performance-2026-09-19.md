# Lossless particle gun performance update — 2026-09-19

Subsequent round: [execution-local workspace optimization](gun-loop-workspace-2026-09-19.md)
adds current-field preparation and buffer reuse. Its new same-thread paired
measurement and regression evidence are separate from the historical numbers
below.

The complete 5,000-particle gun calculation finished with identical stored
results before and after this optimization. At the same four-thread budget,
wall time decreased from **161.05 s to 130.14 s (19.19%)**, and sampled peak
process RSS decreased from **2.455 GB to 1.084 GB (55.83%)**. Sixteen threads
reduced the new implementation to **85.75 s** on this 32-logical-CPU host.
The algorithm improvement and the additional thread improvement are separate.

## Changes and physical scope

- The existing adaptive Boris comparison still evaluates one full step against
  two half steps with the original energy projection and all error thresholds.
  After rejection, the next full step has exactly the duration, starting state,
  static fields and projection of the preceding trial's first half step. The
  serial and parallel kernels retain that already-executed result, reducing
  three advances per attempt to three initially and two per retry. The cache
  lasts only for that call; changing fields or beginning another accepted step
  cannot reuse it. Every particle's error status is still checked.
- History finalization consumes and releases saved momentum rows as it builds
  the public float64 slope arrays. It then consumes position rows into the
  public X/Y/Z arrays. It no longer simultaneously holds the original lists,
  second dense position/momentum histories and additional slope/Z matrices.
  Every existing equal-time sample, coordinate, slope and mask is retained.
- The original chronological axial-record loop is optionally compiled with
  Numba, including the original 1e-12 mm comparison. Without Numba its Python
  implementation remains available.
- Flight-time resampling retains the already ordered NumPy history instead of
  converting each ray's full time/position history into Python lists. Only the
  few exact plane/terminal events are sorted and merged. Stable event priority,
  first arrival through turning/backward paths, missing clocks and readable
  untimed display tails are unchanged.
- History conversion and both resampling stages check cancellation between
  bounded groups of rows/rays, including after physical integration finishes.

No extraction, acceleration, focusing, magnetic alignment, aperture, physical
stop or clock origin was removed. No tolerance or integration step was relaxed.
The public gun-history structure is unchanged. Unsupported/custom field
providers still use the original general integrator. Coherent calculations
remain paused. Grouped asynchronous integration and an analytic drift shortcut
were considered but were not introduced: they would need separate time-grid,
crossing and refinement validation.

## Controlled measurements

These are standalone **complete gun** measurements, not complete microscope,
sample, scan, GUI or High-accuracy pipeline timings. The benchmark directly
calls the physical tip-to-exit tracer without using its executed-result cache.
Numba kernels are warmed before the measured solve; this is not a clean-JIT or
cold-operating-system measurement. Runs were serial, on the same Windows host
and project Python environment, with fixed NumPy and Numba thread limits.

Inputs: flat classical tip; 300 kV final acceleration; 4 kV extractor;
0.3 eV mean launch kinetic energy; 0.3 eV energy FWHM; 1 mrad angular RMS;
5 nm tip-plane emission-footprint FWHM; 0.2 mm requested gun trace step, 2 mm drift cap and
2 mm stored-history setting. The monochromator is uninstalled and stochastic
vacuum transport is disabled. The explicit tracer `count` is 49 or 5,000;
the recorded constructor emitter default of 1,000 is overridden by this call.
Each receipt also retains the full gun input serialization.

| Population and implementation | Threads | Completed solve, s | Analytic step calls, s | Path resampling, s | Time resampling, s | Peak RSS, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000, before | 4 | 161.048 | 101.046 | 6.389 | 3.279 | 2.455 |
| 5,000, after | 4 | 130.137 | 78.258 | 0.712 | 1.685 | 1.084 |
| 5,000, after | 16 | 85.750 | 35.991 | 0.750 | 1.947 | 1.096 |

All three solves accepted **35,336 steps**, retained a **3,536 × 5,000**
equal-time history and delivered all 5,000 particles to the exit. Their digest
over the main trajectory/clock arrays, equal-time history, exit phase space,
identity, weights, survival mask and stop labels is identical:

`a3b1841c60f657aba4b0daf57e57912e4dbffedcd44388650972440026b0c24c`

New history finalization took 0.279 s at four threads and 0.302 s at sixteen.
Nested timings overlap the total and must not be added to it. RSS is sampled
every 20 ms and includes the Python/JIT runtime; brief peaks can be missed.
The large cases are one completed run each, not a statistical scaling study.

The smaller 49-ray case used three complete solves per implementation at four
threads: median **2.672 s before**, **2.387 s after**, with noticeable host-load
variation (before 2.376–2.755 s, after 1.750–2.652 s). All six accepted 8,863
steps and have the same complete-output digest:

`ab289371f1660e894b036fcfd1372911c704e8d250b86d40d37cebc4ba21285e`

Sixteen threads are supported by this measured 5,000-ray case and remain within
half of this host's 32 logical CPUs. This does not establish the optimal count
for every bundle or host. The existing small-bundle serial threshold remains.
The CPU budget is implemented separately from these numerical changes.

## Validation and receipts

The final focused run completed **76 passed** in 168.82 s, with 16 existing
Pydantic deprecation warnings. It covers history/retry acceleration, analytic
CPU/compiled transport, gun TOF and equal-time history, adaptive stepping,
physical aperture planes (including the existing nine-ray curved-tip fixture),
voltage-reference semantics, numerical cache invalidation and cancellation.
This bounded curved-tip regression does not resume coherent-wave development.

The existing analytic-step/reference tests cover nonzero and disabled fields,
rotated stigmation, blanking, active masks, adaptive rejection, parallel/serial
parity, invalid-particle error propagation, custom-provider fallback and the
current/midpoint impulse bound. Eight new tests cover lossless history ownership,
record thresholds and nonfinite samples, cancellation, stable coincident event
ownership and the exact number of executed advances after rejection.

A separate comparison to the saved pre-change implementation tested 100 seeded
histories with 13 rays and 31 time samples each, including backward/turning
paths, unknown terminal clocks and unordered exact events. All coordinates
and clocks matched bit for bit. This is algorithm regression evidence, not
independent qualification of an entire microscope or experimental validation.

An independent read-only review found no substantive equivalence, exception,
cancellation or fallback issue in this diff. Its additional 250 small artificial
history comparisons matched the previous implementation bit for bit, including
NaNs and coincident exact events. A separate process with Numba hidden also
imported successfully and exercised the Python record-index fallback. These
additional checks were reported through the delegated console; no separate
receipt file is claimed for them.

Local ignored receipts and reproducible worker source are in
`tmp/gun-history-20260919/`:

- `before-49.json`, `after-49.json`
- `before-5000-4.json`, `after-5000-4.json`, `after-5000-16.json`
- `randomized-history-parity.json`, `compare_history.py`
- `benchmark_gun.py`, `tracing_before.py`, `analytic_before.py`
- `final-gun-tests.xml` records the focused regression run.

The baseline snapshots are the actual dirty working tree immediately before
this update, rather than an older published revision. Snapshot SHA-256 values:

- Tracer: `e7f246e06654288419d431667c71d15dcdbb1a638a744c7197fafe3108443292`
- Analytic kernel: `f3992bd60b1db05cbe68b3b98d067456752ac7fdc3fb5dd7b25183c94c734ad9`

Measured new source SHA-256 values:

- Tracer: `9acf6b34b8757f6f199671bed2b3fdf52ae6ab183c3ea60216c1bcbe9afac639`
- Analytic kernel: `f346613351e1b278d565d2c1811e489b02de9046d0358e5d7ea05dd538a652d6`

Example bounded reproduction (use a new output path):

```text
.venv\Scripts\python.exe tmp/gun-history-20260919/benchmark_gun.py --variant after --rays 5000 --runs 1 --threads 16 --budget 240 --output tmp/gun-history-20260919/repeat-after-5000-16.json
```

The worker checks the 240 s solve budget cooperatively and terminates only its
own dedicated process after 245 s if finalization cannot return. No running
user application was controlled or interrupted. Calculation arrays and caches
remain local and are not part of the report.
