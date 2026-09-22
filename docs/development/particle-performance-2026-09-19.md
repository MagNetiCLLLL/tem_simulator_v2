# Classical particle performance and section reuse, 2026-09-19

## Executed scope

The original 49-particle baseline below precedes the new always-on inserted
specimen transport in non-scanning particle mode and the change to a no-filter
default assembly. It is a versioned baseline, not a timing claim for the final
expanded live workflow. See [the section implementation report](particle-section-live-tuning-2026-09-19.md).

One bounded process executed the application-facing `calculate` pipeline seven
times with the current classical flat-tip source. All seven calls completed;
their combined measured calculation time was 72.483 seconds. Coherent source,
TEM wave, STEM wave and raster scan execution remained disabled. No running
desktop application or saved instrument setting was changed.

The case uses the default assembled 300 kV instrument, 49 emitted particles,
1 mm requested column step, 5 mm displayed column history spacing, and CPU
execution with four library/Numba threads available. The gun retains its
default 0.2 mm requested trace step, 2 mm history spacing, source distribution,
adaptive error tolerances, extraction, acceleration, focusing and apertures.
Its small compiled particle kernel is serial at this population size.

The inserted specimen is the default 5 nm Si [110] reference with EDS enabled
and 256 overlap sampling points. The installed energy filter is enabled.
Residual-medium scattering is disabled, as in the default state. This is a
complete call for those requested stationary particle products, including EDS
and filter transport; it is not a raster scan, full instrument qualification,
or a 15,000-particle High runtime measurement. All 49 particles reached the
specimen in this particular case.

Host: Windows 11, Python 3.12.3, 32 logical CPUs, approximately 95.6 GiB RAM.
The calculation-only timers exclude interpreter/import startup, snapshot
restore and GUI rendering. "Cold" means the first call in a fresh process;
filesystem and existing compiled-code disk caches were not purged. Thus the
receipt does not measure a fresh compilation installation.

## Measured time

| Call | Total seconds | Preparation | Gun/column stage | EDS stage | Filter stage | Final stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First call | 21.512 | 0.550 | 4.169 | 3.193 | 13.493 | 0.063 |
| Same inputs, no completed-result seed | 18.329 | 0.463 | 0.992 | 3.147 | 13.616 | 0.063 |
| Reuse 1 | 1.104 | 0.497 | reused | reused | reused | 0.561 |
| Reuse 2 | 1.078 | 0.580 | reused | reused | reused | 0.454 |
| Reuse 3 | 0.912 | 0.439 | reused | reused | reused | 0.428 |
| Projector lens 2: +0.1 percentage point | 12.912 | 0.399 | 0.817 | reused | 11.603 | 0.051 |
| Condenser lens 2: +0.1 percentage point | 16.635 | 0.378 | 0.861 | 2.764 | 12.543 | 0.053 |

The reuse median is **1.078 seconds**. A warm library alone saves the initial
gun calculation, but leaves specimen/filter work to execute again. Full result
reuse avoids those calculations. The first and same-input warm/reused column
arrays, including their float64 flight clocks, had identical byte digests.

These stage labels partition pipeline progress, not exact subsystem ownership.
For example, a shared specimen-envelope call occurs after the EDS slot and is
charged to the following stage. Independent wrappers measured these nested
functions during the first call:

| Function scope | Seconds | Calls |
| --- | ---: | ---: |
| Gun transport, including cache lookup | 3.077 | 1 |
| Analytic gun accepted steps, including internal retries | 0.903 | 8,863 |
| Incident-column integration | 0.137 | 1 |
| Post-reference column integration | 0.252 | 1 |
| Specimen-interaction calls, combined | 3.669 | 2 |
| Energy-filter simulation | 13.015 | 1 |
| Calculation signature construction | 0.406 | 1 |

These values overlap their enclosing stage timers and must not be added to
the full wall time. The first call's sampled peak RSS was 212.9 MiB; sampling
every 20 ms can miss brief peaks. The 230 exact incident checkpoints retained
450,800 bytes for this bundle.

## What reuse already saves

Changing the projector reused the entire incident history and specimen/EDS
products. Gun lookup took 0.015 seconds and the post-reference column kernel
took 0.261 seconds; filter transport still took 11.154 seconds. Repeated full
downstream delivery is therefore expensive even when upstream reuse works.

Changing C2 resumed the incident calculation from the existing checkpoint at
z = 555 mm. It reused 106 integration nodes and recomputed 1,049 nodes. Gun
lookup took 0.013 seconds. A early condenser change naturally leaves much of
the column downstream of the changed control; cache reuse cannot remove that
physical dependence.

For full product reuse, the stored `column_cache` record is inherited from the
original simulation. It is not evidence that the current call integrated
anything. Use this call's `reused_products` and timer call counts to distinguish
full result reuse from a newly executed checkpoint suffix.

## Isolated filter hotspot profile

A separate bounded follow-up reconstructed the same 49-particle numerical
configuration from the physical tip and its shared specimen loss distribution.
Column/distribution preparation took 5.513 seconds. Only the filter call was
profiled: 18.858 seconds with cProfile overhead, not an unprofiled timing to
compare against the table above. Twelve physical paths entered the filter and
eight reached the spectroscopy plane. The two trace batches were:

| Batch | Particles | Integrated steps | Profiled seconds |
| --- | ---: | ---: | ---: |
| Independent diagnostic reference | 1 | 3,186 | 9.570 |
| Executed entrance population | 12 | 3,209 | 9.279 |

The 6,395 Boris updates spent 16.415 cumulative seconds in filter magnetic-field
evaluation. The ten multipoles produced 63,950 local field evaluations taking
15.111 cumulative seconds, including 390,095 finite-envelope derivative calls
taking 8.007 seconds. Approximately 818,588 `zeros_like` calls illustrate the
small-array Python/NumPy overhead. These are nested profile values; they must
not be added together. Alignment, Jacobian fitting and transfer construction
were not dominant in this executed filter call.

`EnergyFilterMagneticField.field_at_global_positions_t` evaluates every pole
at each step. `FiniteMultipoleField.field_at_local_positions_t` constructs all
required envelope derivatives before evaluating coefficients, even where the
envelope has exactly zero compact support. Checking validated local coordinates
against that existing support could avoid identically zero arithmetic while
retaining every physical bore, fringe and interception calculation. This needs
boundary/nonfinite-input/provider-fallback parity tests before implementation.
No finite-field evaluation was changed in this work.

The profile receipt is `tmp/particle-performance-20260919/filter-profile-49.json`,
SHA-256 `b598486313c586969c54bdb544f95e16051daaac4882a8ff0201632fe27a822c`.
Its implementation identity is
`b7c33980fa23d7b234ca692402f94d24c588b9bf0136aecf5c72ee88c04afa51`.
It was collected later during parallel development and has its own identity;
it is not a controlled cross-version speed comparison with the first table.

## Implemented reference reuse

`energy_filter_raytrace.py` now keeps a process-local LRU of **executed diagnostic
references**, with at most eight entries and 8 MiB of retained array storage.
The physical incident population continues through the ordinary transport on
every filter calculation. The key contains the actual filter geometry, frames,
field coefficients, fringe model, slit/detector states, beam energy, numerical
step and Boris options, plus source implementation identity. The geometry is
read from executed objects, not the profile serializer that intentionally omits
TOML-owned dimensions. Unknown providers or overridden methods use the normal
uncached route. A changed input during reference execution prevents insertion.

Retained reference arrays are private and read-only, and cache hits return
detached copies. Consumer mutations cannot alter another request. Failed traces
never enter the cache. The change preserves the reference's full trajectory,
interception masks and float64 local arrival clocks. It creates no beam source
and never substitutes the reference clock for a physical particle's clock.

The new cache suite plus existing filter TOF suite passed **27 tests in 19.75 s**;
receipt: `tmp/particle-performance-20260919/filter-cache-tests.xml`. Coverage
includes actual curved-filter cached/uncached equality, clocks, detached arrays,
per-input invalidation, fallback, bounded storage, failure exclusion and a
whole filter call proving that only its reference is reused. Counting-only
invalidation fixtures are distinct from the actual numerical comparisons.

A same-process production-filter measurement with actual 49-particle
tip-origin input subsequently took **13.690 s with a cold reference cache and
6.135 s with a warm reference cache**, a factor of 2.23 for this filter call.
The first call traced a one-particle reference in 6.988 s and the 12-particle
entrance population in 6.510 s. The warm call traced only that physical
population, in 6.056 s. Paths, float64 arrival clocks and particle ancestry
were exactly equal; eight paths reached the spectroscopy plane.

This is a filter-only comparison, not a new whole-pipeline time. Its local
receipt is `tmp/particle-performance-20260919/filter-cache-benchmark-49.json`.
Parallel development changed the repository-wide implementation hash between
the initial snapshot and final receipt, which is recorded rather than hidden.
Both measured calls used the same loaded filter functions and physical inputs;
the recorded batch calls prove that the reference was reused. The isolated
numerical regression above supplies the independent cache-parity check.

## User-sized 5,000-particle bounded run, without a filter

After the requested population was clarified, a fresh worker attempted the
classical pipeline with **5,000 particles and the No Energy Filter assembly**.
The actual default column step was read from the resulting state: **0.5 mm**,
with 2 mm history spacing. Neither value was inferred from a GUI screenshot.
The gun requested step remained 0.2 mm; the beam was 300 kV with classical flat
tip emission, the 5 nm reference specimen inserted, EDS enabled, and scanning,
waves and vacuum scattering disabled. CPU/library concurrency was capped at
four threads. The input choice preserves all modelled gun operations and does
not move the source downstream.

The **180-second process budget expired before the pipeline completed**. Only
the benchmark's own worker tree was terminated. The last durable progress
receipt, at 170.316 seconds of the calculation, recorded:

- Stage 2/5, "Tracing the electron column"; `trace_source_to_exit` had not returned.
- 35,336 completed analytic step calls, including their internal retries.
- 112.686 cumulative seconds inside those step calls.
- 2,373,984,256 bytes of current RSS, approximately 2.21 GiB.
- No completed result and no measured specimen/EDS stage time.

This reproduces a long Stage 2 without an installed filter. It identifies the
gun calculation as the current obstacle at this bundle size, but the final
unreturned tail may include gun-history assembly; these timers do not establish
that the integration loop itself was still running at termination. The remaining
time also includes impulse checks, masks, bookkeeping and storage. It is not a
complete runtime estimate, a proof of deadlock, or a convergence result.

The partial receipt is
`tmp/particle-performance-20260919/stages-5000-no-filter.json`, with status
`SUPERVISOR_TIMEOUT` and SHA-256
`48df2f0b2e2c4b6f940205995ca63b076963a75aab5a637c02a2ba94b094d2fd`.
Its input identity is
`3e66b8ed46cd708ff9a84f97bda522254a097dd4d43843f94f77f9b0fcda2392`,
and initial implementation identity is
`76ea6349dbf6f523962889d7dba1f8ec810f9045ed480c5f7f1b5525eadb9e2c`.
Reproduction, deliberately omitting step overrides:

```text
.venv\Scripts\python.exe scripts/benchmark_particle_stages.py --rays 5000 --energy-filter off --cold-only --output tmp/particle-performance-20260919/stages-5000-no-filter.json --budget-seconds 180
```

This updated configuration makes persistent executed gun/section reuse the
immediate interactive priority. The filter optimization still benefits requests
that explicitly install and reach the filter; it cannot accelerate an absent
filter or the first uncached 5,000-particle gun solve.

## Next gun optimization batch: read-only review

The 5,000-particle receipt attributes 112.69 seconds to accepted analytic-step
calls, including their internal rejected attempts. The remaining elapsed time
also includes step bounds, physical interception checks, history recording and
possibly final history conversion. It is not a measured postprocessing time.
The following priorities are implementation proposals, not completed speedups.

1. **Retain the same history with fewer simultaneous copies.**
   `electron_gun/tracing.py:102–107,352–365` stores position, momentum and two
   masks for every tenth accepted step at these settings. At about 3,535
   retained samples and 5,000 particles, the position/momentum snapshots alone
   occupy about 848 MB. Lines 381–402 then allocate dense copies while the
   original lists still live, plus two slope arrays; lines 419 and 729 also
   construct axial-coordinate arrays. This is consistent with the observed
   late RSS increase, but does not establish where the worker was executing.
   Use bounded blocks or directly owned final-format float64 history storage,
   release converted blocks promptly, and preserve every existing timestamp,
   sampled state, mask and exact event. The public equal-time result already
   stores positions and slopes, so retaining another complete momentum history
   through finalization is unnecessary. Do not reduce history sampling or
   remove equal-time observations as a performance shortcut.
2. **Compile or combine repeated history processing.**
   `tracing.py:761` walks retained samples in Python for each ray; the following
   timing resampler independently gathers, orders and interpolates the same
   ray history (`:888–985`). Keep the exact current comparison and chronological
   first-crossing rules, including backward/turning paths, unknown terminal
   times, exact aperture events and readable untimed tails. A compiled per-ray
   loop or a single shared chronological traversal can reduce overhead without
   touching the accepted trajectories. Add cancellation checks between blocks
   so finalization cannot make the calculation appear unresponsive.
3. **Reduce step workspace allocation before changing the step schedule.**
   `tracing.py:156–157,215–225`, the compiled step's output copies, and
   `RelativisticPhaseSpace.__post_init__` repeatedly copy complete bundles.
   Reusable private workspaces and ownership transfer at validated boundaries
   can preserve the public copy contract and error behavior. Field parameters
   are also packed independently for the impulse cap and adaptive step; share
   one immutable per-step snapshot while retaining all custom-provider guards.
   The outer static-energy projection repeats the accepted fine-step
   projection: fuse that operation if useful, but do not simply delete it
   while claiming identical floating-point results.
4. **Evaluate grouped adaptive stepping as a separate numerical change.**
   The spatial cap uses all active axial positions and the fastest axial speed
   (`tracing.py:152–155`); every ray then shares the most restrictive impulse
   and error-limited time step. Grouping already-emitted independent particles
   by comparable integration requirements could avoid repeatedly advancing
   easy rays at another ray's tiny step. Emit the population once, preserve
   IDs/weights and every field/aperture, and keep the same local error budgets.
   This changes the accepted time grids and requires a new numerical identity,
   convergence and conservation checks, aperture/TOF comparison, and honest
   equal-time reconstruction. Stochastic media additionally require stable
   per-particle random/collision state; unsupported or collective providers
   should retain the general path. No speedup is claimed before measurement.

Priorities 1–3 should first demonstrate unchanged trajectories, clocks, exact
event coordinates, masks and equal-time samples on existing timing fixtures
and a bounded representative bundle, followed by a measured peak-memory and
wall-time comparison. Priority 4 needs refinement-based physical validation
rather than byte equality. The gun files and history format were not changed
in this performance task; executed gun/section reuse is the implemented
interactive mitigation while this separate optimization batch is pending.

## Remaining priorities that preserve the model

1. **Make tuning end at the requested physical section.** A request for a plane
   upstream of the specimen or filter should execute only the upstream path
   needed to reach it. Keep a completed upstream state and recompute only the
   suffix affected by the selected control. The measurement explains why
   running the filter on every upstream slider movement defeats otherwise
   successful incident reuse. This is a deliberately shorter requested path,
   not removal of interactions from a completed path to a downstream detector.
2. **Persist the executed section, not a replacement source.** Retain particle
   identities, weights, energy, phase space, absolute tip-origin clocks,
   interception status and coordinate convention, bound to all consumed tip,
   field, geometry and numerical inputs. A physical continuation resumes from
   those exact states. For stochastic media, phase space alone is insufficient:
   collision clocks, remaining path budgets and random-stream state must also
   be preserved before suffix reuse is admitted.
3. **Optimize the filter before accelerating this small column case.** Its two
   ordinary trace batches (a diagnostic reference and the incident particles)
   currently loop in Python over Boris steps, analytic fields, bores, masks and
   crossing tests. Candidate work is an equivalent compiled batch kernel and
   exact compact-support avoidance. Dependency-bound reference reuse is now
   implemented. The reference consumes
   filter fields/geometry, beam energy and integration settings; it does not
   consume a changed upstream lens's entrance distribution. An unchanged
   reference may be cached, but the changed physical incident beam must still
   be transported. Merely combining reference and beam into one batch changes
   the maximum-speed-based step and therefore needs its own numerical check.
4. **Treat large gun bundles separately.** The earlier bounded 15,000-particle
   run completed 33,335 accepted steps in 360.17 seconds and was cancelled before
   the gun finished. That remains an incomplete, different-version measurement,
   not a full-runtime estimate. Shared-step control lets the most demanding ray
   limit the bundle; persistent executed gun reuse is the immediate saving for
   downstream tuning. Per-particle/group step scheduling and more compact
   history retention are possible later optimizations, requiring conservation,
   aperture-crossing and refinement checks with unchanged error criteria.
5. **Separate physical-state work from readouts and summaries.** Current full
   reuse still spends roughly 0.3–0.5 seconds building identities and 0.4–0.6
   seconds in the shared specimen envelope. A correctly scoped immutable cache
   can avoid rebuilding unchanged summaries. Selecting fewer displayed
   observables must not remove specimen interactions or detector interception
   from the propagated state.

No integration tolerance, source law, material interaction or filter field was
relaxed for these measurements. A coarse live view must record its actual
numerical settings; it is not a substitute for a later validated full run.

## Reproduction and receipts

### Final expanded stationary-particle path

After the focused integration suite passed, a fresh isolated process completed
the final 49-particle no-filter path in **6.625 seconds**. It used the actual
default 0.5 mm column step, inserted 5 nm Si reference, EDS on, scan/waves/vacuum
off, and included the newly required non-scanning elastic/material-exit stages.
No other validation job ran concurrently with this measurement. Interpreter
startup is excluded, and existing compiled disk caches were not purged.

| Progress scope | Seconds |
| --- | ---: |
| State/layout preparation | 0.315 |
| Gun and initial column | 3.019 |
| Explicit specimen electron transport | 0.416 |
| EDS point stage | 1.758 |
| Specimen-exit downstream transport | 0.996 |
| Disabled filter return | <0.001 |
| Final diagnostics | 0.092 |

Progress slots are not exclusive function ownership: nested specimen-envelope
work can occur in the following slot. This verifies a complete small classical
workflow, not a scaling estimate for 5,000 particles or a timing comparison with
the earlier different-filter/different-step baseline. Receipt:
`tmp/particle-performance-20260919/final-validated-49-no-filter.json`.
The earlier `final-particle-49-no-filter.json` was a concurrent integration smoke
run and is not the isolated timing evidence.

### Earlier filter-installed baseline

```text
.venv\Scripts\python.exe scripts/benchmark_particle_stages.py --rays 49 --step-mm 1 --history-step-mm 5 --energy-filter on --output tmp/particle-performance-20260919/stages-49.json --budget-seconds 180
```

The script uses production functions with lightweight timing wrappers, runs
calls serially and writes only scalar receipts. The parent supervises only its
own worker process tree. The executed receipt is local and ignored by Git:

- `tmp/particle-performance-20260919/stages-49.json`
- SHA-256: `53d09213098771f20c405e84c1a321a9a4c08e518c5f10915d1c3cfb25948457`
- Input identity: `f74b3fbef98c307998ab01d6d85f0056a54dbfaa148837c8a42e24916e138804`
- Implementation identity: `fd0a3dada563f3363f2881de0a4d8c55e6d83a316f68a5b18230ec7f7984a37c`
- The source implementation identity was unchanged throughout the seven calls.

After that execution, the script gained explicit Windows worker-tree timeout
cleanup, periodic partial receipts, explicit assembly/default-step options,
a cache-record provenance note and extra filter-count receipt fields.
Those reporting/supervision additions do not change the executed numerical
path. The existing receipt is preserved and was not retroactively enriched.

The earlier large-bundle diagnosis is recorded separately in
`high-accuracy-performance-2026-09-19.md`; its timings must not be combined with
this benchmark as if they were one controlled scaling series.
