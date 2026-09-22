# Classical time-of-flight tracking — 2026-09-19

## Delivered behavior

Transverse Beam now has three usable presets: **Source position tracking**,
**Emission angle tracking**, and **Time of flight**. The first two retain their
fixed emission colours. TOF colours each arriving path by its delay from the
earliest timed arrival in the saved population at the selected plane. The
reference is computed before the 2,000-point display limit. Both plots use the
same colour for each displayed path; the upper plot places it at its original
emission position. Repeated descendants of one source particle retain distinct
times, including in coincident-point hover text.

The TOF upper plot shows arriving display paths only. It is not the complete
emission population shown by the first two presets. Non-arrivals are omitted;
missing clocks are grey, never zero delay. Hover shows absolute laboratory time
in ns and relative delay in fs. The visible delay scale chooses fs, ps or ns.
Colour limits and the earliest-arrival reference can change with plane; compare
numerical values when comparing different planes. Position/angular coordinates
are independently selectable. View changes, projection, zoom and fitting read
completed results and do not execute transport.

This is classical time since a **common simultaneous emission event**. It does
not define a pulsed source, a coherent quantum state, an interference phase or
a direct hardware phase measurement. Coherent development remains paused.

## Physical and numerical contract

Existing emission, extraction, acceleration, focusing, apertures, specimen
interactions and detector absorption remain in the path. No downstream source
is introduced and no physical setting is changed to fit a measured instrument.

- **Gun:** publish the executed exit-arrival clock and interpolate first
  crossings from existing float64 equal-time histories. Resolved DPA/C1/exit
  events override interpolated values. Turning paths are searched in temporal
  order; pre-emission positions and frozen post-stop tails have no arrival.
  Gun display positions/slopes use that same first-crossing segment; a later
  return cannot be interpolated across an earlier axial-maximum envelope.
  Recorded boundary positions override the display interpolation at that event.
  Intermediate gun-Z interpolation has the accuracy of the retained time
  history; this is not a new exact continuous-time solution.
- **Column:** accumulate `dt = sqrt(1 + tx² + ty²) dz / v(K)` with the original
  four RK stages. `dz` is metres, slopes are dimensionless, and `v(K)` uses the
  actual relativistic energy including each particle's energy offset. No
  float32 drawing-path chord is used to estimate time. The existing mapped
  field integrator, NumPy, compiled CPU and CUDA paths retain the same clock.
  Out-of-domain trajectories have no valid time. Time storage is float64.
- **Specimen:** retain executed elastic/material/support/field flight times,
  matched to the inherited incident reference-plane clock. Signed reference
  matching offsets are part of the existing geometric matching approximation,
  not an extra physical path or a new source. Only physical terminal and later
  rows have arrivals. Distinct descendants never share a clock merely because
  they share an ancestor ID.
- **Finite energy loss:** the existing aggregate loss channels do not locate
  each loss event in the specimen. Their absolute arrival times remain unknown.
  A total energy loss is insufficient to infer how long the electron travelled
  at each velocity. Zero-loss paths with complete timing can be displayed.
- **Filter:** use actual Boris-step crossing times, coordinates and slopes at
  the energy slit, output plane and spectrometer detector. Times inherit the
  upstream clock; the diagnostic reference ray cannot provide one. Coordinates
  use the sector-exit dispersive/non-dispersive frame. Arbitrary global Z beyond
  the entrance is explicitly unavailable, because it does not identify an
  outgoing plane in a bent filter. A pre-filter plane uses its upstream path.
  The existing representative-ray limit remains visible in provenance; filter
  timing is not evidence that every original particle was integrated there.
  The pipeline executes this stage after specimen/downstream transport. A
  validated detailed specimen exit is preferred; an optical-reference input
  remains explicitly labelled. Filter cache admission checks this provenance
  and the detailed-exit dependency so the two cannot substitute for each other.

Relativistic energy/momentum relations follow the primary exposition in
[Feynman Lectures I, Chapter 16](https://www.feynmanlectures.caltech.edu/I_16.html).
The per-stage path quadrature is verified separately from the UI. Neither
float64 storage nor a displayed femtosecond unit establishes phase accuracy.

## Cache, memory and speed

Flight clocks travel with incident and downstream branches, full-precision
checkpoints and persistent incident seeds. Gun plane arrivals and equal-time
histories now survive seed round trips. Clock identities invalidate incompatible
active reuse; historical untimed results remain readable, with unknown times.
Incompatible clock dimensions must never broadcast one particle's time onto
other particles. Missing or invalid clocks cannot supply a full/suffix restart.

Identifiers include `tip-origin-lab-clock-v1`, `tip-clock-first-crossing-v1`,
`canonical-rk4-quadrupole-tensor-tof-v4`,
`incident-simulation-seed-v3-flight-time`, and
`incident-propagation-checkpoints-v2-flight-time`.

Each saved ray/plane requires another 8 bytes for time; each restart checkpoint
has five float64 arrays instead of four. Checkpoint selection, GPU retention
and the application RAM estimate include this storage. The clock does not
allocate a velocity matrix over all axial nodes and particles. Deselecting TOF
does not remove timing or other physical state from a new calculation.

Warm bounded column benchmark, 8,192 rays × 1,000 steps, 101 retained rows and
two checkpoints, three repetitions per mode:

| Backend | Without clocks | With clocks | Added time |
| --- | ---: | ---: | ---: |
| Compiled CPU, 4 threads | 37.299 ms | 43.417 ms | 6.118 ms |
| RTX 5090 CUDA | 8.015 ms | 20.527 ms | 12.511 ms |

Trajectories and geometric checkpoints were bitwise identical with and without
clock accumulation in these checks. CUDA's relative overhead is substantial
for this bounded stage, although its absolute time remains below the CPU case.
These measurements do not establish complete High accuracy runtime. The
earlier gun-performance repair and its unresolved large-bundle timing limits
remain documented separately. Receipt: `tmp/tof-20260919/column-timing-benchmark.json`.

## Verification scope

Analytical tests cover relativistic drift, tilted paths, energy-dependent speed,
accelerating gun histories, turning first crossings, apertures, field-curved
paths, RK refinement, CUDA/CPU agreement, checkpoint suffix timing, descendant
identity, unknown inelastic event depth, physical filter crossings and detector
interception. UI/cache checks cover source colours, missing time, no-arrival
clearing, reference selection before display sampling, filter-frame routing,
historical wave-reader restoration and persistent clock precision.

An offscreen three-panel preview used **actual 49-particle tip-to-specimen
transport**, with specimen removed, vacuum off and a coarse 5 mm column step.
Eighteen rays reached the specimen plane. This is a bounded integration/render
check, not a converged full-instrument calculation or a performance benchmark.
Preview: `tmp/tof-20260919/transverse-tof-real-49.png`;
parameters/readout: `tmp/tof-20260919/render-real-49.json`.

- Final display regression: **152 passed**, exit 0; `tmp/tof-20260919/display-final.xml`.
- Final gun/column/specimen/medium/field and memory regression: **103 passed**,
  exit 0; `tmp/tof-20260919/physical-transport-final.xml`. The small CUDA case
  executed on the installed GPU rather than silently using a CPU fallback.
- Final persistent cache, malformed-clock and segmented-restart regression:
  **54 passed**, exit 0, in 114.06 seconds, covering the three files
  `test_time_of_flight_cache.py`, `test_calculation_manifest_artifacts.py` and
  `test_segmented_column_cache.py`.
- Final filter/source-routing/layout/progress and selected controller/cache
  fixtures: **54 passed**, with no skips, in 26.083 seconds;
  `tmp/filter-tof-pipeline-20260919.xml`. Includes rejection of optical-reference
  cache substitution, reuse of a validated detailed exit, no-illumination
  handling, and monotonic stage progress after placing the filter last.
- After the final first-crossing geometry/clock correction, **39 gun/cache/
  segmented-restart/compiled-reference tests passed**, exit 0, in 70.37 seconds;
  `tmp/tof-20260919/final-gun-cache.xml`. This includes four additional display
  tests for a returning trajectory, resolved-plane coordinates and adjacent
  interpolation, unchanged monotone interpolation, and an untimed stopped tail.
  The actual gun exit and integration remain unchanged.
- Final serial `compileall -q src scripts tests`, tracked/additional-file
  whitespace checks and documentation-link checks passed. The refreshed
  three-panel and compact TOF renders were visually checked offscreen; no native
  desktop restart or full-repository run was performed.

Earlier
full-GUI and historical target-qualification failures remain open; no complete
repository or whole-microscope acceptance is claimed. Generated numerical
files/previews stay in ignored `tmp/`. No commit, push or running-application
restart is part of this continuation.

## Remaining work

Finite-loss arrival timing needs physically located loss events and ordered
energy-dependent flights. It must not be completed by assigning the specimen
reference plane as an invented event depth. Arbitrary bent-filter section
readout needs a physical plane definition and saved crossing data. Whole-High
performance still needs the same user request measured in the restarted app.
These are separate from the currently paused coherent/phase development.
