# High accuracy: slow Stage 2 diagnosis and repair

## Observed problem

The running desktop application remained at `Stage 2/5 | Tracing the electron
column` with Ideal Optics selected. Read-only stack sampling of the actual
Python worker (PID 39380; the project venv launcher was PID 40748) located the
work in `trace_feg_to_exit -> _analytic_step -> boris_step`, not a curved-tip
field solve. The base Anaconda executable is the venv launcher's child; NumPy
was loaded from this project's `.venv`.

Two live samples established:

- 15,000 emitted particles, `surface_model=None`, no installed monochromator;
- 38,603 and then 40,087 integration steps; the step counter and physical time
  continued to advance;
- approximately 4 GB resident private memory at the second sample, with one
  CPU core occupied by the original gun implementation;
- later sampling found the same job in specimen EDS photon transport. The gun
  calculation eventually returned. This was very slow work, not a deadlock.
- the final sample showed only the idle GUI main thread. The numerical worker
  had ended; its successful result versus error presentation was not inspected.

No parameters, source model, application state or running process were changed
during inspection. The live job's complete configuration was not recovered;
the separate reproduction uses the repository's default flat-tip gun.

## Cause

Commit `9ea3a7c` (2026-09-17) introduced trajectory-error control for the ordinary
analytic gun as well as the curved-tip work. Each attempted step compares one
full Boris step with two half steps, applies the static-energy invariant to
each, and checks the entire active particle bundle. Thus an ordinary flat tip
also executes this expensive path.

A separate 49-particle default-source reproduction required 8,864 accepted
steps and 70,263 Boris calls. There were 14,557 rejected attempts; 14,336 were
limited by transverse momentum error. Near the axis, the existing floor in
that relative error corresponds to an angular error scale of about 1e-10 rad
per step. A single demanding trajectory constrains the common time step.
These facts explain why increasing the bundle size is more expensive than a
simple particle-count multiplier.

The former NumPy path repeatedly constructs arrays and checks fields during
every retry. Its adaptive steps also produce more history snapshots under the
existing history stride. Stage 2 encompasses the gun, column and diagnostics
without identifying these substages in the progress label.

An independent defect prevented High accuracy from passing its cancellation
token into the gun. Preview/Medium already passed it. Consequently High could
not stop at the gun's existing per-step cancellation checks.

## Implemented changes

- Added a compiled CPU implementation of the **same** analytic fields, Boris
  update, energy projection, impulse bound and full/two-half-step comparison.
  Bundles with at least 1,024 active particles can use the worker's existing
  Numba thread budget. Small bundles use the serial compiled kernel.
- Parallel work is independent per particle. Acceptance still checks all
  active particles and uses one common time step. Numerical errors are reported
  after the parallel section; rejected steps cannot hide invalid particles.
- The original NumPy implementation remains available. Custom/combined or
  overridden field providers, absent Numba and explicitly disabled acceleration
  defer to it. Current field settings are read for each call; there is no cached
  replacement source or field state.
- High accuracy now passes its existing cancellation token to the physical
  transport. Cancelling inside the gun terminates at the next step boundary;
  partial results are neither published nor inserted into gun caches. This
  does not promise immediate cancellation inside every downstream kernel.
- Versioned the analytic step implementation and bound both its version and
  impulse fraction to dependent calculation signatures. Historical files remain
  readable; old executed results cannot stand in for the revised calculation.

The source, geometry, voltages, ray count, requested spatial step, all error
thresholds, history sampling, aperture processing, vacuum participation and
curved-tip implementation are unchanged. Coherent calculations remain paused.

## Evidence and limits

Local receipts and generated arrays are retained under
`tmp/high-accuracy-20260919/` and excluded from Git. In particular:

- `live-stack.json` and `live-stack-2.json`: read-only live snapshots;
- `flat-49.json` / `flat-49-errors.json`: the original step/retry diagnosis;
- `gun-reference-49.npz` / `gun-compiled-49.npz`: initial complete tip-to-exit
  comparison; all particle IDs, weights, energy offsets, stop masks and currents
  match. Maximum exit-coordinate component difference was 3.24e-18 m and
  maximum slope-component difference 4.34e-18 rad;
- a local 15,000-particle update plus impulse-limit benchmark measured 13.178 ms
  for the reference versus 2.465 ms with four compiled CPU threads, about 5.35x.
  These are local kernel timings, not whole-gun or whole-High speed claims.

The original 49-particle diagnostic runs took about 10–16 seconds depending on
concurrent load. The initial compiled complete gun run took 4.19 seconds. The
desktop job continued running during measurements, so these are illustrative
local measurements rather than isolated hardware performance qualification.

An initial **serial compiled** 15,000-particle diagnostic was deliberately
cancelled at its 180-second budget after 4,839 steps. It is incomplete evidence,
and was the reason to add the parallel and fused impulse calculations. Do not
present that diagnostic as a completed High calculation.

The final four-thread default-gun run (`gun-compiled-final-15000.json`) was also
bounded: **33,335 accepted steps in 360.17 seconds**, then the diagnostic's own
cancellation callback stopped it. It was still progressing and had not completed
the full gun. The exception text `Superseded optical tuning request` here is the
existing cancellation mechanism, not a spontaneous solver failure. Its 6-minute
budget and the earlier serial run's 3-minute budget differ; neither certifies a
complete 15,000-particle High runtime. Large-bundle integration, history cost and
the broad Stage 2 label remain performance/usability follow-up work. The repair
does not establish that prior whole-High speed has been restored.

Final regression evidence:

- `final-physics.xml`: **58 passed**, exit 0. Includes compiled/reference and
  parallel parity, full tip-to-exit transport, unchanged current/aperture trends,
  actual launch-potential handoff, tip-origin lens refinement, source admission,
  cache invalidation and High cancellation.
- `transport-cache.xml`: **63 passed**, exit 0, for the preceding integrated
  step implementation, calculation reuse and gun aperture/voltage contracts.
  This ran before the final fused impulse helper; the final 58-case run covers
  the resulting transport path. Counts overlap and must not be added.
- `classical/report.json`: **259 passed**, unchanged source during execution,
  all **965** final source/configuration/input inventory hashes matched. Run ID
  `c5a31d64a673452988bacf33f6e7b82a`, exit 0. This declared classical software
  lane explicitly leaves the full simulator **UNQUALIFIED**.
- Python compilation and whitespace validation passed. Existing Pydantic
  deprecation and pytest XML-property-format warnings are not test failures.

## Applying the repair

The already running desktop process retains its loaded implementation. Save
the current configuration/results and restart the application to load the
repair. Nothing was force-closed or restarted by this investigation. No commit
or push was performed. This is a bounded classical performance repair, not
qualification of the full microscope, coherent imaging or the energy filter.
