# Round 2 optimisation progress

The first receipt below records the R2-00..03 slice. The later **Stigmator and
scan/descan continuation** section supersedes its earlier G1/G2 and "no configs
edited" statements for that later scope; it does not rewrite the old evidence.

## Environment and baseline

- Actual branch / HEAD: `master`, `985463968f365fdb945af1b4f3a744a7a53a5912`.
- Dirty paths preserved: untracked user brief `CODEX_OPTIMIZATION_ROUND2.md`.
  No reset, pull, commit, push, application restart or production-default change.
- Read the entire current brief, root `AGENTS.md` (the only applicable file),
  `PROJECT_FUNCTION_SPEC.md`, README, working-point workflow and the current
  first-round package table/final receipts. Historical wave descriptions do not
  override the user pause or permit downstream illumination sources.
- Interpreter: project `.venv/Scripts/python.exe`, Python 3.12.3; Windows 11 Pro
  10.0.26200. NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, PySide6 6.8.3,
  pyqtgraph 0.14.0, pytest 9.1.1, pytest-qt 4.5.0, threadpoolctl 3.6.0.
- Hardware detected: Ryzen 9 9950X3D, 16 cores / 32 logical processors;
  OS-visible RAM 100270288 KiB; NVIDIA RTX 5090, driver 32.0.16.1062.
  `numba.cuda.is_available()` returned true. This supersedes the old host's
  unavailable-CUDA receipt; availability alone does not prove numerical parity.
- Baseline (before edits), `QT_QPA_PLATFORM=offscreen`:
  `.venv/Scripts/python.exe -m pytest tests/test_source_admission.py tests/test_working_point_contract.py tests/test_alignment_transactions.py tests/test_vacuum_opt_in.py -o addopts= --tb=short --junitxml=tmp/round2-20260917/baseline.xml`.
  Exit 0: **41 passed**, 16 existing Pydantic deprecation warnings, 80.52 s.
  Complete output: `tmp/round2-20260917/baseline.log` (locally ignored).
- Scope exclusions: coherent propagation and production image qualification,
  native GUI, microscope hardware, unbounded calculations and preset changes.

## Package status

| Package | Status | Changed files | Executed evidence | Remaining limit |
| --- | --- | --- | --- | --- |
| R2-00 | VERIFIED_SCOPED | This receipt | 41-test classical baseline | Not a full simulator audit |
| R2-01 | VERIFIED_SCOPED | `acceptance.py`, `acceptance_pytest.py`, `validate_classical_scope.py`, CI, tests | Final 258-case lane; deliberate failing child and legacy blocker report | Full-image gates remain blocked |
| R2-02 | VERIFIED_SCOPED | Existing coordinator/controller/request, `job_events.py`, lifecycle/stress tests | Final 258-case lane; reproduced queue order; two final 21-case stress processes | Offscreen only; no hard real-time guarantee |
| R2-03 | VERIFIED_SCOPED | Existing backend/core/device cache, `backend_execution.py`, GUI/performance log, tests | Final 258-case lane; isolated real-device/utility checks | No full-chain GPU or coherent-image qualification |

## Reconciliation and requirement mapping

The first-round WP-00..11 implementations remain owners: full working points,
lazy portable archives, sampling comparisons/topology, parameter registry,
pinned input assets, GPU plan cache, FIFO jobs, task layouts, constrained alignment
and detached experiments. None is recreated. Their prior receipts remain intact.

| Existing package | Preserved owner / current delivery |
| --- | --- |
| WP-00 | Existing CPU baseline and bounded measurement tools; hardware receipts remain host-specific |
| WP-01 | Complete Working Points, lazy index, Apply/undo, frozen A/B and topology references |
| WP-02 | Independent source/gun/column/mesh sampling comparisons; unresolved populations stay unresolved |
| WP-03 | Existing parameter registry and conservative unknown-input signatures |
| WP-04 | Immutable pinned assets and background captured-state preparation |
| WP-05 | Portable archive resolver, structural/external inputs and historical migration |
| WP-06 | Existing column device plan/buffer cache and measured-cost Auto selector |
| WP-07 | Existing shared FIFO resource coordinator and controller ownership; extended, not replaced |
| WP-08 | Four task layouts and read-only result/A-B presentation |
| WP-09 | Forward-validated constrained condenser and static beam alignment; physical limits retained |
| WP-10 | Detached runtime/geometry experiments, resume and retained failed candidates |
| WP-11 | Previous bounded integration/build receipts; not upgraded to full qualification |

| Selected task | Existing extension point and tests | Requirement |
| --- | --- | --- |
| R2-00 | Source admission, snapshot, alignment and vacuum baseline above | UR-007/009/029/030 |
| R2-01 | Legacy `validate_development_spec.py`, `test_development_acceptance.py`, additive classical runner | UR-030; round2/R2-AT-03..05 |
| R2-02 | `gui/job_coordinator.py`, `calculation_controller.py`, `calculation_request.py`; existing job/background tests | UR-030; round2/R2-AT-06..09 |
| R2-03 | `physics/compute_backend.py`, `core.py`, `ray_device_cache.py`; policy/residency tests | UR-030; round2/R2-AT-10..12 |

## Findings resolved or still open

| Finding | Reproduction / source evidence | Change | Verification | Limit |
| --- | --- | --- | --- | --- |
| F01 | CI invokes full-image acceptance during wave pause | Additive classical lane and manual blocker report | Manual report: exit 1, NOT_RUN/UNQUALIFIED, AT-12/13/14 BLOCKED | Legacy validator unchanged |
| F02/F03 | Implicit workers; reproduced five-second startup failure | Explicit adapters, events, Qt ownership and coalesced priority admission | Cold/warm/rapid/cancel/High/closure fixtures; before/after queue-order regression below | No hard real-time guarantee or general native-GUI qualification |
| F04 | CuPy compile/runtime classes broadly retryable | API/code/stage taxonomy, original error evidence, context quarantine | Policy regressions below | No production wave run |
| Local 01 | Rejected preparation did not close its lease | Exactly-once cleanup for rejected/queued/active jobs | Six terminal scenarios | Reservation is not an RSS cap |
| Local 02 | Decorated Preview could replace retained High | Only canonical High accuracy replaces High | Five label regressions | No physical result changed |
| Local 03 | Backend toolbar rewrote stored policy/enable flag | Display aliases without migration, validate new selections | Both GUI regressions passed in final lane | Historical choices remain readable |

## Acceptance mapping

| Namespaced criterion | Status | Exact test / command | Evidence type | Limitation |
| --- | --- | --- | --- | --- |
| round2/R2-AT-01/02 | VERIFIED_SCOPED | Baseline above | Contract tests and bounded particle persistence | No coherent or assembly qualification |
| round2/R2-AT-03..05 | VERIFIED_SCOPED | `classical-verified/report.json`, exact node IDs and all three phases | Real CLI, fail-closed fixtures, unchanged legacy blockers | Classical software scope only |
| round2/R2-AT-06..09 | VERIFIED_SCOPED | Same final receipt; deterministic barriers and queue-order regression | Offscreen controller ownership and bounded dispatch evidence | Native desktop and arbitrary UI load not qualified |
| round2/R2-AT-10..12 | VERIFIED_SCOPED | Same final receipt; backend-policy/device-cache tests | Error identities, original exceptions, preflight and stage receipts | Most errors use fixtures; no full-chain GPU parity |

## Performance receipts

No speed-up claimed. Baseline elapsed time is test duration, not solver throughput.

## Implemented contracts and intermediate receipts

- Classical acceptance requires its explicit file allowlist, exact pytest
  collection and every setup/call/teardown outcome. Missing, skipped, deselected,
  unknown or duplicate evidence, nonzero test exit, and source/configuration
  drift fail closed. Composite criterion IDs keep the three AT-12 meanings
  distinct. A passing software scope cannot qualify full imaging.
- `.venv/Scripts/python.exe scripts/validate_classical_scope.py --scope full-report --output tmp/round2-20260917/full-report`
  completed with exit **1**, software NOT_RUN, full UNQUALIFIED and the existing
  three source blockers. No physics launched. The legacy validator and its
  allow-no-GPU semantics were not edited. Initial policy check: 35 passed,
  1 deliberately deselected pending creation of the backend test file; not a
  whole-lane PASS.
- Workers now have explicit cancellation/identity/claim/outcome/cleanup adapters.
  The bounded scalar event ring records worker-thread entry separately from
  admission, stages, publication, cancellation and one terminal cleanup. On an
  actual test timeout, `diagnostic_snapshot(include_stacks=True)` records owners
  and thread stacks. No arrays are logged. Owners/runnables survive queued
  delivery until Qt confirms return. New live requests preserve captured High
  jobs and independent experiments.
- New combined lifecycle tests initially crashed with native exit
  `-1073741819`, despite isolated tests passing. Logs remain locally as
  `lifecycle-{first,second,ownership,receivers,sender,no-libraries}.log`.
  Removing numerical setup in a diagnostic child did not prevent the crash;
  suppressing cyclic collection did. Explicit Qt application ownership of
  parentless coordinators prevented it. Production still collects cycles on
  the GUI thread. A regression explicitly collects Python cycles and disposes
  the drained Qt owner. This is a reproduced wrapper/dispatch-owner lifetime
  defect, **not proof of the original F02 five-second timeout cause**.
- `lifecycle-owner.xml`: 52 passed / 1 new assertion failed because a Python
  wrapper can remain after its native QObject is deleted. The test now checks
  native invalidation. `lifecycle-stress.xml`: **28 passed**, 87.58 s, exit 0,
  including five rapid-edit attempts. No entry timeout was increased.
- `.venv/Scripts/python.exe scripts/stress_job_lifecycle.py --repetitions 2 --output tmp/round2-20260917/stress`
  completed two fresh processes, **20 passed each**, 48.391 / 43.015 s supervisor
  wall time. Receipt: `stress/6e8c0137e84643d7aa52abeeef0819eb/report.json`.
  Each process contains five warm queued-High/independent-owner cases and five
  rapid-edit GUI cases. Worker-entry timeout remains **5 seconds**.
- Validation watchdogs terminate only their owned process tree: Windows venv
  launchers can spawn another interpreter. No process-name kill or user-app
  control is used. Preparation estimates captured-buffer/decode costs instead
  of reserving an entire 24 GiB solver. Production budgets/thread counts stay
  unchanged; this remains conservative ownership accounting, not measured RSS.
- GPU retry permits confirmed device absence/OOM, not compiler, launch/input,
  numerical, I/O, cancellation or unknown faults. Evidence retains type,
  message, traceback/cause, attempted backend and stage. Corrupt contexts clear
  and quarantine cached allocations for that process; ordinary cache clearing
  cannot reset quarantine. There is no automatic driver reset/app restart.
  CPU remains explicitly selectable.
- Error identities were checked against NVIDIA's primary
  [Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
  and [Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html).
  Code 701 is not confirmed OOM; illegal access is not repaired by CPU retry.
  These references support error policy, not microscope-model validation.
- Read-only backend preflight rejects strict-GPU mapped/vacuum columns before
  gun execution. Gun, particle specimen, filter and column calls have separate
  requested/actual backend receipts attached to the result and existing GUI log.
  Call receipts may include exact internal reuse; they do not imply new physics
  for every invocation. No filter, sink, extraction/acceleration or source gate
  was bypassed.
- `backend-first.xml`: **62 passed**, 10.57 s. Expanded controller/backend group:
  **79 passed / 1 failed**, 65.83 s (`controller-backend.xml`). A new receipt
  fixture incorrectly supplied an empty external-input inventory and was
  correctly rejected. It now captures actual inputs; the guard was not relaxed.
- In that group, `test_compute_backend.py::test_cuda_ray_trace_matches_cpu_with_energy_spread`
  passed on the RTX 5090: 64-ray finite column segment, existing tolerances
  rel 2e-6 / abs 1e-9. This is an isolated real-device integrator comparison,
  **not** emitter-chain/current/stop parity or R2-AT-30 qualification.
- `policy-integrated.xml`: **62 passed / 1 blocked failure**, 10.91 s. The old
  `test_stem_cuda_pipeline.py::test_resident_cuda_failure_discards_partial_work_and_recomputes_on_cpu`
  fails the existing coherent source-admission gate. No bypass/skip was added,
  and it executed no wave propagation. It remains outside the classical lane;
  full-scope qualification stays incomplete.
- `backend-final.xml`: **58 passed**, 8.65 s: narrow error fixtures, device
  reset/quarantine ownership and two tiny shared FFT/multislice OOM utilities.
  Those utilities are isolated mathematics, not production imaging.
- First integrated classical receipt, `classical-final/report.json`, run
  `29a43375413541adb4370f2856a9d3bd`: **255 passed / 1 failed**, 236.12 s.
  The unchanged five-second rapid-edit preparation assertion timed out. The
  captured trace shows enqueue at 11258.640 s, admission/worker entry at
  11264.312 s, with the worker then inside threadpoolctl library discovery.
  This establishes a 5.672 s pre-entry delay, not that library discovery caused
  that preceding delay. Dispatch, retained-inventory and GUI-collection timing
  were added for the next reproduction. This failed run remains visible and
  does not qualify R2-AT-07. No timeout, physics gate or required test was relaxed.
- Instrumented repeat `classical-dispatch/report.json`, run
  `e93ca7a82af34d018908806d3f08d5b0`, again **255 passed / 1 failed**.
  Enqueue: 11599.437 s; dispatch entry: 11604.843 s; retained inventory finished
  and worker entered: 11605.015 s. The preceding GUI collection ended at
  11599.375 s; the next started at 11605.015 s. Thus the reproduced 5.406 s
  pre-dispatch gap is not retained-memory scanning (0.172 s) or that collector.
- Local stack sampler `tmp/round2-20260917/probe_dispatch.py` reran the four
  preceding files: **91 passed**, 131.13 s (`dispatch-profile.xml`). One-second
  samples in `dispatch-stacks.log` captured pyqtgraph `forgetView` /
  `updateAllViewLists` / `setViewList` during the startup wait, followed by Qt
  event processing before worker entry. This diagnostic changed observation
  timing and did not itself reproduce the failure; it identifies pending plot
  disposal as an observed contributor, not all possible GUI stalls.
- A deterministic queue-order regression then failed against zero-timer
  dispatch: `admission-before.xml`, **1 failed**, 4.23 s. A normal-priority
  display-cleanup event posted first ran before admission. Dispatch now posts
  one coalesced high-priority admission event; work still uses the same FIFO
  queue/pool and cancelled queued work is still cancelled before execution.
  `admission-after.xml`: **30 passed**, 10.07 s, including the new ordering
  regression, terminal ownership, GC affinity and strict-GPU aliases. No test
  timeout or thread/resource default changed. This does not preempt an already
  running GUI callback or make plot teardown itself faster.
- Historical strict-GPU aliases now normalise consistently at discovery,
  column dispatch and retry boundaries, including surrounding whitespace.
  Readable historical aliases remain distinct from validated new selections;
  no saved value is rewritten.

All logs/XML above live under `tmp/round2-20260917`. Pydantic deprecations and
an offscreen pyqtgraph disconnect warning remain visible. Overlapping run totals
are not summed. The later final receipt supersedes intermediate code snapshots.

## Physical and hardware blockers

Paused coherent development; undefined extra stigmator response, dynamic alignment
targets and active-vacuum incident observer remain unavailable. Full-chain GPU
scientific parity and native desktop behavior have not been tested. The small
real-device integrator comparison above does not close those criteria.

## Integrated changes

- Acceptance: `src/temsim/acceptance.py`, `acceptance_pytest.py`,
  `validation_process.py`, `scripts/validate_classical_scope.py`, the existing
  `.github/workflows/tem-p0.yml`, and `test_classical_acceptance.py`.
- Ownership: `job_events.py`; existing GUI coordinator, calculation controller,
  request preparation, garbage-collection diagnostics and result readout;
  `test_job_lifecycle.py`, `scripts/stress_job_lifecycle.py`, and existing
  background-preview/result-readout regressions.
- Backend: existing compute backend, classical core, device cache and simulation
  entry; additive `backend_execution.py`; CPU call receipts at existing gun,
  particle-specimen and filter boundaries; existing performance/UI reporting;
  `test_backend_failure_semantics.py`, strict-policy aliases and two isolated
  utility fixtures changed from unspecified GPU failure to explicit OOM.
- Specification: additive UR-030 and revision entry in
  `PROJECT_FUNCTION_SPEC.md`. Earlier requirement/acceptance IDs and the legacy
  full-development validator remain unchanged. No edits to `configs/`,
  `AGENTS.md`, `pyproject.toml`, dependency locks or the user-supplied brief.

## Exact resumption instruction

R2-00..03 are integrated for the classical software scope. R2-04..11 have not
been implemented or certified by this slice. Continue the next ready package
using existing owners, without duplicating Round 1. If a new startup timeout
appears, use dispatch/worker/stage/GC traces to distinguish already-running GUI
work from admission delay; do not assume all stalls share the observed plot
disposal mechanism. Preserve defaults, live app, paused coherent work and the
user brief. Final build/stress receipts follow below.

## Final integrated receipt

- Command: `.venv/Scripts/python.exe scripts/validate_classical_scope.py --scope classical --output tmp/round2-20260917/classical-verified`.
- Run ID: `4112e595708448d5a864df1c1e2010fe`; process exit **0**;
  **258 passed**, no failed/skipped cases, 16 existing Pydantic warnings,
  243.13 s pytest duration. Exact collection/outcomes, command, software/platform
  identity and source SHA-256 inventory are in `classical-verified/report.json`.
- `software_scope_status=PASS`, `source_unchanged_during_tests=true`,
  `full_simulator_qualification=UNQUALIFIED`. Legacy AT-12/13/14 remain BLOCKED.
  Earlier failed receipts are retained above, not overwritten.
- Final stress command:
  `.venv/Scripts/python.exe scripts/stress_job_lifecycle.py --repetitions 2 --output tmp/round2-20260917/stress-final`.
  Exit **0**, **21 passed in each fresh process**, no skipped cases. Supervisor
  wall times 36.672 / 34.250 s; pytest times 34.58 / 31.99 s. Receipt:
  `stress-final/c4b20c8d9d5d4e0387db8b11f7aa8274/report.json`.
  These small repeat counts are scoped regression evidence, not a reliability
  probability or a cold/warm solver speed benchmark.
- Final full-status command uses `--scope full-report --output tmp/round2-20260917/full-report-final`.
  Exit **1** by design, run `5a6358ada4bf4f71be775e2e6c728ef5`:
  NOT_RUN / UNQUALIFIED with the same legacy blockers; no wave calculation.
- Final isolated numerical regressions: `.venv/Scripts/python.exe -m pytest tests/test_compute_backend.py::test_cuda_ray_trace_matches_cpu_with_energy_spread tests/test_multislice.py::test_cupy_oom_retries_the_complex128_cpu_reference tests/test_wave_fft.py::test_cupy_fft_oom_falls_back_without_losing_the_result -o addopts= --tb=short --junitxml=tmp/round2-20260917/isolated-numerics-final.xml`.
  Exit **0**, **3 passed**, 9.35 s. The RTX 5090 comparison actually executed,
  not skipped. Its one-block low-occupancy warning is expected for 64 rays.
  The other two checks are tiny mathematics/error fixtures, not coherent imaging.
- `.venv/Scripts/python.exe -m compileall -q src scripts tests`: exit **0**,
  executed serially after tests. `git diff --check`: exit **0**.
- Offline wheel command: `.venv/Scripts/python.exe -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir tmp/round2-20260917/wheel .`.
  Exit **0**; 1,932,017-byte wheel SHA-256
  `a935c5771cb3816891ab5c9153adc0137b5e1859d3fd6976d89d184efd16f87a`.
  Installed with `pip install --no-index --no-deps --target tmp/round2-20260917/installed`.
  The first target import failed because a bare `--target` does not supply the
  normal prefix-based configuration layout. Repeated using the existing,
  process-local `TEMSIM_PROJECT_ROOT` override pointing at that target:
  all seven new/touched acceptance/job/backend/controller imports resolved to
  the wheel, and CONFIG_ROOT resolved to the wheel's own `configs` directory.
  Exit **0**. This reuses project dependencies, not a fresh dependency install;
  no GUI/solver, production environment modification or source-path fallback.
- Final process inventory contained no remaining validation Python processes.
  A final SHA-256 inventory comparison matched the successful classical run;
  source/tests/configurations were not edited during or after that run. HEAD is
  still `985463968f365fdb945af1b4f3a744a7a53a5912`. Generated receipts, wheel,
  temporary target installation and diagnostics remain local under ignored
  `tmp/round2-20260917`; no generated numerical cache was staged or committed.
  No commit, push, PR, hardware action or user-application restart occurred.

## Stigmator and scan/descan continuation — 2026-09-17

### Authorization, reconciliation and physical contract

The user subsequently authorized stigmator and scan/descan implementation,
including structural TOML, and continued ready unfinished work. The prohibition
on commit/push, restarting the running application and retuning production
defaults remains. All earlier dirty R2-00..03 edits and the user brief are retained.
UR-031 and `docs/stigmator-and-scan.md` record the opt-in contract.

- G1's missing second basis now has an explicit ideal normal/skew field model.
  Condenser/objective/diffraction X/Y span two trace-free quadrupoles, with
  structural principal-axis angles 0/45 degrees and existing Gaussian effective
  lengths. Legacy difference remains the default and is explicitly rank one.
  Historical profiles lacking the field-model selector cannot inherit the new
  law from the live instrument. No strengths, source settings or lens presets
  are changed automatically. The existing gun stigmator is not replaced.
- TOML additions cover all five columns, the historical NoEnergyFilter recording
  manifest, and the **actual shared projector_stack subassembly** used by the
  split EnergyFilter assembly. They declare two interleaved four-pole windings
  (eight coils), basis angles, reference URLs/evidence, and two axial X/Y stations
  per scan pair using one raster clock. Existing dimensions/positions are retained.
  No unknown winding turns, material properties, CAD or coil-current calibration
  are invented. Housing renderers are unchanged; this is a structural principle
  declaration with effective fields, not new measured coil geometry/FEM.
- Primary references: original FEI Tecnai User Interface Manual, printed p.17
  (two independent stigmator elements at 45 degrees), p.19 and p.52
  (diffraction-focus / beam-shift-pivot coupling):
  https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_alignments_doc.pdf .
  JEOL's illustrated eight-coil principle:
  https://www.jeol.com/words/semterms/20201020.111014.php .
  These references establish topology/operation, not simulator dimensions.
- G2's held-drive contract uses the existing raster law, physical foil centres,
  opposite descan command and executed 4x4 transfer. Probe zero-angle calibration
  and FOV use the selected specimen centre/entrance; descan matches position at
  a separately selected installed physical plane. User pivot offsets are real
  lower-ratio changes. Automatic mode stays the default. Held mode preserves
  calibration when optics change; requested FOV is distinguished from actual
  displacement. Singular/over-limit solves roll back both pairs.
- Filter-crossing targets fail explicitly because the first-order observer has
  no qualified filter response. No filter stage is bypassed. Static matrix
  calibration with a shared raster is not calibrated electronic dynamics,
  automatic diffraction-focus optimization or measured OEM matching.

### Integrated changes

- `optics/stigmator_field.py`, existing stigmator classes and assembly binding
  own the tensor. CPU/Numba/CUDA RK4, active mapped-field transport and specimen
  Lorentz fields consume its off-diagonal term. Mathematical slice helpers
  preserve or reject the term explicitly; paused coherent calculations were not
  enabled or executed. Normal/skew coefficients enter numerical plan identity,
  prefix comparison, device ownership and persistent seeds.
- Incident restart content uses codec `incident-simulation-seed-v2-quadrupole-tensor`.
  Both node and midpoint skew arrays are mandatory. Historical v1 bundles remain
  readable, but are not silently promoted to new active tensor restarts. No
  existing cache directory is deleted or published.
- `physics/scan_calibration.py`, the existing scan geometry/controls and state
  loaders retain a versioned held calibration with captured input identity,
  physical ratios, original FOV and reference. Explicit capture is transactional.
  Existing shared clocks and source-to-column propagation are reused.
- `stigmator_alignment.py` extends the existing detached alignment solver,
  controller, registry and Apply/undo gate. It targets two current-weighted
  position-covariance components at the specimen entrance, not wave A1. Controls
  are limited to the condenser stigmator's two strengths. Current, effective
  samples, D95, rank/conditioning, search bounds and independent gun/column-step
  checks remain mandatory. Other optics cannot be optimized away by this task.
- `physics/phase_space_statistics.py` supplies shared 4-D current-weighted means,
  covariance and projected geometric RMS emittances. The existing Working Points
  Sampling tab displays a compact optional table; SI evidence is retained.
  Empty/historical index data do not acquire invented moments. Pinned A/B handles
  immutable nested metrics and asymmetric available keys without restoring state.
- Registry descriptions distinguish ideal percent strength, structural basis,
  held drives and requested FOV. A cache audit found `last_gun_waist_mm`, assigned
  only as a completed diagnostic, misclassified as an unknown physical input.
  It now shares the existing state-product exclusion path. Historical snapshots
  still read that value, but new input identity/cache keys do not change merely
  because a trace completed. Unrelated unknown inputs still invalidate reuse.

### Package status after this continuation

| Package / gate | Current status | Explicit remaining scope |
| --- | --- | --- |
| R2-00..03 | Existing implementation retained; reverified below | No upgrade to full-image/hardware qualification |
| R2-04 | Partial: existing readout extended, A/B fixed | Full task-summary, coalesced edit undo and populated/scaled layout acceptance remain open |
| R2-05 | Partial: stigmator/scan descriptions, units, profiles and TOML authority | General parameter inventory/hardware calibration audit remains open |
| R2-06 | Multi-level plan/resume integrated above the existing pair engine; explicit axes, policy identities, budgets, scalar journal and existing-panel controls | Detached candidate-search action and broader actual assembly convergence remain open; supported field models only, no invented domain control |
| R2-07 | Partial: shared weighted covariance/emittance table | Full explanatory loss-budget/transport presentation remains open |
| R2-08 | Scoped tensor GPU correctness only | Full-chain measured GPU/performance/resource study remains open |
| R2-09 | Partial: geometric twofold task in existing transaction | Robust experiment decision UI, full response evidence review and qualification-plan links remain open |
| R2-10 | Partial: versioned tensor restart and historical read safety | Recovery journal and broader archive interruption/lifecycle work remain open |
| R2-11 | Scoped integration and receipts below | Native desktop, populated latency/scaling matrix and full workflow acceptance remain open |
| G1 | Ideal independent field implemented | No measured coil map/current calibration or wave-stigmation qualification |
| G2 | Explicit held scan/physical-plane contract implemented | No measured electronics dynamics or filter-crossing observer |
| G3/G4/G5 | Boundaries preserved | Active-vacuum observer unsupported; coherent work paused; OEM/experimental calibration unestablished |

### Intermediate verification and defects found

All XML/receipts below remain local under ignored `tmp/round2-20260917`.
Overlapping totals are not added together. Offscreen fixtures are not native GUI
verification; small actual CUDA kernels are not performance benchmarks.

- Initial 42-case boundary suite: 33 passed / 9 failed. Fixes covered the new
  19-array kernel interface in emulated device fixtures, a mock missing the angular
  response it purported to supply, and optional recording planes in GUI fixtures.
- Next 36-case suite: 35 passed / 1 failed; updated the exact control-set assertion
  for the new AC-only pivot controls. The real CUDA tensor comparison executed.
- Held-drive plus existing scan suite: **26 passed**, 42.00 s.
- First alignment/statistics/registry suite: **69 passed / 3 failed**, 151.06 s.
  All three failures were a stale raw EnergyFilter TOML lookup after subassembly
  splitting; tests now read the same resolved structural owner as the application.
- `stigmator-scan-integrated.xml`: **147 passed**, 197.06 s. Later codec/readout
  additions were not all in that collection; this is intermediate evidence.
- `stigmator-scan-final.xml`: **233 passed / 3 failed**, 345.80 s. Two restart
  failures exposed the derived gun-waist/cache defect; one pinned A/B failure
  exposed immutable nested-metric JSON conversion. Both are fixed, not waived.
- `cache-followup.xml`: **3 passed**, 29.05 s, covering the reproduced cache issue,
  historical diagnostic readback, restarted tensor seed and read-only A/B.
- `held-controls-final.xml`: **10 passed**, 51.31 s. Separate actual transfer
  calculations confirm pivot-only and diffraction-focus-only displacement with
  unchanged held coupling; includes requested-FOV scaling, clock/profile snapshots,
  failed-calibration rollback and refusal to bypass the filter.
- Two attempted commands named non-existent sampling test files and collected
  zero cases; they are not acceptance evidence. The final collection uses actual
  existing independent-emission, product-usability and topology tests instead.
- Mathematical coverage includes independent matrix-exponential comparison,
  rotation/double-angle behavior, CPU/Numba/actual CUDA parity, mapped transport
  and specimen Lorentz signs. Actual 49-ray tip-origin executions establish
  rank-two response; they do **not** certify convergence or a successful real
  geometric-stigmation optimization. Successful inverse/undo/rank/failure fixtures
  are explicitly synthetic, separate from those real particle executions.

### Final source-frozen continuation receipt

- Classical command: `.venv/Scripts/python.exe scripts/validate_classical_scope.py --scope classical --output tmp/round2-20260917/classical-stigmator-scan --timeout-seconds 900`.
  Exit **0**, run `97676ff788c84c64beeec454571f5626`: **259 passed**,
  no failed/skipped cases, 335.44 s pytest duration. Report:
  `classical-stigmator-scan/report.json`; `software_scope_status=PASS`,
  `source_unchanged_during_tests=true`, full simulator **UNQUALIFIED**.
  Legacy BLOCKED/NOT_RUN rows remain present. No acceptance scope was widened.
- Feature/cross-regression command: `.venv/Scripts/python.exe -m pytest tests/test_stigmator_tensor.py tests/test_held_scan_calibration.py tests/test_stigmator_alignment.py tests/test_beam_alignment.py tests/test_phase_space_statistics.py tests/test_parameter_registry.py tests/test_parameter_semantics.py tests/test_scan_system.py tests/test_vector_field_transport.py tests/test_ray_device_residency.py tests/test_first_order_transfer.py tests/test_independent_emission_sampling.py tests/test_calculation_manifest_artifacts.py tests/test_segmented_column_cache.py tests/test_product_usability.py tests/test_topology_evidence.py tests/test_profile_optional_values.py -o addopts= --tb=short --junitxml=tmp/round2-20260917/stigmator-scan-verified.xml`.
  `QT_QPA_PLATFORM=offscreen`; exit **0**, **239 passed**, no skipped cases,
  411.63 s. This supersedes the failed 236-case intermediate receipt. Counts
  overlap the classical lane and must not be summed. The actual CUDA tensor
  test ran; its one-block low-occupancy warning is expected for three rays.
- Sixteen existing Pydantic deprecations remain visible. Pyqtgraph emitted an
  offscreen ViewBox disconnect warning at interpreter teardown; neither run
  suppressed it or claimed a native-desktop pass.
- Separate offscreen widget renders exercised ScanControlView at 1280x720 and
  DirectAlignmentPanel at 600x720 logical pixels at 100/125/150/200% scale. All
  eight renders completed; minimum sizes were 766x386 and 99x240 respectively.
  The scan 100% and alignment 150% renders were visually inspected: new controls
  and scrolling were readable. PNGs are local `*-scale-*.png`. This used Segoe UI
  and empty plots, not the full populated application or native desktop; it does
  not close R2-AT-16/39. The user's running application was untouched.
- `.venv/Scripts/python.exe -m compileall -q src scripts tests`: exit **0**,
  serially after both final suites. `git diff --check`: exit **0**.
- Offline wheel: `.venv/Scripts/python.exe -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir tmp/round2-20260917/stigmator-wheel .`.
  Exit **0**; size **1,945,799 bytes**, SHA-256
  `b55fd2f91fd3344cb1cd623f9a869ff96e621f5b46edbbdcd7bd8058855c2136`.
  Installed only into the new local `stigmator-installed` directory with
  `--no-index --no-deps --target`. Eight new/touched modules imported from that
  target, with process-local `TEMSIM_PROJECT_ROOT` pointing at the target.
  Its own split EnergyFilter TOML resolves the eight-coil stigmator declaration.
  Exit **0**. Existing project dependencies were reused, not reinstalled; this
  was an installed-package import check, not a native application launch.
- HEAD remains `985463968f365fdb945af1b4f3a744a7a53a5912`. No commit, push,
  application restart, hardware acquisition, coherent propagation, source/preset
  retuning or cache deletion took place. Generated arrays/caches, receipts,
  screenshots and wheel artifacts remain local and unstaged.

### Exact remaining work / resumption

Use the final classical command above for the scoped lane, and the feature
command for this continuation. Continue R2-04/05/06/07/08/09/10/11 with the
existing owners and the open items in the current package table; do not recreate
the tensor, calibration, statistics, cache or first-round interfaces. In particular,
the following continuation adds the resumable multi-level qualification plan.
Do not translate successful geometric shape fixtures into real probe/wave A1
qualification or reopen G3/G4/G5 silently. No new preset solve is authorized.

### Continued independent work: multi-level numerical checks

The earlier final source-frozen receipt above remains a dated receipt for that
source state, not proof of the following additions. R2-06 now wraps the existing
pair runner in `sampling_qualification.py` and the existing Sampling panel:

- Explicit distinct axes and at least two refinements, finite cumulative wall
  time, attempt-count, per-run ray and checkpoint-memory budgets. Preflight does
  not execute particles. Unsupported axes/method combinations fail explicitly.
- Captured physical-design identity excludes only implemented numerical controls;
  recipe, conditional sampling method, seed policy, implementation and thresholds
  remain separately fixed. Unknown physical inputs are not ignored.
- Atomic bounded scalar JSON at each comparison boundary, exact input graphs
  and parent-pinned external identities. Verified completed pairs are not repeated.
  Cooperative cancellation preserves completed evidence; attempts count against
  the budget. An abrupt RUNNING-process interruption retains evidence read-only
  because elapsed time is unknown. This is not R2-10 general application recovery.
- Shared diagnostics now expose alpha99, physical current and chief slopes for
  the predeclared comparison policy. Under-resolved or unstable evidence remains
  unresolved. The stronger result is only for the declared numerical scope.
- UI controls are `Multi-level check...` and `Resume check...` in the existing
  panel; no separate scientific solver or live-state writer was introduced.

The new tests distinguish analytically prescribed ring transport from one actual
nine-ray tip/gun/column execution. Neither establishes real assembly convergence.
Executed test results follow after the run completes.

Intermediate new-plan receipt `qualification-first.xml`: **15 passed, 4 failed**,
255.11 s. The failures were not waived: two tests used a nonexistent `State.c1`
alias (changed to the real installed lens list); numerical design identity used
the public quadrature name instead of its persisted `_emission_quadrature`
attribute; the asynchronous UI test incorrectly waited for the full numerical
workflow within 10 seconds. It now uses explicit entered/release barriers to
test cancellation/publication independently, while separate tests retain the
actual numerical engine. `_surface_model` received the same canonical-attribute
correction, with explicit mesh/domain identity coverage. The existing grounded
gun outer numerical boundary is now a supported domain axis; it is not a physical
electrode/source-size control.

#### Final multi-level continuation receipt (2026-09-18)

- Final solver source identity:
  `e8268688a7ba6fc3064cfa8f716347220de4b9f5720ce3abc9e9ff15ae3232aa`.
  Both final commands below ran on the unchanged source for this continuation.
  The earlier 259/239 receipts remain valid for their earlier source state;
  they are not represented as full reruns after these additions.
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest tests/test_sampling_qualification.py -o addopts= --tb=short --junitxml=tmp/round2-20260917/qualification-second.xml`:
  exit **0**, **20 passed**, no skips, 286.82 s. Includes analytic scalar
  comparisons, insufficient nine-ray support, unstable axes, method/tolerance/
  source/implementation mismatch, budgets, cancellation/resume, field-mesh/domain
  identity, bounded GUI lifecycle and a separately labelled actual nine-ray
  tip/gun/column execution. That actual plan ends **UNRESOLVED**, as required.
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest tests/test_product_usability.py tests/test_topology_evidence.py tests/test_independent_emission_sampling.py tests/test_working_point_index.py tests/test_phase_space_statistics.py tests/test_held_scan_calibration.py tests/test_stigmator_tensor.py tests/test_stigmator_alignment.py -o addopts= --tb=short --junitxml=tmp/round2-20260917/qualification-integration.xml`:
  exit **0**, **72 passed**, no skips, 241.84 s. Actual CUDA tensor parity ran.
  Existing Pydantic deprecations, deliberate duplicate-archive fixture warning
  and the tiny CUDA fixture's low-occupancy warning remained visible. Counts
  overlap earlier suites and must not be added as distinct acceptance cases.
- Serial post-test `compileall -q src scripts tests` and `git diff --check`:
  exit **0**. HEAD remains `985463968f365fdb945af1b4f3a744a7a53a5912`.
- Built offline wheel under `tmp/round2-20260918/qualification-wheel`:
  **1,953,973 bytes**, SHA-256
  `74692d42556c854d634775ab4d0e500532747c57ed6b8f7360e354fd39dd7baf`.
  `pip wheel --no-index --no-deps --no-build-isolation` exited **0**.
  `pip install --no-index --no-deps --target tmp/round2-20260918/qualification-installed`
  installed only into that new temporary target. Six qualification/GUI/stigmator/
  scan modules imported from the installed target and its own configuration root;
  exit **0**. Existing runtime dependencies were reused, not modified.
- Separate offscreen 640x620 qualification-dialog render completed; minimum
  hint 405x491. Visually inspected the saved PNG: checkboxes, budgets and actions
  are readable. This is not a full native-desktop or populated high-DPI pass.
- No commit, push, application restart, production-default retuning, coherent
  propagation or generated-cache staging. Prior unrelated working-tree edits and
  the user's running application remain untouched.

R2-06 remains **partial**, not blocked by a fabricated success: the separate
detached candidate-search action and representative assembly convergence study
are still open. Other package limits in the table remain unchanged, including
complete loss diagnostics, recovery journaling and full native/hardware workflow
qualification. Do not interpret analytical fixtures or the multi-level workflow's
implementation as physical/OEM or full TEM/STEM image acceptance.
