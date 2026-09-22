# TEM Simulator v2 — Round 2 Usability and Simulation Optimisation

**Document version:** 2.0\
**Prepared:** 17 September 2026\
**Repository:** `MagNetiCLLLL/tem_simulator_v2`\
**Reviewed branch:** `master`\
**Reviewed commit:** `985463968f365fdb945af1b4f3a744a7a53a5912`\
**Commit subject:** `Integrate reproducible working points and bounded design workflows`\
**Previous baseline:** `9ea3a7cc3940d81a956770e6a3785f3dc9994363`\
**Purpose:** Give Codex an executable, incremental backlog for usability, scientific diagnostics, reliability and performance, without duplicating the first development guide.\
**Review method:** Read-only inspection of current repository files and recorded evidence through GitHub. No repository tests, GUI sessions, GPU kernels or benchmarks were executed for this review. Historical receipts below belong to the repository, not to a new independent verification.

> This is a task brief, not a replacement for `AGENTS.md` or `PROJECT_FUNCTION_SPEC.md`. Keep the existing `CODEX_DEVELOPMENT_GUIDE.md` as history. Use the current checkout and reconcile changes since the reviewed commit; do not reset or check out the reviewed commit automatically. All project-facing additions must be in English (UR-027).

## 0. Instruction to Codex

Copy the following instruction into the Codex session for this repository:

```text
Read CODEX_OPTIMIZATION_ROUND2.md completely, all applicable AGENTS.md files,
and the full current PROJECT_FUNCTION_SPEC.md. Read the latest package table,
final receipts and remaining-scope section of
 docs/development/PRODUCT_USABILITY_PROGRESS.md.

Inspect the actual checkout and preserve unrelated work. Reconcile this brief
against newer code and explicit user requirements. Do not recreate the first
round's working-point manager, snapshots, convergence assistant, parameter
registry, resource coordinator, layouts or experiment engine.

Implement ready Round 2 tasks in dependency order, starting with R2-00 and
R2-01, then R2-02 and R2-03. Deliver integrated code and regression tests, not
another roadmap. Continue independent ready work when hardware or a genuinely
missing physical contract blocks a task. Do not wait for routine design choices.

Preserve the single physical emitter-to-gun-to-column chain, complete captured
instrument state, Direct Alignment-only inverse targets, TOML geometry ownership,
opt-in vacuum policy and the pause on coherent tip-to-column development.
Never weaken physics, source-admission rules, tolerances or legacy acceptance
criteria to obtain a passing report. Never label skipped hardware checks PASS.

For each slice, record actual commands, outcomes, limitations and acceptance
identifiers in docs/development/ROUND2_OPTIMIZATION_PROGRESS.md. Keep this new
receipt separate from the historical first-round receipt. Mark tasks VERIFIED
only in the scope actually exercised. Leave exact resumption instructions.

Do not commit, push, open a PR, rewrite history, alter instrument hardware,
restart the user's running application, change production optical/source defaults,
or perform unbounded computations without separate authorization. Any test
process you launch must be accounted for and terminated or completed before
ending the session. Produce a final changed-file and evidence summary.
```

### 0.1 Reading and execution order

1. Applicable `AGENTS.md`; the complete current `PROJECT_FUNCTION_SPEC.md`.
2. `README.md`, `docs/working-points-and-convergence.md`, and the **current** package table plus final sections of `docs/development/PRODUCT_USABILITY_PROGRESS.md`.
3. This brief, then the actual source and tests for the next selected package.
4. Relevant model documentation and the previous guide for context only.

Historical handoff text, an old permission to publish, and old requests to restart/shut down a machine are not current authorization. Dated intermediate failures and superseded “next step” paragraphs are not current backlog entries. Resolve their status against later receipts. [S01–S05]

## 1. Review conclusions and evidence

### 1.1 Preserve these existing implementations

The repository already implements complete working points and portable input archives; lazy archive indexing; illumination-only Apply/undo; frozen A/B comparison; independent sampling axes and crossover checks; immutable input assets; background request preparation; shared FIFO resource admission; CPU/GPU preference and requirement policies; four task layouts; joint condenser and static beam-centre/direction alignment; and detached, resumable parameter/geometry experiments. These are extension points, not missing foundations. Their numerical, hardware and physical acceptance scopes are different. [S03–S05, S08–S16]

### 1.2 Findings that motivate this round

Evidence labels: **CODE** means observable source structure or logic; **RECEIPT** means reported by the repository; **PROPOSAL** means a new improvement, not a demonstrated defect.

| ID | Finding | Evidence and qualification | Consequence |
| --- | --- | --- | --- |
| F01 | Default CI entry and current development scope are misaligned. | **CODE:** `.github/workflows/tem-p0.yml` invokes `validate_development_spec.py --allow-no-gpu`. That validator runs wave-related tests and retains source-migration blockers for legacy AT-12/13/14. Its success predicate permits only missing GPU AT-25/26, not those blockers. The legacy full-acceptance command therefore cannot become successful merely by passing all its listed tests. This is a control-flow observation, not a claim that current Actions logs were inspected. [S17–S18] | Add an honest, separately named classical acceptance lane; preserve the incomplete full-image qualification report. |
| F02 | Preparation-worker startup intermittently exceeds its test window. | **RECEIPT:** the final progress report leaves a five-second startup issue open despite later passing groups and repeated scenarios; its root cause is not established. [S04] | Reproduce and instrument lifecycle transitions before making a fix or changing timing policy. |
| F03 | Job lifecycle uses several implicit worker conventions. | **CODE:** `job_coordinator.py` discovers cancellation attributes, constructs signal prefixes and dispatches failures through worker-dependent signatures. History records lack stage timings. This does not itself prove a race or memory leak. [S08] | Introduce a small explicit adapter and structured transition evidence around existing workers; do not replace Qt or the coordinator. |
| F04 | Some accelerator errors are classified too broadly relative to the stated fallback contract. | **CODE:** `gpu_failure_category()` categorises CuPy `NVRTCError`, `CompileException`, and broad runtime/driver exception classes as retryable `kernel_or_runtime_failure`. This can treat a compilation/programming failure as a CPU-fallback condition. Actual occurrence on this installation was not tested. The active particle path primarily uses Numba; shared classifier coverage is the immediate scope, not resumed wave execution. [S09] | Add narrow, documented recovery classification and regression tests; preserve original error information. |
| F05 | Parameter descriptions are a pilot, not one complete user-facing contract. | **CODE:** `parameter_definition()` returns `None` outside an explicit subset; runtime discovery still uses scalar/attribute and ownership rules. Conservative unknown-input invalidation already exists. [S10–S11] | Expand semantic coverage without replacing validators, breaking ideal design freedom or narrowing cache dependencies. |
| F06 | Numerical comparison exists; comprehensive qualification is still unresolved. | **RECEIPT/CODE:** independent axes and paired runs are present; real small-population checks remain unresolved. Weighted current, N_eff and alpha diagnostics deliberately do not establish physical qualification. [S03–S04, S12–S13] | Add a bounded multi-axis qualification workflow above existing comparisons, with honest stop reasons and evidence scope. |
| F07 | Important alignment limits are actual model limits. | **RECEIPT:** the present condenser stigmator supplies one independent normal-quadrupole response; dynamic pivot/scan-descan contracts and active-vacuum incident-observer validation are unresolved. [S03–S04] | Improve supported alignment diagnosis and transactions now. Keep unsupported axes unavailable; do not invent missing responses. |
| F08 | Real GPU and native desktop acceptance are absent from the recorded final scope. | **RECEIPT:** CPU/emulated-device and offscreen tests are not CUDA scientific parity or native graphics validation. [S04] | Build runnable hardware checks; execute only on an actually detected suitable device. |
| F09 | Remaining performance priorities should follow stage evidence. | **RECEIPT:** the final nine-ray CPU workflow records a 22.228 s initial run, including 11.555 s in the existing energy-filter stage; exact reuse median is 0.704 s, and P2 edits median 12.682 s. These small-case local results are not large-workload predictions or causal speed-up evidence. [S04] | Profile cold/warm stages, invalidation and populated displays. Do not simply add another cache or bypass the filter. |
| F10 | Users need clearer answers to “what can run?”, “why no signal?” and “which settings produced this?”. | **PROPOSAL** building on existing layouts, readouts, result identity, Model Inspector and aperture/interaction records. [S03, S06–S07, S12–S14] | Integrate guided task actions and diagnostic explanations into existing views, not another parallel interface. |

### 1.3 Product objective

Make the following workflow understandable, responsive and reproducible:

```text
Choose or restore a complete instrument
  -> see current capability and evidence limits
  -> preview from the physical emitter
  -> identify beam losses and numerical limitations
  -> change real controls or run a supported Direct Alignment
  -> inspect affected products and run the appropriate calculation
  -> qualify the selected numerical scope within a declared budget
  -> compare frozen results and save an exact working point/experiment
```

The goal is not to maximise the number of enabled tabs. A disabled feature with an accurate explanation is preferable to a plausible-looking but physically unsupported output.

## 2. Invariants and scope boundaries

| ID | Requirement |
| --- | --- |
| R2-INV-01 | Illumination begins at the modelled physical emitter, including the FEG tip where applicable, and traverses extraction, acceleration, focusing, apertures and column optics. No independent gun-exit, specimen-plane, accelerated or downstream illumination source. |
| R2-INV-02 | An upstream checkpoint is reusable executed data whose identity includes all consumed inputs. A digest or label alone is not transport. Reuse does not allow editing the checkpoint into a new source. |
| R2-INV-03 | Save complete gun, lens, aperture, deflector, stigmator, corrector, specimen, detector and disabled-component state, structural/field identities, numerical settings, seeds and provenance. Never reduce a working point to a few derived beam metrics. |
| R2-INV-04 | Derived quantities remain readable. Supported inverse targets may change physical controls only through Direct Alignment and its existing forward-validation/commit gate. Ordinary physical-control editing and undo must not become another inverse solver. |
| R2-INV-05 | Coherent tip-to-column development remains paused. Preserve historical code and data. No production TEM/STEM phase-image acceptance, synthetic phase, bypassed source or revived long wave run. Shared utility tests may use isolated fixtures without reopening this work. |
| R2-INV-06 | TOML is structural authority. Geometry exploration remains detached unless explicitly applied. Never optimise lens presets implicitly because geometry was edited. |
| R2-INV-07 | Ideal continuous controls stay continuous. Numerical validity bounds are not OEM hardware limits. Optional engineering-constraint models must be explicit; do not silently clamp/snap controls or invent calibration. |
| R2-INV-08 | Vacuum participation remains opt-in and disabled by default; preserve saved choices. Broad invalidation for vacuum changes is allowed. An unsupported observer must reject the request rather than disable vacuum. |
| R2-INV-09 | Preserve absolute weights/current, absorption, aperture losses and detector interception. Deselected readouts do not remove physical sinks. Do not renormalise survivors to manufacture transmission or an alignment target. |
| R2-INV-10 | Do not bypass a requested downstream energy-filter path. A detector physically upstream of the filter does not traverse it. EDS remains off-axis X-ray collection, not an axial electron stop. |
| R2-INV-11 | Current editable state, captured request state, retained result state and presentation state remain distinct. Preview/Medium must not overwrite completed High results. Independent studies must not overwrite the active instrument. |
| R2-INV-12 | Execution success, numerical evidence, physical/model qualification and hardware verification are separate. A preset name, a CPU fixture or an unresolved small-ray run must not become “validated”. |
| R2-INV-13 | Keep existing source-admission, compatibility, corruption and conservative dependency guards. Unknown consumed inputs invalidate affected reuse. Cache boundaries follow field support and shared circuits, not just element centres. |
| R2-INV-14 | Preserve English UI/documentation, stable requirement IDs and historical readability. Add new requirement IDs through the existing ledger process; never recycle legacy IDs. |
| R2-INV-15 | Do not change production optical/source defaults, dependency versions, cache budgets or global user preferences merely to obtain a demonstration or benchmark improvement. |
| R2-INV-16 | No commits, pushes, PRs, live hardware operations, destructive history changes, user-application restarts or generated-array commits are authorised by this brief. Lightweight code, input definitions and evidence summaries are the deliverables. |

## 3. Execution map

Use statuses `NOT_STARTED`, `IN_PROGRESS`, `IMPLEMENTED_UNVERIFIED`, `VERIFIED_SCOPED`, `BLOCKED_HARDWARE`, and `BLOCKED_PHYSICAL_CONTRACT`. “Verified” always names the exact scope. A feature implemented after the reviewed commit should be inspected and reused, not implemented twice.

| Package | Priority | Main outcome | Dependencies |
| --- | --- | --- | --- |
| R2-00 | P0 | Fresh baseline, instructions and evidence inventory | None |
| R2-01 | P0 | Scope-aware acceptance, unambiguous criterion identities | R2-00 |
| R2-02 | P0 | Reproducible job lifecycle and startup diagnosis | R2-00 |
| R2-03 | P0 | Honest accelerator failure and capability reporting | R2-00 |
| R2-04 | P1 | Task-oriented navigation and recoverable editing | R2-01/02 |
| R2-05 | P1 | Consistent parameter units, meaning and actual effects | R2-00; coordinate with R2-04 |
| R2-06 | P1 | Bounded multi-axis numerical qualification | R2-01/02/05 |
| R2-07 | P1 | Beam-loss and phase-space diagnostics | R2-05; reuse R2-06 evidence |
| R2-08 | P1 | Measured stage performance and real-GPU harness | R2-02/03; hardware-dependent execution |
| R2-09 | P1 | Supported alignment diagnosis and robust experiments | R2-05/06 |
| R2-10 | P2 | Recovery and archive lifecycle hardening | R2-02; reuse existing archive contracts |
| R2-11 | P1 integration | End-to-end classical acceptance and handoff | Completed ready packages |

P2 work may proceed when higher-priority work is blocked. R2-11 tests should grow with each package, not wait for a final large integration.

## 4. Work packages

### R2-00 — Establish a fresh, bounded baseline

**Inspect:** `AGENTS.md`, `PROJECT_FUNCTION_SPEC.md`, `pyproject.toml`, current guide/receipts, the actual working tree and relevant tests. [S01–S05, S16]

**Implement/do:**
- Record branch, HEAD, dirty paths, interpreter, relevant package versions, OS, CPU, RAM and detected accelerators. Do not assume the review host or the user's other workstation is the execution host.
- Inventory what each Round 1 package already delivers. Map each selected Round 2 task to source paths, current tests and a requirement ID before editing.
- Run a bounded classical baseline. Distinguish existing failures, new failures and unexecuted checks; retain complete logs locally. Do not invoke the legacy all-development validator as an indiscriminate baseline.
- Create `docs/development/ROUND2_OPTIMIZATION_PROGRESS.md` with a package table, environment, baseline results, blockers and resumption section.

**Acceptance:** R2-AT-01/02. No baseline operation modifies active settings or unrelated files. A failed baseline remains visible; it is not deleted from the final report.

### R2-01 — Separate current classical acceptance from full-image qualification

**Inspect:** `.github/workflows/tem-p0.yml`, `scripts/validate_development_spec.py`, the first-round acceptance map and existing validation tests. [S04, S17–S18]

**Implement:**
- Provide an explicitly named classical-particle acceptance entry point or an additive, validated `--scope` mode. The entry must select bounded applicable tests and record selected/excluded criteria with reasons.
- Preserve the old full-development/full-image acceptance semantics and source-migration blockers. Do not make legacy AT-12/13/14 pass, skip them silently, or redefine `--allow-no-gpu` to waive physics.
- Use composite identifiers such as `legacy-development/AT-12`, `product-usability/AT-12`, and `round2/R2-AT-12`; existing guides use overlapping AT numbers for different meanings. Reports and tooling must not merge them by suffix alone.
- Make the ordinary current-scope CI check explicit. Keep a separate full-scope report/manual lane showing `BLOCKED`, `NOT_RUN` or failure accurately, without launching paused long wave work by default.
- Separate `software_scope_status` from `full_simulator_qualification`. A classical check can succeed while the full simulator remains unqualified. Missing hardware can be reported as a scoped exclusion, never GPU PASS.
- Add tests for report aggregation, missing tests, skipped tests, scope omissions, source changes during execution and exit-code policy.

**Acceptance:** R2-AT-03/04/05. Intentionally fail one classical test: CI must fail. Run a fully passing synthetic classical receipt: only that scope passes. Legacy blockers remain unchanged and visible.

### R2-02 — Make job ownership, progress and cancellation deterministic

**Inspect:** `gui/job_coordinator.py`, `gui/calculation_controller.py`, `gui/calculation_request.py`, `gui/main_window.py`, controller owners and dispatch-related tests. [S06–S08]

**Implement:**
- Add structured events for capture, preparation, enqueue, admission, worker entry, stage entry/exit, publication, cancellation request, cleanup and terminal state. Include job/request identity, owner, generation, monotonic timing, backend and resource claim; logs must not embed large arrays.
- Introduce an explicit small worker adapter/protocol for cancellation, identity, outcome and resource claim. Migrate touched workers incrementally while retaining existing Qt workers, controller ownership and FIFO policy. Do not do a framework rewrite.
- Guarantee exactly one terminal outcome per job and release reservations/pins exactly once on success, rejection, cancellation, error and window closure. Distinguish worker entry from “marked running before dispatch”.
- Reproduce F02 using independently parameterised scenarios: cold imports, warm execution, rapid edits, cancellation during preparation, a queued High request and closure. Capture thread stacks and outstanding owners on an actual timeout.
- Investigate event-loop delivery, thread affinity, queued signals, wrapper lifetime and scoped library setup as hypotheses, not presumed causes. Fix only what reproduction or rigorous invariants support.
- Preserve latest-pending live work, retained completed High results and independent experiment ownership. Acknowledging cancellation must not publish partial scientific results as completed.

**Acceptance:** R2-AT-06/07/08/09. Use deterministic barriers for transition tests and a separately labelled repeated stress run. Do not obtain green tests merely by multiplying the five-second timeout. Distinguish a justified revised timing policy from a proven race fix.

### R2-03 — Tighten backend failure semantics and expose per-stage capability

**Inspect:** `physics/compute_backend.py`, `physics/ray_device_cache.py`, call sites for `gpu_retry_reason`, `test_ray_gpu_policy.py` and `test_ray_device_residency.py`. [S09]

**Implement:**
- Replace blanket exception-class fallback with an explicit recovery taxonomy based on trustworthy error identity/code and current stage. Preserve exception type, message, traceback/cause and attempted backend in diagnostic evidence.
- Confirmed device absence and resource exhaustion may support a declared CPU retry for `Prefer GPU`. Invalid inputs, programming/compilation errors, unsupported physical models, nonfinite scientific outputs, I/O corruption and cancellation must not be disguised as GPU unavailability.
- Treat unknown accelerator failures conservatively. Context corruption or illegal access requires explicit cleanup/recovery semantics, not reusing suspect cached device state. `Require GPU` must never publish an implicit CPU result.
- Test the existing CuPy classifier branch with isolated exceptions/fixtures without executing paused wave calculations. Ensure a compilation error cannot turn into a successful preferred-backend result merely because the CPU branch works.
- Present requested versus actual backend **per stage**. A GPU column does not imply a GPU gun, mapped-field solver, specimen or filter. Report CPU-only requirements before a costly run where knowable.
- Keep legacy backend names readable, but validate new explicit selections and clarify the difference between `CUDA GPU`, `Prefer GPU` and `Require GPU`; do not silently migrate saved choices.

**Acceptance:** R2-AT-10/11/12. Required scenarios include CPU-only host, accelerator OOM, compile failure, invalid input, cancellation, context reset and mixed CPU/GPU stages. Hardware absence remains `NOT_RUN` for scientific parity.

### R2-04 — Improve the user journey without adding another interface

**Inspect:** `gui/main_window.py`, `gui/working_point_panel.py`, existing layouts, compact readouts, parameter widgets and dialogs. [S03, S06]

**Implement:**
- Add a compact task summary to existing Instrument/Alignment/Experiments/Results layouts: selected assembly, source, live revision, result identity, stale/current status, stage capability and next applicable action.
- Make “Open/restore a working point”, “Run Preview”, “Inspect beam losses”, “Supported Direct Alignment”, “Numerical qualification”, and “Save/compare result” reachable using existing controllers. Do not duplicate the Working Points or Design Explorer implementations.
- Explain unavailable actions before execution: the exact unsupported scope, missing numerical prerequisite or absent hardware, and the relevant existing settings/evidence view. Never activate an unsupported feature simply to avoid a disabled button.
- Extend current physical-control editing with bounded, transactional undo/redo where absent. Coalesce a continuous drag into one change. Preserve non-target state and run the same validators/invalidation as a normal edit. Derived optical targets still go through Direct Alignment only.
- Display captured versus current values together when a result is stale. Clicking a retained result must not restore it, solve optics or overwrite current controls.
- Validate practical small-window and scaled layouts, including 1280×720 and 1366×768 where feasible, plus 100/125/150/200% scaling. Use scrolling and presentation changes, not tiny controls or hidden physics.
- Add concise tooltips/keyboard access, error details that can be copied, and explicit Apply/Cancel for multi-field transactions. Preserve current English terminology and saved layouts.

**Acceptance:** R2-AT-13/14/15/16. Switching tabs/layouts, viewing diagnostics and comparing results causes no solver dispatch or physical mutation. Undo reverses one logical edit and marks dependent results stale correctly.

### R2-05 — Make parameters understandable and physically accountable

**Inspect:** `parameter_registry.py`, `runtime_parameters.py`, `parameter_semantics.py`, existing control/calibration adapters and structural editors. [S10–S11, S15]

**Implement:**
- Extend the existing registry first for frequently edited gun, condenser, aperture, deflector, specimen and detector parameters. Each descriptor should state unit, category, active condition, authority, validation source, numerical bounds, optional hardware bounds and affected products.
- Explicitly distinguish operating controls, structural values, derived readouts, execution settings and display-only values. Show why a stored parameter is inactive rather than presenting it as an effective control.
- Show the physical calibration of lens excitation where known: current, ampere-turns or an explicitly model-relative excitation. Unknown coil turns or calibration must stay unknown; never relabel a percentage as amperes.
- Explain electrode potential references using the **implemented** model: electrode-to-ground and electrode-to-tip quantities must not be conflated. Display exact sign/reference conventions; do not invent a commercial gun circuit.
- Audit round trips between units, UI, snapshots, profiles, sweeps and solver inputs. Test that intended physical changes affect actual fields/transport or produce an explicit symmetry/inactive/unsupported explanation.
- Audit legacy 0–100% and similar bounds against ideal-mode requirements. Where an accidental UI bound is identified, change the owning validation path consistently with finite numerical guards; do not simply widen a spinbox while the model still clips values.
- Retain conservative invalidation for unclassified inputs. Expanding descriptions must not grant cache reuse without dependency evidence.

**Acceptance:** R2-AT-17/18/19. Include unit round trips, polarity/reference tests, inactive controls, continuous apertures, unknown calibration, ideal-mode range behaviour and real parameter-to-transport effects. Do not require every observable to change for a symmetric input perturbation.

### R2-06 — Turn paired comparisons into a bounded qualification workflow

**Inspect:** `sampling_convergence.py`, `sampling_diagnostics.py`, `gui/sampling_panel.py`, topology evidence and existing independent-emission tests. Reuse their executed transport and exact checkpoints. [S03–S04, S12–S13]

**Implement:**
- Add a resumable qualification plan around the existing comparison engine. Select applicable axes explicitly: source position, direction and conditional energy sampling; gun/column steps; supported active gun/field meshes; domain extent where the existing model exposes it; and crossover observation spacing.
- Set allowed refinements, memory, evaluation count and wall-time budgets before execution. Stop on exhausted budget, invalid input, unsupported scope or unresolved evidence; do not keep doubling until the machine exhausts resources.
- Keep physical design identity separate from numerical-recipe identity. Every comparison records both actual snapshots and exactly which axis changed. Do not combine results from different geometry, physics, source law, seed policy, tolerance policy or implementation into a single qualification.
- Preserve conditional source-energy sampling and emitted-population weights. If a quadrature method changes as well as a count, report a method change instead of pretending it is a pure one-axis refinement.
- Compare physically relevant observables at exact, named planes: weighted current/transmission, centroids, direction, D95, alpha95/99 and supported topology. Require multiple refinement levels for a stronger numerical conclusion than the existing two-setting agreement, with predeclared tolerances and appropriate absolute floors.
- Retain `STABLE_FOR_CHECKED_AXIS` as scoped evidence. Use a conservative summary such as `NUMERICALLY_CHECKED_FOR_DECLARED_SCOPE` only when all declared prerequisites and comparisons pass; never translate this into experimental/OEM validation.
- Add a candidate-search action that uses existing Design Explorer/alignment services and saves **new detached input candidates**. Do not replace production presets or manufacture a qualified Microprobe/Nanoprobe point. Preserve the distinct topology references and observation-plane definitions.
- Store small scalar receipts and provenance; release paths between comparisons. Resume only compatible completed work. A cancellation leaves prior completed evidence intact.

**Acceptance:** R2-AT-20/21/22/23. Include under-resolved nine-ray evidence, one unstable axis, incompatible inputs, changed tolerances, exhausted budget, conditional quadrature, cancellation/resume and a numerically checked analytical fixture. A fixture is not assembly qualification.

### R2-07 — Explain beam transport, losses and phase space

**Inspect:** `physics/beam_statistics.py`, `sampling_diagnostics.py`, existing selected-plane views, `simulation_pipeline.py` aperture/interaction records, and detector masks. [S12–S13, S03]

**Implement:**
- Add “Why no signal?” and “Where is current lost?” views over existing executed records. Separate: no retained result; no sampled survivors; an established physical interception in this run; a disabled readout; incompatible detector path; numerical failure; and unsupported computation. Do not infer complete physical blockage solely from zero sampled survivors.
- Report ordered component/plane loss summaries using physical weights and source normalisation. Avoid double-counting branches, intermediate monitors and downstream sinks. Identify the first **observed** loss interval; do not name an exact component when checkpoint resolution cannot establish it.
- Overlay aperture opening, beam centroid and footprint using existing coordinate/rotation conventions. Offer a navigation action to the relevant component and its controls, not an automatic opening/retuning action.
- Extend the shared beam-statistics service, where missing, with weighted phase-space covariance, X–slope-X / Y–slope-Y displays, principal spatial axes and projected geometric emittance. Do not build another statistics implementation in the GUI.
- Define coordinates explicitly. For positive-weight surviving rays, use `p_i = w_i / sum(w_survivors)` only to form moments; preserve the original transmitted current separately. For `q = (x, tx, y, ty)`, define `Sigma = sum p_i (q_i - mean_q)(q_i - mean_q)^T` and projected `epsilon_x = sqrt(Sigma_xx * Sigma_txtx - Sigma_xtx^2)`, with the corresponding Y quantity. Retain the actual slope/angle convention; label paraxial slope-based emittance accordingly.
- Treat these as diagnostics of the sampled distribution. Do not claim projected geometric emittance is invariant under aperture clipping, coupling, acceleration or arbitrary magnetic-field conditions. Do not turn geometric ray spot size into diffraction-limited resolution.
- Keep alpha95/99 as angular current-containment quantiles around the weighted chief direction; they are not aperture-edge angles. Label the existing waist offset as a local free-space extrapolation, not an executed focus through intervening lenses.
- Audit consistent treatment of zero-weight and nonfinite rays across `sampling_summary()` and direct callers of `transverse_beam_statistics()`. Report numerical invalidity where applicable instead of silently presenting a healthy beam after removing bad rays. Preserve explicit historical semantics through adapters where needed.

**Acceptance:** R2-AT-24/25/26/27. Test unequal weights, clipped populations, zero weights, nonfinite coordinates, rotations, off-axis beams, coupled covariances and degenerate populations. Read-only diagnostics must not launch a source or propagate new independently defined illumination.

### R2-08 — Optimise measured costs and verify the actual GPU path

**Inspect:** `gui/calculation_controller.py`, `gui/calculation_request.py`, `gui/job_coordinator.py`, product signatures, input assets, device cache, existing benchmark scripts and the final workflow receipt. [S04, S07–S09]

**Implement:**
- Extend existing measurements with cold/warm stage timing, request capture/preparation, field construction, gun integration, column transport, specimen/EDS/filter products, result transfer, display update, archive I/O and cancellation latency. Separate nested timers; never add them as independent elapsed time.
- Record input/implementation identity, numerical budgets, model participation, backend, hardware, repetitions and uncertainty. Compare identical workloads; compilation, disk caches and workstation load are confounders.
- Profile the current expensive energy-filter/downstream path before changing it. Reuse field/transport subproducts only when their consumed inputs and upstream distribution match; P2 changes may legitimately invalidate filter input. Do not bypass filter physics or substitute old output with a new label.
- Measure GUI-thread snapshot/retained-memory inventory cost with representative input assets. Memoise immutable descriptors and stable ownership accounting only after proving mutation and alias handling remain correct. Avoid repeated full graph/array scans for every progress repaint.
- Improve resource estimates for actual worker classes rather than assuming every small preparation request needs a full solver reservation. Keep accounting for queued pinned inputs, active temporaries, shared retained arrays and device resources; make underestimated or rejected claims visible.
- Reuse the existing device plan/buffer cache and observed-cost Auto selection. Optimise residency or transfers only where traces show a cost. Do not introduce another persistent GPU cache or benchmark automatically on every launch.
- Add a real-GPU harness for the active classical stages. Compare CPU/GPU coordinates, weights, alive/stop outcomes, currents and beam metrics for the same captured physical source chain. Cover apertures, polarity, energy spread, nontrivial deflection, cache reuse and context reset within supported model scope.
- Declare comparison tolerances before execution. Near-aperture-boundary disagreement must be reported and investigated/refined, not silently discarded. Record device-to-host and host-to-device bytes and synchronised timing, plus observed device allocation peaks where available.
- A machine without a compatible GPU should run policy/ownership fixtures and emit `BLOCKED_HARDWARE` for hardware acceptance. Do not install or replace CUDA/toolkit/driver packages speculatively.

**Acceptance:** R2-AT-28/29/30/31. A performance improvement is accepted only alongside invariant/numerical checks. No minimum speed-up is promised; “no measured benefit” is a valid finding.

### R2-09 — Improve supported alignment and experiment decisions

**Inspect:** `gui/direct_alignment_controller.py`, `alignment_transaction.py`, current constrained alignment services, Design Explorer and experiment records. [S03–S04]

**Implement:**
- Show target definitions, units, observation plane, allowed physical controls, weights, numerical bounds, evaluation budget, rank/conditioning, active constraints and final residuals before Apply.
- Reuse the existing physical forward model. Scale residuals and control steps explicitly so metres, milliradians and current do not become accidentally incomparable numerical objectives.
- Improve failure explanations: under-resolved observation, zero sampled transmission, ill-conditioned response, control bound, exhausted budget, unsupported model or failed forward verification. Keep the previous instrument/result intact.
- Publish before/after results and exact solved controls as detached evidence. Warm starts may reuse compatible full working points; they may not bypass source transport or invent a target pupil.
- Retain separate optimisation and validation runs. Existing gun/column step checks are not source-sampling or field-mesh qualification; link R2-06 evidence instead of overstating the existing gate.
- Extend the existing experiment plots with evidence filters and explicit robustness summaries over declared perturbations. Preserve invalid/failed draws in denominators and tables; do not rank only survivors and call the result robust.
- Keep signed finite differences, parameter scale and neighbouring executed points visible. A local sensitivity is not a global optimum or an experimental noise estimate.
- Do not enable arbitrary two-axis stigmation, dynamic pivot/scan-descan matching or active-vacuum observer validation under this package. Implement clear capability boundaries and the gated contracts in Section 5.

**Acceptance:** R2-AT-32/33/34. Include rank-deficient response, incompatible/stale result, cancelled solve, infeasible target and failed independent check. None may partially mutate the instrument. Candidate selection alone never applies it.

### R2-10 — Strengthen recovery and archive lifecycle behaviour

**Inspect:** existing `.temwp`/`.temexp` readers/writers, input archive resolver, asset ownership, migration and export tests. Reuse the existing formats and checksums. [S03–S04]

**Implement:**
- Verify file-size/array-shape/decompression bounds, atomic publication and corruption reporting across both working-point and experiment paths. Fix demonstrated gaps; do not create another format or serializer solely for this round.
- Add interrupted-write tests using isolated temporary files/processes owned by the test. An interrupted export must leave either the previous valid archive or no completed replacement, never an apparently valid partial file.
- Provide a small optional recovery journal for unsaved physical-input edits and interrupted experiments where absent. Recovery should offer a detached candidate/record for inspection, never automatically restore optics or rerun a calculation on startup.
- Make “input-only”, “metadata-only”, “retained result”, “portable inputs” and “historical incompatible implementation” unmistakable. Only a verified complete input archive may claim portable recomputation.
- Exercise pinned input ownership through multiple open/compare/export/close cycles and cache-budget changes. Releasing an archive must not invalidate an active request or leak its assets indefinitely.
- Extend existing manifest checks to any new diagnostics/qualification attachments and ensure their input, definition and implementation identities remain bound to the parent evidence.

**Acceptance:** R2-AT-35/36/37. Use temporary copied dependencies for relocation tests; never move/delete original user files. Metadata-only records remain non-executable. Recovery does not imply completion of an interrupted result.

### R2-11 — Integrate, verify and deliver an honest current-scope result

**Inspect:** all changed modules and the new acceptance mapping; preserve the requirements ledger and prior receipts. [S01–S05, S16–S18]

**Implement/do:**
- Exercise a complete classical workflow through actual existing controllers: start with isolated settings; choose/restore a compatible point; Preview from the physical emitter; edit a real control; inspect losses; run supported alignment; compare numerical evidence; save/export/relocate/restore; compare frozen A/B; and cancel/resume a detached experiment.
- Maintain separate tests for pure functions, controller transactions, actual bounded transport, offscreen GUI, installed-package import/launch and hardware checks. Mocks must not be used to declare end-to-end physical acceptance.
- Check small-window layouts and measured event-loop responsiveness with a populated representative scene, not only an empty MainWindow. Native desktop verification remains explicitly unexecuted until actually performed.
- Update user workflow documentation, README links, requirement statuses and the Round 2 receipt. Retain old statuses as historical records, with superseding evidence identified.
- Build and inspect the wheel using the current environment where available; use an isolated install only when dependencies can be satisfied without modifying global environments. Match installed-source identity to the checkout used by the tests.
- Produce a final package-by-package report of changes, tests, numerical results, unmeasured costs, blockers and exact next command. Do not label the entire simulator “complete” because the current classical software scope passes.

**Acceptance:** R2-AT-38/39/40. All changes have a relevant executed test or an explicit unverified explanation. Generated caches/arrays, unrelated user modifications and original input dependencies remain untouched.

## 5. Gated physics work — do not silently implement a substitute

These gates do not block independent usability or classical-diagnostic work. For each gate, record missing information and the smallest necessary design decision in the final receipt. Do not stop the whole session to ask about an unrelated gate.

### G1. Arbitrary two-axis condenser stigmation

The recorded model has only one independent normal-quadrupole response. A second UI slider or a rotated display ellipse is not a second physical response. Before implementation, require an explicit normal/skew field model or supported coil geometry, axial support, strength convention, calibration/ideal-model status, allowed controls and validation observables. Do not assert this missing response is an OEM calibration. Current work may improve rank detection and disable/explain unsupported tasks. [S03–S04]

### G2. Dynamic pivot and scan/descan alignment

Require the time-dependent excitation law, scan clock, physical control locations, target plane(s), centre/direction/conjugacy objective, reference coordinates, residual definition and permitted control set. Reuse the existing dynamic source-to-column path if a later approved contract enables this. Do not call a static beam-centre solve “dynamic pivot alignment”.

### G3. Active-vacuum numerical/alignment observer

Do not disable the saved vacuum choice to make the existing observer run. Extending the observer requires using the actual modelled vacuum path, declaring any stochastic/quadrature treatment, defining reproducible comparisons and validating loss/current accounting. Until that design is explicitly supported, report this scope as unsupported while retaining the main simulator's existing vacuum capability. [S01, S03–S04]

### G4. Coherent TEM/STEM imaging, phase and analytical qualification

Coherent tip-to-column development remains paused. Do not create replacement incident waves, resume long propagation, infer phase from ray intensities, claim atom-resolution HAADF/EELS/4D-STEM qualification, or open production image acceptance. Retain historical data and utility tests within existing boundaries. A future user instruction can reopen this work with a separate physics-and-validation plan; this brief does not do so. [S01–S03]

### G5. Experimental calibration and commercial-instrument fidelity

Geometry reconstructions, analytic fields, finite-pupil fits and self-consistent numerical checks are not experimental calibration. New calibration claims require traceable measurement/reference data, units, uncertainty, domain of validity and an independent validation comparison. Never fill unknown instrument specifications with plausible values presented as fact. [S02, S15]

## 6. Acceptance matrix

All identifiers below are in the **round2** namespace. Existing AT identifiers remain attached to their original guide/scope.

| ID | Required scenario | Pass condition / evidence |
| --- | --- | --- |
| R2-AT-01 | Baseline and instruction reconciliation | Current HEAD/environment/dirty paths and actual bounded baseline recorded; no reset or unrelated edits. |
| R2-AT-02 | Preserved core contracts | Source admission, complete state, Direct Alignment-only inverse writes, vacuum choice and paused-wave boundaries retain regression coverage. |
| R2-AT-03 | Classical acceptance selection | Explicit allowlisted current scope; unknown/missing/skipped tests cannot become PASS. |
| R2-AT-04 | Legacy qualification unchanged | Legacy source/wave blockers remain BLOCKED; no waiver added to manufacture full acceptance. |
| R2-AT-05 | Namespaced evidence/exit status | Different AT-12 definitions do not collide; an actual classical failure produces nonzero current-scope exit. |
| R2-AT-06 | Cold/warm job dispatch | Worker-entry timing and all ownership transitions captured; no unexplained terminal state. |
| R2-AT-07 | Rapid edits during preparation | Only valid latest live request publishes; captured inputs remain immutable. |
| R2-AT-08 | Cancel/reject/fail/close lifecycle | Exactly one terminal event; reservations and owned pins released exactly once; no partial completed result. |
| R2-AT-09 | High and detached work ownership | Preview cannot overwrite High; independent experiments do not overwrite active instrument/results. |
| R2-AT-10 | Backend unavailable/OOM | Prefer policy gives an explicit eligible fallback; Require policy never silently computes on CPU. |
| R2-AT-11 | Compile/input/model/cancellation failures | Original failure is preserved; no successful CPU retry masks it. |
| R2-AT-12 | Mixed backend/context reset | Per-stage actual backend is recorded; context-reset state cannot reuse invalid device buffers. |
| R2-AT-13 | Navigation, result inspection and layouts | No physical mutation or physics dispatch from read-only actions. |
| R2-AT-14 | Logical undo/redo | Coalesced edit reverses exactly, validates normally, and invalidates correct products. |
| R2-AT-15 | Small/high-DPI workflow | Primary actions remain reachable; scrolling/focus works; layout changes preserve results and state. |
| R2-AT-16 | Current versus captured result | Stale results show their own settings/plane/identity, not current values under an old image. |
| R2-AT-17 | Unit and potential-reference round trips | UI/profile/snapshot/sweep/solver agree on units, signs, reference and numerical value. |
| R2-AT-18 | Continuous ideal controls | No unintended clipping/quantisation; numerical limits and optional physical limits are distinct. |
| R2-AT-19 | Control-to-physics dependency | Actual supported effect or explicit inactive/symmetry boundary; unknown inputs cannot grant stale reuse. |
| R2-AT-20 | Incomplete/unstable qualification | Under-resolved or unstable axes keep the overall declared numerical scope unresolved. |
| R2-AT-21 | Identity-bound comparison | Physical-input/method/tolerance mismatches cannot be combined into qualification. |
| R2-AT-22 | Refinement and budget | Applicable axes and real comparisons recorded; budget exhaustion stops cleanly without changing tolerances. |
| R2-AT-23 | Qualification cancel/resume | Completed scalar evidence retained; only compatible unfinished comparisons resume. |
| R2-AT-24 | Weighted loss accounting | No count-as-current substitution, survivor renormalisation or double-counted branches/sinks. |
| R2-AT-25 | No sampled signal / no exact plane | Insufficient evidence remains unavailable/unresolved, not a fabricated zero-size focus or proven blockage. |
| R2-AT-26 | Beam geometry and phase-space diagnostics | Weighted known fixtures/rotations/degeneracies pass with explicit definitions and validity limits. |
| R2-AT-27 | Detector/readout/phase distinction | Unread physical detectors still intercept; ray diagnostics do not produce fake coherent phase/resolution. |
| R2-AT-28 | Performance comparability | Same inputs/model/numerics, separated cold/warm stages, reproducible receipt; no bypassed physics. |
| R2-AT-29 | Memory accounting | Aliases counted once; queued/active pins remain owned; rejected jobs clean up; estimate versus RSS is clear. |
| R2-AT-30 | Actual CPU/GPU scientific parity | Actual hardware, supported stages, predeclared tolerances and coordinate/weight/stop/metric evidence; otherwise BLOCKED_HARDWARE. |
| R2-AT-31 | Transfers/device ownership | Actual transfer/allocation measurements on hardware; fresh host results and bounded device retention. |
| R2-AT-32 | Supported alignment validation | Targets/planes/controls/rank/residuals and independent checks attached to candidate before Apply. |
| R2-AT-33 | Infeasible/stale/cancelled alignment | Transaction fails without partial physical changes or replacement of valid retained results. |
| R2-AT-34 | Experiment evidence and failed draws | Failed/invalid points retained; local sensitivities and robustness summaries do not imply global/calibrated validity. |
| R2-AT-35 | Interrupted/corrupt archive | No apparently complete partial replacement; checksum/schema/size/path checks preserve source files. |
| R2-AT-36 | Metadata/portable/historical modes | Metadata never executes; portability requires complete verified inputs; historical results remain inspectable. |
| R2-AT-37 | Recovery and repeated archive lifecycle | Recovery remains detached; active request assets survive close/eviction; no orphan ownership. |
| R2-AT-38 | Actual bounded classical workflow | GUI/controllers plus real emitter/gun/column transport exercised, separate from mocked fixtures. |
| R2-AT-39 | Build/install/native boundaries | Exact built-source identity and scoped smoke evidence; unrun native/GPU checks explicitly marked. |
| R2-AT-40 | Deliverable and repository hygiene | Changed files, commands, results, exclusions and resumption recorded; no generated arrays committed or user work lost. |

## 7. Implementation and data-contract rules

### 7.1 Extend existing owners

Keep physical state/assembly loaders, runtime setters, snapshot archives, calculation signatures, controllers, the JobCoordinator, shared beam statistics, alignment transactions and experiment records as their respective owners. Add thin adapters or orchestration where a package needs a common interface. Do not create a second `State`, cache manager, simulation engine, unit converter or archive format without a demonstrated incompatibility in the existing one.

Keep numerical/diagnostic services independent of widget presentation. GUI handlers should request existing services and display their results, not contain duplicate physics or recompute scientific metrics from downsampled drawing arrays. New dependency decisions must be tested against actual stage identities and fields with overlapping axial support.

### 7.2 Evidence identity

Prefer extending the existing manifests rather than introducing a parallel database. A new evidence attachment should preserve the following information, mapped to existing field names where available:

```json
{
  "schema": "round2-evidence-v1",
  "scope": "classical-particle/numerical-comparison",
  "criterion_ids": ["round2/R2-AT-21"],
  "design_snapshot_id": "<captured-physical-input-identity>",
  "numerical_recipe_id": "<numerical-input-identity>",
  "execution_id": "<actual-run-identity>",
  "implementation_id": "<tested-source-identity>",
  "definition_id": "<observable-and-plane-definition-version>",
  "parent_evidence_ids": [],
  "varied_axes": ["<declared-axis>"],
  "requested_backend": "<policy>",
  "actual_stage_backends": {},
  "status": "NOT_RUN",
  "tolerance_policy_id": "<predeclared-policy>",
  "command": [],
  "measurements": {},
  "limitations": [],
  "artifacts": []
}
```

This is a **proposed attachment contract**, not a claim that these exact API names exist. Never populate unknown identities or numerical measurements with fabricated values. Implementation identity alone does not establish compatibility of external assets. A read-only historical scalar record is not verified input or result evidence.

### 7.3 Performance targets are design objectives, not claimed measurements

For a declared representative fixture, aim for visible click/cancellation acknowledgement within roughly 100–200 ms, ordinary warm metadata operations without perceptible freezing, and GUI-thread work short enough to keep input processing responsive. Where a numeric gate is introduced, document its fixture, hardware, sampling method and statistical interpretation before testing. Do not apply these values as universal guarantees or use them to remove physics.

Record actual main-thread capture duration, event-loop stalls, cold/warm calculation latency, cancellation-at-safe-boundary latency, peak sampled RSS and device allocation evidence separately. A worker can acknowledge cancellation promptly while a noninterruptible kernel finishes at its next safe boundary; show that distinction. Device and process memory estimates are not guaranteed OS/driver caps.

Use matching workloads and several repetitions for performance comparisons. Retain individual measurements as well as medians/quantiles; do not publish a percentile based on one expensive run. A nine-ray fixture is suitable for integration and latency mechanisms, not representative scientific convergence.

## 8. Bounded commands and validation practice

### 8.1 Windows baseline example

Use the repository's existing supported environment. The README recommends Windows Python 3.12; the project metadata currently allows Python 3.11–3.13. Do not migrate environments simply to run this brief. [S14, S16]

The following is a starting example, not evidence that these commands have already run. Read the selected tests before execution and adapt the scope if newer code changed their meaning.

```powershell
# Run from the repository root, in a development shell, not against microscope hardware.
git status --short
git rev-parse HEAD
git diff --stat

$PY = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $PY)) {
    throw 'Select an existing project-compatible interpreter; do not change global packages.'
}
$OUT = Join-Path 'tmp' ('round2-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $OUT | Out-Null
& $PY --version

$PreviousQtPlatform = $env:QT_QPA_PLATFORM
try {
    $env:QT_QPA_PLATFORM = 'offscreen'
    $BaselineTests = @(
        'tests/test_source_admission.py',
        'tests/test_working_point_contract.py',
        'tests/test_alignment_transactions.py',
        'tests/test_vacuum_opt_in.py'
    )
    & $PY -m pytest @BaselineTests -o 'addopts=' --tb=short `
        "--junitxml=$OUT/baseline.xml" 2>&1 |
        Tee-Object -FilePath "$OUT/baseline.log"
    $BaselineExit = $LASTEXITCODE
    "baseline_exit_code=$BaselineExit" |
        Set-Content "$OUT/baseline-exit.txt"
} finally {
    $env:QT_QPA_PLATFORM = $PreviousQtPlatform
}

git diff --check
```

For an existing WSL/Linux environment, use its project interpreter (commonly `.venv/bin/python`) and process-local `QT_QPA_PLATFORM=offscreen`. Do not assume the Windows virtual environment is executable inside WSL. Preserve the user's existing environment and package lock choices.

### 8.2 Existing targeted tests to inspect and reuse

| Work area | Existing test files recorded in the reviewed source/receipt |
| --- | --- |
| Job lifecycle | `test_job_coordination.py`, `test_job_coordination_gui.py`, `test_background_preview_gui.py`, `test_background_calculation_requests.py` |
| Backend policy/ownership | `test_ray_gpu_policy.py`, `test_ray_device_residency.py` |
| Parameters/cache | `test_parameter_registry.py`, `test_segmented_column_cache.py`, `test_input_assets.py` |
| Working points/UI | `test_product_usability.py`, `test_working_point_index.py`, `test_result_readout.py`, `test_illumination_apply.py` |
| Numerical evidence | `test_independent_emission_sampling.py`, `test_topology_evidence.py` |
| Experiment/geometry | `test_design_explorer.py`, `test_geometry_experiments.py`, `test_experiment_records.py` |
| Archives/integration | `test_working_point_export.py`, `test_portable_inputs.py`, `test_portable_field_maps.py`, `test_report_workflow.py` |
| Physical detector participation | `test_record_plane_detector_masks.py`, `test_stem_detector_control.py` |

Paths above are relative to `tests/`. Their existence/recorded coverage does not establish that all pass on the current checkout. Inspect actual scope before running. Add focused tests for new behaviour rather than changing old expectations to hide a regression.

`scripts/benchmark_product_usability.py` is an existing bounded CPU benchmark entry point; inspect its current options and workload before use. Other timing scripts may already implement needed measures. Reuse them rather than writing competing benchmarks. Do not run `scripts/validate_development_spec.py --full-suite` as a shortcut during the coherent-development pause.

### 8.3 Test-result rules

- Record full commands, interpreter, source identity, exit code and logs. Deduplicate repeated test IDs within a declared scope; do not sum overlapping runs into an inflated total.
- A skipped test is not passed evidence. A synthetic fixture is not actual source-chain transport, and actual source-chain transport is not experimental calibration.
- Fix a failing test expectation only when the authoritative contract demonstrates it was wrong, with a before/after rationale. Never remove the difficult test to claim completion.
- Keep baseline failures visible. Reproduction on the baseline must use an isolated worktree/copy only when necessary and must not discard user work or switch the user's active checkout.
- Do not run unbounded parameter grids, every assembly, large field meshes or long wave calculations. Expand actual physics validation through declared budgets and separate evidence, not uncontrolled overnight work.

## 9. Progress and final-delivery contract

The new progress file should begin with:

```markdown
# Round 2 optimisation progress

## Environment and baseline
- Actual branch / HEAD:
- Dirty paths preserved:
- Interpreter and package versions:
- Hardware actually detected:
- Baseline commands / exit codes / logs:
- Scope exclusions:

## Package status
| Package | Status | Changed files | Executed evidence | Remaining limit |
| --- | --- | --- | --- | --- |
| R2-00 | NOT_STARTED | | | |

## Findings resolved or still open
| Finding | Reproduction / source evidence | Change | Verification | Limit |
| --- | --- | --- | --- | --- |

## Acceptance mapping
| Namespaced criterion | Status | Exact test / command | Evidence type | Limitation |
| --- | --- | --- | --- | --- |

## Performance receipts
Record fixtures, individual timings, memory, identities and confounders.

## Physical and hardware blockers
Preserve unavailable scopes; identify the required decision or environment.

## Exact resumption instruction
Name the next ready package, relevant files, last executed command and open issue.
```

Final delivery must identify what actually changed and what was verified, not merely list all requested packages. Include remaining F02 startup uncertainty if no root cause was demonstrated; GPU parity if hardware was absent; native UI if only offscreen tests ran; numerical/physical qualification limits; and every gated physics item left unavailable. Keep all requirements and prior receipts readable.

## 10. Source index — pinned review evidence

Every link below is pinned to the reviewed commit. Paths/functions, not line numbers, should guide implementation after newer commits. The review examined the relevant files/sections, not every file in the repository. No external commercial instrument data or new physical calibration was assumed.

| ID | Reviewed repository source | Main use |
| --- | --- | --- |
| S01 | [AGENTS.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/AGENTS.md) | Single physical source, complete state, paused coherent development, vacuum policy. |
| S02 | [PROJECT_FUNCTION_SPEC.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/PROJECT_FUNCTION_SPEC.md) | Ideal controls, TOML/physical ownership, English content, requirement governance. |
| S03 | [Working points and numerical comparisons](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/docs/working-points-and-convergence.md) | Current user workflow and supported/unsupported scopes. |
| S04 | [PRODUCT_USABILITY_PROGRESS.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/docs/development/PRODUCT_USABILITY_PROGRESS.md) | Current package table, final receipts, open startup issue, hardware/physics limits and measurements. |
| S05 | [Previous CODEX_DEVELOPMENT_GUIDE.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/CODEX_DEVELOPMENT_GUIDE.md) | Previous baseline and invariants; do not recreate completed first-round work. |
| S06 | [gui/main_window.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/gui/main_window.py) | Existing UI integration, controllers, Working Points and layout owners. |
| S07 | [gui/calculation_controller.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/gui/calculation_controller.py) | Existing asynchronous pipeline, cache/resource estimation and ownership. |
| S08 | [gui/job_coordinator.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/gui/job_coordinator.py) | FIFO admission, worker conventions, lifecycle, retention and resource claims. |
| S09 | [physics/compute_backend.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/physics/compute_backend.py) | Actual backend choices and broad CuPy exception classification. |
| S10 | [parameter_registry.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/parameter_registry.py) | Pilot descriptors and conservative unknown-input dependency guard. |
| S11 | [runtime_parameters.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/runtime_parameters.py) | Editable scalar discovery, active-source conditions and structural ownership. |
| S12 | [sampling_diagnostics.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/sampling_diagnostics.py) | Weighted current/N_eff semantics and historical/index-only evidence. |
| S13 | [physics/beam_statistics.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/src/temsim/physics/beam_statistics.py) | Chief-ray angular quantiles, weighted moments and free-space waist estimate. |
| S14 | [README.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/README.md) | Application purpose, recommended environment and current classical focus. |
| S15 | [SIX_STAGE_PHYSICS_IMPLEMENTATION.md](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/docs/SIX_STAGE_PHYSICS_IMPLEMENTATION.md) | Existing model capabilities/approximations; dated historical evidence, qualified by later scope. |
| S16 | [pyproject.toml](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/pyproject.toml) | Supported Python versions, existing dependencies and package entry points. |
| S17 | [.github/workflows/tem-p0.yml](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/.github/workflows/tem-p0.yml) | Current default CI validation entry. |
| S18 | [scripts/validate_development_spec.py](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/985463968f365fdb945af1b4f3a744a7a53a5912/scripts/validate_development_spec.py) | Legacy criterion meanings, source blockers, test selection and exit-code logic. |

---

**Recommended first delivery:** establish the fresh baseline; separate current classical acceptance from full-image qualification; tighten accelerator error classification; instrument and investigate the preparation-worker lifecycle. Then proceed to the ready user-workflow and scientific-diagnostic packages. Do not use a hardware or physics blocker as a reason to recreate already completed features or to stop unrelated ready work.
