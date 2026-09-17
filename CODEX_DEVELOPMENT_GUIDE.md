# TEM Simulator v2 — Codex Development Guide

**Document version:** 1.0  
**Prepared:** 17 September 2026  
**Repository:** `MagNetiCLLLL/tem_simulator_v2`  
**Reviewed branch:** `master`  
**Reviewed commit:** `9ea3a7cc3940d81a956770e6a3785f3dc9994363`  
**Purpose:** Implement functional and usability improvements, especially working points, alignment, UI, calculation speed and data structures.  
**Status:** Implementation brief. Creating this document did not modify the repository or execute its tests or benchmarks.

> Use the current checkout, not an automatic checkout of the reviewed commit. Reconcile this brief with newer code and explicit user decisions before editing. Existing implementations must be extended, not duplicated. Statements about runtime cost below identify mechanisms to measure, not measured performance problems or promised speed-ups.

## 0. Start here

Place this file in the repository root as `CODEX_DEVELOPMENT_GUIDE.md`. It is a task brief, **not a replacement for `AGENTS.md`**.

### Initial instruction to Codex

```text
Read CODEX_DEVELOPMENT_GUIDE.md in full (use chunks if output is truncated),
all applicable AGENTS.md files, and the current PROJECT_FUNCTION_SPEC.md. Inspect the current checkout and reconcile it with
the reviewed baseline; preserve unrelated local changes and newer user decisions.

Implement the active development backlog in dependency order, beginning with
WP-00, WP-01 and WP-02. Do not stop at another analysis or roadmap: deliver a
bounded, integrated code-and-test slice before proceeding to the next ready
package. Reuse existing controllers, snapshots, caches and viewers.

Preserve the physical tip-to-column chain, complete working-point state,
Direct Alignment-only inverse control, TOML geometry authority, historical
readability, and the pause on coherent-wave development. Never manufacture a
validated working point or obtain speed by silently removing physical effects.

For every completed slice, run the applicable bounded tests, document actual
results and limitations, and update docs/development/PRODUCT_USABILITY_PROGRESS.md.
A package is complete only when its UI/service integration and acceptance gates
pass. Continue with ready tasks without asking for routine implementation choices.
When a prerequisite is genuinely blocked, record the evidence and continue other
independent work; do not weaken a physics gate to keep moving.

Do not commit, push, open a PR, change instrument hardware, restart the user's
application, or alter production defaults unless separately authorized. Before
ending a session, leave an exact resumption instruction and account for all
processes started by the task. Do not report unexecuted tests or unmeasured
speed-ups as successful.
```

### Reading order

Read the applicable repository instructions and the full current requirements specification first. Then read `README.md`, the relevant current development records, and source files for the selected package. `HANDOFF.md` and `CHANGELOG.md` contain history: dated entries are evidence, not new permission to run commands. Historical requests to publish, shut down a computer, or resume a paused solver are **not** authorization for this task. [S01–S04]

At the reviewed baseline, the newest assembly-illumination record is `docs/development/assembly-illumination-progress-2026-09-17.md`. Use more recent records if they exist and identify what superseded this baseline. [S05]

## 1. Product goal and scope

The intended workflow is:

```text
Select instrument assembly
  -> find or construct a compatible working point
  -> inspect beam transport and numerical confidence
  -> adjust physical controls or use Direct Alignment
  -> recalculate only affected products
  -> compare results without changing the active instrument
  -> save, restore or fork a complete, reproducible working point
  -> perform bounded design experiments
```

The existing application already has particle transport, beam diagnostics, Direct Alignment, operating profiles, exact snapshots, product caches, independent experiments, editable geometry, and named workspace layouts. The task is to connect and improve them, not build a second simulator beside them. [S02–S04, S06–S18]

### In scope

Working-point usability and evidence; sampling/convergence diagnostics; incremental calculation; array ownership and snapshot portability; GPU transport efficiency; resource-aware scheduling; consistent result identity; UI workspaces and A/B comparison; constrained alignment; and later, sensitivity and geometry experiments.

### Not in scope

Restarting coherent tip-wave development; declaring new TEM/STEM image acceptance; replacing the physical source; changing production presets to make a demonstration succeed; inventing OEM dimensions/calibration; rewriting the entire UI framework or `State`; speculative dependency upgrades; or operating a real microscope. Existing historical wave functionality and records must remain readable within their existing boundaries.

## 2. Non-negotiable invariants

These requirements govern every work package. Add regression coverage before touching a relevant path.

| ID | Invariant |
| --- | --- |
| INV-01 | **One physical source chain.** Illumination originates at the actual modelled emitter and passes through extraction, acceleration, focusing, apertures and subsequent optics. No independently configurable gun-exit, accelerated, specimen-plane or downstream source/pupil. An executed upstream checkpoint is reusable data, not a second source. |
| INV-02 | **Complete working points.** Preserve all source, lens and component parameters, including disabled hardware, geometry identity, field maps, calibration, aperture states, units, numerical settings, seeds and implementation identity. Do not replace a full snapshot with a few beam moments or a list of lens percentages. |
| INV-03 | **Readouts are not independent controls.** Alpha95/alpha99, D95, magnification, camera length and other derived quantities remain readable from their result. Inverse writes to supported derived optical targets occur only through Direct Alignment, by solving actual component controls and validating forward transport. |
| INV-04 | **Physical weights and losses.** Track absolute/relative current consistently. Do not renormalize surviving rays to force a current target, discard inconvenient rays, bypass stops, shrink the emitter, or invent a source distribution to obtain a preferred spot. Keep the existing ideal scalar total-flux control explicitly distinct from physical brightness/spot-number or space-charge models. |
| INV-05 | **Sampling is not physics.** Numerical sampling proposals may change quadrature or proposal density, but must cover the original source support and retain appropriate physical weights. Changes invalidate relevant numerical identities. No success claim from a few surviving samples or a stable-looking plot. |
| INV-06 | **Respect the paused wave scope.** Do not restart long coherent-wave calculations or open production wave admission. Classical geometric spots are not coherent probe-resolution predictions. Keep per-mode complex-state and phase-reference requirements intact in historical/shared code; do not invent an aggregate phase for an incoherent mixture. |
| INV-07 | **Physical routing stays intact.** Deselecting a readout does not remove detector interception, absorption or other modelled physical effects. EDS remains off-axis photon collection, not an electron stop. A detector before the energy-filter entrance does not traverse it; a path entering the filter must not bypass it. |
| INV-08 | **TOML owns live structure.** Runtime state is not a competing geometry authority. Geometry experiments use detached candidate assemblies. A historical snapshot may resolve archived definitions explicitly, but viewing or restoring must not silently overwrite live TOML files. |
| INV-09 | **Ideal design remains continuous.** Do not add discrete aperture positions or silently enforce hardware current, travel, heat or saturation limits. Keep finite-value and numerical-domain checks. Optional realistic constraints must be clearly identified and explicitly enabled. |
| INV-10 | **Result and UI isolation.** Viewing, pinning, plotting, layout changes and comparisons do not alter the instrument. Preserve completed high-accuracy products when preview work starts, fails or is cancelled; mark them stale when appropriate rather than deleting or relabelling them. |
| INV-11 | **Transactions are atomic.** Apply/restore/alignment either commits the complete intended change or restores the exact preceding live state, including selectors and result associations. Reject obsolete results after intervening edits. A failure must not leave half-applied parameters. |
| INV-12 | **Immutable, exact scientific data.** Preserve array values, dtype, shape, units and ownership. Keep full-precision continuation checkpoints separate from reduced plotting history. Presentation decimation never becomes the population used for weighted scientific metrics. |
| INV-13 | **Evidence has a scope.** Successful execution, a numerical preset named `High accuracy`, numerical convergence and physical-model qualification are different facts. A source, assembly, operating mode or plane validated in one scope does not certify another. |
| INV-14 | **Historical data remain honest.** Keep old results readable. Exact restore requires compatible schemas/solver/dependencies; migration creates a new identity and retains the original. Never silently convert prohibited historical source inputs into active calculations or mark old products as current. |
| INV-15 | **Vacuum policy is retained.** Vacuum scattering/attenuation remains opt-in and disabled by default. Save explicit choices. Broad invalidation after a vacuum-setting change is allowed; do not narrow it without proving equivalent dependencies. |
| INV-16 | **No unrelated or destructive operations.** Preserve local work, records, caches and published history. Do not auto-commit/push, connect to instrument control, force-delete files, reset the checkout, terminate unrelated processes, or restart/shut down the user's application or computer. |
| INV-17 | **English repository content.** New UI strings, code comments, configuration descriptions and project documentation are English. Preserve stable requirement IDs and append development history; do not erase earlier requirements. |

Source basis: existing source/cache and scope policy [S01], product requirements [S03], latest illumination evidence [S05], and the user's explicit complete-cache/Direct Alignment constraints retained in this brief.

## 3. Reviewed baseline: do not rediscover existing features as missing

All observations below refer to the reviewed commit, not an assertion about a later checkout.

| Area | Existing implementation or documented state | Development consequence |
| --- | --- | --- |
| Illumination bank | All-assembly default bank is not complete/installed. Some particle Microprobe cases have scoped evidence; this does not certify downstream images/filter behaviour. [S05] | Implement evidence-aware management; do not mark every assembly validated or auto-install candidates. |
| Crossover policy | The approved no-blanker corrected Microprobe baseline has its own six-crossover reference; the historical Nanoprobe five-crossover constraint remains separate. [S05] | Bind topology evidence to mode/assembly and ordered component intervals, not one global count. |
| Weak transmitted sampling | The latest monochromated-FEG examples retain only 25/53 particles at the sample from 512/1024 emitted particles. [S05] | Surface sampling confidence and independent refinement; do not treat arrival count as current fraction. |
| Source sampling | Optional `apex_stratified_v1` already exists with actual area weights. [S06] | Extend spatial/directional/energy coverage; do not create a parallel source. |
| Beam diagnostics | Position, angular views, histograms, intensity, interactions and phase-space views already exist and use cached plane data. [S07] | Improve linking, definitions and A/B comparison instead of duplicating plots. |
| Working Points UI | Import/export, read-only parameter/observable inspection, compare-with-current, restore, continue and undo are already present. [S22, S23] | Extend discovery, scoped evidence and pinned A/B comparison; do not recreate the browser. |
| Direct Alignment | Catalog-backed target controls, detached solving and transactional validation already exist. [S04, S08] | Extend constraints/tasks and error reporting through those paths. |
| Request capture | High-accuracy capture encodes/decodes the instrument graph and copies editable structures before preparation. [S09] | Measure main-thread latency; use immutable asset handles while preserving capture-time consistency. |
| Snapshot arrays | Snapshot encoding uses exact hexadecimal array bytes; restoration checks current implementation and external paths despite retaining external content. [S10] | Separate bulk assets, preserve exactness, and add explicit archive resolution. |
| Persistent artifacts | Content-addressed metadata and numeric arrays, checksum verification, atomic references and bounded storage already exist. [S11] | Extend this store; do not replace it with an unrelated database/cache. |
| GPU rays | `cuda_rk4()` uploads inputs, allocates outputs, synchronizes and copies outputs back per call. [S12] | Measure repeated transfer/allocation and introduce bounded residency. |
| Backend selection | Auto ray selection uses particle-count thresholds; wave and particle policies already have separate logic. [S13] | Add measured stage-specific cost selection without claiming the whole application is GPU-based. |
| Dependencies | Product signatures and targeted reuse exist; many rules use explicit field sets and `_drop_*` helpers. [S14] | Introduce declared dependencies incrementally, with parity tests. |
| Scheduling/UI | Multiple controllers, debounce, deferred work, generation tokens and named layouts already exist. [S15, S16] | Consolidate coordination; preserve existing cancellation and stale-result protection. |
| Experiments | Runtime-parameter sweeps deliberately exclude TOML-owned geometry. [S17] | Geometry experiments need detached configuration candidates, not a wider unchecked runtime setter. |
| Timing | Stage timing and reuse reporting already exist. [S18] | Extend instrumentation; do not maintain a second independent timing system. |

## 4. Delivery sequence and dependencies

**Priority is not permission to skip dependencies.** Implement small vertical slices: model/service, UI integration, regression tests, measured evidence and documentation. Audit-only or scaffold-only work is not a completed feature.

| Package | Priority | Deliverable | Prerequisites |
| --- | --- | --- | --- |
| WP-00 | P0 | Current baseline, invariant tests and performance harness | None |
| WP-01 | P0 | Unified capability/evidence status and Working Point Manager | WP-00 |
| WP-02 | P0 | Sampling & Convergence assistant | WP-00; reuse WP-01 evidence contracts |
| WP-03 | P1 | Parameter registry and explainable dependency planning | WP-00; WP-01 identities |
| WP-04 | P1 | Immutable assets and lightweight request capture | WP-00; WP-03 pilot mappings |
| WP-05 | P1 | Portable, exact working-point archives | WP-01, WP-04 |
| WP-06 | P1 | GPU plan/buffer reuse and backend selection | WP-00, WP-03, WP-04 |
| WP-07 | P1 | Unified job/resource coordination and incremental execution | WP-03, WP-04; integrate WP-06 when available |
| WP-08 | P1 | Task workspaces, fixed readouts, result identity and A/B comparison | WP-01–WP-03; need not wait for GPU work |
| WP-09 | P1 | Multi-constraint Direct Alignment and supported extra tasks | WP-01–WP-03; integrate WP-07 coordination |
| WP-10 | P2 | Sensitivity/robustness analysis and detached geometry experiments | WP-03, WP-05, WP-07; WP-09 for optional inverse tasks |
| WP-11 | Continuous/final | Regression, integration, packaging and user documentation | Apply to every delivered slice |

Recommended progression: deliver WP-00–WP-02 first, then registry/asset foundations. UI improvements can proceed before GPU completion. Do not block unrelated usability work while attempting to qualify every unresolved source/assembly. WP-10 follows a stable core; it must not become an excuse to defer P0/P1 integration.

## 5. Data and API contracts

These are **proposed logical boundaries**, not a demand to create exactly these class/file names. Reuse compatible existing types; introduce adapters before moving ownership. Avoid a big-bang rewrite of `State`.

### 5.1 Six ownership domains

| Domain | Owns | Must not own |
| --- | --- | --- |
| Instrument definition | Assembly, component IDs, TOML geometry/materials and structural provenance | Mutable viewer state or a second competing geometry definition |
| Operating state | Source/voltage/excitation/deflector/aperture and other actual runtime controls | Cached results masquerading as inputs |
| Execution specification | Numerical budgets, method versions, seeds, backend policy and requested readouts | Independently specified downstream illumination |
| Input assets | Immutable field maps, CIF and external model data with verified identity | Arbitrary untrusted executable objects |
| Result products | Immutable numerical outputs, complete input reference, plane/frame definitions, evidence and actual backend | Mutable pointers to the current GUI's instrument |
| View state | Selected plane, zoom, plot style, layout, camera and comparison selection | Physical transforms or hidden edits to operating state |

Preserve exact units at boundaries. Do not convert all existing units throughout the repository in one change. New transfers must state units, coordinate frame, plane and energy convention explicitly.

### 5.2 Working-point record

A working-point record must reference the full existing instrument snapshot, not replace it. Add stable identity, title, creation metadata, assembly/source/mode compatibility, external assets, result-product references and scoped evidence. Include disabled component parameters and numerical settings.

Store derived observables with value, unit, definition, observation plane, coordinate frame, input/result identity and validity. For example, D95 is not a radius, RMS width or FWHM; alpha95 must retain its existing chief-ray/current-containment definition. Empty transmitted populations have an unavailable value and a reason, not zero diameter or zero angle interpreted as perfect focusing.

Separate actions:

- **View:** read only; never changes current parameters.
- **Apply illumination controls:** apply only the declared upstream control patch to a compatible assembly, preserve downstream settings, then recompute/validate affected products.
- **Restore exact working point:** restore the entire compatible snapshot intentionally.
- **Fork/migrate:** create a new record with new identity and unvalidated current products.

Changing an upstream lens whose field extends downstream still invalidates every physically affected product, even when the applied control patch is described as “illumination only”.

### 5.3 Status is multidimensional

Represent these separately; avoid a single `valid` flag:

| Axis | Example states |
| --- | --- |
| Capability | Available; unavailable for this assembly; paused; unsupported by the current model |
| Execution | Not run; queued; running; completed; cancelled; failed |
| Freshness | Current; stale; historical |
| Numerical evidence | Not checked; insufficient sampling; refinement failed; passed named checks |
| Physical scope | Classical particle illumination to a named plane; other explicitly supported model scopes |
| Compatibility | Exact replay eligible; read-only historical; explicit migration required |

A scoped evidence record includes input and implementation identities, assembly/source/mode, plane range, observed metrics, independent refinements, predeclared tolerances and measured changes. Store a document/report reference and observation date. Validation labels must be derived from compatible evidence, not editable user checkboxes.

### 5.4 Parameter registry

Each registered parameter needs a stable ID, owner, unit, data type, getter, supported write route, model applicability, evidence/provenance category, numerical-domain validation and affected stages. Keep display formatting separate from stored precision.

Write routes distinguish physical operating edits, structural candidate edits, execution settings, display-only state and inverse targets. Inverse targets resolve through Direct Alignment; derived readouts are otherwise read-only.

Unknown/new fields remain conservatively physics-relevant. Do not silently exclude them from signatures. Until a mapping is proven, retain existing conservative invalidation. Register the first representative source, lens, aperture, field-map and display fields before expanding the registry.

### 5.5 Immutable asset reference

An asset reference needs a schema/versioned content identity, dtype including byte order, shape, units/frame when relevant, storage location and verified availability. Hash definitions must be unambiguous: numeric array identity covers dtype/shape/layout semantics and bytes, while an external source file retains its own byte checksum.

A `writeable=False` flag alone is not a sufficient ownership guarantee when a writable alias still exists. Detach once on ingestion or use an existing genuinely immutable owner; retain/pin that owner during active jobs. Do not repeatedly copy large arrays for every read.

Share compatible assets between working points, but retain complete logical snapshot content. A digest does not replace the ability to recover the original bytes.

### 5.6 Execution identity versus product equivalence

Keep the complete execution record, including actual backend, precision, implementation and fallback reason. A fast monotonic UI revision token may protect transactions, but does not replace scientific input identities.

Cross-backend cache reuse requires an explicitly supported equivalence policy and appropriate evidence; do not relabel a CPU result as newly computed on GPU. Preserve origin provenance for reused products.

## 6. Work packages

### WP-00 — Establish a trustworthy baseline

**Goal:** Know what is present, what is currently failing and where time/memory are spent before changing architecture.

**Inspect:** `AGENTS.md`, `PROJECT_FUNCTION_SPEC.md`, `pyproject.toml`, existing acceptance scripts, `calculation_performance.py`, `gui/calculation_controller.py`, and the tests for the selected paths. [S01, S03, S18–S21]

**Implement:**

1. Record current HEAD/branch, working-tree status and any already-modified task files. Do not reset or hide unrelated changes.
2. Reconcile baseline observations in Section 3 with the checkout. Record each as present, changed, already solved or not reproduced.
3. Capture a bounded baseline for request capture, field/gun work, column transport, hashing, cache I/O, CPU/GPU transfers and GUI display separately. Extend existing timers and report real wall time without summing overlapping nested timings.
4. Add/locate invariants for source ownership, exact snapshots, read-only views, transaction rollback and numerical/display separation.
5. Create the progress receipt described in Section 10, initially with truthful `not_started`/`blocked` states.
6. Use temporary configurations and isolated GUI settings; do not modify the user's saved layouts, instruments, running application or real acquisition records.

**Acceptance:** Current-state differences are documented; at least one small representative particle case and relevant tests have actual receipts; baseline failures are distinguished from new failures; no physics/default changes were needed merely to create a benchmark. If execution is unavailable, mark it not run and continue only changes that can be responsibly verified.

**Boundary:** Do not spend the whole implementation session writing an audit. Follow with the first integrated WP-01 slice when its prerequisites are satisfied.

### WP-01 — Capability status and Working Point Manager

**Goal:** A user can find, inspect, apply and compare meaningful working points without mistaking historical/candidate evidence for current qualification.

**Extend:** `instrument_snapshot.py`, `working_point.py`, `gui/working_point_panel.py`, `design_experiments.py`, `design_explorer.py`, `operating_modes.py`, existing result status code and relevant GUI panels. [S05, S08, S10, S15, S17, S22, S23]

**Implement:**

1. Reuse existing snapshot/checkpoint/recipe storage. The separate `gui/instrument_recorder.py` controls actual acquisition and is not the working-point manager; leave its hardware operations outside this task. [S25] Add an evidence-aware index/list rather than a second working-point persistence format with fewer parameters.
2. Expose source, assembly, mode, current, D95, alpha95, plane, numerical status, date and compatibility in a filterable table. Load large products lazily.
3. Separate View, illumination-only Apply, exact Restore and Fork/Migrate. Preview the exact control diff before an explicit apply; preserve non-target controls.
4. Link evidence to its actual source report/input identities. Historical imported evidence remains historical until matching identities can be established.
5. Keep default presets unchanged. Allow saving unresolved candidates with honest status. Do not require all candidates to pass before the manager itself can be used.
6. Bind crossover count **and ordered component intervals** to the correct assembly and mode. Preserve the approved distinct Microprobe/Nanoprobe baselines.
7. Expose unsupported/paused capabilities with a brief reason. Disabled controls must explain the missing prerequisite instead of silently doing nothing.

**Acceptance:**

- Viewing/sorting/pinning a record leaves the complete current state and physics-call count unchanged.
- Illumination-only apply changes exactly the declared compatible controls; downstream controls are equal before/after.
- Incompatible assembly/source/schema is reported without a partial apply or hidden conversion.
- Stale and historical evidence cannot display as a current passed validation.
- Save/restore retains exact full-precision inputs, disabled hardware and source dependencies.

### WP-02 — Sampling & Convergence assistant

**Goal:** Explain numerical confidence and allocate extra sampling to the unresolved dimensions without changing the physical source.

**Extend:** Existing source samplers, `physics/beam_path_audit.py` if still present under that name, illumination validation scripts, beam-plane analysis and working-point evidence. Locate the current implementation before adding modules. [S05–S07]

**Implement:**

1. Display emitted/transmitted sample counts, weighted source current, plane current, transmission, weight concentration, centroid, D95 and alpha95. Always identify the plane and population.
2. Report `N_eff = (sum(w))^2 / sum(w^2)` only as an auxiliary weight-concentration diagnostic. It is not a convergence proof or an automatic error estimate for deterministic quadrature. Handle zero total weight explicitly.
3. Provide bounded, cancellable one-factor-at-a-time refinements: spatial source sampling, directions, energy samples, gun step, column step and field grid. Add joint checks after the individual causes are understood.
4. Keep physical settings and authorized current scale fixed during numerical comparisons. Record seeds/rules and numerical identities. For randomized methods, use appropriate repeated independent samples; do not use ordinary IID bootstrap confidence claims for deterministic quadrature without justification.
5. Extend existing stratification only where needed. Preserve source support, emission law and weights. For proposal-based sampling, retain correct proposal/target accounting and record proposal identity.
6. Distinguish physical blockage, no sampled survivors, insufficient numerical support, out-of-domain model use and numerical failure. Empty sampling is not proof of exactly zero physical transmission.
7. Track the first checkpoint at which independently refined results diverge. Make crossover comparison robust to clipping and compare common positive-weight populations where required by the existing audit.
8. Keep requested thresholds from current target configuration. Declare additional tolerances before evaluation; never loosen them after seeing a failure. For near-zero quantities use justified absolute tolerances, not unstable relative errors.
9. Save lightweight scalar evidence. Keep large ray arrays in local artifact storage, not Git.

**Acceptance:** Deliberately under-resolved fixtures are rejected; weighted metrics differ correctly from raw-count fractions; independent refinement records identify what changed; cancellation preserves previous complete results; every numerical change invalidates appropriate caches. An unresolved real source may remain unresolved—the tool must report that truthfully rather than inventing a successful default.

### WP-03 — Parameter registry and explainable dependencies

**Goal:** One parameter definition can drive editing, model applicability, sweep eligibility and cache invalidation without weakening current reuse correctness.

**Extend:** `calculation_cache.py`, `calculation_manifest.py`, `component_keys.py`, parameter editors and `design_experiments.py`. [S14, S17]

**Implement:**

1. Introduce an adapter-backed registry for a small representative set; keep existing setters and validation until parity is proven.
2. Classify structural, operating, execution, presentation and inverse-target parameters. Preserve stable component IDs and explicit units.
3. Build a read-only reuse/recompute plan from stage dependencies and existing signatures. Surface reasons such as `Gun voltage changed`, `Field support crosses checkpoint`, or `Readout-only change`.
4. Run old and new dependency decisions side-by-side in tests. Any narrowed invalidation must have counterexample coverage; fall back to conservative invalidation on unknown mappings.
5. Determine boundaries from actual field support, masks, vacuum and physical interactions, not component-centre order or a presumed specimen-plane split.
6. Retain model/implementation/numerical identities. Do not remove backend or precision evidence just to increase hit rate.
7. Generate consistent parameter tooltips and sweep eligibility from the same definitions once each mapping is validated.

**Acceptance:** Display-only edits execute no physics; a source/field/active-stop change invalidates every consumer; upstream cache reuse remains correct for downstream edits only when actual support permits it; changed vacuum settings may conservatively invalidate all; unknown fields cannot produce false cache hits.

### WP-04 — Immutable assets and lightweight request capture

**Goal:** Request creation should not repeatedly serialize or copy large immutable inputs on the GUI thread.

**Extend:** `gui/calculation_request.py`, `instrument_snapshot.py`, `artifact_store.py`, `cache_memory.py`, and existing field-map ownership. [S09–S11]

**Implement:**

1. Measure capture, encode, decode, deepcopy, hashing and preparation costs independently on small and large existing asset fixtures.
2. Freeze/register bulk assets once at ingestion. Capture small editable values, complete structural identity and pinned immutable handles at the click-time revision.
3. Do expensive resolution/verification/preparation off the GUI thread only after a consistent snapshot has been captured. Never hand a mutable live `State` to a worker.
4. Keep complete logical snapshots, including disabled hardware and extras attached by supported loaders. Reuse existing allow-listed decoding and object-alias preservation.
5. Introduce versioned asset references with a compatibility reader for existing inline-hex snapshots. Do not silently rewrite old records or discard their embedded bytes.
6. Avoid unnecessary duplicate copies in artifact construction/retrieval, but retain strong ownership and integrity checks. A read-only view backed by mutable live storage is forbidden.
7. Pin assets during active jobs and perform safe eviction after release. Account for shared storage once while exposing resident/working-set memory clearly.
8. Any memory-mapped path must survive file-lifetime and Windows deletion/locking tests before becoming default.

**Acceptance:** Editing live controls immediately after capture cannot change worker inputs; mutating an original input array cannot change an admitted immutable asset; graph round-trip preserves exact values and aliases; capture of already-registered large assets does not repeat a full payload-sized hex/array copy; cancelled or failed capture leaves earlier products valid. Record latency and peak-memory measurements rather than asserting a speed-up from code structure alone.

### WP-05 — Portable exact archives

**Goal:** A compatible working point can be restored after moving its folder or machine, without depending on original external file paths and without silently changing its physics.

**Extend:** Existing `.temwp` package reader/writer, `InstrumentSnapshot.restore()`, artifact storage and external input resolution. Preserve existing packages as supported historical formats. [S10, S11, S22, S23]

**Implement:**

1. Resolve archived assets by verified content identity through an explicit read-only archive resolver. Original paths remain provenance, not mandatory locations when exact embedded content is available.
2. Resolve both bulk field/specimen assets and archived structural definitions. Ensure restored geometry and input loaders consume the declared archive consistently rather than mixing archived and live files.
3. Preserve implementation/schema compatibility checks for exact continuation. Portable input recovery is not a portable Python runtime or an excuse to ignore solver changes.
4. Make metadata-only, inputs-complete and inputs-plus-results exports distinguishable. A metadata-only record must not claim offline reproducibility when assets are absent.
5. Use atomic writes, checksums and bounded parsing. Reject path traversal, escaping links, unreasonable declared shapes, decompression bombs and malformed numeric data. Keep allow-listed object/schema decoding and prohibit pickle-based recovery.
6. Validate all dependencies before publishing a restored live state. Missing/corrupt assets leave the current instrument untouched.
7. Explicit migration creates a new working-point identity, preserves the archived original and requires recalculation. Document a migration diff; do not silently normalize old physical inputs.

**Acceptance:** Export a complete package, move it into an isolated directory, make original fixture paths unavailable, then restore through the archive and reproduce the bounded compatible calculation. Verify byte/value identity of controls and assets. Changed solver/schema remains historical or explicitly migrated; bad checksums/path entries fail cleanly with no live-file writes. Do not remove the user's real files for this test.

### WP-06 — GPU residency, reuse and backend policy

**Goal:** Reduce repeated overhead in real particle workloads while preserving scientific equivalence and honest backend reporting.

**Extend:** `physics/ray_integrator.py`, `physics/core.py`, `physics/compute_backend.py` and performance instrumentation. Do not replace the entire electron-gun solver just because the column path uses CUDA. [S12, S13, S18, S26]

**Implement:**

1. Establish cold/warm end-to-end baselines, including gun work, column work, upload, kernel, synchronization, download and rendering. Keep physics, ray budgets, steps and precision identical for comparisons.
2. Introduce a bounded device-plan cache keyed by exact consumed plan identity, numerical schema, dtype, device and CUDA context. Share unchanged field/coefficient arrays across compatible runs.
3. Reuse capacity-managed particle/checkpoint buffers with clear ownership. Results still referenced by a view/job cannot be overwritten by the next request.
4. Return requested readouts, exact continuation checkpoints and bounded display history as separate products. Avoid full history transfers when unused, but never remove internal state or physical interception required for correct propagation.
5. Keep float64 scientific checkpoints and existing arithmetic precision unless a separately validated precision change is explicitly in scope. Do not market float32 substitution as an equivalent optimization.
6. Respect context reset/device change, cancellation and process lifetime. Evict/rebuild incompatible device objects safely; keep device handles out of persistent archives.
7. Replace particle-count-only Auto decisions incrementally with measured per-stage costs based on ray count, axial steps, field complexity, residency and setup/transfer overhead. Keep a deterministic conservative policy when no timings are available.
8. Expose actual backend per stage and reason for fallback. Preserve CPU reference support. A documented `require_gpu` policy must fail visibly when GPU execution is unavailable; `prefer_gpu` may use an explained fallback. Do not silently turn an explicit CPU selection into GPU work.
9. Retry only eligible accelerator resource/runtime failures. Input errors, invalid physics, failed convergence, cancellation and corrupted assets are not reasons to hide the failure with a CPU retry.

**Acceptance:** Numerical/weight/stop results meet predeclared equivalence tolerances against the reference backend; the second unchanged-plan run demonstrably avoids redundant plan upload/allocation; changed input/context cannot reuse an invalid plan; partial GPU results cannot replace a completed result; fallback and strict-GPU policies are tested. CPU-only success is **not** a passed hardware-GPU benchmark.

### WP-07 — Unified job and resource coordination

**Goal:** Responsive live tuning and bounded experiments without duplicate computation, stale UI updates or competing full-memory claims.

**Extend:** Existing calculation, interactive, sweep, alignment and preset controllers; use an adapter coordinator first. Preserve single-owner live-state mutation. [S15, S16, S19]

**Implement:**

1. Introduce common job identity, captured revision, cancellation token, priority, resource reservation and output association around existing worker operations.
2. Coalesce rapid live edits to the latest request. Use safe cancellation boundaries; do not forcibly interrupt a GPU kernel or discard a user's independent experiment because a preview changed.
3. Permit fair scheduling: continuous previews must not starve explicit work indefinitely. Mark explicit high-accuracy jobs as calculations of their captured state, even when current live controls change.
4. Combine working RAM, retained shared assets, temporary buffers and VRAM reservations across controllers. Avoid double-counting shared storage and avoid each controller independently assuming all free memory is available.
5. Bound worker count and numerical-library threading. Do not nest unlimited Numba/BLAS/process parallelism. Use the existing execution model unless profiling justifies isolation in a process.
6. Execute the dependency plan from WP-03. Report reused/recomputed stages with reasons and retain source identities. Respect physical field support, stops, vacuum and interactions.
7. Deduplicate concurrent identical preparation/tasks where safe. Cancelling one consumer must not invalidate results still needed by another.
8. Report completed work units and stages honestly. Do not display equal-weight stage progress as a reliable elapsed-time estimate.
9. On application exit, account for owned workers and resource release. Do not clear the user's valid persistent products merely because a task was cancelled.

**Acceptance:** Rapid edit tests retain only the latest live result; stale jobs cannot mutate current state; detached experiments remain associated with their own inputs; simultaneous requests respect the shared memory limit; cancellation leaves previous complete results available; failures release reservations; UI thread remains usable while work is queued/running.

### WP-08 — Task-oriented UI and A/B comparison

**Goal:** A user can understand the current instrument, displayed result and effect of an adjustment without repeatedly searching through unrelated pages.

**Extend:** Existing `MainWindow`, workspace layouts, working-point browser, beam analysis and result viewers. Preserve working keyboard shortcuts and object names used by tests where feasible. [S07, S15, S22]

**Implement:**

1. Provide named layouts for `Instrument`, `Alignment`, `Experiments` and `Results` using existing layout persistence. They are arrangements of current pages/docks, not different physical states.
2. Add a shared compact readout: observation plane, current, D95, alpha95, beam centre, transmission and numerical status. Derive it from the displayed result, never partly from live settings and partly from an older product.
3. Display current-settings revision, displayed-result identity, model scope, requested numerical preset, actual backend and validation scope separately. Use text/icons as well as colour; `High accuracy` is not a passed validation badge.
4. Add parameter-state labels: `Active in current model`, `Stored but inactive`, `Derived readout`, `Provisional calibration`, `Display geometry only`. Use WP-03 definitions.
5. Expose voltage reference explicitly for the gun. Do not present the historical analytical `potential_scale` as a physical voltage conversion or imply it is consumed by the solved Laplace model. [S24]
6. Extend existing compare-with-current into `Pin A`, `Pin B`, exact parameter diff and observable/beam comparison. Preserve independent input/result identities. Mark incompatible planes, units, definitions or numerical scopes; avoid misleading direct differences.
7. Reuse shared observation/plane caches. Do not rebuild unchanged meshes or hidden-view plots on every operating edit. Update only visible dependent views, preserving their selected plane and camera.
8. Distinguish plot-history interpolation from exact evaluated checkpoints. Exact diagnostics at a newly requested plane may require a clearly labelled bounded continuation; a viewport drag must not silently pretend an interpolated display point is a newly validated scientific plane.
9. Decimate display data without changing weighted metrics. For many working points/rows, use lazy loading and appropriate Qt model/view patterns instead of filling huge tables synchronously.
10. Test keyboard operation, focus-safe wheel input, readable errors, dark-theme checked states, resizing and supported display scaling. Error details should be copyable.

**Acceptance:** Layout/selection/colour/zoom changes leave instrument and solver-call count unchanged; A/B never combines one record's settings with another record's rays; mismatched metric definitions are flagged; stale results remain visible but marked; common layouts remain usable at tested window sizes; repeated tab changes do not reload unchanged scientific assets.

### WP-09 — Constrained Direct Alignment

**Goal:** Solve useful optical tasks with transparent control changes and forward-validated outcomes, rather than matching one scalar at the expense of everything else.

**Extend:** Existing registered targets, `alignment_transaction.py`, Direct Alignment controller/panel and current forward validators. [S04, S08]

**Implement in two slices:**

**A. Existing target improvements.** Add target, allowed physical controls, fixed controls, constraints, achieved values, forward residuals, numerical evidence and apply/rollback status. A Nanoprobe request can jointly constrain alpha95, D95, specimen current, centroid, entrance-plane focus/waist and the applicable crossover sequence. Do not hard-code one source's valid ranges for every assembly.

**B. Additional model-supported tasks.** Add beam centring, beam tilt/pivot matching, stigmator alignment and scan/descan matching only through implemented physical elements and an appropriate observation model. First identify controllable variables, independent observables, rank/conditioning and available authority. Unsupported combinations remain unavailable with reasons, not fake controls.

**Rules:**

1. The optimizer modifies an isolated candidate. It never writes derived illumination directly or mutates the live instrument during trials.
2. Use explicit finite search bounds for numerical stability, distinguishing them from optional hardware limits. Existing ideal continuous design freedom remains.
3. Scale objectives/constraints by meaningful units/tolerances; record weights and residuals. Poor conditioning is reported rather than hidden by arbitrary parameter changes.
4. Forward validation checks real apertures/walls, current weights, field support, applicable topology and independent numerical refinements. The optimizer's own sampled objective is not independent validation.
5. Freeze the physical source and non-authorized controls. Do not vary emission law/current scale to conceal a bad optical fit.
6. Reject stale solutions; cancellation and validation failure restore exact pre-operation controls, selectors and associations. Successful apply supports undo through the existing transaction path.
7. Report specific failure classes: target missed, constraint violated, numerical check failed, insufficient sampled transmission, unavailable control authority, ill-conditioned solve, stale input or cancelled.
8. Distinguish `No acceptable candidate in this bounded search` from a claim of physical impossibility.

**Acceptance:** Fixtures can meet an angle target yet fail spot/current/topology and must be rejected. All non-authorized inputs remain exact; malformed/cancelled/stale solutions cannot apply; added alignment tasks act through real model components and show an actual measurable forward response.

### WP-10 — Sensitivity, robustness and geometry experiments

**Goal:** Compare design choices systematically while keeping experiments detached from the active microscope.

**Extend:** Existing Design Explorer, experiment recipes/sweeps and component/TOML editing. [S17]

**Implement:**

1. Add bounded one-dimensional response curves, two-dimensional parameter maps and local sensitivity summaries for supported registered controls.
2. Add repeatability/robustness studies for explicitly specified perturbations. A simulated tolerance/noise distribution is a declared assumption, not an OEM specification.
3. Include failed points, failure reasons, numerical status, constraints and objective values. Do not drop unsuccessful points to produce a misleading smooth optimum.
4. Compare multi-objective candidates, including current, size, angle and sensitivity. Let users inspect trade-offs; do not define a universal “best” microscope setting from one metric.
5. For geometry variables, create detached candidate TOML/assembly definitions. Validate topology, clearances and active field dependencies before calculation. Never broaden the runtime sweep setter into an unchecked geometry editor.
6. Separate `Fixed-control comparison` from `Optimize selected controls for each candidate`. Optimization is explicit, and each candidate retains its solved physical controls and numerical evidence.
7. Reuse only truly compatible upstream assets/products. Geometry or field-support changes invalidate every affected stage.
8. Save experiment identity, candidate inputs, results, status and resumable completed points. Resume requires identical relevant configuration and implementation; changed inputs create a new experiment.
9. Export lightweight tables/plots and an input/provenance manifest. Bulk arrays stay in local artifacts. Applying a candidate is a separate explicit operation, not a side effect of selecting it.

**Acceptance:** Fixed-control studies preserve exact operating settings; geometry candidates do not alter live TOMLs; failures remain visible; cancelled experiments resume without mixing identities; viewing/exporting a candidate performs no physical edits.

### WP-11 — Integration and release discipline

**Goal:** Deliver usable features, not disconnected classes, and state precisely what was validated.

For every slice, run relevant tests, inspect the actual UI integration, update current documentation, preserve stable requirement IDs, and append a concise changelog/progress receipt. Do not claim the entire plan completed from unit tests alone.

Before declaring the core complete, exercise this path in an isolated environment:

```text
Open application
  -> choose an assembly
  -> inspect a compatible candidate or validated working point
  -> calculate a bounded particle case
  -> inspect numerical evidence and selected-plane metrics
  -> change one physical control and see the correct reuse plan
  -> attempt an alignment, including a failing/cancelled case
  -> pin and compare two independent results
  -> export a complete working point
  -> restore from a relocated archive
  -> rerun compatible transport and compare scientific observables
```

Build/install checks follow the current repository's packaging workflow, using its pinned validation environment when needed. Inspect acceptance scripts before running them: an historical omnibus script is not permission to restart paused long-wave work. Report CPU, GPU, offscreen GUI, visible GUI, archive/restore and packaging evidence separately. [S20, S21]

## 7. Acceptance test matrix

The IDs below are **new acceptance identifiers**, not claims that test files already exist. Map each to existing tests where suitable; create focused new tests only for uncovered behaviour. Every delivered package must reference actual collected test cases and receipts.

| ID | Test scenario | Required outcome |
| --- | --- | --- |
| AT-01 | Attempt to calculate with an independently configured downstream source | Entry is rejected; historical viewing remains available. |
| AT-02 | Capture a full working point with disabled hardware, shared references and field arrays | Exact parameter/array/ownership semantics survive round-trip; no missing controls. |
| AT-03 | Edit live state after a request is captured | Worker observes only its captured state; result cannot be labelled current after incompatible edits. |
| AT-04 | View, sort, pin, resize, switch layouts or compare records | No change to physical inputs and no unexpected physics execution. |
| AT-05 | Apply illumination-only control patch | Only declared compatible controls change; all affected downstream results are nevertheless invalidated correctly. |
| AT-06 | Fail/cancel/stale an alignment or restore midway | Exact atomic rollback; preceding complete results remain available. |
| AT-07 | Create highly unequal ray weights | Current and containment metrics follow weights, not survivor counts or display samples. |
| AT-08 | Empty or severely under-resolved transmitted population | Unavailable/insufficient-sampling status, never a perfect zero-size focus or fabricated validation. |
| AT-09 | Change one numerical refinement dimension at a time | Evidence records the actual dimension and difference; cache identities change appropriately. |
| AT-10 | Equal crossover count but different ordered component intervals | Topology gate rejects; distinct Microprobe/Nanoprobe references are not interchanged. |
| AT-11 | Modify a field whose support crosses an apparent stage boundary | All affected checkpoints/products are invalidated even if component centre lies downstream. |
| AT-12 | Add an unknown input field or change vacuum participation | Conservative invalidation; no false reuse. |
| AT-13 | Mutate a pre-ingestion array alias or evict an asset used by a job | Admitted data remain immutable/alive; no use-after-eviction or silent corruption. |
| AT-14 | Move complete archive; remove original paths only in the fixture | Content-pinned compatible restoration succeeds without writing to original live files. |
| AT-15 | Corrupt archive/hash/schema or inject an escaping path | Safe rejection; current state unchanged; no arbitrary code execution or path escape. |
| AT-16 | Change implementation/schema, then open old results | Historical read works; exact replay is gated; migration creates a new identity. |
| AT-17 | Repeat unchanged CUDA plan, then change plan/context | Valid repeat reuses device data; changed identity/context rebuilds it. |
| AT-18 | GPU unavailable/OOM versus numerical or input failure | Only eligible runtime failures may follow fallback policy; strict GPU and invalid-input failures remain visible. |
| AT-19 | CPU/GPU equivalence on the same accepted test inputs | Predeclared metric/coordinate/weight/stop tolerances hold; actual backend is recorded. |
| AT-20 | Many rapid live edits plus an independent experiment | Latest live result wins; experiment input/result identity remains intact; no starvation or oversubscription. |
| AT-21 | A/B records have different planes or observable definitions | Comparison flags incompatibility instead of subtracting misleading values. |
| AT-22 | Reduced plot history versus precise continuation checkpoints | Display decimation cannot change scientific metrics or restart precision. |
| AT-23 | Disable a readout at an intercepting detector | Underlying physical interception/loss is preserved. |
| AT-24 | Run a detached geometry sweep including invalid candidates | Live configuration unchanged; failed candidates retained with reasons; fixed-control mode does not optimize implicitly. |
| AT-25 | Request an unsupported/paused product | Clear scoped status; no silent fallback to a differently modelled product with the requested label. |
| AT-26 | Install/build and open representative isolated UI layouts | Entry points/imports remain valid; no user settings or acquisition records are overwritten. |

### Scientific thresholds

Reuse existing applicable acceptance thresholds and definitions from current configuration/evidence. For a new metric/backend comparison, define absolute/relative tolerances and reasons **before** viewing the candidate outcome. Separate agreement with the unchanged baseline from independent numerical convergence and from physical-model validation.

Do not rely only on screenshots, monotonic current conservation or a stable scalar target: those can coexist with wrong distributions, unsupported model domains or incorrect topology. Do not relax assertion thresholds, remove hard examples, substitute synthetic downstream sources, or globally mark failures skipped to obtain green tests.

## 8. Performance measurement contract

Performance measurements are task outputs to be produced by Codex. **No baseline timings or speed-ups have been established by this document.**

### Required scenarios

| Scenario | Report separately |
| --- | --- |
| Startup and first small particle calculation | Import/JIT/setup versus actual source/column work |
| High-accuracy request with a large already-registered field asset | GUI capture latency, copies/bytes, preparation and peak RAM |
| Repeated same-plan particle calculations | Upload/allocation, kernel, download, wall time and VRAM |
| Relevant single-control edit | Reused/recomputed stages and identity evidence |
| Plot-only interaction and tab/layout changes | UI event latency, mesh/plot rebuild count and physics-call count |
| Complete and partial cache hits | Lookup/hash/I/O cost; origin of reused products |
| Export/import and relocated restore | Asset bytes, archive size, memory, verification and restoration time |
| Bounded sweep with cancellation/resume | Throughput, resource peak, completed points and identity continuity |

Record OS, Python, package/solver identity, CPU, actual GPU/driver/runtime, RAM/VRAM, numerical settings and physical input identity. Detect the execution host rather than assuming the user's usual workstation configuration is available.

Compare identical physics/numerics/precision. Show cold and warm results separately. For short repeatable cases, use at least five warm samples for a median; collect enough interactive samples, preferably twenty or more, before reporting a p95. For expensive cases, declare the smaller sample count and avoid unsupported percentile claims. Synchronize correctly for isolated GPU timing, while also measuring real end-to-end wall time; do not add intrusive synchronization to normal execution solely to simplify a benchmark.

Hard acceptance is structural and scientific: no redundant unchanged-plan upload, no full-sized request copy for already-registered assets, no physics on presentation-only changes, bounded resource use and unchanged supported numerical behaviour. Any absolute responsiveness target should be declared against the measured target host. A speed-up percentage is allowed only when measured with the same benchmark inputs and stated calculation scope.

Keep lightweight benchmark summaries in development evidence. Large timing traces, meshes, arrays and calculation caches remain local/ignored. Failed optimizations should be reported and reverted or kept non-default; never retain a regression solely because the new architecture appears more sophisticated.

## 9. Environment and command discipline

The reviewed README recommends 64-bit Python 3.12 on Windows; the reviewed `pyproject.toml` allows `>=3.11,<3.14`. The existing acceptance workflow uses Windows/Python 3.12 and a CPU validation lock file. These are baseline facts, not instructions to overwrite a newer environment. [S02, S20, S21]

Use the project's existing interpreter. Do not run setup/install scripts, change dependency versions, or replace an environment merely because another Python is globally installed. When an isolated test environment is genuinely required, follow current repository instructions and record the difference.

### Initial read-only inspection (PowerShell)

Run in the repository root:

```powershell
$ErrorActionPreference = 'Stop'
$py = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw 'Project interpreter not found. Locate the existing environment before testing.'
}
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short
git diff --stat
& $py --version
Get-Content -LiteralPath AGENTS.md
Get-Content -LiteralPath pyproject.toml
Get-ChildItem -Path tests -Recurse -File -Filter 'test_*.py' |
    Select-Object -ExpandProperty FullName
```

Read any applicable nested instructions as well. On a non-Windows host, use the corresponding existing environment interpreter and document that visible Windows UI/hardware acceptance was not performed there.

### Test discovery and scoped execution

The previous working-point record references these concrete test files; inspect their **current** contents and availability before choosing cases: [S23]

```text
tests/test_working_point_contract.py
tests/test_calculation_cache_reuse.py
tests/test_execution_migration_contract.py
tests/test_calculation_manifest_artifacts.py
tests/test_segmented_column_cache.py
tests/test_background_calculation_requests.py
tests/test_artifact_quota.py
tests/test_cache_settings_integration.py
tests/test_background_preview_gui.py
```

Do not assume dated passing counts apply now. Collect and run the actual relevant cases with the project's pytest configuration. Inspect tests/scripts before executing anything that could launch long calculations, access hardware or depend on retired wave admission. Isolated mathematical/source-policy tests are not production imaging acceptance.

An existing test filename is not permission to ignore its scope. If no intended tests are collected, that is not a passing test run. Keep test logs and include exit status, failures, errors, skips and deselected counts when relevant. A missing plugin/GPU is an environment limitation to report, not a reason to substitute a fabricated pass.

For GUI tests, use isolated temporary settings and the repository's supported offscreen fixtures. Keep desktop/UI quality and GPU evidence separate from offscreen success. Do not blindly change global `QSettings`, graphics drivers or the user's environment variables to make one test pass.

At the end of each slice, run `git diff --check` and inspect the complete change set. Compile changed Python files or run the repository's bounded compile check. A broad full-suite/release command is appropriate only after its current scope is inspected and compatible with the paused wave/hardware boundaries.

## 10. Progress receipts and session handoff

Create/update `docs/development/PRODUCT_USABILITY_PROGRESS.md`. This path is a **proposed new development record**, not an existing claimed artifact. Keep receipts concise and evidence-linked; do not turn the file into a duplicated narrative of this brief.

Each session records:

```text
Session date and actual start HEAD:
Working-tree differences present before this task:
Current applicable requirements / newer decisions:
Package and acceptance IDs addressed:
Implemented user-visible behaviour:
Files changed:
Physics/default changes: none, or exact authorized changes with reasons
Tests executed: command, collected scope, exit code, actual counts, log path
Tests not run: reason and remaining risk
Benchmarks: input identity, host, cold/warm, sample count, measured outcomes
Numerical / model validation scope:
Known limitations or baseline failures:
Assets/caches created and their local ignored locations:
Task-owned processes: completed/cancelled; no unaccounted background work
Next ready package and exact first implementation/test step:
```

Track packages with `not_started`, `in_progress`, `blocked`, `implemented_pending_validation` or `complete`. “Complete” requires the applicable integration and acceptance evidence, not just source files or a green isolated fixture. Partially blocked packages can expose useful honest diagnostics without claiming the underlying physics was solved.

Update `PROJECT_FUNCTION_SPEC.md` without deleting stable requirements, and append concise release notes to `CHANGELOG.md`. Preserve historical records. New task requirement IDs must not collide with the existing ledger.

### Resumption instruction

```text
Read CODEX_DEVELOPMENT_GUIDE.md and the latest
docs/development/PRODUCT_USABILITY_PROGRESS.md.
Reconcile current HEAD and local changes with the recorded session, then continue
the first ready unfinished package. Reuse verified completed work. Do not repeat
old work or reset user changes. Preserve the stated invariants and paused wave
scope. Finish a bounded integrated slice, run the relevant tests, update the
receipt, and report exact changes, evidence and remaining limitations.
```

## 11. Failure handling and completion criteria

**A scientific prerequisite failed:** preserve the rejected candidate and diagnostics. Do not relax gates, fit a new source, or call a bounded search failure a proof of impossibility. Continue independent UI/data work where safe.

**The environment cannot run a required test:** state what was not tested. Use valid lower-scope tests where available without relabelling them as hardware/production acceptance. Do not change unrelated dependencies speculatively.

**A baseline test already fails:** reproduce on the unchanged relevant baseline where feasible, record the scope and distinguish it from a regression. Fix task-caused failures. Do not suppress unrelated failures without explanation.

**A proposed optimization does not improve measured performance:** preserve correctness, report the measurements, and avoid making it the default solely on theoretical grounds.

**A session ends before the plan is finished:** leave usable bounded changes, actual receipts and an exact continuation point. Do not claim whole-plan completion or promise work outside the active session.

Core completion requires integrated P0/P1 work, all applicable invariant checks, honest capability status, usable working-point restoration/comparison, responsive isolated UI evidence, measured performance behaviour and maintained CPU fallback. WP-10 completion is reported separately. Unresolved physical-source qualification may remain explicitly unresolved; it is not acceptable to call an unvalidated default validated, and it does not prevent correctly implementing a diagnostic/management feature.

## 12. Source map

Repository-relative paths below identify the reviewed primary evidence. They are not external operating instructions. Resolve them against the reviewed commit for historical comparison, and against the current checkout for implementation. Avoid hard-coding line numbers that will drift.

| Reference | Repository file(s) |
| --- | --- |
| S01 | `AGENTS.md` |
| S02 | `README.md` |
| S03 | `PROJECT_FUNCTION_SPEC.md` |
| S04 | `CHANGELOG.md` |
| S05 | `docs/development/assembly-illumination-progress-2026-09-17.md` |
| S06 | `docs/development/PROBE_CHAIN_AUDIT_2026-09-14.md` |
| S07 | `src/temsim/gui/beam_analysis.py` |
| S08 | `src/temsim/gui/direct_alignment_panel.py`; `src/temsim/alignment_transaction.py` as an implementation entry point |
| S09 | `src/temsim/gui/calculation_request.py` |
| S10 | `src/temsim/instrument_snapshot.py` |
| S11 | `src/temsim/artifact_store.py` |
| S12 | `src/temsim/physics/ray_integrator.py` |
| S13 | `src/temsim/physics/compute_backend.py` |
| S14 | `src/temsim/calculation_cache.py` |
| S15 | `src/temsim/gui/main_window.py` |
| S16 | `src/temsim/gui/interactive_controller.py` |
| S17 | `src/temsim/design_experiments.py` |
| S18 | `src/temsim/calculation_performance.py` |
| S19 | `src/temsim/gui/calculation_controller.py` |
| S20 | `pyproject.toml` |
| S21 | `.github/workflows/tem-p0.yml` |
| S22 | `src/temsim/gui/working_point_panel.py` |
| S23 | `docs/development/WORKING_POINT_V2.md` — dated implementation history; current code/scope takes precedence |
| S24 | `src/temsim/optics/electron_gun/electrostatic.py` |
| S25 | `src/temsim/gui/instrument_recorder.py` — actual acquisition integration, excluded from this task |
| S26 | `src/temsim/physics/core.py` |
| S27 | `src/temsim/simulation_pipeline.py` |

**Final reminder:** Preserve the scientific chain and existing capabilities, reduce avoidable work rather than physical fidelity, and make the result's identity and limitations visible to the user.
