# New tip-source full-product acceptance work

**Current status: paused by the user.** Current work is limited to classical
particle tip geometry; see [the particle-tip guide](PARTICLE_TIP_GEOMETRY_2026-09-13.md).
The original request and pending acceptance table below are historical context,
not authorization to restart wave propagation or a claim of full completion.

User request: continue until the new source supports Ray Diagram and all
signals available from the historical source. Ask the user before stopping
while required work remains. This document is a work record, not a PASS.

## Preserved scope

Only the physical tip supplies configurable emission. Extraction, acceleration,
focusing, stops, specimen interactions, downstream optics and detector
absorption remain active. Complex fields and phase references are retained per
mode. Historical results remain readable; no profile is silently converted.
The energy filter is integrated last, and a path reaching it cannot bypass it.

## Required checks

| Product / behavior | Status | Required evidence |
| --- | --- | --- |
| Tip-to-gun complex propagation | In progress | Independent radial, axial and interface convergence; every energy |
| Ray Diagram / transverse diagnostics | Pending new-source acceptance | Real executed tip chain; position, angle, interaction and intensity modes |
| TEM image / diffraction | Pending | Sample plus physical projection lenses and camera/screen |
| STEM BF / DF / HAADF and scan | Pending | Actual scan positions, detector intersections, mode weights and specimen contrast |
| Per-mode detector phase | Pending | Full complex phase and references, no phase assigned to an incoherent mixture |
| EDS / specimen interactions | Pending compatibility | Physical incoming source, configured material and detector geometry |
| EELS / outgoing inelastic waves | Pending compatibility | Same incident chain; retained energy loss, phase and downstream transport |
| Cache / live tuning | Pending | Input-bound executed segments; no stale source or lost completed results |
| Energy filter and downstream detector paths | Last stage | Real filter operators; no bypass of an installed intersected filter |
| Historical source / saved profiles | Pending regression | No silently changed source inputs or rewritten historical results |

## Current implementation

`occupied_axial_refinement.py` adds optional source-specific CF4 step-doubling.
It keeps all scattering-matrix channels and both incoming ports. The actual
two-port inputs come from a newly solved tip/gun boundary-value problem.
Each mesh update rebuilds the complete reflected load and solves the driven
tip again. Cached refinement trees are rechecked after the input changes.
The fixed chart and physical component schedule are not altered by refinement.

The error indicator is sqrt(k_ref) times the complex two-port action difference,
relative to the unit-flux reservoir convention of the joint solver. The sum of
local indicators and the change of physical complex fields/derivatives at all
original boundaries are checked separately. This is not a rigorous universal
operator bound or a replacement for independent radial/full-chain convergence.
Operators cannot be certified for a different source just because they passed
the occupied estimate for the current one.

The new option is off by default while under validation. Production source
admission has not been opened. CPU reference tests compare the re-solved
two-way solution with an independent DOP853 transfer solution, exercise
incoming reflected waves, source changes, work/memory limits and cancellation.

Initial scoped receipt: 32 passed, no failures/errors/skips, unchanged inputs:
[receipt](evidence/20260913T153556Z-occupied-axial-initial-02044e58/report.json).
This is not a completed physical-gun or image qualification.

The grounded classical source, source contract, checkpoint observables and
multi-detector bookkeeping scoped suite passed 68 tests with unchanged inputs:
[receipt](evidence/20260913T154406Z-source-classical-preservation-087522d5/report.json).
An earlier invocation named nonexistent test files and ran zero tests; that
failed invocation is retained and is not counted as validation.

`scripts.freeze_physics_run` can retain a verified source/configuration copy
for long reference calculations while development continues in the checkout.
The driver still captures and validates the actual instrument and external
inputs. The frozen copy is neither source admission nor a new beam source.
Its three copying/identity regression tests passed:
[receipt](evidence/20260913T154811Z-frozen-physics-inputs-bf963ed9/report.json).

Initial physical run: `20260913-occupied-joint-quartic01`, three energy modes,
64 radial modes, P2 769 by 257 near-tip mesh, numerical domain factor 5,
one quartic wave-following update, occupied axial amplitude budget 0.001.
Physical source, voltages, lenses and apertures are unchanged. This run must
be checked against an independent coordinate and finer numerical budgets
before its current or complex field can qualify image production.

That local serial run was interrupted before producing a source (see its
`INTERRUPTED.md`). The independently frozen 0.075-width run was preserved.
`20260913-parallel-occupied-joint01` now uses eight bounded interval workers
and initial CF4 slabs against the same physical inputs. Both runs remain
unqualified until their complete evidence exists and convergence is checked.

Parallel intervals own separate refinement trees, use the same globally solved
reflected inputs, share work/memory limits and collect results in physical
order. The 74-case combined regression suite passed, including bitwise equal
serial/parallel complex states and independent potential quadrature:
[receipt](evidence/20260913T160117Z-parallel-batched-gun-8643034b/report.json).
Batched analytic potential moments retain every consumed field knot and both
coordinate products. The single-thread isolated kernel comparison was only
1.18 times faster; this is not a measured full-gun acceleration.

Exact interior static condensation is now available for repeated joint tip
boundary solves. Its cached response is keyed by the complete assembled
matrix and driven RHS, not by a source label. Each new downstream admittance
is solved afresh, with reconstruction and residual checks of the entire
complex interior field. It retains one energy/system response at a time.
The 22-case scoped suite passed, including direct-versus-condensed comparison
with the actual grounded-tip field and a complete coarse round-gun load:
[receipt](evidence/20260913T160851Z-condensed-coupled-tip-01332d2c/report.json).
That one-energy coarse equivalence check is not a convergence certificate.

## Continuing execution and resource repair

The two old serial/threaded frozen diagnostics were subsequently stopped
before producing a source. Their `INTERRUPTED.md` files explain why; their
inputs and evidence remain intact. The eight-process run
`20260913-process-occupied-joint01` failed at its 24 GiB retained-operator
budget after 591.24 seconds while refining an unpublished coordinate pilot.
It produced no complete source or qualified current. Its implementation was
unchanged. This is a recorded failure, not a successful physical computation.

The final physical chart is now always strictly refined when occupied control
is enabled, including the zero-pilot case. Unpublished coordinate pilots are
not refined by default; an explicit numerical option can refine them too.
No pilot current is admitted as a final source. Accepted refinement leaves
retain their complete complex coarse/fine difference operator instead of two
comparison children. Changed incoming reflected/occupied channels are tested
again against that full difference; children are regenerated if needed.
No physical wave channel is removed. Process transfer is bounded and streamed,
with shared evaluation/memory limits and physical-memory reserve checks.

Relevant unchanged-input receipts (overlapping scopes; do not sum counts):

- [34 checks](evidence/20260913T163535Z-compact-occupied-source-03805601/report.json):
  compact comparisons, serial/process equivalence, source rechecks, exact
  tip-boundary condensation and wave-following coordinate checks.
- [20 checks](evidence/20260913T164255Z-complete-energy-evidence-0ab44a4a/report.json):
  complete-energy persistence, actual coarse physical tip/gun output and
  existing near-field/profile integration. Not convergence certification.
- [20 checks](evidence/20260913T164533Z-refinement-failure-evidence-1dce1b2e/report.json):
  failure diagnostics, memory reserve and retained complete-energy evidence.
- [46 checks](evidence/20260913T164816Z-smooth-coordinate-verification-88210305/report.json):
  coordinate derivatives, independent complex free-wave comparison, quartic
  operators and source-specific refinement. Not full physical-gun acceptance.

`inspect_surface_column` now saves each completed energy's full near field,
complex boundary coefficients and covariant derivatives, coordinate history,
exit envelope and phase references. Atomic non-overwriting evidence writes
remain readable if a later energy fails. There is no import route from those
partial files into an active source. A complete energy is not the complete
energy mixture or an image acceptance result.

Two physical diagnostics were launched and require live result inspection:

- `20260913-compact-final01`: unsmoothed final chart, eight processes, 40 GiB.
- `20260913-smooth-final01`: final numerical chart smoothing width 0.05 in
  log Z, four processes, 16 GiB, with complete-energy evidence enabled.

Both retain the same three-energy physical source, 769 by 257 P2 near-tip
mesh, 64 radial modes, 395 physical masks per energy, extraction, acceleration,
gun focusing and shared-column fields. The initial slabs use CF4 and the
occupied axial amplitude tolerance is 0.001. Do not infer convergence from
the execution count, a conserved current, or these numerical inputs.

Optional smoothing changes only the numerical coordinate proposal. It uses
a fixed-width curvature-penalized spline for log radial width and scaled
quadratic/quartic coordinate phase. It never smooths the transported wave,
changes the tip or fits a target current. All transformed derivative terms
remain active, and the complete reflected tip/gun problem is re-executed.
The algorithm follows the fixed-parameter objective documented by
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html).
Its production default is zero/off. Independent coordinate, radial, axial
and complete-product comparisons remain required.

The unsmoothed compact-leaf run then failed its 200,000-evaluation limit
after 1133.05 seconds. The smoothed compact-leaf run failed its 16 GiB
retained-operator budget after 805.77 seconds and 87,792 evaluations. Neither
completed an energy or a source. Its retained-tree diagnostic places large
refinement trees at approximately 109--118 nm and 295--314 nm; a 3 nm root
interval sometimes requires 256 leaves under the unchanged local criterion.
This is not evidence that physical aperture losses are now correct.

### Bounded full-port error records

`occupied_tree_certificate.py` leaves an executed interval's complete complex
scattering matrix unchanged. For each adaptive leaf, its coarse/fine
difference is mapped to the root's complete incoming ports, including all
internal reflected couplings. If M_i is this map and f_i is the leaf's fraction
of action length, Cauchy--Schwarz gives

`sum_i ||M_i x|| <= ||R x||`, where `R` is the QR factor of the vertically
stacked matrices `M_i / sqrt(f_i)` and `sum_i f_i = 1`.

Only R, the full executed root operator and the exact partition are retained.
This conservatively bounds the previous sum of local indicators for any new
root input. Failure regenerates and refines the saved partition; a new source
is never fitted or injected. This numerical-error certificate is **not** a
certificate of continuum accuracy, full-source qualification or image quality.

[26 scoped checks](evidence/20260913T170029Z-full-port-error-certificates-bf0de883/report.json)
passed with unchanged inputs, including independent direct-ODE comparison,
random full-port checks of the conservative bound, unchanged physical root
operators, changed-source re-execution and serial/process agreement.

`20260913-certified-final01` uses this bounded error storage, eight processes,
40 GiB and a one-million-evaluation limit. Its directory name refers to the
local error records, not an accepted source. It must be inspected live.
All physical inputs and the 0.001 amplitude error target remain unchanged.

The previously proposed full-gun engineering comparison remains less than
1% current and 1% unaligned complex L2 discrepancy **per energy**, under
independent coordinate, radial, near-interface and electrode-grid refinements.
These are initial numerical qualification targets, not OEM accuracy claims,
and do not supersede tighter existing operator or flux-ledger tolerances.
Full TEM/STEM and other product checks still follow that gun qualification.

## Continuing numerical comparisons

`20260913-certified-final0075` is an independent 0.075 chart-width execution
using the **same immutable solver directory** as the 0.1-width reference.
`scripts.reuse_frozen_physics` checks every frozen input before and after the
run and writes a separate receipt and output checksums. It does not import a
source. Reusing exact configuration paths keeps comparison identities intact;
it does not weaken the active instrument snapshot restoration rules.
The runner and certificate regression scope passed
[8 checks](evidence/20260913T171152Z-frozen-comparison-reuse-35931d47/report.json).

An optional CF6:5Opt integrator follows Table 6 and Eq. (60) of
[Alvermann and Fehske (2011)](https://arxiv.org/abs/1102.5071).
Four action-Gauss samples form five complete covariant two-way exponentials.
No physical aperture is traversed twice or omitted by its signed numerical
substeps. The internal evanescent fallback now bounds seed growth by absolute
step length while retaining the signed exponential. Public physical layer
width validation remains positive. The initial 12-case suite found two signed
fallback failures; that failed receipt is retained. After correction,
[20 checks passed](evidence/20260913T171735Z-sixth-order-signed-slabs-042087ed/report.json),
including sixth-order complex-ODE convergence, reversible full ports and an
independent negative-step matrix exponential with evanescent coupling.
These results are not a full physical-gun convergence certificate.

`20260913-cf6-actual-gun-audit` re-executes the same physical tip and complete
gun, then observes the first energy's actual incoming ports at approximately
100, 300, 400 and 1000 nm. It compares CF4 and CF6 at increasing subdivisions
without phase, current or source fitting. This deliberately partial diagnostic
publishes no source. Inspect its final evidence before drawing conclusions.

That local diagnostic completed in 630.24 seconds, with unchanged inputs:
[raw comparison](evidence/20260913-cf6-actual-gun-audit/integrator_audit.json).
At 100--100.844279 nm, CF6 with 64 subdivisions differs from CF6/256 by
1.12e-8 in the occupied complex ports; CF4/256 differs by 6.35e-9. Their
measured interval times were 7.81 and 12.63 seconds respectively. At about
300 and 400 nm, even CF4/256 still differs from CF6/256 by 1.24e-5 and
6.43e-5. The finest comparison is not a certified continuum reference.
Higher order helps but does not establish whole-gun convergence by itself.

The optional `initial_indicator` error-budget allocation preserves the **same
total tolerance**. It first measures each full coarse/fine complex difference,
then allocates positive interval budgets using the cost heuristic
`t_i proportional to e_i^(1/(p+1))`, reserving 5% uniformly. Every strict local
check, reflected-boundary re-solve and global complex stability check follows.
No result can be published by the allocation-only probe. Uniform allocation
remains the default and is what the two ongoing CF4 references consume.
An initial edit-placement failure is retained in the test receipts; the
corrected scoped suite passed
[39 checks](evidence/20260913T172630Z-fixed-total-axial-budget-corrected-455d05d8/report.json).

The expanded allocation, CF6 and serial/process equivalence scope passed
[50 checks](evidence/20260913T172802Z-allocated-budget-parallel-equivalence-2349ce44/report.json).
The analytic radial disk-integral recurrence now has an optional compiled
double-precision implementation, with fast-math disabled and the original
NumPy implementation retained. All radial boundaries and polynomial orders
remain. Its independent quadrature, wide-domain and fallback scope passed
[33 checks](evidence/20260913T173156Z-compiled-analytic-radial-moments-b678faa2/report.json).
An isolated warm-kernel measurement at 66 modes and 16 radial boundaries
was 2.028 ms for NumPy and 0.06537 ms compiled (31.0 times). This is not a
whole-gun or image speedup measurement.

`20260913-cf6-allocated-fast` is running from a new immutable solver copy,
with the same physical source, three energies, 64 radial modes, 769 by 257
P2 near-tip mesh and all physical stops. It uses CF6 occupied refinement,
the same total 0.001 amplitude tolerance, initial-indicator budget allocation,
eight processes and a 24 GiB working budget. Its initial chart construction
still uses CF4. Inspect its completed evidence; no acceptance is implied by
the directory name, a running progress counter or this numerical setup.

### Full-pipeline numerical controls and resumable energy modes

The development TEM/STEM driver now accepts the same surface element order,
radial basis, coordinate and occupied-refinement controls used by the gun
diagnostics. It records them in the actual `TipWaveRequest`; they therefore
participate in executed cache identity. A configured coherent surface is
described as such, without calling the historical planar emitter validator.
The profile is not rewritten. The CLI/near-field scope passed
[30 checks](evidence/20260913T174027Z-full-tip-pipeline-numerical-controls-a558545a/report.json).

The exact scattering-interface composition now factors its shared matrix
once for both full right-hand sides. The carrier eigensystem uses its known
triangular Cholesky factor directly. No channel, operator or tolerance was
removed. The reference/ODE/refinement scope passed
[64 checks](evidence/20260913T174337Z-reuse-exact-interface-factorisation-3cf38faf/report.json).

`surface_mode_cache.py` saves a complete executed energy with the existing
atomic, dependency-bound wave store. The exit mode, phase reference, near-tip
field, all complex boundary coefficients/derivatives and numerical coordinates
are committed together. Source/optics/code/numerics/energy identity is checked
before reuse; partial energies are not indexed. The main builder still needs
every energy and all original flux/domain checks before returning a gun.
Read-only development NPZ exports are not admitted as active sources.

The actual coarse-gun resume test interrupted after energy one, reused that
energy without propagation, computed the remaining two, and then reproduced
the identical complete checkpoint and near-field digests from cache. Changing
extractor voltage required new propagation. Cache/checksum/phase/segmentation
scope: [20 checks passed](evidence/20260913T174815Z-complete-tip-energy-resume-0f18b11e/report.json).
This is cache correctness on a coarse physical run, not source convergence.

Further live diagnostics:

- `20260913-far-integrator-audit`: actual first-energy intervals between
  10 micrometres and 400 mm; compares CF4/CF6 subdivision sensitivity. It
  intentionally publishes no source.
- `20260913-cf6-smooth`: same frozen solver and physical inputs as
  `20260913-cf6-allocated-fast`, but numerical chart smoothing 0.05 log-Z,
  four processes and 16 GiB. Every physical operator and the same 0.001
  total amplitude criterion remain active. No full result exists yet.

An isolated RTX 5090 five-matrix 128-by-128 complex eigensystem benchmark was
slower than the single-thread CPU call in the currently contended environment
(about 101 ms versus 31 ms), and is not adopted as a gun acceleration path.
This does not generalise to large grids, ray transport or image FFTs.

The far-gun diagnostic completed in 458.10 seconds with unchanged inputs.
The original CF4 interval at 100--100.5 mm differs from CF6/16 in actual
occupied complex ports by 0.21113; near 10 mm the difference is 0.57687.
These are discrete-comparison fractions, not measured current errors, and
CF6/16 is not certified as converged. The defect cannot be addressed solely
by refining the first few hundred nanometres. See the extended
[current-cause analysis](GUN_CURRENT_CAUSE_2026-09-13.md).

### Independent global-mesh strategy under test

`global_embedded_refinement.py` provides a separate development strategy.
Every candidate halves all retained propagation substeps and solves the
complete reflected gun/tip boundary again. It checks the unfitted complex
field and physical normal derivative at every original plane, plus the
maximum occupied full-port local difference. Two genuinely successive
whole-mesh refinements must satisfy the selected tolerance. No Richardson
extrapolated field is injected. Failed checks mark the largest local
differences for finer execution; this marking is a cost heuristic only.

This is **not** the same criterion as `local_sum` and does not claim that
the sum of local indicators meets that strategy's L1 budget. Both retain
the same physical operators and all wave channels, but estimate errors
differently. The existing local-sum references continue independently.
Production admission remains closed pending actual convergence evidence.
The distinction between local defect control and global error estimation,
including the limitations of coarse-grid estimates, is discussed by
[Boisvert, Muir and Spiteri](https://www.cs.usask.ca/~spiteri/BoisvertEtAl2013.pdf).
Our implementation is a direct complex two-mesh diagnostic, not their
Runge--Kutta code or its proven error controller.

The global/local refinement and CLI scope passed
[54 checks](evidence/20260913T180206Z-global-complex-mesh-comparison-785bd072/report.json).
`20260913-global-embedded01` is now running the complete physical three-energy
gun with this separate strategy: 64 radial modes, 128 quadrature, P2 769x257
near field, domain factor 5, 2048-pixel exit, CF6 refinement, 0.001 tolerance,
48-round/two-million-evaluation limits, four processes and 12 GiB. It uses a
192-GiB complete-energy cache. No energy had finished at the latest check.
The earlier local-sum references remain active; no admission gate is open.

### Full local-current observables (not yet a GUI source)

The radial gun now saves the exact one-sided width/phase coordinate rates at
every boundary, including both sides of coincident physical stops. The
complete-energy cache is versioned as v2 to require this state. Historical v1
files remain readable; missing rates are not silently reconstructed or zeroed.

`radial_wave_observables.py` reconstructs the complex field and full derivative
`B_P d + i B_Q G_QP a`. The second term is essential for the local current,
although it integrates to zero by orthogonality. This is differentiation of
the existing finite-basis field, not an added source or propagated mode.
Scalar intensity, signed axial current density and per-mode phase remain
distinct; incoherent mixtures sum observables and have no aggregate phase.
Reading arbitrary Z is deliberately not done by interpolating across stops.

Independent finite-difference derivatives, integrated flux, standing waves,
mixtures and actual coarse-gun history/cache checks passed
[35 tests](evidence/20260913T181800Z-radial-full-current-observables-573e8c28/report.json),
with unchanged hashed inputs. Serial compilation passed. This establishes
readout/state bookkeeping, not converged source or Ray Diagram acceptance.

### Fixed tip boundary assembly reuse

`joint_boundary_assembly.py` reuses volume, physical reservoir and interface
matrices under a digest of all consumed mesh/potential/drive/sparse arrays,
energy and matching-basis coordinates. It retains one assembly per running
energy and charges retained arrays to the same numerical memory allowance.
Each changed downstream admittance is still solved with the complete driven
tip and all internal complex values reconstructed. No previous boundary
field or current is frozen. A changed physical drive or numerical interface
invalidates reuse even when the Python problem object is the same.

The initial attempt had an indentation failure in three tests (49 other
checks passed); that failed receipt is retained. After repair the entire
scope passed [52 checks](evidence/20260913T183001Z-fixed-tip-boundary-reuse-verified-09219dbe/report.json).
This is cached/direct solver equivalence and source identity validation,
not evidence that the long gun references have converged.

The fixed-input format scope subsequently passed
[20 checks](evidence/20260913T183302Z-fixed-tip-input-format-caa31133/report.json).
Exact right-to-left reflected-load suffix reuse passed
[40 checks](evidence/20260913T183647Z-exact-reflected-suffix-reuse-9e4ff1e4/report.json).
Changed operators invalidate every dependent upstream response, while the
driven tip is solved again. No previous source trace/current is reused.

### Round-column to Cartesian continuation

The development pipeline now executes its axisymmetric gun-output prefix in
the full radial representation, then switches to the existing Cartesian
operator before any non-axisymmetric event. The retained quadratic carrier,
mode identity, energy, scattering history and absolute source weight survive.
Axial flight time and longitudinal action advance over the actual prefix
length; dynamic coils therefore continue from the executed arrival clock.

`radial_cartesian_handoff.py` checks two successive doubled-grid complex
comparisons without fitted phase, plus current against the radial integral.
Finite-window and interpolation errors are numerical diagnostics, not
physical absorption. Neither current nor phase is adjusted to pass.
The initial implementation failed three absolute-phase tests because the
cell-centred lattice changed the analytical carrier's origin. Its exact
constant/linear/quadratic coordinate transformation was corrected, and the
unchanged test tolerances then passed. This is not a new physical tilt.

The round-field, clock and phase scope passed
[39 checks](evidence/20260913T185119Z-round-prefix-clock-phase-2c576479/report.json).
Pipeline routing, cached/direct phase and storage identity subsequently passed
[33 checks](evidence/20260913T185352Z-round-prefix-pipeline-storage-verified-68eb01c4/report.json).
The routing tests are explicitly mocked fixtures, not completed physical
TEM/STEM images. Serial compilation passed. Production source admission is
still closed; the independent full-gun convergence runs remain unfinished.

### Cached transverse wave display and current-coordinate investigation

Forward-column current, complete-carrier canonical angles, exclusive recorded
interaction weights and individual-mode phase now have read-only observables.
The existing transverse panel can present these through its internal
`display_wave_checkpoint` method. It does not generate trajectories, resample
missing Z planes, sum incoherent phases, execute propagation or admit a source.
Returning to the existing particle result restores its controls. User ranges
survive Z/projection changes and newly displayed checkpoints; only Fit beam
resets them. Derived mode buffers are bounded. This is not yet main-calculation
dispatch or a successful new-source Ray Diagram.

Relevant receipts, overlapping scopes (do not sum):

- [21 observable checks](evidence/20260913T185917Z-wave-current-angle-observables-b276518f/report.json).
- [32 CLI/readout checks](evidence/20260913T190140Z-wave-source-readout-cli-e33a2f74/report.json).
- [44 process/refinement checks](evidence/20260913T191113Z-balanced-refinement-processes-verified-d3937242/report.json).
- [39 offline GUI/readout checks](evidence/20260913T191651Z-cached-wave-transverse-view-96fab84b/report.json).
- [60 combined checks](evidence/20260913T192222Z-cached-wave-view-bounds-and-carrier-a603dea9/report.json).

The old local-sum reference `20260913-certified-final01` failed at one million
evaluations after 6900.43 s without a completed energy; no source was published.
The global coordinate-0.075 reference `20260913-global-embedded0075` failed at
a constant-slab current residual of 1.14316e-9, above the unchanged 1e-9 limit.
Its failure receipt and original frozen solver remain intact. The independent
coordinate-0.1 global reference continues; its last completed round 13 had a
maximum unfitted complex change of 0.1855, not a current-change percentage.

`carrier_subspaces.py` supplies a separately checked full invariant-graph
coordinate solve if the spectral current chart fails. The two subspaces are
J-orthogonal graphs [I;X] and [X*;I]. The complete Riccati equation is refined
without adding the small residual to the large carrier. Both directions,
absolute phases and all channels remain. A step-scaled invariance residual
and the original current thresholds are enforced; no S-matrix singular values
or transmitted currents are clipped or renormalised. The initial version
failed the long, tiny-transverse-phase test; solving the invariant equation
repaired that failure at unchanged tolerance. The relevant scope passed
[51 checks](evidence/20260913T192117Z-carrier-current-graph-refined-42d71e5b/report.json),
including independent matrix-exponential, signed-step and small-phase tests.

`20260913-current-graph0075` now re-executes the complete physical reference
with the same source/optics and axial criteria. It also uses bounded cost-aware
process dispatch and exact fixed-boundary/reflected-suffix caches. Failed slabs
retain their actual matrices for isolated reproduction. No complete energy or
full-chain acceptance has yet been reported from this run.

### Current projection and Windows worker transport follow-up

The coordinate-0.075 old local-sum run also exhausted one million evaluations
after 7946.06 s. Both original failure receipts remain intact. Neither run
published a complete energy. The new graph-coordinate 0.075 run failed after
463.65 s with a broken process pool. Its 0.1 counterpart subsequently exposed
the concrete transport failure: Windows `WriteFile` on a multiprocessing
result pipe raised `WinError 1450`. This is a resource/transport failure, not
evidence of a converged or divergent physical source. A logged repeat of
0.075 is running against the identical frozen solver.

Reference runners now retain untruncated stdout/stderr alongside their
receipts. [Six offline runner checks](evidence/20260913T193607Z-untruncated-physics-driver-diagnostics-7cea05de/report.json)
passed, including error output and no-overwrite behavior.

`radial_current_projection.py` evaluates the all-Y projection of the complete
local axial current, including the two outside-basis derivative coefficients.
The common radial phase cancels in this current product; Gauss-Hermite
quadrature integrates the remaining polynomial. Square-root weights are
computed without first underflowing the weights at high order. The method
uses the Gaussian quadrature conventions in
[NIST DLMF 3.5(v)](https://dlmf.nist.gov/3.5#v). It neither samples particles
nor substitutes scalar wave intensity for current. Signed currents, original
source weights, both sides of coincident physical stops, and the unmodified
full transverse current survive visible-window selection. Missing historical
chart rates are explicit; no interpolation across masks or forced current
normalisation is used.

The projection/observable scope passed
[22 checks](evidence/20260913T193840Z-radial-current-projection-0855073d/report.json);
the cached-history and existing transverse-panel scope passed
[31 checks](evidence/20260913T194104Z-cached-current-diagram-3725b63d/report.json).
These include independent real-space quadrature, analytic Gaussian/backflow,
high-order and bounded-chunk checks. They do not admit a source or constitute
a full physical Ray Diagram. The zero-current round-column fix passed
[24 checks](evidence/20260913T192628Z-extinguished-round-column-9b9ed412/report.json).
Scopes overlap and must not be summed. Production admission remains closed.

The file-backed private worker transport passed
[58 numerical/process checks](evidence/20260913T194535Z-file-backed-wave-process-transfer-7148c69c/report.json).
Complete complex results and detailed solver exceptions now travel in owned
temporary files; pipe messages remain small. Process and threaded numerical
results are compared at unchanged tolerances. A new frozen 0.1 reference,
`20260913-file-transport01`, is executing this path. Temporary files are not
source checkpoints and cannot be imported as a replacement beam.

The logged spectral/graph 0.075 repeat has passed round 9, where the older
solver failed its slab-current check; its full complex difference was still
0.4155 at that point. Passing that interval is not full convergence.

Physical detector readouts now retain each mode's energy, scattering history
and **received** optical weight after sensor geometry/pixel acceptance. A
streamed readout combines all modes, rather than keeping the first mode's
metadata. Phase-only requests do not invent collection counts. This is a
readout of the executed waves, not an energy-filter spectrum or a new EDS
transport. The scope passed
[42 checks](evidence/20260913T195022Z-detector-mode-collection-identity-21002d7f/report.json).

An optional direct Riccati initialisation of the same constant-slab graph was
tested; the default spectral path is unchanged. A short-step test initially
exposed a 6.44e-12 absolute discrepancy. Tightening the Newton stopping target
from 1e-10 to 1e-12 repaired it at unchanged comparison tolerances; the scope
passed [54 checks](evidence/20260913T195432Z-direct-riccati-slab-refined-68082ef0/report.json).
A [retained actual-slab timing](evidence/20260913-direct-riccati-actual-slab/report.json)
does **not** demonstrate a speedup: that non-propagating slab routes all three
choices to the same full doubling solver (identical complex results). Its
different wall times are runtime variation, not algorithmic improvement.

### Local axial mesh and display follow-up

The existing tip/ray/profile/cache/readout compatibility scope passed
[242 checks](evidence/20260913T195649Z-source-ray-cache-compatibility-e4b544a8/report.json).
This is scoped offline regression coverage, not a full-suite or full-image claim.

Fixed 128-substep chunk grouping now distributes a long numerical interval
across processes. Serial and process composition use the identical physical
order. The scope passed
[50 checks](evidence/20260913T200325Z-embedded-interval-parallelism-7e7f8d6d/report.json)
and [15 boundary-diagnostic checks](evidence/20260913T200616Z-embedded-boundary-error-location-1b936e6d/report.json).
The frozen `20260913-chunked-refinement0075` reference is running this method.

An additional `spatial_embedded` development strategy refines numerical leaves
inside a physical interval. All fixed interfaces, apertures and their two faces
remain in place. Each comparison halves every retained leaf, re-solves the
driven tip against the complete reflected load, and compares unfitted complex
fields and derivatives at every original physical boundary and every matched
coarse numerical boundary. Two consecutive uniform refinements must pass the
unchanged 0.001 criteria. Bulk marking only chooses where to spend work;
neither local indicators nor small current changes replace these criteria.
No Richardson field extrapolation, flux correction or source retuning is used.

The independent adaptive-BVP scope passed
[9 checks](evidence/20260913T201958Z-spatial-embedded-full-bvp-0ffd1c49/report.json).
The expanded physical-stop/process scope passed
[26 checks](evidence/20260913T202134Z-spatial-embedded-physical-stops-verified-7758e004/report.json),
and shared-sampler retention passed
[16 checks](evidence/20260913T202355Z-spatial-worker-shared-state-6a628a67/report.json).
An earlier scope invocation named a nonexistent test file and ran zero tests;
it is retained as failed invocation evidence, not counted as validation.
On a localised-variation analytic fixture the adaptive method uses less than
half the full-root uniform method's evaluations at the same comparison target.
That fixture does not establish a physical-gun speedup. The complete frozen
`20260913-spatial-refinement0075` source is now executing this method.

The older spectral-only global 0.1 reference failed after 7563.18 seconds,
at 324944 evaluations, with slab-current residual 1.20178e-9. Its final complete
round still had an unfitted complex difference of 0.08086; no source was
published. Newer spectral/graph runs retain the original threshold and
continue separately. The older failure evidence is not overwritten.

Visual inspection of an offline uniform-wave fixture exposed point-histogram
holes and then striping after merely changing display-bin sizes. The latter
passed simple nonzero/sum tests but failed the stronger visual uniformity check.
The display is being replaced with conservative affine-cell overlap. This is
a presentation correction, not a change to propagated wave amplitudes or
an admission of any unfinished physical source.

Conservative affine-cell overlap passed
[31 display/observable checks](evidence/20260913T202936Z-conservative-wave-cell-display-964c9f32/report.json),
including analytic rotated/sheared uniform fields, partial visible cells,
conservation and CPU/Numba agreement. The offline 32 x 32 uniform-wave view
was rendered again and visually checked: neither sparse points nor false
stripes remain. Display-only cell overlap preserves the incoming cell weights;
it does not manufacture additional physical resolution.

The spatial-refinement full-pipeline CLI contract passed
[17 checks](evidence/20260913T203312Z-spatial-imaging-cli-contract-161275d1/report.json).
Phase-only viewing now reads the retained complex mode and its reference
without calculating FFT current observables. Bounded row chunks preserve
the same wrapped phase and leave unoccupied pixels undefined. This passed
[33 checks](evidence/20260913T204228Z-bounded-phase-only-display-ff3c9a91/report.json)
and [22 expanded checks](evidence/20260913T204412Z-high-resolution-phase-only-display-0ef40112/report.json),
including a 2048-square field under a 128 MiB readout budget. Compilation
also completed. These remain display/contract tests, not source admission.

The older unsmoothed CF6 local-budget reference reached its 1,000,000
evaluation budget without converging; it published no source. The newer
global and spatial embedded comparisons continue in separate frozen trees.

### Complete mixed-channel slab acceleration

Mixed propagating/evanescent constant slabs now try a complete modal
scattering chart, referencing decaying solutions at opposite faces instead
of constructing growing long transfer matrices. All 2N solutions, reflection
blocks and absolute complex phases are retained. Degenerate/grazing or
unresolved charts fall back to the previous full scattering-doubling solver;
its original current threshold is unchanged. No singular-value clipping or
flux correction is performed. The approach follows the stable scattering
formulation in [Ko and Inkson (1988)](https://doi.org/10.1103/PhysRevB.38.9945);
this implementation additionally validates its local modal defect/current.

The initial kernel scope passed
[40 checks](evidence/20260913T205431Z-modal-complete-slab-4a47972c/report.json),
and the integrated Magnus/embedded-BVP scope passed
[122 checks](evidence/20260913T205524Z-complete-mixed-slab-integration-1f405172/report.json).
The [actual retained-matrix timing](evidence/20260913-modal-mixed-actual-slab/report.json)
shows 1.12 to 1.50 times kernel speedup over four signed widths, with full
complex block differences below 7.90e-11. Other source jobs were running:
these kernel timings are not a full-gun speedup or a convergence claim.
The separate frozen `20260913-modal-spatial01` physical reference now uses
the new kernel with spatial refinement and the unchanged source parameters.

### Re-executed mesh hints for retries

Spatial refinement can now consume a validated dyadic numerical partition
through `--occupied-mesh-seed`. It accepts numeric-only JSON or the mesh
from a failed spatial diagnostic. Physical interval endpoints must match the
current plan; gaps, overlaps, invalid depths and unmatched intervals fail
before any boundary solve. Every hinted leaf is re-executed from the current
field sampler, and the complete driven tip/reflected-load problem plus two
uniform complex checks still run. No failed wave, source current, phase,
matrix, or source-admission label is imported. The consumed mesh tuple is
included in the numerical checkpoint identity.

The restart/process/CLI scope passed
[73 checks](evidence/20260913T210808Z-numerical-mesh-seed-reexecution-63104782/report.json)
and compilation. A completed adaptive solve exports its final partition;
an independent ODE fixture confirms fresh execution and two new uniform
checks when using it. A failed-round partition was also re-executed and
converged on that fixture. This is not recovery of an accepted physical
source, nor a measured full-gun retry speedup.

### Error classification and complete-port memory reduction

Boundary comparisons now report norm change and complex overlap at the worst
boundary. These are diagnostic scalars only: acceptance still compares the
original, unfitted complex fields and covariant derivatives. A pure phase
error is not hidden by alignment. This passed
[29 checks](evidence/20260913T212002Z-unfitted-boundary-error-classification-b79e1aa5/report.json).

Spatial leaves no longer store an additional composed fine-slab matrix in
addition to both half slabs. Their error indicator applies the two complete
half-slab operators to the current incoming ports, solving their internal
reflection exactly. Full boundary solves still retain every channel and
both half slabs. This removes four dense blocks per refined leaf, not any
physical state. The complete-port and refinement scope passed
[43 checks](evidence/20260913T212224Z-lower-memory-complete-spatial-ports-9840537e/report.json)
and compilation. Existing frozen physical runs do not use these later changes.

Exact downstream suffix reuse now works when adaptive subdivision changes
the number of intervals. Matching uses the full complex operator hashes and
the exit boundary/chart, not array positions. All dependent upstream responses
are rebuilt. The scope passed
[46 checks](evidence/20260913T213444Z-variable-mesh-exact-suffix-21ee89b6/report.json).
Complete cached spatial half slabs now stay in the parent process for fresh
occupied-port readout; only missing half slabs are sent for propagation.
The serial/process/refinement scope passed
[73 checks](evidence/20260913T213647Z-cached-spatial-port-scheduling-601d14b6/report.json).
Both changes compiled. These reduce redundant work without changing the
two-way boundary equations or convergence thresholds.

### Mandatory history versus optional reuse

The pipeline now always supplies an executed-energy store for the coherent
surface gun, including nonsegmented and forced-recalculation requests.
`use_cache=False` forbids reading previous energies but no longer discards
newly executed full boundary histories. The gun result retains dependency-bound
history keys for its complete energy mixture. This passed
[34 checks](evidence/20260913T214204Z-mandatory-history-independent-reuse-41822514/report.json),
including a coarse actual-gun regression that forced a fresh first energy and
preserved its complex history. This is not a converged gun benchmark.

Mesh hints can be scoped to their exact emission energy using
`--occupied-mesh-energy-ev`. A multi-energy source rejects an unscoped hint
or an energy absent from the physical source; other energies still execute
with full refinement. This passed
[57 checks](evidence/20260913T214719Z-energy-scoped-mesh-valid-domain-1aa9b0ce/report.json).
The preceding test attempt stopped two input-policy fixtures at their
independent numerical-domain guard; the fixtures were corrected to use a
valid domain, without changing that guard.

Completed embedded-refinement rounds now stream as immutable diagnostics to
`refinement.jsonl`. The original complex comparison remains the acceptance
criterion, and an observer cannot mutate its wave or mark it successful.
Creating `cancel.request` in that driver's new output directory requests
cooperative cancellation. The diagnostics/CLI scope passed
[46 checks](evidence/20260913T215209Z-immutable-refinement-round-stream-c76c924e/report.json).
The preceding two observer-test failures compared frozen nested tuples with
unfrozen lists; explicit diagnostic deserialization fixed the tests, not the
physics. All changes above compiled.

The older frozen spatial `.075` reference stopped after 64,717 logged step
evaluations at its 16 GiB process-transfer budget. Its last complete check
had complex change 0.137297 and local indicator 0.00198554; it was not
converged. The [failure report](evidence/20260913-spatial-refinement0075/report.json)
retains 19,078 numerical leaves and confirms unchanged implementation inputs.
No energy mode or source was published. Final elapsed time 5383.68 s includes
lengthy failed-buffer cleanup. Read-only native stack inspection identified
Windows heap/NumPy-buffer deallocation during exception-frame release, not
another physical propagation. The diagnostic profiler was installed only in
the ignored `.temsim-wave-cache/diagnostic-tools-pyspy` tools directory.

The driver now writes failed diagnostics before releasing those large frames.
This passed [32 checks](evidence/20260913T215746Z-early-failure-diagnostics-13f58bdc/report.json)
and compilation. The separate frozen `20260913-mesh-resume0075` run re-executes
the numerical hints with a 24 GiB working budget, guarded mixed-channel
modal slabs, reduced leaf storage, exact variable-length suffix reuse and
local cached-port readout. Its hint is explicitly scoped to the unchanged
first emission energy, 0.18744544167837376 eV. Other energies still execute.
No failed field/current is imported and the 0.001 complex threshold is unchanged.

## Paused at the user's request

On 2026-09-13 the user asked to shelve this work because the long-running
validation took approximately four hours. No further calculations are to be
started until the user resumes this work.

The six remaining validation runs were stopped. The mesh-resume run accepted
its cooperative cancellation request and saved an `InterruptedError` report.
The five older drivers without that cancellation interface were terminated
with their own worker trees; their existing logs and runner receipts remain.
Their unfinished, unsaved in-memory wave states are not resumable results.
No project source, existing saved evidence, or completed cache was deleted.
The microscope GUI was not terminated. A process check confirmed that none
of these validation drivers or their launchers remained running.

Full new-source Ray Diagram/TEM/STEM/signal acceptance remains incomplete.
The production admission gate has not been opened, and no interrupted or
unconverged calculation is admitted as a usable source. Existing code changes
are retained, not rolled back. Any future continuation must first address
the computational cost and use explicitly bounded validation runs.
