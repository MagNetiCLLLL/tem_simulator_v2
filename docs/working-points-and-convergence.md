# Working points and numerical comparisons

The current development scope uses classical electrons emitted from the physical
tip and transported through the installed gun and column. Coherent tip-to-column
development remains paused. A named numerical preset is not a validation result.

## Inspect and compare

Open **Working Points** to filter and sort captured source, assembly, mode, plane,
current and sampling readouts. Selecting a record never changes the instrument.
Use **Pin A**, **Pin B** and **Compare A / B** for frozen input and observable
comparisons. Warnings identify differing planes, definitions and numerical scope.
The complete captured parameter graph includes disabled components.

An imported package initially loads its manifest and scalar index only. These
scalars are labelled unverified. **Load retained data** verifies the numerical
payload in the background. It does not restore or start transport.

**Export...** explicitly selects complete inputs with retained results, complete
inputs for recalculation, or metadata only. The first preserves the exact record
identity and requires verified results. An input-only derivative retains every
captured input and links its parent, without carrying old result qualification.
Metadata-only exports omit both bulk input arrays and embedded external content;
they retain inspectable scalar controls, asset checksums and historical readouts.
They cannot restore, migrate, apply illumination or start a sampling calculation.
Inputs/metadata can be exported directly from an indexed archive without loading
result arrays. For offline restoration, first use **Make portable input copy**
to capture verified configuration, structural dependencies, specimen and field
inputs in the background. The separate input record can be calculated and
exported with results. Merely embedding selected files in an older package is
not equivalent to declaring a portable input archive.

**Compare with current** includes the exact parameter diff and a conservative
reuse/recompute explanation. Unmapped dependencies cannot grant reuse. A component
centre is not a safe cache boundary when its field extends across the checkpoint.

## Apply, restore and migrate

**Apply illumination...** first checks the physical source, assembly, mode,
component insertion, geometry and calibration. The confirmation lists the exact
declared lens/aperture/deflector/stigmator/corrector operating changes. Source,
objective, specimen, detectors and other controls are preserved. Recalculation is
required. A stale preview or failed installation leaves the preceding instrument
intact. **Undo last apply** uses the existing transaction history.

**Restore working point** and **Fork compatible point** retain their complete
state and dependency checks. A declared portable archive resolves every input
against its verified contents, including the selected TOML assembly; absent
entries cannot fall back to live files. Other records still require their original
external dependencies. Restoring an archive does not overwrite live files.
Archived structural definitions are read-only until an explicit independent copy
is made. Changed implementation, schema or numerical dependencies require explicit
input migration and fresh calculation; historical results remain inspectable.

**Migrate inputs** creates a separate input-only record with a new identity,
parent identity and migration diff. It leaves the old record readable and the
live instrument unchanged. It does not accept historical downstream sources or
reuse their transport. **Save input candidate** also permits unresolved designs.

## Sampling and Convergence

Choose one numerical axis and an explicit ray budget. Gun and column step checks
retain the existing source quadrature. Independent position, direction and energy
checks use the displayed product factors. Their product is the emitted ray count;
surface energy remains conditional on local direction so the emission law is
preserved. The receipt records any selected-to-baseline sampling-method change.
Grounded-gun mesh checks require an already active surface model.
After completing separate gun-step and column-step checks for the same captured
inputs, choose **Joint gun + column steps / 2**. Its receipt links both preceding
checks. Historical or different-input evidence cannot unlock the joint run;
unresolved preceding checks remain visible and do not become a qualification.
All matching axis receipts are retained when exporting the working point.

Optional **Check crossover topology** adds dense bracketing and exact executions
of proposed roots. **Crossover bracket spacing / 2** checks that observation mesh
independently of the transport step. Set the checkpoint memory budget explicitly;
requests beyond its bound are rejected before tracing. Raw roots remain in the
receipt, and the surface-focus tolerance separates terminal from intermediate
roots. Common surviving positive-weight populations within each bracket prevent
clipping from creating a false waist. The approved six-root Microprobe and
historical five-root Nanoprobe interval targets remain distinct and scoped.

The panel reports emitted/transmitted samples, weighted current/transmission,
centroid, D95, alpha95 and effective sample size. N_eff measures weight
concentration, not convergence or an IID error estimate. No sampled survivors
does not prove complete physical blockage. Pair agreement does not establish
all-axis convergence, source calibration or specimen/detector qualification.
Only scalar evidence is exported; full path buffers are released between runs
and before root refinement. Cancellation preserves previous complete evidence.

## Layouts, readouts and memory

The Layouts menu adds **Instrument**, **Alignment**, **Experiments** and **Results**
arrangements without replacing saved layouts or changing physical settings.
At smaller window sizes the pages scroll while the compact result readout stays
above them. Existing editors keep their usable control sizes.
The compact readout explicitly selects **Ray result** or **Retained High
accuracy**. Each shows its own identity, captured plane/model, numerical preset,
recorded backend and validation scope. Edits mark old results stale. Missing
exact checkpoints stay unavailable rather than becoming plot interpolation.

High-accuracy input preparation runs in the background after capturing editable
values and pinned immutable input arrays. **Cache settings** includes a separate
input-asset retention budget. Pinned arrays remain owned by active requests;
reducing retention cannot invalidate them. This cache bound is not a reservation
for every worker's temporary memory. Shared FIFO admission accounts for retained
arrays and request estimates across the existing numerical workers. Its initial
RAM reservation limit is 40 GiB, configurable within the detected host budget;
one numerical worker and bounded library threads avoid competing full-memory
requests. These estimates are not an operating-system RSS limit.

The Compute selector retains CPU and Auto and adds **Prefer GPU** / **Require
GPU**. Preference reports an eligible CPU fallback. Requirement rejects missing
devices and column stages that require the existing CPU vector-field or medium
solver. It does not turn gun preparation into GPU work. Input, physical-model and
cancellation errors are not retried on CPU. A bounded device cache retains one
compatible column plan and particle capacity, with fresh host results and context
reset checks. Auto can compare medians from repeated actually requested matching
workloads; without enough measurements it retains the deterministic policy.
Hardware equivalence and transfer measurements have not been run on this CPU host.

## Alignment and independent experiments

Direct Alignment offers optional joint condenser constraints on spot size, angle,
current, centroid and waist, with explicit allowed lens controls, numerical bounds,
weights and budgets. The optional topology reference is assembly/mode scoped.
Only a candidate passing independent gun/column checks may apply; apply supports
undo. These checks do not establish sampling or field-mesh convergence. Active
vacuum transport is currently unsupported by this optional incident observer.

**Beam centre and direction** uses the installed two-plane deflector and four
measured entrance-plane coordinates/slopes. Rank and conditioning are checked.
The existing condenser stigmator supplies only one independent quadrupole response,
so arbitrary two-axis stigmation remains unavailable. Dynamic pivot/scan-descan
matching needs a defined time-dependent target and observation-plane contract.

In **Design Explorer**, capture A/B owns the complete instrument graph. Runtime
sweeps use registered controls and a bounded Cartesian product. **Declared normal
perturbations** takes exactly one positive standard deviation per selected control,
sample count and seed. These independent untruncated distributions are explicit
simulation assumptions, not measured noise or OEM tolerances. Invalid draws stay
failed; their reasons are retained.

**Plots / saved experiments** shows executed one-dimensional curves or two-dimensional
point maps. Missing and failed observations are not interpolated into an optimum.
The two-objective comparison lists all nondominated candidates for the selected
min/max objectives. It does not select a universal best setting. Numerical status
and tolerances remain separate from successful execution.
Absolute local sensitivities are additional objectives; signed derivatives and
the executed neighboring points are retained in evidence. Differences use
matching other coordinates and never bridge a failed or missing point. These
are observed finite differences, not convergence or measured-noise estimates.

**Geometry candidates** selects an existing module, component and scalar dimension.
The existing editor derives dependent dimensions and validates the component graph,
module clearances and assembled layout in detached TOML content. Live definitions
are unchanged. Configured mapped fields must consume the candidate geometry;
analytical models remain labelled analytical. Fixed-control comparison performs no
optimization. The separate optimization option requires an explicit condenser target,
allowed controls and joint constraints, retaining solved controls and forward evidence.

Save an experiment as `.temexp` to retain its full input recipe, plan, scalar results,
failed points and execution identity. Resume checks complete inputs, implementation,
numerics, perturbation seed and tolerances before reusing finished points. It does
not retry failed points silently. A changed experiment needs a new plan. Saved
results stay viewable even when their inputs can no longer execute. Table export
includes status, failure reason, numerical status, observables and a provenance
manifest. Selecting, viewing or exporting never applies a candidate.
CSV and PNG exports have distinct `.csv.manifest.json` and `.png.manifest.json`
files containing the exported file's SHA-256. Cancelling before a new point
finishes retains preceding completed results, including an already resumed prefix.

See [the progress receipt](development/PRODUCT_USABILITY_PROGRESS.md) for actual
test/measurement evidence and remaining development-guide packages.
