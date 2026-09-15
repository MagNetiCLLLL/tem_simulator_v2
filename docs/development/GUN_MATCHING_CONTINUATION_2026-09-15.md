# Gun matching: numerical repair and continued qualification

## Status

The detached design passes the declared **classical particle** tolerances below.
It is not a real-instrument calibration or a wave/image qualification. No
default preset was overwritten. All runs are
classical particles from the physical tip, through extraction, acceleration,
alignment fields, real apertures and column walls. No coherent propagation or
TEM/STEM image calculation was started. Existing source and assembly defaults
are retained. Generated arrays and caches remain local and excluded from Git.

Loadable input-only design: `profiles/particle_tip_30mrad_20260915.temwp`.
The package contains complete assembly/source/operating/numerical inputs and
zero cached ray or wave arrays. The lightweight acceptance summary is
[`evidence/gun_matching_20260915_final.json`](evidence/gun_matching_20260915_final.json).

The user permits comparing a 1.0--1.2 kV gun-lens control relative to **tip**
with the historical **extractor-relative** convention. Neither convention is
presented as an OEM-confirmed voltage definition. The extraction range remains
4--5 kV. The target is alpha95 = 30 mrad at the specimen upper surface, a
nanometre-scale geometrical probe and preserved crossover component intervals.

## Physical placement support

`candidate_with_gun_geometry` now supports moving the accelerator as well as
the extractor and gun lens. Accelerator stages and attached components (the
DPA) move with it, exactly once. Body contours, optical references, vacuum
bores and liners remain consistent in the detached assembly snapshot.
Overlapping bodies, changed component order and nonfinite coordinates are
rejected. No downstream source is introduced.

## A concrete cause of failed mesh convergence

Independent corner-grading constructions could create nominally identical
axial nodes separated by only `4.336808689942018e-19 m`. An ordinary exact
`unique` operation retained both. The resulting almost-zero-thickness elements
made the Laplace system ill-conditioned. A small algebraic residual did not
certify the physical field.

`merge_axis_nodes` removes only local floating-point aliases (32 ULPs). It
keeps exact metal/domain boundaries and never averages or displaces them.
Distinct physical boundaries inside that precision limit cause an explicit
error. Genuine tip-scale intervals are not merged using a domain-wide absolute
tolerance. The field/cache mesh identity is now `tip-graded-electrode-local-v2`.
The report includes the minimum cell sizes and alias policy.

Fixed inputs for this comparison: extractor centre 2.1 mm, lens centre 8.2 mm,
accelerator centre 218 mm, extraction 4.5 kV relative to tip, lens 1.1 kV
relative to tip. Tip R=100 nm, cone half-angle 5 degrees, cap half-angle 10
degrees. No physical input was retuned between mesh evaluations.

The following is the dimensionless angle-to-exit-position coefficient B of the
axial variational map, **not** a finite-source spot or acceptance measurement:

| Cells per bore | Corner cells | Before alias repair | After alias repair |
| ---: | ---: | ---: | ---: |
| 8 | 16 | -246.362388 | -994.849448 |
| 16 | 16 | -395.706290 | -995.850234 |
| 32 | 16 | -142.178783 | -996.780131 |
| 32 | 32 | not compared | -995.448825 |
| 64 | 32 | not compared | -996.381021 |

The repaired 32-to-64 comparison changes B by about 0.094%, versus the large,
nonmonotonic pre-repair changes. Its canonical determinant is about
0.999999690. This is encouraging numerical evidence, **not** full finite-source
or nanometre-focus qualification. The largest completed mesh had 11,749,056
vertices and took 116.0 s for field construction plus variational observation.
An earlier, unmerged larger-corner attempt exhausted the sparse-factorization
memory budget; it did not produce a physical result.

## Propagation speed without changed physics

The compiled field evaluator uses the same scalar potential, cut cells,
near-axis interpolation and negative gradient as the NumPy reference. No
fast-math, field cutoff, force multiplier or relaxed error tolerance is used.
The NumPy implementation remains available as the reference/fallback.

- A 193-point field-query microbenchmark measured 9.40x improvement. This is
  not a whole-program speedup.
- Two uncached, full 49-particle gun traces on the same default field measured
  66.55 s (NumPy) and 32.69 s (compiled), a 2.04x improvement. Exit coordinates,
  slopes, transmitted weights and stop identities agreed in that comparison.
- Exactly unpowered gun-deflector coils no longer evaluate an envelope that
  will be multiplied by zero. Nonzero fields, including tiny drives, and the
  independent blanking drive are retained and regression-tested.
- Gun-history observations query many planes in one batch, preserving ray IDs,
  the first forward crossing and the status at that plane. A later stop does
  not remove an earlier observation. No propagation is skipped.

## Full-support numerical tip quadrature

`apex_stratified_v1` remains a partition of the **whole** emitting cap. New
optional `tangent_stratified_v1/v2` partitions both local tangential momentum
components into three Gaussian-CDF intervals each. All nine boxes, including
both exterior tails, keep their exact probabilities. A numerical refinement
centre is not a physical beam tilt, angular cutoff or shifted source mean.

For the existing exponential normal/tangential energy law, the two tangential
amplitudes are independent zero-mean Gaussians with variance `Et_mean/2` in
sqrt(eV) coordinates. The normal energy remains exponential. Tests check
probability in every box, mean/covariance/energy convergence, whole-cap flux,
multiple outgoing directions per surface point, snapshot round trips and
trajectory-cache invalidation without changing the electrostatic field key.

The initial v1 diagnostic repeats the same local CDF sequence at each site.
It remains replayable. v2 continues independent-dimensional sequences across
sites so that adding positions also refines energy and local directions.
The default uniform-CDF emission is unchanged. Unsupported angular/energy
laws and unresolved CDF bins are rejected, not silently approximated by a
narrower distribution.

An optional nine-integer `spatial_stratum_allocation` now directs **numerical
site budgets**, not physical current. Every area stratum receives at least one
site and retains its exact area probability. Empty allocation preserves the
old quadrature. The exploratory allocation `(1,1,1,1,1,4,24,24,1)` concentrates
sites on the two annuli that supply most of the observed specimen current;
it does not certify moments of the whole gun beam at that same budget.

A 0.1-sigma central tangent box proved too narrow for this acceptance region:
executed accepted samples extend to approximately -0.113 through +0.125 sigma
relative to the numerical box centre. Two exterior-box samples each carried
about 1.5e-5 of total emitted current, making the 10,369-ray estimate unstable.
They were NOT dropped. A 0.2-sigma central box passed the comparisons below with
full exterior tails intact. Two unfinished 20,737-ray, 0.1-sigma runs were deliberately
cancelled when this inefficiency and the DPA interval issue were identified.
Their interruption tracebacks are not completed physics results.

An optional nine-integer `angular_stratum_allocation` now allocates sample
counts inside those same nine CDF boxes. Empty allocation exactly preserves
the old sequence. The selected `(1,1,1,1,32,1,1,1,1)` priorities increase the
resolved specimen effective count from about 131 to 855 at 10,369 particles.
No box, tail, area probability or physical interaction is removed. These
priorities invalidate particle execution, not the unchanged electrostatic field.
Tests of whole-source moments compare equal minimum **tail** sample counts;
central refinement is not claimed to resolve whole-gun moments equally well
at the same total budget.

Preview needs at least 81 physical samples for the 9-by-9 full partition,
plus its zero-current axial support probe. This model therefore displays and
executes 82 rays instead of failing at 49. Ordinary source previews retain 49.
Low-budget angular grouping is reduced numerically without altering physical
distributions. A preview with fewer than 16 effective surviving samples retains
its actual trajectories/current but does not advertise a reliable probe angle
or diameter. High-accuracy user budgets are not silently increased.

## Crossover observations are not interchangeable

The historical, isolated analytic gun at a 0.0125 mm step has local envelope
minima near 41.527, 79.759, 115.960, 151.915 and 187.931 mm. Coarse historical
stepping had produced a different count, so the coarse count is not a baseline.

At a 0.025 mm step, comparing the same particle IDs five millimetres before
and after each minimum gives opposite-side fractions of approximately 90.7%,
0%, 0.52%, 0% and 0%, respectively. The first is a pronounced crossover;
the later minima are weak envelope ripples. These observations alone do not
redefine the user's crossover acceptance criterion.

The existing Ray Diagram selects one gun waist; lens crossover assignment
starts with the column lenses. Its displayed count is therefore not the raw
count of all five gun-envelope extrema. Reports retain both meanings rather
than silently replacing one with the other. Raw roots remain available.
The deliberately focused specimen terminal waist is reported separately from
intermediate crossovers within a declared 1 nm terminal-plane tolerance.

The production gun/lens markers also used unweighted sample means. They now
use physical current weights and the same positive-current population across
each three-plane bracket. Splitting one numerical trajectory into copies with
divided weights no longer moves its marker. Zero-current support probes and
population loss cannot manufacture a crossover. Markers remain plot-node
diagnostics, not precise specimen-focus measurements.

### Full-path topology correction

The first whole-pipeline comparison retained the same one gun / eight column
marker labels, but label count alone was insufficient. The historical gun
waist at 41.525 mm preceded the DPA at 120 mm. The new candidate waist near
195--203 mm followed its DPA at 138 mm, despite remaining inside the accelerator
body. That candidate therefore **fails the full component-interval gate**.

The next detached candidate moves the DPA to 210 mm (body 209--211 mm), inside
the accelerator body 48--388 mm and clear of its electrode rings. Its parent,
optical reference, wall and liner move consistently. The aperture's opening is
unchanged. Full-path topology now explicitly includes gun apertures and all
downstream optical components. A regression rejects the old 138 mm DPA case
instead of treating accelerator-body membership as equivalent topology.
The 210 mm DPA candidate passed fresh complete particle transport at both
10,369 and 20,737 particles. The permanent DPA aperture remains present.

For reference, the isolated historical optical-only pipeline at 769 rays and
0.025 mm step displayed the gun waist and these eight lens waists: C1, C2,
TL22, TL12, MC/TL11, Objective, Intermediate, P1. Its exact input component
positions are retained in `historical-full-component-layout.json`; no archived
source was installed into the active simulator.

## Accepted detached particle design

The earlier v1 1,297-particle fit is superseded as a design candidate. Old
sub-nanometre fits on unmerged meshes are not promoted based on spot size alone.

The final design keeps the physical source: R=100 nm, cone half-angle=5 degrees,
emitting-cap half-angle=10 degrees, 1 mm shank, 100 nA full-cap emission,
exponential normal energy mean 0.2 eV and tangential energy mean 0.1 eV. All
locally outgoing angles remain supported. No source size or angular cutoff was
reduced to obtain the small transmitted probe.

| Component/input | New detached design |
| --- | ---: |
| Extractor centre from tip | 2.1 mm |
| Gun lens centre from tip | 8.2 mm |
| Accelerator centre from tip | 218 mm |
| Gun DPA centre from tip | 210 mm |
| Extraction relative to tip | 4.5 kV |
| Gun lens relative to tip | 1.1 kV |
| C1 excitation | 25% |
| C2 excitation | 14.890193391572357% |
| C3 excitation | 26.395215230901794% |
| Objective excitation | 68.89947594537043% |
| Specimen upper surface | 1599.1999975 mm |

At 300 kV, using the final anode as ground, the tip/extractor/gun-lens
potentials are -300.0/-295.5/-298.9 kV. The preserved historical additive
gun-lens convention instead gives -294.4 kV for the same displayed 1.1 kV.
This is a model convention comparison, not confirmation of an OEM supply
wiring diagram. At the same close placement the axial variational B coefficient
is approximately -995.85 for tip-relative and +14894.86 for extractor-relative
at the comparison mesh. That proposal calculation explains why the definitions
produce different focusing; it does not qualify a finite source by itself.

The extractor body is 0.1--4.1 mm, gun lens 4.2--12.2 mm, accelerator
48--388 mm. Accelerator electrode centres span 60--380 mm. DPA body 209--211 mm
does not overlap an electrode. Parent attachments, optical references, vacuum
bores and liners are moved consistently. Aperture openings are unchanged.

### Fixed-settings convergence

Declared engineering tolerances: alpha95 within 1% of 30 mrad; both executed
one-sided and covariance waist estimates within 1 nm of the entrance; d95 below
5 nm; independent d95 changes below 10% and transmitted-current changes below
5%; at least 16 effective positive-current samples. These are finite numerical
accuracy targets, not measurement uncertainty or universal physical validity.
Every row below uses the **same** physical source, placement and lens controls;
comparison rows are not independently refitted.

| Check | alpha95 (mrad) | d95 (nm) | Waist offset (nm) | Specimen current (pA) | Effective samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal 10,369 particles, 72 directions/site | 30.000000 | 1.44737 | approximately 0 | 7.64451 | 854.78 |
| 20,737 particles, 144 directions/site | 30.169951 | 1.40562 | +0.30238 | 7.53424 | 1830.44 |
| 10,369 particles, 36 directions/site | 30.028039 | 1.32876 | +0.57500 | 7.47706 | 736.26 |
| Finer tip/global field mesh | 29.995529 | 1.41291 | +0.13009 | 7.62995 | 846.34 |
| Finer field and 0.0125 mm column step | 29.995529 | 1.41302 | +0.12820 | 7.62995 | 846.34 |
| Ordinary CUDA column backend, same executed gun | 30.000000 | 1.44737 | approximately 0 | 7.64451 | 854.78 |

The 36-direction case doubles emission sites at fixed total budget, so it is
an independent spatial repartition, not a uniform increase of every sampling
dimension. Nominal field numerics are electrode/corner cells 32/16,
apex cells per radius 80, radial/axial base nodes 640/1280. The finer field uses
128 apex cells and 1024/2048 base nodes. Column step is 0.025 mm unless stated.
Comparisons qualify the aperture-selected specimen observables at this design;
they are not a calibration of every full-gun envelope statistic.

### Tip, C1 and specimen are distinct population measurements

The complete physical emitting cap has edge diameter **34.72964 nm**, d95
**33.85674 nm**, RMS radius **12.29443 nm** and depth **1.51922 nm** analytically.
An independent 4,097-particle uniform full-cap check gives d95 **33.83483 nm**.
Do not infer the cap size from the importance-sampled maximum drawn point.

At the nominal whole-current C1 waist, Z=513.752435 mm, the measured d95 is
about **6.13 micrometres** (RMS radius 2.53 micrometres). This full-cap statistic
is diagnostic only; this quadrature preferentially resolves eventual pupil
transmission, not precise moments of the outer full-current gun population.
It must not be labelled a nanometre-sized image of the entire emitting cap.

Observing the **same electron IDs that eventually pass to the specimen**, their
C1 waist is Z=513.946243 mm, d95 **1.30681 nm**, RMS radius **0.34048 nm**. Their
sample-entrance d95 is **1.44737 nm**. This retrospective cohort is only a
diagnostic mask: it does not remove other electrons upstream, invent a new
source, or count an extra full-beam crossover. Aperture selection physically
accounts for the very low accepted current, approximately 7.5 pA out of 100 nA;
no transmitted-current renormalisation was applied.

### Preserved existing crossover chain and projection chamber

The production paths at both particle budgets retain one displayed gun waist
and eight column waists, in the following same historical component intervals:

1. Gun lens -- gun DPA.
2. C1 -- C2.
3. C2 -- C2 aperture.
4. DP22 -- HPC.
5. TL12 -- condenser stigmator.
6. Mini Condenser -- Objective/sample.
7. Objective aperture -- descan.
8. Intermediate -- P1.
9. P1 -- P2.

The 20,737-particle run has **4,637 positive-current particles** at the physical
projection-chamber entrance (Z=2586.9 mm), maximum radius **0.99484 mm**, current
**7.53424 pA**, with finite propagated states. The 10,369-particle run has 2,174
there. These are actual transmitted particles, not zero-current display probes.
This optical-only validation deliberately does not execute specimen interactions.

## Loading and reproducing

1. Open **Working Points**, choose **Import...**, and select
   `profiles/particle_tip_30mrad_20260915.temwp`.
2. Choose **Restore working point**. This restores the complete detached
   geometry and parameters without recalculation or applying lens presets.
3. **Update rays** in Preview for the ray diagram. This source requires at
   least 82 preview rays; small previews are not nanometre-focus measurements.
4. For the documented nominal numerical budget, set **High-accuracy rays** to
   **10369** and **Step (mm)** to **0.025**. Restoring a working point does not
   automatically change the toolbar's next-request budget. Signal/image
   calculation is outside this optical-only acceptance.

Do not press **Load assembly** or **Apply calculated lens preset** to reproduce
this point: those commands intentionally replace parts of the saved design.
The old default configuration and the old voltage definition remain available.
The input package is implementation-pinned; after solver-code changes it remains
viewable but must not be blindly restored as a validated current result.

The acceptance summariser can be rerun against the local scalar report set:
`python scripts/summarise_gun_focus_validation.py --output <new-report.json>`.
Choose a fresh path; existing evidence is never overwritten by that script.

Actual 82-ray cold-start optical preview took 65.8 s in this field configuration.
That includes gun/field construction and is not a live-interaction latency claim.
Expensive executed gun traces were reused while adjusting C3/Objective. Wave
calculation remains paused; no coherent phase, TEM image, STEM image,
space-charge or real-instrument certification is claimed.

Local scalar evidence is under `outputs/crossover-audit-20260914/`, including
`tip-reference-close-field-deduplicated.json`,
`tip-reference-close-field-32-64-merged.json`,
`interpolation-particle-parity49.json`,
`historical-gun-waist-orientation.json` and
`tip-reference-merged-phase1297-30mrad.json`.

## Validation

149 affected physics/source tests passed, followed by 31 optical-tuning,
stopped-history and working-point tests. Input-only package/voltage/contract
checks (25, overlapping) and six scalar acceptance-gate tests also passed.
Final combined validation is recorded below. Actual input-package restoration
in a fresh Python process and an offscreen MainWindow preserved its exact
snapshot, produced no errors and started zero calculations. No visible-desktop
session is claimed. A broad offscreen GUI smoke test did not pass: its tab
inventory was stale (updated to include Vacuum map and Working Points), then
it exposed Physical Layout label collisions. That renderer issue is not fixed
or represented as a physics failure here. No full-project suite, visible GUI
session or wave/image acceptance is claimed.

Final combined affected regression: **193 passed**, 26 warnings, in 137.73 s.
Warnings include existing deprecated Pydantic configuration, arithmetic on
already-stopped nonfinite ray placeholders and Qt disposal. The finite
positive-current trajectories used for physical acceptance were checked
separately. This is not a full-project all-green claim.

Serial `compileall` completed successfully. The ordinary CUDA column backend
was also executed against the same retained physical gun trace: d95 differs
from the tuning kernel by about 3.1e-10 relative, with identical transmitted
current. The final scalar summary includes that additional backend comparison.
