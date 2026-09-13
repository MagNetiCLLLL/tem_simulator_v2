# Gun coordinate convergence: diagnosis, not image acceptance

## Decision

The grounded coherent tip remains unqualified for production TEM/STEM and
the classical Ray Diagram route. A numerical coordinate width is not a
physical source parameter. Neither of the conflicting currents below is a
certified reference. No source parameter, electrode, aperture, admission
limit or legacy calculation path was retuned in this work.

The next implementation target is the propagation from the near-tip face
to the first body-bore event at **6 mm**, not another full imaging feature.
This is a narrower failure interval than an exit-current comparison alone
could identify. It does not yet identify the exact first failing slice or
separate all axial and transverse discretisation errors.

## Same-physics executions

All runs used the existing default physical assembly and grounded coherent
surface: 100 nm tip radius, 10 degree emitting cap, 100 nA reservoir,
0.3 eV mean / 0.1 eV RMS emission energy, 4 kV extraction and 300 kV final
acceleration. Three physical energy modes were executed, including the
jointly loaded near-tip FEM, extraction, focusing, acceleration, body bores,
DPA and C1 apertures, and gun-overlapping column fields. The coordinate
widths below multiply the emitting-cap radius **only to define the basis**.

Numerics: quadratic surface FEM, 769 radial x 257 axial nodes, numerical
surface radius factor 5, 64 radial modes, 128 quadrature parameter,
0.5 mm field step, 0.1 relative near-axis step, 256-pixel exit representation.
Each raw report binds the physical snapshot and solver implementation; all
five completed physical runs report `implementation_unchanged: true`.

| Axial method | Coordinate width | Gun-exit current (nA) | Raw report |
| --- | ---: | ---: | --- |
| Original midpoint | 0.2 | 29.25222659 | [baseline](evidence/20260913-source-convergence-baseline02/report.json) |
| Original midpoint | 0.1 | 58.15723169 | [baseline](evidence/20260913-source-convergence-baseline01/report.json) |
| Midpoint with new diagnostics | 0.2 | 29.25222658 | [loss ledger](evidence/20260913-source-convergence-ledger-02/report.json) |
| Experimental CF4 | 0.2 | 0.0002261274 | [failed convergence](evidence/20260913-source-convergence-cf4-02/report.json) |
| Experimental CF4 | 0.1 | 26.62597928 | [failed convergence](evidence/20260913-source-convergence-cf4-01/report.json) |

The baseline/diagnostic midpoint comparison changes current by 4.97e-10
relative and complex field by 4.95e-7 relative L2. The diagnostics do not
repair or renormalise the output. CF4 is an explicit development option;
**midpoint remains the default**. CF4 is not a demonstrated improvement
for this unresolved physical run.

## What was distinguished

1. At the original near-tip exit face, sampled nodal complex L2 disagreement
   for the two coordinates is about 0.53-0.62%. This is a preliminary face
   comparison, not a FEM convergence certificate.
2. The baseline coordinate-0.2 mask ledger separates 44.0496 nA assigned to
   absorbing disk masks from 2.63856 nA discarded by finite-basis projection.
   Thus **direct mask-projection loss alone cannot explain the 28.905 nA
   difference between baseline outputs**. The absorbing-mask term itself
   is not physically qualified while the incident field is unconverged.
3. The CF4 comparison records fields immediately before/after physical bore
   events. At 6 mm, before the first mask, the three per-mode complex L2
   errors are 1.426-1.464, while amplitude-only L2 errors are 0.614-0.625.
   Their currents still agree closely. This is not merely an arbitrary
   global phase difference: the spatial amplitudes already disagree.
   See the [phase and shape comparison](evidence/20260913-source-convergence-cf4-shape-comparison.json).
4. A midpoint/CF4 boundary comparison additionally rejected its own fixed
   radial comparison quadrature at 422 mm: values 296.09327764 and
   296.09330290 differed by more than the declared 2e-5 absolute tolerance.
   No report was published for that unresolved comparison. The successful
   CF4 coordinate comparison doubles radial sampling and records the change.

## Implementation added

- Optional fourth-order commutator-free Magnus (CF4) propagation, sampled
  at Gauss nodes in the exact Liouville action coordinate, not uniform z.
  The constant-covariant slabs retain both directions, scalar corrections,
  derivative jumps, all consumed fields and all existing mask events.
- An occupied-wave mask ledger for both incoming ports. For the projected
  disk operator P, physical-mask removal is a*(I-P)a and unresolved
  transmitted content is a*(P-P^2)a, multiplied by the chart wave number.
  The sum is independently checked against net boundary current loss.
  No square-root mask, flux correction or phase fit is used.
- Full complex coefficients and covariant derivatives at selected existing
  component boundaries, with physical z, basis width/curvature and retained
  axial phase. These are diagnostic states of an executed source chain,
  not independently configurable sources.
- A boundary comparator that keeps complex phase error separate from
  amplitude/density error and refuses nonmatching physical configurations.

The isolated fourth-order method follows
[Blanes and Moan (2006)](https://personales.upv.es/~serblaza/2006APNUM.pdf).
The coordinate transformation follows
[NIST DLMF 1.13(iv)](https://dlmf.nist.gov/1.13#iv).
Formal fourth order on resolved smooth fixtures is not a large-step
accuracy guarantee for the stiff moving-basis gun problem.

## Required next work and stopping rule

1. Freeze this physical configuration and retain these failed reports.
   Add intermediate checkpoints inside the 2 nm to 6 mm interval to find
   the first growing **complex and amplitude** discrepancy.
2. Replace global field-step-only control in that interval with local error
   estimation: compare an interval with its subdivisions in a common
   physical representation, retaining both directional ports. Do not
   mistake current conservation for an error estimate. Longitudinal carrier
   roundoff, fast coordinate connection, radial basis truncation and boundary
   representation need separate budgets.
3. Refine radial modes and near-face resolution independently of axial
   steps. If fixed-width global Laguerre coordinates remain inefficient,
   implement local/adaptive radial representations with explicit complex
   transfer residuals, including aperture-generated high spatial frequencies.
   Changing the numerical representation must not change the source.
4. Treat each segment as an executed two-port operator. Cache the operator
   with its inputs, numerical method and error evidence; retain downstream
   reflection feedback when composing the full chain. A frozen near-field
   wave with an arbitrary outgoing boundary is not a substitute source.
5. Before further whole-gun runs, predeclare a numerical qualification target
   (proposed engineering starting point: <1% current and <1% complex L2 for
   every energy mode, across two successive refinements and independent
   coordinate widths; domain/projection residuals require their own tighter
   allocated budgets). These numbers are not experimental accuracy claims.
   Only after the short segment qualifies should the same procedure extend
   to the gun exit, complete column, sample and physical detector images.

**No full coherent TEM/STEM image was produced or admitted in this round.**
The failure interval and repeatable error measurements are now explicit;
the propagation convergence defect itself remains open.

## Verification

- [48 scoped numerical tests](evidence/20260913T114226Z-radial-gun-convergence-88807a21/report.json)
  passed, including independent DOP853, Airy, spatial disk-integral,
  flux, moving-frame and radial-moment references.
- [2 updated comparison tests](evidence/20260913T115003Z-gun-boundary-comparison-ed89d681/report.json)
  passed, including the distinction between phase-only and amplitude errors.
- [19 source-admission and cache/transport compatibility tests](evidence/20260913T115217Z-gun-source-compat-scoped-d4bd1d2b/report.json)
  passed in a subsequent unchanged-input run. These are separate from the
  five legacy near-axis fixture failures below, not a rerun that resolves them.
- A broader source-compatibility run is retained as
  [unqualified](evidence/20260913T114546Z-radial-gun-source-compat-baea6e9d/report.json):
  25 passed / 5 failed, and two diagnostic/test files changed during that run.
  The five failures are in the unchanged legacy `test_tip_gun_wave.py` path:
  its 5 nm tip fixture is rejected by the existing paraxial domain bound
  (6.972% > 1%) before reaching the new round-surface solver. No test,
  source parameter or bound was weakened. This is not a full-suite pass.
- Changed Python files compiled successfully. The running user application
  was not restarted, and no commit, push or shutdown was performed.
