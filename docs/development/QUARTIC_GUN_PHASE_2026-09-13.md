# Quartic numerical radial phase in the physical tip/gun chain

Status: implementation and focused numerical checks complete. Both complete
physical comparisons finished after a scattering-roundoff repair, but
independent coordinate convergence FAILED. Production source-to-image
admission remains closed. This continues the failed convergence experiment
in [the wave-following report](WAVE_FOLLOWING_GUN_CHART_2026-09-13.md).

## Physical scope and invariants

No physical source, tip size, electrode potential, extraction/acceleration,
focusing field, aperture or bore has changed. The development scalar orbital
Klein-Gordon operator and its axisymmetric limitations remain the same.
Every configured energy mode still executes the full gun and its reflected
load, jointly coupled to the driven curved-tip finite-element boundary.

The new option is numerical `wave_following.phase_order = 4`; the default
remains 2, with zero adaptation iterations. A quartic coordinate phase is not
a physical aberration coefficient or a new emission model. No downstream
source or fitted image is supplied. Existing profiles are not migrated.

## Operator derivation

The radial basis now optionally includes the explicitly retained phase

    Phi_n = exp(i k c r^2 / 2 + i g r^4)
            * L_n(r^2/b^2) exp(-r^2/2b^2) / (sqrt(pi) b).

Here `r,b` are in nm, `k,c` in nm^-1, and numerical `g` in nm^-4.
Let `q=k*c`, `X=<r^2/b^2>`, `D=<1+r*d/dr>` and `T=<-b^2*Laplacian>`.
Differentiating the physical complex basis gives

    K = T/b^2 + q^2 b^2 X + 8 q g b^4 X^2 + 16 g^2 b^6 X^3
        - 2 i q D - 4 i g b^2 (XD + DX)
    G = k c' b^2 X/2 + g' b^4 X^2 + i (b'/b) D
    closure = G_PQ G_QP.

`K` is the transverse kinetic operator and `G` the Hermitian connection.
The same connection goes into the existing exact Liouville coordinate
transformation. The closure retains the derivative outside the finite radial
subspace; it is no longer only one last-diagonal term. Three extra modes are
used when forming polynomial products **before** projection. Squaring the
already truncated `X` would lose actual boundary matrix entries.

The kinetic, connection and closure matrices are checked independently by
integrating real-space basis derivatives. For zero quartic coefficient and
derivative they reduce to the previous quadratic matrices. The production
quadratic branch retains its previous operator evaluation.

## Selecting coordinates without changing the wave

Full executed gradient moments choose the numerical quadratic and quartic
phase. For the phase-factored envelope, with `rho=r/b`, minimize

    || grad(envelope) - i (d2 rho + d4 rho^3) envelope ||^2.

The 2x2 Gram matrix contains `<X>, <X^2>, <X^3>`; its right-hand side is
`Im<D>, Im<XD>`. Diagonal scaling improves conditioning without changing
the transported state or current. The solved corrections add to the already
executed analytic coefficients, rather than erasing prior phase. All complex
radial coefficients participate. A missing/invalid state or ill-conditioned
fit is rejected.

Log-width, quadratic curvature and quartic coefficient are tabulated over
log axial position. Exact spline derivatives and C2 transition derivatives
are retained. The transition leaves the physical tip matching chart unchanged
up to 10 nm and completes at 100 nm. The full physical BVP is then executed
again; no projected downstream trace is injected.

These are coordinate choices, not a claim that the physical wave consists of
a Gaussian with a quartic aberration. Non-Gaussian coefficients and backward
waves remain in the executed state. The full pilot fields, derivatives and
all analytic coordinate phases are included in chart provenance.

## Phase-preserving export and replay

- Boundary diagnostics store `quartic_phase_per_nm4` beside their full complex
  coefficients. Comparisons reconstruct it explicitly; identical intensity
  with a different quartic phase must not pass the complex-field comparison.
- The gun's Cartesian complex amplitude contains the quartic phase exactly
  once; its existing quadratic and axial phase references remain separate.
- The radial alternative representation stores the quartic coefficient
  explicitly and reconstructs it when loading the executed gun state.
  Missing historical coefficients mean zero, not a rewritten historical file.
- Export checks the analytic phase increment over the occupied grid. A
  too-coarse Cartesian grid raises an error; radial loading refines by
  re-evaluating the same executed expansion within its declared budget.
  The 1e-12 tail is excluded only from sampling diagnostics, not the wave.
- A scaled Laguerre recurrence evaluates the full complex expansion in
  O(modes * points) without an O(modes * points) temporary array. It is
  independently checked against SciPy polynomial evaluation. No coefficient
  threshold, filtering, fitted phase or probability correction is used.

## Focused validation

[Immutable test receipt](evidence/20260913T134412Z-quartic-radial-phase-a594de88/report.json):
**91 passed, zero failures/errors/skips**, with unchanged hashed inputs.
Coverage includes the new operator calculus, numerical chart derivatives,
gradient-moment fit, phase export/replay and sampling failure paths, previous
radial propagation, masks, near-boundary and source-admission regressions.

The independent nonparaxial Bessel angular-spectrum fixture has error
5.0087e-6 with 32 radial modes, practically unchanged after halving the axial
step. With 48 modes its complex L2 error is 6.2580e-7 at the original 0.05 nm
step, below the predeclared 2e-6 threshold. The test therefore refines the
radial basis rather than weakening its tolerance or changing the physical
reference wave. This fixture does not qualify the full physical source.

## Constant-slab roundoff repair

The first full quartic run, pilot terminal coordinate 0.1, failed the existing
1e-9 current-unitarity check. An observer repeated the same execution and
saved the unmodified failing operator in
[the failure capture](evidence/20260913-quartic-failure-capture/failed_slab.json).
It lies at 394.1662850718--398.0924466405 nm for the 0.1874454417 eV mode.
The original 0.075 run completed (67.58085 nA), but one completed run is not
an independent coordinate comparison.

For that 64-channel slab the old norm-based seed required 20 scattering
doublings. Its initial unitarity residual was 6.70e-16; composition amplified
it to 1.60e-9 by the nineteenth doubling. Windows NumPy long-double has no
extra precision on this host. Simply requesting that dtype does not repair
the error.

The seed now distinguishes phase from norm growth. For the unchanged
first-order generator `A`, with Hermitian `Q` and `G`, its Hermitian part has
eigenvalues `+/- eig((kappa I-Q/kappa)/2)`. The connection `-iG` is
skew-Hermitian and does not directly grow the norm. With `h` the seed length,

    ||exp(+/- A h)||_2 <= exp(h ||(A+A*)/2||_2).

The new seed requires log-growth <= 0.5 and separately `||A h||_inf <= 16`.
Thus its transfer condition is bounded by e, and large phase arguments are
still bounded. The exact same generator is exponentiated; physical layers,
all propagating/evanescent channels and both propagation directions remain.
No polar-unitary projection, renormalisation, discarded channel or relaxed
unitarity tolerance is used. Smaller seed-growth bounds are available for
independent numerical checks, and each slab records its scaling diagnostics.

The growth bound follows [Higham's logarithmic-norm result, Theorems 3 and 6](https://nhigham.com/2022/01/18/what-is-the-logarithmic-norm/).
The general risk that excessive scaling accumulates rounding error is
documented by [Al-Mohy and Higham (2009)](https://eprints.maths.manchester.ac.uk/1442/).
Our use in Redheffer scattering composition is a derived implementation,
not a claim that those sources validate this physical gun or our code.

For the captured slab, 17 doublings now give a maximum unitarity residual
1.2137e-10. Reducing seed log-growth from 0.5 to 0.25 gives 18 doublings;
the complete 128x128 complex S matrices differ by 3.5346e-10 in infinity norm,
with neither phase fitting nor current adjustment. The original 1e-9 gates
remain active. Changing internal current-coordinate scales alone was also
tried and did not fix the original failure; those failed trials remain in
[the chart comparison](evidence/20260913-quartic-slab-chart-comparison.json).

[Post-repair immutable receipt](evidence/20260913T140111Z-quartic-growth-kernel-0ef93863/report.json):
**108 passed, zero failures/errors/skips**, unchanged inputs. This includes
the exact captured operator, independent short-slab complex transfer,
large pure-connection phase, strict failure guards, scalar/evanescent current
checks, and the preceding quartic/operator/export coverage. This is a local
numerical repair, not yet evidence of full-gun coordinate convergence.

### Compatibility scope and pre-existing failures

[Broader compatibility receipt](evidence/20260913T140400Z-quartic-legacy-compat-10e7e1df/report.json):
**68 passed, 15 failed**, zero errors/skips, unchanged inputs. This is NOT a
full-suite pass. The failures are retained without changing the tests:

- Ten `test_wave_imaging.py` cases hit the existing coherent-source admission
  rejection, before the numerical code added here. The same ten failure
  names and admission cause appear in the
  [earlier image-entry audit](evidence/20260913T103540Z-legacy-image-entry-audit-b7646d30/pytest.log).
  The TEM/STEM imaging modules and that test module have matching input hashes;
  the admission module hash differs from that earlier audit, so whole-module
  identity is not claimed. No admission module was edited in this round.
- Five `test_tip_gun_wave.py` cases hit the old 5 nm near-axis fixture's
  6.972% generator-error bound, above its unchanged 1% limit. The same failures
  appear in the [earlier source audit](evidence/20260913T114546Z-radial-gun-source-compat-baea6e9d/pytest.log).
  The tested quadratic-gun module, domain guard and test module have matching
  input hashes. The earlier receipt itself was marked input-changed; its raw
  failure log is historical evidence, not a qualified pass.

The current receipt includes passing canonical particle integration,
historical-source readability/policy and isolated column-cache regressions.
Those isolated fixtures are not demonstrations of production coherent images.
No new source-policy exception or retuned tip parameter was introduced.

[Comparison-audit receipt](evidence/20260913T141225Z-quartic-comparison-audit-1ad2c582/report.json):
**30 passed, zero failures/errors/skips**, unchanged inputs (overlaps the
108-test numerical receipt). Synthetic format fixtures confirm that failed
runs, changed implementations, changed physical settings and missing modes
or masks cannot be accepted as a numerical refinement. Even a passing pair
does not grant image admission. The gun-exit comparison now also doubles its
radial integration samples and retains the quartic phase explicitly.

## Complete physical comparison plan

Use the same source/electrode/column settings as the preceding round, all
three energies, 64 radial modes, 769 x 257 quadratic near-field nodes,
domain factor 5, relative axial step 0.01 and field step 0.5 mm. The two
pilots start with coordinate 0.2 and transition to either 0.1 or 0.075.
Each performs one quartic wave-following correction. Cartesian export uses
2048 pixels to resolve the additionally retained analytic phase.

    python -m scripts.inspect_surface_column --output <new-directory> ...
        --wave-following-iterations 1 --wave-following-phase-order 4
        --gun-pixels 2048

Acceptance is unchanged: below 1% relative current and 1% complex-field L2
on successive independent refinements. No phase/intensity fitting is allowed.
Physical identities, energy weights and every physical mask event must match.
Conservation is checked separately from field convergence. A completed run
or a smaller current difference alone does not qualify TEM/STEM imaging.

## Complete physical results: not a convergence cure

| Pilot terminal coordinate | Exit current | Runtime | Raw record |
| --- | ---: | ---: | --- |
| 0.1 | 54.02614528 nA | 934.03 s | [Executed](evidence/20260913-quartic-growth01/report.json) |
| 0.075 | 67.58085044 nA | 915.26 s | [Executed](evidence/20260913-quartic-growth0075/report.json) |

Both reports confirm the implementation remained unchanged during execution.
The source, fields, energy identities/weights and all **395 physical mask
events per energy** match. Complex states and numerical inputs are retained.
The observer only captured a failed operator when one occurred; it returned
the original operator without alteration on these completed runs.

[Input-bound comparison](evidence/20260913-quartic-growth-comparison.json):
relative current disagreement **25.0892%**, full mixture-weighted complex
field L2 disagreement **145.7519%**. Both fail the unchanged 1% thresholds.
Compared with the preceding quadratic wave-following round, current
disagreement increased from 13.25%; complex disagreement is still very large.
Quartic adaptation is therefore **not promoted**, and no images are admitted.

The first energy's local error shows the chart is failing early, before the
first physical bore at 6 mm. The comparison reconstructs each full analytic
phase and doubles integration samples; it does not align phases or intensities.

| Axial plane | Complex L2 error | Amplitude L2 error |
| --- | ---: | ---: |
| 2 nm | 1.0952% | 0.8213% |
| 100 nm | 21.6272% | 2.3268% |
| 1 micrometre | 63.3332% | 24.1068% |
| 10 micrometres | 60.1837% | 46.3808% |
| 100 micrometres | 135.0700% | 53.9152% |

The small change already at the common 2 nm interface includes the changed
downstream reflected load; the tip inputs were not changed. The much larger
later error occurs while net current is constant. Thus conservation alone
cannot diagnose the finite radial representation error. Differing downstream
aperture illumination then produces differing physical absorption, in
addition to separately reported unresolved mask modes.

The successful old 0.075 quartic run was also repeated with only the
constant-slab numerical repair:
[kernel-repeat comparison](evidence/20260913-quartic-growth-kernel-repeat.json).
Current changes by 3.6761e-10 relative and complex field by 4.1556e-8. This
supports the scope of the roundoff repair; it does not make the two independent
radial charts converged. That repeat intentionally compares different solver
versions and is not counted as an independent coordinate-refinement pass.

## Next bounded investigation

1. Freeze these failed physical records and leave production admission closed.
2. Build an independently checked radially partitioned representation for
   the first 100--1000 nm, where quartic adaptation now introduces a large
   phase discrepancy. Keep the same electrode field, tip boundary, both
   propagation directions and downstream reflected load.
3. Measure local complex field and derivative residuals under radial and axial
   refinement. Do not choose coordinate coefficients by matching exit current;
   a finite global Laguerre chart and more phase coefficients alone have not
   demonstrated convergence.
4. Re-execute the full three-energy gun and every physical mask only after the
   local independent comparison improves. Image admission still requires the
   complete tip-to-specimen/detector and scan/inelastic validation, not merely
   this source-kernel work.

The current round did not edit physical configurations, retune source/lens
parameters, restart the user's running application, commit, push or shut down.

## Final verification and complete files

[Final focused receipt](evidence/20260913T142103Z-quartic-final-scoped-41e05c9b/report.json):
**116 passed, zero failures/errors/skips**, unchanged hashed inputs. All changed
Python files compiled serially afterward. Both complete physical reports'
solver identities still match the final source implementation. The 15 broader
compatibility failures above remain unresolved and are not included as passes.

Complete implementation files, not partial patches:

- [Quartic radial operators and envelope](../../src/temsim/physics/quartic_radial_phase.py)
- [Executed-wave chart selection](../../src/temsim/physics/wave_following_chart.py)
- [Covariant scattering boundary](../../src/temsim/physics/covariant_boundary.py)
- [Complete round-gun execution](../../src/temsim/physics/radial_gun_wave.py)
- [Surface/gun coupling and export](../../src/temsim/physics/surface_gun_wave.py)
- [Radial column replay](../../src/temsim/physics/radial_column_wave.py)
- [Refinement audit driver](../../scripts/summarize_gun_refinement.py)
