# Wave-following numerical radial charts

**Final status: NOT a completed convergence repair.** The complete physical
comparison still fails (13.25% current difference and 145.74% complex-field
L2). The new optional method is implemented and tested, but is not promoted
to default. The source-to-image production gate remains closed. This continues
the failed independent-coordinate result in
[the previous radial-chart report](GUN_RADIAL_CHART_REPAIR_2026-09-13.md).

## What changed

The optional `WaveFollowingNumerics` corrects the numerical radial width and
quadratic curvature from the **complete already-executed complex wave** at
every boundary. These two moments choose coordinates; they do not replace
the retained wave with a Gaussian, set an exit source, adjust physical source
size, fit comparison phase or repair current.

For each configured energy mode, the implementation:

1. Executes all gun fields and physical masks, embeds the downstream reflected
   load, and jointly solves the driven tip boundary.
2. Computes radial moments of every complex boundary field. It fits log width
   and curvature over log axial position. Exact spline and transition
   derivatives enter the existing covariant operator and basis closure.
3. Releases the completed operators, then rebuilds the entire gun and jointly
   re-solves its reflected load with the physical tip. No old trace is injected
   at an intermediate plane.

The initial matching chart is unchanged up to 10 nm. A C2 transition ends at
100 nm. Iteration count and a numerical width multiplier are explicit
development inputs, recorded in numerical identities. The default iteration
count is zero, so existing paths retain their method. This does not enable
new production imaging, change legacy ray behavior, or modify profiles.

## Mathematics and units

For normalized expectations in the full m=0 Laguerre expansion, let `b` be
the chart width in nm, `c` its curvature in nm^-1 and `k` the reference wave
number in nm^-1. Existing matrices are `X=<r^2/b^2>`, the anti-Hermitian
dilation `D=<1+r*d/dr>` and `T=<-b^2*Laplacian>`.

    R2 = b^2 <X>
    C0 = Im <D>
    P2_intrinsic = (<T> - C0^2 / <X>) / b^2
    b_opt = (R2 / P2_intrinsic)^(1/4)
    c_opt = c + C0 / (k R2)
    minimum_mean_radial_order = (sqrt(R2 P2_intrinsic) - 1) / 2

The intrinsic momentum variance is evaluated **before** adding the large
analytic quadratic chirp. This avoids subtracting nearly equal large chirp
terms. Field scaling is used only to evaluate expectation ratios robustly;
the transported field/current is never rescaled by these coordinates.

All coefficients contribute to the moments, including non-Gaussian radial
content. A zero/non-finite field or violated uncertainty bound is rejected.
The chart cannot extrapolate beyond its executed interval. At a physical
mask with two traces at the same axial position, the right trace supplies
the chart knot; the physical mask is still executed separately.

The chart digest includes the full pilot field, normal derivative, coordinate
history and pilot numerical configuration. It is ephemeral within one
physical energy-mode execution; the parent checkpoint also binds the actual
source, instrument snapshot and solver implementation. It is not an admitted
physical beam cache, a phase model or a configurable downstream source.

## Validation criteria

Existing acceptance remains unchanged: under 1% relative current and 1%
complex-field L2 on two successive independent numerical refinements, with
all energy modes and unchanged physical inputs. No phase, curvature or
intensity fitting is allowed in comparisons. Small moment occupation and
exact current balance alone are insufficient.

Independent focused tests cover:

- Exact Gaussian and pure radial-mode moments, including extremely large
  analytic chirps and very small coefficient amplitudes.
- Real-space complex-field/gradient quadrature of a non-Gaussian superposition.
- Exact coordinate derivatives through the initial transition and refusal to
  extrapolate or define coordinates from a missing wave.
- Retained absolute complex phase against an independent nonparaxial Bessel
  angular-spectrum propagation, using a tabulated coordinate chart.
- Full pilot field and derivative sensitivity of numerical provenance.

These are numerical fixtures, not physical source-to-image acceptance.

The [immutable focused validation receipt](evidence/20260913T130206Z-wave-following-chart-f1ada46e/report.json)
records **80 passed, zero failures/errors/skips**, with unchanged hashed inputs.
Changed Python files also compiled successfully in a subsequent serial check.
The full repository suite and desktop/GPU image workflows were not run.

## Reproduction

Use the existing physical driver with optional numerical-only controls:

    python -m scripts.inspect_surface_column --output <new-evidence-directory> ...
        --wave-following-iterations 1
        --wave-following-width-multiplier 1

Each correction costs another complete physical tip/gun solve. It is not
advertised as a speed optimization, convergence certificate or GUI feature.
Source files must remain unchanged during a calculation for its result to
qualify as an immutable development observation.

## Complete physical comparison: failed convergence

Two full three-energy calculations used the same physical source, electrodes,
overlapping column fields and masks. Each used 64 radial modes, 769 x 257
quadratic near-field nodes, near-domain factor 5, relative axial step 0.01,
field step 0.5 mm and a 256-pixel exported grid. Each ran one wave-following
correction after its pilot. Pilots began with coordinate 0.2 and transitioned
to either guide 0.1 or 0.075 over 100--1000 nm; no physical parameter changed.

| Numerical method | Pilot guide 0.1 current | Pilot guide 0.075 current | Relative difference |
| --- | ---: | ---: | ---: |
| Previous guide-only calculation | 57.32742854 nA | 68.02389540 nA | 18.65855% |
| One wave-following correction | 57.01456937 nA | 64.56747119 nA | 13.24732% |

The corrected weighted complex L2 difference is **1.45736664**, compared
with 1.46822395 before correction. This is essentially unresolved phase
disagreement, not a reliable improvement. Current closeness alone would be
misleading: the first-energy local error at 10 micrometres actually increased.

| First-energy plane | Corrected complex L2 | Corrected amplitude L2 |
| --- | ---: | ---: |
| Near boundary, 2 nm | 0.0104853 | 0.0056091 |
| 1 micrometre | 0.0280793 | 0.0146057 |
| 10 micrometres | 0.5009232 | 0.1737635 |
| 100 micrometres | 1.0618067 | 0.4854646 |

The previous guide-only pair had 0.332011 complex L2 and 0.110879 amplitude
L2 at 10 micrometres. Thus moment-matched width/curvature alone is **not a
demonstrated cure**. The changed near-boundary field does not mean the
physical source was retuned: the jointly solved reflected downstream load
also changes when its unconverged representation changes.

Evidence:

- [Guide-0.1 pilot plus correction](evidence/20260913-wave-following01/report.json):
  completed in 910.055 s; implementation unchanged during execution.
- [Guide-0.075 pilot plus correction](evidence/20260913-wave-following0075/report.json):
  completed in 914.302 s; implementation unchanged during execution.
- [Phase-preserving comparison](evidence/20260913-wave-following-comparison.json):
  all three exit energy modes compared; local boundary comparison explicitly
  limited to 100 micrometres with quadrature doubled.

The two physical-setting identities, energy values, mixture weights and
all **395 mask events per energy** (component, type, z and radius) match.
Maximum near-boundary flux-balance residual is 1.08e-14 of reference current;
maximum mask-ledger residual is 5.00e-16. These conservation checks pass while
complex-field convergence fails. Neither result is an accepted imaging source.

## Next numerical work

Do not run more width-only iterations and describe them as physical
qualification. The persistent growth between 1 and 10 micrometres calls for
an independently checked richer radial phase representation / local basis
adaptation, including its full covariant derivative, boundary closure and
phase-preserving export. Width and quadratic curvature are insufficient in
this tested implementation. A higher-order phase factor is a candidate to
test, not an established solution. Retain the existing physical fields and
stops, two-way reflected load, per-energy phase, and separate mask versus
unresolved-basis loss records in that work.

No TEM/STEM image, full-column validation, production cache, GUI setting,
commit, push or shutdown was produced in this round.
