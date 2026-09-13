# Grounded gun: local coordinate repair and remaining convergence limits

**Final status: partial numerical improvement, NOT a completed convergence
repair.** An independent terminal-chart comparison still changes current by
18.66%. Neither output is qualified for production source-to-image use.

## Scope

This continues [the earlier convergence investigation](GUN_COORDINATE_CONVERGENCE_2026-09-13.md).
It does not qualify a new coherent TEM/STEM source. Physical tip inputs,
electrodes, extraction/acceleration, gun-overlapping column fields, bore
events and apertures are unchanged. All completed gun runs execute the
three configured energy modes and couple the downstream reflected load
back into the driven near-tip boundary. No exit source is supplied.

Acceptance remains: independently refine axial steps, radial representation
and the FEM boundary; compare every energy mode with no fitted phase. The
proposed engineering target is below 1% current and 1% complex-field L2 on
two successive refinements and independent coordinates. A shared final
chart is NOT an independent final-chart convergence check.

## Located failure

Dense physical-plane observations narrow the first major disagreement from
2 nm--6 mm to **1--10 micrometres**. First energy mode, coordinate widths
0.2 and 0.1, 64 radial modes:

| Plane | Relative complex L2 | Relative amplitude L2 |
| --- | ---: | ---: |
| 1 micrometre | 0.006594 | 0.004780 |
| 10 micrometres | 0.447779 | 0.184289 |

[Dense comparison](evidence/20260913-dense-prefix-comparison.json).
Adding observation nodes also subdivides propagation, so this is not a
bitwise replay of the older grid. The raw reports preserve that distinction.

- Reducing relative axial step from 0.1 to 0.01 left the 10 micrometre
  coordinate disagreement essentially unchanged: complex L2 0.446966,
  amplitude L2 0.184505. The three-energy gun currents remain inconsistent:
  26.81794 versus 57.35274 nA. [Comparison](evidence/20260913-fine-relative-coordinate-comparison.json).
- Doubling modes from 64 to 128 for coordinate 0.2 reduced its 10 micrometre
  amplitude difference against coordinate 0.1/64 to 0.125534, but the full
  gun current became 17.22873 nA, not a converged reference.
  [Prefix comparison](evidence/20260913-basis128-prefix-comparison.json).
- At 10 micrometres, the *best possible* orthogonal projection of the
  coordinate-0.1 executed field into the coordinate-0.2/64 basis leaves
  **0.290900 relative complex L2** unrepresented. The reverse projection
  leaves 0.003260. This is radial phase/amplitude resolution, not a physical
  aperture loss or a justification for correcting the current.
  [Bidirectional best-approximation diagnostic](evidence/20260913-fine-basis-support.json).

The best-approximation calculation projects complex fields without phase
alignment or normalisation. It checks the sampled Gram matrix and doubles
radial integration sampling. Its projected coefficients are diagnostic;
they are never passed to the transport or source-admission APIs.

## Implemented numerical changes

### Local step-doubling control

`adaptive_scattering.py` compares one CF4 interval with two half intervals
in the same complete two-port chart. It retains reflection and uses the raw
complex operator difference, without dividing by 15. Exhausted depth, work
budget or cancellation returns no accepted interval. Local indicators are
not rigorous global error bounds near resonances.

**Not promoted to default:** a full-operator adaptive experiment at tolerance
1e-4 exhausted its 900 s diagnostic deadline near 237 nm in the first energy
mode. It did not finish the gun. [Failed run](evidence/20260913-adaptive-prefix-02-tol4/report.json).
Uniformly resolving every unoccupied/evanescent channel's operator error is
too expensive here and does not address the demonstrated radial limitation.

### Smooth radial coordinate transition

`radial_coordinates.py` blends two numerical ellipse guides following the
same executed optical map. Over the chosen numerical interval, a quintic
C2 weight interpolates **log width** and **real curvature**. The exact
transition-rate terms are included in both derivatives. Those derivatives
enter the existing full covariant operator and finite-basis closure:

    G = (k_ref / 2) c' b^2 X + i (b'/b) D

Thus this is a smooth change of numerical basis, not a projected handoff,
new source, physical lens, phase fit, output renormalisation or deletion of
backward channels. The initial tip chart remains unchanged. Physical stops
act on the current chart with the same physical radii and loss ledger.
All chart settings are recorded in the executed numerical identity.

The experimental driver accepts:

    --coordinate-blend TARGET_WIDTH START_MM END_MM

Widths multiply the *physical cap radius only to define a numerical basis*.
They are not emission sizes. The default remains the existing unblended
midpoint development method; this change is not a production source gate.

## Independent checks

- A transitioned numerical chart converges against the exact nonparaxial
  free-space Bessel angular spectrum, retaining absolute complex phase.
  At the finest tested step its relative complex error is 1.43e-8. This
  fixture is not an emitted source or an imaging acceptance test.
- 100 actual constant gun slabs between approximately 0.5 and 17.8
  micrometres agree with an independent matrix-exponential/doubling route
  to at worst 8.85e-10 in full two-port operator infinity norm. The
  independent route then rejected its own current-unitarity check; the
  diagnostic exited unsuccessfully, not as a complete reference run.
  [Partial raw observations](evidence/20260913-actual-slab-comparison/report.json).
- An additional direct potential check at 2, 10, 100, 1000, 10000 and
  100000 nm compared analytic radial moments with split Gauss integration;
  maximum absolute matrix difference was below 8e-12 nm^-2. This was a
  terminal diagnostic, not an immutable test receipt.
- [68 focused regressions](evidence/20260913T123613Z-smooth-radial-chart-4cd96a2e/report.json)
  passed with unchanged hashed inputs. They cover the new chart and its
  derivatives, independent complex-wave propagation, adaptive integration,
  both-port mask accounting, comparison diagnostics, and source admission.
  No full repository suite, GPU or desktop interaction was run.

## Physical transition run

The unchanged full physical configuration completed with coordinate 0.2 at
the tip, smoothly transitioning to the 0.1 guide over 100--1000 nm. All three
energies, the joint near boundary and all gun operations completed in
340.78 s; the implementation hash remained unchanged.

| Full-gun run | Current (nA) |
| --- | ---: |
| Unblended initial coordinate 0.2, relative step 0.01 | 26.81793638 |
| Unblended initial coordinate 0.1, relative step 0.01 | 57.35274496 |
| Initial coordinate 0.2, smooth transition to 0.1 | 57.32742854 |

Compared with unblended coordinate 0.1, the transitioned result differs by
**0.04414% current** and **0.28799% weighted complex-field L2**, without any
current or phase fit. At 10 micrometres, first-energy amplitude disagreement
falls from 18.4505% to **0.30355%**, with complex L2 0.42811%.
[Full executed report](evidence/20260913-smooth-chart02-01/report.json) and
[phase-preserving comparison](evidence/20260913-smooth-chart-comparison.json).

The physical snapshot, component list and three energy modes are identical
to the control. Each mode retains the same **395** bore/aperture events,
including identical physical z and radius. The maximum mask-ledger residual
is 8.19e-16 of source current; the maximum near-boundary balance residual is
1.03e-14. Conservation is checked separately from complex-field accuracy.

This demonstrates effective repair of the bad *initial-chart continuation*
for this comparison. It does **not** prove physical gun-output convergence:
the compared results share the same final 0.1 chart.

### Independent terminal-chart check: failed

A second complete physical execution kept initial coordinate 0.2 but
transitioned to terminal coordinate **0.075**, with all other numerical and
physical settings unchanged. All three energies completed in 333.65 s with
an unchanged implementation hash. Current was **68.02389540 nA**, differing
by **18.65855%** from the transition-to-0.1 output. Weighted complex L2 was
**1.46822**, with per-mode errors 1.46432, 1.45778 and 1.58836.

The first energy still differs by only 0.5399% complex L2 at 1 micrometre,
but grows to 33.2011% complex L2 and 11.0879% amplitude L2 at 10 micrometres.
Thus the fixed radial representation remains inadequate in essentially the
same region. The useful common-terminal-chart result above must not be
mistaken for an independent physical convergence certificate.

[Independent execution](evidence/20260913-smooth-chart02-0075/report.json)
and [failed independent comparison](evidence/20260913-independent-terminal-chart-comparison.json).
Production source admission remains closed. No full TEM or STEM image was
generated or certified by this work.

### Next implementation boundary

The remaining task is a genuinely adaptive radial representation, with
complex transfer residuals and independent phase, support and mode-count
checks. A wave-following chart or higher-order radial phase representation
needs its own derivation and independent validation before replacing the
fixed ellipse guide. It must retain the complete complex state, both-port
reflection coupling and physical masks. Another manually selected terminal
width, a current rescaling, a fitted comparison phase or a smaller axial
step alone does not resolve the demonstrated defect.

All 68-test input hashes were rechecked unchanged after the receipt, and
the changed Python files passed serial compilation. Existing particle-ray,
legacy image and GUI paths were not rewritten; they were not subjected to
a fresh full image/desktop acceptance run here.

No commit, push or shutdown is part of this repair request. The user's
running application is not restarted. Existing dirty work is preserved.
