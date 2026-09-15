# Tip, crossover and specimen-probe audit

## Decision and user requirement

**The new-source probe is not qualified. No candidate was applied to defaults
or to the open application.** Matching alpha95 alone had accepted an optical
branch with a micrometre spot and the wrong crossover sequence; that branch
is now explicitly labelled as a rejected diagnostic candidate.

The user requires the crossover count, order and intervals between optical
components to remain as before the tip-source change. Electron count is not
the invariant: physical aperture, wall and detector absorption must remain.
At the specimen upper surface, the target remains current-weighted
alpha95 = 30 mrad with a nanometre-scale probe. These conditions must hold
together, including numerical convergence. No source shrink, independent
gun-exit source, hidden ray removal or aperture bypass is permitted.

## Historical reference and scope

An isolated archive of commit `0a71832` (before the curved-particle default)
was executed using its own `src` and `configs`. No checkout or active profile
was replaced. The comparison uses the historical Gaussian tip-emission law
and its analytic gun-field model, followed by its physical column transport.
The new model uses curved surface emission and the solved gun field. Thus
the comparison changes the **source/gun model**, not just one tip-size number.

Initial reference: Nanoprobe / Diffraction, 300 kV, 769 tip samples,
0.05 mm maximum column step, 0.5 mm checkpoint spacing. Reference waists were
then re-executed at exact full-precision planes, not measured from plot pixels
or linearly interpolated display arrays. Vacuum transport was off. No sample
interaction or wave-image calculation was requested.

The audited topology range is gun exit Z=450 mm through the specimen upper
surface Z=1599.1999975 mm. This range does **not** certify gun-internal or
post-specimen crossover topology. A future accepted operating point must
also verify those regions; the already rejected candidate does not warrant
claiming full-column acceptance. A waist inside or beyond the sample is not
counted as an upstream entrance-plane waist.

## Measured sizes: distinguish diameter from radius

All sizes below are **diameters containing 95% of surviving current**, about
the beam centroid. They are not full geometric extents, RMS radii or FWHM.
The physical cap itself has diameter 34.7296 nm; its sampled d95 is smaller.

| Checkpoint | Historical source/gun | New source/gun with the historical lens settings |
| --- | ---: | ---: |
| Tip emission footprint d95 | 10.0296 nm | 33.8299 nm |
| Gun exit, Z=450 mm, d95 | 4.94566 um | 3.95505 mm |
| First C1 crossover Z | 503.248154 mm | 503.066067 mm |
| First C1 crossover d95 | 43.3932 nm | 53.0525 um |
| First C1 crossover alpha95 | 0.483762 mrad | 398.752 mrad |
| Sample upper-surface d95 | 0.479471 nm | No transmitted samples in this uniform 769-ray run |

Both C1 controls are 51.666666666666664%. The new-source C1 result is a
numerical diagnostic, **not a paraxial-accuracy qualification**: the very large
angle invalidates assuming small-angle errors are negligible. Other
aberrations and higher-order field errors have not been bounded at this point.

The historical C1 reduces the incoming beam (d95=2.95311 um at C1 centre)
to 43.3932 nm. That does not mean its image must be smaller than the initial
10.0296 nm physical emission footprint. Physical tip footprint, effective
source image and beam diameter at a lens are different quantities. In
particular, a quoted FEG virtual-source size must not be substituted as a new
configurable source after acceleration.

The historical sample result has alpha95=24.600907 mrad and a covariance
waist 2.70253 nm after the upper surface. It demonstrates the previous model's
small spot; it is **not** an achieved new-source 30 mrad entrance-plane result.

The discrepancy is already present before C1: the gun-exit d95 grows by
about 800 times, whereas the tip footprint grows by about 3.4 times. Therefore
the evidence does not support blaming tip diameter alone. Emission direction,
energy and the gun-field transfer need to be matched as a coupled system.
This is not proof that every physically possible setting is unreachable.

## Crossover intervals: equal total count is insufficient

The following are executed, fixed-sampling radial-variance minima. Positive
current rays shared by both ends of each bracket define its population;
clipping outer rays must not manufacture a false waist. Decorative pole-piece
solids and derived image planes are not separate optical boundaries.

| Historical order | Required component interval | Historical Z (mm) | Historical d95 (nm) |
| ---: | --- | ---: | ---: |
| 1 | C1 -> C2 | 503.248154 | 43.3932 |
| 2 | C2 -> C2 aperture | 668.190651 | 6.45921 |
| 3 | Probe DP22 -> Probe HPC | 1195.925946 | 3490.62 |
| 4 | Probe TL12 -> Condenser stigmator | 1372.960017 | 4.20909 |
| 5 | Mini Condenser -> Objective / sample | 1585.922469 | 0.324406 |

A second historical run increased the particle budget to 1537, halved the
maximum column step to 0.025 mm and checkpoint spacing to 0.25 mm. It found
the same five ordered component intervals. Their Z positions were
503.249376, 668.190864, 1195.951305, 1372.960001 and 1585.922469 mm.
This supports the discrete topology reference, **not quantitative size
convergence**: C1 d95 changed from 43.3932 to 40.0686 nm and specimen d95
from 0.479471 to 0.459218 nm. Particle and step effects were changed together
in this corroboration and have not been independently separated.

The existing Ray Diagram markers also merge supplemental gun/corrector
markers and use display-path node minima. Their displayed total is not this
incident-only, current-weighted audit count. This work does not change those
GUI markers or claim that their node-level coordinates qualify focus sizes.

The rejected Custom 30 mrad candidate has C1=8.012821601837457%,
C2=84.12563904641223%, C3=32.49174967180758% and
Objective=25.275237294853614%. Its upstream minima are at
628.167181, 650.968148, **770.630596**, 1194.573143 and 1368.564079 mm.
The extra third minimum lies between the C2 aperture and condenser deflector;
the Mini Condenser -> Objective minimum is missing. Total count is still
five, but order/interval matching fails. The sample d95 is also 8.90457 um,
and reusing its lens settings with 385 rays shifts focus by 4.67103 um.

This candidate is rejected on **spot size, topology and particle sampling**.
It is not a preset to load for the requested nanometre probe. The additional
Ideal candidate is likewise not qualified; see
[surface-focus evidence](SURFACE_PROBE_FOCUS_2026-09-14.md).

## Numerical changes and bounded diagnostic sampling

`beam_path_audit.py` adds current-weighted spot measurements, exact incident
checkpoints, bracketed/refined waist measurements and ordered component-
interval comparison. Empty transmitted populations have no spot size; they
are not reported as a zero-diameter focused beam. Tip angular measurements
use full 3-D emission directions rather than slopes that would fold backward
emission into forward rays.

An optional `apex_stratified_v1` quadrature resolves nine disjoint area strata
from the apex to the full emitting-cap edge. Each sample retains its actual
area weight and multiple local directions. No emitting area is omitted or
renormalised into a small artificial source. Each annulus has its own azimuth
sequence to avoid angular aliasing. The rule is a numerical budget option,
not a physical tip change, and invalidates the gun-trace cache. Default
uniform-area sampling and physical settings are unchanged.

A 193-ray execution of this optional rule under the historical lens preset
resolves low-current trajectories through C1 and C2 but none to the sample.
187 rays pass the gun, carrying **66.25%**, not 187/193 of source current;
27 sampled trajectories reach the C2 region. These observations are not
particle-converged. They show why zero surviving samples in the uniform
769-ray comparison must not be interpreted as mathematically zero accepted
physical current. Angular/energy acceptance remains under-resolved as well.

## Reproduction and remaining work

The subsequent [gun-field numerical repair](GUN_AXIS_MATCHING_REPAIR_2026-09-14.md)
documents the near-axis force defect, electrode-local mesh refinement and
new executed comparisons. It does not qualify a new lens preset or probe.

Use the project interpreter. The audit writes only scalar JSON reports, not
particle arrays or calculation caches:

```powershell
.\.venv\Scripts\python.exe scripts/check_crossover_chain.py --historical-root <isolated-src-configs-tree> --rays 769 --refine-crossovers --output-report <new-report.json>
.\.venv\Scripts\python.exe scripts/check_crossover_chain.py --profile profiles/particle_tip_surface_30mrad_769_custom_draft.toml --rays 769 --reference-report <historical-report.json> --refine-crossovers --output-report <new-candidate-report.json>
```

The local historical tree and generated reports under
`outputs/crossover-audit-20260914/` are ignored by Git. They are not inputs to
the active application. Reuse requires the exact executed upstream source,
optics, model and numerical identity; a historical snapshot with a changed
implementation digest must not be relabelled as current.

Next matching work must constrain the gun exit/C1 incoming phase space first,
then keep every required crossover in its original component interval while
fitting specimen entrance focus, d95 and alpha95 together. Use bounded
classical-particle runs and convergence checks; do not resume coherent-wave
development or launch a large image calculation to conceal unresolved optics.
No new-source nm-probe or full TEM/STEM imaging acceptance is claimed.

## Validation record

129 affected tests passed in 105.83 s: beam-path audit, importance sampling,
surface focus, multi-direction surface sampling, tip particle geometry,
tip assembly particles, shared-tip behavior and the source contract. The
16 warnings were existing Pydantic `json_encoders` deprecations. These tests
include analytical/isolated fixtures; the historical and new-source particle
runs described above are separate execution evidence. No full-project suite,
visible GUI, coherent wave or TEM/STEM image acceptance was run for this audit.
