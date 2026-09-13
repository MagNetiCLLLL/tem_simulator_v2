# Gun-current attribution and local propagation audit

Status: the current difference is accounted for. The propagated wave remains
coordinate-dependent and is **not qualified for TEM/STEM imaging**. The
source, electrodes, apertures, existing ray paths and production admission
policy have not been changed in this investigation.

## What changed, and what did not

The two completed runs in the preceding
[quartic-phase investigation](QUARTIC_GUN_PHASE_2026-09-13.md) used the same
physical tip reservoir, three-energy distribution, extraction/acceleration,
round fields and 395 physical masks per energy. Their terminal numerical
radial coordinate widths were 0.1 and 0.075 times the emitting-cap radius.
These are coordinate choices, not different emission areas.

The [current ledger](evidence/20260913-gun-current-attribution-validated.json) gives:

| Current contribution (nA) | Coordinate 0.1 | Coordinate 0.075 |
| --- | ---: | ---: |
| Prescribed incoming tip reservoir | 100.000000 | 100.000000 |
| Reflected back to the tip | 24.039137 | 24.057873 |
| Net entering the gun | 75.960858 | 75.942126 |
| Computed physical-mask absorption | 20.543347 | 7.974874 |
| Unresolved wave after finite-basis masks | 1.391365 | 0.386402 |
| Gun exit | 54.026145 | 67.580850 |

Numerical side-port escape is 0.000005124 and 0.000001189 nA, respectively.
The net entering-gun row is an intermediate balance, not an extra loss.
Increasing the exit by 13.554705 nA is accounted for by:

- DPA aperture computed absorption decreasing by 5.838033 nA.
- C1 aperture computed absorption decreasing by 6.730440 nA.
- Unresolved masked wave decreasing by 1.004963 nA.
- A net entering-gun decrease of 0.018731 nA, opposing those increases.

**Computed physical-mask absorption is not a certified physical prediction.**
The opening sizes are identical, but an unconverged incident wave places a
different amount of current outside each opening. The aperture turns that
spatial error into a large exit-current error. This is not an observed change
in tip emission or a legitimate dependence on a user-selected numerical width.

The ledger preserves signed conservation residuals and full energy weights;
it does not normalize away missing current. Historical total-mask removal is
not relabelled as physical absorption. The audit reads saved results only and
does not publish an equivalent source state.

## Why current conservation alone did not detect it

Two complex waves can have the same integrated current but different radial
intensity and phase. Both can pass a current-conservation check in vacuum,
then give different transmission through the same aperture. The present
problem requires agreement of the physical complex wave and its derivative,
not just conservation or a small tail occupation in one coordinate basis.

Before the first physical mask, the first-energy complex-field disagreement
is already 21.6% at 100 nm and 63.3% at 1000 nm. The difference therefore
does not originate only in the mask operation.

## Independently checked candidates

1. [Actual electrode-potential projection](evidence/20260913-actual-gun-potential-projection.json):
   18 comparisons, covering three energies, both coordinates and 100, 1000,
   10000 nm. Analytic piecewise integration agrees with independent positive
   quadrature to maximum relative matrix error 1.372e-13. Maximum occupied
   wave-action error is 5.445e-12 nm^-2. This checks the integration of the
   supplied electrode field, not the accuracy of the electrostatic model itself.
2. [Cross-basis support](evidence/20260913-quartic-cross-basis-support.json):
   at 100 and 1000 nm, best-approximation residuals for the first energy are
   about 0.33--1.61%, much smaller than the difference between propagated
   fields. Integration refinement changes these diagnostics by less than
   1.6e-9. This does not prove dynamical basis convergence; it shows that a
   simple final-plane representation deficit alone is an incomplete explanation.
3. [Local axial comparison](evidence/20260913-actual-gun-axial-refinement/report.json):
   changing only subdivisions of actual gun intervals produces much larger
   complex two-port changes after wave-following coordinate adaptation than
   in the pilot chart. Eight subdivisions are still not a converged reference
   in several intervals. Those raw infinity norms include all 64 channels;
   they must not be misreported as a percent current error.

## Local method and scope

The observer re-executes the recorded physical settings. A complete
first-energy tip/gun boundary-value solve supplies the numerical chart, with
the reflected downstream load and every mask retained. The observer returns
the **original** production operator to that solve; comparison operators are
never silently substituted. It compares midpoint and fourth-order
commutator-free Magnus steps, with independent subdivision in the Liouville
action coordinate, retaining both propagation directions and complex phase.

The CF4 construction follows
[Blanes and Moan (2006), author-hosted paper](https://personales.upv.es/~serblaza/2006APNUM.pdf).
This reference supports the numerical method, not our physical-gun accuracy.
The independently integrated small-matrix test checks its refinement behavior.

The production midpoint is taken in physical z, whereas the comparison's
midpoint is in the action coordinate. Their difference is explicitly labelled
`physical_z_midpoint_vs_action_midpoint`; the two are not assumed identical
in an accelerating field. The observer stops intentionally without publishing
a complete three-energy source checkpoint. No image acceptance is claimed.

## Confirmed error on the occupied physical wave

The [occupied-wave audit](evidence/20260913-actual-gun-occupied-refinement/report.json)
completed in 275.80 s with unchanged implementation. It re-executed both
complete first-energy boundary-value solves (pilot and quartic-following),
including all gun masks and the reflected tip load. At each inspected step,
its actual complex left-incoming and right-incoming amplitudes were retained.
No synthetic downstream beam was injected. Those two incoming traces are
held fixed when testing alternative local propagation operators: this is
**local sensitivity**, not a globally refined boundary-value solution.

For the first energy (0.1874454417 eV), the following-chart results are:

| Interval start (nm) | Production vs CF4/16, occupied complex norm | CF4/8 vs CF4/16, occupied complex norm |
| --- | ---: | ---: |
| 100.000 | 2.2133% | 0.1265% |
| 398.108 | 58.4681% | 2.2415% |
| 1000.000 | 1.6594% | 0.00955% |
| 10000.000 | 0.05460% | 0.0000115% |

The 398.108--402.089 nm interval is only 3.981 nm long. Before the
wave-following chart, its occupied production/CF4 comparison is 8.505e-6
(fraction, not percent); after adaptation it is 0.58468. The discrepancy is
therefore not just a worst-case matrix norm on unoccupied high-order modes.
All compared operators preserve the modeled channels and both directions.
Their individual current conservation does not guarantee correct propagation.

This confirms a concrete defect in this experiment: the old fixed axial
step is inadequate after adapting the numerical radial phase/width. Changing
the coordinate chart changes a rapidly varying discrete operator; freezing
that operator over one step produces a different physical wave. In turn,
the same physical aperture integrates a different incident distribution.

CF4/16 is **not an exact reference**: the rightmost column remains too large
at several locations. The 58.47% figure is a measured difference, not a
certified error relative to the true solution, nor a percent exit-current
error. It does not prove that axial discretization alone accounts for all
25.09% of the full-run current discrepancy. Dynamical radial truncation and
tip-interface convergence still need independent checks.

The next repair must resolve these local complex-field/derivative errors,
recompute the coupled reflected load, and then repeat the full three-energy
comparison with independent radial coordinates. Merely adding a higher
radial phase order without refining propagation is not sufficient. Do not
tune the physical emission current, change apertures, normalize the output,
or promote this diagnostic into a source/image cache to make it pass.

## Reproduce

From the project root, using the project environment:

```powershell
.venv\Scripts\python.exe -m scripts.audit_gun_current `
  docs/development/evidence/20260913-quartic-growth01/report.json `
  docs/development/evidence/20260913-quartic-growth0075/report.json `
  --output <new-ledger.json>
.venv\Scripts\python.exe -m scripts.audit_gun_potential_projection `
  docs/development/evidence/20260913-quartic-growth01/report.json `
  docs/development/evidence/20260913-quartic-growth0075/report.json `
  --output <new-projection-audit.json>
.venv\Scripts\python.exe -m scripts.audit_gun_axial_steps `
  docs/development/evidence/20260913-quartic-growth01/report.json `
  --output <new-evidence-directory> --occupied-waves --timeout 600
```

Use new output paths; raw evidence is not overwritten. Re-execution from a
snapshot requires its source implementation identity to match. Historical
records can still be read for accounting if the implementation later changes;
they must not be admitted as a new active source under a different solver.

## Validation and remaining work

[Refined focused receipt](evidence/20260913T144100Z-gun-current-axial-refined-5b0e3c3a/report.json):
14 passed, no failures/errors/skips, unchanged hashed inputs. Serial compilation
of the three audit drivers and two new test modules also passed. The earlier
13-case attempt had one failure: eight CF4 subdivisions did not meet the
predeclared 1e-7 fixture tolerance (2.647e-7 observed). Sixteen subdivisions
meet the same tolerance; it was not loosened. The failed receipt is retained.

This round has not modified production physical/numerical models, default
source values, legacy ray/image paths or GUI behavior. It does not supersede
the preceding 68-pass/15-failure compatibility receipt with a full-suite pass.
Full energy-mixture, independent-coordinate gun convergence and source-to-TEM/
STEM image acceptance still need to be demonstrated after the propagation
error is repaired. No commit, push, running-application restart or shutdown
was performed.

## Continuing full-source work: remote intervals are also unresolved

The later frozen [actual far-gun diagnostic](evidence/20260913-far-integrator-audit/integrator_audit.json)
completed in 458.10 seconds with unchanged inputs. It re-solved the physical
tip and complete first-energy gun load, then compared actual occupied ports
without fitting phase or current. Each entry below is relative to CF6 with
16 subdivisions, which is **not a certified continuum reference**.

| Approximate Z | CF4, one original interval | CF6, one interval | CF6, eight subdivisions |
| --- | ---: | ---: | ---: |
| 10 micrometres | 6.8302e-4 | 1.0828e-4 | 4.6719e-6 |
| 100 micrometres | 1.5418e-4 | 6.2994e-6 | 1.1622e-6 |
| 1 mm | 5.0045e-2 | 1.0095e-2 | 3.6286e-5 |
| 10 mm | 5.7687e-1 | 8.6401e-2 | 7.5922e-3 |
| 100 mm | 2.1113e-1 | 8.1825e-2 | 3.6531e-3 |
| 400 mm | 6.2318e-2 | 1.4615e-2 | 1.4581e-4 |

Thus refining only the first few hundred nanometres cannot establish a
correct aperture illumination or exit current. Large complex-field changes
also occur far downstream in the moving phase/basis representation. These
changes are much larger than a small floating-point carrier-rounding effect;
that does not independently bound all rounding errors. The complete
three-energy propagation and radial/coordinate convergence remain required.
The ongoing implementation and test receipts are recorded in
[the full-product work log](NEW_SOURCE_FULL_ACCEPTANCE_2026-09-13.md).
